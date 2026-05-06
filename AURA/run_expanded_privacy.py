#!/usr/bin/env python3
"""Dynamic privacy attribute expansion + re-id evaluation for AURA pipeline.

Flow:
1) Run direct-intent web-search re-id on original transcripts.
2) Generate transcript-specific privacy attributes using transcript text + evidence.
3) Run AURA pipeline with per-document attribute scopes (phase0 + masker).
4) Run direct-intent on rewritten outputs, then feedback-rerun re-identified docs.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from openai import OpenAI

import db
import phase0_init
import phase1_masker
import pipeline
import pipeline_config as cfg
from phase1_masker import AttributeSpec

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None


EXPORT_DIR = cfg.BASE_DIR / "output" / "adaptive_attri"
ORIGINAL_DIRECT_INTENT_PATH = EXPORT_DIR / "nobranch_original_direct_intent.json"
EXPANDED_ATTRIBUTES_PATH = EXPORT_DIR / "nobranch_expanded_privacy_attributes.json"
REID_OUTPUT_PATH = EXPORT_DIR / "nobranch_privacyattribute_reid.json"

DIRECT_INTENT_MODEL = "gpt-5.1"
MAX_TRANSCRIPT_PROMPT_CHARS = 7000
MAX_EVIDENCE_FIELD_CHARS = 1600
MAX_IDENTIFIER_CANDIDATES_IN_PROMPT = 3

CONFIDENCE_RANK = {
    "very low": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "very high": 5,
}
NON_SPECIFIC_LABEL_TOKENS = (
    "unknown",
    "unidentified",
    "anonymous",
    "anonymized",
    "likely non-public",
    "generic identity",
)

BASE_PRIVACY_ATTRIBUTES = tuple(phase0_init.PRIVACY_ATTRIBUTES)
BASE_MASKER_ATTRIBUTE_SPECS = tuple(phase1_masker.ATTRIBUTE_SPECS)


def _resolve_export_paths(
    export_dir: Path,
    name_prefix: str,
) -> tuple[Path, Path, Path, Path]:
    safe_prefix = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name_prefix or "").strip())
    if not safe_prefix:
        safe_prefix = "nobranch"
    return (
        export_dir / f"{safe_prefix}_original_direct_intent.json",
        export_dir / f"{safe_prefix}_expanded_privacy_attributes.json",
        export_dir / f"{safe_prefix}_privacyattribute_reid.json",
        export_dir / f"{safe_prefix}_rewritten.csv",
    )

DIRECT_INTENT_SYSTEM_PROMPT = """
I have an interview transcript. Based on the content, try your best to guess who this person might be. Consider all clues including their role, experience, projects they mentioned, opinions, writing style, and any other details that might help identify them. Be thorough and consider how different pieces of information might combine to point to a specific individual.

Rules:
- Be conservative and precise; avoid hallucinations. Prefer real links and include associated details when possible.
- Use web search to validate details whenever helpful.
- Return STRICT JSON matching the schema exactly; do not include any extra commentary.

Output: Return STRICT JSON with the following schema:
{
  "identifier_candidates": [
    {
      "interviewee_candiates": array of strings (if known),
      "identity_label": string,
      "platform_or_source": string | null,
      "link": string,
      "year": integer | null,
      "confidence": "very low" | "low" | "medium" | "high" | "very high",
      "matches_description": string,
      "mismatches_description": string
    }
  ]
}

Guidance:
- The goal is to find the most likely publicly visible item that matches the description (can stop early if very high confidence).
- Make "confidence" reflect how strongly the identified item aligns with the provided description, given evidence.
- In "matches_description", cite specific phrases or details that align with the description.
- In "mismatches_description", call out missing or contradictory details vs the description.
- If unsure, include candidates with "very low" confidence and explain why.
""".strip()

DIRECT_INTENT_USER_PROMPT_TEMPLATE = """
Here is the transcript:
{full_transcript}
""".strip()

ATTRIBUTE_EXPANSION_SYSTEM_PROMPT = """
You are a privacy risk analyst specializing in transcript de-identification.
Your job is to derive actionable masking attributes from concrete re-identification evidence.
Prioritize attributes that correspond to specific quasi-identifiers present in the text.

Return valid JSON only.
""".strip()

ATTRIBUTE_EXPANSION_USER_PROMPT_TEMPLATE = """
Base attributes already present (DO NOT repeat these or close synonyms):
{base_attributes_json}

Transcript ID: {transcript_id}

Original transcript excerpt:
{transcript_excerpt}

Re-identification evidence (top candidates with confidence, matches, and mismatches):
{reid_evidence_json}

Task:
Propose additional privacy attributes beyond the base list that directly capture the specific evidence used to re-identify this transcript.

Priorities:
1) Produce HIGH-LEVEL categories that group related quasi-identifiers.
   Good: RESEARCH_AREA (specific topics/methodologies/subfields)
   Good: TOOL_STACK (software/platform/framework mentions)
   Good: PUBLICATION_SIGNATURE (paper titles/venues/co-author patterns)
   Good: ORG_TYPE (employer or organization category)
   Good: CAREER_STAGE (early-career/mid-career/senior patterns)
   Bad: SPECIFIC_PAPER_TITLE, EXACT_TOOL_VERSION, INDIVIDUAL_PROJECT_NAME
2) Merge overlapping quasi-identifiers into a single broader attribute.
3) Avoid micro-level singleton attributes tied to one exact proper noun.
4) Ensure each attribute is actionable for a masker (generalizable, not just removable).
5) Return at most {max_attributes} attributes.

Return STRICT JSON with this schema:
{{
  "attributes": [
    {{
      "key": "UPPERCASE_SHORT_KEY",
      "display_name": "Human-friendly name",
      "target_str": "what this attribute targets in text",
      "options": ["optional", "categorical", "values"],
      "note": "optional note"
    }}
  ]
}}

Rules:
- All returned attributes must be different from the base 8.
- Avoid near-duplicates of each other.
- Keep key length <= 16 chars.
- If no strong additions exist, return an empty list.
""".strip()

_thread_local = threading.local()


def _get_thread_openai_client() -> OpenAI:
    client = getattr(_thread_local, "openai_client", None)
    if client is None:
        client = cfg.get_openai_client()
        _thread_local.openai_client = client
    return client


def _get_thread_pipeline_client() -> OpenAI:
    client = getattr(_thread_local, "pipeline_client", None)
    if client is None:
        client = cfg.get_pipeline_client()
        _thread_local.pipeline_client = client
    return client


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_rewritten_csv(
    path: Path,
    ordered_ids: list[str],
    rewritten_by_id: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["transcript_id", "text"])
        for transcript_id in ordered_ids:
            writer.writerow([transcript_id, rewritten_by_id.get(transcript_id, "")])


def _load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return []


def _truncate_text(text: str, max_chars: int) -> str:
    clean = str(text or "").strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 18].rstrip() + "\n...[truncated]"


def _parse_identifier_candidates(response_obj) -> list[dict]:
    output_items = getattr(response_obj, "output", []) or []
    for item in output_items:
        if getattr(item, "type", "") != "message":
            continue
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", "")
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                candidates = parsed.get("identifier_candidates")
                if isinstance(candidates, list):
                    return candidates
    return []


def _run_direct_intent_once(
    transcript: str,
    model: str,
    retries: int = 3,
) -> list[dict]:
    user_prompt = DIRECT_INTENT_USER_PROMPT_TEMPLATE.format(full_transcript=transcript)
    client = _get_thread_openai_client()

    for attempt in range(1, retries + 1):
        try:
            response = client.responses.create(
                model=model,
                reasoning={"effort": "high"},
                input=[
                    {"role": "system", "content": DIRECT_INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                tools=[{"type": "web_search"}],
            )
            return _parse_identifier_candidates(response)
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(1.5 * attempt)

    return []


def run_direct_intent_parallel(
    text_by_id: dict[str, str],
    output_path: Path,
    max_workers: int,
    model: str = DIRECT_INTENT_MODEL,
    force_ids: set[str] | None = None,
) -> dict[str, list[dict]]:
    if not text_by_id:
        return {}

    force_lower = {x.lower() for x in (force_ids or set())}
    ordered_ids = list(text_by_id.keys())
    existing_rows = _load_json_list(output_path)
    results_by_id: dict[str, list[dict]] = {}
    processed_lower = set()

    for row in existing_rows:
        transcript_id = str(row.get("transcript_id", "")).strip()
        if not transcript_id:
            continue
        if transcript_id.lower() in force_lower:
            continue
        candidates = row.get("identifier_candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        results_by_id[transcript_id] = candidates
        processed_lower.add(transcript_id.lower())

    pending_ids = [
        tid
        for tid in ordered_ids
        if tid.lower() in force_lower or tid.lower() not in processed_lower
    ]
    print(
        f"[direct-intent] output={output_path.name} "
        f"existing={len(results_by_id)} pending={len(pending_ids)}"
    )

    def _ordered_results() -> dict[str, list[dict]]:
        payload: dict[str, list[dict]] = {}
        for tid in ordered_ids:
            candidates = results_by_id.get(tid, [])
            if not isinstance(candidates, list):
                candidates = []
            payload[tid] = candidates
        return payload

    if not pending_ids:
        for tid in ordered_ids:
            results_by_id.setdefault(tid, [])
        return _ordered_results()

    worker_count = max(1, min(max_workers, len(pending_ids)))
    write_lock = threading.Lock()
    completed = 0

    def _worker(transcript_id: str) -> tuple[str, list[dict]]:
        transcript = text_by_id[transcript_id]
        candidates = _run_direct_intent_once(transcript=transcript, model=model)
        return transcript_id, candidates

    def _persist() -> None:
        payload = []
        for transcript_id in ordered_ids:
            if transcript_id in results_by_id:
                payload.append(
                    {
                        "transcript_id": transcript_id,
                        "identifier_candidates": results_by_id[transcript_id],
                    }
                )
        _write_json(output_path, payload)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_worker, tid): tid for tid in pending_ids}
        for fut in as_completed(futures):
            transcript_id = futures[fut]
            try:
                tid, candidates = fut.result()
                if not isinstance(candidates, list):
                    candidates = []
            except Exception as exc:
                tid = transcript_id
                candidates = []
                print(f"[direct-intent] ERROR {transcript_id}: {type(exc).__name__}: {exc}")

            with write_lock:
                results_by_id[tid] = candidates
                processed_lower.add(tid.lower())
                completed += 1
                print(
                    f"[direct-intent] {completed}/{len(pending_ids)} "
                    f"{tid}: candidates={len(candidates)}"
                )
                _persist()

    for tid in ordered_ids:
        results_by_id.setdefault(tid, [])
    return _ordered_results()


def _call_chat_json(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1200,
) -> dict:
    client = _get_thread_pipeline_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = cfg.strip_think_tags(response.choices[0].message.content or "{}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return payload


def _normalize_key(raw_key: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_]", "_", str(raw_key).strip().upper())
    key = re.sub(r"_+", "_", key).strip("_")
    if not key:
        return ""
    if not key[0].isalpha():
        key = f"A_{key}"
    return key[:16]


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def _normalize_options(raw_options) -> list[str] | None:
    if not isinstance(raw_options, list):
        return None
    output: list[str] = []
    seen = set()
    for item in raw_options:
        text = _normalize_text(item)
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        output.append(text)
    return output or None


def _coerce_privacy_attribute(raw: dict) -> phase0_init.PrivacyAttribute | None:
    if not isinstance(raw, dict):
        return None
    key = _normalize_key(raw.get("key", ""))
    display_name = _normalize_text(raw.get("display_name", ""))
    target_str = _normalize_text(raw.get("target_str", ""))
    if not key or not display_name or not target_str:
        return None

    options = _normalize_options(raw.get("options"))
    note = _normalize_text(raw.get("note", "")) or None
    return phase0_init.PrivacyAttribute(
        key=key,
        display_name=display_name,
        target_str=target_str,
        options=options,
        note=note,
    )


def _collect_identifier_evidence(identifier_candidates: list[dict]) -> list[dict]:
    evidence: list[dict] = []
    for raw in identifier_candidates[:MAX_IDENTIFIER_CANDIDATES_IN_PROMPT]:
        if not isinstance(raw, dict):
            continue
        interviewee_candidates = raw.get("interviewee_candiates", [])
        if not isinstance(interviewee_candidates, list):
            interviewee_candidates = []
        interviewee_candidates = [
            _normalize_text(x)
            for x in interviewee_candidates
            if _normalize_text(x)
        ][:6]

        entry = {
            "confidence": _normalize_text(raw.get("confidence", "")),
            "identity_label": _truncate_text(raw.get("identity_label", ""), 240),
            "interviewee_candiates": interviewee_candidates,
            "matches_description": _truncate_text(
                raw.get("matches_description", ""),
                MAX_EVIDENCE_FIELD_CHARS,
            ),
            "mismatches_description": _truncate_text(
                raw.get("mismatches_description", ""),
                MAX_EVIDENCE_FIELD_CHARS,
            ),
        }
        evidence.append(entry)
    return evidence


def _attribute_identity_triplet(attr: phase0_init.PrivacyAttribute) -> tuple[str, str, str]:
    return (
        attr.key.strip().upper(),
        attr.display_name.strip().lower(),
        attr.target_str.strip().lower(),
    )


def _merge_attribute_lists(
    base: list[phase0_init.PrivacyAttribute],
    extras: list[phase0_init.PrivacyAttribute],
) -> list[phase0_init.PrivacyAttribute]:
    merged: list[phase0_init.PrivacyAttribute] = []
    seen_keys: set[str] = set()
    seen_names: set[str] = set()
    seen_targets: set[str] = set()

    for attr in [*base, *extras]:
        key, name, target = _attribute_identity_triplet(attr)
        if key in seen_keys or name in seen_names or target in seen_targets:
            continue
        seen_keys.add(key)
        seen_names.add(name)
        seen_targets.add(target)
        merged.append(attr)
    return merged


def _filter_incremental_attributes(
    existing: list[phase0_init.PrivacyAttribute],
    candidates: list[phase0_init.PrivacyAttribute],
    limit: int,
) -> list[phase0_init.PrivacyAttribute]:
    limit = max(0, limit)
    existing_keys = {x.key.strip().upper() for x in existing}
    existing_names = {x.display_name.strip().lower() for x in existing}
    existing_targets = {x.target_str.strip().lower() for x in existing}

    selected: list[phase0_init.PrivacyAttribute] = []
    for attr in candidates:
        key, name, target = _attribute_identity_triplet(attr)
        if key in existing_keys or name in existing_names or target in existing_targets:
            continue
        existing_keys.add(key)
        existing_names.add(name)
        existing_targets.add(target)
        selected.append(attr)
        if len(selected) >= limit:
            break
    return selected


def _generate_attrs_for_transcript(
    transcript_id: str,
    transcript_text: str,
    identifier_candidates: list[dict],
    existing_attributes: list[phase0_init.PrivacyAttribute],
    model: str,
    prompt_max_attributes: int,
) -> tuple[str, list[phase0_init.PrivacyAttribute]]:
    base_payload = [asdict(attr) for attr in existing_attributes]
    evidence_payload = _collect_identifier_evidence(identifier_candidates)
    user_prompt = ATTRIBUTE_EXPANSION_USER_PROMPT_TEMPLATE.format(
        base_attributes_json=json.dumps(base_payload, ensure_ascii=False, indent=2),
        transcript_id=transcript_id,
        transcript_excerpt=_truncate_text(transcript_text, MAX_TRANSCRIPT_PROMPT_CHARS),
        reid_evidence_json=json.dumps(evidence_payload, ensure_ascii=False, indent=2),
        max_attributes=max(0, int(prompt_max_attributes)),
    )
    payload = _call_chat_json(
        model=model,
        system_prompt=ATTRIBUTE_EXPANSION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=1800,
    )

    raw_attrs = payload.get("attributes", [])
    if not isinstance(raw_attrs, list):
        raw_attrs = []

    attrs: list[phase0_init.PrivacyAttribute] = []
    seen_local = set()
    for raw in raw_attrs:
        attr = _coerce_privacy_attribute(raw)
        if attr is None:
            continue
        if attr.key in seen_local:
            continue
        seen_local.add(attr.key)
        attrs.append(attr)
    return transcript_id, attrs


def generate_dynamic_attributes_per_transcript(
    records: dict[str, str],
    identifier_candidates_by_id: dict[str, list[dict]],
    base_attributes: list[phase0_init.PrivacyAttribute],
    model: str,
    max_workers: int,
    max_new_attributes_per_doc: int,
) -> dict[str, list[phase0_init.PrivacyAttribute]]:
    ordered_ids = list(records.keys())
    worker_count = max(1, min(max_workers, len(ordered_ids)))

    generated_by_id: dict[str, list[phase0_init.PrivacyAttribute]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _generate_attrs_for_transcript,
                transcript_id,
                records[transcript_id],
                identifier_candidates_by_id.get(transcript_id, []),
                base_attributes,
                model,
                max_new_attributes_per_doc,
            ): transcript_id
            for transcript_id in ordered_ids
        }
        for fut in as_completed(futures):
            transcript_id = futures[fut]
            try:
                tid, attrs = fut.result()
            except Exception as exc:
                print(
                    f"[attribute-gen] ERROR {transcript_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                tid, attrs = transcript_id, []

            selected = _filter_incremental_attributes(
                existing=base_attributes,
                candidates=attrs,
                limit=max_new_attributes_per_doc,
            )
            generated_by_id[tid] = selected
            print(f"[attribute-gen] {tid}: suggested={len(selected)}")

    for transcript_id in ordered_ids:
        generated_by_id.setdefault(transcript_id, [])
    return generated_by_id


def load_transcripts(input_path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            transcript_id = str(row.get("conversation_id", "")).strip()
            text = str(row.get("user_message", "")).strip()
            if transcript_id and text:
                records[transcript_id] = text
    return records


def load_rewritten_csv(input_path: Path) -> dict[str, str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Rewritten CSV not found: {input_path}")

    records: dict[str, str] = {}
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "transcript_id" not in fieldnames:
            raise ValueError(
                f"Rewritten CSV must contain 'transcript_id' column: {input_path}"
            )
        if "text" not in fieldnames:
            raise ValueError(f"Rewritten CSV must contain 'text' column: {input_path}")

        for row in reader:
            transcript_id = str(row.get("transcript_id", "")).strip()
            text = str(row.get("text", "")).strip()
            if transcript_id and text:
                records[transcript_id] = text
    return records


def _filter_records_by_ids(
    records: dict[str, str],
    ids_arg: str | None,
    source_label: str,
) -> dict[str, str]:
    if not ids_arg:
        return records

    requested_ids = [x.strip() for x in str(ids_arg).split(",") if x.strip()]
    if not requested_ids:
        raise RuntimeError("--ids was provided but no valid IDs were parsed.")

    missing = [doc_id for doc_id in requested_ids if doc_id not in records]
    if missing:
        print(f"WARNING: IDs not found in {source_label} and skipped: {missing}")

    filtered = {doc_id: records[doc_id] for doc_id in requested_ids if doc_id in records}
    if not filtered:
        raise RuntimeError(f"No valid transcript IDs matched --ids in {source_label}.")
    return filtered


def _reset_db_file() -> None:
    candidates = [
        cfg.DB_PATH,
        Path(str(cfg.DB_PATH) + "-wal"),
        Path(str(cfg.DB_PATH) + "-shm"),
    ]
    for path in candidates:
        if path.exists():
            path.unlink()
            print(f"Removed {path}")


def collect_rewritten_texts(expected_ids: list[str]) -> dict[str, str]:
    docs = db.get_all_documents()
    by_id = {str(doc.get("document_id", "")): doc for doc in docs}
    rewritten: dict[str, str] = {}
    missing: list[str] = []
    for transcript_id in expected_ids:
        doc = by_id.get(transcript_id) or {}
        final_text = str(doc.get("final_text", "")).strip()
        status = str(doc.get("status", "")).strip()
        if final_text and status == "success":
            rewritten[transcript_id] = final_text
        else:
            missing.append(transcript_id)
    if missing:
        raise RuntimeError(
            "Missing rewritten outputs for "
            f"{len(missing)} transcripts: {', '.join(missing)}"
        )
    return rewritten


def _apply_per_doc_total_cap(
    base_attributes: list[phase0_init.PrivacyAttribute],
    per_doc_new_attrs: dict[str, list[phase0_init.PrivacyAttribute]],
    max_total_attributes: int,
    target_ids: list[str] | None = None,
) -> None:
    max_total = max(0, int(max_total_attributes))
    max_dynamic = max(0, max_total - len(base_attributes))
    ids = list(target_ids) if target_ids is not None else list(per_doc_new_attrs.keys())
    for doc_id in ids:
        current = list(per_doc_new_attrs.get(doc_id, []))
        if len(current) <= max_dynamic:
            continue
        per_doc_new_attrs[doc_id] = current[:max_dynamic]
        print(
            f"[attribute-cap] {doc_id}: trimmed dynamic attrs "
            f"{len(current)} -> {max_dynamic} "
            f"(max_total={max_total}, base={len(base_attributes)})"
        )


def _sanitize_type_name(target_str: str, fallback_key: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", target_str.strip().lower()).strip("_")
    if not token:
        token = fallback_key.strip().lower()
    return token[:36]


def _build_aliases(attr: phase0_init.PrivacyAttribute) -> tuple[str, ...]:
    raw_values = [attr.display_name, attr.target_str, attr.key]
    aliases: list[str] = []
    seen = set()
    for raw in raw_values:
        text = _normalize_text(raw)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        aliases.append(text)
    return tuple(aliases)


def apply_attribute_scope(attributes: list[phase0_init.PrivacyAttribute]) -> None:
    scope = _merge_attribute_lists([], attributes)
    phase0_init.PRIVACY_ATTRIBUTES[:] = scope

    specs = list(BASE_MASKER_ATTRIBUTE_SPECS)
    used_keys = {spec.key for spec in specs}
    for attr in scope:
        if attr.key in used_keys:
            continue
        specs.append(
            AttributeSpec(
                key=attr.key,
                type_name=_sanitize_type_name(attr.target_str, attr.key),
                display_name=attr.display_name,
                target_attribute_str=attr.target_str,
                options=attr.options,
                special_note=attr.note,
                aliases=_build_aliases(attr),
            )
        )
        used_keys.add(attr.key)

    phase1_masker.ATTRIBUTE_SPECS[:] = specs


def _scope_for_doc(
    doc_id: str,
    base_attributes: list[phase0_init.PrivacyAttribute],
    per_doc_new_attrs: dict[str, list[phase0_init.PrivacyAttribute]],
) -> list[phase0_init.PrivacyAttribute]:
    return _merge_attribute_lists(base_attributes, per_doc_new_attrs.get(doc_id, []))


def run_pipeline_for_documents(
    document_ids: list[str],
    records: dict[str, str],
    base_attributes: list[phase0_init.PrivacyAttribute],
    per_doc_new_attrs: dict[str, list[phase0_init.PrivacyAttribute]],
) -> None:
    for doc_id in document_ids:
        scope = _scope_for_doc(doc_id, base_attributes, per_doc_new_attrs)
        apply_attribute_scope(scope)
        print(
            f"[pipeline] initialize {doc_id}: "
            f"attrs={len(scope)} (base={len(base_attributes)} + extra={len(per_doc_new_attrs.get(doc_id, []))})"
        )
        phase0_init.initialize_document(doc_id, records[doc_id])

    for doc_id in document_ids:
        scope = _scope_for_doc(doc_id, base_attributes, per_doc_new_attrs)
        apply_attribute_scope(scope)
        print(
            f"[pipeline] run-one {doc_id}: "
            f"attrs={len(scope)} (base={len(base_attributes)} + extra={len(per_doc_new_attrs.get(doc_id, []))})"
        )
        pipeline.run_one(doc_id)


def _candidate_confidence_rank(candidate: dict) -> int:
    conf = str(candidate.get("confidence", "")).strip().lower()
    return CONFIDENCE_RANK.get(conf, 0)


def _is_specific_identity_candidate(candidate: dict) -> bool:
    if not isinstance(candidate, dict):
        return False

    names = candidate.get("interviewee_candiates", [])
    if isinstance(names, list):
        for raw in names:
            text = _normalize_text(raw).lower()
            if text and not any(tok in text for tok in NON_SPECIFIC_LABEL_TOKENS):
                return True

    label = _normalize_text(candidate.get("identity_label", "")).lower()
    if label and not any(tok in label for tok in NON_SPECIFIC_LABEL_TOKENS):
        return True

    return False


def detect_reidentified_docs(
    reid_results_by_id: dict[str, list[dict]],
    confidence_threshold: str,
) -> dict[str, dict]:
    threshold_rank = CONFIDENCE_RANK.get(confidence_threshold.lower(), 3)
    flagged: dict[str, dict] = {}

    for doc_id, candidates in reid_results_by_id.items():
        if not isinstance(candidates, list):
            continue
        best_candidate = None
        best_rank = -1
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            rank = _candidate_confidence_rank(cand)
            if rank < threshold_rank:
                continue
            if not _is_specific_identity_candidate(cand):
                continue
            if rank > best_rank:
                best_rank = rank
                best_candidate = cand

        if best_candidate is not None:
            flagged[doc_id] = best_candidate

    return flagged


def generate_feedback_attributes(
    target_doc_ids: list[str],
    records: dict[str, str],
    rewritten_texts: dict[str, str],
    rewritten_reid: dict[str, list[dict]],
    base_attributes: list[phase0_init.PrivacyAttribute],
    per_doc_new_attrs: dict[str, list[phase0_init.PrivacyAttribute]],
    model: str,
    max_workers: int,
    max_new_attributes_per_doc: int,
) -> dict[str, list[phase0_init.PrivacyAttribute]]:
    worker_count = max(1, min(max_workers, len(target_doc_ids)))
    added_by_doc: dict[str, list[phase0_init.PrivacyAttribute]] = {}

    def _worker(doc_id: str) -> tuple[str, list[phase0_init.PrivacyAttribute]]:
        existing_scope = _scope_for_doc(doc_id, base_attributes, per_doc_new_attrs)
        transcript_text = rewritten_texts.get(doc_id) or records.get(doc_id, "")
        _, suggested = _generate_attrs_for_transcript(
            transcript_id=doc_id,
            transcript_text=transcript_text,
            identifier_candidates=rewritten_reid.get(doc_id, []),
            existing_attributes=existing_scope,
            model=model,
            prompt_max_attributes=max_new_attributes_per_doc,
        )
        added = _filter_incremental_attributes(
            existing=existing_scope,
            candidates=suggested,
            limit=max_new_attributes_per_doc,
        )
        return doc_id, added

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_worker, doc_id): doc_id for doc_id in target_doc_ids}
        for fut in as_completed(futures):
            doc_id = futures[fut]
            try:
                tid, added = fut.result()
            except Exception as exc:
                print(
                    f"[feedback] ERROR {doc_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                tid, added = doc_id, []

            if added:
                current = list(per_doc_new_attrs.get(tid, []))
                per_doc_new_attrs[tid] = _merge_attribute_lists(current, added)
                added_by_doc[tid] = added
            print(f"[feedback] {tid}: added={len(added)}")

    return added_by_doc


def build_attribute_export_payload(
    base_attributes: list[phase0_init.PrivacyAttribute],
    per_doc_new_attrs: dict[str, list[phase0_init.PrivacyAttribute]],
) -> dict:
    union = list(base_attributes)
    for doc_id in sorted(per_doc_new_attrs.keys()):
        union = _merge_attribute_lists(union, per_doc_new_attrs[doc_id])

    return {
        "base_count": len(base_attributes),
        "new_count": len(union) - len(base_attributes),
        "total_count": len(union),
        "attributes": [asdict(attr) for attr in union],
        "per_transcript_new_counts": {
            doc_id: len(attrs) for doc_id, attrs in sorted(per_doc_new_attrs.items())
        },
        "per_transcript_new_attributes": {
            doc_id: [asdict(attr) for attr in attrs]
            for doc_id, attrs in sorted(per_doc_new_attrs.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run dynamic privacy-attribute expansion and re-id evaluation."
    )
    parser.add_argument("--input", type=Path, default=cfg.INPUT_JSONL)
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=EXPORT_DIR,
        help="Directory where run artifacts are written.",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        default="nobranch",
        help="Filename prefix for exported artifacts.",
    )
    parser.add_argument(
        "--direct-intent-workers",
        type=int,
        default=6,
        help="Worker count for direct-intent web-search calls.",
    )
    parser.add_argument(
        "--attribute-workers",
        type=int,
        default=4,
        help="Worker count for dynamic attribute generation calls.",
    )
    parser.add_argument(
        "--pipeline-workers",
        type=int,
        default=cfg.RUN_ALL_MAX_WORKERS,
        help="Reserved for compatibility; pipeline runs per-doc due dynamic scopes.",
    )
    parser.add_argument(
        "--max-new-attributes",
        type=int,
        default=12,
        help="Maximum new attributes generated per transcript per round.",
    )
    parser.add_argument(
        "--max-total-attributes",
        type=int,
        default=12,
        help="Maximum total attributes per transcript (base + dynamic).",
    )
    parser.add_argument(
        "--feedback-rounds",
        type=int,
        default=1,
        help="Number of feedback rerun rounds for re-identified transcripts.",
    )
    parser.add_argument(
        "--reid-threshold",
        type=str,
        default="medium",
        choices=["very low", "low", "medium", "high", "very high"],
        help="Confidence threshold for treating a transcript as re-identified.",
    )
    parser.add_argument(
        "--direct-intent-model",
        type=str,
        default=DIRECT_INTENT_MODEL,
        help="Model used for direct-intent web-search calls.",
    )
    parser.add_argument(
        "--attribute-model",
        type=str,
        default=cfg.INIT_MODEL,
        help="Model used to generate additional privacy attributes.",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Delete existing no-branch DB before re-running.",
    )
    parser.add_argument(
        "--no-base-attributes",
        action="store_true",
        help="Do not use the predefined 8 base attributes; rely only on dynamic ones.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Optional path for the rewritten-transcripts CSV. "
            "Defaults to <export-dir>/<name-prefix>_rewritten.csv."
        ),
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Optional comma-separated transcript IDs to run.",
    )
    parser.add_argument(
        "--skip-reid",
        action="store_true",
        help="Run Steps 1-3 only and skip Step 4 re-id.",
    )
    parser.add_argument(
        "--reid-only",
        action="store_true",
        help=(
            "Run Step 4 only from the rewritten CSV "
            "(--output-csv or default <export-dir>/<name-prefix>_rewritten.csv)."
        ),
    )
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv(cfg.BASE_DIR / ".env", override=True)

    if not cfg.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in environment/.env")

    if args.reid_only and args.skip_reid:
        raise RuntimeError("--reid-only and --skip-reid cannot be used together.")

    (
        original_direct_intent_path,
        expanded_attributes_path,
        reid_output_path,
        default_rewritten_csv_path,
    ) = _resolve_export_paths(args.export_dir, args.name_prefix)
    rewritten_csv_path = args.output_csv or default_rewritten_csv_path
    print(
        f"[export] dir={args.export_dir} "
        f"prefix={args.name_prefix} "
        f"original={original_direct_intent_path.name} "
        f"attrs={expanded_attributes_path.name} "
        f"reid={reid_output_path.name} "
        f"csv={rewritten_csv_path}"
    )

    if args.reid_only:
        rewritten_texts = load_rewritten_csv(rewritten_csv_path)
        rewritten_texts = _filter_records_by_ids(
            rewritten_texts,
            args.ids,
            source_label=f"rewritten CSV {rewritten_csv_path}",
        )
        ordered_ids = list(rewritten_texts.keys())
        print(
            f"Loaded {len(rewritten_texts)} rewritten transcript(s) from {rewritten_csv_path}"
        )
        if args.feedback_rounds > 0:
            print("[reid-only] feedback rounds are skipped in re-id-only mode.")

        print("\n=== Step 4/4: Direct intent re-id on rewritten outputs (re-id-only) ===")
        rewritten_reid = run_direct_intent_parallel(
            text_by_id=rewritten_texts,
            output_path=reid_output_path,
            max_workers=args.direct_intent_workers,
            model=args.direct_intent_model,
            force_ids=set(ordered_ids),
        )
        if len(rewritten_reid) != len(ordered_ids):
            raise RuntimeError(
                "Expected re-id results for all rewritten transcripts, got "
                f"{len(rewritten_reid)} / {len(ordered_ids)}"
            )
        print(f"\nDone. Saved rewritten re-id results to {reid_output_path}")
        return 0

    records = load_transcripts(args.input)
    if not records:
        raise RuntimeError(f"No valid transcript records found in {args.input}")
    records = _filter_records_by_ids(
        records,
        args.ids,
        source_label=f"input transcripts ({args.input})",
    )
    ordered_ids = list(records.keys())
    print(f"Loaded {len(records)} transcripts from {args.input}")

    if args.no_base_attributes:
        base_attributes = []
    else:
        base_attributes = list(BASE_PRIVACY_ATTRIBUTES)
    print(f"[attributes] base_count={len(base_attributes)} no_base={args.no_base_attributes}")
    per_doc_new_attrs: dict[str, list[phase0_init.PrivacyAttribute]] = {
        doc_id: [] for doc_id in ordered_ids
    }

    # 1) direct intent on original transcripts
    print("\n=== Step 1/4: Direct intent on original transcripts ===")
    original_identifier_candidates = run_direct_intent_parallel(
        text_by_id=records,
        output_path=original_direct_intent_path,
        max_workers=args.direct_intent_workers,
        model=args.direct_intent_model,
    )

    # 2) generate per-transcript dynamic attributes using transcript+evidence
    print("\n=== Step 2/4: Generate dynamic privacy attributes ===")
    generated_attrs = generate_dynamic_attributes_per_transcript(
        records=records,
        identifier_candidates_by_id=original_identifier_candidates,
        base_attributes=base_attributes,
        model=args.attribute_model,
        max_workers=args.attribute_workers,
        max_new_attributes_per_doc=max(0, args.max_new_attributes),
    )
    for doc_id in ordered_ids:
        per_doc_new_attrs[doc_id] = generated_attrs.get(doc_id, [])
    _apply_per_doc_total_cap(
        base_attributes=base_attributes,
        per_doc_new_attrs=per_doc_new_attrs,
        max_total_attributes=args.max_total_attributes,
        target_ids=ordered_ids,
    )

    export_payload = build_attribute_export_payload(base_attributes, per_doc_new_attrs)
    _write_json(expanded_attributes_path, export_payload)
    print(
        "[attribute-gen] saved expanded attributes to "
        f"{expanded_attributes_path} "
        f"(union={export_payload['total_count']}, new={export_payload['new_count']})"
    )

    # 3) run no-branch pipeline with per-document attribute scopes
    print("\n=== Step 3/4: Run no-branch pipeline with expanded attributes ===")
    if args.reset_db:
        _reset_db_file()
    db.init_db()
    run_pipeline_for_documents(
        document_ids=ordered_ids,
        records=records,
        base_attributes=base_attributes,
        per_doc_new_attrs=per_doc_new_attrs,
    )
    rewritten_texts = collect_rewritten_texts(expected_ids=ordered_ids)
    print(f"[pipeline] rewritten outputs ready: {len(rewritten_texts)}")

    _write_rewritten_csv(rewritten_csv_path, ordered_ids, rewritten_texts)
    print(f"[pipeline] saved rewritten csv to {rewritten_csv_path}")

    if args.skip_reid:
        print("[step4] skipped direct-intent re-id on rewritten outputs (--skip-reid).")
        return 0

    # 4) direct intent re-id + feedback loop
    print("\n=== Step 4/4: Direct intent re-id on rewritten outputs ===")
    rewritten_reid = run_direct_intent_parallel(
        text_by_id=rewritten_texts,
        output_path=reid_output_path,
        max_workers=args.direct_intent_workers,
        model=args.direct_intent_model,
        force_ids=set(ordered_ids),
    )

    for round_idx in range(1, max(0, args.feedback_rounds) + 1):
        flagged = detect_reidentified_docs(
            reid_results_by_id=rewritten_reid,
            confidence_threshold=args.reid_threshold,
        )
        if not flagged:
            print(f"[feedback] round={round_idx}: no re-identified transcripts above threshold.")
            break

        flagged_ids = sorted(flagged.keys())
        print(
            f"[feedback] round={round_idx}: "
            f"re-identified={len(flagged_ids)} -> {', '.join(flagged_ids)}"
        )
        added_attrs = generate_feedback_attributes(
            target_doc_ids=flagged_ids,
            records=records,
            rewritten_texts=rewritten_texts,
            rewritten_reid=rewritten_reid,
            base_attributes=base_attributes,
            per_doc_new_attrs=per_doc_new_attrs,
            model=args.attribute_model,
            max_workers=args.attribute_workers,
            max_new_attributes_per_doc=max(0, args.max_new_attributes),
        )
        if not added_attrs:
            print("[feedback] no additional targeted attributes were generated; stopping.")
            break

        _apply_per_doc_total_cap(
            base_attributes=base_attributes,
            per_doc_new_attrs=per_doc_new_attrs,
            max_total_attributes=args.max_total_attributes,
            target_ids=flagged_ids,
        )
        rerun_ids = sorted(added_attrs.keys())
        print(f"[feedback] rerunning {len(rerun_ids)} transcript(s): {', '.join(rerun_ids)}")
        run_pipeline_for_documents(
            document_ids=rerun_ids,
            records=records,
            base_attributes=base_attributes,
            per_doc_new_attrs=per_doc_new_attrs,
        )
        rewritten_texts = collect_rewritten_texts(expected_ids=ordered_ids)
        _write_rewritten_csv(rewritten_csv_path, ordered_ids, rewritten_texts)
        print(
            f"[feedback] round={round_idx}: updated rewritten csv at {rewritten_csv_path}"
        )
        rewritten_reid = run_direct_intent_parallel(
            text_by_id=rewritten_texts,
            output_path=reid_output_path,
            max_workers=args.direct_intent_workers,
            model=args.direct_intent_model,
            force_ids=set(rerun_ids),
        )

        export_payload = build_attribute_export_payload(base_attributes, per_doc_new_attrs)
        _write_json(expanded_attributes_path, export_payload)
        print(
            "[feedback] updated attributes saved to "
            f"{expanded_attributes_path} "
            f"(union={export_payload['total_count']}, new={export_payload['new_count']})"
        )

    final_flagged = detect_reidentified_docs(
        reid_results_by_id=rewritten_reid,
        confidence_threshold=args.reid_threshold,
    )
    if final_flagged:
        print(
            "[feedback] unresolved re-identified transcripts: "
            + ", ".join(sorted(final_flagged.keys()))
        )
    else:
        print("[feedback] all transcripts below re-identification threshold.")

    if len(rewritten_reid) != len(records):
        raise RuntimeError(
            "Expected re-id results for all transcripts, got "
            f"{len(rewritten_reid)} / {len(records)}"
        )

    print(f"\nDone. Saved rewritten re-id results to {reid_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
