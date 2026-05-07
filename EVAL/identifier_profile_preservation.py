#!/usr/bin/env python3
"""Profile analysis helpers for re-id candidates and transcript recoverability.

This script currently supports two workflows:

1. `reid_compare`
   Compare re-identification JSON outputs by extracting atomic profile facts
   from identifier candidates.

2. `profile_recoverability`
   Generate 8 attribute summaries from the original transcript, decompose them
   into a completed profile, and evaluate whether each fact is recoverable from
   each transcript config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _compat import (  # noqa: E402
    ATTRIBUTE_SPECS,
    CFG_DISPLAY as AP_CFG_DISPLAY,
    CFG_ORDER as AP_CFG_ORDER,
    maybe_load_csv_map as ap_maybe_load_csv_map,
    maybe_load_csv_map_flexible as ap_maybe_load_csv_map_flexible,
)


DEFAULT_MODEL = "gpt-4.1"


class _GeminiChatMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _GeminiChatChoice:
    def __init__(self, content: str) -> None:
        self.message = _GeminiChatMessage(content)


class _GeminiChatResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_GeminiChatChoice(content)]


class _GeminiChatCompletions:
    def __init__(self, client: Any) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> _GeminiChatResponse:
        from google.genai import types as _genai_types

        model = kwargs["model"]
        messages = kwargs.get("messages", [])
        system_text = ""
        user_parts: list[str] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                system_text = content
            else:
                user_parts.append(content)
        contents = "\n\n".join(part for part in user_parts if part)
        config_kwargs: dict[str, Any] = {}
        if system_text:
            config_kwargs["system_instruction"] = system_text
        if "temperature" in kwargs:
            config_kwargs["temperature"] = kwargs["temperature"]
        if "max_completion_tokens" in kwargs:
            config_kwargs["max_output_tokens"] = kwargs["max_completion_tokens"]
        response_format = kwargs.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            config_kwargs["response_mime_type"] = "application/json"
        config = _genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        text = getattr(response, "text", None) or ""
        return _GeminiChatResponse(text)


class _GeminiChatNamespace:
    def __init__(self, client: Any) -> None:
        self.completions = _GeminiChatCompletions(client)


class _GeminiChatClient:
    def __init__(self, api_key: str) -> None:
        from google import genai as _genai

        self._client = _genai.Client(api_key=api_key)
        self.chat = _GeminiChatNamespace(self._client)


def _resolve_api_key(model: str) -> str:
    if model.lower().startswith("gemini"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        return api_key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return api_key


def _create_chat_client(model: str, api_key: str | None) -> Any:
    if model.lower().startswith("gemini"):
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        return _GeminiChatClient(api_key)
    return OpenAI(api_key=api_key)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "identifier_profile_preservation_results"
DEFAULT_PROFILE_ROOT = DEFAULT_OUTPUT_ROOT / "profile"
DEFAULT_REFERENCE_CSV: Path | None = None  # supply via --original-csv
MAX_RETRIES = 3
CACHE_SCHEMA_VERSION = "profile_recoverability_v10_strict_duplicate_merge"

CONFIDENCE_RANK = {
    "very low": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "very high": 5,
}

SOURCE_FIELDS = {"unified_description"}
ATTRIBUTE_SUMMARY_SOURCE_FIELD = "attribute_summary"
FACT_OVERLAP_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "and",
    "or",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "at",
    "from",
    "use",
    "ai",
    "tool",
    "work",
    "publish",
    "sell",
}

EXTRACT_SYSTEM_PROMPT = (
    "You are a careful privacy-analysis assistant. "
    "You extract atomic profile facts from re-identification candidate summaries. "
    "Use only the provided unified candidate descriptions and candidate confidence. "
    "Do not use outside knowledge, web search, or unsupported assumptions. "
    "Return valid JSON only."
)

MERGE_SYSTEM_PROMPT = (
    "You merge multiple partial descriptions of the same re-identification candidate into one "
    "unified description. Use only the provided text. Remove redundancy, preserve useful detail, "
    "keep uncertainty markers when supported, and do not add outside knowledge. Return valid JSON only."
)

ATTRIBUTE_SUMMARY_SYSTEM_PROMPT = (
    "You are a careful qualitative researcher. "
    "Given an interview transcript, produce one concise attribute summary "
    "for the participant using only evidence from the transcript. "
    "When the attribute is supported, prefer a richer and more informative summary "
    "rather than a minimal label. "
    "Do not use outside knowledge. Return valid JSON only."
)

ATTRIBUTE_SUMMARY_SANITY_SYSTEM_PROMPT = (
    "You are a careful qualitative researcher cleaning an attribute summary. "
    "Keep only the information that truly belongs to the requested attribute and "
    "remove anything that belongs to the other attributes. "
    "Do not add new information. Return valid JSON only."
)

ATTRIBUTE_FACT_SYSTEM_PROMPT = (
    "You are a careful privacy-analysis assistant. "
    "Given one attribute summary description, break it into distinct atomic facts "
    "for that attribute only. Use only the provided summary description. "
    "Do not use outside knowledge. Return valid JSON only."
)

RECOVERABILITY_SYSTEM_PROMPT = (
    "You are a careful qualitative researcher. "
    "Judge whether a specific profile fact can be recovered from a transcript. "
    "Use only the transcript text. Do not use outside knowledge. "
    "Return valid JSON only."
)

FACT_SANITY_MERGE_SYSTEM_PROMPT = (
    "You are a careful qualitative researcher performing a sanity check on a completed profile. "
    "Merge only truly duplicated or near-duplicated facts that express the same recoverable proposition. "
    "Do not add new information. Return valid JSON only."
)

RECOVERABILITY_DECISIONS = (
    "Yes, the answer can be recovered from the transcript with clear evidence",
    "No, the answer cannot be recovered because the transcripts contain ambigious information and not specific enough to answer the question",
    "I am not sure",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_dataset_id(value: str) -> str:
    return normalize_text(value).lower()


def canonical_fact_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalize_text(text).lower()).strip()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "run"


def parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def confidence_rank(value: str) -> int:
    return CONFIDENCE_RANK.get(normalize_text(value).lower(), 0)


def confidence_weight(value: str) -> float:
    rank = confidence_rank(value)
    return round(rank / 5.0, 3) if rank else 0.0


def parse_input_spec(raw: str) -> tuple[str, Path]:
    label: str
    path_str: str
    if "=" in raw:
        label, path_str = raw.split("=", 1)
        label = slugify(label)
    else:
        path_str = raw
        label = slugify(Path(path_str).stem)
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return label, path.resolve()


def load_reid_rows(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    rows: list[Any]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        maybe_rows = payload.get("results") or payload.get("rows") or payload.get("data")
        if not isinstance(maybe_rows, list):
            raise ValueError(f"Could not find row list in {path}")
        rows = maybe_rows
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        transcript_id = normalize_text(row.get("transcript_id", ""))
        if not transcript_id:
            continue
        normalized_id = normalize_dataset_id(transcript_id)
        identifier_candidates = row.get("identifier_candidates", [])
        out[normalized_id] = {
            "transcript_id": transcript_id,
            "identifier_candidates": identifier_candidates if isinstance(identifier_candidates, list) else [],
        }
    return out


def sanitize_candidate(raw_candidate: dict[str, Any], display_index: int) -> dict[str, Any] | None:
    names: list[str] = []
    raw_names = raw_candidate.get("interviewee_candiates", [])
    if isinstance(raw_names, list):
        for raw_name in raw_names:
            name = normalize_text(raw_name)
            if name and name not in names:
                names.append(name)

    label = normalize_text(raw_candidate.get("identity_label", ""))
    confidence = normalize_text(raw_candidate.get("confidence", "")).lower()
    if not names and not label:
        return None

    return {
        "candidate_index": display_index,
        "interviewee_candiates": names,
        "identity_label": label,
        "confidence": confidence,
    }


def candidate_description_texts(candidate: dict[str, Any]) -> list[str]:
    descriptions: list[str] = []
    for raw_text in candidate.get("interviewee_candiates", []):
        text = normalize_text(raw_text)
        if text and text not in descriptions:
            descriptions.append(text)
    label = normalize_text(candidate.get("identity_label", ""))
    if label and label not in descriptions:
        descriptions.append(label)
    return descriptions


def select_identifier_candidates(
    raw_candidates: list[Any],
    *,
    max_candidates: int | None,
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for raw_candidate in raw_candidates:
        if not isinstance(raw_candidate, dict):
            continue
        provisional = sanitize_candidate(raw_candidate, len(cleaned) + 1)
        if provisional is not None:
            cleaned.append(provisional)

    cleaned.sort(
        key=lambda item: (
            confidence_rank(item.get("confidence", "")),
            1 if item.get("identity_label") else 0,
            len(item.get("interviewee_candiates", [])),
        ),
        reverse=True,
    )

    if max_candidates is not None and max_candidates > 0:
        cleaned = cleaned[:max_candidates]

    normalized: list[dict[str, Any]] = []
    for idx, candidate in enumerate(cleaned, start=1):
        normalized.append({
            "candidate_index": idx,
            "interviewee_candiates": list(candidate.get("interviewee_candiates", [])),
            "identity_label": normalize_text(candidate.get("identity_label", "")),
            "confidence": normalize_text(candidate.get("confidence", "")).lower(),
        })
    return normalized


def format_attribute_spec(spec: Any) -> str:
    lines = [
        f"- {spec.key}: display_name={spec.display_name}, type_name={spec.type_name}, "
        f'target="{spec.target_attribute_str}"',
    ]
    if spec.aliases:
        lines.append(f'  aliases={", ".join(spec.aliases)}')
    if spec.options:
        lines.append(f'  options={", ".join(spec.options)}')
    if spec.special_note:
        lines.append(f"  note={spec.special_note}")
    return "\n".join(lines)


def build_merge_prompt(candidate: dict[str, Any]) -> str:
    descriptions = []
    for idx, text in enumerate(candidate.get("interviewee_candiates", []), start=1):
        normalized = normalize_text(text)
        if normalized:
            descriptions.append({
                "source": f"interviewee_candiates[{idx}]",
                "description": normalized,
            })

    label = normalize_text(candidate.get("identity_label", ""))
    if label:
        descriptions.append({
            "source": "identity_label",
            "description": label,
        })

    descriptions_json = json.dumps(descriptions, ensure_ascii=False, indent=2)
    return (
        "Here are descriptions about the same candidate. Generate a unified description about this candidate.\n\n"
        "Rules:\n"
        "- Merge overlapping details into one coherent description.\n"
        "- Preserve useful specific details that do not conflict.\n"
        "- Keep uncertainty words such as anonymous, unidentified, likely, or exact identity not disclosed when supported.\n"
        "- Remove redundancy.\n"
        "- Do not add outside knowledge.\n"
        "- If descriptions differ in specificity, prefer the more informative wording when compatible.\n"
        "- If descriptions conflict, reflect uncertainty instead of inventing a resolution.\n"
        "- Return a single unified description only.\n\n"
        f"Descriptions:\n{descriptions_json}\n\n"
        'Return JSON exactly: {"unified_description":"..."}\n'
    )


def merge_candidate_descriptions(
    client: OpenAI,
    *,
    model: str,
    candidate: dict[str, Any],
) -> str:
    descriptions = candidate_description_texts(candidate)
    if not descriptions:
        return ""
    if len(descriptions) == 1:
        return descriptions[0]

    prompt = build_merge_prompt(candidate)
    parsed = call_json(
        client,
        model=model,
        system_prompt=MERGE_SYSTEM_PROMPT,
        user_prompt=prompt,
    )
    unified = normalize_text(parsed.get("unified_description", ""))
    if unified:
        return unified
    return max(descriptions, key=len)


def build_unified_candidates(
    client: OpenAI,
    *,
    model: str,
    selected_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unified_candidates: list[dict[str, Any]] = []
    for candidate in selected_candidates:
        unified_candidates.append({
            **candidate,
            "unified_description": merge_candidate_descriptions(
                client,
                model=model,
                candidate=candidate,
            ),
        })
    return unified_candidates


def build_extraction_prompt(
    *,
    sample_id: str,
    config_label: str,
    unified_candidates: list[dict[str, Any]],
) -> str:
    attr_block = "\n".join(format_attribute_spec(spec) for spec in ATTRIBUTE_SPECS)
    candidate_view = [
        {
            "candidate_index": candidate["candidate_index"],
            "unified_description": normalize_text(candidate.get("unified_description", "")),
            "confidence": normalize_text(candidate.get("confidence", "")),
        }
        for candidate in unified_candidates
    ]
    candidates_json = json.dumps(candidate_view, ensure_ascii=False, indent=2)
    output_shape = ",\n".join(
        f'  "{spec.key}": [{{"fact": "...", "supporting_candidates": [1], '
        f'"source_fields": ["unified_description"]}}]'
        for spec in ATTRIBUTE_SPECS
    )

    return (
        f"Analyze identifier-candidate profile information for sample `{sample_id}` "
        f"from config `{config_label}`.\n\n"
        "Each `unified_description` below was merged from that candidate's "
        "`interviewee_candiates` and `identity_label`.\n\n"
        "Only use the fields shown below:\n"
        "- `unified_description`\n"
        "- `confidence`\n\n"
        "Task:\n"
        "For each of the 8 attributes, extract distinct atomic facts that are "
        "explicitly stated or tightly implied by the provided candidate text.\n\n"
        "Rules:\n"
        "1. Use only the provided input. Ignore every other field that may exist in the original JSON.\n"
        "2. An atomic fact must be as small and non-overlapping as possible.\n"
        "3. Be conservative. Do not use outside knowledge.\n"
        "4. A unified description alone still does NOT justify unsupported age, sex, location, education, relationship, income, or place of birth inferences.\n"
        "5. If an attribute is unsupported, return an empty list for that attribute.\n"
        "6. If the same fact appears in multiple candidates, output it once and list all supporting candidate indices.\n"
        "7. `source_fields` must only contain `unified_description`.\n"
        "8. Relationship, income, and place of birth need direct evidence; do not infer them from profession.\n"
        "9. If a phrase like `early-career researcher` appears, you may include a cautious age/life-stage fact, "
        "but do not convert it into a precise age number.\n"
        "10. Keep occupation facts separate from education facts whenever possible.\n"
        "11. Split compound occupation/activity clauses into multiple facts when they contain multiple work aspects "
        "(role, task, material/domain, tool use, or purpose).\n"
        "12. Example: from `chemist or chemical engineer working on catalytic depolymerization of complex "
        "aromatic/ether-linked organic materials, using AI tools for kinetics modeling and literature reviews`, "
        "good OCCP facts include `chemist or chemical engineer`, `works on catalytic depolymerization`, "
        "`works on complex aromatic/ether-linked organic materials`, `uses AI tools for kinetics modeling`, "
        "and `uses AI tools for literature reviews`.\n"
        "13. Do NOT output overlapping facts where one is just a broader or less specific version of another.\n"
        "14. If two facts overlap, keep only the most specific one and drop the broader one.\n"
        "15. Bad example: do not keep both `publishes art books` and `self-published art books`; keep only "
        "`self-published art books`.\n"
        "16. Bad example: do not keep both `works on catalytic depolymerization of complex aromatic materials` "
        "and `works on catalytic depolymerization` if the first is not further decomposed; instead decompose and "
        "keep only the non-overlapping smaller facts.\n\n"
        f"Attributes:\n{attr_block}\n\n"
        f"Input candidates:\n{candidates_json}\n\n"
        "Return JSON exactly as an object with these 8 keys and no extra top-level keys:\n"
        "{\n"
        f"{output_shape}\n"
        "}\n"
    )


def _chat_completion_request(model: str, user_prompt: str) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 4096,
    }
    if not model.lower().startswith("gpt-5"):
        request["temperature"] = 0
    return request


def call_json(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = _chat_completion_request(model, user_prompt)
            request["messages"] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            resp = client.chat.completions.create(**request)
            raw = resp.choices[0].message.content or "{}"
            parsed = parse_json_object(raw)
            if not parsed:
                raise ValueError("Model returned empty or invalid JSON")
            return parsed
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(float(min(2 * attempt, 8)))
    raise RuntimeError(f"OpenAI request failed after {retries} attempts: {last_error}")


def hash_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_cached_result(cache_path: Path, cache_key: str) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if cached.get("cache_key") != cache_key:
        return None
    result = cached.get("result")
    return result if isinstance(result, dict) else None


def write_cached_result(
    cache_path: Path,
    *,
    cache_key: str,
    metadata: dict[str, Any],
    result: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cache_key": cache_key,
                **metadata,
                "result": result,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def evidence_quote_supported(transcript: str, evidence_quote: str) -> bool:
    quote = normalize_for_match(evidence_quote).strip("\"' ")
    if len(quote) < 8:
        return False
    return quote in normalize_for_match(transcript)


def format_attribute_note(spec: Any) -> str:
    parts = [f"Attribute key: {spec.key}", f"Attribute label: {spec.display_name}"]
    if spec.target_attribute_str:
        parts.append(f'Target meaning: "{spec.target_attribute_str}"')
    if spec.options:
        parts.append("Allowed coarse options: " + ", ".join(spec.options))
    if spec.aliases:
        parts.append("Aliases: " + ", ".join(spec.aliases))
    if spec.special_note:
        parts.append("Special note: " + spec.special_note)
    return "\n".join(f"- {part}" for part in parts)


def format_attribute_scope_guidance(spec: Any) -> str:
    guidance_map = {
        "AGE": [
            "Summarize only age, age range, or life-stage information.",
            "Do not include occupation, education, location, institution, project topic, relationship status, income, or birthplace.",
            "Only use direct age statements or very strong life-stage cues such as early-career, student, or retired when explicitly supported.",
        ],
        "SEX": [
            "Summarize only sex or gender information.",
            "Do not include age, occupation, education, location, relationship status, income, or birthplace.",
            "Use only direct evidence; do not infer sex or gender from name, writing style, or profession.",
        ],
        "LOC": [
            "Summarize only the participant's current place of living or current geographic base in geographic terms such as city, region, or country.",
            "Do not include employer, university, institution, occupation, project topic, technical method, or language context unless they directly state the geographic place itself.",
            "Do not infer current place of living from where a project happened, where someone studied, where they worked temporarily, or what language appeared in their environment.",
            "If the transcript does not directly support current place of living, use the unknown fallback.",
        ],
        "OCCP": [
            "Summarize only occupation: role, specialty, domain, tasks, methods, outputs, or professional affiliation that directly refines the role.",
            "Do not include current place of living, age, education level, relationship status, income, or birthplace.",
            "You may include an employer or institution only when it directly clarifies the occupational identity, not as location evidence.",
        ],
        "EDU": [
            "Summarize only level of education or training stage, such as in college, college degree, PhD, postdoc, or other explicit training status.",
            "Do not include occupation, employer, institution of work, project topic, methods, location, relationship status, income, or birthplace unless they directly indicate education level.",
            "Do not infer education level just because the participant is skilled, technical, or works at a university.",
            "If the transcript does not directly support education or training level, use the unknown fallback.",
        ],
        "REL": [
            "Summarize only current relationship or marital status.",
            "Do not include age, occupation, education, location, income, or birthplace.",
            "Use only direct evidence such as spouse, partner, married, divorced, or similar explicit cues.",
        ],
        "INC": [
            "Summarize only yearly income or income status.",
            "Do not include occupation, employer prestige, education, location, or lifestyle details as proxies for income.",
            "Use only direct income evidence; otherwise use the unknown fallback.",
        ],
        "POBP": [
            "Summarize only place of birth in geographic terms.",
            "Do not include current place of living, employer, school, occupation, or ancestry unless the transcript explicitly states birthplace.",
            "If birthplace is not directly stated, use the unknown fallback.",
        ],
    }
    lines = guidance_map.get(spec.key, [])
    return "\n".join(f"- {line}" for line in lines)


def build_attribute_summary_prompt(
    *,
    transcript_id: str,
    transcript_text: str,
    spec: Any,
) -> str:
    unknown_phrase = (
        f"Unknown {spec.target_attribute_str}; the transcript does not provide enough evidence."
    )
    return (
        f"Read the interview transcript and summarize only the participant's {spec.display_name}.\n\n"
        f"{format_attribute_note(spec)}\n\n"
        f"Attribute boundary rules:\n{format_attribute_scope_guidance(spec)}\n\n"
        "Rules:\n"
        "- Produce exactly one description for this single attribute.\n"
        "- Use only evidence from the transcript.\n"
        "- When supported, make the description longer and more detailed than a bare label.\n"
        "- Prefer a compact but information-dense noun phrase or one sentence fragment of about 12-35 words.\n"
        "- Detailed does not mean broad: include only specifics that truly belong to this attribute and exclude content that belongs to the other 7 attributes.\n"
        "- Preserve uncertainty markers like likely, possible, unclear, or affiliated with when the evidence is partial.\n"
        "- Do not overclaim or turn weak hints into definite statements.\n"
        "- If the strongest evidence mainly supports a different attribute, do not use it here; use the unknown fallback instead.\n"
        "- Do not use generic outputs like `College Degree` when a more detailed evidence-grounded summary is possible; but if the transcript does not directly support the attribute, use the unknown fallback instead of guessing.\n"
        f'- If the attribute is missing, ambiguous, or unsupported, set `description` to: "{unknown_phrase}"\n'
        "- Set `supported` to false when using the unknown fallback, otherwise true.\n"
        "- If supported, provide one exact evidence quote from the transcript.\n"
        "- If unsupported, `evidence_quote` must be an empty string.\n\n"
        "Example style for a supported attribute summary:\n"
        "- `Astrophysicist and numerical modeler of colliding radiative plasma flows (University of Rochester), first author of the cooling-and-instabilities simulation series`\n\n"
        f"Transcript ID: {transcript_id}\n\n"
        f"=== TRANSCRIPT ===\n{transcript_text}\n=== END TRANSCRIPT ===\n\n"
        'Return JSON exactly: {"description":"...","supported":true,"evidence_quote":"..."}\n'
    )


def unknown_attribute_description(spec: Any) -> str:
    return f"Unknown {spec.target_attribute_str}; the transcript does not provide enough evidence."


def build_attribute_summary_sanity_prompt(
    *,
    spec: Any,
    summary_description: str,
) -> str:
    unknown_phrase = unknown_attribute_description(spec)
    return (
        f"Clean the following {spec.display_name} summary so it contains only {spec.display_name} information.\n\n"
        f"{format_attribute_note(spec)}\n\n"
        f"Attribute boundary rules:\n{format_attribute_scope_guidance(spec)}\n\n"
        "Rules:\n"
        "- Remove any content that belongs more naturally to another attribute.\n"
        "- Keep detail only if it directly refines this attribute.\n"
        "- Do not add new information.\n"
        "- Keep the cleaned summary detailed when valid information remains.\n"
        f'- If no valid attribute-specific information remains, set `description` to: "{unknown_phrase}" and set `supported` to false.\n'
        "- Otherwise set `supported` to true.\n\n"
        f'Original summary: "{summary_description}"\n\n'
        'Return JSON exactly: {"description":"...","supported":true}\n'
    )


def build_attribute_fact_prompt(
    *,
    spec: Any,
    summary_description: str,
) -> str:
    return (
        f"Break the following {spec.display_name} summary into non-overlapping atomic facts.\n\n"
        f"{format_attribute_note(spec)}\n\n"
        f"Attribute boundary rules:\n{format_attribute_scope_guidance(spec)}\n\n"
        "Rules:\n"
        "- Use only the summary description.\n"
        "- Return facts for this attribute only.\n"
        "- Facts must be distinct, atomic, and non-overlapping.\n"
        "- If the summary contains mixed information, keep only the parts that actually belong to this attribute and discard the rest.\n"
        "- If the summary says the attribute is unknown, unsupported, or lacks evidence, return an empty list.\n"
        "- Do not add outside knowledge.\n\n"
        f'Summary description: "{summary_description}"\n\n'
        'Return JSON exactly: {"facts":["..."]}\n'
    )


def build_recoverability_prompt(
    *,
    transcript_id: str,
    config_label: str,
    transcript_text: str,
    fact_item: dict[str, Any],
) -> str:
    decisions = "\n".join(f"- {item}" for item in RECOVERABILITY_DECISIONS)
    return (
        f"Determine whether the profile fact below can be recovered from the transcript for config `{config_label}`.\n\n"
        f"Transcript ID: {transcript_id}\n"
        f"Attribute: {fact_item['attribute_key']} ({fact_item['attribute_display_name']})\n"
        f'Attribute summary: "{fact_item["summary_description"]}"\n'
        f'Reference fact: "{fact_item["fact"]}"\n\n'
        "Allowed decisions (must match exactly one of these strings):\n"
        f"{decisions}\n\n"
        "Rules:\n"
        "- Judge recoverability from the transcript alone.\n"
        "- Choose `Yes...` only when the transcript contains clear evidence for the fact.\n"
        "- Choose `No...` when the transcript is ambiguous, missing, or not specific enough.\n"
        "- Choose `I am not sure` only when the transcript hints at the fact but the evidence quality is genuinely unclear.\n"
        "- If the decision is Yes, provide one exact evidence quote from the transcript.\n"
        "- If the decision is No or I am not sure, `evidence_quote` may be empty.\n\n"
        f"=== TRANSCRIPT ===\n{transcript_text}\n=== END TRANSCRIPT ===\n\n"
        'Return JSON exactly: {"decision":"...","reasoning":"...","evidence_quote":"..."}\n'
    )


def build_fact_sanity_merge_prompt(
    *,
    transcript_id: str,
    completed_profile: list[dict[str, Any]],
) -> str:
    fact_view = [
        {
            "fact_id": item["fact_id"],
            "attribute_key": item["attribute_key"],
            "attribute_display_name": item["attribute_display_name"],
            "summary_description": item["summary_description"],
            "fact": item["fact"],
        }
        for item in completed_profile
    ]
    facts_json = json.dumps(fact_view, ensure_ascii=False, indent=2)
    return (
        f"Sanity-check the completed profile facts for transcript `{transcript_id}`.\n\n"
        "Task:\n"
        "- Merge facts only when they are true duplicates or near-duplicates expressing the same recoverable proposition.\n"
        "- Keep related but distinct facts separate.\n"
        "- Prefer the clearest and most specific wording among duplicates.\n"
        "- Do not invent new facts.\n"
        "- If two facts come from different attributes but say the same thing, merge them and choose the best primary attribute.\n"
        "- Every original `fact_id` must appear exactly once across the output groups.\n\n"
        "Completed profile facts:\n"
        f"{facts_json}\n\n"
        "Return JSON exactly:\n"
        "{\n"
        '  "merged_facts": [\n'
        '    {"fact": "...", "primary_attribute_key": "OCCP", "source_fact_ids": ["OCCP_01", "EDU_02"]}\n'
        "  ]\n"
        "}\n"
    )


def make_reference_fact_item(
    *,
    fact: str,
    source_fields: list[str],
) -> dict[str, Any]:
    return {
        "fact": fact,
        "source_fields": source_fields,
        "supporting_candidates": [],
        "supporting_candidate_confidences": [],
        "confidence_weight": 0.0,
    }


def postprocess_attribute_summary_facts(
    *,
    spec: Any,
    parsed: dict[str, Any],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    raw_items = parsed.get("facts", [])
    if not isinstance(raw_items, list):
        raw_items = []

    for raw_item in raw_items:
        fact = normalize_text(raw_item.get("fact", "")) if isinstance(raw_item, dict) else normalize_text(raw_item)
        if not fact:
            continue
        for fact_variant in expand_atomic_fact(spec.key, fact):
            fact_key = canonical_fact_key(fact_variant)
            if not fact_key:
                continue
            merged.setdefault(
                fact_key,
                make_reference_fact_item(
                    fact=fact_variant,
                    source_fields=[ATTRIBUTE_SUMMARY_SOURCE_FIELD],
                ),
            )

    finalized = dedupe_overlapping_facts(list(merged.values()))
    cleaned: list[dict[str, Any]] = []
    for item in finalized:
        cleaned.append({
            "fact": item["fact"],
            "source_fields": item["source_fields"],
        })
    return cleaned


def generate_attribute_summary_with_cache(
    *,
    api_key: str,
    model: str,
    transcript_id: str,
    transcript_text: str,
    spec: Any,
    cache_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    cache_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "stage": "attribute_summary",
        "model": model,
        "transcript_id": transcript_id,
        "attribute_key": spec.key,
        "transcript_text": transcript_text,
    }
    cache_key = hash_payload(cache_payload)
    cache_path = cache_root / "attribute_summaries" / slugify(transcript_id) / f"{spec.key}.json"
    if not overwrite:
        cached = load_cached_result(cache_path, cache_key)
        if cached is not None:
            return cached

    client = _create_chat_client(model, api_key)
    parsed = call_json(
        client,
        model=model,
        system_prompt=ATTRIBUTE_SUMMARY_SYSTEM_PROMPT,
        user_prompt=build_attribute_summary_prompt(
            transcript_id=transcript_id,
            transcript_text=transcript_text,
            spec=spec,
        ),
    )
    description = normalize_text(parsed.get("description", ""))
    supported = bool(parsed.get("supported", False))
    evidence_quote = normalize_text(parsed.get("evidence_quote", ""))
    if not description:
        description = unknown_attribute_description(spec)
        supported = False
    elif supported:
        sanitized = call_json(
            client,
            model=model,
            system_prompt=ATTRIBUTE_SUMMARY_SANITY_SYSTEM_PROMPT,
            user_prompt=build_attribute_summary_sanity_prompt(
                spec=spec,
                summary_description=description,
            ),
        )
        description = normalize_text(sanitized.get("description", "")) or unknown_attribute_description(spec)
        supported = bool(sanitized.get("supported", False)) and description != unknown_attribute_description(spec)
        if not supported:
            evidence_quote = ""
    if supported and evidence_quote and not evidence_quote_supported(transcript_text, evidence_quote):
        evidence_quote = ""
    result = {
        "attribute_key": spec.key,
        "attribute_display_name": spec.display_name,
        "target_attribute": spec.target_attribute_str,
        "description": description,
        "supported": supported,
        "evidence_quote": evidence_quote,
    }
    write_cached_result(
        cache_path,
        cache_key=cache_key,
        metadata={"model": model, "stage": "attribute_summary"},
        result=result,
    )
    return result


def extract_attribute_profile_with_cache(
    *,
    api_key: str,
    model: str,
    transcript_id: str,
    spec: Any,
    attribute_summary: dict[str, Any],
    cache_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    cache_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "stage": "attribute_profile",
        "model": model,
        "transcript_id": transcript_id,
        "attribute_key": spec.key,
        "attribute_summary": attribute_summary,
    }
    cache_key = hash_payload(cache_payload)
    cache_path = cache_root / "attribute_profiles" / slugify(transcript_id) / f"{spec.key}.json"
    if not overwrite:
        cached = load_cached_result(cache_path, cache_key)
        if cached is not None:
            return cached

    client = _create_chat_client(model, api_key)
    parsed = call_json(
        client,
        model=model,
        system_prompt=ATTRIBUTE_FACT_SYSTEM_PROMPT,
        user_prompt=build_attribute_fact_prompt(
            spec=spec,
            summary_description=attribute_summary["description"],
        ),
    )
    facts = postprocess_attribute_summary_facts(spec=spec, parsed=parsed)
    result = {
        "attribute_key": spec.key,
        "display_name": spec.display_name,
        "type_name": spec.type_name,
        "target_attribute": spec.target_attribute_str,
        "summary_description": attribute_summary["description"],
        "supported": bool(attribute_summary.get("supported", False)),
        "evidence_quote": attribute_summary.get("evidence_quote", ""),
        "count": len(facts),
        "facts": facts,
    }
    write_cached_result(
        cache_path,
        cache_key=cache_key,
        metadata={"model": model, "stage": "attribute_profile"},
        result=result,
    )
    return result


def flatten_completed_profile(attribute_profiles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    completed_profile: list[dict[str, Any]] = []
    for spec in ATTRIBUTE_SPECS:
        profile = attribute_profiles[spec.key]
        for index, fact_item in enumerate(profile.get("facts", []), start=1):
            completed_profile.append({
                "fact_id": f"{spec.key}_{index:02d}",
                "attribute_key": spec.key,
                "attribute_display_name": spec.display_name,
                "target_attribute": spec.target_attribute_str,
                "summary_description": profile.get("summary_description", ""),
                "fact": fact_item["fact"],
                "source_fields": fact_item.get("source_fields", [ATTRIBUTE_SUMMARY_SOURCE_FIELD]),
            })
    return completed_profile


def _merge_group_is_valid(source_items: list[dict[str, Any]]) -> bool:
    if len(source_items) <= 1:
        return True
    for left_index, left_item in enumerate(source_items):
        for right_item in source_items[left_index + 1 :]:
            if not _facts_overlap(str(left_item.get("fact", "")), str(right_item.get("fact", ""))):
                return False
    return True


def sanity_merge_completed_profile_with_cache(
    *,
    api_key: str,
    model: str,
    transcript_id: str,
    completed_profile: list[dict[str, Any]],
    cache_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    cache_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "stage": "profile_sanity_merge",
        "model": model,
        "transcript_id": transcript_id,
        "completed_profile": completed_profile,
    }
    cache_key = hash_payload(cache_payload)
    cache_path = cache_root / "profile_sanity_merge" / f"{slugify(transcript_id)}.json"
    if not overwrite:
        cached = load_cached_result(cache_path, cache_key)
        if cached is not None:
            return cached

    if not completed_profile:
        result = {
            "transcript_id": transcript_id,
            "merged_completed_profile": [],
            "merge_groups": [],
        }
        write_cached_result(
            cache_path,
            cache_key=cache_key,
            metadata={"model": model, "stage": "profile_sanity_merge"},
            result=result,
        )
        return result

    client = _create_chat_client(model, api_key)
    parsed = call_json(
        client,
        model=model,
        system_prompt=FACT_SANITY_MERGE_SYSTEM_PROMPT,
        user_prompt=build_fact_sanity_merge_prompt(
            transcript_id=transcript_id,
            completed_profile=completed_profile,
        ),
    )

    fact_by_id = {item["fact_id"]: item for item in completed_profile}
    fact_order = {item["fact_id"]: index for index, item in enumerate(completed_profile)}
    spec_by_key = {spec.key: spec for spec in ATTRIBUTE_SPECS}
    merged_entries = parsed.get("merged_facts", [])
    if not isinstance(merged_entries, list):
        merged_entries = []

    groups: list[dict[str, Any]] = []
    used_fact_ids: set[str] = set()
    for raw_entry in merged_entries:
        if not isinstance(raw_entry, dict):
            continue
        raw_source_ids = raw_entry.get("source_fact_ids", [])
        if not isinstance(raw_source_ids, list):
            continue
        source_fact_ids: list[str] = []
        for raw_fact_id in raw_source_ids:
            fact_id = normalize_text(raw_fact_id)
            if fact_id in fact_by_id and fact_id not in used_fact_ids and fact_id not in source_fact_ids:
                source_fact_ids.append(fact_id)
        if not source_fact_ids:
            continue

        primary_attribute_key = normalize_text(raw_entry.get("primary_attribute_key", "")).upper()
        if primary_attribute_key not in spec_by_key:
            primary_attribute_key = fact_by_id[source_fact_ids[0]]["attribute_key"]

        fact_text = normalize_text(raw_entry.get("fact", ""))
        if not fact_text:
            fact_text = max((fact_by_id[fact_id]["fact"] for fact_id in source_fact_ids), key=len)

        source_items = [fact_by_id[fact_id] for fact_id in source_fact_ids]
        if not _merge_group_is_valid(source_items):
            continue
        preferred_source = next(
            (item for item in source_items if item["attribute_key"] == primary_attribute_key),
            source_items[0],
        )
        groups.append({
            "primary_attribute_key": primary_attribute_key,
            "fact": fact_text,
            "source_fact_ids": source_fact_ids,
            "source_attribute_keys": sorted({item["attribute_key"] for item in source_items}),
            "source_fields": sorted({
                source_field
                for item in source_items
                for source_field in item.get("source_fields", [ATTRIBUTE_SUMMARY_SOURCE_FIELD])
            }),
            "summary_description": preferred_source["summary_description"],
            "sort_index": min(fact_order[fact_id] for fact_id in source_fact_ids),
        })
        used_fact_ids.update(source_fact_ids)

    for item in completed_profile:
        if item["fact_id"] in used_fact_ids:
            continue
        groups.append({
            "primary_attribute_key": item["attribute_key"],
            "fact": item["fact"],
            "source_fact_ids": [item["fact_id"]],
            "source_attribute_keys": [item["attribute_key"]],
            "source_fields": list(item.get("source_fields", [ATTRIBUTE_SUMMARY_SOURCE_FIELD])),
            "summary_description": item["summary_description"],
            "sort_index": fact_order[item["fact_id"]],
        })

    groups.sort(key=lambda item: (item["sort_index"], item["primary_attribute_key"], item["fact"].lower()))
    merged_completed_profile: list[dict[str, Any]] = []
    attr_counts: dict[str, int] = {}
    for group in groups:
        primary_attribute_key = group["primary_attribute_key"]
        attr_counts[primary_attribute_key] = attr_counts.get(primary_attribute_key, 0) + 1
        spec = spec_by_key[primary_attribute_key]
        merged_completed_profile.append({
            "fact_id": f"{primary_attribute_key}_{attr_counts[primary_attribute_key]:02d}",
            "attribute_key": primary_attribute_key,
            "attribute_display_name": spec.display_name,
            "target_attribute": spec.target_attribute_str,
            "summary_description": group["summary_description"],
            "fact": group["fact"],
            "source_fields": group["source_fields"],
            "merged_from_fact_ids": group["source_fact_ids"],
            "source_attribute_keys": group["source_attribute_keys"],
        })

    result = {
        "transcript_id": transcript_id,
        "merged_completed_profile": merged_completed_profile,
        "merge_groups": [
            {
                "fact": item["fact"],
                "attribute_key": item["attribute_key"],
                "merged_from_fact_ids": item["merged_from_fact_ids"],
                "source_attribute_keys": item["source_attribute_keys"],
            }
            for item in merged_completed_profile
        ],
    }
    write_cached_result(
        cache_path,
        cache_key=cache_key,
        metadata={"model": model, "stage": "profile_sanity_merge"},
        result=result,
    )
    return result


def rebuild_attribute_profiles_from_completed_profile(
    attribute_profiles: dict[str, dict[str, Any]],
    completed_profile: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rebuilt: dict[str, dict[str, Any]] = {}
    for spec in ATTRIBUTE_SPECS:
        original = attribute_profiles[spec.key]
        rebuilt[spec.key] = {
            **original,
            "facts": [],
            "count": 0,
        }

    for fact_item in completed_profile:
        attribute_key = fact_item["attribute_key"]
        if attribute_key not in rebuilt:
            continue
        rebuilt[attribute_key]["facts"].append({
            "fact": fact_item["fact"],
            "source_fields": fact_item.get("source_fields", [ATTRIBUTE_SUMMARY_SOURCE_FIELD]),
            "merged_from_fact_ids": fact_item.get("merged_from_fact_ids", [fact_item["fact_id"]]),
            "source_attribute_keys": fact_item.get("source_attribute_keys", [attribute_key]),
        })

    for spec in ATTRIBUTE_SPECS:
        rebuilt[spec.key]["count"] = len(rebuilt[spec.key]["facts"])
    return rebuilt


def build_reference_profile_for_transcript(
    *,
    api_key: str,
    model: str,
    transcript_id: str,
    transcript_text: str,
    profile_root: Path,
    attribute_workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    cache_root = profile_root / "_cache"
    attribute_summary_map: dict[str, dict[str, Any]] = {}
    summary_workers = max(1, min(attribute_workers, len(ATTRIBUTE_SPECS)))
    with ThreadPoolExecutor(max_workers=summary_workers) as executor:
        futures = {
            executor.submit(
                generate_attribute_summary_with_cache,
                api_key=api_key,
                model=model,
                transcript_id=transcript_id,
                transcript_text=transcript_text,
                spec=spec,
                cache_root=cache_root,
                overwrite=overwrite,
            ): spec.key
            for spec in ATTRIBUTE_SPECS
        }
        for future in as_completed(futures):
            attribute_summary_map[futures[future]] = future.result()

    attribute_profiles: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=summary_workers) as executor:
        futures = {
            executor.submit(
                extract_attribute_profile_with_cache,
                api_key=api_key,
                model=model,
                transcript_id=transcript_id,
                spec=spec,
                attribute_summary=attribute_summary_map[spec.key],
                cache_root=cache_root,
                overwrite=overwrite,
            ): spec.key
            for spec in ATTRIBUTE_SPECS
        }
        for future in as_completed(futures):
            attribute_profiles[futures[future]] = future.result()

    ordered_summaries = {spec.key: attribute_summary_map[spec.key] for spec in ATTRIBUTE_SPECS}
    raw_attribute_profiles = {spec.key: attribute_profiles[spec.key] for spec in ATTRIBUTE_SPECS}
    raw_completed_profile = flatten_completed_profile(raw_attribute_profiles)
    sanity_merged = sanity_merge_completed_profile_with_cache(
        api_key=api_key,
        model=model,
        transcript_id=transcript_id,
        completed_profile=raw_completed_profile,
        cache_root=cache_root,
        overwrite=overwrite,
    )
    completed_profile = sanity_merged["merged_completed_profile"]
    ordered_profiles = rebuild_attribute_profiles_from_completed_profile(
        raw_attribute_profiles,
        completed_profile,
    )

    summaries_payload = {
        "created_at": now_iso(),
        "model": model,
        "transcript_id": transcript_id,
        "attributes": ordered_summaries,
    }
    summary_path = profile_root / "reference_summaries" / f"{slugify(transcript_id)}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    reference_profile = {
        "created_at": now_iso(),
        "model": model,
        "transcript_id": transcript_id,
        "attributes": ordered_profiles,
        "raw_completed_profile": raw_completed_profile,
        "completed_profile": completed_profile,
        "sanity_merge": {
            "raw_fact_count": len(raw_completed_profile),
            "merged_fact_count": len(completed_profile),
            "merge_groups": sanity_merged.get("merge_groups", []),
        },
        "summary": {
            "total_atomic_facts": len(completed_profile),
            "nonempty_attributes": sum(1 for spec in ATTRIBUTE_SPECS if ordered_profiles[spec.key]["count"] > 0),
        },
    }
    profile_path = profile_root / "reference_profiles" / f"{slugify(transcript_id)}.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(reference_profile, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return reference_profile


def summarize_recoverability_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {decision: 0 for decision in RECOVERABILITY_DECISIONS}
    error_count = 0
    by_attribute: dict[str, dict[str, Any]] = {}
    for spec in ATTRIBUTE_SPECS:
        by_attribute[spec.key] = {
            "attribute_display_name": spec.display_name,
            "total": 0,
            "yes_count": 0,
            "no_count": 0,
            "unsure_count": 0,
            "error_count": 0,
            "recoverable_rate": None,
        }

    for item in items:
        decision = item.get("decision")
        attribute_key = str(item.get("attribute_key", ""))
        if attribute_key not in by_attribute:
            continue
        attr = by_attribute[attribute_key]
        attr["total"] += 1
        if item.get("error"):
            error_count += 1
            attr["error_count"] += 1
            continue
        if decision in counts:
            counts[decision] += 1
        if decision == RECOVERABILITY_DECISIONS[0]:
            attr["yes_count"] += 1
        elif decision == RECOVERABILITY_DECISIONS[1]:
            attr["no_count"] += 1
        elif decision == RECOVERABILITY_DECISIONS[2]:
            attr["unsure_count"] += 1

    for attr in by_attribute.values():
        if attr["total"] > 0:
            attr["recoverable_rate"] = round(attr["yes_count"] / attr["total"], 3)

    total = len(items)
    return {
        "total_facts": total,
        "yes_count": counts[RECOVERABILITY_DECISIONS[0]],
        "no_count": counts[RECOVERABILITY_DECISIONS[1]],
        "unsure_count": counts[RECOVERABILITY_DECISIONS[2]],
        "error_count": error_count,
        "recoverable_rate": round(counts[RECOVERABILITY_DECISIONS[0]] / total, 3) if total else None,
        "by_attribute": by_attribute,
    }


def evaluate_recoverability_fact(
    *,
    api_key: str,
    model: str,
    transcript_id: str,
    config_label: str,
    transcript_text: str,
    fact_item: dict[str, Any],
    retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    last_error: str | None = None
    for attempt in range(1, retries + 1):
        try:
            client = _create_chat_client(model, api_key)
            parsed = call_json(
                client,
                model=model,
                system_prompt=RECOVERABILITY_SYSTEM_PROMPT,
                user_prompt=build_recoverability_prompt(
                    transcript_id=transcript_id,
                    config_label=config_label,
                    transcript_text=transcript_text,
                    fact_item=fact_item,
                ),
            )
            decision = normalize_text(parsed.get("decision", ""))
            reasoning = normalize_text(parsed.get("reasoning", ""))
            evidence_quote = normalize_text(parsed.get("evidence_quote", ""))
            if decision not in RECOVERABILITY_DECISIONS:
                raise ValueError(f"Invalid decision: {decision!r}")
            if decision == RECOVERABILITY_DECISIONS[0] and not evidence_quote_supported(transcript_text, evidence_quote):
                decision = RECOVERABILITY_DECISIONS[2]
                if not reasoning:
                    reasoning = "The fact may be recoverable, but the response did not provide a verifiable evidence quote."
                evidence_quote = ""
            return {
                **fact_item,
                "decision": decision,
                "reasoning": reasoning,
                "evidence_quote": evidence_quote,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < retries:
                time.sleep(float(min(2 * attempt, 8)))

    return {
        **fact_item,
        "decision": RECOVERABILITY_DECISIONS[2],
        "reasoning": "",
        "evidence_quote": "",
        "error": last_error or "unknown",
    }


def evaluate_recoverability_config(
    *,
    dataset_id: str,
    config_name: str,
    transcript_text: str,
    transcript_source: str,
    reference_profile: dict[str, Any],
    api_key: str,
    model: str,
    output_root: Path,
    max_fact_workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    out_dir = output_root / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slugify(dataset_id)}.json"
    cache_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "stage": "recoverability",
        "model": model,
        "dataset_id": dataset_id,
        "config_name": config_name,
        "transcript_text": transcript_text,
        "completed_profile": reference_profile.get("completed_profile", []),
    }
    cache_key = hash_payload(cache_payload)

    if out_path.exists() and not overwrite:
        try:
            with open(out_path, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("cache_key") == cache_key:
                return cached
        except (json.JSONDecodeError, OSError):
            pass

    facts = list(reference_profile.get("completed_profile", []))
    workers = max(1, min(max_fact_workers, len(facts))) if facts else 1
    items: list[dict[str, Any]] = [None] * len(facts)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_recoverability_fact,
                api_key=api_key,
                model=model,
                transcript_id=dataset_id,
                config_label=config_name,
                transcript_text=transcript_text,
                fact_item=fact_item,
            ): idx
            for idx, fact_item in enumerate(facts)
        }
        for future in as_completed(futures):
            items[futures[future]] = future.result()

    result = {
        "cache_key": cache_key,
        "dataset_id": dataset_id,
        "config_name": config_name,
        "transcript_source": transcript_source,
        "model": model,
        "timestamp": now_iso(),
        "summary": summarize_recoverability_items(items),
        "items": items,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return result


def aggregate_recoverability_summary(
    payloads_by_sample: dict[str, dict[str, dict[str, Any]]],
    *,
    config_order: list[str],
) -> dict[str, Any]:
    by_config: dict[str, Any] = {}
    for config_name in config_order:
        sample_payloads = [
            payloads_by_sample[sample_id][config_name]
            for sample_id in payloads_by_sample
            if config_name in payloads_by_sample[sample_id]
        ]
        if not sample_payloads:
            continue

        attr_summary: dict[str, Any] = {}
        total_facts = 0
        yes_count = 0
        no_count = 0
        unsure_count = 0
        error_count = 0
        for spec in ATTRIBUTE_SPECS:
            attr_summary[spec.key] = {
                "attribute_display_name": spec.display_name,
                "total": 0,
                "yes_count": 0,
                "no_count": 0,
                "unsure_count": 0,
                "error_count": 0,
                "recoverable_rate": None,
            }

        for payload in sample_payloads:
            summary = payload.get("summary", {})
            total_facts += int(summary.get("total_facts", 0))
            yes_count += int(summary.get("yes_count", 0))
            no_count += int(summary.get("no_count", 0))
            unsure_count += int(summary.get("unsure_count", 0))
            error_count += int(summary.get("error_count", 0))
            for spec in ATTRIBUTE_SPECS:
                info = summary.get("by_attribute", {}).get(spec.key, {})
                attr = attr_summary[spec.key]
                attr["total"] += int(info.get("total", 0))
                attr["yes_count"] += int(info.get("yes_count", 0))
                attr["no_count"] += int(info.get("no_count", 0))
                attr["unsure_count"] += int(info.get("unsure_count", 0))
                attr["error_count"] += int(info.get("error_count", 0))

        for spec in ATTRIBUTE_SPECS:
            attr = attr_summary[spec.key]
            if attr["total"] > 0:
                attr["recoverable_rate"] = round(attr["yes_count"] / attr["total"], 3)

        by_config[config_name] = {
            "sample_count": len(sample_payloads),
            "total_facts": total_facts,
            "yes_count": yes_count,
            "no_count": no_count,
            "unsure_count": unsure_count,
            "error_count": error_count,
            "recoverable_rate": round(yes_count / total_facts, 3) if total_facts else None,
            "by_attribute": attr_summary,
        }

    by_sample: dict[str, dict[str, Any]] = {}
    for sample_id, config_payloads in payloads_by_sample.items():
        by_sample[sample_id] = {}
        for config_name, payload in config_payloads.items():
            by_sample[sample_id][config_name] = payload.get("summary", {})

    return {
        "by_config": by_config,
        "by_sample": by_sample,
    }


def load_reference_transcripts(path: Path) -> dict[str, str]:
    transcripts = ap_maybe_load_csv_map(path, "transcript_id", "text")
    if not transcripts:
        raise ValueError(f"No transcripts found in {path}")
    return transcripts


def build_profile_config_sources(args: argparse.Namespace, original_map: dict[str, str]) -> list[tuple[str, dict[str, str], str]]:
    configs: list[tuple[str, dict[str, str], str]] = [
        ("original", original_map, str(args.original_csv)),
        ("adaptive_privacy", ap_maybe_load_csv_map(args.adaptive_path, "transcript_id", "text"), str(args.adaptive_path)),
        (
            "anonymized_transcripts",
            ap_maybe_load_csv_map(args.anonymized_path, "Dataset_ID", "Transcript"),
            str(args.anonymized_path),
        ),
    ]

    if args.config_set == "diff_report":
        extras: list[tuple[str, dict[str, str], str]] = [
            ("nobranch", ap_maybe_load_csv_map(args.nobranch_path, "transcript_id", "text"), str(args.nobranch_path)),
            (
                "pure_adaptive_attri",
                ap_maybe_load_csv_map(args.pure_adaptive_attri_path, "transcript_id", "text"),
                str(args.pure_adaptive_attri_path),
            ),
            ("on_device", ap_maybe_load_csv_map(args.on_device_path, "transcript_id", "text"), str(args.on_device_path)),
            (
                "on_device_qwen",
                ap_maybe_load_csv_map(args.on_device_qwen_path, "transcript_id", "text"),
                str(args.on_device_qwen_path),
            ),
            ("remove_2", ap_maybe_load_csv_map(args.remove2_path, "transcript_id", "text"), str(args.remove2_path)),
            ("remove_4", ap_maybe_load_csv_map(args.remove4_path, "transcript_id", "text"), str(args.remove4_path)),
            (
                "presidio",
                ap_maybe_load_csv_map(args.presidio_path, "Dataset_ID", "rewritten_transcript_presidio"),
                str(args.presidio_path),
            ),
            (
                "rewritten_v1",
                ap_maybe_load_csv_map_flexible(
                    args.rewritten_v1_path,
                    [("Dataset_ID", "rewritten transcript"), ("transcript_id", "text")],
                ),
                str(args.rewritten_v1_path),
            ),
            (
                "rewritten_v2",
                ap_maybe_load_csv_map_flexible(
                    args.rewritten_v2_path,
                    [("Dataset_ID", "rewritten transcript"), ("transcript_id", "text")],
                ),
                str(args.rewritten_v2_path),
            ),
            (
                "dpmlm_eps_10",
                ap_maybe_load_csv_map(args.dpmlm_10_path, "transcript_id", "rewritten_text", epsilon="10"),
                f"{args.dpmlm_10_path} (epsilon=10)",
            ),
            (
                "dpmlm_eps_30",
                ap_maybe_load_csv_map(args.dpmlm_30_path, "transcript_id", "rewritten_text", epsilon="30"),
                f"{args.dpmlm_30_path} (epsilon=30)",
            ),
            (
                "dpmlm_eps_50",
                ap_maybe_load_csv_map(args.dpmlm_50_path, "transcript_id", "rewritten_text", epsilon="50"),
                f"{args.dpmlm_50_path} (epsilon=50)",
            ),
            (
                "dpmlm_eps_70",
                ap_maybe_load_csv_map(args.dpmlm_70_path, "transcript_id", "rewritten_text", epsilon="70"),
                f"{args.dpmlm_70_path} (epsilon=70)",
            ),
            (
                "dpmlm_eps_100",
                ap_maybe_load_csv_map(args.dpmlm_path, "Dataset_ID", "rewritten_transcript", epsilon="100"),
                f"{args.dpmlm_path} (epsilon=100)",
            ),
            (
                "dpmlm_eps_120",
                ap_maybe_load_csv_map(args.dpmlm_path, "Dataset_ID", "rewritten_transcript", epsilon="120"),
                f"{args.dpmlm_path} (epsilon=120)",
            ),
            (
                "dpmlm_eps_140",
                ap_maybe_load_csv_map(args.dpmlm_path, "Dataset_ID", "rewritten_transcript", epsilon="140"),
                f"{args.dpmlm_path} (epsilon=140)",
            ),
        ]
        for config_name, text_map, source in extras:
            if text_map:
                configs.append((config_name, text_map, source))

    if args.only_configs:
        requested = list(dict.fromkeys(args.only_configs))
        requested_set = set(requested)
        configs = [cfg for cfg in configs if cfg[0] in requested_set]
        present = {cfg_name for cfg_name, _, _ in configs}
        missing = [cfg_name for cfg_name in requested if cfg_name not in present]
        if missing:
            raise ValueError(f"Requested configs not available: {', '.join(missing)}")

    return configs
def normalize_candidate_indices(raw_value: Any, valid_indices: set[int]) -> list[int]:
    out: list[int] = []
    if isinstance(raw_value, list):
        for item in raw_value:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if idx in valid_indices and idx not in out:
                out.append(idx)
    return sorted(out)


def normalize_source_fields(raw_value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(raw_value, list):
        for item in raw_value:
            value = normalize_text(item)
            if value in SOURCE_FIELDS and value not in out:
                out.append(value)
    return sorted(out)


def _append_fact_variant(variants: list[str], fact: str) -> None:
    normalized = normalize_text(fact).strip(" ,;")
    if not normalized:
        return
    if normalized not in variants:
        variants.append(normalized)


def _split_tail_segments(text: str) -> list[str]:
    stripped = normalize_text(text)
    if not stripped:
        return []
    if " and " not in stripped and "," not in stripped:
        return [stripped]

    raw_segments = re.split(r",| and ", stripped)
    segments = [normalize_text(segment) for segment in raw_segments if normalize_text(segment)]
    return segments or [stripped]


def expand_atomic_fact(spec_key: str, fact: str) -> list[str]:
    derived_variants: list[str] = []
    lowered = normalize_text(fact).lower()
    if spec_key != "OCCP":
        return [normalize_text(fact)] if normalize_text(fact) else []

    if lowered.startswith("works on "):
        tail = normalize_text(fact[9:])
        tail = re.sub(r"\s*\((?:e\.g\.,?|for example)[^)]+\)\s*", "", tail, flags=re.IGNORECASE)
        if tail:
            if " of " in tail:
                left, right = tail.split(" of ", 1)
                _append_fact_variant(derived_variants, f"works on {left}")
                _append_fact_variant(derived_variants, f"works on {right}")

    if lowered.startswith("uses ai tools for "):
        tail = normalize_text(fact[18:])
        for segment in _split_tail_segments(tail):
            _append_fact_variant(derived_variants, f"uses AI tools for {segment}")

    if derived_variants:
        return derived_variants
    return [normalize_text(fact)] if normalize_text(fact) else []


def _normalize_overlap_token(token: str) -> str:
    normalized = token.lower()
    replacements = {
        "publishes": "publish",
        "published": "publish",
        "publishing": "publish",
        "uses": "use",
        "using": "use",
        "used": "use",
        "works": "work",
        "working": "work",
        "sells": "sell",
        "selling": "sell",
        "books": "book",
        "reviews": "review",
        "materials": "material",
        "platforms": "platform",
        "products": "product",
        "designs": "design",
    }
    normalized = replacements.get(normalized, normalized)
    if normalized.endswith("ies") and len(normalized) > 4:
        normalized = normalized[:-3] + "y"
    elif normalized.endswith("s") and len(normalized) > 3 and not normalized.endswith("ss"):
        normalized = normalized[:-1]
    return normalized


def _fact_overlap_text(fact: str) -> str:
    text = normalize_text(fact).lower().replace("self-published", "self published")
    prefixes = (
        "uses ai tools for ",
        "uses ai tool for ",
        "uses ai for ",
        "works on ",
        "work on ",
        "publishes ",
        "publishes ",
        "self published ",
        "sells ",
        "sell ",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def _fact_core_tokens(fact: str) -> set[str]:
    text = _fact_overlap_text(fact)
    raw_tokens = re.findall(r"[a-z0-9]+", text)
    tokens = {_normalize_overlap_token(token) for token in raw_tokens}
    return {token for token in tokens if token and token not in FACT_OVERLAP_STOPWORDS}


def _fact_specificity_sort_key(item: dict[str, Any]) -> tuple[int, int, int]:
    core_tokens = _fact_core_tokens(str(item.get("fact", "")))
    fact_text = normalize_text(item.get("fact", ""))
    return (-len(core_tokens), -len(fact_text), -len(item.get("source_fields", [])))


def _facts_overlap(left_fact: str, right_fact: str) -> bool:
    left_tokens = _fact_core_tokens(left_fact)
    right_tokens = _fact_core_tokens(right_fact)
    if not left_tokens or not right_tokens:
        return False
    if left_tokens == right_tokens:
        return True
    smaller, larger = sorted((left_tokens, right_tokens), key=len)
    return smaller.issubset(larger)


def _merge_confidence_lists(left: list[str], right: list[str]) -> list[str]:
    merged: list[str] = []
    for raw in list(left) + list(right):
        value = normalize_text(raw).lower()
        if value and value not in merged:
            merged.append(value)
    merged.sort(key=confidence_rank, reverse=True)
    return merged


def _merge_fact_items(preferred: dict[str, Any], dropped: dict[str, Any]) -> dict[str, Any]:
    preferred["supporting_candidates"] = sorted(
        set(preferred.get("supporting_candidates", [])) | set(dropped.get("supporting_candidates", []))
    )
    preferred["source_fields"] = sorted(
        set(preferred.get("source_fields", [])) | set(dropped.get("source_fields", []))
    )
    preferred["supporting_candidate_confidences"] = _merge_confidence_lists(
        list(preferred.get("supporting_candidate_confidences", [])),
        list(dropped.get("supporting_candidate_confidences", [])),
    )
    preferred["confidence_weight"] = round(
        max(
            float(preferred.get("confidence_weight", 0.0)),
            float(dropped.get("confidence_weight", 0.0)),
        ),
        3,
    )
    return preferred


def dedupe_overlapping_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = sorted(facts, key=_fact_specificity_sort_key)
    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        overlapped = False
        for idx, existing in enumerate(kept):
            if _facts_overlap(str(candidate.get("fact", "")), str(existing.get("fact", ""))):
                kept[idx] = _merge_fact_items(existing, candidate)
                overlapped = True
                break
        if not overlapped:
            kept.append(candidate)
    kept.sort(key=lambda item: (-item["confidence_weight"], item["fact"].lower()))
    return kept


def postprocess_profile(
    parsed: dict[str, Any],
    *,
    selected_candidates: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    candidate_conf_map = {
        int(candidate["candidate_index"]): str(candidate.get("confidence", ""))
        for candidate in selected_candidates
    }
    valid_indices = set(candidate_conf_map.keys())

    out: dict[str, list[dict[str, Any]]] = {}
    for spec in ATTRIBUTE_SPECS:
        merged: dict[str, dict[str, Any]] = {}
        raw_items = parsed.get(spec.key, [])
        if not isinstance(raw_items, list):
            raw_items = []

        for raw_item in raw_items:
            if isinstance(raw_item, dict):
                fact = normalize_text(raw_item.get("fact", ""))
                supporting_candidates = normalize_candidate_indices(
                    raw_item.get("supporting_candidates", []), valid_indices
                )
                source_fields = normalize_source_fields(raw_item.get("source_fields", []))
            else:
                fact = normalize_text(raw_item)
                supporting_candidates = []
                source_fields = []

            if not fact:
                continue
            if not source_fields:
                source_fields = ["unified_description"]

            fact_variants = expand_atomic_fact(spec.key, fact)
            for fact_variant in fact_variants:
                fact_key = canonical_fact_key(fact_variant)
                if not fact_key:
                    continue

                item = merged.setdefault(
                    fact_key,
                    {
                        "fact": fact_variant,
                        "supporting_candidates": [],
                        "source_fields": [],
                    },
                )
                item["supporting_candidates"] = sorted(
                    set(item["supporting_candidates"]) | set(supporting_candidates)
                )
                item["source_fields"] = sorted(set(item["source_fields"]) | set(source_fields))

        finalized: list[dict[str, Any]] = []
        for item in merged.values():
            confidences = [
                candidate_conf_map[idx]
                for idx in item["supporting_candidates"]
                if idx in candidate_conf_map
            ]
            max_weight = max((confidence_weight(conf) for conf in confidences), default=0.0)
            finalized.append({
                "fact": item["fact"],
                "supporting_candidates": item["supporting_candidates"],
                "source_fields": item["source_fields"],
                "supporting_candidate_confidences": confidences,
                "confidence_weight": round(max_weight, 3),
            })

        out[spec.key] = dedupe_overlapping_facts(finalized)
    return out


def summarize_profile(
    *,
    config_label: str,
    input_path: Path,
    transcript_id: str,
    normalized_sample_id: str,
    selected_candidates: list[dict[str, Any]],
    atomic_profile: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    attribute_summary: dict[str, Any] = {}
    total_count = 0
    total_weighted = 0.0

    for spec in ATTRIBUTE_SPECS:
        facts = atomic_profile.get(spec.key, [])
        count = len(facts)
        weighted_count = round(sum(float(item.get("confidence_weight", 0.0)) for item in facts), 3)
        total_count += count
        total_weighted += weighted_count
        attribute_summary[spec.key] = {
            "display_name": spec.display_name,
            "type_name": spec.type_name,
            "target_attribute": spec.target_attribute_str,
            "count": count,
            "weighted_count": weighted_count,
            "facts": facts,
        }

    return {
        "config_label": config_label,
        "input_path": str(input_path),
        "transcript_id": transcript_id,
        "normalized_transcript_id": normalized_sample_id,
        "candidate_count": len(selected_candidates),
        "candidate_confidences": [candidate.get("confidence", "") for candidate in selected_candidates],
        "best_candidate_confidence": selected_candidates[0].get("confidence", "") if selected_candidates else "",
        "selected_identifier_candidates": selected_candidates,
        "attributes": attribute_summary,
        "summary": {
            "total_atomic_facts": total_count,
            "confidence_weighted_total": round(total_weighted, 3),
            "nonempty_attributes": sum(1 for spec in ATTRIBUTE_SPECS if attribute_summary[spec.key]["count"] > 0),
        },
    }


def build_cache_key(
    *,
    model: str,
    config_label: str,
    transcript_id: str,
    selected_candidates: list[dict[str, Any]],
) -> str:
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "model": model,
        "config_label": config_label,
        "transcript_id": transcript_id,
        "selected_candidates": selected_candidates,
        "attribute_keys": [spec.key for spec in ATTRIBUTE_SPECS],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def extract_profile_with_cache(
    *,
    api_key: str,
    model: str,
    config_label: str,
    input_path: Path,
    sample_id: str,
    transcript_id: str,
    identifier_candidates: list[Any],
    cache_root: Path,
    overwrite_cache: bool,
    max_candidates: int | None,
) -> dict[str, Any]:
    selected_candidates = select_identifier_candidates(identifier_candidates, max_candidates=max_candidates)
    cache_dir = cache_root / slugify(config_label)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = build_cache_key(
        model=model,
        config_label=config_label,
        transcript_id=sample_id,
        selected_candidates=selected_candidates,
    )
    cache_path = cache_dir / f"{slugify(sample_id)}.json"

    if cache_path.exists() and not overwrite_cache:
        try:
            with open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            if (
                cached.get("cache_key") == cache_key
                and cached.get("model") == model
                and cached.get("config_label") == config_label
            ):
                return cached["result"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    atomic_profile = {spec.key: [] for spec in ATTRIBUTE_SPECS}
    if selected_candidates:
        client = _create_chat_client(model, api_key)
        unified_candidates = build_unified_candidates(
            client,
            model=model,
            selected_candidates=selected_candidates,
        )
        prompt = build_extraction_prompt(
            sample_id=transcript_id,
            config_label=config_label,
            unified_candidates=unified_candidates,
        )
        parsed = call_json(
            client,
            model=model,
            system_prompt=EXTRACT_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
        atomic_profile = postprocess_profile(parsed, selected_candidates=unified_candidates)
        selected_candidates = unified_candidates

    result = summarize_profile(
        config_label=config_label,
        input_path=input_path,
        transcript_id=transcript_id,
        normalized_sample_id=sample_id,
        selected_candidates=selected_candidates,
        atomic_profile=atomic_profile,
    )

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cache_key": cache_key,
                "model": model,
                "config_label": config_label,
                "result": result,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    return result


def compare_against_reference(
    sample_profiles: dict[str, dict[str, Any]],
    *,
    ordered_labels: list[str],
    reference_label: str,
) -> dict[str, Any]:
    reference = sample_profiles[reference_label]
    comparison: dict[str, Any] = {
        "reference_label": reference_label,
        "by_config": {},
    }

    for label in ordered_labels:
        profile = sample_profiles.get(label)
        if profile is None:
            continue

        attr_comparison: dict[str, Any] = {}
        for spec in ATTRIBUTE_SPECS:
            ref_attr = reference["attributes"][spec.key]
            cur_attr = profile["attributes"][spec.key]
            ref_count = ref_attr["count"]
            ref_weighted = ref_attr["weighted_count"]
            cur_count = cur_attr["count"]
            cur_weighted = cur_attr["weighted_count"]
            attr_comparison[spec.key] = {
                "count": cur_count,
                "weighted_count": cur_weighted,
                "count_delta_vs_reference": cur_count - ref_count,
                "weighted_delta_vs_reference": round(cur_weighted - ref_weighted, 3),
                "count_ratio_vs_reference": round(cur_count / ref_count, 3) if ref_count else None,
                "weighted_ratio_vs_reference": (
                    round(cur_weighted / ref_weighted, 3) if ref_weighted else None
                ),
            }

        ref_total = reference["summary"]["total_atomic_facts"]
        ref_weighted_total = reference["summary"]["confidence_weighted_total"]
        cur_total = profile["summary"]["total_atomic_facts"]
        cur_weighted_total = profile["summary"]["confidence_weighted_total"]
        comparison["by_config"][label] = {
            "summary": {
                "total_atomic_facts": cur_total,
                "confidence_weighted_total": cur_weighted_total,
                "total_delta_vs_reference": cur_total - ref_total,
                "weighted_total_delta_vs_reference": round(cur_weighted_total - ref_weighted_total, 3),
                "total_ratio_vs_reference": round(cur_total / ref_total, 3) if ref_total else None,
                "weighted_total_ratio_vs_reference": (
                    round(cur_weighted_total / ref_weighted_total, 3) if ref_weighted_total else None
                ),
            },
            "attributes": attr_comparison,
        }
    return comparison


def aggregate_dataset_summary(
    results_by_sample: dict[str, dict[str, dict[str, Any]]],
    *,
    ordered_labels: list[str],
    reference_label: str,
) -> dict[str, Any]:
    config_summary: dict[str, Any] = {}
    for label in ordered_labels:
        available_profiles = [
            sample_profiles[label]
            for sample_profiles in results_by_sample.values()
            if label in sample_profiles
        ]
        if not available_profiles:
            continue

        attr_summary: dict[str, Any] = {}
        for spec in ATTRIBUTE_SPECS:
            counts = [profile["attributes"][spec.key]["count"] for profile in available_profiles]
            weighted = [profile["attributes"][spec.key]["weighted_count"] for profile in available_profiles]
            attr_summary[spec.key] = {
                "total_count": sum(counts),
                "average_count": round(sum(counts) / len(counts), 3),
                "total_weighted_count": round(sum(weighted), 3),
                "average_weighted_count": round(sum(weighted) / len(weighted), 3),
            }

        totals = [profile["summary"]["total_atomic_facts"] for profile in available_profiles]
        weighted_totals = [profile["summary"]["confidence_weighted_total"] for profile in available_profiles]
        config_summary[label] = {
            "sample_count": len(available_profiles),
            "average_total_atomic_facts": round(sum(totals) / len(totals), 3),
            "average_confidence_weighted_total": round(sum(weighted_totals) / len(weighted_totals), 3),
            "attributes": attr_summary,
        }

    preservation_vs_reference: dict[str, Any] = {}
    for label in ordered_labels:
        sample_ids = [
            sample_id
            for sample_id, sample_profiles in results_by_sample.items()
            if reference_label in sample_profiles and label in sample_profiles
        ]
        if not sample_ids:
            continue

        total_ratios: list[float] = []
        weighted_total_ratios: list[float] = []
        for sample_id in sample_ids:
            reference = results_by_sample[sample_id][reference_label]
            current = results_by_sample[sample_id][label]
            ref_total = reference["summary"]["total_atomic_facts"]
            cur_total = current["summary"]["total_atomic_facts"]
            if ref_total:
                total_ratios.append(cur_total / ref_total)

            ref_weighted = reference["summary"]["confidence_weighted_total"]
            cur_weighted = current["summary"]["confidence_weighted_total"]
            if ref_weighted:
                weighted_total_ratios.append(cur_weighted / ref_weighted)

        preservation_vs_reference[label] = {
            "sample_count": len(sample_ids),
            "average_total_ratio": round(sum(total_ratios) / len(total_ratios), 3) if total_ratios else None,
            "average_weighted_total_ratio": (
                round(sum(weighted_total_ratios) / len(weighted_total_ratios), 3)
                if weighted_total_ratios
                else None
            ),
        }

    return {
        "reference_label": reference_label,
        "config_summary": config_summary,
        "preservation_vs_reference": preservation_vs_reference,
    }


def resolve_target_sample_ids(
    rows_by_label: dict[str, dict[str, dict[str, Any]]],
    requested_sample_ids: list[str] | None,
) -> list[str]:
    if requested_sample_ids:
        normalized = [normalize_dataset_id(sample_id) for sample_id in requested_sample_ids if normalize_text(sample_id)]
        unique = list(dict.fromkeys(normalized))
        return unique

    label_order = list(rows_by_label.keys())
    if not label_order:
        return []
    shared_ids = set(rows_by_label[label_order[0]].keys())
    for label in label_order[1:]:
        shared_ids &= set(rows_by_label[label].keys())
    return sorted(shared_ids)


def build_run_paths(
    *,
    output_root: Path,
    sample_ids: list[str],
    ordered_labels: list[str],
) -> tuple[Path, Path]:
    sample_tag = slugify(sample_ids[0]) if len(sample_ids) == 1 else f"{len(sample_ids)}samples"
    label_tag = "_vs_".join(slugify(label) for label in ordered_labels)
    run_dir = output_root / f"{sample_tag}__{label_tag}"
    json_path = run_dir / "comparison.json"
    return run_dir, json_path


def print_console_summary(
    payload: dict[str, Any],
    *,
    ordered_labels: list[str],
) -> None:
    print("")
    print(f"Model: {payload['model']}")
    print(f"Reference config: {payload['reference_label']}")
    print(f"Samples analyzed: {len(payload['sample_ids'])}")

    dataset_summary = payload.get("dataset_summary", {})
    config_summary = dataset_summary.get("config_summary", {})
    if config_summary:
        print("")
        print("Dataset-level average totals:")
        for label in ordered_labels:
            info = config_summary.get(label)
            if not info:
                continue
            print(
                f"  - {label}: avg_total_atomic_facts={info['average_total_atomic_facts']}, "
                f"avg_weighted_total={info['average_confidence_weighted_total']}"
            )

    for sample_id in payload["sample_ids"]:
        sample_payload = payload["samples"][sample_id]
        print("")
        print(f"Sample: {sample_id}")
        print("  Config totals:")
        for label in ordered_labels:
            profile = sample_payload["profiles"].get(label)
            if profile is None:
                continue
            summary = profile["summary"]
            print(
                f"    - {label}: total_atomic_facts={summary['total_atomic_facts']}, "
                f"weighted_total={summary['confidence_weighted_total']}, "
                f"best_confidence={profile['best_candidate_confidence'] or 'n/a'}, "
                f"candidates={profile['candidate_count']}"
            )

        print("  Attribute counts:")
        for spec in ATTRIBUTE_SPECS:
            per_label = []
            for label in ordered_labels:
                profile = sample_payload["profiles"].get(label)
                if profile is None:
                    continue
                count = profile["attributes"][spec.key]["count"]
                per_label.append(f"{label}={count}")
            print(f"    - {spec.key}: " + ", ".join(per_label))

        if len(payload["sample_ids"]) == 1:
            print("  Atomic facts:")
            for label in ordered_labels:
                profile = sample_payload["profiles"].get(label)
                if profile is None:
                    continue
                print(f"    {label}:")
                for spec in ATTRIBUTE_SPECS:
                    facts = [item["fact"] for item in profile["attributes"][spec.key]["facts"]]
                    facts_text = "; ".join(facts) if facts else "[none]"
                    print(f"      - {spec.key}: {facts_text}")


def print_recoverability_console_summary(
    payload: dict[str, Any],
    *,
    config_order: list[str],
) -> None:
    print("")
    print(f"Model: {payload['model']}")
    print(f"Reference transcripts: {len(payload['sample_ids'])}")
    print("Recoverability summary:")
    for config_name in config_order:
        info = payload["summary"]["by_config"].get(config_name)
        if not info:
            continue
        print(
            f"  - {config_name}: samples={info['sample_count']}, facts={info['total_facts']}, "
            f"yes={info['yes_count']}, no={info['no_count']}, unsure={info['unsure_count']}, "
            f"recoverable_rate={info['recoverable_rate']}"
        )


def run_reid_compare(args: argparse.Namespace) -> int:
    if not args.input:
        raise ValueError("--input is required for --workflow reid_compare")

    ordered_labels: list[str] = []
    input_paths_by_label: dict[str, Path] = {}
    rows_by_label: dict[str, dict[str, dict[str, Any]]] = {}

    for raw_input in args.input:
        label, path = parse_input_spec(raw_input)
        if label in input_paths_by_label:
            raise ValueError(f"Duplicate input label: {label}")
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        ordered_labels.append(label)
        input_paths_by_label[label] = path
        rows_by_label[label] = load_reid_rows(path)

    reference_label = slugify(args.reference_label) if args.reference_label else ordered_labels[0]
    if reference_label not in rows_by_label:
        raise ValueError(f"Reference label `{reference_label}` is not one of the inputs: {ordered_labels}")

    target_sample_ids = resolve_target_sample_ids(rows_by_label, args.sample_id)
    if not target_sample_ids:
        raise ValueError("No target sample IDs found to analyze")

    run_dir, json_path = build_run_paths(
        output_root=args.output_root,
        sample_ids=target_sample_ids,
        ordered_labels=ordered_labels,
    )
    cache_root = run_dir / "_cache"
    run_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str, Path, dict[str, Any]]] = []
    missing_samples: dict[str, list[str]] = {sample_id: [] for sample_id in target_sample_ids}
    for sample_id in target_sample_ids:
        for label in ordered_labels:
            row = rows_by_label[label].get(sample_id)
            if row is None:
                missing_samples[sample_id].append(label)
                continue
            tasks.append((sample_id, label, input_paths_by_label[label], row))

    if any(missing_samples.values()) and args.sample_id:
        missing_text = ", ".join(
            f"{sample_id}: missing in {', '.join(labels)}"
            for sample_id, labels in missing_samples.items()
            if labels
        )
        raise ValueError(f"Requested sample IDs were not found in every config: {missing_text}")

    api_key = os.getenv("OPENAI_API_KEY")
    results_by_sample: dict[str, dict[str, dict[str, Any]]] = {sample_id: {} for sample_id in target_sample_ids}

    worker_count = max(1, min(args.max_workers, len(tasks))) if tasks else 1
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {}
        for sample_id, label, input_path, row in tasks:
            future = executor.submit(
                extract_profile_with_cache,
                api_key=api_key,
                model=args.model,
                config_label=label,
                input_path=input_path,
                sample_id=sample_id,
                transcript_id=row["transcript_id"],
                identifier_candidates=row.get("identifier_candidates", []),
                cache_root=cache_root,
                overwrite_cache=args.overwrite_cache,
                max_candidates=args.max_candidates,
            )
            future_map[future] = (sample_id, label)

        for future in as_completed(future_map):
            sample_id, label = future_map[future]
            results_by_sample[sample_id][label] = future.result()
            print(f"Finished {sample_id} / {label}")

    sample_payloads: dict[str, Any] = {}
    for sample_id in target_sample_ids:
        sample_profiles = results_by_sample[sample_id]
        if reference_label not in sample_profiles:
            raise ValueError(f"Reference config `{reference_label}` missing for sample `{sample_id}`")
        sample_payloads[sample_id] = {
            "profiles": sample_profiles,
            "comparison": compare_against_reference(
                sample_profiles,
                ordered_labels=ordered_labels,
                reference_label=reference_label,
            ),
        }

    payload = {
        "created_at": now_iso(),
        "model": args.model,
        "reference_label": reference_label,
        "inputs": {label: str(path) for label, path in input_paths_by_label.items()},
        "sample_ids": target_sample_ids,
        "samples": sample_payloads,
        "dataset_summary": aggregate_dataset_summary(
            results_by_sample,
            ordered_labels=ordered_labels,
            reference_label=reference_label,
        ),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print_console_summary(payload, ordered_labels=ordered_labels)
    print("")
    print(f"Saved comparison JSON: {json_path}")
    return 0


def run_profile_recoverability(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args.model)
    profile_root = args.profile_root
    profile_root.mkdir(parents=True, exist_ok=True)

    original_map = load_reference_transcripts(args.original_csv)
    if args.sample_id:
        target_sample_ids = [
            normalize_dataset_id(sample_id)
            for sample_id in args.sample_id
            if normalize_text(sample_id)
        ]
        target_sample_ids = list(dict.fromkeys(target_sample_ids))
        missing = [sample_id for sample_id in target_sample_ids if sample_id not in original_map]
        if missing:
            raise ValueError(f"Requested sample IDs not found in original CSV: {', '.join(missing)}")
    else:
        target_sample_ids = sorted(original_map.keys())

    sample_workers = max(1, min(args.sample_workers, len(target_sample_ids))) if target_sample_ids else 1
    reference_profiles: dict[str, dict[str, Any]] = {}
    all_profiles_path = profile_root / "reference_profiles" / "all_profiles.json"

    if args.skip_reference_build:
        reference_profile_dir = profile_root / "reference_profiles"
        for sample_id in target_sample_ids:
            ref_path = reference_profile_dir / f"{slugify(sample_id)}.json"
            if not ref_path.exists():
                raise FileNotFoundError(f"Missing reference profile for --skip-reference-build: {ref_path}")
            with open(ref_path, encoding="utf-8") as f:
                reference_profiles[sample_id] = json.load(f)
            print(f"Loaded reference profile {sample_id}")
    else:
        def _build_reference_profile(sample_id: str) -> tuple[str, dict[str, Any]]:
            transcript_text = original_map[sample_id]
            profile = build_reference_profile_for_transcript(
                api_key=api_key,
                model=args.model,
                transcript_id=sample_id,
                transcript_text=transcript_text,
                profile_root=profile_root,
                attribute_workers=args.max_workers,
                overwrite=args.overwrite,
            )
            return sample_id, profile

        with ThreadPoolExecutor(max_workers=sample_workers) as executor:
            future_map = {executor.submit(_build_reference_profile, sample_id): sample_id for sample_id in target_sample_ids}
            for future in as_completed(future_map):
                sample_id, profile = future.result()
                reference_profiles[sample_id] = profile
                print(f"Built reference profile {sample_id}")

        all_profiles_payload = {
            "created_at": now_iso(),
            "model": args.model,
            "original_csv": str(args.original_csv),
            "sample_ids": sorted(reference_profiles.keys()),
            "profiles": {sample_id: reference_profiles[sample_id] for sample_id in sorted(reference_profiles.keys())},
        }
        all_profiles_path.parent.mkdir(parents=True, exist_ok=True)
        with open(all_profiles_path, "w", encoding="utf-8") as f:
            json.dump(all_profiles_payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

        if args.reference_only:
            print("")
            print(f"Saved reference profiles: {all_profiles_path}")
            print("Skipping recoverability evaluation (--reference-only).")
            return 0

    configs = build_profile_config_sources(args, original_map)
    config_lookup = {config_name: (text_map, source) for config_name, text_map, source in configs}
    ordered_configs = [cfg for cfg in AP_CFG_ORDER if cfg in config_lookup]

    recoverability_root = profile_root / "recoverability"
    payloads_by_sample: dict[str, dict[str, dict[str, Any]]] = {}

    def _process_sample(sample_id: str) -> tuple[str, dict[str, dict[str, Any]]]:
        sample_payloads: dict[str, dict[str, Any]] = {}
        config_tasks: list[tuple[str, str, str]] = []
        for config_name in ordered_configs:
            text_map, source = config_lookup[config_name]
            transcript_text = text_map.get(sample_id)
            if not transcript_text:
                continue
            config_tasks.append((config_name, transcript_text, source))

        config_workers = max(1, min(args.max_workers, len(config_tasks))) if config_tasks else 1
        with ThreadPoolExecutor(max_workers=config_workers) as executor:
            future_map = {
                executor.submit(
                    evaluate_recoverability_config,
                    dataset_id=sample_id,
                    config_name=config_name,
                    transcript_text=transcript_text,
                    transcript_source=source,
                    reference_profile=reference_profiles[sample_id],
                    api_key=api_key,
                    model=args.model,
                    output_root=recoverability_root,
                    max_fact_workers=args.max_fact_workers,
                    overwrite=args.overwrite,
                ): config_name
                for config_name, transcript_text, source in config_tasks
            }
            for future in as_completed(future_map):
                config_name = future_map[future]
                sample_payloads[config_name] = future.result()
        return sample_id, sample_payloads

    with ThreadPoolExecutor(max_workers=sample_workers) as executor:
        future_map = {executor.submit(_process_sample, sample_id): sample_id for sample_id in sorted(reference_profiles.keys())}
        for future in as_completed(future_map):
            sample_id, sample_payloads = future.result()
            payloads_by_sample[sample_id] = sample_payloads
            print(f"Evaluated recoverability {sample_id}")

    summary_payload = {
        "created_at": now_iso(),
        "model": args.model,
        "original_csv": str(args.original_csv),
        "reference_profiles_path": str(all_profiles_path),
        "sample_ids": sorted(reference_profiles.keys()),
        "config_sources": {config_name: source for config_name, _, source in configs},
        "summary": aggregate_recoverability_summary(
            payloads_by_sample,
            config_order=ordered_configs,
        ),
    }

    summary_path = recoverability_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print_recoverability_console_summary(summary_payload, config_order=ordered_configs)
    print("")
    print(f"Saved reference profiles: {all_profiles_path}")
    print(f"Saved recoverability summary: {summary_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze re-id profiles or transcript recoverability."
    )
    parser.add_argument(
        "--workflow",
        choices=["reid_compare", "profile_recoverability"],
        default="reid_compare",
    )
    parser.add_argument(
        "--input",
        action="append",
        help="Repeatable input spec. Format: label=path/to/reid.json or just path/to/reid.json.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        help="Case-insensitive transcript ID to analyze. Repeat to pass multiple IDs. Default: shared IDs across all inputs.",
    )
    parser.add_argument("--reference-label", help="Config label to use as preservation reference.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample-workers", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-fact-workers", type=int, default=8)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional limit on candidates per sample/config after sorting by confidence.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument("--original-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--config-set", choices=["core", "diff_report"], default="diff_report")
    parser.add_argument("--adaptive-path", type=Path, default=None)
    parser.add_argument("--anonymized-path", type=Path, default=None)
    parser.add_argument("--nobranch-path", type=Path, default=None)
    parser.add_argument("--on-device-path", type=Path, default=None)
    parser.add_argument("--on-device-qwen-path", type=Path, default=None)
    parser.add_argument("--remove2-path", type=Path, default=None)
    parser.add_argument("--remove4-path", type=Path, default=None)
    parser.add_argument("--pure-adaptive-attri-path", type=Path, default=None)
    parser.add_argument("--presidio-path", type=Path, default=None)
    parser.add_argument("--rewritten-v1-path", type=Path, default=None)
    parser.add_argument("--rewritten-v2-path", type=Path, default=None)
    parser.add_argument("--dpmlm-path", type=Path, default=None)
    parser.add_argument("--dpmlm-10-path", type=Path, default=None)
    parser.add_argument("--dpmlm-30-path", type=Path, default=None)
    parser.add_argument("--dpmlm-50-path", type=Path, default=None)
    parser.add_argument("--dpmlm-70-path", type=Path, default=None)
    parser.add_argument("--only-configs", nargs="*")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--skip-reference-build", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if load_dotenv is not None:
        for candidate in (
            REPO_ROOT / ".env",
            REPO_ROOT.parent / "AURA" / ".env",
            REPO_ROOT.parent / ".env",
        ):
            if candidate.exists():
                load_dotenv(candidate)
                break

    _resolve_api_key(args.model)
    if args.workflow == "profile_recoverability":
        return run_profile_recoverability(args)
    return run_reid_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
