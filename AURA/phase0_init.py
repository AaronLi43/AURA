"""Phase 0: Load transcripts and build privacy + topic-aware utility profiles."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pipeline_config as cfg
import db

# ── Privacy attribute specs (mirrored from Advance_Anonymizer) ─────────

@dataclass(frozen=True)
class PrivacyAttribute:
    key: str
    display_name: str
    target_str: str
    options: list[str] | None = None
    note: str | None = None


PRIVACY_ATTRIBUTES: list[PrivacyAttribute] = [
    PrivacyAttribute("AGE", "Age", "age", note="Use the age of the author when they wrote the transcript."),
    PrivacyAttribute("SEX", "Sex", "sex", options=["Male", "Female"]),
    PrivacyAttribute("LOC", "Location", "current place of living"),
    PrivacyAttribute("OCCP", "Occupation", "occupation"),
    PrivacyAttribute("EDU", "Education", "level of education",
                     options=["No Highschool", "In Highschool", "HS Diploma", "In College", "College Degree", "PhD"]),
    PrivacyAttribute("REL", "Relationship Status", "current relationship status",
                     options=["No relation", "In Relation", "Married", "Divorced"]),
    PrivacyAttribute("INC", "Income", "yearly income",
                     options=["No income", "Low (<30k USD)", "Medium (30-60k USD)", "High (60-150k USD)", "Very High (>150k USD)"]),
    PrivacyAttribute("POBP", "Place of Birth", "place of birth"),
]

# ── Utility attribute specs ────────────────────────────────────────────

@dataclass(frozen=True)
class UtilityAttribute:
    key: str
    display_name: str
    description: str


UTILITY_ATTRIBUTES: list[UtilityAttribute] = [
    UtilityAttribute("THEME", "Thematic Content",
                     "Core topics, recurring themes, and subject domains discussed"),
    UtilityAttribute("EXPER", "Experiential Narratives",
                     "Specific events, stories, anecdotes, and lived experiences"),
    UtilityAttribute("AFFECT", "Emotional/Affective Expressions",
                     "Feelings, attitudes, frustrations, and satisfactions expressed"),
    UtilityAttribute("REASON", "Reasoning & Beliefs",
                     "Opinions, justifications, decision rationale, and values"),
    UtilityAttribute("BEHAV", "Behavioral Patterns",
                     "Workflows, habits, routines, and practices described"),
    UtilityAttribute("RELAT", "Relational Dynamics",
                     "Interactions with others, social roles, and group dynamics"),
    UtilityAttribute("TEMPO", "Temporal Structure",
                     "Chronology, development over time, turning points, and before/after comparisons"),
    UtilityAttribute("EXPERT", "Domain Knowledge",
                     "Professional/technical insights, competencies, and specialized vocabulary"),
]

# ── Prompts ────────────────────────────────────────────────────────────

INFERENCE_SYSTEM_PROMPT = (
    "You are an expert investigator with years of experience in online profiling "
    "and text analysis. You work with an analytical mindset and try to answer "
    "questions as precisely as possible."
)


def build_privacy_inference_prompt(transcript: str) -> str:
    attr_names = ", ".join(a.display_name for a in PRIVACY_ATTRIBUTES)
    footer = (
        "First reason step-by-step for each attribute going over all information "
        "in the text as well as language used. Then give your top 3 guesses for "
        "each attribute. Also identify the exact verbatim spans from the text that "
        "most contributed to each inference.\n\n"
        "Return valid JSON with this structure:\n"
        "{\n"
    )
    for a in PRIVACY_ATTRIBUTES:
        opts = ""
        if a.options:
            opts = f'  Options: {", ".join(a.options)}.'
        note = f"  {a.note}" if a.note else ""
        footer += (
            f'  "{a.key}": {{\n'
            f'    "inference": "your detailed reasoning",\n'
            f'    "guesses": ["guess1", "guess2", "guess3"],{opts}{note}\n'
            f'    "certainty": <1-5>,\n'
            f'    "evidence_spans": ["exact span 1", "exact span 2"]\n'
            f"  }},\n"
        )
    footer += "}\n"

    return (
        f"Below is an interview transcript. Your task is to infer the interviewee's "
        f"{attr_names} from the text.\n\n"
        f"=== TRANSCRIPT ===\n{transcript}\n=== END TRANSCRIPT ===\n\n"
        f"{footer}"
    )


UTILITY_SYSTEM_PROMPT = (
    "You are an expert qualitative researcher specializing in thematic analysis "
    "of interview data. You identify and catalogue the research-valuable content "
    "in transcripts with precision."
)

TOPIC_SYSTEM_PROMPT = (
    "You are a qualitative methods specialist. Identify the PRIMARY interview "
    "topic and distinguish it from contextual domain details. Your output is used "
    "to preserve topic-relevant insight while generalizing unnecessary specifics."
)

FINGERPRINT_SYSTEM_PROMPT = (
    "You are a re-identification risk analyst. Identify concrete phrases or "
    "phrase combinations in a transcript that could be searched online to reveal "
    "the speaker's identity."
)


def build_fingerprint_prompt(transcript: str) -> str:
    return (
        "Below is an interview transcript.\n\n"
        "Identify the top 5-10 re-identification fingerprints: specific verbatim "
        "phrases or phrase combinations that, when searched online or combined, "
        "could identify the speaker.\n\n"
        "Focus on highly distinctive technical content, unique project details, "
        "rare named entities, and unusual combinations.\n\n"
        "Return valid JSON only with this structure:\n"
        "{\n"
        '  "fingerprints": [\n'
        '    "verbatim phrase or short combination 1",\n'
        '    "verbatim phrase or short combination 2"\n'
        "  ]\n"
        "}\n\n"
        f"=== TRANSCRIPT ===\n{transcript}\n=== END TRANSCRIPT ==="
    )


def build_topic_focus_prompt(transcript: str) -> str:
    return (
        "Below is an interview transcript.\n\n"
        "Infer the main analytic topic of this interview and define what should be "
        "preserved vs generalized for anonymized analysis.\n\n"
        "Return valid JSON only with this structure:\n"
        "{\n"
        '  "primary_topic": "short phrase describing the main research topic",\n'
        '  "secondary_context": "domain/work context that supports but is not the main topic",\n'
        '  "preserve_focus": ["content categories that should be preserved in detail"],\n'
        '  "generalize_focus": ["content categories that should be abstracted/generalized"],\n'
        '  "rationale": "brief rationale"\n'
        "}\n\n"
        f"=== TRANSCRIPT ===\n{transcript}\n=== END TRANSCRIPT ==="
    )


def _format_topic_focus(topic_focus: dict | None) -> str:
    if not isinstance(topic_focus, dict):
        return "(not available)"

    primary = str(topic_focus.get("primary_topic") or "").strip() or "(unspecified)"
    secondary = str(topic_focus.get("secondary_context") or "").strip()

    preserve = topic_focus.get("preserve_focus") or []
    if not isinstance(preserve, list):
        preserve = [str(preserve)]
    preserve = [str(x).strip() for x in preserve if str(x).strip()]

    generalize = topic_focus.get("generalize_focus") or []
    if not isinstance(generalize, list):
        generalize = [str(generalize)]
    generalize = [str(x).strip() for x in generalize if str(x).strip()]

    lines = [f"Primary topic: {primary}"]
    if secondary:
        lines.append(f"Secondary context: {secondary}")
    if preserve:
        lines.append("Preserve focus: " + "; ".join(preserve))
    if generalize:
        lines.append("Generalize focus: " + "; ".join(generalize))
    return "\n".join(lines)


def build_utility_summary_prompt(transcript: str, topic_focus: dict | None = None) -> str:
    topic_context = _format_topic_focus(topic_focus)
    footer = (
        "For each utility attribute below, extract a structured summary of what "
        "the transcript contains. Include the key content, specific examples, and "
        "the exact verbatim spans that carry the most research value.\n\n"
        "IMPORTANT: prioritize content that supports the PRIMARY TOPIC and "
        "PRESERVE FOCUS. For content in GENERALIZE FOCUS, treat detailed domain "
        "specifics as lower-priority utility unless required to understand the "
        "topic-relevant AI/workflow insight.\n\n"
        "Return valid JSON with this structure:\n"
        "{\n"
    )
    for a in UTILITY_ATTRIBUTES:
        footer += (
            f'  "{a.key}": {{\n'
            f'    "attribute": "{a.display_name}",\n'
            f'    "summary": "brief description of what the transcript reveals for {a.description}",\n'
            f'    "key_spans": ["verbatim span 1", "verbatim span 2", ...],\n'
            f'    "richness": <1-5 how much content exists for this attribute>\n'
            f"  }},\n"
        )
    footer += "}\n"

    return (
        f"Below is an interview transcript from a qualitative research study. "
        f"Analyze it to create an insight profile cataloguing the research-valuable content.\n\n"
        f"=== TOPIC FOCUS ===\n{topic_context}\n=== END TOPIC FOCUS ===\n\n"
        f"=== TRANSCRIPT ===\n{transcript}\n=== END TRANSCRIPT ===\n\n"
        f"{footer}"
    )


# ── LLM helpers ────────────────────────────────────────────────────────

def _call_llm_json(client, model: str, system: str, user: str, max_tokens: int | None = None) -> dict:
    budget = max_tokens or cfg.JSON_MAX_TOKENS
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=budget,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    parsed = cfg.safe_json_loads(raw)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"items": parsed}
    return {}


def run_privacy_inference(transcript: str, client=None) -> dict:
    client = client or cfg.get_pipeline_client()
    prompt = build_privacy_inference_prompt(transcript)
    return _call_llm_json(client, cfg.INIT_MODEL, INFERENCE_SYSTEM_PROMPT, prompt)


def run_topic_focus(transcript: str, client=None) -> dict:
    client = client or cfg.get_pipeline_client()
    prompt = build_topic_focus_prompt(transcript)
    return _call_llm_json(client, cfg.INIT_MODEL, TOPIC_SYSTEM_PROMPT, prompt)


def run_utility_summary(transcript: str, topic_focus: dict | None = None, client=None) -> dict:
    client = client or cfg.get_pipeline_client()
    prompt = build_utility_summary_prompt(transcript, topic_focus=topic_focus)
    return _call_llm_json(client, cfg.INIT_MODEL, UTILITY_SYSTEM_PROMPT, prompt)


def run_reid_fingerprints(transcript: str, client=None) -> list[str]:
    client = client or cfg.get_pipeline_client()
    prompt = build_fingerprint_prompt(transcript)
    payload = _call_llm_json(client, cfg.INIT_MODEL, FINGERPRINT_SYSTEM_PROMPT, prompt)
    raw = payload.get("fingerprints", []) if isinstance(payload, dict) else []
    if not isinstance(raw, list):
        return []

    fingerprints: list[str] = []
    seen = set()
    for item in raw:
        text = str(item).strip()
        if len(text) < 3:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        fingerprints.append(text)
    return fingerprints[:10]


def extract_evidence_spans_from_inferences(inferences: dict) -> list[str]:
    """Extract high-certainty evidence spans for downstream privacy checks."""
    spans: list[str] = []
    threshold = cfg.CERTAINTY_THRESHOLD_FOR_BLACKLIST
    for attr in PRIVACY_ATTRIBUTES:
        info = inferences.get(attr.key, {})
        certainty = info.get("certainty", 1)
        try:
            certainty = int(certainty)
        except (ValueError, TypeError):
            certainty = 1
        if certainty >= threshold:
            for span in info.get("evidence_spans", []):
                s = str(span).strip()
                if s and len(s) >= 3:
                    spans.append(s)
    seen = set()
    deduped = []
    for s in spans:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    return deduped


def initialize_document(document_id: str, transcript: str):
    """Run privacy inference + topic focus + topic-aware utility summary."""
    client = cfg.get_pipeline_client()

    with ThreadPoolExecutor(max_workers=3) as executor:
        priv_fut = executor.submit(run_privacy_inference, transcript, client)
        topic_fut = executor.submit(run_topic_focus, transcript, client)
        fingerprint_fut = executor.submit(run_reid_fingerprints, transcript, client)
        privacy_inferences = priv_fut.result()
        topic_focus = topic_fut.result()
        reid_fingerprints = fingerprint_fut.result()

    insight_profile = run_utility_summary(transcript, topic_focus=topic_focus, client=client)
    if isinstance(insight_profile, dict):
        insight_profile["__topic_focus__"] = topic_focus
        insight_profile["__reid_fingerprints__"] = reid_fingerprints

    evidence_spans = extract_evidence_spans_from_inferences(privacy_inferences)

    db.upsert_document(
        document_id,
        original_text=transcript,
        evidence_spans=evidence_spans,
        insight_profile=insight_profile,
        privacy_inferences=privacy_inferences,
        status="initialized",
    )
    print(f"  {document_id}: evidence_span_count={len(evidence_spans)}")
    return evidence_spans, insight_profile, privacy_inferences


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 0: Initialize documents")
    parser.add_argument("--input", type=Path, default=cfg.INPUT_JSONL)
    parser.add_argument("--ids", type=str, help="Comma-separated document IDs")
    parser.add_argument("--reset-db", action="store_true")
    args = parser.parse_args()

    if args.reset_db:
        if cfg.DB_PATH.exists():
            cfg.DB_PATH.unlink()
            print("Removed existing DB.")
    db.init_db()

    with open(args.input, encoding="utf-8") as f:
        records = {}
        for line in f:
            rec = json.loads(line)
            cid = rec.get("conversation_id", "")
            text = rec.get("user_message", "")
            if cid and text:
                records[cid] = text

    print(f"Loaded {len(records)} records from {args.input}")

    if args.ids:
        target_ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    else:
        target_ids = list(records.keys())

    missing = [i for i in target_ids if i not in records]
    if missing:
        print(f"WARNING: IDs not found in input: {missing}")
        target_ids = [i for i in target_ids if i in records]

    print(f"Initializing {len(target_ids)} document(s)...")
    for doc_id in target_ids:
        initialize_document(doc_id, records[doc_id])

    print("Phase 0 complete.")


if __name__ == "__main__":
    main()
