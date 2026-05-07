"""Phase 2a: Attacker — adversarial inference on rewritten text.

The attacker only sees the rewritten text (simulating an external adversary)
and attempts to infer the 8 PII attributes. Its output is compared with the
original inferences to identify residual privacy leaks.
"""
from __future__ import annotations

import json
import re

import pipeline_config as cfg
from phase0_init import (
    PRIVACY_ATTRIBUTES,
    INFERENCE_SYSTEM_PROMPT,
    build_privacy_inference_prompt,
)

ATTACKER_SYSTEM_PROMPT = INFERENCE_SYSTEM_PROMPT

VULNERABILITY_SYSTEM_PROMPT = (
    "You are a privacy auditor. Compare the attacker's inferences on the "
    "rewritten text against the original inferences. Identify which attributes "
    "are still leaking and extract the exact spans from the rewritten text that "
    "enable each inference. Output valid JSON only."
)

_COMMON_TOKENS = {
    "about", "after", "again", "against", "among", "because", "before", "being",
    "between", "could", "doing", "during", "first", "from", "have", "their",
    "there", "these", "those", "through", "under", "using", "with", "would",
    "where", "which", "while", "work", "research", "project", "study",
}

ATTRIBUTE_DIMENSIONS = [
    {
        "key": "OCCUPATION",
        "label": "Occupational Domain",
        "too_specific_example": "teaches cooking classes at a local Italian supermarket chain",
        "just_right_example": "works in food education and catering",
        "too_vague_example": "works in a service industry",
    },
    {
        "key": "EXPERTISE",
        "label": "Expertise & Experience Level",
        "too_specific_example": "15-year veteran chef who trained at Le Cordon Bleu",
        "just_right_example": "experienced professional with deep domain expertise",
        "too_vague_example": "someone with work experience",
    },
    {
        "key": "AI_USAGE",
        "label": "AI Interaction Context",
        "too_specific_example": "uses Claude to write menus for wedding catering events",
        "just_right_example": "uses AI for professional writing and content creation in their field",
        "too_vague_example": "uses AI at work",
    },
    {
        "key": "REGION_TYPE",
        "label": "Geographic/Cultural Context",
        "too_specific_example": "lives in St. John's, Newfoundland and references local festivals",
        "just_right_example": "based in a coastal Canadian city with strong local food culture",
        "too_vague_example": "lives somewhere in North America",
    },
    {
        "key": "LIFE_STAGE",
        "label": "Life Stage & Demographics",
        "too_specific_example": "42-year-old married father of two who recently changed careers",
        "just_right_example": "mid-career working adult with a family",
        "too_vague_example": "an adult",
    },
]

SPECIFICITY_SYSTEM_PROMPT = (
    "You are a privacy specificity auditor. Judge whether the rewritten transcript "
    "still reveals participant attributes at a too-specific level. Use the provided "
    "dimension definitions and examples. Output valid JSON only."
)


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _extract_evidence_spans(original_inferences: dict | None) -> list[str]:
    """Collect high-certainty evidence spans from original inferences."""
    if not isinstance(original_inferences, dict):
        return []

    spans: list[str] = []
    threshold = cfg.CERTAINTY_THRESHOLD_FOR_BLACKLIST
    for attr in PRIVACY_ATTRIBUTES:
        info = original_inferences.get(attr.key, {})
        if not isinstance(info, dict):
            continue
        certainty = info.get("certainty", 1)
        try:
            certainty = int(certainty)
        except (TypeError, ValueError):
            certainty = 1
        if certainty < threshold:
            continue
        for span in info.get("evidence_spans", []) or []:
            s = str(span).strip()
            if len(s) >= 3:
                spans.append(s)
    return spans


def _assess_span_leakage(
    rewritten_text: str,
    original_inferences: dict | None = None,
) -> dict:
    """Measure leakage of high-certainty evidence spans in rewrite."""
    candidate_spans = _extract_evidence_spans(original_inferences)

    deduped: list[str] = []
    seen = set()
    for s in candidate_spans:
        key = s.lower()
        if key not in seen and len(s) >= 3:
            seen.add(key)
            deduped.append(s)

    rewritten_norm = _normalize_text(rewritten_text)
    rewritten_tokens = set(re.findall(r"[a-z0-9]+", rewritten_norm))

    leaks: list[dict] = []
    total = 0

    for span in deduped:
        span_norm = _normalize_text(span)
        if not span_norm:
            continue

        exact_present = span_norm in rewritten_norm
        span_tokens = [
            t for t in re.findall(r"[a-z0-9]+", span_norm)
            if len(t) >= 6 and t not in _COMMON_TOKENS
        ]
        unique_tokens = sorted(set(span_tokens))

        overlap = 0.0
        if unique_tokens:
            overlap = (
                sum(1 for t in unique_tokens if t in rewritten_tokens) / len(unique_tokens)
            )

        severity = 0
        reason = ""
        if exact_present:
            severity = 3
            reason = "exact_span_present"
        elif len(unique_tokens) >= 4 and overlap >= 0.80:
            severity = 2
            reason = "high_token_overlap"
        elif len(unique_tokens) >= 5 and overlap >= 0.60:
            severity = 1
            reason = "partial_token_overlap"

        if severity > 0:
            total += severity
            leaks.append(
                {
                    "span": span,
                    "severity": severity,
                    "reason": reason,
                    "token_overlap": round(overlap, 3),
                }
            )

    return {
        "source_span_count": len(deduped),
        "num_span_leaks": len(leaks),
        "total_span_severity": total,
        "leaks": leaks,
    }


def build_vulnerability_prompt(
    original_inferences: dict,
    rewrite_inferences: dict,
    rewritten_text: str,
) -> str:
    return (
        "Compare these two inference reports and identify privacy vulnerabilities.\n\n"
        "ORIGINAL INFERENCES (from the unprotected text):\n"
        f"{json.dumps(original_inferences, indent=2)}\n\n"
        "REWRITE INFERENCES (from the rewritten text):\n"
        f"{json.dumps(rewrite_inferences, indent=2)}\n\n"
        f"REWRITTEN TEXT:\n{rewritten_text}\n\n"
        "For each of the attributes, determine:\n"
        "- leaked: true/false (did the rewrite fail to neutralize this attribute?)\n"
        "- certainty_delta: (original_certainty - rewrite_certainty)\n"
        "- evidence_spans: exact verbatim spans from REWRITTEN TEXT that enable inference\n"
        "- severity: 1-5 (5 = attribute fully exposed, 1 = effectively neutralized)\n\n"
        "Return JSON:\n{\n"
        '  "AGE": {"leaked": bool, "certainty_delta": int, "severity": int, '
        '"evidence_spans": [...], "explanation": "..."},\n'
        "  ... (all 8 attributes)\n"
        '  "total_severity": <sum of all severity scores>\n'
        "}\n"
    )


def build_specificity_prompt(rewritten_text: str) -> str:
    dims = "\n".join(
        (
            f'- {d["key"]} ({d["label"]})\n'
            f'  Too specific example: "{d["too_specific_example"]}"\n'
            f'  Just right example: "{d["just_right_example"]}"\n'
            f'  Too vague example: "{d["too_vague_example"]}"'
        )
        for d in ATTRIBUTE_DIMENSIONS
    )
    return (
        "Evaluate whether this rewritten transcript is still too specific on the "
        "dimensions below.\n\n"
        "Mark too_specific=true only when details are specific enough to materially "
        "increase identity risk. Mild abstraction loss is acceptable.\n\n"
        f"Dimensions:\n{dims}\n\n"
        f"=== REWRITTEN TRANSCRIPT ===\n{rewritten_text}\n=== END REWRITTEN TRANSCRIPT ===\n\n"
        "Return JSON with this structure:\n"
        "{\n"
        '  "dimensions": {\n'
        '    "OCCUPATION": {"too_specific": bool, "rationale": "...", "evidence_spans": ["..."]},\n'
        '    "EXPERTISE": {"too_specific": bool, "rationale": "...", "evidence_spans": ["..."]},\n'
        '    "AI_USAGE": {"too_specific": bool, "rationale": "...", "evidence_spans": ["..."]},\n'
        '    "REGION_TYPE": {"too_specific": bool, "rationale": "...", "evidence_spans": ["..."]},\n'
        '    "LIFE_STAGE": {"too_specific": bool, "rationale": "...", "evidence_spans": ["..."]}\n'
        "  },\n"
        '  "too_specific_count": <int>\n'
        "}\n"
    )


def _call_llm_json(client, model: str, system: str, user: str, max_tokens: int | None = None) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=max_tokens or cfg.JSON_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    raw = cfg.strip_think_tags(resp.choices[0].message.content or "{}")
    return json.loads(raw)


def run_attack(rewritten_text: str, client=None) -> dict:
    """Infer attributes from rewritten text (adversary perspective)."""
    client = client or cfg.get_pipeline_client()
    prompt = build_privacy_inference_prompt(rewritten_text)
    return _call_llm_json(client, cfg.ATTACKER_MODEL, ATTACKER_SYSTEM_PROMPT, prompt)


def compare_inferences(
    original_inferences: dict,
    rewrite_inferences: dict,
    rewritten_text: str,
    client=None,
) -> dict:
    """Produce a structured vulnerability report comparing before/after inferences."""
    client = client or cfg.get_pipeline_client()
    prompt = build_vulnerability_prompt(
        original_inferences, rewrite_inferences, rewritten_text
    )
    return _call_llm_json(
        client, cfg.ATTACKER_MODEL, VULNERABILITY_SYSTEM_PROMPT, prompt
    )


def _normalize_specificity_report(raw: dict | None) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    raw_dims = raw.get("dimensions", {})
    if not isinstance(raw_dims, dict):
        raw_dims = {}

    dims: dict = {}
    too_specific_count = 0
    for dim in ATTRIBUTE_DIMENSIONS:
        key = dim["key"]
        info = raw_dims.get(key, {})
        if not isinstance(info, dict):
            info = {}
        too_specific = bool(info.get("too_specific", False))
        rationale = str(info.get("rationale") or "").strip()
        evidence = info.get("evidence_spans") or []
        if not isinstance(evidence, list):
            evidence = []
        evidence = [str(s).strip() for s in evidence if str(s).strip()]
        dims[key] = {
            "too_specific": too_specific,
            "rationale": rationale,
            "evidence_spans": evidence,
        }
        if too_specific:
            too_specific_count += 1

    if isinstance(raw.get("too_specific_count"), int):
        too_specific_count = max(too_specific_count, int(raw["too_specific_count"]))

    return {
        "dimensions": dims,
        "too_specific_count": too_specific_count,
    }


def run_specificity_audit(rewritten_text: str, client=None) -> dict:
    client = client or cfg.get_pipeline_client()
    prompt = build_specificity_prompt(rewritten_text)
    try:
        raw = _call_llm_json(
            client,
            cfg.ATTACKER_MODEL,
            SPECIFICITY_SYSTEM_PROMPT,
            prompt,
        )
    except Exception as exc:
        print(f"  Specificity audit failed: {exc}")
        raw = {}
    return _normalize_specificity_report(raw)


def attack_and_report(
    rewritten_text: str,
    original_inferences: dict,
    client=None,
) -> dict:
    """Full attacker pipeline: infer on rewrite, then compare with original.

    Returns {"rewrite_inferences": {...}, "vulnerability_report": {...}}.
    """
    client = client or cfg.get_pipeline_client()
    rewrite_inferences = run_attack(rewritten_text, client)
    vulnerability_report = compare_inferences(
        original_inferences, rewrite_inferences, rewritten_text, client
    )
    specificity_report = run_specificity_audit(rewritten_text, client)
    span_leakage = _assess_span_leakage(
        rewritten_text=rewritten_text,
        original_inferences=original_inferences,
    )
    if isinstance(vulnerability_report, dict):
        vulnerability_report["SPAN_LEAKAGE"] = span_leakage

    return {
        "rewrite_inferences": rewrite_inferences,
        "vulnerability_report": vulnerability_report,
        "specificity_report": specificity_report,
    }


def total_severity(report: dict) -> int:
    """Extract total severity score from a vulnerability report."""
    vr = report.get("vulnerability_report", report)
    if "total_severity" in vr:
        try:
            return int(vr["total_severity"])
        except (ValueError, TypeError):
            pass
    total = 0
    for attr in PRIVACY_ATTRIBUTES:
        info = vr.get(attr.key, {})
        if isinstance(info, dict):
            try:
                total += int(info.get("severity", 0))
            except (ValueError, TypeError):
                pass
    span_leak = vr.get("SPAN_LEAKAGE", {})
    if isinstance(span_leak, dict):
        try:
            total += int(span_leak.get("total_span_severity", 0))
        except (ValueError, TypeError):
            pass
    return total
