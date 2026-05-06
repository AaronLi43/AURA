"""Phase 1b: Refiller — rewrite masked spans guided by insight profile."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pipeline_config as cfg

REFILLER_SYSTEM_PROMPT = """\
You are a text refiller for privacy protection.
You receive a masked template with [MASK_i] tokens and must rewrite ONLY those
masked spans to improve fluency and recover safe detail.

=== HARD CONSTRAINTS ===
- You may output replacements for mask keys only.
- You must not modify any non-masked text.
- You must not restore cross-referenceable details that can re-identify a
  person through web search or paper/project lookup.
- Specifically avoid restoring detailed research pipeline signatures:
  method sequences, named instruments/software, named protocols, exact setup
  variants, unique experiment combinations, paper-level fingerprints.
- Do not leave [MASK_i] tokens unresolved.
- Do not invent claims or events not implied by context.

=== REFINEMENT GOAL ===
- Bring masked spans closer to the original wording only when safe.
- If fidelity conflicts with privacy, prioritize privacy.
- Slight abstraction/loss is acceptable for these dimensions:
  OCCUPATION, EXPERTISE, AI_USAGE, REGION_TYPE, LIFE_STAGE.
- Keep the transcript coherent and preserve high-level reasoning, emotion, and
  qualitative insight.

Return JSON only:
{"MASK_1": "replacement text", "MASK_2": "replacement text", ...}
"""

REFILLER_USER_PROMPT = """\
=== INPUT ===
TEMPLATE:
{template}

ORIGINAL MASK MAP (what each token replaced — DO NOT reuse these):
{mask_map}

SEED REPLACEMENTS (from rewrite-first masker; may use or improve):
{seed_replacements}

INSIGHT PROFILE (preserve research value in these dimensions):
{insight_profile}

=== ADAPTIVE RULES ===
{adaptive_rules}

Generate ONE replacement dictionary as JSON: {{"MASK_1": "replacement", ...}}
Rewrite ONLY masked spans and leave non-masked text untouched.
Use SEED REPLACEMENTS as the starting point, then refine safely.
Do not restore cross-referenceable pipeline details.
Prefer general category-level wording for sensitive spans that could enable
identity inference.
"""


def _extract_json(content: str | None) -> dict | None:
    if not content or not content.strip():
        return None
    text = content.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text, i)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def _format_insight_profile(profile: dict | None) -> str:
    if not profile:
        return "(not available)"

    topic_focus = None
    if isinstance(profile, dict):
        maybe_topic = profile.get("__topic_focus__")
        if isinstance(maybe_topic, dict):
            topic_focus = maybe_topic

    parts = []
    if topic_focus:
        primary = str(topic_focus.get("primary_topic") or "").strip() or "(unspecified)"
        secondary = str(topic_focus.get("secondary_context") or "").strip()
        preserve = topic_focus.get("preserve_focus") or []
        generalize = topic_focus.get("generalize_focus") or []
        if not isinstance(preserve, list):
            preserve = [str(preserve)]
        if not isinstance(generalize, list):
            generalize = [str(generalize)]
        preserve = [str(x).strip() for x in preserve if str(x).strip()]
        generalize = [str(x).strip() for x in generalize if str(x).strip()]

        parts.append("TOPIC FOCUS:")
        parts.append(f"- Primary topic: {primary}")
        if secondary:
            parts.append(f"- Secondary context: {secondary}")
        if preserve:
            parts.append("- Preserve focus: " + "; ".join(preserve))
        if generalize:
            parts.append("- Generalize focus: " + "; ".join(generalize))
        parts.append("")

    for key, info in profile.items():
        if key == "__topic_focus__":
            continue
        if not isinstance(info, dict):
            continue
        attr = info.get("attribute", key)
        richness = info.get("richness", "?")
        summary = info.get("summary", "")
        parts.append(f"- {attr} (richness={richness}): {summary}")
    return "\n".join(parts) if parts else "(not available)"


def _format_adaptive_rules(rules: list[str] | None) -> str:
    if not rules:
        return "(no rules yet)"
    return "\n".join(f"[{i+1}] {r}" for i, r in enumerate(rules))


def _generate_single(
    template: str,
    mask_map: dict,
    seed_map: dict | None,
    insight_profile: dict | None,
    adaptive_rules: list[str] | None,
    client,
) -> dict | None:
    profile_str = _format_insight_profile(insight_profile)
    rules_str = _format_adaptive_rules(adaptive_rules)
    mask_map_str = json.dumps(mask_map, indent=2)
    seed_map_str = json.dumps(seed_map or {}, indent=2)

    try:
        resp = client.chat.completions.create(
            model=cfg.REFILLER_MODEL,
            messages=[
                {"role": "system", "content": REFILLER_SYSTEM_PROMPT},
                {"role": "user", "content": REFILLER_USER_PROMPT.format(
                    template=template,
                    mask_map=mask_map_str,
                    seed_replacements=seed_map_str,
                    insight_profile=profile_str,
                    adaptive_rules=rules_str,
                )},
            ],
            temperature=0.7,
            max_tokens=cfg.JSON_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        print(f"  Refiller call failed: {exc}")
        return None

    content = cfg.strip_think_tags(resp.choices[0].message.content or "")
    return _extract_json(content)


def assemble(template: str, fill_dict: dict) -> str:
    """Substitute [MASK_i] tokens with their replacements."""
    result = template
    for key, val in fill_dict.items():
        token = f"[{key}]" if not key.startswith("[") else key
        result = result.replace(token, str(val))
    residual = re.findall(r"\[MASK_\d+\]", result)
    if residual:
        print(f"  WARNING: {len(residual)} residual mask tokens remain")
    return result


def generate_variations(
    template: str,
    mask_map: dict,
    seed_map: dict | None = None,
    insight_profile: dict | None = None,
    adaptive_rules: list[str] | None = None,
    n: int | None = None,
    client=None,
) -> list[dict]:
    """Generate N refill variations in parallel.

    Returns list of {"fill_dict": {...}, "assembled_text": "..."}.
    """
    client = client or cfg.get_pipeline_client()
    n = n or cfg.VARIATIONS_PER_ROUND

    results = [None] * n
    workers = min(n, cfg.REFILLER_MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _generate_single, template, mask_map,
                seed_map, insight_profile, adaptive_rules, client
            ): idx
            for idx in range(n)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            fill_dict = fut.result()
            if fill_dict:
                assembled = assemble(template, fill_dict)
                results[idx] = {"fill_dict": fill_dict, "assembled_text": assembled}

    return [r for r in results if r is not None]
