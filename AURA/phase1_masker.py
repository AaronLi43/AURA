"""Phase 1a: Masker — rewrite text then derive [MASK_i] spans via diff."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import pipeline_config as cfg


@dataclass(frozen=True)
class AttributeSpec:
    key: str
    type_name: str
    display_name: str
    target_attribute_str: str
    options: list[str] | None = None
    special_note: str | None = None
    aliases: tuple[str, ...] = ()


ATTRIBUTE_SPECS: list[AttributeSpec] = [
    AttributeSpec(
        "AGE",
        "age",
        "Age",
        "age",
        special_note="Use the age of the author when they wrote the transcript.",
    ),
    AttributeSpec(
        "SEX",
        "gender",
        "Sex",
        "sex",
        options=["Male", "Female"],
        aliases=("sex",),
    ),
    AttributeSpec(
        "LOC",
        "location",
        "Location",
        "current place of living",
        aliases=("place of living",),
    ),
    AttributeSpec("OCCP", "occupation", "Occupation", "occupation"),
    AttributeSpec(
        "EDU",
        "education",
        "Education",
        "level of education",
        options=["No Highschool", "In Highschool", "HS Diploma", "In College", "College Degree", "PhD"],
    ),
    AttributeSpec(
        "REL",
        "married",
        "Relationship Status",
        "current relationship status",
        options=["No relation", "In Relation", "Married", "Divorced"],
        aliases=("relationship status", "relationship", "marital status"),
    ),
    AttributeSpec(
        "INC",
        "income",
        "Income",
        "yearly income",
        options=["No income", "Low (<30k USD)", "Medium (30-60k USD)", "High (60-150k USD)", "Very High (>150k USD)"],
    ),
    AttributeSpec(
        "POBP",
        "pobp",
        "Place of Birth",
        "place of birth",
        aliases=("birth place", "birth location"),
    ),
]

INFERENCE_SYSTEM_PROMPT = (
    "You are an expert investigator with years of experience in online profiling and text analysis. "
    "You work with an analytical mindset and try to answer questions as precisely as possible."
)

MASKER_SYSTEM_PROMPT = """\
You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous.
You follow the instructions and format precisely and you try to change as little as possible,
keeping the original text intact as much as possible. Only generalize information and do not
invent new information.
Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not.
Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but
'my husband and I have a dog' -> 'my partner and I have a cat' is not.
Example: 'my husband and I' -> 'I' is also valid as it only removes information.
"""

MASKER_USER_PROMPT = """\
Below I give you a transcript as well as personal inferences made about the person that can
be made from the transcript. Your task is to anonymize the transcript to prevent these
inferences, while keeping the rest of the text intact.

Transcript:
{text}

Inferences:
{privacy_inferences}

In a first step briefly explain the changes you will make. In a new line write a single # and
then return the anonymized transcript. Only change relevant parts and keep everything else as
is. Make sure you keep the original meaning, i.e. do not invent new information and instead
generalize information.
"""

_NON_WS_RE = re.compile(r"\S+")


def _extract_rewritten_text(raw: str | None, fallback_text: str) -> str:
    text = (raw or "").strip()
    if not text:
        return fallback_text

    split_marker = "\n#"
    if split_marker in text:
        candidate = text.split(split_marker, 1)[1].strip()
        if candidate:
            return candidate

    if text.startswith("#"):
        candidate = text[1:].strip()
        if candidate:
            return candidate

    hash_index = text.find("\n# ")
    if hash_index != -1:
        candidate = text[hash_index + 3:].strip()
        if candidate:
            return candidate

    return text


def _format_inferences_for_prompt(privacy_inferences: dict | None) -> str:
    if not isinstance(privacy_inferences, dict) or not privacy_inferences:
        return "(not available)"

    parts: list[str] = []
    for spec in ATTRIBUTE_SPECS:
        info = privacy_inferences.get(spec.key, {})
        if not isinstance(info, dict):
            info = {}

        guesses = info.get("guesses")
        if not isinstance(guesses, list):
            guesses = info.get("guess")
        if not isinstance(guesses, list):
            guesses = []
        guesses = [str(g).strip() for g in guesses if str(g).strip()]

        inference = str(info.get("inference") or "").strip()
        certainty = info.get("certainty", 1)
        parts.extend(
            [
                f"Type: {spec.type_name}",
                f"Inference: {inference}",
                f"Guess: {', '.join(guesses) if guesses else 'Unknown'}",
                f"Certainty: {certainty}",
                "",
            ]
        )
    return "\n".join(parts).strip()


def _normalize_label(text: str) -> str:
    return "".join(ch for ch in text.strip().lower() if ch.isalnum())


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize_label(a), _normalize_label(b)).ratio()


def _is_close(a: str, b: str, threshold: float = 0.82) -> bool:
    return _ratio(a, b) >= threshold


def _match_attribute_key(text: str) -> str | None:
    token = _normalize_label(text)
    if not token:
        return None

    labels: list[tuple[str, str]] = []
    for spec in ATTRIBUTE_SPECS:
        labels.extend(
            [
                (spec.key, spec.key),
                (spec.key, spec.type_name),
                (spec.key, spec.display_name),
                (spec.key, spec.target_attribute_str),
            ]
        )
        for alias in spec.aliases:
            labels.append((spec.key, alias))

    normalized_exact = {_normalize_label(label): key for key, label in labels}
    if token in normalized_exact:
        return normalized_exact[token]

    best_key = None
    best_score = 0.0
    for key, label in labels:
        score = SequenceMatcher(None, token, _normalize_label(label)).ratio()
        if score > best_score:
            best_key = key
            best_score = score
    return best_key if best_score >= 0.62 else None


def _key_to_spec(spec_key: str) -> AttributeSpec | None:
    for spec in ATTRIBUTE_SPECS:
        if spec.key == spec_key:
            return spec
    return None


def _match_attribute_type_name(text: str) -> str | None:
    spec_key = _match_attribute_key(text)
    if spec_key is None:
        return None
    spec = _key_to_spec(spec_key)
    return spec.type_name if spec is not None else None


def build_joint_inference_prompt(transcript: str) -> str:
    pii_str = ", ".join(spec.type_name for spec in ATTRIBUTE_SPECS)
    footer = (
        "First reason step-by-step for each value going over all information in the text as well as "
        "language used. Then give your top 3 guesses for each value based on your reasoning. Follow "
        "exactly the format specified below:\n\n"
    )
    for spec in ATTRIBUTE_SPECS:
        footer += f"Type: {spec.type_name}\n"
        footer += "Inference: Your detailed inference\n"
        guess_line = "Guess: Your top 3 guesses in a list separated by ; ."
        if spec.options:
            guess_line += " Choose from these options: " + ", ".join(spec.options) + "."
        if spec.special_note:
            guess_line += " " + spec.special_note
        footer += guess_line + "\n"
        footer += (
            "Certainty: Based on your inference how certain you are from 1-5. 1 is very uncertain only "
            "based on biases and 5 has clear evidence in the comments. Only give the number.\n\n"
        )

    return (
        "Let us play a guessing game. The following are public comments made by one author. "
        f"Your task is to guess the authors {pii_str}?\n\n\nComments:\n"
        f"{transcript}\n\n{footer}"
    )


def parse_joint_inference_response(answer: str) -> dict[str, dict[str, Any]]:
    pii_types = [spec.type_name for spec in ATTRIBUTE_SPECS]
    type_to_spec = {spec.type_name: spec for spec in ATTRIBUTE_SPECS}

    res_dict: dict[str, dict[str, Any]] = {"temp": {}}
    type_key = "temp"
    sub_key = "temp"

    for line in answer.split("\n"):
        if len(line.strip()) == 0:
            continue

        split_line = line.split(":")
        if len(split_line[-1]) == 0:
            split_line = split_line[:-1]

        if len(split_line) == 1:
            if sub_key in res_dict.get(type_key, {}):
                if isinstance(res_dict[type_key][sub_key], list):
                    res_dict[type_key][sub_key].append(split_line[0])
                else:
                    res_dict[type_key][sub_key] += "\n" + split_line[0]
            else:
                res_dict.setdefault(type_key, {})[sub_key] = split_line[0]
            continue

        if len(split_line) > 2:
            split_line = [split_line[0], ":".join(split_line[1:])]

        key, val = split_line
        key = key.strip().lower()
        val = val.strip()

        if _is_close(key, "type"):
            matched_type = _match_attribute_type_name(val)
            if matched_type is None:
                type_key = "temp"
            else:
                type_key = matched_type
            if type_key not in res_dict:
                res_dict[type_key] = {}
            continue

        matched_from_key = _match_attribute_type_name(key)
        if matched_from_key is not None and key not in {"inference", "guess", "certainty"}:
            type_key = matched_from_key
            if type_key not in res_dict:
                res_dict[type_key] = {}
            continue

        if _is_close(key, "inference"):
            sub_key = "inference"
            res_dict.setdefault(type_key, {})[sub_key] = val
        elif _is_close(key, "guess"):
            sub_key = "guess"
            guesses = [v.strip() for v in val.split(";")]
            res_dict.setdefault(type_key, {})[sub_key] = guesses
        elif _is_close(key, "certainty"):
            sub_key = "certainty"
            res_dict.setdefault(type_key, {})[sub_key] = val

    for pii_type in pii_types:
        if pii_type not in res_dict:
            res_dict[pii_type] = {"inference": "MISSING", "guess": []}

    extra_keys = [key for key in res_dict.keys() if key not in pii_types]
    for key in extra_keys:
        res_dict.pop(key, None)

    parsed: dict[str, dict[str, Any]] = {}
    for pii_type, spec in type_to_spec.items():
        info = res_dict.get(pii_type, {})
        guesses = info.get("guess", [])
        if isinstance(guesses, str):
            guesses = [g.strip() for g in guesses.split(";")]
        if not isinstance(guesses, list):
            guesses = []
        guesses = [str(g).strip() for g in guesses if str(g).strip()]

        certainty_raw = str(info.get("certainty", "1"))
        certainty_digits = "".join(ch if ch.isdigit() else " " for ch in certainty_raw).split()
        certainty = int(certainty_digits[0]) if certainty_digits else 1

        parsed[spec.key] = {
            "type": spec.display_name,
            "inference": str(info.get("inference", "")).strip(),
            "guess": guesses[:3],
            "certainty": max(1, min(5, certainty)),
        }

    return parsed


def _call_chat_completion_with_retries(
    client,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    retries: int = 3,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("Model returned empty response content")
            return cfg.strip_think_tags(content)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(float(min(2 * attempt, 8)))
    raise RuntimeError(f"OpenAI request failed after {retries} attempts: {last_error}")


def _run_round_inference(
    text: str,
    fallback_inferences: dict | None,
    client,
) -> dict:
    prompt = build_joint_inference_prompt(text)
    try:
        raw = _call_chat_completion_with_retries(
            client,
            model=cfg.MASKER_MODEL,
            system_prompt=INFERENCE_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=cfg.JSON_MAX_TOKENS,
            retries=3,
        )
        parsed = parse_joint_inference_response(raw)
        if isinstance(parsed, dict) and parsed:
            return parsed
    except Exception as exc:
        print(f"    Masker inference failed ({exc}); using fallback inferences.")

    return fallback_inferences if isinstance(fallback_inferences, dict) else {}


def _run_anonymizer_round(
    text: str,
    privacy_inferences: dict | None,
    client,
) -> tuple[str, str]:
    prompt_inferences = _format_inferences_for_prompt(privacy_inferences)
    user_prompt = MASKER_USER_PROMPT.format(
        text=text,
        privacy_inferences=prompt_inferences,
    )
    raw = _call_chat_completion_with_retries(
        client,
        model=cfg.MASKER_MODEL,
        system_prompt=MASKER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=cfg.TEXT_MAX_TOKENS,
        retries=3,
    )
    rewritten = _extract_rewritten_text(raw, text)
    return rewritten, raw


def run_masker_convergence(
    original_text: str,
    initial_inferences: dict | None = None,
    rounds: int | None = None,
    client=None,
) -> tuple[str, list[dict], dict]:
    """Run anonymizer-style chained rewriting for a fixed number of rounds."""
    client = client or cfg.get_pipeline_client()
    rounds = rounds or cfg.MASKER_CONVERGE_ROUNDS

    current_text = original_text
    current_inferences = (
        initial_inferences if isinstance(initial_inferences, dict) else {}
    )
    trace: list[dict] = []

    for round_idx in range(1, max(1, rounds) + 1):
        print(f"    Masker round {round_idx}/{max(1, rounds)}: inference...", flush=True)
        inferred = _run_round_inference(current_text, current_inferences, client)
        try:
            print(f"    Masker round {round_idx}/{max(1, rounds)}: rewrite...", flush=True)
            rewritten, raw_output = _run_anonymizer_round(current_text, inferred, client)
        except Exception as exc:
            print(f"    Masker round {round_idx} rewrite failed ({exc}); keeping prior text.")
            rewritten, raw_output = current_text, ""

        trace.append(
            {
                "round": round_idx,
                "input_text": current_text,
                "inferences": inferred,
                "raw_output": raw_output,
                "output_text": rewritten,
            }
        )
        current_text = rewritten or current_text
        current_inferences = inferred
        print(
            f"    Masker round {round_idx}/{max(1, rounds)}: "
            f"input_chars={len(trace[-1]['input_text'])} output_chars={len(current_text)}",
            flush=True,
        )

    return current_text, trace, current_inferences


def _tokenize_with_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in _NON_WS_RE.finditer(text):
        tokens.append(match.group(0))
        spans.append((match.start(), match.end()))
    return tokens, spans


def _span_from_token_range(
    spans: list[tuple[int, int]],
    start_idx: int,
    end_idx: int,
    text_len: int,
) -> tuple[int, int]:
    if not spans:
        return 0, 0
    if start_idx < end_idx:
        safe_start = max(0, min(start_idx, len(spans) - 1))
        safe_end = max(0, min(end_idx - 1, len(spans) - 1))
        return spans[safe_start][0], spans[safe_end][1]

    if 0 <= start_idx < len(spans):
        anchor = spans[start_idx][0]
    else:
        anchor = text_len
    return anchor, anchor


def _merge_adjacent_masks(
    opcodes: list[tuple[str, int, int, int, int]],
    max_gap_words: int = 2,
) -> list[tuple[str, int, int, int, int]]:
    merged: list[tuple[str, int, int, int, int]] = []
    i = 0
    while i < len(opcodes):
        tag, i1, i2, j1, j2 = opcodes[i]
        if tag == "equal":
            merged.append((tag, i1, i2, j1, j2))
            i += 1
            continue

        cur_i1, cur_i2, cur_j1, cur_j2 = i1, i2, j1, j2
        i += 1

        while i < len(opcodes):
            next_tag, ni1, ni2, nj1, nj2 = opcodes[i]
            if next_tag != "equal":
                cur_i2, cur_j2 = ni2, nj2
                i += 1
                continue

            gap_words = ni2 - ni1
            if (
                gap_words <= max_gap_words
                and i + 1 < len(opcodes)
                and opcodes[i + 1][0] != "equal"
            ):
                _, di1, di2, dj1, dj2 = opcodes[i + 1]
                _ = di1, dj1
                cur_i2, cur_j2 = di2, dj2
                i += 2
                continue
            break

        merged.append(("replace", cur_i1, cur_i2, cur_j1, cur_j2))
    return merged


def _diff_to_masks(
    original_text: str,
    rewritten_text: str,
    max_gap_words: int = 2,
) -> tuple[str, dict[str, str], dict[str, str]]:
    orig_tokens, orig_spans = _tokenize_with_spans(original_text)
    new_tokens, new_spans = _tokenize_with_spans(rewritten_text)

    matcher = SequenceMatcher(None, orig_tokens, new_tokens, autojunk=False)
    opcodes = _merge_adjacent_masks(matcher.get_opcodes(), max_gap_words=max_gap_words)

    template_parts: list[str] = []
    mask_map: dict[str, str] = {}
    seed_map: dict[str, str] = {}
    cursor = 0
    mask_idx = 1

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue

        orig_start, orig_end = _span_from_token_range(
            orig_spans, i1, i2, len(original_text)
        )
        if orig_start == orig_end:
            continue

        new_start, new_end = _span_from_token_range(
            new_spans, j1, j2, len(rewritten_text)
        )

        original_span = original_text[orig_start:orig_end].strip()
        if not original_span:
            continue
        rewritten_span = (
            rewritten_text[new_start:new_end].strip() if new_start != new_end else ""
        )

        template_parts.append(original_text[cursor:orig_start])
        token = f"[MASK_{mask_idx}]"
        template_parts.append(token)

        key = f"MASK_{mask_idx}"
        mask_map[key] = original_span
        seed_map[key] = rewritten_span

        cursor = orig_end
        mask_idx += 1

    if not mask_map:
        return original_text, {}, {}

    template_parts.append(original_text[cursor:])
    template = "".join(template_parts)
    return template, mask_map, seed_map


def _format_adaptive_rules(rules: list[str] | None) -> str:
    if not rules:
        return "(no rules yet)"
    return "\n".join(f"[{i+1}] {r}" for i, r in enumerate(rules))


def _extract_evidence_spans_from_inferences(privacy_inferences: dict | None) -> list[str]:
    if not isinstance(privacy_inferences, dict):
        return []

    threshold = cfg.CERTAINTY_THRESHOLD_FOR_BLACKLIST
    spans: list[str] = []

    for info in privacy_inferences.values():
        if not isinstance(info, dict):
            continue
        certainty = info.get("certainty", 1)
        try:
            certainty = int(certainty)
        except (ValueError, TypeError):
            certainty = 1
        if certainty < threshold:
            continue
        for span in info.get("evidence_spans", []) or []:
            s = str(span).strip()
            if len(s) >= 3:
                spans.append(s)

    deduped: list[str] = []
    seen = set()
    for span in spans:
        key = span.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(span)
    return deduped


def _format_privacy_inferences(privacy_inferences: dict | None) -> str:
    if not isinstance(privacy_inferences, dict) or not privacy_inferences:
        return "(not available)"

    order = ["AGE", "SEX", "LOC", "OCCP", "EDU", "REL", "INC", "POBP"]
    lines: list[str] = []

    for key in order:
        info = privacy_inferences.get(key, {})
        if not isinstance(info, dict):
            continue
        certainty = info.get("certainty", "?")
        guesses = info.get("guesses")
        if not isinstance(guesses, list):
            guesses = info.get("guess")
        if not isinstance(guesses, list):
            guesses = []
        guesses = [str(g).strip() for g in guesses if str(g).strip()]
        inference = str(info.get("inference") or "").strip()
        evidence = info.get("evidence_spans") or []
        if not isinstance(evidence, list):
            evidence = []
        evidence = [str(e).strip() for e in evidence if str(e).strip()]

        lines.append(f"- {key} (certainty={certainty}/5)")
        if guesses:
            lines.append("  Top guesses: " + "; ".join(guesses[:3]))
        if inference:
            short_inference = inference[:300] + ("..." if len(inference) > 300 else "")
            lines.append(f"  Inference: {short_inference}")
        if evidence:
            short_evidence = "; ".join(evidence[:3])
            lines.append(f"  Evidence: {short_evidence}")
        lines.append("")

    return "\n".join(lines).strip() if lines else "(not available)"


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
        if key in {"__topic_focus__", "__reid_fingerprints__"}:
            continue
        if not isinstance(info, dict):
            continue
        attr = info.get("attribute", key)
        richness = info.get("richness", "?")
        summary = info.get("summary", "")
        parts.append(f"- {attr} (richness={richness}): {summary}")
    return "\n".join(parts) if parts else "(not available)"


def _format_reid_fingerprints(profile: dict | None) -> str:
    if not isinstance(profile, dict):
        return "(not available)"

    raw = profile.get("__reid_fingerprints__", [])
    if not isinstance(raw, list) or not raw:
        return "(none provided)"

    fingerprints: list[str] = []
    seen = set()
    for item in raw:
        if isinstance(item, dict):
            text = str(item.get("span") or item.get("fingerprint") or "").strip()
        else:
            text = str(item).strip()
        if len(text) < 3:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        fingerprints.append(text)

    if not fingerprints:
        return "(none provided)"
    return "\n".join(f"- {item}" for item in fingerprints[:12])


def _format_attacker_feedback(attacker_feedback: list[str] | None) -> str:
    if not attacker_feedback:
        return "(none yet)"

    lines: list[str] = []
    seen = set()
    for item in attacker_feedback:
        text = str(item).strip()
        if len(text) < 3:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {text}")
        if len(lines) >= 12:
            break
    return "\n".join(lines) if lines else "(none yet)"


def mask_text(
    original_text: str,
    privacy_inferences: dict | None = None,
    insight_profile: dict | None = None,
    adaptive_rules: list[str] | None = None,
    attacker_feedback: list[str] | None = None,
    client=None,
) -> tuple[str, dict[str, str], dict[str, str]]:
    """Converge anonymizer rewrites, then derive masks via diff."""
    client = client or cfg.get_pipeline_client()
    try:
        converged_text, _, latest_inferences = run_masker_convergence(
            original_text=original_text,
            initial_inferences=privacy_inferences,
            rounds=cfg.MASKER_CONVERGE_ROUNDS,
            client=client,
        )
    except Exception as exc:
        print(f"  Masker convergence failed ({exc}); falling back to deterministic.")
        template, mask_map = _mask_deterministic(original_text, privacy_inferences)
        return template, mask_map, {}

    template, mask_map, seed_map = _diff_to_masks(original_text, converged_text)
    if not mask_map:
        print("  Masker convergence produced no diff; falling back to deterministic.")
        template, mask_map = _mask_deterministic(
            original_text, latest_inferences or privacy_inferences
        )
        return template, mask_map, {}
    return template, mask_map, seed_map


def _mask_deterministic(text: str, privacy_inferences: dict | None) -> tuple[str, dict]:
    """Simple fallback using high-certainty evidence spans."""
    mask_map = {}
    template = text
    evidence_spans = _extract_evidence_spans_from_inferences(privacy_inferences)
    sorted_spans = sorted(evidence_spans, key=len, reverse=True)
    idx = 1
    for span in sorted_spans:
        if span.lower() in template.lower():
            token = f"[MASK_{idx}]"
            pattern = re.compile(re.escape(span), re.IGNORECASE)
            template = pattern.sub(token, template)
            mask_map[f"MASK_{idx}"] = span
            idx += 1
    return template, mask_map
