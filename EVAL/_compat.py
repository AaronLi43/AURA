"""Vendored helpers for the eval scripts.

The original codebase imported ``ATTRIBUTE_SPECS`` from
``Advance_Anonymizer.anonymizer`` and a small set of helpers from
``baseline.experiments.insight.utility_pilot.run_ap_preserved_mcqa``.
Those packages are part of the authors' private workspace and are not
shipped with this release.  The symbols below are vendored verbatim from
those modules so that ``identifier_profile_preservation.py`` and
``evaluate_code_fact_recoverability.py`` are self-contained.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


# ── 8 base privacy attributes ────────────────────────────────────────────


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
        options=[
            "No Highschool",
            "In Highschool",
            "HS Diploma",
            "In College",
            "College Degree",
            "PhD",
        ],
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
        options=[
            "No income",
            "Low (<30k USD)",
            "Medium (30-60k USD)",
            "High (60-150k USD)",
            "Very High (>150k USD)",
        ],
    ),
    AttributeSpec(
        "POBP",
        "pobp",
        "Place of Birth",
        "place of birth",
        aliases=("birth place", "birth location"),
    ),
]


# ── Configuration registry ───────────────────────────────────────────────
#
# CFG_ORDER and CFG_DISPLAY enumerate every rewriting variant referenced
# in the paper.  The eval scripts expect these names to exist; you only
# need to populate the ones you actually run via ``--<config>-path``
# command-line arguments.


CFG_ORDER = [
    "original",
    "nobranch",
    "adaptive_privacy",
    "pure_adaptive_attri",
    "on_device",
    "on_device_qwen",
    "remove_2",
    "remove_4",
    "anonymized_transcripts",
    "presidio",
    "rewritten_v1",
    "rewritten_v2",
    "dpmlm_eps_10",
    "dpmlm_eps_30",
    "dpmlm_eps_50",
    "dpmlm_eps_70",
    "dpmlm_eps_100",
    "dpmlm_eps_120",
    "dpmlm_eps_140",
]

CFG_DISPLAY = {
    "original": "Original",
    "nobranch": "NoBranch",
    "adaptive_privacy": "Adaptive Privacy",
    "pure_adaptive_attri": "Pure Adaptive Attri",
    "on_device": "On Device",
    "on_device_qwen": "On Device (Qwen)",
    "remove_2": "Remove 2",
    "remove_4": "Remove 4",
    "anonymized_transcripts": "Anonymizer",
    "presidio": "Presidio",
    "rewritten_v1": "Rewritten v1",
    "rewritten_v2": "Rewritten v2",
    "dpmlm_eps_10": "DP-MLM e10",
    "dpmlm_eps_30": "DP-MLM e30",
    "dpmlm_eps_50": "DP-MLM e50",
    "dpmlm_eps_70": "DP-MLM e70",
    "dpmlm_eps_100": "DP-MLM e100",
    "dpmlm_eps_120": "DP-MLM e120",
    "dpmlm_eps_140": "DP-MLM e140",
}


# ── CSV helpers ──────────────────────────────────────────────────────────


def load_csv_map(
    path: Path,
    id_col: str,
    text_col: str,
    epsilon: str | None = None,
) -> dict[str, str]:
    """Load a {transcript_id -> text} map from a CSV with the given columns."""
    out: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = (row.get(id_col) or "").strip().lower()
            if epsilon is not None and str(row.get("epsilon", "")).strip() != epsilon:
                continue
            txt = (row.get(text_col) or "").strip()
            if rid and txt:
                out[rid] = txt
    return out


def maybe_load_csv_map(
    path: Path | None,
    id_col: str,
    text_col: str,
    epsilon: str | None = None,
) -> dict[str, str]:
    if path is None or not Path(path).exists():
        return {}
    return load_csv_map(Path(path), id_col=id_col, text_col=text_col, epsilon=epsilon)


def maybe_load_csv_map_flexible(
    path: Path | None,
    column_candidates: list[tuple[str, str]],
    epsilon: str | None = None,
) -> dict[str, str]:
    """Like ``maybe_load_csv_map`` but tries multiple (id_col, text_col) pairs."""
    if path is None or not Path(path).exists():
        return {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        selected_pair: tuple[str, str] | None = None
        for id_col, text_col in column_candidates:
            if id_col in fieldnames and text_col in fieldnames:
                selected_pair = (id_col, text_col)
                break
        if selected_pair is None:
            return {}

        id_col, text_col = selected_pair
        out: dict[str, str] = {}
        for row in reader:
            rid = (row.get(id_col) or "").strip().lower()
            if epsilon is not None and str(row.get("epsilon", "")).strip() != epsilon:
                continue
            txt = (row.get(text_col) or "").strip()
            if rid and txt:
                out[rid] = txt
        return out
