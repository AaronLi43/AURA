#!/usr/bin/env python3
"""Evaluate recoverability of deterministic code facts across transcript configs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


THIS_DIR = Path(__file__).resolve().parent
MODULE_PATH = THIS_DIR / "identifier_profile_preservation.py"
DEFAULT_CODE_FACT_ROOT = THIS_DIR / "code_fact"
DEFAULT_REFERENCE_ROOT = DEFAULT_CODE_FACT_ROOT / "reference_facts"
DEFAULT_OUTPUT_ROOT = DEFAULT_CODE_FACT_ROOT / "recoverability"
DEFAULT_REFERENCE_CSV: Path | None = None  # supply via --original-csv
CACHE_SCHEMA_VERSION = "code_fact_recoverability_v1"


_ipp_module: Any | None = None


def load_profile_module() -> Any:
    """Lazily load identifier_profile_preservation.py.

    We import the helper module on demand so that `--help` and basic
    introspection do not require the OpenAI SDK or the ``_compat`` module
    to be importable.
    """
    global _ipp_module
    if _ipp_module is not None:
        return _ipp_module
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    spec = importlib.util.spec_from_file_location(
        "identifier_profile_preservation", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("identifier_profile_preservation", module)
    spec.loader.exec_module(module)
    _ipp_module = module
    return module


class _LazyIpp:
    def __getattr__(self, name: str) -> Any:
        return getattr(load_profile_module(), name)


ipp = _LazyIpp()


CODE_FACT_RECOVERABILITY_SYSTEM_PROMPT = (
    "You are a careful qualitative researcher. "
    "Judge whether a specific qualitative code fact can be recovered from a transcript. "
    "Use only the transcript text. Do not use outside knowledge. "
    "Return valid JSON only."
)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def normalize_text_map(text_map: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for transcript_id, text in text_map.items():
        key = ipp.normalize_dataset_id(transcript_id)
        if key and key not in normalized:
            normalized[key] = text
    return normalized


def load_reference_fact_payloads(reference_root: Path) -> dict[str, dict[str, Any]]:
    if not reference_root.exists():
        raise FileNotFoundError(f"Reference facts directory not found: {reference_root}")

    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted(reference_root.glob("*.json")):
        if path.name == "all_code_facts.json":
            continue
        payload = load_json(path)
        transcript_id = ipp.normalize_dataset_id(payload.get("transcript_id", path.stem))
        if transcript_id:
            payloads[transcript_id] = payload
    if not payloads:
        raise ValueError(f"No reference fact payloads found in {reference_root}")
    return payloads


def build_target_sample_ids(
    reference_payloads: dict[str, dict[str, Any]],
    requested_sample_ids: list[str] | None,
) -> list[str]:
    available = sorted(reference_payloads)
    if not requested_sample_ids:
        return available

    requested = []
    for sample_id in requested_sample_ids:
        normalized = ipp.normalize_dataset_id(sample_id)
        if normalized and normalized not in requested:
            requested.append(normalized)
    missing = [sample_id for sample_id in requested if sample_id not in reference_payloads]
    if missing:
        raise ValueError(f"Requested sample IDs not found in reference facts: {', '.join(missing)}")
    return requested


def build_code_fact_recoverability_prompt(
    *,
    transcript_id: str,
    config_label: str,
    transcript_text: str,
    fact_item: dict[str, Any],
) -> str:
    decisions = "\n".join(f"- {item}" for item in ipp.RECOVERABILITY_DECISIONS)
    return (
        f"Determine whether the qualitative code fact below can be recovered from the transcript for config `{config_label}`.\n\n"
        f"Transcript ID: {transcript_id}\n"
        f"Category: {fact_item.get('category_id', '')} ({fact_item.get('category_label', '')})\n"
        f"Code: {fact_item.get('code_id', '')} ({fact_item.get('code_label', '')})\n"
        f"Code definition: \"{fact_item.get('code_definition', '')}\"\n"
        f"Inclusion criteria: \"{fact_item.get('inclusion_criteria', '')}\"\n"
        f"Exclusion criteria: \"{fact_item.get('exclusion_criteria', '')}\"\n"
        f"Reference excerpt from original transcript: \"{fact_item.get('excerpt', '')}\"\n"
        f"Reference code fact: \"{fact_item.get('fact', '')}\"\n\n"
        "Allowed decisions (must match exactly one of these strings):\n"
        f"{decisions}\n\n"
        "Rules:\n"
        "- Judge recoverability from the transcript alone.\n"
        "- The code metadata and reference excerpt clarify the target meaning, but they are not evidence for the current transcript.\n"
        "- Choose `Yes...` only when the transcript contains clear evidence for the same substantive coded insight.\n"
        "- Choose `No...` when the transcript is missing, weaker, broader, or not specific enough to support the code fact.\n"
        "- Choose `I am not sure` only when the transcript hints at the code fact but the evidence quality is genuinely unclear.\n"
        "- If the decision is Yes, provide one exact evidence quote from the transcript.\n"
        "- If the decision is No or I am not sure, `evidence_quote` may be empty.\n\n"
        f"=== TRANSCRIPT ===\n{transcript_text}\n=== END TRANSCRIPT ===\n\n"
        'Return JSON exactly: {"decision":"...","reasoning":"...","evidence_quote":"..."}\n'
    )


def evaluate_code_fact(
    *,
    api_key: str,
    model: str,
    transcript_id: str,
    config_label: str,
    transcript_text: str,
    fact_item: dict[str, Any],
    retries: int = 3,
) -> dict[str, Any]:
    last_error: str | None = None
    for attempt in range(1, retries + 1):
        try:
            client = ipp._create_chat_client(model, api_key)
            parsed = ipp.call_json(
                client,
                model=model,
                system_prompt=CODE_FACT_RECOVERABILITY_SYSTEM_PROMPT,
                user_prompt=build_code_fact_recoverability_prompt(
                    transcript_id=transcript_id,
                    config_label=config_label,
                    transcript_text=transcript_text,
                    fact_item=fact_item,
                ),
            )
            decision = ipp.normalize_text(parsed.get("decision", ""))
            reasoning = ipp.normalize_text(parsed.get("reasoning", ""))
            evidence_quote = ipp.normalize_text(parsed.get("evidence_quote", ""))
            if decision not in ipp.RECOVERABILITY_DECISIONS:
                raise ValueError(f"Invalid decision: {decision!r}")
            if decision == ipp.RECOVERABILITY_DECISIONS[0] and not ipp.evidence_quote_supported(
                transcript_text, evidence_quote
            ):
                decision = ipp.RECOVERABILITY_DECISIONS[2]
                if not reasoning:
                    reasoning = (
                        "The code fact may be recoverable, but the response did not provide a "
                        "verifiable evidence quote."
                    )
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
        "decision": ipp.RECOVERABILITY_DECISIONS[2],
        "reasoning": "",
        "evidence_quote": "",
        "error": last_error or "unknown",
    }


def init_code_bucket(code_id: str, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "code_id": code_id,
        "code_label": item.get("code_label", ""),
        "category_id": item.get("category_id", ""),
        "category_label": item.get("category_label", ""),
        "sample_count": 0,
        "total": 0,
        "yes_count": 0,
        "no_count": 0,
        "unsure_count": 0,
        "error_count": 0,
        "recoverable_rate": None,
    }


def init_category_bucket(category_id: str, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "category_id": category_id,
        "category_label": item.get("category_label", ""),
        "sample_count": 0,
        "total": 0,
        "yes_count": 0,
        "no_count": 0,
        "unsure_count": 0,
        "error_count": 0,
        "recoverable_rate": None,
    }


def apply_item_to_bucket(bucket: dict[str, Any], item: dict[str, Any]) -> None:
    bucket["total"] += 1
    if item.get("error"):
        bucket["error_count"] += 1
        return
    if item.get("decision") == ipp.RECOVERABILITY_DECISIONS[0]:
        bucket["yes_count"] += 1
    elif item.get("decision") == ipp.RECOVERABILITY_DECISIONS[1]:
        bucket["no_count"] += 1
    elif item.get("decision") == ipp.RECOVERABILITY_DECISIONS[2]:
        bucket["unsure_count"] += 1


def finalize_bucket_rates(buckets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for bucket in buckets.values():
        if bucket["total"] > 0:
            bucket["recoverable_rate"] = round(bucket["yes_count"] / bucket["total"], 3)
    return dict(sorted(buckets.items()))


def summarize_code_fact_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {decision: 0 for decision in ipp.RECOVERABILITY_DECISIONS}
    error_count = 0
    by_code: dict[str, dict[str, Any]] = {}
    by_category: dict[str, dict[str, Any]] = {}

    for item in items:
        code_id = ipp.normalize_text(item.get("code_id", "")) or "UNKNOWN"
        category_id = ipp.normalize_text(item.get("category_id", "")) or "UNKNOWN"
        code_bucket = by_code.setdefault(code_id, init_code_bucket(code_id, item))
        category_bucket = by_category.setdefault(category_id, init_category_bucket(category_id, item))
        apply_item_to_bucket(code_bucket, item)
        apply_item_to_bucket(category_bucket, item)

        if item.get("error"):
            error_count += 1
            continue

        decision = item.get("decision")
        if decision in counts:
            counts[decision] += 1

    for bucket in by_code.values():
        bucket["sample_count"] = 1
    for bucket in by_category.values():
        bucket["sample_count"] = 1

    total = len(items)
    return {
        "total_facts": total,
        "yes_count": counts[ipp.RECOVERABILITY_DECISIONS[0]],
        "no_count": counts[ipp.RECOVERABILITY_DECISIONS[1]],
        "unsure_count": counts[ipp.RECOVERABILITY_DECISIONS[2]],
        "error_count": error_count,
        "recoverable_rate": round(counts[ipp.RECOVERABILITY_DECISIONS[0]] / total, 3) if total else None,
        "by_code": finalize_bucket_rates(by_code),
        "by_category": finalize_bucket_rates(by_category),
    }


def merge_summary_bucket(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["sample_count"] += int(source.get("sample_count", 0))
    target["total"] += int(source.get("total", 0))
    target["yes_count"] += int(source.get("yes_count", 0))
    target["no_count"] += int(source.get("no_count", 0))
    target["unsure_count"] += int(source.get("unsure_count", 0))
    target["error_count"] += int(source.get("error_count", 0))


def evaluate_code_fact_config(
    *,
    dataset_id: str,
    config_name: str,
    transcript_text: str,
    transcript_source: str,
    reference_payload: dict[str, Any],
    api_key: str,
    model: str,
    output_root: Path,
    max_fact_workers: int,
    overwrite: bool,
) -> dict[str, Any]:
    out_dir = output_root / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ipp.slugify(dataset_id)}.json"

    facts = list(reference_payload.get("facts", []))
    cache_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "stage": "code_fact_recoverability",
        "model": model,
        "dataset_id": dataset_id,
        "config_name": config_name,
        "transcript_text": transcript_text,
        "facts": facts,
    }
    cache_key = ipp.hash_payload(cache_payload)

    if out_path.exists() and not overwrite:
        try:
            cached = load_json(out_path)
            if cached.get("cache_key") == cache_key:
                return cached
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    workers = max(1, min(max_fact_workers, len(facts))) if facts else 1
    items: list[dict[str, Any]] = [None] * len(facts)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_code_fact,
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
        "timestamp": ipp.now_iso(),
        "summary": summarize_code_fact_items(items),
        "items": items,
    }
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result


def evaluate_config_samples(
    *,
    config_name: str,
    text_map: dict[str, str],
    transcript_source: str,
    target_sample_ids: list[str],
    reference_payloads: dict[str, dict[str, Any]],
    api_key: str,
    model: str,
    output_root: Path,
    sample_workers: int,
    max_fact_workers: int,
    overwrite: bool,
) -> dict[str, dict[str, Any]]:
    sample_ids = [sample_id for sample_id in target_sample_ids if sample_id in text_map]
    if not sample_ids:
        return {}

    payloads: dict[str, dict[str, Any]] = {}
    workers = max(1, min(sample_workers, len(sample_ids)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_code_fact_config,
                dataset_id=sample_id,
                config_name=config_name,
                transcript_text=text_map[sample_id],
                transcript_source=transcript_source,
                reference_payload=reference_payloads[sample_id],
                api_key=api_key,
                model=model,
                output_root=output_root,
                max_fact_workers=max_fact_workers,
                overwrite=overwrite,
            ): sample_id
            for sample_id in sample_ids
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            payloads[sample_id] = future.result()
            print(f"Evaluated code facts {config_name}:{sample_id}")
    return payloads


def aggregate_recoverability_summary(
    payloads_by_sample: dict[str, dict[str, dict[str, Any]]],
    *,
    config_order: list[str],
) -> dict[str, Any]:
    by_config: dict[str, Any] = {}
    top_by_code: dict[str, dict[str, Any]] = {}

    for config_name in config_order:
        sample_payloads = [
            payloads_by_sample[sample_id][config_name]
            for sample_id in payloads_by_sample
            if config_name in payloads_by_sample[sample_id]
        ]
        if not sample_payloads:
            continue

        by_code: dict[str, dict[str, Any]] = {}
        by_category: dict[str, dict[str, Any]] = {}
        total_facts = 0
        yes_count = 0
        no_count = 0
        unsure_count = 0
        error_count = 0

        for payload in sample_payloads:
            summary = payload.get("summary", {})
            total_facts += int(summary.get("total_facts", 0))
            yes_count += int(summary.get("yes_count", 0))
            no_count += int(summary.get("no_count", 0))
            unsure_count += int(summary.get("unsure_count", 0))
            error_count += int(summary.get("error_count", 0))

            for code_id, info in summary.get("by_code", {}).items():
                bucket = by_code.setdefault(code_id, init_code_bucket(code_id, info))
                merge_summary_bucket(bucket, info)
            for category_id, info in summary.get("by_category", {}).items():
                bucket = by_category.setdefault(category_id, init_category_bucket(category_id, info))
                merge_summary_bucket(bucket, info)

        by_code = finalize_bucket_rates(by_code)
        by_category = finalize_bucket_rates(by_category)

        config_info = {
            "sample_count": len(sample_payloads),
            "total_facts": total_facts,
            "yes_count": yes_count,
            "no_count": no_count,
            "unsure_count": unsure_count,
            "error_count": error_count,
            "recoverable_rate": round(yes_count / total_facts, 3) if total_facts else None,
            "by_code": by_code,
            "by_category": by_category,
        }
        by_config[config_name] = config_info

        for code_id, info in by_code.items():
            top_entry = top_by_code.setdefault(
                code_id,
                {
                    "code_id": code_id,
                    "code_label": info.get("code_label", ""),
                    "category_id": info.get("category_id", ""),
                    "category_label": info.get("category_label", ""),
                    "overall_total": 0,
                    "overall_yes_count": 0,
                    "overall_no_count": 0,
                    "overall_unsure_count": 0,
                    "overall_error_count": 0,
                    "configs": {},
                },
            )
            top_entry["overall_total"] += int(info.get("total", 0))
            top_entry["overall_yes_count"] += int(info.get("yes_count", 0))
            top_entry["overall_no_count"] += int(info.get("no_count", 0))
            top_entry["overall_unsure_count"] += int(info.get("unsure_count", 0))
            top_entry["overall_error_count"] += int(info.get("error_count", 0))
            top_entry["configs"][config_name] = {
                "sample_count": int(info.get("sample_count", 0)),
                "total": int(info.get("total", 0)),
                "yes_count": int(info.get("yes_count", 0)),
                "no_count": int(info.get("no_count", 0)),
                "unsure_count": int(info.get("unsure_count", 0)),
                "error_count": int(info.get("error_count", 0)),
                "recoverable_rate": info.get("recoverable_rate"),
            }

    for info in top_by_code.values():
        total = int(info.get("overall_total", 0))
        info["overall_recoverable_rate"] = round(info["overall_yes_count"] / total, 3) if total else None

    by_sample: dict[str, dict[str, Any]] = {}
    for sample_id, config_payloads in payloads_by_sample.items():
        by_sample[sample_id] = {}
        for config_name, payload in config_payloads.items():
            by_sample[sample_id][config_name] = payload.get("summary", {})

    return {
        "by_config": by_config,
        "by_sample": by_sample,
        "by_code": dict(sorted(top_by_code.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate recoverability of code facts across transcript configs."
    )
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=ipp.DEFAULT_MODEL)
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--config-workers", type=int, default=4)
    parser.add_argument("--sample-workers", type=int, default=3)
    parser.add_argument("--max-fact-workers", type=int, default=8)
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if load_dotenv is not None:
        for candidate in (
            THIS_DIR / ".env",
            THIS_DIR.parent / "NIPS_SUBMISSION_CODE" / ".env",
            THIS_DIR.parent / ".env",
        ):
            if candidate.exists():
                load_dotenv(candidate)
                break
    api_key = ipp._resolve_api_key(args.model)

    reference_payloads = load_reference_fact_payloads(args.reference_root.resolve())
    target_sample_ids = build_target_sample_ids(reference_payloads, args.sample_id)
    original_map = normalize_text_map(ipp.load_reference_transcripts(args.original_csv))
    configs = [
        (config_name, normalize_text_map(text_map), source)
        for config_name, text_map, source in ipp.build_profile_config_sources(args, original_map)
    ]
    config_order = [config_name for config_name, _, _ in configs]
    config_sources = {config_name: source for config_name, _, source in configs}

    payloads_by_sample: dict[str, dict[str, dict[str, Any]]] = {sample_id: {} for sample_id in target_sample_ids}
    workers = max(1, min(args.config_workers, len(configs))) if configs else 1
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_config_samples,
                config_name=config_name,
                text_map=text_map,
                transcript_source=source,
                target_sample_ids=target_sample_ids,
                reference_payloads=reference_payloads,
                api_key=api_key,
                model=args.model,
                output_root=args.output_root.resolve(),
                sample_workers=args.sample_workers,
                max_fact_workers=args.max_fact_workers,
                overwrite=args.overwrite,
            ): config_name
            for config_name, text_map, source in configs
        }
        for future in as_completed(futures):
            config_name = futures[future]
            sample_payloads = future.result()
            for sample_id, payload in sample_payloads.items():
                payloads_by_sample[sample_id][config_name] = payload

    summary_payload = {
        "created_at": ipp.now_iso(),
        "model": args.model,
        "reference_root": str(args.reference_root.resolve()),
        "sample_ids": target_sample_ids,
        "config_sources": config_sources,
        "summary": aggregate_recoverability_summary(payloads_by_sample, config_order=config_order),
    }

    summary_path = args.output_root.resolve() / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    ipp.print_recoverability_console_summary(summary_payload, config_order=config_order)
    print("")
    print(f"Saved code fact recoverability summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
