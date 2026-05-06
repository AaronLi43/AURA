"""AURA pipeline orchestration: masker converge -> refill -> attack+keep select."""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pipeline_config as cfg
import db
from phase0_init import initialize_document
from phase1_masker import mask_text
from phase1_refiller import generate_variations
from phase2_attacker import attack_and_report, total_severity
from phase2_keeper import evaluate_preservation, total_loss


def _dedupe_keep_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if len(text) < 3:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _extract_attacker_evidence_spans(
    vulnerability_report: dict | None,
    min_severity: int = 3,
) -> list[str]:
    if not isinstance(vulnerability_report, dict):
        return []

    spans: list[str] = []
    for attr, info in vulnerability_report.items():
        if attr in {"total_severity", "SPAN_LEAKAGE"}:
            continue
        if not isinstance(info, dict):
            continue
        try:
            severity = int(info.get("severity", 0))
        except (TypeError, ValueError):
            severity = 0
        if severity < min_severity:
            continue
        for span in info.get("evidence_spans", []) or []:
            spans.append(str(span).strip())
    return _dedupe_keep_order(spans)


def _map_span_to_original(original_text: str, span: str) -> str | None:
    if not original_text or not span:
        return None

    exact = re.search(re.escape(span), original_text, flags=re.IGNORECASE)
    if exact:
        return original_text[exact.start():exact.end()].strip()

    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", span.lower())
        if len(token) >= 5
    }
    if len(query_tokens) < 2:
        return None

    best_line = None
    best_score = 0.0
    for line in original_text.splitlines():
        candidate = line.strip()
        if len(candidate) < 3:
            continue
        cand_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", candidate.lower())
            if len(token) >= 5
        }
        if not cand_tokens:
            continue
        overlap = len(query_tokens & cand_tokens)
        if overlap == 0:
            continue
        score = overlap / len(query_tokens)
        if score > best_score:
            best_score = score
            best_line = candidate

    if best_line and best_score >= 0.5:
        return best_line
    return None


def _map_feedback_spans_to_original(original_text: str, spans: list[str]) -> list[str]:
    mapped: list[str] = []
    for span in spans:
        backmapped = _map_span_to_original(original_text, span)
        mapped.append(backmapped or span)
    return _dedupe_keep_order(mapped)


def _evaluate_variation(
    variation: dict,
    original_text: str,
    original_inferences: dict,
    mask_map: dict,
    client,
) -> dict:
    """Run attacker + keeper in parallel for one variation."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        atk_fut = executor.submit(
            attack_and_report,
            variation["assembled_text"],
            original_inferences,
            client,
        )
        keep_fut = executor.submit(
            evaluate_preservation,
            original_text,
            variation["assembled_text"],
            mask_map,
            client,
        )
        attacker_result = atk_fut.result()
        keeper_result = keep_fut.result()

    return {
        "attacker": attacker_result,
        "keeper": keeper_result,
        "too_specific_count": int(
            attacker_result.get("specificity_report", {}).get("too_specific_count", 0)
        ),
        "severity": total_severity(attacker_result),
        "loss": total_loss(keeper_result),
    }


def _select_best(evaluations: list[dict], max_too_specific: int) -> int:
    """Select best variation under specificity cap when possible."""
    if not evaluations:
        return 0

    valid = []
    fallback = []
    for i, ev in enumerate(evaluations):
        if not isinstance(ev, dict):
            continue
        too_specific_count = int(ev.get("too_specific_count", 999))
        severity = int(ev.get("severity", 999))
        loss = int(ev.get("loss", 999))
        fallback.append((too_specific_count, severity, loss, i))
        if too_specific_count <= max_too_specific:
            valid.append((severity, loss, i))

    if valid:
        valid.sort()
        return valid[0][2]

    if not fallback:
        return 0

    fallback.sort()
    return fallback[0][3]


def run_one(doc_id: str, max_iter: int | None = None):
    """Run two-stage pipeline for a single document."""
    _ = max_iter  # Retained for CLI compatibility; unused in two-stage mode.
    client = cfg.get_pipeline_client()

    doc = db.get_document(doc_id)
    if doc is None:
        print(f"ERROR: Document {doc_id} not found in DB. Run phase0 first.")
        return

    original_text = doc["original_text"]
    evidence_spans = list(doc.get("evidence_spans") or doc.get("blacklist") or [])
    insight_profile = doc.get("insight_profile")
    privacy_inferences = doc.get("privacy_inferences") or {}

    print(f"Processing {doc_id} | evidence_spans={len(evidence_spans)}")

    # Stage 1: masker convergence + diff-derived masks
    print(f"  Stage 1/2 — Masker convergence ({cfg.MASKER_CONVERGE_ROUNDS} rounds)...")
    template, mask_map, seed_map = mask_text(
        original_text=original_text,
        privacy_inferences=privacy_inferences,
        insight_profile=insight_profile,
        adaptive_rules=None,
        attacker_feedback=None,
        client=client,
    )
    if not mask_map:
        print("  No masks produced; keeping original text as final output.")
        db.upsert_document(
            doc_id,
            final_text=original_text,
            final_privacy_score=float(999),
            evidence_spans=evidence_spans,
            status="success",
        )
        return

    print(f"  Masks: {len(mask_map)} spans")

    # Stage 2: refill + one-shot attacker/keeper selection
    print(f"  Stage 2/2 — Generating {cfg.VARIATIONS_PER_ROUND} refill variations...")
    variations = generate_variations(
        template,
        mask_map,
        seed_map,
        insight_profile,
        adaptive_rules=None,
        n=cfg.VARIATIONS_PER_ROUND,
        client=client,
    )
    if not variations:
        print("  No valid variations; keeping converged masker output.")
        fallback_text = template
        for key, val in (seed_map or {}).items():
            fallback_text = fallback_text.replace(f"[{key}]", str(val))
        for key, val in (mask_map or {}).items():
            fallback_text = fallback_text.replace(f"[{key}]", str(val))
        db.upsert_document(
            doc_id,
            final_text=fallback_text,
            final_privacy_score=float(999),
            evidence_spans=evidence_spans,
            status="success",
        )
        return

    print(f"  Got {len(variations)} valid variations")
    print("  Evaluating variations (attacker + keeper + specificity)...")
    evaluations = [None] * len(variations)
    workers = min(len(variations), cfg.EVAL_MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _evaluate_variation,
                var, original_text, privacy_inferences, mask_map, client,
            ): idx
            for idx, var in enumerate(variations)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                evaluations[idx] = fut.result()
                print(
                    f"    Variation {idx} evaluated "
                    f"({sum(ev is not None for ev in evaluations)}/{len(variations)})",
                    flush=True,
                )
            except Exception as exc:
                print(f"    Variation {idx} eval failed: {exc}")
                evaluations[idx] = {
                    "severity": 999,
                    "loss": 999,
                    "too_specific_count": 999,
                    "attacker": {},
                    "keeper": {},
                }

    for i, ev in enumerate(evaluations):
        sev = ev.get("severity", "?") if ev else "?"
        loss = ev.get("loss", "?") if ev else "?"
        too_spec = ev.get("too_specific_count", "?") if ev else "?"
        print(
            f"    v{i}: too_specific={too_spec}, severity={sev}, loss={loss}"
        )

    best_idx = _select_best(evaluations, cfg.MAX_TOO_SPECIFIC_ATTRS)
    best_var = variations[best_idx]
    best_eval = evaluations[best_idx]
    best_severity = int(best_eval.get("severity", 999))
    best_loss = int(best_eval.get("loss", 999))
    best_too_specific = int(best_eval.get("too_specific_count", 999))
    print(
        f"  Best: v{best_idx} "
        f"(too_specific={best_too_specific}, severity={best_severity}, loss={best_loss})"
    )

    db.insert_iteration(
        doc_id,
        1,
        template_text=template,
        mask_map_json=mask_map,
        variations_json=[v["assembled_text"] for v in variations],
        best_variation_idx=best_idx,
        attacker_report_json=best_eval.get("attacker"),
        keeper_report_json=best_eval.get("keeper"),
        modulator_output_json={"mode": "two_stage_no_modulator"},
        masker_rules_json=[],
        refiller_rules_json=[],
        blacklist_snapshot_json=evidence_spans,
    )

    # Finalize
    db.upsert_document(
        doc_id,
        final_text=best_var["assembled_text"],
        final_privacy_score=float(best_severity),
        evidence_spans=evidence_spans,
        status="success",
    )
    print(f"\n  SUCCESS: {doc_id} | final_severity={best_severity}")


def run_all(max_iter: int | None = None, max_workers: int | None = None):
    """Run pipeline for all initialized documents in parallel."""
    max_workers = max_workers or cfg.RUN_ALL_MAX_WORKERS
    docs = db.get_all_documents(status="initialized")
    if not docs:
        print("No initialized documents to process.")
        return

    print(f"Running {len(docs)} documents (max_workers={max_workers})")

    def _run(doc_id):
        run_one(doc_id, max_iter)
        return doc_id

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run, d["document_id"]): d["document_id"] for d in docs}
        for fut in as_completed(futures):
            doc_id = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                print(f"FAILED: {doc_id}: {exc}")
                db.upsert_document(doc_id, status="failed")


def main():
    parser = argparse.ArgumentParser(description="No-branch privacy pipeline")
    sub = parser.add_subparsers(dest="command")

    p_one = sub.add_parser("run-one", help="Process a single document")
    p_one.add_argument("--doc-id", required=True)
    p_one.add_argument("--max-iter", type=int, default=cfg.MAX_ITERATIONS)

    p_all = sub.add_parser("run-all", help="Process all initialized documents")
    p_all.add_argument("--max-iter", type=int, default=cfg.MAX_ITERATIONS)
    p_all.add_argument("--max-workers", type=int, default=cfg.RUN_ALL_MAX_WORKERS)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run-one":
        run_one(args.doc_id, args.max_iter)
    elif args.command == "run-all":
        run_all(args.max_iter, args.max_workers)


if __name__ == "__main__":
    main()
