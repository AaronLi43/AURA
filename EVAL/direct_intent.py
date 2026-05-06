#!/usr/bin/env python3
"""Direct-intent re-identification with web search.

Reads a CSV of rewritten transcripts (columns: ``transcript_id``, ``text``) and,
for each row, asks an OpenAI model to guess the speaker by combining web search
with the transcript content.  Results are streamed to a JSON file so the script
can be safely interrupted and resumed.

Usage:
    python direct_intent.py <input.csv>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ── Prompts ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
I have an interview transcript. Based on the content, try your best to guess who this person might be. Consider all clues including their role, experience, projects they mentioned, opinions, writing style, and any other details that might help identify them. Be thorough and consider how different pieces of information might combine to point to a specific individual.

Rules:
- Be conservative and precise; avoid hallucinations. Prefer real links and include associated details when possible.
- Use web search to validate details whenever helpful.
- Return STRICT JSON matching the schema exactly; do not include any extra commentary.

Output: Return STRICT JSON with the following schema:
{
  "identifier_candidates": [
    {
      "interviewee_candidates": array of strings (if known),
      "identity_label": string,   # the most likely publicly visible item
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
"""

USER_PROMPT_TEMPLATE = """
Here is the transcript:
{full_transcript}
"""


# ── Core ─────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "input_csv",
        type=Path,
        help="CSV with columns 'transcript_id' and 'text'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: web_search_<input_name>.json next to the script).",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="OpenAI model to call (default: %(default)s).",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        choices=["low", "medium", "high"],
        help="`reasoning.effort` value sent to the model (default: %(default)s).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.2,
        help="Seconds to sleep between requests (default: %(default)s).",
    )
    return parser.parse_args()


def load_existing(output_path: Path) -> tuple[list[dict], set[str]]:
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as fh:
            results = json.load(fh)
        processed = {r["transcript_id"] for r in results if "transcript_id" in r}
        return results, processed
    return [], set()


def extract_candidates(response_obj) -> dict | None:
    """Walk a Responses API output and return the first valid candidate JSON."""
    for item in response_obj.output:
        if item.type == "message":
            for content in getattr(item, "content", []):
                if hasattr(content, "text") and content.text:
                    try:
                        parsed = json.loads(content.text)
                    except json.JSONDecodeError:
                        continue
                    if "identifier_candidates" in parsed:
                        return parsed
    return None


def main() -> int:
    args = parse_args()

    load_dotenv(override=True)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    input_path: Path = args.input_csv
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    output_path = args.output or Path(f"web_search_{input_path.name}.json")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    results, processed = load_existing(output_path)

    with open(input_path, "r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    count = 0
    for entry in rows:
        transcript_id = (entry.get("transcript_id") or "").strip()
        full_transcript = (entry.get("text") or "").strip()
        if not transcript_id or not full_transcript:
            print("Skipping row with missing transcript_id or text")
            continue
        if transcript_id in processed:
            continue
        count += 1

        try:
            print(f"Count: {count} - (Processing ID: {transcript_id})")
            response = client.responses.create(
                model=args.model,
                reasoning={"effort": args.reasoning_effort},
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_PROMPT_TEMPLATE.format(
                            full_transcript=full_transcript
                        ),
                    },
                ],
                tools=[{"type": "web_search"}],
            )

            # Persist the raw response for auditing.
            save_dir = Path(f"responses_{input_path.stem}")
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / f"{transcript_id}_full_response.txt").write_text(
                str(response), encoding="utf-8"
            )

            extracted = extract_candidates(response)
            if extracted:
                results.append(
                    {
                        "transcript_id": transcript_id,
                        "identifier_candidates": extracted.get(
                            "identifier_candidates", []
                        ),
                    }
                )
                with open(output_path, "w", encoding="utf-8") as fh:
                    json.dump(results, fh, ensure_ascii=False, indent=2)
                print(
                    f"  Found {len(results[-1]['identifier_candidates'])} candidates"
                )
            else:
                print(f"  No valid identifier_candidates found. ID: {transcript_id}")

            time.sleep(args.sleep)

        except Exception as exc:  # noqa: BLE001
            print(f"Error processing ID {transcript_id}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
