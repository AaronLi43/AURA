"""Phase 2b: Keeper — check insight preservation across 8 utility attributes.

The keeper has access to the original text, the rewritten text, and the mask
map. It evaluates how much research-valuable content survived rewriting.
"""
from __future__ import annotations

import json

import pipeline_config as cfg
from phase0_init import UTILITY_ATTRIBUTES

KEEPER_SYSTEM_PROMPT = """\
You are a qualitative research analyst evaluating whether a privacy-rewritten
transcript preserves the research-valuable content of the original.

You have access to:
- The ORIGINAL transcript (ground truth)
- The REWRITTEN transcript (privacy-protected version)
- The MASK MAP showing what was replaced

For each of the 8 utility dimensions, assess:
1. What key content existed in the original?
2. Was it preserved, distorted, or lost in the rewrite?
3. If lost, what specifically was lost and how severe is the loss?

Be precise: cite exact spans from both texts to support your assessment.
Output valid JSON only.
"""


def build_keeper_prompt(
    original_text: str,
    rewritten_text: str,
    mask_map: dict,
) -> str:
    attr_block = "\n".join(
        f"- {a.key} ({a.display_name}): {a.description}"
        for a in UTILITY_ATTRIBUTES
    )
    return (
        "=== ORIGINAL TRANSCRIPT ===\n"
        f"{original_text}\n"
        "=== END ORIGINAL ===\n\n"
        "=== REWRITTEN TRANSCRIPT ===\n"
        f"{rewritten_text}\n"
        "=== END REWRITTEN ===\n\n"
        "=== MASK MAP (original → replaced) ===\n"
        f"{json.dumps(mask_map, indent=2)}\n\n"
        "=== UTILITY ATTRIBUTES TO EVALUATE ===\n"
        f"{attr_block}\n\n"
        "For EACH attribute, provide:\n"
        "- preserved: true/false (is the core content still present?)\n"
        "- loss_severity: 1-5 (1 = fully preserved, 5 = completely destroyed)\n"
        "- original_content: brief description of what existed in the original\n"
        "- rewritten_content: brief description of what remains in the rewrite\n"
        "- lost_details: list of specific content items that were lost or distorted\n"
        "- recovery_suggestion: how the refiller could recover lost content\n\n"
        "Return JSON:\n{\n"
        '  "THEME": {"preserved": bool, "loss_severity": int, ...},\n'
        "  ... (all 8 attributes)\n"
        '  "total_loss": <sum of all loss_severity scores>\n'
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


def evaluate_preservation(
    original_text: str,
    rewritten_text: str,
    mask_map: dict,
    client=None,
) -> dict:
    """Evaluate how well research content was preserved in the rewrite."""
    client = client or cfg.get_pipeline_client()
    prompt = build_keeper_prompt(original_text, rewritten_text, mask_map)
    return _call_llm_json(client, cfg.KEEPER_MODEL, KEEPER_SYSTEM_PROMPT, prompt)


def total_loss(report: dict) -> int:
    """Extract total loss score from a keeper report."""
    if "total_loss" in report:
        try:
            return int(report["total_loss"])
        except (ValueError, TypeError):
            pass
    total = 0
    for attr in UTILITY_ATTRIBUTES:
        info = report.get(attr.key, {})
        if isinstance(info, dict):
            try:
                total += int(info.get("loss_severity", 0))
            except (ValueError, TypeError):
                pass
    return total
