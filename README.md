# AURA: Adaptive Utility-preserving Re-identification-resistant Anonymization

> Reference code for the NeurIPS 2026 submission.

| [About](#about) | [Repository layout](#repository-layout) | [Quick start](#quick-start) | [Reproducing the paper](#reproducing-the-paper) | [Privacy notes](#privacy-notes) | [Citation](#citation) |

## About

This repository contains the implementation and evaluation harness for
**AURA**, a privacy-rewriting pipeline that anonymizes interview-style
conversation transcripts while preserving downstream qualitative utility.
The pipeline iterates between a **Masker** that proposes [MASK_i] spans
on a span level, a **Refiller** that fills them in with privacy-preserving
generalizations, and an **Attacker / Keeper** pair that adversarially
probes the rewrite to decide whether further masking is needed.  An
**adaptive-privacy** wrapper additionally probes each transcript with web
search to discover transcript-specific privacy attributes that are not
covered by the eight base attributes (age, sex, location, occupation,
education, relationship status, income, place of birth).

The design and empirical results are described in our NeurIPS 2026 paper.

## Repository layout 

```
NIPS_CODE/
├── README.md                     # this file
├── LICENSE                       # MIT
├── .gitignore
├── requirements.txt              # combined deps for both subfolders
│
├── AURA/         # the AURA pipeline itself
│   ├── README.md
│   ├── requirements.txt
│   ├── .env.example
│   ├── pipeline.py               # 4-phase rewrite over a single transcript
│   ├── run_expanded_privacy.py   # adaptive-privacy wrapper
│   ├── run_pure_adaptive_attri.py
│   ├── run_qwen_expanded_batch.py
│   ├── run_openrouter_sample.py
│   ├── phase{0,1,2}*.py          # phase implementations
│   ├── pipeline_config.py
│   ├── db.py
│   └── input/example_transcripts.jsonl   # synthetic example
│
└── EVAL/         # re-identification & utility evaluation
    ├── README.md
    ├── requirements.txt
    ├── _compat.py                # vendored constants and CSV helpers
    ├── direct_intent.py          # web-search re-id over a CSV of transcripts
    ├── identifier_profile_preservation.py
    ├── evaluate_code_fact_recoverability.py
    └── input/adaptive_attri/example_rewritten.csv   # synthetic example
```

The two subfolders are independent: `EVAL/` operates on
CSVs and JSONs and never imports from `AURA/`.

## Quick start

```bash
git clone <this-repo> aura && cd aura/NIPS_CODE

# 1. Install dependencies (Python 3.10+).
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys.  The pipeline needs OPENAI_API_KEY at minimum;
#    OpenRouter / Gemini are only needed if you exercise those code paths.
cp AURA/.env.example AURA/.env
$EDITOR AURA/.env

# 3. Run the AURA pipeline on the shipped synthetic transcripts.
cd AURA
python pipeline.py --reset-db
# → output/<name-prefix>_rewritten.csv

# 4. Probe the rewrite with web-search re-id.
cd ../EVAL
python direct_intent.py ../AURA/output/<name-prefix>_rewritten.csv
# → web_search_<name-prefix>_rewritten.csv.json
```

For details on every CLI flag, see the per-folder READMEs:

* [`AURA/README.md`](AURA/README.md) — pipeline,
  adaptive variants, and OpenRouter / Qwen on-device runs.
* [`EVAL/README.md`](EVAL/README.md) —
  re-identification and utility evaluation harness.

## Reproducing the paper

The paper reports three families of results, each driven by code in this
repository:

1. **Adaptive-privacy rewrites** under different LLM backbones
   (`gpt-4.1`, `qwen/qwen3.5-27b`, `qwen/qwen3.5-35b-a3b`).  Use
   `run_expanded_privacy.py` with `NB_LLM_PROVIDER=openai`, or
   `run_qwen_expanded_batch.py` with `NB_LLM_PROVIDER=openrouter` and the
   appropriate Qwen model id.
2. **Re-identification rates** under three attacker models
   (`gpt-5.1`, `gpt-5.4-mini`, `gemini-3-flash-preview`).  Use
   `direct_intent.py` for the web-search probe and
   `identifier_profile_preservation.py --workflow reid_compare` for the
   atomic-fact comparison against the original-transcript candidates.
3. **Utility preservation** measured as profile-fact and code-fact
   recoverability and combined into the per-transcript utility grid.
   Use `identifier_profile_preservation.py --workflow profile_recoverability`
   and `evaluate_code_fact_recoverability.py`.

The original Anthropic transcripts and the per-transcript reference fact
files used in the paper are not redistributed; see
[Privacy notes](#privacy-notes) below.

## Privacy notes

The ad-hoc artifacts produced by both subfolders are sensitive:

* **Source transcripts.** AURA was developed against a non-public
  re-identification corpus.  We ship a synthetic
  `example_transcripts.jsonl` so the pipeline is runnable out of the box,
  but you must bring your own data to reproduce the paper numbers.
* **Re-id candidate JSON.** `direct_intent.py` saves the model's free-form
  identification candidates next to the original transcripts.  These
  candidates frequently contain copy-pasted spans of the original text and
  are therefore PII-equivalent.
* **Reference fact files.** The eval harness writes per-transcript
  `excerpt`, `example_quote`, and `evidence_quote` fields that are
  verbatim transcript spans.  See
  [`EVAL/README.md`](EVAL/README.md#privacy-data-we-do-not-ship)
  for the recommended aliasing strategy when you regenerate them on a
  private workspace.

`*.db`, `*.db-shm`, `*.db-wal`, `output/`, `responses_*/`, and
`identifier_profile_preservation_results/` are listed in
[`.gitignore`](.gitignore) to make accidental redistribution harder.

## Acknowledgements

Implementation builds on the OpenAI Responses API, the OpenRouter
inference layer, and the Tavily web-search API for adaptive-attribute
discovery. 

## Citation


(Replace with the camera-ready BibTeX once the paper is accepted.)

## License

This codebase is released under the [MIT License](LICENSE).
