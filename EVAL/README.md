# EVAL

Companion evaluation harness for the AURA privacy-rewriting release in
[`../AURA`](../AURA).  These scripts measure
**re-identification risk** and **utility preservation** on rewritten
transcripts.

| Script | What it does |
|---|---|
| [`direct_intent.py`](direct_intent.py) | Web-search re-identification probe (OpenAI Responses API + `web_search` tool) over a CSV of transcripts. |
| [`identifier_profile_preservation.py`](identifier_profile_preservation.py) | Two workflows: (a) compare re-id JSON outputs and extract atomic profile facts; (b) build an 8-attribute reference profile and judge per-config recoverability. |
| [`evaluate_code_fact_recoverability.py`](evaluate_code_fact_recoverability.py) | LLM-judged recoverability of pre-built deterministic code facts under each rewriting config. |
| [`_compat.py`](_compat.py) | Vendored constants (`ATTRIBUTE_SPECS`, `CFG_ORDER`, `CFG_DISPLAY`) and CSV helpers used by the two `*.py` evaluators above. |

## 📁 Folder layout

```
EVAL/
├── README.md                              # this file
├── _compat.py                             # vendored constants and CSV helpers
├── direct_intent.py                       # web-search re-id over a CSV of transcripts
├── evaluate_code_fact_recoverability.py   # LLM-judged code-fact recoverability per config
├── identifier_profile_preservation.py     # 8-attribute profile facts + recoverability
├── requirements.txt
└── input/
    └── adaptive_attri/
        └── example_rewritten.csv          # synthetic example, see "Privacy" below
```

The scripts emit results next to themselves under
`identifier_profile_preservation_results/` and `code_fact/`.  Both directories
are listed in [`../.gitignore`](../.gitignore).

## 🔒 Privacy: data we do not ship

The full evaluation harness ordinarily produces four families of artifacts
that we cannot redistribute:

| Tree | Why we omit it |
|---|---|
| `code_fact/reference_facts/<sample>.json` | `excerpt` and `example_quote` fields are lifted **verbatim** from the original transcripts. |
| `profile/reference_profiles/<sample>.json`, `profile/reference_summaries/<sample>/<ATTR>.json` | Each attribute summary embeds a verbatim `evidence_quote`. |
| `code_fact/recoverability/<config>/<sample>.json`, `profile/recoverability/<config>/<sample>.json` | Same `evidence_quote` issue, plus filenames encode internal `transcript_id` slugs. |
| `utility_grid/<config>/<sample>/grid.json`, `utility_grid/summary.json` | Filenames and JSON keys encode the internal `transcript_id`. |



The shipped `input/adaptive_attri/example_rewritten.csv` contains two
fully synthetic transcripts to make `direct_intent.py` immediately runnable
without supplying any private data.

## ⚙️ Setup

```bash
cd EVAL
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Add OPENAI_API_KEY (and optionally GEMINI_API_KEY) to:
#   ../AURA/.env   (loaded automatically), OR
#   ./.env                         (also auto-discovered)
cp ../AURA/.env.example ../AURA/.env
nano ../AURA/.env
```

## 📖 Usage

### 1. Web-search re-identification (`direct_intent.py`)

Reads a CSV of transcripts (columns `transcript_id`, `text`) and asks the
configured OpenAI model to identify the speaker, optionally validating
candidates with the `web_search` tool.  Output is streamed to JSON so the
script can be safely interrupted and resumed.

```bash
python direct_intent.py input/adaptive_attri/example_rewritten.csv \
    --output web_search_example.json \
    --model gpt-5.4-mini
```

`responses_<input_stem>/<transcript_id>_full_response.txt` contains the raw
Responses API trace per transcript for auditing.

### 2. Profile recoverability (`identifier_profile_preservation.py`)

#### Workflow A — `reid_compare`

Compare two or more re-identification JSON outputs (such as those produced
by `direct_intent.py`) by extracting atomic profile facts from each
candidate description and ranking configs against a reference label.

```bash
python identifier_profile_preservation.py \
    --workflow reid_compare \
    --input original=path/to/web_search_originals.json \
    --input adaptive_privacy=web_search_example.json \
    --reference-label original \
    --model gpt-4.1
```

#### Workflow B — `profile_recoverability`

Build an 8-attribute reference profile from the original transcripts
(`--reference-only`), then run the full per-config recoverability evaluation.
You must point `--original-csv` at a private CSV with columns
`transcript_id` and `text`.

```bash
# Reference profile build only.
python identifier_profile_preservation.py \
    --workflow profile_recoverability \
    --original-csv path/to/private_originals.csv \
    --reference-only

# Per-config recoverability for the AURA Adaptive-Privacy variant.
python identifier_profile_preservation.py \
    --workflow profile_recoverability \
    --original-csv path/to/private_originals.csv \
    --adaptive-path path/to/aura_rewritten.csv \
    --only-configs adaptive_privacy
```

The full set of `--<config>-path` flags is documented via
`python identifier_profile_preservation.py --workflow profile_recoverability --help`.
Any config that is not pointed at a CSV is silently skipped.

### 3. Code-fact recoverability (`evaluate_code_fact_recoverability.py`)

For each `(sample, config)` pair, judges whether each pre-built code fact
(emitted by your private `build_code_facts.py`) can be recovered from the
rewritten transcript.

```bash
python evaluate_code_fact_recoverability.py \
    --reference-root code_fact/reference_facts \
    --output-root code_fact/recoverability \
    --original-csv path/to/private_originals.csv \
    --adaptive-path path/to/aura_rewritten.csv \
    --only-configs adaptive_privacy \
    --model gpt-4.1
```

Per-sample outputs land at `<output-root>/<config>/<sample>.json`; an
aggregate `<output-root>/summary.json` is written and then merged on
subsequent runs (so adding new configs does not overwrite existing ones).

## 💡 Notes

* `--model` defaults to `gpt-4.1` for the two main evaluators and
  `gpt-5.4-mini` for `direct_intent.py`.  For Gemini judges, install
  `google-genai` (already pinned in `requirements.txt`) and pass
  `--model gemini-3-flash-preview` (or similar).
* `_compat.py` ships verbatim copies of the eight base privacy attributes
  used in the AURA paper, plus the configuration enumeration
  (`CFG_ORDER` / `CFG_DISPLAY`).  Edit it if you add or rename
  rewriting variants.
