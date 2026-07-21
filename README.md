# GABRIEL

**GABRIEL** (Generalized Attribute Based Ratings Information Extraction Library) turns messy qualitative corpora into analysis-ready datasets with GPT. It handles prompting, batching, retries, checkpointing, and audit trails so you can treat “ask the model” workflows like any other measurement instrument. From rating rhetoric across a million speeches to matching product catalogs, you focus on the research question while GABRIEL handles the operations.

📓 **Tutorial notebook** (start here!): [GABRIEL Colab notebook](https://colab.research.google.com/drive/1RMUeAWACpViqiUMlPMMwPTKyGU-OX756?usp=sharing) — also available as `gabriel_tutorial_notebook.ipynb` in this repo if you’d like to download and run it locally.

🖥️ **Slides:** [Presentation deck](https://drive.google.com/file/d/1hpslDIEw1_26OTK1mAt9PcU6sYiJHhb8/view?usp=sharing)

You can install the GABRIEL Python library from [PyPI](https://pypi.org/project/openai-gabriel/) with `pip install openai-gabriel` and then `import gabriel`.

> **Compatibility note:** `max_output_tokens` is deprecated in GABRIEL. Existing
> calls remain valid, but the value is ignored and emits a `FutureWarning`;
> GABRIEL leaves the output budget to the selected model.

> **Using a coding agent?** Read [`AGENTS.md`](https://github.com/openai/GABRIEL/blob/main/AGENTS.md).
> It is the repository's shared guide to model choice, data shape, scale,
> checkpoints, and quality assurance.

Read our blog post [here](https://openai.com/index/scaling-social-science-research/), our paper with validation experiments and applied examples [here](http://www.nber.org/papers/w34834), and submit feedback / bugs / feature requests [here](https://forms.gle/RKnBskuiZ64Wt9D66).

## Table of contents

- [Why GABRIEL?](#why-gabriel)
- [What can you do with GABRIEL?](#what-can-you-do-with-gabriel)
- [Installation](#installation)
- [Using GABRIEL with coding agents](#using-gabriel-with-coding-agents)
- [Quick start](#quick-start)
- [Task highlights](#task-highlights)
- [Multimodal data and web search](#multimodal-data-and-web-search)
- [Custom prompts and model routing](#custom-prompts-and-model-routing)
- [Saving, logging, and resuming](#saving-logging-and-resuming)
- [Development and testing](#development-and-testing)
- [Citation](#citation)

## Why GABRIEL?

Most of the evidence social scientists and analysts care about lives in unstructured formats: interviews, speeches, transcripts, product photos, archival scans. Modern GPT models can judge attributes, extract facts, and reason about this material with high fidelity, but building robust pipelines is still tedious. GABRIEL provides:

- 🧠 **Human-level comprehension on demand** – express the attribute the way you would brief a human coder; GABRIEL packages the prompt, context, and retries for you.
- 📊 **Quantitative outputs** – ratings (0–100), pairwise comparisons, classifications, and structured extractions return as tidy DataFrames with reproducible settings.
- ⚙️ **Operational tooling** – automatic parallelism (hundreds of concurrent calls), resumable runs, raw response logs, and helper UIs make it safe to scale to very large corpora.
- 🧰 **Extensibility** – swap instructions with `additional_instructions`, bring your own templates, or drop down to `gabriel.whatever` + custom `response_fn` for bespoke prompts while still reusing the infrastructure.

The tutorial notebook walks through these ideas step-by-step—from setting up an API key to running multimodal analyses—so skim this README, then dive into the notebook for the full guided tour.

## What can you do with GABRIEL?

### A) Measure attributes on qualitative data

| Function | Purpose & Output Scale | Example Use |
| --- | --- | --- |
| `gabriel.rate` | Asks GPT to score each text / image / audio / item on natural language attributes. Output = 0-100 rating. | Measure “populist rhetoric” in a speech; “toxicity” of tweets; “luxury” in ad images. |
| `gabriel.rank` | Pairwise comparisons yield Bradley–Terry attribute ratings. Output = model-derived z-scores standardized within connected comparison components. | Rank technologies by “bulkiness” or artworks by “fine brushwork”. |
| `gabriel.classify` | Classifies texts / images / audio / items on whether provided labels apply. Output = one or more classes per item. | Tag news articles, product photos, or interview clips into topical categories. |
| `gabriel.extract` | Structured fact or entity extraction. One source can return one row or expand into several extracted-entity rows. | Extract every product and its name, price, description, and color from each catalog page. |
| `gabriel.discover` | Discovers natural language features which discriminate two classes of data. | Identify what distinguishes 5 star vs. 1 star reviews or successful vs. failed startups. |

### B) Clean data

| Function | Purpose & Output Scale | Example Use |
| --- | --- | --- |
| `gabriel.merge` | Creates crosswalks. Output = merged table with GPT-matched identifiers. | Match two distinct job title directories; link patent titles to product names. |
| `gabriel.deduplicate` | Detects conceptual duplicates. Maps all duplicates to one representative term. | Collapse “F-18”, “Super Hornet Fighter Jet”, “f-18 hornet” into “F-18”. |
| `gabriel.filter` | High-throughput boolean screening. Outputs items which meet natural language condition. | Subset 18M Wikipedia titles to only technologies. |
| `gabriel.deidentify` | Replaces PII with realistic, consistent fake PII. Outputs anonymized text + mapping. | Replace names, employers, addresses before sharing interview corpora. |

### C) Helper tools

| Function | Purpose & Output Scale | Example Use |
| --- | --- | --- |
| `gabriel.codify` | Passage coding: highlights snippets in text that match qualitative codes. | Flag sentences about “economic insecurity” in speeches; “stressors” mentioned in interview. |
| `gabriel.compare` | Identifies similarities / differences between paired items. Output = list of differences. | Contrast op-eds from different districts; compare two ad campaigns. |
| `gabriel.bucket` | Proposes a compact taxonomy from many terms. Output = bucket names and definitions, not row-level assignments. | Develop candidate categories for technologies, artworks, or HR complaints, then review and apply them downstream. |
| `gabriel.seed` | Enforces a representative distribution / diversity of seeds. | Initialize unique personas that match US population distribution. |
| `gabriel.poll` | Seeds personas, expands them into full biographies, and surveys them. | Simulate a synthetic opinion poll on policy, values, and open-ended attitudes. |
| `gabriel.ideate` | Generates and filters candidate scientific theories for further evaluation. | Procure novel theories on inflation for potential research. |
| `gabriel.debias` | Post-process measurements to remove inference bias. | Ensure GPT isn't guessing climate opinions in speeches based on general political lean. |
| `gabriel.load` | Prepares a folder of text / image / audio files into a spreadsheet for use in GABRIEL. | Image directory converted into spreadsheet of file paths. |
| `gabriel.view` | UI to view sample texts with ratings / passage coding. | Spot-check classify / rating outputs; view coded passages. |
| `gabriel.paraphrase` | Rewrites texts consistently per instructions. | Summarize earnings call transcripts to remove company specifics. |
| `gabriel.whatever` | Run any GPT prompts, but leverage GABRIEL's parallelization / checkpointing. | Any set of prompts; slots into any pipeline. |

## Installation

```bash
pip install openai-gabriel

# or install directly from GitHub
pip install \
  --force-reinstall \
  git+https://github.com/openai/GABRIEL.git@main

# then run import gabriel
```

Before running a GABRIEL call, declare your GPT API key:

```bash
export OPENAI_API_KEY="sk-..."
```

In Colab, store the key in **Secrets** under `OPENAI_API_KEY` and read it with
`google.colab.userdata`; do not paste a real key into notebook source, chat,
screenshots, or Git. The tutorial shows the complete setup.

## Using GABRIEL with coding agents

The repository's [`AGENTS.md`](https://github.com/openai/GABRIEL/blob/main/AGENTS.md)
is a concise operating guide for coding agents. It
explains task selection, model routing, checkpointing, multimodal inputs, and QA.
Agents should also read `gabriel_tutorial_notebook.ipynb` closely before designing
a workflow; it is the broadest guide to what the library can do.
Three rules are especially important:

- **Verify model slugs live.** At this documentation's last review, the exact general
  models were `gpt-5.6-sol`, `gpt-5.6-terra`, and `gpt-5.6-luna`, and audio used
  `gpt-audio-1.5`. Model IDs change; check the
  [official model catalog](https://developers.openai.com/api/docs/models) and
  [pricing](https://developers.openai.com/api/docs/pricing), plus the models
  actually provisioned for your organization or gateway, instead of guessing or
  copying an old example. Luna is often a sensible starting point for many
  applications and large-scale work. For unusually subtle, knowledge-heavy, multi-constraint,
  or writing-sensitive work, pilot Terra and Sol on representative examples and
  use the smallest model that reliably meets the research standard.
- **Generally keep `n_parallels=650`, including for small tests.** It is a ceiling, not a fixed worker
  count. GABRIEL ramps up, reads rate-limit signals, retries transient failures,
  and lowers actual concurrency when needed. Temporary retries and a few slow
  stragglers are normal: while the run is progressing, let it finish. Raising
  the ceiling can help a large workload when provisioned limits support more
  throughput; lowering it may help a known deployment constraint or persistent
  failure. A small test does not require a lower ceiling.
- **Pilot, checkpoint, then scale.** Test a representative sample, inspect raw
  responses and output shape, and keep `reset_files=False` to resume a compatible
  run. Use a new `save_dir` when the model, prompt, fields, or labels change.

## Quick start

The tutorial notebook walks through many complete projects; here’s the minimal rating flow the notebook starts with. Paste this into Colab or a notebook cell so you can use `await` directly:

```python
import os
import pandas as pd

import gabriel

PATH = os.path.expanduser("~/Documents/gabriel_runs")
toy_data = pd.DataFrame(
    {
        "entity": [
            "turkey",
            "pumpkin pie",
            "green bean casserole",
            "cornbread",
        ]
    }
)

attributes = {
    "savory taste": "How savory the dish is",
    "sweet taste": "Dessert-like sweetness",
    "tangy taste": "Notes of tartness or acidity",
}

rate_results = await gabriel.rate(
    toy_data,
    column_name="entity",
    attributes=attributes,
    save_dir=os.path.join(PATH, "toy_rate"),
    model="gpt-5.6-luna",
    n_parallels=650,  # normally leave this default unchanged
    n_runs=1,
    modality="entity",
    reset_files=True,
)
rate_results.head()
```

The helper returns a `pandas.DataFrame` with one column per attribute and writes raw model responses + configs to `save_dir`. Running the same code in a plain Python script just requires wrapping the coroutine with `asyncio.run(...)`.

## Task highlights

The tutorial notebook covers full projects end-to-end. The list below matches its main use cases so you can jump to the right helper quickly.

### Measurement primitives
- **`gabriel.rate`** – assign 0–100 scores per attribute across text, entities, images, audio, or web-sourced context.
- **`gabriel.rank`** – pairwise tournaments fitted with a Bradley–Terry model, with symmetric pseudo-comparison regularization on observed edges when `learning_rate > 0`; a zero value requires Ford's directed strong-connectivity condition within each nontrivial observed component for a finite fit. Draws are encoded as half-wins in a binary Bradley–Terry working objective (one comparison total), not as a three-outcome tie likelihood. `insufficient_signal_policy="tie"` makes the modeling assumption that missing direct evidence is equality; use `"abstain"` to condition fitting on retained judgments, noting that informative abstention can still bias estimates. In non-recursive runs, z-scores are standardized separately within connected comparison components (with a neutral zero fallback when a component has no score dispersion), and `_component` columns identify which values are mutually comparable; isolated items remain unranked with `NaN` scores and standard errors. `<attribute>_se` is an asymptotic, model-based sandwich standard error for the component-centered raw log-skill `<attribute>_raw`, computed while treating the realized comparison graph as fixed. Observed comparison weights determine the meat under the binary-BT working variance, while fixed pseudo-comparisons add curvature to the bread. The standard error excludes shrinkage bias, shared-judge dependence, misspecification, adaptive-selection uncertainty, and uncertainty for the reported z-score. A non-recursive `<file_name>_diagnostics.csv` sidecar records decoded outcome counts, effective comparison weight, graph components, isolates, and the count of finite standard errors. Resume metadata fingerprints the effective judge settings and input payloads; `judge_version` is required for custom judges and must change whenever their hidden logic changes. Planned rounds and Batch jobs are recovered transactionally rather than silently mixed or resubmitted.
- **`gabriel.classify`** – single- or multi-label tagging with label definitions and consensus columns.
- **`gabriel.extract`** – extract typed fields or multiple entities from each passage, page, image, PDF, or audio item; one source may intentionally produce several rows.
- **`gabriel.discover`** – contrast two labeled corpora to learn discriminating features.

### Qualitative coding and review
- **`gabriel.codify`** highlights snippets that match qualitative codes and pairs with **`gabriel.view`** for UI-based auditing.
- **`gabriel.compare`** contrasts paired items (drafts, policies, campaigns) with concise differences/similarities.
- **`gabriel.bucket`** proposes bucket names and definitions from a large term universe; review that taxonomy, then use `classify` or another mapping step when row-level assignments are needed.

### Data prep and cleanup
- **`gabriel.load`** converts folders of media into spreadsheets with clean IDs and file paths.
- **`gabriel.merge`** / **`gabriel.deduplicate`** produce fuzzy joins and de-duplicated lists using embeddings plus GPT checks.
- **`gabriel.filter`** screens large candidate lists with natural-language conditions.
- **`gabriel.deidentify`** replaces PII with realistic stand-ins to protect privacy.

### Ideation and custom prompts
- **`gabriel.ideate`** and **`gabriel.seed`** generate diverse candidates before deeper measurement.
- **`gabriel.whatever`** runs bespoke prompts (with optional web search or custom `response_fn`) while reusing retries, logging, and checkpointing.

## Multimodal data and web search

Set `modality` to `text`, `entity`, `pdf`, `image`, `audio`, or `web` on any measurement helper. Pair `gabriel.load` with folders of media to build the right DataFrame, and use `web_search=True` when GPT should gather context before rating or extracting. The tutorial’s county-level example shows how to chain web search → rating → mapping in one flow.

For image understanding, pass `image_detail="low"`, `"high"`, `"original"`,
or `"auto"`; this is distinct from image-generation `quality`. Audio inputs need
an audio-capable model and are routed through Chat Completions rather than the
Responses API. At this documentation's 2026-07-13 review, use
`gpt-audio-1.5`; `gpt-audio-mini` is deprecated. Verify the current audio slug
before running an old example. Treat GABRIEL's pre-run cost estimate as rough
for audio, tool calls, and very long contexts; consult live API pricing for the
applicable modality and processing tier.

## Custom prompts and model routing

- Add clarifications with `additional_instructions` (e.g., mandate mutually exclusive labels).
- Swap in your own Jinja `template_path` while keeping retries and checkpoints.
- Drop to `gabriel.whatever` for fully custom prompts, attachments, or routing logic.
- Treat every model name in an example as a dated snapshot; verify the exact current slug and endpoint support before overriding a default.

## Saving, logging, and resuming

Each run expands `save_dir` (tilde and environment variables supported), writes
cleaned outputs, and checkpoints raw calls in task-specific files such as
`*_raw_responses.csv`. Leave `reset_files=False` to resume a compatible partial
run. When the research specification changes, use a fresh directory or pass
`reset_files=True` intentionally so old responses are not mixed with a new
prompt. Keep large generated outputs and private datasets outside the source
repository. `gabriel.view` reads results for quick spot checks, and helpers like
`gabriel.utils.mapmaker.MapMaker` can consume the same files downstream.

## Development and testing

Install development extras and run tests:

```bash
pip install -e .[dev]
pytest
```

Tests rely on the built-in dummy responses, so no API key is necessary. Linting and type checks (`ruff`, `mypy`) are included in the dev extras.

## Citation

If you use GABRIEL in your research, please cite:

- Asirvatham, H., Mokski, E., and Shleifer, A. (2026). *GPT as a Measurement Tool*. NBER Working Paper No. 34834.

- Asirvatham, H. and Mokski, E. (2026). *GABRIEL: Generalized Attribute-Based Ratings Information Extraction Library* (software). GitHub repository: https://github.com/openai/GABRIEL

## License

GABRIEL is released under the Apache 2.0 License. See [LICENSE](LICENSE).
