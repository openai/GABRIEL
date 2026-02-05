# GABRIEL

**GABRIEL** (Generalized Attribute Based Ratings Information Extraction Library) turns messy qualitative corpora into analysis-ready datasets with GPT. It handles prompting, batching, retries, checkpointing, and audit trails so you can treat “ask the model” workflows like any other measurement instrument. From rating rhetoric across a million speeches to matching product catalogs, you focus on the research question while GABRIEL handles the operations.

📓 **Tutorial notebook** (start here!): https://colab.research.google.com/drive/1RMUeAWACpViqiUMlPMMwPTKyGU-OX756?usp=sharing — also available as `gabriel_tutorial_notebook.ipynb` in this repo if you’d like to download and run it locally.

You can install the GABRIEL Python library with `pip install openai-gabriel` and then `import gabriel`.

## Table of contents

- [Why GABRIEL?](#why-gabriel)
- [What can you do with GABRIEL?](#what-can-you-do-with-gabriel)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Task highlights](#task-highlights)
- [Detailed usage](#detailed-usage)
- [Multimodal data and web search](#multimodal-data-and-web-search)
- [Custom prompts and model routing](#custom-prompts-and-model-routing)
- [Saving, logging, and resuming](#saving-logging-and-resuming)
- [Development and testing](#development-and-testing)
- [Citation](#citation)

## Why GABRIEL?

Most of the evidence social scientists and analysts care about lives in unstructured formats: interviews, speeches, transcripts, product photos, archival scans. Modern GPT models can judge attributes, extract facts, and reason about this material with high fidelity, but building robust pipelines is still tedious. GABRIEL provides:

- 🧠 **Human-level comprehension on demand** – express the attribute the way you would brief a human coder; GABRIEL packages the prompt, context, and retries for you.
- 📊 **Quantitative outputs** – ratings (0–100), grounded comparisons, classifications, and structured extractions return as tidy DataFrames with reproducible settings.
- ⚙️ **Operational tooling** – automatic parallelism (hundreds of concurrent calls), resumable runs, raw response logs, and helper UIs make it safe to scale to very large corpora.
- 🧰 **Extensibility** – swap instructions with `additional_instructions`, bring your own templates, or drop down to `gabriel.whatever` + custom `response_fn` for bespoke prompts while still reusing the infrastructure.

The tutorial notebook walks through these ideas step-by-step—from setting up an API key to running multimodal analyses—so skim this README, then dive into the notebook for the full guided tour.

## What can you do with GABRIEL?

### A) Measure attributes on qualitative data

| Function | Purpose & Output Scale | Example Use |
| --- | --- | --- |
| `gabriel.rate` | Asks GPT to score each text / image / audio / item on natural language attributes. Output = 0--100 rating. | Measure “populist rhetoric” in a speech; “toxicity” of tweets; “luxury” in ad images. |
| `gabriel.rank` | Pairwise comparisons between texts yields ELO-like attribute ratings. Output = grounded, relative z scores for each text. | Rank technologies by “bulkiness” or artworks by “fine brushwork”. |
| `gabriel.classify` | Classifies texts / images / audio / items on whether provided labels apply. Output = one or more classes per item. | Tag news articles, product photos, or interview clips into topical categories. |
| `gabriel.extract` | Structured fact extraction on each item. Output = string / numeric values. | For each product, provide the “company”, “CEO”, and “year of invention”. |
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
| `gabriel.bucket` | Builds taxonomies from many terms. Output = bucket/cluster labels. | Group technologies, artworks, or HR complaints into emergent categories. |
| `gabriel.seed` | Enforces a representative distribution / diversity of seeds. | Initialize unique personas that match US population distribution. |
| `gabriel.ideate` | Generates many novel scientific theories and filters the cream of the crop. | Procure novel theories on inflation for potential research. |
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
```

Before running real jobs, point the helpers to your GPT endpoint:

```bash
export OPENAI_API_KEY="sk-..."
# or os.environ['OPENAI_API_KEY'] = "sk-..." inside a Jupyter notebook
```

Every task also accepts `use_dummy=True` for offline dry runs (the tutorial uses this to demonstrate workflows without making API calls).

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
    model="gpt-5-mini",
    n_runs=2,
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
- **`gabriel.rank`** – pairwise tournaments that surface relative winners with grounded z-scores.
- **`gabriel.classify`** – single- or multi-label tagging with label definitions and consensus columns.
- **`gabriel.extract`** – turn passages or multimodal product cards into tidy tables with optional schemas.
- **`gabriel.discover`** – contrast two labeled corpora to learn discriminating features.

### Qualitative coding and review
- **`gabriel.codify`** highlights snippets that match qualitative codes and pairs with **`gabriel.view`** for UI-based auditing.
- **`gabriel.compare`** contrasts paired items (drafts, policies, campaigns) with concise differences/similarities.
- **`gabriel.bucket`** groups terms/entities into emergent taxonomies that feed back into rate/classify flows.

### Data prep and cleanup
- **`gabriel.load`** converts folders of media into spreadsheets with clean IDs and file paths.
- **`gabriel.merge`** / **`gabriel.deduplicate`** produce fuzzy joins and de-duplicated lists using embeddings plus GPT checks.
- **`gabriel.filter`** screens large candidate lists with natural-language conditions.
- **`gabriel.deidentify`** replaces PII with realistic stand-ins to protect privacy.

### Ideation and custom prompts
- **`gabriel.ideate`** and **`gabriel.seed`** generate diverse candidates before deeper measurement.
- **`gabriel.whatever`** runs bespoke prompts (with optional web search or custom `response_fn`) while reusing retries, logging, and checkpointing.

## Multimodal data and web search

Set `modality` to `text`, `entity`, `image`, `audio`, or `web` on any measurement helper. Pair `gabriel.load` with folders of media to build the right DataFrame, and use `web_search=True` when GPT should gather context before rating or extracting. The tutorial’s county-level example shows how to chain web search → rating → mapping in one flow.

## Custom prompts and model routing

- Add clarifications with `additional_instructions` (e.g., mandate mutually exclusive labels).
- Swap in your own Jinja `template_path` while keeping retries and checkpoints.
- Drop to `gabriel.whatever` for fully custom prompts, attachments, or routing logic.

## Saving, logging, and resuming

Each run expands `save_dir` (tilde and environment variables supported), writes structured outputs (`file_name` CSV/Parquet), and saves raw model payloads under `responses/` with metadata for auditability. Leave `reset_files=False` to resume partially completed runs; delete the folder or pass `reset_files=True` to start fresh. `gabriel.view` reads these outputs for quick spot checks, and helpers like `gabriel.utils.mapmaker.MapMaker` can consume the same files for downstream visualization.

## Development and testing

Install development extras and run tests:

```bash
pip install -e .[dev]
pytest
```

Tests rely on the built-in dummy responses, so no API key is necessary. Linting and type checks (`ruff`, `mypy`) are included in the dev extras.

## Citation

If you use GABRIEL in your research, please cite:

> The Generalized Attribute Based Ratings Information Extraction Library (GABRIEL). Hemanth Asirvatham and Elliott Mokski (2026). <https://github.com/openai/GABRIEL>

## License

GABRIEL is released under the Apache 2.0 License. See [LICENSE](LICENSE).
