# GABRIEL agent guide

This is the canonical operating guide for coding agents working in this
checkout or writing code that uses GABRIEL. Keep it concise and update it when
public behavior changes. The README explains the package; the tutorial notebook
contains runnable examples; `src/gabriel/api.py` is the source of truth for
public signatures.

The recommendations below are practical heuristics, not universal rules. They
reflect recurring successful workflows and package behavior at the date shown;
they may be stale, unavailable in a particular organization, too expensive, or
wrong for a specific dataset. User intent, the live API/package, and evidence
from a representative pilot take precedence.

Before improvising a workflow, read `gabriel_tutorial_notebook.ipynb` carefully.
It is the most complete capability map and shows how tasks, modalities, custom
endpoints, checkpoints, visualization, and downstream analysis fit together.

## Before writing or running code

- Inspect the actual environment first. Confirm `gabriel.__file__`, the
  installed `openai-gabriel` version, and the live function signature instead
  of assuming the PyPI package, this checkout, and an old example are identical.
- Inspect unfamiliar scripts and inputs before executing them. For a paid run,
  confirm the row count, prompt count, model, output directory, and expected
  cost, then pilot a small representative sample before scaling.
- Prefer the public `gabriel.*` wrappers. Use `await` in notebooks and
  `asyncio.run(...)` in ordinary Python scripts.
- Never place API keys, private data, machine-specific paths, or generated
  research artifacts in committed examples.

## Models change: verify them live

Never invent a model slug or copy one blindly from an old notebook. Check the
[official model catalog](https://developers.openai.com/api/docs/models), the
[pricing page](https://developers.openai.com/api/docs/pricing), and the models
actually provisioned for the user's organization or gateway before overriding a
default or publishing a release. Availability, aliases, rate limits, and feature
support can vary by organization.

At this guide's last review (2026-07-13), GABRIEL used this snapshot:

| Workload | Exact model slug | Typical use |
| --- | --- | --- |
| Fast and inexpensive | `gpt-5.6-luna` | Often a sensible starting point, especially for large-scale runs; default for rate, classify, rank, extract, filter, and merge |
| Balanced | `gpt-5.6-terra` | Consider when a representative pilot materially improves over Luna and does not require Sol |
| Full capability | `gpt-5.6-sol` | Very complex or subtle work with many moving parts, niche knowledge, difficult synthesis, or important writing quality |
| Audio understanding | `gpt-audio-1.5` | Audio input through Chat Completions |

These are dated facts, not permanent aliases. Verify that a prospective model
supports the required endpoint, modality, reasoning setting, and structured
output behavior. Do not mechanically replace an audio model with a text model.

A common starting point is Luna, particularly at scale. If quality is uncertain,
run a small representative comparison across Luna, Terra, and Sol and inspect
task-specific accuracy, omissions, consistency, latency, and cost. Prefer the
smallest model that reliably meets the research standard. Escalate to Sol when
the work is genuinely complex or subtle, has many interacting constraints,
depends on niche internal knowledge, or when the quality of the reasoning or
writing is itself important. Do not choose model size mechanically from the
GABRIEL function name alone.

## Defaults and scale heuristics

- Generally keep `n_parallels=650`, including for small tests. It is a ceiling,
  not a fixed worker count, so a small job does not create 650 unnecessary
  requests. GABRIEL ramps workers, observes rate limits, retries transient
  failures, and reduces actual concurrency for sustained errors, web search,
  and media. Temporary retries and a few slow stragglers are normal; while a run
  is progressing, it is usually best to let it finish. Raising the ceiling can
  be reasonable when a large workload needs more throughput and the account or
  gateway supports it. Reducing it can be appropriate for a known deployment
  constraint or persistent failure, but is rarely needed merely because a test
  is small. These are operational suggestions, not hard limits.
- Do not pass `max_output_tokens`. GABRIEL accepts it only for compatibility,
  emits a `FutureWarning`, and ignores it.
- Keep `reset_files=False` to resume a compatible checkpoint. If the model,
  prompt, labels, attributes, batching, or meaning of a run changes, use a new
  `save_dir` or intentionally reset; never mix two specifications in one cache.
- Normally leave `n_attributes_per_run=None`. Current models can often handle
  dozens of labels or fields together—including 30, 40, or 50—while retaining
  useful cross-attribute context. The parameter remains available if an actual
  pilot reveals a context, schema-adherence, quality, or provider constraint;
  do not split attributes preemptively.
- Reuse GABRIEL's retries and checkpoints. Do not wrap it in ad hoc thread pools
  or rerun an entire corpus when only a small failed subset needs repair.

For fewer than roughly 100,000 rows, the simplest approach is usually to pass
the full DataFrame directly to one GABRIEL run and let its checkpointing and
adaptive concurrency operate. At or above roughly 100,000 rows, deterministic
batches processed separately may make scheduling, recovery, parallel job
compute, and exports easier. This is a rough operational threshold, not a
quality boundary. If batching helps, preserve stable source identifiers and the
same specification across batches, then union and audit the results.

## Choose the right primitive

`extract`, `classify`, and `rate` cover a large share of real projects.
`extract` is often the most flexible because a single call can combine several
different kinds of structured judgment. The other primitives remain useful
when their specialized behavior matches the question.

| Need | Use | Important behavior |
| --- | --- | --- |
| Mixed structured outputs | `extract` | Often the most flexible choice: one pass can return facts, classifications, ratings, evidence, explanations, and mixed boolean, categorical, numeric, or text fields. It can also expand one source into several extracted-entity rows. |
| One or more label columns | `classify` | Labels are independent booleans by default, so one row can receive several classes. For exactly one class, make labels mutually exclusive and say "choose exactly one" in `additional_instructions`; audit the result because this is prompted behavior, not a hard invariant. |
| Continuous 0-100 measurements | `rate` | Convenient when several attributes share a clearly defined numeric scale and the built-in rating aggregation is useful. |
| Relative ordering | `rank` | Pairwise comparisons produce relative z-scores; potentially useful when fine distinctions are easier than absolute scores. |
| Matching passages inside documents | `codify` | Use when the evidence snippet matters, not merely a document-level label. |
| Broad inexpensive screening | `filter` | Reduce a candidate universe before a more expensive task. |
| Building an emergent taxonomy | `bucket` | Takes a large universe of terms, lists, or term-to-definition mappings and iteratively proposes and votes on a compact set of relatively distinct bucket names and definitions. It returns the taxonomy, not an assignment for every source row; the reviewed buckets can then inform `classify` or another mapping step. |
| Finding features that distinguish groups | `discover` | Contrasts two groups of texts or paired columns to propose recurring discriminating features that can later become reviewed measurement definitions. |
| Fuzzy crosswalks and normalization | `merge`, `deduplicate` | Preserve raw values beside canonical mappings for auditability. |
| Controlled rewriting or privacy transforms | `paraphrase`, `deidentify` | Inspect samples before trusting a large transformed corpus. |
| Diverse generation | `seed`, `poll`, `ideate` | Generation, synthetic-persona, survey, and ranked-idea workflows. |
| Fully custom prompts | `whatever` | Keeps GABRIEL's parallelism, retries, checkpoints, attachments, and optional JSON parsing. |

Attribute, field, and label descriptions are the measurement specification.
Write concrete inclusion/exclusion criteria and operational examples; names alone
are rarely enough. Use `additional_instructions` for cross-label constraints.

For `extract`, the `types` mapping is downstream coercion, not a replacement for
clear field descriptions. If delivery requires one row per original source,
explicitly aggregate multi-entity results while preserving every extracted value
rather than silently dropping extra rows.

`extract` is not limited to literal named entities. It can act as a broad
structured-analysis primitive for multi-field classification, scores plus
explanations, linked evidence, or several records embedded in one source.
`classify` may be simpler when the desired output is only independent boolean
label columns, while `rate` provides purpose-built repeated numeric ratings.

## Useful workflow patterns

Treat these as possibilities to adapt, not required recipes:

- Build a representative pilot that includes easy cases, hard cases, long
  inputs, missing information, multilingual or messy examples, and any edge
  case that could change the conclusion. Inspect rows, not just averages.
- One possible large-scale funnel is `filter` for broad screening, then
  `classify`/`extract`/`rate` on survivors, and `rank` only when relative
  ordering or the extreme top tail matters. Skip stages that add no value.
- For exploratory work, `discover` or `bucket` can propose a vocabulary. Human
  review and sharpen those definitions before treating them as a fixed
  measurement instrument for `classify` or `rate`.
- One possible open-ended workflow is to `extract` free-form entities, events,
  or judgments; `bucket` those values into a candidate taxonomy; review the
  definitions; then `classify` the full corpus and `rate` intensity or quality
  where a continuous measure is useful. Each stage is optional.
- Prefer explicit field and label definitions over clever orchestration.
  Examples, inclusion/exclusion rules, an entity unit, and instructions for
  unknown values often improve results more than adding another pipeline layer.
- Use multiple runs when model variation matters, but first decide whether the
  downstream statistic should be a mean, vote, union, or disagreement flag.
  Preserve per-run outputs when disagreement itself is informative.
- On large jobs, keep stable source IDs, use a new descriptive `save_dir` for a
  changed specification, let checkpointing work, audit the completed table, and
  repair only missing or failed rows/cells. Separately managed batches are an
  operational option at very large scale, not an automatic quality improvement.
- Keep raw responses, the full parsed dataset, and any filtered/ranked shortlist
  as separate artifacts when each serves a different audit purpose. This is
  especially useful for ideation and multi-stage research workflows.
- Cost estimates are planning aids. Recheck the live printed estimate, current
  pricing, prompt count, retries, tool use, long-context surcharges, audio/media
  billing, and output shape before a large run.
- Use GABRIEL's built-in printouts as the first monitoring surface. The startup
  summary and live progress updates expose useful information about the model,
  workload, rate limits, planned concurrency, estimated cost, throughput,
  retries, and remaining work. Keep an eye on those signals during long runs;
  temporary retries and a few slow stragglers are expected while progress
  continues. Preserve the console log for remote or expensive jobs when useful,
  but still audit the completed table afterward because a finished progress bar
  is not proof that every requested field parsed successfully.
- When a result looks odd, inspect the rendered prompt, raw response, cached run
  metadata, installed package path/version, and a handful of source rows before
  rewriting the whole workflow.

## Shape the data for the question

GABRIEL generally accepts a pandas DataFrame and a `column_name` identifying the
content to analyze. Useful input units range from individual terms to long-form
documents and media. These are common patterns, not restrictions:

- Prefer one row per meaningful source unit: for example a speech, article,
  interview response, product listing, archival page, technology, job title,
  legal case, image, audio clip, or PDF. Keep an explicit stable source ID and
  preserve useful provenance or grouping columns beside the analyzed column.
- The model generally analyzes the selected `column_name`, not every metadata
  column in the DataFrame. If date, speaker, location, surrounding text, or
  another field should affect the judgment, construct an explicit context
  column or use the task's supported context parameters rather than assuming
  the model sees the rest of the row.
- Use `modality="text"` for passages or documents whose supplied words are the
  evidence. Use `modality="entity"` for short names or concepts where model
  knowledge is part of the task. `modality="web"` can add current web context
  for entities or questions when that is appropriate.
- A text cell can be a sentence, transcript, article, or substantial document;
  an entity cell can be a short term; image, audio, and PDF modalities normally
  use file paths. Separate mixed modalities into coherent calls unless a custom
  `whatever` workflow is genuinely useful.
- Whole documents can work when full context matters. Split at meaningful
  boundaries only when document length, precision, or the desired unit of
  analysis makes that useful, and retain source/chunk IDs plus an explicit
  recombination plan.
- CSV, TSV, Excel, Parquet, and Feather tables can be loaded directly.
  `gabriel.load(...)` can also construct a DataFrame from folders of text,
  images, audio, or PDFs while retaining filenames and directory layers.
  For a table file with no modality or a text, entity, or web modality,
  `gabriel.load(...)` reads that file rather than an aggregate cache.
- Direct PDF modality preserves layout, figures, and images. Text extraction
  may be preferable when only words matter or when text from several file types
  should share one run. Images and audio can be analyzed directly rather than
  first forcing them through OCR or transcription when visual or acoustic
  evidence matters.
- Preserve the raw source column even when creating cleaned or shortened input.
  Examples, definitions, instructions for unknown values, and the intended unit
  of analysis often matter more than elaborate preprocessing.
- Decide the desired output unit before running. `rate` and `classify` normally
  preserve one row per source; `extract` can either return several fields on one
  source row or intentionally expand one source into multiple entity rows. If a
  later deliverable needs one source row, aggregate explicitly without losing
  the extracted values.

## Modalities and passthroughs

- Use `gabriel.load(...)` to turn folders of supported files into DataFrames,
  and make the call's `modality` match the input column.
- For image understanding, pass `image_detail="low"`, `"high"`, `"original"`,
  or `"auto"`. This becomes `detail` on each API `input_image`. It is different
  from image-generation `quality`.
- Audio input requires an audio-capable model and uses Chat Completions. Verify
  the current audio slug before running; audio is not supported in batch mode.
- Web search and JSON mode cannot currently be combined; GABRIEL warns and
  disables JSON mode. Inspect the raw response shape before parsing a web run.
- When supplying a custom structured-output schema, use the API's supported
  strict JSON Schema subset and pilot the exact schema; a syntactically valid
  Python dictionary is not proof the API will accept every schema feature.
- `response_fn` replaces one model call. `get_all_responses_fn` replaces the
  orchestration layer. Use the narrowest override that solves the problem.

## Outputs, recovery, and QA

- Put paid-run outputs and datasets outside the source checkout. Give each
  experiment a stable, descriptive `save_dir`.
- Preserve the cleaned result, raw-response checkpoint, and run metadata.
- After a run, audit expected identifiers, row counts, duplicate IDs, nulls,
  parse failures, value ranges, and any task-specific `Successful` field.
- A completed process is not proof of a complete table. Repair missing rows or
  cells from the checkpoint rather than paying to regenerate successful work.
- `extract` may legitimately increase row count. Decide explicitly whether the
  downstream unit is the extracted entity or the original source.
- Spot-check examples with `gabriel.view` and read raw responses when the output
  distribution looks surprising.

## When modifying GABRIEL itself

- Preserve unrelated user changes in a dirty worktree.
- Keep public defaults synchronized across `src/gabriel/api.py`, task configs,
  parsing fallbacks, pricing, README, tutorial examples, and tests.
- Deprecate public arguments with a warning and compatibility path before
  removal. Test the exact request payload, not only the returned DataFrame.
- Unit tests must not require paid API calls. Use dummy responses or mocks.
- Run `pytest`, `git diff --check`, validate the notebook JSON, and build both a
  wheel and source distribution before calling a release ready.
- Keep useful deterministic tutorial outputs when they aid readers, but clear
  stale, private, failed, or excessively noisy outputs. Keep generated run
  artifacts out of Git.
