from __future__ import annotations

import os
import re
import inspect
import json
import warnings
import pandas as pd
from dataclasses import fields
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional, Union, Any, List, Mapping, Sequence

from .tasks import (
    Rate,
    RateConfig,
    Classify,
    ClassifyConfig,
    Rank,
    RankConfig,
    Deidentifier,
    DeidentifyConfig,
    Codify,
    CodifyConfig,
    Extract,
    ExtractConfig,
    Paraphrase,
    ParaphraseConfig,
    Compare,
    CompareConfig,
    Merge,
    MergeConfig,
    Deduplicate,
    DeduplicateConfig,
    Bucket,
    BucketConfig,
    Discover,
    DiscoverConfig,
    Seed,
    SeedConfig,
    Poll,
    PollConfig,
    Filter,
    FilterConfig,
    Whatever,
    WhateverConfig,
    Ideate,
    IdeateConfig,
)
from .utils.openai_utils import (
    _discard_deprecated_max_output_tokens,
    _ignore_deprecated_max_output_tokens,
    get_all_responses,
    get_response,
)
from .utils.passage_viewer import view as _view_passages
from .tasks.debias import (
    DebiasConfig,
    DebiasPipeline,
    DebiasResult,
    MeasurementMode,
    RemovalMethod,
)

__all__ = [
    "rate",
    "extract",
    "seed",
    "poll",
    "classify",
    "ideate",
    "id8",
    "deidentify",
    "rank",
    "codify",
    "paraphrase",
    "compare",
    "bucket",
    "discover",
    "deduplicate",
    "merge",
    "filter",
    "debias",
    "whatever",
    "view",
]


def _load_cached_dataframe(
    final_path: Optional[str],
    *,
    task_name: str,
) -> pd.DataFrame:
    def _find_split_parts(path: str) -> List[Path]:
        target = Path(path)
        stem = target.stem
        suffix = target.suffix
        pattern = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$")
        matches: List[tuple[int, Path]] = []
        try:
            for candidate in target.parent.glob(f"{stem}_*{suffix}"):
                match = pattern.match(candidate.name)
                if not match:
                    continue
                matches.append((int(match.group(1)), candidate))
        except Exception:
            return []
        return [path for _, path in sorted(matches, key=lambda item: item[0])]

    if not final_path:
        raise ValueError(
            f"{task_name} does not have a cached final output file; "
            "provide a DataFrame to run the task."
        )
    if os.path.exists(final_path):
        print(
            f"[API] df is None for {task_name}; loading cached results from {final_path}."
        )
        return pd.read_csv(final_path)

    split_parts = _find_split_parts(final_path)
    if split_parts:
        print(
            f"[API] df is None for {task_name}; loading cached split results from "
            f"{len(split_parts)} files with base {final_path}."
        )
        frames = [pd.read_csv(part) for part in split_parts]
        return pd.concat(frames, ignore_index=True)

    raise FileNotFoundError(
        f"{task_name} cached output not found at {final_path}. "
        "Provide a DataFrame to compute results."
    )


def _debias_default_run_name(
    measurement_attribute: Optional[str],
    removal_attribute: Optional[str],
) -> str:
    base_name = measurement_attribute or removal_attribute or "signal"
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", base_name).strip("_")
    prefix = "debias"
    return f"{prefix}_{cleaned}" if cleaned else prefix


_OPENAI_RESPONSE_OVERRIDES: frozenset[str] = frozenset(
    {
        *inspect.signature(get_all_responses).parameters.keys(),
        *inspect.signature(get_response).parameters.keys(),
    }
    - {
        "prompt",
        "prompts",
        "identifiers",
        "prompt_images",
        "prompt_audio",
        "prompt_pdfs",
        "prompt_web_search_filters",
        "save_path",
    }
)


def _split_cfg_and_response_kwargs(
    cfg_cls: Any,
    extra_kwargs: Dict[str, Any],
    *,
    task_name: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split extra kwargs into config overrides and response passthrough kwargs."""

    cfg_fields = {field.name for field in fields(cfg_cls)}
    cfg_overrides: Dict[str, Any] = {}
    response_overrides: Dict[str, Any] = {}
    unknown_keys: List[str] = []

    for key, value in extra_kwargs.items():
        if key == "max_output_tokens":
            _ignore_deprecated_max_output_tokens(value, stacklevel=4)
        elif key in cfg_fields:
            cfg_overrides[key] = value
        elif key in _OPENAI_RESPONSE_OVERRIDES:
            response_overrides[key] = value
        else:
            unknown_keys.append(key)

    if unknown_keys:
        unknown_display = ", ".join(sorted(unknown_keys))
        raise TypeError(
            f"Unknown keyword argument(s) for gabriel.{task_name}: {unknown_display}. "
            "Pass task configuration keys, or OpenAI response kwargs such as "
            "`image_detail` / `reasoning_effort`."
        )

    return cfg_overrides, response_overrides


async def rate(
    df: Optional[pd.DataFrame],
    column_name: str,
    *,
    attributes: Dict[str, str],
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5.6-luna",
    n_parallels: int = 650,
    n_runs: int = 1,
    n_attributes_per_run: Optional[int] = None,
    reset_files: bool = False,
    file_name: str = "ratings.csv",
    modality: str = "text",
    reasoning_effort: Optional[str] = None,
    search_context_size: str = "medium",
    template_path: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Asks GPT to score each text / image / audio / pdf / item on natural language attributes. Output = 0-100 rating.

    Example Use
    -----------
    Measure "populist rhetoric" in a speech; "toxicity" of tweets; "luxury" in ad images.

    Parameters
    ----------
    df:
        Source DataFrame containing the passages to rate. When ``None``, load
        cached results from ``save_dir`` instead of recomputing.
    column_name:
        Column in ``df`` that holds the passages (text, image, audio, or PDF
        references depending on ``modality``).
    attributes:
        Mapping of attribute names to natural-language descriptions that the
        model should evaluate on a 0–100 scale.
    save_dir:
        Directory where raw responses and the aggregated ratings CSV are
        written. Created if it does not exist.
    additional_instructions:
        Optional extra guidance injected into the prompt template.
    model:
        Model name passed through to the OpenAI Responses API. The signature
        default was current when this package version shipped; model IDs change,
        so verify the exact slug in the official OpenAI model catalog before
        overriding it.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    n_runs:
        Number of repeat rating passes to perform for each passage.
    n_attributes_per_run:
        Maximum number of attributes to include in a single prompt. When set
        to an integer, larger attribute sets are split across multiple prompts;
        when ``None``, all attributes are processed in one prompt.
    reset_files:
        When ``True`` existing outputs in ``save_dir`` are ignored and
        regenerated.
    file_name:
        Basename (without the automatic ``_raw_responses`` suffix) for saved
        artifacts.
    modality:
        One of ``"text"``, ``"entity"``, ``"web"``, ``"image"``, ``"audio"``, or ``"pdf"``
        to control how inputs are packaged into prompts.
    reasoning_effort:
        Controls how intensely the model reasons (none/low/medium/high). Higher is smarter but slower.
    search_context_size:
        Size hint forwarded to web-search capable models.
    template_path:
        Override the default rating prompt template with a custom Jinja2 file.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.
    **cfg_kwargs:
        Additional overrides applied to :class:`gabriel.tasks.rate.RateConfig`. Keys matching :func:`gabriel.utils.openai_utils.get_all_responses` / :func:`gabriel.utils.openai_utils.get_response` (for example ``image_detail``) are forwarded to the model call.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with one column per attribute containing the mean score
        across runs.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    if df is None:
        base_name = os.path.splitext(file_name)[0]
        final_path = os.path.join(save_dir, f"{base_name}_cleaned.csv")
        return _load_cached_dataframe(final_path, task_name="Rate")
    cfg_kwargs, response_kwargs = _split_cfg_and_response_kwargs(
        RateConfig,
        dict(cfg_kwargs),
        task_name="rate",
    )
    cfg = RateConfig(
        attributes=attributes,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        n_attributes_per_run=n_attributes_per_run,
        additional_instructions=additional_instructions,
        modality=modality,
        reasoning_effort=reasoning_effort,
        search_context_size=search_context_size,
        **cfg_kwargs,
    )
    return await Rate(cfg, template_path=template_path).run(
        df,
        column_name,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        **response_kwargs,
    )

async def extract(
    df: Optional[pd.DataFrame],
    column_name: str,
    *,
    attributes: Dict[str, str],
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5.6-luna",
    n_parallels: int = 650,
    n_runs: int = 1,
    n_attributes_per_run: Optional[int] = None,
    reset_files: bool = False,
    file_name: str = "extraction.csv",
    modality: str = "entity",
    reasoning_effort: Optional[str] = None,
    types: Optional[Dict[str, Any]] = None,
    template_path: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Structured fact extraction on each item. Output = string / numeric values.

    Example Use
    -----------
    For each product, provide the "company", "CEO", and "year of invention".

    Parameters
    ----------
    df:
        Source DataFrame containing the passages to parse. When ``None``, load
        cached results from ``save_dir`` instead of recomputing.
    column_name:
        Column in ``df`` with the content to extract from.
    attributes:
        Mapping of field names to descriptions of what should be extracted.
    save_dir:
        Directory where extraction outputs will be written. Created if absent.
    additional_instructions:
        Optional extra guidance injected into the extraction prompt.
    model:
        Model used for extraction via the OpenAI Responses API. Luna is the
        scale-first default for most extraction work. Pilot Terra or Sol when
        the task is unusually subtle, knowledge-heavy, or writing-sensitive.
        Model IDs and organization provisioning change, so verify the exact
        available slug before overriding it.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    n_runs:
        Number of extraction passes to perform; results are averaged when
        applicable.
    n_attributes_per_run:
        Maximum number of attributes to include in each prompt. When set to an
        integer, larger attribute sets are split across multiple prompts; when
        ``None``, all attributes are processed in one prompt.
    reset_files:
        When ``True`` forces regeneration of outputs in ``save_dir``.
    file_name:
        CSV name used when saving extraction results.
    modality:
        Indicates whether the content is ``"entity"`` text or another modality
        supported by the templates.
    reasoning_effort:
        Controls how intensely the model reasons (none/low/medium/high). Higher is smarter but slower.
    types:
        Optional mapping of attribute names to explicit Python types for
        stronger downstream typing.
    template_path:
        Custom Jinja2 template path to override the default extraction prompt.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.
    **cfg_kwargs:
        Additional overrides forwarded to :class:`gabriel.tasks.extract.ExtractConfig`. Keys matching :func:`gabriel.utils.openai_utils.get_all_responses` / :func:`gabriel.utils.openai_utils.get_response` (for example ``image_detail``) are forwarded to the model call.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame augmented with one column per requested
        attribute.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    if df is None:
        base_name = os.path.splitext(file_name)[0]
        final_path = os.path.join(save_dir, f"{base_name}_cleaned.csv")
        return _load_cached_dataframe(final_path, task_name="Extract")
    cfg_kwargs, response_kwargs = _split_cfg_and_response_kwargs(
        ExtractConfig,
        dict(cfg_kwargs),
        task_name="extract",
    )
    cfg = ExtractConfig(
        attributes=attributes,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        n_attributes_per_run=n_attributes_per_run,
        additional_instructions=additional_instructions,
        modality=modality,
        reasoning_effort=reasoning_effort,
        **cfg_kwargs,
    )
    return await Extract(cfg, template_path=template_path).run(
        df,
        column_name,
        reset_files=reset_files,
        types=types,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        **response_kwargs,
    )


async def seed(
    instructions: str,
    *,
    save_dir: str,
    file_name: str = "seed_entities.csv",
    model: str = "gpt-5.6-sol",
    n_parallels: int = 650,
    num_entities: int = 1000,
    entities_per_generation: int = 50,
    entity_batch_frac: float = 0.25,
    existing_entities_cap: int = 100,
    deduplicate: bool = False,
    deduplicate_sample_seed: int = 42,
    reasoning_effort: Optional[str] = None,
    template_path: Optional[str] = None,
    existing_entities: Optional[List[str]] = None,
    reset_files: bool = False,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    embedding_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_embeddings_fn: Optional[Callable[..., Awaitable[Dict[str, List[float]]]]] = None,
    **response_kwargs: Any,
) -> pd.DataFrame:
    """Enforces a representative distribution / diversity of seeds.

    Example Use
    -----------
    Initialize unique personas that match US population distribution.

    Parameters
    ----------
    instructions:
        High-level description of the domain and what constitutes a good seed
        entity.
    save_dir:
        Directory where seed entities and raw responses are stored.
    file_name:
        Name of the CSV to write seed entities to.
    model:
        Model used for generation. The signature default was current when this
        package version shipped; model IDs change, so verify the exact slug in
        the official OpenAI model catalog before overriding it.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    num_entities:
        Target number of entities to generate in total.
    entities_per_generation:
        Number of entities requested from each API call.
    entity_batch_frac:
        Fraction of generated entities to keep per batch before deduplication.
    existing_entities_cap:
        Maximum number of prior entities to consider when avoiding duplicates.
    deduplicate:
        When ``True`` over-generate and apply a shallow deduplication pass
        before returning results.
    deduplicate_sample_seed:
        Random seed used when sampling a deterministic subset after
        deduplication.
    reasoning_effort:
        Controls how intensely the model reasons (none/low/medium/high). Higher is smarter but slower.
    template_path:
        Optional Jinja2 template override for the seeding prompt.
    existing_entities:
        List of pre-existing entities to avoid regenerating.
    reset_files:
        When ``True`` ignore any saved state in ``save_dir`` and regenerate.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.
    embedding_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_embeddings`
        within nested deduplication. It replaces only the per-text embedding call.
    get_all_embeddings_fn:
        Optional callable that fully replaces
        :func:`gabriel.utils.openai_utils.get_all_embeddings` within nested
        deduplication. It must accept ``texts`` and ``identifiers``.
    **response_kwargs:
        Additional keyword arguments forwarded to
        :func:`gabriel.utils.openai_utils.get_all_responses`. The deprecated
        ``max_output_tokens`` argument is accepted, warned about, and ignored.

    Returns
    -------
    pandas.DataFrame
        DataFrame of seed entities with provenance metadata.
    """

    response_kwargs = dict(response_kwargs)
    _discard_deprecated_max_output_tokens(response_kwargs, stacklevel=3)
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = SeedConfig(
        instructions=instructions,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        num_entities=num_entities,
        entities_per_generation=entities_per_generation,
        entity_batch_frac=entity_batch_frac,
        existing_entities_cap=existing_entities_cap,
        deduplicate=deduplicate,
        deduplicate_sample_seed=deduplicate_sample_seed,
        reasoning_effort=reasoning_effort,
    )
    task = Seed(cfg, template_path=template_path)
    return await task.run(
        existing_entities=existing_entities,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        embedding_fn=embedding_fn,
        get_all_embeddings_fn=get_all_embeddings_fn,
        **response_kwargs,
    )


async def poll(
    df: Optional[pd.DataFrame] = None,
    column_name: Optional[str] = None,
    *,
    population_description: Optional[str] = None,
    questions: Optional[Union[str, Sequence[str]]] = None,
    save_dir: str,
    file_name: str = "poll_results.csv",
    seed_file_name: str = "poll_seeds.csv",
    persona_file_name: str = "poll_personas.csv",
    model: str = "gpt-5.6-sol",
    seed_model: Optional[str] = None,
    persona_model: Optional[str] = None,
    poll_model: Optional[str] = None,
    n_parallels: int = 650,
    num_personas: int = 1000,
    entities_per_generation: int = 50,
    entity_batch_frac: float = 0.25,
    existing_entities_cap: int = 100,
    deduplicate: bool = False,
    deduplicate_sample_seed: int = 42,
    n_questions_per_run: int = 8,
    seed_additional_instructions: Optional[str] = None,
    additional_instructions: Optional[str] = None,
    web_search: bool = False,
    reasoning_effort: Optional[str] = None,
    seed_template_path: Optional[str] = None,
    persona_template_path: Optional[str] = None,
    answer_template_path: Optional[str] = None,
    reset_files: bool = False,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    embedding_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_embeddings_fn: Optional[Callable[..., Awaitable[Dict[str, List[float]]]]] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Seed a synthetic population, expand it into personas, and survey them.

    Example Use
    -----------
    Survey a representative synthetic sample of the U.S. population on one or
    more poll questions.

    Parameters
    ----------
    df:
        Optional DataFrame containing precomputed respondent seeds. When
        provided, `population_description` is ignored and the seeding stage is
        skipped. If the DataFrame already contains a ``persona`` column, the
        task skips directly to the polling stage and reuses those personas.
    column_name:
        Column in ``df`` containing the seed descriptions. If omitted, the task
        will look for ``"seed"`` and then ``"entity"``. This is optional when
        reusing an existing ``persona`` column.
    population_description:
        Natural-language description of the population to seed when ``df`` is
        not supplied.
    questions:
        A single survey question or a sequence of questions. Questions are
        answered in JSON and become columns in the returned DataFrame.
    save_dir:
        Directory where intermediate and final CSV artifacts are written.
    file_name:
        Final CSV written by the poll task.
    seed_file_name:
        CSV used for the seeded respondent population.
    persona_file_name:
        CSV used for the generated personas before question answering.
    model:
        Default model used for all three stages unless a stage-specific model is
        provided. The signature default was current when this package version
        shipped; model IDs change, so verify the exact slug in the official
        OpenAI model catalog before overriding it.
    seed_model / persona_model / poll_model:
        Optional model overrides for each stage. Verify every override against
        the current official model catalog rather than copying a stale example.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    num_personas:
        Number of synthetic respondents to create when seeding a population.
    entities_per_generation / entity_batch_frac / existing_entities_cap /
    deduplicate / deduplicate_sample_seed:
        Seeding controls forwarded to :class:`gabriel.tasks.seed.Seed`.
    n_questions_per_run:
        Maximum number of questions bundled into one polling prompt.
    seed_additional_instructions:
        Extra guidance appended to the seed-generation instructions.
    additional_instructions:
        Extra guidance appended to the poll-answering prompt.
    web_search:
        Enable web search augmentation for the polling stage only.
    reasoning_effort:
        Controls how intensely the model reasons (none/low/medium/high).
    seed_template_path / persona_template_path / answer_template_path:
        Optional Jinja2 template overrides for each prompt stage.
    reset_files:
        When ``True`` ignore saved checkpoints and regenerate all stages.
    response_fn / get_all_responses_fn:
        Optional overrides for the Responses API execution path.
    embedding_fn / get_all_embeddings_fn:
        Optional embedding overrides used by nested seed deduplication.
    **cfg_kwargs:
        Additional overrides applied to :class:`gabriel.tasks.poll.PollConfig`.
        Keys matching :func:`gabriel.utils.openai_utils.get_all_responses` /
        :func:`gabriel.utils.openai_utils.get_response` are forwarded to model
        calls. The legacy ``max_output_tokens`` key is accepted but ignored
        with a warning.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing respondent seeds, generated personas, and one
        column per question when questions are supplied.
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    if df is None and population_description is None:
        final_path = os.path.join(save_dir, file_name)
        return _load_cached_dataframe(final_path, task_name="Poll")

    cfg_overrides, response_kwargs = _split_cfg_and_response_kwargs(
        PollConfig,
        dict(cfg_kwargs),
        task_name="poll",
    )
    normalized_questions: List[str]
    if questions is None:
        normalized_questions = []
    elif isinstance(questions, str):
        normalized_questions = [questions]
    else:
        normalized_questions = [str(question) for question in questions]

    cfg = PollConfig(
        population_description=population_description,
        questions=normalized_questions,
        save_dir=save_dir,
        file_name=file_name,
        seed_file_name=seed_file_name,
        persona_file_name=persona_file_name,
        seed_model=seed_model or model,
        persona_model=persona_model or model,
        poll_model=poll_model or model,
        n_parallels=n_parallels,
        num_personas=num_personas,
        entities_per_generation=entities_per_generation,
        entity_batch_frac=entity_batch_frac,
        existing_entities_cap=existing_entities_cap,
        deduplicate=deduplicate,
        deduplicate_sample_seed=deduplicate_sample_seed,
        n_questions_per_run=n_questions_per_run,
        seed_additional_instructions=seed_additional_instructions,
        additional_instructions=additional_instructions,
        web_search=web_search,
        reasoning_effort=reasoning_effort,
        **cfg_overrides,
    )
    task = Poll(
        cfg,
        seed_template_path=seed_template_path,
        persona_template_path=persona_template_path,
        answer_template_path=answer_template_path,
    )
    return await task.run(
        df=df,
        column_name=column_name,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        embedding_fn=embedding_fn,
        get_all_embeddings_fn=get_all_embeddings_fn,
        **response_kwargs,
    )


async def classify(
    df: Optional[pd.DataFrame],
    column_name: Optional[str] = None,
    *,
    labels: Dict[str, str],
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5.6-luna",
    differentiate: bool = False,
    circle_column_name: Optional[str] = None,
    square_column_name: Optional[str] = None,
    n_parallels: int = 650,
    n_runs: int = 1,
    n_attributes_per_run: Optional[int] = None,
    min_frequency: float = 0.6,
    reset_files: bool = False,
    file_name: str = "classify_responses.csv",
    modality: str = "text",
    reasoning_effort: Optional[str] = None,
    search_context_size: str = "medium",
    template_path: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Classifies texts / images / audio / pdfs / items on whether provided labels apply. Output = one or more classes per item.

    Example Use
    -----------
    Tag news articles, product photos, or interview clips into topical categories.

    Parameters
    ----------
    df:
        DataFrame containing content to classify. When ``None``, load cached
        results from ``save_dir`` instead of recomputing.
    column_name:
        Column with the main passage text. Can be ``None`` when using paired
        circle/square inputs.
    labels:
        Mapping of label names to definitions the model should follow.
    save_dir:
        Directory where classification artifacts are written.
    additional_instructions:
        Free-form instructions appended to the classification prompt.
    model:
        Model name used for classification. The signature default was current
        when this package version shipped; model IDs change, so verify the exact
        slug in the official OpenAI model catalog before overriding it.
    differentiate:
        When ``True`` use differentiation mode to highlight contrasts.
    circle_column_name, square_column_name:
        Optional paired columns for contrastive classification.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    n_runs:
        Number of repeated classification passes.
    n_attributes_per_run:
        Maximum number of labels to evaluate per prompt. When set to an
        integer, larger label sets are split across multiple prompts; when
        ``None``, all labels are evaluated in one prompt.
    min_frequency:
        Minimum label frequency required to keep a label during aggregation.
    reset_files:
        When ``True`` overwrite any existing outputs in ``save_dir``.
    file_name:
        Basename for saved classification CSVs.
    modality:
        Indicates the content modality for prompt rendering.
    reasoning_effort:
        Controls how intensely the model reasons (none/low/medium/high). Higher is smarter but slower.
    search_context_size:
        Context size hint forwarded to the Responses API.
    template_path:
        Override the default classification prompt template.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.
    **cfg_kwargs:
        Extra configuration passed to :class:`gabriel.tasks.classify.ClassifyConfig`. Keys matching :func:`gabriel.utils.openai_utils.get_all_responses` / :func:`gabriel.utils.openai_utils.get_response` (for example ``image_detail``) are forwarded to the model call.

    Returns
    -------
    pandas.DataFrame
        DataFrame including one column per label plus ``predicted_classes``; aggregates repeated runs using ``min_frequency``.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    if df is None:
        base_name = os.path.splitext(file_name)[0]
        final_path = os.path.join(save_dir, f"{base_name}_cleaned.csv")
        return _load_cached_dataframe(final_path, task_name="Classify")
    cfg_kwargs, response_kwargs = _split_cfg_and_response_kwargs(
        ClassifyConfig,
        dict(cfg_kwargs),
        task_name="classify",
    )
    cfg = ClassifyConfig(
        labels=labels,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        differentiate=differentiate,
        n_parallels=n_parallels,
        n_runs=n_runs,
        n_attributes_per_run=n_attributes_per_run,
        min_frequency=min_frequency,
        additional_instructions=additional_instructions or "",
        modality=modality,
        reasoning_effort=reasoning_effort,
        search_context_size=search_context_size,
        **cfg_kwargs,
    )
    return await Classify(cfg, template_path=template_path).run(
        df,
        column_name,
        circle_column_name=circle_column_name,
        square_column_name=square_column_name,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        **response_kwargs,
    )


async def ideate(
    topic: str,
    *,
    save_dir: str,
    file_name: str = "ideation.csv",
    model: str = "gpt-5.6-terra",
    scientific_theory: bool = True,
    ranking_model: Optional[str] = None,
    n_ideas: int = 1000,
    n_parallels: int = 650,
    evaluation_mode: str = "recursive_rank",
    attributes: Optional[Dict[str, str]] = None,
    rank_attribute: Optional[str] = None,
    recursive_fraction: float = 1.0 / 3.0,
    recursive_min_remaining: int = 30,
    recursive_final_round_multiplier: int = 3,
    recursive_cut_side: str = "top",
    recursive_rate_first_round: bool = True,
    additional_instructions: Optional[str] = None,
    web_search: bool = False,
    reasoning_effort: Optional[str] = None,
    reset_files: bool = False,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    rank_config_updates: Optional[Dict[str, Any]] = None,
    rank_run_kwargs: Optional[Dict[str, Any]] = None,
    rate_config_updates: Optional[Dict[str, Any]] = None,
    rate_run_kwargs: Optional[Dict[str, Any]] = None,
    use_seed_entities: Optional[bool] = None,
    seed_deduplicate: bool = False,
    seed_config_updates: Optional[Dict[str, Any]] = None,
    seed_run_kwargs: Optional[Dict[str, Any]] = None,
    deduplicate_run_kwargs: Optional[Dict[str, Any]] = None,
    template_path: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    embedding_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_embeddings_fn: Optional[Callable[..., Awaitable[Dict[str, List[float]]]]] = None,
) -> pd.DataFrame:
    """Generate and filter candidate scientific theories for further evaluation.

    Example Use
    -----------
    Procure novel theories on inflation for potential research.

    Parameters
    ----------
    topic:
        Subject area or question to ideate on.
    save_dir:
        Directory where generated ideas and intermediate rankings are saved.
    file_name:
        CSV name for the consolidated ideation output.
    model, ranking_model:
        Models used for idea generation and ranking (if different). Their
        defaults were current when this package version shipped; model IDs
        change, so verify exact slugs in the official OpenAI model catalog.
    scientific_theory:
        When ``True`` generate novel scientific theories; when ``False`` generate
        general-purpose ideas with the same output structure.
    n_ideas:
        Target number of ideas to generate before pruning.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    evaluation_mode:
        Strategy used to evaluate ideas (for example ``"recursive_rank"``).
    attributes:
        Optional attributes to rate ideas on during evaluation.
    rank_attribute:
        Name of the attribute used for final ranking when multiple attributes
        are present.
    recursive_*:
        Parameters controlling iterative ranking passes (fraction kept,
        minimum remaining, cut side, etc.).
    additional_instructions:
        Extra guidance injected into prompts for both generation and ranking.
    web_search:
        Enable web search augmentation for generation.
    reasoning_effort:
        Controls how intensely the model reasons (none/low/medium/high). Higher is smarter but slower.
    reset_files:
        Force regeneration of outputs in ``save_dir``.
    *_config_updates, *_run_kwargs:
        Fine-grained overrides for nested Rate/Rank/Seed/Deduplicate tasks.
        A legacy ``max_output_tokens`` entry is accepted, warned about, and
        ignored.
    seed_deduplicate:
        When ``True`` enable deduplication in the nested seed generation.
    template_path:
        Optional template override for the ideation prompts.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.
    embedding_fn:
        Optional callable forwarded to nested deduplication tasks that replaces
        the per-text embedding call while preserving the default
        :func:`gabriel.utils.openai_utils.get_all_embeddings` orchestration.
    get_all_embeddings_fn:
        Optional callable forwarded to nested deduplication tasks that fully
        replaces :func:`gabriel.utils.openai_utils.get_all_embeddings`.

    Returns
    -------
    pandas.DataFrame
        Ranked list of ideas with evaluation metadata.
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)

    cfg_kwargs: Dict[str, Any] = dict(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        scientific_theory=scientific_theory,
        ranking_model=ranking_model,
        n_parallels=n_parallels,
        n_ideas=n_ideas,
        evaluation_mode=evaluation_mode,
        rank_attribute=rank_attribute,
        recursive_fraction=recursive_fraction,
        recursive_min_remaining=recursive_min_remaining,
        recursive_final_round_multiplier=recursive_final_round_multiplier,
        recursive_cut_side=recursive_cut_side,
        recursive_rate_first_round=recursive_rate_first_round,
        additional_instructions=additional_instructions,
        web_search=web_search,
        reasoning_effort=reasoning_effort,
        seed_deduplicate=seed_deduplicate,
    )
    if attributes is not None:
        cfg_kwargs["attributes"] = attributes
    cfg = IdeateConfig(**cfg_kwargs)

    def _with_response_overrides(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        updated = dict(payload or {})
        if response_fn is not None:
            updated.setdefault("response_fn", response_fn)
        if get_all_responses_fn is not None:
            updated.setdefault("get_all_responses_fn", get_all_responses_fn)
        return updated

    def _with_embedding_overrides(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        updated = _with_response_overrides(payload)
        if embedding_fn is not None:
            updated.setdefault("embedding_fn", embedding_fn)
        if get_all_embeddings_fn is not None:
            updated.setdefault("get_all_embeddings_fn", get_all_embeddings_fn)
        return updated

    generation_kwargs = _with_response_overrides(generation_kwargs)
    rank_run_kwargs = _with_response_overrides(rank_run_kwargs)
    rate_run_kwargs = _with_response_overrides(rate_run_kwargs)
    seed_run_kwargs = _with_embedding_overrides(seed_run_kwargs)
    deduplicate_run_kwargs = _with_embedding_overrides(deduplicate_run_kwargs)

    ideator = Ideate(cfg, template_path=template_path)
    return await ideator.run(
        topic,
        additional_instructions=additional_instructions,
        evaluation_mode=evaluation_mode,
        attributes=attributes,
        rank_attribute=rank_attribute,
        reset_files=reset_files,
        generation_kwargs=generation_kwargs,
        rank_config_updates=rank_config_updates,
        rank_run_kwargs=rank_run_kwargs,
        rate_config_updates=rate_config_updates,
        rate_run_kwargs=rate_run_kwargs,
        use_seed_entities=use_seed_entities,
        seed_config_updates=seed_config_updates,
        seed_run_kwargs=seed_run_kwargs,
        deduplicate_run_kwargs=deduplicate_run_kwargs,
    )


async def id8(*args, **kwargs) -> pd.DataFrame:
    """Alias for :func:`ideate`."""

    return await ideate(*args, **kwargs)


async def deidentify(
    df: Optional[pd.DataFrame],
    column_name: str,
    *,
    save_dir: str,
    grouping_column: Optional[str] = None,
    mapping_column: Optional[str] = None,
    model: str = "gpt-5.6-terra",
    n_parallels: int = 650,
    file_name: str = "deidentified.csv",
    max_words_per_call: int = 7500,
    additional_instructions: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    n_passes: int = 1,
    use_existing_mappings_only: bool = False,
    template_path: Optional[str] = None,
    reset_files: bool = False,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Replaces PII with realistic, consistent fake PII. Outputs anonymized text + mapping.

    Example Use
    -----------
    Replace names, employers, addresses before sharing interview corpora.

    Parameters
    ----------
    df:
        DataFrame containing passages to deidentify. When ``None``, load cached
        results from ``save_dir`` instead of recomputing.
    column_name:
        Column in ``df`` holding the text to scrub.
    save_dir:
        Directory where anonymised outputs and mappings are written.
    grouping_column:
        Optional column grouping records that should share replacements.
    mapping_column:
        Optional column providing deterministic replacement tokens.
    model:
        Model name used to perform deidentification. The signature default was
        current when this package version shipped; model IDs change, so verify
        the exact slug in the official OpenAI model catalog before overriding it.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    file_name:
        CSV filename used when persisting deidentified text.
    max_words_per_call:
        Chunk size control for long passages.
    additional_instructions:
        Extra guidance appended to the prompt.
    reasoning_effort:
        Controls how intensely the model reasons (none/low/medium/high). Higher is smarter but slower.
    n_passes:
        Number of deidentification passes to run over each passage.
    use_existing_mappings_only:
        If ``True`` only apply existing mappings and avoid new model calls.
    template_path:
        Custom prompt template path.
    reset_files:
        When ``True`` ignore cached outputs in ``save_dir``.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.
    **cfg_kwargs:
        Additional overrides for :class:`gabriel.tasks.deidentify.DeidentifyConfig`. Keys matching :func:`gabriel.utils.openai_utils.get_all_responses` / :func:`gabriel.utils.openai_utils.get_response` (for example ``image_detail``) are forwarded to the model call.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing deidentified text and replacement mappings.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    if df is None:
        base_name = os.path.splitext(file_name)[0]
        final_path = os.path.join(save_dir, f"{base_name}_cleaned.csv")
        return _load_cached_dataframe(final_path, task_name="Deidentify")
    cfg_kwargs, response_kwargs = _split_cfg_and_response_kwargs(
        DeidentifyConfig,
        dict(cfg_kwargs),
        task_name="deidentify",
    )
    cfg = DeidentifyConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        max_words_per_call=max_words_per_call,
        additional_instructions=additional_instructions,
        reasoning_effort=reasoning_effort,
        n_passes=n_passes,
        use_existing_mappings_only=use_existing_mappings_only,
        **cfg_kwargs,
    )
    return await Deidentifier(cfg, template_path=template_path).run(
        df,
        column_name,
        grouping_column=grouping_column,
        mapping_column=mapping_column,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        **response_kwargs,
    )

async def rank(
    df: Optional[pd.DataFrame],
    column_name: str,
    *,
    attributes: Union[Dict[str, str], List[str]],
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5.6-luna",
    n_rounds: int = 5,
    matches_per_round: int = 3,
    power_matching: bool = True,
    return_raw_scores: bool = False,
    learning_rate: float = 0.1,
    insufficient_signal_policy: str = "tie",
    n_parallels: int = 650,
    n_attributes_per_run: Optional[int] = None,
    file_name: str = "rankings",
    reset_files: bool = False,
    modality: str = "text",
    reasoning_effort: Optional[str] = None,
    template_path: Optional[str] = None,
    recursive: bool = False,
    recursive_fraction: float = 1.0 / 3.0,
    recursive_min_remaining: int = 30,
    recursive_final_round_multiplier: int = 3,
    recursive_cut_attr: Optional[str] = None,
    recursive_cut_side: str = "top",
    recursive_rate_first_round: bool = True,
    recursive_rewrite_func: Optional[Callable[[str, str, int], str]] = None,
    recursive_rewrite_text_col: str = "text",
    recursive_keep_stage_columns: bool = True,
    recursive_add_stage_suffix: bool = True,
    initial_rating_pass: bool = True,
    rate_kwargs: Optional[Dict[str, Any]] = None,
    primer_scores: Optional[Dict[str, Dict[str, float]]] = None,
    primer_scale: float = 1.0,
    primer_center: bool = True,
    judge_version: Optional[str] = None,
    id_column: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Fit model-derived Bradley–Terry ratings from pairwise comparisons.

    Example Use
    -----------
    Rank technologies by "bulkiness" or artworks by "fine brushwork".

    Parameters
    ----------
    df:
        DataFrame containing passages to rank. When ``None``, load cached
        results from ``save_dir`` instead of recomputing.
    column_name:
        Column holding the content to rank.
    attributes:
        Either a mapping of attribute names to descriptions or a list of
        attribute names (descriptions inferred from templates).
    save_dir:
        Directory where ranking artifacts are saved.
    additional_instructions:
        Free-form prompt additions applied to each comparison.
    model:
        Model name used for ranking calls. The signature default was current
        when this package version shipped; model IDs change, so verify the exact
        slug in the official OpenAI model catalog before overriding it.
    n_rounds, matches_per_round, power_matching, learning_rate:
        Parameters controlling the Bradley–Terry tournament mechanics.
        ``matches_per_round`` is a per-item target capped at the number of
        possible opponents. Realized degrees may exceed it because an item can
        also be selected as another item's opponent; this is especially common
        in random mode. ``power_matching=True`` uses an uncertainty-guided
        heuristic, not exact expected information gain; ``False`` uses random
        pairing. ``learning_rate`` is the total symmetric pseudo-comparison
        weight applied to each observed pair. A zero value requires Ford's
        directed strong-connectivity condition within every nontrivial observed
        component for a finite maximum-likelihood fit.
        ``<attribute>_se`` is an asymptotic, model-based sandwich standard error
        for the component-centered raw log-skill, computed while treating the
        realized comparison graph as fixed. Observed comparison weights
        determine the meat under binary-BT working variance; positive fixed
        pseudo-comparison weight adds curvature to the bread. It excludes
        shrinkage bias, shared-judge dependence, misspecification,
        adaptive-selection uncertainty, and uncertainty for the reported
        z-score.
    insufficient_signal_policy:
        ``"tie"`` makes the modeling assumption that lack of direct signal is
        equality and gives each item a half-win. ``"abstain"`` excludes that
        judgment from fitting; informative abstention can still bias the
        retained sample. True draws use half-wins in a binary Bradley–Terry
        working objective, not a three-outcome tie likelihood.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    n_attributes_per_run:
        Maximum number of attributes to compare per prompt. When set to an
        integer, larger attribute sets are split across multiple prompts; when
        ``None``, all attributes are compared in one prompt.
    file_name:
        Base filename for saved rankings (without extension).
    reset_files:
        Force regeneration of any existing outputs in ``save_dir``.
    modality:
        Content modality forwarded to the prompt.
    reasoning_effort:
        Controls how intensely the model reasons (none/low/medium/high). Higher is smarter but slower.
    template_path:
        Path to a custom ranking prompt template.
    recursive_*:
        Settings for recursive pruning (fraction kept, minimum remaining, etc.).
    initial_rating_pass:
        Whether to run a preliminary rating stage before comparisons. Enabled by
        default to give the tournament model-derived starting scores; set to
        ``False`` to skip the rating seed.
    rate_kwargs:
        Additional configuration forwarded to the preliminary rating stage.
        A legacy ``max_output_tokens`` entry is accepted, warned about, and
        ignored. ``attributes``, ``save_dir``, and ``file_name`` are owned by
        Rank and cannot be overridden here.
    primer_scores, primer_scale, primer_center:
        Optional seed ratings to prime the Bradley–Terry loop. Scores are
        centred per attribute when ``primer_center`` is ``True`` and scaled
        by ``primer_scale``.
    judge_version:
        Stable label required when ``response_fn`` or
        ``get_all_responses_fn`` supplies a custom judge. Change it whenever
        hidden judgment logic or configuration changes so resumed tournaments
        reject mixed measurement regimes.
    id_column:
        Optional existing identifier column; otherwise hashes of ``column_name``
        are generated.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.
    **cfg_kwargs:
        Extra parameters passed to :class:`gabriel.tasks.rank.RankConfig`. Keys matching :func:`gabriel.utils.openai_utils.get_all_responses` / :func:`gabriel.utils.openai_utils.get_response` (for example ``image_detail``) are forwarded to the model call.

    Returns
    -------
    pandas.DataFrame
        Ranked outputs. For non-recursive runs, the CSV written to ``save_dir``
        contains raw scores, model-based sandwich standard errors, and
        comparison-graph component labels. Z-scores and raw scores are only
        comparable within a component; components with no score dispersion use
        a neutral z-score fallback of zero, while isolated items have ``NaN``
        scores and standard errors. Raw scores and standard errors are hidden unless
        ``return_raw_scores`` is ``True``. Recursive runs instead return
        stage-relative z-scores, ``exit_stage``, and ``overall_rank``. At
        Rank-produced recursive stages, pruning stops with an error if the cut
        attribute's comparison graph is disconnected; a preliminary Rate stage
        has no comparison graph. Non-recursive runs also save a diagnostics sidecar with
        outcome counts, effective comparison weight, graph topology, and the
        count of finite standard errors.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)

    def _persisted_rank_attribute_keys() -> Optional[List[str]]:
        base_name = os.path.splitext(file_name)[0]
        for path in (
            Path(save_dir) / f"{base_name}_attrs.json",
            Path(save_dir) / "attributes.json",
        ):
            try:
                with path.open(encoding="utf-8") as attribute_file:
                    payload = json.load(attribute_file)
            except Exception:
                continue
            if isinstance(payload, dict) and all(
                isinstance(key, str) for key in payload
            ):
                return list(payload)
            if isinstance(payload, list) and all(
                isinstance(key, str) for key in payload
            ):
                return payload
        return None

    def _project_public_rank_columns(result: pd.DataFrame) -> pd.DataFrame:
        if return_raw_scores or recursive:
            return result
        # Infer persisted Rank attributes from the output schema rather than
        # from this invocation's arguments.  Cached results may have been
        # produced with a different in-memory attribute object, but the public
        # projection must never leak hidden raw estimates or standard errors.
        schema_attr_keys = [
            column[: -len("_component")]
            for column in result.columns
            if column.endswith("_component")
            and column[: -len("_component")] in result.columns
            and f"{column[: -len('_component')]}_raw" in result.columns
            and f"{column[: -len('_component')]}_se" in result.columns
        ]
        caller_attr_keys = (
            list(attributes.keys())
            if isinstance(attributes, dict)
            else list(attributes)
        )
        persisted_attr_keys = _persisted_rank_attribute_keys()
        # Persisted attributes are authoritative and avoid hiding unrelated
        # input columns that happen to share Rank's suffix convention. Caller
        # and schema inference retain safe behavior for legacy cache layouts.
        persisted_matches = [
            attr
            for attr in (persisted_attr_keys or [])
            if f"{attr}_raw" in result.columns and f"{attr}_se" in result.columns
        ]
        attr_keys = (
            persisted_matches
            if persisted_matches
            else list(dict.fromkeys([*schema_attr_keys, *caller_attr_keys]))
        )
        drop_cols = [
            column
            for attr in attr_keys
            for column in (f"{attr}_raw", f"{attr}_se")
            if column in result.columns
        ]
        return result.drop(columns=drop_cols) if drop_cols else result

    if df is None:
        if recursive:
            base_folder = os.path.join(save_dir, f"{file_name}_recursive")
            final_path = os.path.join(base_folder, "recursive_final.csv")
        else:
            base_name = os.path.splitext(file_name)[0]
            final_path = os.path.join(save_dir, f"{base_name}_final.csv")
        return _project_public_rank_columns(
            _load_cached_dataframe(final_path, task_name="Rank")
        )
    cfg_kwargs, response_kwargs = _split_cfg_and_response_kwargs(
        RankConfig,
        dict(cfg_kwargs),
        task_name="rank",
    )
    rate_kwargs = dict(rate_kwargs or {})
    _discard_deprecated_max_output_tokens(rate_kwargs, stacklevel=3)
    cfg = RankConfig(
        attributes=attributes,
        n_rounds=n_rounds,
        matches_per_round=matches_per_round,
        power_matching=power_matching,
        learning_rate=learning_rate,
        insufficient_signal_policy=insufficient_signal_policy,
        model=model,
        n_parallels=n_parallels,
        n_attributes_per_run=n_attributes_per_run,
        save_dir=save_dir,
        file_name=file_name,
        additional_instructions=additional_instructions or "",
        modality=modality,
        reasoning_effort=reasoning_effort,
        recursive=recursive,
        recursive_fraction=recursive_fraction,
        recursive_min_remaining=recursive_min_remaining,
        recursive_final_round_multiplier=recursive_final_round_multiplier,
        recursive_cut_attr=recursive_cut_attr,
        recursive_cut_side=recursive_cut_side,
        recursive_rate_first_round=recursive_rate_first_round,
        recursive_rewrite_func=recursive_rewrite_func,
        recursive_rewrite_text_col=recursive_rewrite_text_col,
        recursive_keep_stage_columns=recursive_keep_stage_columns,
        recursive_add_stage_suffix=recursive_add_stage_suffix,
        initial_rating_pass=initial_rating_pass,
        rate_kwargs=rate_kwargs,
        primer_scores=primer_scores,
        primer_scale=primer_scale,
        primer_center=primer_center,
        judge_version=judge_version,
        **cfg_kwargs,
    )
    result_df = await Rank(cfg, template_path=template_path).run(
        df,
        column_name,
        id_column=id_column,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        **response_kwargs,
    )

    # Keep raw estimates and uncertainty persisted while presenting the same
    # public projection for freshly computed and cached results.
    return _project_public_rank_columns(result_df)


async def codify(
    df: Optional[pd.DataFrame],
    column_name: str,
    *,
    save_dir: str,
    categories: Optional[Dict[str, str]] = None,
    additional_instructions: str = "",
    model: str = "gpt-5.6-terra",
    n_parallels: int = 650,
    max_words_per_call: int = 1000,
    max_categories_per_call: int = 8,
    file_name: str = "coding_results.csv",
    reset_files: bool = False,
    debug_print: bool = False,
    reasoning_effort: Optional[str] = None,
    modality: str = "text",
    n_rounds: int = 2,
    completion_classifier_instructions: Optional[str] = None,
    template_path: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Passage coding: highlights snippets in text that match qualitative codes.

    Example Use
    -----------
    Flag sentences about "economic insecurity" in speeches; "stressors" mentioned in interview.

    Parameters
    ----------
    df:
        DataFrame containing the passages to code. When ``None``, load cached
        results from ``save_dir`` instead of recomputing.
    column_name:
        Column with the text to be coded.
    save_dir:
        Directory where coding outputs are written.
    categories:
        Optional mapping of category names to descriptions. If omitted the model
        infers categories.
    additional_instructions:
        Extra guidance appended to the coding prompt.
    model:
        Model used for coding requests. The signature default was current when
        this package version shipped; model IDs change, so verify the exact slug
        in the official OpenAI model catalog before overriding it.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    max_words_per_call:
        Chunk size control for each request.
    max_categories_per_call:
        Limit on the number of categories evaluated per call.
    file_name:
        Filename for saved coding responses.
    reset_files:
        When ``True`` regenerate outputs even if files exist.
    debug_print:
        Enable verbose logging of prompts and responses.
    reasoning_effort:
        Controls how intensely the model reasons (none/low/medium/high). Higher is smarter but slower.
    modality:
        Content modality hint (text, entity, etc.).
    n_rounds:
        Number of completion passes to refine codes.
    completion_classifier_instructions:
        Optional classifier guidance for completion steps.
    template_path:
        Custom Jinja2 template for coding prompts.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.
    **cfg_kwargs:
        Additional overrides passed to :class:`gabriel.tasks.codify.CodifyConfig`. Keys matching :func:`gabriel.utils.openai_utils.get_all_responses` / :func:`gabriel.utils.openai_utils.get_response` (for example ``image_detail``) are forwarded to the model call.

    Returns
    -------
    pandas.DataFrame
        DataFrame with coded categories and any iterative refinement metadata.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    if df is None:
        final_path = os.path.join(save_dir, "coded_passages.csv")
        return _load_cached_dataframe(final_path, task_name="Codify")
    cfg_kwargs, response_kwargs = _split_cfg_and_response_kwargs(
        CodifyConfig,
        dict(cfg_kwargs),
        task_name="codify",
    )

    cfg = CodifyConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        max_words_per_call=max_words_per_call,
        max_categories_per_call=max_categories_per_call,
        debug_print=debug_print,
        reasoning_effort=reasoning_effort,
        modality=modality,
        n_rounds=n_rounds,
        completion_classifier_instructions=completion_classifier_instructions,
        **cfg_kwargs,
    )
    return await Codify(cfg, template_path=template_path).run(
        df,
        column_name,
        categories=categories,
        additional_instructions=additional_instructions,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        **response_kwargs,
    )


async def paraphrase(
    df: Optional[pd.DataFrame],
    column_name: str,
    *,
    instructions: str,
    save_dir: str,
    model: str = "gpt-5.6-terra",
    modality: str = "text",
    n_rounds: int = 1,
    n_runs: int = 1,
    later_round_run_multiplier: int = 5,
    revised_column_name: Optional[str] = None,
    validation_attribute: Optional[Dict[str, str]] = None,
    use_modified_source: bool = False,
    n_parallels: int = 650,
    reset_files: bool = False,
    json_mode: bool = False,
    reasoning_effort: Optional[str] = None,
    search_context_size: str = "medium",
    file_name: str = "paraphrase_responses.csv",
    template_path: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Rewrites texts consistently per instructions.

    Example Use
    -----------
    Summarize earnings call transcripts to remove company specifics.

    Parameters
    ----------
    df:
        DataFrame containing passages to paraphrase. When ``None``, load cached
        results from ``save_dir`` instead of recomputing.
    column_name:
        Column with text to rewrite.
    instructions:
        Guidance describing how the paraphrase should differ from the source.
    save_dir:
        Directory where paraphrase outputs are written.
    model:
        Model name used for generation. The signature default was current when
        this package version shipped; model IDs change, so verify the exact slug
        in the official OpenAI model catalog before overriding it.
    modality:
        Modality of inputs (text, image, audio, pdf, entity, web).
    n_rounds:
        Maximum number of paraphrase/validation cycles. ``1`` disables recursion.
    n_runs:
        Number of paraphrases to produce per passage and per round.
    later_round_run_multiplier:
        Multiplier applied to ``n_runs`` for rounds after the first one.
    revised_column_name:
        Optional name for the paraphrased column; defaults to a generated one.
    validation_attribute:
        Optional custom classifier label definition used to validate candidates; if multiple
        entries are provided, the first key/value pair is used.
    use_modified_source:
        If ``True`` allow modified source text to be used during validation.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    reset_files:
        When ``True`` regenerate outputs even if files already exist.
    json_mode:
        Whether to request JSON responses.
    reasoning_effort:
        Controls how intensely the model reasons (none/low/medium/high). Higher is smarter but slower.
    search_context_size:
        Web search context size when ``modality="web"``.
    file_name:
        CSV filename for saved paraphrases.
    template_path:
        Custom template path to override the default paraphrase prompt.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.
    **cfg_kwargs:
        Additional configuration passed to :class:`gabriel.tasks.paraphrase.ParaphraseConfig`. Keys matching :func:`gabriel.utils.openai_utils.get_all_responses` / :func:`gabriel.utils.openai_utils.get_response` (for example ``image_detail``) are forwarded to the model call.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing paraphrased text and any validation scores.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    if df is None:
        base_name = os.path.splitext(file_name)[0]
        final_path = os.path.join(save_dir, f"{base_name}_cleaned.csv")
        return _load_cached_dataframe(final_path, task_name="Paraphrase")
    cfg_kwargs, response_kwargs = _split_cfg_and_response_kwargs(
        ParaphraseConfig,
        dict(cfg_kwargs),
        task_name="paraphrase",
    )
    cfg = ParaphraseConfig(
        instructions=instructions,
        revised_column_name=revised_column_name,
        n_rounds=n_rounds,
        n_runs=n_runs,
        later_round_run_multiplier=later_round_run_multiplier,
        use_modified_source=use_modified_source,
        validation_attribute=validation_attribute,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        json_mode=json_mode,
        modality=modality,
        n_parallels=n_parallels,
        search_context_size=search_context_size,
        reasoning_effort=reasoning_effort,
        **cfg_kwargs,
    )
    return await Paraphrase(cfg, template_path=template_path).run(
        df,
        column_name,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        **response_kwargs,
    )


async def compare(
    df: Optional[pd.DataFrame],
    circle_column_name: str,
    square_column_name: str,
    *,
    save_dir: str,
    differentiate: bool = True,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5.6-terra",
    n_parallels: int = 650,
    n_runs: int = 1,
    reset_files: bool = False,
    file_name: str = "comparison_responses.csv",
    modality: str = "text",
    reasoning_effort: Optional[str] = None,
    template_path: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Identifies similarities / differences between paired items. Output = list of differences.

    Example Use
    -----------
    Contrast op-eds from different districts; compare two ad campaigns.

    Parameters
    ----------
    df:
        DataFrame containing the paired passages to compare. When ``None``,
        cached results cannot be loaded because this task does not persist a
        final output DataFrame.
    circle_column_name, square_column_name:
        Columns representing the two sides of each comparison.
    save_dir:
        Directory where comparison outputs are written.
    differentiate:
        Whether to prompt the model to emphasise key differences.
    additional_instructions:
        Extra prompt guidance applied to each comparison.
    model:
        Model name for comparison calls. The signature default was current when
        this package version shipped; model IDs change, so verify the exact slug
        in the official OpenAI model catalog before overriding it.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    n_runs:
        Number of repeated comparisons to gather per pair.
    reset_files:
        When ``True`` regenerate results regardless of existing files.
    file_name:
        CSV filename for saved comparison responses.
    modality:
        Content modality hint for prompt rendering.
    reasoning_effort:
        Controls how intensely the model reasons (none/low/medium/high). Higher is smarter but slower.
    template_path:
        Custom template override for comparison prompts.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.
    **cfg_kwargs:
        Additional configuration passed to :class:`gabriel.tasks.compare.CompareConfig`. Keys matching :func:`gabriel.utils.openai_utils.get_all_responses` / :func:`gabriel.utils.openai_utils.get_response` (for example ``image_detail``) are forwarded to the model call.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by both input columns with one row per attribute and
        an ``explanation`` field describing the preference rationale.
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    if df is None:
        return _load_cached_dataframe(None, task_name="Compare")
    cfg_kwargs, response_kwargs = _split_cfg_and_response_kwargs(
        CompareConfig,
        dict(cfg_kwargs),
        task_name="compare",
    )
    cfg = CompareConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        differentiate=differentiate,
        additional_instructions=additional_instructions or "",
        modality=modality,
        reasoning_effort=reasoning_effort,
        **cfg_kwargs,
    )
    return await Compare(cfg, template_path=template_path).run(
        df,
        circle_column_name,
        square_column_name,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        **response_kwargs,
    )


async def bucket(
    df: Optional[pd.DataFrame],
    column_name: str,
    *,
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5.6-terra",
    n_parallels: int = 650,
    reset_files: bool = False,
    file_name: str = "bucket_definitions.csv",
    bucket_count: int = 10,
    differentiate: bool = False,
    reasoning_effort: Optional[str] = None,
    template_path: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Builds taxonomies from many terms. Output = bucket/cluster labels.

    Example Use
    -----------
    Group technologies, artworks, or HR complaints into emergent categories.

    Parameters
    ----------
    df:
        DataFrame containing passages to bucket. When ``None``, load cached
        results from ``save_dir`` instead of recomputing.
    column_name:
        Column holding the text to cluster.
    save_dir:
        Directory where bucket definitions and intermediate state are saved.
    additional_instructions:
        Extra prompt guidance for bucket creation.
    model:
        Model used to propose bucket definitions. The signature default was
        current when this package version shipped; model IDs change, so verify
        the exact slug in the official OpenAI model catalog before overriding it.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    reset_files:
        When ``True`` regenerate outputs despite existing files.
    file_name:
        Filename for saved bucket definitions.
    bucket_count:
        Target number of buckets to generate.
    differentiate:
        Whether to encourage distinctive bucket descriptions.
    reasoning_effort:
        Controls how intensely the model reasons (none/low/medium/high). Higher is smarter but slower.
    template_path:
        Custom template path for bucket prompts.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.
    **cfg_kwargs:
        Additional overrides forwarded to :class:`gabriel.tasks.bucket.BucketConfig`. Keys matching :func:`gabriel.utils.openai_utils.get_all_responses` / :func:`gabriel.utils.openai_utils.get_response` (for example ``image_detail``) are forwarded to the model call.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the finalized bucket names and definitions (one
        row per bucket).
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    if df is None:
        final_path = os.path.join(save_dir, file_name)
        return _load_cached_dataframe(final_path, task_name="Bucket")
    cfg_kwargs, response_kwargs = _split_cfg_and_response_kwargs(
        BucketConfig,
        dict(cfg_kwargs),
        task_name="bucket",
    )
    cfg = BucketConfig(
        bucket_count=bucket_count,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        additional_instructions=additional_instructions,
        differentiate=differentiate,
        reasoning_effort=reasoning_effort,
        **cfg_kwargs,
    )
    return await Bucket(cfg, template_path=template_path).run(
        df,
        column_name,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        **response_kwargs,
    )


async def discover(
    df: Optional[pd.DataFrame],
    *,
    column_name: Optional[str] = None,
    circle_column_name: Optional[str] = None,
    square_column_name: Optional[str] = None,
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5.6-terra",
    n_parallels: int = 650,
    reset_files: bool = False,
    n_runs: int = 1,
    min_frequency: float = 0.6,
    bucket_count: int = 10,
    differentiate: bool = True,
    max_words_per_call: int = 1000,
    max_categories_per_call: int = 8,
    n_terms_per_prompt: int = 250,
    repeat_bucketing: int = 5,
    repeat_voting: int = 25,
    next_round_frac: float = 0.25,
    top_k_per_round: int = 1,
    raw_term_definitions: bool = True,
    modality: str = "text",
    reasoning_effort: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    **cfg_kwargs,
) -> Dict[str, pd.DataFrame]:
    """Discovers natural language features which discriminate two classes of data.

    Example Use
    -----------
    Identify what distinguishes 5 star vs. 1 star reviews or successful vs. failed startups.

    Parameters
    ----------
    df:
        DataFrame containing the corpus to mine for labels. When ``None``,
        cached results cannot be loaded because this task does not persist a
        final output DataFrame.
    column_name:
        Column with free-form text to analyse. Optional when providing paired
        circle/square columns for contrastive discovery.
    circle_column_name, square_column_name:
        Optional paired columns enabling bidirectional discovery.
    save_dir:
        Directory where intermediate and final discovery outputs are saved.
    additional_instructions:
        Extra guidance applied throughout the discovery pipeline.
    model:
        Model used for bucket definitions and classification. The signature
        default was current when this package version shipped; model IDs change,
        so verify the exact slug in the official OpenAI model catalog before
        overriding it.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    n_runs:
        Number of classification repetitions to stabilise label prevalence.
    min_frequency:
        Minimum frequency threshold for labels to persist.
    bucket_count:
        Target number of buckets to propose in the initial step.
    differentiate:
        Encourage distinctive bucket descriptions when ``True``.
    max_words_per_call, max_categories_per_call:
        Chunking controls for classification prompts.
    n_terms_per_prompt, repeat_bucketing, repeat_voting:
        Parameters that regulate how many discovered terms are evaluated and how
        often bucketing/voting rounds repeat.
    next_round_frac, top_k_per_round:
        Controls for carrying top-performing terms into subsequent rounds.
    raw_term_definitions:
        Whether to keep raw label definitions in the outputs.
    modality:
        Content modality hint forwarded to downstream tasks.
    reasoning_effort:
        Controls how intensely the model reasons (none/low/medium/high). Higher is smarter but slower.
    reset_files:
        When ``True`` regenerate all discovery artifacts.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.
    **cfg_kwargs:
        Additional overrides passed to :class:`gabriel.tasks.discover.DiscoverConfig`. Keys matching :func:`gabriel.utils.openai_utils.get_all_responses` / :func:`gabriel.utils.openai_utils.get_response` (for example ``image_detail``) are forwarded to the model call.

    Returns
    -------
    Dict[str, pandas.DataFrame]
        Intermediate DataFrames from each step of the discovery pipeline. When
        ``circle_column_name`` and ``square_column_name`` are provided,
        classification is performed twice (circle and square directions). A
        ``summary`` key describes label prevalence differences with
        ``difference_pct`` expressed as circle minus square percentage points.
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    if df is None:
        raise ValueError(
            "Discover does not persist a final output DataFrame; "
            "provide a DataFrame to run the task."
        )
    cfg_kwargs, response_kwargs = _split_cfg_and_response_kwargs(
        DiscoverConfig,
        dict(cfg_kwargs),
        task_name="discover",
    )
    cfg = DiscoverConfig(
        save_dir=save_dir,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        min_frequency=min_frequency,
        bucket_count=bucket_count,
        additional_instructions=additional_instructions,
        differentiate=differentiate,
        max_words_per_call=max_words_per_call,
        max_categories_per_call=max_categories_per_call,
        n_terms_per_prompt=n_terms_per_prompt,
        repeat_bucketing=repeat_bucketing,
        repeat_voting=repeat_voting,
        next_round_frac=next_round_frac,
        top_k_per_round=top_k_per_round,
        raw_term_definitions=raw_term_definitions,
        modality=modality,
        reasoning_effort=reasoning_effort,
        **cfg_kwargs,
    )
    return await Discover(cfg).run(
        df,
        column_name=column_name,
        circle_column_name=circle_column_name,
        square_column_name=square_column_name,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        **response_kwargs,
    )


async def deduplicate(
    df: Optional[pd.DataFrame],
    column_name: str,
    *,
    save_dir: str,
    additional_instructions: Optional[str] = None,
    modality: str = "entity",
    max_words_per_text: int = 500,
    model: str = "gpt-5.6-terra",
    n_parallels: int = 650,
    n_runs: int = 3,
    reset_files: bool = False,
    file_name: str = "deduplicate_responses.csv",
    use_embeddings: bool = True,
    group_size: int = 500,
    template_path: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    embedding_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_embeddings_fn: Optional[Callable[..., Awaitable[Dict[str, List[float]]]]] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Detects conceptual duplicates. Maps all duplicates to one representative term.

    Example Use
    -----------
    Collapse "F-18", "Super Hornet Fighter Jet", "f-18 hornet" into "F-18".

    Parameters
    ----------
    df:
        DataFrame containing the passages to deduplicate. When ``None``,
        cached results cannot be loaded because this task does not persist a
        final output DataFrame.
    column_name:
        Column holding the text to deduplicate.
    save_dir:
        Directory where deduplication artifacts are written.
    additional_instructions:
        Extra guidance appended to the deduplication prompt.
    modality:
        Use ``"entity"`` for short entity strings or ``"text"`` for long-form text snippets.
    max_words_per_text:
        Maximum word count for each text snippet when ``modality="text"``.
    model:
        Model name used for overlap detection. The signature default was current
        when this package version shipped; model IDs change, so verify the exact
        slug in the official OpenAI model catalog before overriding it.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    n_runs:
        Number of passes to run; helps stabilise duplicate detection.
    reset_files:
        When ``True`` regenerate outputs regardless of existing files.
    file_name:
        CSV filename for saved deduplication responses.
    use_embeddings:
        Whether to use embedding-based prefiltering prior to model calls.
    group_size:
        Number of passages to evaluate per batch during deduplication.
    template_path:
        Custom template override for deduplication prompts.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.
    embedding_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_embeddings`
        that replaces only the per-text embedding call.
    get_all_embeddings_fn:
        Optional callable that fully replaces
        :func:`gabriel.utils.openai_utils.get_all_embeddings`.
    **cfg_kwargs:
        Additional configuration passed to
        :class:`gabriel.tasks.deduplicate.DeduplicateConfig`. Keys matching
        :func:`gabriel.utils.openai_utils.get_all_responses` /
        :func:`gabriel.utils.openai_utils.get_response` (for example
        ``image_detail``) are forwarded to the model call.

    Returns
    -------
    pandas.DataFrame
        DataFrame including the original content plus ``mapped_<column_name>`` columns
        (per run and final) indicating the canonical representative for each
        detected duplicate cluster.
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    if df is None:
        return _load_cached_dataframe(None, task_name="Deduplicate")
    cfg_kwargs, response_kwargs = _split_cfg_and_response_kwargs(
        DeduplicateConfig,
        dict(cfg_kwargs),
        task_name="deduplicate",
    )
    cfg = DeduplicateConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        additional_instructions=additional_instructions,
        use_embeddings=use_embeddings,
        group_size=group_size,
        modality=modality,
        max_words_per_text=max_words_per_text,
        **cfg_kwargs,
    )
    return await Deduplicate(cfg, template_path=template_path).run(
        df,
        column_name=column_name,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        embedding_fn=embedding_fn,
        get_all_embeddings_fn=get_all_embeddings_fn,
        **response_kwargs,
    )


async def merge(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    save_dir: str,
    on: Optional[str] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    how: str = "left",
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5.6-luna",
    n_parallels: int = 650,
    n_runs: int = 1,
    reset_files: bool = False,
    file_name: str = "merge_responses.csv",
    use_embeddings: bool = True,
    short_list_len: int = 16,
    long_list_len: int = 256,
    max_attempts: int = 4,
    short_list_multiplier: float = 0.5,
    auto_match_threshold: float = 0.75,
    use_best_auto_match: bool = False,
    candidate_scan_chunks: int = 5,
    template_path: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    embedding_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_embeddings_fn: Optional[Callable[..., Awaitable[Dict[str, List[float]]]]] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Creates crosswalks. Output = merged table with GPT-matched identifiers.

    Example Use
    -----------
    Match two distinct job title directories; link patent titles to product names.

    Parameters
    ----------
    df_left, df_right:
        DataFrames to merge.
    save_dir:
        Directory where merge results and diagnostics are saved.
    on, left_on, right_on:
        Column(s) to match on. ``on`` applies to both sides; ``left_on`` and
        ``right_on`` override per side.
    how:
        Merge strategy (``"left"`` or ``"right"``) determining which side is treated as
        the short/base table.
    additional_instructions:
        Extra prompt context for the model.
    model:
        Model used to compare candidate records. The signature default was
        current when this package version shipped; model IDs change, so verify
        the exact slug in the official OpenAI model catalog before overriding it.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    n_runs:
        Number of repeated comparisons per candidate.
    reset_files:
        When ``True`` regenerate outputs even if files exist.
    file_name:
        CSV filename for saved merge responses.
    use_embeddings:
        Whether to use embeddings to shortlist candidates before calling the
        model.
    short_list_len, long_list_len, short_list_multiplier:
        Controls for candidate pool sizes.
    max_attempts:
        Maximum retry attempts per match before giving up.
    auto_match_threshold:
        Confidence threshold for automatically accepting matches.
    use_best_auto_match:
        When ``True`` pick the highest confidence candidate when multiple exceed
        ``auto_match_threshold``.
    candidate_scan_chunks:
        Number of candidate batches to scan when building the shortlist.
    template_path:
        Custom template override for merge prompts.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.
    embedding_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_embeddings`
        that replaces only the per-text embedding call.
    get_all_embeddings_fn:
        Optional callable that fully replaces
        :func:`gabriel.utils.openai_utils.get_all_embeddings`.
    **cfg_kwargs:
        Additional overrides forwarded to :class:`gabriel.tasks.merge.MergeConfig`. Keys matching :func:`gabriel.utils.openai_utils.get_all_responses` / :func:`gabriel.utils.openai_utils.get_response` (for example ``image_detail``) are forwarded to the model call.

    Returns
    -------
    pandas.DataFrame
        Merged result keyed to the ``how``-selected short side, enriched with
        model-evaluated matches from the long side and deduplicated on the
        short key.
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg_kwargs, response_kwargs = _split_cfg_and_response_kwargs(
        MergeConfig,
        dict(cfg_kwargs),
        task_name="merge",
    )
    cfg = MergeConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        additional_instructions=additional_instructions,
        use_embeddings=use_embeddings,
        short_list_len=short_list_len,
        long_list_len=long_list_len,
        max_attempts=max_attempts,
        short_list_multiplier=short_list_multiplier,
        auto_match_threshold=auto_match_threshold,
        use_best_auto_match=use_best_auto_match,
        candidate_scan_chunks=candidate_scan_chunks,
        **cfg_kwargs,
    )
    return await Merge(cfg, template_path=template_path).run(
        df_left,
        df_right,
        on=on,
        left_on=left_on,
        right_on=right_on,
        how=how,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        embedding_fn=embedding_fn,
        get_all_embeddings_fn=get_all_embeddings_fn,
        **response_kwargs,
    )


async def filter(
    df: Optional[pd.DataFrame],
    column_name: str,
    *,
    condition: str,
    save_dir: str,
    entities_per_call: int = 150,
    shuffle: bool = True,
    random_seed: int = 42,
    n_runs: int = 1,
    threshold: float = 0.5,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5.6-luna",
    n_parallels: int = 650,
    reset_files: bool = False,
    file_name: str = "filter_responses.csv",
    template_path: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """High-throughput boolean screening. Outputs items which meet natural language condition.

    Example Use
    -----------
    Subset 18M Wikipedia titles to only technologies.

    Parameters
    ----------
    df:
        DataFrame containing passages to filter. When ``None``, cached results
        cannot be loaded because this task does not persist a final output
        DataFrame.
    column_name:
        Column with the text to evaluate.
    condition:
        Natural-language condition that determines whether a passage is kept.
    save_dir:
        Directory where filter responses are saved.
    entities_per_call:
        Number of passages to send in each API call.
    shuffle:
        Whether to randomise order before batching.
    random_seed:
        Seed used when ``shuffle`` is ``True``.
    n_runs:
        Number of repeated evaluations per passage.
    threshold:
        Probability threshold above which a passage is retained.
    additional_instructions:
        Extra guidance appended to the filter prompt.
    model:
        Model used for filtering. The signature default was current when this
        package version shipped; model IDs change, so verify the exact slug in
        the official OpenAI model catalog before overriding it.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    reset_files:
        When ``True`` regenerate outputs even if files exist.
    file_name:
        CSV filename for saved filter responses.
    template_path:
        Custom prompt template path.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.
    **cfg_kwargs:
        Additional configuration passed to :class:`gabriel.tasks.filter.FilterConfig`. Keys matching :func:`gabriel.utils.openai_utils.get_all_responses` / :func:`gabriel.utils.openai_utils.get_response` (for example ``image_detail``) are forwarded to the model call.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with keep/score columns reflecting model decisions.
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    if df is None:
        return _load_cached_dataframe(None, task_name="Filter")
    cfg_kwargs, response_kwargs = _split_cfg_and_response_kwargs(
        FilterConfig,
        dict(cfg_kwargs),
        task_name="filter",
    )
    cfg = FilterConfig(
        condition=condition,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        entities_per_call=entities_per_call,
        shuffle=shuffle,
        random_seed=random_seed,
        n_runs=n_runs,
        threshold=threshold,
        additional_instructions=additional_instructions or "",
        **cfg_kwargs,
    )
    return await Filter(cfg, template_path=template_path).run(
        df,
        column_name,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        **response_kwargs,
    )


async def debias(
    df: Optional[pd.DataFrame],
    column_name: str,
    *,
    mode: MeasurementMode = "rate",
    measurement_attribute: Optional[str] = None,
    removal_attribute: Optional[str] = None,
    signal_dictionary: Dict[str, str],
    attributes: Optional[Dict[str, str]] = None,
    removal_method: RemovalMethod = "codify",
    save_dir: str = os.path.expanduser("~/Documents/runs"),
    run_name: Optional[str] = None,
    strip_percentages: Optional[List[int]] = None,
    categories_to_strip: Optional[List[str]] = None,
    template_path: Optional[str] = None,
    model: str = "gpt-5.6-terra",
    n_parallels: int = 650,
    measurement_kwargs: Optional[Dict[str, Any]] = None,
    removal_kwargs: Optional[Dict[str, Any]] = None,
    remaining_signal: bool = True,
    max_words_per_call: Optional[int] = 1000,
    n_rounds: Optional[int] = 3,
    robust_regression: bool = True,
    random_seed: int = 12345,
    verbose: bool = True,
    reset_files: bool = False,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
) -> pd.DataFrame:
    """Post-process measurements to remove inference bias.

    Example Use
    -----------
    Ensure GPT isn't guessing climate opinions in speeches based on general political lean.

    Parameters
    ----------
    df:
        DataFrame containing passages to measure and debias. When ``None``,
        load cached results from ``save_dir`` instead of recomputing.
    column_name:
        Column with the text to process.
    mode:
        Measurement mode (e.g., ``"rate"``) determining how bias is estimated.
    measurement_attribute, removal_attribute:
        Specify the attribute used for regression and the key from
        ``signal_dictionary`` that should be removed. When
        ``measurement_attribute`` is omitted the first key from ``attributes``
        is used. ``removal_attribute`` defaults to the measurement attribute
        when present in ``signal_dictionary`` or otherwise the first key from
        ``signal_dictionary``. Notices are printed when inferred and
        ``verbose`` is ``True``.
    signal_dictionary:
        Mapping of bias signals to their definitions.
    attributes:
        Optional rating attributes used during measurement.
    removal_method:
        Strategy for removing bias (for example ``"codify"``).
    save_dir:
        Base directory for all debiasing artifacts.
    run_name:
        Optional run identifier; defaults to a timestamped folder.
    strip_percentages, categories_to_strip:
        Optional controls for category pruning during removal.
    template_path:
        Optional template override used during removal steps.
    model:
        Model used across the measurement and removal stages. The signature
        default was current when this package version shipped; model IDs change,
        so verify the exact slug in the official OpenAI model catalog before
        overriding it.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    measurement_kwargs, removal_kwargs:
        Fine-grained overrides for the measurement and removal tasks. A legacy
        ``max_output_tokens`` entry is accepted, warned about, and ignored.
    remaining_signal:
        When ``True`` (default) measure a remaining-signal prevalence attribute on
        the stripped text and use it in the two-step debiasing regression.
    max_words_per_call, n_rounds:
        Convenience passthroughs for the removal stage. ``max_words_per_call``
        configures the codify task's chunk size, while ``n_rounds`` controls the
        number of completion passes run by codify and any downstream
        paraphrasing steps. Defaults to 3 when not explicitly provided.
    robust_regression:
        Whether to use robust regression when estimating bias coefficients.
    random_seed:
        Seed for deterministic behaviour in sampling-heavy steps.
    verbose:
        When ``True`` print notices about inferred defaults and progress.
    reset_files:
        When ``True`` propagate reset behaviour to all measurement and removal stages.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.

    Returns
    -------
    pandas.DataFrame
        Debiased results with raw, stripped, and debiased columns appended.
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    if df is None:
        run_name = run_name or _debias_default_run_name(
            measurement_attribute, removal_attribute
        )
        run_dir = os.path.join(save_dir, run_name)
        final_path = os.path.join(run_dir, "debias_results.csv")
        return _load_cached_dataframe(final_path, task_name="Debias")
    measurement_kwargs = dict(measurement_kwargs or {})
    removal_kwargs = dict(removal_kwargs or {})
    if response_fn is not None:
        measurement_kwargs.setdefault("response_fn", response_fn)
        removal_kwargs.setdefault("response_fn", response_fn)
    if get_all_responses_fn is not None:
        measurement_kwargs.setdefault("get_all_responses_fn", get_all_responses_fn)
        removal_kwargs.setdefault("get_all_responses_fn", get_all_responses_fn)

    if reset_files:
        measurement_kwargs.setdefault("reset_files", True)
        removal_kwargs.setdefault("reset_files", True)

    if removal_method == "codify" and max_words_per_call is not None:
        removal_kwargs.setdefault("max_words_per_call", max_words_per_call)
    if "completion_max_rounds" in removal_kwargs and "n_rounds" not in removal_kwargs:
        replacement = removal_kwargs.pop("completion_max_rounds")
        warnings.warn(
            "completion_max_rounds in removal_kwargs is deprecated; use n_rounds instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if replacement is not None:
            removal_kwargs.setdefault("n_rounds", replacement)
    if n_rounds is not None:
        removal_kwargs.setdefault("n_rounds", n_rounds)

    cfg = DebiasConfig(
        mode=mode,
        measurement_attribute=measurement_attribute,
        removal_attribute=removal_attribute,
        signal_dictionary=signal_dictionary,
        attributes=attributes or {},
        removal_method=removal_method,
        save_dir=save_dir,
        run_name=run_name,
        strip_percentages=strip_percentages,
        categories_to_strip=categories_to_strip,
        template_path=template_path,
        model=model,
        n_parallels=n_parallels,
        measurement_kwargs=measurement_kwargs,
        removal_kwargs=removal_kwargs,
        remaining_signal=remaining_signal,
        robust_regression=robust_regression,
        random_seed=random_seed,
        verbose=verbose,
    )
    pipeline = DebiasPipeline(cfg)
    result = await pipeline.run(df, column_name, reset_files=reset_files)
    return result.results


async def whatever(
    prompts: Optional[Union[str, List[str], pd.DataFrame]] = None,
    identifiers: Optional[List[str]] = None,
    *,
    save_dir: str,
    df: Optional[pd.DataFrame] = None,
    column_name: Optional[str] = None,
    identifier_column: Optional[str] = None,
    image_column: Optional[str] = None,
    audio_column: Optional[str] = None,
    prompt_images: Optional[Dict[str, List[str]]] = None,
    prompt_audio: Optional[Dict[str, List[Dict[str, str]]]] = None,
    file_name: str = "custom_prompt_responses.csv",
    model: str = "gpt-5.6-terra",
    json_mode: bool = False,
    web_search: Optional[bool] = None,
    web_search_filters: Optional[Dict[str, Any]] = None,
    search_context_size: str = "medium",
    n_parallels: int = 650,
    reset_files: bool = False,
    return_original_columns: bool = True,
    drop_prompts: bool = True,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Run any GPT prompts, but leverage GABRIEL's parallelization / checkpointing.

    Example Use
    -----------
    Any set of prompts; slots into any pipeline.

    Parameters
    ----------
    prompts:
        Single prompt string, list of prompts, or DataFrame of prompts.
    identifiers:
        Optional identifiers to align responses with custom keys.
    save_dir:
        Directory where raw responses are written.
    df:
        Source DataFrame to pull prompts from when ``prompts`` is not provided.
    column_name:
        Column in ``df`` containing prompts to send.
    identifier_column:
        Column providing identifiers for each prompt row.
    image_column, audio_column:
        Optional columns containing image or audio references to include.
    prompt_images, prompt_audio:
        Pre-constructed multimodal payloads keyed by identifier.
    file_name:
        CSV filename for persisted responses.
    model:
        Model name passed to :func:`gabriel.utils.openai_utils.get_all_responses`.
        The signature default was current when this package version shipped;
        model IDs change, so verify the exact slug in the official OpenAI model
        catalog before overriding it.
    json_mode:
        Whether to request JSON-mode responses where supported.
    web_search:
        Enable web search augmentation.
    web_search_filters:
        Filters dict forwarded to the Responses API (allowed domains and optional
        location hints such as ``city`` or ``timezone``).
    search_context_size:
        Context size hint for web-search capable models.
    n_parallels:
        Concurrency ceiling, not a fixed worker count. Keep the default unchanged:
        GABRIEL ramps up and adapts to rate limits and repeated errors. Temporary
        retries and slow stragglers are normal; let a progressing run finish.
        Reduce it only for a persistent failure or known deployment constraint.
    reset_files:
        When ``True`` regenerate outputs even if files already exist.
    return_original_columns:
        When ``True`` and ``df`` is provided, merge response columns back onto
        the input DataFrame using the prompt identifiers.
    drop_prompts:
        When ``True`` and merging back onto ``df``, drop the prompt column
        before saving/returning the result.
    reasoning_effort, reasoning_summary:
        Controls how intensely the model reasons (none/low/medium/high). Higher is smarter but slower.
    response_fn:
        Optional callable forwarded to :func:`gabriel.utils.openai_utils.get_all_responses`
        that replaces the per-prompt model invocation. Ignored when
        ``get_all_responses_fn`` is supplied.
    get_all_responses_fn:
        Optional callable that fully replaces :func:`gabriel.utils.openai_utils.get_all_responses`.
        It must accept ``prompts`` and ``identifiers`` (and ideally ``model`` and
        ``json_mode``) and return a DataFrame containing a ``"Response"`` column.
    **kwargs:
        Additional parameters forwarded directly to
        :func:`gabriel.utils.openai_utils.get_all_responses`. The deprecated
        ``max_output_tokens`` argument is accepted, warned about, and ignored.

    Returns
    -------
    pandas.DataFrame
        DataFrame of prompts, identifiers, and model responses saved to
        ``save_dir/file_name``.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)

    if df is None and prompts is None:
        raise ValueError("Either prompts or df must be provided to `whatever`.")

    kwargs = dict(kwargs)
    _discard_deprecated_max_output_tokens(kwargs, stacklevel=3)
    if response_fn is None:
        response_fn = kwargs.pop("response_fn", None)
    else:
        kwargs.pop("response_fn", None)
    if get_all_responses_fn is None:
        get_all_responses_fn = kwargs.pop("get_all_responses_fn", None)
    else:
        kwargs.pop("get_all_responses_fn", None)

    if web_search is None and "web_search" in kwargs:
        web_search = kwargs.pop("web_search")
    else:
        kwargs.pop("web_search", None)

    if web_search_filters is None and "web_search_filters" in kwargs:
        web_search_filters = kwargs.pop("web_search_filters")
    else:
        kwargs.pop("web_search_filters", None)

    if "search_context_size" in kwargs:
        if search_context_size == "medium":
            search_context_size = kwargs.pop("search_context_size")
        else:
            kwargs.pop("search_context_size")

    cfg = WhateverConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        json_mode=json_mode,
        web_search=web_search,
        web_search_filters=web_search_filters,
        search_context_size=search_context_size,
        n_parallels=n_parallels,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
    )

    runner = Whatever(cfg)
    return await runner.run(
        prompts,
        df=df,
        identifiers=identifiers,
        column_name=column_name,
        identifier_column=identifier_column,
        image_column=image_column,
        audio_column=audio_column,
        prompt_images=prompt_images,
        prompt_audio=prompt_audio,
        web_search_filters=web_search_filters,
        reset_files=reset_files,
        return_original_columns=return_original_columns,
        drop_prompts=drop_prompts,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        **kwargs,
    )


def view(
    df: pd.DataFrame,
    column_name: str,
    attributes: Optional[Union[Mapping[str, Any], Sequence[Any], Any]] = None,
    *,
    header_columns: Optional[Any] = None,
    max_passages: Optional[int] = None,
    font_scale: float = 1.0,
    font_family: Optional[str] = None,
    color_mode: str = "auto",
):
    """UI to view sample texts with ratings / passage coding.

    Example Use
    -----------
    Spot-check classify / rating outputs; view coded passages.

    Parameters
    ----------
    df:
        DataFrame containing passages to display.
    column_name:
        Column with the primary text to render.
    attributes:
        Optional iterable or mapping of attribute columns to include alongside
        the passage text.
    header_columns:
        Optional columns whose values should appear in the viewer header.
    max_passages:
        Optional cap on the number of passages displayed.
    font_scale:
        Scaling factor applied to viewer typography.
    font_family:
        Optional font family override.
    color_mode:
        Either ``"auto"``, ``"light"``, or ``"dark"`` to control the viewer
        theme.

    Returns
    -------
    Any
        The rendered viewer object produced by
        :func:`gabriel.utils.passage_viewer.view`.
    """

    return _view_passages(
        df,
        column_name,
        attributes=attributes,
        header_columns=header_columns,
        max_passages=max_passages,
        font_scale=font_scale,
        font_family=font_family,
        color_mode=color_mode,
    )
