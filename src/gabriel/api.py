import os
import warnings
import pandas as pd
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
    Filter,
    FilterConfig,
    Whatever,
    WhateverConfig,
    Ideate,
    IdeateConfig,
)
from .utils.openai_utils import get_all_responses
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

async def rate(
    df: pd.DataFrame,
    column_name: str,
    *,
    attributes: Dict[str, str],
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 650,
    n_runs: int = 1,
    n_attributes_per_run: int = 8,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "ratings.csv",
    modality: str = "text",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
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
        Source DataFrame containing the passages to rate.
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
        Model name passed through to the OpenAI Responses API.
    n_parallels:
        Maximum number of concurrent requests to issue.
    n_runs:
        Number of repeat rating passes to perform for each passage.
    n_attributes_per_run:
        Maximum number of attributes to include in a single prompt. Attributes
        are split across prompts when this limit is exceeded.
    reset_files:
        When ``True`` existing outputs in ``save_dir`` are ignored and
        regenerated.
    use_dummy:
        If ``True`` use deterministic dummy responses for offline testing.
    file_name:
        Basename (without the automatic ``_raw_responses`` suffix) for saved
        artifacts.
    modality:
        One of ``"text"``, ``"entity"``, ``"web"``, ``"image"``, ``"audio"``, or ``"pdf"``
        to control how inputs are packaged into prompts.
    reasoning_effort, reasoning_summary:
        Optional OpenAI metadata that tunes reasoning depth and summary capture.
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
        Additional overrides applied to :class:`gabriel.tasks.rate.RateConfig`.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with one column per attribute containing the mean score
        across runs.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = RateConfig(
        attributes=attributes,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        n_attributes_per_run=n_attributes_per_run,
        use_dummy=use_dummy,
        additional_instructions=additional_instructions,
        modality=modality,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        search_context_size=search_context_size,
        **cfg_kwargs,
    )
    return await Rate(cfg, template_path=template_path).run(
        df,
        column_name,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
    )

async def extract(
    df: pd.DataFrame,
    column_name: str,
    *,
    attributes: Dict[str, str],
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 650,
    n_runs: int = 1,
    n_attributes_per_run: int = 8,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "extraction.csv",
    modality: str = "entity",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
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
        Source DataFrame containing the passages to parse.
    column_name:
        Column in ``df`` with the content to extract from.
    attributes:
        Mapping of field names to descriptions of what should be extracted.
    save_dir:
        Directory where extraction outputs will be written. Created if absent.
    additional_instructions:
        Optional extra guidance injected into the extraction prompt.
    model:
        Model used for extraction via the OpenAI Responses API.
    n_parallels:
        Maximum number of concurrent extraction calls.
    n_runs:
        Number of extraction passes to perform; results are averaged when
        applicable.
    n_attributes_per_run:
        Maximum number of attributes to include in each prompt. Attributes are
        split into multiple prompts when this threshold is exceeded.
    reset_files:
        When ``True`` forces regeneration of outputs in ``save_dir``.
    use_dummy:
        If ``True`` return deterministic dummy outputs instead of real API
        calls.
    file_name:
        CSV name used when saving extraction results.
    modality:
        Indicates whether the content is ``"entity"`` text or another modality
        supported by the templates.
    reasoning_effort, reasoning_summary:
        Optional OpenAI metadata for reasoning depth and summarisation.
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
        Additional overrides forwarded to :class:`gabriel.tasks.extract.ExtractConfig`.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame augmented with one column per requested
        attribute.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = ExtractConfig(
        attributes=attributes,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        n_attributes_per_run=n_attributes_per_run,
        use_dummy=use_dummy,
        additional_instructions=additional_instructions,
        modality=modality,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        **cfg_kwargs,
    )
    return await Extract(cfg, template_path=template_path).run(
        df,
        column_name,
        reset_files=reset_files,
        types=types,
    )


async def seed(
    instructions: str,
    *,
    save_dir: str,
    file_name: str = "seed_entities.csv",
    model: str = "gpt-5.1",
    n_parallels: int = 650,
    num_entities: int = 1000,
    entities_per_generation: int = 50,
    entity_batch_frac: float = 0.25,
    existing_entities_cap: int = 100,
    use_dummy: bool = False,
    deduplicate: bool = False,
    deduplicate_sample_seed: int = 42,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    max_timeout: Optional[float] = None,
    template_path: Optional[str] = None,
    existing_entities: Optional[List[str]] = None,
    reset_files: bool = False,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
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
        Model used for generation.
    n_parallels:
        Maximum number of concurrent generation calls.
    num_entities:
        Target number of entities to generate in total.
    entities_per_generation:
        Number of entities requested from each API call.
    entity_batch_frac:
        Fraction of generated entities to keep per batch before deduplication.
    existing_entities_cap:
        Maximum number of prior entities to consider when avoiding duplicates.
    use_dummy:
        If ``True`` emit deterministic dummy seeds for offline testing.
    deduplicate:
        When ``True`` over-generate and apply a shallow deduplication pass
        before returning results.
    deduplicate_sample_seed:
        Random seed used when sampling a deterministic subset after
        deduplication.
    reasoning_effort, reasoning_summary:
        Optional OpenAI reasoning controls.
    max_timeout:
        Optional timeout in seconds for each API call.
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
    **response_kwargs:
        Additional keyword arguments forwarded to
        :func:`gabriel.utils.openai_utils.get_all_responses`.

    Returns
    -------
    pandas.DataFrame
        DataFrame of seed entities with provenance metadata.
    """

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
        use_dummy=use_dummy,
        deduplicate=deduplicate,
        deduplicate_sample_seed=deduplicate_sample_seed,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        max_timeout=max_timeout,
    )
    task = Seed(cfg, template_path=template_path)
    return await task.run(
        existing_entities=existing_entities,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
        **response_kwargs,
    )


async def classify(
    df: pd.DataFrame,
    column_name: Optional[str] = None,
    *,
    labels: Dict[str, str],
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    differentiate: bool = False,
    circle_column_name: Optional[str] = None,
    square_column_name: Optional[str] = None,
    n_parallels: int = 650,
    n_runs: int = 1,
    n_attributes_per_run: int = 8,
    min_frequency: float = 0.6,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "classify_responses.csv",
    modality: str = "text",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
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
        DataFrame containing content to classify.
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
        Model name used for classification.
    differentiate:
        When ``True`` use differentiation mode to highlight contrasts.
    circle_column_name, square_column_name:
        Optional paired columns for contrastive classification.
    n_parallels:
        Maximum number of concurrent classification calls.
    n_runs:
        Number of repeated classification passes.
    n_attributes_per_run:
        Maximum number of labels to evaluate per prompt. Labels are split into
        batches when this count is exceeded.
    min_frequency:
        Minimum label frequency required to keep a label during aggregation.
    reset_files:
        When ``True`` overwrite any existing outputs in ``save_dir``.
    use_dummy:
        If ``True`` return deterministic dummy outputs for offline testing.
    file_name:
        Basename for saved classification CSVs.
    modality:
        Indicates the content modality for prompt rendering.
    reasoning_effort, reasoning_summary:
        Optional OpenAI reasoning controls.
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
        Extra configuration passed to :class:`gabriel.tasks.classify.ClassifyConfig`.

    Returns
    -------
    pandas.DataFrame
        DataFrame including one column per label plus ``predicted_classes``; aggregates repeated runs using ``min_frequency``.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
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
        use_dummy=use_dummy,
        modality=modality,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
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
    )


async def ideate(
    topic: str,
    *,
    save_dir: str,
    file_name: str = "ideation.csv",
    model: str = "gpt-5-mini",
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
    use_dummy: bool = False,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    reset_files: bool = False,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    rank_config_updates: Optional[Dict[str, Any]] = None,
    rank_run_kwargs: Optional[Dict[str, Any]] = None,
    rate_config_updates: Optional[Dict[str, Any]] = None,
    rate_run_kwargs: Optional[Dict[str, Any]] = None,
    use_seed_entities: Optional[bool] = None,
    seed_deduplicate: bool = True,
    seed_config_updates: Optional[Dict[str, Any]] = None,
    seed_run_kwargs: Optional[Dict[str, Any]] = None,
    template_path: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
) -> pd.DataFrame:
    """Generates many novel scientific theories and filters the cream of the crop.

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
        Models used for idea generation and ranking (if different).
    scientific_theory:
        When ``True`` generate novel scientific theories; when ``False`` generate
        general-purpose ideas with the same output structure.
    n_ideas:
        Target number of ideas to generate before pruning.
    n_parallels:
        Maximum concurrent calls for generation and ranking phases.
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
    use_dummy:
        When ``True`` perform deterministic offline runs.
    reasoning_effort, reasoning_summary:
        Optional OpenAI reasoning controls.
    reset_files:
        Force regeneration of outputs in ``save_dir``.
    *_config_updates, *_run_kwargs:
        Fine-grained overrides for nested Rate/Rank/Seed tasks.
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
        use_dummy=use_dummy,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        seed_deduplicate=seed_deduplicate,
    )
    if attributes is not None:
        cfg_kwargs["attributes"] = attributes
    cfg = IdeateConfig(**cfg_kwargs)

    def _with_callable_overrides(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        updated = dict(payload or {})
        if response_fn is not None:
            updated.setdefault("response_fn", response_fn)
        if get_all_responses_fn is not None:
            updated.setdefault("get_all_responses_fn", get_all_responses_fn)
        return updated

    generation_kwargs = _with_callable_overrides(generation_kwargs)
    rank_run_kwargs = _with_callable_overrides(rank_run_kwargs)
    rate_run_kwargs = _with_callable_overrides(rate_run_kwargs)
    seed_run_kwargs = _with_callable_overrides(seed_run_kwargs)

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
    )


async def id8(*args, **kwargs) -> pd.DataFrame:
    """Alias for :func:`ideate`."""

    return await ideate(*args, **kwargs)


async def deidentify(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    grouping_column: Optional[str] = None,
    mapping_column: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 650,
    use_dummy: bool = False,
    file_name: str = "deidentified.csv",
    max_words_per_call: int = 7500,
    additional_instructions: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
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
        DataFrame containing passages to deidentify.
    column_name:
        Column in ``df`` holding the text to scrub.
    save_dir:
        Directory where anonymised outputs and mappings are written.
    grouping_column:
        Optional column grouping records that should share replacements.
    mapping_column:
        Optional column providing deterministic replacement tokens.
    model:
        Model name used to perform the deidentification.
    n_parallels:
        Maximum concurrent requests.
    use_dummy:
        When ``True`` produce deterministic dummy replacements for testing.
    file_name:
        CSV filename used when persisting deidentified text.
    max_words_per_call:
        Chunk size control for long passages.
    additional_instructions:
        Extra guidance appended to the prompt.
    reasoning_effort, reasoning_summary:
        Optional OpenAI reasoning controls.
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
        Additional overrides for :class:`gabriel.tasks.deidentify.DeidentifyConfig`.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing deidentified text and replacement mappings.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = DeidentifyConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        use_dummy=use_dummy,
        max_words_per_call=max_words_per_call,
        additional_instructions=additional_instructions,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
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
    )

async def rank(
    df: pd.DataFrame,
    column_name: str,
    *,
    attributes: Union[Dict[str, str], List[str]],
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_rounds: int = 5,
    matches_per_round: int = 3,
    power_matching: bool = True,
    return_raw_scores: bool = False,
    learning_rate: float = 0.1,
    n_parallels: int = 650,
    n_attributes_per_run: int = 8,
    use_dummy: bool = False,
    file_name: str = "rankings",
    reset_files: bool = False,
    modality: str = "text",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
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
    id_column: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Pairwise comparisons between texts yields ELO-like attribute ratings. Output = grounded, relative z scores for each text.

    Example Use
    -----------
    Rank technologies by "bulkiness" or artworks by "fine brushwork".

    Parameters
    ----------
    df:
        DataFrame containing passages to rank.
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
        Model name used for ranking calls.
    n_rounds, matches_per_round, power_matching, learning_rate:
        Parameters controlling the Elo-style tournament mechanics.
    n_parallels:
        Maximum concurrent ranking calls.
    n_attributes_per_run:
        Maximum number of attributes to compare per prompt. Attributes are
        batched across prompts when this cap is exceeded.
    use_dummy:
        When ``True`` run deterministic offline ranking.
    file_name:
        Base filename for saved rankings (without extension).
    reset_files:
        Force regeneration of any existing outputs in ``save_dir``.
    modality:
        Content modality forwarded to the prompt.
    reasoning_effort, reasoning_summary:
        Optional OpenAI reasoning controls.
    template_path:
        Path to a custom ranking prompt template.
    recursive_*:
        Settings for recursive pruning (fraction kept, minimum remaining, etc.).
    initial_rating_pass:
        Whether to run a preliminary rating stage before comparisons. Enabled by
        default to give the tournament grounded starting scores; set to
        ``False`` to skip the rating seed.
    rate_kwargs:
        Additional configuration forwarded to the preliminary rating stage.
    primer_scores, primer_scale, primer_center:
        Optional seed ratings to prime the Bradley–Terry loop. Scores are
        centred per attribute when ``primer_center`` is ``True`` and scaled
        by ``primer_scale``.
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
        Extra parameters passed to :class:`gabriel.tasks.rank.RankConfig`.

    Returns
    -------
    pandas.DataFrame
        Ranked outputs. The CSV written to ``save_dir`` always contains raw
        scores and standard errors, but the returned DataFrame hides those
        columns unless ``return_raw_scores`` is ``True``.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = RankConfig(
        attributes=attributes,
        n_rounds=n_rounds,
        matches_per_round=matches_per_round,
        power_matching=power_matching,
        learning_rate=learning_rate,
        model=model,
        n_parallels=n_parallels,
        n_attributes_per_run=n_attributes_per_run,
        use_dummy=use_dummy,
        save_dir=save_dir,
        file_name=file_name,
        additional_instructions=additional_instructions or "",
        modality=modality,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
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
        rate_kwargs=rate_kwargs or {},
        primer_scores=primer_scores,
        primer_scale=primer_scale,
        primer_center=primer_center,
        **cfg_kwargs,
    )
    result_df = await Rank(cfg, template_path=template_path).run(
        df,
        column_name,
        id_column=id_column,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
    )

    # By default only expose the z-score columns (attribute names without suffixes)
    # to API callers while keeping the raw/SE columns persisted in the CSV output.
    if return_raw_scores:
        return result_df

    if isinstance(attributes, dict):
        attr_keys: List[str] = list(attributes.keys())
    else:
        attr_keys = list(attributes)
    drop_cols: List[str] = []
    for attr in attr_keys:
        raw_col = f"{attr}_raw"
        se_col = f"{attr}_se"
        if raw_col in result_df.columns:
            drop_cols.append(raw_col)
        if se_col in result_df.columns:
            drop_cols.append(se_col)
    if drop_cols:
        result_df = result_df.drop(columns=drop_cols)
    return result_df


async def codify(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    categories: Optional[Dict[str, str]] = None,
    additional_instructions: str = "",
    model: str = "gpt-5-mini",
    n_parallels: int = 650,
    max_words_per_call: int = 1000,
    max_categories_per_call: int = 8,
    file_name: str = "coding_results.csv",
    reset_files: bool = False,
    debug_print: bool = False,
    use_dummy: bool = False,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    modality: str = "text",
    json_mode: bool = True,
    max_timeout: Optional[float] = None,
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
        DataFrame containing the passages to code.
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
        Model used for coding requests.
    n_parallels:
        Maximum number of concurrent coding calls.
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
    use_dummy:
        Use deterministic dummy outputs for testing.
    reasoning_effort, reasoning_summary:
        Optional OpenAI reasoning controls.
    modality:
        Content modality hint (text, entity, etc.).
    json_mode:
        Request JSON-mode responses where supported.
    max_timeout:
        Optional per-call timeout.
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
        Additional overrides passed to :class:`gabriel.tasks.codify.CodifyConfig`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with coded categories and any iterative refinement metadata.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg_kwargs = dict(cfg_kwargs)
    
    cfg = CodifyConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        max_words_per_call=max_words_per_call,
        max_categories_per_call=max_categories_per_call,
        debug_print=debug_print,
        use_dummy=use_dummy,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        modality=modality,
        json_mode=json_mode,
        max_timeout=max_timeout,
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
    )


async def paraphrase(
    df: pd.DataFrame,
    column_name: str,
    *,
    instructions: str,
    save_dir: str,
    revised_column_name: Optional[str] = None,
    n_revisions: int = 1,
    file_name: str = "paraphrase_responses.csv",
    model: str = "gpt-5-mini",
    json_mode: bool = False,
    web_search: Optional[bool] = None,
    n_parallels: int = 650,
    use_dummy: bool = False,
    reset_files: bool = False,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    n_rounds: int = 1,
    recursive_validation: Optional[bool] = None,
    n_initial_candidates: int = 1,
    n_validation_candidates: int = 5,
    use_modified_source: bool = False,
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
        DataFrame containing passages to paraphrase.
    column_name:
        Column with text to rewrite.
    instructions:
        Guidance describing how the paraphrase should differ from the source.
    save_dir:
        Directory where paraphrase outputs are written.
    revised_column_name:
        Optional name for the paraphrased column; defaults to a generated one.
    n_revisions:
        Number of paraphrases to produce per passage.
    file_name:
        CSV filename for saved paraphrases.
    model:
        Model name used for generation.
    json_mode:
        Whether to request JSON responses.
    web_search:
        Enable web search augmentation when supported by the model.
    n_parallels:
        Maximum concurrent paraphrase calls.
    use_dummy:
        Produce deterministic dummy paraphrases.
    reset_files:
        When ``True`` regenerate outputs even if files already exist.
    reasoning_effort, reasoning_summary:
        Optional OpenAI reasoning controls.
    n_rounds:
        Maximum number of paraphrase/validation cycles. ``1`` disables recursion.
    recursive_validation:
        Deprecated boolean flag retained for compatibility; prefer ``n_rounds``.
    n_initial_candidates, n_validation_candidates:
        Control the number of candidates in generation and validation phases.
    use_modified_source:
        If ``True`` allow modified source text to be used during validation.
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
        Additional configuration passed to :class:`gabriel.tasks.paraphrase.ParaphraseConfig`.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing paraphrased text and any validation scores.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = ParaphraseConfig(
        instructions=instructions,
        revised_column_name=revised_column_name,
        n_revisions=n_revisions,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        json_mode=json_mode,
        web_search=web_search,
        n_parallels=n_parallels,
        use_dummy=use_dummy,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        n_rounds=n_rounds,
        recursive_validation=recursive_validation,
        n_initial_candidates=n_initial_candidates,
        n_validation_candidates=n_validation_candidates,
        use_modified_source=use_modified_source,
        **cfg_kwargs,
    )
    return await Paraphrase(cfg, template_path=template_path).run(
        df,
        column_name,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
    )


async def compare(
    df: pd.DataFrame,
    circle_column_name: str,
    square_column_name: str,
    *,
    save_dir: str,
    differentiate: bool = True,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 650,
    n_runs: int = 1,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "comparison_responses.csv",
    modality: str = "text",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
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
        DataFrame containing the paired passages to compare.
    circle_column_name, square_column_name:
        Columns representing the two sides of each comparison.
    save_dir:
        Directory where comparison outputs are written.
    differentiate:
        Whether to prompt the model to emphasise key differences.
    additional_instructions:
        Extra prompt guidance applied to each comparison.
    model:
        Model name for comparison calls.
    n_parallels:
        Maximum number of concurrent comparison requests.
    n_runs:
        Number of repeated comparisons to gather per pair.
    reset_files:
        When ``True`` regenerate results regardless of existing files.
    use_dummy:
        If ``True`` return deterministic dummy comparison outputs.
    file_name:
        CSV filename for saved comparison responses.
    modality:
        Content modality hint for prompt rendering.
    reasoning_effort, reasoning_summary:
        Optional OpenAI reasoning controls.
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
        Additional configuration passed to :class:`gabriel.tasks.compare.CompareConfig`.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by both input columns with one row per attribute and
        an ``explanation`` field describing the preference rationale.
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = CompareConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        use_dummy=use_dummy,
        differentiate=differentiate,
        additional_instructions=additional_instructions or "",
        modality=modality,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        **cfg_kwargs,
    )
    return await Compare(cfg, template_path=template_path).run(
        df,
        circle_column_name,
        square_column_name,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
    )


async def bucket(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 650,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "bucket_definitions.csv",
    bucket_count: int = 10,
    differentiate: bool = False,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
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
        DataFrame containing passages to bucket.
    column_name:
        Column holding the text to cluster.
    save_dir:
        Directory where bucket definitions and intermediate state are saved.
    additional_instructions:
        Extra prompt guidance for bucket creation.
    model:
        Model used to propose bucket definitions.
    n_parallels:
        Maximum number of concurrent bucket definition calls.
    reset_files:
        When ``True`` regenerate outputs despite existing files.
    use_dummy:
        Return deterministic dummy buckets for offline testing.
    file_name:
        Filename for saved bucket definitions.
    bucket_count:
        Target number of buckets to generate.
    differentiate:
        Whether to encourage distinctive bucket descriptions.
    reasoning_effort, reasoning_summary:
        Optional OpenAI reasoning controls.
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
        Additional overrides forwarded to :class:`gabriel.tasks.bucket.BucketConfig`.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the finalized bucket names and definitions (one
        row per bucket).
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = BucketConfig(
        bucket_count=bucket_count,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        use_dummy=use_dummy,
        additional_instructions=additional_instructions,
        differentiate=differentiate,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        **cfg_kwargs,
    )
    return await Bucket(cfg, template_path=template_path).run(
        df,
        column_name,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
    )


async def discover(
    df: pd.DataFrame,
    *,
    column_name: Optional[str] = None,
    circle_column_name: Optional[str] = None,
    square_column_name: Optional[str] = None,
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 650,
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
    use_dummy: bool = False,
    modality: str = "text",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    reset_files: bool = False,
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
        DataFrame containing the corpus to mine for labels.
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
        Model used for bucket definitions and classification.
    n_parallels:
        Maximum concurrent calls per stage.
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
    use_dummy:
        If ``True`` perform deterministic offline discovery.
    modality:
        Content modality hint forwarded to downstream tasks.
    reasoning_effort, reasoning_summary:
        Optional OpenAI reasoning controls.
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
        Additional overrides passed to :class:`gabriel.tasks.discover.DiscoverConfig`.

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
        use_dummy=use_dummy,
        modality=modality,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
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
    )


async def deduplicate(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    additional_instructions: Optional[str] = None,
    modality: str = "entity",
    max_words_per_text: int = 500,
    model: str = "gpt-5-mini",
    n_parallels: int = 650,
    n_runs: int = 3,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "deduplicate_responses.csv",
    use_embeddings: bool = True,
    group_size: int = 500,
    max_timeout: Optional[float] = None,
    template_path: Optional[str] = None,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Detects conceptual duplicates. Maps all duplicates to one representative term.

    Example Use
    -----------
    Collapse "F-18", "Super Hornet Fighter Jet", "f-18 hornet" into "F-18".

    Parameters
    ----------
    df:
        DataFrame containing the passages to deduplicate.
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
        Model name used for overlap detection.
    n_parallels:
        Maximum number of concurrent calls.
    n_runs:
        Number of passes to run; helps stabilise duplicate detection.
    reset_files:
        When ``True`` regenerate outputs regardless of existing files.
    use_dummy:
        Return deterministic dummy outputs for offline testing.
    file_name:
        CSV filename for saved deduplication responses.
    use_embeddings:
        Whether to use embedding-based prefiltering prior to model calls.
    group_size:
        Number of passages to evaluate per batch during deduplication.
    max_timeout:
        Optional timeout per API call.
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
    **cfg_kwargs:
        Additional configuration passed to
        :class:`gabriel.tasks.deduplicate.DeduplicateConfig`.

    Returns
    -------
    pandas.DataFrame
        DataFrame including the original content plus ``mapped_<column_name>`` columns
        (per run and final) indicating the canonical representative for each
        detected duplicate cluster.
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = DeduplicateConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        use_dummy=use_dummy,
        max_timeout=max_timeout,
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
    model: str = "gpt-5-nano",
    n_parallels: int = 650,
    n_runs: int = 1,
    reset_files: bool = False,
    use_dummy: bool = False,
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
        Model used to compare candidate records.
    n_parallels:
        Maximum number of concurrent merge comparisons.
    n_runs:
        Number of repeated comparisons per candidate.
    reset_files:
        When ``True`` regenerate outputs even if files exist.
    use_dummy:
        If ``True`` return deterministic dummy matches.
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
    **cfg_kwargs:
        Additional overrides forwarded to :class:`gabriel.tasks.merge.MergeConfig`.

    Returns
    -------
    pandas.DataFrame
        Merged result keyed to the ``how``-selected short side, enriched with
        model-evaluated matches from the long side and deduplicated on the
        short key.
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = MergeConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        use_dummy=use_dummy,
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
    )


async def filter(
    df: pd.DataFrame,
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
    model: str = "gpt-5-nano",
    n_parallels: int = 650,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "filter_responses.csv",
    max_timeout: Optional[float] = None,
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
        DataFrame containing passages to filter.
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
        Model used for filtering.
    n_parallels:
        Maximum number of concurrent filtering calls.
    reset_files:
        When ``True`` regenerate outputs even if files exist.
    use_dummy:
        Return deterministic dummy outputs instead of real API responses.
    file_name:
        CSV filename for saved filter responses.
    max_timeout:
        Optional per-call timeout.
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
        Additional configuration passed to :class:`gabriel.tasks.filter.FilterConfig`.

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame with keep/score columns reflecting model decisions.
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
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
        use_dummy=use_dummy,
        max_timeout=max_timeout,
        **cfg_kwargs,
    )
    return await Filter(cfg, template_path=template_path).run(
        df,
        column_name,
        reset_files=reset_files,
        response_fn=response_fn,
        get_all_responses_fn=get_all_responses_fn,
    )


async def debias(
    df: pd.DataFrame,
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
    model: str = "gpt-5-mini",
    n_parallels: int = 650,
    measurement_kwargs: Optional[Dict[str, Any]] = None,
    removal_kwargs: Optional[Dict[str, Any]] = None,
    remaining_signal: bool = True,
    max_words_per_call: Optional[int] = 1000,
    n_rounds: Optional[int] = 3,
    use_dummy: bool = False,
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
        DataFrame containing passages to measure and debias.
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
        Model used across the measurement and removal stages.
    n_parallels:
        Maximum concurrent API calls.
    measurement_kwargs, removal_kwargs:
        Fine-grained overrides for the measurement and removal tasks.
    remaining_signal:
        When ``True`` (default) measure a remaining-signal prevalence attribute on
        the stripped text and use it in the two-step debiasing regression.
    max_words_per_call, n_rounds:
        Convenience passthroughs for the removal stage. ``max_words_per_call``
        configures the codify task's chunk size, while ``n_rounds`` controls the
        number of completion passes run by codify and any downstream
        paraphrasing steps. Defaults to 3 when not explicitly provided.
    use_dummy:
        If ``True`` run deterministic offline debiasing.
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
        use_dummy=use_dummy,
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
    model: str = "gpt-5-mini",
    json_mode: bool = False,
    web_search: Optional[bool] = None,
    web_search_filters: Optional[Dict[str, Any]] = None,
    search_context_size: str = "medium",
    n_parallels: int = 650,
    use_dummy: bool = False,
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
        Maximum concurrent response requests.
    use_dummy:
        If ``True`` return deterministic dummy responses.
    reset_files:
        When ``True`` regenerate outputs even if files already exist.
    return_original_columns:
        When ``True`` and ``df`` is provided, merge response columns back onto
        the input DataFrame using the prompt identifiers.
    drop_prompts:
        When ``True`` and merging back onto ``df``, drop the prompt column
        before saving/returning the result.
    reasoning_effort, reasoning_summary:
        Optional OpenAI reasoning controls.
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
        :func:`gabriel.utils.openai_utils.get_all_responses`.

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
    if response_fn is not None:
        kwargs.setdefault("response_fn", response_fn)
    if get_all_responses_fn is not None:
        kwargs.setdefault("get_all_responses_fn", get_all_responses_fn)

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
        use_dummy=use_dummy,
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
