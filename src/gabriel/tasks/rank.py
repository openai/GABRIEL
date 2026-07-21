"""
rank.py
~~~~~~~~

This module implements a simplified yet fully featured ranking engine for
evaluating pairs of passages on a set of attributes.  It draws heavy
inspiration from the existing ``elo.py`` implementation found in the
GABRIEL distribution but removes support for the classic Elo rating
system and focuses solely on the Bradley–Terry (BT) style approach.

Key improvements and changes relative to ``elo.py`` include:

* A streamlined configuration dataclass (`RankConfig`) that exposes the
  parameters most relevant to the BT method.  Irrelevant options
  (e.g. ``rating_method``, ``k_factor``) have been removed, and
  parameter names have been harmonised with the high‑level API
  described in the calling code.  ``file_name`` is now treated as a
  stem; if an extension is provided it will be stripped automatically.

* Support for the new rankings prompt (``rankings_prompt.jinja2``)
  which allows the large language model to return one of four
  outcomes for each attribute: ``"circle"``, ``"square"``, ``"draw``
  or ``"insufficient signal"``. A draw contributes one half-win to each
  item. ``insufficient signal`` can use the same equality interpretation
  or be treated as an abstention via ``insufficient_signal_policy``.

* A cleaned up asynchronous ``run`` method that accepts a pandas
  ``DataFrame`` and the name of the column containing the text to be
  ranked.  Each row receives a stable identifier derived from a hash of its
  contents; no external ``id_col`` argument is required.  The method
  produces a DataFrame with one row per input passage and writes results
  under ``save_dir``. Non-recursive runs report component-relative raw
  scores, z-scores, standard errors, and graph labels; recursive runs report
  stage-relative scores, exit stages, and an overall rank.

The Bradley–Terry implementation fits each connected observed comparison
component, regularizes only observed edges when ``learning_rate > 0``, and
reports component-aware scores and model-based uncertainty diagnostics.
Persisted runs include estimator metadata so outcomes are never silently
replayed under different missing-signal or regularization semantics.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import random
import re
import math
import copy
import shutil
import tempfile
import warnings
from numbers import Integral, Real
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd

# Import helper utilities from the gabriel package.  These modules are
# expected to be available in the runtime environment.  Should you wish
# to run this module outside of the GABRIEL distribution, you may need
# to adjust these imports accordingly.
from gabriel.core.prompt_template import PromptTemplate, resolve_template
from gabriel.utils.openai_utils import get_all_responses
from gabriel.utils import (
    safest_json,
    load_image_inputs,
    load_audio_inputs,
    load_pdf_inputs,
    warn_if_modality_mismatch,
)
from gabriel.utils.logging import announce_prompt_rendering
from .rate import Rate, RateConfig
from ._attribute_utils import load_persisted_attributes
from ._run_utils import (
    hash_identifier,
    load_run_metadata,
    run_metadata_path,
    resolve_attribute_batches,
    resolve_identifier_hash_bits,
    update_run_metadata,
    write_task_run_metadata,
)


WeightedOutcome = Tuple[str, str, float]
_RANK_ESTIMATOR_VERSION = 2
_RANK_OWNED_RATE_KEYS = frozenset({"attributes", "save_dir", "file_name"})
_RANK_OWNED_RESPONSE_KEYS = frozenset(
    {
        "prompts",
        "identifiers",
        "prompt_images",
        "prompt_audio",
        "prompt_pdfs",
        "prompt_web_search_filters",
        "model",
        "n_parallels",
        "json_mode",
        "save_path",
        "reset_files",
        "use_dummy",
        "max_retries",
        "reasoning_effort",
    }
)
_RANK_OPERATIONAL_RESPONSE_KEYS = frozenset(
    {
        "api_key",
        "max_output_tokens",
        "estimated_output_tokens_per_prompt",
        "print_example_prompt",
        "skip_tail_fails",
        "ramp_up_seconds",
        "ramp_up_start_fraction",
        "timeout",
        "timeout_factor",
        "max_timeout",
        "dynamic_timeout",
        "timeout_burst_window",
        "timeout_burst_cooldown",
        "timeout_burst_max_restarts",
        "background_mode",
        "background_poll_interval",
        "cancel_existing_batch",
        "use_batch",
        "batch_completion_window",
        "batch_poll_interval",
        "batch_wait_for_completion",
        "max_batch_requests",
        "max_batch_file_bytes",
        "save_every_x_responses",
        "verbose",
        "quiet",
        "global_cooldown",
        "rate_limit_window",
        "connection_error_window",
        "token_sample_size",
        "status_report_interval",
        "planning_rate_limit_buffer",
        "logging_level",
        "service_tier",
        "include",
        "return_raw",
        "request_phase_callback",
    }
)


def _canonical_rank_spec_value(value: Any) -> Any:
    """Return a deterministic, secret-free representation of judge settings."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Rank judgment settings must contain finite numbers")
        return number
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Rank judgment-setting mappings must use string keys")
        return {
            key: _canonical_rank_spec_value(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_rank_spec_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_rank_spec_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ),
        )
    if callable(value):
        target = value if hasattr(value, "__qualname__") else type(value)
        return {
            "callable_module": getattr(target, "__module__", ""),
            "callable_qualname": getattr(target, "__qualname__", ""),
        }
    raise ValueError(
        "Rank cannot fingerprint a non-serializable judgment setting of type "
        f"{type(value).__name__!r}; pass a stable, serializable value instead"
    )


def _sanitized_rank_endpoint(value: Any) -> str:
    """Canonicalize an API endpoint without persisting credentials or queries."""

    parsed = urlsplit(str(value))
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme.lower(), host.lower(), parsed.path, "", ""))


def _rank_batch_state_has_external_work(path: Path) -> bool:
    """Fail closed on malformed state and identify any durable Batch job."""

    try:
        with path.open(encoding="utf-8") as state_file:
            state = json.load(state_file)
    except Exception as exc:
        raise ValueError(
            f"Could not safely inspect Rank Batch API state {str(path)!r}"
        ) from exc
    batches = state.get("batches", []) if isinstance(state, dict) else None
    if not isinstance(batches, list):
        raise ValueError(f"Rank Batch API state {str(path)!r} is malformed")
    return bool(batches) or bool(state.get("batch_id"))


def _validate_rank_attribute_names(
    attributes: Union[Dict[str, Any], List[str]], *, recursive: bool
) -> List[str]:
    """Validate every column namespace generated by effective Rank attributes."""

    if not isinstance(attributes, (dict, list)) or not attributes:
        raise ValueError("attributes must contain at least one attribute")
    attribute_names = (
        list(attributes.keys()) if isinstance(attributes, dict) else list(attributes)
    )
    if any(
        not isinstance(attribute, str) or not attribute.strip()
        for attribute in attribute_names
    ):
        raise ValueError("attribute names must be nonblank strings")
    canonical_attributes = [
        attribute.strip().lower() for attribute in attribute_names
    ]
    if len(canonical_attributes) != len(set(canonical_attributes)):
        raise ValueError(
            "attribute names must be unique after case and whitespace normalization"
        )
    generated_column_owners: Dict[str, str] = {}
    for attribute in attribute_names:
        for generated_column in (
            attribute,
            f"{attribute}_raw",
            f"{attribute}_se",
            f"{attribute}_component",
        ):
            owner = generated_column_owners.get(generated_column)
            if owner is not None and owner != attribute:
                raise ValueError(
                    "Rank attribute output namespaces overlap: "
                    f"{owner!r} and {attribute!r} both generate "
                    f"{generated_column!r}"
                )
            generated_column_owners[generated_column] = attribute
    if recursive:
        reserved_recursive_names = {"identifier", "overall_rank", "exit_stage"}
        invalid_recursive_names = [
            attribute
            for attribute in attribute_names
            if attribute.strip().lower() in reserved_recursive_names
            or re.match(r"^stage\d+_", attribute.strip(), flags=re.IGNORECASE)
        ]
        if invalid_recursive_names:
            raise ValueError(
                "Recursive Rank attribute names cannot use internal output "
                "names or the stage<number>_ prefix: "
                + ", ".join(repr(name) for name in invalid_recursive_names)
            )
    return attribute_names


def _validate_rank_resume_metadata(
    metadata: Dict[str, Any],
    *,
    artifacts_exist: bool,
    reset_files: bool,
    insufficient_signal_policy: str,
    learning_rate: float,
    modality: str,
    measurement_spec_fingerprint: str,
) -> None:
    """Fail closed when persisted judgments use incompatible semantics."""

    if reset_files or not artifacts_exist:
        return

    mismatches: List[str] = []
    saved_version = metadata.get("rank_estimator_version")
    if type(saved_version) is not int or saved_version != _RANK_ESTIMATOR_VERSION:
        mismatches.append(
            "rank_estimator_version "
            f"(saved={saved_version!r}, required={_RANK_ESTIMATOR_VERSION})"
        )

    saved_policy = metadata.get("insufficient_signal_policy")
    if saved_policy != insufficient_signal_policy:
        mismatches.append(
            "insufficient_signal_policy "
            f"(saved={saved_policy!r}, requested={insufficient_signal_policy!r})"
        )

    saved_learning_rate = metadata.get("learning_rate")
    valid_saved_rate = isinstance(saved_learning_rate, (int, float)) and not isinstance(
        saved_learning_rate, bool
    )
    try:
        saved_rate_float = float(saved_learning_rate)
        valid_saved_rate = valid_saved_rate and math.isfinite(saved_rate_float)
    except (TypeError, ValueError, OverflowError):
        saved_rate_float = math.nan
        valid_saved_rate = False
    if not valid_saved_rate or saved_rate_float != float(learning_rate):
        mismatches.append(
            "learning_rate "
            f"(saved={saved_learning_rate!r}, requested={learning_rate!r})"
        )

    last_completed_round = metadata.get("last_completed_round")
    if (
        type(last_completed_round) is not int
        or last_completed_round < -1
    ):
        mismatches.append(
            "last_completed_round "
            f"(saved={last_completed_round!r}, required=an integer >= -1)"
        )

    if not isinstance(metadata.get("input_fingerprints"), dict):
        mismatches.append("input_fingerprints (missing or malformed)")
    saved_modality = metadata.get("modality")
    if saved_modality != modality:
        mismatches.append(
            f"modality (saved={saved_modality!r}, requested={modality!r})"
        )
    saved_measurement_spec = metadata.get("measurement_spec_fingerprint")
    if saved_measurement_spec != measurement_spec_fingerprint:
        mismatches.append("measurement_spec_fingerprint (changed or missing)")

    if mismatches:
        raise ValueError(
            "Existing Rank artifacts are incompatible with the current "
            "Bradley-Terry estimator configuration: "
            + "; ".join(mismatches)
            + ". Use reset_files=True or a new save_dir to recompute them."
        )


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except Exception:
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _is_valid_identifier(value: Any) -> bool:
    """Return whether a user identifier can be persisted unambiguously."""

    return not _is_missing_scalar(value) and bool(str(value).strip())


def _hash_text_identifier(
    value: Any,
    *,
    strict: bool,
    bits: int = 64,
) -> Optional[str]:
    if _is_missing_scalar(value):
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="ignore")
    elif isinstance(value, str):
        text = value
    elif strict:
        return None
    else:
        text = str(value)
    text = text.strip()
    if not text:
        return None
    return hash_identifier(text, bits=bits)


def _rank_input_fingerprints(
    df: pd.DataFrame,
    *,
    id_column: str,
    payload_column: str,
    modality: str,
) -> Dict[str, str]:
    """Fingerprint the effective comparison payload for each stable ID.

    Local media are encoded before hashing so replacing a file in place cannot
    silently reuse judgments about its previous contents. Remote locators are
    hashed as locators because fetching them here would duplicate model I/O.
    """

    fingerprints: Dict[str, str] = {}
    for item_id, payload in zip(df[id_column], df[payload_column]):
        if modality == "image":
            effective_payload: Any = load_image_inputs(payload)
        elif modality == "audio":
            effective_payload = load_audio_inputs(payload)
        elif modality == "pdf":
            effective_payload = load_pdf_inputs(payload)
        else:
            effective_payload = payload
        serialized = json.dumps(
            {"modality": modality, "payload": effective_payload},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=repr,
        )
        fingerprints[str(item_id)] = hash_identifier(serialized, bits=160)
    return fingerprints


def _current_rank_input_fingerprints(
    df: pd.DataFrame,
    *,
    payload_column: str,
    id_column: Optional[str],
    modality: str,
    identifier_hash_bits: int,
) -> Dict[str, str]:
    """Prepare valid IDs and fingerprint current effective payloads."""

    if payload_column not in df.columns:
        raise ValueError(f"column_name '{payload_column}' not found in DataFrame")
    prepared = df.reset_index(drop=True).copy()
    if id_column is not None:
        if id_column not in prepared.columns:
            raise ValueError(f"id_column '{id_column}' not found in DataFrame")
        valid_mask = pd.Series(
            [_is_valid_identifier(value) for value in prepared[id_column]],
            index=prepared.index,
            dtype=bool,
        )
        prepared = prepared.loc[valid_mask].copy().reset_index(drop=True)
        prepared["_id"] = prepared[id_column].astype(str)
    else:
        strict = modality in {"text", "entity", "web"}
        hashed_ids = prepared[payload_column].map(
            lambda value: _hash_text_identifier(
                value,
                strict=strict,
                bits=identifier_hash_bits,
            )
        )
        valid_mask = hashed_ids.notna()
        prepared = prepared.loc[valid_mask].copy().reset_index(drop=True)
        prepared["_id"] = (
            hashed_ids.loc[valid_mask].astype(str).reset_index(drop=True)
        )
    duplicate_ids = prepared.loc[
        prepared["_id"].duplicated(keep=False), "_id"
    ].unique()
    if len(duplicate_ids) > 0:
        preview = ", ".join(repr(value) for value in duplicate_ids[:3])
        raise ValueError(
            "Rank requires a unique identifier for every row; duplicate "
            f"identifier(s): {preview}. Provide a unique id_column when "
            "ranking duplicate content."
        )
    return _rank_input_fingerprints(
        prepared,
        id_column="_id",
        payload_column=payload_column,
        modality=modality,
    )


@dataclass
class RankConfig:
    """User‑visible configuration for :class:`Rank`.

    Only a minimal set of parameters are exposed to keep the API
    straightforward.  Additional hyperparameters for the underlying
    Bradley–Terry model and pairing heuristics are fixed at sensible
    values and should not generally need to be changed.  See the
    surrounding documentation for more details.

    Parameters
    ----------
    attributes:
        Mapping from attribute names to definitions.  A list of
        attribute names is also accepted; definitions will be set to
        empty strings.
    n_rounds:
        Number of rounds of pairwise comparisons to perform.
    matches_per_round:
        Target matches per item per round, capped at the number of possible
        opponents. Realized degrees may exceed the target because deduplicated
        undirected edges can also select an item as another item's opponent.
    power_matching:
        Whether to use an uncertainty-guided pairing heuristic. If ``False``,
        pairs are sampled randomly; random mode can exceed the target when an
        item is chosen by several other items. Non-recursive rankings include
        per‑attribute z‑scores alongside the raw Bradley–Terry estimates
        (``"<attribute>_raw"``) and their standard errors
        (``"<attribute>_se"``).
    learning_rate:
        Total symmetric pseudo‑comparison weight added to each observed
        pair by the BT model. A larger value shrinks pairwise estimates
        more strongly toward a draw. Reported standard errors are asymptotic,
        model-based sandwich standard errors for component-centered raw
        log-skills, computed while treating the realized comparison graph as
        fixed. They exclude shrinkage bias, adaptive-selection uncertainty, and
        are not standard errors for the reported z-scores. Pseudo-comparison
        curvature is present only when ``learning_rate`` is positive. A zero
        value requires Ford's directed strong-connectivity condition within
        every nontrivial observed component for a finite maximum-likelihood fit.
    insufficient_signal_policy:
        How to use an ``"insufficient signal"`` judgment. ``"tie"``
        records one half‑win per item, making the modeling assumption that
        neither item manifests the attribute more strongly. ``"abstain"``
        records no comparison when absence of direct evidence should be
        treated as missingness; informative abstention can still bias the
        retained sample. A true draw uses half-wins in a binary Bradley–Terry
        working objective rather than a separate three-outcome likelihood.
    model:
        Name of the language model to call via ``get_all_responses``.
    n_parallels:
        Number of parallel API calls to issue.
    save_dir:
        Directory into which result files should be saved.
    file_name:
        Stem for the output CSV files.  If an extension is present it
        will be removed.
    additional_instructions:
        Extra, user‑supplied instructions passed to the prompt.
    recursive:
        When ``True`` run ranking in multiple stages, pruning the pool
        of candidates between stages according to ``recursive_fraction``
        and ``recursive_min_remaining``.
    recursive_fraction, recursive_min_remaining,
    recursive_final_round_multiplier:
        Parameters controlling how many items are kept between stages
        and how many rounds are executed in the final stage when
        ``recursive`` is enabled.
    recursive_cut_attr, recursive_cut_side:
        Select which attribute and direction are used when choosing
        which items survive to the next stage.
    recursive_rate_first_round:
        If ``True`` perform a :class:`Rate` sweep before the first
        recursive stage and seed subsequent rounds with those scores.
        This is enabled by default so the initial culling uses model-derived
        single-pass ratings; set to ``False`` to skip.
    recursive_rewrite_func, recursive_rewrite_text_col:
        Optional hook to rewrite surviving passages between stages and
        the column where rewritten text should be stored.
    recursive_keep_stage_columns, recursive_add_stage_suffix:
        Control whether intermediate stage outputs are merged into the
        final results and whether their columns receive stage prefixes.
    initial_rating_pass:
        Enables a one-off :class:`Rate` pass before standard ranking
        rounds.  The centred scores from that pass seed the initial
        Bradley–Terry ratings which helps pairing focus on refinement.
        Enabled by default; set ``initial_rating_pass=False`` if you
        want to start directly with pairwise comparisons.
    rate_kwargs:
        Optional dictionary of overrides forwarded to the rating task
        whenever it is invoked (either as a seed or during recursion).
        Rank owns ``attributes``, ``save_dir``, and ``file_name`` so those
        keys cannot be overridden here.
    primer_scores, primer_scale, primer_center:
        Optional manual primers to seed the Bradley–Terry rating state.
        Scores are centred per attribute when ``primer_center`` is
        ``True`` and scaled by ``primer_scale``.
    judge_version:
        Stable label required when a custom response function or external judge
        is supplied. Change it whenever that judge's hidden prompt, logic, or
        configuration changes so persisted rounds cannot silently mix
        measurement regimes.
    """

    attributes: Union[Dict[str, str], List[str]]
    n_rounds: int = 5
    matches_per_round: int = 5
    power_matching: bool = True
    learning_rate: float = 0.1
    insufficient_signal_policy: str = "tie"
    model: str = "gpt-5.6-luna"
    n_parallels: int = 650
    use_dummy: bool = False
    save_dir: str = os.path.expanduser("~/Documents/runs")
    file_name: str = "rankings"
    additional_instructions: Optional[str] = None
    circle_first: Optional[bool] = None
    modality: str = "text"
    n_attributes_per_run: Optional[int] = None
    reasoning_effort: Optional[str] = None
    # Recursive execution controls
    recursive: bool = False
    recursive_fraction: float = 1.0 / 3.0
    recursive_min_remaining: int = 30
    recursive_final_round_multiplier: int = 3
    recursive_cut_attr: Optional[str] = None
    recursive_cut_side: str = "top"
    recursive_rate_first_round: bool = True
    recursive_rewrite_func: Optional[Callable[[str, str, int], str]] = None
    recursive_rewrite_text_col: str = "text"
    recursive_keep_stage_columns: bool = True
    recursive_add_stage_suffix: bool = True
    # Optional single pass rating seed controls
    initial_rating_pass: bool = True
    rate_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Optional manual primers to seed ratings (applies to both recursive and
    # non-recursive runs). Mapping from identifier -> {attribute: score}.
    # Scores are centred by attribute and scaled by ``primer_scale`` before
    # being injected into the Bradley–Terry state.
    primer_scores: Optional[Dict[str, Dict[str, float]]] = None
    primer_scale: float = 1.0
    primer_center: bool = True
    judge_version: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_rank_attribute_names(self.attributes, recursive=self.recursive)
        if not isinstance(self.rate_kwargs, dict):
            raise ValueError("rate_kwargs must be a dictionary")
        conflicting_rate_keys = sorted(
            _RANK_OWNED_RATE_KEYS & self.rate_kwargs.keys()
        )
        if conflicting_rate_keys:
            raise ValueError(
                "rate_kwargs cannot override Rank-owned Rate setting(s): "
                + ", ".join(conflicting_rate_keys)
            )
        if self.additional_instructions is not None:
            cleaned = str(self.additional_instructions).strip()
            self.additional_instructions = cleaned or None
        if self.insufficient_signal_policy not in {"tie", "abstain"}:
            raise ValueError(
                "insufficient_signal_policy must be either 'tie' or 'abstain'"
            )
        for name in ("n_rounds", "matches_per_round"):
            value = getattr(self, name)
            if (
                not isinstance(value, Integral)
                or isinstance(value, bool)
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be an integer of at least 1")
            setattr(self, name, int(value))
        if not isinstance(self.learning_rate, Real) or isinstance(
            self.learning_rate, bool
        ):
            raise ValueError("learning_rate must be a finite non-negative number")
        self.learning_rate = float(self.learning_rate)
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("learning_rate must be a finite non-negative number")
        if not isinstance(self.primer_scale, Real) or isinstance(
            self.primer_scale, bool
        ):
            raise ValueError("primer_scale must be a finite number")
        self.primer_scale = float(self.primer_scale)
        if not math.isfinite(self.primer_scale):
            raise ValueError("primer_scale must be a finite number")
        if type(self.primer_center) is not bool:
            raise ValueError("primer_center must be a boolean")
        if self.judge_version is not None:
            if not isinstance(self.judge_version, str) or not self.judge_version.strip():
                raise ValueError("judge_version must be a nonblank string or None")
            self.judge_version = self.judge_version.strip()
        if not isinstance(self.recursive_fraction, Real) or isinstance(
            self.recursive_fraction, bool
        ):
            raise ValueError("recursive_fraction must be between 0 and 1")
        self.recursive_fraction = float(self.recursive_fraction)
        if (
            not math.isfinite(self.recursive_fraction)
            or self.recursive_fraction <= 0
            or self.recursive_fraction >= 1
        ):
            raise ValueError("recursive_fraction must be between 0 and 1")
        for name in (
            "recursive_min_remaining",
            "recursive_final_round_multiplier",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, Integral)
                or isinstance(value, bool)
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be an integer of at least 1")
            setattr(self, name, int(value))


class Rank:
    """Rank items by comparing passages pairwise on multiple attributes.

    An instance of :class:`Ranker` orchestrates the iterative process
    of sampling pairs, calling a language model to adjudicate which
    passage better exhibits each attribute, and then fitting a
    Bradley–Terry model to those outcomes.  Standard errors and
        component-aware z‑scores are computed for every attribute in
        non-recursive mode. A component with no score dispersion receives the
        neutral z-score fallback of zero; a singleton remains unranked. Results
        are persisted to disk after the final round.
    """

    def __init__(
        self,
        cfg: RankConfig,
        template: Optional[PromptTemplate] = None,
        template_path: Optional[str] = None,
    ) -> None:
        """Instantiate a ranking engine.

        Parameters
        ----------
        cfg:
            User‑provided configuration.
        template:
            Optional :class:`gabriel.core.prompt_template.PromptTemplate` to
            render the comparison prompts.  If not supplied, the built‑in
            ``rankings_prompt.jinja2`` template is used.
        template_path:
            Path to a custom prompt template on disk. The template is
            validated to ensure it expects the same variables as the
            built‑in template.
        """
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        self.template = resolve_template(
            template=template,
            template_path=template_path,
            reference_filename="rankings_prompt.jinja2",
        )
        # random state; a seed is intentionally omitted from the public
        # configuration to discourage brittle behaviour.  If
        # reproducibility is required, modify this line to pass a
        # specific seed.
        self.rng = random.Random()
        # place holders for multiway rankings and aggregated standard errors
        self.history_multi: Dict[str, List[List[str]]] = {}
        self._last_se_agg: Optional[Dict[str, float]] = None
        self._warned_disconnected_graph = False

        # internal constants for the pairing and BT algorithms.  These
        # values are deliberately not exposed through the public API as
        # they seldom need tuning and adjusting them can complicate
        # reproducibility.  Should you need to experiment with these
        # hyperparameters, modify the values below.
        self._EXPLORE_FRAC = 0.2  # fraction of random pairings per round
        self._CANDIDATE_NEIGHBORS = 20  # neighbourhood size for info gain pairing
        self._HIGH_SE_FRAC = 0.25  # fraction of high‑uncertainty items
        self._MAX_ITER = 1000  # maximum iterations for BT optimisation
        self._TOL = 1e-6  # convergence tolerance for BT
        # Relative eigenvalue tolerance for the Fisher-information pseudoinverse.
        # This is a numerical cutoff only; it must not add synthetic information.
        self._SE_EIGEN_TOL = 1e-12
        # The maximum number of candidate pairs to consider per pairing round.
        # When the number of items becomes very large (e.g. tens of thousands),
        # evaluating all possible pairs is intractable.  We therefore cap the
        # total number of candidate pairs by limiting the neighbourhood size
        # used when constructing candidate pairs.  The default of 200k ensures
        # that information gain pairing remains tractable even with very
        # large data sets: for example, with 10 000 items and a cap of
        # 200 000, each item will only consider approximately 20 neighbours.
        self._MAX_CANDIDATE_PAIRS_PER_ROUND = 200_000

    def _measurement_spec_fingerprint(
        self, runtime_kwargs: Optional[Mapping[str, Any]] = None
    ) -> str:
        """Hash judgment settings that must not change within a tournament."""

        effective_runtime: Dict[str, Any] = {
            "web_search": self.cfg.modality == "web",
            "image_detail": None,
            "n": 1,
            "temperature": 0.9,
            "expected_schema": None,
            "tools": None,
            "tool_choice": None,
            "reasoning_summary": None,
            "dummy_responses": None,
            "response_fn": None,
            "get_all_responses_fn": None,
        }
        for key, value in dict(runtime_kwargs or {}).items():
            if key in _RANK_OPERATIONAL_RESPONSE_KEYS:
                continue
            effective_runtime[key] = value
        effective_runtime["web_search"] = bool(
            effective_runtime.get("web_search")
        )
        if effective_runtime.get("image_detail") in {None, "none"}:
            effective_runtime["image_detail"] = None
        if effective_runtime.get("web_search"):
            effective_runtime.setdefault("web_search_filters", None)
            context_size = effective_runtime.get("search_context_size", "medium")
            effective_runtime["search_context_size"] = {
                "small": "low",
                "large": "high",
            }.get(context_size, context_size)
        else:
            effective_runtime.pop("web_search_filters", None)
            effective_runtime.pop("search_context_size", None)
        endpoint = effective_runtime.get("base_url") or os.getenv("OPENAI_BASE_URL")
        if endpoint:
            effective_runtime["base_url"] = _sanitized_rank_endpoint(endpoint)
        else:
            effective_runtime.pop("base_url", None)

        semantic_rate_kwargs = {
            key: value
            for key, value in self.cfg.rate_kwargs.items()
            if key not in _RANK_OPERATIONAL_RESPONSE_KEYS
            and key != "n_parallels"
        }
        payload = _canonical_rank_spec_value(
            {
                "model": self.cfg.model,
                "template": self.template.text,
                "attributes": self.cfg.attributes,
                "recursive": self.cfg.recursive,
                "recursive_rate_first_round": self.cfg.recursive_rate_first_round,
                "recursive_rewrite_func": self.cfg.recursive_rewrite_func,
                "initial_rating_pass": self.cfg.initial_rating_pass,
                "rate_kwargs": semantic_rate_kwargs,
                "additional_instructions": self.cfg.additional_instructions,
                "circle_first": self.cfg.circle_first,
                "reasoning_effort": self.cfg.reasoning_effort,
                "use_dummy": self.cfg.use_dummy,
                "judge_version": self.cfg.judge_version,
                "runtime_judgment": effective_runtime,
            }
        )
        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hash_identifier(serialized, bits=160)

    # ------------------------------------------------------------------
    def _apply_primer(
        self,
        ratings: Dict[str, Dict[str, float]],
        primer: Optional[Dict[str, Dict[str, float]]],
        attr_keys: List[str],
    ) -> None:
        """Inject user-provided primer scores into the rating state.

        Primers are centred per-attribute if ``primer_center`` is True and
        scaled by ``primer_scale``. Missing attributes are ignored.
        """

        if not primer:
            return

        # normalise per attribute
        attr_to_vals: Dict[str, List[float]] = {a: [] for a in attr_keys}
        for ident, amap in primer.items():
            if ident not in ratings:
                continue
            for attr in attr_keys:
                if attr in amap and amap[attr] is not None:
                    try:
                        value = float(amap[attr])
                        if math.isfinite(value):
                            attr_to_vals[attr].append(value)
                    except Exception:
                        continue

        attr_offset: Dict[str, float] = {a: 0.0 for a in attr_keys}
        if self.cfg.primer_center:
            for attr, vals in attr_to_vals.items():
                if vals:
                    attr_offset[attr] = float(np.mean(vals))

        scale = self.cfg.primer_scale
        for ident, amap in primer.items():
            if ident not in ratings:
                continue
            for attr in attr_keys:
                if attr not in amap or amap[attr] is None:
                    continue
                try:
                    value = float(amap[attr])
                    if math.isfinite(value):
                        ratings[ident][attr] = (
                            value - attr_offset[attr]
                        ) * scale
                except Exception:
                    continue

        # ------------------------------------------------------------------
        # Public API for adding multiway rankings
        # ------------------------------------------------------------------
    def add_multiway_ranking(self, attr: str, ranking: List[str]) -> None:
        """Record a multiway ranking for a given attribute.

        Multiway rankings are stored but not used by the current BT
        implementation.  They are retained for potential future
        extensions where a Plackett–Luce model could be incorporated.
        """
        if attr not in self.history_multi:
            self.history_multi[attr] = []
        self.history_multi[attr].append(ranking)

    def _attributes_as_dict(self) -> Dict[str, str]:
        if isinstance(self.cfg.attributes, dict):
            return dict(self.cfg.attributes)
        return {attr: "" for attr in self.cfg.attributes}

    def _split_rate_kwargs(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        merged: Dict[str, Any] = {}
        if self.cfg.rate_kwargs:
            merged.update(self.cfg.rate_kwargs)
        if overrides:
            merged.update(overrides)
        conflicting_rate_keys = sorted(_RANK_OWNED_RATE_KEYS & merged.keys())
        if conflicting_rate_keys:
            raise ValueError(
                "rate_kwargs cannot override Rank-owned Rate setting(s): "
                + ", ".join(conflicting_rate_keys)
            )
        config_fields = {f.name for f in fields(RateConfig)}
        cfg_kwargs: Dict[str, Any] = {}
        run_kwargs: Dict[str, Any] = {}
        for key, value in merged.items():
            if key in config_fields:
                cfg_kwargs[key] = value
            else:
                run_kwargs[key] = value
        return cfg_kwargs, run_kwargs

    async def _run_rate_pass(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        save_dir: str,
        file_name: str,
        reset_files: bool,
        rate_kwargs: Optional[Dict[str, Any]] = None,
        runtime_kwargs: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        cfg_overrides, run_kwargs = self._split_rate_kwargs(rate_kwargs)
        rate_cfg = RateConfig(
            attributes=self._attributes_as_dict(),
            save_dir=save_dir,
            file_name=file_name,
            model=self.cfg.model,
            n_parallels=self.cfg.n_parallels,
            n_runs=1,
            use_dummy=self.cfg.use_dummy,
            additional_instructions=self.cfg.additional_instructions or "",
            modality=self.cfg.modality,
            n_attributes_per_run=self.cfg.n_attributes_per_run,
            reasoning_effort=self.cfg.reasoning_effort,
        )
        for key, value in cfg_overrides.items():
            setattr(rate_cfg, key, value)
        combined_kwargs = dict(run_kwargs)
        if runtime_kwargs:
            combined_kwargs.update(runtime_kwargs)
        combined_kwargs.setdefault("web_search", self.cfg.modality == "web")
        rate_task = Rate(rate_cfg)
        return await rate_task.run(
            df,
            column_name,
            reset_files=reset_files,
            **combined_kwargs,
        )

    def _seed_ratings_from_rate(
        self,
        rate_df: pd.DataFrame,
        *,
        id_column: Optional[str],
        text_column: str,
        item_ids: Sequence[str],
        attr_keys: Sequence[str],
        identifier_hash_bits: int,
    ) -> Dict[str, Dict[str, float]]:
        if rate_df.empty:
            return {}
        attr_cols = [attr for attr in attr_keys if attr in rate_df.columns]
        if not attr_cols:
            return {}
        if id_column and id_column in rate_df.columns:
            key_series = rate_df[id_column].astype(str)
        elif text_column in rate_df.columns:
            strict_payload = self.cfg.modality in {"text", "entity", "web"}
            key_series = rate_df[text_column].map(
                lambda x: _hash_text_identifier(
                    x,
                    strict=strict_payload,
                    bits=identifier_hash_bits,
                )
            )
            key_series = key_series.dropna()
        else:
            return {}
        stage_df = pd.DataFrame({"_id": key_series})
        for attr in attr_cols:
            stage_df[attr] = pd.to_numeric(rate_df[attr], errors="coerce")
        grouped = stage_df.groupby("_id")[attr_cols].mean()
        seeds: Dict[str, Dict[str, float]] = {}
        for attr in attr_cols:
            series = grouped[attr].dropna()
            if series.empty:
                continue
            mean_val = float(series.mean())
            centred = series - mean_val
            for item_id, value in centred.items():
                seeds.setdefault(item_id, {})[attr] = float(value)
        # Only retain seeds for items that will appear in the ranking loop
        return {item_id: seeds[item_id] for item_id in item_ids if item_id in seeds}

    # ------------------------------------------------------------------
    # BT / PL fitting utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _matches_outcome_label(value: str, aliases: Sequence[str]) -> bool:
        """Match only explicit, complete labels from the response contract."""

        return value in aliases

    def _decode_pairwise_outcome(
        self,
        id_a: str,
        id_b: str,
        raw_value: Any,
    ) -> Tuple[str, List[WeightedOutcome]]:
        """Decode one model judgment into evidence for the BT model.

        The prompt distinguishes a true draw from a lack of direct signal.
        Keeping those labels separate here lets callers choose whether the
        latter is an equality judgment or an abstention. Fractional wins make
        every non-abstaining response contribute exactly one comparison.
        """

        value = raw_value
        if isinstance(value, dict):
            winner_key = next(
                (key for key in value if str(key).strip().lower() == "winner"),
                None,
            )
            value = value.get(winner_key) if winner_key is not None else None
        if not isinstance(value, str):
            return "invalid", []

        label = " ".join(value.strip().lower().split())
        if self._matches_outcome_label(label, ("insufficient signal", "insufficient")):
            if self.cfg.insufficient_signal_policy == "abstain":
                return "insufficient_signal", []
            return "insufficient_signal", [
                (id_a, id_b, 0.5),
                (id_b, id_a, 0.5),
            ]
        if self._matches_outcome_label(label, ("draw", "tie", "equal")):
            return "draw", [(id_a, id_b, 0.5), (id_b, id_a, 0.5)]
        if self._matches_outcome_label(
            label, ("circle", "circle wins", "text a", "text a wins")
        ):
            return "circle", [(id_a, id_b, 1.0)]
        if self._matches_outcome_label(
            label, ("square", "square wins", "text b", "text b wins")
        ):
            return "square", [(id_b, id_a, 1.0)]
        return "invalid", []

    def _record_pairwise_outcome(
        self,
        history: List[WeightedOutcome],
        id_a: str,
        id_b: str,
        raw_value: Any,
    ) -> str:
        """Append a decoded response to ``history`` and return its category."""

        category, outcomes = self._decode_pairwise_outcome(id_a, id_b, raw_value)
        history.extend(outcomes)
        return category

    @staticmethod
    def _successful_response_mask(frame: pd.DataFrame) -> pd.Series:
        """Identify rows with both a successful request and a usable response."""

        if "Response" not in frame.columns:
            return pd.Series(False, index=frame.index, dtype=bool)
        responses = frame["Response"]
        response_mask = responses.notna() & responses.astype(str).str.strip().ne("")
        if "Successful" not in frame.columns:
            return response_mask.astype(bool)

        success_raw = frame["Successful"]
        success_mask = pd.Series(False, index=frame.index, dtype=bool)
        try:
            success_mask |= success_raw.astype("boolean").fillna(False)
        except (TypeError, ValueError):
            pass
        success_mask |= (
            success_raw.astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "y", "completed", "succeeded", "success"})
        )
        return (response_mask & success_mask).astype(bool)

    @classmethod
    def _validate_requested_responses(
        cls,
        frame: pd.DataFrame,
        requested_ids: Sequence[str],
        *,
        context: str,
        allow_extra: bool,
    ) -> pd.Series:
        """Require one successful, nonblank response for every requested ID."""

        required_columns = {"Identifier", "Response"}
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{context} response table is missing columns: "
                + ", ".join(sorted(missing_columns))
            )
        identifiers = frame["Identifier"].astype(str)
        requested_set = set(requested_ids)
        requested_mask = identifiers.isin(requested_set)
        returned_requested = identifiers.loc[requested_mask]
        if (
            returned_requested.duplicated().any()
            or set(returned_requested) != requested_set
            or (not allow_extra and int(requested_mask.sum()) != len(frame))
        ):
            raise ValueError(
                f"{context} did not return exactly one row for every requested "
                "judgment"
            )
        successful = cls._successful_response_mask(frame)
        failed_requested = requested_mask & ~successful
        if failed_requested.any():
            failed_preview = ", ".join(
                repr(value)
                for value in identifiers.loc[failed_requested].head(3)
            )
            raise ValueError(
                f"{context} returned failed or blank judgment response(s): "
                f"{failed_preview}. The round was not committed and can be retried."
            )
        return requested_mask

    @staticmethod
    async def _coerce_response_dict(raw: Any) -> Dict[str, Any]:
        """Decode the response container forms accepted by the task API."""

        obj = await safest_json(raw)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            nested = await safest_json(obj)
            if isinstance(nested, dict):
                return nested
        if isinstance(obj, list) and obj:
            nested = await safest_json(obj[0])
            if isinstance(nested, dict):
                return nested
        return {}

    async def _validate_pairwise_response_payloads(
        self,
        frame: pd.DataFrame,
        meta_map: Dict[str, Tuple[int, int, str, str]],
        attr_batches: Sequence[Sequence[str]],
        *,
        context: str,
        retry_path: Optional[str] = None,
    ) -> None:
        """Require complete, recognized outcomes before committing a round."""

        invalid: Dict[str, str] = {}
        for identifier, raw_response in zip(
            frame["Identifier"].astype(str), frame["Response"]
        ):
            meta = meta_map.get(identifier)
            if meta is None:
                invalid[identifier] = "missing prompt metadata"
                continue
            batch_idx, _, id_a, id_b = meta
            if batch_idx < 0 or batch_idx >= len(attr_batches):
                invalid[identifier] = "invalid attribute batch"
                continue
            response_obj = await self._coerce_response_dict(raw_response)
            expected = {
                str(attribute).strip().lower(): str(attribute)
                for attribute in attr_batches[batch_idx]
            }
            normalized: Dict[str, Any] = {}
            duplicate_key = False
            for raw_key, value in response_obj.items():
                key = str(raw_key).strip().lower()
                if key in normalized:
                    duplicate_key = True
                normalized[key] = value
            if duplicate_key or set(normalized) != set(expected):
                invalid[identifier] = (
                    "response must contain exactly the requested attributes"
                )
                continue
            for key, value in normalized.items():
                category, _ = self._decode_pairwise_outcome(id_a, id_b, value)
                if category == "invalid":
                    invalid[identifier] = (
                        f"unrecognized outcome for {expected[key]!r}"
                    )
                    break

        if not invalid:
            return
        if retry_path is not None:
            retry_frame = frame.copy()
            if "Successful" not in retry_frame.columns:
                retry_frame["Successful"] = True
            retry_mask = retry_frame["Identifier"].astype(str).isin(invalid)
            retry_frame.loc[retry_mask, "Successful"] = False
            if "Error Log" not in retry_frame.columns:
                retry_frame["Error Log"] = ""
            retry_frame.loc[retry_mask, "Error Log"] = retry_frame.loc[
                retry_mask, "Identifier"
            ].astype(str).map(invalid)
            self._write_dataframe_atomically(retry_frame, retry_path)
        preview = "; ".join(
            f"{identifier!r}: {reason}"
            for identifier, reason in list(invalid.items())[:3]
        )
        raise ValueError(
            f"{context} returned semantically invalid judgment payload(s): "
            f"{preview}. The round was not committed and can be retried."
        )

    @staticmethod
    def _read_rank_checkpoint(path: str, n_attribute_batches: int) -> pd.DataFrame:
        """Read and validate one committed v2 round without coercing IDs."""

        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception as exc:
            raise ValueError(f"Could not read Rank checkpoint {path!r}") from exc

        required_columns = {
            "Identifier",
            "Response",
            "Batch",
            "Pair",
            "IdA",
            "IdB",
        }
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"Rank checkpoint {path!r} is missing structured columns: "
                + ", ".join(sorted(missing_columns))
            )
        if frame.empty:
            raise ValueError(f"Rank checkpoint {path!r} contains no judgments")
        if not Rank._successful_response_mask(frame).all():
            raise ValueError(
                f"Rank checkpoint {path!r} contains failed or blank responses"
            )
        # Checkpoints written before the response collector exposed an explicit
        # success flag are valid when every response is nonblank.  Normalize
        # them before a staged catch-up is concatenated; otherwise pandas adds
        # ``Successful`` only for the new rows and the legacy rows become NaN,
        # causing the atomic checkpoint validator to reject the merge.
        if "Successful" not in frame.columns:
            frame["Successful"] = True
        identifiers = frame["Identifier"].str.strip()
        if identifiers.eq("").any() or identifiers.duplicated().any():
            raise ValueError(
                f"Rank checkpoint {path!r} has blank or duplicate identifiers"
            )
        if frame["IdA"].str.strip().eq("").any() or frame[
            "IdB"
        ].str.strip().eq("").any():
            raise ValueError(f"Rank checkpoint {path!r} has blank item IDs")

        batch_numbers = pd.to_numeric(frame["Batch"], errors="coerce")
        pair_numbers = pd.to_numeric(frame["Pair"], errors="coerce")
        valid_batches = (
            batch_numbers.notna()
            & np.isfinite(batch_numbers)
            & (batch_numbers >= 0)
            & (batch_numbers < n_attribute_batches)
            & (batch_numbers == np.floor(batch_numbers))
        )
        valid_pairs = (
            pair_numbers.notna()
            & np.isfinite(pair_numbers)
            & (pair_numbers >= 0)
            & (pair_numbers == np.floor(pair_numbers))
        )
        if not valid_batches.all() or not valid_pairs.all():
            raise ValueError(
                f"Rank checkpoint {path!r} has invalid Batch or Pair values"
            )
        frame["Batch"] = batch_numbers.astype(int)
        frame["Pair"] = pair_numbers.astype(int)
        return frame

    @classmethod
    def _write_rank_checkpoint(
        cls,
        frame: pd.DataFrame,
        path: str,
        n_attribute_batches: int,
    ) -> None:
        """Validate and atomically replace one structured round checkpoint."""

        save_dir = os.path.dirname(path) or "."
        os.makedirs(save_dir, exist_ok=True)
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{Path(path).name}.",
            suffix=".tmp",
            dir=save_dir,
            text=True,
        )
        os.close(file_descriptor)
        try:
            frame.to_csv(temporary_path, index=False)
            cls._read_rank_checkpoint(temporary_path, n_attribute_batches)
            os.replace(temporary_path, path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _write_dataframe_atomically(frame: pd.DataFrame, path: str) -> None:
        """Durably replace a CSV staging checkpoint."""

        save_dir = os.path.dirname(path) or "."
        os.makedirs(save_dir, exist_ok=True)
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{Path(path).name}.",
            suffix=".tmp",
            dir=save_dir,
            text=True,
        )
        try:
            with os.fdopen(
                file_descriptor, "w", encoding="utf-8", newline=""
            ) as temporary_file:
                frame.to_csv(temporary_file, index=False)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _write_json_atomically(payload: Dict[str, Any], path: str) -> None:
        """Durably replace a JSON sidecar without exposing a partial file."""

        save_dir = os.path.dirname(path) or "."
        os.makedirs(save_dir, exist_ok=True)
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{Path(path).name}.",
            suffix=".tmp",
            dir=save_dir,
            text=True,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _comparison_components(n_ij: np.ndarray) -> List[np.ndarray]:
        """Return deterministic components of the observed comparison graph."""

        n = n_ij.shape[0]
        if n_ij.shape != (n, n):
            raise ValueError("n_ij must be a square matrix")
        visited = np.zeros(n, dtype=bool)
        components: List[np.ndarray] = []
        for start in range(n):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            component: List[int] = []
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbour in np.flatnonzero(n_ij[node] > 0):
                    neighbour_int = int(neighbour)
                    if not visited[neighbour_int]:
                        visited[neighbour_int] = True
                        stack.append(neighbour_int)
            components.append(np.array(sorted(component), dtype=int))
        return components

    @classmethod
    def _comparison_component_labels(cls, n_ij: np.ndarray) -> np.ndarray:
        """Map every item to its observed comparison-graph component."""

        labels = np.empty(n_ij.shape[0], dtype=int)
        for label, component in enumerate(cls._comparison_components(n_ij)):
            labels[component] = label
        return labels

    @staticmethod
    def _component_zscores(
        values: np.ndarray, component_labels: np.ndarray
    ) -> np.ndarray:
        """Standardize scores within identifiable comparison components."""

        if values.shape != component_labels.shape:
            raise ValueError("values and component_labels must have the same shape")
        zscores = np.full(len(values), np.nan, dtype=float)
        for component in np.unique(component_labels):
            members = component_labels == component
            local_values = values[members]
            if int(np.sum(members)) == 1:
                continue
            local_std = float(local_values.std(ddof=0))
            if local_std > 0:
                zscores[members] = (local_values - local_values.mean()) / local_std
            else:
                zscores[members] = 0.0
        return zscores

    @staticmethod
    def _is_strongly_connected(
        observed_wins: np.ndarray, component: np.ndarray
    ) -> bool:
        """Check Ford's directed-win condition within one component."""

        if len(component) < 2:
            return False
        local = observed_wins[np.ix_(component, component)] > 0

        def reaches_all(adjacency: np.ndarray) -> bool:
            visited = np.zeros(len(component), dtype=bool)
            stack = [0]
            visited[0] = True
            while stack:
                node = stack.pop()
                for neighbour in np.flatnonzero(adjacency[node]):
                    neighbour_int = int(neighbour)
                    if not visited[neighbour_int]:
                        visited[neighbour_int] = True
                        stack.append(neighbour_int)
            return bool(np.all(visited))

        return reaches_all(local) and reaches_all(local.T)

    @staticmethod
    def _build_bt_win_matrices(
        item_ids: Sequence[str],
        outcomes: Sequence[Union[Tuple[str, str], WeightedOutcome]],
        pseudo: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build separate observed and regularized directed-win matrices.

        ``pseudo`` is the total symmetric pseudo-comparison mass for each
        observed undirected edge. Half is assigned as a win in each direction.
        Unobserved edges and the diagonal receive no artificial comparisons.
        """

        if len(set(item_ids)) != len(item_ids):
            raise ValueError("item_ids must contain unique values")
        if not np.isfinite(pseudo) or pseudo < 0:
            raise ValueError("pseudo must be a finite non-negative number")

        n = len(item_ids)
        idx = {item: i for i, item in enumerate(item_ids)}
        observed_wins = np.zeros((n, n), dtype=float)
        for outcome in outcomes:
            if len(outcome) == 2:
                winner, loser = outcome
                weight = 1.0
            elif len(outcome) == 3:
                winner, loser, raw_weight = outcome
                try:
                    weight = float(raw_weight)
                except (TypeError, ValueError) as exc:
                    raise ValueError("outcome weights must be numeric") from exc
            else:
                raise ValueError(
                    "outcomes must be (winner, loser) or (winner, loser, weight)"
                )
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("outcome weights must be finite and non-negative")
            if weight == 0:
                continue
            if winner not in idx or loser not in idx:
                raise ValueError(
                    "outcome identifiers must be present in item_ids; "
                    f"received ({winner!r}, {loser!r})"
                )
            if winner == loser:
                continue
            observed_wins[idx[winner], idx[loser]] += weight

        np.fill_diagonal(observed_wins, 0.0)
        observed_matches = observed_wins + observed_wins.T
        edge_mask = observed_matches > 0
        fit_wins = observed_wins + 0.5 * pseudo * edge_mask
        np.fill_diagonal(fit_wins, 0.0)
        return observed_wins, fit_wins

    @staticmethod
    def _bt_logit_spanning_tree_start(
        local_fit_wins: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Construct a scale-stable empirical-logit start for a BT component.

        Each bidirectionally observed edge supplies the empirical equation
        ``s_i - s_j = log(w_ij) - log(w_ji)``.  Integrating those equations
        along a maximum-curvature spanning tree avoids squaring the condition
        number in normal equations. On a tree this is the exact regularized
        optimum, even when pseudo mass is 1e-300 or a long score chain lies far
        outside the range in which abilities can be exponentiated.
        """

        size = int(local_fit_wins.shape[0])
        edge_data: List[Tuple[int, int, float, float]] = []
        for left in range(size):
            for right in range(left + 1, size):
                left_wins = float(local_fit_wins[left, right])
                right_wins = float(local_fit_wins[right, left])
                if left_wins <= 0 or right_wins <= 0:
                    continue
                high = max(left_wins, right_wins)
                low = min(left_wins, right_wins)
                weight = low / (1.0 + low / high)
                log_odds = math.log(left_wins) - math.log(right_wins)
                if not np.isfinite(weight) or not np.isfinite(log_odds):
                    return None
                edge_data.append((left, right, weight, log_odds))
        if not edge_data:
            return None

        parent = list(range(size))

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        adjacency: List[List[Tuple[int, float]]] = [
            [] for _ in range(size)
        ]
        tree_edges = 0
        for left, right, _weight, log_odds in sorted(
            edge_data, key=lambda edge: edge[2], reverse=True
        ):
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                continue
            parent[right_root] = left_root
            adjacency[left].append((right, -log_odds))
            adjacency[right].append((left, log_odds))
            tree_edges += 1
            if tree_edges == size - 1:
                break
        if tree_edges != size - 1:
            return None

        scores = np.full(size, np.nan, dtype=float)
        scores[0] = 0.0
        stack = [0]
        while stack:
            node = stack.pop()
            for neighbor, offset in adjacency[node]:
                if np.isfinite(scores[neighbor]):
                    continue
                scores[neighbor] = scores[node] + offset
                stack.append(neighbor)
        scores -= scores.mean()
        if np.any(~np.isfinite(scores)):
            return None
        return scores

    @staticmethod
    def _bt_strong_components(local_observed_wins: np.ndarray) -> List[np.ndarray]:
        """Return deterministic SCCs of an observed directed-win graph.

        Tiny edge regularization can put SCC-wide location contrasts hundreds
        of orders of magnitude below the within-SCC curvature.  Keeping the
        observed SCCs explicit lets the numerical fallback optimize those weak
        contrasts without subtracting the O(1) information internal to an SCC.
        """

        size = int(local_observed_wins.shape[0])
        adjacency = local_observed_wins > 0

        def reachable(start: int, graph: np.ndarray) -> Set[int]:
            visited = {start}
            stack = [start]
            while stack:
                node = stack.pop()
                for neighbor in np.flatnonzero(graph[node]):
                    neighbor_int = int(neighbor)
                    if neighbor_int not in visited:
                        visited.add(neighbor_int)
                        stack.append(neighbor_int)
            return visited

        remaining = set(range(size))
        components: List[np.ndarray] = []
        while remaining:
            start = min(remaining)
            component = reachable(start, adjacency) & reachable(start, adjacency.T)
            ordered = np.array(sorted(component), dtype=int)
            components.append(ordered)
            remaining.difference_update(component)
        return components

    @classmethod
    def _bt_scc_coordinate_directions(
        cls,
        local_observed_wins: np.ndarray,
        include_within_scc: bool = True,
    ) -> List[np.ndarray]:
        """Build a basis of SCC-wide and within-SCC score contrasts."""

        size = int(local_observed_wins.shape[0])
        components = cls._bt_strong_components(local_observed_wins)
        directions: List[np.ndarray] = []
        if len(components) > 1:
            reference = components[-1]
            for component in components[:-1]:
                direction = np.zeros(size, dtype=float)
                direction[component] = 1.0 / len(component)
                direction[reference] = -1.0 / len(reference)
                directions.append(direction)
        if include_within_scc:
            for component in components:
                reference = int(component[-1])
                for member in component[:-1]:
                    direction = np.zeros(size, dtype=float)
                    direction[int(member)] = 1.0
                    direction[reference] = -1.0
                    directions.append(direction)
        expected_count = size - 1 if include_within_scc else len(components) - 1
        if len(directions) != expected_count:
            raise RuntimeError("Could not construct Bradley-Terry SCC coordinates")
        return directions

    @staticmethod
    def _bt_stable_edge_gradient(
        scores: np.ndarray,
        local_fit_wins: np.ndarray,
    ) -> np.ndarray:
        """Evaluate pairwise BT score residuals without tail cancellation."""

        differences = scores[:, None] - scores[None, :]
        tail_probability = np.exp(-np.abs(differences))
        winner_probability = np.where(
            differences >= 0,
            1.0 / (1.0 + tail_probability),
            tail_probability / (1.0 + tail_probability),
        )
        loser_probability = np.where(
            differences >= 0,
            tail_probability / (1.0 + tail_probability),
            1.0 / (1.0 + tail_probability),
        )
        edge_gradient = (
            -local_fit_wins * loser_probability
            + local_fit_wins.T * winner_probability
        )
        # Enforce pairwise cancellation exactly enough that summing over an SCC
        # removes its O(1) internal score terms.  Otherwise roundoff from a
        # balanced internal edge can overwhelm a 1e-300 cross-SCC residual.
        return 0.5 * (edge_gradient - edge_gradient.T)

    @classmethod
    def _bt_stable_gradient(
        cls,
        scores: np.ndarray,
        local_fit_wins: np.ndarray,
    ) -> np.ndarray:
        """Evaluate BT score equations without saturated-probability loss."""

        return cls._bt_stable_edge_gradient(scores, local_fit_wins).sum(axis=1)

    @classmethod
    def _bt_stable_directional_gradient(
        cls,
        scores: np.ndarray,
        local_fit_wins: np.ndarray,
        direction: np.ndarray,
    ) -> float:
        """Contract pairwise residuals before summing over score directions."""

        edge_gradient = cls._bt_stable_edge_gradient(scores, local_fit_wins)
        direction_differences = direction[:, None] - direction[None, :]
        return float(0.5 * np.sum(edge_gradient * direction_differences))

    @classmethod
    def _bt_scc_solution_is_certified(
        cls,
        scores: np.ndarray,
        local_observed_wins: np.ndarray,
        local_fit_wins: np.ndarray,
        tol: float,
        include_within_scc: bool = True,
    ) -> bool:
        """Check that every SCC-coordinate optimum is within score tolerance."""

        convergence_tol = max(float(tol), 1e-10)
        for direction in cls._bt_scc_coordinate_directions(
            local_observed_wins,
            include_within_scc=include_within_scc,
        ):
            radius = convergence_tol / float(np.max(np.abs(direction)))
            lower_derivative = cls._bt_stable_directional_gradient(
                scores - radius * direction,
                local_fit_wins,
                direction,
            )
            upper_derivative = cls._bt_stable_directional_gradient(
                scores + radius * direction,
                local_fit_wins,
                direction,
            )
            if (
                not np.isfinite(lower_derivative)
                or not np.isfinite(upper_derivative)
                or lower_derivative > 0
                or upper_derivative < 0
            ):
                return False
        return True

    @classmethod
    def _refine_bt_scc_offsets_with_newton(
        cls,
        initial_scores: np.ndarray,
        local_observed_wins: np.ndarray,
        local_fit_wins: np.ndarray,
        tol: float,
        max_iter: int = 2000,
    ) -> np.ndarray:
        """Refine only SCC-wide offsets on the condensation graph.

        Internal SCC comparisons are constant under these offsets and are
        removed before forming the objective and Hessian.  The remaining
        Newton system therefore retains regularized cross-SCC curvature even
        when it is hundreds of orders of magnitude below internal curvature.
        """

        scores = np.asarray(initial_scores, dtype=float).copy()
        scores -= scores.mean()
        components = cls._bt_strong_components(local_observed_wins)
        if len(components) <= 1:
            return scores

        labels = np.empty(len(scores), dtype=int)
        for component_index, component in enumerate(components):
            labels[component] = component_index
        cross_mask = labels[:, None] != labels[None, :]
        cross_fit_wins = np.where(cross_mask, local_fit_wins, 0.0)
        cross_n_ij = cross_fit_wins + cross_fit_wins.T
        convergence_tol = max(float(tol), 1e-10)

        def objective(values: np.ndarray) -> float:
            directed_log_losses = np.logaddexp(
                0.0, values[None, :] - values[:, None]
            )
            return float(np.sum(cross_fit_wins * directed_log_losses))

        for _ in range(max_iter):
            if cls._bt_scc_solution_is_certified(
                scores,
                local_observed_wins,
                local_fit_wins,
                tol,
                include_within_scc=False,
            ):
                return scores

            edge_gradient = cls._bt_stable_edge_gradient(
                scores, cross_fit_wins
            )
            item_gradient = edge_gradient.sum(axis=1)
            group_gradient = np.array(
                [item_gradient[component].sum() for component in components]
            )

            differences = scores[:, None] - scores[None, :]
            tail_probability = np.exp(-np.abs(differences))
            variance = tail_probability / np.square(1.0 + tail_probability)
            item_curvature = cross_n_ij * variance
            group_curvature = np.zeros(
                (len(components), len(components)), dtype=float
            )
            for left, left_component in enumerate(components):
                for right in range(left + 1, len(components)):
                    right_component = components[right]
                    curvature = float(
                        item_curvature[
                            np.ix_(left_component, right_component)
                        ].sum()
                    )
                    group_curvature[left, right] = curvature
                    group_curvature[right, left] = curvature
            hessian = (
                np.diag(group_curvature.sum(axis=1)) - group_curvature
            )
            scale = float(np.max(np.abs(hessian)))
            if not np.isfinite(scale) or scale <= 0:
                break
            ground = int(np.argmax(np.diag(hessian)))
            keep = [
                index for index in range(len(components)) if index != ground
            ]
            try:
                reduced_direction = np.linalg.solve(
                    hessian[np.ix_(keep, keep)] / scale,
                    group_gradient[keep] / scale,
                )
            except np.linalg.LinAlgError:
                break
            group_direction = np.zeros(len(components), dtype=float)
            group_direction[keep] = reduced_direction
            item_direction = group_direction[labels]
            item_direction -= item_direction.mean()
            if np.any(~np.isfinite(item_direction)):
                break
            direction_norm = float(np.max(np.abs(item_direction)))
            decrement = float(np.dot(group_gradient, group_direction))
            if (
                direction_norm <= convergence_tol
                or not np.isfinite(decrement)
                or decrement <= 0
            ):
                break

            current_objective = objective(scores)
            step_size = 1.0
            accepted = False
            for _ in range(60):
                candidate = scores - step_size * item_direction
                candidate -= candidate.mean()
                candidate_objective = objective(candidate)
                if (
                    np.isfinite(candidate_objective)
                    and candidate_objective
                    <= current_objective - 1e-4 * step_size * decrement
                ):
                    scores = candidate
                    accepted = True
                    break
                step_size *= 0.5
            if not accepted:
                break

        raise RuntimeError(
            "Bradley-Terry SCC-offset optimization failed to converge"
        )

    @classmethod
    def _fit_bt_component_with_scc_coordinates(
        cls,
        initial_scores: np.ndarray,
        local_observed_wins: np.ndarray,
        local_fit_wins: np.ndarray,
        tol: float,
        max_sweeps: int = 4000,
        include_within_scc: bool = True,
    ) -> np.ndarray:
        """Finish an ill-conditioned BT fit by exact coordinate minimization.

        The coordinates consist of SCC-wide mean contrasts plus within-SCC
        contrasts, and therefore span the component's sum-to-zero space.  Each
        one-dimensional convex subproblem is solved by bracketing its stable
        score equation.  This avoids a global Hessian inversion when a weak
        regularized cut coexists with high-information comparisons inside an
        SCC, while still optimizing the original regularized likelihood.
        """

        scores = np.asarray(initial_scores, dtype=float).copy()
        scores -= scores.mean()
        convergence_tol = max(float(tol), 1e-10)
        directions = cls._bt_scc_coordinate_directions(
            local_observed_wins,
            include_within_scc=include_within_scc,
        )

        def directional_gradient(direction: np.ndarray, shift: float) -> float:
            candidate = scores + shift * direction
            if np.any(~np.isfinite(candidate)):
                return math.copysign(math.inf, shift)
            return cls._bt_stable_directional_gradient(
                candidate,
                local_fit_wins,
                direction,
            )

        for _ in range(max_sweeps):
            max_update = 0.0
            for direction in directions:
                coordinate_tol = max(
                    convergence_tol * 1e-3,
                    8.0
                    * np.finfo(float).eps
                    * max(1.0, float(np.max(np.abs(scores)))),
                )
                certification_radius = convergence_tol / float(
                    np.max(np.abs(direction))
                )
                lower_derivative = directional_gradient(
                    direction, -certification_radius
                )
                upper_derivative = directional_gradient(
                    direction, certification_radius
                )
                if (
                    np.isfinite(lower_derivative)
                    and np.isfinite(upper_derivative)
                    and lower_derivative <= 0
                    and upper_derivative >= 0
                ):
                    continue
                derivative = directional_gradient(direction, 0.0)
                if not np.isfinite(derivative):
                    raise RuntimeError(
                        "Bradley-Terry SCC coordinate gradient was non-finite"
                    )
                if derivative == 0.0:
                    continue

                if derivative > 0:
                    lower, upper = -1.0, 0.0
                    for _ in range(2048):
                        if directional_gradient(direction, lower) <= 0:
                            break
                        lower *= 2.0
                    else:
                        raise RuntimeError(
                            "Could not bracket a Bradley-Terry SCC coordinate"
                        )
                else:
                    lower, upper = 0.0, 1.0
                    for _ in range(2048):
                        if directional_gradient(direction, upper) >= 0:
                            break
                        upper *= 2.0
                    else:
                        raise RuntimeError(
                            "Could not bracket a Bradley-Terry SCC coordinate"
                        )

                for _ in range(256):
                    midpoint = 0.5 * (lower + upper)
                    midpoint_derivative = directional_gradient(direction, midpoint)
                    if midpoint_derivative > 0:
                        upper = midpoint
                    elif midpoint_derivative < 0:
                        lower = midpoint
                    else:
                        lower = upper = midpoint
                        break
                    if upper - lower <= coordinate_tol:
                        break
                shift = 0.5 * (lower + upper)
                scores += shift * direction
                scores -= scores.mean()
                max_update = max(
                    max_update,
                    abs(shift) * float(np.max(np.abs(direction))),
                )
            if max_update <= convergence_tol:
                return scores

        raise RuntimeError(
            "Bradley-Terry SCC coordinate optimization failed to converge"
        )

    @staticmethod
    def _refine_bt_component_with_newton(
        initial_scores: np.ndarray,
        local_fit_wins: np.ndarray,
        local_n_ij: np.ndarray,
        tol: float,
        max_iter: int = 2000,
    ) -> np.ndarray:
        """Polish a slow MM fit with constrained, damped Newton steps.

        The Bradley–Terry negative log-likelihood is convex in log-skills and
        has one location null direction. Each Newton system is therefore
        solved in the positive-eigenvalue subspace of its Laplacian Hessian,
        preserving the component sum-to-zero constraint. Backtracking keeps
        every accepted step likelihood-improving.
        """

        scores = np.asarray(initial_scores, dtype=float).copy()
        scores -= scores.mean()
        convergence_tol = max(float(tol), 1e-10)

        def objective(values: np.ndarray) -> float:
            directed_log_losses = np.logaddexp(
                0.0, values[None, :] - values[:, None]
            )
            return float(np.sum(local_fit_wins * directed_log_losses))

        for _ in range(max_iter):
            differences = scores[:, None] - scores[None, :]
            tail_probability = np.exp(-np.abs(differences))
            # Form the score equations from directed edge residuals.  Writing
            # them as ``n_ij * p_ij - wins_i`` subtracts nearly equal O(1)
            # terms after separation and loses pseudo-counts as small as
            # 1e-300 to cancellation.
            gradient = Rank._bt_stable_gradient(scores, local_fit_wins)
            variance = tail_probability / np.square(1.0 + tail_probability)
            curvature = local_n_ij * variance
            hessian = np.diag(curvature.sum(axis=1)) - curvature
            hessian = 0.5 * (hessian + hessian.T)
            scale = float(np.max(np.abs(hessian)))
            if not np.isfinite(scale) or scale <= 0:
                break
            # Ground the highest-curvature vertex and solve the reduced
            # Laplacian.  Unlike a full eigendecomposition, this preserves a
            # weak cut next to a high-information subgraph instead of losing
            # its tiny positive eigenvalue to cancellation with the location
            # null direction.
            ground = int(np.argmax(np.diag(hessian)))
            keep = [index for index in range(len(scores)) if index != ground]
            reduced_hessian = hessian[np.ix_(keep, keep)] / scale
            reduced_gradient = gradient[keep] / scale
            try:
                reduced_direction = np.linalg.solve(
                    reduced_hessian, reduced_gradient
                )
            except np.linalg.LinAlgError:
                break
            direction = np.zeros(len(scores), dtype=float)
            direction[keep] = reduced_direction
            if np.any(~np.isfinite(direction)):
                break
            direction -= direction.mean()
            direction_norm = float(np.max(np.abs(direction)))
            if direction_norm <= convergence_tol:
                return scores
            decrement = float(np.dot(gradient, direction))
            if not np.isfinite(decrement) or decrement <= 0:
                break

            current_objective = objective(scores)
            step_size = 1.0
            accepted = False
            for _ in range(60):
                candidate = scores - step_size * direction
                candidate -= candidate.mean()
                candidate_objective = objective(candidate)
                if (
                    np.isfinite(candidate_objective)
                    and candidate_objective
                    <= current_objective - 1e-4 * step_size * decrement
                ):
                    scores = candidate
                    accepted = True
                    break
                step_size *= 0.5
            if not accepted:
                break
            if step_size * direction_norm <= convergence_tol:
                return scores

        raise RuntimeError(
            "Bradley-Terry optimization failed to converge; no scores or "
            "standard errors were produced"
        )

    def _fit_bt(
        self,
        item_ids: List[str],
        outcomes: Sequence[Union[Tuple[str, str], WeightedOutcome]],
        pseudo: float,
        max_iter: int,
        tol: float,
        return_info: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
        """Fit a Bradley–Terry model given pairwise outcomes.

        Parameters
        ----------
        item_ids:
            List of unique item identifiers.
        outcomes:
            Pairwise ``(winner, loser)`` tuples, or weighted
            ``(winner, loser, weight)`` tuples. A tie is represented by two
            directions with weight ``0.5`` each.
        pseudo:
            Total symmetric pseudo-comparison weight added to each observed
            edge. Unobserved pairs receive no pseudo-comparisons.
        max_iter, tol:
            Control convergence of the iterative fixed‑point updates.
        return_info:
            If ``True`` return the intermediate matrices ``n_ij`` and
            ``p_ij`` for downstream standard error computation.

        Returns
        -------
        scores : dict
            Mapping from item identifier to estimated log‑skill.
        (scores, n_ij, p_ij) : tuple
            When ``return_info`` is ``True``, also return total observed match
            counts and predicted win probabilities for observed components.
            Pseudo-comparisons are intentionally excluded from the counts;
            cross-component probabilities are ``NaN`` because they are not
            identified by the comparison data.
        """
        if max_iter < 1:
            raise ValueError("max_iter must be at least 1")
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be a finite positive number")

        observed_wins, fit_wins = self._build_bt_win_matrices(
            item_ids, outcomes, pseudo
        )
        n = len(item_ids)
        observed_n_ij = observed_wins + observed_wins.T
        fit_n_ij = fit_wins + fit_wins.T
        scores = np.zeros(n, dtype=float)

        for component in self._comparison_components(observed_n_ij):
            if len(component) == 1:
                continue
            if pseudo == 0 and not self._is_strongly_connected(
                observed_wins, component
            ):
                raise ValueError(
                    "learning_rate must be positive when the directed win graph "
                    "is not strongly connected"
                )
            local_observed_wins = observed_wins[np.ix_(component, component)]
            local_fit_wins = fit_wins[np.ix_(component, component)]
            local_wins = local_fit_wins.sum(axis=1)
            local_n_ij = fit_n_ij[np.ix_(component, component)]
            requires_scc_coordinates = (
                len(self._bt_strong_components(local_observed_wins)) > 1
            )
            if np.any(local_wins <= 0):
                raise ValueError(
                    "learning_rate must be positive when the observed win graph "
                    "does not have a finite Bradley-Terry maximum-likelihood estimate"
                )

            def scc_solution_is_certified(candidate_scores: np.ndarray) -> bool:
                return not requires_scc_coordinates or (
                    self._bt_scc_solution_is_certified(
                        candidate_scores,
                        local_observed_wins,
                        local_fit_wins,
                        tol,
                    )
                )

            def finish_with_scc_coordinates(
                initial_scores: np.ndarray,
            ) -> np.ndarray:
                if requires_scc_coordinates:
                    try:
                        offset_scores = self._refine_bt_scc_offsets_with_newton(
                            initial_scores,
                            local_observed_wins,
                            local_fit_wins,
                            tol,
                        )
                    except RuntimeError:
                        try:
                            offset_scores = (
                                self._fit_bt_component_with_scc_coordinates(
                                    initial_scores,
                                    local_observed_wins,
                                    local_fit_wins,
                                    tol,
                                    include_within_scc=False,
                                )
                            )
                        except RuntimeError:
                            offset_scores = initial_scores
                    if scc_solution_is_certified(offset_scores):
                        return offset_scores
                    try:
                        polished_scores = (
                            self._refine_bt_component_with_newton(
                                offset_scores,
                                local_fit_wins,
                                local_n_ij,
                                tol,
                            )
                        )
                    except RuntimeError:
                        polished_scores = offset_scores
                    if scc_solution_is_certified(polished_scores):
                        return polished_scores
                else:
                    polished_scores = initial_scores
                return self._fit_bt_component_with_scc_coordinates(
                    polished_scores,
                    local_observed_wins,
                    local_fit_wins,
                    tol,
                )

            def refine_component(
                initial_scores: np.ndarray,
                fallback_on_newton_failure: bool,
            ) -> np.ndarray:
                try:
                    refined_scores = self._refine_bt_component_with_newton(
                        initial_scores,
                        local_fit_wins,
                        local_n_ij,
                        tol,
                    )
                except RuntimeError:
                    if not fallback_on_newton_failure:
                        raise
                    return finish_with_scc_coordinates(initial_scores)
                if not scc_solution_is_certified(refined_scores):
                    return finish_with_scc_coordinates(refined_scores)
                return refined_scores

            log_abilities = np.zeros(len(component), dtype=float)
            converged = False
            logit_start = self._bt_logit_spanning_tree_start(local_fit_wins)
            if logit_start is not None:
                try:
                    log_abilities = refine_component(
                        logit_start,
                        fallback_on_newton_failure=False,
                    )
                    converged = True
                except RuntimeError:
                    # Retain Hunter's globally convergent MM update as the
                    # conservative path when direct refinement is unresolved
                    # from the log-odds projection.
                    pass

            abilities = np.ones(len(component), dtype=float)
            for _ in range(max_iter):
                if converged:
                    break
                denom = (
                    local_n_ij
                    / (abilities[:, None] + abilities[None, :])
                ).sum(axis=1)
                if np.any(~np.isfinite(denom)) or np.any(denom <= 0):
                    raise FloatingPointError(
                        "Bradley-Terry update encountered an invalid denominator"
                    )
                next_abilities = local_wins / denom
                if np.any(~np.isfinite(next_abilities)) or np.any(
                    next_abilities <= 0
                ):
                    raise FloatingPointError(
                        "Bradley-Terry update encountered an invalid ability"
                    )

                next_log_abilities = np.log(next_abilities)
                next_log_abilities -= next_log_abilities.mean()
                next_abilities = np.exp(next_log_abilities)
                delta = float(
                    np.max(np.abs(next_log_abilities - log_abilities))
                )
                abilities = next_abilities
                log_abilities = next_log_abilities
                if delta < tol:
                    converged = True
                    break
            if not converged:
                log_abilities = refine_component(
                    log_abilities,
                    fallback_on_newton_failure=True,
                )
            elif not scc_solution_is_certified(log_abilities):
                log_abilities = finish_with_scc_coordinates(log_abilities)
            scores[component] = log_abilities

        if not return_info:
            return {item: float(val) for item, val in zip(item_ids, scores)}
        p_ij = np.full((n, n), np.nan, dtype=float)
        for component in self._comparison_components(observed_n_ij):
            local_scores = scores[component]
            score_diff = np.clip(
                local_scores[:, None] - local_scores[None, :], -700, 700
            )
            p_ij[np.ix_(component, component)] = 1.0 / (
                1.0 + np.exp(-score_diff)
            )
        return (
            {item: float(val) for item, val in zip(item_ids, scores)},
            observed_n_ij,
            p_ij,
        )

    def _bt_standard_errors(
        self,
        s: np.ndarray,
        n_ij: np.ndarray,
        p_ij: np.ndarray,
        rcond: float,
        regularization_strength: float = 0.0,
    ) -> np.ndarray:
        """Estimate sandwich standard errors for regularized BT scores.

        The observed information for the Bradley–Terry working model is given by
        ``I = diag(q 1) - q`` where ``q = n_ij * p_ij * (1 - p_ij)`` encodes the
        model-based binary-BT variability determined by observed comparison
        weights. Because
        the point estimator includes fixed pseudo-comparisons, its frequentist
        covariance uses the penalized-information matrix as the sandwich bread
        and observed information as the meat. Thus regularization affects the
        estimator's sampling variance without being treated as random data.
        The calculation treats the realized comparison graph as fixed. This
        model-based variance excludes regularization bias, shared-judge
        dependence, likelihood misspecification, and adaptive-selection
        uncertainty. It is therefore not a total-error or mean-squared-error
        estimate.

        The calculation is performed separately under a sum-to-zero constraint
        within each observed component. Isolated items remain ``NaN``. Draws
        use the package's fractional-binomial half-win approximation rather than
        a separate three-outcome tie likelihood.

        Parameters
        ----------
        s : np.ndarray
            Array of estimated log-skills for each item.
        n_ij : np.ndarray
            Matrix of observed total match weights between items.
        p_ij : np.ndarray
            Matrix of predicted win probabilities between items.
        rcond : float
            Relative eigenvalue tolerance used when taking the constrained
            inverse. It does not add information to the Fisher matrix.
        regularization_strength : float
            Total fixed pseudo-comparison weight added to each observed edge
            by the point estimator. This enters the sandwich bread, not the
            observed-data meat.

        Returns
        -------
        np.ndarray
            Array of standard errors corresponding to each element of ``s``.
        """

        n = len(s)
        if n == 0:
            return np.array([], dtype=float)
        if n_ij.shape != (n, n) or p_ij.shape != (n, n):
            raise ValueError("n_ij and p_ij must be square matrices matching s")
        if not np.isfinite(rcond) or rcond < 0:
            raise ValueError("rcond must be a finite non-negative number")
        if (
            not np.isfinite(regularization_strength)
            or regularization_strength < 0
        ):
            raise ValueError(
                "regularization_strength must be a finite non-negative number"
            )
        components = self._comparison_components(n_ij)
        if (
            len(components) > 1
            and np.any(n_ij > 0)
            and not self._warned_disconnected_graph
        ):
            warnings.warn(
                "The observed comparison graph is disconnected. Scores and "
                "standard errors are component-relative; inspect the saved "
                "'<attribute>_component' columns before comparing items.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_disconnected_graph = True

        se = np.full(n, np.nan, dtype=float)
        for component in components:
            if len(component) == 1:
                continue
            local_n_ij = n_ij[np.ix_(component, component)]
            local_s = s[component]
            local_p_ij = p_ij[np.ix_(component, component)]
            if np.any(~np.isfinite(local_p_ij)):
                continue
            # exp(-|delta|) / (1 + exp(-|delta|))^2 is algebraically
            # p * (1 - p), but remains symmetric when a direct sigmoid rounds
            # to exactly zero or one at an extreme score gap.
            tail_probability = np.exp(
                -np.abs(local_s[:, None] - local_s[None, :])
            )
            variance = tail_probability / np.square(1.0 + tail_probability)
            observed_q = local_n_ij * variance
            regularized_n_ij = local_n_ij + regularization_strength * (
                local_n_ij > 0
            )
            bread_q = regularized_n_ij * variance
            meat = np.diag(observed_q.sum(axis=1)) - observed_q
            bread = np.diag(bread_q.sum(axis=1)) - bread_q
            meat = 0.5 * (meat + meat.T)
            bread = 0.5 * (bread + bread.T)
            scale = float(np.max(np.abs(bread)))
            if not np.isfinite(scale) or scale <= 0:
                continue
            scaled_bread = bread / scale
            scaled_meat = meat / scale
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(scaled_bread)
            except np.linalg.LinAlgError:
                continue
            max_eigenvalue = float(np.max(eigenvalues))
            threshold = max(
                float(rcond), np.finfo(float).eps * len(component)
            ) * max(0.0, max_eigenvalue)
            identified = eigenvalues > threshold
            if int(np.sum(identified)) != len(component) - 1:
                continue
            basis = eigenvectors[:, identified]
            local_meat = basis.T @ scaled_meat @ basis
            local_meat = 0.5 * (local_meat + local_meat.T)
            try:
                meat_eigenvalues, meat_eigenvectors = np.linalg.eigh(local_meat)
            except np.linalg.LinAlgError:
                continue
            meat_root = meat_eigenvectors * np.sqrt(
                np.clip(meat_eigenvalues, 0.0, None)
            )[None, :]
            covariance_factor = basis @ (
                meat_root / eigenvalues[identified, None]
            )
            se[component] = (
                np.hypot.reduce(covariance_factor, axis=1) / np.sqrt(scale)
            )
        return se

    def _fit_pl(
        self,
        item_ids: List[str],
        rankings: List[List[str]],
        pseudo: float,
        max_iter: int,
        tol: float,
    ) -> Dict[str, float]:
        """Fit a Plackett–Luce model for multiway rankings.

        When every ranking is of length two this reduces to the BT
        model and defers to :meth:`_fit_bt`.  If no rankings are
        provided a zero‑centred score is returned for each item.  See
        Hunter (2004) for details on the fitting procedure.
        """
        if not rankings:
            return {i: 0.0 for i in item_ids}
        # if all rankings are of length 2, delegate to BT
        if all(len(r) == 2 for r in rankings):
            outcomes = [(r[0], r[1]) for r in rankings]
            return self._fit_bt(
                item_ids, outcomes, pseudo, max_iter, tol, return_info=False
            )
        n = len(item_ids)
        idx = {item: i for i, item in enumerate(item_ids)}
        w_i = np.zeros(n, dtype=float)
        rankings_idx = []
        for r in rankings:
            r_idx = [idx[x] for x in r if x in idx]
            if len(r_idx) < 2:
                continue
            rankings_idx.append(r_idx)
            for i_ in r_idx:
                w_i[i_] += 1.0
        if len(rankings_idx) == 0:
            return {i: 0.0 for i in item_ids}
        w_i += pseudo
        p = np.ones(n, dtype=float)
        for _ in range(max_iter):
            denom = np.zeros(n, dtype=float)
            for r_idx in rankings_idx:
                remaining = np.array(r_idx, dtype=int)
                sum_p = p[remaining].sum()
                for i_ in r_idx:
                    denom[i_] += 1.0 / sum_p
                    sum_p -= p[i_]
            denom[denom == 0] = 1e-12
            p_new = w_i / denom
            if np.max(np.abs(p_new - p)) < tol:
                p = p_new
                break
            p = p_new
        s = np.log(p)
        s -= s.mean()
        return {item: float(val) for item, val in zip(item_ids, s)}

    # ------------------------------------------------------------------
    # Pairing strategies
    # ------------------------------------------------------------------
    def _pairs_random(
        self, item_ids: List[str], texts_by_id: Dict[str, str], mpr: int
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """Return a set of random, unique pairs for the given items."""
        pairs_set: Set[Tuple[str, str]] = set()
        for a in item_ids:
            others = [x for x in item_ids if x != a]
            if not others:
                continue
            k = min(mpr, len(others))
            opponents = self.rng.sample(others, k)
            for b in opponents:
                pairs_set.add(tuple(sorted((a, b))))
        return [((a, texts_by_id[a]), (b, texts_by_id[b])) for a, b in pairs_set]

    def _pairs_adjacent(
        self,
        item_ids: List[str],
        texts_by_id: Dict[str, str],
        current_ratings: Dict[str, float],
        mpr: int,
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """Pair each item with its nearest neighbours in rating space."""
        pairs_set: Set[Tuple[str, str]] = set()
        sorted_ids = sorted(item_ids, key=lambda i: current_ratings[i])
        n = len(sorted_ids)
        for i, a in enumerate(sorted_ids):
            for off in range(1, mpr + 1):
                b = sorted_ids[(i + off) % n]
                if a == b:
                    continue
                pairs_set.add(tuple(sorted((a, b))))
        # small amount of random exploration to avoid pathological pairings
        n_random_targets = int(self._EXPLORE_FRAC * n * mpr)
        for _ in range(n_random_targets):
            if n < 2:
                break
            a, b = self.rng.sample(item_ids, 2)
            pairs_set.add(tuple(sorted((a, b))))
        return [((a, texts_by_id[a]), (b, texts_by_id[b])) for a, b in pairs_set]

    def _pairs_info_gain(
        self,
        item_ids: List[str],
        texts_by_id: Dict[str, str],
        current_ratings: Dict[str, float],
        se_agg: Dict[str, float],
        mpr: int,
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """Select pairs with an uncertainty-guided scheduling heuristic.

        A bounded candidate set scales with the number of items. The heuristic
        favors uncertain, similarly rated pairs using current model-based
        standard errors and binary-BT outcome variance; it is not an exact
        expected-information-gain calculation. ``mpr`` is a capped target
        degree, not a guarantee: an item can exceed it when selected as another
        item's opponent.
        """
        n = len(item_ids)
        if n < 2:
            return []
        finite_se = [
            float(value)
            for value in se_agg.values()
            if np.isfinite(value) and float(value) >= 0
        ]
        unidentified_se = 2.0 * max(finite_se, default=1.0)

        def item_se(item_id: str) -> float:
            value = se_agg.get(item_id, unidentified_se)
            if not np.isfinite(value) or float(value) < 0:
                return unidentified_se
            return float(value)

        max_pairs = max(1, self._MAX_CANDIDATE_PAIRS_PER_ROUND)
        desired_neighbors = max_pairs // max(1, n)
        candidate_neighbors = max(
            mpr, min(self._CANDIDATE_NEIGHBORS, desired_neighbors)
        )

        def logistic_clip(x: float) -> float:
            if x > 50:
                return 1.0
            if x < -50:
                return 0.0
            return 1.0 / (1.0 + np.exp(-x))

        ids_sorted = sorted(item_ids, key=lambda i: current_ratings[i])
        idx_of = {i_id: k for k, i_id in enumerate(ids_sorted)}
        num_high_se = max(1, int(self._HIGH_SE_FRAC * n))
        high_se_ids = sorted(item_ids, key=item_se, reverse=True)[
            :num_high_se
        ]
        candidate_pairs_set: Set[Tuple[str, str]] = set()
        for i_id in item_ids:
            pos = idx_of[i_id]
            lower = max(0, pos - candidate_neighbors)
            upper = min(n, pos + candidate_neighbors + 1)
            for j in ids_sorted[lower:upper]:
                if i_id == j:
                    continue
                candidate_pairs_set.add(tuple(sorted((i_id, j))))
        for hs in high_se_ids:
            others = [x for x in item_ids if x != hs]
            k = min(candidate_neighbors, len(others))
            samp = self.rng.sample(others, k)
            for j in samp:
                candidate_pairs_set.add(tuple(sorted((hs, j))))
        remaining_capacity = max_pairs - len(candidate_pairs_set)
        n_random_targets = int(self._EXPLORE_FRAC * n * mpr)
        if remaining_capacity > 0:
            n_random_targets = min(n_random_targets, remaining_capacity)
            for _ in range(n_random_targets):
                if n < 2:
                    break
                a, b = self.rng.sample(item_ids, 2)
                candidate_pairs_set.add(tuple(sorted((a, b))))
        partners_count = {i: 0 for i in item_ids}
        for a, b in candidate_pairs_set:
            partners_count[a] += 1
            partners_count[b] += 1
        for i_id in item_ids:
            while partners_count[i_id] < mpr:
                potential = [x for x in item_ids if x != i_id]
                if not potential:
                    break
                j = self.rng.choice(potential)
                pair = tuple(sorted((i_id, j)))
                if pair not in candidate_pairs_set:
                    candidate_pairs_set.add(pair)
                    partners_count[i_id] += 1
                    partners_count[j] += 1
                else:
                    partners_count[i_id] += 1
                    partners_count[j] += 1
        scored_pairs: List[Tuple[float, str, str]] = []
        for a, b in candidate_pairs_set:
            diff = current_ratings[a] - current_ratings[b]
            p = logistic_clip(diff)
            outcome_var = p * (1 - p)
            var_a = item_se(a) ** 2
            var_b = item_se(b) ** 2
            param_unc = var_a + var_b
            # Favor uncertain, similarly rated pairs. The closeness term dampens
            # pairings with large rating gaps to probe subtle ordering differences.
            closeness = 1.0 / (1.0 + abs(diff))
            score = outcome_var * param_unc * closeness
            scored_pairs.append((score, a, b))
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        needed: Dict[str, int] = {i: mpr for i in item_ids}
        pairs_selected: List[Tuple[str, str]] = []
        pairs_seen: Set[Tuple[str, str]] = set()
        for score, a, b in scored_pairs:
            if needed[a] > 0 and needed[b] > 0:
                tup = (a, b) if a < b else (b, a)
                if tup in pairs_seen:
                    continue
                pairs_selected.append(tup)
                pairs_seen.add(tup)
                needed[a] -= 1
                needed[b] -= 1
        while any(cnt > 0 for cnt in needed.values()):
            ids_needing = [i for i, cnt in needed.items() if cnt > 0]
            if not ids_needing:
                break
            # Choose an item that still needs matches
            a = self.rng.choice(ids_needing)
            # Prefer an unscheduled partner that also needs a comparison. If
            # none exists, allow one endpoint to exceed its target by one.
            potential = [
                x
                for x in item_ids
                if x != a and tuple(sorted((a, x))) not in pairs_seen
            ]
            if not potential:
                break
            partners_needing = [x for x in potential if needed[x] > 0]
            b = self.rng.choice(partners_needing or potential)
            tup = (a, b) if a < b else (b, a)
            pairs_selected.append(tup)
            pairs_seen.add(tup)
            needed[a] -= 1
            if needed[b] > 0:
                needed[b] -= 1
        return [((a, texts_by_id[a]), (b, texts_by_id[b])) for a, b in pairs_selected]

    def _generate_pairs(
        self,
        item_ids: List[str],
        texts_by_id: Dict[str, str],
        current_ratings: Optional[Dict[str, float]],
        se_agg: Optional[Dict[str, float]],
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """Dispatch to the appropriate pairing strategy."""
        if len(item_ids) < 2:
            return []
        mpr = min(self.cfg.matches_per_round, len(item_ids) - 1)
        if not self.cfg.power_matching:
            return self._pairs_random(item_ids, texts_by_id, mpr)
        if current_ratings is None:
            current_ratings = {i: 0.0 for i in item_ids}
        if se_agg is None or len(se_agg) != len(item_ids):
            se_full = {i: 1.0 for i in item_ids}
        else:
            se_full = se_agg
        return self._pairs_info_gain(
            item_ids, texts_by_id, current_ratings, se_full, mpr
        )

    async def _catch_up_existing_rounds(
        self,
        candidate_ids: List[str],
        round_indices: List[int],
        item_ids: List[str],
        texts_by_id: Dict[str, str],
        images_by_id: Dict[str, List[str]],
        audio_by_id: Dict[str, List[Dict[str, str]]],
        pdfs_by_id: Dict[str, List[Dict[str, str]]],
        attr_batches: List[List[str]],
        attr_keys: List[str],
        history_pairs: Dict[str, List[WeightedOutcome]],
        outcome_counts: Dict[str, Dict[str, int]],
        ratings: Dict[str, Dict[str, float]],
        se_store: Dict[str, Dict[str, float]],
        component_store: Dict[str, Dict[str, int]],
        base_name: str,
        df_proc: pd.DataFrame,
        _write_checkpoint: Callable[[], None],
        current_ratings: Optional[Dict[str, float]],
        se_agg_local: Optional[Dict[str, float]],
        reset_files: bool,
        identifier_hash_bits: int,
        **kwargs: Any,
    ) -> None:
        if not candidate_ids:
            return
        for rnd in round_indices:
            round_path = os.path.join(self.cfg.save_dir, f"{base_name}_round{rnd}.csv")
            if not os.path.exists(round_path):
                continue
            df_round = self._read_rank_checkpoint(
                round_path, len(attr_batches)
            )
            counts: Dict[str, int] = {}
            current_id_set = set(item_ids)
            committed_pairs: Set[Tuple[str, str]] = set()
            if {"IdA", "IdB"}.issubset(df_round.columns):
                counts_by_batch: Dict[int, Dict[str, int]] = {
                    batch_idx: {} for batch_idx in range(len(attr_batches))
                }
                for batch_raw, a, b in zip(
                    df_round.get("Batch", pd.Series(dtype=float)),
                    df_round["IdA"],
                    df_round["IdB"],
                ):
                    try:
                        batch_idx = int(batch_raw)
                    except (TypeError, ValueError):
                        continue
                    if batch_idx not in counts_by_batch:
                        continue
                    id_a = str(a)
                    id_b = str(b)
                    if id_a not in current_id_set or id_b not in current_id_set:
                        continue
                    committed_pairs.add(tuple(sorted((id_a, id_b))))
                    batch_counts = counts_by_batch[batch_idx]
                    batch_counts[id_a] = batch_counts.get(id_a, 0) + 1
                    batch_counts[id_b] = batch_counts.get(id_b, 0) + 1
                for item_id in item_ids:
                    counts[item_id] = min(
                        (
                            batch_counts.get(item_id, 0)
                            for batch_counts in counts_by_batch.values()
                        ),
                        default=0,
                    )
            else:
                for ident in df_round.get("Identifier", []):
                    parts = str(ident).split("|")
                    if len(parts) != 5:
                        continue
                    _, _, _, id_a, id_b = parts
                    if id_a not in current_id_set or id_b not in current_id_set:
                        continue
                    committed_pairs.add(tuple(sorted((id_a, id_b))))
                    counts[id_a] = counts.get(id_a, 0) + 1
                    counts[id_b] = counts.get(id_b, 0) + 1
            catchup_path = os.path.join(
                self.cfg.save_dir, f".{base_name}_catchup_round{rnd}.csv"
            )
            plan_path = os.path.join(
                self.cfg.save_dir, f".{base_name}_catchup_round{rnd}_plan.json"
            )
            batch_state_path = f"{catchup_path}.batch_state.json"
            committed_response_ids = set(df_round["Identifier"].astype(str))
            expected_target_matches = min(
                self.cfg.matches_per_round, max(0, len(item_ids) - 1)
            )
            canonical_item_ids = sorted(item_ids)
            plan_records: Optional[List[Dict[str, Any]]] = None
            if os.path.exists(plan_path):
                try:
                    with open(plan_path, encoding="utf-8") as plan_file:
                        plan_payload = json.load(plan_file)
                except Exception as exc:
                    raise ValueError(
                        f"Could not read Rank catch-up plan {plan_path!r}"
                    ) from exc
                planned_item_ids = (
                    plan_payload.get("item_ids")
                    if isinstance(plan_payload, dict)
                    else None
                )
                if (
                    not isinstance(plan_payload, dict)
                    or plan_payload.get("version") != 1
                    or plan_payload.get("round") != rnd
                    or plan_payload.get("attribute_batches") != attr_batches
                    or not isinstance(planned_item_ids, list)
                    or any(not isinstance(item_id, str) for item_id in planned_item_ids)
                    or len(planned_item_ids) != len(set(planned_item_ids))
                    or sorted(planned_item_ids) != canonical_item_ids
                    or plan_payload.get("target_matches")
                    != expected_target_matches
                    or not isinstance(plan_payload.get("records"), list)
                    or not plan_payload["records"]
                ):
                    raise ValueError(
                        f"Rank catch-up plan {plan_path!r} is incompatible or malformed"
                    )
                candidate_records = plan_payload["records"]
                required_record_keys = {
                    "identifier",
                    "batch",
                    "pair",
                    "id_a",
                    "id_b",
                    "circle_first",
                }
                if any(
                    not isinstance(record, dict)
                    or not required_record_keys.issubset(record)
                    for record in candidate_records
                ):
                    raise ValueError(
                        f"Rank catch-up plan {plan_path!r} has malformed records"
                    )
                for record in candidate_records:
                    if (
                        type(record["batch"]) is not int
                        or type(record["pair"]) is not int
                        or type(record["circle_first"]) is not bool
                        or not str(record["id_a"]).strip()
                        or not str(record["id_b"]).strip()
                    ):
                        raise ValueError(
                            f"Rank catch-up plan {plan_path!r} has invalid fields"
                        )
                    expected_identifier = hash_identifier(
                        "catchup|"
                        f"{rnd}|{record['batch']}|{record['pair']}|"
                        f"{record['id_a']}|{record['id_b']}",
                        bits=identifier_hash_bits,
                    )
                    if str(record["identifier"]) != expected_identifier:
                        raise ValueError(
                            f"Rank catch-up plan {plan_path!r} has an invalid identifier"
                        )
                planned_ids = [str(record["identifier"]) for record in candidate_records]
                if len(planned_ids) != len(set(planned_ids)):
                    raise ValueError(
                        f"Rank catch-up plan {plan_path!r} has duplicate identifiers"
                    )
                committed_plan_ids = set(planned_ids) & committed_response_ids
                if committed_plan_ids and committed_plan_ids != set(planned_ids):
                    raise ValueError(
                        "Rank found a partially committed catch-up plan. Use "
                        "reset_files=True or a new save_dir to recompute safely."
                    )
                if committed_plan_ids == set(planned_ids):
                    for completed_artifact in (
                        batch_state_path,
                        catchup_path,
                        plan_path,
                    ):
                        try:
                            Path(completed_artifact).unlink(missing_ok=True)
                        except OSError:
                            pass
                else:
                    planned_endpoints = {
                        str(record[key])
                        for record in candidate_records
                        for key in ("id_a", "id_b")
                    }
                    if not planned_endpoints.issubset(current_id_set):
                        raise ValueError(
                            "Rank has an unfinished catch-up plan whose items are "
                            "not all present. Restore the same input set before "
                            "resuming, or use reset_files=True after confirming no "
                            "external batch remains."
                        )
                    plan_records = candidate_records

            if os.path.exists(batch_state_path) and not os.path.exists(
                catchup_path
            ):
                try:
                    with open(batch_state_path, encoding="utf-8") as state_file:
                        batch_state = json.load(state_file)
                except Exception as exc:
                    raise ValueError(
                        "Could not read Rank catch-up Batch API state "
                        f"{batch_state_path!r}"
                    ) from exc
                batches = (
                    batch_state.get("batches", [])
                    if isinstance(batch_state, dict)
                    else None
                )
                if not isinstance(batches, list):
                    raise ValueError(
                        f"Rank catch-up Batch API state {batch_state_path!r} "
                        "is malformed"
                    )
                unresolved_submission = any(
                    not isinstance(batch, dict)
                    or batch.get("status") == "submitting"
                    or not batch.get("batch_id")
                    for batch in batches
                )
                has_active_batch = bool(batch_state.get("batch_id")) or bool(
                    batches
                )
                if unresolved_submission:
                    raise ValueError(
                        "Rank found an unresolved catch-up Batch API submission. "
                        "The server may already have accepted a paid job; "
                        "reconcile or cancel it before resetting local state."
                    )
                if not has_active_batch or plan_records is None:
                    raise ValueError(
                        "Rank found catch-up Batch API state without a durable "
                        "response checkpoint and recoverable plan. Paid results "
                        "may already exist; use reset_files=True only after "
                        "confirming that it is safe to discard the external state."
                    )
                # A submitted batch ID plus the durable plan is sufficient to
                # reconstruct the request set.  The collector will poll that
                # batch and atomically create the staging CSV when rows arrive.

            if plan_records is None:
                deficits = {
                    item_id: max(
                        0, expected_target_matches - counts.get(item_id, 0)
                    )
                    for item_id in candidate_ids
                }
                pairs_needed: List[Tuple[str, str]] = []
                scheduled_pairs: Set[Tuple[str, str]] = set(committed_pairs)
                while any(deficit > 0 for deficit in deficits.values()):
                    max_deficit = max(deficits.values())
                    focal_candidates = [
                        item_id
                        for item_id, deficit in deficits.items()
                        if deficit == max_deficit
                    ]
                    id_a = self.rng.choice(focal_candidates)
                    available = [
                        item_id
                        for item_id in item_ids
                        if item_id != id_a
                        and tuple(sorted((id_a, item_id))) not in scheduled_pairs
                    ]
                    if not available:
                        break
                    partners_needing = [
                        item_id
                        for item_id in available
                        if deficits.get(item_id, 0) > 0
                    ]
                    id_b = self.rng.choice(partners_needing or available)
                    pair = tuple(sorted((id_a, id_b)))
                    scheduled_pairs.add(pair)
                    pairs_needed.append((id_a, id_b))
                    deficits[id_a] -= 1
                    if deficits.get(id_b, 0) > 0:
                        deficits[id_b] -= 1
                if not pairs_needed:
                    continue
                plan_records = []
                for batch_idx, _batch in enumerate(attr_batches):
                    for pair_idx, (id_a, id_b) in enumerate(pairs_needed):
                        raw_ident = (
                            f"catchup|{rnd}|{batch_idx}|{pair_idx}|{id_a}|{id_b}"
                        )
                        hashed_ident = hash_identifier(
                            raw_ident, bits=identifier_hash_bits
                        )
                        circle_first_flag = (
                            self.cfg.circle_first
                            if self.cfg.circle_first is not None
                            else self.rng.random() < 0.5
                        )
                        plan_records.append(
                            {
                                "identifier": hashed_ident,
                                "batch": batch_idx,
                                "pair": pair_idx,
                                "id_a": id_a,
                                "id_b": id_b,
                                "circle_first": circle_first_flag,
                            }
                        )
                self._write_json_atomically(
                    {
                        "version": 1,
                        "round": rnd,
                        "attribute_batches": attr_batches,
                        # Pair prompts and plan endpoints are ID-keyed.  Store a
                        # canonical set representation so harmless DataFrame row
                        # reordering does not strand already-paid staged rows.
                        "item_ids": canonical_item_ids,
                        "target_matches": expected_target_matches,
                        "records": plan_records,
                    },
                    plan_path,
                )

            announce_prompt_rendering("Rank:catchup", len(plan_records))
            prompts: List[str] = []
            ids: List[str] = []
            pair_images: Dict[str, List[str]] = {}
            pair_audio: Dict[str, List[Dict[str, str]]] = {}
            pair_pdfs: Dict[str, List[Dict[str, str]]] = {}
            meta_map: Dict[str, Tuple[int, int, str, str]] = {}
            for record in plan_records:
                batch_idx = int(record["batch"])
                pair_idx = int(record["pair"])
                id_a = str(record["id_a"])
                id_b = str(record["id_b"])
                hashed_ident = str(record["identifier"])
                circle_first_flag = bool(record["circle_first"])
                if batch_idx < 0 or batch_idx >= len(attr_batches):
                    raise ValueError(
                        f"Rank catch-up plan {plan_path!r} has an invalid batch"
                    )
                batch = attr_batches[batch_idx]
                attr_def_map = (
                    {a: self.cfg.attributes[a] for a in batch}
                    if isinstance(self.cfg.attributes, dict)
                    else {a: "" for a in batch}
                )
                prompts.append(
                    self.template.render(
                        entry_circle=texts_by_id[id_a],
                        entry_square=texts_by_id[id_b],
                        attributes=attr_def_map,
                        additional_instructions=self.cfg.additional_instructions or "",
                        modality=self.cfg.modality,
                        circle_first=circle_first_flag,
                    )
                )
                ids.append(hashed_ident)
                meta_map[hashed_ident] = (batch_idx, pair_idx, id_a, id_b)
                if images_by_id:
                    imgs = []
                    ia = images_by_id.get(id_a, [])
                    ib = images_by_id.get(id_b, [])
                    if circle_first_flag:
                        if ia:
                            imgs.extend(ia)
                        if ib:
                            imgs.extend(ib)
                    else:
                        if ib:
                            imgs.extend(ib)
                        if ia:
                            imgs.extend(ia)
                    if imgs:
                        pair_images[hashed_ident] = imgs
                if audio_by_id:
                    auds = []
                    aa = audio_by_id.get(id_a, [])
                    ab = audio_by_id.get(id_b, [])
                    if circle_first_flag:
                        if aa:
                            auds.extend(aa)
                        if ab:
                            auds.extend(ab)
                    else:
                        if ab:
                            auds.extend(ab)
                        if aa:
                            auds.extend(aa)
                    if auds:
                        pair_audio[hashed_ident] = auds
                if pdfs_by_id:
                    pdfs: List[Dict[str, str]] = []
                    pa = pdfs_by_id.get(id_a, [])
                    pb = pdfs_by_id.get(id_b, [])
                    if circle_first_flag:
                        if pa:
                            pdfs.extend(pa)
                        if pb:
                            pdfs.extend(pb)
                    else:
                        if pb:
                            pdfs.extend(pb)
                        if pa:
                            pdfs.extend(pa)
                    if pdfs:
                        pair_pdfs[hashed_ident] = pdfs
            if not prompts:
                continue
            if len(ids) != len(set(ids)):
                raise ValueError(
                    "Rank prompt identifier collision; use reset_files=True "
                    "or a save_dir configured for 64-bit identifiers"
                )
            response_kwargs = dict(kwargs)
            # Rank commits a round only when every planned judgment is
            # present.  The generic collector's large-run tail shortcut is
            # therefore incompatible with transactional tournament replay.
            response_kwargs["skip_tail_fails"] = False
            if self.cfg.use_dummy:
                response_kwargs.setdefault(
                    "dummy_responses",
                    {
                        identifier: {
                            "responses": [
                                json.dumps(
                                    {
                                        attribute: "draw"
                                        for attribute in attr_batches[
                                            meta_map[identifier][0]
                                        ]
                                    }
                                )
                            ]
                        }
                        for identifier in ids
                    },
                )
            resp_df = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                prompt_images=pair_images or None,
                prompt_audio=pair_audio or None,
                prompt_pdfs=pair_pdfs or None,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                json_mode=self.cfg.modality != "audio",
                # Never let response collection append raw rows to an already
                # committed structured round. The staging checkpoint can resume
                # interrupted or failed requests, then the validated rows are
                # merged into the committed round atomically below.
                save_path=catchup_path,
                reset_files=reset_files,
                use_dummy=self.cfg.use_dummy,
                max_retries=1,
                reasoning_effort=self.cfg.reasoning_effort,
                **response_kwargs,
            )
            new_mask = self._validate_requested_responses(
                resp_df,
                ids,
                context="Rank catch-up",
                allow_extra=True,
            )
            resp_df = resp_df.loc[new_mask].copy()
            await self._validate_pairwise_response_payloads(
                resp_df,
                meta_map,
                attr_batches,
                context="Rank catch-up",
                retry_path=catchup_path,
            )
            if "Successful" not in resp_df.columns:
                resp_df["Successful"] = True
            resp_df["Batch"] = resp_df.Identifier.map(
                lambda x: meta_map.get(str(x), (np.nan, np.nan, "", ""))[0]
            )
            resp_df["Pair"] = resp_df.Identifier.map(
                lambda x: meta_map.get(str(x), (np.nan, np.nan, "", ""))[1]
            )
            resp_df["IdA"] = resp_df.Identifier.map(
                lambda x: meta_map.get(str(x), (np.nan, np.nan, "", ""))[2]
            )
            resp_df["IdB"] = resp_df.Identifier.map(
                lambda x: meta_map.get(str(x), (np.nan, np.nan, "", ""))[3]
            )
            combined_round = pd.concat([df_round, resp_df], ignore_index=True)
            combined_round = combined_round.drop_duplicates(
                subset=["Identifier"], keep="last"
            )
            self._write_rank_checkpoint(
                combined_round, round_path, len(attr_batches)
            )
            for completed_artifact in (
                batch_state_path,
                catchup_path,
                plan_path,
            ):
                try:
                    Path(completed_artifact).unlink(missing_ok=True)
                except OSError:
                    pass

            async def _coerce_dict(raw: Any) -> Dict[str, Any]:
                obj = await safest_json(raw)
                if isinstance(obj, dict):
                    return obj
                if isinstance(obj, str):
                    obj2 = await safest_json(obj)
                    if isinstance(obj2, dict):
                        return obj2
                if isinstance(obj, list) and obj:
                    inner = await safest_json(obj[0])
                    if isinstance(inner, dict):
                        return inner
                return {}

            for ident, resp in zip(resp_df.Identifier, resp_df.Response):
                meta = meta_map.get(str(ident))
                if not meta:
                    continue
                batch_idx, _, id_a, id_b = meta
                safe_obj = await _coerce_dict(resp)
                if not safe_obj:
                    continue
                batch = attr_batches[batch_idx]
                batch_attr_map = {str(k).strip().lower(): k for k in batch}
                for attr_raw, winner_raw in safe_obj.items():
                    attr_key_l = str(attr_raw).strip().lower()
                    if attr_key_l not in batch_attr_map:
                        continue
                    real_attr = batch_attr_map[attr_key_l]
                    category = self._record_pairwise_outcome(
                        history_pairs[real_attr], id_a, id_b, winner_raw
                    )
                    outcome_counts[real_attr][category] += 1
            se_agg_next: Dict[str, float] = {i: 0.0 for i in item_ids}
            se_agg_counts: Dict[str, int] = {i: 0 for i in item_ids}
            for attr in attr_keys:
                outcomes = history_pairs[attr]
                if len(outcomes) == 0:
                    continue
                bt_scores, n_ij, p_ij = self._fit_bt(
                    item_ids=item_ids,
                    outcomes=outcomes,
                    pseudo=self.cfg.learning_rate,
                    max_iter=self._MAX_ITER,
                    tol=self._TOL,
                    return_info=True,
                )
                for i in item_ids:
                    ratings[i][attr] = bt_scores[i]
                s_vec = np.array([bt_scores[i] for i in item_ids])
                se_vec = self._bt_standard_errors(
                    s=s_vec,
                    n_ij=n_ij,
                    p_ij=p_ij,
                    rcond=self._SE_EIGEN_TOL,
                    regularization_strength=self.cfg.learning_rate,
                )
                component_labels = self._comparison_component_labels(n_ij)
                for i, se_val in zip(item_ids, se_vec):
                    se_store[attr][i] = float(se_val)
                    if np.isfinite(se_val):
                        se_agg_next[i] += float(se_val)
                        se_agg_counts[i] += 1
                for i, component in zip(item_ids, component_labels):
                    component_store[attr][i] = int(component)
            for i in item_ids:
                if se_agg_counts[i] > 0:
                    se_agg_next[i] /= se_agg_counts[i]
                else:
                    se_agg_next[i] = 1.0
            self._last_se_agg = se_agg_next
            for attr in attr_keys:
                vals = [ratings[i][attr] for i in item_ids]
                mean_val = float(np.mean(vals))
                for i in item_ids:
                    ratings[i][attr] -= mean_val
            _write_checkpoint()

    async def _run_recursive(
        self,
        df: pd.DataFrame,
        text_column: str,
        *,
        id_column: Optional[str],
        reset_files: bool,
        identifier_hash_bits: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        attr_dict = self._attributes_as_dict()
        attr_list = list(attr_dict.keys())
        if not attr_list:
            raise ValueError("No attributes provided for ranking")
        cut_attr = self.cfg.recursive_cut_attr or attr_list[0]
        if cut_attr not in attr_list:
            raise ValueError(
                f"recursive_cut_attr '{self.cfg.recursive_cut_attr}' not present in attributes"
            )
        cut_side = (self.cfg.recursive_cut_side or "top").lower()
        if cut_side not in {"top", "bottom"}:
            raise ValueError("recursive_cut_side must be 'top' or 'bottom'")
        reserved_input_columns = [
            str(column)
            for column in df.columns
            if str(column).strip().lower()
            in {"identifier", "overall_rank", "exit_stage"}
            or re.match(
                r"^stage\d+_", str(column).strip(), flags=re.IGNORECASE
            )
        ]
        if reserved_input_columns:
            raise ValueError(
                "Recursive Rank input columns cannot use internal output names "
                "or the stage<number>_ prefix: "
                + ", ".join(repr(name) for name in reserved_input_columns)
            )

        work_df = df.reset_index(drop=True).copy()
        strict_text_mode = self.cfg.modality in {"text", "entity", "web"}
        if id_column is not None:
            if id_column not in work_df.columns:
                raise ValueError(f"id_column '{id_column}' not found in DataFrame")
            valid_mask = pd.Series(
                [_is_valid_identifier(value) for value in work_df[id_column]],
                index=work_df.index,
                dtype=bool,
            )
            dropped = len(work_df) - int(valid_mask.sum())
            if dropped > 0:
                total = len(work_df)
                pct = (dropped / total * 100.0) if total else 0.0
                print(
                    f"[Rank] Dropping {dropped}/{total} rows ({pct:.1f}%) with malformed '{id_column}' values in recursive mode."
                )
            work_df = work_df.loc[valid_mask].copy().reset_index(drop=True)
            work_df["identifier"] = work_df[id_column].astype(str)
        else:
            hashed = work_df[text_column].map(
                lambda x: _hash_text_identifier(
                    x,
                    strict=strict_text_mode,
                    bits=identifier_hash_bits,
                )
            )
            valid_mask = hashed.notna()
            dropped = int((~valid_mask).sum())
            if dropped > 0:
                total = len(work_df)
                pct = (dropped / total * 100.0) if total else 0.0
                print(
                    f"[Rank] Dropping {dropped}/{total} rows ({pct:.1f}%) with malformed '{text_column}' values in recursive mode."
                )
            work_df = work_df.loc[valid_mask].copy().reset_index(drop=True)
            work_df["identifier"] = hashed.loc[valid_mask].astype(str).reset_index(drop=True)
        if text_column != "text":
            # Keep the caller's original column for the returned DataFrame and
            # use an internal text view for prompt rendering.
            work_df["text"] = work_df[text_column]
        rewrite_col = self.cfg.recursive_rewrite_text_col or "text"
        if rewrite_col not in work_df.columns:
            work_df[rewrite_col] = work_df["text"]
        work_df["identifier"] = work_df["identifier"].astype(str)

        duplicate_ids = work_df.loc[
            work_df["identifier"].duplicated(keep=False), "identifier"
        ].unique()
        if len(duplicate_ids) > 0:
            preview = ", ".join(repr(value) for value in duplicate_ids[:3])
            raise ValueError(
                "Rank requires a unique identifier for every row; duplicate "
                f"identifier(s): {preview}. Provide a unique id_column when "
                "ranking duplicate content."
            )

        base_folder = os.path.join(
            self.cfg.save_dir, f"{self.cfg.file_name}_recursive"
        )
        base_folder_path = Path(base_folder)
        existing_recursive_artifacts = base_folder_path.exists() and any(
            path.is_file() for path in base_folder_path.rglob("*")
        )
        if work_df.empty:
            empty_out = work_df[[c for c in df.columns if c in work_df.columns]].copy()
            for attr in attr_list:
                empty_out[attr] = pd.Series(dtype="float64")
            empty_out["overall_rank"] = pd.Series(dtype="float64")
            empty_out["exit_stage"] = pd.Series(dtype="float64")
            if existing_recursive_artifacts and not reset_files:
                return empty_out
            os.makedirs(base_folder, exist_ok=True)
            final_file = os.path.join(base_folder, "recursive_final.csv")
            empty_out.to_csv(final_file, index=False)
            return empty_out

        # A requested attribute replaces any same-named input column. Exclude
        # those columns here so the final column selection cannot duplicate the
        # generated stage-relative score. Recursive Rank does not emit the
        # non-recursive raw/SE/component suffixes, so unrelated user suffix
        # columns remain untouched.
        generated_recursive_cols = set(attr_list)
        original_cols = [
            c
            for c in df.columns
            if c in work_df.columns and c not in generated_recursive_cols
        ]
        original_df = work_df[original_cols + ["identifier"]].copy()
        latest_text: Dict[str, str] = {
            ident: txt for ident, txt in zip(work_df["identifier"], work_df["text"])
        }

        os.makedirs(base_folder, exist_ok=True)

        def _compute_stage_zscores(
            stage_df: pd.DataFrame,
        ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
            zscores: Dict[str, Dict[str, float]] = {attr: {} for attr in attr_list}
            scales: Dict[str, float] = {attr: 1.0 for attr in attr_list}
            for attr in attr_list:
                raw_col = f"{attr}_raw"
                component_col = f"{attr}_component"
                # Rank already returns component-wise z-scores. Preserve them
                # instead of globally re-standardizing arbitrary component
                # locations. Rate stages do not have component labels and are
                # normalized here as before.
                if attr in stage_df.columns and component_col in stage_df.columns:
                    source_col = attr
                else:
                    source_col = raw_col if raw_col in stage_df.columns else attr
                if source_col not in stage_df.columns:
                    continue
                series = pd.to_numeric(stage_df[source_col], errors="coerce")
                if source_col == attr and component_col in stage_df.columns:
                    normed = series
                    if raw_col in stage_df.columns:
                        raw_std = pd.to_numeric(
                            stage_df[raw_col], errors="coerce"
                        ).std(ddof=0)
                        scales[attr] = (
                            float(raw_std)
                            if np.isfinite(raw_std) and raw_std > 0
                            else 1.0
                        )
                else:
                    mean = series.mean()
                    std = series.std(ddof=0)
                    if std == 0 or np.isnan(std):
                        normed = pd.Series(
                            [0.0] * len(series), index=stage_df.index
                        )
                        scales[attr] = 1.0
                    else:
                        normed = (series - mean) / std
                        scales[attr] = (
                            float(std) if raw_col in stage_df.columns else 1.0
                        )
                for ident, val in zip(stage_df["identifier"], normed):
                    zscores[attr][str(ident)] = float(val)
            return zscores, scales

        def _require_identifiable_cut_scores(stage_df: pd.DataFrame) -> None:
            component_col = f"{cut_attr}_component"
            if component_col not in stage_df.columns:
                return
            components = pd.to_numeric(
                stage_df[component_col], errors="coerce"
            ).dropna()
            if components.nunique() > 1:
                raise ValueError(
                    "Recursive Rank cannot prune or globally order a disconnected "
                    f"comparison graph for '{cut_attr}'. Component-relative "
                    "Bradley-Terry scores have no identified ordering across "
                    "components. Increase n_rounds or matches_per_round, or use "
                    "a comparison design that connects every surviving item."
                )

        def _select_next_ids(
            active_ids: Sequence[str],
            stage_zs: Dict[str, Dict[str, float]],
        ) -> List[str]:
            n = len(active_ids)
            if n <= self.cfg.recursive_min_remaining:
                return list(active_ids)
            keep_n = min(
                n - 1,
                max(
                    int(math.ceil(n * self.cfg.recursive_fraction)),
                    self.cfg.recursive_min_remaining,
                ),
            )
            scores = {
                item_id: stage_zs.get(cut_attr, {}).get(item_id, 0.0)
                for item_id in active_ids
            }
            ascending = cut_side == "bottom"
            ranked = sorted(
                active_ids,
                key=lambda item_id: scores.get(item_id, 0.0),
                reverse=not ascending,
            )
            return ranked[:keep_n]

        def _maybe_rewrite_texts(
            df_local: pd.DataFrame,
            ids_to_keep: Sequence[str],
            stage_idx: int,
        ) -> pd.DataFrame:
            if self.cfg.recursive_rewrite_func is None:
                return df_local
            mask = df_local["identifier"].isin(ids_to_keep)
            rewritten: List[str] = []
            for _, row in df_local[mask].iterrows():
                new_text = self.cfg.recursive_rewrite_func(
                    row[self.cfg.recursive_rewrite_text_col],
                    row["identifier"],
                    stage_idx,
                )
                rewritten.append(new_text)
                latest_text[str(row["identifier"])] = new_text
            df_local.loc[mask, self.cfg.recursive_rewrite_text_col] = rewritten
            if (
                self.cfg.recursive_rewrite_text_col != "text"
                and "text" in df_local.columns
            ):
                df_local.loc[mask, "text"] = df_local.loc[
                    mask, self.cfg.recursive_rewrite_text_col
                ]
            return df_local

        stage_idx = 0
        final_stage_df: Optional[pd.DataFrame] = None
        stage_z_history: Dict[int, Dict[str, Dict[str, float]]] = {}
        exit_stage: Dict[str, Optional[int]] = {ident: None for ident in work_df["identifier"]}
        current_ids = list(work_df["identifier"])
        stage_primer = self.cfg.primer_scores or None

        while current_ids:
            stage_idx += 1
            n_current = len(current_ids)
            is_final_stage = False
            if n_current <= self.cfg.recursive_min_remaining:
                is_final_stage = True
            else:
                next_keep = min(
                    n_current - 1,
                    max(
                        int(math.ceil(n_current * self.cfg.recursive_fraction)),
                        self.cfg.recursive_min_remaining,
                    ),
                )
                if next_keep <= self.cfg.recursive_min_remaining:
                    is_final_stage = True

            stage_rounds = self.cfg.n_rounds
            if is_final_stage:
                final_multiplier = self.cfg.recursive_final_round_multiplier or 3
                stage_rounds = max(1, stage_rounds * final_multiplier)

            stage_folder = os.path.join(base_folder, f"stage{stage_idx}")
            os.makedirs(stage_folder, exist_ok=True)
            stage_cfg = copy.deepcopy(self.cfg)
            stage_cfg.recursive = False
            stage_cfg.recursive_rate_first_round = False
            stage_cfg.save_dir = stage_folder
            stage_cfg.n_rounds = stage_rounds
            stage_cfg.file_name = self.cfg.file_name
            stage_cfg.rate_kwargs = dict(self.cfg.rate_kwargs)
            stage_cfg.initial_rating_pass = False
            stage_cfg.primer_scores = stage_primer
            stage_cfg.primer_center = False

            stage_df_in = work_df[work_df["identifier"].isin(current_ids)].copy()

            if stage_idx == 1 and self.cfg.recursive_rate_first_round:
                print(
                    "[Rank] Recursive stage 1: running Rate for initial culling "
                    "(disable with recursive_rate_first_round=False)."
                )
                stage_df_out = await self._run_rate_pass(
                    stage_df_in,
                    column_name="text",
                    save_dir=stage_folder,
                    file_name=f"stage{stage_idx}_ratings.csv",
                    reset_files=reset_files,
                    runtime_kwargs=kwargs,
                )
                stage_df_out["identifier"] = stage_df_in["identifier"].values
            else:
                stage_ranker = Rank(stage_cfg, template=self.template)
                stage_df_out = await stage_ranker.run(
                    stage_df_in,
                    column_name="text",
                    id_column="identifier",
                    reset_files=reset_files,
                    **kwargs,
                )

            _require_identifiable_cut_scores(stage_df_out)
            stage_zs, stage_scales = _compute_stage_zscores(stage_df_out)
            stage_z_history[stage_idx] = stage_zs

            if is_final_stage:
                for ident in current_ids:
                    exit_stage[ident] = stage_idx
                final_stage_df = stage_df_out
                break

            next_ids = _select_next_ids(current_ids, stage_zs)
            removed = set(current_ids) - set(next_ids)
            for ident in removed:
                exit_stage[ident] = stage_idx
            stage_primer = {
                ident: {
                    attr: stage_zs.get(attr, {}).get(ident, 0.0) * stage_scales.get(attr, 1.0)
                    for attr in attr_list
                }
                for ident in next_ids
            }
            work_df = _maybe_rewrite_texts(work_df, next_ids, stage_idx)
            current_ids = next_ids

        if final_stage_df is None:
            final_stage_df = work_df[work_df["identifier"].isin(current_ids)].copy()

        # Build final output
        stage_cols: Dict[str, List[Optional[float]]] = {}
        final_attr_cols: Dict[str, List[Optional[float]]] = {a: [] for a in attr_list}
        exit_col: List[Optional[int]] = []

        # build a consolidated map of stage-wise z-scores per identifier
        stage_order = sorted(stage_z_history.keys())
        id_list = list(original_df["identifier"])
        for ident in id_list:
            ident_stage = exit_stage.get(ident)
            exit_col.append(ident_stage)
            final_attr_vals: Dict[str, Optional[float]] = {a: None for a in attr_list}
            for stage in stage_order:
                zs = stage_z_history.get(stage, {})
                for attr in attr_list:
                    col_name = f"stage{stage}_{attr}"
                    stage_cols.setdefault(col_name, []).append(zs.get(attr, {}).get(ident))
                    if ident_stage is not None and stage == ident_stage:
                        final_attr_vals[attr] = zs.get(attr, {}).get(ident)
            for attr in attr_list:
                final_attr_cols[attr].append(final_attr_vals[attr])

        ordered_df = original_df.copy()
        ordered_df[text_column] = ordered_df["identifier"].map(latest_text)
        ordered_df["exit_stage"] = exit_col
        for attr, vals in final_attr_cols.items():
            ordered_df[attr] = vals
        for col, vals in stage_cols.items():
            ordered_df[col] = vals

        # Compute overall ranking: later stages outrank earlier; within a stage, sort by cut_attr z-score
        cut_scores = {i: ordered_df.loc[idx, cut_attr] if cut_attr in ordered_df else None for idx, i in enumerate(id_list)}
        def _rank_key(idx: int) -> Tuple[int, float]:
            ident = id_list[idx]
            stage_num = ordered_df.loc[idx, "exit_stage"] or 0
            score = cut_scores.get(ident)
            if score is None or np.isnan(score):
                score = -np.inf if cut_side == "top" else np.inf
            if cut_side == "bottom":
                score = -score
            return (stage_num, score)

        order_indices = sorted(range(len(id_list)), key=_rank_key, reverse=True)
        ordered_df = ordered_df.iloc[order_indices].reset_index(drop=True)
        ordered_df.insert(0, "overall_rank", range(1, len(ordered_df) + 1))

        final_columns: List[str] = []
        if text_column in original_cols:
            for col in original_cols:
                final_columns.append(col)
                if col == text_column:
                    final_columns.append("overall_rank")
        else:
            final_columns = ["overall_rank"] + [c for c in original_cols]
        for attr in attr_list:
            final_columns.append(attr)
        final_columns.append("exit_stage")
        final_columns.extend(sorted(stage_cols.keys()))
        final_columns = [c for c in final_columns if c in ordered_df.columns and c != "identifier"]
        ordered_df = ordered_df[final_columns]

        final_path = os.path.join(base_folder, "recursive_final.csv")
        ordered_df.to_csv(final_path, index=False)
        return ordered_df

    # ------------------------------------------------------------------
    # Main ranking loop
    # ------------------------------------------------------------------
    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        id_column: Optional[str] = None,
        reset_files: bool = False,
        n_runs: Optional[int] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Execute the ranking procedure.

        Parameters
        ----------
        df:
            Input DataFrame containing the passages to be ranked.
        column_name:
            Name of the column in ``df`` that holds the text for each
            passage.
        id_column:
            Optional name of a column that contains stable identifiers
            for each row. When provided, these identifiers are used to
            track passages across rounds instead of hashing the text
            itself.  Supplying ``id_column`` is recommended when texts
            may be rewritten between stages (e.g., during recursive
            runs).
        reset_files:
            If ``True``, ignore any previously saved results and
            recompute the rankings.  Otherwise, if the final output
            file already exists on disk it will be loaded and returned
            immediately.
        n_runs:
            Deprecated/ignored parameter provided for compatibility
            with :class:`Rate`. When supplied, a message is printed
            noting that ``n_rounds`` controls the number of iterations
            and that ``n_runs`` has no effect.
        **kwargs:
            Additional keyword arguments forwarded to
            :func:`get_all_responses`.  When ``initial_rating_pass`` is
            enabled these arguments are also forwarded to the rating
            stage.  Useful for passing through authentication tokens or
            tracing settings.

        Returns
        -------
        pandas.DataFrame
            In non-recursive mode, a DataFrame with one row per input passage. For each
            attribute the DataFrame contains a ``"<attribute>"`` column
            holding the z‑score, a ``"<attribute>_raw"`` column with the
            centred Bradley–Terry estimate, and a ``"<attribute>_se"``
            column with its model-based sandwich standard error. The
            ``"<attribute>_component"`` column identifies disconnected
            comparison-graph components; z-scores and raw scores are only
            comparable within the same component. Recursive mode instead
            returns stage-relative scores, exit stages, and an overall rank.
            The DataFrame is also written to ``save_dir``.
        """
        has_custom_judge = any(
            kwargs.get(key) is not None
            for key in ("response_fn", "get_all_responses_fn")
        ) or any(
            self.cfg.rate_kwargs.get(key) is not None
            for key in ("response_fn", "get_all_responses_fn")
        )
        if has_custom_judge and self.cfg.judge_version is None:
            raise ValueError(
                "judge_version is required when response_fn or "
                "get_all_responses_fn supplies a custom Rank judge"
            )
        conflicting_response_keys = sorted(
            _RANK_OWNED_RESPONSE_KEYS & kwargs.keys()
        )
        if conflicting_response_keys:
            raise TypeError(
                "Rank owns response setting(s) that cannot be overridden at "
                "runtime: " + ", ".join(conflicting_response_keys)
            )
        kwargs.setdefault("web_search", self.cfg.modality == "web")
        base_name = os.path.splitext(self.cfg.file_name)[0]
        final_path = os.path.join(self.cfg.save_dir, f"{base_name}_final.csv")
        recursive_base_folder = Path(self.cfg.save_dir) / (
            f"{self.cfg.file_name}_recursive"
        )
        initial_rate_folder = Path(self.cfg.save_dir) / f"{base_name}_initial_rate"
        if reset_files:
            if self.cfg.recursive:
                reset_batch_states = (
                    list(recursive_base_folder.rglob("*.batch_state.json"))
                    if recursive_base_folder.exists()
                    else []
                )
            else:
                root = Path(self.cfg.save_dir)
                reset_batch_states = []
                for pattern in (
                    f"{base_name}_round*.csv.batch_state.json",
                    f".{base_name}_round*.csv.batch_state.json",
                    f".{base_name}_catchup_round*.csv.batch_state.json",
                ):
                    reset_batch_states.extend(root.glob(pattern))
            if initial_rate_folder.exists():
                reset_batch_states.extend(
                    initial_rate_folder.rglob("*.batch_state.json")
                )
            for state_path in dict.fromkeys(reset_batch_states):
                try:
                    has_external_work = _rank_batch_state_has_external_work(
                        state_path
                    )
                except Exception as exc:
                    raise ValueError(
                        "Rank cannot safely reset unreadable Batch API state "
                        f"{str(state_path)!r}; reconcile the external job first."
                    ) from exc
                if has_external_work:
                    raise ValueError(
                        "Rank cannot reset while durable Batch API state may "
                        f"reference external work ({str(state_path)!r}). Resume, "
                        "reconcile, or cancel that batch before resetting files."
                    )
            if initial_rate_folder.exists():
                try:
                    shutil.rmtree(initial_rate_folder)
                except OSError as exc:
                    raise OSError(
                        "Could not clear stale Rank initial-rating artifacts in "
                        f"{str(initial_rate_folder)!r}"
                    ) from exc
        if reset_files and self.cfg.recursive and recursive_base_folder.exists():
            try:
                shutil.rmtree(recursive_base_folder)
            except OSError as exc:
                raise OSError(
                    "Could not clear stale recursive Rank artifacts in "
                    f"{str(recursive_base_folder)!r}"
                ) from exc
        if reset_files and not self.cfg.recursive:
            stale_paths = list(
                Path(self.cfg.save_dir).glob(f"{base_name}_round*.csv")
            )
            stale_paths.extend(
                Path(self.cfg.save_dir).glob(
                    f"{base_name}_round*.csv.batch_state.json"
                )
            )
            stale_paths.extend(
                Path(self.cfg.save_dir).glob(f".{base_name}_round*.csv")
            )
            stale_paths.extend(
                Path(self.cfg.save_dir).glob(
                    f".{base_name}_round*.csv.batch_state.json"
                )
            )
            stale_paths.extend(
                Path(self.cfg.save_dir).glob(
                    f".{base_name}_round*_plan.json"
                )
            )
            stale_paths.extend(
                Path(self.cfg.save_dir).glob(
                    f".{base_name}_catchup_round*.csv"
                )
            )
            stale_paths.extend(
                Path(self.cfg.save_dir).glob(
                    f".{base_name}_catchup_round*.csv.batch_state.json"
                )
            )
            stale_paths.extend(
                Path(self.cfg.save_dir).glob(
                    f".{base_name}_catchup_round*_plan.json"
                )
            )
            stale_paths.extend(
                [
                    Path(final_path),
                    Path(self.cfg.save_dir) / f"{base_name}_diagnostics.csv",
                ]
            )
            for stale_path in stale_paths:
                try:
                    stale_path.unlink(missing_ok=True)
                except OSError as exc:
                    raise OSError(
                        f"Could not clear stale Rank artifact {str(stale_path)!r}"
                    ) from exc
        checkpoint_paths = [
            str(path)
            for path in Path(self.cfg.save_dir).glob(f"{base_name}_round*.csv")
        ]
        normal_staging_paths = [
            str(path)
            for path in Path(self.cfg.save_dir).glob(f".{base_name}_round*.csv")
        ]
        normal_plan_paths = [
            str(path)
            for path in Path(self.cfg.save_dir).glob(
                f".{base_name}_round*_plan.json"
            )
        ]
        round_batch_state_paths = [
            str(path)
            for path in Path(self.cfg.save_dir).glob(
                f"{base_name}_round*.csv.batch_state.json"
            )
        ]
        round_batch_state_paths.extend(
            str(path)
            for path in Path(self.cfg.save_dir).glob(
                f".{base_name}_round*.csv.batch_state.json"
            )
        )
        run_metadata = load_run_metadata(
            self.cfg.save_dir, base_name, reset_files=reset_files
        )
        recursive_artifacts_exist = self.cfg.recursive and (
            recursive_base_folder.exists()
            and any(path.is_file() for path in recursive_base_folder.rglob("*"))
        )
        initial_rate_artifacts_exist = (
            initial_rate_folder.exists()
            and any(path.is_file() for path in initial_rate_folder.rglob("*"))
        )
        rank_resume_artifacts_exist = (
            bool(checkpoint_paths)
            or bool(normal_staging_paths)
            or bool(normal_plan_paths)
            or bool(round_batch_state_paths)
            or os.path.exists(final_path)
            or os.path.exists(run_metadata_path(self.cfg.save_dir, base_name))
            or recursive_artifacts_exist
            or initial_rate_artifacts_exist
        )
        identifier_hash_bits = resolve_identifier_hash_bits(
            task_name="Rank",
            metadata=run_metadata,
            reset_files=reset_files,
            checkpoint_paths=checkpoint_paths,
        )
        attribute_sidecar_paths = [
            Path(self.cfg.save_dir) / "attributes.json",
            Path(self.cfg.save_dir) / f"{base_name}_attrs.json",
        ]
        has_readable_attribute_sidecar = False
        if rank_resume_artifacts_exist and not reset_files:
            for attribute_path in attribute_sidecar_paths:
                try:
                    with attribute_path.open(encoding="utf-8") as attribute_file:
                        json.load(attribute_file)
                    has_readable_attribute_sidecar = True
                    break
                except Exception:
                    continue
        self.cfg.attributes = load_persisted_attributes(
            save_dir=self.cfg.save_dir,
            incoming=self.cfg.attributes,
            reset_files=reset_files,
            task_name="Rank",
            item_name="attributes",
            legacy_filename=f"{base_name}_attrs.json",
            persist_missing=reset_files or not rank_resume_artifacts_exist,
        )
        _validate_rank_attribute_names(
            self.cfg.attributes, recursive=self.cfg.recursive
        )
        measurement_spec_fingerprint = self._measurement_spec_fingerprint(kwargs)
        _validate_rank_resume_metadata(
            run_metadata,
            artifacts_exist=rank_resume_artifacts_exist,
            reset_files=reset_files,
            insufficient_signal_policy=self.cfg.insufficient_signal_policy,
            learning_rate=self.cfg.learning_rate,
            modality=self.cfg.modality,
            measurement_spec_fingerprint=measurement_spec_fingerprint,
        )
        if (
            rank_resume_artifacts_exist
            and not reset_files
            and not has_readable_attribute_sidecar
        ):
            # Restore missing/corrupt attribute sidecars only after metadata has
            # proved that the effective measurement definition is compatible.
            self.cfg.attributes = load_persisted_attributes(
                save_dir=self.cfg.save_dir,
                incoming=self.cfg.attributes,
                reset_files=False,
                task_name="Rank",
                item_name="attributes",
                legacy_filename=f"{base_name}_attrs.json",
                persist_missing=True,
            )
            _validate_rank_attribute_names(
                self.cfg.attributes, recursive=self.cfg.recursive
            )
        if not reset_files:
            for batch_state_path in round_batch_state_paths:
                try:
                    with open(batch_state_path, encoding="utf-8") as state_file:
                        batch_state = json.load(state_file)
                except Exception as exc:
                    raise ValueError(
                        f"Could not read Rank Batch API state {batch_state_path!r}"
                    ) from exc
                if not isinstance(batch_state, dict):
                    raise ValueError(
                        f"Rank Batch API state {batch_state_path!r} is malformed"
                    )
                round_checkpoint_path = batch_state_path.removesuffix(
                    ".batch_state.json"
                )
                batches = batch_state.get("batches", [])
                if not isinstance(batches, list):
                    raise ValueError(
                        f"Rank Batch API state {batch_state_path!r} is malformed"
                    )
                has_active_batch = bool(batch_state.get("batch_id")) or bool(
                    batches
                )
                unresolved_submission = any(
                    not isinstance(batch, dict)
                    or batch.get("status") == "submitting"
                    or not batch.get("batch_id")
                    for batch in batches
                )
                if unresolved_submission:
                    raise ValueError(
                        "Rank found an unresolved Batch API submission. The "
                        "server may already have accepted a paid job; reconcile "
                        "or cancel it before resetting local state."
                    )
                if has_active_batch:
                    normal_plan_path = (
                        f"{round_checkpoint_path.removesuffix('.csv')}_plan.json"
                    )
                    is_journaled_staging = Path(round_checkpoint_path).name.startswith(
                        f".{base_name}_round"
                    ) and os.path.exists(normal_plan_path)
                    if not is_journaled_staging:
                        raise ValueError(
                            "Rank found active Batch API state without a "
                            "recoverable round plan. Reconcile or cancel the "
                            "external batch before resetting local state."
                        )
                    # The durable plan reconstructs the exact prompt IDs, so
                    # get_all_responses can safely poll this submitted batch.
                    continue
                if not os.path.exists(round_checkpoint_path):
                    raise ValueError(
                        "Rank found completed/empty Batch API state without a "
                        "durable response checkpoint. Paid results may already "
                        "exist; use reset_files=True only after confirming that "
                        "it is safe to discard that state."
                    )

        kwargs.setdefault("web_search", self.cfg.modality == "web")
        if self.cfg.recursive:
            current_input_fingerprints = _current_rank_input_fingerprints(
                df,
                payload_column=column_name,
                id_column=id_column,
                modality=self.cfg.modality,
                identifier_hash_bits=identifier_hash_bits,
            )
            if (
                rank_resume_artifacts_exist
                and not reset_files
                and run_metadata.get("recursive_input_fingerprint_version") != 1
            ):
                raise ValueError(
                    "Existing recursive Rank artifacts lack content-aware input "
                    "fingerprints. Use reset_files=True or a new save_dir to "
                    "recompute them safely."
                )
            saved_input_fingerprints = (
                run_metadata.get("input_fingerprints", {})
                if rank_resume_artifacts_exist and not reset_files
                else {}
            )
            if not isinstance(saved_input_fingerprints, dict):
                saved_input_fingerprints = {}
            recursive_transaction_plans: List[Path] = []
            if recursive_base_folder.exists():
                normal_stage_plan = re.compile(
                    rf"^\.{re.escape(base_name)}_round(\d+)_plan\.json$"
                )
                catchup_stage_plan = re.compile(
                    rf"^\.{re.escape(base_name)}_catchup_round(\d+)_plan\.json$"
                )
                for plan_path in recursive_base_folder.rglob("*_plan.json"):
                    stage_metadata = load_run_metadata(
                        str(plan_path.parent), base_name, reset_files=False
                    )
                    marker = stage_metadata.get("last_completed_round", -1)
                    marker = marker if type(marker) is int else -1
                    normal_match = normal_stage_plan.match(plan_path.name)
                    if normal_match is not None:
                        if int(normal_match.group(1)) > marker:
                            recursive_transaction_plans.append(plan_path)
                        continue
                    catchup_match = catchup_stage_plan.match(plan_path.name)
                    if catchup_match is None:
                        recursive_transaction_plans.append(plan_path)
                        continue
                    round_index = int(catchup_match.group(1))
                    round_path = plan_path.parent / (
                        f"{base_name}_round{round_index}.csv"
                    )
                    try:
                        plan_payload = json.loads(plan_path.read_text())
                        records = plan_payload["records"]
                        if not isinstance(records, list) or not records:
                            raise ValueError("malformed catch-up plan records")
                        planned_response_ids = {
                            str(record["identifier"]) for record in records
                        }
                        if len(planned_response_ids) != len(records):
                            raise ValueError("duplicate catch-up plan identifiers")
                        committed_response_ids = set(
                            pd.read_csv(
                                round_path,
                                usecols=["Identifier"],
                                dtype=str,
                                keep_default_na=False,
                            )["Identifier"].astype(str)
                        )
                    except Exception:
                        recursive_transaction_plans.append(plan_path)
                        continue
                    committed_plan_ids = (
                        planned_response_ids & committed_response_ids
                    )
                    if committed_plan_ids == planned_response_ids:
                        continue
                    if committed_plan_ids:
                        raise ValueError(
                            "Recursive Rank found a partially committed stage "
                            "catch-up plan"
                        )
                    recursive_transaction_plans.append(plan_path)
            recursive_batch_states = (
                list(recursive_base_folder.rglob("*.batch_state.json"))
                if recursive_base_folder.exists()
                else []
            )
            if initial_rate_folder.exists():
                recursive_batch_states.extend(
                    initial_rate_folder.rglob("*.batch_state.json")
                )
            has_active_recursive_batch = any(
                _rank_batch_state_has_external_work(path)
                for path in recursive_batch_states
            )
            if (
                (recursive_transaction_plans or has_active_recursive_batch)
                and set(current_input_fingerprints)
                != set(saved_input_fingerprints)
            ):
                raise ValueError(
                    "Recursive Rank has an unfinished stage transaction. "
                    "Restore the exact top-level input set before resuming."
                )
            changed_ids = [
                item_id
                for item_id, fingerprint in current_input_fingerprints.items()
                if item_id in saved_input_fingerprints
                and saved_input_fingerprints[item_id] != fingerprint
            ]
            if changed_ids:
                preview = ", ".join(repr(value) for value in changed_ids[:3])
                raise ValueError(
                    "Recursive Rank comparison payload changed for persisted "
                    f"identifier(s): {preview}. Existing stage judgments are "
                    "no longer valid; use reset_files=True or a new save_dir."
                )
            merged_input_fingerprints = {
                **saved_input_fingerprints,
                **current_input_fingerprints,
            }
            update_run_metadata(
                self.cfg.save_dir,
                base_name,
                strict=True,
                task="Rank",
                output_base_name=base_name,
                model=self.cfg.model,
                identifier_hash_bits=identifier_hash_bits,
                rank_estimator_version=_RANK_ESTIMATOR_VERSION,
                insufficient_signal_policy=self.cfg.insufficient_signal_policy,
                learning_rate=self.cfg.learning_rate,
                last_completed_round=-1,
                modality=self.cfg.modality,
                input_fingerprints=merged_input_fingerprints,
                recursive_input_fingerprint_version=1,
                measurement_spec_fingerprint=measurement_spec_fingerprint,
                recursive=True,
            )
            return await self._run_recursive(
                df,
                column_name,
                id_column=id_column,
                reset_files=reset_files,
                identifier_hash_bits=identifier_hash_bits,
                **kwargs,
            )

        # prepare file paths
        if n_runs is not None:
            print(
                "Parameter 'n_runs' is ignored. Use 'n_rounds' to control the number of iterations. "
                f"Current n_rounds={self.cfg.n_rounds}."
            )

        df_proc = df.reset_index(drop=True).copy()
        warn_if_modality_mismatch(df_proc[column_name].tolist(), self.cfg.modality, column_name=column_name)
        strict_text_mode = self.cfg.modality in {"text", "entity", "web"}
        if id_column is not None:
            if id_column not in df_proc.columns:
                raise ValueError(f"id_column '{id_column}' not found in DataFrame")
            valid_mask = pd.Series(
                [_is_valid_identifier(value) for value in df_proc[id_column]],
                index=df_proc.index,
                dtype=bool,
            )
            dropped = len(df_proc) - int(valid_mask.sum())
            if dropped > 0:
                total = len(df_proc)
                pct = (dropped / total * 100.0) if total else 0.0
                print(
                    f"[Rank] Dropping {dropped}/{total} rows ({pct:.1f}%) with malformed '{id_column}' values."
                )
            df_proc = df_proc.loc[valid_mask].copy().reset_index(drop=True)
            df_proc["_id"] = df_proc[id_column].astype(str)
        else:
            hashed_ids = df_proc[column_name].map(
                lambda x: _hash_text_identifier(
                    x,
                    strict=strict_text_mode,
                    bits=identifier_hash_bits,
                )
            )
            valid_mask = hashed_ids.notna()
            dropped = int((~valid_mask).sum())
            if dropped > 0:
                total = len(df_proc)
                pct = (dropped / total * 100.0) if total else 0.0
                print(
                    f"[Rank] Dropping {dropped}/{total} rows ({pct:.1f}%) with malformed '{column_name}' values before ranking."
                )
            df_proc = df_proc.loc[valid_mask].copy().reset_index(drop=True)
            df_proc["_id"] = hashed_ids.loc[valid_mask].astype(str).reset_index(drop=True)

        if df_proc.empty:
            if isinstance(self.cfg.attributes, dict):
                attr_keys_empty = list(self.cfg.attributes.keys())
            else:
                attr_keys_empty = list(self.cfg.attributes)
            df_out = df_proc[[c for c in df.columns if c in df_proc.columns]].copy()
            for attr in attr_keys_empty:
                df_out[attr] = pd.Series(dtype="float64")
                df_out[f"{attr}_raw"] = pd.Series(dtype="float64")
                df_out[f"{attr}_se"] = pd.Series(dtype="float64")
                df_out[f"{attr}_component"] = pd.Series(dtype="int64")
            if rank_resume_artifacts_exist and not reset_files:
                # An empty view of an existing tournament must not erase its
                # checkpoints, fingerprints, or committed-round marker.
                return df_out
            empty_attr_items = [(attr, attr) for attr in attr_keys_empty]
            empty_batches, effective_empty_batch_size = resolve_attribute_batches(
                task_name="Rank",
                items=empty_attr_items,
                requested_n=self.cfg.n_attributes_per_run,
                metadata=run_metadata,
                reset_files=reset_files,
                checkpoint_paths=checkpoint_paths,
            )
            write_task_run_metadata(
                save_dir=self.cfg.save_dir,
                base_name=base_name,
                task_name="Rank",
                model=self.cfg.model,
                identifier_hash_bits=identifier_hash_bits,
                n_attributes_per_run=effective_empty_batch_size,
                attribute_batches=empty_batches,
                strict=True,
                rank_estimator_version=_RANK_ESTIMATOR_VERSION,
                insufficient_signal_policy=self.cfg.insufficient_signal_policy,
                learning_rate=self.cfg.learning_rate,
                last_completed_round=-1,
                modality=self.cfg.modality,
                input_fingerprints={},
                measurement_spec_fingerprint=measurement_spec_fingerprint,
            )
            df_out.to_csv(final_path, index=False)
            pd.DataFrame(
                {
                    "attribute": attr_keys_empty,
                    "circle_count": 0,
                    "square_count": 0,
                    "draw_count": 0,
                    "insufficient_signal_count": 0,
                    "invalid_count": 0,
                    "effective_comparison_weight": 0.0,
                    "comparison_components": 0,
                    "isolated_items": 0,
                    "finite_standard_errors": 0,
                }
            ).to_csv(
                os.path.join(self.cfg.save_dir, f"{base_name}_diagnostics.csv"),
                index=False,
            )
            return df_out

        duplicate_ids = df_proc.loc[
            df_proc["_id"].duplicated(keep=False), "_id"
        ].unique()
        if len(duplicate_ids) > 0:
            preview = ", ".join(repr(value) for value in duplicate_ids[:3])
            raise ValueError(
                "Rank requires a unique identifier for every row; duplicate "
                f"identifier(s): {preview}. Provide a unique id_column when "
                "ranking duplicate content."
            )
        current_input_fingerprints = _rank_input_fingerprints(
            df_proc,
            id_column="_id",
            payload_column=column_name,
            modality=self.cfg.modality,
        )
        saved_input_fingerprints = (
            run_metadata.get("input_fingerprints", {})
            if rank_resume_artifacts_exist
            else {}
        )
        if not isinstance(saved_input_fingerprints, dict):
            saved_input_fingerprints = {}
        initial_rate_batch_states = (
            list(initial_rate_folder.rglob("*.batch_state.json"))
            if initial_rate_folder.exists()
            else []
        )
        if (
            any(
                _rank_batch_state_has_external_work(path)
                for path in initial_rate_batch_states
            )
            and set(current_input_fingerprints) != set(saved_input_fingerprints)
        ):
            raise ValueError(
                "Rank has an unfinished initial-rating Batch transaction. "
                "Restore the exact input set before resuming."
            )
        completed_marker = run_metadata.get("last_completed_round", -1)
        completed_marker = (
            completed_marker if type(completed_marker) is int else -1
        )
        persisted_endpoint_ids: Set[str] = set()
        current_id_set = set(current_input_fingerprints)
        saved_attribute_batches = run_metadata.get("attribute_batches")
        saved_batch_count = (
            len(saved_attribute_batches)
            if isinstance(saved_attribute_batches, list)
            and saved_attribute_batches
            else 0
        )

        # A durable plan is a transaction intent.  Validate its input universe
        # before merging new fingerprints into metadata; otherwise a rejected
        # accidental superset can permanently bind never-judged payloads.
        normal_plan_pattern = re.compile(
            rf"^\.{re.escape(base_name)}_round(\d+)_plan\.json$"
        )
        for raw_plan_path in normal_plan_paths:
            plan_path = Path(raw_plan_path)
            match = normal_plan_pattern.match(plan_path.name)
            if match is None:
                raise ValueError(f"Rank round plan {str(plan_path)!r} is noncanonical")
            round_index = int(match.group(1))
            try:
                with plan_path.open(encoding="utf-8") as plan_file:
                    plan = json.load(plan_file)
            except Exception as exc:
                raise ValueError(
                    f"Could not read Rank round plan {str(plan_path)!r}"
                ) from exc
            planned_ids = plan.get("item_ids") if isinstance(plan, dict) else None
            records = plan.get("records") if isinstance(plan, dict) else None
            if (
                not isinstance(plan, dict)
                or plan.get("version") != 1
                or plan.get("round") != round_index
                or not isinstance(planned_ids, list)
                or any(not isinstance(item_id, str) for item_id in planned_ids)
                or len(planned_ids) != len(set(planned_ids))
                or not isinstance(records, list)
                or not records
            ):
                raise ValueError(
                    f"Rank round plan {str(plan_path)!r} is incompatible or malformed"
                )
            round_path = Path(self.cfg.save_dir) / (
                f"{base_name}_round{round_index}.csv"
            )
            promotion_safe = False
            if round_path.exists() and round_index <= completed_marker:
                promotion_safe = True
            elif round_path.exists() and round_index == completed_marker + 1:
                if saved_batch_count == 0:
                    raise ValueError(
                        "Rank cannot validate an uncommitted round without its "
                        "persisted attribute-batch schema"
                    )
                try:
                    self._read_rank_checkpoint(str(round_path), saved_batch_count)
                except Exception as exc:
                    raise ValueError(
                        "Rank found an uncommitted round checkpoint that is not "
                        "safe to promote"
                    ) from exc
                promotion_safe = True
            if promotion_safe:
                continue
            if (
                round_index != completed_marker + 1
                or set(planned_ids) != current_id_set
            ):
                raise ValueError(
                    "Rank has an unfinished round plan for a different input "
                    "set. Restore the exact planned items before resuming."
                )
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Rank round plan {str(plan_path)!r} has malformed records"
                    )
                for key in ("id_a", "id_b"):
                    endpoint = record.get(key)
                    if not isinstance(endpoint, str) or endpoint not in current_id_set:
                        raise ValueError(
                            f"Rank round plan {str(plan_path)!r} has invalid endpoints"
                        )
                    persisted_endpoint_ids.add(endpoint)

        catchup_plan_pattern = re.compile(
            rf"^\.{re.escape(base_name)}_catchup_round(\d+)_plan\.json$"
        )
        catchup_plan_paths = list(
            Path(self.cfg.save_dir).glob(
                f".{base_name}_catchup_round*_plan.json"
            )
        )
        for plan_path in catchup_plan_paths:
            match = catchup_plan_pattern.match(plan_path.name)
            if match is None:
                raise ValueError(
                    f"Rank catch-up plan {str(plan_path)!r} is noncanonical"
                )
            round_index = int(match.group(1))
            try:
                with plan_path.open(encoding="utf-8") as plan_file:
                    plan = json.load(plan_file)
            except Exception as exc:
                raise ValueError(
                    f"Could not read Rank catch-up plan {str(plan_path)!r}"
                ) from exc
            planned_ids = plan.get("item_ids") if isinstance(plan, dict) else None
            records = plan.get("records") if isinstance(plan, dict) else None
            if (
                not isinstance(plan, dict)
                or plan.get("version") != 1
                or plan.get("round") != round_index
                or not isinstance(planned_ids, list)
                or any(not isinstance(item_id, str) for item_id in planned_ids)
                or len(planned_ids) != len(set(planned_ids))
                or not isinstance(records, list)
                or not records
            ):
                raise ValueError(
                    f"Rank catch-up plan {str(plan_path)!r} is incompatible or malformed"
                )
            round_path = Path(self.cfg.save_dir) / (
                f"{base_name}_round{round_index}.csv"
            )
            try:
                committed_ids = set(
                    pd.read_csv(
                        round_path,
                        usecols=["Identifier"],
                        dtype=str,
                        keep_default_na=False,
                    )["Identifier"].astype(str)
                )
            except Exception as exc:
                raise ValueError(
                    "Rank catch-up plan does not have a readable committed "
                    f"round checkpoint: {str(plan_path)!r}"
                ) from exc
            plan_response_ids = {
                str(record.get("identifier"))
                for record in records
                if isinstance(record, dict)
            }
            if len(plan_response_ids) != len(records) or "None" in plan_response_ids:
                raise ValueError(
                    f"Rank catch-up plan {str(plan_path)!r} has malformed records"
                )
            already_committed = plan_response_ids & committed_ids
            if already_committed == plan_response_ids:
                continue
            if already_committed:
                raise ValueError("Rank found a partially committed catch-up plan")
            if set(planned_ids) != current_id_set:
                raise ValueError(
                    "Rank has an unfinished catch-up plan for a different input "
                    "set. Restore the exact planned items before resuming."
                )
            for record in records:
                for key in ("id_a", "id_b"):
                    endpoint = record.get(key)
                    if not isinstance(endpoint, str) or endpoint not in current_id_set:
                        raise ValueError(
                            f"Rank catch-up plan {str(plan_path)!r} has invalid endpoints"
                        )
                    persisted_endpoint_ids.add(endpoint)

        round_name_pattern = re.compile(
            rf"^{re.escape(base_name)}_round(\d+)\.csv$"
        )
        for checkpoint_path in checkpoint_paths:
            match = round_name_pattern.match(Path(checkpoint_path).name)
            if match is None or int(match.group(1)) > completed_marker:
                continue
            try:
                endpoint_frame = pd.read_csv(
                    checkpoint_path,
                    usecols=["IdA", "IdB"],
                    dtype=str,
                    keep_default_na=False,
                )
            except Exception as exc:
                raise ValueError(
                    f"Could not verify persisted Rank endpoints in "
                    f"{checkpoint_path!r}"
                ) from exc
            persisted_endpoint_ids.update(endpoint_frame["IdA"].astype(str))
            persisted_endpoint_ids.update(endpoint_frame["IdB"].astype(str))
        missing_or_malformed_fingerprints = sorted(
            item_id
            for item_id in persisted_endpoint_ids
            if not isinstance(saved_input_fingerprints.get(item_id), str)
            or re.fullmatch(
                r"[0-9a-f]{40}", str(saved_input_fingerprints.get(item_id, ""))
            )
            is None
        )
        if missing_or_malformed_fingerprints:
            preview = ", ".join(
                repr(value) for value in missing_or_malformed_fingerprints[:3]
            )
            raise ValueError(
                "Rank metadata lacks valid content fingerprints for persisted "
                f"checkpoint identifier(s): {preview}. Use reset_files=True or "
                "a new save_dir to recompute the tournament safely."
            )
        changed_ids = [
            item_id
            for item_id, fingerprint in current_input_fingerprints.items()
            if item_id in saved_input_fingerprints
            and saved_input_fingerprints[item_id] != fingerprint
        ]
        if changed_ids:
            preview = ", ".join(repr(value) for value in changed_ids[:3])
            raise ValueError(
                "Rank comparison payload changed for persisted identifier(s): "
                f"{preview}. Existing pairwise judgments are no longer valid; "
                "use reset_files=True or a new save_dir."
            )
        merged_input_fingerprints = {
            **saved_input_fingerprints,
            **current_input_fingerprints,
        }
        update_run_metadata(
            self.cfg.save_dir,
            base_name,
            strict=True,
            task="Rank",
            output_base_name=base_name,
            model=self.cfg.model,
            identifier_hash_bits=identifier_hash_bits,
            rank_estimator_version=_RANK_ESTIMATOR_VERSION,
            insufficient_signal_policy=self.cfg.insufficient_signal_policy,
            learning_rate=self.cfg.learning_rate,
            last_completed_round=(
                run_metadata["last_completed_round"]
                if type(run_metadata.get("last_completed_round")) is int
                and run_metadata["last_completed_round"] >= -1
                else -1
            ),
            modality=self.cfg.modality,
            input_fingerprints=merged_input_fingerprints,
            measurement_spec_fingerprint=measurement_spec_fingerprint,
            recursive=False,
        )
        # Determine how many rounds have already been processed when
        # `reset_files` is False.  We look for files named
        # ``<base_name>_round<k>.csv`` to infer progress.  If a final
        # checkpoint exists for the last round, reuse it; otherwise we
        # resume from the next incomplete round.  When ``reset_files``
        # is ``True``, all progress is ignored and the computation
        # restarts from round 0.
        start_round = 0
        existing_rounds: List[int] = []
        if not reset_files:
            try:
                for fname in os.listdir(self.cfg.save_dir):
                    if fname.startswith(f"{base_name}_round") and fname.endswith(
                        ".csv"
                    ):
                        idx_str = fname[
                            len(base_name) + 6 : -4
                        ]  # len("_round") == 6
                        try:
                            rnd_idx = int(idx_str)
                        except (TypeError, ValueError):
                            continue
                        if fname != f"{base_name}_round{rnd_idx}.csv":
                            raise ValueError(
                                "Existing Rank checkpoint has a noncanonical "
                                f"filename: {fname!r}. Rename or remove it, "
                                "or use reset_files=True."
                            )
                        existing_rounds.append(rnd_idx)
            except ValueError:
                raise
            except Exception:
                existing_rounds = []
        completed_round = (
            int(run_metadata.get("last_completed_round", -1))
            if existing_rounds or os.path.exists(final_path)
            else -1
        )
        if existing_rounds:
            existing_rounds = sorted(set(existing_rounds))
            if existing_rounds != list(range(existing_rounds[-1] + 1)):
                raise ValueError(
                    "Existing Rank round checkpoints are not contiguous from "
                    "round 0. Use reset_files=True or a new save_dir to "
                    "recompute them."
                )
            if completed_round > existing_rounds[-1]:
                raise ValueError(
                    "Rank metadata marks a completed round whose checkpoint "
                    "file is missing. Use reset_files=True or a new save_dir "
                    "to recompute the run."
                )
            if existing_rounds[-1] > completed_round + 1:
                raise ValueError(
                    "Rank has more than one uncommitted round checkpoint. "
                    "Use reset_files=True or a new save_dir to recompute the "
                    "run."
                )
            if existing_rounds[-1] == completed_round + 1:
                uncommitted_path = Path(self.cfg.save_dir) / (
                    f"{base_name}_round{completed_round + 1}.csv"
                )
                saved_attribute_batches = run_metadata.get("attribute_batches")
                if (
                    not isinstance(saved_attribute_batches, list)
                    or not saved_attribute_batches
                ):
                    raise ValueError(
                        "Rank found an uncommitted round but cannot validate its "
                        "attribute-batch schema. Use reset_files=True or a new "
                        "save_dir to recompute the run."
                    )
                try:
                    self._read_rank_checkpoint(
                        str(uncommitted_path), len(saved_attribute_batches)
                    )
                except Exception as exc:
                    raise ValueError(
                        "Rank found an uncommitted round checkpoint that is not "
                        "safe to promote. Use reset_files=True or a new save_dir "
                        "to recompute the run."
                    ) from exc
                completed_round += 1
                update_run_metadata(
                    self.cfg.save_dir,
                    base_name,
                    strict=True,
                    last_completed_round=completed_round,
                )
                print(
                    "[Rank] Promoted a complete uncommitted round checkpoint "
                    f"without repeating its judgments (round {completed_round})."
                )
                for completed_artifact in (
                    Path(self.cfg.save_dir)
                    / f".{base_name}_round{completed_round}.csv.batch_state.json",
                    Path(self.cfg.save_dir)
                    / f".{base_name}_round{completed_round}.csv",
                    Path(self.cfg.save_dir)
                    / f".{base_name}_round{completed_round}_plan.json",
                ):
                    completed_artifact.unlink(missing_ok=True)
            if completed_round + 1 > self.cfg.n_rounds:
                raise ValueError(
                    "Rank cannot resume with fewer n_rounds than are already "
                    f"committed (requested={self.cfg.n_rounds}, "
                    f"committed={completed_round + 1}). Use reset_files=True "
                    "or a new save_dir to recompute the shorter tournament."
                )
            start_round = completed_round + 1
        elif completed_round >= 0:
            raise ValueError(
                "Rank metadata marks completed rounds, but no round checkpoint "
                "files exist. Use reset_files=True or a new save_dir to "
                "recompute the run."
            )
        # extract contents and build lookup
        if self.cfg.modality in {"image", "audio", "pdf"}:
            texts = list(zip(df_proc["_id"], ["" for _ in df_proc[column_name]]))
        else:
            texts = list(zip(df_proc["_id"], df_proc[column_name].astype(str)))
        texts_by_id = {i: t for i, t in texts}
        item_ids = [i for i, _ in texts]

        images_by_id: Dict[str, List[str]] = {}
        audio_by_id: Dict[str, List[Dict[str, str]]] = {}
        pdfs_by_id: Dict[str, List[Dict[str, str]]] = {}
        if self.cfg.modality == "image":
            for rid, imgs in zip(df_proc["_id"], df_proc[column_name]):
                encoded = load_image_inputs(imgs)
                if encoded:
                    images_by_id[rid] = encoded
        elif self.cfg.modality == "audio":
            for rid, auds in zip(df_proc["_id"], df_proc[column_name]):
                encoded = load_audio_inputs(auds)
                if encoded:
                    audio_by_id[rid] = encoded
        elif self.cfg.modality == "pdf":
            for rid, pdfs in zip(df_proc["_id"], df_proc[column_name]):
                encoded = load_pdf_inputs(pdfs)
                if encoded:
                    pdfs_by_id[rid] = encoded
        # derive list of attributes
        if isinstance(self.cfg.attributes, dict):
            attr_keys = list(self.cfg.attributes.keys())
        else:
            attr_keys = list(self.cfg.attributes)
        # initialise ratings for each item/attribute
        ratings: Dict[str, Dict[str, float]] = {
            i: {a: 0.0 for a in attr_keys} for i in item_ids
        }
        rate_seed: Dict[str, Dict[str, float]] = {}
        if self.cfg.primer_scores:
            self._apply_primer(ratings, self.cfg.primer_scores, attr_keys)
        if self.cfg.initial_rating_pass and attr_keys and len(item_ids) > 1:
            print(
                "[Rank] Running initial rating pass to seed pairwise comparisons "
                "(disable with initial_rating_pass=False)."
            )
            rate_dir = os.path.join(self.cfg.save_dir, f"{base_name}_initial_rate")
            os.makedirs(rate_dir, exist_ok=True)
            rate_df = await self._run_rate_pass(
                df_proc,
                column_name,
                save_dir=rate_dir,
                file_name=f"{base_name}_initial_rate.csv",
                reset_files=reset_files,
                runtime_kwargs=kwargs,
            )
            rate_seed = self._seed_ratings_from_rate(
                rate_df,
                id_column=id_column,
                text_column=column_name,
                item_ids=item_ids,
                attr_keys=attr_keys,
                identifier_hash_bits=identifier_hash_bits,
            )
            if rate_seed:
                print(
                    "[Rank] Initial rating pass complete. Seeding tournament with "
                    "centred ratings from the rate stage."
                )
            for item_id, attr_map in rate_seed.items():
                for attr, val in attr_map.items():
                    ratings[item_id][attr] = val
        has_seed_ratings = bool(rate_seed)
        # maintain a history of pairwise outcomes for each attribute
        history_pairs: Dict[str, List[WeightedOutcome]] = {
            a: [] for a in attr_keys
        }
        outcome_categories = (
            "circle",
            "square",
            "draw",
            "insufficient_signal",
            "invalid",
        )
        outcome_counts: Dict[str, Dict[str, int]] = {
            attr: {category: 0 for category in outcome_categories}
            for attr in attr_keys
        }
        # store per‑attribute standard errors across items
        se_store: Dict[str, Dict[str, float]] = {
            a: {i: np.nan for i in item_ids} for a in attr_keys
        }
        component_store: Dict[str, Dict[str, int]] = {
            a: {item_id: index for index, item_id in enumerate(item_ids)}
            for a in attr_keys
        }
        # Define attribute batches once to reuse across replay and new rounds
        attr_items = [(attr, attr) for attr in attr_keys]
        attr_batch_items, effective_n_attributes_per_run = resolve_attribute_batches(
            task_name="Rank",
            items=attr_items,
            requested_n=self.cfg.n_attributes_per_run,
            metadata=run_metadata,
            reset_files=reset_files,
            checkpoint_paths=checkpoint_paths,
        )
        attr_batches: List[List[str]] = [
            [name for name, _ in batch] for batch in attr_batch_items
        ]
        attr_count = len(attr_keys)
        if (
            effective_n_attributes_per_run is not None
            and attr_count > effective_n_attributes_per_run
        ):
            batches = (
                attr_count + effective_n_attributes_per_run - 1
            ) // effective_n_attributes_per_run
            print(
                f"[Rank] {attr_count} attributes provided. n_attributes_per_run={effective_n_attributes_per_run}. "
                f"Splitting into {batches} prompt batches; set n_attributes_per_run=None to process them together."
            )
        write_task_run_metadata(
            save_dir=self.cfg.save_dir,
            base_name=base_name,
            task_name="Rank",
            model=self.cfg.model,
            identifier_hash_bits=identifier_hash_bits,
            n_attributes_per_run=effective_n_attributes_per_run,
            attribute_batches=attr_batch_items,
            strict=True,
            rank_estimator_version=_RANK_ESTIMATOR_VERSION,
            insufficient_signal_policy=self.cfg.insufficient_signal_policy,
            learning_rate=self.cfg.learning_rate,
            last_completed_round=completed_round,
            modality=self.cfg.modality,
            input_fingerprints=merged_input_fingerprints,
            measurement_spec_fingerprint=measurement_spec_fingerprint,
        )

        # Helper function to write the current results to the final CSV.  This
        # builds the output DataFrame from the current ``df_proc`` and
        # ``ratings``/``se_store``/``zscores`` and writes it to
        # ``final_path``.
        checkpoint_result: Optional[pd.DataFrame] = None

        def _write_checkpoint() -> None:
            nonlocal checkpoint_result
            # Compute z-scores within observed comparison components. A global
            # normalization would make even within-component values change when
            # an unrelated disconnected group is added.
            zscores_local: Dict[str, Dict[str, float]] = {}
            for attr in attr_keys:
                vals = np.array([ratings[i][attr] for i in item_ids])
                components = np.array(
                    [component_store[attr].get(i, -1) for i in item_ids],
                    dtype=int,
                )
                zscores = self._component_zscores(vals, components)
                zscores_local[attr] = {
                    i: float(value) for i, value in zip(item_ids, zscores)
                }
            # Merge computed results back into the original DataFrame copy.
            for attr in attr_keys:
                raw_col = f"{attr}_raw"
                # ratings
                component_sizes: Dict[int, int] = {}
                for component in component_store[attr].values():
                    component_sizes[component] = component_sizes.get(component, 0) + 1
                val_map = {
                    i: (
                        ratings[i][attr]
                        if component_sizes.get(component_store[attr].get(i, -1), 0)
                        > 1
                        else np.nan
                    )
                    for i in item_ids
                }
                df_proc[raw_col] = df_proc["_id"].map(val_map)
                # standard errors
                se_map = {i: se_store[attr].get(i, np.nan) for i in item_ids}
                df_proc[f"{attr}_se"] = df_proc["_id"].map(se_map)
                # Connected-component labels make it explicit when scores and
                # standard errors are only comparable within a graph component.
                component_map = {
                    i: component_store[attr].get(i, -1) for i in item_ids
                }
                df_proc[f"{attr}_component"] = df_proc["_id"].map(component_map)
                # z‑scores
                z_map = zscores_local.get(attr, {i: np.nan for i in item_ids})
                df_proc[attr] = df_proc["_id"].map(z_map)

            # Reorder columns: original user columns first (excluding the internal ``_id``),
            # then for each attribute the z‑score column followed by raw scores and
            # standard errors.
            generated_rank_cols = {
                generated
                for attr in attr_keys
                for generated in (
                    attr,
                    f"{attr}_raw",
                    f"{attr}_se",
                    f"{attr}_component",
                )
            }
            original_cols = [
                c for c in df.columns if c not in generated_rank_cols
            ]  # preserve the order of unaffected user columns
            new_cols: List[str] = []
            for attr in attr_keys:
                new_cols.append(attr)
                new_cols.append(f"{attr}_raw")
                new_cols.append(f"{attr}_se")
                new_cols.append(f"{attr}_component")
            final_cols = original_cols + new_cols
            final_cols = [c for c in final_cols if c in df_proc.columns]
            df_out_local = df_proc[final_cols].copy()
            checkpoint_result = df_out_local
            # Write the final results to disk in CSV format.  Using CSV avoids
            # Excel row limits and unnecessary overhead.
            df_out_local.to_csv(final_path, index=False)

            diagnostic_rows: List[Dict[str, Any]] = []
            for attr in attr_keys:
                components = list(component_store[attr].values())
                component_sizes: Dict[int, int] = {}
                for component in components:
                    component_sizes[component] = component_sizes.get(component, 0) + 1
                counts = outcome_counts[attr]
                diagnostic_rows.append(
                    {
                        "attribute": attr,
                        **{f"{category}_count": counts[category] for category in outcome_categories},
                        "effective_comparison_weight": float(
                            sum(weight for _, _, weight in history_pairs[attr])
                        ),
                        "comparison_components": len(component_sizes),
                        "isolated_items": sum(
                            size == 1 for size in component_sizes.values()
                        ),
                        "finite_standard_errors": sum(
                            np.isfinite(se_store[attr].get(item_id, np.nan))
                            for item_id in item_ids
                        ),
                    }
                )
            pd.DataFrame(diagnostic_rows).to_csv(
                os.path.join(self.cfg.save_dir, f"{base_name}_diagnostics.csv"),
                index=False,
            )

        if len(item_ids) < 2:
            _write_checkpoint()
            assert checkpoint_result is not None
            return checkpoint_result.copy()

        # If there are completed rounds and we're resuming, replay them to
        # reconstruct the ratings and uncertainties.  After each replayed
        # round we write a checkpoint to ``final_path``.
        if start_round > 0:
            for replay_rnd in range(start_round):
                round_path = os.path.join(
                    self.cfg.save_dir, f"{base_name}_round{replay_rnd}.csv"
                )
                if not os.path.exists(round_path):
                    break
                try:
                    df_round = self._read_rank_checkpoint(
                        round_path, len(attr_batches)
                    )
                    df_round["Response"] = df_round["Response"].apply(
                        lambda x: None if pd.isna(x) else x
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Could not replay Rank checkpoint {round_path!r}. "
                        "Use reset_files=True or a new save_dir to recompute it."
                    ) from exc
                replay_meta = {
                    str(identifier): (
                        int(batch_idx),
                        int(pair_idx),
                        str(id_a),
                        str(id_b),
                    )
                    for identifier, batch_idx, pair_idx, id_a, id_b in zip(
                        df_round["Identifier"],
                        df_round["Batch"],
                        df_round["Pair"],
                        df_round["IdA"],
                        df_round["IdB"],
                    )
                }
                await self._validate_pairwise_response_payloads(
                    df_round,
                    replay_meta,
                    attr_batches,
                    context=f"Rank checkpoint {round_path!r}",
                )

                # Parse each response to build history_pairs
                async def _coerce_dict_replay(raw: Any) -> Dict[str, Any]:
                    obj = await safest_json(raw)
                    if isinstance(obj, dict):
                        return obj
                    if isinstance(obj, str):
                        obj2 = await safest_json(obj)
                        if isinstance(obj2, dict):
                            return obj2
                    if isinstance(obj, list) and obj:
                        inner = await safest_json(obj[0])
                        if isinstance(inner, dict):
                            return inner
                    return {}

                if {"Batch", "IdA", "IdB"}.issubset(df_round.columns):
                    for batch_idx_raw, id_a, id_b, resp_raw in zip(
                        df_round["Batch"],
                        df_round["IdA"],
                        df_round["IdB"],
                        df_round["Response"],
                    ):
                        if pd.isna(batch_idx_raw):
                            continue
                        try:
                            batch_idx = int(batch_idx_raw)
                        except (TypeError, ValueError):
                            continue
                        id_a = str(id_a)
                        id_b = str(id_b)
                        if id_a not in ratings or id_b not in ratings:
                            continue
                        if batch_idx < 0 or batch_idx >= len(attr_batches):
                            continue
                        batch = attr_batches[batch_idx]
                        batch_attr_map = {str(k).strip().lower(): k for k in batch}
                        safe_obj = await _coerce_dict_replay(resp_raw)
                        if not safe_obj:
                            continue
                        for attr_raw, winner_raw in safe_obj.items():
                            attr_key_l = str(attr_raw).strip().lower()
                            if attr_key_l not in batch_attr_map:
                                continue
                            real_attr = batch_attr_map[attr_key_l]
                            category = self._record_pairwise_outcome(
                                history_pairs[real_attr], id_a, id_b, winner_raw
                            )
                            outcome_counts[real_attr][category] += 1
                else:
                    for ident, resp_raw in zip(
                        df_round["Identifier"], df_round["Response"]
                    ):
                        parts = str(ident).split("|")
                        if len(parts) != 5:
                            continue
                        _, batch_idx_str, _, id_a, id_b = parts
                        id_a = str(id_a)
                        id_b = str(id_b)
                        try:
                            batch_idx = int(batch_idx_str)
                        except (TypeError, ValueError):
                            continue
                        if id_a not in ratings or id_b not in ratings:
                            continue
                        if batch_idx < 0 or batch_idx >= len(attr_batches):
                            continue
                        batch = attr_batches[batch_idx]
                        batch_attr_map = {str(k).strip().lower(): k for k in batch}
                        safe_obj = await _coerce_dict_replay(resp_raw)
                        if not safe_obj:
                            continue
                        for attr_raw, winner_raw in safe_obj.items():
                            attr_key_l = str(attr_raw).strip().lower()
                            if attr_key_l not in batch_attr_map:
                                continue
                            real_attr = batch_attr_map[attr_key_l]
                            category = self._record_pairwise_outcome(
                                history_pairs[real_attr], id_a, id_b, winner_raw
                            )
                            outcome_counts[real_attr][category] += 1
                # After parsing all pairs for this round, update ratings
                se_agg_next: Dict[str, float] = {i: 0.0 for i in item_ids}
                se_agg_counts: Dict[str, int] = {i: 0 for i in item_ids}
                for attr in attr_keys:
                    outcomes = history_pairs[attr]
                    if len(outcomes) == 0:
                        continue
                    bt_scores, n_ij, p_ij = self._fit_bt(
                        item_ids=item_ids,
                        outcomes=outcomes,
                        pseudo=self.cfg.learning_rate,
                        max_iter=self._MAX_ITER,
                        tol=self._TOL,
                        return_info=True,
                    )
                    for i in item_ids:
                        ratings[i][attr] = bt_scores[i]
                    s_vec = np.array([bt_scores[i] for i in item_ids])
                    se_vec = self._bt_standard_errors(
                        s=s_vec,
                        n_ij=n_ij,
                        p_ij=p_ij,
                        rcond=self._SE_EIGEN_TOL,
                        regularization_strength=self.cfg.learning_rate,
                    )
                    component_labels = self._comparison_component_labels(n_ij)
                    for i, se_val in zip(item_ids, se_vec):
                        se_store[attr][i] = float(se_val)
                        if np.isfinite(se_val):
                            se_agg_next[i] += float(se_val)
                            se_agg_counts[i] += 1
                    for i, component in zip(item_ids, component_labels):
                        component_store[attr][i] = int(component)
                for i in item_ids:
                    if se_agg_counts[i] > 0:
                        se_agg_next[i] /= se_agg_counts[i]
                    else:
                        se_agg_next[i] = 1.0
                self._last_se_agg = se_agg_next
                # Centre ratings to zero mean for each attribute
                for attr in attr_keys:
                    vals = [ratings[i][attr] for i in item_ids]
                    mean_val = float(np.mean(vals))
                    for i in item_ids:
                        ratings[i][attr] -= mean_val
                # Write checkpoint after this replayed round
                _write_checkpoint()

        # Determine if any new items were added and need to catch up on existing rounds
        await self._catch_up_existing_rounds(
            candidate_ids=item_ids,
            round_indices=list(range(start_round)),
            item_ids=item_ids,
            texts_by_id=texts_by_id,
            images_by_id=images_by_id,
            audio_by_id=audio_by_id,
            pdfs_by_id=pdfs_by_id,
            attr_batches=attr_batches,
            attr_keys=attr_keys,
            history_pairs=history_pairs,
            outcome_counts=outcome_counts,
            ratings=ratings,
            se_store=se_store,
            component_store=component_store,
            base_name=base_name,
            df_proc=df_proc,
            _write_checkpoint=_write_checkpoint,
            current_ratings=None,
            se_agg_local=self._last_se_agg,
            reset_files=reset_files,
            identifier_hash_bits=identifier_hash_bits,
            **kwargs,
        )

        # Now proceed with new rounds starting from ``start_round``
        for rnd in range(start_round, self.cfg.n_rounds):
            # aggregate current ratings across attributes for pairing
            current_agg = {
                i: float(np.mean(list(ratings[i].values()))) for i in item_ids
            }
            se_agg_local = self._last_se_agg
            use_current = rnd > 0 or start_round > 0 or has_seed_ratings
            se_source = se_agg_local if (rnd > 0 or start_round > 0 or se_agg_local is not None) else None
            round_path = os.path.join(
                self.cfg.save_dir, f"{base_name}_round{rnd}.csv"
            )
            staging_path = os.path.join(
                self.cfg.save_dir, f".{base_name}_round{rnd}.csv"
            )
            plan_path = os.path.join(
                self.cfg.save_dir, f".{base_name}_round{rnd}_plan.json"
            )
            batch_state_path = f"{staging_path}.batch_state.json"
            plan_records: Optional[List[Dict[str, Any]]] = None
            canonical_item_ids = sorted(item_ids)
            if os.path.exists(plan_path):
                try:
                    with open(plan_path, encoding="utf-8") as plan_file:
                        plan_payload = json.load(plan_file)
                except Exception as exc:
                    raise ValueError(
                        f"Could not read Rank round plan {plan_path!r}"
                    ) from exc
                planned_item_ids = (
                    plan_payload.get("item_ids")
                    if isinstance(plan_payload, dict)
                    else None
                )
                if (
                    not isinstance(plan_payload, dict)
                    or plan_payload.get("version") != 1
                    or plan_payload.get("round") != rnd
                    or plan_payload.get("attribute_batches") != attr_batches
                    or not isinstance(planned_item_ids, list)
                    or any(
                        not isinstance(item_id, str)
                        for item_id in planned_item_ids
                    )
                    or len(planned_item_ids) != len(set(planned_item_ids))
                    or sorted(planned_item_ids) != canonical_item_ids
                    or not isinstance(plan_payload.get("records"), list)
                    or not plan_payload["records"]
                ):
                    raise ValueError(
                        f"Rank round plan {plan_path!r} is incompatible or malformed"
                    )
                candidate_records = plan_payload["records"]
                required_record_keys = {
                    "identifier",
                    "batch",
                    "pair",
                    "id_a",
                    "id_b",
                    "circle_first",
                }
                pair_endpoints: Dict[int, Tuple[str, str]] = {}
                pair_indices_by_batch: Dict[int, Set[int]] = {
                    batch_idx: set() for batch_idx in range(len(attr_batches))
                }
                planned_identifiers: List[str] = []
                for record in candidate_records:
                    if (
                        not isinstance(record, dict)
                        or not required_record_keys.issubset(record)
                        or type(record["batch"]) is not int
                        or type(record["pair"]) is not int
                        or type(record["circle_first"]) is not bool
                    ):
                        raise ValueError(
                            f"Rank round plan {plan_path!r} has malformed records"
                        )
                    batch_idx = record["batch"]
                    pair_idx = record["pair"]
                    id_a = str(record["id_a"])
                    id_b = str(record["id_b"])
                    if (
                        batch_idx not in pair_indices_by_batch
                        or pair_idx < 0
                        or id_a == id_b
                        or id_a not in texts_by_id
                        or id_b not in texts_by_id
                    ):
                        raise ValueError(
                            f"Rank round plan {plan_path!r} has invalid endpoints"
                        )
                    endpoints = (id_a, id_b)
                    if (
                        pair_idx in pair_endpoints
                        and pair_endpoints[pair_idx] != endpoints
                    ):
                        raise ValueError(
                            f"Rank round plan {plan_path!r} changes pair endpoints"
                        )
                    pair_endpoints[pair_idx] = endpoints
                    if pair_idx in pair_indices_by_batch[batch_idx]:
                        raise ValueError(
                            f"Rank round plan {plan_path!r} duplicates a pair"
                        )
                    pair_indices_by_batch[batch_idx].add(pair_idx)
                    expected_identifier = hash_identifier(
                        f"{rnd}|{batch_idx}|{pair_idx}|{id_a}|{id_b}",
                        bits=identifier_hash_bits,
                    )
                    if str(record["identifier"]) != expected_identifier:
                        raise ValueError(
                            f"Rank round plan {plan_path!r} has an invalid identifier"
                        )
                    planned_identifiers.append(expected_identifier)
                if (
                    len(planned_identifiers) != len(set(planned_identifiers))
                    or len(set(map(frozenset, pair_indices_by_batch.values())))
                    != 1
                ):
                    raise ValueError(
                        f"Rank round plan {plan_path!r} is incomplete or colliding"
                    )
                plan_records = candidate_records
            else:
                pairs = self._generate_pairs(
                    item_ids=item_ids,
                    texts_by_id=texts_by_id,
                    current_ratings=current_agg if use_current else None,
                    se_agg=se_source,
                )
                if not pairs:
                    break
                plan_records = []
                for batch_idx, _batch in enumerate(attr_batches):
                    for pair_idx, ((id_a, _), (id_b, _)) in enumerate(pairs):
                        raw_identifier = (
                            f"{rnd}|{batch_idx}|{pair_idx}|{id_a}|{id_b}"
                        )
                        plan_records.append(
                            {
                                "identifier": hash_identifier(
                                    raw_identifier, bits=identifier_hash_bits
                                ),
                                "batch": batch_idx,
                                "pair": pair_idx,
                                "id_a": id_a,
                                "id_b": id_b,
                                "circle_first": (
                                    self.cfg.circle_first
                                    if self.cfg.circle_first is not None
                                    else self.rng.random() < 0.5
                                ),
                            }
                        )
                self._write_json_atomically(
                    {
                        "version": 1,
                        "round": rnd,
                        "attribute_batches": attr_batches,
                        "item_ids": canonical_item_ids,
                        "records": plan_records,
                    },
                    plan_path,
                )

            announce_prompt_rendering("Rank", len(plan_records))
            prompts: List[str] = []
            ids: List[str] = []
            pair_images: Dict[str, List[str]] = {}
            pair_audio: Dict[str, List[Dict[str, str]]] = {}
            pair_pdfs: Dict[str, List[Dict[str, str]]] = {}
            meta_map: Dict[str, Tuple[int, int, str, str]] = {}
            for record in plan_records:
                batch_idx = int(record["batch"])
                pair_idx = int(record["pair"])
                id_a = str(record["id_a"])
                id_b = str(record["id_b"])
                hashed_ident = str(record["identifier"])
                circle_first_flag = bool(record["circle_first"])
                batch = attr_batches[batch_idx]
                attr_def_map = (
                    {a: self.cfg.attributes[a] for a in batch}
                    if isinstance(self.cfg.attributes, dict)
                    else {a: "" for a in batch}
                )
                prompts.append(
                    self.template.render(
                        entry_circle=texts_by_id[id_a],
                        entry_square=texts_by_id[id_b],
                        attributes=attr_def_map,
                        additional_instructions=self.cfg.additional_instructions
                        or "",
                        modality=self.cfg.modality,
                        circle_first=circle_first_flag,
                    )
                )
                ids.append(hashed_ident)
                meta_map[hashed_ident] = (batch_idx, pair_idx, id_a, id_b)
                if images_by_id:
                    imgs = []
                    ia = images_by_id.get(id_a, [])
                    ib = images_by_id.get(id_b, [])
                    if circle_first_flag:
                        if ia:
                            imgs.extend(ia)
                        if ib:
                            imgs.extend(ib)
                    else:
                        if ib:
                            imgs.extend(ib)
                        if ia:
                            imgs.extend(ia)
                    if imgs:
                        pair_images[hashed_ident] = imgs
                if audio_by_id:
                    auds = []
                    aa = audio_by_id.get(id_a, [])
                    ab = audio_by_id.get(id_b, [])
                    if circle_first_flag:
                        if aa:
                            auds.extend(aa)
                        if ab:
                            auds.extend(ab)
                    else:
                        if ab:
                            auds.extend(ab)
                        if aa:
                            auds.extend(aa)
                    if auds:
                        pair_audio[hashed_ident] = auds
                if pdfs_by_id:
                    pdfs: List[Dict[str, str]] = []
                    pa = pdfs_by_id.get(id_a, [])
                    pb = pdfs_by_id.get(id_b, [])
                    if circle_first_flag:
                        if pa:
                            pdfs.extend(pa)
                        if pb:
                            pdfs.extend(pb)
                    else:
                        if pb:
                            pdfs.extend(pb)
                        if pa:
                            pdfs.extend(pa)
                    if pdfs:
                        pair_pdfs[hashed_ident] = pdfs
            # obtain responses from the language model for this round
            if len(ids) != len(set(ids)):
                raise ValueError(
                    "Rank prompt identifier collision; use reset_files=True "
                    "or a save_dir configured for 64-bit identifiers"
                )
            response_kwargs = dict(kwargs)
            # A partial tail cannot be committed or safely treated as a
            # completed tournament round, even for very large collections.
            response_kwargs["skip_tail_fails"] = False
            if self.cfg.use_dummy:
                response_kwargs.setdefault(
                    "dummy_responses",
                    {
                        identifier: {
                            "responses": [
                                json.dumps(
                                    {
                                        attribute: "draw"
                                        for attribute in attr_batches[
                                            meta_map[identifier][0]
                                        ]
                                    }
                                )
                            ]
                        }
                        for identifier in ids
                    },
                )
            resp_df = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                prompt_images=pair_images or None,
                prompt_audio=pair_audio or None,
                prompt_pdfs=pair_pdfs or None,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                json_mode=self.cfg.modality != "audio",
                save_path=staging_path,
                reset_files=reset_files,
                use_dummy=self.cfg.use_dummy,
                max_retries=1,
                reasoning_effort=self.cfg.reasoning_effort,
                **response_kwargs,
            )
            self._validate_requested_responses(
                resp_df,
                ids,
                context="Rank",
                allow_extra=False,
            )
            await self._validate_pairwise_response_payloads(
                resp_df,
                meta_map,
                attr_batches,
                context="Rank",
                retry_path=staging_path,
            )
            # attach metadata columns and overwrite the round CSV
            resp_df["Batch"] = resp_df.Identifier.map(
                lambda x: meta_map.get(str(x), (np.nan, np.nan, "", ""))[0]
            )
            resp_df["Pair"] = resp_df.Identifier.map(
                lambda x: meta_map.get(str(x), (np.nan, np.nan, "", ""))[1]
            )
            resp_df["IdA"] = resp_df.Identifier.map(
                lambda x: meta_map.get(str(x), (np.nan, np.nan, "", ""))[2]
            )
            resp_df["IdB"] = resp_df.Identifier.map(
                lambda x: meta_map.get(str(x), (np.nan, np.nan, "", ""))[3]
            )
            self._write_rank_checkpoint(resp_df, round_path, len(attr_batches))

            # parse each response
            # reuse the _coerce_dict function defined in the original implementation
            async def _coerce_dict(raw: Any) -> Dict[str, Any]:
                obj = await safest_json(raw)
                if isinstance(obj, dict):
                    return obj
                if isinstance(obj, str):
                    obj2 = await safest_json(obj)
                    if isinstance(obj2, dict):
                        return obj2
                if isinstance(obj, list) and obj:
                    inner = await safest_json(obj[0])
                    if isinstance(inner, dict):
                        return inner
                return {}

            for ident, resp in zip(resp_df.Identifier, resp_df.Response):
                meta = meta_map.get(str(ident))
                if not meta:
                    continue
                batch_idx, _, id_a, id_b = meta
                safe_obj = await _coerce_dict(resp)
                if not safe_obj:
                    continue
                batch = attr_batches[batch_idx]
                batch_attr_map = {str(k).strip().lower(): k for k in batch}
                for attr_raw, winner_raw in safe_obj.items():
                    attr_key_l = str(attr_raw).strip().lower()
                    if attr_key_l not in batch_attr_map:
                        continue
                    real_attr = batch_attr_map[attr_key_l]
                    category = self._record_pairwise_outcome(
                        history_pairs[real_attr], id_a, id_b, winner_raw
                    )
                    outcome_counts[real_attr][category] += 1
            # update ratings using the BT model for this round
            se_agg_next: Dict[str, float] = {i: 0.0 for i in item_ids}
            se_agg_counts: Dict[str, int] = {i: 0 for i in item_ids}
            for attr in attr_keys:
                outcomes = history_pairs[attr]
                if len(outcomes) == 0:
                    continue
                bt_scores, n_ij, p_ij = self._fit_bt(
                    item_ids=item_ids,
                    outcomes=outcomes,
                    pseudo=self.cfg.learning_rate,
                    max_iter=self._MAX_ITER,
                    tol=self._TOL,
                    return_info=True,
                )
                for i in item_ids:
                    ratings[i][attr] = bt_scores[i]
                s_vec = np.array([bt_scores[i] for i in item_ids])
                se_vec = self._bt_standard_errors(
                    s=s_vec,
                    n_ij=n_ij,
                    p_ij=p_ij,
                    rcond=self._SE_EIGEN_TOL,
                    regularization_strength=self.cfg.learning_rate,
                )
                component_labels = self._comparison_component_labels(n_ij)
                for i, se_val in zip(item_ids, se_vec):
                    se_store[attr][i] = float(se_val)
                    if np.isfinite(se_val):
                        se_agg_next[i] += float(se_val)
                        se_agg_counts[i] += 1
                for i, component in zip(item_ids, component_labels):
                    component_store[attr][i] = int(component)
            for i in item_ids:
                if se_agg_counts[i] > 0:
                    se_agg_next[i] /= se_agg_counts[i]
                else:
                    se_agg_next[i] = 1.0
            self._last_se_agg = se_agg_next
            # Centre ratings to zero mean for each attribute
            for attr in attr_keys:
                vals = [ratings[i][attr] for i in item_ids]
                mean_val = float(np.mean(vals))
                for i in item_ids:
                    ratings[i][attr] -= mean_val
            # Write checkpoint after this new round
            _write_checkpoint()
            update_run_metadata(
                self.cfg.save_dir,
                base_name,
                strict=True,
                last_completed_round=rnd,
            )
            for completed_artifact in (
                batch_state_path,
                staging_path,
                plan_path,
            ):
                try:
                    Path(completed_artifact).unlink(missing_ok=True)
                except OSError:
                    pass
        # After processing all rounds, return the final DataFrame
        # The checkpoint has already been written in the final iteration
        if checkpoint_result is None:
            _write_checkpoint()
        assert checkpoint_result is not None
        return checkpoint_result.copy()
