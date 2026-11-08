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
  or ``"insufficient signal"``.  ``draw`` and ``insufficient signal``
  are both treated as a tie and contribute equally to both items when
  fitting the BT model.

* A cleaned up asynchronous ``run`` method that accepts a pandas
  ``DataFrame`` and the name of the column containing the text to be
  ranked.  Each row receives a stable identifier derived from a hash of its
  contents; no external ``id_col`` argument is required.  The method
  produces a DataFrame with one row per input passage, a numeric
  rating for each attribute, along with z‑scores and standard errors,
  and writes the results to disk under ``save_dir``.

The core ranking logic remains largely unchanged from ``elo.py``
because the underlying mathematics of the BT model and the pairing
strategies continue to work well.  However, comments have been added
throughout the code to clarify intent and to highlight areas where
further experimentation (e.g. alternative information gain metrics) can
be incorporated.
"""

from __future__ import annotations

import os
from pathlib import Path
import random
import math
import copy
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse.csgraph import connected_components
from scipy.special import expit

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
    resolve_attribute_batches,
    resolve_identifier_hash_bits,
    write_task_run_metadata,
)


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except Exception:
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


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
        Number of matches per item per round.
    power_matching:
        Whether to use an information‑theoretic pairing heuristic.  The
        resulting rankings always include per‑attribute z‑scores alongside the
        raw Bradley–Terry estimates (``"<attribute>_raw"``) and their
        standard errors (``"<attribute>_se"``).
    learning_rate:
        Total pseudo-count per observed pair, split equally between both
        directions. Zero requests an unregularized fit, which requires
        bidirectional reachability through wins in each comparison component.
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
        This is enabled by default so the initial culling uses grounded
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
    primer_scores, primer_scale, primer_center:
        Optional manual primers to seed the Bradley–Terry rating state.
        Scores are centred per attribute when ``primer_center`` is
        ``True`` and scaled by ``primer_scale``.
    """

    attributes: Union[Dict[str, str], List[str]]
    n_rounds: int = 5
    matches_per_round: int = 5
    power_matching: bool = True
    learning_rate: float = 0.1
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

    def __post_init__(self) -> None:
        if self.additional_instructions is not None:
            cleaned = str(self.additional_instructions).strip()
            self.additional_instructions = cleaned or None


class Rank:
    """Rank items by comparing passages pairwise on multiple attributes.

    An instance of :class:`Ranker` orchestrates the iterative process
    of sampling pairs, calling a language model to adjudicate which
    passage better exhibits each attribute, and then fitting a
    Bradley–Terry model to those outcomes.  Standard errors and
    z‑scores are computed for every attribute.  Results are persisted to disk
    after the final round.
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
        if not math.isfinite(cfg.learning_rate) or cfg.learning_rate < 0:
            raise ValueError("learning_rate must be finite and non-negative")
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

        # internal constants for the pairing and BT algorithms.  These
        # values are deliberately not exposed through the public API as
        # they seldom need tuning and adjusting them can complicate
        # reproducibility.  Should you need to experiment with these
        # hyperparameters, modify the values below.
        self._EXPLORE_FRAC = 0.2  # fraction of random pairings per round
        self._CANDIDATE_NEIGHBORS = 20  # neighbourhood size for info gain pairing
        self._HIGH_SE_FRAC = 0.25  # fraction of high‑uncertainty items
        self._MAX_ITER = 1000  # maximum iterations for BT optimisation
        self._TOL = 1e-6  # gradient tolerance for BT
        # A small ridge term stabilises the inversion of the Fisher information
        # matrix when computing standard errors.  The previous value (1e‑9)
        # occasionally led to extremely large uncertainties for items with
        # limited or contradictory comparisons.  Increasing this value
        # regularises the covariance estimate and prevents unreasonably
        # large standard errors.  If you observe inflated SE values,
        # consider increasing this further (e.g. to 1e‑4).
        self._SE_RIDGE = 1e-5
        # The maximum number of candidate pairs to consider per pairing round.
        # When the number of items becomes very large (e.g. tens of thousands),
        # evaluating all possible pairs is intractable.  We therefore cap the
        # total number of candidate pairs by limiting the neighbourhood size
        # used when constructing candidate pairs.  The default of 200k ensures
        # that information gain pairing remains tractable even with very
        # large data sets: for example, with 10 000 items and a cap of
        # 200 000, each item will only consider approximately 20 neighbours.
        self._MAX_CANDIDATE_PAIRS_PER_ROUND = 200_000

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
                        attr_to_vals[attr].append(float(amap[attr]))
                    except Exception:
                        continue

        attr_offset: Dict[str, float] = {a: 0.0 for a in attr_keys}
        if self.cfg.primer_center:
            for attr, vals in attr_to_vals.items():
                if vals:
                    attr_offset[attr] = float(np.mean(vals))

        scale = self.cfg.primer_scale or 1.0
        for ident, amap in primer.items():
            if ident not in ratings:
                continue
            for attr in attr_keys:
                if attr not in amap or amap[attr] is None:
                    continue
                try:
                    val = (float(amap[attr]) - attr_offset[attr]) * scale
                    ratings[ident][attr] = val
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

    def _split_rate_kwargs(self, overrides: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        merged: Dict[str, Any] = {}
        if self.cfg.rate_kwargs:
            merged.update(self.cfg.rate_kwargs)
        if overrides:
            merged.update(overrides)
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
            key_series = rate_df[text_column].map(
                lambda x: _hash_text_identifier(
                    x,
                    strict=True,
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
    def _fit_bt(
        self,
        item_ids: List[str],
        outcomes: List[Tuple[str, str]],
        pseudo: float,
        max_iter: int,
        tol: float,
        return_info: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
        """Fit log skills within observed comparison components.

        Each outcome contributes one win. Smoothing adds half of ``pseudo``
        to each direction of an observed edge. The returned match counts
        contain only observed comparisons, for uncertainty estimation.
        """
        n = len(item_ids)
        idx = {item: i for i, item in enumerate(item_ids)}
        wins = np.zeros((n, n), dtype=float)
        for winner, loser in outcomes:
            if winner != loser and winner in idx and loser in idx:
                wins[idx[winner], idx[loser]] += 1

        observed = wins + wins.T
        scores = np.zeros(n, dtype=float)
        count, labels = connected_components(observed, directed=False)
        for label in range(count):
            members = np.flatnonzero(labels == label)
            if len(members) < 2:
                continue
            block = np.ix_(members, members)
            local_wins = wins[block]
            if pseudo == 0 and connected_components(
                local_wins, directed=True, connection="strong", return_labels=False
            ) != 1:
                raise ValueError(
                    "Unregularized Bradley–Terry scores are undefined for a "
                    "separated comparison component. Use a positive learning_rate."
                )

            left, right = np.nonzero(np.triu(observed[block], k=1))
            forward = local_wins[left, right] + pseudo / 2
            reverse = local_wins[right, left] + pseudo / 2
            if len(members) == 2:
                contrast = np.log(forward[0]) - np.log(reverse[0])
                scores[members] = [contrast / 2, -contrast / 2]
                continue

            def objective(free_scores: np.ndarray) -> Tuple[float, np.ndarray]:
                # Fix one score at zero to remove the location indeterminacy.
                local = np.append(free_scores, 0.0)
                delta = local[left] - local[right]
                loss = forward @ np.logaddexp(0, -delta) + reverse @ np.logaddexp(
                    0, delta
                )
                residual = reverse * expit(delta) - forward * expit(-delta)
                gradient = np.bincount(left, residual, minlength=len(local))
                gradient -= np.bincount(right, residual, minlength=len(local))
                return float(loss), gradient[:-1]

            result = minimize(
                objective,
                np.zeros(len(members) - 1),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": max_iter, "gtol": tol, "ftol": np.finfo(float).eps},
            )
            if not result.success:
                raise RuntimeError(f"Bradley–Terry fit did not converge: {result.message}")
            local = np.append(result.x, 0.0)
            scores[members] = local - local.mean()

        by_item = dict(zip(item_ids, scores.tolist()))
        if not return_info:
            return by_item
        probabilities = expit(scores[:, None] - scores[None, :])
        return by_item, observed, probabilities

    def _bt_standard_errors(
        self,
        n_ij: np.ndarray,
        p_ij: np.ndarray,
        ridge: float,
        pseudo: float,
    ) -> np.ndarray:
        """Compute model-based uncertainty within observed components.

        Pseudo-counts contribute curvature to the fit. Sampling variation
        comes from observed comparisons, so use observed information in
        the middle of the sandwich. Isolates have no standard error.
        """
        errors = np.full(len(n_ij), np.nan)
        count, labels = connected_components(n_ij, directed=False)
        for label in range(count):
            members = np.flatnonzero(labels == label)
            if len(members) < 2:
                continue
            block = np.ix_(members, members)
            counts = n_ij[block]
            probabilities = p_ij[block]
            variance = probabilities * probabilities.T

            observed = counts * variance
            fitted = (counts + pseudo * (counts > 0)) * variance
            meat = np.diag(observed.sum(axis=1)) - observed
            bread = np.diag(fitted.sum(axis=1)) - fitted
            projector = np.eye(len(members)) - 1.0 / len(members)
            inverse = np.linalg.pinv(bread + ridge * projector, hermitian=True)
            covariance = inverse @ meat @ inverse
            errors[members] = np.sqrt(np.clip(np.diag(covariance), 0, None))
        return errors

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
        """Select pairs by maximising expected information gain while ensuring
        that every item participates in the prescribed number of matches.

        This implementation differs from the original heuristics by
        considering a bounded set of candidate pairs that scales with the
        number of items.  Each pair is assigned a score based on the
        expected reduction in uncertainty (estimated from the current
        ratings and aggregated standard errors).  Pairs with larger
        scores are chosen first, subject to the constraint that each
        item is matched exactly ``mpr`` times.  If some items remain
        unmatched after exhausting the scored pairs, additional pairs
        are filled in randomly to satisfy the per‑item quota.
        """
        n = len(item_ids)
        if n < 2:
            return []
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
        high_se_ids = sorted(item_ids, key=lambda i: se_agg.get(i, 1.0), reverse=True)[
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
            var_a = se_agg.get(a, 1.0) ** 2
            var_b = se_agg.get(b, 1.0) ** 2
            param_unc = var_a + var_b
            # Encourage comparisons between similarly‑rated items (high information
            # gain) while still prioritising uncertain pairs.  The closeness term
            # dampens pairings with large rating gaps to tease out subtle ordering
            # differences.
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
            # Try to pair it with any other item (not just those needing matches) to avoid self‑pairs
            potential = [x for x in item_ids if x != a]
            if not potential:
                # Degenerate case: only one item exists; cannot form a valid pair
                break
            b = self.rng.choice(potential)
            tup = (a, b) if a < b else (b, a)
            pairs_selected.append(tup)
            needed[a] -= 1
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
        mpr = max(1, self.cfg.matches_per_round)
        # Always use information gain pairing to guarantee exact match counts
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
        new_ids: List[str],
        round_indices: List[int],
        item_ids: List[str],
        texts_by_id: Dict[str, str],
        images_by_id: Dict[str, List[str]],
        audio_by_id: Dict[str, List[Dict[str, str]]],
        pdfs_by_id: Dict[str, List[Dict[str, str]]],
        attr_batches: List[List[str]],
        attr_keys: List[str],
        history_pairs: Dict[str, List[Tuple[str, str]]],
        ratings: Dict[str, Dict[str, float]],
        se_store: Dict[str, Dict[str, float]],
        base_name: str,
        df_proc: pd.DataFrame,
        _write_checkpoint: Callable[[], None],
        current_ratings: Optional[Dict[str, float]],
        se_agg_local: Optional[Dict[str, float]],
        reset_files: bool,
        identifier_hash_bits: int,
        **kwargs: Any,
    ) -> None:
        if not new_ids:
            return
        for rnd in round_indices:
            round_path = os.path.join(self.cfg.save_dir, f"{base_name}_round{rnd}.csv")
            if not os.path.exists(round_path):
                continue
            try:
                df_round = pd.read_csv(round_path)
            except Exception:
                continue
            counts: Dict[str, int] = {}
            if {"IdA", "IdB"}.issubset(df_round.columns):
                for a, b in zip(df_round["IdA"], df_round["IdB"]):
                    counts[str(a)] = counts.get(str(a), 0) + 1
                    counts[str(b)] = counts.get(str(b), 0) + 1
            else:
                for ident in df_round.get("Identifier", []):
                    parts = str(ident).split("|")
                    if len(parts) != 5:
                        continue
                    _, _, _, id_a, id_b = parts
                    counts[id_a] = counts.get(id_a, 0) + 1
                    counts[id_b] = counts.get(id_b, 0) + 1
            pairs_needed: List[Tuple[str, str]] = []
            for nid in new_ids:
                needed = self.cfg.matches_per_round - counts.get(nid, 0)
                if needed <= 0:
                    continue
                opponents = [i for i in item_ids if i != nid]
                self.rng.shuffle(opponents)
                for opp in opponents[:needed]:
                    pairs_needed.append((nid, opp))
            if not pairs_needed:
                continue
            announce_prompt_rendering(
                "Rank:catchup", len(attr_batches) * len(pairs_needed)
            )
            prompts: List[str] = []
            ids: List[str] = []
            pair_images: Dict[str, List[str]] = {}
            pair_audio: Dict[str, List[Dict[str, str]]] = {}
            pair_pdfs: Dict[str, List[Dict[str, str]]] = {}
            meta_map: Dict[str, Tuple[int, int, str, str]] = {}
            id_to_circle_first: Dict[str, bool] = {}
            for batch_idx, batch in enumerate(attr_batches):
                attr_def_map = (
                    {a: self.cfg.attributes[a] for a in batch}
                    if isinstance(self.cfg.attributes, dict)
                    else {a: "" for a in batch}
                )
                for pair_idx, (id_a, id_b) in enumerate(pairs_needed):
                    raw_ident = f"catchup|{rnd}|{batch_idx}|{pair_idx}|{id_a}|{id_b}"
                    hashed_ident = hash_identifier(raw_ident, bits=identifier_hash_bits)
                    circle_first_flag = (
                        self.cfg.circle_first
                        if self.cfg.circle_first is not None
                        else self.rng.random() < 0.5
                    )
                    id_to_circle_first[hashed_ident] = circle_first_flag
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
            resp_df = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                prompt_images=pair_images or None,
                prompt_audio=pair_audio or None,
                prompt_pdfs=pair_pdfs or None,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                json_mode=self.cfg.modality != "audio",
                save_path=round_path,
                reset_files=reset_files,
                use_dummy=self.cfg.use_dummy,
                max_retries=1,
                reasoning_effort=self.cfg.reasoning_effort,
                **kwargs,
            )
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
            resp_df.to_csv(round_path, index=False)

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
                    val = winner_raw
                    if isinstance(val, dict) and "winner" in val:
                        val = val.get("winner")
                    if isinstance(val, str):
                        v = val.strip().lower()
                    else:
                        v = ""
                    if v.startswith(("cir", "c", "left", "text a")):
                        history_pairs[real_attr].append((id_a, id_b))
                    elif v.startswith(("squ", "b", "right", "text b")):
                        history_pairs[real_attr].append((id_b, id_a))
                    elif v.startswith("draw") or v.startswith("insufficient"):
                        history_pairs[real_attr].append((id_a, id_b))
                        history_pairs[real_attr].append((id_b, id_a))
                    else:
                        continue
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
                se_vec = self._bt_standard_errors(
                    n_ij=n_ij,
                    p_ij=p_ij,
                    ridge=self._SE_RIDGE,
                    pseudo=self.cfg.learning_rate,
                )
                for i, se_val in zip(item_ids, se_vec):
                    se_store[attr][i] = float(se_val)
                    if np.isfinite(se_val):
                        se_agg_next[i] += float(se_val)
                        se_agg_counts[i] += 1
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

        work_df = df.reset_index(drop=True).copy()
        strict_text_mode = self.cfg.modality in {"text", "entity", "web"}
        if id_column is not None:
            if id_column not in work_df.columns:
                raise ValueError(f"id_column '{id_column}' not found in DataFrame")
            valid_mask = ~work_df[id_column].map(_is_missing_scalar)
            dropped = int((~valid_mask).sum())
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
            work_df = work_df.rename(columns={text_column: "text"})
        rewrite_col = self.cfg.recursive_rewrite_text_col or "text"
        if rewrite_col not in work_df.columns:
            work_df[rewrite_col] = work_df["text"]
        work_df["identifier"] = work_df["identifier"].astype(str)

        if work_df.empty:
            empty_out = work_df[[c for c in df.columns if c in work_df.columns]].copy()
            for attr in attr_list:
                empty_out[attr] = pd.Series(dtype="float64")
                empty_out[f"{attr}_raw"] = pd.Series(dtype="float64")
                empty_out[f"{attr}_se"] = pd.Series(dtype="float64")
            empty_out["overall_rank"] = pd.Series(dtype="float64")
            empty_out["exit_stage"] = pd.Series(dtype="float64")
            final_file = os.path.join(self.cfg.save_dir, f"{self.cfg.file_name}_recursive_final.csv")
            empty_out.to_csv(final_file, index=False)
            return empty_out

        original_cols = [c for c in df.columns if c in work_df.columns]
        original_df = work_df[original_cols + ["identifier"]].copy()
        latest_text: Dict[str, str] = {
            ident: txt for ident, txt in zip(work_df["identifier"], work_df["text"])
        }

        base_folder = os.path.join(
            self.cfg.save_dir, f"{self.cfg.file_name}_recursive"
        )
        os.makedirs(base_folder, exist_ok=True)

        def _compute_stage_zscores(
            stage_df: pd.DataFrame,
        ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
            zscores: Dict[str, Dict[str, float]] = {attr: {} for attr in attr_list}
            scales: Dict[str, float] = {attr: 1.0 for attr in attr_list}
            for attr in attr_list:
                raw_col = f"{attr}_raw"
                source_col = raw_col if raw_col in stage_df.columns else attr
                if source_col not in stage_df.columns:
                    continue
                series = pd.to_numeric(stage_df[source_col], errors="coerce")
                mean = series.mean()
                std = series.std(ddof=0)
                if std == 0 or np.isnan(std):
                    normed = pd.Series([0.0] * len(series), index=stage_df.index)
                    scales[attr] = 1.0
                else:
                    normed = (series - mean) / std
                    scales[attr] = float(std) if raw_col in stage_df.columns else 1.0
                for ident, val in zip(stage_df["identifier"], normed):
                    zscores[attr][str(ident)] = float(val)
            return zscores, scales

        def _select_next_ids(active_ids: Sequence[str], stage_zs: Dict[str, Dict[str, float]]) -> List[str]:
            n = len(active_ids)
            if n <= self.cfg.recursive_min_remaining:
                return list(active_ids)
            keep_n = max(
                int(math.ceil(n * self.cfg.recursive_fraction)),
                self.cfg.recursive_min_remaining,
            )
            scores = {i: stage_zs.get(cut_attr, {}).get(i, 0.0) for i in active_ids}
            ascending = cut_side == "bottom"
            ranked = sorted(active_ids, key=lambda x: scores.get(x, 0.0), reverse=not ascending)
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
        final_stage_idx: Optional[int] = None
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
                next_keep = max(
                    int(math.ceil(n_current * self.cfg.recursive_fraction)),
                    self.cfg.recursive_min_remaining,
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

            stage_zs, stage_scales = _compute_stage_zscores(stage_df_out)
            stage_z_history[stage_idx] = stage_zs

            if is_final_stage:
                for ident in current_ids:
                    exit_stage[ident] = stage_idx
                final_stage_df = stage_df_out
                final_stage_idx = stage_idx
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
            if stage_idx:
                final_stage_idx = stage_idx

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
                    if final_stage_idx is not None and stage == final_stage_idx:
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
            A DataFrame with one row per input passage.  For each
            attribute the DataFrame contains a ``"<attribute>"`` column
            holding the z‑score, a ``"<attribute>_raw"`` column with the
            centred Bradley–Terry estimate, and a ``"<attribute>_se"``
            column with the standard error.  The DataFrame is also written
            to ``save_dir``.
        """
        base_name = os.path.splitext(self.cfg.file_name)[0]
        checkpoint_paths = [
            str(path)
            for path in Path(self.cfg.save_dir).glob(f"{base_name}_round*.csv")
        ]
        run_metadata = load_run_metadata(
            self.cfg.save_dir, base_name, reset_files=reset_files
        )
        identifier_hash_bits = resolve_identifier_hash_bits(
            task_name="Rank",
            metadata=run_metadata,
            reset_files=reset_files,
            checkpoint_paths=checkpoint_paths,
        )
        self.cfg.attributes = load_persisted_attributes(
            save_dir=self.cfg.save_dir,
            incoming=self.cfg.attributes,
            reset_files=reset_files,
            task_name="Rank",
            item_name="attributes",
            legacy_filename=f"{base_name}_attrs.json",
        )

        kwargs.setdefault("web_search", self.cfg.modality == "web")
        if self.cfg.recursive:
            return await self._run_recursive(
                df,
                column_name,
                id_column=id_column,
                reset_files=reset_files,
                identifier_hash_bits=identifier_hash_bits,
                **kwargs,
            )

        # prepare file paths
        final_path = os.path.join(self.cfg.save_dir, f"{base_name}_final.csv")
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
            valid_mask = ~df_proc[id_column].map(_is_missing_scalar)
            dropped = int((~valid_mask).sum())
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
            df_out.to_csv(final_path, index=False)
            return df_out
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
                        try:
                            idx_str = fname[
                                len(base_name) + 6 : -4
                            ]  # len("_round") == 6
                            rnd_idx = int(idx_str)
                            existing_rounds.append(rnd_idx)
                        except Exception:
                            continue
            except Exception:
                existing_rounds = []
        if existing_rounds:
            last_completed = max(existing_rounds)
            if os.path.exists(final_path):
                try:
                    final_df = pd.read_csv(final_path)
                    identifier_col = (
                        id_column
                        if id_column and id_column in final_df.columns
                        else column_name
                    )
                    if identifier_col not in final_df.columns:
                        raise ValueError(
                            "Existing ranking output is missing identifier column "
                            f"'{identifier_col}'."
                        )
                    if id_column:
                        final_ids = set(final_df[identifier_col].astype(str))
                    else:
                        final_ids = set(
                            final_df[identifier_col]
                            .map(
                                lambda x: _hash_text_identifier(
                                    x,
                                    strict=True,
                                    bits=identifier_hash_bits,
                                )
                            )
                            .dropna()
                            .astype(str)
                        )
                    if last_completed >= self.cfg.n_rounds - 1 and set(df_proc["_id"]) <= final_ids:
                        return final_df
                except Exception:
                    pass
            start_round = last_completed + 1
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
        if self.cfg.initial_rating_pass and attr_keys:
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
        history_pairs: Dict[str, List[Tuple[str, str]]] = {a: [] for a in attr_keys}
        # store per‑attribute standard errors across items
        se_store: Dict[str, Dict[str, float]] = {
            a: {i: np.nan for i in item_ids} for a in attr_keys
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
        )

        # Helper function to write the current results to the final CSV.  This
        # builds the output DataFrame from the current ``df_proc`` and
        # ``ratings``/``se_store``/``zscores`` and writes it to
        # ``final_path``.
        def _write_checkpoint() -> None:
            # Compute z‑scores for each attribute so that we can expose centred
            # BT scores alongside their normalised variants.
            zscores_local: Dict[str, Dict[str, float]] = {}
            for attr in attr_keys:
                vals = np.array([ratings[i][attr] for i in item_ids])
                mean = vals.mean()
                std = vals.std(ddof=0)
                if std == 0:
                    zscores_local[attr] = {i: 0.0 for i in item_ids}
                else:
                    zscores_local[attr] = {
                        i: float((ratings[i][attr] - mean) / std) for i in item_ids
                    }
            # Merge computed results back into the original DataFrame copy.
            for attr in attr_keys:
                raw_col = f"{attr}_raw"
                # ratings
                val_map = {i: ratings[i][attr] for i in item_ids}
                df_proc[raw_col] = df_proc["_id"].map(val_map)
                # standard errors
                se_map = {i: se_store[attr].get(i, np.nan) for i in item_ids}
                df_proc[f"{attr}_se"] = df_proc["_id"].map(se_map)
                # z‑scores
                z_map = zscores_local.get(attr, {i: np.nan for i in item_ids})
                df_proc[attr] = df_proc["_id"].map(z_map)

            # Reorder columns: original user columns first (excluding the internal ``_id``),
            # then for each attribute the z‑score column followed by raw scores and
            # standard errors.
            original_cols = [
                c for c in df.columns
            ]  # preserve the order provided by the user
            new_cols: List[str] = []
            for attr in attr_keys:
                new_cols.append(attr)
                new_cols.append(f"{attr}_raw")
                new_cols.append(f"{attr}_se")
            final_cols = original_cols + new_cols
            final_cols = [c for c in final_cols if c in df_proc.columns]
            df_out_local = df_proc[final_cols].copy()
            # Write the final results to disk in CSV format.  Using CSV avoids
            # Excel row limits and unnecessary overhead.
            df_out_local.to_csv(final_path, index=False)

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
                    # Load existing responses for this round
                    df_round = pd.read_csv(round_path)
                    df_round["Response"] = df_round["Response"].apply(
                        lambda x: None if pd.isna(x) else x
                    )
                except Exception:
                    continue

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
                            val = winner_raw
                            if isinstance(val, dict) and "winner" in val:
                                val = val.get("winner")
                            if isinstance(val, str):
                                v = val.strip().lower()
                            else:
                                v = ""
                            if v.startswith(("cir", "c", "left", "text a")):
                                history_pairs[real_attr].append((id_a, id_b))
                            elif v.startswith(("squ", "b", "right", "text b")):
                                history_pairs[real_attr].append((id_b, id_a))
                            elif v.startswith("draw") or v.startswith("insufficient"):
                                history_pairs[real_attr].append((id_a, id_b))
                                history_pairs[real_attr].append((id_b, id_a))
                            else:
                                continue
                else:
                    for ident, resp_raw in zip(
                        df_round["Identifier"], df_round["Response"]
                    ):
                        parts = str(ident).split("|")
                        if len(parts) != 5:
                            continue
                        _, batch_idx_str, _, id_a, id_b = parts
                        try:
                            batch_idx = int(batch_idx_str)
                        except (TypeError, ValueError):
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
                            val = winner_raw
                            if isinstance(val, dict) and "winner" in val:
                                val = val.get("winner")
                            if isinstance(val, str):
                                v = val.strip().lower()
                            else:
                                v = ""
                            if v.startswith(("cir", "c", "left", "text a")):
                                history_pairs[real_attr].append((id_a, id_b))
                            elif v.startswith(("squ", "b", "right", "text b")):
                                history_pairs[real_attr].append((id_b, id_a))
                            elif v.startswith("draw") or v.startswith("insufficient"):
                                history_pairs[real_attr].append((id_a, id_b))
                                history_pairs[real_attr].append((id_b, id_a))
                            else:
                                continue
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
                    se_vec = self._bt_standard_errors(
                        n_ij=n_ij,
                        p_ij=p_ij,
                        ridge=self._SE_RIDGE,
                        pseudo=self.cfg.learning_rate,
                    )
                    for i, se_val in zip(item_ids, se_vec):
                        se_store[attr][i] = float(se_val)
                        if np.isfinite(se_val):
                            se_agg_next[i] += float(se_val)
                            se_agg_counts[i] += 1
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
        seen_ids: Set[str] = set()
        for pair_list in history_pairs.values():
            for a, b in pair_list:
                seen_ids.add(a)
                seen_ids.add(b)
        new_ids = [i for i in item_ids if i not in seen_ids]
        await self._catch_up_existing_rounds(
            new_ids=new_ids,
            round_indices=list(range(start_round)),
            item_ids=item_ids,
            texts_by_id=texts_by_id,
            images_by_id=images_by_id,
            audio_by_id=audio_by_id,
            pdfs_by_id=pdfs_by_id,
            attr_batches=attr_batches,
            attr_keys=attr_keys,
            history_pairs=history_pairs,
            ratings=ratings,
            se_store=se_store,
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
            pairs = self._generate_pairs(
                item_ids=item_ids,
                texts_by_id=texts_by_id,
                current_ratings=current_agg if use_current else None,
                se_agg=se_source,
            )
            if not pairs:
                break
            announce_prompt_rendering(
                "Rank", len(attr_batches) * len(pairs)
            )
            prompts: List[str] = []
            ids: List[str] = []
            pair_images: Dict[str, List[str]] = {}
            pair_audio: Dict[str, List[Dict[str, str]]] = {}
            pair_pdfs: Dict[str, List[Dict[str, str]]] = {}
            meta_map: Dict[str, Tuple[int, int, str, str]] = {}
            id_to_circle_first: Dict[str, bool] = {}
            for batch_idx, batch in enumerate(attr_batches):
                attr_def_map = (
                    {a: self.cfg.attributes[a] for a in batch}
                    if isinstance(self.cfg.attributes, dict)
                    else {a: "" for a in batch}
                )
                for pair_idx, ((id_a, t_a), (id_b, t_b)) in enumerate(pairs):
                    raw_ident = f"{rnd}|{batch_idx}|{pair_idx}|{id_a}|{id_b}"
                    hashed_ident = hash_identifier(raw_ident, bits=identifier_hash_bits)
                    circle_first_flag = (
                        self.cfg.circle_first
                        if self.cfg.circle_first is not None
                        else self.rng.random() < 0.5
                    )
                    id_to_circle_first[hashed_ident] = circle_first_flag
                    prompts.append(
                        self.template.render(
                            entry_circle=t_a,
                            entry_square=t_b,
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
            round_path = os.path.join(self.cfg.save_dir, f"{base_name}_round{rnd}.csv")
            resp_df = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                prompt_images=pair_images or None,
                prompt_audio=pair_audio or None,
                prompt_pdfs=pair_pdfs or None,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                json_mode=self.cfg.modality != "audio",
                save_path=round_path,
                reset_files=reset_files,
                use_dummy=self.cfg.use_dummy,
                max_retries=1,
                reasoning_effort=self.cfg.reasoning_effort,
                **kwargs,
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
            resp_df.to_csv(round_path, index=False)

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
                    val = winner_raw
                    if isinstance(val, dict) and "winner" in val:
                        val = val.get("winner")
                    if isinstance(val, str):
                        v = val.strip().lower()
                    else:
                        v = ""
                    if v.startswith(("cir", "c", "left", "text a")):
                        history_pairs[real_attr].append((id_a, id_b))
                    elif v.startswith(("squ", "b", "right", "text b")):
                        history_pairs[real_attr].append((id_b, id_a))
                    elif v.startswith("draw") or v.startswith("insufficient"):
                        history_pairs[real_attr].append((id_a, id_b))
                        history_pairs[real_attr].append((id_b, id_a))
                    else:
                        continue
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
                se_vec = self._bt_standard_errors(
                    n_ij=n_ij,
                    p_ij=p_ij,
                    ridge=self._SE_RIDGE,
                    pseudo=self.cfg.learning_rate,
                )
                for i, se_val in zip(item_ids, se_vec):
                    se_store[attr][i] = float(se_val)
                    if np.isfinite(se_val):
                        se_agg_next[i] += float(se_val)
                        se_agg_counts[i] += 1
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
        # After processing all rounds, return the final DataFrame
        # The checkpoint has already been written in the final iteration
        return pd.read_csv(final_path)
