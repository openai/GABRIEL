"""
Enhanced Gabriel Visualisation Utilities
=======================================

This module refines the original plotting utilities to provide:

* OLS regressions via statsmodels with meaningful coefficient names
  (no more ``x1``, ``x2``) and optional robust standard errors.
* Binned scatter plots that support multiple independent variables via
  ``controls`` and allow custom axis limits.
* Bar, box and line plots with a variety of customisation options.

The functions mirror the earlier API but with cleaner parameter names
and additional features.  For Python 3.12 and SciPy 1.16+, use
``statsmodels>=0.14.5`` to avoid import errors.
"""

from __future__ import annotations

import math
import os
import random
import re
import textwrap
from collections import OrderedDict
from itertools import combinations
from pathlib import Path
from typing import Iterable, Dict, Any, Optional, List, Tuple, Sequence, Union, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from scipy.stats import sem, norm

try:
    from tabulate import tabulate  # type: ignore
except ModuleNotFoundError:
    tabulate = None  # fallback when tabulate isn't installed


class _MissingStatsmodels:
    """Lazily raise an informative error when statsmodels isn't available."""

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial
        raise ImportError(
            "statsmodels is required for this functionality. Install statsmodels>=0.14 to enable it."
        )


try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover - exercised when statsmodels is missing
    sm = _MissingStatsmodels()
    smf = _MissingStatsmodels()


def _ensure_list(values: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Return ``values`` as a list, accepting strings or iterables."""

    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return list(values)


def _to_native(value: Any) -> Any:
    """Convert NumPy scalar types to native Python scalars for metadata."""

    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_save_format(save_format: Optional[str]) -> Optional[str]:
    """Normalize a requested save format to a lowercase file extension."""

    if save_format is None:
        return None
    fmt = str(save_format).strip().lower().lstrip(".")
    return fmt or None


def _resolve_plot_save_path(
    save_path: Optional[Union[str, Path]],
    *,
    default_name: str,
    save_format: Optional[str] = None,
    batch_idx: int = 0,
    total_batches: int = 1,
) -> Optional[Path]:
    """Return a fully qualified save path for a plot, creating directories."""

    if save_path is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(save_path)))
    path = Path(expanded)
    fmt = _normalize_save_format(save_format)
    suffix = f"_{batch_idx + 1}" if total_batches > 1 else ""
    if path.suffix:
        ext = fmt or path.suffix.lstrip(".")
        stem = path.stem + suffix
        target = path.with_name(f"{stem}.{ext}")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
    path.mkdir(parents=True, exist_ok=True)
    ext = fmt or "png"
    safe_name = default_name or "plot"
    return path / f"{safe_name}{suffix}.{ext}"


def _save_figure(
    fig: plt.Figure,
    *,
    save_path: Optional[Union[str, Path]],
    default_name: str,
    save_format: Optional[str] = None,
    batch_idx: int = 0,
    total_batches: int = 1,
    bbox_inches: Optional[str] = "tight",
    dpi: Optional[int] = None,
    print_save_path: bool = True,
) -> Optional[Path]:
    """Save ``fig`` if ``save_path`` is provided and return the path."""

    target = _resolve_plot_save_path(
        save_path,
        default_name=default_name,
        save_format=save_format,
        batch_idx=batch_idx,
        total_batches=total_batches,
    )
    if target is None:
        return None
    save_kwargs: Dict[str, Any] = {}
    if bbox_inches:
        save_kwargs["bbox_inches"] = bbox_inches
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    fig.savefig(target, **save_kwargs)
    if print_save_path:
        print(f"Saved plot to {target}")
    return target


def _prepare_fixed_effect_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    min_share: float,
) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    """Normalise fixed-effect columns and return base/rare level metadata."""

    base_levels: Dict[str, Any] = {}
    rare_levels: Dict[str, List[Any]] = {}
    total_rows = len(data)
    min_share = max(float(min_share), 0.0)
    for col in columns:
        if col not in data.columns:
            raise KeyError(f"Fixed-effect column '{col}' not found in dataframe.")
        series = pd.Series(data[col], index=data.index)
        if not series.empty:
            series = series.astype(object)
        counts = series.dropna().value_counts()
        if counts.empty:
            base_levels[col] = None
            rare_levels[col] = []
            data[col] = series
            continue
        rare: List[Any] = []
        if min_share > 0 and total_rows > 0:
            shares = counts / float(total_rows)
            rare = shares[shares < min_share].index.tolist()
        placeholder = None
        if rare:
            placeholder = f"__rare__{col}__"
            existing = {str(v) for v in counts.index}
            while placeholder in existing:
                placeholder += "_"
            series = series.where(~series.isin(rare), placeholder)
        non_missing = series.dropna()
        if non_missing.empty:
            base = None
            ordered_levels: List[Any] = []
        else:
            unique_levels = list(dict.fromkeys(non_missing))
            if placeholder is not None:
                base = placeholder
                ordered_levels = [placeholder]
                ordered_levels.extend(lvl for lvl in unique_levels if lvl != placeholder)
            else:
                counts_after = pd.Series(non_missing).value_counts()
                base = counts_after.idxmax()
                ordered_levels = [base]
                ordered_levels.extend(lvl for lvl in unique_levels if lvl != base)
        if ordered_levels:
            cat = pd.Categorical(series, categories=ordered_levels)
            data[col] = pd.Series(cat, index=data.index)
        else:
            data[col] = series
        base_levels[col] = _to_native(base)
        rare_levels[col] = [_to_native(val) for val in rare]
    return base_levels, rare_levels


def _cluster_groups(df: pd.DataFrame, columns: Sequence[str]) -> Union[np.ndarray, pd.Series]:
    """Return an array of cluster identifiers suitable for statsmodels."""

    if len(columns) == 1:
        col = columns[0]
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            return series.values
        return pd.Categorical(series).codes
    group_df = pd.DataFrame(index=df.index)
    for col in columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            group_df[col] = series.values
        else:
            group_df[col] = pd.Categorical(series).codes
    return group_df.values


def _apply_year_excess(
    df: pd.DataFrame,
    *,
    year_col: str,
    window: int,
    columns: Sequence[str],
    mode: str = "difference",
    replace: bool = True,
    prefix: str = "",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Compute excess/ratio values relative to a rolling window of years.

    Parameters
    ----------
    df : DataFrame
        Data containing the original variables and ``year_col``.
    year_col : str
        Column representing the temporal dimension.
    window : int
        Number of years on each side used when computing the rolling mean.
    columns : Sequence[str]
        Variables for which to compute excess or ratios.
    mode : {"difference", "ratio", "percent"}
        Whether to subtract (excess), divide (ratio), or express the
        percent change relative to the window mean.
    replace : bool, default True
        If True, the returned mapping replaces each original column name with
        the derived excess/ratio column in downstream analyses.
    prefix : str, default ""
        Optional prefix for new columns (useful when running multiple configs).

    Returns
    -------
    df_out : DataFrame
        Copy of ``df`` containing the additional columns.
    replacements : dict
        Mapping of original column names to new excess/ratio columns that can
        be used to update variable lists.
    """

    if year_col not in df.columns:
        raise KeyError(f"Year column '{year_col}' not found in dataframe.")
    resolved_mode = (mode or "difference").lower()
    valid_modes = {"difference", "ratio", "percent"}
    if resolved_mode not in valid_modes:
        raise ValueError("mode must be 'difference', 'ratio', or 'percent'.")
    df_out = df.copy()
    missing = [col for col in columns if col not in df_out.columns]
    if missing:
        raise KeyError(f"Columns {missing} not found in dataframe for excess calculation.")
    df_out = df_out.sort_values(year_col)
    unique_years = df_out[year_col].dropna().unique()
    unique_years.sort()
    replacements: Dict[str, str] = {}
    means: Dict[str, Dict[Any, float]] = {col: {} for col in columns}
    for i, year in enumerate(unique_years):
        lower_idx = max(0, i - window)
        upper_idx = min(len(unique_years) - 1, i + window)
        relevant_years = unique_years[lower_idx : upper_idx + 1]
        subset = df_out[df_out[year_col].isin(relevant_years)]
        year_means = subset[columns].mean()
        for col in columns:
            means[col][year] = year_means.get(col, np.nan)
    for col in columns:
        mean_col = f"{prefix}{col}__year_mean"
        df_out[mean_col] = df_out[year_col].map(means[col])
        suffix = {
            "difference": "excess",
            "ratio": "ratio",
            "percent": "percent",
        }[resolved_mode]
        new_col = f"{prefix}{col}_{suffix}"
        if resolved_mode == "difference":
            df_out[new_col] = df_out[col] - df_out[mean_col]
        elif resolved_mode == "ratio":
            df_out[new_col] = df_out[col] / df_out[mean_col]
        else:
            ratio = df_out[col] / df_out[mean_col]
            df_out[new_col] = (ratio - 1.0) * 100.0
        if replace:
            replacements[col] = new_col
    return df_out, replacements


def _build_default_joint_plan(
    y_vars: Sequence[str],
    x_vars: Sequence[str],
    control_vars: Sequence[str],
    entity_fixed_effects: Sequence[str],
    time_fixed_effects: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Return the default joint-regression plan used for LaTeX tables."""

    plan: Dict[str, List[Dict[str, Any]]] = {}
    fe_order = list(entity_fixed_effects) + list(time_fixed_effects)
    for y_var in y_vars:
        specs: List[Dict[str, Any]] = []
        base_spec = {
            "label": "All X",
            "x": list(x_vars),
            "controls": [],
            "entity_fe": [],
            "time_fe": [],
        }
        specs.append(base_spec)
        if control_vars:
            specs.append(
                {
                    "label": "All X + controls",
                    "x": list(x_vars),
                    "controls": list(control_vars),
                    "entity_fe": [],
                    "time_fe": [],
                }
            )
        if fe_order:
            first = fe_order[0]
            specs.append(
                {
                    "label": f"All X + {first}",
                    "x": list(x_vars),
                    "controls": list(control_vars),
                    "entity_fe": [first] if first in entity_fixed_effects else [],
                    "time_fe": [first] if first in time_fixed_effects else [],
                }
            )
            if len(fe_order) > 1:
                specs.append(
                    {
                        "label": "All X + all FE",
                        "x": list(x_vars),
                        "controls": list(control_vars),
                        "entity_fe": list(entity_fixed_effects),
                        "time_fe": list(time_fixed_effects),
                    }
                )
        plan[y_var] = specs
    return plan


def _normalise_joint_plan(
    plan: Optional[Dict[str, Any]],
    *,
    default_plan: Dict[str, List[Dict[str, Any]]],
    x_vars: Sequence[str],
    control_vars: Sequence[str],
    entity_fixed_effects: Sequence[str],
    time_fixed_effects: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Normalise user-provided LaTeX column plans."""

    if plan is None:
        return default_plan
    normalised: Dict[str, List[Dict[str, Any]]] = {}
    x_set = set(x_vars)
    ctrl_set = set(control_vars)
    entity_set = set(entity_fixed_effects)
    time_set = set(time_fixed_effects)
    for y_key, raw_specs in plan.items():
        specs_iter: List[Any]
        if isinstance(raw_specs, dict):
            specs_iter = list(raw_specs.values())
        else:
            specs_iter = list(raw_specs or [])
        if not specs_iter:
            normalised[y_key] = default_plan.get(y_key, [])
            continue
        resolved: List[Dict[str, Any]] = []
        for entry in specs_iter:
            if isinstance(entry, dict):
                vars_entry = entry.get("x") or entry.get("vars") or entry.get("independent")
                controls_entry = entry.get("controls") or entry.get("control") or []
                label = entry.get("label")
                entity_entry = entry.get("entity_fe") or entry.get("entity_fixed_effects") or []
                time_entry = entry.get("time_fe") or entry.get("time_fixed_effects") or []
            else:
                vars_entry = entry
                controls_entry = []
                label = None
                entity_entry = []
                time_entry = []
            vars_list = _ensure_list(vars_entry)
            controls_list = _ensure_list(controls_entry)
            entity_list = _ensure_list(entity_entry)
            time_list = _ensure_list(time_entry)
            inferred_x: List[str] = []
            inferred_ctrl: List[str] = []
            inferred_entity = list(dict.fromkeys(entity_list))
            inferred_time = list(dict.fromkeys(time_list))
            if not vars_list and not inferred_entity and not inferred_time:
                vars_list = list(x_vars)
            for name in vars_list:
                if name in x_set:
                    inferred_x.append(name)
                elif name in ctrl_set:
                    inferred_ctrl.append(name)
                elif name in entity_set:
                    if name not in inferred_entity:
                        inferred_entity.append(name)
                elif name in time_set:
                    if name not in inferred_time:
                        inferred_time.append(name)
                else:
                    inferred_x.append(name)
            for ctrl in controls_list:
                if ctrl not in inferred_ctrl:
                    inferred_ctrl.append(ctrl)
            if not inferred_x:
                raise ValueError("Each LaTeX column specification must include at least one regressor.")
            resolved.append(
                {
                    "label": label,
                    "x": inferred_x,
                    "controls": inferred_ctrl,
                    "entity_fe": inferred_entity,
                    "time_fe": inferred_time,
                }
            )
        normalised[y_key] = resolved
    for y_var, specs in default_plan.items():
        normalised.setdefault(y_var, specs)
    return normalised


def _format_coefficient(
    coef: float,
    se: Optional[float],
    pval: Optional[float],
    *,
    float_fmt: str,
) -> Tuple[str, str]:
    """Return formatted coefficient and standard error strings with stars."""

    if coef is None or (isinstance(coef, float) and np.isnan(coef)):
        return "-", ""
    stars = ""
    if pval is not None:
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.1:
            stars = "*"
    coef_part = f"{float_fmt.format(coef)}{stars}"
    if se is None or (isinstance(se, float) and np.isnan(se)):
        return coef_part, ""
    se_part = float_fmt.format(se)
    return coef_part, f"({se_part})"


def _cache_f_stat_failure(res: sm.regression.linear_model.RegressionResultsWrapper) -> None:
    """Store NaN F-statistics on the statsmodels results cache."""

    cache = getattr(res, "_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(res, "_cache", cache)
    cache["fvalue"] = np.nan
    cache["f_pvalue"] = np.nan


def _safe_fvalue(res: sm.regression.linear_model.RegressionResultsWrapper) -> float:
    """Return ``res.fvalue`` while gracefully handling statsmodels failures."""

    try:
        return float(res.fvalue)
    except ValueError:
        _cache_f_stat_failure(res)
        return np.nan


def _results_to_dict(
    res: sm.regression.linear_model.RegressionResultsWrapper,
    *,
    display_varnames: List[str],
    param_lookup: Dict[str, str],
) -> Dict[str, Any]:
    """Convert a statsmodels result object to the dictionary structure used here."""

    params = res.params
    se = res.bse
    f_value = _safe_fvalue(res)
    return {
        "coef": params,
        "se": se,
        "t": res.tvalues,
        "p": res.pvalues,
        "r2": getattr(res, "rsquared", np.nan),
        "adj_r2": getattr(res, "rsquared_adj", np.nan),
        "n": int(res.nobs),
        "k": len(params) - 1 if "Intercept" in params.index else len(params),
        "rse": np.sqrt(res.mse_resid) if hasattr(res, "mse_resid") else np.nan,
        "F": f_value,
        "resid": res.resid,
        "varnames": list(params.index),
        "display_varnames": display_varnames,
        "param_lookup": param_lookup,
        "sm_results": res,
    }


def _fit_formula_model(
    data: pd.DataFrame,
    *,
    y: str,
    main_vars: Sequence[str],
    main_display: Sequence[str],
    controls: Sequence[str],
    control_display: Sequence[str],
    robust: bool,
    entity_fe: Sequence[str],
    time_fe: Sequence[str],
    interaction_terms: bool,
    include_intercept: bool,
    cluster_cols: Sequence[str],
) -> Tuple[Dict[str, Any], str]:
    """Fit an OLS model via formulas, optionally with fixed effects."""

    rhs_terms = [f"Q('{var}')" for var in main_vars]
    rhs_terms.extend(f"Q('{var}')" for var in controls)
    for entity in entity_fe:
        rhs_terms.append(f"C(Q('{entity}'))")
    for time in time_fe:
        rhs_terms.append(f"C(Q('{time}'))")
    if interaction_terms:
        if entity_fe and time_fe:
            for entity in entity_fe:
                for time in time_fe:
                    rhs_terms.append(f"C(Q('{entity}')):C(Q('{time}'))")
        else:
            # Allow interaction terms among same-type fixed effects when only one
            # class is provided (e.g. two entity effects).
            fe_collection = entity_fe if entity_fe else time_fe
            for first, second in combinations(fe_collection, 2):
                rhs_terms.append(f"C(Q('{first}')):C(Q('{second}'))")
    if not rhs_terms:
        rhs_terms = ["1"]
    formula = f"Q('{y}') ~ " + " + ".join(rhs_terms)
    if not include_intercept:
        formula += " - 1"
    model = smf.ols(formula=formula, data=data)
    fit_kwargs: Dict[str, Any] = {}
    if cluster_cols:
        groups = _cluster_groups(data, cluster_cols)
        fit_kwargs["cov_type"] = "cluster"
        fit_kwargs["cov_kwds"] = {"groups": groups}
    elif robust:
        fit_kwargs["cov_type"] = "HC3"
    try:
        res = model.fit(**fit_kwargs)
    except ValueError:
        if fit_kwargs.get("cov_type") == "HC3":
            fit_kwargs["cov_type"] = "HC1"
            res = model.fit(**fit_kwargs)
        else:
            raise
    display_varnames: List[str] = []
    param_lookup: Dict[str, str] = {}
    if include_intercept and "Intercept" in res.params.index:
        display_varnames.append("Intercept")
        param_lookup["Intercept"] = "Intercept"
    for var, disp in zip(main_vars, main_display):
        key = f"Q('{var}')"
        if key in res.params.index:
            display_varnames.append(disp)
            param_lookup[disp] = key
    for var, disp in zip(controls, control_display):
        key = f"Q('{var}')"
        if key in res.params.index:
            display_varnames.append(disp)
            param_lookup[disp] = key
    result = _results_to_dict(res, display_varnames=display_varnames, param_lookup=param_lookup)
    return result, formula


def build_regression_latex(
    results: Dict[Tuple[str, str], Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None,
    *,
    rename_map: Optional[Dict[str, str]] = None,
) -> str:
    """Create a LaTeX regression table from ``regression_plot`` results.

    ``build_regression_latex`` is designed to work out-of-the-box using the
    metadata produced by :func:`regression_plot`.  Passing ``options=None`` or
    ``{}`` will emit a sensible default table that lists every model (simple and
    with controls) that was estimated.  Advanced layouts can still be achieved
    by providing a configuration dictionary.  The most common keys are:

    ``columns``
        A list describing which models should appear as columns.  Each entry is
        a mapping with ``key`` (tuple of ``(y, x)``) and ``model`` (``"simple"``
        or ``"with_controls"``).  ``label`` and ``dependent_label`` override the
        column heading and dependent variable label respectively.
    ``row_order``
        Ordered list of variable display names to keep.  By default all
        available coefficients are displayed.
    ``include_intercept``
        Whether the intercept should be shown (defaults to ``False``).
    ``float_format``
        Format string used for coefficients and summary statistics.
    ``include_controls_row``
        Toggle the summary row that indicates whether controls are present.
    ``include_fe_rows``
        When ``True`` (default) each supplied fixed-effect column is listed as its
        own row labelled by the variable name.  Rows are omitted entirely when
        no fixed effects are specified.
    ``include_cluster_row``
        Controls whether cluster indicators appear.  Each cluster column is
        displayed on its own row when at least one model clusters on it.
    ``include_model_numbers``
        When ``True`` (default) a numbered header row ``(1)``, ``(2)``, … is
        added above the coefficient block.
    ``show_model_labels``
        Display the human-readable labels for each model beneath the numbered
        headers.  Useful when you still want descriptive titles in addition to
        the standard numbering.
    ``notes``
        Text displayed in the footnote row.  Defaults to the conventional
        significance legend.  Set to ``None`` or ``""`` to omit the row.
    ``column_spacing``
        Point value used in ``\\extracolsep`` to control column spacing
        (default ``5``).
    ``max_width``
        Value inserted into the surrounding ``adjustbox`` environment to cap
        the table width (default ``\textwidth``).
    ``caption`` / ``label``
        Metadata for the LaTeX table environment.
    ``save_path``
        Path on disk to write the LaTeX string to (optional).

    Examples
    --------
    >>> summary = regression_plot(  # doctest: +SKIP
    ...     df,
    ...     x="treatment",
    ...     y="outcome",
    ...     controls=["age", "income"],
    ...     latex_options=True,
    ... )
    >>> print(summary["latex_table"])  # doctest: +SKIP

    Passing ``latex_options=True`` during :func:`regression_plot` automatically
    adds a ``"latex_table"`` entry to the returned dictionary.  To customise the
    layout, supply a dictionary such as ``{"caption": "My Results"}`` or a
    detailed ``{"columns": [...], "row_order": [...]}`` specification.  The
    ``options`` argument here mirrors that structure so you can also refine the
    table post-hoc.

    Parameters
    ----------
    results : dict
        Output of :func:`regression_plot`.
    options : dict, optional
        Table configuration overriding the defaults listed above.
    rename_map : dict, optional
        Mapping from original variable names to pretty labels.
    """

    rename_map = rename_map or {}
    if options is None:
        options = {}
    elif not isinstance(options, dict):
        raise TypeError("options must be a mapping or None")
    else:
        options = dict(options)
    if "caption" not in options:
        options["caption"] = "Regression results"
    if "label" not in options:
        options["label"] = "tab:regression_results"
    columns_spec = options.get("columns")
    joint_meta = results.get("_joint_columns") if isinstance(results, dict) else None
    if not columns_spec:
        columns_spec = []
        if isinstance(joint_meta, dict) and joint_meta:
            for entries in joint_meta.values():
                for entry in entries:
                    entry_spec = {
                        "key": tuple(entry["key"]),
                        "model": entry.get("model", "joint"),
                        "label": entry.get("label"),
                        "dependent_label": entry.get("dependent_label"),
                    }
                    columns_spec.append(entry_spec)
        if not columns_spec:
            for key, model_dict in results.items():
                if (
                    not isinstance(key, tuple)
                    or len(key) != 2
                    or not isinstance(model_dict, dict)
                ):
                    continue
                y_var, x_var = key
                dep_label = rename_map.get(y_var, y_var)
                indep_label = rename_map.get(x_var, x_var)
                if model_dict.get("simple") is not None:
                    columns_spec.append({
                        "key": (y_var, x_var),
                        "model": "simple",
                        "label": f"{dep_label} ~ {indep_label}",
                        "dependent_label": dep_label,
                    })
                if model_dict.get("with_controls") is not None:
                    columns_spec.append({
                        "key": (y_var, x_var),
                        "model": "with_controls",
                        "label": f"{dep_label} ~ {indep_label} + controls",
                        "dependent_label": dep_label,
                    })
    models: List[Dict[str, Any]] = []
    for spec in columns_spec:
        key = tuple(spec.get("key", ()))
        if len(key) != 2:
            raise ValueError("Each column specification must include a (y, x) key tuple.")
        if key not in results:
            raise KeyError(f"Result for key {key} not found.")
        model_name = spec.get("model", "simple")
        model_entry = results[key].get(model_name)
        if model_entry is None:
            continue
        label = spec.get("label") or f"Model: {key[0]} ~ {key[1]}"
        dependent_label = spec.get("dependent_label", rename_map.get(key[0], key[0]))
        models.append({
            "label": label,
            "dependent": dependent_label,
            "result": model_entry,
        })
    if not models:
        raise ValueError("No models found for LaTeX table generation.")
    include_intercept = bool(options.get("include_intercept", False))
    float_fmt = options.get("float_format", "{:.3f}")
    row_order = options.get("row_order")
    if row_order is None:
        row_order = []
        for model in models:
            for name in model["result"].get("display_varnames", []):
                if not include_intercept and name == "Intercept":
                    continue
                if name not in row_order:
                    row_order.append(name)
    caption = options.get("caption", "Regression results")
    label = options.get("label", "tab:regression_results")
    include_stats = options.get("include_stats", True)
    include_adj_r2 = options.get("include_adj_r2", False)
    include_controls_row = options.get("include_controls_row", False)
    include_fe_rows = options.get("include_fe_rows", True)
    include_cluster_row = options.get("include_cluster_row", True)
    # Track display labels for fixed effects and cluster columns that appear in
    # any model so that we only emit rows for the variables that are actually
    # specified.
    fe_display_lookup: OrderedDict[str, str] = OrderedDict()
    cluster_display_lookup: OrderedDict[str, str] = OrderedDict()
    if include_fe_rows or include_cluster_row:
        for model in models:
            meta = model["result"].get("metadata", {})
            fe_meta = meta.get("fixed_effects", {})
            if include_fe_rows:
                for key in ("entity", "time"):
                    for fe_name in _ensure_list(fe_meta.get(key)):
                        display = rename_map.get(fe_name, fe_name)
                        if fe_name not in fe_display_lookup:
                            fe_display_lookup[fe_name] = display
            if include_cluster_row:
                for cluster_name in _ensure_list(fe_meta.get("cluster")):
                    display = rename_map.get(cluster_name, cluster_name)
                    if cluster_name not in cluster_display_lookup:
                        cluster_display_lookup[cluster_name] = display
    show_dependent = options.get("show_dependent", True)
    include_model_numbers = bool(options.get("include_model_numbers", True))
    show_model_labels = bool(options.get("show_model_labels", False))
    notes_text = options.get(
        "notes",
        r"\textsuperscript{*} p\textless{}0.1; "
        r"\textsuperscript{**} p\textless{}0.05; "
        r"\textsuperscript{***} p\textless{}0.01",
    )
    max_width = options.get("max_width", r"\textwidth")
    column_spacing = options.get("column_spacing", 5)
    column_spec = f"@{{\\extracolsep{{{column_spacing}pt}}}}l" + "c" * len(models)
    row_end = " " + "\\\\"
    lines = [
        r"\begin{table}[!htbp] \centering",
        rf"\begin{{adjustbox}}{{max width={max_width}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\\[-1.8ex]\hline \hline \\[-1.8ex]",
    ]
    if show_dependent:
        dependent_labels = {model["dependent"] for model in models}
        if len(dependent_labels) == 1:
            dep_label_text = next(iter(dependent_labels))
        else:
            dep_label_text = ", ".join(sorted(dependent_labels))
        lines.append(
            "& "
            + rf"\multicolumn{{{len(models)}}}{{c}}{{\textit{{Dependent variable: {dep_label_text}}}}}"
            + row_end
        )
        lines.append(r"\cline{2-" + str(len(models) + 1) + "}")
    if include_model_numbers:
        number_row = [f"({idx})" for idx in range(1, len(models) + 1)]
        lines.append(r"\\[-1.8ex] & " + " & ".join(number_row) + row_end)
    if show_model_labels:
        label_cells = ["", *[model["label"] for model in models]]
        if include_model_numbers:
            lines.append(" & ".join(label_cells) + row_end)
        else:
            lines.append(r"\\[-1.8ex] " + " & ".join(label_cells) + row_end)
    lines.append(r"\hline \\[-1.8ex]")
    for row in row_order:
        if not include_intercept and row == "Intercept":
            continue
        display_row = rename_map.get(row, row)
        row_entries = [f" {display_row}"]
        se_entries = ["  "]
        show_se_row = False
        for model in models:
            res = model["result"]
            coef_series = res["coef"]
            se_series = res["se"]
            pvals = res["p"]
            lookup = res.get("param_lookup", {})
            if not isinstance(coef_series, pd.Series):
                coef_series = pd.Series(coef_series, index=res.get("display_varnames"))
            if not isinstance(se_series, pd.Series):
                se_series = pd.Series(se_series, index=res.get("display_varnames"))
            if not isinstance(pvals, pd.Series):
                pvals = pd.Series(pvals, index=res.get("display_varnames"))
            key = lookup.get(row, row)
            if key in coef_series.index:
                coef_val = float(coef_series[key])
                se_val = float(se_series[key]) if key in se_series.index else None
                p_val = float(pvals[key]) if key in pvals.index else None
                coef_text, se_text = _format_coefficient(
                    coef_val,
                    se_val,
                    p_val,
                    float_fmt=float_fmt,
                )
            else:
                coef_text, se_text = "-", ""
            row_entries.append(coef_text)
            se_entries.append(se_text)
            show_se_row = show_se_row or bool(se_text)
        lines.append(" & ".join(row_entries) + row_end)
        if show_se_row:
            lines.append(" & ".join(se_entries) + row_end)
    if include_stats or include_controls_row or include_fe_rows or include_cluster_row:
        lines.append(r"\hline \\[-1.8ex]")
    if include_stats:
        def _fmt_stat(val: Any) -> str:
            return float_fmt.format(val) if pd.notnull(val) else "-"
        obs_row = [" Observations"]
        r2_row = [" $R^2$"]
        adj_row = [" Adjusted $R^2$"]
        for model in models:
            res = model["result"]
            n_val = res.get("n", np.nan)
            obs_row.append(str(int(n_val)) if pd.notnull(n_val) else "-")
            r2_row.append(_fmt_stat(res.get("r2", np.nan)))
            if include_adj_r2:
                adj_row.append(_fmt_stat(res.get("adj_r2", np.nan)))
        lines.append(" & ".join(obs_row) + row_end)
        lines.append(" & ".join(r2_row) + row_end)
        if include_adj_r2:
            lines.append(" & ".join(adj_row) + row_end)
    if include_controls_row:
        ctrl_row = [" Controls"]
        for model in models:
            meta = model["result"].get("metadata", {})
            has_controls = bool(meta.get("controls_included"))
            ctrl_row.append(r"\checkmark" if has_controls else "-")
        lines.append(" & ".join(ctrl_row) + row_end)
    if include_fe_rows and fe_display_lookup:
        for fe_name, fe_display in fe_display_lookup.items():
            row = [f" {fe_display}"]
            for model in models:
                meta = model["result"].get("metadata", {})
                fe_meta = meta.get("fixed_effects", {})
                fe_values = set()
                for key in ("entity", "time"):
                    fe_values.update(_ensure_list(fe_meta.get(key)))
                row.append(r"\checkmark" if fe_name in fe_values else "-")
            lines.append(" & ".join(row) + row_end)
    if include_cluster_row and cluster_display_lookup:
        for cluster_name, cluster_display in cluster_display_lookup.items():
            row = [f" {cluster_display}"]
            for model in models:
                meta = model["result"].get("metadata", {})
                fe_meta = meta.get("fixed_effects", {})
                clusters = set(_ensure_list(fe_meta.get("cluster")))
                row.append(r"\checkmark" if cluster_name in clusters else "-")
            lines.append(" & ".join(row) + row_end)
    lines.append(r"\hline \hline \\[-1.8ex]")
    if notes_text:
        note_row = (
            r"\textit{Note:} & "
            + rf"\multicolumn{{{len(models)}}}{{r}}{{{notes_text}}}"
            + row_end
        )
        lines.append(note_row)
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    latex = "\n".join(lines)
    save_path = options.get("save_path")
    if save_path:
        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(latex)
    return latex

# Set monospace font for consistency
plt.rcParams["font.family"] = "monospace"


def _z(s: pd.Series) -> pd.Series:
    """Return a z‑scored version of a pandas Series (population std)."""
    return (s - s.mean()) / s.std(ddof=0)


def fit_ols(
    y: np.ndarray,
    X: np.ndarray,
    *,
    robust: bool = True,
    varnames: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fit an OLS regression using statsmodels and return a dictionary of results.

    When ``varnames`` is supplied and its length matches the number of columns
    in ``X``, the design matrix is converted to a pandas DataFrame with those
    column names so that parameter estimates in the statsmodels summary carry
    meaningful names instead of ``x1``, ``x2``.  Robust HC3 standard errors
    are applied by default.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Dependent variable values.
    X : ndarray of shape (n, k+1)
        Design matrix including an intercept column.  If ``varnames`` is
        provided, it should contain ``k+1`` names corresponding to the
        columns of ``X``.
    robust : bool, default True
        Use HC3 robust covariance estimates.  If False, classical OLS
        standard errors are used.
    varnames : list of str, optional
        Column names for ``X``.  If provided and valid, these names are used
        in the statsmodels regression and stored in the returned dictionary.

    Returns
    -------
    dict
        Contains coefficient arrays, standard errors, t‑values, p‑values, R²,
        adjusted R², residuals, the fitted statsmodels results object, and
        the variable names used.  If ``varnames`` is ``None`` or mismatched,
        parameter names default to ``const``, ``x1``, ``x2``, etc.
    """
    # Wrap exogenous matrix in DataFrame when names are provided
    if varnames is not None and len(varnames) == X.shape[1]:
        exog = pd.DataFrame(X, columns=varnames)
    else:
        exog = X
        varnames = None
    n, k_plus1 = X.shape
    k = k_plus1 - 1
    model = sm.OLS(y, exog)
    res = model.fit()
    # Apply robust covariance if requested
    if robust:
        try:
            use = res.get_robustcov_results(cov_type="HC3")
        except Exception:
            use = res.get_robustcov_results(cov_type="HC1")
    else:
        use = res
    # Ensure statsmodels returns Series even when given raw ndarrays
    params = use.params
    bse = use.bse
    tvalues = use.tvalues
    pvalues = use.pvalues
    if isinstance(params, np.ndarray):
        if varnames is None:
            try:
                varnames = list(res.model.exog_names)
            except Exception:  # pragma: no cover - very unlikely branch
                varnames = [f"x{i}" for i in range(params.shape[0])]
        params = pd.Series(params, index=varnames)
        bse = pd.Series(np.asarray(bse), index=varnames)
        tvalues = pd.Series(np.asarray(tvalues), index=varnames)
        pvalues = pd.Series(np.asarray(pvalues), index=varnames)
    # Extract statistics
    adj_r2 = res.rsquared_adj
    resid = res.resid
    df_resid = n - k_plus1
    rse = np.sqrt((resid @ resid) / df_resid) if df_resid > 0 else np.nan
    F_stat = _safe_fvalue(res) if k > 0 else np.nan
    display_names = varnames or list(params.index)
    return {
        "coef": params,
        "se": bse,
        "t": tvalues,
        "p": pvalues,
        "r2": res.rsquared,
        "adj_r2": adj_r2,
        "n": n,
        "k": k,
        "rse": rse,
        "F": F_stat,
        "resid": resid,
        "varnames": varnames,
        "display_varnames": display_names,
        "param_lookup": {name: name for name in display_names},
        "sm_results": res,
    }


def _print_table(res: Dict[str, Any], *, tablefmt: str = "github") -> None:
    """Print a statsmodels summary and a compact coefficient table.

    If the ``tabulate`` library is available, it is used for formatting; otherwise
    pandas' string representation is used.  ``varnames`` should match the
    ordering of the coefficient vector.
    """
    # Print the full statsmodels summary for context
    sm_results = res.get("sm_results")
    if sm_results is not None and hasattr(sm_results, "summary"):
        print(sm_results.summary())
    display_names = res.get("display_varnames") or list(res["coef"].index)
    lookup = res.get("param_lookup", {name: name for name in display_names})
    rows = []
    for name in display_names:
        param_name = lookup.get(name, name)
        if param_name not in res["coef"].index:
            continue
        rows.append({
            "variable": name,
            "coef": res["coef"][param_name],
            "se(HC3)": res["se"][param_name],
            "t": res["t"][param_name],
            "p": res["p"][param_name],
        })
    tbl = pd.DataFrame(rows).set_index("variable") if rows else pd.DataFrame()
    if tabulate is not None:
        print(tabulate(tbl.round(7), headers="keys", tablefmt=tablefmt, showindex=True))
    else:
        print(tbl.round(7).to_string())
    print(f"\nR² = {res['r2']:.4f}, adj. R² = {res['adj_r2']:.4f}, n = {res['n']}")
    print("-" * 60)


def regression_plot(
    df: pd.DataFrame,
    *,
    x: Union[str, Iterable[str]],
    y: Union[str, Iterable[str]],
    controls: Optional[Union[str, Iterable[str]]] = None,
    rename_map: Optional[Dict[str, str]] = None,
    zscore_x: bool = False,
    zscore_y: bool = False,
    bins: int = 20,
    cmap: str = "rainbow",
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 300,
    wrap_width: int = 60,
    show_plots: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    save_format: Optional[str] = None,
    bbox_inches: Optional[str] = "tight",
    print_save_path: bool = True,
    tablefmt: str = "github",
    robust: bool = True,
    print_summary: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    excess_year_col: Optional[str] = None,
    excess_window: Optional[int] = None,
    excess_mode: str = "difference",
    excess_columns: Optional[Union[str, Iterable[str]]] = None,
    excess_replace: bool = True,
    excess_prefix: str = "",
    entity_fixed_effects: Optional[Union[str, Iterable[str]]] = None,
    time_fixed_effects: Optional[Union[str, Iterable[str]]] = None,
    fe_interactions: bool = False,
    cluster: Optional[Union[str, Iterable[str]]] = None,
    include_intercept: Optional[bool] = None,
    use_formula: Optional[bool] = None,
    latex_column_plan: Optional[Dict[str, Any]] = None,
    latex_options: Union[bool, Dict[str, Any], None] = True,
    fixed_effect_min_share: float = 0.01,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Run OLS regressions for each combination of ``y`` and ``x`` variables.

    Parameters accept either a string (single variable) or an iterable of
    strings.  For each pair, two models are estimated: one with just the
    independent variable and one including any specified ``controls``.  When
    ``show_plots`` is True, a binned scatter plot with quantile bins and error
    bars is displayed.  Provide ``save_path`` to save plots to a directory or
    filename (format inferred from the extension or overridden via
    ``save_format``).  If ``zscore_x`` or ``zscore_y`` is True, the respective
    variables are standardised before analysis (but the original variables
    remain untouched in the output).

    ``excess_year_col`` turns on peer-adjusted ("excess") outcome variables.
    Provide the column that defines peer groups (typically a year column) and a
    positive ``excess_window`` describing how many peers before/after should be
    used when computing the rolling mean.  By default every dependent variable
    in ``y`` is adjusted; override with ``excess_columns`` if you also want the
    adjustment applied to other variables.  ``excess_mode`` switches between the
    default difference-from-mean, a ratio-to-mean calculation, or a
    percent-change-from-mean transformation, while
    ``excess_replace`` controls whether the adjusted columns are automatically
    used in the regression.  ``excess_prefix`` can be used to disambiguate the
    derived columns when running several specifications in succession.

    ``entity_fixed_effects`` and ``time_fixed_effects`` allow inclusion of one or
    multiple fixed-effect dimensions via statsmodels' formula API.  Provide a
    string or list of column names for each.  ``fe_interactions`` adds
    interaction terms between every specified entity/time pair (or pairwise
    interactions within a single group when only entity or time effects are
    supplied).  ``fixed_effect_min_share`` controls how rare categories are
    handled: levels appearing in fewer than the specified share of rows (default
    1%) are pooled into a combined "rare" bucket that serves as the baseline
    category.  ``cluster`` can provide columns for clustered standard errors.
    ``include_intercept`` overrides the automatic intercept handling when fixed
    effects are present, while ``use_formula`` forces the formula-based path even
    without fixed effects.  ``latex_column_plan`` customises which joint
    regressions populate the LaTeX table: by default, every dependent variable is
    paired with (i) a specification containing all ``x`` variables (and
    ``controls`` when provided), (ii) if fixed effects exist, a version with the
    first available fixed effect, and (iii) a version with every supplied fixed
    effect.  Supply a dictionary such as ``{"adoption lag": [["var_a", "var_b"],
    ["var_a", "primary category"]]}`` to override those defaults.  Entries may be
    either sequences of variable names (``x``/``controls``/fixed-effect
    columns) or dictionaries with ``{"label": ..., "x": [...], "controls": [...],
    "entity_fe": [...], "time_fe": [...]}``.

    ``latex_options`` controls LaTeX output.  By default this is ``True``,
    which means :func:`build_regression_latex` is run automatically and the
    resulting string is stored under ``"latex_table"`` in the returned
    dictionary (and printed unless ``{"print": False}`` is supplied).  Pass a
    configuration dictionary to customise the table or ``False``/``None`` to
    disable LaTeX creation entirely.

    Returns a dictionary keyed by ``(y_var, x_var)`` with entries ``'simple'``,
    ``'with_controls'``, and ``'binned_df'`` along with metadata for downstream
    table construction.  When controls are not provided, ``'with_controls'``
    will be ``None``.  Additional joint specifications used in the LaTeX table
    appear under keys of the form ``(y_var, 'joint_{i}')`` with a ``'joint'``
    entry describing the combined regression.  ``results['_joint_columns']``
    lists the default (or user-supplied) LaTeX column plan for reference.

    Examples
    --------
    >>> results = regression_plot(  # doctest: +SKIP
    ...     df,
    ...     x="treatment",
    ...     y=["outcome"],
    ...     controls=["age", "income"],
    ...     excess_year_col="year",
    ...     excess_window=2,
    ...     latex_options=True,
    ... )
    >>> sorted(results.keys())  # doctest: +SKIP
    [('outcome', 'joint_0'), ('outcome', 'treatment'), 'latex_table']
    """
    x_vars = _ensure_list(x)
    y_vars = _ensure_list(y)
    control_vars = _ensure_list(controls)
    if not x_vars or not y_vars:
        raise ValueError("At least one x and one y variable must be provided.")
    rename_map = dict(rename_map or {})
    prepared_df = df.copy()
    replacements: Dict[str, str] = {}
    if excess_year_col is not None:
        columns = _ensure_list(excess_columns)
        if not columns:
            columns = list(dict.fromkeys(y_vars))  # preserve order, default to y variables
        if excess_window is None or int(excess_window) <= 0:
            raise ValueError("excess_window must be a positive integer when excess_year_col is provided.")
        mode = (excess_mode or "difference").lower()
        prepared_df, replacements = _apply_year_excess(
            prepared_df,
            year_col=excess_year_col,
            window=int(excess_window),
            columns=columns,
            mode=mode,
            replace=bool(excess_replace),
            prefix=excess_prefix,
        )
        suffix = " (excess)" if mode == "difference" else " (ratio)"
        for original, new in replacements.items():
            if original in rename_map and new not in rename_map:
                rename_map[new] = rename_map[original] + suffix
            elif new not in rename_map:
                rename_map[new] = original + suffix
    x_actual = {var: replacements.get(var, var) for var in x_vars}
    y_actual = {var: replacements.get(var, var) for var in y_vars}
    controls_actual = {var: replacements.get(var, var) for var in control_vars}
    processed_df = prepared_df.copy()
    for original, actual in x_actual.items():
        rename_map.setdefault(original, original)
        rename_map.setdefault(actual, rename_map.get(original, actual))
    for original, actual in y_actual.items():
        rename_map.setdefault(original, original)
        rename_map.setdefault(actual, rename_map.get(original, actual))
    for original, actual in controls_actual.items():
        rename_map.setdefault(original, original)
        rename_map.setdefault(actual, rename_map.get(original, actual))
    if zscore_x:
        for var, actual in list(x_actual.items()):
            numeric = pd.to_numeric(processed_df[actual], errors="coerce")
            new_col = f"{actual}_z"
            processed_df[new_col] = _z(numeric)
            base_label = rename_map.get(var, var)
            rename_map[new_col] = f"{base_label} (z)"
            x_actual[var] = new_col
    if zscore_y:
        for var, actual in list(y_actual.items()):
            numeric = pd.to_numeric(processed_df[actual], errors="coerce")
            new_col = f"{actual}_z"
            processed_df[new_col] = _z(numeric)
            base_label = rename_map.get(var, var)
            rename_map[new_col] = f"{base_label} (z)"
            y_actual[var] = new_col
    fe_entity = list(dict.fromkeys(_ensure_list(entity_fixed_effects)))
    fe_time = list(dict.fromkeys(_ensure_list(time_fixed_effects)))
    cluster_cols = list(dict.fromkeys(_ensure_list(cluster)))
    fe_min_share = max(float(fixed_effect_min_share or 0.0), 0.0)
    fe_min_share = min(fe_min_share, 1.0)
    entity_base_levels: Dict[str, Any] = {}
    entity_rare_levels: Dict[str, List[Any]] = {}
    time_base_levels: Dict[str, Any] = {}
    time_rare_levels: Dict[str, List[Any]] = {}
    if fe_entity:
        entity_base_levels, entity_rare_levels = _prepare_fixed_effect_columns(
            processed_df, fe_entity, min_share=fe_min_share
        )
    if fe_time:
        time_base_levels, time_rare_levels = _prepare_fixed_effect_columns(
            processed_df, fe_time, min_share=fe_min_share
        )

    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    save_path_is_file = False
    if save_path is not None:
        expanded = os.path.expandvars(os.path.expanduser(str(save_path)))
        save_path_is_file = bool(Path(expanded).suffix)
    total_plot_batches = len(x_vars) * len(y_vars) if save_path_is_file else 1
    plot_counter = 0

    def _resolve_column(name: str) -> Tuple[str, str]:
        """Return the actual dataframe column and display label for ``name``."""

        if name in x_actual:
            actual_col = x_actual[name]
        elif name in controls_actual:
            actual_col = controls_actual[name]
        elif name in replacements:
            actual_col = replacements[name]
        else:
            actual_col = name
        if actual_col not in processed_df.columns:
            raise KeyError(f"Column '{name}' (resolved to '{actual_col}') not found in dataframe.")
        display = rename_map.get(actual_col, rename_map.get(name, name))
        rename_map.setdefault(actual_col, display)
        return actual_col, display
    if include_intercept is None:
        include_intercept = True
    else:
        include_intercept = bool(include_intercept)
    if use_formula is None:
        use_formula = bool(fe_entity or fe_time or cluster_cols)
    if cluster_cols:
        use_formula = True
    use_formula = bool(use_formula)
    for y_var in y_vars:
        y_col = y_actual[y_var]
        for x_var in x_vars:
            x_col = x_actual[x_var]
            # Create a copy for each pair to avoid side effects
            data = processed_df.copy()
            # Pretty names for axes and tables
            y_disp = rename_map.get(y_col, rename_map.get(y_var, y_var))
            x_disp = rename_map.get(x_col, rename_map.get(x_var, x_var))
            ctrl_disp = [rename_map.get(controls_actual[c], rename_map.get(c, c)) for c in control_vars]
            # Ensure variables are numeric; non-numeric rows dropped
            numeric_needed = [x_col, y_col] + [controls_actual[c] for c in control_vars]
            data[numeric_needed] = data[numeric_needed].apply(pd.to_numeric, errors="coerce")
            drop_subset = list(numeric_needed)
            drop_subset.extend(fe_entity)
            drop_subset.extend(fe_time)
            data = data.dropna(subset=drop_subset)
            x_use = x_col
            y_use = y_col
            # Binned scatter plot
            data["_bin"] = pd.qcut(data[x_use], q=bins, duplicates="drop")
            grp = data.groupby("_bin", observed=True)
            xm = grp[x_use].mean()
            ym = grp[y_use].mean()
            yerr = grp[y_use].apply(sem)
            should_plot = show_plots or save_path is not None
            if should_plot:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                ax.errorbar(xm, ym, yerr=yerr, fmt="o", color="black",
                            ecolor="black", capsize=3, markersize=6)
                colours = plt.get_cmap(cmap)(np.linspace(0, 1, len(xm)))
                ax.scatter(xm, ym, c=colours, s=50, zorder=3)
                title = f"{y_disp} vs. {x_disp}"
                ax.set_title(textwrap.fill(title, wrap_width))
                ax.set_xlabel(x_disp)
                ax.set_ylabel(y_disp)
                ax.grid(alpha=0.3)
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                safe_title = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_") or "regression_plot"
                _save_figure(
                    fig,
                    save_path=save_path,
                    default_name=safe_title,
                    save_format=save_format,
                    batch_idx=plot_counter,
                    total_batches=total_plot_batches,
                    bbox_inches=bbox_inches,
                    print_save_path=print_save_path,
                )
                plot_counter += 1
                if show_plots:
                    plt.show()
                else:
                    plt.close(fig)
            # Prepare design matrices and variable names
            y_arr = data[y_use].values
            # Simple model: intercept and primary x variable
            varnames_simple = ["Intercept", x_disp]
            ctrl_columns = [controls_actual[c] for c in control_vars]
            ctrl_display = ctrl_disp
            simple_res: Dict[str, Any]
            simple_formula = None
            metadata_base = {
                "y": y_var,
                "y_column": y_use,
                "y_display": y_disp,
                "x": x_var,
                "x_column": x_use,
                "x_display": x_disp,
                "controls": control_vars,
                "control_columns": ctrl_columns,
                "control_display": ctrl_disp,
                "controls_included": [],
                "fixed_effects": {
                    "entity": list(fe_entity),
                    "time": list(fe_time),
                    "cluster": list(cluster_cols),
                    "include_intercept": include_intercept,
                    "interaction_terms": fe_interactions,
                    "min_share": fe_min_share if fe_entity or fe_time else None,
                    "entity_base_levels": dict(entity_base_levels),
                    "time_base_levels": dict(time_base_levels),
                    "entity_rare_levels": {k: list(v) for k, v in entity_rare_levels.items()},
                    "time_rare_levels": {k: list(v) for k, v in time_rare_levels.items()},
                },
                "excess_replacements": replacements,
            }
            if use_formula:
                simple_res, simple_formula = _fit_formula_model(
                    data,
                    y=y_use,
                    main_vars=[x_use],
                    main_display=[x_disp],
                    controls=[],
                    control_display=[],
                    robust=robust,
                    entity_fe=fe_entity,
                    time_fe=fe_time,
                    interaction_terms=fe_interactions,
                    include_intercept=include_intercept,
                    cluster_cols=cluster_cols,
                )
            else:
                X_simple = np.column_stack([np.ones(len(data)), data[x_use].values])
                simple_res = fit_ols(y_arr, X_simple, robust=robust, varnames=varnames_simple)
                simple_res["varnames"] = varnames_simple
            if "varnames" not in simple_res and "display_varnames" in simple_res:
                simple_res["varnames"] = simple_res["display_varnames"]
            simple_res.setdefault("metadata", {}).update(metadata_base | {"model": "simple", "formula": simple_formula})
            if print_summary:
                print(f"\n=== Model: {y_disp} ~ {x_disp} ===")
                _print_table(simple_res, tablefmt=tablefmt)
            # Fit controlled model if controls exist
            ctrl_res = None
            ctrl_formula = None
            if control_vars:
                if use_formula:
                    ctrl_res, ctrl_formula = _fit_formula_model(
                        data,
                        y=y_use,
                        main_vars=[x_use],
                        main_display=[x_disp],
                        controls=ctrl_columns,
                        control_display=ctrl_display,
                        robust=robust,
                        entity_fe=fe_entity,
                        time_fe=fe_time,
                        interaction_terms=fe_interactions,
                        include_intercept=include_intercept,
                        cluster_cols=cluster_cols,
                    )
                else:
                    arrays = [np.ones(len(data)), data[x_use].values]
                    for c in ctrl_columns:
                        arrays.append(data[c].values)
                    X_ctrl = np.column_stack(arrays)
                    varnames_ctrl = ["Intercept", x_disp] + ctrl_disp
                    ctrl_res = fit_ols(y_arr, X_ctrl, robust=robust, varnames=varnames_ctrl)
                    ctrl_res["varnames"] = varnames_ctrl
                if ctrl_res is not None:
                    if "varnames" not in ctrl_res and "display_varnames" in ctrl_res:
                        ctrl_res["varnames"] = ctrl_res["display_varnames"]
                    ctrl_res.setdefault("metadata", {}).update(
                        metadata_base
                        | {
                            "model": "with_controls",
                            "formula": ctrl_formula,
                            "controls_included": ctrl_columns,
                        }
                    )
                    if print_summary:
                        print(f"\n=== Model: {y_disp} ~ {x_disp} + controls ===")
                        _print_table(ctrl_res, tablefmt=tablefmt)
            # Store results keyed by (original y, original x)
            results[(y_var, x_var)] = {
                "simple": simple_res,
                "with_controls": ctrl_res,
                "binned_df": grp[[x_use, y_use]].mean(),
            }
            results[(y_var, x_var)]["metadata"] = metadata_base
    default_joint_plan = _build_default_joint_plan(
        y_vars,
        x_vars,
        control_vars,
        fe_entity,
        fe_time,
    )
    joint_plan = _normalise_joint_plan(
        latex_column_plan,
        default_plan=default_joint_plan,
        x_vars=x_vars,
        control_vars=control_vars,
        entity_fixed_effects=fe_entity,
        time_fixed_effects=fe_time,
    )
    joint_columns_meta: Dict[str, List[Dict[str, Any]]] = {}
    joint_counter = 0
    for y_var in y_vars:
        y_col = y_actual[y_var]
        y_disp = rename_map.get(y_col, rename_map.get(y_var, y_var))
        specs = joint_plan.get(y_var, [])
        column_entries: List[Dict[str, Any]] = []
        for spec in specs:
            spec_x_names = spec.get("x", [])
            spec_ctrl_names = spec.get("controls", [])
            entity_spec = [col for col in spec.get("entity_fe", []) if col in fe_entity]
            time_spec = [col for col in spec.get("time_fe", []) if col in fe_time]
            x_columns: List[str] = []
            x_display: List[str] = []
            for name in spec_x_names:
                actual, disp = _resolve_column(name)
                if actual not in x_columns:
                    x_columns.append(actual)
                    x_display.append(disp)
            ctrl_columns_spec: List[str] = []
            ctrl_display_spec: List[str] = []
            for name in spec_ctrl_names:
                actual, disp = _resolve_column(name)
                if actual not in ctrl_columns_spec:
                    ctrl_columns_spec.append(actual)
                    ctrl_display_spec.append(disp)
            if not x_columns:
                continue
            data = processed_df.copy()
            numeric_needed = [y_col] + x_columns + ctrl_columns_spec
            data[numeric_needed] = data[numeric_needed].apply(pd.to_numeric, errors="coerce")
            drop_subset = list(numeric_needed)
            drop_subset.extend(entity_spec)
            drop_subset.extend(time_spec)
            data = data.dropna(subset=drop_subset)
            if data.empty:
                continue
            joint_use_formula = use_formula or bool(entity_spec) or bool(time_spec) or bool(cluster_cols)
            joint_formula = None
            if joint_use_formula:
                joint_res, joint_formula = _fit_formula_model(
                    data,
                    y=y_col,
                    main_vars=x_columns,
                    main_display=x_display,
                    controls=ctrl_columns_spec,
                    control_display=ctrl_display_spec,
                    robust=robust,
                    entity_fe=entity_spec,
                    time_fe=time_spec,
                    interaction_terms=fe_interactions,
                    include_intercept=include_intercept,
                    cluster_cols=cluster_cols,
                )
            else:
                y_arr = data[y_col].values
                arrays = [np.ones(len(data))]
                for col in x_columns:
                    arrays.append(data[col].values)
                for col in ctrl_columns_spec:
                    arrays.append(data[col].values)
                design = np.column_stack(arrays)
                varnames = ["Intercept"] + x_display + ctrl_display_spec
                joint_res = fit_ols(y_arr, design, robust=robust, varnames=varnames)
                joint_res["varnames"] = varnames
            joint_res.setdefault("metadata", {}).update(
                {
                    "y": y_var,
                    "y_column": y_col,
                    "y_display": y_disp,
                    "x": list(spec_x_names),
                    "x_columns": list(x_columns),
                    "x_display": list(x_display),
                    "controls": list(spec_ctrl_names),
                    "control_columns": list(ctrl_columns_spec),
                    "control_display": list(ctrl_display_spec),
                    "controls_included": list(ctrl_columns_spec),
                    "fixed_effects": {
                        "entity": list(entity_spec),
                        "time": list(time_spec),
                        "cluster": list(cluster_cols),
                        "include_intercept": include_intercept,
                        "interaction_terms": fe_interactions,
                        "min_share": fe_min_share if entity_spec or time_spec else None,
                        "entity_base_levels": dict(entity_base_levels),
                        "time_base_levels": dict(time_base_levels),
                        "entity_rare_levels": {k: list(v) for k, v in entity_rare_levels.items()},
                        "time_rare_levels": {k: list(v) for k, v in time_rare_levels.items()},
                    },
                    "model": "joint",
                    "formula": joint_formula,
                    "excess_replacements": replacements,
                }
            )
            spec_label = spec.get("label")
            if not spec_label:
                pieces = [" + ".join(x_display)] if x_display else []
                if ctrl_display_spec:
                    pieces.append(" + ".join(ctrl_display_spec))
                fe_bits = entity_spec + time_spec
                if fe_bits:
                    pieces.append(" + ".join(fe_bits))
                spec_label = " | ".join(pieces) if pieces else f"Model {joint_counter + 1}"
            joint_key = (y_var, f"joint_{joint_counter}")
            joint_counter += 1
            results[joint_key] = {"joint": joint_res}
            results[joint_key]["metadata"] = joint_res.get("metadata", {})
            column_entries.append(
                {
                    "key": joint_key,
                    "model": "joint",
                    "label": spec_label,
                    "dependent_label": y_disp,
                }
            )
        if column_entries:
            joint_columns_meta[y_var] = column_entries
    if joint_columns_meta:
        results["_joint_columns"] = joint_columns_meta
    latex_opts: Optional[Dict[str, Any]]
    if isinstance(latex_options, bool):
        latex_opts = {} if latex_options else None
    elif latex_options is None:
        latex_opts = None
    else:
        latex_opts = dict(latex_options)
    if latex_opts is not None:
        latex = build_regression_latex(results, latex_opts, rename_map=rename_map)
        results["latex_table"] = latex
        if latex_opts.get("print", True):
            print(latex)
    return results


def bar_plot(
    categories: Optional[Iterable[str]] = None,
    values: Optional[Iterable[float]] = None,
    *,
    data: Optional[pd.DataFrame] = None,
    category_column: Optional[str] = None,
    value_column: Optional[str] = None,
    value_agg: Union[str, Callable[[pd.Series], float]] = "mean",
    category_order: Optional[Iterable[str]] = None,
    title: str = "Bar Chart",
    x_label: str = "Category",
    y_label: str = "Value",
    as_percent: bool = False,
    cmap: str = "Reds",
    gradient_start: float = 0.3,
    gradient_end: float = 1.0,
    background_color: str = "#ffffff",
    font_family: str = "monospace",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 400,
    label_font_size: int = 12,
    tick_label_size: int = 11,
    title_font_size: int = 14,
    wrap_width: Optional[int] = 18,
    label_wrap_mode: str = "auto",
    min_wrap_chars: int = 12,
    rotate_xlabels: bool = False,
    annotation_font_size: int = 10,
    annotation_fontweight: str = "bold",
    precision: int = 3,
    value_axis_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
    orientation: str = "vertical",
    horizontal_label_fraction: float = 0.28,
    series_labels: Optional[Iterable[str]] = None,
    title_wrap: Optional[int] = None,
    error_bars: Optional[Union[Iterable[float], Dict[str, Iterable[float]], str, bool]] = None,
    error_bar_capsize: float = 4.0,
    max_bars_per_plot: Optional[int] = 12,
    sort_mode: Optional[str] = "descending",
    save_path: Optional[Union[str, Path]] = None,
    save_format: Optional[str] = None,
    bbox_inches: Optional[str] = "tight",
    print_save_path: bool = True,
    vertical_bar_width: float = 0.92,
    horizontal_bar_height: float = 0.7,
    min_category_fraction: float = 0.0,
    category_cap: Optional[int] = 12,
    excess_year_col: Optional[str] = None,
    excess_window: Optional[int] = None,
    excess_mode: str = "difference",
    excess_columns: Optional[Union[str, Iterable[str]]] = None,
    excess_replace: bool = True,
    excess_prefix: str = "",
    **legacy_kwargs: Any,
) -> None:
    """Draw a bar chart with flexible sizing, wrapping and optional extras.

    Parameters
    ----------
    categories, values : optional
        Pre-computed category labels and bar values.  ``values`` may be a
        one-dimensional iterable for single-series plots or a sequence of
        iterables (one per category) for grouped bars.  When omitted,
        ``data``/``category_column`` are used to aggregate the bars
        automatically: provide ``value_column`` for standard aggregations or
        omit it to plot category counts directly.  ``value_column`` may be a
        string or a sequence of strings for grouped bars.
    data : DataFrame, optional
        Raw data used to compute bar heights.  Requires ``category_column`` and
        ``value_column``.  When supplied, the ``value_column`` is aggregated by
        ``value_agg`` for each category and may optionally be transformed via
        the excess utilities (``excess_year_col`` et al.).
    value_agg : str or callable, default "mean"
        Aggregation applied to the ``value_column`` when ``data`` is provided.
    category_order : iterable of str, optional
        Explicit order for categories when aggregating from ``data``.  When
        omitted, categories follow the order returned by the aggregation.
    value_axis_limits : tuple, optional
        Explicit lower/upper bounds for the axis showing bar magnitudes (y-axis
        for vertical bars, x-axis for horizontal bars).
    orientation : {"vertical", "horizontal"}, default "vertical"
        Direction of the bars.  Horizontal bars flip the axes and swap the role
        of ``x_label``/``y_label``.
    horizontal_label_fraction : float, default 0.28
        Portion of the figure width reserved for y-axis labels when rendering
        horizontal charts.  Increase the fraction when labels are long and need
        more breathing room on the left side of the figure.
    series_labels : iterable of str, optional
        Labels used in the legend when plotting grouped/multi-value bars.  When
        omitted the column names (data) or ``Series i`` placeholders (values)
        are used.
    tick_label_size : int, default 11
        Font size applied to the tick labels along the categorical axis.
    auto sizing :
        When ``figsize`` is omitted, the function widens vertical charts or
        increases the height of horizontal charts based on how many categories
        are rendered in the current chunk.  The heuristic is intentionally
        gentle so wide charts stay legible without becoming excessively tall.
    max_bars_per_plot : int, optional
        Maximum number of categories to display per figure.  Additional
        categories are wrapped into subsequent plots.  When ``orientation`` is
        ``"horizontal"`` the limit is doubled to account for the additional
        vertical space.  Set to ``None`` or ``<= 0`` to disable batching.
    category_cap : int, optional
        When counting categories (``value_column`` omitted), retain only the
        ``category_cap`` most frequent categories by default.  Set to ``None``
        or ``<= 0`` to disable the cap.
    wrap_width : int, optional
        Base width (in characters) used when wrapping category labels.  Values
        ``<= 0`` disable wrapping entirely.  When ``None`` a default width of
        ``18`` characters is used before scaling.
    label_wrap_mode : {"auto", "fixed", "none"}, default "auto"
        Controls how category labels are wrapped.  ``"auto"`` retains the
        adaptive behaviour that widens the wrap width for long labels, while
        ``"fixed"`` enforces the value provided by ``wrap_width``.  Pass
        ``"none"`` to disable wrapping altogether regardless of
        ``wrap_width``.
    min_wrap_chars : int, default 12
        Minimum wrap width applied after auto-scaling so labels never collapse
        into unreadably narrow columns.
    title_wrap : optional
        Explicit wrap width (in characters) for the title.  When ``None`` a
        reasonable width is derived from the figure width.
    min_category_fraction : float, default 0.0
        Minimum share of the underlying observations required for a category to
        be included when aggregating directly from ``data``.  Categories with a
        relative frequency below this threshold are dropped before plotting.
        Set to ``0`` to keep all categories.
    sort_mode : {"descending", "ascending", "none", "random"}, optional
        Determines the automatic ordering of categories when ``category_order``
        is not provided.  Defaults to descending order of the aggregated bar
        totals.  Pass ``"none"`` to preserve the existing order or ``"random"``
        for a shuffled arrangement.
    save_path : path-like, optional
        Directory or filename where generated figures should be saved.  When
        omitted, plots are only displayed.  If a directory is supplied, files
        are named using the title plus a numerical suffix when multiple panels
        are created.  If a filename is supplied, the extension determines the
        output format unless ``save_format`` overrides it.
    save_format : str, optional
        Override the output format (e.g., ``"png"``, ``"pdf"``).  When omitted
        the format is inferred from ``save_path`` or defaults to PNG.
    bbox_inches : str, optional
        Optional ``bbox_inches`` argument passed through to ``fig.savefig``.
    print_save_path : bool, default True
        Whether to print the resolved save path when saving figures.
    vertical_bar_width, horizontal_bar_height : float, default (0.92, 0.7)
        Width/height of each bar group for the respective orientations.  For
        grouped bars the value is split evenly across the series.
    error_bars : iterable, dict, str or bool, optional
        Adds error bars to each bar.  Provide a sequence of symmetric error
        magnitudes, a mapping with ``{"lower": ..., "upper": ...}`` for
        asymmetric bars, a string (``"std"``, ``"sem"``, ``"ci90"``,
        ``"ci95"``, ``"ci99"``) to compute errors from ``data``, or pass
        ``True`` to automatically display 95% confidence intervals when
        ``data`` is supplied.
    excess_* : optional
        Match the ``regression_plot`` excess arguments, enabling automated
        rolling difference/ratio/percent-change calculations before
        aggregating the bars.

    Notes
    -----
    Values formatted as percentages simply append a ``%`` sign; supply values in
    the desired scale (e.g. 42 for ``42%``).  Large values are abbreviated using
    ``K``/``M`` suffixes when ``as_percent`` is False.
    """

    orientation = (orientation or "vertical").strip().lower()
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be 'vertical' or 'horizontal'.")
    min_wrap_chars = max(int(min_wrap_chars), 1)
    label_wrap_mode = (label_wrap_mode or "auto").strip().lower()
    if label_wrap_mode not in {"auto", "fixed", "none"}:
        raise ValueError("label_wrap_mode must be 'auto', 'fixed', or 'none'.")
    try:
        horizontal_label_fraction = float(horizontal_label_fraction)
    except (TypeError, ValueError):
        horizontal_label_fraction = 0.28
    horizontal_label_fraction = max(0.05, min(horizontal_label_fraction, 0.85))
    if legacy_kwargs:
        renamed = {"x_label_font_size": "tick_label_size"}
        removed = {
            "wrap_auto_scale": "wrap_auto_scale has been removed; label wrapping now adapts automatically.",
            "wrap_scale_reference": "wrap_scale_reference has been removed; the heuristics no longer need manual tuning.",
            "wrap_scale_limits": "wrap_scale_limits has been removed; labels use a softer built-in scaling.",
            "title_wrap_per_inch": "title_wrap_per_inch has been removed; pass title_wrap for explicit control.",
            "title_wrap_auto_scale": "title_wrap_auto_scale has been removed; the automatic width uses the figure size directly.",
            "title_wrap_reference": "title_wrap_reference has been removed; the automatic width uses the figure size directly.",
            "title_wrap_scale_limits": "title_wrap_scale_limits has been removed; the automatic width uses the figure size directly.",
            "auto_size": "auto_size has been removed; omit figsize to use the streamlined auto-sizing heuristics.",
            "size_per_category": "size_per_category has been removed; omit figsize to use the streamlined auto-sizing heuristics.",
            "min_category_axis": "min_category_axis has been removed; auto-sizing now keeps widths reasonable by default.",
            "max_category_axis": "max_category_axis has been removed; auto-sizing now keeps widths reasonable by default.",
            "category_axis_padding": "category_axis_padding has been removed; a consistent padding is now applied automatically.",
        }
        for key, value in list(legacy_kwargs.items()):
            if key in renamed:
                target = renamed[key]
                if target == "tick_label_size":
                    tick_label_size = value  # type: ignore[assignment]
                legacy_kwargs.pop(key)
                continue
            if key in removed:
                raise TypeError(removed[key])
        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    try:
        tick_label_size = int(tick_label_size)
    except (TypeError, ValueError):
        tick_label_size = 11

    using_dataframe = data is not None or category_column is not None or value_column is not None
    min_category_fraction = 0.0 if min_category_fraction is None else float(min_category_fraction)
    if min_category_fraction < 0.0:
        raise ValueError("min_category_fraction must be greater than or equal to 0.")
    resolved_series_labels: Optional[List[str]] = list(series_labels) if series_labels is not None else None

    if isinstance(error_bars, bool):
        if error_bars:
            if not using_dataframe or value_column is None:
                raise ValueError(
                    "error_bars=True requires supplying `data` with a value_column so confidence intervals can be computed."
                )
            error_bars = "ci95"
        else:
            error_bars = None

    if using_dataframe:
        if data is None or category_column is None:
            raise ValueError("When supplying raw data you must also provide data and category_column.")
        working_df = data.copy()
        working_df = working_df.dropna(subset=[category_column])
        if working_df.empty:
            raise ValueError("No rows remain after dropping missing category values.")
        aggregated_map: "OrderedDict[str, List[float]]"
        error_array: Optional[np.ndarray] = None
        counting_categories = value_column is None
        if counting_categories:
            if error_bars not in (None, False):
                raise ValueError("error_bars are not supported when plotting category counts.")
            category_series = working_df[category_column].astype(str)
            overall_total = float(len(category_series))
            counts = category_series.value_counts()
            if min_category_fraction > 0.0 and overall_total > 0.0:
                counts = counts[(counts / overall_total) >= min_category_fraction]
            if counts.empty:
                raise ValueError("No categories remain after applying min_category_fraction.")
            counts = counts.sort_values(ascending=False)
            if category_order is None and categories is None:
                if category_cap is not None and int(category_cap) > 0:
                    counts = counts.iloc[: int(category_cap)]
            aggregated_map = OrderedDict((str(idx), [float(count)]) for idx, count in counts.items())
            if as_percent and overall_total > 0.0:
                for key, values in aggregated_map.items():
                    values[0] = values[0] / overall_total * 100.0
            default_series_labels: List[str] = []
        else:
            value_columns = _ensure_list(value_column)
            if not value_columns:
                raise ValueError("value_column must be provided when using data.")
            replacements: Dict[str, str] = {}
            if excess_year_col is not None:
                columns = _ensure_list(excess_columns)
                if not columns:
                    columns = value_columns
                if excess_window is None or int(excess_window) <= 0:
                    raise ValueError(
                        "excess_window must be a positive integer when excess_year_col is provided."
                    )
                working_df, replacements = _apply_year_excess(
                    working_df,
                    year_col=excess_year_col,
                    window=int(excess_window),
                    columns=columns,
                    mode=(excess_mode or "difference").lower(),
                    replace=bool(excess_replace),
                    prefix=excess_prefix,
                )
            resolved_columns = [replacements.get(col, col) for col in value_columns]
            keep_columns = list(dict.fromkeys([category_column, *resolved_columns]))
            working_df = working_df[keep_columns].copy()
            for col in resolved_columns:
                working_df[col] = pd.to_numeric(working_df[col], errors="coerce")
            working_df = working_df.dropna(subset=[category_column] + resolved_columns, how="any")
            grouped = working_df.groupby(category_column, observed=True)[resolved_columns]
            group_sizes = grouped.size()
            total_group_size = float(group_sizes.sum())
            try:
                aggregated = grouped.aggregate(value_agg)
            except TypeError as exc:
                raise TypeError("Failed to aggregate value_column with the provided value_agg.") from exc
            if isinstance(aggregated, pd.Series):
                aggregated = aggregated.to_frame(name=resolved_columns[0])
            aggregated = aggregated.dropna(how="all")
            aggregated_map = OrderedDict()
            for key, row in aggregated.iterrows():
                aggregated_map[str(key)] = [float(row[col]) for col in aggregated.columns]
            if min_category_fraction > 0.0 and total_group_size > 0.0:
                frequency_map = {
                    str(idx): float(count) / total_group_size
                    for idx, count in group_sizes.items()
                    if str(idx) in aggregated_map
                }
                filtered_map: "OrderedDict[str, List[float]]" = OrderedDict(
                    (cat, values)
                    for cat, values in aggregated_map.items()
                    if frequency_map.get(cat, 0.0) >= min_category_fraction
                )
                aggregated_map = filtered_map
            if not aggregated_map:
                raise ValueError("Aggregation produced no bars to plot.")
            default_series_labels = list(aggregated.columns)
        if category_order is not None and categories is not None:
            raise ValueError("Provide at most one of categories or category_order when aggregating from data.")
        if category_order is not None:
            desired_order = [str(cat) for cat in category_order]
        elif categories is not None:
            desired_order = [str(cat) for cat in categories]
        else:
            desired_order = list(aggregated_map.keys())
        category_keys: List[str] = []
        bar_matrix: List[List[float]] = []
        for cat in desired_order:
            if cat not in aggregated_map:
                if min_category_fraction > 0.0:
                    continue
                raise KeyError(f"Category '{cat}' not present in aggregated data.")
            category_keys.append(cat)
            bar_matrix.append(aggregated_map[cat])
        if not category_keys:
            raise ValueError(
                "No categories remain after applying min_category_fraction and desired ordering."
            )

        if not counting_categories:
            def _compute_error_from_string(kind: str) -> List[float]:
                mode = (kind or "").strip().lower()
                reference_column = aggregated.columns[0]
                grouped_series = working_df.groupby(category_column, observed=True)[reference_column]
                if mode == "std":
                    series = grouped_series.std(ddof=1)
                elif mode == "sem":
                    series = grouped_series.apply(lambda s: sem(s, nan_policy="omit"))
                elif mode.startswith("ci"):
                    digits = mode[2:] or "95"
                    try:
                        level = float(digits) / 100.0
                    except ValueError as exc:
                        raise ValueError("ci error bars must be followed by a percentage, e.g. 'ci95'.") from exc
                    level = max(0.0, min(level, 0.999))
                    z_score = norm.ppf(0.5 + level / 2.0)
                    sem_series = grouped_series.apply(lambda s: sem(s, nan_policy="omit"))
                    series = sem_series * z_score
                else:
                    raise ValueError(
                        "String error_bars must be one of 'std', 'sem', 'ci90', 'ci95', or 'ci99'."
                    )
                result_map = {
                    str(idx): float(val) if pd.notna(val) else float("nan") for idx, val in series.items()
                }
                return [abs(result_map.get(cat, float("nan"))) for cat in category_keys]

            if error_bars is None:
                error_array = None
            else:
                if len(aggregated.columns) > 1:
                    raise ValueError("error_bars are currently only supported for single-series bar plots.")
                if isinstance(error_bars, str):
                    error_array = np.asarray(_compute_error_from_string(error_bars), dtype=float)
                elif isinstance(error_bars, dict):
                    lower = list(error_bars.get("lower", []))
                    upper = list(error_bars.get("upper", []))
                    if len(lower) != len(category_keys) or len(upper) != len(category_keys):
                        raise ValueError(
                            "Asymmetric error bars must provide 'lower' and 'upper' lists matching the bar count."
                        )
                    error_array = np.vstack([
                        np.abs(np.asarray(lower, dtype=float)),
                        np.abs(np.asarray(upper, dtype=float)),
                    ])
                else:
                    array = np.asarray(list(error_bars), dtype=float)
                    if array.shape[0] != len(category_keys):
                        raise ValueError("error_bars iterable length must match the number of categories.")
                    error_array = np.abs(array)
            if resolved_series_labels is None:
                resolved_series_labels = default_series_labels
        else:
            if resolved_series_labels is None:
                resolved_series_labels = default_series_labels
    else:
        if categories is None or values is None:
            raise ValueError("categories and values must be provided when data is not supplied.")
        category_keys = [str(cat) for cat in categories]
        raw_values = list(values)
        if len(category_keys) != len(raw_values):
            raise ValueError("categories and values must be the same length.")
        if not raw_values:
            raise ValueError("No bars to plot.")
        bar_matrix = []
        for val in raw_values:
            if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
                bar_matrix.append([float(v) for v in val])
            else:
                bar_matrix.append([float(val)])
        n_series = len(bar_matrix[0]) if bar_matrix else 0
        for row in bar_matrix:
            if len(row) != n_series:
                raise ValueError("Each category must provide the same number of series values.")
        if category_order is not None:
            desired = [str(cat) for cat in category_order]
            missing = [cat for cat in desired if cat not in category_keys]
            if missing:
                raise KeyError(f"Categories {missing} not present in provided data.")
            index_map = {cat: idx for idx, cat in enumerate(category_keys)}
            ordered_indices = [index_map[cat] for cat in desired]
            category_keys = [category_keys[idx] for idx in ordered_indices]
            bar_matrix = [bar_matrix[idx] for idx in ordered_indices]
        if isinstance(error_bars, dict):
            lower = list(error_bars.get("lower", []))
            upper = list(error_bars.get("upper", []))
            if len(lower) != len(bar_matrix) or len(upper) != len(bar_matrix):
                raise ValueError("Asymmetric error bars must match the number of bars.")
            error_array = np.vstack([
                np.abs(np.asarray(lower, dtype=float)),
                np.abs(np.asarray(upper, dtype=float)),
            ])
        elif error_bars is None:
            error_array = None
        else:
            array = np.asarray(list(error_bars), dtype=float)
            if array.shape[0] != len(bar_matrix):
                raise ValueError("error_bars iterable length must match the number of bars.")
            error_array = np.abs(array)
        if resolved_series_labels is None and n_series > 1:
            resolved_series_labels = [f"Series {idx + 1}" for idx in range(n_series)]

    display_categories = ["other" if key.strip().lower() == "none" else key for key in category_keys]

    bar_array = np.asarray(bar_matrix, dtype=float)
    if bar_array.size == 0:
        raise ValueError("No bars to plot.")
    if bar_array.ndim == 1:
        bar_array = bar_array[:, np.newaxis]
    n_categories, n_series = bar_array.shape
    if error_array is not None and n_series > 1:
        raise ValueError("error_bars are only supported for single-series bar plots.")
    if resolved_series_labels is None and n_series == 1:
        resolved_series_labels = []
    elif resolved_series_labels is None:
        resolved_series_labels = [f"Series {idx + 1}" for idx in range(n_series)]
    else:
        resolved_series_labels = list(resolved_series_labels)
        if n_series > 1 and len(resolved_series_labels) != n_series:
            raise ValueError("Number of series_labels must match the number of value series.")
        if n_series == 1 and len(resolved_series_labels) not in (0, 1):
            raise ValueError("Single-series bar plots accept at most one series label.")

    if category_order is None and sort_mode:
        mode = sort_mode.strip().lower() if isinstance(sort_mode, str) else ""
        indices = list(range(n_categories))
        if mode in {"descending", "ascending"}:
            totals = bar_array.sum(axis=1)
            reverse = mode == "descending"
            indices.sort(key=lambda idx: totals[idx], reverse=reverse)
        elif mode == "random":
            random.shuffle(indices)
        elif mode in {"none", ""}:
            indices = list(range(n_categories))
        else:
            raise ValueError("sort_mode must be 'descending', 'ascending', 'none', or 'random'.")
        bar_array = bar_array[indices]
        display_categories = [display_categories[idx] for idx in indices]
        if error_array is not None:
            if error_array.ndim == 1:
                error_array = np.asarray(error_array)[indices]
            else:
                error_array = np.asarray(error_array)[:, indices]

    def fmt(val: float) -> str:
        if as_percent:
            return f"{val:.{precision}g}%"
        if abs(val) >= 1e6:
            return f"{val / 1e6:.{precision}g}M"
        if abs(val) >= 1e3:
            return f"{val / 1e3:.{precision}g}K"
        return f"{val:.{precision}g}"

    plt.style.use("default")
    plt.rcParams["font.family"] = font_family

    manual_figsize: Optional[Tuple[float, float]]
    if figsize is None:
        manual_figsize = None
    else:
        manual_figsize = (float(figsize[0]), float(figsize[1]))
    default_figsize = (13.0, 6.5)
    if wrap_width is None:
        configured_wrap_width: Optional[float] = 18
    else:
        try:
            configured_wrap_width = float(wrap_width)
        except (TypeError, ValueError) as exc:
            raise TypeError("wrap_width must be a numeric value or None.") from exc
    if max_bars_per_plot is None or int(max_bars_per_plot) <= 0:
        effective_limit = n_categories
    else:
        base_limit = max(int(max_bars_per_plot), 1)
        effective_limit = base_limit * (2 if orientation == "horizontal" else 1)
        effective_limit = max(effective_limit, 1)
    total_chunks = (n_categories + effective_limit - 1) // effective_limit if effective_limit else 1

    def _label_wrap_width(raw_labels: Sequence[str], chunk_count: int) -> Optional[int]:
        if label_wrap_mode == "none":
            return None
        if configured_wrap_width is None:
            return None
        try:
            base_width = int(round(configured_wrap_width))
        except (TypeError, ValueError) as exc:
            raise TypeError("wrap_width must be a numeric value or None.") from exc
        if base_width <= 0:
            return None
        base_width = max(base_width, min_wrap_chars)
        if label_wrap_mode == "fixed":
            return base_width
        if not raw_labels:
            return base_width
        longest = max(len(label) for label in raw_labels)
        overflow = max(longest - base_width, 0)
        relief_cap = int(round(max(base_width * 0.5, min_wrap_chars)))
        relief = 0
        if overflow > 0:
            relief = int(round(min(overflow * 0.18, relief_cap)))
        penalty_scale = 0.4 if orientation == "vertical" else 0.2
        penalty = int(round(max(chunk_count - 8, 0) * penalty_scale))
        effective_width = base_width + relief - penalty
        return max(effective_width, min_wrap_chars)

    def _label_density_scale(
        raw_labels: Sequence[str], wrapped_labels: Sequence[str], chunk_count: int
    ) -> float:
        """Return a multiplier for figure width based on label complexity."""

        if not raw_labels:
            return 1.0

        reference = configured_wrap_width
        if reference is None or reference <= 0:
            reference = 18
        reference = max(reference, min_wrap_chars)

        longest_raw = max(len(label) for label in raw_labels)
        overflow = max(longest_raw - reference, 0)
        overflow_ratio = overflow / max(reference, 1)

        has_wrapping = bool(
            wrapped_labels and any("\n" in label for label in wrapped_labels if label)
        )
        if has_wrapping:
            overflow_ratio *= 0.03

        if wrapped_labels:
            line_counts = [label.count("\n") + 1 for label in wrapped_labels]
            max_lines = max(line_counts)
            max_line_len = max(
                max(len(segment) for segment in label.split("\n")) if label else 0
                for label in wrapped_labels
            )
        else:
            max_lines = 1
            max_line_len = 0

        line_overflow = max(max_line_len - reference, 0)
        line_ratio = line_overflow / max(reference, 1)

        line_weight = 0.18 if has_wrapping else 0.25
        multiline_weight = 0.07 if has_wrapping else 0.1

        scale = 1.0 + 0.35 * overflow_ratio + line_weight * line_ratio + multiline_weight * max(max_lines - 1, 0)
        clamped_count = max(1, min(int(chunk_count), 12))
        if has_wrapping:
            dynamic_cap = min(1.6, 1.1 + clamped_count * 0.04)
        else:
            dynamic_cap = min(2.4, 1.25 + clamped_count * 0.07)
        return max(1.0, min(scale, dynamic_cap))

    def _auto_figsize(chunk_count: int) -> Tuple[float, float]:
        width, height = default_figsize
        count = max(chunk_count, 1)
        if orientation == "vertical":
            width = min(24.0, max(width, 0.85 * count + 6.0))
        else:
            height = min(18.0, max(height, 0.6 * count + 4.0))
            width = max(width, 11.0)
        return width, height

    chunk_sizes: List[int] = []
    if total_chunks > 0:
        base_chunk = n_categories // total_chunks
        remainder = n_categories % total_chunks
        for idx in range(total_chunks):
            size = base_chunk + (1 if idx < remainder else 0)
            if size <= 0:
                continue
            chunk_sizes.append(size)

    if not chunk_sizes:
        chunk_sizes = [n_categories]

    total_chunks = len(chunk_sizes)

    safe_title = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_") or "bar_plot"

    figures: List[Tuple[plt.Figure, plt.Axes]] = []

    start = 0
    for chunk_idx, chunk_size in enumerate(chunk_sizes):
        end = start + chunk_size
        chunk_values = bar_array[start:end]
        raw_labels = display_categories[start:end]
        if error_array is None:
            chunk_error = None
        else:
            if error_array.ndim == 1:
                chunk_error = error_array[start:end]
            else:
                chunk_error = error_array[:, start:end]

        chunk_count = chunk_values.shape[0]
        resolved_wrap_width = _label_wrap_width(raw_labels, chunk_count)
        if resolved_wrap_width is None:
            chunk_labels = raw_labels
        else:
            chunk_labels = [
                textwrap.fill(label, width=resolved_wrap_width) if resolved_wrap_width > 0 else label
                for label in raw_labels
            ]
        if manual_figsize is None:
            fig_width, fig_height = _auto_figsize(chunk_count)
            if orientation == "vertical":
                density_scale = _label_density_scale(raw_labels, chunk_labels, chunk_count)
                fig_width = min(30.0, fig_width * density_scale)
        else:
            fig_width, fig_height = manual_figsize

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        ax.set_facecolor(background_color)
        fig.patch.set_facecolor(background_color)

        indices = np.arange(chunk_count, dtype=float)
        bar_containers: List[mpl.container.BarContainer] = []
        if n_series == 1:
            colours = plt.get_cmap(cmap)(np.linspace(gradient_start, gradient_end, chunk_count))
            values_slice = chunk_values[:, 0]
            if orientation == "vertical":
                container = ax.bar(
                    indices,
                    values_slice,
                    width=vertical_bar_width,
                    color=colours,
                    edgecolor="black",
                    yerr=chunk_error if chunk_error is not None else None,
                    capsize=error_bar_capsize if chunk_error is not None else None,
                )
                bar_containers.append(container)
            else:
                container = ax.barh(
                    indices,
                    values_slice,
                    height=horizontal_bar_height,
                    color=colours,
                    edgecolor="black",
                    xerr=chunk_error if chunk_error is not None else None,
                    capsize=error_bar_capsize if chunk_error is not None else None,
                )
                bar_containers.append(container)
        else:
            cmap_obj = plt.get_cmap(cmap)
            series_colours = cmap_obj(np.linspace(gradient_start, gradient_end, n_series))
            if orientation == "vertical":
                group_width = vertical_bar_width
                bar_width = group_width / n_series
                offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_width
                for series_idx in range(n_series):
                    container = ax.bar(
                        indices + offsets[series_idx],
                        chunk_values[:, series_idx],
                        width=bar_width,
                        color=series_colours[series_idx],
                        edgecolor="black",
                        label=resolved_series_labels[series_idx] if resolved_series_labels else None,
                    )
                    bar_containers.append(container)
            else:
                group_height = horizontal_bar_height
                bar_height = group_height / n_series
                offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_height
                for series_idx in range(n_series):
                    container = ax.barh(
                        indices + offsets[series_idx],
                        chunk_values[:, series_idx],
                        height=bar_height,
                        color=series_colours[series_idx],
                        edgecolor="black",
                        label=resolved_series_labels[series_idx] if resolved_series_labels else None,
                    )
                    bar_containers.append(container)

        positive_errors: Optional[np.ndarray]
        negative_errors: Optional[np.ndarray]
        if chunk_error is not None and n_series == 1:
            chunk_err_arr = np.asarray(chunk_error, dtype=float)
            if chunk_err_arr.ndim == 1:
                positive_errors = np.nan_to_num(chunk_err_arr.astype(float), nan=0.0)
                negative_errors = positive_errors
            elif chunk_err_arr.ndim == 2 and chunk_err_arr.shape[0] == 2:
                negative_errors = np.nan_to_num(chunk_err_arr[0].astype(float), nan=0.0)
                positive_errors = np.nan_to_num(chunk_err_arr[1].astype(float), nan=0.0)
            else:
                flat = np.nan_to_num(np.atleast_1d(chunk_err_arr.squeeze()).astype(float), nan=0.0)
                positive_errors = flat
                negative_errors = flat
        else:
            positive_errors = None
            negative_errors = None

        point_offset = 6 if chunk_error is not None and n_series == 1 else 3
        annotation_size = annotation_font_size + (1 if chunk_error is not None and n_series == 1 else 0)

        for series_idx, container in enumerate(bar_containers):
            series_col = series_idx if n_series > 1 else 0
            for bar_idx, (bar, value) in enumerate(zip(container, chunk_values[:, series_col])):
                if orientation == "vertical":
                    height = bar.get_height()
                    err_up = float(positive_errors[bar_idx]) if positive_errors is not None else 0.0
                    err_down = float(negative_errors[bar_idx]) if negative_errors is not None else 0.0
                    base_height = height + err_up if height >= 0 else height - err_down
                    offset = point_offset if height >= 0 else -point_offset
                    ax.annotate(
                        fmt(value),
                        xy=(bar.get_x() + bar.get_width() / 2, base_height),
                        xytext=(0, offset),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if height >= 0 else "top",
                        fontsize=annotation_size,
                        fontweight=annotation_fontweight,
                    )
                else:
                    width_val = bar.get_width()
                    err_up = float(positive_errors[bar_idx]) if positive_errors is not None else 0.0
                    err_down = float(negative_errors[bar_idx]) if negative_errors is not None else 0.0
                    base_width = width_val + err_up if width_val >= 0 else width_val - err_down
                    offset = point_offset if width_val >= 0 else -point_offset
                    ax.annotate(
                        fmt(value),
                        xy=(base_width, bar.get_y() + bar.get_height() / 2),
                        xytext=(offset, 0),
                        textcoords="offset points",
                        ha="left" if width_val >= 0 else "right",
                        va="center",
                        fontsize=annotation_size,
                        fontweight=annotation_fontweight,
                    )

        axis_padding = 0.08

        if orientation == "vertical":
            ax.set_xticks(indices)
            ax.set_xticklabels(chunk_labels, rotation=45 if rotate_xlabels else 0, ha="right" if rotate_xlabels else "center")
            ax.set_xlabel(x_label, fontsize=label_font_size, fontweight="bold")
            ax.set_ylabel(y_label, fontsize=label_font_size, fontweight="bold")
            ax.tick_params(axis="x", labelsize=tick_label_size)
            if chunk_count > 0 and axis_padding:
                ax.margins(x=axis_padding * 0.5)
            for tick_label in ax.get_xticklabels():
                tick_label.set_multialignment("center")
            if value_axis_limits is not None:
                lower, upper = value_axis_limits
                current_lower, current_upper = ax.get_ylim()
                ax.set_ylim(
                    current_lower if lower is None else lower,
                    current_upper if upper is None else upper,
                )
        else:
            ax.set_yticks(indices)
            ax.set_yticklabels(chunk_labels)
            ax.set_ylabel(x_label, fontsize=label_font_size, fontweight="bold")
            ax.set_xlabel(y_label, fontsize=label_font_size, fontweight="bold")
            ax.tick_params(axis="y", labelsize=tick_label_size)
            if value_axis_limits is not None:
                lower, upper = value_axis_limits
                current_lower, current_upper = ax.get_xlim()
                ax.set_xlim(
                    current_lower if lower is None else lower,
                    current_upper if upper is None else upper,
                )
            if chunk_count > 0:
                group_span = horizontal_bar_height
                pad = group_span / 2.0
                extra = group_span * (axis_padding + 0.08)
                lower_bound = indices[0] - pad - extra
                upper_bound = indices[-1] + pad + extra
                ax.set_ylim(lower_bound, upper_bound)
                ax.invert_yaxis()
            else:
                ax.margins(y=axis_padding)
            if value_axis_limits is None:
                if chunk_count > 0:
                    ax.margins(x=0.04)
                else:
                    ax.margins(x=0.04, y=axis_padding)

        if resolved_series_labels and n_series > 1:
            handles = []
            labels = []
            for container, label in zip(bar_containers, resolved_series_labels):
                if label is None:
                    continue
                handles.append(container.patches[0] if container.patches else container)
                labels.append(label)
            if handles:
                ax.legend(handles, labels, frameon=False)

        if title_wrap is None:
            computed_wrap = max(int(round(fig.get_figwidth() * 5.5)), 1)
            title_width = computed_wrap
        else:
            title_width = max(int(title_wrap), 1)
        title_text = textwrap.fill(title, width=title_width) if title_width > 0 else title
        ax.set_title(title_text, fontsize=title_font_size, fontweight="bold")

        fig.tight_layout()
        if orientation == "horizontal":
            right_margin = 0.98
            if horizontal_label_fraction >= right_margin:
                right_margin = min(horizontal_label_fraction + 0.01, 0.99)
            fig.subplots_adjust(left=horizontal_label_fraction, right=right_margin)
        figures.append((fig, ax))

        _save_figure(
            fig,
            save_path=save_path,
            default_name=safe_title or "bar_plot",
            save_format=save_format,
            batch_idx=chunk_idx,
            total_batches=total_chunks,
            bbox_inches=bbox_inches,
            print_save_path=print_save_path,
        )
        start = end

    if figures:
        plt.show()


def box_plot(
    data: Union[pd.DataFrame, Dict[str, Iterable[float]], Iterable[Iterable[float]]],
    *,
    labels: Optional[Iterable[str]] = None,
    title: str = "Distribution by Group",
    x_label: str = "Group",
    y_label: str = "Value",
    cmap: str = "viridis",
    gradient_start: float = 0.25,
    gradient_end: float = 0.9,
    background_color: str = "#ffffff",
    font_family: str = "monospace",
    figsize: Tuple[float, float] = (12, 6),
    dpi: int = 300,
    notch: bool = False,
    showfliers: bool = False,
    patch_alpha: float = 0.9,
    line_color: str = "#2f2f2f",
    box_linewidth: float = 1.6,
    median_linewidth: float = 2.4,
    annotate_median: bool = True,
    annotation_font_size: int = 10,
    annotation_fontweight: str = "bold",
    wrap_width: int = 22,
    summary_precision: int = 2,
    print_summary: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    save_format: Optional[str] = None,
    bbox_inches: Optional[str] = "tight",
    print_save_path: bool = True,
) -> Dict[str, Any]:
    """Render a high-DPI box plot that matches the house style.

    ``data`` may be a tidy DataFrame (columns interpreted as groups), a mapping
    from labels to iterables, or a sequence of iterables.  When ``labels`` is
    omitted, column names or dictionary keys are used automatically.  Numeric
    data are coerced with ``pd.to_numeric`` and missing values are dropped.

    Returns a dictionary containing the Matplotlib ``figure`` and ``ax`` along
    with a ``summary`` DataFrame of descriptive statistics (count, median,
    quartiles and whiskers).  This mirrors the ergonomics of :func:`regression_plot`
    and friends by providing both a styled visual and machine-friendly output.
    Provide ``save_path`` to persist the figure (optionally overriding the
    format via ``save_format``).  When ``save_path`` is a directory, the filename
    defaults to a sanitized version of the title.

    Examples
    --------
    >>> out = box_plot(df[["group_a", "group_b"]], title="Score dispersion")  # doctest: +SKIP
    >>> out["summary"].loc["group_a", "median"]  # doctest: +SKIP
    0.42
    """

    if isinstance(data, pd.DataFrame):
        available = list(data.columns)
        if labels is None:
            labels_list = available
        else:
            labels_list = list(labels)
            missing = [label for label in labels_list if label not in data.columns]
            if missing:
                raise KeyError(f"Columns {missing} not found in provided DataFrame.")
        value_arrays = [pd.to_numeric(data[label], errors="coerce").dropna().to_numpy() for label in labels_list]
    elif isinstance(data, dict):
        if labels is None:
            labels_list = list(data.keys())
        else:
            labels_list = list(labels)
        missing = [label for label in labels_list if label not in data]
        if missing:
            raise KeyError(f"Keys {missing} not found in data mapping.")
        value_arrays = [pd.to_numeric(pd.Series(data[label]), errors="coerce").dropna().to_numpy() for label in labels_list]
    else:
        if labels is None:
            try:
                length = len(data)  # type: ignore[arg-type]
            except TypeError:
                raise TypeError("When supplying a sequence of iterables, please provide labels or ensure it has a length.")
            labels_list = [f"Series {i + 1}" for i in range(length)]
        else:
            labels_list = list(labels)
        value_arrays = []
        for idx, series in enumerate(data):
            arr = pd.to_numeric(pd.Series(series), errors="coerce").dropna().to_numpy()
            value_arrays.append(arr)
        if len(value_arrays) != len(labels_list):
            raise ValueError("Number of provided labels does not match the number of series in `data`.")

    if not value_arrays:
        raise ValueError("No data provided for box_plot.")

    plt.style.use("default")
    plt.rcParams["font.family"] = font_family
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)

    bp = ax.boxplot(
        value_arrays,
        patch_artist=True,
        notch=notch,
        showfliers=showfliers,
    )
    ax.set_xticks(
        range(1, len(labels_list) + 1),
        labels=[textwrap.fill(str(label), wrap_width) for label in labels_list],
    )

    cmap_obj = plt.get_cmap(cmap)
    colours = cmap_obj(np.linspace(gradient_start, gradient_end, len(value_arrays)))
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_edgecolor(line_color)
        patch.set_alpha(patch_alpha)
        patch.set_linewidth(box_linewidth)
    for element in ("whiskers", "caps"):
        for artist in bp[element]:
            artist.set(color=line_color, linewidth=box_linewidth, alpha=0.9)
    for median in bp["medians"]:
        median.set(color=line_color, linewidth=median_linewidth)

    ax.set_title(textwrap.fill(title, width=80), fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    medians = [np.nanmedian(values) if len(values) else np.nan for values in value_arrays]
    if annotate_median:
        for idx, median in enumerate(medians):
            if np.isnan(median):
                continue
            ax.annotate(
                f"{median:.{summary_precision}f}",
                xy=(idx + 1, median),
                xytext=(0, -12),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=annotation_font_size,
                fontweight=annotation_fontweight,
                color=line_color,
            )

    summary_rows = []
    whisker_pairs = [(bp["whiskers"][i], bp["whiskers"][i + 1]) for i in range(0, len(bp["whiskers"]), 2)]
    for label, values, whiskers in zip(labels_list, value_arrays, whisker_pairs):
        if len(values) == 0:
            summary = {"count": 0, "median": np.nan, "mean": np.nan, "std": np.nan, "q1": np.nan, "q3": np.nan, "whisker_low": np.nan, "whisker_high": np.nan}
        else:
            summary = {
                "count": len(values),
                "median": float(np.nanmedian(values)),
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                "q1": float(np.nanpercentile(values, 25)),
                "q3": float(np.nanpercentile(values, 75)),
                "whisker_low": float(np.min(whiskers[0].get_ydata())),
                "whisker_high": float(np.max(whiskers[1].get_ydata())),
            }
        summary_rows.append(summary)
    summary_df = pd.DataFrame(summary_rows, index=labels_list)
    if print_summary:
        display_df = summary_df.round(summary_precision)
        if tabulate is not None:
            print(tabulate(display_df, headers="keys", tablefmt="github", showindex=True))
        else:
            print(display_df.to_string())

    plt.tight_layout()
    safe_title = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_") or "box_plot"
    _save_figure(
        fig,
        save_path=save_path,
        default_name=safe_title,
        save_format=save_format,
        bbox_inches=bbox_inches,
        print_save_path=print_save_path,
    )
    plt.show()

    return {"figure": fig, "ax": ax, "summary": summary_df}

def line_plot(
    df,
    x,                                  # x-axis column (year, date, etc.)
    y=None,                             # numeric column (long); if None with `by`, counts per (x, by)
    by=None,                            # LONG format: category column (mutually exclusive with `series`)
    series=None,                        # WIDE format: list/str of numeric columns to plot
    include=None, exclude=None,         # LONG: filter groups by values in `by`
    top_k=None,                         # keep top-k series by overall plotted weight
    mode='value',                       # 'value' or 'proportion'
    agg='mean',                         # aggregator for duplicates at (x, by): 'mean','median','std','var','cv','se','sum','count'
    smoothing_window=None,              # int (rolling mean window, centered)
    smoothing_method='rolling',         # 'rolling' or 'spline'
    spline_k=3,                         # spline degree (if using 'spline')
    interpolation_points=None,          # optional: upsample to N points across x-range
    # --- presentation ---
    title=None,
    xlabel=None, ylabel=None,
    x_range=None, y_range=None,         # soft clamps for view; aliases below
    xlim=None, ylim=None,
    dpi=400,
    font_family='monospace',
    wrap_width=96,
    grid=True,
    linewidth=2.0,
    cmap_names=None,                    # list of colormap names for distinct series (when no color_map)
    gradient_mode='value',              # 'value' or 'linear' (used only if gradient=True)
    gradient_start=0.35, gradient_end=0.75,
    gradient=True,                      # if False, draw solid lines (respecting color_map/colors)
    color_map=None,                     # dict: {series_name: color_hex}; overrides colormaps
    legend_order=None,                  # list to control legend order
    legend_loc='best',
    alpha=1.0,
    max_lines_per_plot=8,               # batch panels if many series
    save_path=None,                     # file or dir; batches get suffix _1, _2, ...
    save_format=None,                   # override extension (e.g., "png", "pdf")
    bbox_inches="tight",                # pass through to fig.savefig
    print_save_path=True,
    show=True,
):
    """
    Multi-line plotter for *long* or *wide* data with optional proportions, aggregation, smoothing, and batching.

    Quick recipes
    -------------
    LONG (group column):
        line_plot(df, x='year', y='score', by='party', agg='mean')
        # share within each year:
        line_plot(df, x='year', by='party', mode='proportion')  # y=None => counts

    WIDE (several numeric columns already):
        line_plot(df, x='year', y=['dem_score','gop_score'])  # quick shorthand
        line_plot(df, x='year', series=['dem_score','gop_score'])  # explicit

    Key behaviors
    -------------
    • mode='value'     : plot values (after aggregating duplicates if long).
    • mode='proportion': within each x, divide each series' value by total across series at that x.
                         (Works for long and wide.)
    • Aggregation     : duplicates at each (x, series) are combined with `agg`
                        (works for both long and wide data).
    • Smoothing         : centered rolling mean (or B-spline if SciPy available).
    • Colors            : deterministic; prefer `color_map={'A':'#...', 'B':'#...'}` to pin exact hues.
                          If not provided, falls back to colormaps in `cmap_names`.
    • Batching          : if many series, panels are split into sets of `max_lines_per_plot`.

    Parameters worth remembering
    ----------------------------
    - `legend_order=['Democrat','Republican']` for stable legend order.
    - `gradient=False` for solid lines; `True` for aesthetic gradient lines.
    - `top_k=...` to focus on the most important series by overall plotted mass.
    - `save_path='charts/lines.pdf'` to save output (format inferred from the extension
      or overridden via `save_format`).
    """

    # ---- optional SciPy spline
    try:
        from scipy.interpolate import make_interp_spline
        _spline_available = True
    except Exception:
        _spline_available = False
        if smoothing_method == 'spline':
            print("SciPy not available; using rolling smoothing instead.")
            smoothing_method = 'rolling'

    # Matplotlib basics
    plt.rcParams.update({'font.family': font_family})
    if cmap_names is None:
        cmap_names = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys"]

    def _is_non_string_sequence(value: Any) -> bool:
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes))

    series_columns: Optional[List[Any]] = None
    if series is not None:
        if by is not None:
            raise ValueError("Specify either `by` for long-form data or `series`/`y` for wide data, not both.")
        series_columns = _ensure_list(series)
    elif by is None:
        if y is None:
            raise ValueError("Provide one or more columns via `y` or `series` when `by` is omitted.")
        if _is_non_string_sequence(y):
            series_columns = list(y)
        else:
            series_columns = [y]
        y = None
    elif y is not None and _is_non_string_sequence(y):
        raise ValueError("When `by` is provided, `y` must reference a single column name.")

    if by is None and series_columns is None:
        raise ValueError("Specify `by` or supply one or more columns via `y`/`series`.")
    if series_columns is not None and not series_columns:
        raise ValueError("No columns were supplied to plot.")

    agg_fns = {
        'mean':   lambda arr: float(np.mean(arr)) if len(arr) else np.nan,
        'median': lambda arr: float(np.median(arr)) if len(arr) else np.nan,
        'std':    lambda arr: float(np.std(arr)) if len(arr) else np.nan,
        'var':    lambda arr: float(np.var(arr)) if len(arr) else np.nan,
        'cv':     lambda arr: float(np.std(arr) / np.mean(arr)) if len(arr) and np.mean(arr) != 0 else np.nan,
        'se':     lambda arr: float(np.std(arr) / np.sqrt(max(len(arr), 1))) if len(arr) else np.nan,
        'sum':    lambda arr: float(np.sum(arr)) if len(arr) else 0.0,
        'count':  lambda arr: len(arr),
    }
    if callable(agg):
        agg_callable = lambda arr: agg(np.asarray(arr))
    else:
        if agg not in agg_fns:
            raise ValueError(f"Unsupported agg '{agg}'.")
        agg_callable = agg_fns[agg]

    def _apply_agg(values: Union[pd.Series, np.ndarray]) -> float:
        if hasattr(values, "to_numpy"):
            arr = values.to_numpy()
        else:
            arr = np.asarray(values)
        return agg_callable(arr)

    work = df.copy()

    # ---- standardize x to something plottable
    if pd.api.types.is_datetime64_any_dtype(work[x]):
        pass
    else:
        # try to coerce numeric; if that fails, leave as-is
        try:
            work[x] = pd.to_numeric(work[x], errors='ignore')
        except Exception:
            pass

    # ---- represent everything as long: (x, _series, _value)
    if series_columns is not None:
        missing = [c for c in series_columns if c not in work.columns]
        if missing:
            raise KeyError(f"Missing columns for wide plot: {missing}")
        subset = work[[x] + series_columns].copy()
        for col in series_columns:
            subset[col] = pd.to_numeric(subset[col], errors='coerce')
        grouped = subset.groupby(x, dropna=False)[series_columns]
        aggregated = grouped.agg(lambda s: _apply_agg(s)).reset_index()
        long_all = aggregated.melt(id_vars=[x], var_name="_series", value_name="_value")

    else:
        if by not in work.columns:
            raise KeyError(f"`by` column '{by}' not found.")

        if include is not None:
            work = work[work[by].isin(include)]
        if exclude is not None:
            work = work[~work[by].isin(exclude)]

        if y is None:
            long_all = (work.groupby([x, by], dropna=False)
                            .size().rename("_value").reset_index())
        else:
            if y not in work.columns:
                raise KeyError(f"`y` column '{y}' not found.")
            work[y] = pd.to_numeric(work[y], errors='coerce')
            tmp = work.rename(columns={y: '_value'})
            grouped = tmp.groupby([x, by], dropna=False)["_value"]
            long_all = grouped.apply(lambda s: _apply_agg(s)).reset_index()
        long_all = long_all.rename(columns={by: "_series"})

    # ---- compute plotted value
    if mode not in ('value', 'proportion'):
        raise ValueError("mode must be 'value' or 'proportion'.")

    if mode == 'proportion':
        denom = long_all.groupby(x)["_value"].transform(lambda s: s.replace(0, np.nan).sum())
        long_all["_plotval"] = long_all["_value"] / denom
    else:
        long_all["_plotval"] = long_all["_value"]

    # ---- select top_k series (by total plotted value)
    if top_k is not None:
        keep = (long_all.groupby("_series")["_plotval"]
                        .sum(numeric_only=True)
                        .sort_values(ascending=False)
                        .head(int(top_k)).index)
        long_all = long_all[long_all["_series"].isin(set(keep))].copy()

    # ---- sort for plotting
    long_all = long_all.sort_values([x, "_series"])

    # ---- order series
    series_order = legend_order if legend_order else (
        long_all.groupby("_series")["_plotval"].mean(numeric_only=True).sort_values(ascending=False).index.tolist()
    )

    # ---- batching
    if max_lines_per_plot is None or max_lines_per_plot <= 0:
        batches = [series_order]
    else:
        step = int(max_lines_per_plot)
        batches = [series_order[i:i+step] for i in range(0, len(series_order), step)]

    # ---- color resolver
    def _series_color(s, idx):
        if color_map and s in color_map:
            return color_map[s]
        # fall back to palette families by index
        cmap = plt.get_cmap(cmap_names[idx % len(cmap_names)])
        return cmap(0.6)

    figs_axes = []

    def _plot_one(batch_series, batch_idx):
        fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=dpi)
        global_min, global_max = float('inf'), float('-inf')

        for idx, s in enumerate(batch_series):
            sdf = long_all[long_all["_series"] == s].sort_values(x)
            xs = sdf[x].to_numpy()
            ys = sdf["_plotval"].to_numpy()

            # smoothing
            x_s, y_s = xs, ys
            if smoothing_window and smoothing_window > 1 and len(xs) > 1:
                if smoothing_method == 'rolling':
                    y_s = (pd.Series(ys)
                           .rolling(window=int(smoothing_window), min_periods=1, center=True)
                           .mean().to_numpy())
                elif smoothing_method == 'spline' and _spline_available and len(xs) > 2:
                    order = np.argsort(xs)
                    xs_o, ys_o = xs[order], ys[order]
                    k = max(1, min(int(spline_k), len(xs_o) - 1))
                    x_s = np.linspace(xs_o.min(), xs_o.max(),
                                      max(len(xs_o), interpolation_points or len(xs_o)))
                    y_s = make_interp_spline(xs_o, ys_o, k=k)(x_s)

            if interpolation_points and (len(x_s) < interpolation_points):
                xi = np.linspace(np.min(x_s), np.max(x_s), int(interpolation_points))
                yi = np.interp(xi, x_s, y_s)
                x_s, y_s = xi, yi

            if len(y_s) and np.isfinite(y_s).any():
                global_min = min(global_min, np.nanmin(y_s))
                global_max = max(global_max, np.nanmax(y_s))

            color = _series_color(s, idx)

            if gradient:
                # gradient along the line
                pts = np.array([x_s, y_s]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                if isinstance(color, str):
                    # solid color requested but gradient=True → use a subtle light→full alpha ramp
                    from matplotlib.colors import to_rgba
                    base = np.array(to_rgba(color))
                    alphas = np.linspace(gradient_start, gradient_end, max(len(segs), 2))
                    cols = np.tile(base, (len(segs), 1))
                    cols[:, -1] = alphas
                else:
                    # use colormap-driven gradient
                    cmap = plt.get_cmap(cmap_names[idx % len(cmap_names)])
                    if gradient_mode == 'value' and len(y_s) > 1:
                        ymin, ymax = np.nanmin(y_s), np.nanmax(y_s)
                        denom = max((ymax - ymin), 1e-12)
                        norm = (y_s - ymin) / denom
                        seg_vals = (norm[:-1] + norm[1:]) / 2
                        seg_vals = gradient_start + seg_vals * (gradient_end - gradient_start)
                        cols = cmap(seg_vals)
                    else:
                        cols = cmap(np.linspace(gradient_start, gradient_end, max(len(segs), 2)))
                lc = LineCollection(segs, colors=cols, linewidth=linewidth, alpha=alpha, label=str(s))
                ax.add_collection(lc)
            else:
                ax.plot(x_s, y_s, linewidth=linewidth, alpha=alpha, label=str(s), color=color)

        # axis limits (data-driven, then user overrides)
        if not np.isfinite(global_min) or not np.isfinite(global_max):
            global_min, global_max = 0.0, 1.0
        if global_max == global_min:
            pad = 1.0 if global_max == 0 else 0.05 * abs(global_max)
            global_min, global_max = global_min - pad, global_max + pad

        xr = xlim if xlim is not None else x_range
        yr = ylim if ylim is not None else y_range
        if xr is None:
            xr = (pd.Series(long_all[x]).min(), pd.Series(long_all[x]).max())
        if yr is None:
            span = (global_max - global_min)
            yr = (global_min - 0.05*span, global_max + 0.05*span)

        ax.set_xlim(xr[0], xr[1])
        ax.set_ylim(yr[0], yr[1])

        if grid:
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        else:
            for sp in ['top','right']:
                ax.spines[sp].set_visible(False)

        # labels & legend
        ttl = title or "line plot"
        if len(batches) > 1:
            ttl = f"{ttl} (set {batch_idx+1}/{len(batches)})"
        ax.set_title(textwrap.fill(ttl, width=wrap_width))
        ax.set_xlabel(xlabel if xlabel is not None else str(x))
        default_ylabel = "share" if mode == 'proportion' else (agg if by is not None else "value")
        ax.set_ylabel(ylabel if ylabel is not None else default_ylabel)

        # legend in requested order
        handles, labels = ax.get_legend_handles_labels()
        if legend_order:
            order_index = {lbl:i for i,lbl in enumerate(labels)}
            order = [order_index[lbl] for lbl in legend_order if lbl in order_index]
            handles = [handles[i] for i in order] + [h for j,h in enumerate(handles) if j not in order]
            labels  = [labels[i] for i in order]  + [l for j,l in enumerate(labels) if j not in order]
        ax.legend(handles, labels, loc=legend_loc, ncol=1, frameon=True)

        plt.tight_layout()

        safe_title = re.sub(r"[^A-Za-z0-9]+", "_", title or "line_plot").strip("_") or "line_plot"
        _save_figure(
            fig,
            save_path=save_path,
            default_name=safe_title,
            save_format=save_format,
            batch_idx=batch_idx,
            total_batches=len(batches),
            bbox_inches=bbox_inches,
            dpi=dpi,
            print_save_path=print_save_path,
        )

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    figs_axes = [ _plot_one(batch, i) for i, batch in enumerate(batches) ]
    return figs_axes
