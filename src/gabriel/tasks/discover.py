from __future__ import annotations

import io
import json
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .codify import Codify, CodifyConfig
from .compare import Compare, CompareConfig
from .bucket import Bucket, BucketConfig
from .classify import Classify, ClassifyConfig


@dataclass
class DiscoverConfig:
    """Configuration for :class:`Discover`."""

    save_dir: str = "discover"
    model: str = "gpt-5-mini"
    n_parallels: int = 650
    n_runs: int = 1
    min_frequency: float = 0.6
    bucket_count: int = 10
    additional_instructions: Optional[str] = None
    differentiate: bool = True
    max_words_per_call: int = 1000
    max_categories_per_call: int = 8
    use_dummy: bool = False
    modality: str = "text"
    n_terms_per_prompt: int = 250
    repeat_bucketing: int = 5
    repeat_voting: int = 25
    next_round_frac: float = 0.25
    top_k_per_round: int = 1
    raw_term_definitions: bool = True
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None
    max_timeout: Optional[float] = None

    def __post_init__(self) -> None:
        if self.additional_instructions is not None:
            cleaned = str(self.additional_instructions).strip()
            self.additional_instructions = cleaned or None


class Discover:
    """High-level feature discovery pipeline.

    Depending on the inputs, the pipeline will either:
    1. Use :class:`Codify` to discover raw feature candidates from a single column, or
    2. Use :class:`Compare` to surface differentiating attributes between two columns.

    The discovered terms are then grouped into buckets via :class:`Bucket` and finally
    applied back onto the dataset using :class:`Classify`.
    """

    def __init__(self, cfg: DiscoverConfig) -> None:
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg

    def _to_serializable(self, value: Any) -> Any:
        if isinstance(value, pd.DataFrame):
            safe = value.copy()
            safe = safe.where(pd.notna(safe), None)
            return safe.to_dict(orient="records")
        if isinstance(value, dict):
            return {str(k): self._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        if isinstance(value, pd.Series):
            safe_series = value.where(pd.notna(value), None)
            return safe_series.tolist()
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                return None
        except Exception:
            pass
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return value.isoformat()
        return value

    def _persist_result_snapshot(self, result: Dict[str, Any]) -> None:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "results": {k: self._to_serializable(v) for k, v in result.items()},
        }
        out_path = os.path.join(self.cfg.save_dir, "discover_results_snapshot.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _value_to_dataframe(self, name: str, value: Any) -> Optional[pd.DataFrame]:
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if isinstance(value, pd.Series):
            return value.to_frame().reset_index(drop=True)
        if isinstance(value, dict):
            rows = [{"key": str(k), "value": v} for k, v in value.items()]
            df = pd.DataFrame(rows)
            if name == "buckets":
                df = df.rename(columns={"key": "bucket", "value": "definition"})
            return df
        if isinstance(value, list):
            if not value:
                return pd.DataFrame()
            if all(isinstance(item, dict) for item in value):
                return pd.DataFrame(value)
            return pd.DataFrame({"value": value})
        return None

    def _export_result_archive(self, result: Dict[str, Any]) -> None:
        tables: Dict[str, pd.DataFrame] = {}
        for key, value in result.items():
            df = self._value_to_dataframe(key, value)
            if df is not None:
                tables[key] = df
        if not tables:
            return
        archive_path = os.path.join(self.cfg.save_dir, "discover_results_export.zip")
        try:
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                name_counts: Dict[str, int] = {}
                for key, df in tables.items():
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", key).strip("._-") or "table"
                    count = name_counts.get(safe, 0)
                    name_counts[safe] = count + 1
                    if count:
                        safe = f"{safe}_{count}"
                    filename = f"{safe}.csv"
                    zf.writestr(filename, csv_buffer.getvalue())
        except Exception:
            pass


    async def run(
        self,
        df: pd.DataFrame,
        *,
        column_name: Optional[str] = None,
        circle_column_name: Optional[str] = None,
        square_column_name: Optional[str] = None,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the discovery pipeline.

        Args:
            df: Input dataframe.
            column_name: Column to analyse when using a single column pipeline.
            circle_column_name: First column when contrasting two columns.
            square_column_name: Second column when contrasting two columns.
            reset_files: Forwarded to underlying tasks to control caching.

        Returns:
            Dictionary with intermediate and final results. Keys include:
            ``candidates`` (raw candidate terms), ``buckets`` (bucket definitions),
            ``classification`` (original dataframe with label columns), ``summary`` (if
            circle/square columns were provided) containing per-label differences (``difference_pct``
            expresses circle minus square in percentage points),
            and optionally
            ``compare`` or ``codify`` depending on which stage was used for candidate
            generation.
        """

        single = column_name is not None
        pair = circle_column_name is not None and square_column_name is not None
        if single == pair:
            raise ValueError(
                "Provide either column_name or both circle_column_name and square_column_name"
            )

        if single:
            self.cfg.differentiate = False
        elif pair:
            self.cfg.differentiate = True

        compare_df: Optional[pd.DataFrame] = None
        codify_df: Optional[pd.DataFrame] = None

        # ── 1. candidate discovery ─────────────────────────────────────
        if single:
            coder_cfg = CodifyConfig(
                save_dir=os.path.join(self.cfg.save_dir, "codify"),
                model=self.cfg.model,
                n_parallels=self.cfg.n_parallels,
                max_words_per_call=self.cfg.max_words_per_call,
                max_categories_per_call=self.cfg.max_categories_per_call,
                debug_print=False,
                use_dummy=self.cfg.use_dummy,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
                max_timeout=self.cfg.max_timeout,
            )
            coder = Codify(coder_cfg)
            codify_df = await coder.run(
                df,
                column_name,  # type: ignore[arg-type]
                categories=None,
                additional_instructions=self.cfg.additional_instructions or "",
                reset_files=reset_files,
                **kwargs,
            )
            term_defs: Dict[str, str] = {}
            if "coded_passages" in codify_df:
                for entry in codify_df["coded_passages"].dropna():
                    if isinstance(entry, dict):
                        for k, v in entry.items():
                            if k not in term_defs:
                                if isinstance(v, list) and v:
                                    term_defs[k] = str(v[0])
                                else:
                                    term_defs[k] = str(v) if v is not None else ""
            if self.cfg.raw_term_definitions:
                candidate_df = (
                    pd.DataFrame({"term": [term_defs]})
                    if term_defs
                    else pd.DataFrame({"term": []})
                )
            else:
                candidate_df = pd.DataFrame({"term": sorted(set(term_defs.keys()))})
        else:
            cmp_cfg = CompareConfig(
                save_dir=os.path.join(self.cfg.save_dir, "compare"),
                model=self.cfg.model,
                n_parallels=self.cfg.n_parallels,
                use_dummy=self.cfg.use_dummy,
                max_timeout=self.cfg.max_timeout,
                differentiate=self.cfg.differentiate,
                additional_instructions=self.cfg.additional_instructions,
                modality=self.cfg.modality,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
            )
            cmp = Compare(cmp_cfg)
            compare_df = await cmp.run(
                df,
                circle_column_name,  # type: ignore[arg-type]
                square_column_name,  # type: ignore[arg-type]
                reset_files=reset_files,
                **kwargs,
            )
            term_defs = {}
            for attr, expl in zip(
                compare_df["attribute"], compare_df["explanation"]
            ):
                if pd.notna(attr) and attr not in term_defs:
                    term_defs[attr] = str(expl) if pd.notna(expl) else ""
            if self.cfg.raw_term_definitions:
                candidate_df = (
                    pd.DataFrame({"term": [term_defs]})
                    if term_defs
                    else pd.DataFrame({"term": []})
                )
            else:
                candidate_df = pd.DataFrame({"term": sorted(set(term_defs.keys()))})

        # ── 2. bucketisation ───────────────────────────────────────────
        bucket_df: pd.DataFrame
        if candidate_df.empty:
            bucket_df = pd.DataFrame(columns=["bucket", "definition"])
        else:
            buck_cfg = BucketConfig(
                bucket_count=self.cfg.bucket_count,
                save_dir=os.path.join(self.cfg.save_dir, "bucket"),
                model=self.cfg.model,
                n_parallels=self.cfg.n_parallels,
                use_dummy=self.cfg.use_dummy,
                additional_instructions=self.cfg.additional_instructions,
                differentiate=self.cfg.differentiate if pair else False,
                n_terms_per_prompt=self.cfg.n_terms_per_prompt,
                repeat_bucketing=self.cfg.repeat_bucketing,
                repeat_voting=self.cfg.repeat_voting,
                next_round_frac=self.cfg.next_round_frac,
                top_k_per_round=self.cfg.top_k_per_round,
                raw_term_definitions=self.cfg.raw_term_definitions,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
                max_timeout=self.cfg.max_timeout,
            )
            buck = Bucket(buck_cfg)
            bucket_df = await buck.run(
                candidate_df,
                "term",
                reset_files=reset_files,
                **kwargs,
            )

        labels = (
            dict(zip(bucket_df["bucket"], bucket_df["definition"]))
            if not bucket_df.empty
            else {}
        )

        # ── 3. classification ──────────────────────────────────────────
        classify_result: pd.DataFrame
        summary_df: Optional[pd.DataFrame] = None
        if not labels:
            classify_result = df.reset_index(drop=True).copy()
        elif pair:
            base_cfg = {
                "model": self.cfg.model,
                "n_parallels": self.cfg.n_parallels,
                "n_runs": self.cfg.n_runs,
                "min_frequency": self.cfg.min_frequency,
                "use_dummy": self.cfg.use_dummy,
                "modality": self.cfg.modality,
                "reasoning_effort": self.cfg.reasoning_effort,
                "reasoning_summary": self.cfg.reasoning_summary,
                "n_attributes_per_run": 8,
                "differentiate": True,
                "additional_instructions": self.cfg.additional_instructions or "",
                "max_timeout": self.cfg.max_timeout,
            }

            def swap_cs(text: str) -> str:
                def repl(match: re.Match[str]) -> str:
                    word = match.group(0)
                    return "square" if word.lower() == "circle" else "circle"
                return re.sub(r"(?i)circle|square", repl, text)

            def build_combined_metadata(
                base_labels: Dict[str, str]
            ) -> Tuple[Dict[str, str], Dict[str, str]]:
                combined_local: Dict[str, str] = {}
                rename_local: Dict[str, str] = {}
                for lab, desc in base_labels.items():
                    actual_key = lab
                    swapped_lab = swap_cs(lab)
                    if swapped_lab == lab:
                        swapped_lab = f"{lab} (inverted)"
                    combined_local[actual_key] = desc
                    rename_local[actual_key] = f"{lab}_actual"
                    combined_local[swapped_lab] = swap_cs(desc)
                    rename_local[swapped_lab] = f"{lab}_inverted"
                return combined_local, rename_local

            def derive_base_from_combined(
                combined_local: Dict[str, str]
            ) -> Tuple[Dict[str, str], Dict[str, str]]:
                base_local: Dict[str, str] = {}
                rename_local: Dict[str, str] = {}
                processed: Set[str] = set()
                for key, desc in combined_local.items():
                    if key in processed:
                        continue
                    swapped_key = swap_cs(key)
                    inverted_key: Optional[str] = None
                    if swapped_key != key and swapped_key in combined_local:
                        inverted_key = swapped_key
                    else:
                        candidate = f"{key} (inverted)"
                        if candidate in combined_local:
                            inverted_key = candidate
                    canonical_key = key
                    canonical_desc = desc
                    if canonical_key.endswith(" (inverted)") and swapped_key in combined_local:
                        canonical_key = swapped_key
                        canonical_desc = combined_local[canonical_key]
                        inverted_key = key
                    base_local[canonical_key] = canonical_desc
                    rename_local[canonical_key] = f"{canonical_key}_actual"
                    if inverted_key and inverted_key in combined_local:
                        rename_local[inverted_key] = f"{canonical_key}_inverted"
                        processed.add(inverted_key)
                    processed.add(canonical_key)
                return base_local, rename_local

            combined_labels, rename_map = build_combined_metadata(labels)

            clf_cfg = ClassifyConfig(
                labels=combined_labels,
                save_dir=os.path.join(self.cfg.save_dir, "classify"),
                **base_cfg,  # type: ignore[arg-type]
            )

            clf = Classify(clf_cfg)

            combined_df = await clf.run(
                df,
                circle_column_name=circle_column_name,  # type: ignore[arg-type]
                square_column_name=square_column_name,  # type: ignore[arg-type]
                reset_files=reset_files,
                **kwargs,
            )

            actual_combined_labels = dict(clf.cfg.labels)
            if set(actual_combined_labels.keys()) != set(combined_labels.keys()):
                print(
                    "[Discover] Detected mismatch between cached classifier labels and generated buckets; "
                    "using cached labels from the classification cache instead."
                )
                combined_labels = actual_combined_labels
                labels, rename_map = derive_base_from_combined(combined_labels)
                if labels:
                    bucket_df = pd.DataFrame(
                        {
                            "bucket": list(labels.keys()),
                            "definition": list(labels.values()),
                        }
                    )
                else:
                    bucket_df = pd.DataFrame(columns=["bucket", "definition"])

            classify_result = combined_df.rename(columns=rename_map)

            available: Dict[str, str] = {}
            missing: List[str] = []
            for lab, desc in labels.items():
                actual_col = f"{lab}_actual"
                inverted_col = f"{lab}_inverted"
                if (
                    actual_col not in classify_result.columns
                    or inverted_col not in classify_result.columns
                ):
                    missing.append(lab)
                    continue
                available[lab] = desc
            if missing:
                print(
                    "[Discover] Warning: classification cache is missing the following labels, "
                    "so they were skipped:",
                    ", ".join(missing),
                )
                if not bucket_df.empty:
                    bucket_df = bucket_df[bucket_df["bucket"].isin(available.keys())]
            labels = available

            summary_records: List[Dict[str, Any]] = []
            for lab in labels:
                actual_col = f"{lab}_actual"
                inverted_col = f"{lab}_inverted"
                actual_true = (
                    classify_result[actual_col]
                    .fillna(False)
                    .infer_objects()
                    .sum()
                )
                inverted_true = (
                    classify_result[inverted_col]
                    .fillna(False)
                    .infer_objects()
                    .sum()
                )
                total = classify_result[[actual_col, inverted_col]].notna().any(axis=1).sum()
                actual_pct = (actual_true / total * 100) if total else None
                inverted_pct = (inverted_true / total * 100) if total else None
                net_pct = (
                    (actual_pct - inverted_pct)
                    if actual_pct is not None and inverted_pct is not None
                    else None
                )
                summary_records.append({
                    "label": lab,
                    "actual_true": actual_true,
                    "inverted_true": inverted_true,
                    "total": total,
                    "actual_pct": actual_pct,
                    "inverted_pct": inverted_pct,
                    "net_pct": net_pct,
                })
            summary_df = pd.DataFrame(summary_records)
        else:
            clf_cfg = ClassifyConfig(
                labels=labels,
                save_dir=os.path.join(self.cfg.save_dir, "classify"),
                model=self.cfg.model,
                n_parallels=self.cfg.n_parallels,
                n_runs=self.cfg.n_runs,
                min_frequency=self.cfg.min_frequency,
                additional_instructions=self.cfg.additional_instructions or "",
                use_dummy=self.cfg.use_dummy,
                modality=self.cfg.modality,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
                n_attributes_per_run=8,
                max_timeout=self.cfg.max_timeout,
            )
            clf = Classify(clf_cfg)
            classify_result = await clf.run(
                df,
                column_name,  # type: ignore[arg-type]
                reset_files=reset_files,
                **kwargs,
            )
            actual_labels = dict(clf.cfg.labels)
            if actual_labels != labels:
                print(
                    "[Discover] Detected mismatch between cached classifier labels and generated buckets; "
                    "using cached labels from the classification cache instead."
                )
                labels = actual_labels
                if labels:
                    bucket_df = pd.DataFrame(
                        {"bucket": list(labels.keys()), "definition": list(labels.values())}
                    )
                else:
                    bucket_df = pd.DataFrame(columns=["bucket", "definition"])
            else:
                labels = actual_labels

        result: Dict[str, Any] = {
            "candidates": candidate_df,
            "buckets": labels,
            "classification": classify_result,
        }
        if not bucket_df.empty:
            result["bucket_df"] = bucket_df
        if summary_df is not None:
            result["summary"] = summary_df
        if compare_df is not None:
            result["compare"] = compare_df
        if codify_df is not None:
            result["codify"] = codify_df
        self._persist_result_snapshot(result)
        self._export_result_archive(result)
        return result
