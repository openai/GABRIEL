from __future__ import annotations

import asyncio
import os
import re
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import html
import unicodedata

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

from ..core.prompt_template import PromptTemplate, resolve_template
from ..utils.openai_utils import get_all_responses
from ..utils import safest_json, safe_json, get_all_embeddings
from ..utils.logging import announce_prompt_rendering


@dataclass
class MergeConfig:
    """Configuration options for :class:`Merge`."""

    save_dir: str = "merge"
    file_name: str = "merge_responses.csv"
    model: str = "gpt-5-nano"
    n_parallels: int = 650
    n_runs: int = 1
    use_dummy: bool = False
    additional_instructions: Optional[str] = None
    use_embeddings: bool = True
    short_list_len: int = 16
    long_list_len: int = 256
    max_attempts: int = 4
    short_list_multiplier: float = 0.5
    auto_match_threshold: float = 0.75
    use_best_auto_match: bool = False
    candidate_scan_chunks: int = 5
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.additional_instructions is not None:
            cleaned = str(self.additional_instructions).strip()
            self.additional_instructions = cleaned or None


class Merge:
    """Fuzzy merge between two DataFrames using LLM assistance."""

    def __init__(
        self,
        cfg: MergeConfig,
        template: Optional[PromptTemplate] = None,
        template_path: Optional[str] = None,
    ) -> None:
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        self.template = resolve_template(
            template=template,
            template_path=template_path,
            reference_filename="merge_prompt.jinja2",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(val: str) -> str:
        """Normalize strings for fuzzy matching."""
        # Convert HTML entities and strip accents before keeping alphanumeric
        txt = html.unescape(val).lower()
        txt = unicodedata.normalize("NFKD", txt)
        return "".join(ch for ch in txt if ch.isalnum())

    @classmethod
    def _deduplicate(
        cls, series: pd.Series
    ) -> Tuple[List[str], Dict[str, List[str]], Dict[str, str]]:
        """Return (unique_values, rep_to_group, norm_to_rep) for a Series."""
        norm_map: Dict[str, str] = {}
        groups: Dict[str, List[str]] = {}
        for val in series.dropna().astype(str):
            norm = cls._normalize(val)
            if norm in norm_map:
                rep = norm_map[norm]
                groups[rep].append(val)
            else:
                norm_map[norm] = val
                groups[val] = [val]
        uniques = list(groups.keys())
        return uniques, groups, norm_map

    # ------------------------------------------------------------------
    async def run(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        *,
        on: Optional[str] = None,
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        how: str = "left",
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Perform an LLM-assisted merge between two DataFrames."""

        if on:
            left_key = right_key = on
        elif left_on and right_on:
            left_key, right_key = left_on, right_on
        else:
            raise ValueError("Specify `on` or both `left_on` and `right_on`.")

        how = how.lower()
        if how not in {"left", "right"}:
            raise ValueError("`how` must be either 'left' or 'right'.")

        if how == "left":
            short_df, long_df = df_left.reset_index(drop=True), df_right.reset_index(drop=True)
            short_key, long_key = left_key, right_key
        else:  # right merge
            short_df, long_df = df_right.reset_index(drop=True), df_left.reset_index(drop=True)
            short_key, long_key = right_key, left_key

        # Deduplicate keys and track normalized maps
        short_uniques, short_groups, short_norm_map = self._deduplicate(short_df[short_key])
        long_uniques, long_groups, long_norm_map = self._deduplicate(long_df[long_key])

        # Build a global norm→representative map for the left-hand keys.
        global_short_norm_map = {self._normalize(s): s for s in short_uniques}

        use_embeddings = self.cfg.use_embeddings and len(long_uniques) >= self.cfg.long_list_len

        if reset_files:
            for p in Path(self.cfg.save_dir).glob("merge_groups_attempt*.json"):
                try:
                    p.unlink()
                except OSError:
                    pass

        short_emb: Dict[str, List[float]] = {}
        long_emb: Dict[str, List[float]] = {}
        if use_embeddings:
            short_emb = await get_all_embeddings(
                texts=short_uniques,
                identifiers=short_uniques,
                save_path=os.path.join(self.cfg.save_dir, "short_embeddings.pkl"),
                reset_file=reset_files,
                use_dummy=self.cfg.use_dummy,
                verbose=self.cfg.verbose,
            )
            long_emb = await get_all_embeddings(
                texts=long_uniques,
                identifiers=long_uniques,
                save_path=os.path.join(self.cfg.save_dir, "long_embeddings.pkl"),
                reset_file=reset_files,
                use_dummy=self.cfg.use_dummy,
                verbose=self.cfg.verbose,
            )

        matches: Dict[str, str] = {}
        remaining = short_uniques[:]
        if use_embeddings and self.cfg.auto_match_threshold > 0:
            short_matrix = np.array([short_emb[s] for s in remaining], dtype=float)
            long_matrix = np.array([long_emb[t] for t in long_uniques], dtype=float)
            short_norms = np.linalg.norm(short_matrix, axis=1) + 1e-8
            long_norms = np.linalg.norm(long_matrix, axis=1) + 1e-8
            sims = (short_matrix @ long_matrix.T) / (short_norms[:, None] * long_norms[None, :])
            for i, s in enumerate(remaining):
                row = sims[i]
                above = np.where(row >= self.cfg.auto_match_threshold)[0]
                if len(above) == 1:
                    matches[s] = long_uniques[above[0]]
                elif len(above) > 1 and self.cfg.use_best_auto_match:
                    best_idx = above[np.argmax(row[above])]
                    matches[s] = long_uniques[best_idx]
            remaining = [s for s in remaining if s not in matches]

        def _build_groups(
            remaining_short: List[str], short_len: int, extra_scans: int = 1
        ) -> Tuple[List[List[str]], List[List[str]]]:
            clusters_out: List[List[str]] = []
            candidates: List[List[str]] = []
            if use_embeddings and short_emb:
                arr = np.array([short_emb[s] for s in remaining_short], dtype=float)
                k = max(1, int(np.ceil(len(remaining_short) / short_len)))
                _, labels = kmeans2(arr, k, minit="points")
                cluster_sets: List[List[str]] = []
                for cluster_id in range(k):
                    members = [remaining_short[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
                    if not members:
                        continue
                    for j in range(0, len(members), short_len):
                        subset = members[j : j + short_len]
                        cluster_sets.append(subset)

                long_matrix = np.array([long_emb[t] for t in long_uniques], dtype=float)
                long_norms = np.linalg.norm(long_matrix, axis=1) + 1e-8
                for subset in cluster_sets:
                    short_vecs = [np.array(short_emb[s], dtype=float) for s in subset]
                    short_norms = [np.linalg.norm(vec) + 1e-8 for vec in short_vecs]
                    orders: List[np.ndarray] = []
                    sims_list: List[np.ndarray] = []
                    for vec, norm in zip(short_vecs, short_norms):
                        sims = long_matrix @ vec / (long_norms * norm)
                        order = np.argsort(sims)[::-1]
                        orders.append(order)
                        sims_list.append(sims)

                    per_term = max(1, math.ceil(self.cfg.long_list_len / len(subset)))
                    for scan in range(extra_scans):
                        combined: Dict[int, float] = {}
                        start = scan * per_term
                        end = start + per_term
                        for order, sims in zip(orders, sims_list):
                            if start >= len(order):
                                continue
                            idx_slice = order[start:end]
                            for idx in idx_slice:
                                score = float(sims[idx])
                                if idx not in combined or score > combined[idx]:
                                    combined[idx] = score
                        if not combined:
                            continue
                        sorted_idx = sorted(combined.keys(), key=lambda i: combined[i], reverse=True)
                        candidate = [long_uniques[i] for i in sorted_idx[: self.cfg.long_list_len]]
                        candidates.append(candidate)
                        clusters_out.append(subset)
            else:
                short_sorted = sorted(remaining_short, key=lambda x: x.lower())
                long_sorted = sorted(long_uniques, key=lambda x: x.lower())
                if len(long_sorted) <= self.cfg.long_list_len:
                    base_candidate = list(long_sorted)
                    for i in range(0, len(short_sorted), short_len):
                        subset = short_sorted[i : i + short_len]
                        for _ in range(extra_scans):
                            clusters_out.append(subset)
                            candidates.append(base_candidate)
                else:
                    import bisect

                    lower_long = [s.lower() for s in long_sorted]
                    for i in range(0, len(short_sorted), short_len):
                        subset = short_sorted[i : i + short_len]
                        mid = subset[len(subset) // 2].lower()
                        idx = bisect.bisect_left(lower_long, mid)
                        start = max(0, idx - self.cfg.long_list_len // 2)
                        for scan in range(extra_scans):
                            scan_start = start + scan * self.cfg.long_list_len
                            scan_end = scan_start + self.cfg.long_list_len
                            if scan_end > len(long_sorted):
                                scan_end = len(long_sorted)
                                scan_start = max(0, scan_end - self.cfg.long_list_len)
                            clusters_out.append(subset)
                            candidates.append(list(long_sorted[scan_start:scan_end]))
            return clusters_out, candidates

        def _parse_response(res: Any) -> Dict[str, str]:
            """Normalize raw model output into a dictionary."""
            if isinstance(res, list):
                combined: Dict[str, str] = {}
                for item in res:
                    if isinstance(item, dict):
                        for k, v in item.items():
                            if isinstance(k, str) and isinstance(v, str):
                                combined[k] = v
                    elif isinstance(item, str):
                        inner = safe_json(item)
                        if isinstance(inner, dict):
                            for k, v in inner.items():
                                if isinstance(k, str) and isinstance(v, str):
                                    combined[k] = v
                res = combined
            elif isinstance(res, str):
                res = safe_json(res)

            if isinstance(res, dict):
                return {k: v for k, v in res.items() if isinstance(k, str) and isinstance(v, str)}
            return {}

        save_path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        progress_path = os.path.join(self.cfg.save_dir, "merge_progress.csv")
        if reset_files and os.path.exists(progress_path):
            try:
                os.remove(progress_path)
            except OSError:
                pass
        for attempt in range(self.cfg.max_attempts):
            if not remaining:
                break
            prev_total = len(matches)
            cur_short_len = max(1, int(self.cfg.short_list_len * (self.cfg.short_list_multiplier ** attempt)))
            group_path = os.path.join(self.cfg.save_dir, f"merge_groups_attempt{attempt}.json")
            if os.path.exists(group_path) and not reset_files:
                with open(group_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                clusters = data.get("clusters", [])
                candidates = data.get("candidates", [])
            else:
                extra = self.cfg.candidate_scan_chunks if attempt >= 1 else 1
                clusters, candidates = _build_groups(remaining, cur_short_len, extra)
                with open(group_path, "w", encoding="utf-8") as f:
                    json.dump({"clusters": clusters, "candidates": candidates}, f)

            prompts: List[str] = []
            identifiers: List[str] = []
            base_ids: List[str] = []
            announce_prompt_rendering("Merge", len(clusters) * max(1, self.cfg.n_runs))
            for idx, (short_terms, long_terms) in enumerate(zip(clusters, candidates)):
                short_dict = {s: "" for s in short_terms}
                prompt = self.template.render(
                    short_list=short_dict,
                    long_list=list(long_terms),
                    additional_instructions=self.cfg.additional_instructions or "",
                )
                base_id = f"merge_{attempt:02d}_{idx:05d}"
                base_ids.append(base_id)
                if self.cfg.n_runs > 1:
                    for run in range(self.cfg.n_runs):
                        prompts.append(prompt)
                        identifiers.append(f"{base_id}_run{run}")
                else:
                    prompts.append(prompt)
                    identifiers.append(base_id)

            if prompts:
                resp_df = await get_all_responses(
                    prompts=prompts,
                    identifiers=identifiers,
                    n_parallels=self.cfg.n_parallels,
                    model=self.cfg.model,
                    save_path=save_path,
                    use_dummy=self.cfg.use_dummy,
                    json_mode=True,
                    reset_files=reset_files if attempt == 0 else False,
                    **kwargs,
                )
            else:
                resp_df = pd.DataFrame(columns=["Identifier", "Response"])

            resp_map = dict(zip(resp_df.get("Identifier", []), resp_df.get("Response", [])))
            parsed = await asyncio.gather(
                *[safest_json(resp_map.get(i, "")) for i in identifiers]
            )

            responses_by_base: Dict[str, List[Dict[str, str]]] = {bid: [] for bid in base_ids}
            for ident, res in zip(identifiers, parsed):
                base_id = ident.rsplit("_run", 1)[0] if self.cfg.n_runs > 1 else ident
                responses_by_base.setdefault(base_id, []).append(_parse_response(res))

            for clus, base_id in zip(clusters, base_ids):
                results = responses_by_base.get(base_id, [])
                normalized_results = [
                    {
                        self._normalize(k): v
                        for k, v in res.items()
                        if isinstance(k, str) and isinstance(v, str)
                    }
                    for res in results
                ]
                for s in clus:
                    counts: Dict[str, int] = {}
                    s_norm = self._normalize(s)
                    for res_map in normalized_results:
                        val = res_map.get(s_norm)
                        if val and self._normalize(val) != "nocertainmatch":
                            counts[val] = counts.get(val, 0) + 1
                    if not counts:
                        continue
                    max_count = max(counts.values())
                    top_candidates = [v for v, c in counts.items() if c == max_count]
                    chosen: Optional[str] = None
                    if len(top_candidates) == 1:
                        chosen = top_candidates[0]
                    elif use_embeddings and short_emb and long_emb:
                        s_vec = np.array(short_emb.get(s, []), dtype=float)
                        if s_vec.size:
                            sims: Dict[str, float] = {}
                            s_norm_val = np.linalg.norm(s_vec) + 1e-8
                            for cand in top_candidates:
                                l_vec = np.array(long_emb.get(cand, []), dtype=float)
                                if l_vec.size:
                                    sims[cand] = float(
                                        s_vec @ l_vec / (s_norm_val * (np.linalg.norm(l_vec) + 1e-8))
                                    )
                            if sims:
                                chosen = max(sims, key=sims.get)
                    if chosen:
                        k_norm = s_norm
                        v_norm = self._normalize(chosen)
                        if k_norm in global_short_norm_map and v_norm in long_norm_map:
                            short_rep = global_short_norm_map[k_norm]
                            long_rep = long_norm_map[v_norm]
                            matches[short_rep] = long_rep

            remaining = [s for s in remaining if s not in matches]
            round_matches = len(matches) - prev_total
            total_matches = len(matches)
            missing = len(remaining)
            print(
                f"[Merge] Attempt {attempt}: {round_matches} matches this round, "
                f"{total_matches} total, {missing} remaining"
            )
            progress_df = pd.DataFrame(
                [
                    {
                        "attempt": attempt,
                        "matches_this_round": round_matches,
                        "total_matches": total_matches,
                        "remaining": missing,
                    }
                ]
            )
            progress_df.to_csv(
                progress_path,
                mode="a",
                header=not os.path.exists(progress_path),
                index=False,
            )

        records: List[Dict[str, str]] = []
        if short_key == long_key:
            temp_col = f"{long_key}_match"
            for short_rep, long_rep in matches.items():
                for s in short_groups.get(short_rep, []):
                    for l in long_groups.get(long_rep, []):
                        records.append({short_key: s, temp_col: l})
            map_df = pd.DataFrame(records, columns=[short_key, temp_col])
            map_df[short_key] = map_df[short_key].astype(object)
            map_df[temp_col] = map_df[temp_col].astype(object)
            merged = short_df.merge(map_df, how="left", on=short_key)
            merged = merged.merge(
                long_df,
                how="left",
                left_on=temp_col,
                right_on=long_key,
                suffixes=("", "_y"),
            )
            merged = merged.drop(columns=[temp_col])
        else:
            for short_rep, long_rep in matches.items():
                for s in short_groups.get(short_rep, []):
                    for l in long_groups.get(long_rep, []):
                        records.append({short_key: s, long_key: l})
            map_df = pd.DataFrame(records, columns=[short_key, long_key])
            map_df[short_key] = map_df[short_key].astype(object)
            map_df[long_key] = map_df[long_key].astype(object)
            merged = short_df.merge(map_df, how="left", on=short_key)
            merged = merged.merge(
                long_df,
                how="left",
                left_on=long_key,
                right_on=long_key,
                suffixes=("", "_y"),
            )
        merged = merged.drop_duplicates(subset=[short_key])
        return merged
