from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from gabriel.core.prompt_template import PromptTemplate, resolve_template
from gabriel.utils.openai_utils import get_all_responses, response_to_text
from gabriel.utils.logging import announce_prompt_rendering
from gabriel.tasks.rank import Rank, RankConfig
from gabriel.tasks.rate import Rate, RateConfig
from gabriel.tasks.deduplicate import Deduplicate, DeduplicateConfig
from gabriel.tasks.seed import Seed, SeedConfig


_SCI_ATTR_LABEL = "major new contribution to literature"
_SCI_ATTR_DESCRIPTION = (
    "Measures how original, well-reasoned, and consequential the proposed theory is. "
    "High scores correspond to ideas that introduce novel and creative thought, "
    "but above all are just genuinely superior scientific theory pursuant to the topic. "
    "Use your best professor hat to judge 'major new contribution to literature' just as a high quality journal would, seeing past basic tricks and rehashes to identify true brilliance and novel cleverness. "
    "Theories that contribute to the literature say something usefully new and interesting, capturing something in the real world better "
    "than existing thought. Novel yet specific, testable, non-trivial, and brilliant/inspired such that top professors would admire it deeply; "
    "a high standard requiring deep thought and consideration, worthy of evaluating frontier research theories. "
    "Give low ratings to anything but a truly exceptional new theory that goes beyond existing work; penalize uncreative and unambitious theories that parrot existing work, "
    "while rewarding clearly new ideas that are clever, logically sensible, and explain important (NOT trivial and vague) things about the world that existing theories don't. "
    "Don't reward focus on niches and fads that don't really matter. Winning theories can be old school or new school, as long as they speak to something genuinely important in the topic and the world. "
    "Reward interesting and important and clever, not just slapping old work onto something new like quantum or smartphones if it is just for the sake of it. "
    "Penalize lack of clarity, where jargon or complex writing obfuscates the underlying ideas. Penalize proposals that just try to sound smart by being complicated / are unreadable. Penalize if core ideas aren't truly clear, parsimonious, well written, or presented with the intention to convey understanding. "
    "Parsimony and clarity are key. "
    "A major contribution to the literature MUST explain something big and significant, NOT tiny effects that don't really matter. "
    "Default to low ratings unless you are fully convinced this is truly brilliant work deserving of research and publication."
)

_GEN_ATTR_LABEL = "overall idea quality"
_GEN_ATTR_DESCRIPTION = (
    "Measures how useful, clear, and high-leverage the idea is for the stated topic. "
    "Reward ideas that are non-trivial in scope, tailored to the specific context described in the topic, explain the mechanism clearly, "
    "and would be genuinely valuable if executed or tested. "
    "Favor solutions that fit the situation at hand while remaining adaptable to adjacent settings. "
    "Prioritize clarity, practicality, and impact over novelty for novelty's sake. "
    "Penalize tiny scope, vague wording, or ideas that are mostly buzzwords. "
    "Penalize obfuscation and jargon-heavy writing that hides the core idea. "
    "A top score requires a concrete, high-quality idea with an explainable path to success and clear evaluation signals."
)


def _default_attributes(scientific_theory: bool) -> Dict[str, str]:
    return (
        {_SCI_ATTR_LABEL: _SCI_ATTR_DESCRIPTION}
        if scientific_theory
        else {_GEN_ATTR_LABEL: _GEN_ATTR_DESCRIPTION}
    )


@dataclass
class IdeateConfig:
    """Configuration for :class:`Ideate`."""

    save_dir: str = os.path.expanduser("~/Documents/runs")
    file_name: str = "ideation.csv"
    model: str = "gpt-5-mini"
    ranking_model: Optional[str] = None
    n_parallels: int = 650
    n_ideas: int = 1000
    evaluation_mode: str = "recursive_rank"
    scientific_theory: bool = True
    attributes: Optional[Dict[str, str]] = None
    rank_attribute: Optional[str] = None
    recursive_fraction: float = 1.0 / 3.0
    recursive_min_remaining: int = 30
    recursive_final_round_multiplier: int = 3
    recursive_cut_side: str = "top"
    recursive_rate_first_round: bool = True
    additional_instructions: Optional[str] = None
    use_dummy: bool = False
    web_search: bool = False
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None
    use_seed_entities: bool = True
    seed_num_entities: Optional[int] = None
    seed_entities_per_generation: Optional[int] = None
    seed_entity_batch_frac: Optional[float] = None
    seed_existing_entities_cap: Optional[int] = None
    seed_additional_instructions: Optional[str] = None
    seed_template_path: Optional[str] = None
    seed_deduplicate: bool = True
    deduplicate_ideas: bool = True

    def __post_init__(self) -> None:
        if self.attributes is None:
            self.attributes = _default_attributes(self.scientific_theory)
        if self.additional_instructions is not None:
            cleaned = str(self.additional_instructions).strip()
            self.additional_instructions = cleaned or None
        if self.seed_additional_instructions is not None:
            cleaned_seed = str(self.seed_additional_instructions).strip()
            self.seed_additional_instructions = cleaned_seed or None


class Ideate:
    """Generate and optionally score ideas or scientific theories."""

    def __init__(
        self,
        cfg: IdeateConfig,
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
            reference_filename="ideation_prompt.jinja2",
        )

    async def run(
        self,
        topic: str,
        *,
        additional_instructions: Optional[str] = None,
        evaluation_mode: Optional[str] = None,
        attributes: Optional[Dict[str, str]] = None,
        rank_attribute: Optional[str] = None,
        reset_files: bool = False,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        rank_config_updates: Optional[Dict[str, Any]] = None,
        rank_run_kwargs: Optional[Dict[str, Any]] = None,
        rate_config_updates: Optional[Dict[str, Any]] = None,
        rate_run_kwargs: Optional[Dict[str, Any]] = None,
        use_seed_entities: Optional[bool] = None,
        seed_config_updates: Optional[Dict[str, Any]] = None,
        seed_run_kwargs: Optional[Dict[str, Any]] = None,
        deduplicate_ideas: Optional[bool] = None,
        deduplicate_config_updates: Optional[Dict[str, Any]] = None,
        deduplicate_run_kwargs: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Generate a large batch of theories and optionally score them."""

        base_name = os.path.splitext(self.cfg.file_name)[0]
        final_path = os.path.join(self.cfg.save_dir, f"{base_name}_final.csv")

        if not reset_files and os.path.exists(final_path):
            try:
                print(f"[Ideate] Loading cached results from {final_path}")
                cached = pd.read_csv(final_path)
                return cached
            except Exception:
                print("[Ideate] Failed to load cached results; recomputing.")

        attrs = attributes or self.cfg.attributes
        if not attrs:
            raise ValueError("At least one attribute must be provided for scoring")
        attr_key = str(
            rank_attribute or self.cfg.rank_attribute or next(iter(attrs))
        ).strip()

        mode = (evaluation_mode or self.cfg.evaluation_mode or "none").lower()
        if mode not in {"recursive_rank", "rank", "rate", "none"}:
            raise ValueError(
                "evaluation_mode must be one of 'recursive_rank', 'rank', 'rate', or 'none'"
            )

        gen_kwargs = dict(generation_kwargs or {})
        rank_cfg_updates = dict(rank_config_updates or {})
        rank_run_kwargs = dict(rank_run_kwargs or {})
        rate_cfg_updates = dict(rate_config_updates or {})
        rate_run_kwargs = dict(rate_run_kwargs or {})

        use_seed = (
            self.cfg.use_seed_entities if use_seed_entities is None else use_seed_entities
        )
        use_dedup = (
            self.cfg.deduplicate_ideas if deduplicate_ideas is None else deduplicate_ideas
        )
        dedup_cfg_updates = dict(deduplicate_config_updates or {})
        dedup_run_kwargs = dict(deduplicate_run_kwargs or {})

        raw_df, _ = await self._generate_reports(
            topic,
            additional_instructions or self.cfg.additional_instructions,
            reset_files=reset_files,
            use_seed_entities=use_seed,
            **gen_kwargs,
            seed_config_updates=seed_config_updates or {},
            seed_run_kwargs=seed_run_kwargs or {},
        )
        parsed_df = self._parse_reports(raw_df, topic)
        self._print_random_previews(parsed_df)

        if use_dedup:
            parsed_df = await self._deduplicate_ideas(
                parsed_df,
                reset_files=reset_files,
                config_updates=dedup_cfg_updates,
                run_kwargs=dedup_run_kwargs,
            )

        topic_instruction = (
            "Research field/topic the theories are situated in, and should be judged in the context of: "
            f"{topic}"
        )

        if mode == "none":
            parsed_df.to_csv(final_path, index=False)
            return parsed_df

        if mode == "rate":
            scored_df = await self._apply_rate(
                parsed_df,
                attrs,
                attr_key,
                topic_instruction,
                reset_files=reset_files,
                config_updates=rate_cfg_updates,
                run_kwargs=rate_run_kwargs,
            )
        else:
            recursive = mode == "recursive_rank"
            scored_df = await self._apply_rank(
                parsed_df,
                attrs,
                attr_key,
                topic_instruction,
                recursive=recursive,
                reset_files=reset_files,
                config_updates=rank_cfg_updates,
                run_kwargs=rank_run_kwargs,
            )

        self._print_rank_summaries(scored_df, attr_key)
        scored_df.to_csv(final_path, index=False)
        return scored_df

    async def _generate_reports(
        self,
        topic: str,
        additional_instructions: Optional[str],
        *,
        reset_files: bool,
        use_seed_entities: bool,
        seed_config_updates: Dict[str, Any],
        seed_run_kwargs: Dict[str, Any],
        **generation_kwargs: Any,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        base_name = os.path.splitext(self.cfg.file_name)[0]
        raw_path = os.path.join(self.cfg.save_dir, f"{base_name}_raw_responses.csv")
        print(
            f"[Ideate] Generating {self.cfg.n_ideas} theories with model {self.cfg.model}."
        )

        seed_assignments: List[Optional[str]] = []
        seed_df: Optional[pd.DataFrame] = None
        seeds_enabled = use_seed_entities
        if seeds_enabled:
            seed_df = await self._generate_seed_entities(
                topic,
                additional_instructions,
                reset_files=reset_files,
                config_updates=seed_config_updates,
                run_kwargs=seed_run_kwargs,
            )
            seed_assignments = (
                seed_df["entity"].astype(str).str.strip().tolist() if seed_df is not None else []
            )
            seed_assignments = [s for s in seed_assignments if s]
            if len(seed_assignments) < self.cfg.n_ideas:
                print(
                    "[Ideate] Warning: insufficient unique seeds; recycling to cover all prompts."
                )
                if not seed_assignments:
                    seeds_enabled = False
            if seeds_enabled:
                while len(seed_assignments) < self.cfg.n_ideas:
                    deficit = self.cfg.n_ideas - len(seed_assignments)
                    seed_assignments.extend(seed_assignments[:deficit])
                seed_assignments = seed_assignments[: self.cfg.n_ideas]

        prompts: List[str] = []
        identifiers: List[str] = []
        announce_prompt_rendering("Ideate", self.cfg.n_ideas)
        for idx in range(self.cfg.n_ideas):
            seed_text = seed_assignments[idx] if seeds_enabled and idx < len(seed_assignments) else None
            prompts.append(
                self.template.render(
                    topic=topic,
                    additional_instructions=additional_instructions or "",
                    seed=seed_text,
                    scientific_theory=self.cfg.scientific_theory,
                )
            )
            identifiers.append(f"idea-{idx:05d}")

        kwargs = dict(
            model=self.cfg.model,
            n_parallels=self.cfg.n_parallels,
            save_path=raw_path,
            reset_files=reset_files,
            use_dummy=self.cfg.use_dummy,
            reasoning_effort=self.cfg.reasoning_effort,
            reasoning_summary=self.cfg.reasoning_summary,
            print_example_prompt=True,
        )
        kwargs.update(generation_kwargs)
        if "web_search" not in kwargs:
            kwargs["web_search"] = self.cfg.web_search

        df_resp = await get_all_responses(
            prompts=prompts,
            identifiers=identifiers,
            **kwargs,
        )
        if not isinstance(df_resp, pd.DataFrame):
            raise RuntimeError("get_all_responses returned no DataFrame")
        df_resp = df_resp.copy()
        df_resp["idea_id"] = df_resp["Identifier"].astype(str)
        df_resp["topic"] = topic
        df_resp["report_text"] = df_resp["Response"].apply(response_to_text)
        df_resp["report_text"] = df_resp["report_text"].astype(str).str.strip()
        if seeds_enabled:
            df_resp["seed_text"] = seed_assignments[: len(df_resp)]
        else:
            df_resp["seed_text"] = None
        return df_resp, seed_df

    async def _generate_seed_entities(
        self,
        topic: str,
        additional_instructions: Optional[str],
        *,
        reset_files: bool,
        config_updates: Dict[str, Any],
        run_kwargs: Dict[str, Any],
    ) -> pd.DataFrame:
        config_updates = dict(config_updates)
        instructions = self._build_seed_instruction(topic, additional_instructions)
        base_name = os.path.splitext(self.cfg.file_name)[0]
        seed_save = os.path.join(self.cfg.save_dir, "seed")
        template_override = config_updates.pop("template_path", None)
        cfg_kwargs: Dict[str, Any] = dict(
            instructions=instructions,
            save_dir=seed_save,
            file_name=f"{base_name}_seed_entities.csv",
            model=self.cfg.model,
            n_parallels=self.cfg.n_parallels,
            num_entities=self.cfg.seed_num_entities or self.cfg.n_ideas,
            entities_per_generation=self.cfg.seed_entities_per_generation or 20,
            entity_batch_frac=self.cfg.seed_entity_batch_frac or 0.25,
            existing_entities_cap=self.cfg.seed_existing_entities_cap or 100,
            use_dummy=self.cfg.use_dummy,
            deduplicate=self.cfg.seed_deduplicate,
            reasoning_effort=self.cfg.reasoning_effort,
            reasoning_summary=self.cfg.reasoning_summary,
        )
        if self.cfg.seed_additional_instructions:
            cfg_kwargs["instructions"] = (
                f"{cfg_kwargs['instructions'].rstrip()}\n\nAdditional guidance:\n{self.cfg.seed_additional_instructions}"
            )
        cfg_kwargs.update(config_updates)
        seed_cfg = SeedConfig(**cfg_kwargs)
        template_path = template_override or self.cfg.seed_template_path
        seed_task = Seed(seed_cfg, template_path=template_path)
        run_opts = dict(run_kwargs)
        seed_df = await seed_task.run(reset_files=reset_files, **run_opts)
        if not isinstance(seed_df, pd.DataFrame):
            raise RuntimeError("Seed generation did not return a DataFrame")
        return seed_df

    def _build_seed_instruction(
        self, topic: str, additional_instructions: Optional[str]
    ) -> str:
        if self.cfg.scientific_theory:
            base_lines = [
                "Generate concise, specific seed concepts that can anchor frontier scientific theories. ",
                "Each seed should describe a sharply defined angle, mechanism, dataset, real world phenomena or scenario, expressed in 1-2 specific sentences. ",
                "Seeds must be mutually unique and grounded in the topic. ",
                "Do not draft the full theory—provide only the inspirational seed or scenario to explore. ",
                "Be genuinely novel and creative; think deeply about the topic and provide interesting seeds for frontier work that are clearly distinct from one another ",
                "and would lead to completely different theories and ideas if fully explored. ",
                "Again: don't describe a theory, just some details/a domain that would be interesting to pursue a novel theory. ",
                "For each seed, just give some light nudges towards a research focus, NOT the full theory. ",
                "Each seed should touch on important, non-trivial specific subdomains for research; avoid niches, fads, etc that don't have real significance in the research field or the real world. ",
                "Don't obsess with recent events like quantum or DeFi; can be old school too, not necessarily anything to do with current events. ",
                "Can be anything, old events or more recent, wacky or traditional, as long as interesting research focus related to the topic. Present a broad range of seeds across very different interesting angles.",
            ]
        else:
            base_lines = [
                "Generate concise, specific seed concepts that can anchor strong, useful ideas. ",
                "Each seed should describe a sharply defined angle, mechanism, workflow, dataset, or real-world scenario, expressed in 1-2 specific sentences. ",
                "Seeds must be mutually unique and grounded in the topic. ",
                "Do not draft the full idea—provide only the inspirational seed or scenario to explore. ",
                "Favor ideas that are non-trivial in scope and would be valuable if executed. ",
                "Again: don't describe the full solution, just a clear direction or nucleus to explore. ",
                "For each seed, give light nudges toward a focus (strategy, product, policy, operational change, research direction, etc.), NOT the full plan. ",
                "Avoid tiny tweaks or niche fads; aim for significant, practical, and clearly distinct directions. ",
                "Present a broad range of seeds across very different angles so they would lead to genuinely different ideas if explored fully.",
            ]
        base_lines.append("Primary topic focus:")
        base_lines.append(topic.strip())
        if additional_instructions:
            base_lines.append("Contextual guidance from the user:")
            base_lines.append(additional_instructions.strip())
        return "\n".join(line for line in base_lines if line)

    def _parse_reports(self, df: pd.DataFrame, topic: str) -> pd.DataFrame:
        print("[Ideate] Parsing structured sections from each report.")
        df_proc = df.copy()
        df_proc["report_text"] = df_proc["report_text"].apply(response_to_text)
        df_proc["report_text"] = df_proc["report_text"].astype(str).str.strip()

        sections: Dict[str, List[Optional[str]]] = {
            "title": [],
            "in_a_nutshell": [],
            "in_one_paragraph": [],
            "illustrative_examples": [],
            "testable_predictions": [],
            "full_thinking": [],
            "summary_preview": [],
            "report_preview": [],
        }

        for text in df_proc["report_text"].astype(str):
            parsed = self._extract_sections(text)
            sections["title"].append(parsed.get("title"))
            sections["in_a_nutshell"].append(parsed.get("in_a_nutshell"))
            sections["in_one_paragraph"].append(parsed.get("in_one_paragraph"))
            sections["illustrative_examples"].append(parsed.get("illustrative_examples"))
            sections["testable_predictions"].append(parsed.get("testable_predictions"))
            sections["full_thinking"].append(parsed.get("full_thinking"))
            preview_parts: List[str] = []
            for key, label in [
                ("title", "Title"),
                ("in_a_nutshell", "In a nutshell"),
                ("in_one_paragraph", "In one paragraph"),
                ("illustrative_examples", "Illustrative examples"),
                ("testable_predictions", "Testable predictions"),
            ]:
                value = parsed.get(key)
                if value:
                    preview_parts.append(f"{label}: {value}")
            preview_text = "\n\n".join(preview_parts) if preview_parts else None
            sections["summary_preview"].append(preview_text)
            sections["report_preview"].append(preview_text)

        for key, values in sections.items():
            df_proc[key] = values

        df_proc["topic"] = topic
        return self._clean_columns(df_proc)

    async def _deduplicate_ideas(
        self,
        df: pd.DataFrame,
        *,
        reset_files: bool,
        config_updates: Dict[str, Any],
        run_kwargs: Dict[str, Any],
    ) -> pd.DataFrame:
        print("[Ideate] Deduplicating ideas before scoring.")
        dedup_save = os.path.join(self.cfg.save_dir, "ideate_deduplicate")
        base_name = os.path.splitext(self.cfg.file_name)[0]
        dedup_instruction = (
            "You do not need exact matches. Deduplicate ideas that are highly similar, "
            "operate in the same conceptual space, or describe the same underlying theory. "
            "Pick the representative text as the clearest, best-stated, and most complete version."
        )
        extra_instruction = config_updates.get("additional_instructions")
        cfg_kwargs: Dict[str, Any] = dict(
            save_dir=dedup_save,
            file_name=f"{base_name}_deduplicate.csv",
            model=self.cfg.model,
            n_parallels=self.cfg.n_parallels,
            n_runs=1,
            use_dummy=self.cfg.use_dummy,
            modality="text",
            max_words_per_text=500,
            group_size=25,
            additional_instructions=dedup_instruction,
        )
        cfg_kwargs.update(config_updates)
        if extra_instruction:
            cfg_kwargs["additional_instructions"] = (
                f"{dedup_instruction}\n\n{extra_instruction}"
            )
        df_proc = df.copy()
        if "report_text_original" not in df_proc.columns:
            df_proc["report_text_original"] = df_proc["report_text"]
        dedup_cfg = DeduplicateConfig(**cfg_kwargs)
        dedup_task = Deduplicate(dedup_cfg)
        dedup_run_opts = dict(run_kwargs)
        dedup_df = await dedup_task.run(
            df_proc,
            column_name="report_text",
            reset_files=reset_files,
            **dedup_run_opts,
        )
        if "mapped_report_text" in dedup_df.columns:
            dedup_df["report_text"] = dedup_df["mapped_report_text"]
        return dedup_df

    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop raw response metadata and present a consistent column order."""

        raw_columns = {
            "Identifier",
            "Response",
            "Time Taken",
            "Input Tokens",
            "Reasoning Tokens",
            "Output Tokens",
            "Reasoning Effort",
            "Reasoning Summary",
            "Successful",
            "Error Log",
            "Response IDs",
            "Response ID",
        }
        cleaned = df.drop(columns=[col for col in raw_columns if col in df.columns])

        preferred_order = [
            "idea_id",
            "topic",
            "seed_text",
            "report_text",
            "report_text_original",
            "title",
            "in_a_nutshell",
            "in_one_paragraph",
            "illustrative_examples",
            "testable_predictions",
            "full_thinking",
            "summary_preview",
            "report_preview",
        ]

        ordered = [col for col in preferred_order if col in cleaned.columns]
        remaining = [col for col in cleaned.columns if col not in ordered]
        return cleaned.loc[:, ordered + remaining]

    def _extract_sections(self, text: str) -> Dict[str, Optional[str]]:
        headers = {
            "title": "title",
            "in a nutshell": "in_a_nutshell",
            "in one paragraph": "in_one_paragraph",
            "illustrative examples": "illustrative_examples",
            "testable predictions": "testable_predictions",
            "the full thinking": "full_thinking",
        }
        result: Dict[str, Optional[str]] = {v: None for v in headers.values()}
        current_key: Optional[str] = None
        buffer: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line and current_key is None:
                continue
            lowered = line.lower()
            matched = None
            for header_text, key in headers.items():
                if lowered.startswith(f"{header_text}:"):
                    matched = key
                    content = line[len(header_text) + 1 :].strip()
                    if current_key is not None:
                        result[current_key] = "\n".join(buffer).strip() or None
                    buffer = [content] if content else []
                    current_key = key
                    break
            if matched is None:
                if current_key is not None:
                    buffer.append(raw_line.rstrip())
        if current_key is not None:
            result[current_key] = "\n".join(buffer).strip() or None
        return result

    async def _apply_rate(
        self,
        df: pd.DataFrame,
        attributes: Dict[str, str],
        attr_key: str,
        topic_instruction: str,
        *,
        reset_files: bool,
        config_updates: Dict[str, Any],
        run_kwargs: Dict[str, Any],
    ) -> pd.DataFrame:
        print("[Ideate] Scoring reports with Rate task.")
        rate_save = os.path.join(self.cfg.save_dir, "rate")
        base_name = os.path.splitext(self.cfg.file_name)[0]
        cfg_kwargs: Dict[str, Any] = dict(
            attributes=attributes,
            save_dir=rate_save,
            file_name=f"{base_name}_ratings.csv",
            model=self.cfg.ranking_model or self.cfg.model,
            n_parallels=self.cfg.n_parallels,
            use_dummy=self.cfg.use_dummy,
            reasoning_effort=self.cfg.reasoning_effort,
            reasoning_summary=self.cfg.reasoning_summary,
        )
        cfg_kwargs.update(config_updates)
        existing_instruction = cfg_kwargs.get("additional_instructions")
        if existing_instruction:
            cfg_kwargs["additional_instructions"] = (
                f"{existing_instruction.rstrip()}\n\n{topic_instruction}"
            )
        else:
            cfg_kwargs["additional_instructions"] = topic_instruction
        rate_cfg = RateConfig(**cfg_kwargs)
        rate_task = Rate(rate_cfg)
        rate_run_opts = dict(run_kwargs)
        rate_run_opts.setdefault("web_search", False)
        df_scored = await rate_task.run(
            df,
            "report_text",
            reset_files=reset_files,
            **rate_run_opts,
        )
        return self._sort_results(df_scored, attr_key)

    async def _apply_rank(
        self,
        df: pd.DataFrame,
        attributes: Dict[str, str],
        attr_key: str,
        topic_instruction: str,
        *,
        recursive: bool,
        reset_files: bool,
        config_updates: Dict[str, Any],
        run_kwargs: Dict[str, Any],
    ) -> pd.DataFrame:
        print("[Ideate] Ranking reports with Rank task.")
        rank_save = os.path.join(self.cfg.save_dir, "rank")
        base_name = os.path.splitext(self.cfg.file_name)[0]
        cfg_kwargs: Dict[str, Any] = dict(
            attributes=attributes,
            save_dir=rank_save,
            file_name=f"{base_name}_rankings",
            model=self.cfg.ranking_model or self.cfg.model,
            n_parallels=self.cfg.n_parallels,
            use_dummy=self.cfg.use_dummy,
            reasoning_effort=self.cfg.reasoning_effort,
            reasoning_summary=self.cfg.reasoning_summary,
            recursive=recursive,
            recursive_fraction=self.cfg.recursive_fraction,
            recursive_min_remaining=self.cfg.recursive_min_remaining,
            recursive_final_round_multiplier=self.cfg.recursive_final_round_multiplier,
            recursive_cut_side=self.cfg.recursive_cut_side,
            recursive_rate_first_round=self.cfg.recursive_rate_first_round,
        )
        if attr_key and cfg_kwargs.get("recursive"):
            cfg_kwargs.setdefault("recursive_cut_attr", attr_key)
        cfg_kwargs.update(config_updates)
        existing_instruction = cfg_kwargs.get("additional_instructions")
        if existing_instruction:
            cfg_kwargs["additional_instructions"] = (
                f"{existing_instruction.rstrip()}\n\n{topic_instruction}"
            )
        else:
            cfg_kwargs["additional_instructions"] = topic_instruction
        rank_cfg = RankConfig(**cfg_kwargs)
        rank_task = Rank(rank_cfg)
        rank_run_opts = dict(run_kwargs)
        rank_run_opts.setdefault("web_search", False)
        df_ranked = await rank_task.run(
            df,
            "report_text",
            id_column="idea_id",
            reset_files=reset_files,
            **rank_run_opts,
        )
        return self._sort_results(df_ranked, attr_key)

    def _sort_results(self, df: pd.DataFrame, attr_key: str) -> pd.DataFrame:
        resolved = self._resolve_attr_column(df, attr_key)
        if resolved is None:
            return df.reset_index(drop=True)
        df_sorted = df.copy()
        if resolved != attr_key:
            df_sorted = df_sorted.rename(columns={resolved: attr_key})
            resolved = attr_key
        if not pd.api.types.is_numeric_dtype(df_sorted[resolved]):
            df_sorted[resolved] = pd.to_numeric(df_sorted[resolved], errors="coerce")
        df_sorted = df_sorted.sort_values(by=resolved, ascending=False, na_position="last").copy()
        df_sorted.reset_index(drop=True, inplace=True)
        rank_col = f"{attr_key}_rank"
        positions: List[Optional[int]] = []
        counter = 1
        for value in df_sorted[resolved]:
            if pd.isna(value):
                positions.append(None)
            else:
                positions.append(counter)
                counter += 1
        df_sorted[rank_col] = pd.Series(positions, dtype="Int64")
        return df_sorted

    def _print_random_previews(self, df: pd.DataFrame, count: int = 5) -> None:
        if df.empty:
            return
        preview_columns = [
            "summary_preview",
            "title",
            "in_a_nutshell",
            "in_one_paragraph",
            "illustrative_examples",
            "testable_predictions",
        ]
        missing_columns = [col for col in preview_columns if col not in df.columns]
        if missing_columns:
            return
        mask = df[preview_columns].notna().any(axis=1)
        available = df[mask]
        if available.empty:
            return
        sample_count = min(count, len(available))
        print(f"[Ideate] Showing {sample_count} random generated ideas:")
        samples = available.sample(n=sample_count, replace=False)
        for idx, (_, row) in enumerate(samples.iterrows(), start=1):
            preview = self._build_preview(row)
            print(f"\n--- Random Idea {idx} ({row.get('idea_id', 'N/A')}) ---")
            print(preview)

    def _print_rank_summaries(
        self, df: pd.DataFrame, attr_key: str, count: int = 5
    ) -> None:
        if df.empty:
            print("[Ideate] Skipping ranked summaries (missing score column or empty data).")
            return
        resolved = self._resolve_attr_column(df, attr_key)
        if resolved is None:
            print("[Ideate] Skipping ranked summaries (missing score column or empty data).")
            return
        df_local = df.copy()
        if resolved != attr_key:
            df_local = df_local.rename(columns={resolved: attr_key})
            resolved = attr_key
        if not pd.api.types.is_numeric_dtype(df_local[resolved]):
            df_local[resolved] = pd.to_numeric(df_local[resolved], errors="coerce")
        non_null = df_local[df_local[resolved].notna()].copy()
        non_null = non_null.sort_values(by=resolved, ascending=False, na_position="last")
        if non_null.empty:
            print("[Ideate] Skipping ranked summaries (no scored entries available).")
            return
        top_count = min(count, len(non_null))
        print(f"\n[Ideate] Top {top_count} ideas by '{attr_key}':")
        for position, (_, row) in enumerate(non_null.head(top_count).iterrows(), start=1):
            preview = self._build_preview(row)
            score = row.get(attr_key, row.get(resolved, "N/A"))
            print(f"\n#{position} (Score: {score}) - {row.get('idea_id', 'N/A')}")
            print(preview)

        bottom_count = min(count, len(non_null))
        print(f"\n[Ideate] Bottom {bottom_count} ideas by '{attr_key}':")
        tail_rows = non_null.tail(bottom_count).iloc[::-1]
        start_position = len(non_null) - bottom_count + 1
        for offset, (_, row) in enumerate(tail_rows.iterrows()):
            position = start_position + offset
            preview = self._build_preview(row)
            score = row.get(attr_key, row.get(resolved, "N/A"))
            print(f"\n#{position} (Score: {score}) - {row.get('idea_id', 'N/A')}")
            print(preview)

    @staticmethod
    def _normalize_label(label: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(label).lower())

    def _resolve_attr_column(self, df: pd.DataFrame, attr_key: str) -> Optional[str]:
        target = self._normalize_label(attr_key)
        for column in df.columns:
            if self._normalize_label(column) == target:
                return column
        prefixes = ["cumulative_", "final_"]
        pattern = re.compile(r"^(stage\d+_|round\d+_)")
        for column in df.columns:
            stripped = column
            for prefix in prefixes:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix) :]
                    break
            stripped = pattern.sub("", stripped)
            if self._normalize_label(stripped) == target:
                return column
        return None

    def _build_preview(self, row: pd.Series) -> str:
        parts: List[str] = []
        if "summary_preview" in row and isinstance(row["summary_preview"], str):
            return row["summary_preview"].strip()
        if "report_preview" in row and isinstance(row["report_preview"], str):
            return row["report_preview"].strip()
        title = row.get("title")
        nutshell = row.get("in_a_nutshell")
        paragraph = row.get("in_one_paragraph")
        examples = row.get("illustrative_examples")
        predictions = row.get("testable_predictions")
        seed = row.get("seed_text")
        if seed:
            parts.append(f"Seed: {seed}")
        if title:
            parts.append(f"Title: {title}")
        if nutshell:
            parts.append(f"In a nutshell: {nutshell}")
        if paragraph:
            parts.append(f"In one paragraph: {paragraph}")
        if examples:
            parts.append(f"Illustrative examples: {examples}")
        if predictions:
            parts.append(f"Testable predictions: {predictions}")
        return "\n\n".join(parts) if parts else "(No preview available)"
