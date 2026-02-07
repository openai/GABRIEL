"""
This module reimplements the original GABRIEL `openai_utils.py` for the
OpenAI Responses API with several improvements:

* Rate limit introspection – a helper fetches the current token/request
  budget from the ``x‑ratelimit-*`` response headers returned by a cheap
  ``GET /v1/models`` call.  These values are used to display how many
  tokens and requests remain per minute.
* User‑friendly summary – before a long job starts, the module prints a
  summary showing the number of prompts, input words, remaining rate‑limit
  capacity, usage tier qualifications, and an estimated cost.
* Improved rate‑limit gating – the token limiter now estimates the worst
  possible output length when the cutoff is unspecified by assuming
  the response could be as long as the input.  This avoids grossly
  underestimating throughput while still honouring the per‑minute token
  budget.
* Exponential backoff with jitter – the retry logic uses a random
  exponential backoff when rate‑limit errors occur, following OpenAI’s
  guidelines for handling 429 responses.

The overall API surface remains compatible with the original file: the
public functions ``get_response`` and ``get_all_responses`` still
exist.
"""

from __future__ import annotations

import asyncio
import csv
import functools
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import re
import tempfile
import time
import subprocess
import sys
import textwrap
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Hashable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from collections import defaultdict, deque
from collections.abc import Iterable
import pickle

from gabriel.utils.logging import get_logger, set_log_level
import logging
import math
import pandas as pd
from aiolimiter import AsyncLimiter
from tqdm.auto import tqdm
import openai
import statistics
import numpy as np
import tiktoken
from dataclasses import dataclass, fields
from pydantic import BaseModel

logger = get_logger(__name__)

# Track whether the verbose usage sheet has been shown to avoid repeating the
# static "info sheet" content on subsequent runs.
_USAGE_SHEET_PRINTED = False
_DEPENDENCIES_VERIFIED = False

# Cap the number of prompts we fully scan when estimating words/tokens.  Large
# datasets are sampled to keep start-up time predictable.
_ESTIMATION_SAMPLE_SIZE = 5000

_TIMEOUT_BURST_RATIO = 1.25

DEFAULT_SYSTEM_INSTRUCTION = (
    "Please provide a helpful response to this inquiry for purposes of academic research."
)

# Try to import requests/httpx for rate‑limit introspection
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore
try:
    import httpx  # type: ignore
except Exception:
    httpx = None  # type: ignore

# Bring in specific error classes for granular handling
try:
    from openai import (
        APIConnectionError,
        APIError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        InvalidRequestError,
        RateLimitError,
    )  # type: ignore
except Exception:
    APIConnectionError = Exception  # type: ignore
    APIError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    AuthenticationError = Exception  # type: ignore
    BadRequestError = Exception  # type: ignore
    InvalidRequestError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore

from gabriel.utils.parsing import parse_json_with_status, safe_json

# single connection pool per process, keyed by base URL and created lazily
_clients_async: Dict[Optional[str], openai.AsyncOpenAI] = {}


def _progress_bar(*args: Any, verbose: bool = True, **kwargs: Any):
    """Construct a tqdm progress bar that degrades gracefully."""

    disable = kwargs.pop("disable", False) or not verbose
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(*args, disable=disable, **kwargs)


def _display_example_prompt(example_prompt: str, *, verbose: bool = True) -> None:
    """Print the full example prompt in plain text for easy copying."""

    if not verbose or not example_prompt:
        return

    print("\n===== Example prompt =====")
    print(textwrap.indent(example_prompt.strip("\n"), "  "))


def _get_client(base_url: Optional[str] = None) -> openai.AsyncOpenAI:
    """Return a cached ``AsyncOpenAI`` client for ``base_url``.

    When ``base_url`` is ``None`` the default OpenAI endpoint is used.  A client
    is created on first use and reused for subsequent calls with the same base
    URL to benefit from connection pooling.
    """

    url = base_url or os.getenv("OPENAI_BASE_URL")
    key: Optional[str] = url
    client = _clients_async.get(key)
    if client is None:
        kwargs: Dict[str, Any] = {}
        if url:
            kwargs["base_url"] = url
        if httpx is not None:
            try:
                kwargs.setdefault(
                    "timeout",
                    httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
                )
            except Exception:
                # Fall back to the SDK default if constructing the timeout fails
                pass
        client = openai.AsyncOpenAI(**kwargs)
        _clients_async[key] = client
    return client

# Estimated output tokens per prompt used for cost estimation when no cutoff is specified.
# When a user does not explicitly set ``max_output_tokens``, we assume that each response
# will contain roughly this many tokens.  This value is used solely for estimating cost
# and determining how many parallel requests can safely run under the token budget.
ESTIMATED_OUTPUT_TOKENS_PER_PROMPT = 500
# Extra input tokens to add per prompt when estimating non-text inputs or web search.
NON_TEXT_INPUT_TOKEN_BUFFER = 2000

# Conservative headroom when translating observed rate limits into concurrency and limiter budgets.
# Using less than the reported limit provides a buffer for short spikes and accounting inaccuracies.
RATE_LIMIT_HEADROOM = 0.85
# Additional planning buffer applied when translating reported rate limits into budgets.
PLANNING_RATE_LIMIT_BUFFER = 0.85
# Cushion applied to expected output tokens during initial planning. This headroom is relaxed to
# ``OUTPUT_TOKEN_HEADROOM_STEADY`` after we accumulate real usage samples.
OUTPUT_TOKEN_HEADROOM_INITIAL = 2.0
OUTPUT_TOKEN_HEADROOM_STEADY = 1.0

# ---------------------------------------------------------------------------
# Helper dataclasses and token utilities


@dataclass
class StatusTracker:
    """Simple container for bookkeeping counters."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_connection_errors: int = 0
    num_api_errors: int = 0
    num_timeout_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: float = 0.0


@dataclass
class DummyResponseSpec:
    """Configuration object describing synthetic responses for dummy runs."""

    responses: Optional[Any] = None
    duration: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    reasoning_summary: Optional[str] = None
    response_id: Optional[str] = None
    successful: Optional[bool] = None
    error_log: Optional[Union[str, List[str]]] = None
    warning: Optional[str] = None


class BackgroundTimeoutError(asyncio.TimeoutError):
    """Timeout raised while polling a background response."""

    def __init__(self, response_id: Optional[str], last_response: Any, message: str):
        super().__init__(message)
        self.response_id = response_id
        self.last_response = last_response


class JSONParseError(ValueError):
    """Raised when JSON parsing fails during JSON-mode requests."""

    def __init__(self, message: str, snippet: Optional[str] = None):
        super().__init__(message)
        self.snippet = snippet


def _extract_retry_after_seconds(error: Exception) -> Optional[float]:
    """Return a retry-after duration in seconds when available."""

    for attr in ("retry_after", "retry_after_s", "retry_after_seconds"):
        retry_value = getattr(error, attr, None)
        if isinstance(retry_value, (int, float)) and retry_value > 0:
            return float(retry_value)
    retry_ms = getattr(error, "retry_after_ms", None)
    if isinstance(retry_ms, (int, float)) and retry_ms > 0:
        return float(retry_ms) / 1000.0
    message = str(error)
    if not message:
        return None
    match = re.search(r"after\s+([0-9]+(?:\.[0-9]+)?)\s*seconds", message)
    if match:
        try:
            parsed = float(match.group(1))
        except ValueError:
            return None
        if parsed > 0:
            return parsed
    return None


def _is_quota_error_message(message: str) -> bool:
    """Return True when the error text indicates an exhausted quota."""

    return bool(message) and "quota" in message.lower()


def _get_tokenizer(model_name: str) -> tiktoken.Encoding:
    """Return a tiktoken encoding for the model or a sensible default."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        class _ApproxEncoder:
            def encode(self, text: str) -> List[int]:
                return [0] * max(1, _approx_tokens(text))

        return _ApproxEncoder()


def _uses_legacy_system_instruction(model_name: str) -> bool:
    """Return True when the model expects legacy system-message prompts."""

    lowered = (model_name or "").lower()
    return lowered.startswith("gpt-3") or lowered.startswith("gpt-4")


def _is_audio_model(model_name: str) -> bool:
    """Return True when the model name indicates audio support."""

    return "audio" in (model_name or "").lower()


def _has_media_payloads(
    prompt_images: Optional[Dict[str, List[str]]],
    prompt_audio: Optional[Dict[str, List[Dict[str, str]]]],
    prompt_pdfs: Optional[Dict[str, List[Dict[str, str]]]],
    identifiers: Iterable[Any],
) -> bool:
    """Return True when any prompt includes image/audio/pdf payloads."""

    if not (prompt_images or prompt_audio or prompt_pdfs):
        return False
    for ident in identifiers:
        key = str(ident)
        if prompt_images and prompt_images.get(key):
            return True
        if prompt_audio and prompt_audio.get(key):
            return True
        if prompt_pdfs and prompt_pdfs.get(key):
            return True
    return False

# Usage tiers with qualifications and monthly limits for printing
TIER_INFO = [
    {
        "tier": "Free",
        "qualification": "User must be in an allowed geography",
        "monthly_quota": "$100 / month",
    },
    {"tier": "Tier 1", "qualification": "$5 added", "monthly_quota": None},
    {"tier": "Tier 2", "qualification": "$50 added and 7+ days since first payment", "monthly_quota": None},
    {"tier": "Tier 3", "qualification": "$100 added and 7+ days since first payment", "monthly_quota": None},
    {"tier": "Tier 4", "qualification": "$250 added and 14+ days since first payment", "monthly_quota": None},
    {"tier": "Tier 5", "qualification": "$1 000 added and 30+ days since first payment", "monthly_quota": None},
]

# Truncated pricing table (USD per million tokens) for a few common models
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # model family       input   cached_input   output   batch_factor
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00, "batch": 0.5},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60, "batch": 0.5},
    "gpt-4.1-nano": {
        "input": 0.10,
        "cached_input": 0.025,
        "output": 0.40,
        "batch": 0.5,
    },
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00, "batch": 0.5},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60, "batch": 0.5},
    "o3": {"input": 2.00, "cached_input": 0.50, "output": 8.00, "batch": 0.5},
    "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40, "batch": 0.5},
    "gpt-audio-mini": {
        "input": 0.60,
        "cached_input": 0.15,
        "output": 2.40,
        "batch": 0.5,
    },
    "gpt-audio": {
        "input": 2.50,
        "cached_input": 0.625,
        "output": 10.00,
        "batch": 0.5,
    },
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00, "batch": 0.5},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00, "batch": 0.5},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00, "batch": 0.5},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00, "batch": 0.5},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40, "batch": 0.5},
    "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40, "batch": 0.5},
    "o3-deep-research": {
        "input": 10.00,
        "cached_input": 2.50,
        "output": 40.00,
        "batch": 0.5,
    },
    "o4-mini-deep-research": {
        "input": 2.00,
        "cached_input": 0.50,
        "output": 8.00,
        "batch": 0.5,
    },
}


def _print_tier_explainer(verbose: bool = True) -> None:
    """Print a helpful explanation of usage tiers and how to increase them.

    This helper can be called when a user encounters errors that may be
    related to low quotas or tier limitations.  It summarises the
    qualifications for each tier and encourages users to check their
    payment status and billing page.  The message is only printed when
    ``verbose`` is ``True``.
    """
    if not verbose:
        return
    print("\n===== Tier explainer =====")
    print(
        "Your ability to call the OpenAI API is governed by usage tiers. Runs on lower usage tiers will be slower."
    )
    print(
        "As you spend more on the API, you are automatically graduated to higher tiers with larger token and request limits."
    )
    print("Here are the current tiers and how to qualify:")
    for tier in TIER_INFO:
        monthly = tier.get("monthly_quota")
        monthly_text = f"; monthly quota {monthly}" if monthly else ""
        print(f"  • {tier['tier']}: qualify by {tier['qualification']}{monthly_text}")
    print("If you are encountering rate limits or truncated outputs, consider:")
    print(
        "  – Checking your current spend and ensuring you have met the payment criteria for a higher tier."
    )
    print(
        "  – Adding funds at https://platform.openai.com/settings/organization/billing/."
    )


def _approx_tokens(text: str) -> int:
    """Roughly estimate the token count from a string by assuming ~1.5 tokens per word."""
    return int(len(str(text).split()) * 1.5)


def _decide_default_max_output_tokens(
    max_output_tokens: Optional[int],
    rate_headers: Optional[Dict[str, str]],
) -> Optional[int]:
    """Choose a default max output token cap when the user leaves it unset."""

    if max_output_tokens is not None:
        return max_output_tokens
    return None


def _lookup_model_pricing(model: str) -> Optional[Dict[str, float]]:
    """Find a pricing entry for ``model`` by prefix match (case‑insensitive)."""
    key = model.lower()
    # Find the most specific prefix match by selecting the longest matching prefix.
    best_match: Optional[Dict[str, float]] = None
    best_len = -1
    for prefix, pricing in MODEL_PRICING.items():
        if key.startswith(prefix) and len(prefix) > best_len:
            best_match = pricing
            best_len = len(prefix)
    return best_match


def _estimate_cost(
    prompts: List[str],
    n: int,
    max_output_tokens: Optional[int],
    model: str,
    use_batch: bool,
    *,
    sample_size: int = _ESTIMATION_SAMPLE_SIZE,
    estimated_output_tokens_per_prompt: int = ESTIMATED_OUTPUT_TOKENS_PER_PROMPT,
    extra_input_tokens_per_prompt: int = 0,
) -> Optional[Dict[str, float]]:
    """Estimate input/output tokens and cost for a set of prompts.

    Returns a dict with keys ``input_tokens``, ``output_tokens``, ``input_cost``, ``output_cost``, and ``total_cost``.
    If the model pricing is unavailable, returns ``None``.
    ``estimated_output_tokens_per_prompt`` controls the assumed output length when no
    explicit ``max_output_tokens`` is supplied.
    """
    pricing = _lookup_model_pricing(model)
    if pricing is None:
        return None
    # Estimate tokens: sample large datasets to avoid long start-up times.
    total_prompts = len(prompts)
    if total_prompts == 0:
        return None
    if sample_size and total_prompts > sample_size:
        # Deterministic sampling keeps estimates stable across runs.
        rng = random.Random(total_prompts)
        sampled = rng.sample(prompts, sample_size)
        avg_tokens = sum(_approx_tokens(p) for p in sampled) / float(sample_size)
        input_tokens = int(avg_tokens * total_prompts * max(1, n))
    else:
        input_tokens = sum(_approx_tokens(p) for p in prompts) * max(1, n)
    if extra_input_tokens_per_prompt > 0:
        input_tokens += extra_input_tokens_per_prompt * total_prompts * max(1, n)
    # Estimate output tokens: when no cutoff is provided we assume a reasonable default
    # number of output tokens per prompt.  This prevents the cost estimate from
    # ballooning for long inputs, which previously assumed the output could be as long
    # as the input.
    if max_output_tokens is None:
        # Use the per‑prompt estimate for each response
        output_tokens = estimated_output_tokens_per_prompt * max(1, n) * len(prompts)
    else:
        output_tokens = max_output_tokens * max(1, n) * len(prompts)
    cost_in = (input_tokens / 1_000_000) * pricing["input"]
    cost_out = (output_tokens / 1_000_000) * pricing["output"]
    if use_batch:
        cost_in *= pricing["batch"]
        cost_out *= pricing["batch"]
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": cost_in,
        "output_cost": cost_out,
        "total_cost": cost_in + cost_out,
    }


def _estimate_tokens_per_call(
    avg_input_tokens: float,
    expected_output_tokens: Optional[int],
    n: int,
    *,
    estimated_output_tokens_per_prompt: int = ESTIMATED_OUTPUT_TOKENS_PER_PROMPT,
    output_headroom: float = OUTPUT_TOKEN_HEADROOM_INITIAL,
) -> float:
    """Return a conservative token estimate per call for planning throughput."""

    gating_output = (
        expected_output_tokens
        if expected_output_tokens is not None
        else estimated_output_tokens_per_prompt
    )
    gating_output *= max(1.0, float(output_headroom))
    tokens_per_call = max(1.0, (avg_input_tokens + gating_output) * max(1, n))
    return tokens_per_call


def _estimate_prompts_per_minute(
    allowed_req_pm: Optional[float],
    allowed_tok_pm: Optional[float],
    tokens_per_call: Optional[float],
) -> Tuple[Optional[float], Dict[str, float]]:
    """Estimate achievable prompts/min given budgets and per-call token needs."""

    details: Dict[str, float] = {}
    if tokens_per_call is None or tokens_per_call <= 0:
        return None, details
    if allowed_req_pm is not None and allowed_req_pm > 0:
        details["request_bound"] = float(allowed_req_pm)
    if allowed_tok_pm is not None and allowed_tok_pm > 0:
        details["token_bound"] = float(allowed_tok_pm) / float(tokens_per_call)
    candidates = [v for v in details.values() if v is not None]
    if not candidates:
        return None, details
    throughput = min(candidates)
    details["throughput"] = throughput
    return throughput, details


def _planned_ppm_and_details(
    allowed_req_pm: Optional[float],
    allowed_tok_pm: Optional[float],
    tokens_per_call: Optional[float],
) -> Tuple[Optional[int], Dict[str, float]]:
    """Return rounded prompts/minute with the underlying limiter details."""

    throughput, details = _estimate_prompts_per_minute(
        allowed_req_pm, allowed_tok_pm, tokens_per_call
    )
    if throughput is None or throughput <= 0:
        return None, details
    return max(1, int(round(throughput))), details


def _safe_planned_ppm_and_details(
    allowed_req_pm: Optional[float],
    allowed_tok_pm: Optional[float],
    tokens_per_call: Optional[float],
    *,
    context: str,
) -> Tuple[Optional[int], Dict[str, float]]:
    """Wrapper around `_planned_ppm_and_details` that never raises."""

    try:
        return _planned_ppm_and_details(
            allowed_req_pm, allowed_tok_pm, tokens_per_call
        )
    except Exception as exc:
        logger.error("Error while %s: %s", context, exc)
        if context == "tuning concurrency":
            print(f"Error while updating concurrency cap dynamically: {exc}")
        return None, {}


def _resolve_limiting_factor(
    throughput_details: Dict[str, float],
    *,
    allowed_req_pm: Optional[float],
    allowed_tok_pm: Optional[float],
) -> Tuple[Optional[str], Optional[float]]:
    """Return the primary limiter (requests/tokens per minute) and its value."""

    req_bound = throughput_details.get("request_bound")
    tok_bound = throughput_details.get("token_bound")
    if req_bound is None and tok_bound is None:
        return None, None
    if tok_bound is None or (req_bound is not None and req_bound <= tok_bound):
        return "requests per minute", allowed_req_pm if allowed_req_pm is not None else req_bound
    return "token budget per minute", allowed_tok_pm if allowed_tok_pm is not None else tok_bound


def _format_throughput_plan(
    *,
    planned_ppm: Optional[int],
    throughput_details: Dict[str, float],
    remaining_prompts: int,
    allowed_req_pm: Optional[float],
    allowed_tok_pm: Optional[float],
    include_upgrade_hint: bool = True,
    tokens_per_call: Optional[float] = None,
    parallel_ceiling: Optional[int] = None,
    n_parallels: Optional[int] = None,
    ultimate_parallel_cap: Optional[int] = None,
) -> List[str]:
    """Build human-friendly throughput summary lines."""

    del include_upgrade_hint, tokens_per_call
    parallel_cap = parallel_ceiling if parallel_ceiling is not None else n_parallels
    ultimate_parallel_cap = (
        max(1, ultimate_parallel_cap) if ultimate_parallel_cap is not None else n_parallels
    )
    fallback_line = "If running into API or timeout errors, try reducing n_parallels."

    if planned_ppm is None or planned_ppm <= 0:
        return [
            "Expected prompts per minute: unknown (rate-limit data unavailable; running with conservative defaults).",
            fallback_line,
        ]
    estimated_minutes = math.ceil(remaining_prompts / planned_ppm) if remaining_prompts > 0 else 0
    minimum_minutes = max(1, estimated_minutes)
    limiter, limiter_value = _resolve_limiting_factor(
        throughput_details,
        allowed_req_pm=allowed_req_pm,
        allowed_tok_pm=allowed_tok_pm,
    )
    lines = [
        f"Expected prompts per minute: maximum of {planned_ppm:,}",
        f"Estimated total mins: minimum of {minimum_minutes} minute{'s' if minimum_minutes != 1 else ''}",
    ]
    rate_label = limiter or "current rate limits"
    rate_val = f"~{int(limiter_value):,}/min" if limiter_value is not None else "rate limits"
    meets_parallel_cap = (
        parallel_cap is not None
        and planned_ppm >= parallel_cap
        and (limiter_value is None or limiter_value >= parallel_cap)
    )
    at_ultimate_parallel_cap = (
        ultimate_parallel_cap is not None
        and meets_parallel_cap
        and parallel_cap == ultimate_parallel_cap
    )
    if at_ultimate_parallel_cap:
        lines.append(
            f"Rate currently limited by n_parallels = {ultimate_parallel_cap}. Increase n_parallels for faster runs, if your machine allows."
        )
    elif limiter:
        lines.append(
            f"Rate currently limited by {limiter} ({rate_val}). Moving to a higher usage tier can raise these limits and allow faster runs."
        )
    else:
        lines.append(
            f"Rate currently limited by {rate_label} ({rate_val}). Moving to a higher usage tier can raise these limits and allow faster runs."
        )
    lines.append(fallback_line)
    return lines


def _estimate_dataset_stats(
    prompts: List[str],
    *,
    sample_size: int = _ESTIMATION_SAMPLE_SIZE,
    extra_input_tokens_per_prompt: int = 0,
) -> Dict[str, Any]:
    """Return rough totals for words and tokens without scanning massive datasets.

    The helper samples up to ``sample_size`` prompts and scales the totals to the
    full dataset.  This keeps initial reporting fast even for hundreds of
    thousands of prompts.
    """

    total_prompts = len(prompts)
    if total_prompts == 0:
        return {"word_count": 0, "token_count": 0, "sampled": False, "sample_size": 0}
    if sample_size and total_prompts > sample_size:
        rng = random.Random(total_prompts)
        sample = rng.sample(prompts, sample_size)
        avg_words = sum(len(str(p).split()) for p in sample) / float(sample_size)
        avg_tokens = sum(_approx_tokens(p) for p in sample) / float(sample_size)
        if extra_input_tokens_per_prompt > 0:
            avg_tokens += float(extra_input_tokens_per_prompt)
        return {
            "word_count": int(avg_words * total_prompts),
            "token_count": int(avg_tokens * total_prompts),
            "sampled": True,
            "sample_size": sample_size,
        }
    extra_tokens = extra_input_tokens_per_prompt * total_prompts
    return {
        "word_count": sum(len(str(p).split()) for p in prompts),
        "token_count": sum(_approx_tokens(p) for p in prompts) + extra_tokens,
        "sampled": False,
        "sample_size": total_prompts,
    }


def _ensure_runtime_dependencies(packages: Optional[List[str]] = None, *, verbose: bool = True) -> None:
    """Install missing runtime dependencies in a best-effort manner.

    The function is intentionally lightweight: it checks for a small set of
    packages and silently returns when everything is already present.  When a
    package is missing, ``pip`` is invoked to install only the missing items so
    the helper works in local, Colab, Databricks, and CI environments without
    user intervention.
    """

    global _DEPENDENCIES_VERIFIED
    if _DEPENDENCIES_VERIFIED:
        return
    pkgs = packages or ["wheel", "tiktoken", "aiolimiter", "httpx", "requests"]
    missing = [pkg for pkg in pkgs if importlib.util.find_spec(pkg) is None]
    if not missing:
        _DEPENDENCIES_VERIFIED = True
        return
    if verbose:
        print(
            "Installing missing dependencies for GABRIEL (once per session): "
            + ", ".join(sorted(missing))
        )
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", *missing],
            check=True,
        )
        _DEPENDENCIES_VERIFIED = True
    except Exception as exc:
        logger.warning("Automatic dependency installation failed: %s", exc)


def _print_run_banner(
    *,
    prompts: List[str],
    model: str,
    n: int,
    use_batch: bool,
    modality: Optional[str],
    web_search: bool,
    estimated_cost: Optional[Dict[str, float]],
    max_output_tokens: Optional[int],
    stats: Dict[str, Any],
    estimated_output_tokens_per_prompt: int = ESTIMATED_OUTPUT_TOKENS_PER_PROMPT,
    verbose: bool = True,
) -> None:
    """Print an immediate run overview so users see progress right away."""

    if not verbose:
        return
    print("\n===== Run kickoff =====")
    total_words = stats.get("word_count", 0) or 0
    words_per_prompt = (
        int(round(total_words / max(len(prompts), 1))) if prompts else 0
    )
    print(
        f"Prompts: {len(prompts):,} | Words: ~{total_words:,} | Words per prompt: ~{words_per_prompt:,}"
    )
    modality_segment = f" | modality: {modality}" if modality else ""
    print(
        f"Model: {model} | Mode: {'batch' if use_batch else 'streaming'}{modality_segment}"
    )
    pricing = _lookup_model_pricing(model)
    if pricing:
        print(
            f"Pricing for model '{model}': input ${pricing['input']}/1M, output ${pricing['output']}/1M"
        )
    tokens_per_call = None
    if stats and prompts:
        avg_input_tokens = (stats.get("token_count") or 0) / max(1, len(prompts))
        tokens_per_call = _estimate_tokens_per_call(
            avg_input_tokens,
            max_output_tokens,
            n,
            estimated_output_tokens_per_prompt=estimated_output_tokens_per_prompt,
            output_headroom=OUTPUT_TOKEN_HEADROOM_INITIAL,
        )
    if estimated_cost:
        token_usage = (
            f"Estimated token usage: input {estimated_cost['input_tokens']:,}, output {estimated_cost['output_tokens']:,}"
        )
        if tokens_per_call:
            token_usage += f" | ~{int(round(tokens_per_call)):,} tokens per call"
        print(token_usage)
        print(
            f"Estimated {'batch' if use_batch else 'synchronous'} cost: ${estimated_cost['total_cost']:.2f} "
            f"(input: ${estimated_cost['input_cost']:.2f}, output: ${estimated_cost['output_cost']:.2f})"
        )
        if _is_multimodal_estimate(modality=modality, web_search=web_search):
            print(
                "Note: multimedia/web inputs can make cost estimates unreliable. Monitor usage in the OpenAI dashboard."
            )
    else:
        if pricing:
            print("Estimated token usage unavailable for this model.")
        print("Estimated cost unavailable for this model.")


def _infer_modality_from_inputs(
    prompt_images: Optional[Dict[str, List[str]]],
    prompt_audio: Optional[Dict[str, List[Dict[str, str]]]],
    prompt_pdfs: Optional[Dict[str, List[Dict[str, str]]]],
) -> str:
    present = []
    if prompt_images:
        present.append("image")
    if prompt_audio:
        present.append("audio")
    if prompt_pdfs:
        present.append("pdf")
    if not present:
        return "text"
    if len(present) > 1:
        return "mixed"
    return present[0]


def _estimate_extra_input_tokens_per_prompt(
    *,
    modality: Optional[str],
    web_search: bool,
    has_media: bool,
) -> int:
    """Return the extra input tokens to add per prompt for non-text inputs."""

    mode = (modality or "").lower()
    if mode and mode not in {"text", "entity"}:
        return NON_TEXT_INPUT_TOKEN_BUFFER
    if web_search or has_media:
        return NON_TEXT_INPUT_TOKEN_BUFFER
    return 0


def _is_multimodal_estimate(
    *,
    modality: Optional[str],
    web_search: bool,
    has_media: bool = False,
) -> bool:
    mode = (modality or "").lower()
    if web_search:
        return True
    if mode and mode not in {"text", "entity"}:
        return True
    if has_media:
        return True
    return False



def _require_api_key() -> str:
    """Return the API key or raise a runtime error if missing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable must be set or passed via OpenAIClient(api_key)."
        )
    return api_key


def _get_rate_limit_headers(
    model: str = "gpt-5-mini", base_url: Optional[str] = None
) -> Optional[Dict[str, str]]:
    """Retrieve rate‑limit headers via a cheap API request.

    The OpenAI platform does not yet expose a dedicated endpoint for
    checking how many requests or tokens remain in your minute quota.  In
    practice, these values are only communicated via ``x‑ratelimit-*``
    headers on API responses.  The newer *Responses* API does not
    consistently include these headers as of mid‑2025【360365694688557†L209-L243】, but it
    may in the future.  To accommodate current and future behaviour, this
    helper first tries a minimal call against the Responses endpoint and
    falls back to a tiny call against the Chat completions endpoint when
    the headers are absent.  Both calls cap generation at one token to
    minimise usage.

    :param model: The model to use for the dummy request.  Matching the
      model you intend to use yields the most accurate limits.
    :returns: A dictionary containing limit and remaining values for
      requests and tokens if successful, otherwise ``None``.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    base = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    base = base.rstrip("/")
    # Define two candidate endpoints: the Responses API and the Chat
    # completions API.  In mid‑2025 the Responses API often omits rate‑limit
    # headers【360365694688557†L209-L243】, but OpenAI may add them in the future.  We try
    # the Responses endpoint first to see if headers are now included; if
    # missing, we fall back to a minimal call to the chat completions
    # endpoint.  Both calls cap generation at one token to minimise usage.
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    # Responses API payload (first attempt)
    candidates.append(
        (
            f"{base}/responses",
            {
                "model": model,
                "input": [
                    {"role": "user", "content": "Hello"},
                ],
                "truncation": "auto",
                "max_output_tokens": 1,
            },
        )
    )
    # Chat completions API payload (fallback)
    candidates.append(
        (
            f"{base}/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
                "max_tokens": 1,
            },
        )
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for url, payload in candidates:
        for client in (requests, httpx):
            if client is None:
                continue
            try:
                resp = client.post(url, headers=headers, json=payload, timeout=10)  # type: ignore
                h = getattr(resp, "headers", {})  # type: ignore
                new_h = {k.lower(): v for k, v in h.items()}
                # Collect both standard and usage‑based headers.  If the
                # responses API is missing them, continue to the next
                # candidate.
                limit_requests = new_h.get("x-ratelimit-limit-requests")
                remaining_requests = new_h.get("x-ratelimit-remaining-requests")
                reset_requests = new_h.get("x-ratelimit-reset-requests")
                limit_tokens = new_h.get("x-ratelimit-limit-tokens") or new_h.get(
                    "x-ratelimit-limit-tokens_usage_based"
                )
                remaining_tokens = new_h.get("x-ratelimit-remaining-tokens") or new_h.get(
                    "x-ratelimit-remaining-tokens_usage_based"
                )
                reset_tokens = new_h.get("x-ratelimit-reset-tokens") or new_h.get(
                    "x-ratelimit-reset-tokens_usage_based"
                )
                # If any of the primary values are present, return them.  Some
                # providers may omit remaining values until you are close to
                # the limit, so we treat the presence of a limit value as
                # success.
                if limit_requests or limit_tokens:
                    return {
                        "limit_requests": limit_requests,
                        "remaining_requests": remaining_requests,
                        "reset_requests": reset_requests,
                        "limit_tokens": limit_tokens,
                        "remaining_tokens": remaining_tokens,
                        "reset_tokens": reset_tokens,
                    }
            except Exception:
                # Ignore any errors and try the next client or candidate
                continue
    return None


def _print_usage_overview(
    prompts: List[str],
    n: int,
    max_output_tokens: Optional[int],
    model: str,
    use_batch: bool,
    n_parallels: int,
    *,
    verbose: bool = True,
    rate_headers: Optional[Dict[str, str]] = None,
    base_url: Optional[str] = None,
    web_search_warning: Optional[str] = None,
    web_search_parallel_note: Optional[str] = None,
    show_static_sections: bool = True,
    stats: Optional[Dict[str, Any]] = None,
    sample_size: int = _ESTIMATION_SAMPLE_SIZE,
    estimated_output_tokens_per_prompt: int = ESTIMATED_OUTPUT_TOKENS_PER_PROMPT,
    extra_input_tokens_per_prompt: int = 0,
    heading: Optional[str] = "OpenAI API usage summary",
    show_prompt_stats: bool = True,
) -> None:
    """Print a summary of usage limits, cost estimate and tier information.

    Optionally takes a pre‑fetched ``rate_headers`` dict to avoid calling
    ``_get_rate_limit_headers`` multiple times per job.  When ``rate_headers``
    is ``None``, the helper will fetch the headers itself.  Optional web-search
    warnings can be provided to display additional caveats alongside the
    usage overview. ``heading`` controls the section title (set to ``None`` to
    skip the header) and ``show_prompt_stats`` suppresses the redundant prompt
    counts when the run banner already printed them.
    """
    if not verbose:
        return
    if heading:
        print(f"\n===== {heading} =====")
    if web_search_warning:
        print(web_search_warning)
    if web_search_parallel_note:
        print(web_search_parallel_note)
    stats = stats or _estimate_dataset_stats(
        prompts,
        sample_size=sample_size,
        extra_input_tokens_per_prompt=extra_input_tokens_per_prompt,
    )
    if show_prompt_stats:
        print(f"Prompts: {len(prompts)}")
        print(f"Approx. input words: {stats.get('word_count', 0):,}")
    # Fetch fresh headers if not supplied.  Pass the model and base_url so the
    # helper knows which endpoint to probe when performing the dummy call.
    rl = rate_headers if rate_headers is not None else _get_rate_limit_headers(model, base_url=base_url)
    # Determine whether the headers include any meaningful limit values.  Some
    # endpoints (or API tiers) may omit rate‑limit headers, or return zero
    # values, which should be treated as unknown.
    def _parse_float(val: Optional[str]) -> Optional[float]:
        try:
            if val is None:
                return None
            # Treat empty strings as None
            s = str(val).strip()
            if not s:
                return None
            f = float(s)
            return f if f > 0 else None
        except Exception:
            return None

    # Parse rate‑limit values from the response headers.  If no headers are
    # returned or a value is zero/negative, leave the variable as None.
    lim_r_val = rem_r_val = None
    lim_t_val = rem_t_val = None
    reset_r = reset_t = None
    if rl:
        lim_r_val = _parse_float(rl.get("limit_requests"))
        rem_r_val = _parse_float(rl.get("remaining_requests"))
        lim_t_val = _parse_float(rl.get("limit_tokens"))
        rem_t_val = _parse_float(rl.get("remaining_tokens"))
        # Some accounts report usage‑based token limits
        if lim_t_val is None:
            lim_t_val = _parse_float(rl.get("limit_tokens_usage_based"))
        if rem_t_val is None:
            rem_t_val = _parse_float(rl.get("remaining_tokens_usage_based"))
        reset_r = rl.get("reset_requests")
        reset_t = rl.get("reset_tokens") or rl.get("reset_tokens_usage_based")
    # Print raw rate limit information without falling back to configured defaults.
    # If a value is unavailable, display "unknown" instead of substituting another number.
    def fmt(val: Optional[float]) -> str:
        return f"{int(val):,}" if val is not None else "unknown"
    concurrency_message_lines: List[str] = []
    concurrency_possible: Optional[int] = None
    concurrency_cap = max(1, n_parallels)
    concurrency_possible_from_requests: Optional[int] = None
    concurrency_possible_from_tokens: Optional[int] = None
    allowed_req_source: Optional[str] = None
    allowed_tok_source: Optional[str] = None
    allowed_req: Optional[float] = None
    allowed_tok: Optional[float] = None
    tokens_per_call: Optional[float] = None
    try:
        token_total = stats.get("token_count") if stats is not None else None
        avg_input_tokens = (token_total or sum(_approx_tokens(p) for p in prompts)) / max(
            1, len(prompts)
        )
        tokens_per_call = _estimate_tokens_per_call(
            avg_input_tokens,
            max_output_tokens,
            n,
            estimated_output_tokens_per_prompt=estimated_output_tokens_per_prompt,
            output_headroom=OUTPUT_TOKEN_HEADROOM_INITIAL,
        )
        def _pf(val: Optional[str]) -> Optional[float]:
            try:
                if val is None:
                    return None
                s = str(val).strip()
                if not s:
                    return None
                f = float(s)
                return f if f > 0 else None
            except Exception:
                return None
        if rl:
            lim_r_val2 = _pf(rl.get("limit_requests"))
            rem_r_val2 = _pf(rl.get("remaining_requests"))
            lim_t_val2 = _pf(rl.get("limit_tokens")) or _pf(rl.get("limit_tokens_usage_based"))
            rem_t_val2 = _pf(rl.get("remaining_tokens")) or _pf(rl.get("remaining_tokens_usage_based"))
            if rem_r_val2 is not None:
                allowed_req = rem_r_val2
                allowed_req_source = "remaining"
            else:
                allowed_req = lim_r_val2
                allowed_req_source = "limit" if lim_r_val2 is not None else None
            if rem_t_val2 is not None:
                allowed_tok = rem_t_val2
                allowed_tok_source = "remaining"
            else:
                allowed_tok = lim_t_val2
                allowed_tok_source = "limit" if lim_t_val2 is not None else None
        if allowed_req is None or allowed_req <= 0:
            concurrency_possible_from_requests = None
        else:
            concurrency_possible_from_requests = int(max(1, allowed_req))
        if allowed_tok is None or allowed_tok <= 0 or tokens_per_call is None:
            concurrency_possible_from_tokens = None
        else:
            concurrency_possible_from_tokens = int(max(1, allowed_tok // tokens_per_call))
        if concurrency_possible_from_requests is None and concurrency_possible_from_tokens is None:
            concurrency_possible = None
        elif concurrency_possible_from_requests is None:
            concurrency_possible = concurrency_possible_from_tokens
        elif concurrency_possible_from_tokens is None:
            concurrency_possible = concurrency_possible_from_requests
        else:
            concurrency_possible = min(concurrency_possible_from_requests, concurrency_possible_from_tokens)
        if concurrency_possible is None:
            concurrency_cap = max(1, n_parallels)
        else:
            concurrency_cap = max(1, min(n_parallels, concurrency_possible))
    except Exception:
        concurrency_message_lines = []
        concurrency_cap = max(1, n_parallels)
        concurrency_possible = None
        tokens_per_call = None
    # Print concise rate limit information.  Only the total per‑minute
    # capacities are shown; remaining quotas and reset timers are omitted
    # to reduce clutter.  Unknown values are labelled as "unknown".
    if lim_r_val is not None:
        print(f"Requests per minute: {fmt(lim_r_val)}")
    else:
        print("Requests per minute: unknown (API did not share a request limit)")
    if lim_t_val is not None:
        print(f"Tokens per minute: {fmt(lim_t_val)}")
        words_per_min = int(lim_t_val) // 2
        print(f"Approx. words per minute: {words_per_min:,}")
    else:
        print("Tokens per minute: unknown (API did not share a token limit)")
        print("Approx. words per minute: unknown")
    if concurrency_possible is not None and concurrency_possible > n_parallels:
        concurrency_message_lines.append(
            f"We are running with {n_parallels:,} parallel requests, but your current limits could allow up to {int(concurrency_possible):,} concurrent requests if desired."
        )
    elif concurrency_cap < n_parallels:
        limiting_messages: List[str] = []
        suggest_upgrade = False
        if (
            concurrency_possible is not None
            and concurrency_possible_from_requests is not None
            and concurrency_possible == concurrency_possible_from_requests
        ):
            if allowed_req_source == "remaining" and allowed_req is not None:
                limiting_messages.append(
                    f"the API reported only {int(allowed_req):,} request slots remaining in the current minute"
                )
            elif allowed_req_source == "limit" and allowed_req is not None:
                limiting_messages.append(
                    f"your per-minute request limit is {int(allowed_req):,}"
                )
                suggest_upgrade = True
        if (
            concurrency_possible is not None
            and concurrency_possible_from_tokens is not None
            and concurrency_possible == concurrency_possible_from_tokens
        ):
            approx_tokens = int(max(1, allowed_tok)) if allowed_tok is not None else None
            if allowed_tok_source == "remaining" and approx_tokens is not None:
                limiting_messages.append(
                    f"about {approx_tokens:,} tokens remain in the current minute"
                )
            elif allowed_tok_source == "limit" and approx_tokens is not None:
                limiting_messages.append(
                    f"your per-minute token limit is about {approx_tokens:,}"
                )
                suggest_upgrade = True
        if not limiting_messages:
            limiting_messages.append("of the reported rate limits")
        reason = " and ".join(limiting_messages)
        concurrency_message_lines.append(
            f"Note: running at most {concurrency_cap:,} concurrent requests (vs. {n_parallels:,} requested) because {reason}."
        )
        if suggest_upgrade:
            concurrency_message_lines.append(
                "Upgrading your tier would allow more parallel requests and speed up processing."
            )
    else:
        concurrency_message_lines.append(
            f"We can run up to {concurrency_cap:,} requests at the same time with your current settings."
        )
    for line in concurrency_message_lines:
        print(line)
    if lim_r_val is None or lim_t_val is None:
        warning_msg = (
            "⚠️ API did not return complete rate-limit headers. Running with conservative defaults. "
            "If you are on a free/low-balance plan, add funds to avoid quota blocks: "
            "https://platform.openai.com/settings/organization/billing/"
        )
        print(warning_msg)
        logger.warning(warning_msg)
    if show_static_sections:
        print("\nUsage tiers (higher tier = faster runs):")
        for tier in TIER_INFO:
            monthly = tier.get("monthly_quota")
            monthly_text = f"; monthly quota {monthly}" if monthly else ""
            print(f"  • {tier['tier']}: qualify by {tier['qualification']}{monthly_text}")
        print("\nAdd funds or manage your billing here: https://platform.openai.com/settings/organization/billing/")
    if max_output_tokens is not None:
        print(
            f"\nmax_output_tokens: {max_output_tokens} (safety cutoff; generation will stop if this is reached)"
        )


def _rate_limit_decrement(concurrency_cap: int) -> int:
    """Return a downward step that curbs rate-limit thrash aggressively."""

    return max(2, int(math.ceil(max(concurrency_cap * 0.4, 3))))


def _connection_error_decrement(concurrency_cap: int) -> int:
    """Return a step-down size for connection errors matching rate-limit aggression."""

    return _rate_limit_decrement(concurrency_cap)


def _smooth_wait_based_cap(
    current_cap: int,
    candidate_cap: int,
    *,
    now: float,
    last_adjust: float,
    limiter_pressure: bool,
    min_delta: int,
    cooldown_up: float,
    cooldown_down: float,
    max_step_up_ratio: float = 0.18,
    max_step_down_ratio: float = 0.08,
) -> Tuple[int, float, bool]:
    """Dampen wait-based cap changes and space them out.

    Returns a tuple of (new_cap, updated_last_adjust, changed?).
    """

    if candidate_cap < 1:
        candidate_cap = 1
    delta = candidate_cap - current_cap
    if abs(delta) < min_delta:
        return current_cap, last_adjust, False
    if delta < 0 and not limiter_pressure:
        return current_cap, last_adjust, False

    cooldown = cooldown_up if delta > 0 else cooldown_down
    if (now - last_adjust) < cooldown:
        return current_cap, last_adjust, False

    step_ratio = max_step_up_ratio if delta > 0 else max_step_down_ratio
    step = max(1, int(math.ceil(current_cap * step_ratio)))
    bounded_delta = max(-step, min(step, delta))
    new_cap = max(1, current_cap + bounded_delta)

    if new_cap == current_cap:
        return current_cap, last_adjust, False
    return new_cap, now, True


def _normalise_web_search_filters(
    filters: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Convert caller-friendly web-search filters to the Responses schema.

    ``filters`` mirrors the keyword arguments exposed by :func:`get_response`
    and the higher-level task wrappers.  Callers can supply an
    ``"allowed_domains"`` iterable together with optional location hints –
    ``city``, ``country``, ``region``, ``timezone`` and ``type`` (currently the
    API accepts ``"approximate"``).  Falsy values are stripped so the outgoing
    payload stays compact.

    The Responses API expects domain restrictions under ``filters`` and
    geography hints under ``user_location``.  This helper reshapes the mapping
    accordingly and ignores unknown keys to avoid forwarding unsupported
    filters.
    """

    if not filters:
        return {}

    result: Dict[str, Any] = {}

    allowed_domains = filters.get("allowed_domains")
    if allowed_domains:
        if isinstance(allowed_domains, (str, bytes)) or not isinstance(allowed_domains, Iterable):
            raise TypeError(
                "web_search_filters['allowed_domains'] must be an iterable of domain strings"
            )
        cleaned = [str(d) for d in allowed_domains if d]
        if cleaned:
            result["filters"] = {"allowed_domains": cleaned}

    location_keys = ("city", "country", "region", "timezone", "type")
    location = {
        key: str(value)
        for key, value in ((k, filters.get(k)) for k in location_keys)
        if value
    }
    if location:
        location.setdefault("type", "approximate")
        result["user_location"] = location

    return result


def _merge_web_search_filters(
    base: Optional[Dict[str, Any]], override: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Combine global and per-prompt web-search filter dictionaries.

    Both inputs follow the caller-facing schema accepted by
    :func:`get_all_responses`.  The override takes precedence, but falsy values
    are skipped so callers can opt out of specific fields.  ``allowed_domains``
    entries are normalised to a list of non-empty strings regardless of whether
    they were supplied as comma-separated text, tuples, or lists.
    """

    if not base and not override:
        return None

    merged: Dict[str, Any] = {}

    def _normalise_allowed(val: Any) -> Optional[List[str]]:
        if not val:
            return None
        if isinstance(val, (str, bytes)):
            candidates = [s.strip() for s in str(val).split(",") if s.strip()]
            return candidates or None
        if isinstance(val, Iterable):
            items = [str(item).strip() for item in val if str(item).strip()]
            return items or None
        return None

    for source in (base or {}, override or {}):
        for key, value in source.items():
            if not value:
                continue
            if key == "allowed_domains":
                normalised = _normalise_allowed(value)
                if normalised:
                    merged[key] = normalised
            else:
                merged[key] = value

    return merged or None


def _normalise_include_values(value: Optional[Union[str, Iterable[str]]]) -> List[str]:
    """Normalise ``include`` into an ordered list of unique strings."""

    if value is None:
        return []
    if isinstance(value, str):
        items: Iterable[str] = [value]
    elif isinstance(value, Iterable):
        items = value
    else:
        return []
    seen: Set[str] = set()
    normalised: List[str] = []
    for item in items:
        if item is None:
            continue
        try:
            text = str(item).strip()
        except Exception:
            continue
        if not text or text in seen:
            continue
        seen.add(text)
        normalised.append(text)
    return normalised


def _normalise_search_context_size(value: Optional[str]) -> str:
    """Clamp web-search context size to the Responses API's allowed values.

    The Responses schema accepts ``"low"``, ``"medium"`` and ``"high"`` for
    ``search_context_size``.  Earlier versions of this library and some
    notebooks used ``"small"``/``"large"``; these aliases are mapped to
    ``"low"`` and ``"high"`` respectively to preserve backwards
    compatibility. Any other value raises ``ValueError`` so misconfigurations
    surface early instead of generating pydantic warnings when serialising the
    request.
    """

    if value is None:
        return "medium"

    try:
        normalised = str(value).strip().lower()
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("search_context_size must be a string") from exc

    aliases = {"small": "low", "medium": "medium", "large": "high", "high": "high"}
    resolved = aliases.get(normalised, normalised)

    if resolved not in {"low", "medium", "high"}:
        raise ValueError(
            "search_context_size must be one of {'low', 'medium', 'high'}"
        )

    # Preserve the canonical casing used by the API
    return resolved


def _extract_web_search_sources(raw_items: List[Any]) -> Optional[List[Any]]:
    """Retrieve any web-search sources returned in ``raw_items``."""

    collected: List[Any] = []
    seen: Set[str] = set()
    visited: Set[int] = set()

    def _record(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, list):
            for entry in value:
                _record(entry)
            return
        if isinstance(value, dict):
            try:
                fingerprint = json.dumps(value, sort_keys=True, ensure_ascii=False)
            except Exception:
                fingerprint = None
            if fingerprint and fingerprint in seen:
                return
            if fingerprint:
                seen.add(fingerprint)
            collected.append(value)
            return
        if isinstance(value, (str, bytes)):
            text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
            text = text.strip()
            if text and text not in seen:
                seen.add(text)
                collected.append(text)

    def _as_maybe_mapping(obj: Any) -> Any:
        if isinstance(obj, BaseModel):
            try:
                converted = obj.model_dump(exclude_none=True, warnings=False)
            except TypeError:
                converted = obj.model_dump(exclude_none=True)
            except Exception:
                converted = None
            if isinstance(converted, (dict, list)):
                return converted
        for attr in ("model_dump", "dict"):
            meth = getattr(obj, attr, None)
            if callable(meth):
                try:
                    converted = meth(exclude_none=True)
                except TypeError:
                    converted = meth()
                if isinstance(converted, (dict, list)):
                    return converted
        data = getattr(obj, "__dict__", None)
        if isinstance(data, dict) and data:
            return data
        return obj

    def _walk(obj: Any) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)
        direct_sources = getattr(obj, "sources", None)
        if direct_sources is not None:
            _record(direct_sources)
        action_obj = getattr(obj, "action", None)
        if action_obj is not None:
            _record(getattr(action_obj, "sources", None))
        if isinstance(obj, (list, tuple, set, frozenset)):
            for item in obj:
                _walk(item)
            return
        obj = _as_maybe_mapping(obj)
        if isinstance(obj, dict):
            if "sources" in obj:
                _record(obj.get("sources"))
            for value in obj.values():
                _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    for item in raw_items:
        _walk(item)
    return collected or None


def _build_params(
    *,
    model: str,
    input_data: List[Dict[str, Any]],
    max_output_tokens: Optional[int],
    temperature: float,
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[dict] = None,
    web_search: bool = False,
    web_search_filters: Optional[Dict[str, Any]] = None,
    search_context_size: str = "medium",
    json_mode: bool = False,
    expected_schema: Optional[Dict[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    include: Optional[Union[str, Iterable[str]]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Compose the keyword arguments for an OpenAI Responses API call.

    The function gathers together the many optional features supported by the
    Responses endpoint – such as tool use, web search, JSON formatting and
    reasoning controls – and emits a plain ``dict`` that mirrors the expected
    JSON payload.  ``None`` values are omitted so the underlying SDK can apply
    its own defaults.  This helper keeps the main :func:`get_response` function
    relatively small and easy to read.

    Parameters
    ----------
    model:
        Identifier of the model to query (e.g. ``"gpt-5-mini"``).
    input_data:
        A list representing the conversation so far.  Each element is a mapping
        with ``role`` and ``content`` keys in the format required by the API.
    max_output_tokens:
        Soft cap on the number of tokens the model may generate.  When ``None``
        the parameter is omitted and the model's server-side default applies.
    temperature:
        Sampling temperature controlling randomness for legacy models that
        honour it.
    tools, tool_choice:
        Optional tool specifications following the Responses API schema.
    web_search:
        When ``True`` a built-in web search tool is appended to the tool list.
    web_search_filters:
        Optional mapping with keys ``allowed_domains`` and/or any of
        ``city``, ``country``, ``region``, ``timezone`` and ``type``.  ``type``
        should match the Responses API expectation (currently ``"approximate"``
        for geographic hints).  Allowed domains are placed under
        ``filters.allowed_domains`` and location hints under ``user_location``
        to match the Responses API schema.  Keys with falsey values are
        ignored.
    search_context_size:
        Size of the search context when ``web_search`` is enabled.  Accepts
        ``"low"``, ``"medium"`` or ``"high"``; legacy aliases
        ``"small"``/``"large"`` are normalised to ``"low"``/``"high"``.
    json_mode:
        If ``True`` the model is asked to produce structured JSON output.
    expected_schema:
        Optional JSON schema supplied when ``json_mode`` is requested.
    reasoning_effort, reasoning_summary:
        ``reasoning_effort`` controls how intensely the model reasons
        (``none``, ``low``, ``medium``, ``high``). Higher values are typically
        smarter but slower. ``reasoning_summary`` requests a concise reasoning
        summary when supported.
    include:
        Optional list (or comma-separated string) of ``include`` fields to
        request from the Responses API. When ``web_search`` is enabled, the
        ``web_search_call.action.sources`` include is added automatically. If
        no ``include`` values were supplied, the helper defaults to requesting
        only ``\"web_search_call.action.sources\"``.
    **extra:
        Any additional key-value pairs to forward directly to the API.

    Returns
    -------
    dict
        Dictionary ready to be expanded into
        :meth:`openai.AsyncOpenAI.responses.create`.
    """
    params: Dict[str, Any] = {
        "model": model,
        "input": input_data,
        "truncation": "auto",
    }
    if max_output_tokens is not None:
        params["max_output_tokens"] = max_output_tokens
    if json_mode:
        params["text"] = (
            {"format": {"type": "json_schema", "schema": expected_schema}}
            if expected_schema
            else {"format": {"type": "json_object"}}
        )
    all_tools = list(tools) if tools else []
    if web_search:
        context_size = _normalise_search_context_size(search_context_size)
        tool: Dict[str, Any] = {"type": "web_search", "search_context_size": context_size}
        filters = _normalise_web_search_filters(web_search_filters)
        if filters:
            domains = filters.get("filters")
            if domains:
                tool["filters"] = domains
            user_location = filters.get("user_location")
            if user_location:
                tool["user_location"] = user_location
        all_tools.append(tool)
    if all_tools:
        params["tools"] = all_tools
    if tool_choice is not None:
        params["tool_choice"] = tool_choice
    if _uses_legacy_system_instruction(model):
        params["temperature"] = temperature
    else:
        reasoning: Dict[str, Any] = {}
        if reasoning_effort is not None:
            reasoning["effort"] = reasoning_effort
        if reasoning_summary is not None:
            reasoning["summary"] = reasoning_summary
        if reasoning:
            params["reasoning"] = reasoning
        if temperature != 0.9:
            logger.warning(
                f"Model {model} does not support temperature; ignoring provided value."
            )
    include_values = _normalise_include_values(include)
    extra_include = _normalise_include_values(extra.pop("include", None))
    include_values.extend([val for val in extra_include if val not in include_values])
    include_explicitly_requested = bool(include_values or include is not None or extra_include)
    if web_search:
        if include_explicitly_requested:
            if "web_search_call.action.sources" not in include_values:
                include_values.append("web_search_call.action.sources")
        else:
            include_values = ["web_search_call.action.sources"]
    if include_values:
        params["include"] = include_values
    params.update(extra)
    return params


async def get_response(
    prompt: str,
    *,
    model: str = "gpt-5-mini",
    n: int = 1,
    max_output_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    temperature: float = 0.9,
    json_mode: bool = False,
    expected_schema: Optional[Dict[str, Any]] = None,
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[dict] = None,
    web_search: bool = False,
    web_search_filters: Optional[Dict[str, Any]] = None,
    search_context_size: str = "medium",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    include: Optional[Union[str, Iterable[str]]] = None,
    use_dummy: bool = False,
    base_url: Optional[str] = None,
    verbose: bool = True,
    images: Optional[List[str]] = None,
    audio: Optional[List[Dict[str, str]]] = None,
    pdfs: Optional[List[Dict[str, str]]] = None,
    return_raw: bool = False,
    logging_level: Optional[Union[str, int]] = None,
    background_mode: Optional[bool] = None,
    background_poll_interval: float = 10.0,
    **kwargs: Any,
):
    """Request one or more model completions from the OpenAI API.

    This coroutine is the main entry point for sending a single prompt to an
    OpenAI model.  It supports text-only prompts as well as prompts that include
    images or audio.  When ``use_dummy`` is ``True`` no network requests are
    made; instead a predictable placeholder response is returned, which is
    useful for tests.

    Internally the function prepares a parameter dictionary via
    :func:`_build_params` and dispatches the request using the asynchronous
    OpenAI SDK.  Audio inputs are routed through the chat-completions API,
    whereas all other requests use the newer Responses API.  Multiple
    completions can be retrieved in parallel by setting ``n`` greater than one.

    Parameters
    ----------
    prompt:
        The user question or instruction to send to the model.
    model:
        Name of the model to query.
    n:
        Number of completions to request.  Each completion is retrieved in
        parallel.
    max_output_tokens:
        Optional cap on the length of each completion in tokens.
    timeout:
        Maximum time in seconds to wait for the API to respond.  ``None``
        disables client-side timeouts.
    temperature:
        Randomness control for legacy models that accept it.
    json_mode, expected_schema:
        When ``json_mode`` is ``True`` the model is instructed to output JSON.
        ``expected_schema`` may provide a JSON schema to validate against.
    tools, tool_choice:
        Optional tool specifications to pass through to the API.
    web_search, search_context_size:
        Enable and configure the built-in web-search tool.
    web_search_filters:
        Optional mapping with ``allowed_domains`` and/or user location hints
        (``city``, ``country``, ``region``, ``timezone`` and ``type`` – typically
        ``"approximate"``) to guide search results when ``web_search`` is
        enabled.
    include:
        Optional ``include`` values forwarded to the Responses API. When
        ``web_search`` is enabled, ``web_search_call.action.sources`` is added
        automatically (without duplicating caller-provided values) so sources
        are available in the returned payloads.
    reasoning_effort, reasoning_summary:
        Additional reasoning controls for modern models.
    use_dummy:
        If ``True`` return deterministic dummy responses instead of calling the
        external API.
    base_url:
        Optional custom OpenAI-compatible endpoint. If omitted, the default
        ``api.openai.com/v1`` or ``OPENAI_BASE_URL`` environment variable is
        used.
    verbose:
        When set, progress information is printed via the module logger.
    images, audio, pdfs:
        Lists of base64-encoded media to include alongside the text prompt.
    return_raw:
        If ``True`` the raw SDK response objects are returned alongside the
        extracted text and timing information.
    logging_level:
        Optional override for the module's log level.
    background_mode:
        When ``True`` the helper submits the request in background mode and
        polls :meth:`openai.AsyncOpenAI.responses.retrieve` until completion.
        When ``None`` (default) the helper automatically enables background
        mode whenever ``timeout`` is ``None`` so long-running calls are resilient
        to transient HTTP disconnects.  Set to ``False`` to force the legacy
        behaviour of waiting on the initial HTTP response.
    background_poll_interval:
        How frequently (in seconds) to poll for background completion when
        background mode is active. Defaults to 10 seconds and automatically
        lengthens when rate-limit responses instruct a longer pause.
    **kwargs:
        Any additional parameters understood by the OpenAI SDK are forwarded
        transparently.

    Returns
    -------
    tuple
        ``([text, ...], duration)`` when ``return_raw`` is ``False``.  If
        ``return_raw`` is ``True`` the third element contains the raw response
        objects from the SDK.
    """
    if web_search_filters and not web_search:
        logger.debug(
            "web_search_filters were supplied but web_search is disabled; ignoring filters."
        )
    if web_search and json_mode:
        logger.warning(
            "Web search cannot be combined with JSON mode; disabling JSON mode."
        )
        json_mode = False
    # Use dummy for testing without calling the API
    if use_dummy:
        dummy = [f"DUMMY {prompt}" for _ in range(max(n, 1))]
        if return_raw:
            return dummy, 0.0, []
        return dummy, 0.0
    if logging_level is not None:
        set_log_level(logging_level)
    _require_api_key()
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    client_async = _get_client(base_url)

    try:
        poll_interval = float(background_poll_interval)
    except (TypeError, ValueError):
        poll_interval = 10.0
    if poll_interval <= 0:
        poll_interval = 10.0

    explicit_background = kwargs.pop("background", None)
    if explicit_background is not None:
        effective_background = bool(explicit_background)
    elif background_mode is not None:
        effective_background = bool(background_mode)
    else:
        effective_background = False
    background_argument: Optional[bool] = None
    if explicit_background is not None:
        background_argument = bool(explicit_background)
    elif effective_background:
        background_argument = True

    failure_statuses = {"failed", "cancelled", "expired"}

    def _background_error_message(resp: Any) -> str:
        err = _safe_get(resp, "error")
        if isinstance(err, dict):
            for key in ("message", "code", "type"):
                val = err.get(key)
                if isinstance(val, str) and val.strip():
                    return val
            try:
                return json.dumps(err, ensure_ascii=False)
            except Exception:
                return str(err)
        if err:
            return str(err)
        status = _safe_get(resp, "status")
        identifier = _safe_get(resp, "id")
        return f"Response {identifier or '<unknown>'} failed with status {status}."

    async def _await_background_completion(
        response_obj: Any,
        start_time: float,
        *,
        poll: Optional[float] = None,
        client: Optional[openai.AsyncOpenAI] = None,
        should_poll: bool = False,
    ) -> Any:
        if not should_poll:
            return response_obj
        status = _safe_get(response_obj, "status")
        if status in (None, "completed"):
            return response_obj
        response_id = _safe_get(response_obj, "id")
        if not response_id:
            return response_obj
        poll_every = poll if poll is not None else poll_interval
        poll_every = max(0.5, float(poll_every))
        local_client = client or client_async
        last = response_obj
        consecutive_errors = 0
        while True:
            status = _safe_get(last, "status")
            if status == "completed":
                return last
            if status in failure_statuses or status == "requires_action":
                message = _background_error_message(last)
                raise APIError(message)
            if status not in {"queued", "in_progress", "cancelling"}:
                return last
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    raise BackgroundTimeoutError(
                        response_id,
                        last,
                        f"Background response {response_id} exceeded timeout of {timeout} s",
                    )
                sleep_for = min(poll_every, max(0.1, remaining))
            else:
                sleep_for = poll_every
            await asyncio.sleep(sleep_for)
            retrieve_kwargs: Dict[str, Any] = {}
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    raise BackgroundTimeoutError(
                        response_id,
                        last,
                        f"Background response {response_id} exceeded timeout of {timeout} s",
                    )
                retrieve_kwargs["timeout"] = max(1.0, min(30.0, remaining))
            try:
                last = await local_client.responses.retrieve(response_id, **retrieve_kwargs)
                consecutive_errors = 0
            except asyncio.CancelledError:
                raise
            except RateLimitError:
                logger.warning(
                    "[get_response] Polling %s hit rate limit; aborting background checks.",
                    response_id,
                )
                raise
            except Exception as exc:
                consecutive_errors += 1
                logger.warning(
                    "[get_response] Polling %s failed on attempt %d: %r",
                    response_id,
                    consecutive_errors,
                    exc,
                )
                if timeout is not None and (time.time() - start_time) >= timeout:
                    raise BackgroundTimeoutError(
                        response_id,
                        last,
                        f"Background response {response_id} exceeded timeout of {timeout} s",
                    )
                if consecutive_errors >= 5:
                    raise
                continue
    # Derive the effective cutoff
    cutoff = max_output_tokens
    system_instruction = DEFAULT_SYSTEM_INSTRUCTION
    legacy_system_instruction = _uses_legacy_system_instruction(model)
    if audio:
        audio_model = _is_audio_model(model)
        if not audio_model:
            logger.warning(
                "Audio inputs detected but model '%s' does not include 'audio' in its name; the API may reject the request.",
                model,
            )
        contents: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if pdfs:
            logger.warning("PDF inputs are ignored for audio-only requests.")
        if images:
            for img in images:
                img_url = (
                    img if str(img).startswith("data:") else f"data:image/jpeg;base64,{img}"
                )
                contents.append(
                    {"type": "input_image", "image_url": {"url": img_url}}
                )
        for a in audio:
            contents.append({"type": "input_audio", "input_audio": a})
        messages = [{"role": "user", "content": contents}]
        # ``chat.completions`` infers the output modality from the request
        # content.  Audio-capable models may require explicitly requesting
        # text output via ``modalities`` so we default to text when possible.
        params_chat: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if audio_model:
            params_chat.setdefault("modalities", ["text"])
        if tools is not None:
            params_chat["tools"] = tools
        if tool_choice is not None:
            params_chat["tool_choice"] = tool_choice
        if cutoff is not None:
            params_chat["max_completion_tokens"] = cutoff
        params_chat.update(kwargs)
        start = time.time()
        tasks = [
            asyncio.create_task(
                client_async.chat.completions.create(
                    **params_chat, **({"timeout": timeout} if timeout is not None else {})
                )
            )
            for _ in range(max(n, 1))
        ]
        try:
            raw = await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise
        except asyncio.TimeoutError as exc:
            message = (
                f"API call timed out after {timeout} s"
                if timeout is not None
                else "API call timed out"
            )
            logger.error(f"[get_response] {message}")
            raise asyncio.TimeoutError(message) from exc
        except Exception as e:
            logger.error(
                "[get_response] API call resulted in exception: %r", e, exc_info=True
            )
            raise
        texts = []
        for r in raw:
            msg = r.choices[0].message
            parts = getattr(msg, "content", None)
            if isinstance(parts, list):
                texts.append(
                    "".join(p.get("text", "") for p in parts if p.get("type") == "text")
                )
            else:
                texts.append(parts)
        duration = time.time() - start
        if return_raw:
            return texts, duration, raw
        return texts, duration
    else:
        if images or pdfs:
            contents: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
            if images:
                for img in images:
                    img_url = (
                        img if str(img).startswith("data:") else f"data:image/jpeg;base64,{img}"
                    )
                    contents.append(
                        {"type": "input_image", "image_url": img_url}
                    )
            if pdfs:
                for pdf in pdfs:
                    file_data = pdf.get("file_data")
                    file_url = pdf.get("file_url")
                    filename = pdf.get("filename")
                    if file_data and not str(file_data).startswith("data:"):
                        file_data = f"data:application/pdf;base64,{file_data}"
                    file_payload: Dict[str, Any] = {"type": "input_file"}
                    if filename:
                        file_payload["filename"] = filename
                    if file_data:
                        file_payload["file_data"] = file_data
                    elif file_url:
                        file_payload["file_url"] = file_url
                    else:
                        continue
                    contents.append(file_payload)
            input_data = (
                [{"role": "user", "content": contents}]
                if not legacy_system_instruction
                else [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": contents},
                ]
            )
        else:
            input_data = (
                [{"role": "user", "content": prompt}]
                if not legacy_system_instruction
                else [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ]
            )

        params = _build_params(
            model=model,
            input_data=input_data,
            max_output_tokens=cutoff,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            web_search=web_search,
            web_search_filters=web_search_filters,
            search_context_size=search_context_size,
            json_mode=json_mode,
            expected_schema=expected_schema,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            include=include,
            **kwargs,
        )
        if background_argument is not None:
            params["background"] = background_argument
        total_needed = max(n, 1)
        start = time.time()
        raw_new: List[Any] = []
        new_tasks: List[asyncio.Task] = [
            asyncio.create_task(
                client_async.responses.create(
                    **params, **({"timeout": timeout} if timeout is not None else {})
                )
            )
            for _ in range(total_needed)
        ]
        try:
            raw_new = await asyncio.gather(*new_tasks)
        except asyncio.CancelledError:
            for t in new_tasks:
                t.cancel()
            raise
        except asyncio.TimeoutError as exc:
            message = (
                f"API call timed out after {timeout} s"
                if timeout is not None
                else "API call timed out"
            )
            logger.error(f"[get_response] {message}")
            raise asyncio.TimeoutError(message) from exc
        except Exception as e:
            logger.error(
                "[get_response] API call resulted in exception: %r", e, exc_info=True
            )
            raise
        def _should_poll_response(resp: Any) -> bool:
            status = _safe_get(resp, "status")
            if status in failure_statuses or status == "requires_action":
                message = _background_error_message(resp)
                raise APIError(message)
            return status not in (None, "completed")

        completed_raw: List[Any] = []
        watcher_tasks: List[asyncio.Task] = []
        completion_durations: List[float] = []

        for response_obj in raw_new:
            needs_poll = _should_poll_response(response_obj)
            if needs_poll:
                watcher_tasks.append(
                    asyncio.create_task(
                        _await_background_completion(
                            response_obj,
                            start,
                            poll=poll_interval,
                            client=client_async,
                            should_poll=True,
                        )
                    )
                )
            else:
                completed_raw.append(response_obj)

        poll_error: Optional[BaseException] = None
        if watcher_tasks:
            try:
                for task in asyncio.as_completed(watcher_tasks):
                    try:
                        result_obj = await task
                        completion_durations.append(max(0.0, time.time() - start))
                        completed_raw.append(result_obj)
                        if len(completed_raw) >= total_needed:
                            break
                    except (BackgroundTimeoutError, RateLimitError, APIError) as exc:
                        poll_error = exc
                        break
                    except Exception as exc:
                        poll_error = exc
                        break
            finally:
                for task in watcher_tasks:
                    if task.done():
                        try:
                            task.result()
                        except Exception:
                            pass
                        continue
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, BackgroundTimeoutError, RateLimitError, APIError):
                        pass
                    except Exception:
                        pass
        if poll_error is not None:
            raise poll_error
        if len(completed_raw) < total_needed:
            raise asyncio.TimeoutError("Background responses did not complete")
        raw = completed_raw[:total_needed]
        # Extract ``output_text`` from the responses.  For Responses API
        # the SDK returns an object with an ``output_text`` attribute.
        texts = [r.output_text for r in raw]
        duration = time.time() - start
        if completion_durations:
            duration = max(duration, max(completion_durations))
        if return_raw:
            return texts, duration, raw
        return texts, duration


def _ser(x: Any) -> Optional[str]:
    """Serialize Python objects deterministically."""
    return None if x is None else json.dumps(x, ensure_ascii=False)


def _de(x: Any) -> Any:
    """Deserialize JSON strings back to Python objects."""
    if pd.isna(x):
        return None
    parsed = safe_json(x)
    return parsed if parsed else None


def response_to_text(value: Any) -> str:
    """Coerce a Response payload into plain text.

    The OpenAI Responses API frequently wraps the textual output in one or
    more layers of lists and dictionaries.  This helper mirrors the extraction
    heuristics used when the responses are first collected, but in a reusable
    form so downstream tasks can reliably access the human-readable text.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        for item in value:
            text = response_to_text(item)
            if text:
                return text
        return ""

    if isinstance(value, dict):
        for key in ("text", "content", "output_text"):
            if key in value:
                text = response_to_text(value.get(key))
                if text:
                    return text
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


async def get_embedding(
    text: str,
    *,
    model: str = "text-embedding-3-small",
    timeout: Optional[float] = None,
    use_dummy: bool = False,
    base_url: Optional[str] = None,
    return_raw: bool = False,
    logging_level: Optional[Union[str, int]] = None,
    **kwargs: Any,
) -> Tuple[List[float], float]:
    """Retrieve a numeric embedding vector for ``text``.

    The OpenAI embedding endpoint converts a piece of text into a list of
    floating‑point numbers that capture semantic meaning.  This helper wraps
    that API in a small asynchronous function and returns both the embedding and
    the time taken to obtain it.  When ``use_dummy`` is ``True`` a synthetic
    embedding is produced instead of contacting the network – handy for unit
    tests or offline experimentation.

    Parameters
    ----------
    text:
        The string to embed.
    model:
        Which embedding model to use.  Defaults to ``"text-embedding-3-small"``.
    timeout:
        Optional request timeout in seconds.  ``None`` waits indefinitely.
    use_dummy:
        Return a short deterministic vector instead of calling the API.
    base_url:
        Optional custom endpoint for the OpenAI-compatible API.
    return_raw:
        When ``True`` the raw SDK response object is also returned.
    logging_level:
        Optional log level override for this call.
    **kwargs:
        Additional parameters forwarded to
        :meth:`openai.AsyncOpenAI.embeddings.create`.

    Returns
    -------
    tuple
        ``(embedding, duration)`` or ``(embedding, duration, raw)`` when
        ``return_raw`` is ``True``.
    """

    if use_dummy:
        dummy = [float(len(text))]
        return (dummy, 0.0, {}) if return_raw else (dummy, 0.0)

    if logging_level is not None:
        set_log_level(logging_level)
    _require_api_key()

    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    client_async = _get_client(base_url)

    start = time.time()
    try:
        raw = await client_async.embeddings.create(
            model=model,
            input=text,
            **({"timeout": timeout} if timeout is not None else {}),
            **kwargs,
        )
    except asyncio.TimeoutError as exc:
        message = (
            f"API call timed out after {timeout} s"
            if timeout is not None
            else "API call timed out"
        )
        logger.error(f"[get_embedding] {message}")
        raise asyncio.TimeoutError(message) from exc
    except APITimeoutError as e:
        logger.error(
            "[get_embedding] API call resulted in client timeout: %r", e, exc_info=True
        )
        raise
    except Exception as e:
        logger.error(
            "[get_embedding] API call resulted in exception: %r", e, exc_info=True
        )
        raise

    embed = raw.data[0].embedding
    duration = time.time() - start
    if return_raw:
        return embed, duration, raw
    return embed, duration


async def get_all_embeddings(
    texts: List[str],
    identifiers: Optional[List[str]] = None,
    *,
    model: str = "text-embedding-3-small",
    save_path: str = "embeddings.pkl",
    reset_file: bool = False,
    n_parallels: int = 150,
    timeout: float = 30.0,
    save_every_x: int = 5000,
    use_dummy: bool = False,
    dummy_embeddings: Optional[Dict[str, List[float]]] = None,
    base_url: Optional[str] = None,
    verbose: bool = True,
    logging_level: Union[str, int] = "warning",
    max_retries: int = 3,
    global_cooldown: int = 15,
    **get_embedding_kwargs: Any,
) -> Dict[str, List[float]]:
    """Compute embeddings for many pieces of text and persist the results.

    The function accepts a list of input strings and queries the OpenAI
    embedding API concurrently.  Progress is periodically written to
    ``save_path`` so long‑running jobs can be resumed.  The routine adapts the
    number of parallel workers based on observed successes and handles common
    failure modes such as timeouts or rate‑limit errors by retrying with an
    exponential backoff.

    Parameters
    ----------
    texts:
        Iterable of strings to embed.
    identifiers:
        Optional identifiers corresponding to ``texts``; defaults to using the
        text itself.  These keys are used when saving and resuming work.
    model:
        Embedding model name.
    save_path:
        File path of a pickle used to store intermediate and final results.
    reset_file:
        When ``True`` any existing ``save_path`` is ignored and overwritten.
    n_parallels:
        Upper bound on the number of concurrent API calls.
    timeout:
        Per‑request timeout in seconds.
    save_every_x:
        Frequency (in processed texts) at which the pickle file is updated.
    use_dummy:
        Generate fake embeddings instead of calling the API.
    dummy_embeddings:
        Optional mapping from identifiers (or ``"*"`` for a fallback) to
        deterministic vectors used when ``use_dummy`` is ``True``.  Supplying
        this allows tests to control the synthetic embeddings instead of
        relying on the default ``[len(text)]`` stub.
    base_url:
        Optional custom OpenAI-compatible endpoint used for requests.
    verbose:
        If ``True`` a progress bar is displayed.
    logging_level:
        Logging verbosity for this helper.
    max_retries:
        Number of times to retry a failed request before giving up.
    global_cooldown:
        Seconds to pause new work after encountering a rate‑limit error.
    **get_embedding_kwargs:
        Additional keyword arguments passed to :func:`get_embedding`.

    Returns
    -------
    dict
        Mapping from identifier to embedding vector.
    """

    if not use_dummy:
        _require_api_key()
    set_log_level(logging_level)
    logger = get_logger(__name__)
    base_url = base_url or os.getenv("OPENAI_BASE_URL")

    if identifiers is None:
        identifiers = texts
    dummy_embeddings_map: Dict[str, List[float]] = {}
    dummy_embedding_default: Optional[List[float]] = None
    if dummy_embeddings:
        for key, value in dummy_embeddings.items():
            if value is None:
                continue
            vector = [float(v) for v in value]
            dummy_embeddings_map[str(key)] = vector
        if "*" in dummy_embeddings_map:
            dummy_embedding_default = list(dummy_embeddings_map["*"])
        elif "__default__" in dummy_embeddings_map:
            dummy_embedding_default = list(dummy_embeddings_map["__default__"])

    save_path = os.path.expanduser(os.path.expandvars(save_path))
    embeddings: Dict[str, List[float]] = {}
    if not reset_file and os.path.exists(save_path):
        try:
            with open(save_path, "rb") as f:
                embeddings = pickle.load(f)
            print(
                f"[get_all_embeddings] Loaded {len(embeddings)} existing embeddings from {save_path}"
            )
        except Exception:
            embeddings = {}

    if len(texts) > 50_000:
        msg = (
            "[get_all_embeddings] Warning: more than 50k texts supplied; the"
            " resulting embeddings file may be very large."
        )
        print(msg)
        logger.warning(msg)

    items = [
        (i, t) for i, t in zip(identifiers, texts) if i not in embeddings
    ]
    if not items:
        print(
            f"[get_all_embeddings] Using cached embeddings from {save_path}; no new texts to process"
        )
        return embeddings

    tokenizer = _get_tokenizer(model)
    get_embedding_kwargs.setdefault("base_url", base_url)
    error_logs: Dict[str, List[str]] = defaultdict(list)
    queue: asyncio.Queue[Tuple[str, str, int]] = asyncio.Queue()
    first_timeout_logged = False
    first_rate_limit_logged = False
    first_connection_logged = False
    for item in items:
        queue.put_nowait((item[1], item[0], max_retries))

    processed = 0
    pbar = _progress_bar(
        total=len(items),
        desc="Getting embeddings",
        leave=True,
        verbose=verbose,
    )
    cooldown_until = 0.0
    active_workers = 0
    concurrency_cap = max(1, min(n_parallels, queue.qsize()))
    print(f"[init] Starting with {concurrency_cap} parallel workers")
    logger.info(f"[init] Starting with {concurrency_cap} parallel workers")
    rate_limit_errors_since_adjust = 0
    successes_since_adjust = 0

    def maybe_adjust_concurrency() -> None:
        nonlocal concurrency_cap, rate_limit_errors_since_adjust, successes_since_adjust
        total_events = rate_limit_errors_since_adjust + successes_since_adjust
        if rate_limit_errors_since_adjust > 0:
            min_samples = max(20, int(math.ceil(concurrency_cap * 0.3)))
            if total_events >= min_samples:
                error_ratio = rate_limit_errors_since_adjust / max(1, total_events)
                if error_ratio >= 0.25 or rate_limit_errors_since_adjust >= max(8, int(math.ceil(concurrency_cap * 0.2))):
                    decrement = max(1, int(math.ceil(max(concurrency_cap * 0.15, 1))))
                    new_cap = max(1, concurrency_cap - decrement)
                    if new_cap != concurrency_cap:
                        msg = (
                            f"[scale down] Reducing parallel workers from {concurrency_cap} to {new_cap} due to repeated rate limit errors."
                        )
                        print(msg)
                        logger.warning(msg)
                    concurrency_cap = new_cap
                    rate_limit_errors_since_adjust = 0
                    successes_since_adjust = 0
                    return
        if rate_limit_errors_since_adjust == 0 and concurrency_cap < n_parallels:
            success_threshold = max(15, int(math.ceil(concurrency_cap * 0.75)))
            if successes_since_adjust >= success_threshold:
                increment = max(1, int(math.ceil(max(concurrency_cap * 0.25, 1))))
                new_cap = min(n_parallels, concurrency_cap + increment)
                if new_cap != concurrency_cap:
                    msg = (
                        f"[scale up] Increasing parallel workers from {concurrency_cap} to {new_cap} after sustained success."
                    )
                    print(msg)
                    logger.info(msg)
                concurrency_cap = new_cap
                successes_since_adjust = 0
                rate_limit_errors_since_adjust = 0

    def _log_embedding_timeout_once(message: str) -> None:
        nonlocal first_timeout_logged
        if not first_timeout_logged:
            logger.warning(
                "Encountered first timeout error. Future timeout errors will be silenced."
            )
            first_timeout_logged = True
        else:
            logger.debug("Timeout error: %s", message)

    def _log_embedding_rate_limit_once(detail: Optional[str] = None) -> None:
        nonlocal first_rate_limit_logged
        if not first_rate_limit_logged:
            logger.warning(
                "Encountered first rate limit error. Future rate limit errors will be silenced."
            )
            first_rate_limit_logged = True
        else:
            if detail:
                logger.debug("Rate limit error: %s", detail)
            else:
                logger.debug("Rate limit error encountered.")

    def _log_embedding_connection_once(detail: Optional[str] = None) -> None:
        nonlocal first_connection_logged
        if not first_connection_logged:
            logger.warning(
                "Encountered first connection error. Future connection errors will be silenced."
            )
            first_connection_logged = True
        else:
            if detail:
                logger.debug("Connection error: %s", detail)
            else:
                logger.debug("Connection error encountered.")

    async def worker() -> None:
        nonlocal processed, cooldown_until, active_workers, concurrency_cap
        nonlocal rate_limit_errors_since_adjust, successes_since_adjust
        while True:
            try:
                text, ident, attempts_left = await queue.get()
            except asyncio.CancelledError:
                break
            try:
                now = time.time()
                if now < cooldown_until:
                    await asyncio.sleep(cooldown_until - now)
                while active_workers >= concurrency_cap:
                    await asyncio.sleep(0.01)
                active_workers += 1
                error_logs.setdefault(ident, [])
                call_timeout = timeout
                start = time.time()
                override_embedding: Optional[List[float]] = None
                if use_dummy and (dummy_embeddings_map or dummy_embedding_default):
                    override_embedding = dummy_embeddings_map.get(str(ident))
                    if override_embedding is None:
                        override_embedding = dummy_embedding_default
                    if override_embedding is not None:
                        embeddings[ident] = list(override_embedding)
                        processed += 1
                        successes_since_adjust += 1
                        rate_limit_errors_since_adjust = 0
                        maybe_adjust_concurrency()
                        if processed % save_every_x == 0:
                            with open(save_path, "wb") as f:
                                pickle.dump(embeddings, f)
                        pbar.update(1)
                        continue
                task = asyncio.create_task(
                    get_embedding(
                        text,
                        model=model,
                        timeout=call_timeout,
                        use_dummy=use_dummy,
                        **get_embedding_kwargs,
                    )
                )
                emb, _ = await task
                embeddings[ident] = emb
                processed += 1
                successes_since_adjust += 1
                rate_limit_errors_since_adjust = 0
                maybe_adjust_concurrency()
                if processed % save_every_x == 0:
                    with open(save_path, "wb") as f:
                        pickle.dump(embeddings, f)
                pbar.update(1)
            except (asyncio.TimeoutError, APITimeoutError) as e:
                elapsed = time.time() - start
                if isinstance(e, APITimeoutError):
                    error_message = (
                        f"OpenAI client timed out after {elapsed:.2f} s; "
                        "consider reducing concurrency."
                    )
                    detail = str(e)
                    if detail:
                        error_logs[ident].append(detail)
                else:
                    error_message = f"API call timed out after {elapsed:.2f} s"
                error_logs[ident].append(error_message)
                _log_embedding_timeout_once(error_message)
                if attempts_left - 1 > 0:
                    backoff = random.uniform(1, 2) * (2 ** (max_retries - attempts_left))
                    await asyncio.sleep(backoff)
                    queue.put_nowait((text, ident, attempts_left - 1))
                else:
                    logger.error(f"[get_all_embeddings] {ident} failed: {e}")
                    processed += 1
                    pbar.update(1)
                    if processed % save_every_x == 0:
                        with open(save_path, "wb") as f:
                            pickle.dump(embeddings, f)
            except RateLimitError as e:
                error_logs[ident].append(str(e))
                _log_embedding_rate_limit_once(str(e))
                cooldown_until = time.time() + global_cooldown
                rate_limit_errors_since_adjust += 1
                successes_since_adjust = 0
                maybe_adjust_concurrency()
                if attempts_left - 1 > 0:
                    backoff = random.uniform(1, 2) * (2 ** (max_retries - attempts_left))
                    await asyncio.sleep(backoff)
                    queue.put_nowait((text, ident, attempts_left - 1))
                else:
                    logger.error(f"[get_all_embeddings] {ident} failed: {e}")
                    processed += 1
                    pbar.update(1)
                    if processed % save_every_x == 0:
                        with open(save_path, "wb") as f:
                            pickle.dump(embeddings, f)
            except APIConnectionError as e:
                error_logs[ident].append(str(e))
                _log_embedding_connection_once(str(e))
                if attempts_left - 1 > 0:
                    backoff = random.uniform(1, 2) * (2 ** (max_retries - attempts_left))
                    await asyncio.sleep(backoff)
                    queue.put_nowait((text, ident, attempts_left - 1))
                else:
                    logger.error(f"[get_all_embeddings] {ident} failed: {e}")
                    processed += 1
                    pbar.update(1)
                    if processed % save_every_x == 0:
                        with open(save_path, "wb") as f:
                            pickle.dump(embeddings, f)
            except (
                APIError,
                BadRequestError,
                AuthenticationError,
                InvalidRequestError,
            ) as e:
                error_logs[ident].append(str(e))
                logger.warning(f"API error for {ident}: {e}")
                processed += 1
                pbar.update(1)
                if processed % save_every_x == 0:
                    with open(save_path, "wb") as f:
                        pickle.dump(embeddings, f)
            except Exception as e:
                error_logs[ident].append(str(e))
                logger.error(f"Unexpected error for {ident}: {e}")
                raise
            finally:
                active_workers -= 1
                queue.task_done()

    workers = [
        asyncio.create_task(worker())
        for _ in range(max(1, min(n_parallels, queue.qsize())))
    ]
    try:
        await queue.join()
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Cancellation requested, shutting down workers...")
        raise
    finally:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        pbar.close()
        with open(save_path, "wb") as f:
            pickle.dump(embeddings, f)

    return embeddings


def _coerce_to_list(value: Any) -> List[Any]:
    """Return ``value`` as a list while preserving common container types."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    """Retrieve ``attr`` from ``obj`` supporting dicts and objects uniformly."""

    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _resolve_effective_timeout(
    nonlocal_timeout: float, task_timeout: float, dynamic_timeout: bool
) -> float:
    """Return the timeout that should apply when evaluating in-flight tasks."""

    if not dynamic_timeout:
        return task_timeout
    if math.isinf(task_timeout):
        return nonlocal_timeout
    return task_timeout


def _should_cancel_inflight_task(
    start_time: float,
    now: float,
    nonlocal_timeout: float,
    task_timeout: float,
    dynamic_timeout: bool,
) -> bool:
    """Determine whether an in-flight task should be cancelled for timeout."""

    limit = _resolve_effective_timeout(nonlocal_timeout, task_timeout, dynamic_timeout)
    if math.isinf(limit):
        return False
    return now - start_time > limit


def _normalize_response_result(result: Any) -> Tuple[List[Any], Optional[float], List[Any]]:
    """Normalize outputs from ``response_fn`` into ``(responses, duration, raw)``."""

    responses_obj: Any = None
    duration: Optional[float] = None
    raw_obj: Any = []
    if isinstance(result, dict):
        responses_obj = result.get("responses") or result.get("response")
        duration = result.get("duration")
        raw_obj = result.get("raw", result.get("raw_responses", []))
    elif isinstance(result, tuple):
        seq = list(result)
        responses_obj = seq[0] if seq else None
        if len(seq) >= 2:
            candidate = seq[1]
            if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                duration = float(candidate)
                tail = seq[2:]
                if len(tail) == 1:
                    raw_obj = tail[0]
                elif tail:
                    raw_obj = tail
            elif candidate is None:
                duration = None
                tail = seq[2:]
                if len(tail) == 1:
                    raw_obj = tail[0]
                elif tail:
                    raw_obj = tail
            else:
                raw_obj = seq[1:]
    else:
        responses_obj = result
    if responses_obj is None:
        responses_obj = [] if isinstance(result, tuple) else result
    responses_list = _coerce_to_list(responses_obj)
    raw_list = _coerce_to_list(raw_obj)
    if duration is not None:
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            duration = None
    if not raw_list or (len(raw_list) == 1 and raw_list[0] is None):
        raw_list = []
    return responses_list, duration, raw_list


def _coerce_dummy_response_spec(
    value: Optional[Union[DummyResponseSpec, Dict[str, Any]]]
) -> Optional[DummyResponseSpec]:
    """Return ``value`` as a :class:`DummyResponseSpec` instance when possible."""

    if value is None:
        return None
    if isinstance(value, DummyResponseSpec):
        return value
    if isinstance(value, dict):
        allowed = {field.name for field in fields(DummyResponseSpec)}
        filtered = {k: v for k, v in value.items() if k in allowed}
        return DummyResponseSpec(**filtered)
    raise TypeError(
        "dummy_responses values must be DummyResponseSpec instances or dictionaries"
    )


def _merge_dummy_specs(
    primary: Optional[DummyResponseSpec], fallback: Optional[DummyResponseSpec]
) -> Optional[DummyResponseSpec]:
    """Combine ``primary`` with ``fallback`` preferring explicit ``primary`` values."""

    if primary is None:
        return fallback
    if fallback is None:
        return primary
    return DummyResponseSpec(
        responses=primary.responses
        if primary.responses is not None
        else fallback.responses,
        duration=primary.duration
        if primary.duration is not None
        else fallback.duration,
        input_tokens=primary.input_tokens
        if primary.input_tokens is not None
        else fallback.input_tokens,
        output_tokens=primary.output_tokens
        if primary.output_tokens is not None
        else fallback.output_tokens,
        reasoning_tokens=primary.reasoning_tokens
        if primary.reasoning_tokens is not None
        else fallback.reasoning_tokens,
        reasoning_summary=primary.reasoning_summary
        if primary.reasoning_summary is not None
        else fallback.reasoning_summary,
        response_id=primary.response_id
        if primary.response_id is not None
        else fallback.response_id,
        successful=primary.successful
        if primary.successful is not None
        else fallback.successful,
        error_log=primary.error_log
        if primary.error_log is not None
        else fallback.error_log,
        warning=primary.warning
        if primary.warning is not None
        else fallback.warning,
    )


def _auto_dummy_usage(prompt: str, responses: List[Any]) -> DummyResponseSpec:
    """Generate a fallback :class:`DummyResponseSpec` based on prompt/response length."""

    approx_in = max(1, _approx_tokens(str(prompt)))
    approx_out = max(
        1,
        sum(max(1, _approx_tokens(str(resp))) for resp in responses) or 1,
    )
    return DummyResponseSpec(
        input_tokens=approx_in,
        output_tokens=approx_out,
        reasoning_tokens=0,
    )


def _listify_error_log(value: Optional[Union[str, List[str]]]) -> List[str]:
    """Normalise ``value`` into a list of human-readable error log entries."""

    if value is None:
        return []
    return [str(item) for item in _coerce_to_list(value)]


def _synthesise_dummy_raw(
    identifier: str, spec: DummyResponseSpec, responses: List[Any]
) -> List[Dict[str, Any]]:
    """Create a faux Responses payload so downstream code sees usage metrics."""

    usage = {
        "input_tokens": int(spec.input_tokens or 0),
        "output_tokens": int(spec.output_tokens or 0),
        "output_tokens_details": {
            "reasoning_tokens": int(spec.reasoning_tokens or 0)
        },
    }
    output_blocks: List[Dict[str, Any]] = []
    if responses:
        content = [
            {"type": "output_text", "text": str(resp)} for resp in responses
        ]
        output_blocks.append(
            {"type": "message", "role": "assistant", "content": content}
        )
    if spec.reasoning_summary:
        output_blocks.append(
            {
                "type": "reasoning",
                "summary": [
                    {"type": "output_text", "text": spec.reasoning_summary}
                ],
            }
        )
    response_id = spec.response_id or f"dummy-{identifier}"
    return [
        {
            "id": response_id,
            "status": "completed" if spec.successful is not False else "failed",
            "output": output_blocks,
            "usage": usage,
        }
    ]

async def get_all_responses(
    prompts: List[str],
    identifiers: Optional[List[str]] = None,
    prompt_images: Optional[Dict[str, List[str]]] = None,
    prompt_audio: Optional[Dict[str, List[Dict[str, str]]]] = None,
    prompt_pdfs: Optional[Dict[str, List[Dict[str, str]]]] = None,
    prompt_web_search_filters: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    model: str = "gpt-5-mini",
    modality: Optional[str] = None,
    n: int = 1,
    max_output_tokens: Optional[int] = None,
    estimated_output_tokens_per_prompt: Optional[int] = ESTIMATED_OUTPUT_TOKENS_PER_PROMPT,
    temperature: float = 0.9,
    json_mode: bool = False,
    expected_schema: Optional[Dict[str, Any]] = None,
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[dict] = None,
    web_search: Optional[bool] = None,
    web_search_filters: Optional[Dict[str, Any]] = None,
    search_context_size: str = "medium",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    include: Optional[Union[str, Iterable[str]]] = None,
    dummy_responses: Optional[Dict[str, Union[DummyResponseSpec, Dict[str, Any]]]] = None,
    use_dummy: bool = False,
    response_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    get_all_responses_fn: Optional[Callable[..., Awaitable[pd.DataFrame]]] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    print_example_prompt: bool = True,
    save_path: str = "responses.csv",
    reset_files: bool = False,
    # Maximum number of parallel worker tasks to spawn.  This value
    # represents a ceiling; the actual number of concurrent requests
    # will be adjusted downward based on your API rate limits and
    # average prompt length.  See `_print_usage_overview` for more
    # details on how the concurrency cap is calculated.  When web
    # search or media inputs are enabled the helper automatically lowers
    # this ceiling to half of the requested value to avoid overwhelming
    # the API or tool backends.
    n_parallels: int = 650,
    max_retries: int = 3,
    timeout_factor: float = 2.5,
    max_timeout: Optional[float] = None,
    dynamic_timeout: bool = True,
    timeout_burst_window: float = 60.0,
    timeout_burst_cooldown: float = 60.0,
    timeout_burst_max_restarts: int = 3,
    background_mode: Optional[bool] = None,
    background_poll_interval: float = 2.0,
    cancel_existing_batch: bool = False,
    use_batch: bool = False,
    batch_completion_window: str = "24h",
    batch_poll_interval: int = 10,
    batch_wait_for_completion: bool = False,
    max_batch_requests: int = 50_000,
    max_batch_file_bytes: int = 100 * 1024 * 1024,
    save_every_x_responses: int = 100,
    verbose: bool = True,
    quiet: bool = False,
    global_cooldown: int = 15,
    rate_limit_window: float = 30.0,
    connection_error_window: float = 30.0,
    token_sample_size: int = 20,
    status_report_interval: Optional[float] = 120.0,
    planning_rate_limit_buffer: float = PLANNING_RATE_LIMIT_BUFFER,
    logging_level: Union[str, int] = "warning",
    **get_response_kwargs: Any,
) -> pd.DataFrame:
    """Retrieve model responses for a collection of prompts.

    For each prompt the function contacts the OpenAI API and stores the returned
    text, token counts and timing information in ``save_path``.  It can either
    send requests directly using an asynchronous worker pool or, when
    ``use_batch`` is ``True``, upload the prompts to the OpenAI Batch API and
    periodically poll for completion.  In both modes the helper automatically
    obeys rate limits, retries transient failures with exponential backoff and
    writes partial results to disk so interrupted runs can be resumed.
    The API base URL can be overridden per call via ``base_url`` or globally
    with the ``OPENAI_BASE_URL`` environment variable.

    When no ``max_output_tokens`` cutoff is supplied, the helper assumes each
    response will contain roughly ``estimated_output_tokens_per_prompt`` tokens
    (default: 400) for cost and throughput planning.  Adjust this parameter if
    you expect substantially longer or shorter generations.

    A dynamic timeout mechanism keeps long‑running jobs efficient: the function
    initially allows unlimited time for each request, then observes how long
    successful responses take and sets a timeout based on the 90th percentile of
    observed durations.  Subsequent calls use this timeout (capped by
    ``max_timeout`` if provided) and it is increased if later responses are
    slower.  Any request exceeding the current limit is cancelled and retried.
    While the
    timeout is unbounded the helper automatically submits requests in
    background mode and polls for completion so that connections closed by the
    server or networking layer do not strand in-flight prompts.  You can force
    or disable this behaviour with ``background_mode`` and adjust the polling
    cadence via ``background_poll_interval``.

    Concurrency adapts gently to sustained rate‑limit pressure.  A rolling
    window (``rate_limit_window``, default 30 seconds) tracks recent rate‑limit
    errors and only reduces the parallel worker cap when many errors occur
    within that window or when a long streak of consecutive errors is
    observed.  After a reduction the helper waits for another full window
    before scaling down again so brief spikes do not trigger runaway
    throttling, while successful calls reset the counters and allow the pool to
    scale back up.

    Connection errors (e.g., transient network drops, Wi‑Fi/VPN instability, or bandwidth limitations)
    are handled similarly: the helper tracks recent connection failures over
    ``connection_error_window`` seconds and reduces parallelism when repeated
    failures occur, while logging a hint to check network stability.

    Timeout bursts now scale with the active level of parallelism.  The helper
    triggers a protective restart only after observing roughly 1.25× the
    current parallel worker count worth of timeouts within
    ``timeout_burst_window`` seconds.

    Because every prompt that uses web search fans out into additional tool
    calls, and because media inputs (images/audio/PDFs) are heavier, the helper
    automatically lowers the requested ``n_parallels`` to half of its original
    value whenever web search or media payloads are detected.  This guard
    reduces the chance of exhausting tool quotas and keeps the Responses API
    from being flooded with longer prompts.  You can still request a smaller
    value manually if needed, and the message printed at the start of each run
    explains the adjustment so it can be revisited in the future if the
    limitation becomes unnecessary.

    Long‑running jobs can also emit periodic status updates.  The
    ``status_report_interval`` parameter controls how frequently the helper
    prints the current concurrency cap, number of active workers, queue size and
    failure counts (default: every five minutes).  Set the interval to ``None``
    or ``0`` to disable these reports.

    The worker pool responds promptly to user cancellation (e.g. pressing
    stop/``Ctrl+C``) by signalling all workers to halt before any new API
    requests are issued.  Transient network disruptions such as lost
    connections are retried with exponential backoff so long‑running jobs can
    resume automatically once connectivity returns.

    For organisations that route prompts through an internal LLM gateway the
    ``response_fn`` parameter exposes a lightweight dependency‑injection
    point.  When provided, the callable is awaited for every prompt instead of
    :func:`get_response`.  Only the keyword arguments accepted by the callable
    are forwarded, allowing simple signatures (e.g. ``async def fn(prompt)``)
    while still supporting advanced features for fully compatible adapters.
    The callable may return a list of responses, a ``(responses, duration)``
    pair, or the traditional ``(responses, duration, raw)`` tuple.  Missing
    duration or raw values simply disable the associated timeout and
    token‑tracking heuristics, keeping the worker resilient to alternative
    backends without forcing callers to mirror the OpenAI API exactly.

    Offline test runs frequently rely on ``use_dummy`` to avoid network calls.
    The optional ``dummy_responses`` mapping refines this mode by letting you
    describe the synthetic payload for each identifier (or ``"*"`` as a
    fallback) via :class:`DummyResponseSpec`.  These specs control the response
    text, duration, token usage, warnings and error logs so tests can exercise
    cost reporting and failure handling paths deterministically.

    Additional web search options (allowed domains and user location hints such
    as ``city``, ``country``, ``region``, ``timezone`` and ``type`` – usually
    ``"approximate"``) can be supplied together via ``web_search_filters``.
    Per-identifier overrides can be passed through
    ``prompt_web_search_filters`` where the mapping keys correspond to prompt
    identifiers and values follow the same schema as ``web_search_filters``.
    These overrides are merged with the global filters before each request,
    enabling DataFrame-driven location hints without hand-crafting separate
    dictionaries.

    To bypass the built-in orchestration entirely, supply ``get_all_responses_fn``.
    This callable is invoked at the start of the function and receives as many
    keyword arguments from this signature as it accepts (including values from
    ``**get_response_kwargs``).  It must handle prompt dispatch on its own and
    return a :class:`pandas.DataFrame` containing at least ``"Identifier"`` and
    ``"Response"`` columns, where ``"Response"`` mirrors this helper’s output
    structure.  The callable must accept ``prompts`` and ``identifiers`` and
    should ideally accept ``json_mode`` and ``model`` if relevant.  When
    provided, ``get_all_responses_fn`` takes precedence over ``response_fn`` and
    all other internal processing.
"""
    global _USAGE_SHEET_PRINTED
    message_verbose = bool(verbose and not quiet)
    set_log_level(logging_level)
    logger = get_logger(__name__)
    identifiers = prompts if identifiers is None else identifiers
    if get_all_responses_fn is not None:
        if response_fn is not None:
            logger.info(
                "Both get_all_responses_fn and response_fn were supplied; "
                "deferring to get_all_responses_fn and ignoring response_fn."
            )
        candidate = get_all_responses_fn
        underlying_callable: Callable[..., Any] = candidate
        if isinstance(underlying_callable, functools.partial):
            underlying_callable = underlying_callable.func  # type: ignore[attr-defined]
        try:
            sig = inspect.signature(underlying_callable)
        except (TypeError, ValueError):
            sig = None
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ) if sig is not None else True
        param_lookup = sig.parameters if sig is not None else {}
        if not (accepts_var_kw or "prompts" in param_lookup):
            raise TypeError("Custom get_all_responses_fn must accept a `prompts` argument.")
        if not (accepts_var_kw or "identifiers" in param_lookup):
            raise TypeError("Custom get_all_responses_fn must accept an `identifiers` argument.")
        base_kwargs: Dict[str, Any] = {}
        for name in inspect.signature(get_all_responses).parameters:
            if name in {"get_all_responses_fn", "get_response_kwargs"}:
                continue
            base_kwargs[name] = locals()[name]
        extra_kwargs = dict(get_response_kwargs)
        available_kwargs: Dict[str, Any] = {**extra_kwargs, **base_kwargs}
        used_keys: Set[str] = set()
        positional_args: List[Any] = []
        call_kwargs: Dict[str, Any] = {}
        if sig is not None:
            for param in sig.parameters.values():
                if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                    if param.name in available_kwargs:
                        positional_args.append(available_kwargs[param.name])
                        used_keys.add(param.name)
                    elif param.default is inspect._empty:
                        raise TypeError(
                            f"Custom get_all_responses_fn is missing required parameter `{param.name}`."
                        )
                elif param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                    if param.name in available_kwargs:
                        call_kwargs[param.name] = available_kwargs[param.name]
                        used_keys.add(param.name)
            if accepts_var_kw:
                for key, value in available_kwargs.items():
                    if key not in used_keys:
                        call_kwargs[key] = value
        else:
            call_kwargs = available_kwargs
        result = await get_all_responses_fn(*positional_args, **call_kwargs)
        if not isinstance(result, pd.DataFrame):
            raise TypeError("Custom get_all_responses_fn must return a pandas DataFrame.")
        if "Response" not in result.columns or "Identifier" not in result.columns:
            raise ValueError(
                "Custom get_all_responses_fn must return a DataFrame containing 'Identifier' and 'Response' columns."
            )
        return result
    if message_verbose:
        print("Initializing model calls and loading data...")
    if api_key is not None:
        os.environ["OPENAI_API_KEY"] = api_key
    response_callable = response_fn or get_response
    provided_api_key = api_key
    underlying_callable = response_callable
    if isinstance(underlying_callable, functools.partial):
        underlying_callable = underlying_callable.func  # type: ignore[attr-defined]
    try:
        underlying_callable = inspect.unwrap(underlying_callable)  # type: ignore[arg-type]
    except Exception:
        pass
    using_custom_response_fn = response_fn is not None and underlying_callable is not get_response
    manage_rate_limits = not using_custom_response_fn
    planning_buffer = float(min(max(planning_rate_limit_buffer, 0.1), 1.0))
    if not use_dummy and not using_custom_response_fn:
        _require_api_key()
    _ensure_runtime_dependencies(verbose=message_verbose)
    try:
        estimated_output_tokens_per_prompt = int(
            ESTIMATED_OUTPUT_TOKENS_PER_PROMPT
            if estimated_output_tokens_per_prompt is None
            else estimated_output_tokens_per_prompt
        )
    except Exception:
        estimated_output_tokens_per_prompt = ESTIMATED_OUTPUT_TOKENS_PER_PROMPT
    if estimated_output_tokens_per_prompt <= 0:
        estimated_output_tokens_per_prompt = ESTIMATED_OUTPUT_TOKENS_PER_PROMPT
    inferred_modality = modality or _infer_modality_from_inputs(
        prompt_images,
        prompt_audio,
        prompt_pdfs,
    )
    web_search_requested = bool(get_response_kwargs.get("web_search", web_search))
    has_media_payloads = _has_media_payloads(
        prompt_images,
        prompt_audio,
        prompt_pdfs,
        identifiers,
    )
    extra_input_tokens_per_prompt = _estimate_extra_input_tokens_per_prompt(
        modality=inferred_modality,
        web_search=web_search_requested,
        has_media=has_media_payloads,
    )
    dataset_stats = _estimate_dataset_stats(
        prompts,
        extra_input_tokens_per_prompt=extra_input_tokens_per_prompt,
    )
    cost_estimate = _estimate_cost(
        prompts,
        n,
        max_output_tokens,
        model,
        use_batch,
        sample_size=_ESTIMATION_SAMPLE_SIZE,
        estimated_output_tokens_per_prompt=estimated_output_tokens_per_prompt,
        extra_input_tokens_per_prompt=extra_input_tokens_per_prompt,
    )
    _print_run_banner(
        prompts=prompts,
        model=model,
        n=n,
        use_batch=use_batch,
        modality=inferred_modality,
        web_search=web_search_requested,
        estimated_cost=cost_estimate,
        max_output_tokens=max_output_tokens,
        stats=dataset_stats,
        estimated_output_tokens_per_prompt=estimated_output_tokens_per_prompt,
        verbose=message_verbose,
    )
    response_param_names: Set[str] = set()
    response_accepts_var_kw = False
    response_accepts_return_raw = False
    prompt_param_kind: Optional[inspect._ParameterKind] = None
    has_generic_positional_slot = False
    has_var_positional = False
    has_var_keyword = False
    prompt_call_via_keyword = False
    try:
        sig = inspect.signature(response_callable)
    except (TypeError, ValueError):
        response_accepts_var_kw = True
        response_accepts_return_raw = True
        prompt_call_via_keyword = False
    else:
        for name, param in sig.parameters.items():
            if name in {"self", "cls"}:
                continue
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                has_var_positional = True
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                response_accepts_var_kw = True
                has_var_keyword = True
                continue
            if name == "prompt":
                prompt_param_kind = param.kind
                continue
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                response_param_names.add(name)
            if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY):
                has_generic_positional_slot = True
        response_accepts_return_raw = response_accepts_var_kw or ("return_raw" in response_param_names)
        prompt_can_be_positional = False
        prompt_can_be_keyword = False
        if prompt_param_kind is not None:
            if prompt_param_kind == inspect.Parameter.POSITIONAL_ONLY:
                prompt_can_be_positional = True
            elif prompt_param_kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                prompt_can_be_positional = True
                prompt_can_be_keyword = True
            elif prompt_param_kind == inspect.Parameter.KEYWORD_ONLY:
                prompt_can_be_keyword = True
            elif prompt_param_kind == inspect.Parameter.VAR_POSITIONAL:
                prompt_can_be_positional = True
            elif prompt_param_kind == inspect.Parameter.VAR_KEYWORD:
                prompt_can_be_keyword = True
        else:
            if has_generic_positional_slot or has_var_positional:
                prompt_can_be_positional = True
            if has_var_keyword:
                prompt_can_be_keyword = True
        if not prompt_can_be_positional and not prompt_can_be_keyword:
            raise TypeError(
                "Custom response_fn must accept a `prompt` argument as a positional or keyword parameter."
            )
        prompt_call_via_keyword = prompt_can_be_keyword and not prompt_can_be_positional
    dummy_response_specs: Dict[str, DummyResponseSpec] = {}
    dummy_default_spec: Optional[DummyResponseSpec] = None
    if dummy_responses:
        for key, value in dummy_responses.items():
            spec = _coerce_dummy_response_spec(value)
            if spec is None:
                continue
            dummy_response_specs[str(key)] = spec
        dummy_default_spec = dummy_response_specs.get("*") or dummy_response_specs.get("__default__")
        if not use_dummy:
            logger.warning(
                "`dummy_responses` were provided but `use_dummy` is False; ignoring synthetic payloads."
            )
    else:
        dummy_response_specs = {}
    if status_report_interval is not None:
        try:
            status_report_interval = float(status_report_interval)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid `status_report_interval=%r`; disabling periodic status reports.",
                status_report_interval,
            )
            status_report_interval = None
        else:
            if status_report_interval <= 0:
                status_report_interval = None
    if quiet:
        status_report_interval = None
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    if web_search_filters and not web_search and not get_response_kwargs.get("web_search", False):
        logger.debug(
            "web_search_filters were supplied but web_search is disabled; filters will be ignored."
        )

    if using_custom_response_fn and use_batch:
        logger.warning(
            "Custom response_fn cannot be combined with batch mode; falling back to per-request execution."
        )
        use_batch = False

    if get_response_kwargs.get("web_search", web_search) and get_response_kwargs.get(
        "json_mode", json_mode
    ):
        logger.warning(
            "Web search cannot be combined with JSON mode; disabling JSON mode."
        )
        get_response_kwargs["json_mode"] = False
    # httpx logs a success line for every request at INFO level, which
    # interferes with tqdm's progress display.  Silence these messages
    # so only warnings and errors surface.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    status = StatusTracker()
    requested_n_parallels = max(1, n_parallels)
    user_requested_n_parallels = requested_n_parallels
    tokenizer = _get_tokenizer(model)
    # Backwards compatibility for identifiers
    if identifiers is None:
        identifiers = prompts
    # Pull default values into kwargs for get_response
    get_response_kwargs.setdefault("web_search", web_search)
    if web_search_filters is not None:
        get_response_kwargs.setdefault("web_search_filters", web_search_filters)
    get_response_kwargs.setdefault("search_context_size", search_context_size)
    get_response_kwargs.setdefault("tools", tools)
    get_response_kwargs.setdefault("tool_choice", tool_choice)
    get_response_kwargs.setdefault("json_mode", json_mode)
    get_response_kwargs.setdefault("expected_schema", expected_schema)
    get_response_kwargs.setdefault("temperature", temperature)
    get_response_kwargs.setdefault("reasoning_effort", reasoning_effort)
    get_response_kwargs.setdefault("reasoning_summary", reasoning_summary)
    # Pass the chosen model through to get_response by default
    get_response_kwargs.setdefault("model", model)
    get_response_kwargs.setdefault("base_url", base_url)
    if background_mode is not None:
        get_response_kwargs.setdefault("background_mode", background_mode)
    get_response_kwargs.setdefault("background_poll_interval", background_poll_interval)
    base_web_search_filters = get_response_kwargs.get("web_search_filters")
    web_search_active = bool(get_response_kwargs.get("web_search"))
    media_active = _has_media_payloads(
        prompt_images, prompt_audio, prompt_pdfs, identifiers
    )
    web_search_warning_text: Optional[str] = None
    web_search_parallel_note: Optional[str] = None
    if web_search_active:
        web_search_warning_text = (
            "⚠️ Web search is enabled: tool lookups incur extra fees and tokens beyond this estimate, "
            "so actual costs may be significantly higher. Reduce `n_parallels` manually if tool errors occur."
        )
        logger.warning(web_search_warning_text)
    if not use_batch:
        reduction_reasons: List[str] = []
        if web_search_active:
            reduction_reasons.append("web search")
        if media_active:
            reduction_reasons.append("media inputs (image/audio/pdf)")
        if reduction_reasons:
            reduced = max(1, int(math.ceil(requested_n_parallels * 0.5)))
            if reduced < requested_n_parallels:
                requested_n_parallels = reduced
                reasons = " and ".join(reduction_reasons)
                web_search_parallel_note = (
                    f"{reasons.capitalize()} detected; automatically capped parallel workers at {requested_n_parallels} "
                    f"(requested {user_requested_n_parallels}). You can increase n_parallels for faster runs, "
                    "or decrease further if running into API errors."
                )
                logger.info(web_search_parallel_note)
    web_search_warning_displayed = False
    web_search_note_displayed = False
    # Decide default cutoff once per job using cached rate headers
    # Fetch rate headers once to avoid multiple API calls
    # Retrieve rate‑limit headers for the chosen model.  Passing the model
    # ensures the helper performs a dummy call with the correct model
    # rather than probing the unsupported ``/v1/models`` endpoint.
    rate_headers = (
        _get_rate_limit_headers(model, base_url=base_url)
        if manage_rate_limits
        else {}
    )
    cutoff = max_output_tokens
    get_response_kwargs.setdefault("max_output_tokens", cutoff)
    initial_estimated_output_tokens = (
        cutoff if cutoff is not None else estimated_output_tokens_per_prompt
    )
    planning_output_tokens = initial_estimated_output_tokens
    output_headroom_live = OUTPUT_TOKEN_HEADROOM_INITIAL
    output_headroom_reduced = False
    # Always load or initialise the CSV
    # Expand variables in save_path and ensure the parent directory exists.
    save_path = os.path.expandvars(os.path.expanduser(save_path))
    save_dir = Path(save_path).expanduser().resolve().parent
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.debug("Could not create directory %s", save_dir)
    if reset_files:
        for p in (save_path, save_path + ".batch_state.json"):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    csv_header_written = os.path.exists(save_path) and not reset_files and os.path.getsize(save_path) > 0
    if os.path.exists(save_path) and not reset_files:
        if message_verbose:
            print(f"Reading from existing files at {save_path}...")
        df = pd.read_csv(save_path)
        df = df.drop_duplicates(subset=["Identifier"], keep="last")
        df["Response"] = df["Response"].apply(_de)
        if "Error Log" in df.columns:
            df["Error Log"] = df["Error Log"].apply(_de)
        if "Web Search Sources" in df.columns:
            df["Web Search Sources"] = df["Web Search Sources"].apply(_de)
        else:
            df["Web Search Sources"] = pd.NA
        expected_cols = [
            "Input Tokens",
            "Reasoning Tokens",
            "Output Tokens",
            "Reasoning Effort",
            "Successful",
            "Error Log",
            "Web Search Sources",
        ]
        if reasoning_summary is not None:
            expected_cols.insert(4, "Reasoning Summary")
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.NA
        if reasoning_summary is None and "Reasoning Summary" in df.columns:
            df = df.drop(columns=["Reasoning Summary"])
        # Only skip identifiers that previously succeeded so failures can be retried
        if "Successful" in df.columns:
            done = set(df.loc[df["Successful"] == True, "Identifier"])
        else:
            done = set(df["Identifier"])
        if message_verbose:
            print(f"Loaded {len(df):,} rows; {len(done):,} already marked complete.")
    else:
        cols = [
            "Identifier",
            "Response",
            "Web Search Sources",
            "Time Taken",
            "Input Tokens",
            "Reasoning Tokens",
            "Output Tokens",
            "Reasoning Effort",
            "Successful",
            "Error Log",
        ]
        if reasoning_summary is not None:
            cols.insert(7, "Reasoning Summary")
        df = pd.DataFrame(columns=cols)
        done = set()
    written_identifiers: Set[Any] = set(df["Identifier"]) if not df.empty else set()
    # Helper to calculate and report final run cost
    def _report_cost() -> None:
        nonlocal df
        pricing = _lookup_model_pricing(model)
        required_cols = {"Input Tokens", "Output Tokens"}
        if not pricing or not required_cols.issubset(df.columns):
            return
        inp = pd.to_numeric(df["Input Tokens"], errors="coerce").fillna(0)
        out = pd.to_numeric(df["Output Tokens"], errors="coerce").fillna(0)
        if "Reasoning Tokens" in df:
            reason = pd.to_numeric(df["Reasoning Tokens"], errors="coerce").fillna(0)
        else:
            reason = pd.Series([0] * len(df))
        df["Cost"] = (inp / 1_000_000) * pricing["input"] + ((out + reason) / 1_000_000) * pricing["output"]
        total_cost = df["Cost"].sum()
        if len(df) > 0:
            avg_row = total_cost / len(df)
            avg_1000 = avg_row * 1000
        else:
            avg_row = 0.0
            avg_1000 = 0.0
        msg = (
            f"Actual total cost: ${total_cost:.2f}; average per row: ${avg_row:.2f}; average per 1000 rows: ${avg_1000:.2f}"
        )
        if message_verbose:
            print(msg)
        logger.info(msg)
    # Filter prompts/identifiers based on what is already completed
    todo_pairs = [(p, i) for p, i in zip(prompts, identifiers) if i not in done]
    if not todo_pairs:
        _report_cost()
        return df
    if len(todo_pairs) >= 10_000:
        effective_save_every = save_every_x_responses
        if len(todo_pairs) >= 50_000:
            effective_save_every = max(save_every_x_responses, 2000)
        elif len(todo_pairs) >= 20_000:
            effective_save_every = max(save_every_x_responses, 1000)
        elif len(todo_pairs) >= 10_000:
            effective_save_every = max(save_every_x_responses, 500)
        if effective_save_every != save_every_x_responses:
            logger.debug(
                "Large run detected (%d rows); autoscaling checkpoint frequency to every %d responses (was %d).",
                len(todo_pairs),
                effective_save_every,
                save_every_x_responses,
            )
            save_every_x_responses = effective_save_every
    status.num_tasks_started = len(todo_pairs)
    status.num_tasks_in_progress = len(todo_pairs)
    if prompt_audio and any(prompt_audio.get(str(i)) for _, i in todo_pairs):
        if use_batch:
            logger.warning(
                "Batch mode is not supported for audio inputs; falling back to non-batch processing."
            )
        use_batch = False
    # Warn the user if the input dataset is very large.  Processing more
    # than 50,000 prompts in a single run can lead to very long execution
    # times and increased risk of rate‑limit throttling.  We still proceed
    # with the run, but advise the user to split the input into smaller
    # batches when possible.
    if len(todo_pairs) > 50_000:
        logger.warning(
            f"You are attempting to process {len(todo_pairs):,} prompts in one go. For better performance and reliability, we recommend splitting jobs into 50k‑row chunks or fewer."
        )
    show_example_prompt = bool(print_example_prompt and not quiet)
    prompt_list = [p for p, _ in todo_pairs]
    todo_stats = _estimate_dataset_stats(
        prompt_list,
        sample_size=_ESTIMATION_SAMPLE_SIZE,
        extra_input_tokens_per_prompt=extra_input_tokens_per_prompt,
    )
    if todo_pairs and not using_custom_response_fn and message_verbose:
        _print_usage_overview(
            prompts=prompt_list,
            n=n,
            max_output_tokens=cutoff,
            model=model,
            use_batch=use_batch,
            n_parallels=requested_n_parallels,
            verbose=message_verbose,
            rate_headers=rate_headers,
            base_url=base_url,
            web_search_warning=web_search_warning_text,
            web_search_parallel_note=web_search_parallel_note,
            show_static_sections=not _USAGE_SHEET_PRINTED,
            stats=todo_stats,
            sample_size=_ESTIMATION_SAMPLE_SIZE,
            estimated_output_tokens_per_prompt=estimated_output_tokens_per_prompt,
            extra_input_tokens_per_prompt=extra_input_tokens_per_prompt,
            heading="Run limits",
            show_prompt_stats=False,
        )
        _USAGE_SHEET_PRINTED = True
        if web_search_warning_text:
            web_search_warning_displayed = True
        if web_search_parallel_note:
            web_search_note_displayed = True
    elif message_verbose and todo_pairs:
        print(
            "\n===== Job summary ====="
            f"\nNumber of prompts: {len(prompt_list)}"
            f"\nParallel workers (requested): {requested_n_parallels}"
        )
        if web_search_warning_text:
            print(web_search_warning_text)
            web_search_warning_displayed = True
        if web_search_parallel_note:
            print(web_search_parallel_note)
            web_search_note_displayed = True
        logger.info(
            "Skipping OpenAI usage overview because a custom response_fn was supplied."
        )
    if message_verbose and web_search_warning_text and not web_search_warning_displayed:
        print(web_search_warning_text)
        web_search_warning_displayed = True
    if message_verbose and web_search_parallel_note and not web_search_note_displayed:
        print(web_search_parallel_note)
        web_search_note_displayed = True
    if show_example_prompt and todo_pairs:
        example_prompt, _ = todo_pairs[0]
        _display_example_prompt(example_prompt, verbose=message_verbose)
        if not message_verbose:
            logger.info("Example prompt omitted from logs because verbose output is disabled.")
    # Dynamically adjust the maximum number of parallel workers based on rate
    # limits.  We base the concurrency on your API’s per‑minute request and
    # token budgets and the average prompt length.  This calculation only
    # runs once at the start of a non‑batch run.  The resulting value acts
    # as the true upper bound on parallelism; it will be used to size the
    # worker pool and to configure the request/token limiters below.
    max_parallel_ceiling = requested_n_parallels
    concurrency_cap = requested_n_parallels
    allowed_req_pm = max(1, requested_n_parallels)
    estimated_tokens_per_call = max(
        1.0, (planning_output_tokens * output_headroom_live + 1) * max(1, n)
    )
    allowed_tok_pm = int(max(1, requested_n_parallels * estimated_tokens_per_call))
    if not use_batch and manage_rate_limits:
        try:
            # Estimate the average number of tokens per call using tiktoken
            # for more accurate gating.  We include the expected output length
            # to ensure that long prompts reduce available parallelism.
            sample_for_tokens = (
                random.sample(todo_pairs, min(len(todo_pairs), _ESTIMATION_SAMPLE_SIZE))
                if len(todo_pairs) > _ESTIMATION_SAMPLE_SIZE
                else todo_pairs
            )
            avg_input_tokens = (
                sum(len(tokenizer.encode(p)) for p, _ in sample_for_tokens)
                / max(1, len(sample_for_tokens))
            )
            gating_output = planning_output_tokens * output_headroom_live
            tokens_per_call = max(1.0, (avg_input_tokens + gating_output) * max(1, n))

            def _pf(val: Optional[str]) -> Optional[float]:
                try:
                    if val is None:
                        return None
                    s = str(val).strip()
                    if not s:
                        return None
                    f = float(s)
                    return f if f > 0 else None
                except Exception:
                    return None

            lim_r: Optional[float] = None
            rem_r: Optional[float] = None
            lim_t: Optional[float] = None
            rem_t: Optional[float] = None
            if rate_headers:
                lim_r = _pf(rate_headers.get("limit_requests"))
                rem_r = _pf(rate_headers.get("remaining_requests"))
                lim_t = _pf(rate_headers.get("limit_tokens")) or _pf(
                    rate_headers.get("limit_tokens_usage_based")
                )
                rem_t = _pf(rate_headers.get("remaining_tokens")) or _pf(
                    rate_headers.get("remaining_tokens_usage_based")
                )
            def _with_headroom(val: Optional[float], *, buffer: float = RATE_LIMIT_HEADROOM) -> Optional[int]:
                if val is None:
                    return None
                return int(max(1, math.floor(val * buffer)))

            def _select_budget(limit_val: Optional[float], remaining_val: Optional[float]) -> Optional[float]:
                candidates = [v for v in (remaining_val, limit_val) if v is not None and v > 0]
                if not candidates:
                    return None
                return min(candidates)

            def _select_ceiling(limit_val: Optional[float], remaining_val: Optional[float]) -> Optional[float]:
                candidates = [v for v in (limit_val, remaining_val) if v is not None and v > 0]
                if not candidates:
                    return None
                return max(candidates)

            initial_req_budget = _select_budget(lim_r, rem_r)
            initial_tok_budget = _select_budget(lim_t, rem_t)
            ceiling_req = _select_ceiling(lim_r, rem_r)
            ceiling_tok = _select_ceiling(lim_t, rem_t)
            req_budget_for_cap = _with_headroom(initial_req_budget, buffer=planning_buffer)
            tok_budget_for_cap = _with_headroom(initial_tok_budget, buffer=planning_buffer)
            concurrency_candidates = [requested_n_parallels]
            if req_budget_for_cap is not None:
                concurrency_candidates.append(req_budget_for_cap)
            if tok_budget_for_cap is not None:
                concurrency_candidates.append(
                    int(max(1, tok_budget_for_cap // tokens_per_call))
                )
            concurrency_cap = max(1, min(concurrency_candidates))
            ceiling_candidates = [requested_n_parallels]
            ceiling_req_budget = _with_headroom(ceiling_req, buffer=planning_buffer)
            ceiling_tok_budget = _with_headroom(ceiling_tok, buffer=planning_buffer)
            if ceiling_req_budget is not None:
                ceiling_candidates.append(ceiling_req_budget)
            if ceiling_tok_budget is not None:
                ceiling_candidates.append(
                    int(max(1, ceiling_tok_budget // tokens_per_call))
                )
            max_parallel_ceiling = max(1, min(ceiling_candidates))
            if max_parallel_ceiling < concurrency_cap:
                max_parallel_ceiling = concurrency_cap
            if concurrency_cap < requested_n_parallels:
                logger.info(
                    f"[parallel reduction] Limiting parallel workers from {requested_n_parallels} to {concurrency_cap} based on your current rate limits. Consider upgrading your plan for faster processing."
                )
            if ceiling_req_budget is not None:
                allowed_req_pm = ceiling_req_budget
            elif req_budget_for_cap is not None:
                allowed_req_pm = req_budget_for_cap
            else:
                allowed_req_pm = max(1, max_parallel_ceiling)
            if ceiling_tok_budget is not None:
                allowed_tok_pm = ceiling_tok_budget
            elif tok_budget_for_cap is not None:
                allowed_tok_pm = tok_budget_for_cap
            else:
                allowed_tok_pm = int(max(1, max_parallel_ceiling * tokens_per_call))
            estimated_tokens_per_call = tokens_per_call
        except Exception:
            concurrency_cap = max(1, requested_n_parallels)
            max_parallel_ceiling = concurrency_cap
            allowed_req_pm = max(1, requested_n_parallels)
            allowed_tok_pm = int(max(1, requested_n_parallels * estimated_tokens_per_call))
    elif use_batch:
        # In batch mode we don't set concurrency or limiters here; they are
        # handled by the batch API submission.
        allowed_req_pm = 1
        allowed_tok_pm = 1
    planned_ppm, throughput_details_plan = _safe_planned_ppm_and_details(
        allowed_req_pm if manage_rate_limits else None,
        allowed_tok_pm if manage_rate_limits else None,
        estimated_tokens_per_call,
        context="planning parallelization",
    )
    throughput_ceiling_ppm = (
        max_parallel_ceiling if planned_ppm is None else planned_ppm
    )
    planned_parallel_workers = min(
        concurrency_cap,
        throughput_ceiling_ppm if throughput_ceiling_ppm is not None else concurrency_cap,
    )
    planned_parallel_workers = max(1, int(planned_parallel_workers))
    if message_verbose and not use_batch:
        print("\n===== Parallelization plan =====")
        print(f"# of parallel threads: {planned_parallel_workers}")
        for line in _format_throughput_plan(
            planned_ppm=planned_ppm,
            throughput_details=throughput_details_plan,
            remaining_prompts=len(todo_pairs),
            allowed_req_pm=allowed_req_pm if manage_rate_limits else None,
            allowed_tok_pm=allowed_tok_pm if manage_rate_limits else None,
            include_upgrade_hint=True,
            tokens_per_call=estimated_tokens_per_call,
            parallel_ceiling=max_parallel_ceiling,
            n_parallels=user_requested_n_parallels,
            ultimate_parallel_cap=user_requested_n_parallels,
        ):
            print(line)

    # Batch submission path
    if use_batch:
        state_path = save_path + ".batch_state.json"

        # Helper to append batch rows
        def _append_results(rows: List[Dict[str, Any]]) -> None:
            nonlocal df, csv_header_written, written_identifiers
            if not rows:
                return
            batch_df = pd.DataFrame(rows)
            if "Web Search Sources" not in batch_df.columns:
                batch_df["Web Search Sources"] = pd.NA
            batch_df = batch_df[~batch_df["Identifier"].isin(written_identifiers)]
            if batch_df.empty:
                return
            to_save = batch_df.copy()
            for col in ("Response", "Error Log", "Web Search Sources"):
                if col in to_save:
                    to_save[col] = to_save[col].apply(_ser)
            to_save.to_csv(
                save_path,
                mode="a" if csv_header_written else "w",
                header=not csv_header_written,
                index=False,
                quoting=csv.QUOTE_MINIMAL,
            )
            csv_header_written = True
            if df.empty:
                df = batch_df.reset_index(drop=True)
            else:
                df = pd.concat([df, batch_df], ignore_index=True)
            written_identifiers.update(batch_df["Identifier"])

        client = _get_client(base_url)
        # Load existing state
        if os.path.exists(state_path) and not reset_files:
            with open(state_path, "r") as f:
                state = json.load(f)
        else:
            state = {}
        # Convert single batch format
        if state.get("batch_id"):
            state = {
                "batches": [
                    {
                        "batch_id": state["batch_id"],
                        "input_file_id": state.get("input_file_id"),
                        "total": None,
                        "submitted_at": None,
                    }
                ]
            }
        # Cancel unfinished batches if requested
        if cancel_existing_batch and state.get("batches"):
            logger.info("Cancelling unfinished batch jobs...")
            for b in state["batches"]:
                bid = b.get("batch_id")
                try:
                    await client.batches.cancel(bid)
                    logger.info(f"Cancelled batch {bid}.")
                except Exception as exc:
                    logger.warning(f"Failed to cancel batch {bid}: {exc}")
            try:
                os.remove(state_path)
            except OSError:
                pass
            state = {}
        # If there are no unfinished batches, create new ones
        if not state.get("batches"):
            tasks: List[Dict[str, Any]] = []
            for prompt, ident in todo_pairs:
                imgs = prompt_images.get(str(ident)) if prompt_images else None
                pdfs = prompt_pdfs.get(str(ident)) if prompt_pdfs else None
                model_name = get_response_kwargs.get("model", "gpt-5-mini")
                legacy_system_instruction = _uses_legacy_system_instruction(model_name)
                if imgs or pdfs:
                    contents: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
                    if imgs:
                        for img in imgs:
                            img_url = img if str(img).startswith("data:") else f"data:image/jpeg;base64,{img}"
                            contents.append({"type": "input_image", "image_url": img_url})
                    if pdfs:
                        for pdf in pdfs:
                            file_data = pdf.get("file_data")
                            file_url = pdf.get("file_url")
                            filename = pdf.get("filename")
                            if file_data and not str(file_data).startswith("data:"):
                                file_data = f"data:application/pdf;base64,{file_data}"
                            payload: Dict[str, Any] = {"type": "input_file"}
                            if filename:
                                payload["filename"] = filename
                            if file_data:
                                payload["file_data"] = file_data
                            elif file_url:
                                payload["file_url"] = file_url
                            else:
                                continue
                            contents.append(payload)
                    input_data = (
                        [{"role": "user", "content": contents}]
                        if not legacy_system_instruction
                        else [
                            {
                                "role": "system",
                                "content": DEFAULT_SYSTEM_INSTRUCTION,
                            },
                            {"role": "user", "content": contents},
                        ]
                    )
                else:
                    input_data = (
                        [{"role": "user", "content": prompt}]
                        if not legacy_system_instruction
                        else [
                            {
                                "role": "system",
                                "content": DEFAULT_SYSTEM_INSTRUCTION,
                            },
                            {"role": "user", "content": prompt},
                        ]
                    )
                per_prompt_filters = (
                    prompt_web_search_filters.get(str(ident))
                    if prompt_web_search_filters
                    else None
                )
                merged_filters = _merge_web_search_filters(
                    base_web_search_filters, per_prompt_filters
                )
                body = _build_params(
                    model=model_name,
                    input_data=input_data,
                    max_output_tokens=cutoff,
                    temperature=get_response_kwargs.get("temperature", 0.9),
                    tools=get_response_kwargs.get("tools"),
                    tool_choice=get_response_kwargs.get("tool_choice"),
                    web_search=get_response_kwargs.get("web_search", False),
                    web_search_filters=merged_filters,
                    search_context_size=get_response_kwargs.get(
                        "search_context_size", "medium"
                    ),
                    json_mode=get_response_kwargs.get("json_mode", False),
                    expected_schema=get_response_kwargs.get("expected_schema"),
                    reasoning_effort=get_response_kwargs.get("reasoning_effort"),
                    reasoning_summary=get_response_kwargs.get(
                        "reasoning_summary"
                    ),
                    include=get_response_kwargs.get("include"),
                )
                tasks.append(
                    {
                        "custom_id": str(ident),
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": body,
                    }
                )
            if tasks:
                batches: List[List[Dict[str, Any]]] = []
                current_batch: List[Dict[str, Any]] = []
                current_size = 0
                for obj in tasks:
                    line_bytes = (
                        len(json.dumps(obj, ensure_ascii=False).encode("utf-8")) + 1
                    )
                    if (
                        len(current_batch) >= max_batch_requests
                        or current_size + line_bytes > max_batch_file_bytes
                    ):
                        if current_batch:
                            batches.append(current_batch)
                        current_batch = []
                        current_size = 0
                    current_batch.append(obj)
                    current_size += line_bytes
                if current_batch:
                    batches.append(current_batch)
                state["batches"] = []
                for batch_tasks in batches:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".jsonl"
                    ) as tmp:
                        for obj in batch_tasks:
                            tmp.write(json.dumps(obj).encode("utf-8") + b"\n")
                        input_filename = tmp.name
                    uploaded = await client.files.create(
                        file=open(input_filename, "rb"), purpose="batch"
                    )
                    batch = await client.batches.create(
                        input_file_id=uploaded.id,
                        endpoint="/v1/responses",
                        completion_window=batch_completion_window,
                    )
                    state["batches"].append(
                        {
                            "batch_id": batch.id,
                            "input_file_id": uploaded.id,
                            "total": len(batch_tasks),
                            "submitted_at": int(time.time()),
                        }
                    )
                    logger.info(
                        f"Submitted batch {batch.id} with {len(batch_tasks)} requests."
                    )
                with open(state_path, "w") as f:
                    json.dump(state, f)
        # Return immediately if not waiting for completion
        if not batch_wait_for_completion:
            return df
        unfinished_batches: List[Dict[str, Any]] = list(state.get("batches", []))
        completed_rows: List[Dict[str, Any]] = []
        while unfinished_batches:
            for b in list(unfinished_batches):
                bid = b.get("batch_id")
                try:
                    job = await client.batches.retrieve(bid)
                except Exception as exc:
                    logger.warning(f"Failed to retrieve batch {bid}: {exc}")
                    continue
                status = job.status
                if status == "completed":
                    output_file_id = job.output_file_id
                    error_file_id = job.error_file_id
                    logger.info(f"Batch {bid} completed. Downloading results...")
                    try:
                        file_response = await client.files.content(output_file_id)
                    except Exception as exc:
                        logger.warning(
                            f"Failed to download output file for batch {bid}: {exc}"
                        )
                        unfinished_batches.remove(b)
                        continue
                    # Normalize file response to plain text
                    text_data: Optional[str] = None
                    try:
                        if isinstance(file_response, str):
                            text_data = file_response
                        elif isinstance(file_response, bytes):
                            text_data = file_response.decode("utf-8", errors="replace")
                        elif hasattr(file_response, "text"):
                            attr = getattr(file_response, "text")
                            text_data = await attr() if callable(attr) else attr  # type: ignore
                        if text_data is None and hasattr(file_response, "read"):
                            content_bytes = await file_response.read()  # type: ignore
                            text_data = (
                                content_bytes.decode("utf-8", errors="replace")
                                if isinstance(content_bytes, bytes)
                                else str(content_bytes)
                            )
                    except Exception:
                        pass
                    if text_data is None:
                        logger.warning(f"No data found in output file for batch {bid}.")
                        unfinished_batches.remove(b)
                        continue
                    errors: Dict[str, Any] = {}
                    if error_file_id:
                        try:
                            err_response = await client.files.content(error_file_id)
                        except Exception as exc:
                            logger.warning(
                                f"Failed to download error file for batch {bid}: {exc}"
                            )
                            err_response = None
                        if err_response is not None:
                            err_text: Optional[str] = None
                            try:
                                if isinstance(err_response, str):
                                    err_text = err_response
                                elif isinstance(err_response, bytes):
                                    err_text = err_response.decode(
                                        "utf-8", errors="replace"
                                    )
                                elif hasattr(err_response, "text"):
                                    attr = getattr(err_response, "text")
                                    err_text = await attr() if callable(attr) else attr  # type: ignore
                                if err_text is None and hasattr(err_response, "read"):
                                    content_bytes = await err_response.read()  # type: ignore
                                    err_text = (
                                        content_bytes.decode("utf-8", errors="replace")
                                        if isinstance(content_bytes, bytes)
                                        else str(content_bytes)
                                    )
                            except Exception:
                                err_text = None
                            if err_text:
                                for line in err_text.splitlines():
                                    try:
                                        rec = json.loads(line)
                                        errors[rec.get("custom_id")] = rec.get("error")
                                    except Exception:
                                        pass
                    for line in text_data.splitlines():
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        ident = rec.get("custom_id")
                        if not ident:
                            continue
                        if rec.get("response") is None:
                            err = rec.get("error") or errors.get(ident)
                            row = {
                                "Identifier": ident,
                                "Response": None,
                                "Web Search Sources": None,
                                "Time Taken": None,
                                "Input Tokens": None,
                                "Reasoning Tokens": None,
                                "Output Tokens": None,
                                "Reasoning Effort": get_response_kwargs.get(
                                    "reasoning_effort", reasoning_effort
                                ),
                                "Successful": False,
                                "Error Log": [err] if err else [],
                            }
                            if reasoning_summary is not None:
                                row["Reasoning Summary"] = None
                            completed_rows.append(row)
                            continue
                        resp_obj = rec["response"]
                        resp_text: Optional[str] = None
                        summary_text: Optional[str] = None
                        usage = {}
                        if isinstance(resp_obj, dict):
                            usage = resp_obj.get("usage", {}) or {}
                        input_tok = usage.get("input_tokens") if isinstance(usage, dict) else None
                        output_tok = usage.get("output_tokens") if isinstance(usage, dict) else None
                        reason_tok = None
                        if isinstance(usage, dict):
                            otd = usage.get("output_tokens_details") or {}
                            if isinstance(otd, dict):
                                reason_tok = otd.get("reasoning_tokens")
                        # Determine candidate payload
                        candidate = (
                            resp_obj.get("body", resp_obj)
                            if isinstance(resp_obj, dict)
                            else None
                        )
                        search_objs: List[Dict[str, Any]] = []
                        if isinstance(candidate, dict):
                            search_objs.append(candidate)
                        if isinstance(resp_obj, dict):
                            search_objs.append(resp_obj)
                        sources_data = _extract_web_search_sources(search_objs)
                        for obj in search_objs:
                            if resp_text is None and isinstance(
                                obj.get("output_text"), (str, bytes)
                            ):
                                resp_text = obj["output_text"]
                                break
                            if resp_text is None and isinstance(
                                obj.get("choices"), list
                            ):
                                choices = obj.get("choices")
                                if choices:
                                    choice = choices[0]
                                    if isinstance(choice, dict):
                                        msg = (
                                            choice.get("message")
                                            or choice.get("delta")
                                            or {}
                                        )
                                        if isinstance(msg, dict):
                                            content = msg.get("content")
                                            if isinstance(content, str):
                                                resp_text = content
                                                break
                                    if resp_text is None and isinstance(
                                        obj.get("output"), list
                                    ):
                                        out_list = obj.get("output")
                                        for item in out_list:
                                            if not isinstance(item, dict):
                                                continue
                                            content_list = item.get("content")
                                            if isinstance(content_list, list):
                                                for piece in content_list:
                                                    if (
                                                        isinstance(piece, dict)
                                                        and "text" in piece
                                                    ):
                                                        txt = piece.get("text")
                                                        if isinstance(txt, str):
                                                            resp_text = txt
                                                            break
                                                if resp_text is not None:
                                                    break
                                            if resp_text is None and isinstance(
                                                item.get("text"), str
                                            ):
                                                resp_text = item["text"]
                                                break
                                            if resp_text is not None:
                                                break
                                        if resp_text is not None:
                                            break
                        for obj in search_objs:
                            out_list = obj.get("output")
                            if isinstance(out_list, list):
                                for piece in out_list:
                                    if (
                                        isinstance(piece, dict)
                                        and piece.get("type") == "reasoning"
                                    ):
                                        summ = piece.get("summary")
                                        if isinstance(summ, list) and summ:
                                            first = summ[0]
                                            if isinstance(first, dict):
                                                txt = first.get("text")
                                                if isinstance(txt, str):
                                                    summary_text = txt
                                                    break
                                if summary_text is not None:
                                    break
                        row = {
                            "Identifier": ident,
                            "Response": [resp_text],
                            "Web Search Sources": sources_data,
                            "Time Taken": None,
                            "Input Tokens": input_tok,
                            "Reasoning Tokens": reason_tok,
                            "Output Tokens": output_tok,
                            "Reasoning Effort": get_response_kwargs.get(
                                "reasoning_effort", reasoning_effort
                            ),
                            "Successful": True,
                            "Error Log": [],
                        }
                        if reasoning_summary is not None:
                            row["Reasoning Summary"] = summary_text
                        completed_rows.append(row)
                    unfinished_batches.remove(b)
                    state["batches"] = [
                        bb
                        for bb in state.get("batches", [])
                        if bb.get("batch_id") != bid
                    ]
                    with open(state_path, "w") as f:
                        json.dump(state, f)
                elif status in {"failed", "cancelled", "expired"}:
                    logger.warning(f"Batch {bid} finished with status {status}.")
                    unfinished_batches.remove(b)
                    state["batches"] = [
                        bb
                        for bb in state.get("batches", [])
                        if bb.get("batch_id") != bid
                    ]
                    with open(state_path, "w") as f:
                        json.dump(state, f)
                else:
                    rc = job.request_counts
                    logger.info(
                        f"Batch {bid} in progress: {status}; completed {rc.completed}/{rc.total}."
                    )
            if unfinished_batches:
                await asyncio.sleep(batch_poll_interval)
        # Append and return
        _append_results(completed_rows)
        _report_cost()
        return df
    # Non‑batch path
    # Initialise limiters using the per‑minute budgets derived above.  These
    # limiters control the rate of API requests and the number of tokens
    # consumed per minute.  By setting the budgets based on your account’s
    # remaining limits (or sensible defaults when limits are unknown), we
    # ensure that tasks yield gracefully when the budget is exhausted rather
    # than overrunning the API’s quota.  We do not apply any dynamic
    # scaling factor here; concurrency has already been capped based on
    # the budgets and average prompt length.
    max_timeout_val = float("inf") if max_timeout is None else float(max_timeout)
    nonlocal_timeout: float = float("inf") if dynamic_timeout else max_timeout_val
    req_lim: Optional[AsyncLimiter] = None
    tok_lim: Optional[AsyncLimiter] = None
    if not use_batch and manage_rate_limits:
        req_lim = AsyncLimiter(allowed_req_pm, 60)
        tok_lim = AsyncLimiter(allowed_tok_pm, 60)
    success_times: List[float] = []
    timeout_initialized = False
    observed_latency_p90 = float("inf")
    inflight: Dict[str, Tuple[float, asyncio.Task, float]] = {}
    error_logs: Dict[str, List[str]] = defaultdict(list)
    call_count = 0
    samples_for_timeout = max(
        1,
        int(
            0.90
            * min(
                len(todo_pairs),
                max_parallel_ceiling,
                requested_n_parallels,
            )
        ),
    )
    queue: asyncio.Queue[Tuple[str, str, int]] = asyncio.Queue()
    for item in todo_pairs:
        queue.put_nowait((item[0], item[1], max_retries))
    results: List[Dict[str, Any]] = []
    processed = 0
    pbar = _progress_bar(
        total=len(todo_pairs),
        desc="Processing prompts",
        leave=True,
        verbose=verbose,
    )
    cooldown_until = 0.0
    stop_event = asyncio.Event()
    timeout_cancellations: Set[str] = set()
    seen_error_messages: Set[Hashable] = set()
    # Counters used for the gentle concurrency adaptation below
    rate_limit_errors_since_adjust = 0
    successes_since_adjust = 0
    active_workers = 0
    rate_limit_window = max(1.0, float(rate_limit_window))
    connection_error_window = max(1.0, float(connection_error_window))
    rate_limit_error_times: Deque[float] = deque()
    last_concurrency_scale_down = 0.0
    last_concurrency_scale_up = 0.0
    connection_error_times: Deque[float] = deque()
    connection_errors_since_adjust = 0
    last_connection_scale_down = 0.0
    usage_samples: List[Tuple[int, int, int]] = []
    estimated_output_tokens = planning_output_tokens
    estimated_output_tokens_per_prompt_live = float(planning_output_tokens)
    estimated_input_tokens_per_prompt = float(
        (dataset_stats.get("token_count") or 0) / max(1, len(prompts))
    )

    def _effective_parallel_ceiling() -> int:
        ceiling = max_parallel_ceiling
        if throughput_ceiling_ppm is not None:
            ceiling = min(ceiling, throughput_ceiling_ppm)
        return max(1, ceiling)

    observed_input_tokens_total = 0.0
    observed_output_tokens_total = 0.0
    observed_reasoning_tokens_total = 0.0
    observed_usage_count = 0
    estimate_update_target = max(1, min(_effective_parallel_ceiling(), len(todo_pairs)))
    estimate_update_done = False
    estimate_refresh_cooldown = 60.0
    last_estimate_refresh = 0.0
    timeout_error_times: Deque[float] = deque()
    limiter_wait_durations: Deque[float] = deque(maxlen=max(10, token_sample_size))
    limiter_wait_ratios: Deque[float] = deque(maxlen=max(10, token_sample_size))
    limiter_wait_ratio_threshold = 0.3
    limiter_wait_duration_threshold = 0.6
    wait_adjust_cooldown_up = max(20.0, rate_limit_window * 0.7)
    wait_adjust_cooldown_down = max(45.0, rate_limit_window * 1.25)
    wait_adjust_min_delta = max(1, int(math.ceil(max_parallel_ceiling * 0.04)))
    last_wait_adjust = 0.0
    timeout_notes: Deque[str] = deque(maxlen=3)
    timeout_errors_since_last_status = 0
    connection_errors_since_last_status = 0
    first_timeout_logged = False
    first_rate_limit_logged = False
    first_connection_logged = False
    current_tokens_per_call = float(estimated_tokens_per_call)
    last_rate_limit_concurrency_change = 0.0
    restart_requested = False
    restart_count = int(get_response_kwargs.pop("_timeout_restart_count", 0))
    timeout_burst_window = max(1.0, float(timeout_burst_window))
    timeout_burst_cooldown = max(1.0, float(timeout_burst_cooldown))
    timeout_burst_max_restarts = max(0, int(timeout_burst_max_restarts))

    def _maybe_reduce_output_headroom(*, allow_reduce: bool) -> bool:
        """Lower output headroom after the first token estimate refresh."""

        nonlocal output_headroom_live, output_headroom_reduced, estimated_output_tokens
        nonlocal estimated_tokens_per_call, current_tokens_per_call, throughput_ceiling_ppm
        nonlocal estimated_input_tokens_per_prompt
        if output_headroom_reduced or not allow_reduce:
            return False
        if observed_usage_count <= 0:
            return False
        previous_headroom = output_headroom_live
        output_headroom_live = OUTPUT_TOKEN_HEADROOM_STEADY
        output_headroom_reduced = True
        avg_output_observed = (
            observed_output_tokens_total + observed_reasoning_tokens_total
        ) / max(1, observed_usage_count)
        base_output_estimate = (
            avg_output_observed
            if avg_output_observed > 0
            else estimated_output_tokens / max(1.0, previous_headroom)
        )
        estimated_output_tokens = max(
            1.0,
            base_output_estimate * output_headroom_live,
        )
        estimated_tokens_per_call = max(
            1.0,
            (estimated_input_tokens_per_prompt + estimated_output_tokens) * max(1, n),
        )
        current_tokens_per_call = estimated_tokens_per_call
        planned_ppm_live, _ = _safe_planned_ppm_and_details(
            allowed_req_pm if manage_rate_limits else None,
            allowed_tok_pm if manage_rate_limits else None,
            current_tokens_per_call,
            context="updating token estimates",
        )
        if planned_ppm_live is not None:
            throughput_ceiling_ppm = planned_ppm_live
        msg = (
            f"[token headroom] Lowered output headroom from {previous_headroom:.2f} "
            f"to {output_headroom_live:.2f} after refreshing token estimates."
        )
        logger.info(msg)
        return True

    def _effective_timeout_burst_threshold() -> int:
        parallel_workers = max(1, max(active_workers, concurrency_cap))
        return int(math.ceil(parallel_workers * _TIMEOUT_BURST_RATIO))

    def _emit_first_error(
        message: str,
        *,
        level: int = logging.WARNING,
        dedup_key: Optional[Hashable] = None,
    ) -> bool:
        key = dedup_key if dedup_key is not None else message
        first_time = key not in seen_error_messages
        if first_time:
            seen_error_messages.add(key)
            formatted = (
                f"{message} (subsequent occurrences of this error will be silenced in logs)"
            )
            logger.log(level, formatted)
        else:
            logger.debug(message)

    def _log_timeout_once(message: str, note: str, *, dedup_key: Hashable) -> None:
        nonlocal first_timeout_logged
        if not first_timeout_logged:
            _emit_first_error(message, dedup_key=dedup_key)
            logger.warning(
                "First timeout encountered (%s). Subsequent timeouts will be summarised in periodic status updates.",
                note,
            )
            first_timeout_logged = True
        else:
            logger.debug("Timeout error: %s", message)

    def _log_rate_limit_once(detail: Optional[str] = None) -> None:
        nonlocal first_rate_limit_logged
        if not first_rate_limit_logged:
            logger.warning(
                "Encountered first rate limit error. Future rate limit errors will be silenced and tracked in periodic updates."
            )
            first_rate_limit_logged = True
        else:
            if detail:
                logger.debug("Rate limit error: %s", detail)
            else:
                logger.debug("Rate limit error encountered.")

    def _log_connection_once(detail: Optional[str] = None) -> None:
        nonlocal first_connection_logged
        if not first_connection_logged:
            logger.warning(
                "Encountered first connection error. Future connection errors will be silenced and tracked in periodic updates."
            )
            first_connection_logged = True
        else:
            if detail:
                logger.debug("Connection error: %s", detail)
            else:
                logger.debug("Connection error encountered.")

    def _record_timeout_event(now: Optional[float] = None) -> None:
        ts = now if now is not None else time.time()
        timeout_error_times.append(ts)
        window_start = ts - timeout_burst_window
        while timeout_error_times and timeout_error_times[0] < window_start:
            timeout_error_times.popleft()

    def _trigger_timeout_burst(now: Optional[float] = None) -> None:
        nonlocal restart_requested
        _record_timeout_event(now)
        threshold = _effective_timeout_burst_threshold()
        if (
            restart_requested
            or len(timeout_error_times) < threshold
            or restart_count >= timeout_burst_max_restarts
        ):
            return
        restart_requested = True
        stop_event.set()
        msg = (
            f"[timeouts] {len(timeout_error_times)} timeouts in the last {int(timeout_burst_window)}s "
            f"(threshold={threshold}). "
            f"Pausing workers for {int(timeout_burst_cooldown)}s and resuming from the last checkpoint."
        )
        if message_verbose:
            print(msg)
        logger.warning(msg)
        for _, (_, task, _) in list(inflight.items()):
            try:
                task.cancel()
            except Exception:
                continue
        drained = 0
        while True:
            try:
                _, _, _ = queue.get_nowait()
                queue.task_done()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained and message_verbose:
            print(f"[timeouts] Drained {drained} queued prompts before restart.")

    def _cost_progress_snapshot() -> Optional[Tuple[float, bool]]:
        """Return total cost so far and whether the value is sampled."""

        pricing = _lookup_model_pricing(model)
        if not pricing:
            return None
        frames: List[pd.DataFrame] = []
        if not df.empty:
            frames.append(df[["Input Tokens", "Output Tokens", "Reasoning Tokens"]].copy())
        if results:
            temp = pd.DataFrame(results)
            for col in ("Input Tokens", "Output Tokens", "Reasoning Tokens"):
                if col not in temp.columns:
                    temp[col] = 0
            frames.append(temp[["Input Tokens", "Output Tokens", "Reasoning Tokens"]])
        if not frames:
            return (0.0, False)
        combined = pd.concat(frames, ignore_index=True)
        total_rows = len(combined)
        if total_rows == 0:
            return (0.0, False)
        sample = False
        scale = 1.0
        if total_rows > 50000:
            sample = True
            sampled = combined.sample(n=min(5000, total_rows), random_state=42)
            scale = total_rows / max(1, len(sampled))
            combined = sampled
        inp = pd.to_numeric(combined["Input Tokens"], errors="coerce").fillna(0)
        out = pd.to_numeric(combined["Output Tokens"], errors="coerce").fillna(0)
        reason = pd.to_numeric(combined["Reasoning Tokens"], errors="coerce").fillna(0)
        total_cost = (
            (inp / 1_000_000) * pricing["input"]
            + ((out + reason) / 1_000_000) * pricing["output"]
        ).sum()
        return (float(total_cost) * scale, sample)

    def _aggregate_usage(raw_items: List[Any]) -> Tuple[int, int, int]:
        total_in = total_out = total_reason = 0
        for item in raw_items:
            usage = _safe_get(item, "usage")
            if usage is None:
                continue
            input_tokens = _safe_get(usage, "input_tokens")
            if input_tokens in (None, 0):
                input_tokens = _safe_get(usage, "prompt_tokens")
            output_tokens = _safe_get(usage, "output_tokens")
            if output_tokens in (None, 0):
                output_tokens = _safe_get(usage, "completion_tokens")
            details = _safe_get(usage, "output_tokens_details")
            if isinstance(details, dict):
                reasoning_tokens = details.get("reasoning_tokens") or 0
            else:
                reasoning_tokens = _safe_get(details, "reasoning_tokens", 0)
            try:
                total_in += int(input_tokens or 0)
            except Exception:
                pass
            try:
                total_out += int(output_tokens or 0)
            except Exception:
                pass
            try:
                total_reason += int(reasoning_tokens or 0)
            except Exception:
                pass
        return total_in, total_out, total_reason

    concurrency_cap = min(concurrency_cap, _effective_parallel_ceiling())

    def emit_parallelization_status(
        reason: str, *, force: bool = False, label: Optional[str] = "parallelization"
    ) -> None:
        """Print and log a snapshot of the current worker utilisation."""

        if not force and status_report_interval is None and not verbose:
            return
        if quiet and not force:
            return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        reason_clean = reason.rstrip(" .:;")
        planned_ppm, throughput_details = _safe_planned_ppm_and_details(
            allowed_req_pm if manage_rate_limits else None,
            allowed_tok_pm if manage_rate_limits else None,
            current_tokens_per_call,
            context="reporting parallelization status",
        )
        timeout_text = ""
        connection_text = ""
        total_completed = processed
        if status.num_timeout_errors or total_completed:
            denom = max(total_completed, 1)
            timeout_text = f"timeouts={status.num_timeout_errors}/{denom}"
            if status_report_interval is not None and timeout_errors_since_last_status:
                timeout_text += f" (+{timeout_errors_since_last_status} since last)"
            burst_alert_threshold = _effective_timeout_burst_threshold()
            if timeout_errors_since_last_status >= burst_alert_threshold:
                burst_msg = (
                    f"[timeouts] {timeout_errors_since_last_status} timeouts since last update; "
                    "consider reducing concurrency or checking network stability if this persists."
                )
                logger.warning(burst_msg)
                if message_verbose:
                    print(burst_msg)
        cost_text = ""
        cost_snapshot = _cost_progress_snapshot()
        if cost_snapshot is not None:
            total_cost, sampled = cost_snapshot
            cost_text = f"cost_so_far={'~' if sampled else ''}${total_cost:.2f}"
        prefix = f"[{label}] " if label else ""
        if status.num_connection_errors or connection_errors_since_last_status:
            connection_text = f"connection_errors={status.num_connection_errors}"
            if status_report_interval is not None and connection_errors_since_last_status:
                connection_text += f" (+{connection_errors_since_last_status} since last)"
        status_bits: List[str] = [
            f"cap={concurrency_cap}",
            f"active={active_workers}",
            f"inflight={len(inflight)}",
            f"queue={queue.qsize()}",
            f"processed={processed}/{status.num_tasks_started}",
            f"rate_limit_errors={status.num_rate_limit_errors}",
        ]
        if cost_text:
            status_bits.insert(0, cost_text)
        if planned_ppm is not None:
            ppm_piece = f"throughput<={planned_ppm} prompts/min"
            status_bits.append(ppm_piece)
        if timeout_text:
            status_bits.append(timeout_text)
        if connection_text:
            status_bits.append(connection_text)
        msg = f"{prefix}{timestamp} | {reason_clean}: " + ", ".join(status_bits)
        if message_verbose:
            print(msg)
        logger.info(msg)

    emit_parallelization_status("Initial parallelization settings", force=True)

    def _maybe_refresh_estimates(trigger_reason: str, *, force: bool = False) -> bool:
        """Update token and cost estimates from observed usage."""

        nonlocal estimated_output_tokens, estimated_tokens_per_call, current_tokens_per_call
        nonlocal estimated_output_tokens_per_prompt_live, estimated_input_tokens_per_prompt
        nonlocal throughput_ceiling_ppm, estimate_update_done, last_estimate_refresh
        nonlocal output_headroom_live, output_headroom_reduced
        now = time.time()
        if not force and (now - last_estimate_refresh) < estimate_refresh_cooldown:
            return False
        if observed_usage_count <= 0:
            return False
        _maybe_reduce_output_headroom(allow_reduce=force)
        avg_input = observed_input_tokens_total / max(1, observed_usage_count)
        avg_output = (
            (observed_output_tokens_total + observed_reasoning_tokens_total)
            / max(1, observed_usage_count)
        )
        if avg_input <= 0 and avg_output <= 0:
            return False
        estimated_input_tokens_per_prompt = max(1.0, avg_input)
        if avg_output > 0:
            estimated_output_tokens_per_prompt_live = max(1.0, avg_output)
            estimated_output_tokens = max(1.0, avg_output * output_headroom_live)
        updated_tokens_per_call = max(
            1.0,
            (estimated_input_tokens_per_prompt + estimated_output_tokens) * max(1, n),
        )
        estimated_tokens_per_call = updated_tokens_per_call
        current_tokens_per_call = updated_tokens_per_call
        planned_ppm_live, _ = _safe_planned_ppm_and_details(
            allowed_req_pm if manage_rate_limits else None,
            allowed_tok_pm if manage_rate_limits else None,
            current_tokens_per_call,
            context="refreshing token estimates",
        )
        if planned_ppm_live is not None:
            throughput_ceiling_ppm = planned_ppm_live
        pricing = _lookup_model_pricing(model)
        cost_line = None
        total_rows = max(1, status.num_tasks_started)
        updated_parallel_workers = min(
            concurrency_cap,
            throughput_ceiling_ppm if throughput_ceiling_ppm is not None else concurrency_cap,
        )
        updated_parallel_workers = max(1, int(updated_parallel_workers))
        if pricing is not None and total_rows:
            total_input_tokens = estimated_input_tokens_per_prompt * max(1, n) * total_rows
            total_output_tokens = max(avg_output, 0.0) * max(1, n) * total_rows
            input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
            output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
            batch_multiplier = pricing.get("batch", 1.0) if use_batch else 1.0
            input_cost *= batch_multiplier
            output_cost *= batch_multiplier
            cost_line = (
                f"Updated estimated total cost: ~${(input_cost + output_cost):.2f} "
                f"(input ${input_cost:.2f}, output ${output_cost:.2f}). "
                f"Updated parallel threads: {updated_parallel_workers} based on refreshed "
                "token usage."
            )
        msg_lines = [
            "[token estimate] Refreshed per-prompt estimates from observed usage "
            f"({observed_usage_count} sample{'' if observed_usage_count == 1 else 's'}): "
            f"input ≈ {int(round(estimated_input_tokens_per_prompt)):,} tokens, "
            f"output (incl. reasoning) ≈ {int(round(estimated_output_tokens_per_prompt_live)):,} tokens."
        ]
        if cost_line:
            msg_lines.append("[token estimate] " + cost_line)
            if _is_multimodal_estimate(
                modality=inferred_modality,
                web_search=web_search_requested,
                has_media=has_media_payloads,
            ):
                msg_lines.append(
                    "[token estimate] Note: multimedia/web inputs can make cost estimates unreliable. "
                    "Monitor usage in the OpenAI dashboard."
                )
            if planned_ppm_live is not None and planned_ppm_live > 0:
                remaining_prompts = max(0, status.num_tasks_started - processed)
                estimated_minutes = math.ceil(remaining_prompts / planned_ppm_live)
                minimum_minutes = max(1, estimated_minutes)
                msg_lines.append(
                    "[token estimate] Updated time estimate: minimum of "
                    f"{minimum_minutes} minute{'s' if minimum_minutes != 1 else ''}. Moving to a higher usage tier raises rate limits and allows much faster runs."
                )
        if not quiet:
            for line in msg_lines:
                print(line)
        for line in msg_lines:
            logger.info(line)
        if force:
            estimate_update_done = True
        last_estimate_refresh = now
        return True

    def _maybe_trigger_threshold_refresh() -> None:
        if not estimate_update_done and processed >= estimate_update_target:
            _maybe_refresh_estimates("parallel-cap reached", force=True)

    async def flush() -> None:
        nonlocal results, df, processed, csv_header_written, written_identifiers
        if results:
            batch_df = pd.DataFrame(results)
            if "Web Search Sources" not in batch_df.columns:
                batch_df["Web Search Sources"] = pd.NA
            batch_df = batch_df[~batch_df["Identifier"].isin(written_identifiers)]
            if not batch_df.empty:
                to_save = batch_df.copy()
                for col in ("Response", "Error Log", "Web Search Sources"):
                    if col in to_save:
                        to_save[col] = to_save[col].apply(_ser)
                to_save.to_csv(
                    save_path,
                    mode="a" if csv_header_written else "w",
                    header=not csv_header_written,
                    index=False,
                    quoting=csv.QUOTE_MINIMAL,
                )
                csv_header_written = True
                if df.empty:
                    df = batch_df.reset_index(drop=True)
                else:
                    df = pd.concat([df, batch_df], ignore_index=True)
                written_identifiers.update(batch_df["Identifier"])
            results = []
        if logger.isEnabledFor(logging.INFO) and processed:
            logger.info(
                f"Processed {processed}/{status.num_tasks_started} prompts; "
                f"failures: {status.num_tasks_failed} "
                f"(timeouts: {status.num_timeout_errors}, "
                f"rate limits: {status.num_rate_limit_errors}, "
                f"API: {status.num_api_errors}, other: {status.num_other_errors})"
            )

    async def adjust_timeout() -> None:
        nonlocal nonlocal_timeout, timeout_initialized, observed_latency_p90
        if not dynamic_timeout:
            return
        if len(success_times) < samples_for_timeout:
            return
        try:
            p90 = float(np.percentile(success_times, 90))
            observed_latency_p90 = p90
            new_timeout = min(max_timeout_val, timeout_factor * p90)
            if math.isinf(nonlocal_timeout):
                nonlocal_timeout = new_timeout
                if not timeout_initialized:
                    timeout_display = (
                        "inf"
                        if math.isinf(nonlocal_timeout)
                        else f"{nonlocal_timeout:.1f}s"
                    )
                    p90_display = (
                        "inf" if math.isinf(p90) else f"{p90:.1f}s"
                    )
                    msg = (
                        "[dynamic timeout] Initialized timeout to "
                        f"{timeout_display} (p90={p90_display}, factor={timeout_factor:.2f})."
                    )
                    print(msg)
                    logger.info(msg)
                    timeout_initialized = True
            elif new_timeout > nonlocal_timeout:
                p90_display = "inf" if math.isinf(p90) else f"{p90:.1f}s"
                logger.debug(
                    "[dynamic timeout] Updating timeout to %s (p90=%s, factor=%.2f).",
                    "inf" if math.isinf(new_timeout) else f"{new_timeout:.1f}s",
                    p90_display,
                    timeout_factor,
                )
                nonlocal_timeout = new_timeout
            if not math.isinf(nonlocal_timeout):
                now = time.time()
                for ident, (start, task, t_out) in list(inflight.items()):
                    limit = _resolve_effective_timeout(
                        nonlocal_timeout, t_out, dynamic_timeout
                    )
                    if now - start > limit and not task.done():
                        timeout_cancellations.add(ident)
                        task.cancel()
        except Exception:
            pass

    # The per‑minute AsyncLimiter budgets remain fixed.  When rate limits are
    # hit we only adapt the number of in‑flight worker tasks without
    # rebuilding the limiters themselves, keeping the gating logic simple.
    async def rebuild_limiters() -> None:
        return None

    def maybe_adjust_concurrency() -> None:
        nonlocal concurrency_cap, rate_limit_errors_since_adjust, successes_since_adjust, max_parallel_ceiling, last_concurrency_scale_down, last_concurrency_scale_up, last_rate_limit_concurrency_change, throughput_ceiling_ppm
        if not manage_rate_limits:
            return
        now = time.time()
        window_start = now - rate_limit_window
        while rate_limit_error_times and rate_limit_error_times[0] < window_start:
            rate_limit_error_times.popleft()
        recent_errors = len(rate_limit_error_times)
        error_window_threshold = max(6, int(math.ceil(concurrency_cap * 0.08)))
        consecutive_threshold = max(3, int(math.ceil(concurrency_cap * 0.05)))
        should_scale_down = False
        if recent_errors >= error_window_threshold:
            should_scale_down = True
        elif rate_limit_errors_since_adjust >= consecutive_threshold:
            should_scale_down = True
        if should_scale_down and (now - last_concurrency_scale_down) >= max(1.0, rate_limit_window * 0.75):
            decrement = _rate_limit_decrement(concurrency_cap)
            new_cap = max(1, concurrency_cap - decrement)
            if new_cap != concurrency_cap:
                old_cap = concurrency_cap
                concurrency_cap = new_cap
                reason = (
                    f"[rate-limit recovery] Cutting workers from {old_cap} to {new_cap} "
                    f"after {recent_errors} rate-limit errors in the last {int(round(rate_limit_window))}s."
                )
                logger.warning(reason)
                emit_parallelization_status(reason, force=True)
            else:
                concurrency_cap = new_cap
            rate_limit_errors_since_adjust = 0
            successes_since_adjust = 0
            rate_limit_error_times.clear()
            last_concurrency_scale_down = now
            last_rate_limit_concurrency_change = now
            return
        quiet_since_last_error = (now - status.time_of_last_rate_limit_error) >= rate_limit_window
        ceiling_cap = _effective_parallel_ceiling()
        if (
            rate_limit_errors_since_adjust == 0
            and concurrency_cap < ceiling_cap
            and quiet_since_last_error
            and (now - last_concurrency_scale_down) >= rate_limit_window
            and (now - last_concurrency_scale_up) >= rate_limit_window
        ):
            growth_headroom_limit = max(1, int(math.floor(ceiling_cap * 0.9)))
            success_threshold = max(50, int(math.ceil(concurrency_cap * 1.5)))
            if (
                concurrency_cap < growth_headroom_limit
                and successes_since_adjust >= success_threshold
            ):
                increment = max(1, int(math.ceil(max(concurrency_cap * 0.08, 1))))
                new_cap = min(ceiling_cap, concurrency_cap + increment)
                if new_cap != concurrency_cap:
                    old_cap = concurrency_cap
                    concurrency_cap = new_cap
                    reason = (
                        f"[rate-limit recovery] Increasing workers from {old_cap} to {new_cap} after sustained success."
                    )
                    logger.info(reason)
                    emit_parallelization_status(reason, force=True)
                    last_concurrency_scale_up = now
                    last_rate_limit_concurrency_change = now
                else:
                    concurrency_cap = new_cap
                successes_since_adjust = 0
                rate_limit_errors_since_adjust = 0

    def maybe_adjust_for_connection_errors() -> None:
        nonlocal concurrency_cap, connection_errors_since_adjust, last_connection_scale_down
        nonlocal last_rate_limit_concurrency_change
        now = time.time()
        window_start = now - connection_error_window
        while connection_error_times and connection_error_times[0] < window_start:
            connection_error_times.popleft()
        recent_errors = len(connection_error_times)
        error_window_threshold = max(3, int(math.ceil(concurrency_cap * 0.06)))
        consecutive_threshold = max(2, int(math.ceil(concurrency_cap * 0.04)))
        should_scale_down = False
        if recent_errors >= error_window_threshold:
            should_scale_down = True
        elif connection_errors_since_adjust >= consecutive_threshold:
            should_scale_down = True
        if should_scale_down and (
            (now - last_connection_scale_down) >= max(1.0, connection_error_window * 0.75)
        ):
            decrement = _connection_error_decrement(concurrency_cap)
            new_cap = max(1, concurrency_cap - decrement)
            if new_cap != concurrency_cap:
                old_cap = concurrency_cap
                concurrency_cap = new_cap
                reason = (
                    f"[network recovery] Cutting workers from {old_cap} to {new_cap} "
                    f"after {recent_errors} connection errors in the last "
                    f"{int(round(connection_error_window))}s. "
                    "If this persists, check network stability or bandwidth limits, "
                    "or reduce `n_parallels`."
                )
                logger.warning(reason)
                emit_parallelization_status(reason, force=True)
            connection_errors_since_adjust = 0
            connection_error_times.clear()
            last_connection_scale_down = now
            last_rate_limit_concurrency_change = now

    async def worker() -> None:
        nonlocal processed, call_count, nonlocal_timeout, active_workers, concurrency_cap, cooldown_until
        nonlocal estimated_output_tokens, rate_limit_errors_since_adjust, successes_since_adjust, stop_event
        nonlocal max_parallel_ceiling, last_wait_adjust, current_tokens_per_call, timeout_errors_since_last_status
        nonlocal throughput_ceiling_ppm, observed_input_tokens_total, observed_output_tokens_total
        nonlocal observed_reasoning_tokens_total, observed_usage_count, connection_errors_since_adjust
        nonlocal connection_errors_since_last_status
        while True:
            if stop_event.is_set():
                break
            try:
                prompt, ident, attempts_left = await queue.get()
            except asyncio.CancelledError:
                break
            if stop_event.is_set():
                queue.task_done()
                break
            try:
                async def _maybe_retry(backoff: float) -> bool:
                    if restart_requested or stop_event.is_set():
                        return False
                    await asyncio.sleep(backoff)
                    queue.put_nowait((prompt, ident, attempts_left - 1))
                    return True

                now = time.time()
                if now < cooldown_until:
                    await asyncio.sleep(cooldown_until - now)
                while active_workers >= concurrency_cap:
                    await asyncio.sleep(0.01)
                active_workers += 1
                input_tokens = len(tokenizer.encode(prompt))
                gating_output = estimated_output_tokens
                limiter_wait_time = 0.0
                sources_data: Optional[List[Any]] = None
                if req_lim is not None:
                    wait_start = time.perf_counter()
                    await req_lim.acquire()
                    limiter_wait_time += time.perf_counter() - wait_start
                if tok_lim is not None:
                    wait_start = time.perf_counter()
                    await tok_lim.acquire((input_tokens + gating_output) * n)
                    limiter_wait_time += time.perf_counter() - wait_start
                call_count += 1
                error_logs.setdefault(ident, [])
                start = time.time()
                base_timeout = nonlocal_timeout
                multiplier = 1.5 ** (max_retries - attempts_left)
                call_timeout = None if math.isinf(base_timeout) else base_timeout * multiplier
                if call_timeout is not None and dynamic_timeout and not math.isinf(max_timeout_val):
                    call_timeout = min(call_timeout, max_timeout_val)
                images_payload = prompt_images.get(str(ident)) if prompt_images else None
                audio_payload = prompt_audio.get(str(ident)) if prompt_audio else None
                pdf_payload = prompt_pdfs.get(str(ident)) if prompt_pdfs else None
                call_kwargs = dict(get_response_kwargs)
                per_prompt_filters = (
                    prompt_web_search_filters.get(str(ident))
                    if prompt_web_search_filters
                    else None
                )
                merged_filters = _merge_web_search_filters(
                    base_web_search_filters, per_prompt_filters
                )
                if merged_filters is not None:
                    call_kwargs["web_search_filters"] = merged_filters
                else:
                    call_kwargs.pop("web_search_filters", None)
                if images_payload is not None:
                    call_kwargs["images"] = images_payload
                if audio_payload is not None:
                    call_kwargs["audio"] = audio_payload
                if pdf_payload is not None:
                    call_kwargs["pdfs"] = pdf_payload
                call_kwargs.update(
                    {
                        "n": n,
                        "timeout": call_timeout,
                        "use_dummy": use_dummy,
                    }
                )
                if using_custom_response_fn and provided_api_key is not None:
                    call_kwargs.setdefault("api_key", provided_api_key)
                if response_accepts_return_raw:
                    call_kwargs.setdefault("return_raw", True)
                else:
                    call_kwargs.pop("return_raw", None)
                if not response_accepts_var_kw:
                    call_kwargs = {
                        k: v for k, v in call_kwargs.items() if k in response_param_names
                    }
                if prompt_call_via_keyword:
                    call_kwargs = dict(call_kwargs)
                    call_kwargs["prompt"] = prompt
                    call_args = ()
                else:
                    call_args = (prompt,)
                task = asyncio.create_task(response_callable(*call_args, **call_kwargs))
                inflight[ident] = (
                    start,
                    task,
                    call_timeout if call_timeout is not None else float("inf"),
                )
                response_ids: List[str] = []
                try:
                    result = await task
                except asyncio.CancelledError:
                    inflight.pop(ident, None)
                    if ident in timeout_cancellations:
                        timeout_cancellations.discard(ident)
                        if call_timeout is None or math.isinf(call_timeout):
                            raise asyncio.TimeoutError("API call exceeded timeout")
                        raise asyncio.TimeoutError(
                            f"API call timed out after {call_timeout} s"
                        )
                    raise
                inflight.pop(ident, None)
                resps, duration, raw = _normalize_response_result(result)
                sources_data = _extract_web_search_sources(raw)
                success_override: Optional[bool] = None
                if use_dummy:
                    selected_spec = dummy_response_specs.get(str(ident))
                    if selected_spec is None:
                        selected_spec = dummy_default_spec
                    auto_spec: Optional[DummyResponseSpec] = None
                    if not raw:
                        auto_spec = _auto_dummy_usage(prompt, resps)
                    selected_spec = _merge_dummy_specs(selected_spec, auto_spec)
                    if selected_spec is not None:
                        override_responses = selected_spec.responses
                        if override_responses is not None:
                            resps = _coerce_to_list(override_responses)
                        if selected_spec.duration is not None:
                            duration = selected_spec.duration
                        if selected_spec.warning:
                            logger.warning(selected_spec.warning)
                        extra_errors = _listify_error_log(selected_spec.error_log)
                        if extra_errors:
                            error_logs[ident].extend(extra_errors)
                        raw = _synthesise_dummy_raw(str(ident), selected_spec, resps)
                        success_override = selected_spec.successful
                limiter_wait_ratio = 0.0
                if (
                    limiter_wait_time > 0
                    and duration is not None
                    and duration > 0
                ):
                    limiter_wait_ratio = min(
                        1.0, limiter_wait_time / max(duration, 1e-6)
                    )
                limiter_wait_durations.append(limiter_wait_time)
                limiter_wait_ratios.append(limiter_wait_ratio)
                new_cap = concurrency_cap
                total_input, total_output, total_reasoning = _aggregate_usage(raw)
                for item in _coerce_to_list(raw):
                    rid = _safe_get(item, "id")
                    if rid:
                        response_ids.append(rid)
                summary_text = None
                try:
                    for r in raw:
                        out_items = _coerce_to_list(_safe_get(r, "output", []))
                        if not out_items:
                            continue
                        for item in out_items:
                            if _safe_get(item, "type") == "reasoning":
                                summary_list = _coerce_to_list(
                                    _safe_get(item, "summary", [])
                                )
                                if summary_list:
                                    txt = _safe_get(summary_list[0], "text")
                                    if isinstance(txt, str):
                                        summary_text = txt
                                        break
                        if summary_text is not None:
                            break
                except Exception:
                    summary_text = None
                usage_samples.append((total_input, total_output, total_reasoning))
                if len(usage_samples) > token_sample_size:
                    usage_samples.pop(0)
                if any(val > 0 for val in (total_input, total_output, total_reasoning)):
                    observed_input_tokens_total += float(total_input)
                    observed_output_tokens_total += float(total_output)
                    observed_reasoning_tokens_total += float(total_reasoning)
                    observed_usage_count += 1
                try:
                    if manage_rate_limits and len(usage_samples) >= token_sample_size:
                        avg_in = statistics.mean(u[0] for u in usage_samples)
                        avg_out = statistics.mean(u[1] for u in usage_samples)
                        avg_reason = statistics.mean(u[2] for u in usage_samples)
                        observed_output = avg_out + avg_reason
                        if observed_output > 0:
                            estimated_output_tokens = max(
                                1.0, observed_output * output_headroom_live
                            )
                        tokens_per_call_est = (
                            avg_in + max(estimated_output_tokens, observed_output)
                        ) * max(1, n)
                        current_tokens_per_call = max(1.0, tokens_per_call_est)
                        planned_ppm_live, _ = _safe_planned_ppm_and_details(
                            allowed_req_pm if manage_rate_limits else None,
                            allowed_tok_pm if manage_rate_limits else None,
                            current_tokens_per_call,
                            context="tuning concurrency",
                        )
                        if planned_ppm_live is not None:
                            throughput_ceiling_ppm = planned_ppm_live
                        ceiling_cap = _effective_parallel_ceiling()
                        concurrency_cap = min(concurrency_cap, ceiling_cap)
                        token_limited = int(
                            max(1, allowed_tok_pm // max(1, tokens_per_call_est))
                        )
                        req_limited = int(max(1, allowed_req_pm))
                        new_cap = min(ceiling_cap, req_limited, token_limited)
                        if new_cap < 1:
                            new_cap = 1
                        now = time.time()
                        last_connection_error = (
                            connection_error_times[-1] if connection_error_times else 0.0
                        )
                        safe_to_increase = (
                            (now - status.time_of_last_rate_limit_error) >= rate_limit_window
                            and (now - last_connection_error) >= connection_error_window
                        )
                        if new_cap > concurrency_cap:
                            max_increase = max(1, int(math.ceil(concurrency_cap * 0.12)))
                            if not safe_to_increase:
                                new_cap = concurrency_cap
                            else:
                                new_cap = min(new_cap, concurrency_cap + max_increase)
                        limiter_pressure = False
                        if (
                            limiter_wait_ratio >= limiter_wait_ratio_threshold
                            or limiter_wait_time >= limiter_wait_duration_threshold
                        ):
                            limiter_pressure = True
                        else:
                            sample_count = len(limiter_wait_durations)
                            min_samples = max(5, min(token_sample_size, 20))
                            if sample_count >= min_samples:
                                try:
                                    avg_ratio = statistics.mean(limiter_wait_ratios)
                                    avg_wait = statistics.mean(limiter_wait_durations)
                                except statistics.StatisticsError:
                                    avg_ratio = 0.0
                                    avg_wait = 0.0
                                high_ratio_events = sum(
                                    1
                                    for r in limiter_wait_ratios
                                    if r >= limiter_wait_ratio_threshold
                                )
                                high_wait_events = sum(
                                    1
                                    for d in limiter_wait_durations
                                    if d >= limiter_wait_duration_threshold
                                )
                                limiter_pressure = (
                                    avg_ratio >= limiter_wait_ratio_threshold
                                    or avg_wait >= limiter_wait_duration_threshold
                                    or high_ratio_events >= max(3, math.ceil(sample_count * 0.35))
                                    or high_wait_events >= max(3, math.ceil(sample_count * 0.35))
                                )
                        if new_cap < concurrency_cap and not limiter_pressure:
                            logger.debug(
                                "[throughput tuning] Computed concurrency cap %d but keeping %d since limiter waits (%.2fs, %.0f%%) remain below thresholds.",
                                new_cap,
                                concurrency_cap,
                                limiter_wait_time,
                                limiter_wait_ratio * 100,
                            )
                            new_cap = concurrency_cap
                        smoothed_cap, last_wait_adjust, changed = _smooth_wait_based_cap(
                            concurrency_cap,
                            new_cap,
                            now=now,
                            last_adjust=last_wait_adjust,
                            limiter_pressure=limiter_pressure,
                            min_delta=wait_adjust_min_delta,
                            cooldown_up=wait_adjust_cooldown_up,
                            cooldown_down=wait_adjust_cooldown_down,
                        )
                        if smoothed_cap > concurrency_cap and (now - last_rate_limit_concurrency_change) < max(rate_limit_window, wait_adjust_cooldown_up):
                            smoothed_cap = concurrency_cap
                            changed = False
                        if changed:
                            old_cap = concurrency_cap
                            concurrency_cap = smoothed_cap
                            direction = "Raising" if concurrency_cap > old_cap else "Lowering"
                            reason = (
                                f"[throughput tuning] {direction} worker cap from {old_cap} to {concurrency_cap} "
                                f"based on limiter waits (recent wait ≈ {limiter_wait_time:.2f}s, {limiter_wait_ratio * 100:.0f}% of call)."
                            )
                            significant = abs(concurrency_cap - old_cap) >= max(
                                wait_adjust_min_delta, int(math.ceil(old_cap * 0.15)), 3
                            )
                            if significant:
                                logger.info(reason)
                                emit_parallelization_status(reason, force=True)
                            else:
                                logger.debug(reason)
                except Exception as e:
                    _emit_first_error(
                        f"Error while updating concurrency cap dynamically: {e}",
                        level=logging.ERROR,
                        dedup_key=("cap-adjustment", type(e).__name__, str(e)),
                    )
                    logger.debug("Throughput tuning failed; retaining existing cap.", exc_info=True)
                concurrency_cap = new_cap
                if resps and all((isinstance(r, str) and not r.strip()) for r in resps):
                    if call_timeout is not None:
                        elapsed = time.time() - start
                        warning_msg = f"Timeout for {ident} after {call_timeout:.1f}s."
                        status.num_timeout_errors += 1
                        timeout_errors_since_last_status += 1
                        timeout_note = f"{ident} timed out after {elapsed:.2f}s"
                        timeout_notes.append(timeout_note)
                        _log_timeout_once(
                            warning_msg,
                            timeout_note,
                            dedup_key=("timeout", "call-timeout"),
                        )
                        error_logs[ident].append(warning_msg)
                        if attempts_left - 1 > 0:
                            backoff = random.uniform(1, 2) * (
                                2 ** (max_retries - attempts_left)
                            )
                            await _maybe_retry(backoff)
                        else:
                            row = {
                                "Identifier": ident,
                                "Response": None,
                                "Web Search Sources": sources_data,
                                "Time Taken": duration,
                                "Input Tokens": total_input,
                                "Reasoning Tokens": total_reasoning,
                                "Output Tokens": total_output,
                                "Reasoning Effort": get_response_kwargs.get(
                                    "reasoning_effort", reasoning_effort
                                ),
                                "Successful": False,
                                "Error Log": error_logs.get(ident, []),
                            }
                            if response_ids:
                                row["Response IDs"] = response_ids
                            if reasoning_summary is not None:
                                row["Reasoning Summary"] = None
                            results.append(row)
                            processed += 1
                            status.num_tasks_failed += 1
                            status.num_tasks_in_progress -= 1
                            pbar.update(1)
                            error_logs.pop(ident, None)
                            await flush()
                        continue
                json_mode_active = bool(get_response_kwargs.get("json_mode", False))
                should_validate_json = (
                    json_mode_active and not use_dummy and not using_custom_response_fn
                )
                if should_validate_json:
                    invalid_payloads = []
                    for resp in _coerce_to_list(resps):
                        _, ok = parse_json_with_status(resp)
                        if not ok:
                            invalid_payloads.append(resp)
                    if invalid_payloads:
                        snippet = str(invalid_payloads[0])[:200]
                        raise JSONParseError(
                            f"JSON parsing failed for identifier {ident}.",
                            snippet=snippet,
                        )
                row = {
                    "Identifier": ident,
                    "Response": resps,
                    "Web Search Sources": sources_data,
                    "Time Taken": duration,
                    "Input Tokens": total_input,
                    "Reasoning Tokens": total_reasoning,
                    "Output Tokens": total_output,
                    "Reasoning Effort": get_response_kwargs.get(
                        "reasoning_effort", reasoning_effort
                    ),
                    "Error Log": error_logs.get(ident, []),
                }
                if response_ids:
                    row["Response IDs"] = response_ids
                if reasoning_summary is not None:
                    row["Reasoning Summary"] = summary_text
                is_success = True if success_override is None else bool(success_override)
                if is_success and duration is not None:
                    success_times.append(duration)
                    await adjust_timeout()
                row["Successful"] = is_success
                results.append(row)
                processed += 1
                status.num_tasks_in_progress -= 1
                pbar.update(1)
                error_logs.pop(ident, None)
                if is_success:
                    status.num_tasks_succeeded += 1
                    successes_since_adjust += 1
                    rate_limit_errors_since_adjust = 0
                    connection_errors_since_adjust = 0
                    maybe_adjust_concurrency()
                    _maybe_trigger_threshold_refresh()
                    if processed % save_every_x_responses == 0:
                        await flush()
                else:
                    status.num_tasks_failed += 1
                    _maybe_trigger_threshold_refresh()
                    await flush()
            except asyncio.CancelledError:
                raise
            except (asyncio.TimeoutError, APITimeoutError) as e:
                elapsed = time.time() - start
                error_detail = str(e).strip()
                detail_lower = error_detail.lower()
                looks_like_timeout = (
                    isinstance(e, (asyncio.TimeoutError, APITimeoutError))
                    and (
                        ident in timeout_cancellations
                        or call_timeout is not None
                        or "timeout" in detail_lower
                        or "timed out" in detail_lower
                    )
                )
                if looks_like_timeout:
                    status.num_timeout_errors += 1
                    _trigger_timeout_burst(time.time())
                    inflight.pop(ident, None)
                    await adjust_timeout()
                    if isinstance(e, APITimeoutError):
                        base_message = "OpenAI client timed out; consider reducing concurrency."
                    else:
                        base_message = "API call timed out."
                    if error_detail:
                        base_message = f"{base_message} Details: {error_detail}"
                        error_logs[ident].append(error_detail)
                    error_message = f"{base_message} (elapsed {elapsed:.2f}s)"
                    timeout_errors_since_last_status += 1
                    timeout_note = f"{ident} timed out after {elapsed:.2f}s"
                    timeout_notes.append(timeout_note)
                    timeout_key: Hashable = (
                        "timeout",
                        type(e).__name__,
                        error_detail or None,
                    )
                    _log_timeout_once(
                        error_message,
                        timeout_note,
                        dedup_key=timeout_key,
                    )
                    if restart_requested:
                        continue
                    error_logs[ident].append(error_message)
                    if attempts_left - 1 > 0:
                        backoff = random.uniform(1, 2) * (2 ** (max_retries - attempts_left))
                        # Retry the same prompt after a delay.  We sleep within the
                        # worker so the task remains accounted for in ``queue.join``
                        # and ensure the new task is enqueued before ``task_done``
                        # is called.  This mirrors the legacy retry behaviour and
                        # prevents retries from being dropped prematurely.
                        await _maybe_retry(backoff)
                    else:
                        row = {
                            "Identifier": ident,
                            "Response": None,
                            "Web Search Sources": sources_data,
                            "Time Taken": None,
                            "Input Tokens": input_tokens,
                            "Reasoning Tokens": None,
                            "Output Tokens": None,
                            "Reasoning Effort": get_response_kwargs.get(
                                "reasoning_effort", reasoning_effort
                            ),
                            "Successful": False,
                            "Error Log": error_logs.get(ident, []),
                        }
                        if response_ids:
                            row["Response IDs"] = response_ids
                        if reasoning_summary is not None:
                            row["Reasoning Summary"] = None
                        results.append(row)
                        processed += 1
                        status.num_tasks_failed += 1
                        status.num_tasks_in_progress -= 1
                        pbar.update(1)
                        error_logs.pop(ident, None)
                        await flush()
                else:
                    inflight.pop(ident, None)
                    status.num_other_errors += 1
                    base_message = "API call failed before timeout could be applied"
                    if error_detail:
                        base_message = f"{base_message}: {error_detail}"
                        error_logs[ident].append(error_detail)
                    if "connection error" in detail_lower:
                        status.num_connection_errors += 1
                        connection_errors_since_adjust += 1
                        connection_errors_since_last_status += 1
                        connection_error_times.append(time.time())
                        maybe_adjust_for_connection_errors()
                        _log_connection_once(base_message)
                    error_key: Hashable = (
                        "non-timeout-error",
                        type(e).__name__,
                        error_detail or None,
                    )
                    if "connection error" not in detail_lower:
                        _emit_first_error(base_message, dedup_key=error_key)
                        logger.warning(base_message)
                    else:
                        logger.debug(base_message)
                    if attempts_left - 1 > 0:
                        backoff = random.uniform(1, 2) * (2 ** (max_retries - attempts_left))
                        await _maybe_retry(backoff)
                    else:
                        row = {
                            "Identifier": ident,
                            "Response": None,
                            "Web Search Sources": sources_data,
                            "Time Taken": None,
                            "Input Tokens": input_tokens,
                            "Reasoning Tokens": None,
                            "Output Tokens": None,
                            "Reasoning Effort": get_response_kwargs.get(
                                "reasoning_effort", reasoning_effort
                            ),
                            "Successful": False,
                            "Error Log": error_logs.get(ident, []),
                        }
                        if response_ids:
                            row["Response IDs"] = response_ids
                        if reasoning_summary is not None:
                            row["Reasoning Summary"] = None
                        results.append(row)
                        processed += 1
                        status.num_tasks_failed += 1
                        status.num_tasks_in_progress -= 1
                        pbar.update(1)
                        error_logs.pop(ident, None)
                        _maybe_trigger_threshold_refresh()
                        await flush()
            except JSONParseError as e:
                inflight.pop(ident, None)
                status.num_other_errors += 1
                error_detail = str(e).strip()
                if e.snippet:
                    error_detail = f"{error_detail} Snippet: {e.snippet}"
                logger.warning(f"JSON parse error for {ident}: {error_detail}")
                _emit_first_error(
                    f"JSON parse error encountered: {error_detail}",
                    dedup_key=("json-parse-error", error_detail),
                )
                error_logs[ident].append(error_detail)
                if attempts_left - 1 > 0:
                    backoff = random.uniform(1, 2) * (2 ** (max_retries - attempts_left))
                    await _maybe_retry(backoff)
                else:
                    row = {
                        "Identifier": ident,
                        "Response": None,
                        "Web Search Sources": sources_data,
                        "Time Taken": duration,
                        "Input Tokens": total_input,
                        "Reasoning Tokens": total_reasoning,
                        "Output Tokens": total_output,
                        "Reasoning Effort": get_response_kwargs.get(
                            "reasoning_effort", reasoning_effort
                        ),
                        "Successful": False,
                        "Error Log": error_logs.get(ident, []),
                    }
                    if response_ids:
                        row["Response IDs"] = response_ids
                    if reasoning_summary is not None:
                        row["Reasoning Summary"] = None
                    results.append(row)
                    processed += 1
                    status.num_tasks_failed += 1
                    status.num_tasks_in_progress -= 1
                    pbar.update(1)
                    error_logs.pop(ident, None)
                    _maybe_trigger_threshold_refresh()
                    await flush()
            except RateLimitError as e:
                inflight.pop(ident, None)
                status.num_rate_limit_errors += 1
                status.time_of_last_rate_limit_error = time.time()
                cooldown_until = status.time_of_last_rate_limit_error + global_cooldown
                error_text = str(e)
                _log_rate_limit_once(error_text)
                error_logs[ident].append(error_text)
                rate_limit_error_times.append(time.time())
                rate_limit_errors_since_adjust += 1
                successes_since_adjust = 0
                if _is_quota_error_message(error_text):
                    fatal_msg = (
                        "Quota exceeded (billing or credit balance likely exhausted). "
                        "Add funds at https://platform.openai.com/settings/organization/billing/. "
                        "Stopping remaining requests."
                    )
                    logger.error(fatal_msg)
                    error_logs[ident].append(fatal_msg)
                    row = {
                        "Identifier": ident,
                        "Response": None,
                        "Time Taken": None,
                        "Input Tokens": input_tokens,
                        "Reasoning Tokens": None,
                        "Output Tokens": None,
                        "Reasoning Effort": get_response_kwargs.get(
                            "reasoning_effort", reasoning_effort
                        ),
                        "Successful": False,
                        "Error Log": error_logs.get(ident, []),
                    }
                    if response_ids:
                        row["Response IDs"] = response_ids
                    if reasoning_summary is not None:
                        row["Reasoning Summary"] = None
                    results.append(row)
                    processed += 1
                    status.num_tasks_failed += 1
                    status.num_tasks_in_progress -= 1
                    pbar.update(1)
                    drained = 0
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                            queue.task_done()
                            drained += 1
                        except asyncio.QueueEmpty:
                            break
                    if drained:
                        status.num_tasks_failed += drained
                        status.num_tasks_in_progress -= drained
                        processed += drained
                        pbar.update(drained)
                    stop_event.set()
                    await flush()
                    raise RuntimeError(fatal_msg)
                _maybe_refresh_estimates("rate-limit encountered", force=False)
                maybe_adjust_concurrency()
                if attempts_left - 1 > 0:
                    backoff = random.uniform(1, 2) * (2 ** (max_retries - attempts_left))
                    await _maybe_retry(backoff)
                else:
                    row = {
                        "Identifier": ident,
                        "Response": None,
                        "Time Taken": None,
                        "Input Tokens": input_tokens,
                        "Reasoning Tokens": None,
                        "Output Tokens": None,
                        "Reasoning Effort": get_response_kwargs.get(
                            "reasoning_effort", reasoning_effort
                        ),
                        "Successful": False,
                        "Error Log": error_logs.get(ident, []),
                    }
                    if response_ids:
                        row["Response IDs"] = response_ids
                    if reasoning_summary is not None:
                        row["Reasoning Summary"] = None
                    results.append(row)
                    processed += 1
                    status.num_tasks_failed += 1
                    status.num_tasks_in_progress -= 1
                    pbar.update(1)
                    error_logs.pop(ident, None)
                    _maybe_trigger_threshold_refresh()
                    await flush()
            except APIConnectionError as e:
                inflight.pop(ident, None)
                status.num_api_errors += 1
                status.num_connection_errors += 1
                _log_connection_once(str(e))
                error_logs[ident].append(str(e))
                connection_error_times.append(time.time())
                connection_errors_since_adjust += 1
                connection_errors_since_last_status += 1
                maybe_adjust_for_connection_errors()
                if attempts_left - 1 > 0 and not stop_event.is_set():
                    backoff = random.uniform(1, 2) * (2 ** (max_retries - attempts_left))
                    await _maybe_retry(backoff)
                else:
                    row = {
                        "Identifier": ident,
                        "Response": None,
                        "Time Taken": None,
                        "Input Tokens": input_tokens,
                        "Reasoning Tokens": None,
                        "Output Tokens": None,
                        "Reasoning Effort": get_response_kwargs.get(
                            "reasoning_effort", reasoning_effort
                        ),
                        "Successful": False,
                        "Error Log": error_logs.get(ident, []),
                    }
                    if reasoning_summary is not None:
                        row["Reasoning Summary"] = None
                    results.append(row)
                    processed += 1
                    status.num_tasks_failed += 1
                    status.num_tasks_in_progress -= 1
                    pbar.update(1)
                    error_logs.pop(ident, None)
                    _maybe_trigger_threshold_refresh()
                    await flush()
            except (
                APIError,
                BadRequestError,
                AuthenticationError,
                InvalidRequestError,
            ) as e:
                inflight.pop(ident, None)
                status.num_api_errors += 1
                logger.warning(f"API error for {ident}: {e}")
                _emit_first_error(f"API error encountered: {e}")
                error_logs[ident].append(str(e))
                row = {
                    "Identifier": ident,
                    "Response": None,
                    "Time Taken": None,
                    "Input Tokens": input_tokens,
                    "Reasoning Tokens": None,
                    "Output Tokens": None,
                    "Reasoning Effort": get_response_kwargs.get(
                        "reasoning_effort", reasoning_effort
                    ),
                    "Successful": False,
                    "Error Log": error_logs.get(ident, []),
                }
                if response_ids:
                    row["Response IDs"] = response_ids
                if reasoning_summary is not None:
                    row["Reasoning Summary"] = None
                results.append(row)
                processed += 1
                status.num_tasks_failed += 1
                status.num_tasks_in_progress -= 1
                pbar.update(1)
                error_logs.pop(ident, None)
                _maybe_trigger_threshold_refresh()
                await flush()
            except Exception as e:
                inflight.pop(ident, None)
                status.num_other_errors += 1
                logger.error(f"Unexpected error for {ident}: {e}")
                _emit_first_error(f"Unexpected error encountered: {e}", level=logging.ERROR)
                await flush()
                raise
            finally:
                active_workers -= 1
                queue.task_done()

    async def timeout_watcher() -> None:
        try:
            while True:
                await asyncio.sleep(0.5)
                if stop_event.is_set():
                    break
                now = time.time()
                current_timeout = nonlocal_timeout
                for ident, (start, task, t_out) in list(inflight.items()):
                    if task.done():
                        continue
                    if _should_cancel_inflight_task(
                        start,
                        now,
                        current_timeout,
                        t_out,
                        dynamic_timeout,
                    ):
                        timeout_cancellations.add(ident)
                        task.cancel()
        except asyncio.CancelledError:
            pass

    async def status_reporter() -> None:
        if status_report_interval is None:
            return
        try:
            nonlocal timeout_errors_since_last_status, connection_errors_since_last_status
            while not stop_event.is_set():
                await asyncio.sleep(status_report_interval)
                if stop_event.is_set() or processed >= status.num_tasks_started:
                    break
                emit_parallelization_status(
                    "Periodic status update", force=True, label=None
                )
                timeout_errors_since_last_status = 0
                connection_errors_since_last_status = 0
        except asyncio.CancelledError:
            pass

    # Spawn workers and ensure they are cleaned up on exit or cancellation
    watcher = asyncio.create_task(timeout_watcher())
    status_task: Optional[asyncio.Task] = None
    if status_report_interval is not None:
        status_task = asyncio.create_task(status_reporter())
    initial_worker_count = max(1, min(_effective_parallel_ceiling(), queue.qsize()))
    workers = [asyncio.create_task(worker()) for _ in range(initial_worker_count)]
    try:
        await queue.join()
    except (asyncio.CancelledError, KeyboardInterrupt):
        stop_event.set()
        logger.info("Cancellation requested, shutting down workers...")
        raise
    finally:
        stop_event.set()
        for w in workers:
            w.cancel()
        watcher.cancel()
        if status_task is not None:
            status_task.cancel()
        worker_results = await asyncio.gather(*workers, return_exceptions=True)
        await asyncio.gather(watcher, return_exceptions=True)
        if status_task is not None:
            await asyncio.gather(status_task, return_exceptions=True)
        for res in worker_results:
            if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                # flush partial results before raising
                await flush()
                pbar.close()
                raise res
        # Flush remaining results and close progress bar
        await flush()
        pbar.close()

    if restart_requested:
        cooldown_msg = (
            f"[timeouts] Cooldown triggered after {len(timeout_error_times)} timeouts in "
            f"{int(timeout_burst_window)}s. Sleeping for {int(timeout_burst_cooldown)}s before resuming from checkpoint."
        )
        if message_verbose:
            print(cooldown_msg)
        logger.warning(cooldown_msg)
        await asyncio.sleep(timeout_burst_cooldown)
        restart_kwargs = dict(get_response_kwargs)
        for duplicate_key in (
            "model",
            "n",
            "max_output_tokens",
            "temperature",
            "json_mode",
            "expected_schema",
            "tools",
            "tool_choice",
            "web_search",
            "web_search_filters",
            "search_context_size",
            "reasoning_effort",
            "reasoning_summary",
            "use_dummy",
            "response_fn",
            "get_all_responses_fn",
            "base_url",
            "background_mode",
            "background_poll_interval",
            "api_key",
        ):
            restart_kwargs.pop(duplicate_key, None)
        restart_kwargs["_timeout_restart_count"] = restart_count + 1
        return await get_all_responses(
            prompts=prompts,
            identifiers=identifiers,
            prompt_images=prompt_images,
            prompt_audio=prompt_audio,
            prompt_web_search_filters=prompt_web_search_filters,
            model=model,
            n=n,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            json_mode=json_mode,
            expected_schema=expected_schema,
            tools=tools,
            tool_choice=tool_choice,
            web_search=web_search,
            web_search_filters=web_search_filters,
            search_context_size=search_context_size,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            dummy_responses=dummy_responses,
            use_dummy=use_dummy,
            response_fn=response_fn,
            get_all_responses_fn=get_all_responses_fn,
            base_url=base_url,
            api_key=api_key,
            print_example_prompt=print_example_prompt,
            save_path=save_path,
            reset_files=False,
            n_parallels=n_parallels,
            max_retries=max_retries,
            timeout_factor=timeout_factor,
            max_timeout=max_timeout,
            dynamic_timeout=dynamic_timeout,
            timeout_burst_window=timeout_burst_window,
            timeout_burst_cooldown=timeout_burst_cooldown,
            timeout_burst_max_restarts=timeout_burst_max_restarts,
            background_mode=background_mode,
            background_poll_interval=background_poll_interval,
            cancel_existing_batch=cancel_existing_batch,
            use_batch=use_batch,
            batch_completion_window=batch_completion_window,
            batch_poll_interval=batch_poll_interval,
            batch_wait_for_completion=batch_wait_for_completion,
            max_batch_requests=max_batch_requests,
            max_batch_file_bytes=max_batch_file_bytes,
            save_every_x_responses=save_every_x_responses,
            verbose=verbose,
            quiet=quiet,
            global_cooldown=global_cooldown,
            rate_limit_window=rate_limit_window,
            connection_error_window=connection_error_window,
            token_sample_size=token_sample_size,
            status_report_interval=status_report_interval,
            planning_rate_limit_buffer=planning_rate_limit_buffer,
            logging_level=logging_level,
            **restart_kwargs,
        )

    logger.info(
        f"Processing complete. {status.num_tasks_succeeded}/{status.num_tasks_started} requests succeeded."
    )
    if status.num_tasks_failed > 0:
        logger.warning(f"{status.num_tasks_failed} requests failed.")
    if status.num_rate_limit_errors > 0:
        logger.warning(
            f"{status.num_rate_limit_errors} rate limit errors encountered; consider reducing concurrency via lower n_parallels."
        )
    if status.num_connection_errors > 0:
        logger.warning(
            f"{status.num_connection_errors} API connection errors encountered, indicating network instability or bandwidth limitations; "
            "consider reducing concurrency via lower n_parallels."
        )
    if status.num_timeout_errors > 0:
        logger.warning(f"{status.num_timeout_errors} timeouts encountered.")
    if status.num_api_errors > 0:
        logger.warning(f"{status.num_api_errors} API errors encountered.")
    if status.num_other_errors > 0:
        logger.warning(f"{status.num_other_errors} unexpected errors encountered.")
    _report_cost()
    return df
