import asyncio
from pathlib import Path
from collections import defaultdict

import pandas as pd

from gabriel.utils import openai_utils


def test_timeout_burst_restarts_and_resumes(tmp_path: Path) -> None:
    attempts = defaultdict(int)
    timeout_count = {"count": 0}

    async def flaky_responder(prompt: str, **_: object):
        attempts[prompt] += 1
        # The first two overall calls time out to trigger the burst logic, then all succeed.
        if timeout_count["count"] < 2:
            timeout_count["count"] += 1
            await asyncio.sleep(0)
            raise asyncio.TimeoutError("simulated timeout")
        return [f"ok-{prompt}"], 0.01, []

    save_path = tmp_path / "responses.csv"
    df: pd.DataFrame = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["p1", "p2"],
            identifiers=["p1", "p2"],
            response_fn=flaky_responder,
            use_dummy=False,
            save_path=str(save_path),
            reset_files=True,
            timeout_burst_window=10.0,
            timeout_burst_cooldown=0.01,
            timeout_burst_max_restarts=2,
            dynamic_timeout=False,  # keep deterministic for the test
            max_retries=1,
            n_parallels=1,
            logging_level="error",
        )
    )

    assert set(df["Identifier"]) == {"p1", "p2"}
    assert len(df) == 2
    # At least one timeout should have fired, triggering the burst handler, and the run should return without hanging.
    assert timeout_count["count"] == 2
    assert all(count >= 2 for count in attempts.values())


def test_timeout_burst_threshold_tracks_concurrency(tmp_path: Path) -> None:
    attempts = defaultdict(int)
    timeout_count = {"count": 0}

    async def flaky_responder(prompt: str, **_: object):
        attempts[prompt] += 1
        if timeout_count["count"] < 2:
            timeout_count["count"] += 1
            await asyncio.sleep(0)
            raise asyncio.TimeoutError("simulated timeout")
        return [f"ok-{prompt}"], 0.01, []

    save_path = tmp_path / "responses.csv"
    df: pd.DataFrame = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["p1", "p2"],
            identifiers=["p1", "p2"],
            response_fn=flaky_responder,
            use_dummy=False,
            save_path=str(save_path),
            reset_files=True,
            timeout_burst_window=10.0,
            timeout_burst_cooldown=1.0,
            timeout_burst_max_restarts=1,
            dynamic_timeout=False,
            max_retries=3,
            n_parallels=2,
            logging_level="error",
        )
    )

    # Threshold scales with concurrency (n_parallels=2 -> threshold=3), so two
    # timeouts should not trigger a restart and retries should succeed.
    assert timeout_count["count"] == 2
    assert all(count >= 2 for count in attempts.values())
    assert df["Successful"].all()


def test_timeout_burst_restart_recovers_with_other_errors(tmp_path: Path) -> None:
    attempts = defaultdict(int)

    async def flaky_responder(prompt: str, **_: object):
        attempts[prompt] += 1
        if prompt == "p1" and attempts[prompt] == 1:
            raise asyncio.TimeoutError("simulated timeout")
        if prompt == "p2" and attempts[prompt] == 1:
            raise openai_utils.RateLimitError("rate limit")
        return [f"ok-{prompt}-{attempts[prompt]}"], 0.01, []

    save_path = tmp_path / "responses.csv"
    df: pd.DataFrame = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["p1", "p2"],
            identifiers=["p1", "p2"],
            response_fn=flaky_responder,
            use_dummy=False,
            save_path=str(save_path),
            reset_files=True,
            timeout_burst_window=5.0,
            timeout_burst_cooldown=0.01,
            timeout_burst_max_restarts=1,
            dynamic_timeout=False,
            max_retries=2,
            n_parallels=2,
            logging_level="error",
        )
    )

    assert df["Successful"].all()
    assert attempts["p1"] >= 2
    assert attempts["p2"] >= 2


def test_non_timeout_error_preserved(tmp_path: Path) -> None:
    attempts = defaultdict(int)

    async def flaky_responder(prompt: str, **_: object):
        attempts[prompt] += 1
        if attempts[prompt] == 1:
            raise RuntimeError("boom")
        return [f"ok-{prompt}-{attempts[prompt]}"], 0.01, []

    save_path = tmp_path / "responses.csv"
    df: pd.DataFrame = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["p1"],
            identifiers=["p1"],
            response_fn=flaky_responder,
            use_dummy=False,
            save_path=str(save_path),
            reset_files=True,
            dynamic_timeout=False,
            max_retries=2,
            n_parallels=1,
            logging_level="error",
        )
    )

    assert df["Successful"].iloc[0]
    error_logs = df["Error Log"].iloc[0]
    assert any("boom" in str(msg) for msg in error_logs)
    assert not any("timeout" in str(msg).lower() for msg in error_logs)


def test_connection_timeout_does_not_count_as_timeout(tmp_path: Path) -> None:
    async def flaky_responder(prompt: str, **_: object):
        raise openai_utils.APITimeoutError("Connection error")

    save_path = tmp_path / "responses.csv"
    df: pd.DataFrame = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["p1"],
            identifiers=["p1"],
            response_fn=flaky_responder,
            use_dummy=False,
            save_path=str(save_path),
            reset_files=True,
            dynamic_timeout=False,
            max_retries=1,
            n_parallels=1,
            logging_level="error",
        )
    )

    error_logs = df["Error Log"].iloc[0]
    assert any("connection error" in str(msg).lower() for msg in error_logs)
    assert not any("timed out" in str(msg).lower() for msg in error_logs)
