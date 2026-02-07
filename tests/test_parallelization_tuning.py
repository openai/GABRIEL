import asyncio
import time

from gabriel.utils import openai_utils


def test_example_prompt_is_plain_text(capsys):
    prompt = "Line one\nLine two"
    openai_utils._display_example_prompt(prompt, verbose=True)
    output = capsys.readouterr().out
    assert "===== Example prompt =====" in output
    assert "Line one" in output and "Line two" in output
    assert "<details" not in output


def test_usage_overview_compact_printout(capsys):
    openai_utils._print_usage_overview(
        prompts=["hello", "world"],
        n=1,
        max_output_tokens=32,
        model="gpt-5-mini",
        use_batch=False,
        n_parallels=4,
        verbose=True,
        rate_headers={"limit_requests": "20", "limit_tokens": "2000"},
        heading="Usage check",
        show_prompt_stats=False,
    )
    output = capsys.readouterr().out
    assert "Usage check" in output
    assert "Prompts:" not in output
    assert "<summary" not in output


def test_wait_based_cap_dampens_reductions():
    now = time.time()
    cap, last_adjust, changed = openai_utils._smooth_wait_based_cap(
        current_cap=100,
        candidate_cap=40,
        now=now,
        last_adjust=0.0,
        limiter_pressure=True,
        min_delta=4,
        cooldown_up=10.0,
        cooldown_down=30.0,
    )
    assert changed
    assert cap == 92  # 8% step down

    cap2, last_adjust2, changed2 = openai_utils._smooth_wait_based_cap(
        current_cap=cap,
        candidate_cap=40,
        now=now + 5.0,
        last_adjust=last_adjust,
        limiter_pressure=True,
        min_delta=4,
        cooldown_up=10.0,
        cooldown_down=30.0,
    )
    assert not changed2
    assert cap2 == cap
    assert last_adjust2 == last_adjust


def test_wait_based_cap_allows_gentle_growth():
    now = time.time()
    cap, last_adjust, changed = openai_utils._smooth_wait_based_cap(
        current_cap=10,
        candidate_cap=30,
        now=now,
        last_adjust=0.0,
        limiter_pressure=True,
        min_delta=2,
        cooldown_up=0.0,
        cooldown_down=30.0,
    )
    assert changed
    assert cap == 12  # 18% step up from 10 (ceil)

    cap2, _, changed2 = openai_utils._smooth_wait_based_cap(
        current_cap=cap,
        candidate_cap=30,
        now=now + 1.0,
        last_adjust=last_adjust,
        limiter_pressure=True,
        min_delta=2,
        cooldown_up=0.0,
        cooldown_down=30.0,
    )
    assert changed2
    assert cap2 > cap


def test_rate_limit_decrement_is_aggressive():
    assert openai_utils._rate_limit_decrement(50) == 20
    assert openai_utils._rate_limit_decrement(3) == 3


def test_ramp_up_increases_parallelism(tmp_path):
    active = {"current": 0, "peak": 0}
    lock = asyncio.Lock()

    async def responder(prompt: str, **_: object):
        async with lock:
            active["current"] += 1
            active["peak"] = max(active["peak"], active["current"])
        await asyncio.sleep(0.05)
        async with lock:
            active["current"] -= 1
        return [f"ok-{prompt}"], 0.01, []

    asyncio.run(
        openai_utils.get_all_responses(
            prompts=[f"p{i}" for i in range(12)],
            identifiers=[f"p{i}" for i in range(12)],
            response_fn=responder,
            use_dummy=False,
            save_path=str(tmp_path / "responses.csv"),
            reset_files=True,
            dynamic_timeout=False,
            max_retries=1,
            n_parallels=8,
            ramp_up_seconds=0.1,
            ramp_up_start_fraction=0.25,
            status_report_interval=None,
            logging_level="error",
        )
    )

    assert active["peak"] > 2


def test_ramp_up_halts_on_rate_limit(tmp_path):
    active = {"current": 0, "peak": 0}
    lock = asyncio.Lock()
    first_error = {"raised": False}

    async def responder(prompt: str, **_: object):
        async with lock:
            active["current"] += 1
            active["peak"] = max(active["peak"], active["current"])
        if not first_error["raised"]:
            first_error["raised"] = True
            async with lock:
                active["current"] -= 1
            raise openai_utils.RateLimitError("rate limit")
        await asyncio.sleep(0.2)
        async with lock:
            active["current"] -= 1
        return [f"ok-{prompt}"], 0.01, []

    asyncio.run(
        openai_utils.get_all_responses(
            prompts=[f"p{i}" for i in range(10)],
            identifiers=[f"p{i}" for i in range(10)],
            response_fn=responder,
            use_dummy=False,
            save_path=str(tmp_path / "responses.csv"),
            reset_files=True,
            dynamic_timeout=False,
            max_retries=1,
            n_parallels=8,
            ramp_up_seconds=0.1,
            ramp_up_start_fraction=0.25,
            status_report_interval=None,
            global_cooldown=0,
            logging_level="error",
        )
    )

    assert active["peak"] <= 3
