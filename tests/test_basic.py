import asyncio
from typing import Any, Dict, List, Optional, Tuple

import json
from pathlib import Path

import pandas as pd
import numpy as np
import openai
import pytest

from gabriel.core.prompt_template import PromptTemplate
from gabriel.utils import DummyResponseSpec, openai_utils, safest_json
from gabriel.tasks.rate import Rate, RateConfig
from gabriel.tasks.deidentify import Deidentifier, DeidentifyConfig
from gabriel.tasks.classify import Classify, ClassifyConfig, _collect_predictions
from gabriel.tasks.extract import Extract, ExtractConfig
from gabriel.tasks.rank import Rank, RankConfig
from gabriel.tasks.discover import Discover, DiscoverConfig
from gabriel.tasks.bucket import Bucket, BucketConfig
import gabriel.tasks.discover as discover_module
import gabriel


def test_decide_default_max_output_tokens_respects_user_choice():
    assert (
        openai_utils._decide_default_max_output_tokens(4096, {"remaining_tokens": "10"})
        == 4096
    )


def test_decide_default_max_output_tokens_no_longer_caps_by_default():
    cutoff = openai_utils._decide_default_max_output_tokens(
        None,
        {"remaining_tokens": "500000", "limit_tokens": "1000000"},
    )
    assert cutoff is None


def test_normalise_web_search_filters_supports_location_type():
    filters = {
        "allowed_domains": {"news.com", "openai.com"},
        "city": "London",
        "timezone": "",  # Should be stripped
        "type": "approximate",
    }
    normalised = openai_utils._normalise_web_search_filters(filters)
    assert set(normalised["filters"]["allowed_domains"]) == {"news.com", "openai.com"}
    assert normalised["user_location"] == {"city": "London", "type": "approximate"}


def test_normalise_web_search_filters_sets_default_location_type():
    filters = {"country": "US"}
    normalised = openai_utils._normalise_web_search_filters(filters)
    assert normalised["user_location"]["type"] == "approximate"
    assert normalised["user_location"]["country"] == "US"


def test_build_params_embeds_web_search_tool_payload():
    params = openai_utils._build_params(
        model="gpt-4o-mini",
        input_data=[{"role": "user", "content": "hello"}],
        max_output_tokens=None,
        system_instruction="",
        temperature=0.7,
        tools=[{"type": "retrieval"}],
        tool_choice=None,
        web_search=True,
        web_search_filters={
            "allowed_domains": ["openai.com"],
            "country": "GB",
            "type": "approximate",
        },
        search_context_size="high",
        json_mode=False,
        expected_schema=None,
        reasoning_effort=None,
        reasoning_summary=None,
    )
    assert any(tool["type"] == "retrieval" for tool in params["tools"])
    web_tool = next(tool for tool in params["tools"] if tool["type"] == "web_search")
    assert web_tool["search_context_size"] == "high"
    assert web_tool["filters"]["allowed_domains"] == ["openai.com"]
    assert web_tool["user_location"] == {"country": "GB", "type": "approximate"}
    assert "include" in params
    assert "web_search_call.action.sources" in params["include"]


def test_build_params_respects_user_include_and_dedup():
    params = openai_utils._build_params(
        model="gpt-4o-mini",
        input_data=[{"role": "user", "content": "hello"}],
        max_output_tokens=None,
        system_instruction="",
        temperature=0.7,
        tools=None,
        tool_choice=None,
        web_search=False,
        web_search_filters=None,
        search_context_size="medium",
        json_mode=False,
        expected_schema=None,
        reasoning_effort=None,
        reasoning_summary=None,
        include=["message.output_text.logprobs", "message.output_text.logprobs"],
    )
    assert params["include"] == ["message.output_text.logprobs"]

    params_search = openai_utils._build_params(
        model="gpt-4o-mini",
        input_data=[{"role": "user", "content": "hello"}],
        max_output_tokens=None,
        system_instruction="",
        temperature=0.7,
        tools=None,
        tool_choice=None,
        web_search=True,
        web_search_filters=None,
        search_context_size="medium",
        json_mode=False,
        expected_schema=None,
        reasoning_effort=None,
        reasoning_summary=None,
        include=["web_search_call.action.sources", "message.output_text.logprobs"],
    )
    # Should preserve user include, but not duplicate the sources entry
    assert params_search["include"].count("web_search_call.action.sources") == 1
    assert "message.output_text.logprobs" in params_search["include"]


def test_build_params_defaults_include_to_web_sources_only_when_web_search_enabled():
    params = openai_utils._build_params(
        model="gpt-4o-mini",
        input_data=[{"role": "user", "content": "hello"}],
        max_output_tokens=None,
        system_instruction="",
        temperature=0.7,
        tools=None,
        tool_choice=None,
        web_search=True,
        web_search_filters=None,
        search_context_size="medium",
        json_mode=False,
        expected_schema=None,
        reasoning_effort=None,
        reasoning_summary=None,
    )
    assert params["include"] == ["web_search_call.action.sources"]


def test_build_params_normalises_search_context_size_aliases():
    params = openai_utils._build_params(
        model="gpt-4o-mini",
        input_data=[{"role": "user", "content": "hello"}],
        max_output_tokens=None,
        system_instruction="",
        temperature=0.7,
        tools=None,
        tool_choice=None,
        web_search=True,
        web_search_filters=None,
        search_context_size="large",  # backwards compatible alias
        json_mode=False,
        expected_schema=None,
        reasoning_effort=None,
        reasoning_summary=None,
    )

    web_tool = next(tool for tool in params["tools"] if tool["type"] == "web_search")
    assert web_tool["search_context_size"] == "high"


def test_extract_web_search_sources_recurses_nested_payload():
    raw = [
        {
            "output": [
                {
                    "type": "web_search_call",
                    "web_search_call": {
                        "id": "call-1",
                        "action": {
                            "query": "python",
                            "sources": [
                                {"url": "https://example.com/a", "title": "Result A"},
                                {"url": "https://example.com/b", "title": "Result B"},
                            ],
                        },
                    },
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Answer"}],
                },
            ]
        }
    ]
    sources = openai_utils._extract_web_search_sources(raw)
    assert sources is not None
    assert {"url": "https://example.com/a", "title": "Result A"} in sources


def test_prompt_template():
    tmpl = PromptTemplate.from_package("ratings_prompt.jinja2")
    text = tmpl.render(attributes=["a"], descriptions=["desc"], passage="x", object_category="obj", attribute_category="att", format="json")
    assert "desc" in text


def test_ratings_default_scale_prompt():
    tmpl = PromptTemplate.from_package("ratings_prompt.jinja2")
    rendered = tmpl.render(text="x", attributes=["clarity"], scale=None)
    assert "Use integers 0-100" in rendered


def test_shuffled_dict_rendering():
    tmpl = PromptTemplate.from_package("classification_prompt.jinja2")
    rendered = tmpl.render(text="x", attributes={"clarity": "Is the text clear?"})
    assert "OrderedDict" not in rendered
    assert "{" in rendered and "}" in rendered


def test_get_response_dummy():
    responses, _ = asyncio.run(openai_utils.get_response("hi", use_dummy=True))
    assert responses and responses[0].startswith("DUMMY")


def test_get_response_images_dummy():
    responses, _ = asyncio.run(
        openai_utils.get_response("hi", images=["abcd"], use_dummy=True)
    )
    assert responses and responses[0].startswith("DUMMY")


def test_get_response_audio_dummy():
    responses, _ = asyncio.run(
        openai_utils.get_response(
            "hi", audio=[{"data": "abcd", "format": "mp3"}], use_dummy=True
        )
    )
    assert responses and responses[0].startswith("DUMMY")


def test_get_response_background_poll(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    openai_utils._clients_async.clear()

    class DummyResponse:
        def __init__(self, status: str, text: str = "", error: Optional[Dict[str, Any]] = None):
            self.status = status
            self.id = "resp-test"
            self.output_text = text
            self.usage = {
                "input_tokens": 1,
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
            }
            self.output = []
            self.error = error

    class FakeResponses:
        def __init__(self):
            self._retrieve_calls = 0

        async def create(self, **kwargs):
            assert kwargs.get("background") is True
            return DummyResponse("in_progress")

        async def retrieve(self, response_id: str, **kwargs):
            self._retrieve_calls += 1
            if self._retrieve_calls < 2:
                return DummyResponse("in_progress")
            return DummyResponse("completed", text="final-answer")

    class FakeClient:
        def __init__(self):
            self.responses = FakeResponses()

    fake_client = FakeClient()
    monkeypatch.setattr(openai_utils, "_get_client", lambda base_url=None: fake_client)

    async def _runner():
        return await openai_utils.get_response(
            "hello",
            use_dummy=False,
            timeout=None,
            background_mode=True,
            background_poll_interval=0.01,
            return_raw=True,
        )

    texts, duration, raw = asyncio.run(_runner())

    assert texts == ["final-answer"]
    assert duration >= 0
    assert raw and raw[0].status == "completed"
    assert fake_client.responses._retrieve_calls >= 1


def test_get_response_polls_only_when_needed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    openai_utils._clients_async.clear()

    class DummyResponse:
        def __init__(self, status: str, text: str, rid: str):
            self.status = status
            self.id = rid
            self.output_text = text
            self.output = []
            self.error = None
            self.usage = {
                "input_tokens": 1,
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
            }

    class FakeResponses:
        def __init__(self):
            self.create_calls = 0
            self.retrieve_calls = 0

        async def create(self, **kwargs):
            self.create_calls += 1
            return DummyResponse("in_progress", "new-answer", "new-1")

        async def retrieve(self, response_id: str, **kwargs):
            self.retrieve_calls += 1
            return DummyResponse("completed", "old-answer", response_id)

    fake_responses = FakeResponses()
    fake_client = type("FakeClient", (), {"responses": fake_responses})()
    monkeypatch.setattr(openai_utils, "_get_client", lambda base_url=None: fake_client)

    texts, duration, raw = asyncio.run(
        openai_utils.get_response(
            "hi",
            use_dummy=False,
            timeout=None,
            background_mode=False,
            background_poll_interval=0.01,
            return_raw=True,
        )
    )

    assert texts == ["old-answer"]
    assert fake_responses.create_calls == 1
    assert fake_responses.retrieve_calls == 1
    assert duration >= 0


def test_gpt_audio_modalities(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    openai_utils._clients_async.clear()

    class DummyClient:
        def __init__(self):
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            DummyClient.captured = kwargs

            class Msg:
                content = ""

            class Choice:
                message = Msg()

            class Resp:
                choices = [Choice()]

            return Resp()

    dummy = DummyClient()
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **_: dummy)

    asyncio.run(
        openai_utils.get_response(
            "hi",
            model="gpt-audio",
            audio=[{"data": "abcd", "format": "mp3"}],
            use_dummy=False,
        )
    )
    assert DummyClient.captured["modalities"] == ["text"]


def test_custom_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    openai_utils._clients_async.clear()
    client = openai_utils._get_client("https://example.com/v1")
    assert str(client.base_url) == "https://example.com/v1/"
    openai_utils._clients_async.clear()
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.org/v1")
    client2 = openai_utils._get_client()
    assert str(client2.base_url) == "https://example.org/v1/"


def test_get_embedding_dummy():
    emb, _ = asyncio.run(openai_utils.get_embedding("hi", use_dummy=True))
    assert isinstance(emb, list) and emb and isinstance(emb[0], float)


def test_safest_json_codeblock_list():
    raw = ["```json\n{\n \"speech\": true,\n \"music\": false\n}\n```"]
    parsed = asyncio.run(safest_json(raw))
    assert parsed == {"speech": True, "music": False}


def test_safest_json_invalid_without_fallback():
    parsed = asyncio.run(safest_json("not json"))
    assert parsed is None


def test_gpt5_temperature_warning(caplog):
    """Ensure gpt-5 models ignore temperature and log a warning."""
    with caplog.at_level("WARNING"):
        params = openai_utils._build_params(
            model="gpt-5-mini",
            input_data=[{"role": "user", "content": "hi"}],
            max_output_tokens=None,
            system_instruction="test",
            temperature=0.2,
            tools=None,
            tool_choice=None,
            web_search=False,
            search_context_size="medium",
            json_mode=False,
            expected_schema=None,
            reasoning_effort="medium",
        )
    assert "temperature" not in params
    assert any("does not support temperature" in r.message for r in caplog.records)


def test_get_all_responses_dummy(tmp_path):
    df = asyncio.run(openai_utils.get_all_responses(
        prompts=["a", "b"],
        identifiers=["1", "2"],
        save_path=str(tmp_path / "out.csv"),
        use_dummy=True,
    ))
    assert len(df) == 2
    assert set(["Successful", "Error Log"]).issubset(df.columns)
    assert df["Successful"].all()


def test_cap_adjustment_error_is_reported_and_recovers(monkeypatch, capsys, tmp_path):
    """Ensure concurrency tuning errors are surfaced once and runs continue."""

    async def run() -> pd.DataFrame:
        prompts = [f"prompt {i}" for i in range(4)]
        identifiers = [f"id{i}" for i in range(4)]

        real_planner = openai_utils._planned_ppm_and_details
        call_count = {"n": 0}

        def boom(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                return real_planner(*args, **kwargs)
            raise RuntimeError("synthetic cap failure")

        monkeypatch.setattr(openai_utils, "_planned_ppm_and_details", boom)
        return await openai_utils.get_all_responses(
            prompts=prompts,
            identifiers=identifiers,
            use_dummy=True,
            token_sample_size=1,
            quiet=True,
            verbose=False,
            print_example_prompt=False,
            save_path=str(tmp_path / "cap_failure.csv"),
        )

    df = asyncio.run(run())
    captured = capsys.readouterr().out
    assert len(df) == 4
    assert "Error while updating concurrency cap dynamically" in captured


def test_get_all_responses_quiet_minimizes_output(tmp_path, capsys):
    asyncio.run(
        openai_utils.get_all_responses(
            prompts=["a"],
            identifiers=["1"],
            save_path=str(tmp_path / "quiet.csv"),
            use_dummy=True,
            quiet=True,
            status_report_interval=0.01,
        )
    )
    captured = capsys.readouterr().out
    assert "Initializing model calls" not in captured


def test_get_all_responses_images_dummy(tmp_path):
    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["a"],
            identifiers=["1"],
            prompt_images={"1": ["abcd"]},
            save_path=str(tmp_path / "img.csv"),
            use_dummy=True,
        )
    )
    assert len(df) == 1


def test_get_all_responses_audio_dummy(tmp_path):
    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["a"],
            identifiers=["1"],
            prompt_audio={"1": [{"data": "abcd", "format": "mp3"}]},
            save_path=str(tmp_path / "aud.csv"),
            use_dummy=True,
        )
    )
    assert len(df) == 1


def test_get_all_responses_custom_callable(tmp_path):
    calls = []

    async def custom(prompt: str, *, n: int) -> list:
        calls.append((prompt, n))
        return [f"CUSTOM::{prompt}"]

    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["x", "y"],
            identifiers=["1", "2"],
            save_path=str(tmp_path / "custom.csv"),
            response_fn=custom,
            reset_files=True,
        )
    )
    assert sorted(calls) == [("x", 1), ("y", 1)]
    df = df.sort_values("Identifier").reset_index(drop=True)
    assert df.loc[0, "Response"] == ["CUSTOM::x"]


def test_get_all_responses_keyword_only_prompt(tmp_path):
    seen = []

    async def custom(*, prompt: str, model: str, json_mode: bool):
        seen.append((prompt, model, json_mode))
        return [prompt.upper()]

    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["hello"],
            identifiers=["row-1"],
            save_path=str(tmp_path / "kw.csv"),
            response_fn=custom,
            json_mode=True,
            model="alt-model",
            reset_files=True,
        )
    )
    assert seen == [("hello", "alt-model", True)]
    assert df.loc[0, "Response"] == ["HELLO"]


def test_get_all_responses_custom_callable_requires_prompt(tmp_path):
    async def missing_prompt_parameter():
        return ["oops"]

    with pytest.raises(TypeError, match="must accept a `prompt` argument"):
        asyncio.run(
            openai_utils.get_all_responses(
                prompts=["hello"],
                identifiers=["row-1"],
                save_path=str(tmp_path / "missing.csv"),
                response_fn=missing_prompt_parameter,  # type: ignore[arg-type]
            )
        )


def test_get_all_responses_custom_driver_receives_kwargs(tmp_path):
    calls: List[Dict[str, Any]] = []

    async def custom_driver(prompts, identifiers, json_mode=False, model=None, extra=None, **kwargs):
        calls.append(
            {
                "prompts": prompts,
                "identifiers": identifiers,
                "json_mode": json_mode,
                "model": model,
                "extra": extra,
                "kwargs": kwargs,
            }
        )
        return pd.DataFrame(
            {"Identifier": identifiers, "Response": [[f"resp:{ident}"] for ident in identifiers]}
        )

    save_path = str(tmp_path / "custom_driver.csv")

    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["p1", "p2"],
            identifiers=None,
            json_mode=True,
            model="demo-model",
            extra="value",
            get_all_responses_fn=custom_driver,
            save_path=save_path,
        )
    )

    assert calls and calls[0]["prompts"] == ["p1", "p2"]
    assert calls[0]["identifiers"] == ["p1", "p2"]
    assert calls[0]["json_mode"] is True
    assert calls[0]["model"] == "demo-model"
    assert calls[0]["extra"] == "value"
    assert calls[0]["kwargs"]["save_path"] == save_path
    assert df.loc[df["Identifier"] == "p1", "Response"].iloc[0] == ["resp:p1"]


def test_get_all_responses_custom_driver_requires_identifiers(tmp_path):
    async def missing_identifiers(prompts):
        return pd.DataFrame({"Identifier": prompts, "Response": [["ok"] for _ in prompts]})

    with pytest.raises(TypeError, match="identifiers"):
        asyncio.run(
            openai_utils.get_all_responses(
                prompts=["a"],
                identifiers=["a"],
                get_all_responses_fn=missing_identifiers,  # type: ignore[arg-type]
                save_path=str(tmp_path / "custom_driver_error.csv"),
            )
        )


def test_get_all_responses_cancellation_stops_workers(tmp_path):
    prompts = [f"p{i}" for i in range(6)]
    call_log: List[str] = []
    post_cancel_calls: List[str] = []
    cancel_flag = asyncio.Event()
    blocker = asyncio.Event()

    async def stalled_response(prompt: str, **kwargs):
        call_log.append(prompt)
        if cancel_flag.is_set():
            post_cancel_calls.append(prompt)
        await blocker.wait()

    async def runner():
        task = asyncio.create_task(
            openai_utils.get_all_responses(
                prompts=prompts,
                identifiers=[f"id{i}" for i in range(len(prompts))],
                save_path=str(tmp_path / "cancel.csv"),
                response_fn=stalled_response,
                n_parallels=3,
                verbose=False,
            )
        )
        while len(call_log) < 3:
            await asyncio.sleep(0.01)
        cancel_flag.set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not post_cancel_calls

    asyncio.run(runner())


def test_get_all_responses_keyboard_interrupt_cleanup(tmp_path):
    prompts = [f"p{i}" for i in range(5)]
    call_log: List[str] = []
    blocker = asyncio.Event()
    counts = {"at_interrupt": 0}

    async def stalled_response(prompt: str, **kwargs):
        call_log.append(prompt)
        await blocker.wait()

    async def driver():
        task = asyncio.create_task(
            openai_utils.get_all_responses(
                prompts=prompts,
                identifiers=[f"kid{i}" for i in range(len(prompts))],
                save_path=str(tmp_path / "kbint.csv"),
                response_fn=stalled_response,
                n_parallels=2,
                verbose=False,
            )
        )
        while len(call_log) < 2:
            await asyncio.sleep(0.01)
        counts["at_interrupt"] = len(call_log)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(driver())

    assert len(call_log) == counts["at_interrupt"]


def test_usage_overview_reports_remaining_budget_reason(capsys):
    openai_utils._print_usage_overview(
        prompts=["hello"],
        n=1,
        max_output_tokens=None,
        model="gpt-5-mini",
        use_batch=False,
        n_parallels=250,
        rate_headers={
            "limit_requests": "10000",
            "remaining_requests": "13",
            "limit_tokens": "30000000",
            "remaining_tokens": "29000000",
        },
    )
    captured = capsys.readouterr().out
    assert "13 request slots remaining" in captured
    assert "Upgrading your tier" not in captured


def test_web_search_warning_and_parallel_cap(tmp_path, capsys):
    asyncio.run(
        openai_utils.get_all_responses(
            prompts=["search"],
            identifiers=["1"],
            save_path=str(tmp_path / "web.csv"),
            use_dummy=True,
            web_search=True,
            n_parallels=12,
        )
    )
    captured = capsys.readouterr().out
    assert "Web search is enabled" in captured
    assert "automatically capped parallel workers" in captured


def test_get_all_responses_custom_usage(tmp_path):
    recorded_kwargs = []

    async def custom(prompt: str, **kwargs) -> tuple:
        recorded_kwargs.append(kwargs)
        return (
            [f"ANS:{prompt}"],
            0.5,
            [
                {
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 7,
                        "output_tokens_details": {"reasoning_tokens": 2},
                    },
                    "output": [
                        {
                            "type": "reasoning",
                            "summary": [{"text": f"summary-{prompt}"}],
                        }
                    ],
                }
            ],
        )

    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["z"],
            identifiers=["id"],
            save_path=str(tmp_path / "custom_usage.csv"),
            response_fn=custom,
            reasoning_summary="short",
            reset_files=True,
        )
    )
    assert recorded_kwargs and recorded_kwargs[0].get("return_raw") is True
    row = df.iloc[0]
    assert row["Response"] == ["ANS:z"]
    assert row["Input Tokens"] == 10
    assert row["Output Tokens"] == 7
    assert row["Reasoning Tokens"] == 2
    assert row["Reasoning Summary"] == "summary-z"
    assert bool(row["Successful"])
    assert row["Time Taken"] == pytest.approx(0.5)


def test_get_all_responses_prompt_web_filters(tmp_path):
    seen_filters: List[Optional[Dict[str, Any]]] = []

    async def capture(prompt: str, **kwargs):
        seen_filters.append(kwargs.get("web_search_filters"))
        return (
            [f"OK:{prompt}"],
            0.1,
            [
                {
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "output_tokens_details": {"reasoning_tokens": 0},
                    }
                }
            ],
        )

    asyncio.run(
        openai_utils.get_all_responses(
            prompts=["a", "b"],
            identifiers=["one", "two"],
            save_path=str(tmp_path / "web.csv"),
            response_fn=capture,
            web_search=True,
            web_search_filters={"allowed_domains": ["example.com"]},
            prompt_web_search_filters={"two": {"city": "Paris"}},
        )
    )

    assert seen_filters[0] == {"allowed_domains": ["example.com"]}
    assert seen_filters[1] == {"allowed_domains": ["example.com"], "city": "Paris"}

def test_ratings_dummy(tmp_path):
    cfg = RateConfig(attributes={"helpfulness": ""}, save_dir=str(tmp_path), file_name="ratings.csv", use_dummy=True)
    task = Rate(cfg)
    data = pd.DataFrame({"text": ["hello"]})
    df = asyncio.run(task.run(data, column_name="text"))
    assert not df.empty
    assert "helpfulness" in df.columns


def test_ratings_multirun(tmp_path):
    cfg = RateConfig(attributes={"helpfulness": ""}, save_dir=str(tmp_path), file_name="ratings.csv", use_dummy=True, n_runs=2)
    task = Rate(cfg)
    data = pd.DataFrame({"text": ["hello"]})
    df = asyncio.run(task.run(data, column_name="text"))
    assert "helpfulness" in df.columns
    disagg = pd.read_csv(tmp_path / "ratings_full_disaggregated.csv", index_col=[0, 1])
    assert set(disagg.index.names) == {"id", "run"}


def test_ratings_ignore_stale_ids(tmp_path):
    """Ensure stale identifiers in existing files are ignored."""
    cfg = RateConfig(
        attributes={"helpfulness": ""},
        save_dir=str(tmp_path),
        file_name="ratings.csv",
        use_dummy=True,
    )
    # Pre-create a raw responses file with an unrelated identifier
    raw_path = tmp_path / "ratings_raw_responses.csv"
    stale = pd.DataFrame(
        [
            {
                "Identifier": "stale_batch0",
                "Response": openai_utils._ser(["{\"helpfulness\": 1}"]),
                "Time Taken": 0.1,
                "Input Tokens": 1,
                "Reasoning Tokens": 0,
                "Output Tokens": 1,
                "Reasoning Effort": None,
                "Successful": True,
                "Error Log": openai_utils._ser(None),
            }
        ]
    )
    stale.to_csv(raw_path, index=False)
    task = Rate(cfg)
    data = pd.DataFrame({"text": ["hello"]})
    df = asyncio.run(task.run(data, column_name="text"))
    assert "helpfulness" in df.columns


def test_ratings_audio_dummy(tmp_path):
    cfg = RateConfig(
        attributes={"clarity": ""},
        save_dir=str(tmp_path),
        file_name="ratings.csv",
        use_dummy=True,
        modality="audio",
    )
    task = Rate(cfg)
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"abcd")
    data = pd.DataFrame({"audio": [str(audio_path)]})
    df = asyncio.run(task.run(data, column_name="audio"))
    assert "clarity" in df.columns


def test_ratings_image_dummy(tmp_path):
    cfg = RateConfig(
        attributes={"clarity": ""},
        save_dir=str(tmp_path),
        file_name="ratings.csv",
        use_dummy=True,
        modality="image",
    )
    task = Rate(cfg)
    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"abcd")
    data = pd.DataFrame({"image": [str(img_path)]})
    df = asyncio.run(task.run(data, column_name="image"))
    assert "clarity" in df.columns


def test_rank_audio_dummy(tmp_path):
    cfg = RankConfig(
        attributes={"clear": "", "inspiring": ""},
        save_dir=str(tmp_path),
        file_name="rankings.csv",
        use_dummy=True,
        modality="audio",
        n_rounds=1,
        matches_per_round=1,
        n_parallels=5,
    )
    task = Rank(cfg)
    # Provide pre-encoded audio so no actual files are needed
    data = pd.DataFrame(
        {
            "audio": [
                [{"data": "abcd", "format": "mp3"}],
                [{"data": "efgh", "format": "mp3"}],
            ]
        }
    )
    df = asyncio.run(task.run(data, column_name="audio"))
    assert "clear" in df.columns and "inspiring" in df.columns


def test_rank_outputs_zscores_and_raw_columns(tmp_path):
    cfg = RankConfig(
        attributes={"clarity": "", "originality": ""},
        save_dir=str(tmp_path),
        file_name="rankings.csv",
        use_dummy=True,
        n_rounds=1,
        matches_per_round=1,
        n_parallels=4,
    )
    task = Rank(cfg)
    data = pd.DataFrame({"text": ["first", "second"]})
    df = asyncio.run(task.run(data, column_name="text"))
    for attr in ("clarity", "originality"):
        assert attr in df.columns
        assert f"{attr}_raw" in df.columns
        assert f"{attr}_se" in df.columns
        assert np.isfinite(df[f"{attr}_se"].fillna(0.0)).all()


def test_rank_outputs_standard_errors(tmp_path):
    cfg = RankConfig(
        attributes={"clarity": ""},
        save_dir=str(tmp_path),
        file_name="rankings.csv",
        use_dummy=True,
        n_rounds=1,
        matches_per_round=1,
        n_parallels=4,
    )
    task = Rank(cfg)
    data = pd.DataFrame({"text": ["first", "second"]})
    df = asyncio.run(task.run(data, column_name="text"))
    assert "clarity_se" in df.columns
    assert np.isfinite(df["clarity_se"].fillna(0.0)).all()


def test_rank_primer_centering():
    cfg = RankConfig(attributes={"clarity": ""})
    task = Rank(cfg)
    ratings = {"a": {"clarity": 0.0}, "b": {"clarity": 0.0}}
    primer = {"a": {"clarity": 10.0}, "b": {"clarity": 30.0}}
    task._apply_primer(ratings, primer, ["clarity"])
    assert ratings["a"]["clarity"] == -10.0
    assert ratings["b"]["clarity"] == 10.0


def test_recursive_rank_outputs(tmp_path):
    cfg = RankConfig(
        attributes={"clarity": ""},
        save_dir=str(tmp_path),
        file_name="rankings.csv",
        use_dummy=True,
        recursive=True,
        recursive_fraction=0.5,
        recursive_min_remaining=1,
        recursive_rate_first_round=False,
        n_rounds=1,
        matches_per_round=1,
        n_parallels=4,
    )
    task = Rank(cfg)
    data = pd.DataFrame({"text": ["alpha", "beta", "gamma"]})
    df = asyncio.run(task.run(data, column_name="text"))
    assert "overall_rank" in df.columns
    assert "exit_stage" in df.columns
    assert "clarity" in df.columns
    assert any(col.startswith("stage1_") for col in df.columns)
    assert "identifier" not in df.columns
    assert not any(col.startswith("final_") for col in df.columns)
    assert not any(col.startswith("cumulative_") for col in df.columns)


def test_api_rank_hides_raw_columns(tmp_path):
    data = pd.DataFrame({"text": ["first", "second"]})
    df = asyncio.run(
        gabriel.rank(
            data,
            "text",
            attributes={"clarity": "", "originality": ""},
            save_dir=str(tmp_path),
            file_name="rankings.csv",
            use_dummy=True,
            n_rounds=1,
            matches_per_round=1,
            n_parallels=4,
        )
    )
    for attr in ("clarity", "originality"):
        assert attr in df.columns
        assert f"{attr}_raw" not in df.columns
        assert f"{attr}_se" not in df.columns
    final_path = tmp_path / "rankings_final.csv"
    saved = pd.read_csv(final_path)
    for attr in ("clarity", "originality"):
        assert f"{attr}_raw" in saved.columns
        assert f"{attr}_se" in saved.columns


def test_api_rank_can_return_raw_scores_when_requested(tmp_path):
    data = pd.DataFrame({"text": ["first", "second"]})
    df = asyncio.run(
        gabriel.rank(
            data,
            "text",
            attributes={"clarity": "", "originality": ""},
            save_dir=str(tmp_path),
            file_name="rankings.csv",
            use_dummy=True,
            n_rounds=1,
            matches_per_round=1,
            n_parallels=4,
            return_raw_scores=True,
        )
    )
    for attr in ("clarity", "originality"):
        assert attr in df.columns
        assert f"{attr}_raw" in df.columns
        assert f"{attr}_se" in df.columns


def test_deidentifier_dummy(tmp_path):
    cfg = DeidentifyConfig(save_dir=str(tmp_path), file_name="deid.csv", use_dummy=True)
    task = Deidentifier(cfg)
    data = pd.DataFrame({"text": ["John went to Paris."]})
    df = asyncio.run(task.run(data, column_name="text"))
    assert "deidentified_text" in df.columns


def test_deidentifier_respects_punctuation_in_real_forms(tmp_path):
    cfg = DeidentifyConfig(
        save_dir=str(tmp_path),
        file_name="deid.csv",
        use_dummy=True,
        use_existing_mappings_only=True,
    )
    task = Deidentifier(cfg)
    mapping = {
        "person's name": {
            "real forms": ["Gabriel R.", "Gabriel R"],
            "casted form": "Miles P.",
        }
    }
    data = pd.DataFrame(
        {
            "text": ["Gabriel R. met with Gabriel R at the library."],
            "existing_map": [mapping],
        }
    )
    df = asyncio.run(
        task.run(
            data,
            column_name="text",
            mapping_column="existing_map",
        )
    )
    output = df["deidentified_text"].iloc[0]
    assert output.count("Miles P.") == 2
    assert "Gabriel R" not in output


def test_classification_dummy(tmp_path):
    cfg = ClassifyConfig(labels={"yes": ""}, save_dir=str(tmp_path), use_dummy=True)
    task = Classify(cfg)
    df = pd.DataFrame({"txt": ["a", "b"]})
    res = asyncio.run(task.run(df, column_name="txt"))
    assert "yes" in res.columns
    assert "predicted_classes" in res.columns
    assert res.predicted_classes.tolist() == [[], []]


def test_extraction_dummy(tmp_path):
    cfg = ExtractConfig(attributes={"year": ""}, save_dir=str(tmp_path), use_dummy=True)
    task = Extract(cfg)
    df = pd.DataFrame({"txt": ["a"]})
    res = asyncio.run(task.run(df, column_name="txt"))
    assert "year" in res.columns
    assert "entity_name" in res.columns
    assert res["entity_name"].isna().all()


def test_extraction_multiple_entities(tmp_path):
    cfg = ExtractConfig(
        attributes={"year": "", "price": ""},
        save_dir=str(tmp_path),
        use_dummy=True,
    )
    task = Extract(cfg)
    df = pd.DataFrame({"txt": ["listing"]})
    payload = json.dumps(
        {
            "Alpha": {"year": "1990", "price": "10"},
            "Beta": {"year": "2000", "price": "20"},
        }
    )
    specs = {"*": DummyResponseSpec(responses=[payload])}
    res = asyncio.run(
        task.run(
            df,
            column_name="txt",
            dummy_responses=specs,
            reset_files=True,
        )
    )
    assert "entity_name" in res.columns
    assert res.shape[0] == 2
    assert sorted(res["entity_name"].dropna()) == ["Alpha", "Beta"]
    assert set(res["year"].dropna()) == {"1990", "2000"}
    assert set(res["price"].dropna()) == {"10", "20"}


def test_classification_multirun(tmp_path):
    cfg = ClassifyConfig(labels={"yes": ""}, save_dir=str(tmp_path), use_dummy=True, n_runs=2)
    task = Classify(cfg)
    df = pd.DataFrame({"txt": ["a"]})
    res = asyncio.run(task.run(df, column_name="txt"))
    assert "yes" in res.columns
    assert res.predicted_classes.iloc[0] == []
    disagg = pd.read_csv(tmp_path / "classify_responses_full_disaggregated.csv", index_col=[0, 1])
    assert set(disagg.index.names) == {"text", "run"}


def test_discover_uses_cached_labels_when_definitions_diverge(monkeypatch, tmp_path):
    df = pd.DataFrame({"circle": ["alpha"], "square": ["beta"]})
    cached_combined = {
        "circle cached label more than square": "cached desc circle",
        "square cached label more than circle": "cached desc square",
    }
    new_bucket = {
        "circle fresh label more than square": "fresh desc",
    }

    class DummyCompare:
        def __init__(self, cfg: Any):
            self.cfg = cfg

        async def run(
            self,
            df: pd.DataFrame,
            circle_column_name: str,
            square_column_name: str,
            *,
            reset_files: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {"attribute": list(new_bucket.keys()), "explanation": list(new_bucket.values())}
            )

    class DummyBucket:
        def __init__(self, cfg: Any):
            self.cfg = cfg

        async def run(
            self,
            df: pd.DataFrame,
            column_name: str,
            *,
            reset_files: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {"bucket": list(new_bucket.keys()), "definition": list(new_bucket.values())}
            )

    class DummyClassify:
        def __init__(self, cfg: Any):
            self.cfg = cfg

        async def run(
            self,
            df: pd.DataFrame,
            *,
            circle_column_name: Optional[str] = None,
            square_column_name: Optional[str] = None,
            reset_files: bool = False,
            **_: Any,
        ) -> pd.DataFrame:
            self.cfg.labels = dict(cached_combined)
            subset = df[[circle_column_name, square_column_name]].copy()
            for lab in cached_combined:
                subset[lab] = [True] * len(subset)
            subset["predicted_classes"] = [[] for _ in range(len(subset))]
            return subset

    monkeypatch.setattr(discover_module, "Compare", DummyCompare)
    monkeypatch.setattr(discover_module, "Bucket", DummyBucket)
    monkeypatch.setattr(discover_module, "Classify", DummyClassify)

    cfg = DiscoverConfig(save_dir=str(tmp_path / "disc"), use_dummy=True, bucket_count=1)
    result = asyncio.run(
        Discover(cfg).run(
            df,
            circle_column_name="circle",
            square_column_name="square",
        )
    )

    expected_label = "circle cached label more than square"
    assert list(result["buckets"].keys()) == [expected_label]
    classification_cols = result["classification"].columns
    assert f"{expected_label}_actual" in classification_cols
    assert f"{expected_label}_inverted" in classification_cols


def test_discover_single_column_uses_cached_labels(monkeypatch, tmp_path):
    df = pd.DataFrame({"text": ["alpha"]})
    cached_labels = {"cached": "cached desc"}
    fresh_labels = {"fresh": "fresh desc"}

    class DummyCodify:
        def __init__(self, cfg: Any):
            self.cfg = cfg

        async def run(
            self,
            df: pd.DataFrame,
            column_name: str,
            *,
            categories: Optional[Any] = None,
            additional_instructions: str = "",
            reset_files: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame({"coded_passages": [{"fresh": ["desc"]}]})

    class DummyBucket:
        def __init__(self, cfg: Any):
            self.cfg = cfg

        async def run(
            self,
            df: pd.DataFrame,
            column_name: str,
            *,
            reset_files: bool = False,
        ) -> pd.DataFrame:
            return pd.DataFrame(
                {"bucket": list(fresh_labels.keys()), "definition": list(fresh_labels.values())}
            )

    class DummyClassify:
        def __init__(self, cfg: Any):
            self.cfg = cfg

        async def run(
            self,
            df: pd.DataFrame,
            column_name: Optional[str] = None,
            *,
            reset_files: bool = False,
            **_: Any,
        ) -> pd.DataFrame:
            self.cfg.labels = dict(cached_labels)
            subset = df[[column_name]].copy() if column_name else df.copy()
            for lab in cached_labels:
                subset[lab] = [True] * len(subset)
            subset["predicted_classes"] = [[] for _ in range(len(subset))]
            return subset

    monkeypatch.setattr(discover_module, "Codify", DummyCodify)
    monkeypatch.setattr(discover_module, "Bucket", DummyBucket)
    monkeypatch.setattr(discover_module, "Classify", DummyClassify)

    cfg = DiscoverConfig(
        save_dir=str(tmp_path / "disc_single"),
        use_dummy=True,
        bucket_count=1,
        raw_term_definitions=False,
    )
    result = asyncio.run(Discover(cfg).run(df, column_name="text"))
    assert result["buckets"] == cached_labels
    assert set(result["classification"].columns).issuperset(cached_labels.keys())


def test_collect_predictions_np_bool():
    row = pd.Series({"speech": np.bool_(True), "beeps": np.bool_(False), "space": None})
    assert _collect_predictions(row) == ["speech"]


def test_bucket_reuses_final_state(tmp_path):
    cfg = BucketConfig(save_dir=str(tmp_path / "bucket_state"), use_dummy=True, bucket_count=1)
    task = Bucket(cfg)
    df = pd.DataFrame({"term": ["alpha", "beta"]})
    term_map = {"alpha": "", "beta": ""}
    signature = task._terms_signature(list(term_map.keys()), term_map)
    state = {
        "terms_signature": signature,
        "finalized": True,
        "final_buckets": [{"bucket": "cached", "definition": "saved"}],
    }
    state_path = Path(cfg.save_dir) / "bucket_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state))

    result = asyncio.run(task.run(df, column_name="term"))
    assert result.to_dict(orient="records") == state["final_buckets"]


def test_classify_parse_dict(tmp_path):
    cfg = ClassifyConfig(labels={"yes": ""}, save_dir=str(tmp_path), use_dummy=True)
    task = Classify(cfg)
    parsed = asyncio.run(task._parse({"yes": True}, ["yes"]))
    assert parsed["yes"] is True
    

def test_api_wrappers(tmp_path):
    df = pd.DataFrame({"txt": ["hello"]})
    rated = asyncio.run(
        gabriel.rate(
            df,
            "txt",
            attributes={"clarity": ""},
            save_dir=str(tmp_path / "rate"),
            use_dummy=True,
        )
    )
    assert "clarity" in rated.columns

    classified = asyncio.run(
        gabriel.classify(
            df,
            "txt",
            labels={"yes": ""},
            save_dir=str(tmp_path / "cls"),
            use_dummy=True,
        )
    )
    assert "yes" in classified.columns

    extracted = asyncio.run(
        gabriel.extract(
            df,
            "txt",
            attributes={"year": ""},
            save_dir=str(tmp_path / "extr"),
            use_dummy=True,
        )
    )
    assert "year" in extracted.columns

    deidentified = asyncio.run(
        gabriel.deidentify(
            df,
            "txt",
            save_dir=str(tmp_path / "deid"),
            use_dummy=True,
        )
    )
    assert "deidentified_text" in deidentified.columns

    custom = asyncio.run(
        gabriel.whatever(
            prompts=["hello"],
            identifiers=["1"],
            save_dir=str(tmp_path / "cust"),
            file_name="out.csv",
            use_dummy=True,
        )
    )
    assert len(custom) == 1


def test_extract_custom_response_functions(tmp_path):
    df = pd.DataFrame({"txt": ["hello"]})
    seen: Dict[str, Any] = {}

    async def custom_response_fn(prompt: str, **_kwargs: Any):
        seen["response_fn"] = prompt
        return ['{"hello": {"year": "2024"}}']

    custom_response = asyncio.run(
        gabriel.extract(
            df,
            "txt",
            attributes={"year": ""},
            save_dir=str(tmp_path / "extr-response"),
            response_fn=custom_response_fn,
        )
    )
    assert "hello" in seen["response_fn"]
    assert custom_response.loc[0, "entity_name"] == "hello"
    assert custom_response.loc[0, "year"] == "2024"

    calls: List[Tuple[List[str], List[str]]] = []

    async def custom_driver(prompts, identifiers, **_kwargs):
        calls.append((prompts, identifiers))
        return pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": [['{"hello": {"year": "1999"}}'] for _ in identifiers],
            }
        )

    custom_driver_response = asyncio.run(
        gabriel.extract(
            df,
            "txt",
            attributes={"year": ""},
            save_dir=str(tmp_path / "extr-driver"),
            get_all_responses_fn=custom_driver,
        )
    )
    assert len(calls) == 1
    assert calls[0][1] == ["aaf4c61d_batch0"]
    assert "hello" in calls[0][0][0]
    assert custom_driver_response.loc[0, "entity_name"] == "hello"
    assert custom_driver_response.loc[0, "year"] == "1999"


def test_whatever_dataframe_inputs(tmp_path, monkeypatch):
    captured: Dict[str, Any] = {}

    async def fake_get_all_responses(**kwargs):
        captured.update(kwargs)
        identifiers = kwargs["identifiers"]
        df = pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": [["OK"] for _ in identifiers],
                "Successful": [True for _ in identifiers],
                "Error Log": [[] for _ in identifiers],
                "Time Taken": [0.1 for _ in identifiers],
                "Input Tokens": [1 for _ in identifiers],
                "Reasoning Tokens": [0 for _ in identifiers],
                "Output Tokens": [1 for _ in identifiers],
                "Reasoning Effort": [None for _ in identifiers],
            }
        )
        return df

    monkeypatch.setattr("gabriel.tasks.whatever.get_all_responses", fake_get_all_responses)

    data = pd.DataFrame(
        {
            "prompt": ["Hi", "Bye"],
            "img": [["img1"], None],
            "aud": [None, [{"data": "a", "format": "mp3"}]],
            "city_col": ["Austin", "Paris"],
            "domains": [["example.com"], ["news.com", "blog.com"]],
            "ident": ["row1", "row2"],
        }
    )

    result = asyncio.run(
        gabriel.whatever(
            data,
            save_dir=str(tmp_path / "whatever"),
            column_name="prompt",
            identifier_column="ident",
            image_column="img",
            audio_column="aud",
            web_search_filters={"city": "city_col", "allowed_domains": "domains"},
            use_dummy=True,
        )
    )

    assert captured["prompts"] == ["Hi", "Bye"]
    assert captured["identifiers"] == ["row1", "row2"]
    assert captured["prompt_images"]["row1"] == ["img1"]
    assert "row2" not in captured["prompt_images"]
    assert captured["prompt_audio"]["row2"][0]["format"] == "mp3"
    assert captured["prompt_web_search_filters"]["row1"] == {
        "city": "Austin",
        "allowed_domains": ["example.com"],
    }
    assert captured["prompt_web_search_filters"]["row2"]["allowed_domains"] == [
        "news.com",
        "blog.com",
    ]
    assert result.shape[0] == 2


def test_whatever_dataframe_kwarg(tmp_path, monkeypatch):
    captured: Dict[str, Any] = {}

    async def fake_get_all_responses(**kwargs):
        captured.update(kwargs)
        identifiers = kwargs["identifiers"]
        df = pd.DataFrame(
            {
                "Identifier": identifiers,
                "Response": [["OK"] for _ in identifiers],
                "Successful": [True for _ in identifiers],
                "Error Log": [[] for _ in identifiers],
                "Time Taken": [0.1 for _ in identifiers],
                "Input Tokens": [1 for _ in identifiers],
                "Reasoning Tokens": [0 for _ in identifiers],
                "Output Tokens": [1 for _ in identifiers],
                "Reasoning Effort": [None for _ in identifiers],
            }
        )
        return df

    monkeypatch.setattr("gabriel.tasks.whatever.get_all_responses", fake_get_all_responses)

    data = pd.DataFrame(
        {
            "prompt": ["Hi", "Bye"],
            "img": [["img1"], None],
            "aud": [None, [{"data": "a", "format": "mp3"}]],
            "city_col": ["Austin", "Paris"],
            "domains": [["example.com"], ["news.com", "blog.com"]],
            "ident": ["row1", "row2"],
        }
    )

    result = asyncio.run(
        gabriel.whatever(
            save_dir=str(tmp_path / "whatever_kwarg"),
            df=data,
            column_name="prompt",
            identifier_column="ident",
            image_column="img",
            audio_column="aud",
            web_search_filters={"city": "city_col", "allowed_domains": "domains"},
            use_dummy=True,
        )
    )

    assert captured["prompts"] == ["Hi", "Bye"]
    assert captured["identifiers"] == ["row1", "row2"]
    assert captured["prompt_images"]["row1"] == ["img1"]
    assert "row2" not in captured["prompt_images"]
    assert captured["prompt_audio"]["row2"][0]["format"] == "mp3"
    assert captured["prompt_web_search_filters"]["row1"] == {
        "city": "Austin",
        "allowed_domains": ["example.com"],
    }
    assert captured["prompt_web_search_filters"]["row2"]["allowed_domains"] == [
        "news.com",
        "blog.com",
    ]
    assert result.shape[0] == 2


def test_paraphrase_api(tmp_path):
    data = pd.DataFrame({"txt": ["hello"]})
    df = asyncio.run(
        gabriel.paraphrase(
            data,
            "txt",
            instructions="reword",
            save_dir=str(tmp_path / "para"),
            use_dummy=True,
        )
    )
    assert "txt_revised" in df.columns and len(df) == 1
    df_multi = asyncio.run(
        gabriel.paraphrase(
            data,
            "txt",
            instructions="reword",
            save_dir=str(tmp_path / "para_multi"),
            use_dummy=True,
            n_runs=2,
        )
    )
    assert "txt_revised_1" in df_multi.columns and "txt_revised_2" in df_multi.columns


def test_paraphrase_modalities_forward_media(monkeypatch, tmp_path):
    captured = {}

    async def fake_get_all_responses(*, prompts, identifiers, **kwargs):
        captured["prompts"] = list(prompts)
        captured["identifiers"] = list(identifiers)
        captured["prompt_images"] = kwargs.get("prompt_images")
        captured["prompt_audio"] = kwargs.get("prompt_audio")
        captured["prompt_pdfs"] = kwargs.get("prompt_pdfs")
        captured["web_search"] = kwargs.get("web_search")
        captured["search_context_size"] = kwargs.get("search_context_size")
        return pd.DataFrame({"Identifier": identifiers, "Response": prompts})

    monkeypatch.setattr("gabriel.tasks.paraphrase.get_all_responses", fake_get_all_responses)

    df_image = pd.DataFrame({"media": ["data:image/png;base64,xyz"]})
    asyncio.run(
        gabriel.paraphrase(
            df_image,
            "media",
            instructions="caption it",
            save_dir=str(tmp_path / "para_image"),
            modality="image",
        )
    )
    assert captured["prompt_images"]["row_0_rev1"] == ["data:image/png;base64,xyz"]

    df_audio = pd.DataFrame({"media": [{"format": "mp3", "data": "abcd"}]})
    asyncio.run(
        gabriel.paraphrase(
            df_audio,
            "media",
            instructions="summarize it",
            save_dir=str(tmp_path / "para_audio"),
            modality="audio",
        )
    )
    assert captured["prompt_audio"]["row_0_rev1"][0]["format"] == "mp3"

    df_pdf = pd.DataFrame({"media": ["data:application/pdf;base64,abcd"]})
    asyncio.run(
        gabriel.paraphrase(
            df_pdf,
            "media",
            instructions="rewrite",
            save_dir=str(tmp_path / "para_pdf"),
            modality="pdf",
        )
    )
    assert captured["prompt_pdfs"]["row_0_rev1"][0]["file_data"].startswith(
        "data:application/pdf"
    )

    df_web = pd.DataFrame({"query": ["Mount Fuji"]})
    asyncio.run(
        gabriel.paraphrase(
            df_web,
            "query",
            instructions="summarize",
            save_dir=str(tmp_path / "para_web"),
            modality="web",
        )
    )
    assert captured["web_search"] is True
    assert captured["search_context_size"] == "medium"


def test_paraphrase_n_rounds_not_forwarded(monkeypatch, tmp_path):
    captured = {}

    async def fake_get_all_responses(*, prompts, identifiers, **kwargs):
        captured["kwargs"] = dict(kwargs)
        return pd.DataFrame({"Identifier": identifiers, "Response": prompts})

    monkeypatch.setattr("gabriel.tasks.paraphrase.get_all_responses", fake_get_all_responses)

    data = pd.DataFrame({"txt": ["hello"]})
    df = asyncio.run(
        gabriel.paraphrase(
            data,
            "txt",
            instructions="reword",
            save_dir=str(tmp_path / "para_rounds"),
            n_rounds=1,
        )
    )

    assert "n_rounds" not in captured["kwargs"]
    assert df["txt_revised_approved"].tolist() == [True]


def test_paraphrase_approval_column_recursive(monkeypatch, tmp_path):
    async def fake_recursive_validate(
        self,
        original_texts,
        original_values,
        resp_map,
        approval_map,
        *,
        reset_files,
        max_rounds,
    ):
        for idx in range(len(original_texts)):
            resp_map[(idx, 0)] = f"approved {idx}"
            approval_map[(idx, 0)] = (idx % 2 == 0)

    monkeypatch.setattr("gabriel.tasks.paraphrase.Paraphrase._recursive_validate", fake_recursive_validate)

    data = pd.DataFrame({"txt": ["a", "b"]})
    df = asyncio.run(
        gabriel.paraphrase(
            data,
            "txt",
            instructions="reword",
            save_dir=str(tmp_path / "para_recursive"),
            n_rounds=2,
        )
    )

    assert df["txt_revised"].tolist() == ["approved 0", "approved 1"]
    assert df["txt_revised_approved"].tolist() == [True, False]
