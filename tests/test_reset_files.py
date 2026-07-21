import asyncio
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from gabriel.utils import openai_utils


def test_get_all_responses_reset_files(tmp_path):
    save_path = tmp_path / "out.csv"
    asyncio.run(
        openai_utils.get_all_responses(
            prompts=["a", "b"],
            identifiers=["1", "2"],
            save_path=str(save_path),
            use_dummy=True,
        )
    )
    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["b"],
            identifiers=["2"],
            save_path=str(save_path),
            use_dummy=True,
            reset_files=True,
        )
    )
    assert set(df["Identifier"]) == {"2"}


def test_resume_treats_string_success_values_as_completed(tmp_path):
    save_path = tmp_path / "out.csv"
    pd.DataFrame(
        {
            "Identifier": ["1", "2", "3"],
            "Response": ["[]", "[]", "[]"],
            "Web Search Sources": ["[]", "[]", "[]"],
            "Time Taken": [0.1, 0.1, 0.1],
            "Input Tokens": [1, 1, 1],
            "Reasoning Tokens": [0, 0, 0],
            "Output Tokens": [1, 1, 1],
            "Reasoning Effort": ["default", "default", "default"],
            "Successful": ["True", "true", "1"],
            "Error Log": ["[]", "[]", "[]"],
            "Response IDs": ["[]", "[]", "[]"],
            "Reasoning Summary": ["", "", ""],
        }
    ).to_csv(save_path, index=False)

    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["a", "b", "c"],
            identifiers=["1", "2", "3"],
            save_path=str(save_path),
            use_dummy=True,
            reset_files=False,
        )
    )

    assert len(df) == 3


def test_successful_retry_replaces_failed_checkpoint_row(tmp_path):
    save_path = tmp_path / "out.csv"
    pd.DataFrame(
        {
            "Identifier": ["retry-me"],
            "Response": [""],
            "Web Search Sources": ["[]"],
            "Time Taken": [0.1],
            "Input Tokens": [1],
            "Reasoning Tokens": [0],
            "Output Tokens": [0],
            "Reasoning Effort": ["default"],
            "Successful": [False],
            "Error Log": ["previous failure"],
        }
    ).to_csv(save_path, index=False)
    calls = []

    async def responder(prompt: str, **kwargs):
        calls.append(prompt)
        return ["retried successfully"]

    result = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["prompt"],
            identifiers=["retry-me"],
            save_path=str(save_path),
            response_fn=responder,
            reset_files=False,
            skip_tail_fails=False,
            n_parallels=1,
            ramp_up_seconds=0,
            dynamic_timeout=False,
            manage_rate_limits=False,
            status_report_interval=None,
            verbose=False,
        )
    )

    assert calls == ["prompt"]
    assert len(result) == 1
    assert bool(result.iloc[0]["Successful"])
    persisted = pd.read_csv(save_path)
    assert len(persisted) == 1
    assert persisted.iloc[0]["Identifier"] == "retry-me"
    assert bool(persisted.iloc[0]["Successful"])


def test_batch_submission_intent_prevents_ambiguous_resubmission(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    class FakeFiles:
        def __init__(self):
            self.count = 0

        async def create(self, **kwargs):
            self.count += 1
            return SimpleNamespace(id=f"file-{self.count}")

    class FakeBatches:
        def __init__(self):
            self.create_calls = 0

        async def create(self, **kwargs):
            self.create_calls += 1
            if self.create_calls == 2:
                raise ConnectionError("ambiguous submission failure")
            return SimpleNamespace(id="batch-1")

    client = SimpleNamespace(files=FakeFiles(), batches=FakeBatches())
    monkeypatch.setattr(openai_utils, "_get_client", lambda *args: client)
    save_path = tmp_path / "batch.csv"

    with pytest.raises(ConnectionError, match="ambiguous"):
        asyncio.run(
            openai_utils.get_all_responses(
                prompts=["first", "second"],
                identifiers=["1", "2"],
                save_path=str(save_path),
                use_batch=True,
                batch_wait_for_completion=False,
                max_batch_requests=1,
                print_example_prompt=False,
                verbose=False,
            )
        )

    state_path = tmp_path / "batch.csv.batch_state.json"
    state = json.loads(state_path.read_text())
    assert state["batches"][0]["batch_id"] == "batch-1"
    assert state["batches"][1]["status"] == "submitting"
    create_calls = client.batches.create_calls
    with pytest.raises(RuntimeError, match="unresolved Batch API submission"):
        asyncio.run(
            openai_utils.get_all_responses(
                prompts=["first", "second"],
                identifiers=["1", "2"],
                save_path=str(save_path),
                use_batch=True,
                batch_wait_for_completion=False,
                max_batch_requests=1,
                print_example_prompt=False,
                verbose=False,
            )
        )
    assert client.batches.create_calls == create_calls


def test_completed_batch_state_is_retained_until_rows_are_durable(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    class FakeFiles:
        async def content(self, output_file_id):
            return json.dumps(
                {
                    "custom_id": "1",
                    "response": {
                        "body": {"output_text": "paid result"},
                        "usage": {},
                    },
                }
            )

    class FakeBatches:
        async def retrieve(self, batch_id):
            return SimpleNamespace(
                status="completed",
                output_file_id="output-1",
                error_file_id=None,
            )

    client = SimpleNamespace(files=FakeFiles(), batches=FakeBatches())
    monkeypatch.setattr(openai_utils, "_get_client", lambda *args: client)
    save_path = tmp_path / "batch.csv"
    state_path = tmp_path / "batch.csv.batch_state.json"
    original_state = {
        "batches": [
            {
                "batch_id": "batch-1",
                "input_file_id": "input-1",
                "total": 1,
                "status": "submitted",
            }
        ]
    }
    state_path.write_text(json.dumps(original_state))

    def fail_csv_write(self, *args, **kwargs):
        raise OSError("simulated durable CSV failure")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_csv_write)
    with pytest.raises(OSError, match="durable CSV"):
        asyncio.run(
            openai_utils.get_all_responses(
                prompts=["first"],
                identifiers=["1"],
                save_path=str(save_path),
                use_batch=True,
                batch_wait_for_completion=True,
                batch_poll_interval=0,
                print_example_prompt=False,
                verbose=False,
            )
        )

    assert json.loads(state_path.read_text()) == original_state


def test_resume_skip_tail_fails_returns_checkpoint_without_retry(tmp_path, capsys):
    save_path = tmp_path / "out.csv"
    total_rows = 5_001
    identifiers = [str(i) for i in range(total_rows)]
    successful = [True] * total_rows
    successful[-1] = False
    pd.DataFrame(
        {
            "Identifier": identifiers,
            "Response": [openai_utils._ser(["cached"])] * total_rows,
            "Web Search Sources": [openai_utils._ser([])] * total_rows,
            "Time Taken": [0.1] * total_rows,
            "Input Tokens": [1] * total_rows,
            "Reasoning Tokens": [0] * total_rows,
            "Output Tokens": [1] * total_rows,
            "Reasoning Effort": ["default"] * total_rows,
            "Successful": successful,
            "Error Log": [openai_utils._ser([])] * total_rows,
        }
    ).to_csv(save_path, index=False)

    async def should_not_retry(prompt: str, **kwargs):
        raise AssertionError("Tail failures should have been skipped.")

    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["prompt"] * total_rows,
            identifiers=identifiers,
            save_path=str(save_path),
            response_fn=should_not_retry,
            verbose=False,
        )
    )

    output = capsys.readouterr().out
    assert "skip_tail_fails=False" in output
    assert "1/5,001" in output
    assert len(df) == total_rows
    success_mask = df["Successful"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    assert int(success_mask.sum()) == total_rows - 1


def test_resume_skip_tail_fails_false_retries_incomplete_rows(tmp_path):
    save_path = tmp_path / "out.csv"
    total_rows = 5_001
    identifiers = [str(i) for i in range(total_rows)]
    successful = [True] * total_rows
    successful[-1] = False
    pd.DataFrame(
        {
            "Identifier": identifiers,
            "Response": [openai_utils._ser(["cached"])] * total_rows,
            "Web Search Sources": [openai_utils._ser([])] * total_rows,
            "Time Taken": [0.1] * total_rows,
            "Input Tokens": [1] * total_rows,
            "Reasoning Tokens": [0] * total_rows,
            "Output Tokens": [1] * total_rows,
            "Reasoning Effort": ["default"] * total_rows,
            "Successful": successful,
            "Error Log": [openai_utils._ser([])] * total_rows,
        }
    ).to_csv(save_path, index=False)

    calls = []

    async def responder(prompt: str, **kwargs):
        calls.append(prompt)
        return ["retried"]

    asyncio.run(
        openai_utils.get_all_responses(
            prompts=["prompt"] * total_rows,
            identifiers=identifiers,
            save_path=str(save_path),
            response_fn=responder,
            skip_tail_fails=False,
            verbose=False,
        )
    )

    assert calls == ["prompt"]
