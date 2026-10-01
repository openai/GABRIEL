"""Complete split output must not be hidden by a failed primary CSV."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from gabriel.api import _load_cached_dataframe
from gabriel.utils import file_utils
from gabriel.utils.file_utils import save_dataframe_with_fallback


def test_split_fallback_removes_partial_primary_before_reload(tmp_path, monkeypatch):
    expected = pd.DataFrame({"row": range(5), "value": ["x"] * 5})
    target = tmp_path / "ratings_cleaned.csv"
    original_to_csv = pd.DataFrame.to_csv

    def partial_primary(self, path_or_buf=None, *args, **kwargs):
        if str(path_or_buf) == str(target):
            original_to_csv(self.iloc[:1], path_or_buf, *args, **kwargs)
            raise OSError("primary write failed after one row")
        return original_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", partial_primary)

    saved = save_dataframe_with_fallback(expected, str(target), chunk_size=2)

    assert saved == [str(tmp_path / f"ratings_cleaned_{i}.csv") for i in range(1, 4)]
    pd.testing.assert_frame_equal(_load_cached_dataframe(str(target), task_name="Rate"), expected)
    assert not target.exists()


def test_failed_split_attempt_preserves_primary(tmp_path, monkeypatch):
    target = tmp_path / "result.csv"
    original_to_csv = pd.DataFrame.to_csv

    def failed_writes(self, path_or_buf=None, *args, **kwargs):
        if str(path_or_buf) == str(target):
            original_to_csv(self.iloc[:1], path_or_buf, *args, **kwargs)
        raise OSError("write failed")

    monkeypatch.setattr(pd.DataFrame, "to_csv", failed_writes)

    assert save_dataframe_with_fallback(pd.DataFrame({"row": [1, 2]}), str(target)) == []
    assert target.read_text() == "row\n1\n"


def test_failed_primary_cleanup_does_not_report_success(tmp_path, monkeypatch, capsys):
    target = tmp_path / "result.csv"
    original_to_csv = pd.DataFrame.to_csv
    original_remove = os.remove
    warnings = []

    def partial_primary(self, path_or_buf=None, *args, **kwargs):
        if str(path_or_buf) == str(target):
            original_to_csv(self.iloc[:1], path_or_buf, *args, **kwargs)
            raise OSError("partial write")
        return original_to_csv(self, path_or_buf, *args, **kwargs)

    def blocked_cleanup(path):
        if str(path) == str(target):
            raise PermissionError("primary cannot be removed")
        return original_remove(path)

    monkeypatch.setattr(pd.DataFrame, "to_csv", partial_primary)
    monkeypatch.setattr(os, "remove", blocked_cleanup)
    monkeypatch.setattr(
        file_utils.logger, "warning", lambda message, *args, **kwargs: warnings.append(message)
    )

    assert save_dataframe_with_fallback(pd.DataFrame({"row": [1, 2]}), str(target)) == []
    assert target.exists()
    assert (tmp_path / "result_1.csv").exists()
    assert any("primary" in message.lower() for message in warnings)
    assert "could not remove the primary CSV" in capsys.readouterr().out


def test_split_fallback_removes_older_primary_when_open_fails(tmp_path, monkeypatch):
    target = tmp_path / "result.csv"
    pd.DataFrame({"row": [99]}).to_csv(target, index=False)
    expected = pd.DataFrame({"row": [1, 2]})
    original_to_csv = pd.DataFrame.to_csv

    def blocked_primary(self, path_or_buf=None, *args, **kwargs):
        if str(path_or_buf) == str(target):
            raise PermissionError("cannot open primary for writing")
        return original_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", blocked_primary)

    assert save_dataframe_with_fallback(expected, str(target)) == [str(tmp_path / "result_1.csv")]
    assert not target.exists()
    pd.testing.assert_frame_equal(_load_cached_dataframe(str(target), task_name="Rate"), expected)


def test_split_fallback_retries_then_removes_partial_primary(tmp_path, monkeypatch):
    expected = pd.DataFrame({"row": range(5)})
    target = tmp_path / "result.csv"
    original_to_csv = pd.DataFrame.to_csv

    def size_sensitive(self, path_or_buf=None, *args, **kwargs):
        if str(path_or_buf) == str(target) or len(self) > 2:
            original_to_csv(self.iloc[:1], path_or_buf, *args, **kwargs)
            raise OSError("file too large")
        return original_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", size_sensitive)

    saved = save_dataframe_with_fallback(
        expected, str(target), chunk_size=4, fallback_chunk_sizes=[2]
    )

    assert len(saved) == 3
    assert not target.exists()
    pd.testing.assert_frame_equal(_load_cached_dataframe(str(target), task_name="Rate"), expected)


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux RLIMIT_FSIZE")
@pytest.mark.parametrize("existing_primary", [False, True])
def test_split_fallback_round_trip_after_real_file_size_limit(tmp_path, existing_primary):
    # Limit only a child process, never the test runner. No pandas or filesystem mocks.
    script = """
import resource
import signal
import sys
from pathlib import Path
import pandas as pd
from gabriel.api import _load_cached_dataframe
from gabriel.utils.file_utils import save_dataframe_with_fallback

target = Path(sys.argv[1]) / 'result.csv'
expected = pd.DataFrame({'row': range(256), 'value': ['x' * 100] * 256})
if sys.argv[2] == 'True':
    pd.DataFrame({'row': [999], 'value': ['previous']}).to_csv(target, index=False)
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (8192, 8192))
saved = save_dataframe_with_fallback(expected, str(target), chunk_size=16)
assert len(saved) == 16
pd.testing.assert_frame_equal(_load_cached_dataframe(str(target), task_name='Rate'), expected)
assert not target.exists()
"""
    # Select this checkout even when pytest is launched from an editable sibling install.
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source_root, env.get("PYTHONPATH")]))
    # Do not write interpreter caches under the child's file-size limit.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(existing_primary)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
