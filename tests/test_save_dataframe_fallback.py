from pathlib import Path

import pandas as pd

from gabriel.utils.file_utils import save_dataframe_with_fallback


def test_save_dataframe_with_fallback_splits_large_files(tmp_path: Path, monkeypatch) -> None:
    df = pd.DataFrame({"value": range(200_005)})
    target = tmp_path / "final.csv"

    original_to_csv = pd.DataFrame.to_csv

    def flaky_to_csv(self, path_or_buf=None, *args, **kwargs):
        if path_or_buf is not None and str(path_or_buf) == str(target):
            raise OSError("simulated save failure")
        return original_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", flaky_to_csv)

    saved_paths = save_dataframe_with_fallback(df, str(target), index=False, label="Test")

    expected = [
        str(tmp_path / "final_1.csv"),
        str(tmp_path / "final_2.csv"),
        str(tmp_path / "final_3.csv"),
    ]
    assert saved_paths == expected
    for part in expected:
        assert Path(part).exists()


def test_save_dataframe_with_fallback_returns_empty_when_all_writes_fail(
    tmp_path: Path, monkeypatch
) -> None:
    df = pd.DataFrame({"value": [1, 2, 3]})

    def always_fail(self, path_or_buf=None, *args, **kwargs):
        raise OSError("all writes fail")

    monkeypatch.setattr(pd.DataFrame, "to_csv", always_fail)

    saved_paths = save_dataframe_with_fallback(
        df,
        str(tmp_path / "never.csv"),
        index=False,
        label="Test",
    )

    assert saved_paths == []


def test_save_dataframe_with_fallback_retries_smaller_chunk_size(
    tmp_path: Path, monkeypatch
) -> None:
    df = pd.DataFrame({"value": range(20_005)})
    target = tmp_path / "final.csv"

    original_to_csv = pd.DataFrame.to_csv

    def size_sensitive_to_csv(self, path_or_buf=None, *args, **kwargs):
        path_text = str(path_or_buf) if path_or_buf is not None else ""
        if path_or_buf is not None and path_text == str(target):
            raise OSError("simulated full save failure")
        if path_text.endswith("_1.csv") and len(self) > 10_000:
            raise OSError("simulated 100k chunk failure")
        return original_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", size_sensitive_to_csv)

    saved_paths = save_dataframe_with_fallback(df, str(target), index=False, label="Test")

    expected = [
        str(tmp_path / "final_1.csv"),
        str(tmp_path / "final_2.csv"),
        str(tmp_path / "final_3.csv"),
    ]
    assert saved_paths == expected
    for part in expected:
        assert Path(part).exists()


def test_save_dataframe_with_fallback_removes_stale_split_parts_on_primary_success(
    tmp_path: Path,
) -> None:
    df = pd.DataFrame({"value": [1, 2, 3]})
    target = tmp_path / "final.csv"
    stale_part = tmp_path / "final_1.csv"
    stale_part.write_text("old,data\n")

    saved_paths = save_dataframe_with_fallback(df, str(target), index=False, label="Test")

    assert saved_paths == [str(target)]
    assert target.exists()
    assert not stale_part.exists()


def test_relative_primary_save_removes_stale_parts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stale = Path("final_1.csv")
    stale.write_text("value\nold\n")
    unrelated = Path("other_1.csv")
    unrelated.write_text("value\nkeep\n")

    saved = save_dataframe_with_fallback(pd.DataFrame({"value": [1]}), "final.csv")

    assert saved == ["final.csv"]
    assert not stale.exists()
    assert unrelated.read_text() == "value\nkeep\n"


def test_relative_split_save_removes_stale_tail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("final_1.csv").write_text("value\nold\n")
    Path("final_2.csv").write_text("value\nold-tail\n")
    original_to_csv = pd.DataFrame.to_csv

    def fail_primary(self, path_or_buf=None, *args, **kwargs):
        if str(path_or_buf) == "final.csv":
            raise OSError("primary write failed")
        return original_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_primary)
    saved = save_dataframe_with_fallback(pd.DataFrame({"value": [1]}), "final.csv")

    assert saved == ["final_1.csv"]
    assert pd.read_csv("final_1.csv")["value"].tolist() == [1]
    assert not Path("final_2.csv").exists()
