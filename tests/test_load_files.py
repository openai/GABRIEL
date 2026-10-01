import os
import pandas as pd
import pytest

from gabriel import load


def test_load_basic(tmp_path):
    base = tmp_path / "data"
    sub = base / "round"
    base.mkdir()
    sub.mkdir()

    (base / "baseline.txt").write_text("hello")
    (sub / "round_1.txt").write_text("world")
    (sub / "round_2.txt").write_text("!")

    tag_dict = {"baseline": 1, "round_1": 2, "round_2": 2}
    df = load(
        str(base),
        extensions=["txt"],
        tag_dict=tag_dict,
        save_name="agg.csv",
        reset_files=True,
    )

    assert set(df["name"]) == {"baseline.txt", "round_1.txt", "round_2.txt"}
    assert set(df["tag"]) == {1, 2}
    assert set(df["text"]) == {"hello", "world", "!"}
    assert "layer_1" in df.columns
    assert os.path.exists(base / "agg.csv")


def test_load_existing_file_reuse(tmp_path):
    base = tmp_path / "data"
    base.mkdir()
    (base / "a.txt").write_text("a")
    load(str(base), save_name="out.csv", reset_files=True)
    df = load(str(base), save_name="out.csv")
    assert df.loc[0, "text"] == "a"


def test_load_csv_direct(tmp_path):
    df_in = pd.DataFrame({"a": [1, 2]})
    csv_path = tmp_path / "input.csv"
    df_in.to_csv(csv_path, index=False)
    df = load(str(csv_path), save_name="copy.csv", reset_files=True)
    assert (df == df_in).all().all()
    assert not os.path.exists(tmp_path / "copy.csv")


@pytest.mark.parametrize("suffix", ["csv", "tsv", "xlsx", "parquet", "pq", "feather"])
def test_load_direct_table_ignores_aggregate_cache(tmp_path, suffix):
    expected = pd.DataFrame({"record_id": [10, 20], "label": ["wanted", "also wanted"]})
    source = tmp_path / f"requested.{suffix}"
    if suffix == "csv":
        expected.to_csv(source, index=False)
    elif suffix == "tsv":
        expected.to_csv(source, sep="\t", index=False)
    elif suffix == "xlsx":
        pytest.importorskip("openpyxl")
        expected.to_excel(source, index=False)
    elif suffix in {"parquet", "pq"}:
        pytest.importorskip("pyarrow")
        expected.to_parquet(source, index=False)
    else:
        pytest.importorskip("pyarrow")
        expected.to_feather(source)
    cache = tmp_path / "gabriel_aggregated_content.csv"
    cache.write_text("unrelated\nSTALE\n")
    original_source = source.read_bytes()
    original_cache = cache.read_bytes()

    pd.testing.assert_frame_equal(load(str(source)), expected)
    assert source.read_bytes() == original_source
    assert cache.read_bytes() == original_cache


@pytest.mark.parametrize("modality", [None, "text", "entity", "web", "TEXT"])
def test_load_direct_table_ignores_custom_save_cache(tmp_path, modality):
    source = tmp_path / "requested.CSV"
    source.write_text("record_id\n42\n")
    output = tmp_path / "output"
    output.mkdir()
    cache = output / "custom.csv"
    cache.write_text("wrong\n99\n")

    result = load(str(source), save_dir=str(output), save_name="custom.csv", modality=modality)

    assert result.to_dict("list") == {"record_id": [42]}
    assert cache.read_text() == "wrong\n99\n"


def test_load_direct_table_without_cache_does_not_create_copy(tmp_path):
    source = tmp_path / "requested.csv"
    source.write_text("record_id\n42\n")

    assert load(str(source)).to_dict("list") == {"record_id": [42]}
    assert not (tmp_path / "gabriel_aggregated_content.csv").exists()


@pytest.mark.parametrize("modality", ["image", "audio", "pdf"])
def test_load_direct_table_preserves_path_modality_cache(tmp_path, modality):
    source = tmp_path / "requested.csv"
    source.write_text("record_id\n42\n")
    # A prior tagged load leaves a distinguishable cache. Rebuilding without
    # tag_dict would omit the tag column, so equality proves cache reuse.
    expected = load(
        str(source), modality=modality, tag_dict={"requested": 7}, reset_files=True
    )

    pd.testing.assert_frame_equal(load(str(source), modality=modality), expected)


def test_load_modality_paths(tmp_path):
    base = tmp_path / "media"
    base.mkdir()
    (base / "a.png").write_bytes(b"img")
    (base / "b.wav").write_bytes(b"aud")

    df_img = load(
        str(base),
        extensions=["png"],
        save_name="imgs.csv",
        reset_files=True,
        modality="image",
    )
    assert "path" in df_img.columns and "text" not in df_img.columns
    assert df_img["path"].tolist() == [str(base / "a.png")]

    df_aud = load(
        str(base),
        extensions=["wav"],
        save_name="aud.csv",
        reset_files=True,
        modality="audio",
    )
    assert "path" in df_aud.columns and "text" not in df_aud.columns
    assert df_aud["path"].tolist() == [str(base / "b.wav")]


def test_load_entity_behaves_like_text(tmp_path):
    path = tmp_path / "entity.txt"
    path.write_text("entity value")
    df = load(str(path), modality="entity", save_name="entity.csv", reset_files=True)
    assert df.loc[0, "text"] == "entity value"
    assert df.loc[0, "path"] == str(path)


def test_load_auto_detects_image(tmp_path):
    base = tmp_path / "imgs"
    base.mkdir()
    (base / "a.png").write_bytes(b"img")
    (base / "notes.txt").write_text("skip")
    df = load(str(base), save_name="imgs.csv", reset_files=True)
    assert "text" not in df.columns
    assert df["name"].tolist() == ["a.png"]


def test_load_supports_save_dir(tmp_path):
    base = tmp_path / "inputs"
    base.mkdir()
    (base / "a.txt").write_text("alpha")
    save_dir = tmp_path / "outputs"
    df = load(
        str(base),
        save_dir=str(save_dir),
        save_name="custom.csv",
        reset_files=True,
    )
    assert os.path.exists(save_dir / "custom.csv")
    assert df.loc[0, "text"] == "alpha"


def test_load_image_modality_filters_non_images(tmp_path):
    base = tmp_path / "mixed"
    base.mkdir()
    (base / "img.png").write_bytes(b"img")
    (base / "doc.txt").write_text("note")
    df = load(
        str(base),
        modality="image",
        save_name="imgs.csv",
        reset_files=True,
    )
    assert df["name"].tolist() == ["img.png"]


def test_load_audio_modality_filters_non_audio(tmp_path):
    base = tmp_path / "mixed_audio"
    base.mkdir()
    (base / "clip.wav").write_bytes(b"aud")
    (base / "img.png").write_bytes(b"img")
    df = load(
        str(base),
        modality="audio",
        save_name="aud.csv",
        reset_files=True,
    )
    assert df["name"].tolist() == ["clip.wav"]


def test_load_text_modality_skips_media(tmp_path):
    base = tmp_path / "text_folder"
    base.mkdir()
    (base / "doc.txt").write_text("text")
    (base / "photo.jpg").write_bytes(b"img")
    df = load(
        str(base),
        modality="text",
        save_name="texts.csv",
        reset_files=True,
    )
    assert df["name"].tolist() == ["doc.txt"]
