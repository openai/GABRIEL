from __future__ import annotations

import importlib
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd

from .logging import get_logger

logger = get_logger(__name__)

TEXTUAL_MODALITIES = {"text", "entity", "web"}
PATH_MODALITIES = {"image", "audio", "pdf"}
ALL_MODALITIES = TEXTUAL_MODALITIES | PATH_MODALITIES
TABULAR_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".pq", ".feather"}
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".svg",
}
PDF_EXTENSIONS = {".pdf"}
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rtf",
    ".html",
    ".htm",
    ".xml",
    ".json",
    ".csv",
    ".tsv",
}
AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".oga",
    ".opus",
    ".aiff",
    ".aif",
    ".aifc",
    ".wma",
    ".alac",
}
IMAGE_EXTENSION_SUFFIXES = {ext.lstrip(".") for ext in IMAGE_EXTENSIONS}
AUDIO_EXTENSION_SUFFIXES = {ext.lstrip(".") for ext in AUDIO_EXTENSIONS}
PDF_EXTENSION_SUFFIXES = {ext.lstrip(".") for ext in PDF_EXTENSIONS}


def _find_split_csv_parts(path: str) -> List[str]:
    """Return sorted split CSV parts matching ``<stem>_<n><suffix>``."""

    target = os.path.expandvars(os.path.expanduser(path))
    base_dir = os.path.dirname(target)
    stem = os.path.splitext(os.path.basename(target))[0]
    suffix = os.path.splitext(target)[1] or ".csv"
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$")
    matches: List[tuple[int, str]] = []
    if not base_dir or not os.path.isdir(base_dir):
        return []
    for name in os.listdir(base_dir):
        match = pattern.match(name)
        if not match:
            continue
        try:
            order = int(match.group(1))
        except Exception:
            continue
        matches.append((order, os.path.join(base_dir, name)))
    return [candidate for _, candidate in sorted(matches, key=lambda item: item[0])]


def _remove_paths(paths: Iterable[str]) -> None:
    """Best-effort path removal helper."""

    for candidate in paths:
        try:
            os.remove(candidate)
        except Exception:
            continue


def _resolve_chunk_sizes(
    primary_chunk_size: int,
    fallback_chunk_sizes: Optional[Iterable[int]],
) -> List[int]:
    """Build ordered, de-duplicated chunk-size attempts."""

    candidates: List[int] = []

    def _add(value: Any) -> None:
        try:
            parsed = int(value)
        except Exception:
            return
        if parsed <= 0:
            return
        if parsed not in candidates:
            candidates.append(parsed)

    _add(primary_chunk_size)
    if fallback_chunk_sizes is None:
        _add(10_000)
    else:
        for candidate in fallback_chunk_sizes:
            _add(candidate)
    if not candidates:
        candidates = [100_000, 10_000]
    return candidates


def save_dataframe_with_fallback(
    df: pd.DataFrame,
    path: str,
    *,
    index: bool = False,
    chunk_size: int = 100_000,
    fallback_chunk_sizes: Optional[Iterable[int]] = None,
    label: Optional[str] = None,
) -> List[str]:
    """Persist ``df`` to CSV, falling back to progressively smaller split files.

    The function first attempts to save to ``path``. If that fails (for example
    due to very large output files), it retries by splitting the DataFrame into
    numbered parts using dynamic row chunk sizes. By default it attempts 100k
    rows per file and then 10k rows per file.

    Returns
    -------
    list[str]
        Paths that were successfully written. An empty list indicates all save
        attempts failed.
    """

    resolved_path = os.path.expandvars(os.path.expanduser(path))
    prefix = f"[{label}] " if label else ""
    try:
        parent = os.path.dirname(resolved_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    except Exception:
        logger.debug("Could not ensure parent directory for %s", resolved_path)

    try:
        df.to_csv(resolved_path, index=index)
        stale_parts = _find_split_csv_parts(resolved_path)
        if stale_parts:
            _remove_paths(stale_parts)
        return [resolved_path]
    except Exception as primary_exc:
        print(
            f"{prefix}Unable to save final CSV to {resolved_path}. "
            "Trying chunked backup files."
        )
        logger.warning(
            "Failed to save CSV to %s; attempting chunked backup files.",
            resolved_path,
            exc_info=primary_exc,
        )

    stem, ext = os.path.splitext(resolved_path)
    ext = ext or ".csv"
    total_rows = len(df)
    chunk_attempts = _resolve_chunk_sizes(chunk_size, fallback_chunk_sizes)

    for chunk in chunk_attempts:
        attempt_paths: List[str] = []
        part_count = max(1, int(math.ceil(total_rows / max(1, chunk))))
        print(
            f"{prefix}Trying chunked backup with {chunk:,} rows per file "
            f"({part_count} part(s))."
        )
        _remove_paths(_find_split_csv_parts(resolved_path))
        try:
            for idx in range(part_count):
                start = idx * chunk
                stop = start + chunk
                part_path = f"{stem}_{idx + 1}{ext}"
                df.iloc[start:stop].to_csv(part_path, index=index)
                attempt_paths.append(part_path)
        except Exception as split_exc:
            _remove_paths(attempt_paths)
            logger.warning(
                "Chunked CSV fallback failed for %s at chunk size %s.",
                resolved_path,
                chunk,
                exc_info=split_exc,
            )
            continue

        print(
            f"{prefix}Saved fallback split files ({part_count} part(s)) with base path "
            f"{stem}_*.{ext.lstrip('.')}"
        )
        logger.warning(
            "Saved chunked CSV fallback for %s into %d part(s) using chunk size %d.",
            resolved_path,
            part_count,
            chunk,
        )
        return attempt_paths

    print(
        f"{prefix}Final DataFrame could not be saved to disk. "
        "Returning the DataFrame in memory only."
    )
    logger.warning(
        "All chunked CSV fallback attempts failed for %s.",
        resolved_path,
    )
    return []


def load(
    folder_path: str,
    extensions: Optional[Iterable[str]] = None,
    *,
    tag_dict: Optional[Dict[str, Any]] = None,
    save_name: str = "gabriel_aggregated_content.csv",
    save_dir: Optional[str] = None,
    reset_files: bool = False,
    modality: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate files from a folder into a single CSV.

    Parameters
    ----------
    folder_path:
        Path to a directory containing media files or to a single file. When a
        CSV/Excel file is provided, it is loaded directly without creating a
        copy.
    extensions:
        Optional iterable of file extensions (without leading dots) to include.
        When ``None`` all files are processed.
    tag_dict:
        Optional mapping of substrings to tag values. The first matching
        substring found in a file name determines the ``tag`` column value.
    save_name:
        Name of the output CSV written inside ``save_dir``. Defaults to
        ``"gabriel_aggregated_content.csv"``.
    save_dir:
        Optional directory for the aggregated CSV. When omitted, the data is
        saved inside ``folder_path`` (or the parent directory if
        ``folder_path`` points to a file).
    reset_files:
        When ``False`` (default), an existing file at ``save_path`` is reused
        instead of being regenerated. Set to ``True`` to overwrite the file.
    modality:
        Optional modality hint. ``"text"``, ``"entity"``, and ``"web"`` are
        treated as text; ``"image"``, ``"audio"``, and ``"pdf"`` collect file paths. When
        ``None`` (default) the modality is inferred from the first matching file.

    Returns
    -------
    DataFrame
        The aggregated contents or file paths of the processed files. Empty
        results retain the columns for the selected modality.
    """

    folder_path = os.path.expanduser(os.path.expandvars(folder_path))
    target_dir = _resolve_save_directory(folder_path, save_dir)
    save_path = os.path.join(target_dir, save_name)

    if os.path.exists(save_path) and not reset_files:
        logger.info("Loading existing aggregated file from %s", save_path)
        df = _read_tabular_file(save_path)
        print(df.head())
        print(f"Loaded existing aggregated file from {save_path}")
        return df

    extset = {e.lower().lstrip(".") for e in extensions} if extensions else None
    modality = _resolve_modality(folder_path, extset, save_name, modality)
    is_textual = _is_textual_modality(modality)

    path_key = "path"
    rows: List[Dict[str, Any]] = []
    max_layers = 0

    warned_pdf = False
    warned_image = False
    warned_audio = False
    warned_doc = False
    has_non_pdf = False
    has_pdf = False

    if os.path.isfile(folder_path):
        ext = os.path.splitext(folder_path)[1].lower()
        if ext == ".doc":
            if not warned_doc:
                print(
                    "[gabriel.load] Ignoring legacy .doc files. Please convert them "
                    "to .docx or PDF before loading."
                )
                warned_doc = True
        if ext == ".pdf":
            has_pdf = True
        if is_textual and ext in TABULAR_EXTENSIONS:
            logger.info(
                "Input path %s is a tabular file; loading it without creating a copy.",
                folder_path,
            )
            df = _read_tabular_file(folder_path)
            print(df.head())
            print(f"Loaded existing file from {folder_path}")
            return df
        name = os.path.basename(folder_path)
        if ext != ".doc":
            warned_pdf, warned_image, warned_audio = _warn_for_media_mismatch(
                ext,
                modality,
                warned_pdf,
                warned_image,
                warned_audio,
                folder_path,
            )
            rows.append(
                _build_row(
                    file_path=folder_path,
                    name=name,
                    layers=(),
                    tag_dict=tag_dict,
                    is_textual=is_textual,
                )
            )
    else:
        for root, _, files in os.walk(folder_path):
            for fname in files:
                if fname == save_name:
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext == ".doc":
                    if not warned_doc:
                        print(
                            "[gabriel.load] Ignoring legacy .doc files. Please convert "
                            "them to .docx or PDF before loading."
                        )
                        warned_doc = True
                    continue
                short_ext = ext.lstrip(".")
                if ext == ".pdf":
                    has_pdf = True
                if modality == "pdf" and ext != ".pdf":
                    has_non_pdf = True
                warned_pdf, warned_image, warned_audio = _warn_for_media_mismatch(
                    ext,
                    modality,
                    warned_pdf,
                    warned_image,
                    warned_audio,
                    folder_path,
                )
                if not _should_include_file(short_ext, modality, extset):
                    continue
                file_path = os.path.join(root, fname)
                rel = os.path.relpath(file_path, folder_path)
                parts = rel.split(os.sep)
                name = parts[-1]
                layers = parts[:-1]
                max_layers = max(max_layers, len(layers))
                rows.append(
                    _build_row(
                        file_path=file_path,
                        name=name,
                        layers=layers,
                        tag_dict=tag_dict,
                        is_textual=is_textual,
                    )
                )

    if modality == "pdf" and has_non_pdf:
        print(
            "[gabriel.load] Detected non-PDF files in a PDF run. Only PDFs were "
            "ingested. Set modality='text' (or 'entity'/'web') in gabriel.load if you "
            "need to extract text from PDFs and include non-PDF files."
        )
    if modality == "pdf" and has_pdf:
        print(
            "[gabriel.load] PDF modality attaches PDFs directly (richer layout, figures, and "
            "images). Set modality='text' (or 'entity'/'web') to extract text-only "
            "versions of PDFs."
        )

    df = pd.DataFrame(rows)
    for i in range(1, max_layers + 1):
        col = f"layer_{i}"
        if col not in df.columns:
            df[col] = None

    cols = ["name", path_key] + [f"layer_{i}" for i in range(1, max_layers + 1)]
    if tag_dict:
        cols.append("tag")
    else:
        df.drop(columns=["tag"], inplace=True, errors="ignore")
    if is_textual:
        cols.append("text")
    else:
        df.drop(columns=["text"], inplace=True, errors="ignore")
    if df.empty:
        df = pd.DataFrame(columns=cols)
    else:
        df = df[cols]
    df.to_csv(save_path, index=False)
    print(df.head())
    print(f"Saved aggregated file to {save_path}")
    return df


def _build_row(
    *,
    file_path: str,
    name: str,
    layers: Iterable[str],
    tag_dict: Optional[Dict[str, Any]],
    is_textual: bool,
) -> Dict[str, Any]:
    tag = _match_tag(name, tag_dict)
    row: Dict[str, Any] = {
        "name": name,
        "path": file_path,
        "tag": tag,
    }
    if is_textual:
        row["text"] = _extract_text(file_path)
    for i, layer in enumerate(layers, start=1):
        row[f"layer_{i}"] = layer
    return row


def _extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in TEXT_EXTENSIONS or not ext:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    if ext == ".pdf":
        pypdf = _optional_import("pypdf", "pypdf")
        reader = pypdf.PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if ext == ".docx":
        docx = _optional_import("docx", "python-docx")
        document = docx.Document(file_path)
        return "\n".join(p.text for p in document.paragraphs).strip()
    if ext == ".doc":
        return ""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def _optional_import(module_name: str, package_name: str):
    if importlib.util.find_spec(module_name) is None:
        raise ImportError(
            f"Missing optional dependency '{package_name}'. Install it to "
            f"extract {module_name} documents."
        )
    return importlib.import_module(module_name)


def _match_tag(name: str, tag_dict: Optional[Dict[str, Any]]) -> Optional[Any]:
    if not tag_dict:
        return None
    lower_name = name.lower()
    for key, val in tag_dict.items():
        if key.lower() in lower_name:
            return val
    return None


def _resolve_modality(
    folder_path: str,
    extset: Optional[Set[str]],
    save_name: str,
    requested_modality: Optional[str],
) -> str:
    if requested_modality:
        normalized = requested_modality.lower()
        if normalized not in ALL_MODALITIES:
            logger.info(
                "Unknown modality '%s'; defaulting to text-style processing.",
                normalized,
            )
        return normalized
    detected = _detect_modality(folder_path, extset, save_name)
    logger.info("Detected %s modality for %s", detected, folder_path)
    return detected


def _detect_modality(
    folder_path: str,
    extset: Optional[Set[str]],
    save_name: str,
) -> str:
    detected: Set[str] = set()
    media_detected: Set[str] = set()
    saw_text = False
    for file_path in _iter_candidate_files(folder_path, extset, save_name):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in PDF_EXTENSIONS:
            detected.add("pdf")
            media_detected.add("pdf")
        elif ext in IMAGE_EXTENSIONS:
            detected.add("image")
            media_detected.add("image")
        elif ext in AUDIO_EXTENSIONS:
            detected.add("audio")
            media_detected.add("audio")
        else:
            detected.add("text")
            saw_text = True
    if not detected:
        return "text"
    if media_detected:
        if len(media_detected) == 1:
            return next(iter(media_detected))
        return "text"
    if saw_text or "text" in detected:
        return "text"
    if len(detected) == 1:
        return detected.pop()
    return "text"


def _iter_candidate_files(
    folder_path: str,
    extset: Optional[Set[str]],
    save_name: str,
) -> Iterable[str]:
    if os.path.isfile(folder_path):
        yield folder_path
        return
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname == save_name:
                continue
            short_ext = os.path.splitext(fname)[1].lower().lstrip(".")
            if extset and short_ext not in extset:
                continue
            yield os.path.join(root, fname)


def _is_textual_modality(modality: str) -> bool:
    if modality in TEXTUAL_MODALITIES:
        return True
    if modality in PATH_MODALITIES:
        return False
    return True


def _should_include_file(
    short_ext: str,
    modality: str,
    extset: Optional[Set[str]],
) -> bool:
    if extset and short_ext not in extset:
        return False
    if modality in TEXTUAL_MODALITIES:
        if short_ext in IMAGE_EXTENSION_SUFFIXES or short_ext in AUDIO_EXTENSION_SUFFIXES:
            return False
        return True
    if modality == "image":
        return short_ext in IMAGE_EXTENSION_SUFFIXES
    if modality == "audio":
        return short_ext in AUDIO_EXTENSION_SUFFIXES
    if modality == "pdf":
        return short_ext in PDF_EXTENSION_SUFFIXES
    return True


def _warn_for_media_mismatch(
    ext: str,
    modality: str,
    warned_pdf: bool,
    warned_image: bool,
    warned_audio: bool,
    folder_path: str,
) -> tuple[bool, bool, bool]:
    if ext in PDF_EXTENSIONS and modality != "pdf" and not warned_pdf:
        print(
            f"[gabriel.load] Found PDF files in {folder_path} while modality='{modality}'. "
            "PDFs will be extracted into plain text. For best PDF fidelity (layout, "
            "figures, and images), set modality='pdf' here and in the downstream "
            "gabriel call."
        )
        warned_pdf = True
    if ext in IMAGE_EXTENSIONS and modality != "image" and not warned_image:
        print(
            f"[gabriel.load] Found image files in {folder_path}. "
            "Set modality='image' to attach images directly to GPT calls."
        )
        warned_image = True
    if ext in AUDIO_EXTENSIONS and modality != "audio" and not warned_audio:
        print(
            f"[gabriel.load] Found audio files in {folder_path}. "
            "Set modality='audio' to attach audio directly to GPT calls."
        )
        warned_audio = True
    return warned_pdf, warned_image, warned_audio


def _resolve_save_directory(folder_path: str, save_dir: Optional[str]) -> str:
    if save_dir:
        resolved = os.path.expanduser(os.path.expandvars(save_dir))
    else:
        if os.path.isdir(folder_path):
            resolved = folder_path
        else:
            parent = os.path.dirname(folder_path)
            if not parent:
                parent = os.path.dirname(os.path.abspath(folder_path))
            resolved = parent
    if not resolved:
        resolved = os.getcwd()
    if os.path.isfile(resolved):
        raise ValueError(f"save_dir must be a directory path, got file {resolved}")
    os.makedirs(resolved, exist_ok=True)
    return resolved


def _read_tabular_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if ext == ".feather":
        return pd.read_feather(path)
    return pd.read_csv(path)
