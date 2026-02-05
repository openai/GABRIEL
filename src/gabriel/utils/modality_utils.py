from __future__ import annotations

import os
import re
from typing import Any, Iterable, List, Tuple

from .file_utils import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, PDF_EXTENSIONS

IMAGE_SUFFIXES = {ext.lower() for ext in IMAGE_EXTENSIONS}
AUDIO_SUFFIXES = {ext.lower() for ext in AUDIO_EXTENSIONS}
PDF_SUFFIXES = {ext.lower() for ext in PDF_EXTENSIONS}


def warn_if_modality_mismatch(
    values: Iterable[Any],
    modality: str,
    *,
    column_name: str,
    sample_size: int = 100,
) -> None:
    """Inspect a sample of values and warn if modality likely mismatches."""

    sample = list(values)[:sample_size]
    if not sample:
        return

    pdf_hits = 0
    image_hits = 0
    audio_hits = 0
    text_word_counts: List[int] = []

    for value in sample:
        candidate = _coerce_first_value(value)
        if candidate is None:
            continue
        kind = _detect_media_kind(candidate)
        if kind == "pdf":
            pdf_hits += 1
            continue
        if kind == "image":
            image_hits += 1
            continue
        if kind == "audio":
            audio_hits += 1
            continue
        if isinstance(candidate, str):
            words = re.findall(r"\b\w+\b", candidate)
            text_word_counts.append(len(words))

    total = len(sample)
    if total == 0:
        return

    if pdf_hits == total and modality != "pdf":
        print(
            f"[gabriel] Detected PDFs in column '{column_name}'. "
            "Set modality='pdf' to attach files directly, or set modality='text' "
            "(or 'entity'/'web') to extract plain text instead."
        )
    if image_hits == total and modality != "image":
        print(
            f"[gabriel] Detected image-like inputs in column '{column_name}'. "
            "Set modality='image' to attach image files correctly."
        )
    if audio_hits == total and modality != "audio":
        print(
            f"[gabriel] Detected audio-like inputs in column '{column_name}'. "
            "Set modality='audio' to attach audio files correctly."
        )
    if modality == "pdf" and pdf_hits == 0:
        print(
            f"[gabriel] Column '{column_name}' doesn't look like PDF inputs. "
            "If this is text, consider modality='text' (or 'entity'/'web')."
        )
    if modality == "image" and image_hits == 0:
        print(
            f"[gabriel] Column '{column_name}' doesn't look like image inputs. "
            "If this is text, consider modality='text' (or 'entity'/'web')."
        )
    if modality == "audio" and audio_hits == 0:
        print(
            f"[gabriel] Column '{column_name}' doesn't look like audio inputs. "
            "If this is text, consider modality='text' (or 'entity'/'web')."
        )

    if modality == "text" and text_word_counts:
        avg_words = sum(text_word_counts) / max(1, len(text_word_counts))
        if avg_words < 10:
            print(
                f"[gabriel] Average word count in column '{column_name}' is {avg_words:.1f} "
                "words. Confirm you intended modality='text' (rather than 'entity' or 'web')."
            )
    if modality in {"entity", "web"} and text_word_counts:
        avg_words = sum(text_word_counts) / max(1, len(text_word_counts))
        if avg_words > 30:
            print(
                f"[gabriel] Average word count in column '{column_name}' is {avg_words:.1f} "
                "words. Consider modality='text' for long-form passages."
            )


def _coerce_first_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _detect_media_kind(value: Any) -> str:
    if isinstance(value, dict):
        file_data = value.get("file_data")
        file_url = value.get("file_url")
        if isinstance(file_data, str) and file_data.startswith("data:application/pdf"):
            return "pdf"
        if isinstance(file_url, str) and file_url.lower().endswith(".pdf"):
            return "pdf"
        if value.get("format"):
            return "audio"
    if not isinstance(value, str):
        return ""
    lowered = value.lower().strip()
    if lowered.startswith("data:application/pdf"):
        return "pdf"
    if lowered.startswith("data:image/"):
        return "image"
    if lowered.startswith("data:audio/"):
        return "audio"
    if _looks_like_url(lowered):
        if lowered.endswith(tuple(PDF_SUFFIXES)):
            return "pdf"
        if lowered.endswith(tuple(IMAGE_SUFFIXES)):
            return "image"
        if lowered.endswith(tuple(AUDIO_SUFFIXES)):
            return "audio"
    if os.path.exists(value):
        ext = os.path.splitext(value)[1].lower()
        if ext in PDF_SUFFIXES:
            return "pdf"
        if ext in IMAGE_SUFFIXES:
            return "image"
        if ext in AUDIO_SUFFIXES:
            return "audio"
    return ""


def _looks_like_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")
