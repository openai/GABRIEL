"""Utility helpers for working with audio files."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, Optional

from .logging import get_logger

logger = get_logger(__name__)

def encode_audio(audio_path: str) -> Optional[Dict[str, str]]:
    """Return the audio at ``audio_path`` as a dict suitable for the OpenAI API.

    The returned dictionary has two keys:

    ``data``
        Base64 encoded contents of the file.

    ``format``
        The lowercase file extension (e.g. ``"mp3"``, ``"wav"``).  If the
        extension cannot be determined ``"wav"`` is used as a fallback.

    Parameters
    ----------
    audio_path:
        Path to the audio file to encode.

    Returns
    -------
    dict or None
        A mapping with ``data`` and ``format`` keys, or ``None`` if reading the
        file fails.
    """

    try:
        path = Path(audio_path)
        with path.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = path.suffix.lstrip(".").lower() or "wav"
        return {"data": b64, "format": ext}
    except Exception as exc:
        logger.warning("Failed to encode audio file %s: %s", audio_path, exc)
        return None
