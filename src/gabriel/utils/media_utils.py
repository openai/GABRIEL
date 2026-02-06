from __future__ import annotations

import os
from typing import Any, Dict, List

from .image_utils import encode_image
from .audio_utils import encode_audio
from .pdf_utils import encode_pdf
from .logging import get_logger

logger = get_logger(__name__)


def load_image_inputs(val: Any) -> List[str]:
    """Return a list of base64-encoded images from a DataFrame cell.

    ``val`` may be a single file path, a list of file paths, or a list of
    pre-encoded base64 strings. Non-existing paths are ignored.
    """
    if not val:
        return []
    imgs = val if isinstance(val, list) else [val]
    encoded: List[str] = []
    for img in imgs:
        if isinstance(img, str) and os.path.exists(img):
            enc = encode_image(img)
            if enc:
                encoded.append(enc)
        elif isinstance(img, str):
            encoded.append(img)
    return encoded


def load_audio_inputs(val: Any) -> List[Dict[str, str]]:
    """Return a list of audio dicts from a DataFrame cell.

    ``val`` may be a single file path, a list of file paths, or a list of
    already-encoded dicts. Non-existing paths are ignored.
    """
    if not val:
        return []
    auds = val if isinstance(val, list) else [val]
    encoded: List[Dict[str, str]] = []
    for aud in auds:
        if isinstance(aud, str) and os.path.exists(aud):
            enc = encode_audio(aud)
            if enc:
                encoded.append(enc)
            else:
                logger.warning(
                    "Audio encoding failed for %s; the file may be corrupted or unreadable.",
                    aud,
                )
        elif isinstance(aud, str) and not os.path.exists(aud):
            logger.warning("Audio path not found: %s", aud)
        elif isinstance(aud, dict):
            encoded.append(aud)
    return encoded


def load_pdf_inputs(val: Any) -> List[Dict[str, str]]:
    """Return a list of PDF dicts from a DataFrame cell."""
    if not val:
        return []
    pdfs = val if isinstance(val, list) else [val]
    encoded: List[Dict[str, str]] = []
    for pdf in pdfs:
        if isinstance(pdf, str) and os.path.exists(pdf):
            enc = encode_pdf(pdf)
            if enc:
                encoded.append(enc)
        elif isinstance(pdf, dict):
            encoded.append(pdf)
        elif isinstance(pdf, str):
            lowered = pdf.lower()
            if lowered.startswith("data:application/pdf"):
                encoded.append({"filename": "document.pdf", "file_data": pdf})
            elif lowered.startswith("http://") or lowered.startswith("https://"):
                if lowered.endswith(".pdf"):
                    encoded.append({"file_url": pdf})
            else:
                encoded.append(
                    {
                        "filename": "document.pdf",
                        "file_data": f"data:application/pdf;base64,{pdf}",
                    }
                )
    return encoded
