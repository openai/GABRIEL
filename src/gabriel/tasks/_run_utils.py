from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar

import pandas as pd

from ..utils.parsing import safe_json

T = TypeVar("T")

DEFAULT_IDENTIFIER_HASH_BITS = 64
LEGACY_IDENTIFIER_HASH_BITS = 32
ATTRIBUTE_WARNING_THRESHOLD = 12

_IDENTIFIER_BITS_RE = {
    64: re.compile(r"^[0-9a-f]{16}(?:$|[_-])"),
    32: re.compile(r"^[0-9a-f]{8}(?:$|[_-])"),
}
_BATCH_RE = re.compile(r"_batch(\d+)(?:_|$)")


def run_metadata_path(save_dir: str, base_name: str) -> str:
    return os.path.join(save_dir, f"{base_name}_run_metadata.json")


def load_run_metadata(save_dir: str, base_name: str, *, reset_files: bool) -> Dict[str, Any]:
    path = run_metadata_path(save_dir, base_name)
    if reset_files:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def update_run_metadata(
    save_dir: str,
    base_name: str,
    *,
    strict: bool = False,
    **updates: Any,
) -> None:
    """Merge and atomically persist task metadata.

    Most tasks historically treated metadata as best-effort bookkeeping, so
    the default retains that behavior. Tasks whose checkpoint correctness
    depends on a commit marker can set ``strict=True`` to surface read or write
    failures rather than silently advancing with ambiguous state.
    """

    path = run_metadata_path(save_dir, base_name)
    payload: Dict[str, Any] = {}
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                payload.update(loaded)
            elif strict:
                raise ValueError(f"Run metadata {path!r} is not a JSON object")
    except Exception:
        if strict:
            raise
        payload = {}
    payload.setdefault("metadata_version", 1)
    payload.update(updates)
    temporary_path: Optional[str] = None
    try:
        os.makedirs(save_dir, exist_ok=True)
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{base_name}_run_metadata.",
            suffix=".tmp",
            dir=save_dir,
            text=True,
        )
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    except Exception:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
        if strict:
            raise


def hash_identifier(value: Any, *, bits: int = DEFAULT_IDENTIFIER_HASH_BITS) -> str:
    digest_len = max(1, bits // 4)
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:digest_len]


def _identifier_bits(identifier: Any) -> Optional[int]:
    text = str(identifier)
    if _IDENTIFIER_BITS_RE[64].match(text):
        return 64
    if _IDENTIFIER_BITS_RE[32].match(text):
        return 32
    return None


def infer_identifier_hash_bits_from_paths(paths: Sequence[str]) -> Optional[int]:
    found: set[int] = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, usecols=["Identifier"], nrows=5000)
        except Exception:
            continue
        for ident in df["Identifier"].dropna().astype(str).head(5000):
            bits = _identifier_bits(ident)
            if bits:
                found.add(bits)
        if 64 in found:
            return 64
    if 32 in found:
        return 32
    return None


def resolve_identifier_hash_bits(
    *,
    task_name: str,
    metadata: Dict[str, Any],
    reset_files: bool,
    checkpoint_paths: Sequence[str],
) -> int:
    if not reset_files:
        saved = metadata.get("identifier_hash_bits")
        if saved in {32, 64}:
            return int(saved)

    inferred = (
        None if reset_files else infer_identifier_hash_bits_from_paths(checkpoint_paths)
    )
    if inferred == LEGACY_IDENTIFIER_HASH_BITS:
        print(
            f"[{task_name}] Existing 8-char identifiers found; using legacy "
            "32-bit IDs so this save_dir resumes. On large datasets, these "
            "IDs collide more often; use reset_files=True or a new save_dir "
            "to switch to 16-char IDs."
        )
        return LEGACY_IDENTIFIER_HASH_BITS
    return DEFAULT_IDENTIFIER_HASH_BITS


def _coerce_json_obj(raw: Any) -> Any:
    obj = safe_json(raw)
    if isinstance(obj, list) and len(obj) == 1:
        obj = safe_json(obj[0])
    if isinstance(obj, str):
        obj = safe_json(obj)
    return obj


def _collect_attribute_keys(obj: Any, lookup: Dict[str, str]) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            norm = str(key).strip().lower()
            if norm in lookup:
                found.add(lookup[norm])
            if isinstance(value, (dict, list)):
                found.update(_collect_attribute_keys(value, lookup))
    elif isinstance(obj, list):
        for item in obj:
            found.update(_collect_attribute_keys(item, lookup))
    return found


def _batch_index(identifier: Any, row: Optional[pd.Series] = None) -> Optional[int]:
    if row is not None and "Batch" in row:
        try:
            value = row.get("Batch")
            if not pd.isna(value):
                return int(value)
        except Exception:
            pass
    match = _BATCH_RE.search(str(identifier))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def infer_attribute_batches_from_paths(
    paths: Sequence[str],
    attribute_names: Sequence[str],
) -> Optional[List[List[str]]]:
    if not attribute_names:
        return []
    lookup = {str(attr).strip().lower(): str(attr) for attr in attribute_names}
    batch_attrs: Dict[int, set[str]] = {}
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(
                path,
                usecols=lambda col: col in {"Identifier", "Response", "Batch"},
                nrows=5000,
            )
        except Exception:
            continue
        if "Identifier" not in df.columns or "Response" not in df.columns:
            continue
        for _, row in df.head(5000).iterrows():
            batch_idx = _batch_index(row.get("Identifier"), row)
            if batch_idx is None:
                continue
            keys = _collect_attribute_keys(_coerce_json_obj(row.get("Response")), lookup)
            if keys:
                batch_attrs.setdefault(batch_idx, set()).update(keys)
    if not batch_attrs:
        return None
    max_idx = max(batch_attrs)
    if set(batch_attrs) != set(range(max_idx + 1)):
        return None
    attr_set = set(map(str, attribute_names))
    if set().union(*batch_attrs.values()) != attr_set:
        return None
    ordered: List[List[str]] = []
    for batch_idx in range(max_idx + 1):
        attrs = [str(attr) for attr in attribute_names if str(attr) in batch_attrs[batch_idx]]
        if not attrs:
            return None
        ordered.append(attrs)
    return ordered


def _chunk_items(items: Sequence[T], n: Optional[int]) -> List[List[T]]:
    if not items:
        return []
    if n is None:
        return [list(items)]
    return [list(items[i : i + n]) for i in range(0, len(items), n)]


def resolve_attribute_batches(
    *,
    task_name: str,
    items: Sequence[Tuple[str, T]],
    requested_n: Optional[int],
    metadata: Dict[str, Any],
    reset_files: bool,
    checkpoint_paths: Sequence[str],
) -> Tuple[List[List[Tuple[str, T]]], Optional[int]]:
    item_lookup = {name: value for name, value in items}
    attr_names = [name for name, _ in items]

    if requested_n is not None and (
        not isinstance(requested_n, int) or requested_n < 1
    ):
        raise ValueError("n_attributes_per_run must be None or an integer >= 1")

    if attr_names and not reset_files:
        saved_batches = metadata.get("attribute_batches")
        if isinstance(saved_batches, list):
            batches: List[List[Tuple[str, T]]] = []
            seen: List[str] = []
            valid = True
            for batch in saved_batches:
                if not isinstance(batch, list):
                    valid = False
                    break
                names = [str(name) for name in batch]
                if any(name not in item_lookup for name in names):
                    valid = False
                    break
                seen.extend(names)
                batches.append([(name, item_lookup[name]) for name in names])
            if valid and seen and set(seen) == set(attr_names):
                return batches, metadata.get("n_attributes_per_run")

        saved_n = metadata.get("n_attributes_per_run")
        if saved_n is None and "n_attributes_per_run" in metadata:
            chunks = _chunk_items(items, None)
            return chunks, None
        if isinstance(saved_n, int) and saved_n >= 1:
            chunks = _chunk_items(items, saved_n)
            return chunks, saved_n

    if requested_n is None and attr_names and not reset_files:
        inferred = infer_attribute_batches_from_paths(checkpoint_paths, attr_names)
        if inferred and len(inferred) > 1:
            print(
                f"[{task_name}] Existing checkpoint uses multiple attribute "
                "batches; reusing them so cached responses line up. Use "
                "reset_files=True or a new save_dir for the new all-in-one "
                "default."
            )
            inferred_batches = [
                [(name, item_lookup[name]) for name in batch] for batch in inferred
            ]
            return inferred_batches, None

    if requested_n is None and len(items) > ATTRIBUTE_WARNING_THRESHOLD:
        print(
            f"[{task_name}] Processing {len(items)} attributes in one prompt. "
            "For more consistent outputs, "
            "set n_attributes_per_run=12 (or lower) to split them across prompts."
        )

    chunks = _chunk_items(items, requested_n)
    return chunks, requested_n


def metadata_attribute_batches(
    batches: Sequence[Sequence[Tuple[str, Any]]],
) -> List[List[str]]:
    return [[name for name, _ in batch] for batch in batches]


def write_task_run_metadata(
    *,
    save_dir: str,
    base_name: str,
    task_name: str,
    model: Optional[str],
    identifier_hash_bits: int,
    n_attributes_per_run: Optional[int],
    attribute_batches: Sequence[Sequence[Tuple[str, Any]]],
    strict: bool = False,
    **task_metadata: Any,
) -> None:
    attr_batches = metadata_attribute_batches(attribute_batches)
    update_run_metadata(
        save_dir,
        base_name,
        strict=strict,
        task=task_name,
        output_base_name=base_name,
        model=model,
        identifier_hash_bits=identifier_hash_bits,
        n_attributes_per_run=n_attributes_per_run,
        attribute_count=sum(len(batch) for batch in attr_batches),
        attribute_batches=attr_batches,
        **task_metadata,
    )
