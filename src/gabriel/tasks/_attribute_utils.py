import json
import os
from typing import Any, Dict, Optional


def load_persisted_attributes(
    *,
    save_dir: str,
    incoming: Dict[str, Any],
    reset_files: bool,
    task_name: str,
    item_name: str = "attributes",
    legacy_filename: Optional[str] = None,
    persist_missing: bool = True,
) -> Dict[str, Any]:
    """Load attributes/labels from disk for reproducibility.

    Preference order:
    1) ``attributes.json`` in ``save_dir``
    2) ``legacy_filename`` (e.g., ``ratings_attrs.json``) in ``save_dir``
    When neither exists and ``persist_missing`` is true, ``incoming`` is
    written to both paths (when applicable) for future runs.
    """

    primary_path = os.path.join(save_dir, "attributes.json")
    legacy_path = os.path.join(save_dir, legacy_filename) if legacy_filename else None
    candidate_paths = [primary_path] + ([legacy_path] if legacy_path else [])

    if reset_files:
        for path in candidate_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    loaded: Optional[Dict[str, Any]] = None
    source_path: Optional[str] = None
    for path in candidate_paths:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                loaded = json.load(f)
            source_path = path
            break
        except Exception:
            continue

    if loaded is not None:
        message = (
            f"[{task_name}] Found saved {item_name} in {source_path}. Using them for consistency."
        )
        if loaded != incoming:
            message += (
                f" The provided {item_name} differ; set reset_files=True or use a new save_dir to update them."
            )
        print(message)
        return loaded

    if persist_missing:
        for path in candidate_paths:
            if not path:
                continue
            try:
                with open(path, "w") as f:
                    json.dump(incoming, f, indent=2)
            except Exception:
                pass

    return incoming
