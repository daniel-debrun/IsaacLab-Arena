# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import yaml
from pathlib import Path
from typing import Any

_YAML_INCLUDE_KEY = "external_yaml"


def load_env_graph_spec_dict(path: str | Path) -> dict[str, Any]:
    """Load an env-graph YAML file, resolving its top-level ``external_yaml`` include(s).

    ``external_yaml`` is a file path or a list of file paths, relative to the including file. Only the
    entry YAML may declare ``external_yaml``; included files must not contain a further include (no nesting).
    The files are merged without overrides, so no value may be defined in more than one file.

    Args:
        path: Path to the root YAML file.

    Returns:
        A merged mapping ready for :class:`ArenaEnvGraphSpec` validation.
    """
    path = Path(path).resolve()
    raw = _load_yaml_dict(path)

    include = raw.pop(_YAML_INCLUDE_KEY, None)
    if include is None:
        return raw
    include_entries = [include] if isinstance(include, str) else include
    if not (
        isinstance(include_entries, list)
        and len(include_entries) > 0
        and all(isinstance(entry, str) for entry in include_entries)
    ):
        raise ValueError(
            f"'{_YAML_INCLUDE_KEY}' must be a file path string or a non-empty list of file path strings, "
            f"got {include!r} in {path}"
        )

    sources: list[tuple[str, dict[str, Any]]] = []
    for entry in include_entries:
        include_path = (path.parent / entry).resolve()
        included = _load_yaml_dict(include_path)
        if _YAML_INCLUDE_KEY in included:
            raise ValueError(
                f"Nested '{_YAML_INCLUDE_KEY}' is not allowed; only the top-level env graph spec YAML "
                f"may declare an include: {include_path}"
            )
        sources.append((str(include_path), included))
    sources.append((str(path), raw))
    return _merge_env_graph_dicts(sources)


def _load_yaml_dict(path: str | Path) -> dict[str, Any]:
    """Read a single YAML file that must parse to a mapping."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Env graph spec YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Env graph spec must be a dict, got {type(raw).__name__} in {path}")
    return raw


def _merge_env_graph_dicts(sources: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Merge env-graph dicts so that no value is defined by more than one source.

    Mappings merge recursively. Lists whose elements are all mappings with an ``id`` concatenate, and an
    ``id`` may appear only once. Every other value (scalar, ``null``, any other list) may be defined by
    only one source, even if the sources agree. A conflict fails with the path of the value and both sources.

    Args:
        sources: ``(label, dict)`` pairs, where the label (normally a file path) names the dict in error messages.

    Returns:
        The merged dict. The inputs are not modified.
    """
    merged: dict[str, Any] = {}
    origins: dict[str, str] = {}
    for label, source_dict in sources:
        _merge_mapping(merged, source_dict, "", label, origins)
    return merged


def _merge_mapping(
    target: dict[str, Any], incoming: dict[str, Any], path: str, label: str, origins: dict[str, str]
) -> None:
    """Merge ``incoming`` into ``target`` in place, recording the source of each path in ``origins``."""
    for key, value in incoming.items():
        key_path = f"{path}.{key}" if path else str(key)
        if key not in target:
            target[key] = copy.deepcopy(value)
            _record_origins(value, key_path, label, origins)
        elif isinstance(target[key], dict) and isinstance(value, dict):
            _merge_mapping(target[key], value, key_path, label, origins)
        elif _is_id_list(target[key]) and _is_id_list(value):
            _merge_id_lists(target[key], value, key_path, label, origins)
        else:
            raise ValueError(
                f"Conflicting env graph spec value at '{key_path}': defined in both {origins[key_path]} "
                f"(value {target[key]!r}) and {label} (value {value!r})"
            )


def _merge_id_lists(
    target: list[dict[str, Any]], incoming: list[dict[str, Any]], path: str, label: str, origins: dict[str, str]
) -> None:
    """Append the elements of ``incoming`` to ``target``, failing if an ``id`` repeats."""
    existing_ids = {element["id"] for element in target}
    for element in incoming:
        element_path = f"{path}[id={element['id']}]"
        if element["id"] in existing_ids:
            raise ValueError(
                f"Conflicting env graph spec entry at '{element_path}': an element with this id is defined in both "
                f"{origins[element_path]} and {label}"
            )
        existing_ids.add(element["id"])
        target.append(copy.deepcopy(element))
        origins[element_path] = label


def _is_id_list(value: Any) -> bool:
    """Whether ``value`` is a list whose elements are all mappings with an ``id`` (an empty list qualifies)."""
    return isinstance(value, list) and all(isinstance(e, dict) and "id" in e for e in value)


def _record_origins(value: Any, path: str, label: str, origins: dict[str, str]) -> None:
    """Record ``label`` as the source of ``path`` and of everything below it that can later conflict."""
    origins[path] = label
    if isinstance(value, dict):
        for key, child in value.items():
            _record_origins(child, f"{path}.{key}", label, origins)
    elif _is_id_list(value):
        for element in value:
            origins[f"{path}[id={element['id']}]"] = label
