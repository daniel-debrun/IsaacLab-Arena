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

    named_dicts: list[tuple[str, dict[str, Any]]] = []
    for entry in include_entries:
        include_path = (path.parent / entry).resolve()
        included = _load_yaml_dict(include_path)
        if _YAML_INCLUDE_KEY in included:
            raise ValueError(
                f"Nested '{_YAML_INCLUDE_KEY}' is not allowed; only the top-level env graph spec YAML "
                f"may declare an include: {include_path}"
            )
        named_dicts.append((str(include_path), included))
    named_dicts.append((str(path), raw))
    return _merge_env_graph_dicts(named_dicts)


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


def _merge_env_graph_dicts(named_dicts: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Merge env-graph dicts so that no value is defined by more than one dict.

    Mappings merge recursively. Lists whose elements are all mappings with an ``id`` concatenate, and an
    ``id`` may appear only once. Every other value (scalar, ``null``, any other list) may be defined by
    only one dict, even if the dicts agree. A conflict fails with the path of the value and the names of both dicts.

    Args:
        named_dicts: Dicts to merge. ``(source_name, dict)`` pairs, where ``source_name`` (normally a file path) names the dict
            in error messages.

    Returns:
        The merged dict. The inputs are not modified.
    """
    merged: dict[str, Any] = {}
    source_name_by_value_path: dict[str, str] = {}
    for source_name, source_dict in named_dicts:
        _merge_mapping(merged, source_dict, "", source_name, source_name_by_value_path)
    return merged


def _merge_mapping(
    merged_mapping: dict[str, Any],
    new_mapping: dict[str, Any],
    parent_path: str,
    source_name: str,
    source_name_by_value_path: dict[str, str],
) -> None:
    """Recursively merge ``new_mapping`` into ``merged_mapping`` in place, recording where each new value came from."""
    for key, new_value in new_mapping.items():
        value_path = f"{parent_path}.{key}" if parent_path else str(key)
        if key not in merged_mapping:
            merged_mapping[key] = copy.deepcopy(new_value)
            _record_source_name(new_value, value_path, source_name, source_name_by_value_path)
            continue
        merged_value = merged_mapping[key]
        if isinstance(merged_value, dict) and isinstance(new_value, dict):
            _merge_mapping(merged_value, new_value, value_path, source_name, source_name_by_value_path)
        elif _is_id_list(merged_value) and _is_id_list(new_value):
            _merge_id_lists(merged_value, new_value, value_path, source_name, source_name_by_value_path)
        else:
            raise ValueError(
                f"Conflicting env graph spec value at '{value_path}': defined in both "
                f"{source_name_by_value_path[value_path]} (value {merged_value!r}) and "
                f"{source_name} (value {new_value!r})"
            )


def _merge_id_lists(
    merged_list: list[dict[str, Any]],
    new_list: list[dict[str, Any]],
    list_path: str,
    source_name: str,
    source_name_by_value_path: dict[str, str],
) -> None:
    """Append the elements of ``new_list`` to ``merged_list``, failing if an ``id`` repeats."""
    merged_ids = {element["id"] for element in merged_list}
    for new_element in new_list:
        element_path = f"{list_path}[id={new_element['id']}]"
        if new_element["id"] in merged_ids:
            raise ValueError(
                f"Conflicting env graph spec entry at '{element_path}': an element with this id is defined in both "
                f"{source_name_by_value_path[element_path]} and {source_name}"
            )
        merged_ids.add(new_element["id"])
        merged_list.append(copy.deepcopy(new_element))
        source_name_by_value_path[element_path] = source_name


def _is_id_list(value: Any) -> bool:
    """Whether ``value`` is a list whose elements are all mappings with an ``id`` (an empty list qualifies)."""
    return isinstance(value, list) and all(isinstance(element, dict) and "id" in element for element in value)


def _record_source_name(
    value: Any, value_path: str, source_name: str, source_name_by_value_path: dict[str, str]
) -> None:
    """Record ``source_name`` as the source of ``value_path`` and of everything below it that can later conflict."""
    source_name_by_value_path[value_path] = source_name
    if isinstance(value, dict):
        for key, child_value in value.items():
            _record_source_name(child_value, f"{value_path}.{key}", source_name, source_name_by_value_path)
    elif _is_id_list(value):
        for element in value:
            source_name_by_value_path[f"{value_path}[id={element['id']}]"] = source_name
