from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_build_backend_metadata_matches_pyproject_dependencies() -> None:
    project_requirements, project_extras = _pyproject_requirements()
    metadata_lines = _build_backend()._metadata().splitlines()

    backend_requirements = [
        line.removeprefix("Requires-Dist: ")
        for line in metadata_lines
        if line.startswith("Requires-Dist: ")
    ]
    backend_extras = [
        line.removeprefix("Provides-Extra: ")
        for line in metadata_lines
        if line.startswith("Provides-Extra: ")
    ]

    assert backend_requirements == project_requirements
    assert backend_extras == project_extras


def test_matplotlib_is_declared_for_dev_and_evaluation() -> None:
    lines = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    requirement = "matplotlib>=3.7,<4"

    assert requirement in _read_toml_array(lines, "dev")
    assert _read_toml_array(lines, "evaluation") == [requirement]

    metadata_lines = _build_backend()._metadata().splitlines()
    assert "Provides-Extra: evaluation" in metadata_lines
    assert f'Requires-Dist: {requirement}; extra == "evaluation"' in metadata_lines


def _pyproject_requirements() -> tuple[list[str], list[str]]:
    lines = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    requirements = _read_toml_array(lines, "dependencies")
    extras = _optional_dependency_names(lines)
    for extra in extras:
        requirements.extend(
            f'{requirement}; extra == "{extra}"' for requirement in _read_toml_array(lines, extra)
        )
    return requirements, extras


def _optional_dependency_names(lines: list[str]) -> list[str]:
    names: list[str] = []
    in_optional_dependencies = False
    for line in lines:
        stripped = line.strip()
        if stripped == "[project.optional-dependencies]":
            in_optional_dependencies = True
            continue
        if in_optional_dependencies and stripped.startswith("["):
            break
        if in_optional_dependencies and stripped.endswith("= ["):
            names.append(stripped.split("=", maxsplit=1)[0].strip())
    return names


def _read_toml_array(lines: list[str], key: str) -> list[str]:
    start = f"{key} = ["
    for index, line in enumerate(lines):
        if line.strip() != start:
            continue
        values: list[str] = []
        for item_line in lines[index + 1 :]:
            item = item_line.strip()
            if item == "]":
                return values
            if item:
                values.append(item.removesuffix(",").strip('"'))
        break
    raise AssertionError(f"Could not find pyproject array for {key!r}.")


def _build_backend() -> ModuleType:
    path = PROJECT_ROOT / "build_backend" / "fieldbridge_build_backend.py"
    spec = importlib.util.spec_from_file_location("fieldbridge_build_backend_for_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
