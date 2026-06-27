from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    tables_small: Path
    tables_full: Path
    titles: Path
    nuts: Path
    intermediate: Path
    processed: Path
    outputs: Path
    logs: Path


def _resolve_project_path(value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH

    if not path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML configuration: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ConfigurationError(
            "The configuration file must contain a mapping."
        )

    if "paths" not in config:
        raise ConfigurationError(
            "The 'paths' section is missing from config.yaml."
        )

    return config


def get_project_paths(config: dict[str, Any]) -> ProjectPaths:
    paths_config = config["paths"]

    required_paths = {
        "tables_small",
        "tables_full",
        "titles",
        "nuts",
        "intermediate",
        "processed",
        "outputs",
        "logs",
    }

    missing = required_paths.difference(paths_config)

    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ConfigurationError(
            f"Missing paths in config.yaml: {missing_text}"
        )

    return ProjectPaths(
        project_root=PROJECT_ROOT,
        tables_small=_resolve_project_path(paths_config["tables_small"]),
        tables_full=_resolve_project_path(paths_config["tables_full"]),
        titles=_resolve_project_path(paths_config["titles"]),
        nuts=_resolve_project_path(paths_config["nuts"]),
        intermediate=_resolve_project_path(paths_config["intermediate"]),
        processed=_resolve_project_path(paths_config["processed"]),
        outputs=_resolve_project_path(paths_config["outputs"]),
        logs=_resolve_project_path(paths_config["logs"]),
    )


def ensure_output_directories(paths: ProjectPaths) -> None:
    directories = [
        paths.intermediate,
        paths.processed,
        paths.outputs,
        paths.logs,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)