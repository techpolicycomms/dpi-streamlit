from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


REQUIRED_FILES = {
    "dpi_v2_scores_all_years": "dpi_v2_scores_all_years.csv",
    "dpi_v2_scores_latest_year": "dpi_v2_scores_latest_year.csv",
    "doughnut_pillar_scores": "doughnut_pillar_scores.csv",
    "dpi_ready_trends": "dpi_ready_trends.csv",
    "indicator_correlation_report": "indicator_correlation_report.csv",
    "source_coverage_summary": "source_coverage_summary.csv",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_legacy_root() -> Path:
    env_path = os.getenv("DPI_LEGACY_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (repo_root() / ".." / "DPI").resolve()


def outputs_dir_from_legacy() -> Path:
    return resolve_legacy_root() / "outputs"


def outputs_dir_from_repo() -> Path:
    return repo_root() / "outputs"


def outputs_dir_from_snapshot() -> Path:
    return repo_root() / "data" / "legacy_snapshot" / "outputs"


def outputs_dir_from_env() -> Path | None:
    path = os.getenv("DPI_OUTPUTS_DIR")
    if not path:
        return None
    return Path(path).expanduser().resolve()


def list_missing_files(base_dir: Path) -> list[str]:
    missing = []
    for file_name in REQUIRED_FILES.values():
        if not (base_dir / file_name).exists():
            missing.append(file_name)
    return missing


def resolve_outputs_dir(source_mode: str = "auto") -> Path:
    if source_mode == "env":
        env_dir = outputs_dir_from_env()
        if env_dir is None:
            raise ValueError("DPI_OUTPUTS_DIR is not set")
        return env_dir
    if source_mode == "repo":
        return outputs_dir_from_repo()
    if source_mode == "snapshot":
        return outputs_dir_from_snapshot()
    if source_mode == "legacy":
        return outputs_dir_from_legacy()
    if source_mode != "auto":
        raise ValueError(f"Unknown source_mode: {source_mode}")

    candidates: list[Path] = []
    env_dir = outputs_dir_from_env()
    if env_dir is not None:
        candidates.append(env_dir)
    candidates.extend(
        [
            outputs_dir_from_repo(),
            outputs_dir_from_snapshot(),
            outputs_dir_from_legacy(),
        ]
    )
    for candidate in candidates:
        if candidate.exists() and len(list_missing_files(candidate)) == 0:
            return candidate
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return outputs_dir_from_repo()


def load_csv(name: str, base_dir: Path) -> pd.DataFrame:
    if name not in REQUIRED_FILES:
        raise ValueError(f"Unknown dataset key: {name}")
    path = base_dir / REQUIRED_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)
