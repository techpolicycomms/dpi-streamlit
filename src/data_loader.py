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


def resolve_legacy_root() -> Path:
    env_path = os.getenv("DPI_LEGACY_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / ".." / "DPI").resolve()


def outputs_dir() -> Path:
    return resolve_legacy_root() / "outputs"


def load_csv(name: str) -> pd.DataFrame:
    if name not in REQUIRED_FILES:
        raise ValueError(f"Unknown dataset key: {name}")
    path = outputs_dir() / REQUIRED_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)
