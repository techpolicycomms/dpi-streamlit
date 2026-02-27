from __future__ import annotations

import os
import shutil
from pathlib import Path


REQUIRED_FILES = {
    "dpi_v2_scores_all_years.csv",
    "dpi_v2_scores_latest_year.csv",
    "doughnut_pillar_scores.csv",
    "dpi_ready_trends.csv",
    "indicator_correlation_report.csv",
    "source_coverage_summary.csv",
}


def outputs_dir() -> Path:
    env_path = os.getenv("DPI_LEGACY_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve() / "outputs"
    return (Path(__file__).resolve().parents[2] / "DPI" / "outputs").resolve()


def main() -> None:
    source_dir = outputs_dir()
    target_dir = Path(__file__).resolve().parents[1] / "data" / "legacy_snapshot" / "outputs"
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []

    for file_name in REQUIRED_FILES:
        src = source_dir / file_name
        dst = target_dir / file_name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(file_name)
        else:
            missing.append(file_name)

    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Copied ({len(copied)}): {', '.join(copied) if copied else 'none'}")
    if missing:
        print(f"Missing ({len(missing)}): {', '.join(missing)}")


if __name__ == "__main__":
    main()
