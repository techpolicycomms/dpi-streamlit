from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NEW_OUT = ROOT / "outputs"
LEGACY_OUT = (ROOT / ".." / "DPI" / "outputs").resolve()

FILES = [
    "dpi_ready_scores_latest_year.csv",
    "dpi_ready_pillar_scores_all_years.csv",
    "dpi_ready_trends.csv",
    "dpi_v2_scores_all_years.csv",
    "dpi_v2_scores_latest_year.csv",
    "doughnut_pillar_scores.csv",
    "doughnut_scores_latest_year.csv",
    "doughnut_green_actions_scores.csv",
]


def compare_frames(name: str, left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float | int]:
    common_cols = [c for c in left.columns if c in right.columns]
    l = left[common_cols].copy().sort_values(common_cols).reset_index(drop=True)
    r = right[common_cols].copy().sort_values(common_cols).reset_index(drop=True)

    row_diff = abs(len(l) - len(r))
    numeric_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(l[c]) or pd.api.types.is_numeric_dtype(r[c])]
    max_abs = 0.0
    if numeric_cols and len(l) == len(r):
        for col in numeric_cols:
            lv = pd.to_numeric(l[col], errors="coerce").to_numpy(dtype=float)
            rv = pd.to_numeric(r[col], errors="coerce").to_numpy(dtype=float)
            diff = np.nanmax(np.abs(lv - rv)) if lv.size else 0.0
            if np.isfinite(diff):
                max_abs = max(max_abs, float(diff))
    return {
        "common_cols": len(common_cols),
        "left_rows": len(left),
        "right_rows": len(right),
        "row_diff": row_diff,
        "max_abs_numeric_diff": max_abs,
    }


def main() -> None:
    reports = []
    for file_name in FILES:
        new_path = NEW_OUT / file_name
        old_path = LEGACY_OUT / file_name
        if not new_path.exists() or not old_path.exists():
            reports.append({"file": file_name, "status": "missing", "details": "file missing on one side"})
            continue
        new_df = pd.read_csv(new_path)
        old_df = pd.read_csv(old_path)
        cmp = compare_frames(file_name, new_df, old_df)
        reports.append({"file": file_name, "status": "ok", **cmp})

    rep_df = pd.DataFrame(reports)
    out_path = NEW_OUT / "parity_check_core_report.csv"
    rep_df.to_csv(out_path, index=False)
    print(rep_df.to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
