from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


def geometric_mean_with_missing(row: np.ndarray, eps: float = 1e-6) -> float:
    keep = np.isfinite(row) & (row > 0)
    if not keep.any():
        return np.nan
    vals = np.maximum(eps, row[keep])
    w = np.ones(vals.shape[0], dtype=float) / vals.shape[0]
    return float(np.exp(np.sum(w * np.log(vals))))


def main() -> None:
    pillar_path = OUTPUT_DIR / "dpi_ready_pillar_scores_all_years.csv"
    latest_path = OUTPUT_DIR / "dpi_ready_scores_latest_year.csv"
    output_path = OUTPUT_DIR / "dpi_ready_trends.csv"
    if not pillar_path.exists() or not latest_path.exists():
        raise FileNotFoundError("Run build_dpi_ready_index.py first")

    pillars = pd.read_csv(pillar_path, na_values=["", "NA"])
    _ = pd.read_csv(latest_path, na_values=["", "NA"])

    pillar_cols = [c for c in pillars.columns if c.startswith("pillar_")]
    score_col = "dpi_ready_score"
    if score_col not in pillars.columns:
        pillars[score_col] = pillars[pillar_cols].apply(
            lambda r: geometric_mean_with_missing(r.to_numpy(dtype=float)), axis=1
        )

    pillars = pillars.sort_values(["economy", "year"]).copy()
    pieces = []
    for economy, sub in pillars.groupby("economy", sort=False):
        sub = sub.sort_values("year").copy()
        if len(sub) < 2:
            continue
        s = pd.to_numeric(sub[score_col], errors="coerce")
        y = pd.to_numeric(sub["year"], errors="coerce")
        yoy_change = s.diff()
        prev = s.shift(1)
        yoy_pct = (yoy_change / prev) * 100
        yoy_pct = yoy_pct.replace([np.inf, -np.inf], np.nan)

        slope = np.nan
        valid = s.notna() & y.notna()
        if len(sub) >= 3 and int(valid.sum()) >= 3:
            coeffs = np.polyfit(y[valid].to_numpy(dtype=float), s[valid].to_numpy(dtype=float), 1)
            slope = float(coeffs[0])

        out = pd.DataFrame(
            {
                "economy": economy,
                "year": y.astype("Int64"),
                "dpi_score": s,
                "yoy_change": yoy_change,
                "yoy_pct_change": yoy_pct,
                "trend_slope": slope,
            }
        )
        pieces.append(out)

    trends = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    trends.to_csv(output_path, index=False)
    print(f"Trend analysis complete: {output_path}")


if __name__ == "__main__":
    main()
