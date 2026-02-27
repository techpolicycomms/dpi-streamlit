from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
LEGACY_ROOT = (ROOT / ".." / "DPI").resolve()
LEGACY_DATA_DIR = LEGACY_ROOT / "data"
LEGACY_FOCUS_USED_PATH = LEGACY_ROOT / "outputs" / "dpi_focus_economies_used.csv"


def read_focus_economies(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    body = lines[1:] if lines[0].strip().lower() == "economy" else lines
    economies: list[str] = []
    for line in body:
        val = line.strip().strip('"')
        if val:
            economies.append(val)
    return economies


def read_focus_economies_from_csv_column(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if "economy" not in df.columns:
        return []
    vals = df["economy"].astype(str).str.strip()
    vals = vals[vals != ""]
    return vals.unique().tolist()


def robust_minmax(x: pd.Series) -> pd.Series:
    if x.dropna().empty:
        return pd.Series(np.nan, index=x.index)
    lo, hi = np.nanquantile(x.to_numpy(dtype=float), [0.05, 0.95], method="linear")
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(50.0, index=x.index)
    scaled = 100.0 * (x - lo) / (hi - lo)
    return scaled.clip(0, 100)


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "economy": ["economy", "country", "entityname"],
        "indicator_code": ["indicator_code", "indicator", "series_code"],
        "year": ["year", "time", "datayear"],
        "value": ["value", "obs_value", "datavalue"],
    }
    out = df.copy()
    lower = {c.lower(): c for c in out.columns}
    for target, names in aliases.items():
        for name in names:
            if name in lower:
                out = out.rename(columns={lower[name]: target})
                break
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    master = LEGACY_DATA_DIR / "master_export.csv"
    if not master.exists():
        master = LEGACY_DATA_DIR / "itu_export.csv"
    mapping = LEGACY_DATA_DIR / "doughnut_indicator_mapping.csv"
    top50_path = LEGACY_DATA_DIR / "top50_economies.csv"
    dpi_v2_path = OUTPUT_DIR / "dpi_v2_scores_all_years.csv"

    if not master.exists() or not mapping.exists():
        raise FileNotFoundError("Missing required legacy data/mapping files for doughnut pipeline")

    raw = normalize_cols(pd.read_csv(master, na_values=["", "NA", "NaN", ".."]))
    map_df = pd.read_csv(mapping, na_values=["", "NA"])
    if "action_group" not in map_df.columns:
        map_df["action_group"] = "none"
    map_df["action_group"] = map_df["action_group"].astype(str).str.strip().str.lower()
    map_df.loc[(map_df["action_group"].isna()) | (map_df["action_group"] == ""), "action_group"] = "none"

    required = {"economy", "indicator_code", "year", "value"}
    if not required.issubset(set(raw.columns)):
        raise ValueError("Master data missing required columns")

    df = raw.merge(map_df, on="indicator_code", how="inner")
    if df.empty:
        raise ValueError("No overlap between master data and doughnut mapping")
    df["economy"] = df["economy"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["value"].notna() & df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)

    all_economies = sorted(df["economy"].unique().tolist())
    focus_economies = all_economies
    legacy_focus = read_focus_economies_from_csv_column(LEGACY_FOCUS_USED_PATH)
    if legacy_focus:
        matched = sorted(set(legacy_focus).intersection(set(all_economies)))
        if matched:
            focus_economies = matched
    if top50_path.exists():
        top50_list = read_focus_economies(top50_path)
        matched = sorted(set(top50_list).intersection(set(all_economies)))
        if len(matched) >= 30 and not legacy_focus:
            focus_economies = matched
    df = df[df["economy"].isin(focus_economies)].copy()

    df["value_oriented"] = np.where(df["direction"] == "negative", -df["value"], df["value"])
    df["norm_score"] = np.nan
    for ind, idx in df.groupby("indicator_code").groups.items():
        df.loc[idx, "norm_score"] = robust_minmax(df.loc[idx, "value_oriented"]).to_numpy()

    dim_scores = (
        df.groupby(["economy", "year", "dimension_type", "doughnut_dimension"], dropna=False)["norm_score"]
        .mean()
        .reset_index()
    )
    agg = (
        dim_scores.groupby(["economy", "year", "dimension_type"], dropna=False)["norm_score"]
        .mean()
        .reset_index()
    )

    expected_dims = (
        map_df[["dimension_type", "doughnut_dimension"]]
        .drop_duplicates()
        .groupby("dimension_type", dropna=False)["doughnut_dimension"]
        .nunique()
        .reset_index(name="expected_dimensions")
    )
    observed_dims = (
        dim_scores[dim_scores["norm_score"].notna()][["economy", "year", "dimension_type", "doughnut_dimension"]]
        .drop_duplicates()
        .groupby(["economy", "year", "dimension_type"], dropna=False)["doughnut_dimension"]
        .nunique()
        .reset_index(name="observed_dimensions")
        .merge(expected_dims, on="dimension_type", how="left")
    )
    observed_dims["dimension_coverage"] = np.where(
        observed_dims["expected_dimensions"] > 0,
        (observed_dims["observed_dimensions"] / observed_dims["expected_dimensions"]).clip(0, 1),
        np.nan,
    )

    social = agg[agg["dimension_type"] == "social_foundation"][["economy", "year", "norm_score"]].rename(
        columns={"norm_score": "social_foundation_score"}
    )
    eco = agg[agg["dimension_type"] == "ecological_ceiling"][["economy", "year", "norm_score"]].rename(
        columns={"norm_score": "ecological_ceiling_score"}
    )
    doughnut = social.merge(eco, on=["economy", "year"], how="outer")

    cov_social = observed_dims[observed_dims["dimension_type"] == "social_foundation"][
        ["economy", "year", "dimension_coverage"]
    ].rename(columns={"dimension_coverage": "social_dimension_coverage"})
    cov_eco = observed_dims[observed_dims["dimension_type"] == "ecological_ceiling"][
        ["economy", "year", "dimension_coverage"]
    ].rename(columns={"dimension_coverage": "ecological_dimension_coverage"})
    doughnut = doughnut.merge(cov_social, on=["economy", "year"], how="left")
    doughnut = doughnut.merge(cov_eco, on=["economy", "year"], how="left")

    eps = 1e-6
    vals = []
    for _, row in doughnut.iterrows():
        soc = row.get("social_foundation_score")
        eco_score = row.get("ecological_ceiling_score")
        eco_safety = max(0.0, 100.0 - eco_score) if pd.notna(eco_score) else np.nan
        parts = [p for p in [soc, eco_safety] if pd.notna(p)]
        if len(parts) == 0:
            vals.append(np.nan)
        elif len(parts) == 1:
            vals.append(float(parts[0]))
        else:
            vals.append(float(np.exp(np.mean(np.log(np.maximum(eps, np.array(parts, dtype=float)))))))
    doughnut["doughnut_score"] = vals

    green_df = df[df["action_group"] == "green_transition_actions"]
    if not green_df.empty:
        green_scores = (
            green_df.groupby(["economy", "year"], dropna=False)["norm_score"]
            .mean()
            .reset_index(name="green_transition_actions_score")
        )
    else:
        green_scores = pd.DataFrame(columns=["economy", "year", "green_transition_actions_score"])
    doughnut = doughnut.merge(green_scores, on=["economy", "year"], how="left")

    target_year = 2030
    doughnut["green_expected_2030_path"] = np.nan
    doughnut["green_distance_to_target"] = np.nan
    for economy, sub in doughnut[doughnut["green_transition_actions_score"].notna()].groupby("economy", sort=False):
        idx = sub.index
        baseline_idx = sub["year"].idxmin()
        baseline_year = int(doughnut.loc[baseline_idx, "year"])
        baseline_val = float(doughnut.loc[baseline_idx, "green_transition_actions_score"])
        ys = doughnut.loc[idx, "year"].to_numpy(dtype=float)
        span = target_year - baseline_year
        if span > 0:
            expected = baseline_val + (80.0 - baseline_val) * ((ys - baseline_year) / span)
        else:
            expected = np.repeat(80.0, len(idx))
        doughnut.loc[idx, "green_expected_2030_path"] = expected
        doughnut.loc[idx, "green_distance_to_target"] = (
            doughnut.loc[idx, "green_transition_actions_score"] - doughnut.loc[idx, "green_expected_2030_path"]
        )

    focus_year = None
    for yr in sorted(doughnut["year"].dropna().unique().tolist(), reverse=True):
        sub = doughnut[doughnut["year"] == yr]
        both_present = int((sub["social_foundation_score"].notna() & sub["ecological_ceiling_score"].notna()).sum())
        if both_present >= 3:
            focus_year = int(yr)
            break
    if focus_year is None:
        for yr in sorted(doughnut["year"].dropna().unique().tolist(), reverse=True):
            if not doughnut[(doughnut["year"] == yr) & (doughnut["doughnut_score"].notna())].empty:
                focus_year = int(yr)
                break
    if focus_year is None:
        focus_year = int(doughnut["year"].max())

    latest = doughnut[(doughnut["year"] == focus_year) & (doughnut["doughnut_score"].notna())].copy()
    latest["rank"] = latest["doughnut_score"].rank(ascending=False, method="min")

    dpi_green = pd.DataFrame()
    if dpi_v2_path.exists():
        dpi_v2 = pd.read_csv(dpi_v2_path)
        if {"economy", "year", "dpi_composite_v2"}.issubset(set(dpi_v2.columns)):
            dpi_green = dpi_v2[["economy", "year", "dpi_composite_v2"]].merge(
                doughnut[["economy", "year", "green_transition_actions_score", "green_distance_to_target"]],
                on=["economy", "year"],
                how="inner",
            )
            if not dpi_green.empty:
                dpi_green["climate_action_efficiency"] = (
                    100 * dpi_green["green_transition_actions_score"] / np.maximum(dpi_green["dpi_composite_v2"], 1)
                )
                dpi_green.to_csv(OUTPUT_DIR / "dpi_vs_green_actions.csv", index=False)

                gap = pd.DataFrame()
                for yr in sorted(dpi_green["year"].dropna().unique().tolist(), reverse=True):
                    candidate = dpi_green[
                        (dpi_green["year"] == yr)
                        & dpi_green["dpi_composite_v2"].notna()
                        & dpi_green["green_transition_actions_score"].notna()
                    ]
                    if not candidate.empty:
                        gap = candidate.copy()
                        break
                if not gap.empty:
                    m_dpi = float(gap["dpi_composite_v2"].median())
                    m_green = float(gap["green_transition_actions_score"].median())
                    gap["quadrant"] = np.where(
                        (gap["dpi_composite_v2"] >= m_dpi) & (gap["green_transition_actions_score"] >= m_green),
                        "High DPI / High Green",
                        np.where(
                            (gap["dpi_composite_v2"] >= m_dpi) & (gap["green_transition_actions_score"] < m_green),
                            "High DPI / Low Green",
                            np.where(
                                (gap["dpi_composite_v2"] < m_dpi) & (gap["green_transition_actions_score"] >= m_green),
                                "Low DPI / High Green",
                                "Low DPI / Low Green",
                            ),
                        ),
                    )
                    gap["dpi_green_gap"] = gap["green_transition_actions_score"] - gap["dpi_composite_v2"]
                    gap.to_csv(OUTPUT_DIR / "climate_action_gap_report.csv", index=False)

    doughnut.to_csv(OUTPUT_DIR / "doughnut_pillar_scores.csv", index=False)
    latest.to_csv(OUTPUT_DIR / "doughnut_scores_latest_year.csv", index=False)
    doughnut[
        ["economy", "year", "green_transition_actions_score", "green_expected_2030_path", "green_distance_to_target"]
    ].to_csv(OUTPUT_DIR / "doughnut_green_actions_scores.csv", index=False)

    print("Doughnut Python build complete.")
    print(f"Focus year: {focus_year}")
    print(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
