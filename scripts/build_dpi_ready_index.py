from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
LEGACY_ROOT = (ROOT / ".." / "DPI").resolve()
LEGACY_DATA_DIR = LEGACY_ROOT / "data"

FOCUS_ECONOMIES_PATH = LEGACY_DATA_DIR / "top50_economies.csv"
LEGACY_FOCUS_USED_PATH = LEGACY_ROOT / "outputs" / "dpi_focus_economies_used.csv"
MC_RUNS = 500
PILLAR_WEIGHT_MAP = {
    "access_usage": 0.30,
    "affordability_inclusion": 0.25,
    "trust_governance": 0.25,
    "sustainability_resilience": 0.20,
}


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


def clamp01(x: pd.Series) -> pd.Series:
    return x.clip(0, 1)


def safe_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    keep = values.notna() & weights.notna()
    if not keep.any():
        return np.nan
    vals = values[keep].astype(float)
    wts = weights[keep].astype(float)
    return float((vals * wts).sum() / wts.sum())


def infer_layer(indicator_type: str, indicator_name: str, indicator_code: str) -> str:
    type_txt = str(indicator_type).strip().lower()
    txt = f"{indicator_name} {indicator_code}".lower()
    if type_txt == "outcome":
        return "impact"
    if any(
        token in txt
        for token in [
            "use",
            "usage",
            "adoption",
            "digital payment",
            "upi",
            "account use",
            "e-government",
            "e government",
            "service delivery",
            "citizen engagement",
            "gtmi_gt2",
            "gtmi_gt3",
            "psdi",
            "dcei",
        ]
    ):
        return "adoption"
    return "readiness"


def infer_equity_group(indicator_name: str, indicator_code: str) -> str:
    txt = f"{indicator_name} {indicator_code}".lower()
    if "gender" in txt:
        return "gender"
    if "rural" in txt or "urban" in txt:
        return "rural"
    if any(token in txt for token in ["literacy", "school", "education", "skills"]):
        return "education"
    return "none"


def infer_risk_group(pillar: str, indicator_name: str, indicator_code: str) -> str:
    txt = f"{indicator_name} {indicator_code}".lower()
    p = str(pillar).strip().lower()
    if any(token in txt for token in ["cyber", "secure", "security", "privacy", "protection"]):
        return "cyber"
    if p == "trust_governance" or any(
        token in txt for token in ["govern", "rule of law", "regulatory", "corruption", "voice", "stability"]
    ):
        return "governance"
    return "none"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    names_map = {
        "economy": ["economy", "country", "economy_name", "entityname", "entityiso", "iso3"],
        "indicator_code": ["indicator_code", "indicator", "indicator_id", "series_code", "seriescode"],
        "indicator_name": ["indicator_name", "series_name", "indicator_label", "seriesname"],
        "year": ["year", "time", "period", "datayear"],
        "value": ["value", "obs_value", "observation", "val", "datavalue"],
    }
    out = df.copy()
    lower = {c.lower(): c for c in out.columns}
    for target, aliases in names_map.items():
        for alias in aliases:
            if alias in lower:
                out = out.rename(columns={lower[alias]: target})
                break
    required = {"economy", "indicator_code", "year", "value"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Input data missing required columns: {', '.join(sorted(missing))}")
    if "indicator_name" not in out.columns:
        out["indicator_name"] = out["indicator_code"]
    return out


def impute_series(years: pd.Series, values: pd.Series) -> tuple[pd.Series, pd.Series]:
    y = years.to_numpy(dtype=float)
    v = values.to_numpy(dtype=float)
    original_na = np.isnan(v)
    out = v.copy()
    flags = np.array(["observed"] * len(v), dtype=object)
    non_missing = np.where(~np.isnan(v))[0]
    if len(non_missing) == 0:
        return pd.Series(out, index=years.index), pd.Series(["missing_unresolved"] * len(v), index=years.index)
    if len(non_missing) >= 2:
        interp = np.interp(y, y[non_missing], v[non_missing])
        fill_idx = np.where(original_na & ~np.isnan(interp))[0]
        out[fill_idx] = interp[fill_idx]
        flags[fill_idx] = "imputed_interp_or_locf"
    else:
        out[original_na] = v[non_missing[0]]
        flags[original_na] = "imputed_single_value"
    return pd.Series(out, index=years.index), pd.Series(flags, index=years.index)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_data_path = LEGACY_DATA_DIR / "master_export.csv"
    if not input_data_path.exists():
        input_data_path = LEGACY_DATA_DIR / "itu_export.csv"
    mapping_path = LEGACY_DATA_DIR / "indicator_mapping.csv"
    if not input_data_path.exists():
        raise FileNotFoundError("Missing input data file in legacy DPI data dir")
    if not mapping_path.exists():
        raise FileNotFoundError("Missing indicator_mapping.csv in legacy DPI data dir")

    raw_df = normalize_columns(pd.read_csv(input_data_path, na_values=["", "NA", "NaN", ".."]))
    map_df = pd.read_csv(mapping_path, na_values=["", "NA"])

    required_map = {"indicator_code", "pillar", "direction", "weight_within_pillar"}
    missing_map = required_map - set(map_df.columns)
    if missing_map:
        raise ValueError(f"Indicator mapping missing columns: {', '.join(sorted(missing_map))}")

    map_df["direction"] = map_df["direction"].astype(str).str.strip().str.lower()
    if not map_df["direction"].isin(["positive", "negative"]).all():
        raise ValueError("direction must be either positive or negative")
    map_df["weight_within_pillar"] = pd.to_numeric(map_df["weight_within_pillar"], errors="coerce").fillna(1.0)
    if "indicator_name" not in map_df.columns:
        map_df["indicator_name"] = map_df["indicator_code"]
    if "indicator_type" not in map_df.columns:
        map_df["indicator_type"] = "facilitation"
    if "layer" not in map_df.columns:
        map_df["layer"] = map_df.apply(
            lambda r: infer_layer(r.get("indicator_type"), r.get("indicator_name"), r.get("indicator_code")), axis=1
        )
    if "equity_group" not in map_df.columns:
        map_df["equity_group"] = map_df.apply(
            lambda r: infer_equity_group(r.get("indicator_name"), r.get("indicator_code")), axis=1
        )
    if "risk_group" not in map_df.columns:
        map_df["risk_group"] = map_df.apply(
            lambda r: infer_risk_group(r.get("pillar"), r.get("indicator_name"), r.get("indicator_code")), axis=1
        )
    if "include_v2" not in map_df.columns:
        map_df["include_v2"] = "yes"

    for col, allowed, fallback in [
        ("layer", {"readiness", "adoption", "impact"}, "readiness"),
        ("equity_group", {"gender", "rural", "education", "none"}, "none"),
        ("risk_group", {"cyber", "governance", "none"}, "none"),
        ("include_v2", {"yes", "no"}, "yes"),
    ]:
        map_df[col] = map_df[col].astype(str).str.strip().str.lower()
        map_df.loc[~map_df[col].isin(allowed), col] = fallback

    df = raw_df.merge(map_df, on="indicator_code", how="inner")
    if df.empty:
        raise ValueError("No overlap between input indicator data and indicator_mapping.csv")

    df["economy"] = df["economy"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)

    focus_economies: list[str] | None = None
    legacy_focus = read_focus_economies_from_csv_column(LEGACY_FOCUS_USED_PATH)
    if legacy_focus:
        matched = sorted(set(legacy_focus).intersection(set(df["economy"].unique())))
        if matched:
            focus_economies = matched

    if FOCUS_ECONOMIES_PATH.exists() and not focus_economies:
        top50_list = read_focus_economies(FOCUS_ECONOMIES_PATH)
        matched = sorted(set(top50_list).intersection(set(df["economy"].unique())))
        if len(matched) >= 30:
            focus_economies = matched
    if not focus_economies:
        focus_economies = sorted(df["economy"].unique().tolist())
    df = df[df["economy"].isin(focus_economies)].copy()

    # Stage 1 imputation: by economy + indicator
    df = df.sort_values(["economy", "indicator_code", "year"]).reset_index(drop=True)
    grouped = []
    for (_, _), part in df.groupby(["economy", "indicator_code"], sort=False):
        part = part.sort_values("year").copy()
        imputed, flags = impute_series(part["year"], part["value"])
        part["value_stage1"] = imputed.to_numpy()
        part["impute_flag"] = flags.to_numpy()
        grouped.append(part)
    df2 = pd.concat(grouped, ignore_index=True)

    # Stage 2 median by indicator-year
    med_iy = (
        df2.groupby(["indicator_code", "year"], dropna=False)["value_stage1"]
        .median()
        .reset_index(name="med_iy")
    )
    df2 = df2.merge(med_iy, on=["indicator_code", "year"], how="left")
    fill2 = df2["value_stage1"].isna() & df2["med_iy"].notna()
    df2.loc[fill2, "value_stage1"] = df2.loc[fill2, "med_iy"]
    df2.loc[fill2, "impute_flag"] = "imputed_indicator_year_median"
    df2 = df2.drop(columns=["med_iy"])

    # Stage 3 median by indicator
    med_i = df2.groupby("indicator_code", dropna=False)["value_stage1"].median().reset_index(name="med_i")
    df2 = df2.merge(med_i, on="indicator_code", how="left")
    fill3 = df2["value_stage1"].isna() & df2["med_i"].notna()
    df2.loc[fill3, "value_stage1"] = df2.loc[fill3, "med_i"]
    df2.loc[fill3, "impute_flag"] = "imputed_indicator_global_median"
    df2 = df2.drop(columns=["med_i"])

    df2["value_filled"] = df2["value_stage1"]
    df2["was_imputed"] = df2["impute_flag"] != "observed"
    df2["value_oriented"] = np.where(df2["direction"] == "negative", -df2["value_filled"], df2["value_filled"])

    # Normalize per indicator across all rows
    df2["norm_score"] = np.nan
    for ind, idx in df2.groupby("indicator_code").groups.items():
        df2.loc[idx, "norm_score"] = robust_minmax(df2.loc[idx, "value_oriented"]).to_numpy()

    # Pillar scores
    pillar_frames: list[pd.DataFrame] = []
    for pillar, part in df2.groupby("pillar", sort=False):
        rows = []
        for (economy, year), sub in part.groupby(["economy", "year"], sort=False):
            rows.append(
                {
                    "economy": economy,
                    "year": year,
                    f"pillar_{pillar}": safe_weighted_mean(sub["norm_score"], sub["weight_within_pillar"]),
                    f"imputed_share_{pillar}": float(sub["was_imputed"].mean()),
                }
            )
        pillar_frames.append(pd.DataFrame(rows))
    pillar_scores = pillar_frames[0]
    for frame in pillar_frames[1:]:
        pillar_scores = pillar_scores.merge(frame, on=["economy", "year"], how="outer")

    # Select focus year
    pillar_cols_all = [c for c in pillar_scores.columns if c.startswith("pillar_")]
    chosen_year = None
    chosen_coverage = -1
    for yr in sorted(pillar_scores["year"].dropna().unique().tolist(), reverse=True):
        sub = pillar_scores[(pillar_scores["year"] == yr) & (pillar_scores["economy"].isin(focus_economies))]
        if sub.empty:
            continue
        usable = (sub[pillar_cols_all].notna().sum(axis=1) >= 2).sum()
        if usable > chosen_coverage:
            chosen_coverage = int(usable)
            chosen_year = int(yr)
    if chosen_year is None:
        chosen_year = int(pillar_scores["year"].max())

    latest = pillar_scores[(pillar_scores["year"] == chosen_year) & (pillar_scores["economy"].isin(focus_economies))].copy()
    pillar_cols = [c for c in latest.columns if c.startswith("pillar_")]
    pillar_names = [c.replace("pillar_", "") for c in pillar_cols]
    pillar_weights = np.array([PILLAR_WEIGHT_MAP.get(name, np.nan) for name in pillar_names], dtype=float)
    if np.isnan(pillar_weights).any():
        pillar_weights = np.ones(len(pillar_cols), dtype=float) / max(len(pillar_cols), 1)
    else:
        pillar_weights = pillar_weights / pillar_weights.sum()

    eps = 1e-6
    scores = []
    for _, row in latest[pillar_cols].iterrows():
        vals = row.to_numpy(dtype=float)
        keep = np.isfinite(vals) & (vals > 0)
        if not keep.any():
            scores.append(np.nan)
            continue
        vals_use = np.maximum(eps, vals[keep])
        w_use = pillar_weights[keep] / pillar_weights[keep].sum()
        scores.append(float(np.exp(np.sum(w_use * np.log(vals_use)))))
    latest["dpi_ready_score"] = scores
    latest = latest[latest["dpi_ready_score"].notna()].copy()
    latest["rank"] = latest["dpi_ready_score"].rank(ascending=False, method="min")

    dq_sub = df2[df2["year"] == chosen_year]
    if not dq_sub.empty:
        dq = dq_sub.groupby("economy", dropna=False)["was_imputed"].mean().reset_index(name="imputation_share_total")
        latest = latest.merge(dq, on="economy", how="left")
    else:
        latest["imputation_share_total"] = np.nan

    # Monte Carlo
    np.random.seed(42)
    mc = []
    if len(pillar_cols) > 1 and not latest.empty:
        mat = latest[pillar_cols].to_numpy(dtype=float)
        for run in range(1, MC_RUNS + 1):
            w = np.random.exponential(scale=1.0, size=len(pillar_cols))
            w = w / w.sum()
            row_scores = np.exp(np.sum(w * np.log(np.maximum(eps, mat)), axis=1))
            mc.append(pd.DataFrame({"run": run, "economy": latest["economy"].to_numpy(), "score": row_scores}))
    mc_summary = pd.DataFrame()
    if mc:
        mc_df = pd.concat(mc, ignore_index=True)
        mc_summary = mc_df.groupby("economy", dropna=False)["score"].agg(
            mc_mean="mean",
            mc_p10=lambda s: float(np.nanquantile(s, 0.10)),
            mc_p90=lambda s: float(np.nanquantile(s, 0.90)),
        ).reset_index()

    # V2 outputs
    v2_map = map_df[map_df["include_v2"] == "yes"][
        ["indicator_code", "layer", "equity_group", "risk_group", "weight_within_pillar"]
    ]
    df_v2 = df2.merge(v2_map, on="indicator_code", how="inner", suffixes=("_base", "_v2"))
    if not df_v2.empty:
        for base in ["layer", "equity_group", "risk_group"]:
            col = base
            if col not in df_v2.columns:
                if f"{base}_v2" in df_v2.columns:
                    df_v2[col] = df_v2[f"{base}_v2"]
                elif f"{base}_base" in df_v2.columns:
                    df_v2[col] = df_v2[f"{base}_base"]

        layer_scores = (
            df_v2.groupby(["economy", "year", "layer"], dropna=False)["norm_score"]
            .mean()
            .reset_index()
        )
        layer_wide = layer_scores.pivot(index=["economy", "year"], columns="layer", values="norm_score").reset_index()
        for nm in ["readiness", "adoption", "impact"]:
            if nm not in layer_wide.columns:
                layer_wide[nm] = np.nan
        layer_wide = layer_wide.rename(
            columns={
                "readiness": "dpi_readiness_v2",
                "adoption": "dpi_adoption_v2",
                "impact": "dpi_impact_v2",
            }
        )
        base_w = np.array([0.5, 0.3, 0.2], dtype=float)
        v2_vals = layer_wide[["dpi_readiness_v2", "dpi_adoption_v2", "dpi_impact_v2"]].to_numpy(dtype=float)
        comp = []
        for row in v2_vals:
            keep = np.isfinite(row)
            if not keep.any():
                comp.append(np.nan)
                continue
            vals_use = np.maximum(eps, row[keep])
            w_use = base_w[keep] / base_w[keep].sum()
            comp.append(float(np.exp(np.sum(w_use * np.log(vals_use)))))
        layer_wide["dpi_composite_v2"] = comp
        layer_wide["dpi_utilization_efficiency"] = (
            100 * layer_wide["dpi_adoption_v2"] / np.maximum(layer_wide["dpi_readiness_v2"], 1)
        ).clip(0, 200)

        eq_data = df_v2[df_v2["equity_group"].isin(["gender", "rural", "education"])]
        if not eq_data.empty:
            eq_agg = eq_data.groupby(["economy", "year"], dropna=False)["norm_score"].mean().reset_index(name="equity_score")
            layer_wide = layer_wide.merge(eq_agg, on=["economy", "year"], how="left")
        else:
            layer_wide["equity_score"] = np.nan
        layer_wide["inclusion_penalty"] = np.where(
            layer_wide["equity_score"].isna(), 0.0, 1.0 - clamp01(layer_wide["equity_score"] / 100.0)
        )
        alpha = 0.25
        layer_wide["dpi_inclusion_adjusted_v2"] = layer_wide["dpi_composite_v2"] * (
            1.0 - alpha * layer_wide["inclusion_penalty"]
        )

        risk_data = df_v2[df_v2["risk_group"].isin(["cyber", "governance"])]
        if not risk_data.empty:
            risk_agg = risk_data.groupby(["economy", "year"], dropna=False)["norm_score"].mean().reset_index(name="risk_score")
            layer_wide = layer_wide.merge(risk_agg, on=["economy", "year"], how="left")
        else:
            layer_wide["risk_score"] = np.nan
        layer_wide["risk_factor"] = np.where(
            layer_wide["risk_score"].isna(), 0.7, 0.7 + 0.3 * clamp01(layer_wide["risk_score"] / 100.0)
        )
        layer_wide["dpi_risk_adjusted_v2"] = layer_wide["dpi_composite_v2"] * layer_wide["risk_factor"]

        year_expected = (
            df_v2[df_v2["value"].notna()]
            .groupby("year", dropna=False)["indicator_code"]
            .nunique()
            .reset_index(name="expected_indicator_points_year")
        )
        obs_cov = (
            df_v2[df_v2["value"].notna()]
            .groupby(["economy", "year"], dropna=False)["indicator_code"]
            .nunique()
            .reset_index(name="observed_indicator_points")
        )
        imp_cov = (
            df_v2[df_v2["was_imputed"] == 1]
            .groupby(["economy", "year"], dropna=False)["indicator_code"]
            .nunique()
            .reset_index(name="imputed_indicator_points")
        )
        layer_wide = layer_wide.merge(year_expected, on="year", how="left")
        layer_wide = layer_wide.merge(obs_cov, on=["economy", "year"], how="left")
        layer_wide = layer_wide.merge(imp_cov, on=["economy", "year"], how="left")
        layer_wide["observed_indicator_points"] = layer_wide["observed_indicator_points"].fillna(0)
        layer_wide["imputed_indicator_points"] = layer_wide["imputed_indicator_points"].fillna(0)
        layer_wide["coverage_ratio"] = np.where(
            (layer_wide["expected_indicator_points_year"].notna()) & (layer_wide["expected_indicator_points_year"] > 0),
            (layer_wide["observed_indicator_points"] / layer_wide["expected_indicator_points_year"]).clip(0, 1),
            np.nan,
        )
        denom = layer_wide["observed_indicator_points"] + layer_wide["imputed_indicator_points"]
        layer_wide["imputation_share"] = np.where(
            denom > 0, (layer_wide["imputed_indicator_points"] / denom).clip(0, 1), 0
        )
        layer_wide["dpi_confidence_score"] = 100 * (
            0.7 * layer_wide["coverage_ratio"].fillna(0) + 0.3 * (1 - layer_wide["imputation_share"].fillna(0))
        )
        layer_wide["confidence_tier"] = np.where(
            layer_wide["dpi_confidence_score"] >= 80,
            "High",
            np.where(layer_wide["dpi_confidence_score"] >= 60, "Medium", "Low"),
        )

        v2_latest_year = int(layer_wide["year"].max())
        v2_latest = layer_wide[
            (layer_wide["year"] == v2_latest_year) & (layer_wide["dpi_composite_v2"].notna())
        ].copy()
        if not v2_latest.empty:
            v2_latest["rank_v2"] = v2_latest["dpi_composite_v2"].rank(ascending=False, method="min")

        layer_wide.to_csv(OUTPUT_DIR / "dpi_v2_scores_all_years.csv", index=False)
        v2_latest.to_csv(OUTPUT_DIR / "dpi_v2_scores_latest_year.csv", index=False)
        layer_wide[
            ["economy", "year", "coverage_ratio", "imputation_share", "dpi_confidence_score", "confidence_tier"]
        ].to_csv(OUTPUT_DIR / "dpi_v2_confidence.csv", index=False)

    # Persist core outputs
    df2.to_csv(OUTPUT_DIR / "dpi_ready_long_with_imputation.csv", index=False)
    pillar_scores.to_csv(OUTPUT_DIR / "dpi_ready_pillar_scores_all_years.csv", index=False)
    latest.to_csv(OUTPUT_DIR / "dpi_ready_scores_latest_year.csv", index=False)
    pd.DataFrame({"economy": sorted(set(focus_economies))}).to_csv(OUTPUT_DIR / "dpi_focus_economies_used.csv", index=False)
    map_df[
        ["indicator_code", "indicator_name", "layer", "equity_group", "risk_group", "include_v2"]
    ].drop_duplicates().to_csv(OUTPUT_DIR / "dpi_v2_indicator_metadata.csv", index=False)
    if not mc_summary.empty:
        mc_summary.to_csv(OUTPUT_DIR / "dpi_ready_sensitivity_mc.csv", index=False)

    print("DPI-ready Python build complete.")
    print(f"Focus year: {chosen_year}")
    print(f"Output dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
