from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import (
    CORE_REQUIRED_FILES,
    OPTIONAL_FILES,
    REQUIRED_FILES,
    list_missing_files,
    load_csv,
    resolve_outputs_dir,
)


st.set_page_config(page_title="DPI Streamlit V2", page_icon="📊", layout="wide")


@st.cache_data
def load_all(source_mode: str) -> tuple[dict[str, pd.DataFrame], str]:
    base_dir = resolve_outputs_dir(source_mode=source_mode)
    data = {
        "v2_all": load_csv("dpi_v2_scores_all_years", base_dir=base_dir),
        "v2_latest": load_csv("dpi_v2_scores_latest_year", base_dir=base_dir),
        "doughnut": load_csv("doughnut_pillar_scores", base_dir=base_dir),
        "trends": load_csv("dpi_ready_trends", base_dir=base_dir),
        "corr": pd.DataFrame(),
        "coverage": pd.DataFrame(),
    }
    for key in OPTIONAL_FILES:
        try:
            data["corr" if key == "indicator_correlation_report" else "coverage"] = load_csv(key, base_dir=base_dir)
        except FileNotFoundError:
            continue
    return data, str(base_dir)


def latest_for_economy(df: pd.DataFrame, economy: str) -> pd.Series | None:
    part = df[df["economy"] == economy].copy()
    if part.empty:
        return None
    part["year"] = pd.to_numeric(part["year"], errors="coerce")
    return part.sort_values("year").iloc[-1]


def select_score_column(score_mode: str) -> str:
    if score_mode == "Inclusion-adjusted":
        return "dpi_inclusion_adjusted_v2"
    if score_mode == "Risk-adjusted":
        return "dpi_risk_adjusted_v2"
    return "dpi_composite_v2"


def add_confidence_weighted_score(df: pd.DataFrame, base_col: str) -> pd.DataFrame:
    out = df.copy()
    if base_col not in out.columns:
        return out
    confidence = pd.to_numeric(out.get("dpi_confidence_score"), errors="coerce").fillna(0) / 100.0
    out["score_confidence_weighted"] = pd.to_numeric(out[base_col], errors="coerce") * confidence
    return out


def supports_ols_trendline() -> bool:
    try:
        import statsmodels.api  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


@st.cache_data
def load_india_pipeline_data() -> pd.DataFrame:
    path = Path(__file__).resolve().parent / "data" / "india_pipeline" / "dclo_state_year.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce")
    return df


def minmax_0_100(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    lo = vals.min(skipna=True)
    hi = vals.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series([50.0] * len(vals), index=series.index)
    return 100.0 * (vals - lo) / (hi - lo)


def build_policy_signal_explainer(v2_sel: pd.DataFrame, score_col: str, latest: pd.Series | None) -> str:
    if latest is None or score_col not in v2_sel.columns:
        return "Policy signal is not available for the current selection."
    series = (
        v2_sel[["year", score_col]]
        .dropna()
        .sort_values("year")
    )
    if series.empty:
        return "Policy signal is not available for the current selection."

    latest_score = float(pd.to_numeric(latest.get(score_col), errors="coerce")) if score_col in latest.index else float(
        pd.to_numeric(series[score_col].iloc[-1], errors="coerce")
    )
    confidence = float(pd.to_numeric(latest.get("dpi_confidence_score"), errors="coerce")) if latest is not None else float("nan")
    yoy = float(series[score_col].iloc[-1] - series[score_col].iloc[-2]) if len(series) >= 2 else float("nan")

    if pd.isna(yoy):
        trend_label = "insufficient trend history"
    elif yoy >= 2:
        trend_label = "strong positive momentum"
    elif yoy >= 0:
        trend_label = "modest positive momentum"
    elif yoy > -2:
        trend_label = "mild deterioration"
    else:
        trend_label = "material deterioration"

    if pd.isna(confidence):
        confidence_label = "unknown confidence"
    elif confidence >= 80:
        confidence_label = "high confidence"
    elif confidence >= 60:
        confidence_label = "medium confidence"
    else:
        confidence_label = "low confidence"

    return (
        f"Latest score is {latest_score:.2f} with {trend_label}. "
        f"The current evidence quality is {confidence_label}. "
        "Use this signal as a prioritization input, not as causal proof."
    )


def main() -> None:
    st.title("DPI Dashboard V2 (Streamlit)")
    st.caption("Improved app with data-source routing, V2 score controls, and policy diagnostics.")

    source_options = {
        "Auto": "auto",
        "Env path (DPI_OUTPUTS_DIR)": "env",
        "Repo outputs": "repo",
        "Snapshot outputs": "snapshot",
        "Legacy DPI outputs": "legacy",
    }
    with st.sidebar:
        source_label = st.selectbox("Data source", list(source_options.keys()), index=0)
        source_mode = source_options[source_label]

    try:
        selected_dir = resolve_outputs_dir(source_mode=source_mode)
    except ValueError:
        # Most common case: source_mode='env' but DPI_OUTPUTS_DIR is not defined.
        source_mode = "auto"
        selected_dir = resolve_outputs_dir(source_mode=source_mode)
        st.warning("`DPI_OUTPUTS_DIR` is not set; switched data source to Auto.")
    selected_missing = list_missing_files(selected_dir)
    effective_mode = source_mode
    source_dir = selected_dir
    missing_files = selected_missing

    # If user-selected mode is broken, try safer fallbacks in priority order.
    if source_mode != "auto" and selected_missing:
        fallback_modes = ["repo", "snapshot", "auto"]
        switched = False
        for mode in fallback_modes:
            if mode == source_mode:
                continue
            candidate_dir = resolve_outputs_dir(source_mode=mode)
            candidate_missing = list_missing_files(candidate_dir)
            if not candidate_missing:
                effective_mode = mode
                source_dir = candidate_dir
                missing_files = candidate_missing
                switched = True
                break
        if switched:
            st.warning(
                "Selected source is missing core files; automatically switched to a valid source."
            )

    st.info(f"Using outputs directory: `{source_dir}`")
    if missing_files:
        st.warning("Missing files in selected source: " + ", ".join(missing_files))

    try:
        data, loaded_dir = load_all(source_mode=effective_mode)
    except Exception as exc:
        st.error(f"Failed to load outputs: {exc}")
        st.stop()
    st.caption(f"Loaded from: `{loaded_dir}`")

    v2_all = data["v2_all"]
    v2_latest = data["v2_latest"]
    doughnut = data["doughnut"]
    trends = data["trends"]
    corr = data["corr"]
    coverage = data["coverage"]
    india_raw = load_india_pipeline_data()

    v2_all["year"] = pd.to_numeric(v2_all["year"], errors="coerce")
    v2_latest["year"] = pd.to_numeric(v2_latest["year"], errors="coerce")
    doughnut["year"] = pd.to_numeric(doughnut["year"], errors="coerce")
    trends["year"] = pd.to_numeric(trends.get("year"), errors="coerce")

    economies_all = sorted(v2_all["economy"].dropna().astype(str).unique().tolist())
    focus_economies = [e for e in ["India", "Switzerland"] if e in economies_all]
    economies = focus_economies if focus_economies else economies_all
    if not economies:
        st.error("No economies found in v2 dataset.")
        st.stop()
    default_economy = "India" if "India" in economies else economies[0]

    with st.sidebar:
        economy = st.selectbox("Economy (focused)", economies, index=economies.index(default_economy))
        year_min = int(v2_all["year"].min())
        year_max = int(v2_all["year"].max())
        year_range = st.slider("Year range", year_min, year_max, (max(year_min, 2014), year_max))
        score_mode = st.selectbox("V2 score mode", ["Composite", "Inclusion-adjusted", "Risk-adjusted"], index=0)
        use_confidence_weight = st.checkbox("Confidence-weight score", value=False)
        trust_tiers = st.multiselect("Trust tier filter", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        top_n = st.slider("Top N ranking rows", 5, 30, 15)

    base_score_col = select_score_column(score_mode=score_mode)

    v2_sel = v2_all[(v2_all["economy"] == economy)].copy()
    v2_sel = v2_sel[(v2_sel["year"] >= year_range[0]) & (v2_sel["year"] <= year_range[1])]
    v2_sel = add_confidence_weighted_score(v2_sel, base_col=base_score_col)
    score_col = "score_confidence_weighted" if use_confidence_weight else base_score_col

    doughnut_sel = doughnut[(doughnut["economy"] == economy)].copy()
    doughnut_sel = doughnut_sel[(doughnut_sel["year"] >= year_range[0]) & (doughnut_sel["year"] <= year_range[1])]

    latest = latest_for_economy(v2_latest, economy)
    if latest is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Composite v2", f"{float(latest.get('dpi_composite_v2', float('nan'))):.2f}")
        c2.metric("Inclusion-adjusted", f"{float(latest.get('dpi_inclusion_adjusted_v2', float('nan'))):.2f}")
        c3.metric("Risk-adjusted", f"{float(latest.get('dpi_risk_adjusted_v2', float('nan'))):.2f}")
        conf = latest.get("dpi_confidence_score")
        c4.metric("Confidence", f"{float(conf):.1f}" if pd.notna(conf) else "N/A")

    with st.expander("Policy signal explainer", expanded=True):
        st.write(build_policy_signal_explainer(v2_sel=v2_sel, score_col=score_col, latest=latest))
        st.markdown(
            """
            - **What it means**: A compact decision signal combining score level, direction of change, and confidence.
            - **How to use it**: Prioritize reforms and sequencing, then validate with sector diagnostics and local evidence.
            - **How not to use it**: Do not interpret score changes as direct causal effect without identification strategy.
            """
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["DPI ↔ Doughnut Linkage", "Doughnut", "Trends", "Diagnostics", "Policy Research Pack"]
    )

    with tab1:
        st.subheader("DPI Enabling Pillars and Doughnut Linkage")
        plot_cols = [c for c in ["dpi_readiness_v2", "dpi_adoption_v2", "dpi_impact_v2", score_col] if c in v2_sel.columns]
        if plot_cols and not v2_sel.empty:
            long_df = v2_sel.melt(id_vars=["year"], value_vars=plot_cols, var_name="metric", value_name="score")
            fig = px.line(long_df, x="year", y="score", color="metric", markers=True)
            st.plotly_chart(fig, use_container_width=True)

        linkage = v2_sel.merge(
            doughnut_sel[
                [
                    c
                    for c in [
                        "year",
                        "social_foundation_score",
                        "ecological_ceiling_score",
                        "doughnut_score",
                    ]
                    if c in doughnut_sel.columns
                ]
            ],
            on="year",
            how="inner",
        )
        if not linkage.empty:
            st.subheader("Pillar-to-Doughnut Trend Overlay")
            trend_cols = [
                c
                for c in [
                    "dpi_readiness_v2",
                    "dpi_adoption_v2",
                    "dpi_impact_v2",
                    "social_foundation_score",
                    "doughnut_score",
                ]
                if c in linkage.columns
            ]
            if trend_cols:
                trend_long = linkage.melt(id_vars=["year"], value_vars=trend_cols, var_name="metric", value_name="score")
                fig_link_trend = px.line(trend_long, x="year", y="score", color="metric", markers=True)
                st.plotly_chart(fig_link_trend, use_container_width=True)

            st.subheader("Pillar-to-Doughnut Relationship")
            if "dpi_readiness_v2" in linkage.columns and "doughnut_score" in linkage.columns:
                trendline_mode = "ols" if supports_ols_trendline() else None
                fig_scatter = px.scatter(
                    linkage,
                    x="dpi_readiness_v2",
                    y="doughnut_score",
                    text="year",
                    trendline=trendline_mode,
                    title="Readiness vs Doughnut Score",
                )
                fig_scatter.update_traces(textposition="top center")
                st.plotly_chart(fig_scatter, use_container_width=True)
                if trendline_mode is None:
                    st.caption("Trendline disabled: `statsmodels` not installed in current environment.")

        st.dataframe(v2_sel.sort_values("year", ascending=False), use_container_width=True)

        st.subheader("Latest Year Focus Comparison")
        latest_year = int(v2_all["year"].max())
        rank_df = v2_all[(v2_all["year"] == latest_year) & (v2_all["economy"].isin(economies))].copy()
        if "confidence_tier" in rank_df.columns:
            rank_df = rank_df[rank_df["confidence_tier"].isin(trust_tiers)]
        rank_df = add_confidence_weighted_score(rank_df, base_col=base_score_col)
        rank_score_col = "score_confidence_weighted" if use_confidence_weight else base_score_col
        rank_df[rank_score_col] = pd.to_numeric(rank_df[rank_score_col], errors="coerce")
        rank_df = rank_df.dropna(subset=[rank_score_col]).sort_values(rank_score_col, ascending=False).head(top_n)
        if not rank_df.empty:
            fig_rank = px.bar(rank_df, x=rank_score_col, y="economy", orientation="h", title=f"Top {top_n} in {latest_year}")
            fig_rank.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_rank, use_container_width=True)

        st.subheader("Policy Signal: Year-over-Year Movement")
        if score_col in v2_sel.columns and len(v2_sel) >= 2:
            yoy = v2_sel[["year", score_col]].dropna().sort_values("year")
            yoy["yoy_change"] = yoy[score_col].diff()
            st.dataframe(yoy.sort_values("year", ascending=False), use_container_width=True)

        if economy == "India" and not india_raw.empty:
            st.subheader("India Boost from Data Pipelines Project")
            india_signal_col = "DCLO_score_context_adjusted" if "DCLO_score_context_adjusted" in india_raw.columns else "DCLO_score"
            india_agg = (
                india_raw.groupby("year", dropna=True)
                .agg(
                    india_state_signal=(india_signal_col, "mean"),
                    india_state_count=("state_name", "nunique"),
                    india_acc=("ACC_score", "mean"),
                    india_skl=("SKL_score", "mean"),
                    india_srv=("SRV_score", "mean"),
                    india_agr=("AGR_score", "mean"),
                    india_eco=("ECO_score", "mean"),
                    india_out=("OUT_score", "mean"),
                )
                .reset_index()
            )
            boost = v2_sel.merge(india_agg, on="year", how="inner")
            if not boost.empty and score_col in boost.columns:
                boost = boost.sort_values("year")
                boost["dpi_norm"] = minmax_0_100(boost[score_col])
                boost["india_state_norm"] = minmax_0_100(boost["india_state_signal"])
                boost["boosted_signal"] = 0.7 * boost["dpi_norm"] + 0.3 * boost["india_state_norm"]
                boost["boost_uplift"] = boost["boosted_signal"] - boost["dpi_norm"]

                latest_boost = boost.iloc[-1]
                b1, b2, b3 = st.columns(3)
                b1.metric("India state coverage", f"{int(latest_boost['india_state_count'])}")
                b2.metric("Boosted signal (latest)", f"{float(latest_boost['boosted_signal']):.2f}")
                b3.metric("Uplift vs DPI-only", f"{float(latest_boost['boost_uplift']):+.2f}")

                boost_long = boost.melt(
                    id_vars=["year"],
                    value_vars=["dpi_norm", "india_state_norm", "boosted_signal"],
                    var_name="metric",
                    value_name="score",
                )
                fig_boost = px.line(
                    boost_long,
                    x="year",
                    y="score",
                    color="metric",
                    markers=True,
                    title="India boost view: DPI + state-level DCLO signal",
                )
                st.plotly_chart(fig_boost, use_container_width=True)

                if "ACC_score" in india_raw.columns:
                    latest_year_india = int(india_raw["year"].dropna().max())
                    state_latest = india_raw[india_raw["year"] == latest_year_india].copy()
                    state_score_col = (
                        "DCLO_score_context_adjusted"
                        if "DCLO_score_context_adjusted" in state_latest.columns
                        else "DCLO_score"
                    )
                    state_latest[state_score_col] = pd.to_numeric(state_latest[state_score_col], errors="coerce")
                    state_latest = state_latest.dropna(subset=[state_score_col]).sort_values(state_score_col, ascending=False)
                    st.caption(f"Latest state snapshot ({latest_year_india})")
                    st.dataframe(
                        state_latest[["state_name", state_score_col, "ACC_score", "SKL_score", "SRV_score", "AGR_score", "ECO_score", "OUT_score"]]
                        .head(10),
                        use_container_width=True,
                    )
            else:
                st.info("India boost view is unavailable for the selected year range.")

    with tab2:
        st.subheader("Doughnut Scores")
        if not doughnut_sel.empty:
            plot_cols = [c for c in ["social_foundation_score", "ecological_ceiling_score", "doughnut_score"] if c in doughnut_sel.columns]
            long_df = doughnut_sel.melt(id_vars=["year"], value_vars=plot_cols, var_name="metric", value_name="score")
            fig = px.line(long_df, x="year", y="score", color="metric", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(doughnut_sel.sort_values("year", ascending=False), use_container_width=True)

    with tab3:
        st.subheader("Trend Output")
        trend_sel = trends[trends["economy"] == economy].copy() if "economy" in trends.columns else pd.DataFrame()
        if not trend_sel.empty and "dpi_score" in trend_sel.columns:
            fig = px.line(trend_sel.sort_values("year"), x="year", y="dpi_score", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            if "yoy_change" in trend_sel.columns:
                fig_yoy = px.bar(trend_sel.sort_values("year"), x="year", y="yoy_change", title="YoY Change")
                st.plotly_chart(fig_yoy, use_container_width=True)
        st.dataframe(trend_sel.sort_values("year", ascending=False), use_container_width=True)

    with tab4:
        st.subheader("Indicator Correlations")
        if not corr.empty:
            corr["correlation"] = pd.to_numeric(corr.get("correlation"), errors="coerce")
            corr["abs_correlation"] = corr["correlation"].abs()
            threshold = st.slider("Absolute correlation threshold", 0.0, 1.0, 0.7, 0.05)
            sign = st.selectbox("Correlation sign", ["Any", "Positive", "Negative"], index=0)
            filtered = corr[corr["abs_correlation"] >= threshold].copy()
            if sign == "Positive":
                filtered = filtered[filtered["correlation"] > 0]
            elif sign == "Negative":
                filtered = filtered[filtered["correlation"] < 0]
            st.dataframe(filtered.sort_values("abs_correlation", ascending=False).head(50), use_container_width=True)

        st.subheader("Source Coverage")
        if coverage.empty:
            st.info("`source_coverage_summary.csv` not found. Core app functionality is still available.")
        else:
            st.dataframe(coverage, use_container_width=True)

        st.subheader("Dataset Health")
        health = pd.DataFrame(
            {
                "required_file": list(REQUIRED_FILES.values()),
                "is_present": [
                    (file_name not in missing_files)
                    if file_name in CORE_REQUIRED_FILES.values()
                    else (source_dir / file_name).exists()
                    for file_name in REQUIRED_FILES.values()
                ],
                "type": [
                    "core" if file_name in CORE_REQUIRED_FILES.values() else "optional"
                    for file_name in REQUIRED_FILES.values()
                ],
            }
        )
        st.dataframe(health, use_container_width=True)

    with tab5:
        st.subheader("Policy Problem Statement")
        st.markdown(
            """
            Countries invest in DPI but often track readiness, adoption, impact, and sustainability in disconnected systems.
            This creates a policy blind spot: governments cannot easily identify where capability exists, where uptake lags,
            and where digital progress may conflict with inclusion or ecological outcomes.
            """
        )

        st.subheader("PhD Research Questions")
        st.markdown(
            """
            1. How can DPI capability be measured as a multidimensional construct across readiness, adoption, and impact?
            2. Which capability layers are most associated with downstream socioeconomic outcomes?
            3. How does DPI progress relate to Doughnut-style social foundation and ecological ceiling indicators?
            4. How sensitive are rankings and policy conclusions to weighting and imputation choices?
            5. How should confidence and evidence quality be integrated into policy interpretation?
            6. What practical policy sequencing follows from high-level score patterns and gaps?
            """
        )

        st.subheader("Methodology Summary")
        st.markdown(
            """
            - **Construct design**: Formative multidimensional index architecture.
            - **Normalization**: Robust min-max scaling (P5-P95), with directionality handling.
            - **Aggregation**: Pillar-level weighted means; cross-layer geometric composite.
            - **Imputation**: Hierarchical fill strategy with explicit imputation flags.
            - **Uncertainty**: Coverage and imputation-based confidence score + sensitivity checks.
            - **Interpretation stance**: Descriptive/diagnostic, not causal identification.
            """
        )

        st.subheader("Complete Data Calculation Process")
        st.markdown(
            """
            **Step 1 — Ingestion and mapping**
            - Load harmonized indicator panel (`economy`, `year`, `indicator_code`, `value`).
            - Apply indicator metadata: pillar, direction, weights, layer, inclusion/risk tags.

            **Step 2 — Missing-data treatment**
            - Within-series interpolation/forward-backward fill where possible.
            - Cross-sectional indicator-year fallback medians.
            - Global indicator fallback median as last resort.
            - Track `was_imputed` flags for every record.

            **Step 3 — Orientation and normalization**
            - Reorient negative indicators (higher should always mean better after orientation).
            - Apply robust min-max normalization to 0-100.

            **Step 4 — Pillar and V2 layer scores**
            - Compute weighted pillar scores by economy-year.
            - Compute V2 readiness/adoption/impact layer means.

            **Step 5 — Composite and adjustments**
            - Compute geometric composite score (penalizes imbalance across layers).
            - Compute inclusion-adjusted and risk-adjusted variants.
            - Compute utilization efficiency and confidence score.

            **Step 6 — Comparative diagnostics**
            - Produce trends, rank tables, and policy signal narratives.
            - Produce Doughnut social/ecological diagnostics and gap views.
            """
        )

        st.subheader("Policy Interpretation Guardrails")
        st.markdown(
            """
            - Compare score changes with confidence tier before taking major policy action.
            - Treat small rank differences as non-material unless sustained across years.
            - Combine quantitative signals with institutional and implementation context.
            - Use indicators as a steering system for prioritization, not as a final impact claim.
            """
        )


if __name__ == "__main__":
    main()
