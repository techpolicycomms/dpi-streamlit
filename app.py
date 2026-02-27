from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_csv, outputs_dir, resolve_legacy_root


st.set_page_config(page_title="DPI Streamlit Parallel", page_icon="📊", layout="wide")


@st.cache_data
def load_all() -> dict[str, pd.DataFrame]:
    return {
        "v2_all": load_csv("dpi_v2_scores_all_years"),
        "v2_latest": load_csv("dpi_v2_scores_latest_year"),
        "doughnut": load_csv("doughnut_pillar_scores"),
        "trends": load_csv("dpi_ready_trends"),
        "corr": load_csv("indicator_correlation_report"),
        "coverage": load_csv("source_coverage_summary"),
    }


def latest_for_economy(df: pd.DataFrame, economy: str) -> pd.Series | None:
    part = df[df["economy"] == economy].copy()
    if part.empty:
        return None
    part["year"] = pd.to_numeric(part["year"], errors="coerce")
    return part.sort_values("year").iloc[-1]


def main() -> None:
    st.title("DPI Dashboard (Parallel Streamlit Migration)")
    st.caption("Parity-first Python dashboard using the same output artifacts as the Shiny app.")

    st.info(f"Legacy source: `{resolve_legacy_root()}` | Outputs: `{outputs_dir()}`")

    try:
        data = load_all()
    except Exception as exc:
        st.error(f"Failed to load legacy outputs: {exc}")
        st.stop()

    v2_all = data["v2_all"]
    v2_latest = data["v2_latest"]
    doughnut = data["doughnut"]
    trends = data["trends"]
    corr = data["corr"]
    coverage = data["coverage"]

    economies = sorted(v2_all["economy"].dropna().astype(str).unique().tolist())
    default_economy = "India" if "India" in economies else economies[0]

    with st.sidebar:
        economy = st.selectbox("Economy", economies, index=economies.index(default_economy))
        year_min = int(pd.to_numeric(v2_all["year"], errors="coerce").min())
        year_max = int(pd.to_numeric(v2_all["year"], errors="coerce").max())
        year_range = st.slider("Year Range", year_min, year_max, (max(year_min, 2014), year_max))

    v2_sel = v2_all[(v2_all["economy"] == economy)].copy()
    v2_sel["year"] = pd.to_numeric(v2_sel["year"], errors="coerce")
    v2_sel = v2_sel[(v2_sel["year"] >= year_range[0]) & (v2_sel["year"] <= year_range[1])]

    doughnut_sel = doughnut[(doughnut["economy"] == economy)].copy()
    doughnut_sel["year"] = pd.to_numeric(doughnut_sel["year"], errors="coerce")
    doughnut_sel = doughnut_sel[(doughnut_sel["year"] >= year_range[0]) & (doughnut_sel["year"] <= year_range[1])]

    latest = latest_for_economy(v2_latest, economy)
    if latest is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Composite v2", f"{float(latest.get('dpi_composite_v2', float('nan'))):.2f}")
        c2.metric("Inclusion-adjusted", f"{float(latest.get('dpi_inclusion_adjusted_v2', float('nan'))):.2f}")
        c3.metric("Risk-adjusted", f"{float(latest.get('dpi_risk_adjusted_v2', float('nan'))):.2f}")
        conf = latest.get("dpi_confidence_score")
        c4.metric("Confidence", f"{float(conf):.1f}" if pd.notna(conf) else "N/A")

    tab1, tab2, tab3, tab4 = st.tabs(["Capability Layers", "Doughnut", "Trends", "Diagnostics"])

    with tab1:
        st.subheader("Readiness vs Adoption vs Impact")
        plot_cols = [c for c in ["dpi_readiness_v2", "dpi_adoption_v2", "dpi_impact_v2"] if c in v2_sel.columns]
        if plot_cols and not v2_sel.empty:
            long_df = v2_sel.melt(id_vars=["year"], value_vars=plot_cols, var_name="metric", value_name="score")
            fig = px.line(long_df, x="year", y="score", color="metric", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(v2_sel.sort_values("year", ascending=False), use_container_width=True)

    with tab2:
        st.subheader("Doughnut Scores")
        if not doughnut_sel.empty:
            plot_cols = [c for c in ["social_foundation_score", "ecological_ceiling_score", "doughnut_score"] if c in doughnut_sel.columns]
            long_df = doughnut_sel.melt(id_vars=["year"], value_vars=plot_cols, var_name="metric", value_name="score")
            fig = px.line(long_df, x="year", y="score", color="metric", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(doughnut_sel.sort_values("year", ascending=False), use_container_width=True)

    with tab3:
        st.subheader("Legacy Trend Output")
        trend_sel = trends[trends["economy"] == economy].copy() if "economy" in trends.columns else pd.DataFrame()
        if not trend_sel.empty and "dpi_score" in trend_sel.columns:
            fig = px.line(trend_sel.sort_values("year"), x="year", y="dpi_score", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(trend_sel.sort_values("year", ascending=False), use_container_width=True)

    with tab4:
        st.subheader("Indicator Correlations")
        if not corr.empty:
            corr["correlation"] = pd.to_numeric(corr.get("correlation"), errors="coerce")
            corr["abs_correlation"] = corr["correlation"].abs()
            top = corr.sort_values("abs_correlation", ascending=False).head(20)
            st.dataframe(top, use_container_width=True)
        st.subheader("Source Coverage")
        st.dataframe(coverage, use_container_width=True)


if __name__ == "__main__":
    main()
