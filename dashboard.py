"""
dashboard.py
Streamlit interactive dashboard for Teiko technical analysis.

Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from analysis import (
    get_frequency_table,
    get_timepoint_data,
    run_statistical_tests,
    run_timepoint_sensitivity,
    get_baseline_samples,
    get_samples_per_project,
    get_responder_counts,
    get_sex_counts,
    get_avg_bcell_melanoma_males,
    CELL_POPULATIONS,
)

st.set_page_config(
    page_title="Teiko: Immune Cell Analysis",
    page_icon="🧬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_freq_table():
    return get_frequency_table()

@st.cache_data
def load_baseline():
    return get_baseline_samples()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🧬 Immune Cell Population Analysis")
st.caption("Clinical trial data | Teiko technical")
st.divider()

tab1, tab2, tab3 = st.tabs(
    ["📊 Part 2 - Cell Frequencies", "📈 Part 3 - Responder Analysis", "🔍 Part 4 - Subset Queries"]
)


# ---------------------------------------------------------------------------
# Tab 1: Frequency Table
# ---------------------------------------------------------------------------

with tab1:
    st.header("Cell Population Frequencies per Sample")
    st.markdown(
        "Relative frequency of each immune cell population as a percentage of total cell count per sample."
    )

    freq_df = load_freq_table()

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        sample_filter = st.text_input("Filter by sample ID (prefix match)", placeholder="e.g. sample00000")
    with col2:
        pop_filter = st.multiselect(
            "Filter by population",
            options=CELL_POPULATIONS,
            default=CELL_POPULATIONS,
        )

    filtered = freq_df[freq_df["population"].isin(pop_filter)]
    if sample_filter:
        filtered = filtered[filtered["sample"].str.startswith(sample_filter)]

    st.dataframe(
        filtered,
        use_container_width=True,
        height=500,
        column_config={
            "percentage": st.column_config.NumberColumn("percentage", format="%.4f"),
            "count": st.column_config.NumberColumn("count", format="%d"),
            "total_count": st.column_config.NumberColumn("total_count", format="%d"),
        },
    )

    st.caption(f"Showing {len(filtered):,} rows of {len(freq_df):,} total")

    # Summary stats
    st.subheader("Average frequency (%) across all samples")
    avg_pct = (
        freq_df.groupby("population")["percentage"]
        .agg(["mean", "median", "std"])
        .round(2)
        .reset_index()
        .rename(columns={"mean": "Mean %", "median": "Median %", "std": "Std Dev"})
    )
    st.dataframe(avg_pct, use_container_width=False)


# ---------------------------------------------------------------------------
# Tab 2: Statistical Analysis
# ---------------------------------------------------------------------------

with tab2:
    st.header("Responders vs Non-Responders: Baseline (t=0)")
    st.markdown(
        """
        **Goal:** identify immune cell populations that predict treatment response *before* treatment starts.

        - **Filter:** melanoma, miraclib, PBMC, t=0 (baseline) only
        - **Unit of analysis:** one observation per subject (single pre-treatment sample)
        - **Test:** Mann-Whitney U, two-sided, non-parametric
        - **Multiple testing correction:** BY FDR across 5 populations (valid under any correlation structure; BH FDR is anti-conservative here due to compositional negative correlations between populations)
        - **Effect size:** rank-biserial correlation r (r > 0 = responders rank higher; r < 0 = non-responders rank higher; |r| < 0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium, >0.5 large)
        """
    )

    # Primary data: t=0 only — the only timepoint causally prior to treatment
    t0_df = get_timepoint_data(0)
    test_results = run_statistical_tests(t0_df)

    # --- Significance table ---
    st.subheader("Baseline Statistical Test Results")

    def sig_flag(v):
        return "✅ Yes" if v else "❌ No"

    display_results = test_results[[
        "population", "n_responders", "n_non_responders",
        "p_value", "p_bonferroni", "p_by_fdr",
        "sig_uncorrected", "sig_bonferroni", "sig_by_fdr", "effect_size_r"
    ]].copy()
    for col in ["sig_uncorrected", "sig_bonferroni", "sig_by_fdr"]:
        display_results[col] = display_results[col].map(sig_flag)
    display_results.columns = [
        "Population", "n (resp.)", "n (non-resp.)",
        "p (raw)", "p (Bonferroni)", "p (BY FDR)",
        "Sig. raw", "Sig. Bonferroni", "Sig. BY FDR", "Effect size (r)"
    ]
    st.dataframe(display_results, use_container_width=True, hide_index=True)

    st.info(
        "**Why BY FDR?** Cell population frequencies are compositional (sum to 100%), "
        "inducing negative correlations between populations. BH FDR assumes independence or "
        "positive correlation and can be anti-conservative here. "
        "BY FDR is valid under any dependence structure and is the primary correction used."
    )

    st.success(
        "**Primary conclusion:** At baseline, no cell population significantly differentiates "
        "responders from non-responders. All raw p-values exceed 0.20 and all effect sizes are "
        "negligible (|r| < 0.06). There is no evidence of a pre-existing immune cell frequency "
        "signature that predicts miraclib response in this melanoma cohort."
    )

    # --- Boxplots (t=0 data) ---
    st.subheader("Baseline Distributions by Cell Population")
    st.caption(
        "Boxplots show t=0 (pre-treatment) measurements only. "
        "The instructions do not specify a timepoint for Part 3, but the stated aim is response prediction — "
        "only pre-treatment frequencies can inform a decision made before dosing."
    )

    COLORS = {"yes": "#4C9BE8", "no": "#E87E4C"}
    LABELS = {"yes": "Responder", "no": "Non-responder"}

    p_raw_map = dict(zip(test_results["population"], test_results["p_value"]))
    p_by_map  = dict(zip(test_results["population"], test_results["p_by_fdr"]))
    r_map     = dict(zip(test_results["population"], test_results["effect_size_r"]))

    cols = st.columns(len(CELL_POPULATIONS))
    for col_ui, pop in zip(cols, CELL_POPULATIONS):
        pct_col = f"{pop}_pct"
        fig = go.Figure()

        for response_val in ["yes", "no"]:
            subset = t0_df[t0_df["response"] == response_val][pct_col].dropna()
            fig.add_trace(
                go.Box(
                    y=subset,
                    name=LABELS[response_val],
                    marker_color=COLORS[response_val],
                    boxmean=True,
                    boxpoints="outliers",
                    pointpos=0,
                )
            )

        p_raw = p_raw_map[pop]
        p_by  = p_by_map[pop]
        r     = r_map[pop]
        sig   = "✱ (raw only)" if p_raw < 0.05 and p_by >= 0.05 else ("✱✱" if p_by < 0.05 else "ns")

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>{pop.replace('_', ' ').title()}</b><br>"
                    f"<sub>p={p_raw:.4f} | BY p={p_by:.4f} | r={r:+.3f} {sig}</sub>"
                ),
                x=0.5,
            ),
            yaxis_title="Frequency (%)",
            showlegend=False,
            height=420,
            margin=dict(l=10, r=10, t=70, b=10),
        )
        col_ui.plotly_chart(fig, use_container_width=True)

    st.download_button(
        label="⬇️ Download baseline data (CSV)",
        data=t0_df.to_csv(index=False),
        file_name="melanoma_miraclib_pbmc_t0.csv",
        mime="text/csv",
    )

    # --- Temporal context ---
    st.divider()
    st.subheader("Temporal Context: Effect Sizes Across Timepoints")
    st.markdown(
        """
        A genuine response predictor should show a signal **at t=0**, before treatment begins.
        Signals appearing only at t=7 or t=14 reflect treatment-induced changes, not predictive biology.
        """
    )

    sensitivity_df = run_timepoint_sensitivity()
    TIMEPOINT_ORDER = [c for c in ["t=0", "t=7", "t=14"]
                       if c in sensitivity_df["timepoint"].unique()]

    def highlight_pval(val):
        if isinstance(val, float) and val < 0.05:
            return "background-color: #d4edda; color: #155724; font-weight: bold"
        return ""

    for metric, label, do_highlight in [
        ("effect_size_r", "Effect size r (rank-biserial correlation)", False),
        ("p_value",       "Raw p-value",                               True),
        ("p_by_fdr",      "BY FDR p-value",                            True),
    ]:
        pivot = (
            sensitivity_df.pivot(index="population", columns="timepoint", values=metric)
            .reindex(columns=TIMEPOINT_ORDER)
            .round(4)
        )
        st.markdown(f"**{label}**")
        styled = pivot.style.applymap(highlight_pval) if do_highlight else pivot
        st.dataframe(styled, use_container_width=False)

    st.caption("Green cells = p < 0.05 (uncorrected). No population reaches significance after BY FDR correction at any timepoint.")

    st.info(
        "**Key insight:** cd4_t_cell shows essentially no baseline effect (r = +0.01 at t=0, p = 0.796), "
        "but a modest positive signal emerges post-treatment (r = +0.098 at t=7, raw p = 0.030; r = +0.080 at t=14). "
        "This temporal pattern indicates cd4_t_cell is a **marker of treatment response** "
        "(responders' CD4 T cell frequencies rise during treatment) rather than a predictor of who will respond. "
        "Neither the t=7 nor t=14 signal survives BY FDR correction. "
        "A true predictor must be measurable before treatment begins."
    )


# ---------------------------------------------------------------------------
# Tab 3: Subset Queries
# ---------------------------------------------------------------------------

with tab3:
    st.header("Melanoma Baseline Subset Analysis")
    st.markdown(
        """
        **Filter applied:** condition = melanoma · sample_type = PBMC · time_from_treatment_start = 0 · treatment = miraclib
        """
    )

    baseline_df = load_baseline()

    st.metric("Total baseline samples", len(baseline_df))

    st.divider()

    col1, col2, col3 = st.columns(3)

    # Samples per project
    with col1:
        st.subheader("Samples per Project")
        proj_counts = get_samples_per_project(baseline_df)
        fig_proj = px.bar(
            proj_counts,
            x="project",
            y="sample_count",
            color="project",
            text="sample_count",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_proj.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
        fig_proj.update_traces(textposition="outside")
        st.plotly_chart(fig_proj, use_container_width=True)
        st.dataframe(proj_counts, hide_index=True, use_container_width=True)

    # Responder / non-responder counts
    with col2:
        st.subheader("Subjects by Response")
        resp_counts = get_responder_counts(baseline_df)
        resp_counts["label"] = resp_counts["response"].map({"yes": "Responder", "no": "Non-responder"})
        fig_resp = px.pie(
            resp_counts,
            names="label",
            values="subjects",
            color="label",
            color_discrete_map={"Responder": "#4C9BE8", "Non-responder": "#E87E4C"},
            hole=0.4,
        )
        fig_resp.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_resp, use_container_width=True)
        st.dataframe(resp_counts[["label", "subjects"]].rename(columns={"label": "response"}), hide_index=True, use_container_width=True)

    # Male / female counts
    with col3:
        st.subheader("Subjects by Sex")
        sex_counts = get_sex_counts(baseline_df)
        sex_counts["label"] = sex_counts["sex"].map({"M": "Male", "F": "Female"})
        fig_sex = px.pie(
            sex_counts,
            names="label",
            values="subjects",
            color="label",
            color_discrete_map={"Male": "#6abf69", "Female": "#f48fb1"},
            hole=0.4,
        )
        fig_sex.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_sex, use_container_width=True)
        st.dataframe(sex_counts[["label", "subjects"]].rename(columns={"label": "sex"}), hide_index=True, use_container_width=True)

    st.divider()

    st.subheader("Average B Cell Count")
    st.markdown("**Melanoma males · responders · time = 0**")
    avg_bcell = get_avg_bcell_melanoma_males(baseline_df)

    st.metric(
        label="Avg B cells (melanoma males, responders, baseline)",
        value=f"{avg_bcell:.2f}",
    )

    # Raw baseline table
    with st.expander("View raw baseline sample data"):
        st.dataframe(baseline_df, use_container_width=True, height=400)
        st.download_button(
            label="⬇️ Download baseline samples (CSV)",
            data=baseline_df.to_csv(index=False),
            file_name="melanoma_pbmc_baseline_miraclib.csv",
            mime="text/csv",
        )
