"""
dashboard.py
Streamlit interactive dashboard for Teiko technical analysis.

Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from analysis import (
    get_frequency_table,
    get_timepoint_data,
    run_statistical_tests,
    run_timepoint_sensitivity,
    get_longitudinal_data,
    run_mixed_effects_models,
    run_mixed_effects_models_clr,
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

@st.cache_data
def load_longitudinal_data():
    return get_longitudinal_data()

@st.cache_data
def load_lmm_results():
    return run_mixed_effects_models(get_longitudinal_data())

@st.cache_data
def load_lmm_results_clr():
    return run_mixed_effects_models_clr(get_longitudinal_data())


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
    st.header("Responder Analysis: Immune Cell Frequencies and Treatment Response")
    st.markdown(
        "**Cohort:** melanoma patients, miraclib treatment, PBMC samples "
        "(n = 656: 331 responders, 325 non-responders). "
        "**Test:** Mann-Whitney U, two-sided. "
        "**Correction:** Benjamini-Yekutieli (BY) FDR - valid under any correlation structure, "
        "including the negative correlations that arise because cell percentages sum to 100%. "
        "**Effect size:** rank-biserial r (positive = responders rank higher; "
        "|r| < 0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium, >0.5 large)."
    )

    COLORS = {"yes": "#4C9BE8", "no": "#E87E4C"}
    LABELS = {"yes": "Responder", "no": "Non-responder"}

    t0_df = get_timepoint_data(0)
    test_results = run_statistical_tests(t0_df)

    # ================================================================
    # Section 1: Before treatment
    # ================================================================
    st.subheader("Before Treatment: Can Baseline Frequencies Predict Who Will Respond?")
    st.markdown(
        "Only measurements taken *before* treatment begins can inform response prediction. "
        "The table below tests whether t=0 cell frequencies differ between groups. "
        "Use the timepoint selector to also view distributions at t=7 and t=14."
    )

    # Stats table fixed to t=0
    disp = test_results[[
        "population", "n_responders", "n_non_responders",
        "p_value", "p_by_fdr", "sig_by_fdr", "effect_size_r",
    ]].copy()
    disp["sig_by_fdr"] = disp["sig_by_fdr"].map({True: "Yes", False: "No"})
    disp.columns = [
        "Population", "n (resp.)", "n (non-resp.)",
        "p (raw)", "p (BY FDR)", "Significant?", "Effect size r",
    ]

    def _hl_sig(row):
        if row["Significant?"] == "Yes":
            return ["background-color: #d4edda"] * len(row)
        return [""] * len(row)

    st.dataframe(disp.style.apply(_hl_sig, axis=1), use_container_width=False, hide_index=True)

    st.warning(
        "**No cell population distinguishes responders from non-responders at baseline.** "
        "All raw p-values exceed 0.20 and all effect sizes are negligible (|r| < 0.06). "
        "There is no pre-treatment immune cell frequency signature predicting miraclib response "
        "in this melanoma cohort."
    )

    # Boxplots with timepoint radio
    selected_t = st.radio(
        "View distributions at timepoint:",
        options=[0, 7, 14],
        format_func=lambda t: f"t={t} (pre-treatment, prediction-relevant)" if t == 0 else f"t={t}",
        horizontal=True,
    )

    if selected_t == 0:
        plot_df    = t0_df
        plot_stats = test_results
    else:
        plot_df    = get_timepoint_data(selected_t)
        plot_stats = run_statistical_tests(plot_df)

    p_raw_map = dict(zip(plot_stats["population"], plot_stats["p_value"]))
    p_by_map  = dict(zip(plot_stats["population"], plot_stats["p_by_fdr"]))
    r_map     = dict(zip(plot_stats["population"], plot_stats["effect_size_r"]))

    bcols = st.columns(len(CELL_POPULATIONS))
    for col_ui, pop in zip(bcols, CELL_POPULATIONS):
        pct_col = f"{pop}_pct"
        fig = go.Figure()
        for response_val in ["yes", "no"]:
            subset = plot_df[plot_df["response"] == response_val][pct_col].dropna()
            fig.add_trace(go.Box(
                y=subset,
                name=LABELS[response_val],
                marker_color=COLORS[response_val],
                boxmean=True,
                boxpoints="outliers",
                pointpos=0,
            ))
        p_raw = p_raw_map[pop]
        p_by  = p_by_map[pop]
        r     = r_map[pop]
        fig.update_layout(
            title=dict(
                text=(
                    f"<b>{pop.replace('_', ' ').title()}</b><br>"
                    f"<sub>p={p_raw:.3f} | BY={p_by:.3f} | r={r:+.3f}</sub>"
                ),
                x=0.5,
            ),
            yaxis_title="Frequency (%)",
            showlegend=False,
            height=400,
            margin=dict(l=10, r=10, t=70, b=10),
        )
        col_ui.plotly_chart(fig, use_container_width=True)

    if selected_t != 0:
        st.caption(
            f"Note: signals at t={selected_t} reflect treatment-induced changes, "
            "not pre-treatment predictors. A genuine predictor must be detectable at t=0."
        )

    st.download_button(
        label=f"Download t={selected_t} data (CSV)",
        data=plot_df.to_csv(index=False),
        file_name=f"melanoma_miraclib_pbmc_t{selected_t}.csv",
        mime="text/csv",
    )

    # ================================================================
    # Section 2: Over time
    # ================================================================
    st.divider()
    st.subheader("During Treatment: Do Cell Populations Change Differently Over Time?")
    st.markdown(
        "Even without a baseline difference, immune cell populations may evolve differently "
        "in responders vs non-responders as treatment progresses. "
        "The line plots below show mean frequency (+/- 95% CI) at t=0, t=7, and t=14. "
        "The mixed effects model formally tests whether the two groups' trajectories diverge, "
        "accounting for the fact that the same subjects appear at all three timepoints."
    )

    long_df = load_longitudinal_data()
    TIMEPOINTS = [0, 7, 14]
    TRAJ_COLORS = {"yes": "#4C9BE8", "no": "#E87E4C"}
    TRAJ_LABELS = {"yes": "Responder", "no": "Non-responder"}

    traj_fig = make_subplots(
        rows=1, cols=len(CELL_POPULATIONS),
        subplot_titles=[p.replace("_", " ").title() for p in CELL_POPULATIONS],
        shared_yaxes=False,
    )
    for i, pop in enumerate(CELL_POPULATIONS):
        col_name = f"{pop}_pct"
        for response_val in ["yes", "no"]:
            grp = long_df[long_df["response"] == response_val].groupby("time_from_treatment_start")[col_name]
            means = grp.mean().reindex(TIMEPOINTS)
            stds  = grp.std().reindex(TIMEPOINTS)
            ns    = grp.count().reindex(TIMEPOINTS)
            ci    = 1.96 * stds / np.sqrt(ns)
            traj_fig.add_trace(
                go.Scatter(
                    x=TIMEPOINTS,
                    y=means.tolist(),
                    error_y=dict(type="data", array=ci.tolist(), visible=True, thickness=1.5, width=4),
                    mode="lines+markers",
                    name=TRAJ_LABELS[response_val],
                    legendgroup=response_val,
                    showlegend=(i == 0),
                    line=dict(color=TRAJ_COLORS[response_val], width=2),
                    marker=dict(color=TRAJ_COLORS[response_val], size=7),
                ),
                row=1, col=i + 1,
            )
        traj_fig.update_xaxes(tickvals=TIMEPOINTS, title_text="Days", row=1, col=i + 1)
        if i == 0:
            traj_fig.update_yaxes(title_text="Mean frequency (%)", row=1, col=1)

    traj_fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
    )
    st.plotly_chart(traj_fig, use_container_width=True)
    st.caption("Error bars = 95% CI (mean +/- 1.96 x SE). Blue = Responders, Orange = Non-responders.")

    # LMM table
    st.markdown("**Mixed effects model: do trajectories diverge? (time x response interaction)**")
    st.caption(
        "Model per population: frequency ~ time * response + (1 | subject). "
        "The interaction p-value tests whether the rate of change over time differs between groups. "
        "Green = p < 0.05."
    )

    lmm_df = load_lmm_results()

    lmm_display = lmm_df[[
        "population", "p_time", "p_response", "p_interaction", "p_interaction_by_fdr", "sig_interaction_by_fdr"
    ]].rename(columns={
        "population":              "Population",
        "p_time":                  "p (time trend)",
        "p_response":              "p (group diff. at t=0)",
        "p_interaction":           "p (time x response, raw)",
        "p_interaction_by_fdr":    "p (time x response, BY FDR)",
        "sig_interaction_by_fdr":  "Significant?",
    })
    lmm_display["Significant?"] = lmm_display["Significant?"].map({True: "Yes", False: "No"})

    def _highlight_interaction(row):
        if row["Significant?"] == "Yes":
            return ["background-color: #d4edda; color: #155724; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        lmm_display.style.apply(_highlight_interaction, axis=1),
        use_container_width=False,
        hide_index=True,
    )

    bcell_row = lmm_df[lmm_df["population"] == "b_cell"].iloc[0]
    bcell_raw = bcell_row["p_interaction"]
    bcell_fdr = bcell_row["p_interaction_by_fdr"]
    any_significant = lmm_df["sig_interaction_by_fdr"].any()

    if any_significant:
        sig_pops = lmm_df[lmm_df["sig_interaction_by_fdr"]]["population"].tolist()
        st.success(
            f"**Significant trajectory divergence after BY FDR correction: {', '.join(sig_pops)}.** "
            "See table above for details."
        )
    else:
        st.warning(
            "**No population shows a significant trajectory difference after BY FDR correction.** "
            f"B cells have the smallest raw interaction p-value ({bcell_raw:.4f}), and their "
            "trajectories visually diverge in the plot above, but this does not survive correction "
            f"for 5 simultaneous tests (BY FDR p = {bcell_fdr:.4f}). "
            "The pattern is a candidate for follow-up but cannot be claimed as a statistically "
            "significant finding with this dataset."
        )

    with st.expander("Sensitivity analysis: CLR-transformed frequencies"):
        st.markdown(
            "Cell population frequencies are compositional - they sum to 100% by construction, "
            "which introduces artificial negative correlations between populations. "
            "The centered log-ratio (CLR) transform removes this constraint: "
            "`CLR(x) = log(x) - mean(log(x))` computed per sample across all five populations. "
            "The same mixed effects model is then fit on CLR-transformed values. "
            "No zeros exist in this dataset so no pseudocount is needed."
        )

        clr_df = load_lmm_results_clr()

        clr_display = clr_df[[
            "population", "p_interaction", "p_interaction_by_fdr", "sig_interaction_by_fdr"
        ]].rename(columns={
            "population":             "Population",
            "p_interaction":          "p (time x response, raw)",
            "p_interaction_by_fdr":   "p (time x response, BY FDR)",
            "sig_interaction_by_fdr": "Significant?",
        })
        clr_display["Significant?"] = clr_display["Significant?"].map({True: "Yes", False: "No"})

        def _hl_clr(row):
            if row["Significant?"] == "Yes":
                return ["background-color: #d4edda; color: #155724; font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(
            clr_display.style.apply(_hl_clr, axis=1),
            use_container_width=False,
            hide_index=True,
        )

        clr_bcell = clr_df[clr_df["population"] == "b_cell"].iloc[0]
        st.caption(
            f"CLR result for b_cell: raw p = {clr_bcell['p_interaction']:.4f}, "
            f"BY FDR p = {clr_bcell['p_interaction_by_fdr']:.4f}. "
            "Consistent with the primary analysis - the null conclusion holds on the CLR scale."
        )

    # Per-timepoint detail (collapsed)
    with st.expander("Per-timepoint effect sizes and p-values"):
        st.caption(
            "Breakdown of effect size r and p-values at each individual timepoint. "
            "A genuine predictor shows a signal at t=0. "
            "Signals appearing only at t=7 or t=14 are treatment effects, not predictors."
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

        st.caption("Green = p < 0.05 (uncorrected). No population reaches BY FDR significance at any timepoint.")


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
