"""
analysis.py
Analysis logic for Parts 2-4 of the Teiko technical.

Functions:
  get_frequency_table()   -> Part 2: per-sample cell population frequencies
  get_stats_data()        -> Part 3: filtered data for responder comparison
  run_statistical_tests() -> Part 3: Mann-Whitney U per population
  get_boxplot_figure()    -> Part 3: matplotlib figure of boxplots
  get_baseline_samples()  -> Part 4: melanoma PBMC baseline miraclib samples
  get_samples_per_project()     -> Part 4: count per project
  get_responder_counts()        -> Part 4: responder/non-responder counts
  get_sex_counts()              -> Part 4: male/female counts
  get_avg_bcell_melanoma_males() -> Part 4: avg B cell count query
"""

import sqlite3
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DB_PATH = "teiko.db"
CELL_POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Part 2: Frequency table
# ---------------------------------------------------------------------------

def get_frequency_table() -> pd.DataFrame:
    """
    Returns a long-format DataFrame with one row per (sample, population):
      sample, total_count, population, count, percentage
    """
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT sample_id, b_cell, cd8_t_cell, cd4_t_cell, nk_cell, monocyte FROM samples",
        conn,
    )
    conn.close()

    df["total_count"] = df[CELL_POPULATIONS].sum(axis=1)

    rows = []
    for _, row in df.iterrows():
        for pop in CELL_POPULATIONS:
            rows.append(
                {
                    "sample": row["sample_id"],
                    "total_count": int(row["total_count"]),
                    "population": pop,
                    "count": int(row[pop]),
                    "percentage": round(row[pop] / row["total_count"] * 100, 4),
                }
            )

    return pd.DataFrame(rows, columns=["sample", "total_count", "population", "count", "percentage"])


# ---------------------------------------------------------------------------
# Part 3: Statistical analysis
# ---------------------------------------------------------------------------

def get_stats_data() -> pd.DataFrame:
    """
    Returns one row per subject (aggregated across timepoints) for:
      condition=melanoma, treatment=miraclib, sample_type=PBMC

    Each subject's per-sample cell percentages are averaged across all
    their timepoints before comparison. This gives one independent
    observation per subject, satisfying the independence assumption
    required by Mann-Whitney U.

    Columns: subject_id, response, {pop}_pct (mean % per population)
    """
    conn = _get_conn()
    df = pd.read_sql_query(
        """
        SELECT
            sub.subject_id,
            s.response,
            s.b_cell,
            s.cd8_t_cell,
            s.cd4_t_cell,
            s.nk_cell,
            s.monocyte
        FROM samples s
        JOIN subjects sub ON s.subject_id = sub.subject_id
        WHERE sub.condition = 'melanoma'
          AND s.treatment   = 'miraclib'
          AND s.sample_type = 'PBMC'
          AND s.response IN ('yes', 'no')
        """,
        conn,
    )
    conn.close()

    # Compute per-sample percentages first
    df["total_count"] = df[CELL_POPULATIONS].sum(axis=1)
    for pop in CELL_POPULATIONS:
        df[f"{pop}_pct"] = df[pop] / df["total_count"] * 100

    # Aggregate to one row per subject by averaging across timepoints
    pct_cols = [f"{pop}_pct" for pop in CELL_POPULATIONS]
    subject_df = (
        df.groupby(["subject_id", "response"])[pct_cols]
        .mean()
        .reset_index()
    )

    return subject_df


def run_statistical_tests(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs Mann-Whitney U test for each cell population comparing
    responders (response='yes') vs non-responders (response='no').

    Applies Bonferroni and Benjamini-Hochberg FDR corrections for
    multiple testing across the 5 populations.

    Also computes rank-biserial correlation as effect size.

    Returns a DataFrame with columns:
      population, p_value, p_bonferroni, p_bh_fdr,
      sig_uncorrected, sig_bonferroni, sig_bh_fdr,
      effect_size_r, n_responders, n_non_responders
    """
    responders = stats_df[stats_df["response"] == "yes"]
    non_responders = stats_df[stats_df["response"] == "no"]

    rows = []
    raw_pvals = []
    for pop in CELL_POPULATIONS:
        col = f"{pop}_pct"
        r_vals = responders[col].dropna()
        nr_vals = non_responders[col].dropna()
        n1_pop, n2_pop = len(r_vals), len(nr_vals)

        U, p = mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
        r_rb = (2 * U) / (n1_pop * n2_pop) - 1

        raw_pvals.append(p)
        rows.append(
            {
                "population": pop,
                "p_value": round(p, 6),
                "effect_size_r": round(r_rb, 4),
                "n_responders": n1_pop,
                "n_non_responders": n2_pop,
            }
        )

    result_df = pd.DataFrame(rows)
    pvals = raw_pvals

    # Bonferroni (FWER, no assumption on correlation structure)
    _, pvals_bonf, _, _ = multipletests(pvals, alpha=0.05, method="bonferroni")
    result_df["p_bonferroni"] = pvals_bonf.round(6)
    result_df["sig_uncorrected"] = [p < 0.05 for p in pvals]
    result_df["sig_bonferroni"] = pvals_bonf < 0.05

    # Benjamini-Hochberg FDR (assumes independence or positive correlation)
    # NOTE: cell percentages are compositional (sum to 100%) so they are
    # negatively correlated — BH can be anti-conservative here.
    reject_bh, pvals_bh, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    result_df["p_bh_fdr"] = pvals_bh.round(6)
    result_df["sig_bh_fdr"] = reject_bh

    # Benjamini-Yekutieli FDR (valid under any correlation structure,
    # including negative correlations from compositional data)
    reject_by, pvals_by, _, _ = multipletests(pvals, alpha=0.05, method="fdr_by")
    result_df["p_by_fdr"] = pvals_by.round(6)
    result_df["sig_by_fdr"] = reject_by

    return result_df


def get_timepoint_data(timepoint: int) -> pd.DataFrame:
    """
    Returns per-sample frequency data for a single timepoint, filtered to
    melanoma + miraclib + PBMC + response in (yes, no).
    One row per subject (since each subject has one sample per timepoint).
    """
    conn = _get_conn()
    df = pd.read_sql_query(
        """
        SELECT
            sub.subject_id,
            s.response,
            s.time_from_treatment_start,
            s.b_cell,
            s.cd8_t_cell,
            s.cd4_t_cell,
            s.nk_cell,
            s.monocyte
        FROM samples s
        JOIN subjects sub ON s.subject_id = sub.subject_id
        WHERE sub.condition = 'melanoma'
          AND s.treatment   = 'miraclib'
          AND s.sample_type = 'PBMC'
          AND s.response IN ('yes', 'no')
          AND s.time_from_treatment_start = ?
        """,
        conn,
        params=(timepoint,),
    )
    conn.close()

    df["total_count"] = df[CELL_POPULATIONS].sum(axis=1)
    for pop in CELL_POPULATIONS:
        df[f"{pop}_pct"] = df[pop] / df["total_count"] * 100

    return df


def run_timepoint_sensitivity() -> pd.DataFrame:
    """
    Runs Mann-Whitney U + BY FDR at each individual timepoint (0, 7, 14).

    Returns a long-format DataFrame with columns:
      timepoint, population, p_value, p_by_fdr, sig_by_fdr, effect_size_r,
      n_responders, n_non_responders
    """
    conn = _get_conn()
    timepoints_available = pd.read_sql_query(
        """
        SELECT DISTINCT s.time_from_treatment_start
        FROM samples s
        JOIN subjects sub ON s.subject_id = sub.subject_id
        WHERE sub.condition = 'melanoma'
          AND s.treatment   = 'miraclib'
          AND s.sample_type = 'PBMC'
          AND s.response IN ('yes', 'no')
        ORDER BY s.time_from_treatment_start
        """,
        conn,
    )["time_from_treatment_start"].tolist()
    conn.close()

    results = []
    for t in timepoints_available:
        df = get_timepoint_data(t)
        _append_test_rows(results, df, label=f"t={t}")

    return pd.DataFrame(results)


def _append_test_rows(results: list, df: pd.DataFrame, label: str) -> None:
    """Helper: run Mann-Whitney U + BY FDR on df and append rows to results."""
    resp  = df[df["response"] == "yes"]
    nresp = df[df["response"] == "no"]

    raw_pvals = []
    row_data  = []
    for pop in CELL_POPULATIONS:
        col = f"{pop}_pct"
        r_vals  = resp[col].dropna()
        nr_vals = nresp[col].dropna()
        n1_pop, n2_pop = len(r_vals), len(nr_vals)
        U, p = mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
        r_rb = (2 * U) / (n1_pop * n2_pop) - 1
        raw_pvals.append(p)
        row_data.append({"population": pop, "p_value": round(p, 6),
                         "effect_size_r": round(r_rb, 4),
                         "n_responders": n1_pop, "n_non_responders": n2_pop})

    _, pvals_by, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_by")

    for row, p_by in zip(row_data, pvals_by):
        row["timepoint"]   = label
        row["p_by_fdr"]    = round(p_by, 6)
        row["sig_by_fdr"]  = p_by < 0.05
        results.append(row)


def get_boxplot_figure(stats_df: pd.DataFrame, test_results: pd.DataFrame) -> plt.Figure:
    """
    Returns a matplotlib Figure with one boxplot per cell population,
    comparing responders vs non-responders. Annotates p-values.
    """
    n_pops = len(CELL_POPULATIONS)
    fig, axes = plt.subplots(1, n_pops, figsize=(4 * n_pops, 6), sharey=False)
    fig.suptitle(
        "Cell Population Frequencies: Responders vs Non-Responders\n"
        "(Melanoma, Miraclib, PBMC)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    palette = {"yes": "#4C9BE8", "no": "#E87E4C"}
    label_map = {"yes": "Responder", "no": "Non-responder"}

    stats_df = stats_df.copy()
    stats_df["Response"] = stats_df["response"].map(label_map)


    p_map = dict(zip(test_results["population"], test_results["p_value"]))

    for ax, pop in zip(axes, CELL_POPULATIONS):
        col = f"{pop}_pct"
        plot_df = stats_df[["Response", col]].rename(columns={col: "Percentage (%)"})

        sns.boxplot(
            data=plot_df,
            x="Response",
            y="Percentage (%)",
            hue="Response",
            palette={"Responder": "#4C9BE8", "Non-responder": "#E87E4C"},
            order=["Responder", "Non-responder"],
            hue_order=["Responder", "Non-responder"],
            width=0.5,
            ax=ax,
            legend=False,
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
        )

        p = p_map[pop]
        sig_label = f"p = {p:.4f}"
        if p < 0.001:
            sig_label += " ***"
        elif p < 0.01:
            sig_label += " **"
        elif p < 0.05:
            sig_label += " *"
        else:
            sig_label += " (ns)"

        ax.set_title(pop.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=9)

        y_max = plot_df["Percentage (%)"].max()
        ax.text(
            0.5, 0.97, sig_label,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=8.5,
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", edgecolor="gray", alpha=0.8),
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Part 4: Subset queries
# ---------------------------------------------------------------------------

def get_baseline_samples() -> pd.DataFrame:
    """
    Returns all melanoma PBMC baseline (time=0) miraclib samples
    with subject metadata.
    """
    conn = _get_conn()
    df = pd.read_sql_query(
        """
        SELECT
            s.sample_id,
            sub.project_id,
            sub.subject_id,
            sub.condition,
            sub.age,
            sub.sex,
            s.treatment,
            s.response,
            s.sample_type,
            s.time_from_treatment_start,
            s.b_cell,
            s.cd8_t_cell,
            s.cd4_t_cell,
            s.nk_cell,
            s.monocyte
        FROM samples s
        JOIN subjects sub ON s.subject_id = sub.subject_id
        WHERE sub.condition              = 'melanoma'
          AND s.sample_type             = 'PBMC'
          AND s.time_from_treatment_start = 0
          AND s.treatment               = 'miraclib'
        """,
        conn,
    )
    conn.close()
    return df


def get_samples_per_project(baseline_df: pd.DataFrame) -> pd.DataFrame:
    return (
        baseline_df.groupby("project_id")
        .size()
        .reset_index(name="sample_count")
        .rename(columns={"project_id": "project"})
    )


def get_responder_counts(baseline_df: pd.DataFrame) -> pd.DataFrame:
    return (
        baseline_df[baseline_df["response"].isin(["yes", "no"])]
        .groupby("response")
        .agg(subjects=("subject_id", "nunique"))
        .reset_index()
        .rename(columns={"response": "response"})
    )


def get_sex_counts(baseline_df: pd.DataFrame) -> pd.DataFrame:
    return (
        baseline_df.groupby("sex")
        .agg(subjects=("subject_id", "nunique"))
        .reset_index()
    )


def get_avg_bcell_melanoma_males(baseline_df: pd.DataFrame) -> float:
    """
    Average B cell count for melanoma male responders at time=0.
    (baseline_df is already filtered to melanoma + PBMC + time=0 + miraclib)
    """
    subset = baseline_df[
        (baseline_df["sex"] == "M") & (baseline_df["response"] == "yes")
    ]
    return round(subset["b_cell"].mean(), 2)
