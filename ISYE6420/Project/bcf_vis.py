"""
BCF Binary Probit — Post-estimation Analyses
=============================================
A. GATES  — Group Average Treatment Effects by RD decile
B. CLAN   — Classification Analysis covariate profile (Table 1 style)
C. HTE    — Covariate-specific forest plots on the absolute RD scale

Prerequisites (already in your notebook before these cells):
    trace_bcf        : ArviZ InferenceData from pm.sample()
    mu_samples       : (n_draws, N)  — already reshaped in block 25
    tau_samples      : (n_draws, N)  — already reshaped in block 25
    ite_rd           : (n_draws, N)  — risk_1 - risk_0, from block 25
    mean_rd_treated  : (n_treated,) — posterior mean RD per treated patient
    T                : (N,) int array, 1 = treated
    X                : pd.DataFrame of baseline covariates (N rows)
    y_data           : (N,) binary outcome array
    constant_intercept : norm.ppf(num_exac_pats)  ≈ -1.645
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import arviz as az
from scipy.special import ndtr          # Φ(·) — standard normal CDF
from scipy.stats import chi2_contingency, ttest_ind
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Shared utility: rebuild ITE-RD from posterior (idempotent — safe to re-run)
# ─────────────────────────────────────────────────────────────────────────────

def build_ite_rd(trace, X_mu_arr, X_tau_arr, T_arr, constant_intercept):
    """
    Return ite_rd (n_draws × N), rd_treated (n_draws × n_treated),
    mean_rd_treated (n_treated,), treated_mask (N,).

    Uses the probit formula:
        RD_i = Φ(c + μ_i + τ_i) − Φ(c + μ_i)
    """
    N = X_mu_arr.shape[0]
    mu_s  = trace.posterior["mu"].values.reshape(-1, N)   # (D, N)
    tau_s = trace.posterior["tau"].values.reshape(-1, N)  # (D, N)

    eta_0   = mu_s + constant_intercept
    eta_1   = mu_s + tau_s + constant_intercept
    ite_rd  = ndtr(eta_1) - ndtr(eta_0)                  # (D, N)

    treated_mask = T_arr.astype(bool)
    rd_treated   = ite_rd[:, treated_mask]                # (D, n_treated)
    mean_rd_t    = rd_treated.mean(axis=0)                # (n_treated,)

    return ite_rd, rd_treated, mean_rd_t, treated_mask


# ─────────────────────────────────────────────────────────────────────────────
# A. GATES — Group Average Treatment Effects on the RD scale
# ─────────────────────────────────────────────────────────────────────────────

def plot_gates(rd_treated, mean_rd_treated, n_deciles=10,
               hdi_prob=0.95, figsize=(9, 5), save_path=None):
    """
    Sorts treated patients by their posterior-mean RD, bins into deciles,
    computes the posterior GATE distribution per decile, and plots
    point estimates + HDI bands.

    Parameters
    ----------
    rd_treated      : (n_draws, n_treated) array of per-draw, per-patient RDs
    mean_rd_treated : (n_treated,) posterior mean RD
    n_deciles       : number of bins (default 10)
    hdi_prob        : HDI width (default 0.95)
    save_path       : if provided, saves figure to this path

    Returns
    -------
    gates_df : pd.DataFrame with columns [decile, n, gate_mean, lo, hi, p_benefit]
    """
    # Assign each treated patient to a decile based on their posterior mean RD
    decile_labels = pd.qcut(mean_rd_treated, q=n_deciles, labels=False)

    rows = []
    for d in range(n_deciles):
        mask_d = (decile_labels == d)
        # GATE posterior: average RD across patients in this decile, per draw
        gate_draws = rd_treated[:, mask_d].mean(axis=1)  # (n_draws,)
        hdi_vals   = az.hdi(gate_draws, hdi_prob=hdi_prob)
        rows.append({
            "decile"    : d + 1,
            "n"         : mask_d.sum(),
            "gate_mean" : gate_draws.mean(),
            "lo"        : hdi_vals[0],
            "hi"        : hdi_vals[1],
            "p_benefit" : (gate_draws < 0).mean(),   # P(RD < 0)
        })

    gates_df = pd.DataFrame(rows)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    colours = ["#2ecc71" if m < 0 else "#e74c3c"
               for m in gates_df["gate_mean"]]

    for _, row in gates_df.iterrows():
        x = row["decile"]
        ax.plot([x, x], [row["lo"], row["hi"]],
                color="#aaaaaa", lw=1.5, zorder=1)
        ax.scatter(x, row["gate_mean"], color=colours[int(row["decile"]) - 1],
                   s=80, zorder=3, edgecolors="white", linewidths=0.8)

    ax.axhline(0, color="black", lw=1, linestyle="--", alpha=0.6)

    # Overall ATT reference line
    att_mean = rd_treated.mean()
    ax.axhline(att_mean, color="#3498db", lw=1.5, linestyle=":",
               label=f"Overall ATT = {att_mean:.4f}")

    # Shade benefit / no-benefit zones
    ax.axhspan(ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -0.05,
               0, alpha=0.04, color="green")

    ax.set_xticks(range(1, n_deciles + 1))
    ax.set_xticklabels(
        [f"D{d}\n(n={int(gates_df.loc[gates_df.decile==d,'n'].values[0])})"
         for d in range(1, n_deciles + 1)],
        fontsize=8.5)
    ax.set_xlabel("Decile of predicted risk reduction\n(D1 = largest benefit → D10 = least benefit)",
                  fontsize=10)
    ax.set_ylabel(f"GATE — Absolute Risk Difference\n({int(hdi_prob*100)}% HDI)", fontsize=10)
    ax.set_title("GATES: Group Average Treatment Effects by Predicted Benefit Decile",
                 fontsize=11, fontweight="bold")

    benefit_patch = mpatches.Patch(color="#2ecc71", label="Posterior mean RD < 0 (benefit)")
    harm_patch    = mpatches.Patch(color="#e74c3c", label="Posterior mean RD ≥ 0 (no benefit)")
    ax.legend(handles=[benefit_patch, harm_patch,
                        mpatches.Patch(color="#3498db", label=f"Overall ATT = {att_mean:.4f}")],
              fontsize=8.5, loc="upper left")

    fig.tight_layout()
    plt.show()
    return gates_df


# ─────────────────────────────────────────────────────────────────────────────
# B. CLAN — Classification Analysis covariate profile table
# ─────────────────────────────────────────────────────────────────────────────

# ── Human-readable display labels for CLAN table ─────────────────────────────
# Maps raw column names → publication-ready labels.
# Extend this dict if you add new covariates.
CLAN_DISPLAY_LABELS = {
    # Healthcare utilisation
    "num_er_event"             : "ER Visits",
    "num_outpatient_event"     : "Outpatient Visits",
    "total_exac_count"         : "Exacerbation Count - Baseline",
    "cp_count"                 : "Consultation/Procedure Count",
    # Demographics
    "age"                      : "Age",
    "sex_Female"               : "Female Sex",
    # Disease severity / treatment intensity
    "gina_step"                : "GINA Step",
    "num_treatments_baseline"  : "Unique Treatments - Baseline",
    "ICS_LABA_flag"            : "ICS + LABA",
    "ICS_LABA_LAMA_flag"       : "ICS + LABA + LAMA",
    "ICS_flag"                 : "ICS (alone)",
    "SABA_flag"                : "SABA Use",
    "OCS_flag"                 : "OCS Use",
    "Biologics_flag"           : "Biologic Therapy",
    # Cardiometabolic comorbidities
    "ckd_flag"                 : "CKD",
    "dyslipidemia_flag"        : "Dyslipidaemia",
    "hypertension_flag"        : "Hypertension",
    "heart_failure_flag"       : "Heart Failure",
    "CAD_flag"                 : "Coronary Artery Disease",
    "PVD_flag"                 : "Peripheral Vascular Disease",
    "t2dm_flag"                : "Type 2 Diabetes",
    # Metabolic / anthropometric
    "bmi_above_25"             : "BMI > 25",
    "obstructive_sleep_apnea_flag": "Obstructive Sleep Apnoea",
    # Atopic profile
    "atopic_disease_flag"      : "Atopic Disease",
    "eosinophilic_disease_flag": "Eosinophilic Disease",
    # Propensity score (informational only)
    "logit_ps"                 : "Logit Propensity Score",
}
 
 
def build_clan_table(mean_rd_treated, X_full, T_arr,
                     top_pct=0.10,
                     binary_cols=None,
                     continuous_cols=None,
                     display_labels=None):
    """
    Compares the top `top_pct` maximum-benefit treated patients against the
    remaining treated patients on baseline covariates.
 
    Produces a publication-ready table matching the format:
 
        Variable | Type | D1 - high benefit (n=N) | Rest of treated (n=M) | Absolute Difference
 
    For continuous variables: mean ± SD (2 d.p.), abs. difference to 2 d.p.
    For binary variables    : XX.X%,               abs. difference as proportion to 2 d.p.
    Rows are sorted by absolute difference descending.
    No p-value column is included (add separately if required by journal).
 
    Parameters
    ----------
    mean_rd_treated : (n_treated,) posterior mean RD, treated patients only
    X_full          : pd.DataFrame (N rows, all patients)
    T_arr           : (N,) int treatment indicator
    top_pct         : float  — fraction defining the high-benefit group (default 0.10)
    binary_cols     : list of binary column names to summarise as proportions
    continuous_cols : list of continuous column names to summarise as mean ± SD
    display_labels  : dict mapping raw column names to display labels;
                      defaults to CLAN_DISPLAY_LABELS
 
    Returns
    -------
    clan_df : pd.DataFrame with columns
              [Variable, Type, D1 - high benefit (n=N), Rest of treated (n=M),
               Absolute Difference, _abs_diff_raw]
              _abs_diff_raw is a float kept for downstream sorting; drop before display.
    """
    if display_labels is None:
        display_labels = CLAN_DISPLAY_LABELS
 
    treated_mask = T_arr.astype(bool)
    X_treated    = X_full[treated_mask].reset_index(drop=True)
 
    threshold = np.percentile(mean_rd_treated, top_pct * 100)
    hb_mask   = mean_rd_treated <= threshold   # most negative RD = highest benefit
 
    X_hb   = X_treated[hb_mask]
    X_rest = X_treated[~hb_mask]
    n_hb   = int(hb_mask.sum())
    n_rest = int((~hb_mask).sum())
 
    # Column header strings — mirror the target table exactly
    col_hb   = f"D1 - high benefit (n={n_hb})"
    col_rest = f"Rest of treated (n={n_rest})"
 
    # Default column sets — ordered to match target table (sorted overrides below)
    if continuous_cols is None:
        continuous_cols = [c for c in [
            "num_er_event", "age", "num_outpatient_event",
            "num_treatments_baseline", "gina_step", "total_exac_count",
            "logit_ps", "cp_count",
        ] if c in X_treated.columns]
 
    if binary_cols is None:
        binary_cols = [c for c in [
            "ckd_flag", "dyslipidemia_flag", "hypertension_flag",
            "heart_failure_flag", "CAD_flag", "PVD_flag",
            "obstructive_sleep_apnea_flag", "bmi_above_25",
            "ICS_LABA_flag", "ICS_LABA_LAMA_flag", "ICS_flag",
            "SABA_flag", "OCS_flag", "Biologics_flag",
            "t2dm_flag", "sex_Female",
            "atopic_disease_flag", "eosinophilic_disease_flag",
        ] if c in X_treated.columns]
 
    rows = []
 
    # ── Continuous variables ──────────────────────────────────────────────
    for col in continuous_cols:
        hb_vals   = X_hb[col].dropna()
        rest_vals = X_rest[col].dropna()
        abs_diff  = abs(hb_vals.mean() - rest_vals.mean())
        label     = display_labels.get(col, col)
        rows.append({
            "Variable"          : label,
            "Type"              : "continuous",
            col_hb              : f"{hb_vals.mean():.2f} \u00b1 {hb_vals.std():.2f}",
            col_rest            : f"{rest_vals.mean():.2f} \u00b1 {rest_vals.std():.2f}",
            "Absolute Difference": f"{abs_diff:.2f}",
            "_abs_diff_raw"     : abs_diff,
        })
 
    # ── Binary variables ──────────────────────────────────────────────────
    for col in binary_cols:
        hb_prop   = X_hb[col].mean()
        rest_prop = X_rest[col].mean()
        abs_diff  = abs(hb_prop - rest_prop)
        label     = display_labels.get(col, col)
        rows.append({
            "Variable"          : label,
            "Type"              : "binary",
            col_hb              : f"{hb_prop * 100:.1f}%",
            col_rest            : f"{rest_prop * 100:.1f}%",
            "Absolute Difference": f"{abs_diff:.2f}",
            "_abs_diff_raw"     : abs_diff,
        })
 
    clan_df = (pd.DataFrame(rows)
               .sort_values("_abs_diff_raw", ascending=False)
               .drop(columns="_abs_diff_raw")
               .reset_index(drop=True))
 
    return clan_df
 
 
def display_clan_table(clan_df, top_pct=0.10):
    """
    Render the CLAN table in Jupyter using pandas Styler for a clean,
    publication-ready appearance.
 
    Call display_clan_table(clan_df) in a notebook cell; the styled
    DataFrame renders as an HTML table.
 
    For plain-text environments use print_clan_table(clan_df) instead.
    """
    from IPython.display import display as ipy_display
 
    pct_label = f"Top {int(top_pct * 100)}%"
 
    styled = (
        clan_df.style
        .set_caption(
            f"CLAN Table — {pct_label} Maximum-Benefit Subgroup (D1) "
            "vs Rest of Treated"
        )
        .set_table_styles([
            # Caption
            {"selector": "caption",
             "props": [("font-size", "13px"), ("font-weight", "bold"),
                       ("text-align", "left"), ("padding-bottom", "6px")]},
            # Header
            {"selector": "thead th",
             "props": [("background-color", "#2c3e50"), ("color", "white"),
                       ("font-size", "11px"), ("text-align", "center"),
                       ("padding", "6px 10px"), ("border", "1px solid #ddd")]},
            # Body cells
            {"selector": "tbody td",
             "props": [("font-size", "11px"), ("padding", "5px 10px"),
                       ("border", "1px solid #ddd"), ("text-align", "center")]},
            # First column left-aligned
            {"selector": "tbody td:first-child",
             "props": [("text-align", "left"), ("font-weight", "500")]},
            # Alternating row colours
            {"selector": "tbody tr:nth-child(even)",
             "props": [("background-color", "#f7f9fc")]},
            {"selector": "tbody tr:nth-child(odd)",
             "props": [("background-color", "#ffffff")]},
            # Hover
            {"selector": "tbody tr:hover",
             "props": [("background-color", "#eaf2ff")]},
        ])
        .hide(axis="index")
    )
 
    ipy_display(styled)
 
 
def print_clan_table(clan_df, top_pct=0.10):
    """Plain-text fallback for non-Jupyter environments."""
    pct_label = f"Top {int(top_pct * 100)}%"
    hdr = clan_df.columns.tolist()
    w = [36, 12, 26, 26, 20]
    sep = "─" * sum(w)
    print(f"\n{sep}")
    print(f"  CLAN Table — {pct_label} Maximum-Benefit Subgroup (D1) vs Rest of Treated")
    print(sep)
    print(f"{'Variable':<{w[0]}} {'Type':<{w[1]}} {hdr[2]:<{w[2]}} "
          f"{hdr[3]:<{w[3]}} {'Absolute Difference':>{w[4]}}")
    print(sep)
    for _, row in clan_df.iterrows():
        print(f"{row['Variable']:<{w[0]}} {row['Type']:<{w[1]}} "
              f"{row[hdr[2]]:<{w[2]}} {row[hdr[3]]:<{w[3]}} "
              f"{row['Absolute Difference']:>{w[4]}}")
    print(sep + "\n")
 


# ─────────────────────────────────────────────────────────────────────────────
# C. HTE Forest Plots — absolute RD scale for key clinical covariates
# ─────────────────────────────────────────────────────────────────────────────

def _rd_subgroup_posterior(rd_all, mask_all, hdi_prob=0.95):
    """
    Average RD posterior draws across patients in `mask_all`.

    Parameters
    ----------
    rd_all  : (n_draws, N) full-cohort RD array
    mask_all: (N,) boolean mask selecting the subgroup
    """
    draws = rd_all[:, mask_all].mean(axis=1)   # (n_draws,)
    hdi_v = az.hdi(draws, hdi_prob=hdi_prob)
    lo50, hi50 = az.hdi(draws, hdi_prob=0.50)
    return {
        "mean"      : draws.mean(),
        "lo95"      : hdi_v[0],
        "hi95"      : hdi_v[1],
        "lo50"      : lo50,
        "hi50"      : hi50,
        "p_benefit" : (draws < 0).mean(),
        "n"         : mask_all.sum(),
    }


def plot_hte_forest(ite_rd, X_full, T_arr,
                    features_config,
                    hdi_prob=0.95,
                    figsize=None,
                    colour="#2c7bb6",
                    save_path=None):
    """
    Covariate-stratified HTE forest plot on the absolute Risk Difference scale.

    Parameters
    ----------
    ite_rd          : (n_draws, N) per-patient RD posterior
    X_full          : pd.DataFrame (N rows)
    T_arr           : (N,) treatment indicator (1 = treated)
    features_config : list of dicts defining strata. Each dict:
        {
          "label"  : str,          # display name
          "mask"   : array (N,),   # boolean — which patients are in this stratum
          "group"  : str,          # group heading (e.g. "Prior Exacerbations")
        }
    hdi_prob        : HDI width
    figsize         : (width, height) — auto-computed if None
    colour          : hex colour for the point/interval markers
    save_path       : optional save path

    Returns
    -------
    forest_df : pd.DataFrame with all computed statistics
    """
    treated_mask = T_arr.astype(bool)

    rows = []
    for cfg in features_config:
        # Intersect with treated patients for CATE among treated
        subgroup_mask = cfg["mask"] & treated_mask
        if subgroup_mask.sum() < 20:
            continue
        stats = _rd_subgroup_posterior(ite_rd, subgroup_mask, hdi_prob=hdi_prob)
        rows.append({**stats,
                     "label" : cfg["label"],
                     "group" : cfg.get("group", "")})

    forest_df = pd.DataFrame(rows)

    # ── Plot ──────────────────────────────────────────────────────────────
    n_rows = len(forest_df)
    if figsize is None:
        figsize = (9, max(5, 0.55 * n_rows + 2))

    fig, ax = plt.subplots(figsize=figsize)

    y_positions = np.arange(n_rows)[::-1]
    current_group = None
    group_offset  = 0    # extra vertical space between groups

    for i, (_, row) in enumerate(forest_df.iterrows()):
        y = y_positions[i] + group_offset

        # Group header
        if row["group"] != current_group:
            current_group = row["group"]
            ax.text(-0.005, y + 0.65, row["group"],
                    fontsize=9.5, fontweight="bold",
                    transform=ax.get_yaxis_transform(),
                    ha="right", va="center", color="#333333")

        # 95% HDI — thin line
        ax.plot([row["lo95"], row["hi95"]], [y, y],
                color=colour, lw=1.4, alpha=0.5, solid_capstyle="round")
        # 50% HDI — thick line
        ax.plot([row["lo50"], row["hi50"]], [y, y],
                color=colour, lw=5, alpha=0.65, solid_capstyle="round")
        # Posterior mean dot
        dot_colour = "#27ae60" if row["mean"] < 0 else "#e74c3c"
        ax.scatter(row["mean"], y, s=55, color=dot_colour,
                   zorder=5, edgecolors="white", linewidths=0.8)

        # Annotations on the right
        ann = (f"  RD={row['mean']:.4f}  "
               f"[{row['lo95']:.4f}, {row['hi95']:.4f}]  "
               f"P(benefit)={row['p_benefit']:.2f}  "
               f"n={int(row['n'])}")
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] != 0 else row["hi95"] + 0.005,
                y, ann, va="center", fontsize=7.8, color="gray")

    ax.axvline(0, color="black", lw=1, linestyle="--", alpha=0.5)

    # Overall ATT reference
    att_draws = ite_rd[:, treated_mask].mean(axis=1)
    att_mean  = att_draws.mean()
    ax.axvline(att_mean, color="#e67e22", lw=1.2, linestyle=":",
               label=f"Overall ATT = {att_mean:.4f}")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(forest_df["label"].tolist(), fontsize=9.5)
    ax.set_xlabel("Absolute Risk Difference (treated − control)\nNegative = benefit from GLP1RA",
                  fontsize=10)
    ax.set_title(
        "Covariate-Stratified HTE — Absolute Risk Difference\n"
        f"(thick bar = 50% HDI, thin bar = {int(hdi_prob*100)}% HDI)",
        fontsize=11, fontweight="bold")

    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    plt.show()

    return forest_df


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════  MAIN EXECUTION — drop these cells into your notebook  ══════
# ─────────────────────────────────────────────────────────────────────────────

def run_all(trace_bcf, X, T, y_data, X_mu, X_tau,
            constant_intercept, num_exac_pats):
    """
    Master function — call from notebook after sampling.

    Usage
    -----
    from bcf_gates_clan_hte import run_all
    run_all(trace_bcf, X, T, y_data, X_mu, X_tau,
            constant_intercept, num_exac_pats)
    """

    # ── Step 0: (Re)build ITE-RD posterior ───────────────────────────────
    print("Building ITE-RD posterior arrays …")
    ite_rd, rd_treated, mean_rd_treated, treated_mask = build_ite_rd(
        trace_bcf, X_mu, X_tau, T, constant_intercept)
    print(f"  ite_rd shape        : {ite_rd.shape}")
    print(f"  rd_treated shape    : {rd_treated.shape}")
    print(f"  mean(CATT)          : {rd_treated.mean():.5f}")

    # ── A: GATES ─────────────────────────────────────────────────────────
    print("\n── A: GATES plot ──")
    gates_df = plot_gates(
        rd_treated, mean_rd_treated,
        n_deciles=10,
        save_path="fig_gates_rd.png"
    )
    print(gates_df.to_string(index=False))

    # ── B: CLAN ──────────────────────────────────────────────────────────
    print("\n── B: CLAN table ──")
    clan_df = build_clan_table(
        mean_rd_treated, X, T,
        top_pct=0.10
    )
    print_clan_table(clan_df, top_pct=0.10)

    # ── C: HTE Forest Plot ───────────────────────────────────────────────
    print("\n── C: HTE forest plot ──")

    # Build feature configuration matching Wang et al.'s key splits
    # + additional clinical strata
    # All masks apply over the FULL cohort (N rows), not just treated
    X_arr = X.values
    X_df  = X.reset_index(drop=True)

    features_config = [
        # ── Wang et al. first split ───────────────────────────────────
        {"group": "Prior ED Visits (Wang primary split)",
         "label": "≥2 ED visits",
         "mask" : (X_df["num_er_event"] >= 2).values},
        {"group": "Prior ED Visits (Wang primary split)",
         "label": "1 ED visit",
         "mask" : (X_df["num_er_event"] == 1).values},
        {"group": "Prior ED Visits (Wang primary split)",
         "label": "0 ED visits",
         "mask" : (X_df["num_er_event"] == 0).values},

        # ── Wang et al. SABA split ────────────────────────────────────
        {"group": "SABA Use (Wang second split)",
         "label": "SABA user",
         "mask" : (X_df["SABA_flag"] == 1).values},
        {"group": "SABA Use (Wang second split)",
         "label": "No SABA",
         "mask" : (X_df["SABA_flag"] == 0).values},

        # ── Wang et al. ICS+LABA split ────────────────────────────────
        {"group": "ICS/LABA (Wang third split)",
         "label": "ICS+LABA",
         "mask" : (X_df["ICS_LABA_flag"] == 1).values},
        {"group": "ICS/LABA (Wang third split)",
         "label": "ICS alone",
         "mask" : ((X_df["ICS_flag"] == 1) & (X_df["ICS_LABA_flag"] == 0)).values},
        {"group": "ICS/LABA (Wang third split)",
         "label": "ICS+LABA+LAMA",
         "mask" : (X_df["ICS_LABA_LAMA_flag"] == 1).values},
        {"group": "ICS/LABA (Wang third split)",
         "label": "No ICS/LABA",
         "mask" : (X_df["ICS_LABA_flag"] == 0).values},

        # ── Wang et al. age split ─────────────────────────────────────
        {"group": "Age (Wang fourth split)",
         "label": "Age >50",
         "mask" : (X_df["age"] > 50).values},
        {"group": "Age (Wang fourth split)",
         "label": "Age 40–50",
         "mask" : ((X_df["age"] >= 40) & (X_df["age"] <= 50)).values},
        {"group": "Age (Wang fourth split)",
         "label": "Age <40",
         "mask" : (X_df["age"] < 40).values},

        # ── GINA disease severity ─────────────────────────────────────
        {"group": "GINA Step",
         "label": "GINA step 0",
         "mask" : (X_df["gina_step"] == 0).values},
        {"group": "GINA Step",
         "label": "GINA step 1–2",
         "mask" : ((X_df["gina_step"] >= 1) & (X_df["gina_step"] <= 2)).values},
        {"group": "GINA Step",
         "label": "GINA step 3–4",
         "mask" : ((X_df["gina_step"] >= 3) & (X_df["gina_step"] <= 4)).values},
        {"group": "GINA Step",
         "label": "GINA step 5",
         "mask" : (X_df["gina_step"] == 5).values},

        # ── Diabetes / metabolic context ─────────────────────────────
        {"group": "Metabolic",
         "label": "BMI >25",
         "mask" : (X_df["bmi_above_25"] == 1).values},
        {"group": "Metabolic",
         "label": "Obstructive sleep apnoea",
         "mask" : (X_df["obstructive_sleep_apnea_flag"] == 1).values},
        {"group": "Metabolic",
         "label": "Dyslipidaemia",
         "mask" : (X_df["dyslipidemia_flag"] == 1).values},

        # ── Prior exacerbation count ──────────────────────────────────
        {"group": "Prior Exacerbations (total_exac_count)",
         "label": "0 prior exacerbations",
         "mask" : (X_df["total_exac_count"] == 0).values},
        {"group": "Prior Exacerbations (total_exac_count)",
         "label": "1 prior exacerbation",
         "mask" : (X_df["total_exac_count"] == 1).values},
        {"group": "Prior Exacerbations (total_exac_count)",
         "label": "≥2 prior exacerbations",
         "mask" : (X_df["total_exac_count"] >= 2).values},

        # ── Sex ───────────────────────────────────────────────────────
        {"group": "Sex",
         "label": "Female",
         "mask" : (X_df["sex_Female"] == 1).values},
        {"group": "Sex",
         "label": "Male",
         "mask" : (X_df["sex_Female"] == 0).values},
    ]

    forest_df = plot_hte_forest(
        ite_rd, X_df, T,
        features_config=features_config,
        hdi_prob=0.95,
        save_path="fig_hte_forest_rd.png"
    )

    print("\nForest plot statistics:")
    print(forest_df[["group", "label", "n", "mean", "lo95", "hi95", "p_benefit"]]
          .to_string(index=False))

    return gates_df, clan_df, forest_df

