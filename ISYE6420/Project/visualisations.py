"""
Enter script name

Enter short description of the script
"""

__date__ = "2026-04-13"
__author__ = "NedeeshaWeerasuriya"
__version__ = "0.1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import arviz as az
from scipy import stats

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_tau_samples(trace):
    """Flatten posterior tau samples → (total_draws, N)"""
    tau = trace.posterior["tau"].values          # (chains, draws, N)
    return tau.reshape(-1, tau.shape[-1])        # (4000, N)


def CATT_summary(tau_flat, idx):
    """Return (mean, lo94, hi94, P(tau>0)) for a subgroup index array."""
    draws = tau_flat[:, idx].mean(axis=1)        # (total_draws,) — CATT posterior
    hdi   = az.hdi(draws, hdi_prob=0.94)
    return draws.mean(), hdi[0], hdi[1], (draws > 0).mean()


# ═══════════════════════════════════════════════════════════════════════════
# PLOT A — Binary / CATTgorical variable: posterior CATT per level
# Shows a forest-plot style panel: one row per CATTgory level,
# dot = posterior mean, line = 94% HDI, shaded band = 50% HDI
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# CATT by subgroup for continuous covariate (violin + scatter)
# ═══════════════════════════════════════════════════════════════════════════
# X_tau must be a DataFrame or numpy array with feature names

def plot_CATT_by_feature(X, X_tau, tau_mean, tau_hdi, feature_cols, quants=[0.33, 0.67]):
    feature_index = X.columns.get_loc(feature_cols) 
    feature_values = X_tau[:, feature_index]

    # Recompute tertile cut-points for the selected feature
    q_low, q_high = np.percentile(feature_values, [quants[0]*100, quants[1]*100])

    df_tau = pd.DataFrame({"tau_mean": tau_mean})

    # Pick a CATTgorical moderator (e.g. age group, sex, comorbidity)
    df_tau["tertile"] = pd.cut(
        feature_values,
        bins=[-np.inf, q_low, q_high, np.inf],
        labels=["Low", "Mid", "High"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Violin plot
    ax = axes[0]
    groups = ["Low", "Mid", "High"]
    data   = [df_tau.loc[df_tau["tertile"] == g, "tau_mean"].values for g in groups]
    parts  = ax.violinplot(data, positions=range(3), showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#5B5EA6"); pc.set_alpha(0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(groups)
    ax.axhline(0, color="gray", lw=1, linestyle="--")
    ax.set_title(f"Treatment effect distribution by {feature_cols} tertiles", fontsize=12, fontweight="bold")
    # Scatter: τ vs selected covariate with uncertainty ribbon
    ax = axes[1]
    sort_x = np.argsort(feature_values)
    ax.scatter(feature_values[sort_x], tau_mean[sort_x], s=10, alpha=0.4, color="#5B5EA6", label="Posterior mean τ")
    ax.fill_between(feature_values[sort_x], tau_hdi[sort_x, 0], tau_hdi[sort_x, 1],
                    alpha=0.15, color="#5B5EA6", label="95% HDI")
    sort_x = np.argsort(X_tau[:, feature_index])
    ax.scatter(X_tau[sort_x, feature_index], tau_mean[sort_x], s=10, alpha=0.4, color="#5B5EA6", label="Posterior mean τ")
    ax.fill_between(X_tau[sort_x, feature_index], tau_hdi[sort_x, 0], tau_hdi[sort_x, 1],
                    alpha=0.15, color="#5B5EA6", label="95% HDI")
    ax.axhline(0, color="gray", lw=1, linestyle="--")
    ax.set_xlabel(f"Covariate {feature_cols}", fontsize=11)
    ax.set_ylabel("τ(xᵢ)", fontsize=11)
    ax.set_title(f"Treatment effect vs {feature_cols}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    fig.tight_layout()
    plt.show()


def plot_CATE_categorical(trace, covariate_array, feature_name,
                          level_labels=None, ax=None, color="#5B5EA6"):
    """
    Parameters
    ----------
    covariate_array : 1-D array of category labels/codes for each unit
    level_labels    : dict mapping raw value → display label (optional)
    """
    tau_flat = get_tau_samples(trace)
    levels   = np.unique(covariate_array)

    if level_labels is None:
        level_labels = {v: str(v) for v in levels}

    rows = []
    for lv in levels:
        idx  = np.where(covariate_array == lv)[0]
        mean, lo94, hi94, p_pos = CATT_summary(tau_flat, idx)
        # Also compute 50% HDI for inner band
        draws_lv = tau_flat[:, idx].mean(axis=1)
        lo50, hi50 = az.hdi(draws_lv, hdi_prob=0.50)
        rows.append(dict(level=level_labels[lv], n=len(idx),
                         mean=mean, lo94=lo94, hi94=hi94,
                         lo50=lo50, hi50=hi50, p_pos=p_pos))

    df = pd.DataFrame(rows)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 1.2 * len(df) + 1.5))

    y_pos = np.arange(len(df))[::-1]            # top = first level

    for i, (_, row) in enumerate(df.iterrows()):
        y = y_pos[i]
        # 94% HDI — thin line
        ax.plot([row.lo94, row.hi94], [y, y], lw=1.5,
                color=color, alpha=0.5, solid_capstyle="round")
        # 50% HDI — thick line
        ax.plot([row.lo50, row.hi50], [y, y], lw=5,
                color=color, alpha=0.7, solid_capstyle="round")
        # Posterior mean dot
        ax.scatter(row["mean"], y, s=60, color=color,
                   zorder=5, edgecolors="white", linewidths=0.8)
        # P(τ > 0) annotation on right
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] != 0 else row.hi94 + 0.01,
                y, f"  P(τ>0)={row.p_pos:.2f}  n={row.n}",
                va="center", fontsize=8.5, color="gray")

    ax.axvline(0, color="gray", lw=1, linestyle="--", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["level"].values, fontsize=10)
    ax.set_xlabel("CATT — posterior mean τ  [log-count scale]", fontsize=10)
    ax.set_title(f"CATT by {feature_name}", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    return ax, df


# ═══════════════════════════════════════════════════════════════════════════
# PLOT B — Continuous variable: tertile split with full posterior densities
# Overlapping KDE of the CATT posterior per tertile group
# ═══════════════════════════════════════════════════════════════════════════

def plot_CATT_continuous(trace, covariate_array, feature_name,
                         n_quantiles=3, ax=None):
    tau_flat  = get_tau_samples(trace)
    quantiles = np.linspace(0, 100, n_quantiles + 1)
    breakpoints = np.percentile(covariate_array, quantiles)
    labels    = [f"Q{i+1} ({breakpoints[i]:.1f}–{breakpoints[i+1]:.1f})"
                 for i in range(n_quantiles)]
    palette   = ["#5B5EA6", "#1D9E75", "#D85A30"][:n_quantiles]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    summary_rows = []
    for i in range(n_quantiles):
        lo, hi = breakpoints[i], breakpoints[i + 1]
        idx    = np.where((covariate_array >= lo) & (covariate_array <= hi))[0]
        draws  = tau_flat[:, idx].mean(axis=1)  # CATT posterior for this group

        # KDE of CATT posterior
        kde = stats.gaussian_kde(draws, bw_method="scott")
        xs  = np.linspace(draws.min(), draws.max(), 300)
        ax.plot(xs, kde(xs), lw=2, color=palette[i], label=labels[i])
        ax.fill_between(xs, kde(xs), alpha=0.12, color=palette[i])

        mean, lo94, hi94, p_pos = CATT_summary(tau_flat, idx)
        summary_rows.append(dict(group=labels[i], n=len(idx),
                                 mean=mean, lo94=lo94, hi94=hi94, p_pos=p_pos))

    ax.axvline(0, color="gray", lw=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("CATT  [log-count scale]", fontsize=10)
    ax.set_ylabel("Posterior density", fontsize=10)
    ax.set_title(f"CATT distribution by {feature_name} quantile", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    return ax, pd.DataFrame(summary_rows)


# ═══════════════════════════════════════════════════════════════════════════
# PLOT C — Two binary variables: 2×2 interaction grid
# Shows how the treatment effect varies across combinations of two moderators
# ═══════════════════════════════════════════════════════════════════════════

def plot_CATT_interaction(trace, var1_array, var2_array,
                          var1_name, var2_name,
                          var1_labels=None, var2_labels=None, ax=None):
    tau_flat = get_tau_samples(trace)
    lv1 = np.unique(var1_array)
    lv2 = np.unique(var2_array)

    if var1_labels is None: var1_labels = {v: str(v) for v in lv1}
    if var2_labels is None: var2_labels = {v: str(v) for v in lv2}

    n1, n2 = len(lv1), len(lv2)
    means  = np.zeros((n1, n2))
    lo94   = np.zeros((n1, n2))
    hi94   = np.zeros((n1, n2))
    ns     = np.zeros((n1, n2), dtype=int)

    for i, v1 in enumerate(lv1):
        for j, v2 in enumerate(lv2):
            idx = np.where((var1_array == v1) & (var2_array == v2))[0]
            if len(idx) < 5:                    # too few — mark as NaN
                means[i, j] = np.nan
                continue
            m, l, h, _ = CATT_summary(tau_flat, idx)
            means[i, j] = m
            lo94[i, j]  = l
            hi94[i, j]  = h
            ns[i, j]    = len(idx)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5 + n2, 2 + n1 * 1.4))

    # Heatmap-style with text annotations
    vmax = np.nanmax(np.abs(means))
    im   = ax.imshow(means, cmap="RdBu_r", aspect="auto",
                     vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Posterior mean τ", shrink=0.8)

    ax.set_xticks(range(n2))
    ax.set_xticklabels([var2_labels[v] for v in lv2], fontsize=10)
    ax.set_yticks(range(n1))
    ax.set_yticklabels([var1_labels[v] for v in lv1], fontsize=10)
    ax.set_xlabel(var2_name, fontsize=11)
    ax.set_ylabel(var1_name, fontsize=11)
    ax.set_title(f"CATT interaction: {var1_name} × {var2_name}",
                 fontsize=12, fontweight="bold")

    for i in range(n1):
        for j in range(n2):
            if np.isnan(means[i, j]):
                ax.text(j, i, "n<5", ha="center", va="center",
                        fontsize=9, color="gray")
            else:
                ax.text(j, i,
                        f"{means[i,j]:.3f}\n[{lo94[i,j]:.2f}, {hi94[i,j]:.2f}]\nn={ns[i,j]}",
                        ha="center", va="center", fontsize=8.5,
                        color="white" if abs(means[i,j]) > vmax * 0.5 else "black")

    return ax


# ═══════════════════════════════════════════════════════════════════════════
# PLOT D — Sorted ITE strip coloured by a CATTgorical moderator
# Shows individual-level heterogeneity and whether a CATTgory explains it
# ═══════════════════════════════════════════════════════════════════════════

def plot_ite_strip(trace, covariate_array, feature_name,
                  level_labels=None, ax=None):
    tau_flat = get_tau_samples(trace)
    tau_mean = tau_flat.mean(axis=0)             # (N,)
    tau_hdi  = az.hdi(tau_flat, hdi_prob=0.94)   # (N, 2)

    levels  = np.unique(covariate_array)
    if level_labels is None:
        level_labels = {v: str(v) for v in levels}

    palette = ["#5B5EA6", "#1D9E75", "#D85A30", "#D4537E",
               "#BA7517", "#378ADD"][:len(levels)]
    color_map = {lv: palette[i] for i, lv in enumerate(levels)}

    sort_idx = np.argsort(tau_mean)
    xs       = np.arange(len(tau_mean))

    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 5))

    # HDI ribbon
    ax.fill_between(xs, tau_hdi[sort_idx, 0], tau_hdi[sort_idx, 1],
                    alpha=0.12, color="gray")

    # Scatter coloured by CATTgory
    for lv in levels:
        mask = covariate_array[sort_idx] == lv
        ax.scatter(xs[mask], tau_mean[sort_idx][mask],
                   s=12, alpha=0.6, color=color_map[lv],
                   label=level_labels[lv], linewidths=0)

    ax.axhline(0, color="gray", lw=1, linestyle="--", alpha=0.7)
    ax.axhline(tau_mean.mean(), color="crimson", lw=1.5,
               linestyle=":", label=f"ATE = {tau_mean.mean():.3f}")
    ax.set_xlabel("Unit (sorted by τ)", fontsize=10)
    ax.set_ylabel("Posterior mean τ(xᵢ)", fontsize=10)
    ax.set_title(f"Individual treatment effects coloured by {feature_name}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, markerscale=1.8)
    ax.spines[["top", "right"]].set_visible(False)

    return ax


# ═══════════════════════════════════════════════════════════════════════════
# ASSEMBLE — one figure per variable, choosing the right plot automatically
# ═══════════════════════════════════════════════════════════════════════════

def auto_hte_plot(trace, covariate_array, feature_name,
                  level_labels=None, n_unique_threshold=8):
    """
    Automatically picks the right plot type:
      binary (2 levels)     → forest plot  (plot A)
      CATTgorical (3–8)     → forest plot  (plot A) + ITE strip (plot D)
      continuous (>8 unique)→ quantile KDE (plot B) + ITE strip (plot D)
    """
    n_unique = len(np.unique(covariate_array))

    if n_unique <= n_unique_threshold:
        fig, axes = plt.subplots(1, 2, figsize=(15, max(4, n_unique * 0.9 + 2)))
        plot_CATT_CATTgorical(trace, covariate_array, feature_name,
                              level_labels=level_labels, ax=axes[0])
        plot_ite_strip(trace, covariate_array, feature_name,
                       level_labels=level_labels, ax=axes[1])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        plot_CATT_continuous(trace, covariate_array, feature_name, ax=axes[0])
        plot_ite_strip(trace, covariate_array, feature_name,
                       level_labels=level_labels, ax=axes[1])

    fig.suptitle(f"HTE analysis — {feature_name}", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig