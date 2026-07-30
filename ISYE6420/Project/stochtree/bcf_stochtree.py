# =============================================================================
# Bayesian Causal Forest — stochtree (verified against official vignette API)
# Binary outcome (probit), HTE / CATE estimation
# Hahn, Murray & Carvalho (2020)
# =============================================================================
# pip install stochtree scikit-learn arviz seaborn shap

import json
import os
import warnings

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import norm
from sklearn.tree import DecisionTreeRegressor
from stochtree import BCFModel, OutcomeModel

warnings.filterwarnings("ignore")

rng           = np.random.default_rng(123)
output_folder = "ISYE6420/Project/"
dx_name       = "asthma"


# =============================================================================
# 1. DATA
# =============================================================================
cohort = pd.read_csv(f"{output_folder}data/asthma_weighted_cohort.csv")

y_data = (cohort["outcome"].values > 0).astype(float)   # must be float 0/1
T      = cohort["exposure"].values.astype(float)         # must be float

eps    = 1e-6
pi_hat = np.clip(cohort["ps_score"].values, eps, 1 - eps).astype(float)

num_exac_pats = y_data.mean()
print(f"Cohort: {len(cohort):,}  |  Events: {num_exac_pats:.2%}  |  Treated: {T.mean():.2%}")

# Feature matrix: NO treatment column, NO propensity column.
# stochtree receives T via Z_train and PS via propensity_train separately.
drop_cols     = ["patid", "outcome", "exposure", "follow_up_time", "weight", "ps_score"]
X_train       = cohort.drop(columns=drop_cols).astype(float)   # keep as DataFrame
feature_names = X_train.columns.tolist()

print(f"X_train: {X_train.shape}  |  Features: {feature_names}")


# =============================================================================
# 2. BCF MODEL
# BCFModel() takes NO constructor args.
# All configuration is passed as dicts to .sample().
# =============================================================================

general_params = {
    "outcome_model"       : OutcomeModel(outcome="binary", link="probit"),
    "propensity_covariate": "prognostic",   # PS enters mu forest only, never tau
    "adaptive_coding"     : True,
    "random_seed"         : 123,
    "keep_every"          : 5,
    "num_chains"          : 4,
    "standardize"         : True,
}

# Hahn defaults: alpha=0.95, beta=2; 250 trees keeps individuals shallow
prognostic_forest_params = {
    "num_trees"       : 250,
    "alpha"           : 0.95,
    "beta"            : 2.0,
    "min_samples_leaf": 5,
    "max_depth"       : 10,
}

# Strong regularisation toward homogeneous CATE: alpha=0.25, beta=3
# delta_max: max plausible risk difference; conservative for ~5.5% prevalence
treatment_effect_forest_params = {
    "num_trees"       : 100,
    "alpha"           : 0.25,
    "beta"            : 3.0,
    "min_samples_leaf": 5,
    "max_depth"       : 5,
    "delta_max"       : 0.2,
    "sample_intercept": True,
}

bcf = BCFModel()   # no args here

bcf.sample(
    X_train                        = X_train,
    Z_train                        = T,
    y_train                        = y_data,
    propensity_train               = pi_hat,
    num_gfr                        = 20,    # warm-start; must be >= num_chains
    num_burnin                     = 0,
    num_mcmc                       = 500,   # retained draws per chain
    general_params                 = general_params,
    prognostic_forest_params       = prognostic_forest_params,
    treatment_effect_forest_params = treatment_effect_forest_params,
)

print("Sampling complete.")
bcf.summary()


# =============================================================================
# 3. EXTRACT POSTERIOR DRAWS
#
# KEY API FACTS (from official vignette):
#   predict(terms=<single string>) → plain np.array, NOT a dict
#   Correct term names:
#     "prognostic_function"  — mu(X, pi_hat), latent probit scale
#     "cate"                 — tau(X), latent probit scale (= tau_0 + tau forest)
#     "y_hat"                — full prediction mu + Z*tau
#   scale="probit"  → latent probit scale
#   scale="linear"  → original outcome scale (standardised back)
#
#   extract_parameter("tau_hat_train") → same as predict(..., terms="cate")
#   Shape convention: (N, num_chains * num_mcmc)
# =============================================================================

# Each call returns a plain np.array of shape (N, total_draws)
mu_draws   = bcf.predict(X=X_train, Z=T, propensity=pi_hat,
                         terms="prognostic_function", scale="probit")
tau_draws  = bcf.predict(X=X_train, Z=T, propensity=pi_hat,
                         terms="cate", scale="probit")
yhat_prob  = bcf.predict(X=X_train, Z=T, propensity=pi_hat,
                         terms="y_hat", scale="linear")

# Verify shapes
assert mu_draws.ndim == 2,  f"Expected 2D, got {mu_draws.shape}"
assert tau_draws.ndim == 2, f"Expected 2D, got {tau_draws.shape}"
print(f"mu_draws  : {mu_draws.shape}  — (N, total_draws)")
print(f"tau_draws : {tau_draws.shape}")
print(f"yhat_prob : {yhat_prob.shape}")

# Posterior summaries
tau_mean  = tau_draws.mean(axis=1)
tau_sd    = tau_draws.std(axis=1)
tau_lo95  = np.percentile(tau_draws, 2.5,  axis=1)
tau_hi95  = np.percentile(tau_draws, 97.5, axis=1)
tau_lo50  = np.percentile(tau_draws, 25,   axis=1)
tau_hi50  = np.percentile(tau_draws, 75,   axis=1)
mu_mean   = mu_draws.mean(axis=1)
yhat_mean = yhat_prob.mean(axis=1)

treated_idx = np.where(T == 1)[0]
control_idx = np.where(T == 0)[0]
n_draws     = tau_draws.shape[1]

print(f"\nTotal draws          : {n_draws}  (4 chains × 500)")
print(f"ATE (latent probit)  : {tau_mean.mean():.4f}  SD={tau_mean.std():.4f}")
print(f"ATT (latent probit)  : {tau_mean[treated_idx].mean():.4f}")

# Risk difference: compute_contrast with scale="probability"
# Returns plain np.array (N, total_draws)
T1 = np.ones(len(y_data))
T0 = np.zeros(len(y_data))
rd_draws = bcf.compute_contrast(
    X_0=X_train, Z_0=T0, propensity_0=pi_hat,
    X_1=X_train, Z_1=T1, propensity_1=pi_hat,
    type="posterior", scale="probability",
)

ird_mean = rd_draws.mean(axis=1)
ird_lo95 = np.percentile(rd_draws, 2.5,  axis=1)
ird_hi95 = np.percentile(rd_draws, 97.5, axis=1)

# Native 95% CI for tau
ci_tau    = bcf.compute_posterior_interval(
    X=X_train, Z=T, propensity=pi_hat,
    terms="cate", level=0.95, scale="probit"
)   # returns dict {"lower": (N,), "upper": (N,)}
tau_ci_lo = ci_tau["lower"]
tau_ci_hi = ci_tau["upper"]


# =============================================================================
# 4. CONVERGENCE DIAGNOSTICS
# Reshape (total_draws,) → (num_chains, num_mcmc) for multi-chain R̂ / ESS
# =============================================================================
NUM_CHAINS = 4
NUM_MCMC   = 500

ate_trace_flat = tau_draws.mean(axis=0)                          # (total_draws,)
ate_chains     = ate_trace_flat.reshape(NUM_CHAINS, NUM_MCMC)    # (chains, draws)

rhat_val = az.rhat(ate_chains)
ess_val  = az.ess(ate_chains)
print(f"\nATE R̂   : {float(rhat_val):.4f}  (target < 1.01)")
print(f"ATE ESS : {float(ess_val):.0f} / {NUM_CHAINS * NUM_MCMC} total draws")
if float(rhat_val) > 1.01:
    print("⚠  R̂ > 1.01 — consider increasing num_gfr or num_mcmc")
else:
    print("✓  Chains well-mixed")

chain_colours = ["#5B5EA6", "darkorange", "teal", "crimson"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for c in range(NUM_CHAINS):
    axes[0].plot(ate_chains[c], lw=0.7, alpha=0.8,
                 color=chain_colours[c], label=f"Chain {c+1}")
axes[0].axhline(ate_trace_flat.mean(), color="black", lw=1.5, ls="--",
                label=f"Mean = {ate_trace_flat.mean():.3f}")
axes[0].set_title("ATE trace (4 chains)", fontweight="bold")
axes[0].set_xlabel("MCMC draw"); axes[0].set_ylabel("τ̄ (latent probit scale)")
axes[0].legend(fontsize=8)

for c in range(NUM_CHAINS):
    axes[1].hist(ate_chains[c], bins=40, alpha=0.4, density=True,
                 color=chain_colours[c], label=f"Chain {c+1}")
axes[1].axvline(0, color="gray", lw=1, ls=":")
axes[1].set_xlabel("τ̄ (latent probit scale)")
axes[1].set_title("ATE posterior density (per chain)", fontweight="bold")
axes[1].legend(fontsize=8)

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(pd.Series(ate_chains[0]), ax=axes[2], color="#5B5EA6")
axes[2].set_xlim(0, 80)
axes[2].set_title("ATE autocorrelation (chain 1)", fontweight="bold")

fig.suptitle("BCF sampler diagnostics — stochtree (4 chains)", fontsize=13)
fig.tight_layout(); plt.show()


# =============================================================================
# 5. POSTERIOR PREDICTIVE CHECK
# =============================================================================
ppc_draws       = bcf.sample_posterior_predictive(X=X_train, Z=T, propensity=pi_hat)
ppc_event_rates = ppc_draws.mean(axis=0)
obs_rate        = y_data.mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(ppc_event_rates, bins=50, color="teal", alpha=0.7, density=True, edgecolor="white")
axes[0].axvline(obs_rate, color="black", lw=2, label=f"Observed = {obs_rate:.3f}")
axes[0].set_xlabel("Event rate")
axes[0].set_title("PPC: simulated event rate vs observed", fontweight="bold")
axes[0].legend()

df_cal = pd.DataFrame({"pred": yhat_mean, "obs": y_data.astype(int)})
df_cal["decile"] = pd.qcut(df_cal["pred"], 10, labels=False, duplicates="drop")
cal_s = (df_cal.groupby("decile")
               .agg(pred_mean=("pred","mean"), obs_mean=("obs","mean"))
               .reset_index())
lims = [0, max(cal_s["pred_mean"].max(), cal_s["obs_mean"].max()) * 1.15]
axes[1].scatter(cal_s["pred_mean"], cal_s["obs_mean"], color="teal", s=60, zorder=3)
axes[1].plot(lims, lims, "k--", lw=1, label="Perfect calibration")
axes[1].set_xlabel("Mean predicted P(Y=1)"); axes[1].set_ylabel("Observed event rate")
axes[1].set_title("Calibration by decile", fontweight="bold"); axes[1].legend()

fig.suptitle("Posterior predictive checks", fontsize=13)
fig.tight_layout(); plt.show()

print(f"Observed rate : {obs_rate:.4f}")
print(f"PPC mean      : {ppc_event_rates.mean():.4f}  "
      f"[{np.percentile(ppc_event_rates,2.5):.4f}, {np.percentile(ppc_event_rates,97.5):.4f}]")


# =============================================================================
# 6. ATE / ATT SUMMARY
# =============================================================================
def summarise(trace, level=0.95):
    lo = np.percentile(trace, (1-level)/2*100)
    hi = np.percentile(trace, (1+level)/2*100)
    return trace.mean(), lo, hi

ate_trace = tau_draws.mean(axis=0)
att_trace = tau_draws[treated_idx].mean(axis=0)
ate_rd_tr = rd_draws.mean(axis=0)
att_rd_tr = rd_draws[treated_idx].mean(axis=0)

print("\n" + "="*65)
print(f"{'Estimand':<35} {'Mean':>8}  {'95% CI':>20}")
print("-"*65)
for label, tr in [
    ("ATE (latent probit)", ate_trace),
    ("ATT (latent probit)", att_trace),
    ("ATE (risk difference)", ate_rd_tr),
    ("ATT (risk difference)", att_rd_tr),
]:
    m, lo, hi = summarise(tr)
    print(f"{label:<35} {m:>8.4f}  [{lo:.4f}, {hi:.4f}]")
print("="*65)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, ate_t, att_t, xlabel, title in [
    (axes[0], ate_trace, att_trace,
     "τ̄ (latent probit scale)", "ATE vs ATT — latent scale"),
    (axes[1], ate_rd_tr, att_rd_tr,
     "Risk difference Φ(μ+τ) − Φ(μ)", "ATE vs ATT — risk difference"),
]:
    ax.hist(ate_t, bins=60, alpha=0.6, density=True,
            color="#5B5EA6", label="ATE", edgecolor="white")
    ax.hist(att_t, bins=60, alpha=0.5, density=True,
            color="darkorange", label="ATT", edgecolor="white")
    ax.axvline(0, color="gray", lw=1, ls=":")
    ax.set_xlabel(xlabel); ax.set_ylabel("Density")
    ax.set_title(title, fontweight="bold"); ax.legend(fontsize=9)
fig.suptitle("Treatment effect posteriors", fontsize=13)
fig.tight_layout(); plt.show()


# =============================================================================
# 7. ITE DISTRIBUTIONS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(tau_mean, bins=40, color="#5B5EA6", alpha=0.75, edgecolor="white")
axes[0].axvline(tau_mean.mean(), color="crimson", lw=2,
                label=f"ATE = {tau_mean.mean():.3f}")
axes[0].axvline(tau_mean[treated_idx].mean(), color="darkorange", lw=2, ls="--",
                label=f"ATT = {tau_mean[treated_idx].mean():.3f}")
axes[0].axvline(0, color="gray", lw=1, ls=":")
axes[0].set_xlabel("Posterior mean τ(xᵢ) — latent probit scale")
axes[0].set_title("ITE distribution — latent scale", fontweight="bold")
axes[0].legend(fontsize=9)

axes[1].hist(ird_mean, bins=40, color="teal", alpha=0.75, edgecolor="white")
axes[1].axvline(ird_mean.mean(), color="crimson", lw=2,
                label=f"ATE = {ird_mean.mean():.3f}")
axes[1].axvline(ird_mean[treated_idx].mean(), color="darkorange", lw=2, ls="--",
                label=f"ATT = {ird_mean[treated_idx].mean():.3f}")
axes[1].axvline(0, color="gray", lw=1, ls=":")
axes[1].set_xlabel("Posterior mean risk difference")
axes[1].set_title("ITE distribution — risk difference scale", fontweight="bold")
axes[1].legend(fontsize=9)

fig.suptitle("Individual treatment effects — BCF (stochtree)", fontsize=13)
fig.tight_layout(); plt.show()


# =============================================================================
# 8. WATERFALL PLOT
# =============================================================================
sort_idx = np.argsort(ird_mean)
ird_s    = ird_mean[sort_idx]
ird_lo_s = ird_lo95[sort_idx]
ird_hi_s = ird_hi95[sort_idx]
xs       = np.arange(len(ird_s))

fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(xs, 0, ird_s, where=(ird_s < 0),  color="green", alpha=0.3,
                label="Benefit (RD < 0)")
ax.fill_between(xs, 0, ird_s, where=(ird_s >= 0), color="red",   alpha=0.3,
                label="No benefit (RD ≥ 0)")
ax.plot(xs, ird_s, color="black", lw=1.5)
ax.fill_between(xs, ird_lo_s, ird_hi_s, alpha=0.12, color="steelblue", label="95% CI")
ax.axhline(0, color="black", ls="--", lw=1)
ax.axhline(ird_mean.mean(), color="crimson", ls=":", lw=1.5,
           label=f"ATE = {ird_mean.mean():.3f}")
ax.set_ylabel("Treatment effect (absolute risk difference)")
ax.set_xlabel("Patients (sorted by estimated effect)")
ax.set_title("CATT Waterfall Plot", fontweight="bold")
ax.legend(fontsize=9)
fig.tight_layout(); plt.show()

print(f"Proportion with estimated benefit (RD < 0): {(ird_mean < 0).mean():.1%}")


# =============================================================================
# 9. CATERPILLAR PLOT
# =============================================================================
sort_idx_tau = np.argsort(tau_mean)
xs           = np.arange(len(tau_mean))

fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(xs, tau_ci_lo[sort_idx_tau], tau_ci_hi[sort_idx_tau],
                alpha=0.2, color="steelblue", label="95% CI (stochtree native)")
ax.fill_between(xs, tau_lo50[sort_idx_tau],  tau_hi50[sort_idx_tau],
                alpha=0.45, color="steelblue", label="50% CI")
ax.plot(xs, tau_mean[sort_idx_tau], lw=1.2, color="steelblue",
        label="Posterior mean τ(xᵢ)")
ax.axhline(0, color="gray", lw=1, ls="--")
ax.axhline(tau_mean.mean(), color="crimson", lw=1.5, ls=":",
           label=f"ATE = {tau_mean.mean():.3f}")
ax.set_xlabel("Unit (sorted by posterior mean τ)")
ax.set_ylabel("τ(xᵢ) — latent probit scale")
ax.set_title("Individual treatment effects with credible intervals", fontweight="bold")
ax.legend(fontsize=9)
fig.tight_layout(); plt.show()


# =============================================================================
# 10. TAU vs MU SCATTER
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 6))
vmax = np.abs(tau_mean).max()
sc = ax.scatter(mu_mean, tau_mean, c=tau_mean, cmap="RdBu_r",
                alpha=0.4, s=14, vmin=-vmax, vmax=vmax)
plt.colorbar(sc, ax=ax, label="τ(xᵢ)")
ax.scatter(mu_mean[treated_idx], tau_mean[treated_idx],
           s=16, edgecolors="darkorange", facecolors="none", lw=0.8,
           label="Treated", zorder=5)
ax.axhline(0, color="gray", lw=1, ls="--")
ax.set_xlabel("μ(xᵢ, π̂ᵢ) — prognostic score")
ax.set_ylabel("τ(xᵢ) — treatment effect")
ax.set_title("Treatment effect vs prognostic score", fontweight="bold")
ax.legend(fontsize=9)
fig.tight_layout(); plt.show()


# =============================================================================
# 11. HTE CALIBRATION
# =============================================================================
df_eval = pd.DataFrame({
    "pred_rd": ird_mean, "obs_y": y_data.astype(int), "T": T.astype(int)
})
df_eval["decile"] = pd.qcut(df_eval["pred_rd"], 10, labels=False, duplicates="drop")

def safe_rr(x):
    r_t = x[x["T"]==1]["obs_y"].mean()
    r_c = x[x["T"]==0]["obs_y"].mean()
    return r_t / r_c if (r_c > 0 and not np.isnan(r_t)) else np.nan

decile_rr = df_eval.groupby("decile").apply(safe_rr).reset_index()
decile_rr.columns = ["decile", "rate_ratio"]

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=decile_rr, x="decile", y="rate_ratio", palette="mako", ax=ax)
ax.axhline(1, color="red", ls="--", lw=1.5)
ax.set_xlabel("Predicted benefit decile (low → high)")
ax.set_ylabel("Observed rate ratio (treated / control)")
ax.set_title("HTE Calibration — rate ratios by predicted benefit decile", fontweight="bold")
fig.tight_layout(); plt.show()


# =============================================================================
# 12. SUBGROUP CATTs — FOREST PLOT
# =============================================================================
X_df = pd.DataFrame(X_train, columns=feature_names) if not isinstance(X_train, pd.DataFrame) \
       else X_train.copy()

def cate_summary(tau_draws, idx, level=0.94):
    draws = tau_draws[idx].mean(axis=0)
    lo    = np.percentile(draws, (1-level)/2*100)
    hi    = np.percentile(draws, (1+level)/2*100)
    return draws.mean(), lo, hi, np.percentile(draws,25), np.percentile(draws,75), \
           (draws < 0).mean(), len(idx)

feature_list = [
    "ICS_flag", "ICS_LABA_flag", "ICS_LABA_LAMA_flag",
    "num_treatments_baseline", "hba1c_missing", "dyslipidemia_flag",
    "PVD_flag", "CAD_flag", "t2dm_flag",
    "obstructive_sleep_apnea_flag", "hypertension_flag", "ckd_flag",
    "bmi_20_25", "bmi_above_25", "sex_Female",
]

rows = []
for feat in feature_list:
    if feat not in X_df.columns:
        continue
    idx = np.where(X_df[feat].values == 1)[0]
    if len(idx) < 10:
        continue
    m, lo, hi, lo50, hi50, p_b, n = cate_summary(tau_draws, idx)
    rows.append(dict(feature=feat, mean=m, lo94=lo, hi94=hi,
                     lo50=lo50, hi50=hi50, p_benefit=p_b, n=n))

df_forest = pd.DataFrame(rows).set_index("feature")
print(df_forest[["mean","lo94","hi94","p_benefit","n"]].round(3).to_string())

color = "#5B5EA6"
y_pos = np.arange(len(df_forest))[::-1]

fig, ax = plt.subplots(figsize=(10, 1.3*len(df_forest) + 1.5))
for i, (feat, row) in enumerate(df_forest.iterrows()):
    y = y_pos[i]
    ax.plot([row.lo94, row.hi94], [y,y], lw=1.5, color=color,
            alpha=0.5, solid_capstyle="round")
    ax.plot([row.lo50, row.hi50], [y,y], lw=5, color=color,
            alpha=0.7, solid_capstyle="round")
    ax.scatter(row["mean"], y, s=60, color=color, zorder=5,
               edgecolors="white", lw=0.8)
    ax.text(df_forest["hi94"].max() + 0.005, y,
            f"  P(benefit)={row.p_benefit:.2f}  n={row.n:,}",
            va="center", fontsize=8.5, color="gray")

ax.axvline(0, color="gray", lw=1, ls="--", alpha=0.7)
ax.set_yticks(y_pos); ax.set_yticklabels(df_forest.index, fontsize=10)
ax.set_xlabel("CATT — posterior mean τ  [latent probit scale]", fontsize=10)
ax.set_title("Subgroup CATTs with 50% and 94% HDI", fontsize=12, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); plt.show()


# =============================================================================
# 13. DECISION TREE PROXY + SHAP
# =============================================================================
X_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train

proxy_tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=50, random_state=123)
proxy_tree.fit(X_np, tau_mean)

fi_df = (pd.DataFrame({"feature": feature_names,
                        "importance": proxy_tree.feature_importances_})
           .sort_values("importance", ascending=False))
print("\nTop 10 features driving HTE:")
print(fi_df.head(10).reset_index(drop=True).to_string(index=False))

explainer   = shap.TreeExplainer(proxy_tree)
shap_values = explainer.shap_values(X_np)
shap.summary_plot(shap_values, X_np, feature_names=feature_names, max_display=15,
                  plot_title="SHAP — drivers of HTE")


# =============================================================================
# 14. SAVE OUTPUTS
# =============================================================================
os.makedirs(output_folder, exist_ok=True)

# Full model JSON — reload without re-running MCMC via BCFModel().from_json(s)
with open(f"{output_folder}bcf_{dx_name}_model.json", "w") as f:
    json.dump(bcf.to_json(), f)

np.savez_compressed(
    f"{output_folder}bcf_{dx_name}_posterior.npz",
    tau_draws=tau_draws, mu_draws=mu_draws, rd_draws=rd_draws, T=T, y=y_data
)

pd.DataFrame({
    "tau_mean" : tau_mean,  "tau_sd"   : tau_sd,
    "tau_lo95" : tau_lo95,  "tau_hi95" : tau_hi95,
    "tau_ci_lo": tau_ci_lo, "tau_ci_hi": tau_ci_hi,
    "rd_mean"  : ird_mean,  "rd_lo95"  : ird_lo95, "rd_hi95": ird_hi95,
    "mu_mean"  : mu_mean,   "yhat_mean": yhat_mean,
    "T"        : T.astype(int), "y": y_data.astype(int),
}).to_csv(f"{output_folder}bcf_{dx_name}_ite_summary.csv", index=False)

print(f"\nSaved model  : {output_folder}bcf_{dx_name}_model.json")
print(f"Saved draws  : {output_folder}bcf_{dx_name}_posterior.npz")
print(f"Saved summary: {output_folder}bcf_{dx_name}_ite_summary.csv")

# To reload later:
# with open(f"{output_folder}bcf_{dx_name}_model.json") as f:
#     s = json.load(f)
# bcf2 = BCFModel()
# bcf2.from_json(s)
# tau2 = bcf2.predict(X=X_train, Z=T, propensity=pi_hat, terms="cate", scale="probit")
