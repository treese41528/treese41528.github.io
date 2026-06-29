"""
STAT 418 · Chapter 5.6 — Bayesian Poisson regression with a log link (slide 79).

Second GLM example (after logistic). A count outcome y_i with a log link and an exposure
offset:   y_i ~ Poisson( exp( x_i . beta + log t_i ) ).   Coefficients live on the
log-rate scale, so exp(beta_j) is a RATE RATIO: > 1 risk factor, < 1 protective, ~ 1 no
effect. Same PyMC workflow as every GLM: priors -> linear predictor -> link -> likelihood
-> sample.

DATA (synthetic): there is no clean real count dataset in BDA/, so the data are simulated
with KNOWN true effects (seed 42) so the recovered rate ratios tell a clear
risk / protective / null story we can check against ground truth:
  n = 400 observations, three standardized covariates with true coefficients
  beta = (+0.45, -0.31, 0.00), intercept 0.50, and a random exposure (person-time) offset.
At n = 400 the two real effects come out clearly (RR ~ 1.6 risk, ~ 0.7 protective) while the
null coefficient stays genuinely uncertain (rate-ratio interval comfortably straddling 1).

Run:  python 10_poisson_regression.py      # needs numpy, pymc, arviz
"""
from __future__ import annotations

import numpy as np

# --- Simulate count data with known effects (seed 42) ---
N = 400
TRUE_INTERCEPT = 0.5
TRUE_BETA = np.array([0.45, -0.31, 0.00])   # risk, protective, null (log-rate scale)


def simulate():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(N, 3))                          # 3 standardized covariates
    exposure = rng.uniform(0.5, 2.0, size=N)            # person-time per observation
    log_mu = TRUE_INTERCEPT + X @ TRUE_BETA + np.log(exposure)
    y = rng.poisson(np.exp(log_mu))                     # Poisson counts
    return X, exposure, y


def main() -> None:
    import arviz as az
    import pymc as pm

    X, exposure, counts = simulate()
    print(f"n = {N} counts; mean {counts.mean():.2f}, max {counts.max()}; "
          f"true beta = {TRUE_BETA.tolist()} (intercept {TRUE_INTERCEPT})\n")

    # --- Same PyMC pattern as any GLM: priors -> linear predictor -> link -> likelihood ---
    with pm.Model():
        intercept = pm.Normal("intercept", mu=0, sigma=5)
        beta = pm.Normal("beta", mu=0, sigma=2.5, shape=X.shape[1])
        # Log-rate with the exposure offset (log person-time), then exp() as the link.
        log_mu = intercept + pm.math.dot(X, beta) + np.log(exposure)
        pm.Poisson("y_obs", mu=pm.math.exp(log_mu), observed=counts)
        idata = pm.sample(2000, chains=4, target_accept=0.9,
                          random_seed=42, progressbar=False)

    s = az.summary(idata, var_names=["beta"], hdi_prob=0.95)
    print("Posterior (log-rate scale):")
    print(s[["mean", "sd", "hdi_2.5%", "hdi_97.5%"]].to_string())

    # --- Rate ratios: exponentiate the coefficient and its interval endpoints ---
    print("\nRate ratios  RR_j = exp(beta_j):")
    labels = {0: "risk factor", 1: "protective", 2: "null (CI covers 1)"}
    for j in range(X.shape[1]):
        m = np.exp(s["mean"].iloc[j])
        lo = np.exp(s["hdi_2.5%"].iloc[j])
        hi = np.exp(s["hdi_97.5%"].iloc[j])
        print(f"  RR[{j}] = {m:.2f}  ({lo:.2f}, {hi:.2f})   {labels[j]}")


if __name__ == "__main__":
    main()
