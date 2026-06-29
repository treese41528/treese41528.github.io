"""
STAT 418 · Chapter 5.6 — The PyMC -> InferenceData -> ArviZ workflow (slides 74-75).

Slides 74 and 75 are the PyMC reference pair: 74 shows how a declarative model becomes a
computation graph with three node types and how pm.sample() drives NUTS; 75 shows what
pm.sample() RETURNS (an InferenceData object) and how ArviZ reads it. This one script walks
the whole pipeline end to end on a tiny known-answer model, so every concept on those two
slides is something a student can run and inspect:

  build model (free / deterministic / observed nodes)  ->  sample (NUTS)  ->  InferenceData
  ->  az.summary diagnostics (ESS, R-hat, MCSE)  ->  divergence check / interpret-or-stop.

DATA (synthetic, known answer): y ~ Normal(mu=5, sigma=2), n = 400 (seed 42), so the
posterior recovers the sample mean and sd (close to the true mu = 5, sigma = 2) -- you see
the diagnostics on a model whose right answer you already know.

Run:  python 11_pymc_arviz_workflow.py      # needs numpy, pymc, arviz
"""
from __future__ import annotations

import numpy as np

TRUE_MU, TRUE_SIGMA, N = 5.0, 2.0, 400


def main() -> None:
    import arviz as az
    import pymc as pm

    rng = np.random.default_rng(42)
    y_data = rng.normal(TRUE_MU, TRUE_SIGMA, size=N)
    print(f"Data: y ~ Normal({TRUE_MU}, {TRUE_SIGMA}), n = {N};  "
          f"sample mean {y_data.mean():.2f}, sd {y_data.std():.2f}\n")

    # --- Slide 74: declarative model -> computation graph with three node types ---
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=10)                     # free node (sampled by NUTS)
        sigma = pm.HalfNormal("sigma", sigma=5)                  # free node (positive -> log-transformed)
        pm.Deterministic("precision", 1 / sigma**2)             # deterministic (computed + tracked)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data)  # observed node (fixed to data)

    # PyMC assigns the node TYPE from the API call: no `observed=` -> free; Deterministic -> tracked.
    print("Computation-graph nodes (slide 74):")
    print(f"  free (sampled)  : {[v.name for v in model.free_RVs]}")
    print(f"  deterministic   : {[v.name for v in model.deterministics]}")
    print(f"  observed (data) : {[v.name for v in model.observed_RVs]}\n")

    # --- Slide 74: pm.sample() drives NUTS (defaults shown; override via kwargs) ---
    with model:
        idata = pm.sample(1000, tune=1000, chains=4, target_accept=0.9,
                          random_seed=42, progressbar=False)

    # --- Slide 75: pm.sample() returns an InferenceData object (xarray groups) ---
    print("InferenceData groups (slide 75):", list(idata.groups()))
    post = idata.posterior
    print(f"  posterior dims: chains = {post.sizes['chain']}, draws = {post.sizes['draw']}\n")

    # --- Slide 75: az.summary() is the first thing to check (ESS, R-hat, MCSE) ---
    s = az.summary(idata, var_names=["mu", "sigma"], hdi_prob=0.95)
    print("az.summary(idata, hdi_prob=0.95):")
    print(s[["mean", "sd", "hdi_2.5%", "hdi_97.5%", "mcse_mean",
             "ess_bulk", "ess_tail", "r_hat"]].to_string())

    # --- The rule: if diagnostics fail, do NOT interpret -- fix the sampler first ---
    n_div = int(idata.sample_stats["diverging"].sum())
    ok = bool((s["r_hat"] < 1.01).all() and (s["ess_bulk"] > 400).all() and n_div == 0)
    print(f"\nDivergences: {n_div};  all R-hat < 1.01 and ESS_bulk > 400: {ok}  ->  "
          f"{'safe to interpret' if ok else 'STOP, fix the sampler'}")


if __name__ == "__main__":
    main()
