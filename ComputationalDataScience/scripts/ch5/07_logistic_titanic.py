"""
STAT 418 · Chapter 5.6 — Bayesian logistic regression on the Titanic (slides 76/77/78).

The same model coded by hand with Metropolis-Hastings on Slide 64, now in PyMC, fit to
real data: did a passenger survive? We regress

    Survived ~ Sex + Pclass + Age        (the Ch3 §3.5 logistic-GLM family; frequentist MLE there, full posteriors here)

on the 714 passengers with a recorded age (177 of the 891 have a blank Age; we use
complete cases). Weakly informative Normal(0, 10) priors on every coefficient, so the
posterior is data-driven and the posterior means track the frequentist MLE.

Design matrix columns: [const, male (1 if male), Pclass (1-3), Age (years)].
Odds ratio for coefficient j is exp(beta_j): the multiplicative change in the odds of
survival per unit increase in that predictor.

Run:  python 07_logistic_titanic.py        # needs pandas, numpy, pymc, arviz
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import load_titanic  # noqa: E402

NAMES = ["const", "male", "Pclass", "Age"]


def design_matrix():
    """Complete-case design matrix and survival outcome for the Titanic logistic model."""
    df = pd.DataFrame(load_titanic())[["Survived", "Sex", "Pclass", "Age"]].copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")          # blanks -> NaN
    df = df.dropna(subset=["Age"])                                  # 891 -> 714 complete cases
    X = np.column_stack([
        np.ones(len(df)),                                           # const
        (df["Sex"] == "male").astype(float),                        # male indicator
        df["Pclass"].astype(float),                                 # passenger class 1-3
        df["Age"].astype(float),                                    # age in years
    ])
    y = df["Survived"].astype(int).to_numpy()
    return X, y


def main() -> None:
    X, y = design_matrix()
    n = len(y)
    print(f"complete-case n={n}; survived {y.sum()}/{n} = {y.mean():.3f}\n")

    # Bayesian logistic regression: y ~ Bernoulli(sigmoid(X beta)), Normal(0, 10) priors.
    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=10, shape=X.shape[1])
        eta = pm.math.dot(X, beta)
        pm.Bernoulli("y_obs", p=pm.math.sigmoid(eta), observed=y)
        idata = pm.sample(2000, tune=3000, chains=4, random_seed=42,
                          target_accept=0.99, progressbar=False)

    # --- Slide 77: convergence summary (one row per coefficient) ---
    summ = az.summary(idata, hdi_prob=0.95)
    summ.index = NAMES
    print("[77] az.summary (Survived ~ male + Pclass + Age)")
    print(summ[["mean", "sd", "hdi_2.5%", "hdi_97.5%", "ess_bulk", "r_hat"]].to_string())
    divs = int(idata.sample_stats["diverging"].values.sum())
    print(f"   divergences: {divs}\n")

    # --- Slide 78: posterior odds ratios exp(beta) with 95% credible intervals ---
    print("[78] posterior odds ratios  OR_j = exp(beta_j)")
    draws = idata.posterior["beta"].values.reshape(-1, X.shape[1])
    for j, name in enumerate(NAMES):
        if name == "const":
            continue
        orj = np.exp(draws[:, j])
        med, lo, hi = np.median(orj), np.percentile(orj, 2.5), np.percentile(orj, 97.5)
        note = "lowers survival odds" if hi < 1 else ("raises survival odds" if lo > 1 else "uncertain")
        print(f"   {name:7s} OR median {med:.3f}  95% [{lo:.3f}, {hi:.3f}]  ({note})")


if __name__ == "__main__":
    main()
