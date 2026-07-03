"""
STAT 418 · Chapter 5 — §5.8 hierarchical shrinkage on real data (NBA team scoring).

A webbook companion to the Eight Schools example: same Normal hierarchy, but on a dataset
with a built-in OUT-OF-SAMPLE TARGET, so we can SHOW that partial pooling actually predicts
better. This is the basketball cousin of Efron & Morris's famous baseball demonstration.

Data (``NBA_data.csv``, 30 teams): each team's mean points per game over its first
``number_games_played`` (~9) games -- ``Y_bar``, a noisy early-season estimate -- plus its
``sample_sd`` and its mean over the REMAINING games, ``average_remaining_games`` (the held-out
truth we never let the model see).

Model (empirical-Bayes Normal-Normal, the closed form behind §5.8's shrinkage factor):
    Y_bar_j ~ N(theta_j, V_j),   V_j = sample_sd_j^2 / n_j      (early-season sampling variance)
    theta_j ~ N(mu, tau^2)                                       (between-team distribution)
mu and tau^2 are estimated from the 30 teams; the posterior mean of theta_j is the shrinkage
estimate theta_hat_j = mu + (1 - B_j)(Y_bar_j - mu) with B_j = V_j / (V_j + tau^2). (A full
PyMC hierarchical fit gives the same shrinkage; the closed form keeps the demo exact.)

Payoff: the shrunk estimates beat the raw early-season means at predicting each team's
rest-of-season average -- the hierarchical model borrows strength across teams and regresses
flukey hot/cold starts toward the league mean.

Run:  python 12_hierarchical_nba.py        # needs numpy, pandas
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import load_bda  # noqa: E402


def main() -> None:
    df = pd.DataFrame(load_bda("NBA_data.csv", chapter=5))   # Supabase Data/Chapter5
    for c in ("number_games_played", "Y_bar", "sample_sd", "average_remaining_games"):
        df[c] = pd.to_numeric(df[c])
    J = len(df)
    y = df["Y_bar"].to_numpy()
    n = df["number_games_played"].to_numpy()
    V = df["sample_sd"].to_numpy() ** 2 / n          # sampling variance of each early mean
    target = df["average_remaining_games"].to_numpy()  # held-out rest-of-season truth

    # Empirical Bayes: estimate mu (precision-weighted) and tau^2 (iterate to convergence)
    tau2 = np.var(y, ddof=1)
    for _ in range(200):
        w = 1.0 / (V + tau2)
        mu = np.sum(w * y) / np.sum(w)
        tau2 = max(1e-6, np.sum(w ** 2 * ((y - mu) ** 2 - V)) / np.sum(w ** 2))

    B = V / (V + tau2)                               # §5.8 shrinkage factor (toward mu)
    shrunk = mu + (1.0 - B) * (y - mu)               # hierarchical (posterior-mean) estimate

    sse_raw = np.sum((y - target) ** 2)
    sse_shr = np.sum((shrunk - target) ** 2)

    print(f"{J} NBA teams, first ~{int(round(n.mean()))} games each; league mean = {y.mean():.1f} pts")
    print(f"empirical Bayes:  mu = {mu:.1f},  tau = {np.sqrt(tau2):.2f},  "
          f"mean shrinkage B = {B.mean():.2f}")
    print(f"\nPredicting each team's REMAINING-season average:")
    print(f"  raw early means : total squared error = {sse_raw:.0f}")
    print(f"  shrunk estimates: total squared error = {sse_shr:.0f}")
    print(f"  -> shrinkage cuts prediction error by {100 * (1 - sse_shr / sse_raw):.0f}%")

    print("\nThe most extreme early starts (regressed toward the mean):")
    df = df.assign(shrunk=shrunk, B=B)
    extreme = df.iloc[np.argsort(-np.abs(y - y.mean()))[:4]]
    for _, r in extreme.iterrows():
        print(f"  {r['team']:11s} early {r['Y_bar']:6.1f}  ->  shrunk {r['shrunk']:6.1f}   "
              f"(actual rest of season {r['average_remaining_games']:6.1f})")


if __name__ == "__main__":
    main()
