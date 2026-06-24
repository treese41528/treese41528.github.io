"""
STAT 418 · Chapter 5.2 — Dirichlet-Multinomial conjugacy: election forecasting (slide 33).

The last of the five conjugate families. A poll of n = 500 voters records vote counts for
three candidates; the Dirichlet is the multinomial's conjugate prior, so the posterior is
again Dirichlet and the update is just "add the observed counts to the prior alphas":
  Dir(alpha0) prior  +  counts  ->  Dir(alpha0 + counts) posterior.

DATA (synthetic, illustrative -- NOT a real poll): vote counts (215, 180, 105) for
candidates A/B/C (n = 500), and a Dir(42, 38, 20) prior built from a previous election's
vote shares scaled to n0 = 100 pseudo-votes. Two teaching points:

  1. Data weight w = n / (n + n0) = 500 / 600 = 83.3% -- the same prior-pseudo-counts-vs-
     real-counts weighted average as every other conjugate family.

  2. A point estimate cannot answer a probabilistic question. Two DIFFERENT questions give
     very different answers, both computed by Monte Carlo over posterior draws (Chapter 2):
       - P(A wins a MAJORITY,  theta_A > 0.5)                ~ 0.0002
       - P(A wins a PLURALITY, theta_A > max(theta_B, theta_C)) ~ 0.96
     A almost surely gets the MOST votes, but almost surely NOT more than half. The MLE
     (0.43) answers neither; the full posterior does.

Marginal intervals: each component of a Dirichlet is marginally Beta(alpha_i, sum-alpha_i),
so a per-candidate 95% credible interval is a one-line Beta quantile call.

Run:  python 08_dirichlet_multinomial_election.py      # needs numpy, scipy (pymc optional)
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# --- Poll data and prior (synthetic) ---
CANDIDATES = ("A", "B", "C")
COUNTS = np.array([215, 180, 105])       # observed votes, n = 500
ALPHA0 = np.array([42.0, 38.0, 20.0])    # Dirichlet prior, n0 = 100 pseudo-votes


def main() -> None:
    n = int(COUNTS.sum())
    n0 = int(ALPHA0.sum())
    alpha_n = ALPHA0 + COUNTS                       # Dirichlet posterior concentration
    w = n / (n + n0)                                # data weight
    print(f"Poll n = {n} voters; prior n0 = {n0} pseudo-votes; "
          f"data weight w = n/(n+n0) = {w:.3f} ({w * 100:.1f}%)\n")

    # --- Per-candidate summaries (slide 33 table) ---
    prior_mean = ALPHA0 / n0
    mle = COUNTS / n
    post_mean = alpha_n / alpha_n.sum()
    print(f"{'cand':4s} {'a0':>4s} {'votes':>6s} {'a_n':>5s} "
          f"{'prior':>6s} {'MLE':>5s} {'post':>6s}   95% marginal interval")
    for i, c in enumerate(CANDIDATES):
        # A Dirichlet component is marginally Beta(alpha_i, sum(alpha) - alpha_i).
        marg = stats.beta(alpha_n[i], alpha_n.sum() - alpha_n[i])
        lo, hi = marg.ppf(0.025), marg.ppf(0.975)
        print(f"{c:4s} {ALPHA0[i]:4.0f} {COUNTS[i]:6d} {alpha_n[i]:5.0f} "
              f"{prior_mean[i]:6.3f} {mle[i]:5.3f} {post_mean[i]:6.3f}   "
              f"({lo:.3f}, {hi:.3f})")

    # --- Monte Carlo over posterior draws (slide 33 results panel) ---
    # Two DIFFERENT questions; the point estimate (MLE_A = 0.43) answers neither.
    rng = np.random.default_rng(42)
    draws = rng.dirichlet(alpha_n, size=100_000)             # (100000, 3) posterior samples
    p_majority = (draws[:, 0] > 0.5).mean()                  # theta_A > 0.5
    p_plurality = (draws[:, 0] > draws[:, 1:].max(axis=1)).mean()  # theta_A is the largest
    print("\nMonte Carlo (S = 100,000 Dirichlet draws, seed 42):")
    print(f"  P(A wins a MAJORITY,  theta_A > 0.5)                   = {p_majority:.4f}")
    print(f"  P(A wins a PLURALITY, theta_A > max(theta_B, theta_C)) = {p_plurality:.4f}")

    # --- Same model in PyMC (optional): MCMC reaches the same Dirichlet posterior ---
    try:
        import arviz as az
        import pymc as pm
        with pm.Model():
            theta = pm.Dirichlet("theta", a=ALPHA0)
            pm.Multinomial("y", n=n, p=theta, observed=COUNTS)
            idata = pm.sample(2000, chains=4, progressbar=False, random_seed=42)
        s = az.summary(idata, hdi_prob=0.95)
        print(f"\nPyMC (MCMC): posterior mean theta_A = {s['mean'].iloc[0]:.3f} "
              f"-- matches the analytical {post_mean[0]:.3f}; this is the workflow for the "
              f"non-conjugate models in §5.5-5.6.")
    except Exception:
        print("\n[PyMC check skipped -- install pymc + arviz to run the MCMC version]")


if __name__ == "__main__":
    main()
