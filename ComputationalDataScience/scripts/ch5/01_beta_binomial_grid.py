"""
STAT 418 · Chapter 5.1 — Grid approximation of a posterior (the slide-13 example).

Grid approximation replaces the continuous posterior with a fine grid of theta
values carrying normalized weights. Its payoff: once you have the weights, EVERY
summary is a simple weighted sum (mean, variance, mode, credible interval via the
empirical CDF, exceedance probability) -- no per-quantity derivation.

We run it on a conjugate Beta-Binomial so there is a closed-form posterior to
validate against. The data are real: Stephen Curry's 2023-24 free throws (299 of
324). Free throws are used throughout the Beta-Binomial thread because they satisfy
the model's assumption of a constant success probability over exchangeable iid
trials (same distance, no defender, repeatable routine) -- unlike three-pointers.
With a flat Beta(1, 1) prior the exact posterior is Beta(300, 26); the grid matches
it, and P(theta > 0.90) ~ 0.91 answers "is Curry truly an elite (90%+) shooter?".

Run:  python 01_beta_binomial_grid.py        # needs numpy, scipy
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats
from scipy.special import logsumexp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import ft_cumulative  # noqa: E402


def grid_posterior(k, n, a0=1.0, b0=1.0, n_grid=1000):
    """Grid-approximate the Beta-Binomial posterior for k makes in n trials.

    Returns (grid, weights) where weights sum to 1. Done on the log scale for
    numerical stability, then normalized with logsumexp.
    """
    grid = np.linspace(1e-4, 1 - 1e-4, n_grid)
    log_unnorm = stats.beta.logpdf(grid, a0, b0) + stats.binom.logpmf(k, n, grid)
    weights = np.exp(log_unnorm - logsumexp(log_unnorm))
    return grid, weights


def summaries(grid, weights, thresh=0.90):
    """Every posterior summary as a weighted sum over the grid."""
    mean = np.sum(grid * weights)
    var = np.sum((grid - mean) ** 2 * weights)
    mode = grid[np.argmax(weights)]
    cdf = np.cumsum(weights)                      # empirical CDF for the credible interval
    lo, hi = grid[np.searchsorted(cdf, [0.025, 0.975])]
    p_exceed = weights[grid > thresh].sum()       # exceedance prob = mass past the threshold
    return {"mean": mean, "var": var, "mode": mode, "ci": (lo, hi), "p_exceed": p_exceed}


def main() -> None:
    # --- DATA --------------------------------------------------------------------
    # Stephen Curry's 2023-24 regular-season free throws, vendored from
    # basketball-reference (fetched 2026-06-22; see BDA/DATA_CARD.md and
    # BDA/nba_freethrows_2023_24.csv). Full season: k=299 makes of n=324 attempts,
    # so the MLE is 299/324 = 0.923. For context his CAREER mark is 4236/4645 = 0.912
    # (BDA/nba_freethrow_career.csv), so a single season near 0.92 is squarely in
    # character. We use the FULL season here (large n): the data dominate, so the
    # posterior sits on the MLE and the grid has a clean closed form to match. The
    # early-season cutoff (first ~30 FTA, 34/36) is used on the sensitivity slide,
    # where the small n lets the prior still move the posterior.
    #
    # WHY FREE THROWS, NOT THREE-POINTERS: the Beta-Binomial assumes ONE constant
    # success probability theta across exchangeable, iid Bernoulli trials. Free throws
    # satisfy that -- fixed 15-ft distance, no defender, identical routine each time --
    # so theta is genuinely constant. Three-point difficulty swings shot to shot (open
    # catch-and-shoot vs contested step-back, varying distance past the line), so theta
    # is NOT constant and the conjugate model would be misspecified on its own terms.
    k, n = ft_cumulative("Stephen Curry")          # (299, 324)

    # --- PRIOR -------------------------------------------------------------------
    # Flat Beta(1, 1) = Uniform(0, 1): no prior preference among free-throw rates.
    # With n = 324 the likelihood swamps it, so the posterior is driven by the data --
    # the cleanest setting to SEE the Bayesian update and to validate the grid against
    # the exact conjugate posterior Beta(1+k, 1+n-k) = Beta(300, 26).
    a0, b0 = 1.0, 1.0
    print(f"Curry 2023-24 free throws: {k}/{n} = {k / n:.3f}  (flat Beta(1,1) prior)")

    grid, w = grid_posterior(k, n, a0, b0)
    s = summaries(grid, w, thresh=0.90)

    # Exact conjugate posterior for validation: Beta(a0+k, b0+n-k)
    exact = stats.beta(a0 + k, b0 + (n - k))
    print(f"Exact posterior: Beta({a0 + k:.0f}, {b0 + (n - k):.0f})\n")
    print(f"  {'quantity':>14}  {'grid':>9}  {'exact':>9}")
    print(f"  {'mean':>14}  {s['mean']:>9.4f}  {exact.mean():>9.4f}")
    print(f"  {'variance':>14}  {s['var']:>9.5f}  {exact.var():>9.5f}")
    print(f"  {'mode':>14}  {s['mode']:>9.4f}  {(a0 + k - 1) / (a0 + b0 + n - 2):>9.4f}")
    print(f"  {'95% CI low':>14}  {s['ci'][0]:>9.4f}  {exact.ppf(0.025):>9.4f}")
    print(f"  {'95% CI high':>14}  {s['ci'][1]:>9.4f}  {exact.ppf(0.975):>9.4f}")
    print(f"  {'P(theta>0.90)':>14}  {s['p_exceed']:>9.4f}  {exact.sf(0.90):>9.4f}")
    print("\nGrid matches the closed form; P(theta>0.90) ~ 0.91 -> Curry is, with high "
          "posterior probability, a truly elite free-throw shooter.")


if __name__ == "__main__":
    main()
