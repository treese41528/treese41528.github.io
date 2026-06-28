"""
STAT 418 · Chapter 3 — Thread C: a near-boundary binomial CI on real data (slide 45).

Backs §3.2 Slide 45 "Confidence Intervals": never use the Wald interval when p_hat is
near 0 or 1 -- it collapses or runs outside [0, 1]. Curry's 2023-24 OPENING game is a
naturally occurring boundary case: he went 7 for 7 from the line, so p_hat = 1 with
n = 7. The three intervals diverge dramatically:

  - Wald      [1.00, 1.00]  -- SE = sqrt(p_hat(1-p_hat)/n) = 0, so the interval has
                               ZERO width: a useless "we are 95% sure it is exactly 1".
  - Wilson    [0.646, 1.00] -- the score interval stays inside [0, 1] and is honest
                               about the uncertainty from only 7 attempts.
  - Profile/LRT[0.760, 1.00]-- invert the likelihood-ratio test; also respects [0, 1].

This replaces the generic textbook "x = 2, n = 20" with a real proportion on the
course's NBA free-throw thread; the Wald pathology is genuine, not contrived.

CAVEAT: free-throw attempts are treated as iid Bernoulli; n = 7 is a single game (high
variance, not a season estimate), and choosing a low-volume game is mild selection --
but Curry really did go 7/7 on opening night, so the boundary case is not fabricated.

Data: nba_freethrows_2023_24.csv already lives on the course Supabase under
Data/Chapter5 (the Ch5 free-throw thread vendored it); reused here -- no new upload.

Run:  python 03_freethrows_binomial_ci.py        # needs numpy, scipy, statsmodels
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import load_freethrows  # noqa: E402


def profile_lrt_interval(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% interval by inverting the likelihood-ratio test for a binomial proportion.

    Keep p with 2[l(p_hat) - l(p)] <= chi^2_{1, 1-alpha}. At x = n the log-likelihood is
    l(p) = x*log(p), l(p_hat=1) = 0, so the test reduces to -2 x log(p) <= c, giving the
    closed-form lower bound exp(-c / (2x)); the upper bound is the boundary 1.
    """
    c = stats.chi2.ppf(1 - alpha, 1)
    phat = x / n
    ll_max = (x * np.log(phat) if x > 0 else 0.0) + ((n - x) * np.log(1 - phat) if x < n else 0.0)

    def in_interval(p: float) -> bool:
        ll = (x * np.log(p) if x > 0 else 0.0) + ((n - x) * np.log(1 - p) if x < n else 0.0)
        return 2 * (ll_max - ll) <= c

    grid = np.linspace(1e-6, 1.0, 1_000_001)
    inside = [p for p in grid if in_interval(p)]
    return min(inside), max(inside)


def main() -> None:
    games = load_freethrows("Stephen Curry")
    g1 = games[0]                      # opening game of 2023-24
    x, n = g1["ft"], g1["fta"]
    phat = x / n
    print(f"Curry opening game {g1['date']}: ft = {x} / fta = {n}  ->  "
          f"p_hat = {phat:.2f},  n = {n}")

    # Wald (normal approximation) -- collapses at the boundary
    se = np.sqrt(phat * (1 - phat) / n)
    z = stats.norm.ppf(0.975)
    wald_lo, wald_hi = max(0.0, phat - z * se), min(1.0, phat + z * se)
    print(f"  Wald     95% CI: [{wald_lo:.3f}, {wald_hi:.3f}]   "
          f"(SE = {se:.3f}  ->  ZERO width, degenerate at the boundary)")

    # Wilson (score) interval -- respects [0, 1]
    wil_lo, wil_hi = proportion_confint(x, n, alpha=0.05, method="wilson")
    print(f"  Wilson   95% CI: [{wil_lo:.3f}, {wil_hi:.3f}]   (score interval, honest width)")

    # Profile-likelihood / LRT interval
    lrt_lo, lrt_hi = profile_lrt_interval(x, n)
    print(f"  Profile  95% CI: [{lrt_lo:.3f}, {lrt_hi:.3f}]   "
          f"(invert the LRT; closed form lower = e^(-chi2/(2x)) = {np.exp(-z**2 / (2 * x)):.3f})")


if __name__ == "__main__":
    main()
