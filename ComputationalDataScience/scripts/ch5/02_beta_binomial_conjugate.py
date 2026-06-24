"""
STAT 418 · Chapter 5.2 — Beta-Binomial conjugate update in Python (slide-25 example).

The conjugate update is one line: a Beta(a0, b0) prior plus k successes in n trials
gives a Beta(a0 + k, b0 + n - k) posterior -- add successes to a0, failures to b0.
SciPy returns the exact posterior; PyMC (optional, below) reaches the same answer by
MCMC, which is the workflow you fall back on when no conjugate form exists.

DATA (real): Stephen Curry's FIRST 36 free throws of 2023-24 -- 34 of 36 (MLE 0.944);
vendored from basketball-reference, see BDA/DATA_CARD.md. We use this EARLY-season
sample (not the full 299/324 season) because §5.2 is about how the prior interacts
with the data, and a small n is where that interaction is visible. Free throws, not
three-pointers: the Beta-Binomial assumes ONE constant success probability across
exchangeable iid trials -- free throws satisfy that (fixed distance, no defender,
identical routine); 3PT difficulty varies shot to shot, so it would not.

With a flat Beta(1,1) prior the posterior is Beta(35, 3): mean 0.921, but heavily
LEFT-SKEWED (mode 0.944, hard against the upper boundary). That skew is exactly why
the 95% equal-tail interval and the 95% highest-density interval (HDI) DIFFER -- the
HDI shifts rightward toward the mode. (Symmetric posteriors -> the two coincide.)

Run:  python 02_beta_binomial_conjugate.py      # needs numpy, scipy (pymc optional)
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import ft_cumulative  # noqa: E402


def hdi(dist, mass=0.95, grid=200_000):
    """Highest-density interval of a scipy continuous dist: the NARROWEST interval
    holding `mass` probability. (The equal-tail interval instead chops (1-mass)/2 off
    each tail; for a skewed posterior the two differ.)"""
    xs = np.linspace(dist.ppf(1e-4), dist.ppf(1 - 1e-4), grid)
    cdf = dist.cdf(xs)
    best = None
    for i, lo in enumerate(xs):
        j = np.searchsorted(cdf, cdf[i] + mass)      # smallest hi with >= mass between
        if j >= len(xs):
            break
        if best is None or (xs[j] - lo) < (best[1] - best[0]):
            best = (lo, xs[j])
    return best


def main() -> None:
    # --- DATA: Curry's first 36 free throws, 2023-24 (real) ---
    k, n = ft_cumulative("Stephen Curry", min_attempts=30)   # (34, 36)
    a0, b0 = 1, 1                                             # flat prior
    print(f"Curry's first {n} FTA: {k}/{n} = {k / n:.3f}  (flat Beta(1,1) prior)\n")

    # --- Conjugate update: add successes to a0, failures to b0 ---
    a, b = a0 + k, b0 + (n - k)                               # Beta(35, 3)
    post = stats.beta(a, b)
    print(f"Posterior: Beta({a}, {b})")
    print(f"  mean {post.mean():.3f}   sd {post.std():.3f}   mode {(a - 1) / (a + b - 2):.3f}  (left-skewed)")

    # --- Two kinds of 95% interval -- they differ for a skewed posterior ---
    et_lo, et_hi = post.ppf(0.025), post.ppf(0.975)          # equal-tailed (SciPy)
    hd_lo, hd_hi = hdi(post, 0.95)                            # highest-density (narrowest)
    print(f"  95% equal-tail : [{et_lo:.3f}, {et_hi:.3f}]")
    print(f"  95% HDI        : [{hd_lo:.3f}, {hd_hi:.3f}]   <- shifted toward the mode")
    print(f"  P(theta > 0.90): {post.sf(0.90):.3f}")

    # --- Same model in PyMC (optional): MCMC reaches the exact answer ---
    try:
        import arviz as az
        import pymc as pm
        with pm.Model():
            theta = pm.Beta("theta", alpha=a0, beta=b0)
            pm.Binomial("y", n=n, p=theta, observed=k)
            idata = pm.sample(2000, chains=4, progressbar=False, random_seed=42)
        s = az.summary(idata, hdi_prob=0.95)
        print(f"\nPyMC (MCMC): mean {s['mean'].iloc[0]:.3f}, sd {s['sd'].iloc[0]:.3f} -- matches SciPy; "
              f"this is the workflow for the non-conjugate models in §5.5-5.6.")
    except Exception:
        print("\n[PyMC check skipped -- install pymc + arviz to run the MCMC version]")


if __name__ == "__main__":
    main()
