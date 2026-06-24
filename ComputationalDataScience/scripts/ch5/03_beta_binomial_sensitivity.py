"""
STAT 418 · Chapter 5.2 — Prior sensitivity for the Beta-Binomial (slide-26 example).

How much a prior moves the posterior is governed NOT by "small n" but by the prior's
strength n0 = a0 + b0 relative to the data n, through the data weight

    w = n / (n + n0),    posterior mean = w * MLE + (1 - w) * prior_mean,

with posterior precision ~ (n + n0). Same data, different n0 -> different conclusions;
the n-slider (early vs full season) changes n, hence w, for a FIXED prior.

DATA (real): Curry's first 36 free throws of 2023-24 (34/36, MLE 0.944), with the full
season (299/324) shown for contrast (see BDA/DATA_CARD.md). Four priors spanning n0
from 1 to 4321:
  - Flat Beta(1,1), Jeffreys Beta(.5,.5): n0 ~ 1-2 -> w ~ 1 at any n -> invisible.
  - Skeptical Beta(28,8): league-average mean 0.78, n0 = 36. At the early n it is
    halfway (w = 0.5, drags the hot start to ~0.86) -- but that pull is INDEFENSIBLE
    for Curry (career 0.91); it is the right default only for an UNKNOWN shooter, and
    as data accumulate (full season) it is correctly overwhelmed (w -> 0.90).
  - Career Beta(3937,384): Curry's prior-season record, n0 = 4321. w stays tiny
    (0.008 early, 0.07 full) -- it dominates and pins the SD near 0.004. A defensible
    strong prior, because it is backed by 4321 real prior attempts.

Lesson: set n0 to the prior EVIDENCE you actually have. The career prior earns its n0;
generic skepticism does not.

Run:  python 03_beta_binomial_sensitivity.py     # needs numpy, scipy
"""
from __future__ import annotations

import os
import sys

from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import ft_cumulative, load_freethrow_career, load_freethrows  # noqa: E402


def build_priors():
    """The four (alpha0, beta0) priors. n0 = alpha0 + beta0 is the prior strength."""
    career = load_freethrow_career()["Stephen Curry"]
    kf, nf = ft_cumulative("Stephen Curry")                 # 2023-24 full season
    prior_mk = career["ft"] - kf                             # career makes BEFORE 2023-24
    prior_at = career["fta"] - nf                            # career attempts before 2023-24
    return {
        "Flat (.50)":      (1.0, 1.0),                       # n0 = 2
        "Jeffreys (.50)":  (0.5, 0.5),                       # n0 = 1
        "Skeptical (.78)": (28.0, 8.0),                      # league average, n0 = 36
        "Career (.91)":    (float(prior_mk), float(prior_at - prior_mk)),  # n0 = 4321
    }


def compare(k, n, label, priors):
    """Print posterior mean/sd/P(>.90) and the data weight w for each prior."""
    print(f"\n{label}: data {k}/{n}, MLE = {k / n:.3f}")
    print(f"  {'prior':16s} {'n0':>6} {'w=n/(n+n0)':>11} {'mean':>7} {'sd':>7} {'P(>.90)':>8}")
    for name, (a0, b0) in priors.items():
        n0 = a0 + b0
        w = n / (n + n0)                                     # data's share of total evidence
        post = stats.beta(a0 + k, b0 + (n - k))
        print(f"  {name:16s} {n0:>6.0f} {w:>11.3f} {post.mean():>7.3f} {post.std():>7.3f} {post.sf(0.90):>8.2f}")


def main() -> None:
    priors = build_priors()
    # Early-season sample: small n, where a comparable-n0 prior bites.
    g = next(x for x in load_freethrows("Stephen Curry") if x["cum_fta"] >= 30)
    compare(g["cum_ft"], g["cum_fta"], "EARLY (first 36 FTA)", priors)
    # Full season: the SAME fixed priors, now against far more data.
    kf, nf = ft_cumulative("Stephen Curry")
    compare(kf, nf, "FULL season", priors)
    print("\nThe fixed skeptical prior (n0=36) is halfway at n=36 (w=.5) but overwhelmed at "
          "n=324 (w=.9); the career prior (n0=4321) dominates at both. It is n0 vs n, via w.")


if __name__ == "__main__":
    main()
