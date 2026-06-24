"""
STAT 418 · Chapter 5.2 — Weakly informative priors: the standardize-first recipe (slide 36).

A weakly informative prior regularizes (rules out absurd values) while staying vague enough
that a reasonable amount of data dominates. The practical recipe (Gelman et al. 2008; Gabry
et al. 2019) is: STANDARDIZE outcome and predictors first, then put the SAME default priors
on the standardized coefficients -- and CHECK with a prior predictive simulation.

EXAMPLE (synthetic, illustrative): exam score y (mean 70, SD 15) ~ study hours x (mean 10,
SD 5). A realistic tutoring effect is ~3 points/hour. This script shows the two things the
slide claims, by computation rather than assertion:

  1. Effect-size mapping. A raw 3 pt/hr slope is, on the standardized scale,
        beta* = beta_raw * SD_x / SD_y = 3 * 5 / 15 = 1.0   (one SD-of-y per SD-of-x).

  2. The SAME N(0, 2.5) prior is appropriate standardized but WRONG raw. Standardized, its
     95% range is +/- ~5 (in SD_y per SD_x), i.e. +/- ~15 raw pts/hr -- a genuine 3 pt/hr
     effect (beta* = 1.0) sits comfortably inside, while an absurd 15 pt/hr is at the edge.
     On the RAW slope the same N(0, 2.5) allows only +/- ~5 pts/hr, so 3 pt/hr is already in
     the tail. The prior is identical; the SCALE makes it reasonable or not. A prior
     predictive simulation (draw params from the prior, simulate data) is how you catch this.

Run:  python 09_weakly_informative_priors.py      # needs numpy, scipy (pymc optional)
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# --- Synthetic problem constants (no data file; the point is the PRIOR logic) ---
SD_Y, SD_X = 15.0, 5.0          # outcome / predictor scales (exam points, study hours)
REAL_SLOPE = 3.0                # a realistic tutoring effect, points per hour
PRIOR_SD_BETA = 2.5            # the default weakly-informative prior SD on the slope


def raw_to_std(beta_raw: float) -> float:
    """Standardized slope = raw slope rescaled by the predictor/outcome SD ratio."""
    return beta_raw * SD_X / SD_Y


def std_to_raw(beta_std: float) -> float:
    """Inverse map: standardized slope back to raw points-per-hour."""
    return beta_std * SD_Y / SD_X


def main() -> None:
    # --- 1. Effect-size mapping: a real 3 pt/hr effect on the standardized scale ---
    beta_star = raw_to_std(REAL_SLOPE)
    print(f"Real effect {REAL_SLOPE:.0f} pt/hr  ->  standardized beta* = "
          f"{REAL_SLOPE:.0f} * {SD_X:.0f} / {SD_Y:.0f} = {beta_star:.1f} "
          f"(SD of outcome per SD of predictor)\n")

    # --- 2. The same N(0, 2.5) prior on two different scales ---
    z = stats.norm.ppf(0.975)                          # 1.96
    half95 = z * PRIOR_SD_BETA                          # ~4.9 ("+/- ~5")
    print(f"Prior beta ~ N(0, {PRIOR_SD_BETA}):  95% within +/- {half95:.2f}\n")

    print("  On the STANDARDIZED slope:")
    print(f"    +/- {half95:.2f} SD_y per SD_x  ->  +/- {std_to_raw(half95):.1f} raw pt/hr")
    print(f"    genuine effect beta* = {beta_star:.1f} is well inside; "
          f"an absurd 15 pt/hr (beta* = {raw_to_std(15):.1f}) sits at the edge -> good.\n")

    print("  On the RAW slope (no standardizing):")
    print(f"    +/- {half95:.2f} raw pt/hr only  ->  a real {REAL_SLOPE:.0f} pt/hr effect is "
          f"at the {stats.norm.sf(REAL_SLOPE / PRIOR_SD_BETA) * 100:.0f}% tail -> too tight.\n")

    # --- 3. Prior predictive on the quantity of interest (Gabry et al. 2019) ---
    # The check that matters here is on the TUTORING EFFECT itself: draw the slope from its
    # prior, convert to raw points-per-hour, and confirm the prior spans realistic effects
    # while putting little mass on absurd ones. (Checking on the implied EFFECT, not the raw
    # score level, keeps the diagnostic on the bounded, interpretable scale.)
    rng = np.random.default_rng(42)
    effect_raw = rng.normal(0, PRIOR_SD_BETA, 50_000) * SD_Y / SD_X   # b* -> raw pt/hr
    lo, hi = np.percentile(effect_raw, [2.5, 97.5])
    p_absurd = np.mean(np.abs(effect_raw) > 15)                       # |effect| > 15 pt/hr
    print("Prior predictive on the tutoring effect (raw points/hour):")
    print(f"    95% within [{lo:.1f}, {hi:.1f}] pt/hr;  P(|effect| > 15 pt/hr) = {p_absurd:.2f}")
    print(f"    -> spans realistic effects (incl. {REAL_SLOPE:.0f} pt/hr) while ruling out the absurd.")

    # --- Optional: draw the same prior in PyMC and confirm it agrees ---
    try:
        import pymc as pm
        with pm.Model():
            b_ = pm.Normal("b", 0, PRIOR_SD_BETA)
            pp = pm.sample_prior_predictive(samples=50_000, random_seed=42)
        eff = pp.prior["b"].values.ravel() * SD_Y / SD_X
        print(f"    [PyMC prior: 95% within "
              f"[{np.percentile(eff, 2.5):.1f}, {np.percentile(eff, 97.5):.1f}] pt/hr -- matches]")
    except Exception:
        print("    [PyMC check skipped -- install pymc to run sample_prior_predictive]")


if __name__ == "__main__":
    main()
