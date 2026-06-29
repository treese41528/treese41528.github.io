"""
STAT 418 · Chapter 3 — Thread D: Gamma GLM on real data (Stroop reaction times).

Backs §3.5 Slide 120 "Gamma Applications" and the webbook Gamma-regression example.
Reaction times are strictly positive and right-skewed---a textbook Gamma response. With a
**log link**, exp(beta) is a multiplicative effect on the MEAN (a "mean ratio"):

  Gamma GLM (log link)  RT ~ condition  on the Stroop data (50 subjects, 3058 trials):
    Congruent mean ~ 563 ms, Incongruent mean ~ 619 ms,  exp(beta) ~ 1.10 (about 10% slower).

TWO load-bearing diagnostics that the example must NOT hide:
  1. The Gamma family assumes a CONSTANT coefficient of variation (CV = sd/mean). Here it is
     VIOLATED: CV ~ 0.35 (Congruent) vs ~ 0.53 (Incongruent). Show it as a diagnostic check.
  2. Trials are repeated WITHIN subject, so they are not iid. But because `condition` varies
     WITHIN each subject, cluster-robust SEs for the condition effect are ~ the naive SEs
     (the dependence inflates the INTERCEPT / grand-mean SE instead). State it that way --
     do NOT claim the condition SE is understated. (This corrects a draft caveat.)

Data: stroop_dataset.csv lives in the repo BDA/ folder; for students it must also be on the
course Supabase under Data/Chapter3 (loader resolves local first, Supabase otherwise).

Run:  python 04_stroop_gamma_glm.py        # needs numpy, pandas, statsmodels
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import load_bda  # noqa: E402


def main() -> None:
    rows = load_bda("stroop_dataset.csv", chapter=3)   # Supabase Data/Chapter3 (or local BDA)
    df = pd.DataFrame(rows)
    df["RT"] = pd.to_numeric(df["RT"], errors="coerce")
    df["subj"] = pd.to_numeric(df["subj"], errors="coerce")
    df = df.dropna(subset=["RT", "subj", "condition"])
    print(f"Stroop: {df['subj'].nunique()} subjects, {len(df)} trials")

    # Per-condition mean / sd / CV (the constant-CV diagnostic)
    for cond, s in df.groupby("condition")["RT"]:
        print(f"  {cond:12s} mean = {s.mean():6.1f} ms   sd = {s.std():6.1f}   "
              f"CV = {s.std() / s.mean():.3f}   n = {len(s)}")

    # Gamma GLM with log link -> exp(beta) is a mean ratio
    gamma = sm.families.Gamma(link=sm.families.links.Log())
    fit = smf.glm("RT ~ C(condition)", data=df, family=gamma).fit()
    b = fit.params["C(condition)[T.Incongruent]"]
    print(f"\nGamma GLM (log link)  RT ~ condition  (Congruent = reference):")
    print(f"  beta_Incongruent = {b:.4f}  ->  mean ratio exp(beta) = {np.exp(b):.3f}  "
          f"(~{(np.exp(b) - 1) * 100:.0f}% slower on incongruent trials)")

    # Naive vs cluster-by-subject SE (the corrected within-subject caveat)
    fit_cl = smf.glm("RT ~ C(condition)", data=df, family=gamma).fit(
        cov_type="cluster", cov_kwds={"groups": df["subj"]})
    se_c_naive = fit.bse["C(condition)[T.Incongruent]"]
    se_c_clust = fit_cl.bse["C(condition)[T.Incongruent]"]
    se_i_naive = fit.bse["Intercept"]
    se_i_clust = fit_cl.bse["Intercept"]
    print(f"\nWithin-subject dependence (repeated trials per subject):")
    print(f"  SE(condition): naive = {se_c_naive:.4f}, cluster = {se_c_clust:.4f}  "
          f"(ratio {se_c_clust / se_c_naive:.2f} -- condition is WITHIN-subject, barely changes)")
    print(f"  SE(intercept): naive = {se_i_naive:.4f}, cluster = {se_i_clust:.4f}  "
          f"(ratio {se_i_clust / se_i_naive:.2f} -- the grand-mean SE is what inflates)")


if __name__ == "__main__":
    main()
