"""
STAT 418 · Chapter 3 — Thread A: OLS inference on real data (ISLR Advertising).

Backs the §3.4 inference slides 85 (t-tests), 86 (F-tests), 92 (statsmodels output).
ONE fit -- Sales ~ TV + Radio + Newspaper (n = 200 markets) -- threads all three:

  - Slide 85 (t-tests):  individual H0: beta_j = 0. Newspaper FAILS to reject
    (t = -0.18, p = 0.86) -- a memorable real fail-to-reject, the opposite of the
    fabricated "all significant" toy.
  - Slide 86 (F-tests):  the reduced model Sales ~ TV (R^2 = 0.61) vs the full model
    (R^2 = 0.90); the joint test that Radio AND Newspaper add nothing (F = 272, p ~ 0);
    and the F = t^2 identity for the single-restriction drop-Newspaper test.
  - Slide 92 (statsmodels):  one real res.summary() -- map .params/.bse/.tvalues/
    .pvalues/.rsquared/.fvalue to concrete numbers.

Confounding hook (reinforces Slide 69 "holding the others fixed"): Newspaper IS
significant on its own (simple-regression t = 3.30) but vanishes in the full model,
because Newspaper is correlated with Radio (corr = 0.35).

Data: Advertising.csv lives on the course Supabase under Data/Chapter3 (the
Ch4 bootstrap thread vendored it); we reuse that copy here, so no new upload is needed.

Run:  python 01_advertising_ols.py        # needs numpy, pandas, statsmodels
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import load_bda  # noqa: E402


def main() -> None:
    rows = load_bda("Advertising.csv", chapter=3)   # Supabase Data/Chapter3/Advertising.csv
    df = pd.DataFrame(rows)
    for c in ("TV", "Radio", "Newspaper", "Sales"):
        df[c] = df[c].astype(float)
    n = len(df)

    full = smf.ols("Sales ~ TV + Radio + Newspaper", data=df).fit()

    # ---- Slide 92: the statsmodels output map ----
    print(f"=== [Slide 92] full OLS  Sales ~ TV + Radio + Newspaper   (n = {n}) ===")
    print(f"  const     = {full.params['Intercept']:+.4f}")
    print(f"  TV        = {full.params['TV']:+.4f}")
    print(f"  Radio     = {full.params['Radio']:+.4f}")
    print(f"  Newspaper = {full.params['Newspaper']:+.4f}")
    print(f"  R^2 = {full.rsquared:.4f}   F = {full.fvalue:.1f} "
          f"(df {int(full.df_model)}, {int(full.df_resid)})   n = {int(full.nobs)}")

    # ---- Slide 85: individual t-tests, H0: beta_j = 0 ----
    print("\n=== [Slide 85] individual t-tests  (H0: beta_j = 0) ===")
    for v in ("TV", "Radio", "Newspaper"):
        verdict = "fail to reject" if full.pvalues[v] > 0.05 else "reject"
        print(f"  {v:9s}: t = {full.tvalues[v]:+6.2f}   p = {full.pvalues[v]:.4f}   -> {verdict}")

    # ---- Slide 86: F-tests (nested model comparison) ----
    print("\n=== [Slide 86] F-tests (nested model comparison) ===")
    simple_tv = smf.ols("Sales ~ TV", data=df).fit()
    print(f"  reduced  Sales ~ TV          R^2 = {simple_tv.rsquared:.3f}")
    print(f"  full     Sales ~ TV+Rad+News R^2 = {full.rsquared:.3f}")
    joint = full.f_test("Radio = 0, Newspaper = 0")
    print(f"  joint  H0: beta_Radio = beta_Newspaper = 0  ->  "
          f"F = {float(joint.fvalue):.1f}  (df {int(joint.df_num)}, {int(joint.df_denom)})  "
          f"p = {float(joint.pvalue):.1e}")
    drop_news = full.f_test("Newspaper = 0")
    t_news_sq = full.tvalues["Newspaper"] ** 2
    print(f"  single  H0: beta_Newspaper = 0  ->  F = {float(drop_news.fvalue):.4f}  "
          f"= t^2 = {t_news_sq:.4f}   (F = t^2 for a single restriction)")

    # ---- confounding hook (Slide 69 callback) ----
    simple_news = smf.ols("Sales ~ Newspaper", data=df).fit()
    print("\n=== confounding hook (holding the others fixed) ===")
    print(f"  Newspaper ALONE : t = {simple_news.tvalues['Newspaper']:.2f}  "
          f"p = {simple_news.pvalues['Newspaper']:.4f}  (significant!)")
    print(f"  Newspaper in FULL model : t = {full.tvalues['Newspaper']:.2f}  "
          f"p = {full.pvalues['Newspaper']:.2f}  (vanishes)")
    print(f"  corr(Radio, Newspaper) = {df['Radio'].corr(df['Newspaper']):.3f}  "
          f"-> Newspaper rode on Radio's effect")


if __name__ == "__main__":
    main()
