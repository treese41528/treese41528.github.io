"""
STAT 418 · Chapter 3 — Thread B: logistic GLM on real data (Kaggle Titanic).

Backs the §3.5 GLM slides 96 (why OLS fails for binary y), 111 (odds ratios), and
114 (logistic in Python). ONE familiar dataset -- Survived (0/1) on 891 passengers --
threads the motivation, interpretation, and how-to:

  - Slide 96 (Problem with OLS):  the Linear Probability Model gives IMPOSSIBLE fitted
    probabilities. Survived ~ Fare alone predicts > 1; Survived ~ Pclass + Age + Fare
    predicts < 0. A reproducible demo of the structural failure (replaces asserted
    "y_hat = 1.3" scenarios with shown numbers).
  - Slide 111 (Odds Ratios):  exp(beta) as an adjusted odds ratio -- OR_female ~ 12.5
    (vs male), OR per Pclass level ~ 0.28, OR per year of Age ~ 0.96.
  - Slide 114 (Logistic in Python):  a real sm.GLM(..., Binomial()) fit whose
    odds-ratio extraction produces meaningful numbers (replaces a rng(42) simulation).

CAVEATS (teach them, do not hide them): Age is ~20% missing (177 rows) and NOT MCAR
(missing-Age passengers had lower survival), so listwise deletion drops n to 714 and
introduces mild selection bias; the data are observational, so odds ratios are
associational, not causal; statsmodels uses FEMALE as the reference category, so the
sex odds ratio is reported as "female vs male".

Data: titanic.csv lives on the course Supabase under Data/Chapter3 (the Ch4
bootstrap thread vendored it); reused here, so no new upload is needed.

Run:  python 02_titanic_logistic.py        # needs numpy, pandas, statsmodels
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import load_titanic  # noqa: E402


def main() -> None:
    rows = load_titanic(chapter=3, filename="titanic.csv")   # Supabase Data/Chapter3
    df = pd.DataFrame(rows)
    for c in ("Survived", "Pclass", "Age", "Fare"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n = len(df)
    n_age_missing = int(df["Age"].isna().sum())
    print(f"=== Titanic: n = {n}, Age missing = {n_age_missing} "
          f"({100 * n_age_missing / n:.1f}%) ===")

    # ---- Slide 96: the Linear Probability Model gives impossible probabilities ----
    print("\n=== [Slide 96] OLS (LPM) fails structurally for binary y ===")
    lpm1 = smf.ols("Survived ~ Fare", data=df).fit()
    fit1 = lpm1.fittedvalues
    print(f"  LPM  Survived ~ Fare            : max p_hat = {fit1.max():.2f}, "
          f"#(p_hat > 1) = {(fit1 > 1).sum()}  (probabilities above 1)")
    df_age = df.dropna(subset=["Age"]).copy()
    lpm2 = smf.ols("Survived ~ Pclass + Age + Fare", data=df_age).fit()
    fit2 = lpm2.fittedvalues
    print(f"  LPM  Survived ~ Pclass+Age+Fare : min p_hat = {fit2.min():.2f}, "
          f"#(p_hat < 0) = {(fit2 < 0).sum()}  (probabilities below 0; n = {len(df_age)})")

    # ---- Slides 111 / 114: logistic regression + odds ratios ----
    print("\n=== [Slides 111 / 114] logistic  Survived ~ Sex + Pclass + Age "
          f"(n = {len(df_age)}) ===")
    logit = smf.logit("Survived ~ C(Sex) + Pclass + Age", data=df_age).fit(disp=0)
    b_male = logit.params["C(Sex)[T.male]"]   # female is the reference level
    or_female = np.exp(-b_male)               # female vs male
    or_pclass = np.exp(logit.params["Pclass"])
    or_age = np.exp(logit.params["Age"])
    print(f"  beta_male = {b_male:+.3f}  ->  OR female vs male = e^{-b_male:.3f} "
          f"= {or_female:.1f}")
    print(f"  beta_Pclass = {logit.params['Pclass']:+.3f}  ->  OR per class = {or_pclass:.3f}")
    print(f"  beta_Age = {logit.params['Age']:+.4f}  ->  OR per year = {or_age:.3f}")
    print(f"  (converged = {logit.mle_retvals['converged']}, no separation)")


if __name__ == "__main__":
    main()
