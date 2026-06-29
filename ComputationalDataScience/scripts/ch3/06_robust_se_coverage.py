"""
STAT 418 · Chapter 3 — Slide 90: robust (sandwich) SE coverage, seeded & reproducible.

Backs the §3.4 "Robust Standard Errors" coverage panel. Under heteroskedasticity the
classical OLS standard errors are biased, so 95% confidence intervals built from them
UNDER-cover; the HC3 sandwich SEs restore close-to-nominal coverage. This is a Monte
Carlo *coverage study*, so it needs a KNOWN true beta -- hence a simulation, not real
data (real data never reveals the truth a CI is supposed to trap).

DGP (the slide's):  y_i = 2 + 3 x_i + e_i,  e_i ~ N(0, sigma_i^2) with sigma_i = (0.5 + x_i)^2,
x_i ~ U(0,3), n = 100, B = 10,000 replications, seed = 42 (the squared SD makes the
heteroskedasticity strong enough that the OLS undercoverage is visible). For each
replication we form the 95% CI for the slope under (a) classical OLS SEs and (b) HC3
sandwich SEs, and record whether it covers the true slope beta_1 = 3.

OLS and HC3 are computed in closed form (fast enough for 10,000 fits):
  SE_OLS(b)^2 = s^2 (X'X)^{-1},        s^2 = e'e/(n-2)
  HC3 sandwich = (X'X)^{-1} X' diag(e_i^2/(1-h_ii)^2) X (X'X)^{-1},  h = diag(X(X'X)^{-1}X')

Run:  python 06_robust_se_coverage.py        # needs numpy, scipy
"""
from __future__ import annotations

import numpy as np
from scipy import stats

SEED = 42
B = 10_000
N = 100
BETA = np.array([2.0, 3.0])   # intercept, slope


def main() -> None:
    rng = np.random.default_rng(SEED)
    tcrit = stats.t.ppf(0.975, N - 2)
    cover_ols = 0
    cover_hc3 = 0
    for _ in range(B):
        x = rng.uniform(0.0, 3.0, N)
        sd = (0.5 + x) ** 2                            # heteroskedastic: SD grows steeply with x
        y = BETA[0] + BETA[1] * x + rng.normal(0.0, sd)

        X = np.column_stack([np.ones(N), x])
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        e = y - X @ beta

        # classical OLS SE of the slope
        s2 = (e @ e) / (N - 2)
        se_ols = np.sqrt(s2 * XtX_inv[1, 1])

        # HC3 sandwich SE of the slope
        h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)     # leverages h_ii
        meat = X.T @ (X * (e ** 2 / (1 - h) ** 2)[:, None])
        sandwich = XtX_inv @ meat @ XtX_inv
        se_hc3 = np.sqrt(sandwich[1, 1])

        if abs(beta[1] - BETA[1]) <= tcrit * se_ols:
            cover_ols += 1
        if abs(beta[1] - BETA[1]) <= tcrit * se_hc3:
            cover_hc3 += 1

    print(f"DGP: y = 2 + 3x + e,  e ~ N(0, sigma^2), sigma=(0.5+x)^2,  x ~ U(0,3);  "
          f"n={N}, B={B:,}, seed={SEED}")
    print(f"  classical OLS SE : 95% CI coverage of beta_1 = {100 * cover_ols / B:.1f}%")
    print(f"  HC3 sandwich SE  : 95% CI coverage of beta_1 = {100 * cover_hc3 / B:.1f}%")
    print(f"  nominal target   : 95.0%")
    print(f"  -> OLS under-covers by {95.0 - 100 * cover_ols / B:.1f} pp; HC3 restores near-nominal coverage.")


if __name__ == "__main__":
    main()
