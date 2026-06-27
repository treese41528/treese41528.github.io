"""
STAT 418 · Chapter 4 — Regression bootstrap on real heteroscedastic data (slides 36, 37, 63).

ISLR Advertising (n = 200). Sales ~ TV is genuinely heteroscedastic: corr(|residual|, TV)
is clearly positive, so the variance of Sales grows with the TV budget. That makes it the
ideal real example for the residual / pairs / wild trio and for a regression coefficient test.

  - Slides 36/37: bootstrap SE of the slope b1 under residual, pairs, and wild resampling.
    The residual bootstrap UNDERSTATES the SE (it assumes homoscedasticity); pairs and wild
    correctly inflate it. The gap is modest (~6%) because the slope is robust -- the variance
    fan is the real evidence of heteroscedasticity, not the SE gap alone.
  - Slide 63: test H0: beta_Newspaper = 0 in Sales ~ TV + Radio + Newspaper via a
    null-enforced WILD bootstrap (valid under heteroscedasticity, predictors fixed).

Run:  python 03_regression_bootstrap_advertising.py        # needs numpy, scipy
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import load_bda  # noqa: E402

SEED = 42
B = 10_000


def ols(X: np.ndarray, y: np.ndarray):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    s2 = resid @ resid / (n - p)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return beta, se, resid


def main() -> None:
    rows = load_bda("Advertising.csv", chapter=4)   # Supabase Data/Chapter4/Advertising.csv
    TV = np.array([float(r["TV"]) for r in rows])
    radio = np.array([float(r["Radio"]) for r in rows])
    news = np.array([float(r["Newspaper"]) for r in rows])
    sales = np.array([float(r["Sales"]) for r in rows])
    n = len(sales)

    # ---- Slides 36/37: Sales ~ TV, residual / pairs / wild SE of the slope ----
    X = np.column_stack([np.ones(n), TV])
    beta, se, resid = ols(X, sales)
    fitted = X @ beta
    het = np.corrcoef(np.abs(resid), TV)[0, 1]
    print(f"[36/37] Sales ~ TV:  b0={beta[0]:.4f}  b1={beta[1]:.5f}  classical SE(b1)={se[1]:.6f}")
    print(f"        corr(|resid|, TV) = {het:.2f}  (>0 -> variance grows with TV: heteroscedastic)")

    rng = np.random.default_rng(SEED)
    rc = resid - resid.mean()
    br, bp, bw = np.empty(B), np.empty(B), np.empty(B)
    for b in range(B):
        br[b] = np.linalg.lstsq(X, fitted + rng.choice(rc, n, replace=True), rcond=None)[0][1]
        idx = rng.integers(0, n, n)
        bp[b] = np.linalg.lstsq(X[idx], sales[idx], rcond=None)[0][1]
        v = rng.choice([-1.0, 1.0], n)
        bw[b] = np.linalg.lstsq(X, fitted + resid * v, rcond=None)[0][1]
    print(f"        bootstrap SE(b1) (B={B}): residual={br.std(ddof=1):.5f}  "
          f"pairs={bp.std(ddof=1):.5f}  wild={bw.std(ddof=1):.5f}")
    print("        -> residual ~ classical (both understate); pairs ~ wild inflate it correctly")

    # ---- Slide 63: test H0 beta_Newspaper = 0 via null-enforced wild bootstrap ----
    Xf = np.column_stack([np.ones(n), TV, radio, news])
    bf, sef, _ = ols(Xf, sales)
    names = ["const", "TV", "Radio", "Newspaper"]
    print("\n[63] Sales ~ TV + Radio + Newspaper:")
    for nm, bb, ss in zip(names, bf, sef):
        t = bb / ss
        print(f"     {nm:10} coef={bb:+.5f}  SE={ss:.5f}  t={t:+.2f}  p={2*stats.t.sf(abs(t), n-4):.4f}")
    j = 3
    t_obs = bf[j] / sef[j]
    Xr = np.column_stack([np.ones(n), TV, radio])         # restricted: drop Newspaper (H0)
    br_, _, residr = ols(Xr, sales)
    fitr = Xr @ br_
    rng = np.random.default_rng(SEED)
    cnt = 0
    for b in range(B):
        ystar = fitr + residr * rng.choice([-1.0, 1.0], n)
        bb, ssb, _ = ols(Xf, ystar)
        if abs(bb[j] / ssb[j]) >= abs(t_obs):
            cnt += 1
    p_wild = (cnt + 1) / (B + 1)
    print(f"     test H0: beta_Newspaper = 0  ->  t_obs={t_obs:+.2f}  "
          f"classical p={2*stats.t.sf(abs(t_obs), n-4):.3f}  wild-bootstrap p={p_wild:.3f}")
    print("     Both agree (newspaper adds nothing); the wild bootstrap gives VALID inference")
    print("     under the documented heteroscedasticity, without assuming homoscedasticity.")


if __name__ == "__main__":
    main()
