"""
STAT 418 · Chapter 3 — Slide 37: why supplying the gradient cuts optimizer work.

Backs the §3.2 "scipy.optimize" note that providing an exact gradient (by hand, or via
automatic differentiation such as JAX's ``jax.grad``) slashes the number of objective
*function evaluations* a quasi-Newton optimizer needs. Without a gradient, scipy must
approximate it by finite differences -- roughly p+1 objective evaluations per gradient,
every iteration. An exact gradient (autodiff or analytic) is supplied directly, so the
optimizer only evaluates the objective for its line search.

Benchmark (fully reproducible, seed = 0): maximum-likelihood logistic regression,
n = 400 observations, p = 15 parameters, minimized with BFGS.

  finite-difference gradient (jac=None) : nfev = 688
  exact gradient (jac supplied)         : nfev =  43

Autodiff produces the SAME exact gradient as the hand-coded analytic gradient used here,
so its function-evaluation count matches the ``jac``-supplied run -- the ~16x reduction is
the cost of finite-difference probing, not of the gradient formula. Source for the claim
on the slide: this script.

Run:  python 07_gradient_eval_benchmark.py        # needs numpy, scipy
"""
from __future__ import annotations

import numpy as np
from scipy import optimize

SEED = 0
N, P = 400, 15


def main() -> None:
    rng = np.random.default_rng(SEED)
    X = np.column_stack([np.ones(N), rng.normal(size=(N, P - 1))])
    beta_true = 0.5 * rng.normal(size=P)
    prob = 1.0 / (1.0 + np.exp(-(X @ beta_true)))
    y = (rng.uniform(size=N) < prob).astype(float)

    def nll(b: np.ndarray) -> float:
        z = X @ b
        return float(np.sum(np.log1p(np.exp(z)) - y * z))   # negative log-likelihood

    def grad(b: np.ndarray) -> np.ndarray:
        return X.T @ (1.0 / (1.0 + np.exp(-(X @ b))) - y)   # exact gradient

    fd = optimize.minimize(nll, np.zeros(P), method="BFGS")             # finite differences
    an = optimize.minimize(nll, np.zeros(P), method="BFGS", jac=grad)   # exact gradient

    print(f"Logistic MLE  (n = {N}, p = {P}),  BFGS,  seed = {SEED}")
    print(f"  finite-difference gradient (jac=None) : nfev = {fd.nfev}")
    print(f"  exact gradient (autodiff / analytic)  : nfev = {an.nfev}")
    print(f"  reduction: {fd.nfev / an.nfev:.0f}x fewer objective evaluations")
    print(f"  (both converge to the same optimum: ||grad|| = "
          f"{np.linalg.norm(grad(an.x)):.1e}, max|beta diff| = {np.max(np.abs(fd.x - an.x)):.1e})")


if __name__ == "__main__":
    main()
