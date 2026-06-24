"""
STAT 418 · Chapter 5.5 — MCMC Algorithms
Script: Bayesian logistic regression via Metropolis-Hastings (the slide-64 sampler).

This is the chapter's first NON-conjugate model: the sigmoid likelihood times a
Gaussian prior has no closed-form posterior, so MCMC is required. The sampler is
just the multivariate generalization of the scalar random-walk Metropolis-Hastings
in 05_random_walk_mh.py — propose a Gaussian step in d dimensions, accept on the
log scale.

The data are SYNTHETIC on purpose: with known true coefficients we can check that
the sampler recovers them, which is the point of a method demonstration (no real
dataset is needed here — the real-data Bayesian fit lives in the §5.6 PyMC scripts).
Reproduces the numbers shown on slide 64.

Run:  python 05_logistic_regression_mh.py     # needs numpy
"""
from __future__ import annotations

import numpy as np


def log_unnorm_posterior(beta, X, y, tau=10):
    """Log unnormalized posterior for Bayesian logistic regression.

    Log-likelihood uses the numerically stable logaddexp; the N(0, tau^2) prior
    contributes a quadratic penalty.
    """
    eta = X @ beta
    log_lik = np.sum(y * eta - np.logaddexp(0, eta))
    log_prior = -0.5 * np.sum(beta ** 2) / tau ** 2
    return log_lik + log_prior


def metropolis_rw_mv(log_target, theta_0, Sigma_prop, S=20_000, seed=42):
    """Multivariate random-walk Metropolis-Hastings. Returns (chain, accept_rate)."""
    rng = np.random.default_rng(seed)
    d = len(theta_0)
    chain = np.zeros((S, d))
    chain[0] = theta_0
    accepted = 0

    for s in range(1, S):
        prop = rng.multivariate_normal(chain[s - 1], Sigma_prop)
        log_R = log_target(prop) - log_target(chain[s - 1])

        if np.log(rng.uniform()) < log_R:
            chain[s] = prop
            accepted += 1
        else:
            chain[s] = chain[s - 1]

    return chain, accepted / (S - 1)


def main() -> None:
    # --- Synthetic data with correlated features ---
    rng = np.random.default_rng(0)
    n = 200
    Sigma_x = np.array([[1.0, 0.6],
                        [0.6, 1.0]])
    Z = rng.multivariate_normal([0, 0], Sigma_x, size=n)
    X = np.column_stack([np.ones(n), Z])
    d = X.shape[1]
    beta_true = np.array([0.5, -1.0, 0.8])
    prob = 1 / (1 + np.exp(-X @ beta_true))
    y = rng.binomial(1, prob)

    # --- Run sampler (diagonal proposal ignores the feature correlation) ---
    log_target = lambda b: log_unnorm_posterior(b, X, y)
    Sigma_prop = 0.01 * np.eye(d)
    theta_0 = np.zeros(d)
    chain, rate = metropolis_rw_mv(log_target, theta_0, Sigma_prop)

    # --- Results (discard 5,000 warm-up draws) ---
    samples = chain[5000:]
    print(f"Acceptance rate: {rate:.3f}")
    for j in range(d):
        print(f"beta_{j}: {samples[:, j].mean():.3f} (true: {beta_true[j]:.1f})")


if __name__ == "__main__":
    main()
