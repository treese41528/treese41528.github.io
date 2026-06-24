"""
STAT 418 · Chapter 5.5 — MCMC Algorithms
Script: Random-walk Metropolis-Hastings (the slide-62 sampler, runnable).

The whole algorithm is ~12 lines: propose a symmetric random-walk step, accept
with probability min(1, pi(theta')/pi(theta)) on the log scale, otherwise stay.
Because the Gaussian proposal is symmetric, the proposal density cancels from the
acceptance ratio, leaving only the target.

We run it on an UNNORMALIZED Beta(16, 6) posterior — chosen because it has a known
closed form, so we can verify the sampler against the exact mean and SD. This needs
no real data and no external services: it is a pure-Python/NumPy/SciPy demonstration
of the method. (Reproduces the numbers shown on slide 62.)

Run:  python 05_random_walk_mh.py        # needs numpy, scipy
"""
from __future__ import annotations

import numpy as np


def metropolis_rw(log_target, theta_0, sigma_prop, S=10_000, seed=42):
    """Random-walk Metropolis-Hastings for a scalar parameter.

    log_target: callable returning log of the (unnormalized) target density.
    Returns (chain, acceptance_rate).
    """
    rng = np.random.default_rng(seed)
    chain = np.zeros(S)
    chain[0] = theta_0
    accepted = 0

    for s in range(1, S):
        # Propose a symmetric random-walk step
        prop = chain[s - 1] + rng.normal(0, sigma_prop)

        # Log acceptance ratio (the symmetric proposal q cancels)
        log_R = log_target(prop) - log_target(chain[s - 1])

        # Accept / reject
        if np.log(rng.uniform()) < log_R:
            chain[s] = prop
            accepted += 1
        else:
            chain[s] = chain[s - 1]

    return chain, accepted / (S - 1)


def log_unnorm_beta(theta, a, b):
    """Log of the unnormalized Beta(a, b) density on (0, 1)."""
    if theta <= 0 or theta >= 1:
        return -np.inf
    return (a - 1) * np.log(theta) + (b - 1) * np.log(1 - theta)


def main() -> None:
    from scipy import stats

    # Run the sampler on Beta(16, 6) (known closed form for verification)
    log_target = lambda theta: log_unnorm_beta(theta, 16, 6)
    chain, rate = metropolis_rw(log_target, theta_0=0.5, sigma_prop=0.1)

    # Compare MCMC to the exact posterior after discarding warm-up
    burn = 1000
    samples = chain[burn:]
    exact = stats.beta(16, 6)

    print(f"Acceptance rate: {rate:.3f}")
    print(f"MCMC  mean: {samples.mean():.4f}  SD: {samples.std():.4f}")
    print(f"Exact mean: {exact.mean():.4f}  SD: {exact.std():.4f}")


if __name__ == "__main__":
    main()
