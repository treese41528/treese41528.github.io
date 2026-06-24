"""
STAT 418 · Chapter 5 — Normal models on Michelson's 1879 speed of light (slides 14/28/29/67).

ONE real dataset runs the whole Normal thread: A. A. Michelson's 100 determinations of the
speed of light, June 5 - July 2 1879 (R `morley`; `Speed` = km/s - 299000). We use a
**Jeffreys / flat working prior** throughout, so each posterior is data-driven and matches
its frequentist counterpart -- the point is NOT prior sensitivity but the spine below.

THE SPINE (precise but inaccurate). The likelihood is tight (SE ~ 7.9 km/s) but centred in
the WRONG place. Michelson measured in AIR, so the honest comparison value is Stigler's
(1977) air-corrected target 734.5 (= modern vacuum 299,792.5 carried through Michelson's
own correction), NOT the vacuum value. The posterior mean sits ~118 km/s -- about 15
standard errors -- above 734.5, and no credibility level you would report contains it:
systematic error, which a credible interval (random scatter only) cannot see.

Caveat: the `Expt`/`Run` 5x20 layout is Stigler's analytical convenience, NOT Michelson's
design (he measured on 18 days), so we never treat a sub-experiment as a separate study.

  - Slide 28: Normal-Normal, sigma known -> Jeffreys posterior = frequentist z-interval.
  - Slide 29: Normal-Inverse-Gamma (mu, sigma^2 unknown) -> marginal Student-t for mu.
  - Slide 14: 2D grid posterior over (mu, sigma); marginals by summation.
  - Slide 67: Gibbs sampler for the NIG model, checked against the closed form.

Run:  python 06_normal_michelson.py        # needs numpy, scipy
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import load_morley  # noqa: E402

# Air-corrected comparison target (Stigler 1977); the "truth is a modelling choice" spectrum.
TARGETS = {"naive full-refraction": 705.0, "Stigler air-corrected": 734.5, "modern vacuum": 792.5}


def gibbs_nig(y, mu0, kappa0, alpha0, beta0, S=20_000, burn=2_000, seed=42):
    """Gibbs sampler for the Normal-Inverse-Gamma model (conjugate conditionals).

    Alternates two exact draws:
      mu | sigma^2, y ~ Normal( (kappa0 mu0 + n ybar)/(kappa0+n),  sigma^2/(kappa0+n) )
      sigma^2 | mu, y ~ Inv-Gamma( alpha0 + (n+1)/2,
                                   beta0 + 0.5 sum (y-mu)^2 + 0.5 kappa0 (mu-mu0)^2 )
    The (n+1)/2 shape and the kappa0(mu-mu0)^2 term come from p(mu | sigma^2), whose
    variance depends on sigma^2 -- the term students most often drop. NumPy has no
    Inverse-Gamma: draw g ~ Gamma(alpha, 1/beta) and set sigma^2 = 1/g.
    """
    rng = np.random.default_rng(seed)
    n, ybar = len(y), y.mean()
    draws = np.empty((S, 2))
    mu, sig2 = ybar, y.var()
    kn = kappa0 + n
    mun = (kappa0 * mu0 + n * ybar) / kn
    for s in range(S):
        mu = rng.normal(mun, np.sqrt(sig2 / kn))                       # mu | sigma^2, y
        an = alpha0 + (n + 1) / 2
        bn = beta0 + 0.5 * np.sum((y - mu) ** 2) + 0.5 * kappa0 * (mu - mu0) ** 2
        sig2 = 1.0 / rng.gamma(an, 1.0 / bn)                           # sigma^2 | mu, y
        draws[s] = (mu, sig2)
    return draws[burn:]


def bias_report(post, label):
    """Print the bias vs each candidate target and whether it falls in a 99.99% interval."""
    ybar = post.mean()
    print(f"  {label}: posterior mean {299000 + ybar:.1f} km/s")
    lo, hi = post.ppf(5e-5), post.ppf(1 - 5e-5)                        # 99.99% interval
    for name, t in TARGETS.items():
        inside = lo <= t <= hi
        print(f"    vs {name:22s} {299000 + t:9.1f}: bias {ybar - t:+6.1f} km/s "
              f"= {(ybar - t) / post.std():4.1f} SE  in 99.99% CI? {inside}")


def main() -> None:
    speed = np.array([r["Speed"] for r in load_morley()], dtype=float)
    n = len(speed)
    ybar = speed.mean()
    s = speed.std(ddof=1)                  # sample SD ~ 79 km/s
    se = s / np.sqrt(n)                    # ~ 7.9 km/s
    SS = np.sum((speed - ybar) ** 2)      # sum of squares about the mean
    print(f"Michelson 1879: n={n}, mean {299000 + ybar:.1f} km/s, SD {s:.2f}, SE {se:.2f}\n")

    # --- Slide 28: Normal-Normal, sigma KNOWN, flat (Jeffreys) prior on mu ---
    # Posterior collapses to N(ybar, sigma^2/n) -- identical to the frequentist z-interval.
    print("[28] Normal-Normal (sigma known = s), Jeffreys prior")
    post28 = stats.norm(ybar, se)
    lo, hi = post28.ppf(0.025), post28.ppf(0.975)
    print(f"  posterior N({299000 + ybar:.1f}, {se:.2f}^2)  95% [{299000 + lo:.1f}, {299000 + hi:.1f}]")
    # Even a real, distant historical prior (Foucault 1862, 298,000 +/- 500) is swamped:
    mu0_f, sd0_f = 298000 - 299000, 500.0
    w = (n / s**2) / (1 / sd0_f**2 + n / s**2)
    mun_f = w * ybar + (1 - w) * mu0_f
    print(f"  Foucault 1862 prior (298,000 +/- 500): data weight w={w:.4f} -> posterior "
          f"{299000 + mun_f:.1f}  (indistinguishable from Jeffreys)")
    bias_report(post28, "bias")

    # --- Slide 29: Normal-Inverse-Gamma (Jeffreys reference 1/sigma^2), mu and sigma^2 unknown ---
    # Marginal posterior for the mean is Student-t with n-1 df; sigma^2 ~ Inv-Gamma((n-1)/2, SS/2).
    print("\n[29] Normal-Inverse-Gamma (Jeffreys), both unknown")
    df = n - 1
    post29 = stats.t(df, loc=ybar, scale=s / np.sqrt(n))               # marginal t for mu
    lo, hi = post29.ppf(0.025), post29.ppf(0.975)
    print(f"  mu | y ~ t_{df}({299000 + ybar:.1f}, (s/sqrt(n))^2)  95% [{299000 + lo:.1f}, {299000 + hi:.1f}]")
    ig = stats.invgamma(df / 2, scale=SS / 2)                          # sigma^2 | y
    print(f"  sigma^2 | y ~ Inv-Gamma({df/2:.0f}, {SS/2:.0f})  E[sigma^2]={ig.mean():.0f} "
          f"(sigma ~ {np.sqrt(ig.mean()):.1f})")
    bias_report(post29, "bias (marginal t)")

    # --- Slide 14: 2D grid posterior over (mu, sigma), Jeffreys 1/sigma; marginals by summation ---
    print("\n[14] 2D grid posterior over (mu, sigma)")
    ng = 200
    mu_grid = np.linspace(820.0, 885.0, ng)
    sig_grid = np.linspace(60.0, 105.0, ng)
    MU = mu_grid[:, None]
    SIG = sig_grid[None, :]
    ss_mu = np.sum((speed[None, None, :] - MU[:, :, None]) ** 2, axis=2)   # (ng, ng)
    log_joint = -n * np.log(SIG) - 0.5 * ss_mu / SIG**2 - np.log(SIG)      # likelihood + 1/sigma
    log_joint -= log_joint.max()
    joint = np.exp(log_joint)
    joint /= joint.sum()
    marg_mu = joint.sum(axis=1)
    marg_sig = joint.sum(axis=0)
    i, j = np.unravel_index(joint.argmax(), joint.shape)
    mu_mean = np.sum(mu_grid * marg_mu)
    mu_sd = np.sqrt(np.sum((mu_grid - mu_mean) ** 2 * marg_mu))
    sig_mean = np.sum(sig_grid * marg_sig)
    print(f"  grid {ng}x{ng}; mu in [{mu_grid[0]:.0f},{mu_grid[-1]:.0f}], sigma in [{sig_grid[0]:.0f},{sig_grid[-1]:.0f}]")
    print(f"  joint mode: mu={mu_grid[i]:.1f}, sigma={sig_grid[j]:.1f}")
    print(f"  posterior mean mu={mu_mean:.1f} (SD {mu_sd:.2f}); posterior mean sigma={sig_mean:.1f}")
    print(f"  air target 734.5 lies {(mu_mean - 734.5) / mu_sd:.0f} SE below the mu marginal "
          f"(far in the left tail) -> the bias is invisible to the model")

    # --- Slide 67: Gibbs for the NIG model, checked against the closed form ---
    # Weakly-informative NIG (kappa0 = 1 pseudo-observation) -- swamped by n=100, so the
    # posterior is effectively the Jeffreys result; here it exercises the sampler.
    print("\n[67] Gibbs NIG vs closed form (weakly-informative prior, swamped by n=100)")
    mu0, kappa0, alpha0, beta0 = 800.0, 1.0, 1.0, 1.0
    draws = gibbs_nig(speed, mu0, kappa0, alpha0, beta0)
    kn = kappa0 + n
    mun = (kappa0 * mu0 + n * ybar) / kn
    an = alpha0 + n / 2
    bn = beta0 + 0.5 * SS + 0.5 * (kappa0 * n / kn) * (ybar - mu0) ** 2
    print(f"  Gibbs  mean(mu)={draws[:, 0].mean():.2f}   sigma^2={draws[:, 1].mean():.0f}")
    print(f"  Exact  mean(mu)={mun:.2f}   sigma^2={bn / (an - 1):.0f}")


if __name__ == "__main__":
    main()
