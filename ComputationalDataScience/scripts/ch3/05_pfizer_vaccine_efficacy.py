"""
STAT 418 · Chapter 3 — Thread E: the delta method for vaccine efficacy (Pfizer-BioNTech).

Backs §3.3 Slide 57 "Vaccine Efficacy" and the webbook "Vaccine Efficacy Problem" example.
Vaccine efficacy is a NONLINEAR function of two estimated rates,
:math:`\\text{VE} = 1 - p_v / p_p`, so its standard error needs the **delta method** -- the
section's payoff application. We attach the numbers to their real source and disclose how
the resulting interval differs from the famous published one.

REAL DATA (no synthetic study): Pfizer-BioNTech BNT162b2 Phase 3 primary efficacy endpoint
(Polack et al., NEJM 2020), evaluable-efficacy population, >= 7 days after dose 2, no
evidence of prior infection:  8 COVID-19 cases among 18,198 vaccine recipients vs 162 among
18,325 placebo recipients.

  VE = 1 - p_v/p_p = 0.950 (95.0%);  delta-method SE(VE) ~ 0.018;  Wald 95% CI ~ (0.915, 0.986).

CRITICAL DISCLOSURE: the headline Pfizer interval **(90.3%, 97.6%)** is a *Bayesian*
beta-binomial **credible** interval (the trial's pre-specified analysis used a Beta(0.700102,
1) prior on theta = p_v/(p_v+p_p)), NOT the Wald delta interval. The two do not coincide --
with only 8 vaccine-arm events the symmetric Wald interval is the least accurate of the
options and its lower bound (~0.915) sits above the published 90.3%. We compute both so the
slide/book can show the gap honestly.

Run:  python 05_pfizer_vaccine_efficacy.py        # needs numpy, scipy
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# Polack et al. (2020) NEJM, primary efficacy endpoint (evaluable, no prior infection):
CASES_VAX, N_VAX = 8, 18198
CASES_PBO, N_PBO = 162, 18325


def main() -> None:
    p_v = CASES_VAX / N_VAX
    p_p = CASES_PBO / N_PBO
    rr = p_v / p_p
    ve = 1 - rr
    print(f"Pfizer-BioNTech BNT162b2 (Polack et al. 2020):")
    print(f"  vaccine: {CASES_VAX} cases / {N_VAX}    placebo: {CASES_PBO} cases / {N_PBO}")
    print(f"  p_vaccine = {p_v:.6f}   p_placebo = {p_p:.6f}")
    print(f"  VE = 1 - p_v/p_p = {ve:.4f}  ({ve * 100:.1f}%)")

    # Delta method: SE(VE) = RR * sqrt[ (1-p_v)/(n_v p_v) + (1-p_p)/(n_p p_p) ]
    se_ve = rr * np.sqrt((1 - p_v) / (N_VAX * p_v) + (1 - p_p) / (N_PBO * p_p))
    z = stats.norm.ppf(0.975)
    print(f"  delta-method SE(VE) = {se_ve:.4f}")
    print(f"  Wald 95% CI: ({ve - z * se_ve:.3f}, {ve + z * se_ve:.3f})   <- symmetric, normal-based")

    # Pfizer's pre-specified BAYESIAN analysis: theta = p_v/(p_v+p_p), prior Beta(0.700102, 1),
    # posterior Beta(0.700102 + cases_vax, 1 + cases_pbo); VE = (1 - 2 theta)/(1 - theta).
    post = stats.beta(0.700102 + CASES_VAX, 1.0 + CASES_PBO)
    ve_of_theta = lambda th: (1 - 2 * th) / (1 - th)        # decreasing in theta
    lo = ve_of_theta(post.ppf(0.975))
    hi = ve_of_theta(post.ppf(0.025))
    print(f"\n  Bayesian beta-binomial 95% CREDIBLE interval (the trial's pre-specified method):")
    print(f"    VE in ({lo:.3f}, {hi:.3f})   = the published (90.3%, 97.6%)")
    print(f"  The published interval is BAYESIAN, not the Wald delta interval above -- they differ,")
    print(f"  and with only {CASES_VAX} vaccine-arm events the Wald lower bound is optimistic.")


if __name__ == "__main__":
    main()
