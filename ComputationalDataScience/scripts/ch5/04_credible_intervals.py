"""
STAT 418 · Chapter 5.3 — Credible intervals & posterior decisions (slides 41/43/44/45).

All four §5.3 ideas read off ONE Beta posterior, computed once:
  - Slide 41: equal-tail interval (ETI) vs highest-density interval (HDI).
  - Slide 43: directional probabilities P(theta > t) -- one CDF eval per threshold.
  - Slide 44: ROPE (region of practical equivalence) -> reject / accept / undecided.
  - Slide 45: posterior-predictive interval for future free throws (Beta-Binomial).

DATA (real): Stephen Curry's FIRST 36 free throws of 2023-24 (34/36; the §5.2 running
sample) gives the flat-prior posterior Beta(35, 3) -- left-skewed (mode 0.944), so ETI
and HDI visibly differ. The ROPE example also uses real career lines (see BDA/DATA_CARD.md):
Curry (elite), Westbrook (a genuine career-average ~0.77 shooter), and Holiday's early
small sample. Free throws because the Beta-Binomial needs a constant theta over
exchangeable iid trials.

Run:  python 04_credible_intervals.py        # needs numpy, scipy
"""
from __future__ import annotations

import os
import sys

from scipy import stats
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import ft_cumulative, load_freethrow_career, load_freethrows  # noqa: E402


def hdi(dist, mass=0.95):
    """Highest-density interval: the narrowest interval holding `mass` probability.

    For a unimodal density the narrowest such interval is the HDI, and its endpoints
    share equal density, f(lo) = f(hi). We minimize the width ppf(p+mass) - ppf(p)
    over the lower-tail probability p; the minimizer lands exactly on that
    equal-density interval (a grid search only approximates it, leaving the endpoint
    densities slightly unequal)."""
    res = minimize_scalar(lambda p: dist.ppf(p + mass) - dist.ppf(p),
                          bounds=(1e-9, 1 - mass - 1e-9), method="bounded")
    return dist.ppf(res.x), dist.ppf(res.x + mass)


def main() -> None:
    # Running posterior: Curry's first 36 FTA (34/36), flat Beta(1,1) -> Beta(35, 3).
    k, n = ft_cumulative("Stephen Curry", min_attempts=30)       # (34, 36)
    post = stats.beta(1 + k, 1 + (n - k))                        # Beta(35, 3)
    print(f"Curry first {n} FTA = {k}/{n}; flat-prior posterior Beta(35, 3), mode 0.944\n")

    # --- Slide 41: ETI vs HDI. HDI endpoints share equal density; ETI's don't. ---
    et = (post.ppf(0.025), post.ppf(0.975))
    hd = hdi(post)
    print("[41] 95% intervals")
    print(f"  ETI [{et[0]:.3f}, {et[1]:.3f}] width {et[1]-et[0]:.3f}  "
          f"f(lo)={post.pdf(et[0]):.3f}  f(hi)={post.pdf(et[1]):.3f}  (unequal)")
    print(f"  HDI [{hd[0]:.3f}, {hd[1]:.3f}] width {hd[1]-hd[0]:.3f}  "
          f"f(lo)={post.pdf(hd[0]):.3f}  f(hi)={post.pdf(hd[1]):.3f}  (equal -> narrower)")

    # --- Slide 43: directional probabilities; one CDF eval per threshold ---
    print("\n[43] directional probabilities  P(theta > t) = 1 - F(t)")
    for t, lbl in [(0.50, "better than a coin"), (0.78, "above league average"),
                   (0.90, "elite"), (0.95, "near-perfect")]:
        print(f"  P(theta > {t:.2f}) = {post.sf(t):.3f}   ({lbl})")
    # flat vs the informative career prior (the running n0 contrast, folded in here)
    career = load_freethrow_career()["Stephen Curry"]
    kf, nf = ft_cumulative("Stephen Curry")
    pmk, pms = career["ft"] - kf, (career["fta"] - nf) - (career["ft"] - kf)
    career_post = stats.beta(pmk + k, pms + (n - k))
    print(f"  P(>0.90): flat = {post.sf(0.90):.3f}  vs  career prior = {career_post.sf(0.90):.3f}")

    # --- Slide 44: ROPE = "practically league-average" [0.76, 0.80] (0.78 +/- 0.02) ---
    print("\n[44] ROPE [0.76, 0.80] (practically league-average): HDI vs ROPE")
    rope = (0.76, 0.80)

    def rope_decision(a, b, label):
        d = stats.beta(a, b)
        lo, hi = hdi(d)
        if lo > rope[1] or hi < rope[0]:
            out = "Reject (meaningfully different)"
        elif lo > rope[0] and hi < rope[1]:
            out = "Accept (practically average)"
        else:
            out = "Undecided"
        print(f"  {label:32s} HDI [{lo:.3f}, {hi:.3f}] -> {out}")

    rope_decision(1 + 299, 1 + 25, "Curry, full season 299/324")        # elite
    wc = load_freethrow_career()["Russell Westbrook"]
    rope_decision(1 + wc["ft"], 1 + (wc["fta"] - wc["ft"]), "Westbrook, career 6109/7928")  # average
    g = next(x for x in load_freethrows("Jrue Holiday") if x["cum_fta"] >= 30)
    rope_decision(1 + g["cum_ft"], 1 + (g["cum_fta"] - g["cum_ft"]),
                  f"Holiday, first {g['cum_fta']} FTA")                  # small sample

    # --- Slide 45: posterior-predictive interval for the next m free throws ---
    m = 20
    pred = stats.betabinom(m, 1 + k, 1 + (n - k))                # BetaBinom(20, 35, 3)
    print(f"\n[45] next {m} free throws: predictive mean {pred.mean():.1f} makes, "
          f"95% PI [{pred.ppf(0.025):.0f}, {pred.ppf(0.975):.0f}]")
    print(f"     credible interval on theta: [{et[0]:.3f}, {et[1]:.3f}] (epistemic only); "
          f"the predictive PI also carries aleatoric (shot-to-shot) randomness.")


if __name__ == "__main__":
    main()
