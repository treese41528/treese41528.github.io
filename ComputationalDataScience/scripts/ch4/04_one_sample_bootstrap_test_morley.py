"""
STAT 418 · Chapter 4 — One-sample bootstrap hypothesis test on real data (slide id 51).

Backs the restored §4.6 worked example "One-Sample Bootstrap Test: Speed of Light".
This is the payoff of the null-enforcement *Location* strategy introduced one slide
earlier (H0: mu = mu_0  ->  shift each x_i to x_i - xbar + mu_0), and it mirrors the
webbook's "One-Sample Location Test: A Complete Example" (ch4_6) -- but on REAL data
instead of the book's synthetic n=30 sample.

Why morley (real) is the better worked example here:
  - Michelson's 1879 runs (R `morley`, n=100, Speed = km/s - 299000) are already vendored
    and already anchor slides 18 (DKW) and 32 (bias), so this extends one coherent thread.
  - The modern speed of light c = 299792.458 km/s is 792.458 in the data's units, giving a
    genuine, non-arbitrary null value mu_0. Michelson's mean sits ~60 km/s above it -- his
    apparatus' historically documented systematic bias -- so the test rejects DECISIVELY.
  - Because T_obs lands far past every bootstrap replicate, the bootstrap p-value bottoms
    out at its resolution floor 1/(B+1): a live demonstration of the Phipson-Smyth "+1"
    rule (never report p = 0) that the previous slide states. The classical t-test, with a
    closed-form tail, resolves all the way to ~1e-11.
  - The studentized statistic is (approximately) pivotal: the bootstrap null distribution
    of T* is ~ N(0,1), which the slide shows via its percentiles.

Run:  python 04_one_sample_bootstrap_test_morley.py        # needs numpy, scipy
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import load_morley  # noqa: E402

SEED = 42
B = 10_000
MU0 = 792.458   # modern c = 299792.458 km/s, in morley's (km/s - 299000) units


def bootstrap_one_sample_t(x: np.ndarray, mu0: float, B: int, seed: int):
    """Null-enforced one-sample bootstrap test of H0: mu = mu0 (two-sided).

    Steps (mirroring the webbook algorithm):
      1. observed studentized statistic  T_obs = (xbar - mu0) / (s/sqrt(n))
      2. enforce H0 by centering          x_tilde = x - xbar + mu0   (mean now exactly mu0)
      3. resample the centered data, recompute the studentized statistic each time
      4. two-sided p-value with the Phipson-Smyth +1 correction
    Returns (t_obs, t_boot array, p_value).
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    xbar = x.mean()
    s = x.std(ddof=1)
    t_obs = (xbar - mu0) / (s / np.sqrt(n))

    centered = x - xbar + mu0                      # null enforcement: location shift
    t_boot = np.empty(B)
    for b in range(B):
        bs = rng.choice(centered, size=n, replace=True)
        se = bs.std(ddof=1) / np.sqrt(n)
        t_boot[b] = (bs.mean() - mu0) / se
    # two-sided p; +1 in num & denom => smallest possible value is 1/(B+1), never 0
    p_value = (np.sum(np.abs(t_boot) >= np.abs(t_obs)) + 1) / (B + 1)
    return t_obs, t_boot, p_value


def make_null_dist_figure(t_obs: float, t_boot: np.ndarray, p_boot: float, out_path: str) -> None:
    """Webbook figure (Section 4.6): the bootstrap null distribution of the studentized T
    (~ N(0,1)) with T_obs marked far out in the right tail -- literally off the bulk of the
    null, which is why the p-value bottoms out at the 1/(B+1) resolution floor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=150)
    ax.hist(t_boot, bins=60, density=True, color="#2E86AB", alpha=0.55,
            edgecolor="white", label=r"bootstrap null $T^*$ (B = 10,000)")
    grid = np.linspace(-4.2, 4.2, 300)
    ax.plot(grid, stats.norm.pdf(grid), color="#1f2937", lw=1.6, ls="--", label=r"$N(0,1)$")
    # T_obs sits far to the right of the entire null distribution
    ax.axvline(t_obs, color="#C73E1D", lw=2.4)
    ax.annotate(fr"$T_{{\mathrm{{obs}}}} = {t_obs:.2f}$" + "\n" + fr"$p = {p_boot:.4f}$",
                xy=(t_obs, 0.015), xytext=(t_obs - 2.0, 0.27), color="#C73E1D",
                fontsize=11, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#C73E1D", lw=1.6))
    ax.set_xlim(-4.5, t_obs + 0.9)
    ax.set_xlabel(r"studentized statistic $T = (\bar{x}-\mu_0)\,/\,(s/\sqrt{n})$")
    ax.set_ylabel("density")
    ax.set_title("One-sample bootstrap test: Michelson 1879 vs the true speed of light")
    ax.legend(loc="upper center", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figure -> {out_path}")


def main() -> None:
    speed = np.array([r["Speed"] for r in load_morley(chapter=4)], dtype=float)
    n = len(speed)
    xbar, s = speed.mean(), speed.std(ddof=1)

    t_obs, t_boot, p_boot = bootstrap_one_sample_t(speed, MU0, B, SEED)
    _, p_classical = stats.ttest_1samp(speed, MU0)
    n_exceed = int(np.sum(np.abs(t_boot) >= np.abs(t_obs)))

    print("=== One-sample bootstrap test: Michelson speed of light ===")
    print(f"n = {n}   xbar = {xbar:.2f}  ->  {299000 + xbar:.1f} km/s")
    print(f"s = {s:.2f}   mu0 = {MU0}  (true c, in km/s - 299000)")
    print(f"systematic bias  xbar - mu0 = {xbar - MU0:+.2f}  km/s")
    print(f"T_obs = ({xbar:.2f} - {MU0}) / ({s:.2f}/sqrt({n})) = {t_obs:.3f}")
    print()
    print(f"Bootstrap (B = {B:,}, seed = {SEED}); null enforced by centering at mu0")
    print(f"  null T* :  mean = {t_boot.mean():+.3f}   sd = {t_boot.std(ddof=1):.3f}"
          f"   (studentized => ~ N(0,1))")
    # percentiles displayed in the slide's dark null-distribution panel
    pct = {q: np.percentile(np.abs(t_boot), q) for q in (25, 50, 75, 90, 95, 99)}
    print("  sorted |T*| percentiles:  " +
          "  ".join(f"P{q}={v:.3f}" for q, v in pct.items()) +
          f"  max={np.abs(t_boot).max():.3f}")
    print(f"  # of |T*| >= |T_obs|={abs(t_obs):.2f} :  {n_exceed}")
    print(f"  bootstrap p = ({n_exceed}+1)/({B}+1) = {p_boot:.5f}"
          f"   (resolution floor 1/(B+1) = {1/(B+1):.5f})")
    print(f"  classical one-sample t-test p = {p_classical:.3e}")
    print()
    print("Conclusion: reject H0 decisively -- Michelson's apparatus overshoots the true")
    print("speed of light by ~60 km/s. The bootstrap p cannot resolve below 1/(B+1); the")
    print("studentized null ~ N(0,1) confirms the statistic is (approximately) pivotal.")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Website",
                       "assets", "Part2", "Chapter4", "bootstrap-hypothesis-testing_ch4_6",
                       "ch4_6_fig09_one_sample_test_morley.png")
    make_null_dist_figure(t_obs, t_boot, p_boot, os.path.normpath(out))


if __name__ == "__main__":
    main()
