"""
STAT 418 · Chapter 4 — Bootstrap CIs + diagnostics on real data: law-school correlation
(Slides 34 "CI Example: Correlation Coefficient" and 38 "Bootstrap Diagnostics").

Replaces the old SYNTHETIC bivariate-normal example (rng.multivariate_normal, rho=0.9,
n=15) with the REAL Efron-Tibshirani law-school sample (n=15 schools, LSAT vs GPA). This
is the dataset Efron & Tibshirani (1993) use to motivate BCa, and it makes the boundary
lesson authentic: the correlation is bounded on [-1, 1], yet on this small, left-skewed
sample two of the three core CI methods produce IMPOSSIBLE intervals (upper bound > 1).

What each slide shows (all numbers below come from this script, B = 10,000, seed = 42):
  - Slide 34: r_hat and the three CORE CIs -- percentile, basic, normal. Basic and normal
    push the upper limit past 1.0 (impossible); only percentile respects the boundary, but
    at n = 15 it is first-order accurate only. BCa (the second-order remedy) is computed
    here for the webbook / optional Section 4.7; the lecture slide only POINTS to it.
  - Slide 38: the four diagnostics evaluated on THIS bootstrap so the flags actually fire --
    a left-skewed histogram (skew ~ -0.88), a curved Q-Q (the BCa flag), an SE-stability
    curve that flattens, and a negligible bias ratio |bias|/SE (so the issue is SKEW, not
    bias -- which is exactly why BCa, not the basic interval, is the proper fix).

Run:  python 05_correlation_ci_law_school.py        # needs numpy, scipy
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import load_law_school  # noqa: E402

SEED = 42
B = 10_000


def bootstrap_corr(lsat: np.ndarray, gpa: np.ndarray, B: int, seed: int) -> np.ndarray:
    """Nonparametric pairs bootstrap of Pearson's correlation: resample the (LSAT, GPA)
    PAIRS with replacement (rows move together) and recompute r each time."""
    rng = np.random.default_rng(seed)
    n = len(lsat)
    r_star = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        r_star[b] = np.corrcoef(lsat[idx], gpa[idx])[0, 1]
    return r_star


def bca_ci(r_star: np.ndarray, r_hat: float, jack: np.ndarray, alpha: float = 0.05):
    """BCa interval. z0 = bias-correction from the share of replicates below r_hat;
    a = acceleration from the jackknife skewness of the statistic. Computed for the
    webbook / optional Section 4.7 -- NOT featured on the lecture slide."""
    z0 = stats.norm.ppf(np.mean(r_star < r_hat))
    jbar = jack.mean()
    a = np.sum((jbar - jack) ** 3) / (6.0 * np.sum((jbar - jack) ** 2) ** 1.5)

    def endpoint(p: float) -> float:
        zp = stats.norm.ppf(p)
        adj = z0 + (z0 + zp) / (1 - a * (z0 + zp))
        return float(np.quantile(r_star, stats.norm.cdf(adj)))

    return (endpoint(alpha / 2), endpoint(1 - alpha / 2)), z0, a


def make_correlation_ci_figure(r_star, r_hat, cis, out_path):
    """Webbook figure (Sections 4.3 / 4.7): the bootstrap distribution of the law-school
    correlation (left-skewed, bounded at 1) with the four 95% CIs drawn as bars. Basic and
    Normal cross the rho = 1 boundary (impossible); percentile and BCa respect it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(9.0, 5.8), dpi=150)
    gs = GridSpec(2, 1, height_ratios=[3, 2], hspace=0.10)
    ax0, ax1 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    ax0.hist(r_star, bins=60, density=True, color="#F18F01", alpha=0.55,
             edgecolor="white", label=r"bootstrap $\hat{r}^*$ (B = 10,000)")
    ax0.axvline(r_hat, color="#C73E1D", lw=2.0, label=fr"$\hat{{r}} = {r_hat:.3f}$")
    ax0.axvline(1.0, color="#7F8C8D", lw=1.6, ls="--", label=r"boundary $\rho = 1$")
    ax0.axvspan(1.0, 1.16, color="#C73E1D", alpha=0.08)
    ax0.set_ylabel("density")
    ax0.set_title("Bootstrap CIs for a bounded parameter: law-school correlation (n = 15)")
    ax0.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax0.grid(alpha=0.25)
    plt.setp(ax0.get_xticklabels(), visible=False)

    order = [("Percentile", cis["perc"], "#2E86AB"), ("Basic", cis["basic"], "#9B59B6"),
             ("Normal", cis["norm"], "#20B2AA"), ("BCa", cis["bca"], "#3A7D44")]
    yticks, ylabels = [], []
    for i, (name, (lo, hi), col) in enumerate(order):
        y = len(order) - i
        yticks.append(y); ylabels.append(name)
        ax1.plot([lo, hi], [y, y], color=col, lw=5, solid_capstyle="round")
        ax1.plot([r_hat], [y], "|", color="#C73E1D", ms=13, mew=2, zorder=5)
        if hi > 1.0:                                   # highlight the impossible overshoot
            ax1.plot([1.0, hi], [y, y], color="#C73E1D", lw=5, solid_capstyle="round")
            ax1.text(hi + 0.012, y, f"{hi:.2f} > 1", ha="left", va="center",
                     fontsize=8, color="#C73E1D")
    ax1.axvline(1.0, color="#7F8C8D", lw=1.6, ls="--")
    ax1.set_yticks(yticks); ax1.set_yticklabels(ylabels)
    ax1.set_ylim(0.3, len(order) + 0.7)
    ax1.set_xlim(*ax0.get_xlim())
    ax1.set_xlabel(r"correlation $\rho$")
    ax1.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figure -> {out_path}")


def make_diagnostics_figure(r_star, r_hat, out_path):
    """Webbook figure (Section 4.3): the four-panel bootstrap diagnostic suite on the
    law-school correlation -- skewed histogram, curved Q-Q, SE stability, bias ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), dpi=150)
    boot_mean, se = float(r_star.mean()), float(r_star.std(ddof=1))

    ax[0].hist(r_star, bins=50, density=True, color="#2E86AB", alpha=0.6, edgecolor="white")
    ax[0].axvline(r_hat, color="#C73E1D", lw=2, label=fr"$\hat r = {r_hat:.3f}$")
    ax[0].axvline(boot_mean, color="#F18F01", lw=2, ls="--", label=f"boot mean {boot_mean:.3f}")
    ax[0].set_title(f"Histogram (skew {stats.skew(r_star):+.2f})")
    ax[0].set_xlabel(r"$\hat r^*$"); ax[0].legend(fontsize=8)

    stats.probplot(r_star, dist="norm", plot=ax[1])
    ax[1].set_title("Normal Q-Q (curved = skew)")
    ax[1].get_lines()[0].set(color="#2E86AB", markersize=2)
    ax[1].get_lines()[1].set_color("#C73E1D")

    bgrid = np.arange(100, len(r_star) + 1, 100)
    run_se = [float(r_star[:b].std(ddof=1)) for b in bgrid]
    ax[2].plot(bgrid, run_se, color="#20B2AA")
    ax[2].axhline(se, color="#7F8C8D", ls="--", lw=1)
    ax[2].set_title("SE vs B (flattens by ~2k)"); ax[2].set_xlabel("B"); ax[2].set_ylabel("running SE")

    bias = abs(boot_mean - r_hat)
    ax[3].bar(["|bias|", "SE"], [bias, se], color=["#C73E1D", "#2E86AB"])
    ax[3].set_title(f"|bias|/SE = {bias / se:.3f}")

    fig.suptitle("Bootstrap diagnostics: law-school correlation (n = 15, B = 10,000)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figure -> {out_path}")


def main() -> None:
    rows = load_law_school(chapter=4)
    lsat = np.array([r["LSAT"] for r in rows])
    gpa = np.array([r["GPA"] for r in rows])
    n = len(lsat)
    r_hat = float(np.corrcoef(lsat, gpa)[0, 1])

    r_star = bootstrap_corr(lsat, gpa, B, SEED)
    se = float(r_star.std(ddof=1))
    Q = np.quantile(r_star, [0.025, 0.975])
    ci_perc = (float(Q[0]), float(Q[1]))
    ci_basic = (2 * r_hat - float(Q[1]), 2 * r_hat - float(Q[0]))
    z = stats.norm.ppf(0.975)
    ci_norm = (r_hat - z * se, r_hat + z * se)

    # jackknife (leave-one-out) correlations -> acceleration for BCa
    jack = np.array([np.corrcoef(np.delete(lsat, i), np.delete(gpa, i))[0, 1]
                     for i in range(n)])
    ci_bca, z0, a = bca_ci(r_star, r_hat, jack)

    # diagnostics (Slide 38)
    skew = float(stats.skew(r_star))
    exkurt = float(stats.kurtosis(r_star))             # excess kurtosis
    bias = float(r_star.mean() - r_hat)
    bias_ratio = abs(bias) / se
    se_grid = [500, 1000, 2000, 5000, 10000]           # SE-stability check
    se_running = [float(r_star[:b].std(ddof=1)) for b in se_grid]

    print("=== Bootstrap CIs for the law-school correlation (Slide 34) ===")
    print(f"n = {n}   r_hat = {r_hat:.4f}   bootstrap SE = {se:.4f}   (B = {B:,}, seed = {SEED})")
    print(f"  Percentile : [{ci_perc[0]:.3f}, {ci_perc[1]:.3f}]   respects boundary")
    print(f"  Basic      : [{ci_basic[0]:.3f}, {ci_basic[1]:.3f}]   "
          f"{'UPPER > 1 -> IMPOSSIBLE (worst)' if ci_basic[1] > 1 else ''}")
    print(f"  Normal     : [{ci_norm[0]:.3f}, {ci_norm[1]:.3f}]   "
          f"{'UPPER > 1 -> IMPOSSIBLE' if ci_norm[1] > 1 else ''}")
    print(f"  BCa (opt 4.7/webbook): [{ci_bca[0]:.3f}, {ci_bca[1]:.3f}]   z0={z0:+.3f}  a={a:+.3f}")
    print()
    print("=== Diagnostics on the same bootstrap (Slide 38) ===")
    print(f"  1. Histogram : skew(r*) = {skew:+.3f}  (left-skewed) -> percentile over normal")
    print(f"  2. Normal Q-Q: excess kurtosis = {exkurt:+.3f}; the left skew curves the Q-Q -> BCa flag")
    print(f"  3. SE vs B   : " + ", ".join(f"B={b}:{s:.4f}" for b, s in zip(se_grid, se_running))
          + "  (flattens by ~2,000)")
    print(f"  4. Bias ratio: |bias|/SE = {bias_ratio:.3f}  (negligible -> issue is SKEW, not bias)")

    figdir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "..", "..", "Website", "assets", "Part2", "Chapter4",
                                           "nonparametric-bootstrap_ch4_3"))
    make_correlation_ci_figure(
        r_star, r_hat, {"perc": ci_perc, "basic": ci_basic, "norm": ci_norm, "bca": ci_bca},
        os.path.join(figdir, "ch4_3_fig09_correlation_ci_law_school.png"))
    make_diagnostics_figure(
        r_star, r_hat, os.path.join(figdir, "ch4_3_fig10_diagnostics_law_school.png"))


if __name__ == "__main__":
    main()
