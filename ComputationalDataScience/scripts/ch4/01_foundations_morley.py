"""
STAT 418 · Chapter 4 — Foundations on real data: Michelson speed-of-light (slides 18, 32).

ONE already-vendored dataset (R `morley`, n = 100, Speed = km/s - 299000) anchors the
ECDF/plug-in arc, extending the Ch5 Normal thread:

  - Slide 18 (DKW band): the 95% Dvoretzky-Kiefer-Wolfowitz band around the ECDF needs
    ONLY the sample (no known true F), so real data is honest. eps_100 is exact.
    This script regenerates  Website/assets/Part2/Chapter4/empirical-distribution-plugin_ch4_2/ch4_2_fig05_dkw_confidence_band.png.
  - Slide 32 (bias): the plug-in variance (1/n)Sum(x-xbar)^2 has bias -sigma^2/n KNOWN
    exactly. morley makes that a real, reproducible number, and the bootstrap recovers it.

Run:  python 01_foundations_morley.py        # needs numpy, matplotlib
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import load_morley  # noqa: E402

SEED = 42
B = 10_000


def dkw_epsilon(n: int, alpha: float = 0.05) -> float:
    """Half-width of the (1-alpha) DKW band: solve 2 exp(-2 n eps^2) = alpha."""
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


def bootstrap_plugin_variance(x: np.ndarray, B: int, seed: int):
    """Bootstrap bias and SE of the plug-in variance T = (1/n) Sum (x-xbar)^2."""
    rng = np.random.default_rng(seed)
    n = len(x)
    var_plugin = x.var(ddof=0)
    vstar = np.empty(B)
    for b in range(B):
        vstar[b] = rng.choice(x, size=n, replace=True).var(ddof=0)
    bias_boot = vstar.mean() - var_plugin          # ideal value = -var_plugin / n exactly
    se_boot = vstar.std(ddof=1)
    return var_plugin, bias_boot, se_boot


def make_dkw_figure(speed: np.ndarray, eps: float, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    x = np.sort(speed)
    n = len(x)
    ecdf = np.arange(1, n + 1) / n
    grid = np.linspace(x.min() - 40, x.max() + 40, 400)

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=150)
    # ECDF as a step function
    ax.step(np.concatenate([[grid[0]], x]), np.concatenate([[0], ecdf]),
            where="post", color="#1f2937", lw=2, label=r"ECDF $\hat F_n$ (n=100)")
    # DKW band (clipped to [0,1]), simultaneous 95% coverage
    lo = np.clip(np.concatenate([[0], ecdf]) - eps, 0, 1)
    hi = np.clip(np.concatenate([[0], ecdf]) + eps, 0, 1)
    xs = np.concatenate([[grid[0]], x])
    ax.fill_between(xs, lo, hi, step="post", color="#34d399", alpha=0.30,
                    label=fr"95% DKW band  ($\varepsilon_{{100}}={eps:.3f}$)")
    # Normal fit as a plausible reference F (NOT a known truth)
    mu, sd = speed.mean(), speed.std(ddof=1)
    ax.plot(grid, stats.norm.cdf(grid, mu, sd), color="#b91c1c", ls="--", lw=1.6,
            label=fr"Normal fit  $N({mu:.0f},\,{sd:.0f}^2)$")
    ax.set_xlabel("Speed of light  (km/s − 299000)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("DKW 95% confidence band — Michelson 1879 (n = 100)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote figure -> {out_path}")


def main() -> None:
    speed = np.array([r["Speed"] for r in load_morley(chapter=4)], dtype=float)
    n = len(speed)
    print(f"morley: n={n}  mean={speed.mean():.4f}  sd={speed.std(ddof=1):.4f}")

    eps = dkw_epsilon(n)
    print(f"\n[Slide 18] DKW 95% band half-width  eps_100 = sqrt(ln(2/0.05)/200) = {eps:.5f}")

    var_plugin, bias_boot, se_boot = bootstrap_plugin_variance(speed, B, SEED)
    analytic_bias = -var_plugin / n
    print(f"\n[Slide 32] plug-in variance (1/n)Sum(x-xbar)^2 = {var_plugin:.2f}")
    print(f"           analytic bias  -sigma^2/n            = {analytic_bias:.2f}  (known exactly)")
    print(f"           bootstrap bias (B={B}, seed={SEED})  = {bias_boot:.2f}")
    print(f"           bootstrap SE                          = {se_boot:.1f}")
    print(f"           ratio R = |analytic bias| / SE        = {abs(analytic_bias)/se_boot:.3f}  "
          f"(<0.25 -> negligible, do NOT bias-correct)")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Website",
                       "assets", "Part2", "Chapter4", "empirical-distribution-plugin_ch4_2",
                       "ch4_2_fig05_dkw_confidence_band.png")
    make_dkw_figure(speed, eps, os.path.normpath(out))


if __name__ == "__main__":
    main()
