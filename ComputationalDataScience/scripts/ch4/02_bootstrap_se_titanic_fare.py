"""
STAT 418 · Chapter 4 — Bootstrap SE on real, skewed data: Titanic passenger fares (slide 31).

Replaces the old synthetic "MEPS-calibrated" healthcare sample. The Titanic fares are
genuinely right-skewed (a few first-class fares near $512 against a typical $7-15), so the
bootstrap shows the MEDIAN is estimated more precisely than the MEAN -- and there is no
simple closed-form SE for the median, which is the bootstrap's whole value proposition.

We use the n = 876 passengers with a positive recorded fare (drop the 15 zero/complimentary
fares). Sanity check: bootstrap SE(mean) should match the analytic s/sqrt(n).

Run:  python 02_bootstrap_se_titanic_fare.py        # needs numpy
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_data"))
from loader import load_titanic  # noqa: E402

SEED = 42
B = 10_000


def bootstrap_se(x: np.ndarray, stat, B: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = np.empty(B)
    for b in range(B):
        vals[b] = stat(rng.choice(x, size=n, replace=True))
    return vals.std(ddof=1)


def main() -> None:
    fares = np.array([float(r["Fare"]) for r in load_titanic() if r["Fare"] not in ("", "NA")])
    fares = fares[fares > 0]                       # drop 15 zero fares -> n = 876
    n = len(fares)
    s = fares.std(ddof=1)
    print(f"Titanic Fare>0: n={n}  mean={fares.mean():.2f}  median={np.median(fares):.2f}  "
          f"skew={float(__import__('pandas').Series(fares).skew()):.2f}")
    print(f"  five-number: min={fares.min():.2f}  Q1={np.percentile(fares,25):.2f}  "
          f"med={np.median(fares):.2f}  Q3={np.percentile(fares,75):.2f}  max={fares.max():.2f}")

    se_mean = bootstrap_se(fares, np.mean, B, SEED)
    se_med = bootstrap_se(fares, np.median, B, SEED)
    print(f"\nbootstrap (B={B}, seed={SEED}):")
    print(f"  SE(mean)   = {se_mean:.4f}   (analytic s/sqrt(n) = {s/np.sqrt(n):.4f}  -- sanity check)")
    print(f"  SE(median) = {se_med:.4f}   (no simple closed form)")
    print(f"  ratio SE(mean)/SE(median) = {se_mean/se_med:.2f}x  "
          f"-> the median is estimated ~{se_mean/se_med:.1f}x more precisely under heavy right skew")


if __name__ == "__main__":
    main()
