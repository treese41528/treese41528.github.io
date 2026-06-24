"""
STAT 418 · Chapter 6 — shared loader for the real reviews corpus.

`reviews.jsonl` is 9,000 real Amazon product reviews (6 modern categories,
balanced across 1-5 stars) — see DATA_CARD.md for provenance and terms. The
canonical copy is hosted on the course Supabase bucket; this loader fetches it
(caching locally) so any Chapter 6 script can pull the same data with one call:

    from _data.loader import load_reviews, balanced_sample
    reviews = load_reviews()
    train = balanced_sample(reviews, per_class=200)   # 200 pos / 200 neu / 200 neg

`sentiment` is derived from the real star rating (1-2 negative, 3 neutral,
4-5 positive); `rating` is the real 1-5 star value.
"""
from __future__ import annotations

import json
import os
import urllib.request

CORPUS_URL = ("https://pqyjaywwccbnqpwgeiuv.supabase.co/storage/v1/object/public/"
              "STAT%20418%20Images/Data/Chapter6/reviews.jsonl")
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reviews.jsonl")

SENTIMENTS = ("positive", "neutral", "negative")


def load_reviews(url: str = CORPUS_URL, cache: str = _CACHE) -> list[dict]:
    """Return the corpus as a list of dicts. Uses the local cache if present,
    otherwise downloads from the course Supabase bucket and caches it."""
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        print(f"Downloading reviews corpus -> {cache}")
        urllib.request.urlretrieve(url, cache)
    with open(cache, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def balanced_sample(reviews: list[dict], per_class: int,
                    key: str = "sentiment", classes=SENTIMENTS, seed: int = 42) -> list[dict]:
    """Reproducible class-balanced sample: `per_class` rows from each class."""
    import random
    rng = random.Random(seed)
    out: list[dict] = []
    for c in classes:
        pool = [r for r in reviews if r[key] == c]
        out += rng.sample(pool, min(per_class, len(pool)))
    rng.shuffle(out)
    return out


def stratified_by_rating(reviews: list[dict], per_rating: int, seed: int = 42) -> list[dict]:
    """Reproducible sample balanced across the real 1-5 star ratings."""
    import random
    rng = random.Random(seed)
    out: list[dict] = []
    for star in (1, 2, 3, 4, 5):
        pool = [r for r in reviews if r["rating"] == star]
        out += rng.sample(pool, min(per_rating, len(pool)))
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    revs = load_reviews()
    import collections
    print(f"Loaded {len(revs)} reviews")
    print("by sentiment:", dict(collections.Counter(r["sentiment"] for r in revs)))
    print("by rating:", dict(sorted(collections.Counter(r["rating"] for r in revs).items())))
