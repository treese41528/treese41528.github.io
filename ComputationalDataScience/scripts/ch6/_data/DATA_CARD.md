# STAT 418 — Chapter 6 Reviews Corpus (`reviews.jsonl`)

A small, balanced, **real** product-review corpus used as the shared dataset across
the Chapter 6 (LLMs in Data Science) examples: embeddings → classification →
regression → annotation → evaluation. Students work with genuine Amazon reviews
rather than synthetic toy data.

## Source & attribution

- **Dataset:** Amazon Reviews 2023, McAuley Lab, UC San Diego.
  <https://amazon-reviews-2023.github.io/>
- **Citation:** Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, Julian
  McAuley. *Bridging Language and Items for Retrieval and Recommendation at Scale.*
  arXiv:2403.03952, 2024.
- **Terms:** The McAuley Lab releases these data for **academic / research use**
  (cite the paper above). This file is a **small curated educational subset**
  (9,000 of ~571M reviews) vendored for STAT 418 instruction; it is not the full
  dataset and is not redistributed for any commercial purpose.

## What this is

9,000 reviews sampled from **6 modern product categories**, balanced so every
category and every star level is equally represented (so classifiers can't win by
always predicting 5★, and the rating regression has a real spread).

| Facet | Composition |
|---|---|
| **Total** | 9,000 reviews |
| **Categories** (1,500 each) | Electronics · Cell_Phones_and_Accessories · Software · Video_Games · Office_Products · Beauty_and_Personal_Care |
| **Star rating** (1,800 each) | 1★ · 2★ · 3★ · 4★ · 5★ |
| **Sentiment** | 3,600 positive · 1,800 neutral · 3,600 negative |
| **Year** | 2002–2023 (5,647 from 2018+, bulk 2019–2022) |
| **Verified purchases** | 7,656 / 9,000 |
| **Text length** | 20–600 chars (median 154) |

## Schema (one JSON object per line)

| Field | Type | Notes |
|---|---|---|
| `id` | string | stable row id (`r00001` …) |
| `category` | string | one of the 6 categories above |
| `rating` | int | real star rating, 1–5 |
| `sentiment` | string | **derived** from `rating`: 1–2 → `negative`, 3 → `neutral`, 4–5 → `positive` |
| `title` | string | review headline (real) |
| `text` | string | review body (real, whitespace-normalized) |
| `helpful_vote` | int | helpful votes on Amazon |
| `verified_purchase` | bool | Amazon verified-purchase flag |
| `year` | int | year of the review (from the original timestamp) |

`sentiment` is a documented mapping from the real star rating, not a separate
human annotation — a teachable point in §6.4 (gold labels derived from a proxy
signal). For LLM-vs-gold agreement (Cohen's κ), treat `sentiment` as the gold
label and the LLM's predicted sentiment as the comparison.

## How it was built

`build_reviews.py` (in this folder) **streams** each category's `.jsonl.gz`
directly from the McAuley Lab host and **reservoir-samples** a balanced set,
without downloading the multi-GB full category files. Per category it scans the
first 80,000 records and keeps 300 per star level (`random.seed(42)`).

Cleaning filters: text whitespace-normalized; length 20–600 chars; ≥90% ASCII
(drops non-English/emoji-heavy text); exact-duplicate texts removed.

Re-run with: `python3 scripts/ch6/_data/build_reviews.py`
(Note: it samples a streamed window over a live remote file, so a re-run yields a
statistically equivalent — not byte-identical — corpus. The committed
`reviews.jsonl` is the canonical version the slides and fixtures are computed from.)

## Intended use in Chapter 6

- **§6.2 Embeddings** — embed reviews; cosine similarity; decode what each PC
  encodes (length / sentiment / category) via PCA + LDA; logistic-regression
  sentiment classification (CV accuracy); embedding-PC regression on `rating`
  + bootstrap CIs.
- **§6.4 Annotation** — LLM sentiment annotation vs the derived gold label;
  Cohen's κ + bootstrap CI (computed against a committed LLM-output fixture).
- **§6.8 Reliability / §6.10 Summary** — accuracy / F1 / ECE over a frozen
  prediction fixture; end-to-end "embed → classify → annotate → evaluate".

Deterministic statistics (embeddings, cosine, κ/ECE over fixed labels) reproduce
exactly; any number that comes from a live LLM generation is one stochastic draw
and is frozen to a committed fixture before being shown on a slide.
