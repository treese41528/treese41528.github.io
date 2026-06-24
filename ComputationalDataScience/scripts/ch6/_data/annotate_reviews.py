"""
STAT 418 · Chapter 6.4 — generate the frozen LLM-annotation fixture.

Annotates a class-balanced sample of the real reviews corpus with an LLM
sentiment label, then compares to the gold label (derived from the star rating)
with Cohen's kappa + a bootstrap CI. The LLM call is stochastic, so we FREEZE the
outputs to a committed fixture; downstream (03_quality_control.py) recomputes the
deterministic statistics from that fixture.

Strict prompt: a system message forces exactly one lowercase word from the 3-class
set and routes "mixed/unclear" to neutral, so outputs parse cleanly.

Run:  GENAI_STUDIO_API_KEY=... python3 annotate_reviews.py
      ANNOTATE_MODEL=gemma3:12b python3 annotate_reviews.py   # override model
"""
import os, json, time, random, requests, collections
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

random.seed(42)
KEY = os.environ['GENAI_STUDIO_API_KEY']; BASE = 'https://genai.rcac.purdue.edu/api'
H = {'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'}
MODEL = os.environ.get('ANNOTATE_MODEL', 'qwen2.5:72b')
PER_CLASS = 30                   # 30 each of pos/neu/neg -> n = 90
CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reviews.jsonl")
FIXDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
os.makedirs(FIXDIR, exist_ok=True)
SLUG = MODEL.replace(':', '-')

SYSTEM = ("You are a precise sentiment classifier for product reviews. "
          "Reply with EXACTLY one lowercase word and nothing else — one of: "
          "positive, negative, neutral. If the review is mixed or unclear, answer neutral.")

recs = [json.loads(l) for l in open(CORPUS, encoding='utf-8')]
by = {'positive': [], 'neutral': [], 'negative': []}
for r in recs:
    by[r['sentiment']].append(r)
sample = []
for s in ('positive', 'neutral', 'negative'):
    sample += random.sample(by[s], PER_CLASS)
random.shuffle(sample)


def annotate(text):
    body = {'model': MODEL, 'temperature': 0, 'stream': False,
            'messages': [{'role': 'system', 'content': SYSTEM},
                         {'role': 'user', 'content': 'Review: ' + text}]}
    for _ in range(3):
        try:
            r = requests.post(f'{BASE}/chat/completions', headers=H, json=body, timeout=120)
            if r.ok:
                return r.json()['choices'][0]['message']['content']
        except Exception:
            pass
        time.sleep(3)
    return None


def parse(raw):
    if not raw:
        return None
    t = raw.strip().lower()
    for lab, short in (('positive', 'pos'), ('negative', 'neg'), ('neutral', 'neu')):
        if lab in t:
            return short
    if 'mixed' in t or 'unclear' in t:   # strict prompt routes these to neutral
        return 'neu'
    return None


GMAP = {'positive': 'pos', 'neutral': 'neu', 'negative': 'neg'}
fixture, t0 = [], time.time()
for i, r in enumerate(sample, 1):
    raw = annotate(r['text'])
    fixture.append({'id': r['id'], 'category': r['category'], 'rating': r['rating'],
                    'gold': GMAP[r['sentiment']], 'llm_raw': raw, 'llm': parse(raw), 'text': r['text']})
    if i % 15 == 0:
        import sys; sys.stderr.write(f"  annotated {i}/{len(sample)} ({time.time()-t0:.0f}s)\n"); sys.stderr.flush()
    time.sleep(0.3)

fixpath = os.path.join(FIXDIR, f'annotation_{SLUG}.jsonl')
with open(fixpath, 'w', encoding='utf-8') as f:
    for row in fixture:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

ok = [x for x in fixture if x['llm'] is not None]
gold = [x['gold'] for x in ok]; pred = [x['llm'] for x in ok]
labels = ['pos', 'neu', 'neg']
acc = accuracy_score(gold, pred); kappa = cohen_kappa_score(gold, pred, labels=labels)
agree = sum(1 for g, p in zip(gold, pred) if g == p)
rng = np.random.default_rng(42); n = len(gold); boot = []
for _ in range(1000):
    idx = rng.choice(n, n, replace=True)
    boot.append(cohen_kappa_score([gold[i] for i in idx], [pred[i] for i in idx], labels=labels))
lo, hi = np.percentile(boot, [2.5, 97.5])
result = {'model': MODEL, 'n_sampled': len(sample), 'n_parsed': len(ok),
          'unparsed': len(fixture) - len(ok), 'agree': agree,
          'accuracy': round(acc, 3), 'kappa': round(float(kappa), 3),
          'kappa_ci95': [round(float(lo), 3), round(float(hi), 3)],
          'confusion_labels': labels, 'confusion': confusion_matrix(gold, pred, labels=labels).tolist(),
          'per_gold_counts': dict(collections.Counter(gold)), 'fixture': fixpath,
          'seconds': round(time.time() - t0, 1)}
print(json.dumps(result, indent=2))
with open(os.path.join(FIXDIR, f'annotation_result_{SLUG}.json'), 'w') as f:
    json.dump(result, f, indent=2)
