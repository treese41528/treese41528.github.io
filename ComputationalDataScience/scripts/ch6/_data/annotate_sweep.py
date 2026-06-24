"""
STAT 418 · Chapter 6.4 — kappa MODEL SWEEP on a fixed review subset.

Annotates the SAME fixed n=300 balanced sample with several fast chat models, then
computes Cohen's kappa (+ bootstrap CI) per model — a clean apples-to-apples sweep.

Design (learned from a prior hung run):
  * SINGLE-THREADED, paced to <= 20 requests/min (the gateway's hard cap) — so no
    429s, no concurrency timeouts: every row gets labeled.
  * CHECKPOINTED + RESUMABLE: each annotation is appended to the model's fixture
    immediately; re-running skips rows already labeled (so a stall never costs progress).

Run:  python3 annotate_sweep.py        (resumable — safe to re-run)
"""
import os, json, time, random, requests, collections
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

KEY = os.environ['GENAI_STUDIO_API_KEY']; BASE = 'https://genai.rcac.purdue.edu/api'
H = {'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'}
DATA = os.path.dirname(os.path.abspath(__file__)); FIX = os.path.join(DATA, 'fixtures')
MODELS = ['llama3.2:latest', 'mistral:latest', 'gemma3:12b', 'phi4:latest']
PER_CLASS = 100                      # n = 300 (100 pos / 100 neu / 100 neg)
MIN_INTERVAL = 3.2                   # seconds between call starts -> <= ~19/min (under the 20/min cap)
random.seed(42)

SYSTEM = ("You are a precise sentiment classifier for product reviews. Reply with EXACTLY one "
          "lowercase word and nothing else — one of: positive, negative, neutral. If the review "
          "is mixed or unclear, answer neutral.")
GMAP = {'positive': 'pos', 'neutral': 'neu', 'negative': 'neg'}

recs = [json.loads(l) for l in open(os.path.join(DATA, 'reviews.jsonl'), encoding='utf-8')]
by = {s: [r for r in recs if r['sentiment'] == s] for s in ('positive', 'neutral', 'negative')}
sample = sum([random.sample(by[s], PER_CLASS) for s in ('positive', 'neutral', 'negative')], [])
random.shuffle(sample)

_last = [0.0]
def throttled_post(body):
    wait = MIN_INTERVAL - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    return requests.post(f'{BASE}/chat/completions', headers=H, json=body, timeout=90)

def annotate(model, text):
    body = {'model': model, 'temperature': 0, 'stream': False,
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': 'Review: ' + text}]}
    for _ in range(3):
        try:
            r = throttled_post(body)
            if r.status_code == 429:
                time.sleep(10); continue
            if r.ok:
                return r.json()['choices'][0]['message']['content']
        except Exception:
            pass
        time.sleep(4)
    return None

def parse(raw):
    if not raw:
        return None
    t = raw.strip().lower()
    for lab, sh in (('positive', 'pos'), ('negative', 'neg'), ('neutral', 'neu')):
        if lab in t:
            return sh
    if 'mixed' in t or 'unclear' in t:
        return 'neu'
    return None

def fixture_path(model):
    return os.path.join(FIX, f'sweep_{model.replace(":", "-")}.jsonl')

def load_done(path):
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            if r.get('llm') is not None:        # only count clean labels as done
                done[r['id']] = r
    return done

results = {}
for model in MODELS:
    path = fixture_path(model); done = load_done(path)
    todo = [r for r in sample if r['id'] not in done]
    print(f"[{time.strftime('%H:%M:%S')}] {model}: {len(done)} done, {len(todo)} to do", flush=True)
    t0 = time.time()
    with open(path, 'a', encoding='utf-8') as f:
        for i, r in enumerate(todo, 1):
            raw = annotate(model, r['text'])
            row = {'id': r['id'], 'category': r['category'], 'rating': r['rating'],
                   'gold': GMAP[r['sentiment']], 'llm_raw': raw, 'llm': parse(raw), 'text': r['text']}
            f.write(json.dumps(row, ensure_ascii=False) + '\n'); f.flush()
            if i % 25 == 0:
                print(f"    {model}: {i}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    # metrics over the full sample (reload everything for this model)
    allrows = {json.loads(l)['id']: json.loads(l) for l in open(path, encoding='utf-8')}
    rows = [allrows[r['id']] for r in sample if r['id'] in allrows]
    ok = [x for x in rows if x['llm']]
    g = [x['gold'] for x in ok]; p = [x['llm'] for x in ok]; labs = ['pos', 'neu', 'neg']
    if len(set(g)) < 2:
        results[model] = {'error': 'insufficient labels'}; continue
    kap = cohen_kappa_score(g, p, labels=labs)
    rng = np.random.default_rng(42); m = len(g); bb = []
    for _ in range(1000):
        idx = rng.choice(m, m, replace=True)
        bb.append(cohen_kappa_score([g[i] for i in idx], [p[i] for i in idx], labels=labs))
    lo, hi = np.percentile(bb, [2.5, 97.5])
    results[model] = {'n_labeled': len(ok), 'n_sample': len(sample),
                      'accuracy': round(accuracy_score(g, p), 3), 'kappa': round(float(kap), 3),
                      'kappa_ci95': [round(float(lo), 3), round(float(hi), 3)],
                      'clears_0.6': bool(lo >= 0.6),
                      'confusion_labels': labs, 'confusion': confusion_matrix(g, p, labels=labs).tolist()}
    print(f"    -> {model}: kappa={results[model]['kappa']} CI={results[model]['kappa_ci95']} "
          f"acc={results[model]['accuracy']} labeled={len(ok)}/{len(sample)}", flush=True)

json.dump(results, open(os.path.join(FIX, 'kappa_sweep_results.json'), 'w'), indent=2)
print('\n=== KAPPA SWEEP ===', flush=True)
print(json.dumps(results, indent=2))
