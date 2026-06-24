"""
STAT 418 · Chapter 6 — FULL-SWEEP compute over the real reviews corpus.

Embeds the FULL corpus (9,000 real Amazon reviews) once with the course model and
computes every §6.2 example number on real data, then runs the §6.4 annotation
kappa SWEEP across two chat models. Writes fixtures + a results JSON the slides
read from. Embeddings/cosine/clustering/regression numbers are deterministic and
reproduce exactly; the LLM annotation is frozen to a fixture.

Run (service must be up):  python3 compute_full_sweep.py
"""
import os, json, time, random, requests, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ['GENAI_STUDIO_API_KEY']; BASE = 'https://genai.rcac.purdue.edu/api'
H = {'Authorization': f'Bearer {KEY}', 'Content-Type': 'application/json'}
DATA = os.path.dirname(os.path.abspath(__file__)); FIX = os.path.join(DATA, 'fixtures')
os.makedirs(FIX, exist_ok=True)
EMB_MODEL = 'llama3.2:latest'
CHAT_MODELS = ['gemma3:12b', 'qwen2.5:72b']
random.seed(42)

recs = [json.loads(l) for l in open(os.path.join(DATA, 'reviews.jsonl'), encoding='utf-8')]
texts = [r['text'] for r in recs]
ratings = np.array([r['rating'] for r in recs])
sentiments = np.array([r['sentiment'] for r in recs])
categories = np.array([r['category'] for r in recs])
log = lambda m: (print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True))

# ───────────────────────── PART A: embed full corpus ─────────────────────────
EMB_PATH = os.path.join(FIX, 'embeddings_llama3.2_full.npy')
def embed_batch(batch, to=120):
    for _ in range(4):
        try:
            r = requests.post(f'{BASE}/embeddings', headers=H,
                              json={'model': EMB_MODEL, 'input': batch}, timeout=to)
            if r.ok:
                return [d['embedding'] for d in r.json()['data']]
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError('embed batch failed after retries')

if os.path.exists(EMB_PATH):
    X = np.load(EMB_PATH); log(f'loaded cached embeddings {X.shape}')
else:
    log(f'embedding {len(texts)} reviews with {EMB_MODEL} ...')
    vecs, B, t0 = [], 128, time.time()
    for i in range(0, len(texts), B):
        vecs.extend(embed_batch(texts[i:i + B]))
        if (i // B) % 10 == 0:
            log(f'  {min(i + B, len(texts))}/{len(texts)} ({time.time()-t0:.0f}s)')
    X = np.asarray(vecs, dtype=np.float32)
    np.save(EMB_PATH, X); log(f'embedded + cached {X.shape} in {time.time()-t0:.0f}s')

results = {'corpus_n': len(recs), 'embed_model': EMB_MODEL, 'embed_dim': int(X.shape[1])}

# ───────────────────────── PART B: §6.2 numbers ─────────────────────────
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

# s26 — classification (positive vs negative)
mask = np.isin(sentiments, ['positive', 'negative'])
Xc, yc = X[mask], (sentiments[mask] == 'positive').astype(int)
cv = cross_val_score(LogisticRegression(max_iter=1000, random_state=42), Xc, yc, cv=5, scoring='accuracy')
results['s26_classification'] = {'n': int(mask.sum()), 'cv_acc_mean': round(float(cv.mean()), 3),
                                 'cv_acc_std': round(float(cv.std()), 3)}
clf = LogisticRegression(max_iter=1000, random_state=42).fit(Xc, yc)
held = {}
for star in (5, 1, 3):
    idx = np.where(ratings == star)[0][0]
    p = clf.predict_proba(X[idx:idx + 1])[0]
    pred = 'positive' if p[1] >= 0.5 else 'negative'
    held[f'{star}star'] = {'pred': pred, 'p': round(float(max(p)), 3), 'text': texts[idx][:80]}
results['s26_examples'] = held
log(f"s26 classification CV acc = {cv.mean():.3f} +/- {cv.std():.3f} (n={mask.sum()})")

# s27/s28 — regression of star rating on embedding PCs + review length, bootstrap CIs
lengths = np.array([len(t.split()) for t in texts])
pcs = PCA(n_components=5, random_state=42).fit_transform(X)
Xr = np.column_stack([pcs, lengths]); names = [f'PC{i+1}' for i in range(5)] + ['review_length']
reg = LinearRegression().fit(Xr, ratings)
results['s27_regression'] = {'n': len(recs), 'r2': round(float(reg.score(Xr, ratings)), 3),
                             'coef': {n: round(float(c), 4) for n, c in zip(names, reg.coef_)},
                             'intercept': round(float(reg.intercept_), 4)}
rng = np.random.default_rng(42); n = len(ratings); B = np.zeros((1000, Xr.shape[1]))
for b in range(1000):
    idx = rng.choice(n, n, replace=True)
    B[b] = LinearRegression().fit(Xr[idx], ratings[idx]).coef_
results['s28_bootstrap'] = {names[i]: [round(float(np.percentile(B[:, i], 2.5)), 4),
                                       round(float(np.percentile(B[:, i], 97.5)), 4)] for i in range(len(names))}
log(f"s27 regression R2 = {results['s27_regression']['r2']}  PC1 coef = {results['s27_regression']['coef']['PC1']}")

# s24 — clustering vs category; s25 — PCA variance + silhouette
km = KMeans(n_clusters=6, n_init=10, random_state=42).fit(X)
ari = adjusted_rand_score(categories, km.labels_)
purity = 0
for c in range(6):
    m = km.labels_ == c
    if m.sum():
        purity += collections.Counter(categories[m]).most_common(1)[0][1]
purity /= len(recs)
results['s24_clustering'] = {'k': 6, 'ari': round(float(ari), 3), 'purity': round(float(purity), 3)}
pca_full = PCA(n_components=10, random_state=42).fit(X)
sil = silhouette_score(X, km.labels_, sample_size=2000, random_state=42)
results['s25_pca'] = {'pc1_var': round(float(pca_full.explained_variance_ratio_[0]), 3),
                      'pc2_var': round(float(pca_full.explained_variance_ratio_[1]), 3),
                      'top10_var': round(float(pca_full.explained_variance_ratio_[:10].sum()), 3),
                      'silhouette': round(float(sil), 3)}
log(f"s24 clustering ARI={ari:.3f} purity={purity:.3f} | s25 PC1var={results['s25_pca']['pc1_var']}")

# s22 — sentiment-word cosine ranking (small fixed list, domain words)
def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
words = ['excellent', 'great', 'good', 'decent', 'okay', 'mediocre', 'poor', 'terrible', 'awful', 'camera', 'battery']
wv = embed_batch(words)
anchor = wv[0]
results['s22_word_cosine'] = {'anchor': 'excellent',
                              'ranking': [[w, round(cos(anchor, v), 3)] for w, v in sorted(
                                  zip(words[1:], wv[1:]), key=lambda kv: -cos(anchor, kv[1]))]}

# s23 — nearest-neighbor retrieval over real reviews
q = np.where(ratings == 1)[0][0]
sims = X @ X[q] / (np.linalg.norm(X, axis=1) * np.linalg.norm(X[q]) + 1e-9)
order = np.argsort(-sims)
top = [i for i in order if i != q][:5]
results['s23_retrieval'] = {'query': texts[q][:90], 'query_rating': int(ratings[q]),
                            'neighbors': [{'sim': round(float(sims[i]), 3), 'rating': int(ratings[i]),
                                           'text': texts[i][:70]} for i in top]}
log('s22/s23 cosine + retrieval done')

# ───────────────────────── PART C: §6.4 kappa sweep ─────────────────────────
by = {s: [r for r in recs if r['sentiment'] == s] for s in ('positive', 'neutral', 'negative')}
ksample = sum([random.sample(by[s], 100) for s in ('positive', 'neutral', 'negative')], [])
random.shuffle(ksample)
SYS = ("You are a precise sentiment classifier for product reviews. Reply with EXACTLY one "
       "lowercase word and nothing else — one of: positive, negative, neutral. If the review "
       "is mixed or unclear, answer neutral.")
GMAP = {'positive': 'pos', 'neutral': 'neu', 'negative': 'neg'}

def annotate_one(model, text):
    body = {'model': model, 'temperature': 0, 'stream': False,
            'messages': [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': 'Review: ' + text}]}
    for _ in range(4):
        try:
            r = requests.post(f'{BASE}/chat/completions', headers=H, json=body, timeout=120)
            if r.status_code == 429:
                time.sleep(8); continue
            if r.ok:
                return r.json()['choices'][0]['message']['content']
        except Exception:
            pass
        time.sleep(4)
    return None

def parse(raw):
    if not raw: return None
    t = raw.strip().lower()
    for lab, sh in (('positive', 'pos'), ('negative', 'neg'), ('neutral', 'neu')):
        if lab in t: return sh
    if 'mixed' in t or 'unclear' in t: return 'neu'
    return None

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
results['kappa_sweep'] = {}
for model in CHAT_MODELS:
    log(f'annotating {len(ksample)} reviews with {model} ...')
    t0 = time.time(); rows = [None] * len(ksample)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(annotate_one, model, r['text']): i for i, r in enumerate(ksample)}
        done = 0
        for fut in list(futs):
            i = futs[fut]; raw = fut.result()
            rows[i] = {'id': ksample[i]['id'], 'rating': ksample[i]['rating'], 'category': ksample[i]['category'],
                       'gold': GMAP[ksample[i]['sentiment']], 'llm_raw': raw, 'llm': parse(raw), 'text': ksample[i]['text']}
            done += 1
            if done % 50 == 0: log(f'  {model}: {done}/{len(ksample)} ({time.time()-t0:.0f}s)')
    slug = model.replace(':', '-')
    with open(os.path.join(FIX, f'annotation_{slug}_n{len(ksample)}.jsonl'), 'w', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + '\n')
    ok = [x for x in rows if x['llm']]
    g = [x['gold'] for x in ok]; p = [x['llm'] for x in ok]; labs = ['pos', 'neu', 'neg']
    kap = cohen_kappa_score(g, p, labels=labs)
    rng2 = np.random.default_rng(42); m = len(g); bb = []
    for _ in range(1000):
        idx = rng2.choice(m, m, replace=True)
        bb.append(cohen_kappa_score([g[i] for i in idx], [p[i] for i in idx], labels=labs))
    results['kappa_sweep'][model] = {
        'n_sampled': len(ksample), 'n_parsed': len(ok), 'timeouts': sum(1 for x in rows if x['llm_raw'] is None),
        'accuracy': round(accuracy_score(g, p), 3), 'kappa': round(float(kap), 3),
        'kappa_ci95': [round(float(np.percentile(bb, 2.5)), 3), round(float(np.percentile(bb, 97.5)), 3)],
        'confusion_labels': labs, 'confusion': confusion_matrix(g, p, labels=labs).tolist(),
        'clears_0.6': bool(np.percentile(bb, 2.5) >= 0.6), 'seconds': round(time.time() - t0, 1)}
    log(f"  {model}: kappa={results['kappa_sweep'][model]['kappa']} CI={results['kappa_sweep'][model]['kappa_ci95']} "
        f"parsed={len(ok)}/{len(ksample)}")

with open(os.path.join(FIX, 'full_sweep_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
log('DONE — wrote fixtures/full_sweep_results.json')
print(json.dumps({k: v for k, v in results.items() if k != 's23_retrieval'}, indent=2))
