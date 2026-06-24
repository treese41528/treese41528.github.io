"""Recompute all §6.2 numbers + gemma kappa from CACHED data (no API)."""
import json, os, collections
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, accuracy_score, cohen_kappa_score, confusion_matrix

DATA = os.path.dirname(os.path.abspath(__file__)); FIX = os.path.join(DATA, 'fixtures')
recs = [json.loads(l) for l in open(os.path.join(DATA, 'reviews.jsonl'), encoding='utf-8')]
texts = [r['text'] for r in recs]
ratings = np.array([r['rating'] for r in recs])
sentiments = np.array([r['sentiment'] for r in recs])
categories = np.array([r['category'] for r in recs])
X = np.load(os.path.join(FIX, 'embeddings_llama3.2_full.npy'))
assert X.shape[0] == len(recs)
R = {'corpus_n': len(recs), 'embed_model': 'llama3.2:latest', 'embed_dim': int(X.shape[1])}

# s26 classification
mask = np.isin(sentiments, ['positive', 'negative'])
Xc, yc = X[mask], (sentiments[mask] == 'positive').astype(int)
cv = cross_val_score(LogisticRegression(max_iter=1000, random_state=42), Xc, yc, cv=5, scoring='accuracy')
R['s26_classification'] = {'n': int(mask.sum()), 'cv_acc_mean': round(float(cv.mean()), 3), 'cv_acc_std': round(float(cv.std()), 3)}
clf = LogisticRegression(max_iter=1000, random_state=42).fit(Xc, yc)
R['s26_examples'] = {}
for star in (5, 1, 3):
    i = int(np.where(ratings == star)[0][0]); p = clf.predict_proba(X[i:i+1])[0]
    R['s26_examples'][f'{star}star'] = {'pred': 'positive' if p[1] >= .5 else 'negative', 'p': round(float(max(p)), 3), 'text': texts[i][:80]}

# s27/s28 regression + bootstrap
lengths = np.array([len(t.split()) for t in texts])
pcs = PCA(n_components=5, random_state=42).fit_transform(X)
Xr = np.column_stack([pcs, lengths]); names = [f'PC{i+1}' for i in range(5)] + ['review_length']
reg = LinearRegression().fit(Xr, ratings)
R['s27_regression'] = {'n': len(recs), 'r2': round(float(reg.score(Xr, ratings)), 3),
                       'coef': {n: round(float(c), 4) for n, c in zip(names, reg.coef_)}, 'intercept': round(float(reg.intercept_), 4)}
rng = np.random.default_rng(42); n = len(ratings); Bc = np.zeros((1000, Xr.shape[1]))
for b in range(1000):
    idx = rng.choice(n, n, replace=True); Bc[b] = LinearRegression().fit(Xr[idx], ratings[idx]).coef_
R['s28_bootstrap'] = {names[i]: [round(float(np.percentile(Bc[:, i], 2.5)), 4), round(float(np.percentile(Bc[:, i], 97.5)), 4),
                                 bool(np.percentile(Bc[:, i], 2.5) > 0 or np.percentile(Bc[:, i], 97.5) < 0)] for i in range(len(names))}

# s24/s25 clustering by category AND by sentiment
def cluster_eval(k, truth):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    ari = adjusted_rand_score(truth, km.labels_)
    pur = sum(collections.Counter(truth[km.labels_ == c]).most_common(1)[0][1] for c in range(k) if (km.labels_ == c).any()) / len(truth)
    return round(float(ari), 3), round(float(pur), 3), km.labels_
ari_cat, pur_cat, _ = cluster_eval(6, categories)
ari_sent, pur_sent, lab_sent = cluster_eval(3, sentiments)
R['s24_clustering'] = {'by_category_k6': {'ari': ari_cat, 'purity': pur_cat},
                       'by_sentiment_k3': {'ari': ari_sent, 'purity': pur_sent}}
pca10 = PCA(n_components=10, random_state=42).fit(X)
R['s25_pca'] = {'pc1_var': round(float(pca10.explained_variance_ratio_[0]), 3),
                'pc2_var': round(float(pca10.explained_variance_ratio_[1]), 3),
                'top10_var': round(float(pca10.explained_variance_ratio_[:10].sum()), 3),
                'silhouette_sentiment': round(float(silhouette_score(X, lab_sent, sample_size=2000, random_state=42)), 3)}

# s23 NN retrieval
q = int(np.where(ratings == 1)[0][0])
sims = X @ X[q] / (np.linalg.norm(X, axis=1) * np.linalg.norm(X[q]) + 1e-9)
top = [int(i) for i in np.argsort(-sims) if i != q][:5]
R['s23_retrieval'] = {'query': texts[q][:90], 'query_rating': int(ratings[q]),
                      'neighbors': [{'sim': round(float(sims[i]), 3), 'rating': int(ratings[i]), 'text': texts[i][:70]} for i in top]}

# kappa from fixtures
def kappa_from(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    ok = [x for x in rows if x.get('llm')]
    g = [x['gold'] for x in ok]; p = [x['llm'] for x in ok]; labs = ['pos', 'neu', 'neg']
    rng2 = np.random.default_rng(42); m = len(g); bb = []
    for _ in range(1000):
        idx = rng2.choice(m, m, replace=True); bb.append(cohen_kappa_score([g[i] for i in idx], [p[i] for i in idx], labels=labs))
    return {'n_sampled': len(rows), 'n_parsed': len(ok), 'timeouts': sum(1 for x in rows if x['llm_raw'] is None),
            'accuracy': round(accuracy_score(g, p), 3), 'kappa': round(float(cohen_kappa_score(g, p, labels=labs)), 3),
            'kappa_ci95': [round(float(np.percentile(bb, 2.5)), 3), round(float(np.percentile(bb, 97.5)), 3)],
            'confusion_labels': labs, 'confusion': confusion_matrix(g, p, labels=labs).tolist()}
R['kappa'] = {'gemma3:12b_n300': kappa_from(os.path.join(FIX, 'annotation_gemma3-12b_n300.jsonl')),
              'qwen2.5:72b_n90': kappa_from(os.path.join(FIX, 'annotation_qwen2.5-72b.jsonl'))}

json.dump(R, open(os.path.join(FIX, 'full_sweep_results.json'), 'w'), indent=2)
print(json.dumps({k: v for k, v in R.items() if k not in ('s23_retrieval', 's26_examples')}, indent=2))
