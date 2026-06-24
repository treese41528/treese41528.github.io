import urllib.request, gzip, json, time, random, os, sys, collections
random.seed(42)

BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories"
CATS = ["Electronics", "Cell_Phones_and_Accessories", "Software",
        "Video_Games", "Office_Products", "Beauty_and_Personal_Care"]
PER_CELL = 300          # target per (category x rating) cell -> balanced
SCAN_CAP = 80000        # rows streamed per category (curl stops after this)
MIN_LEN, MAX_LEN = 20, 600
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reviews.jsonl")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def clean(t):
    return " ".join(str(t or "").split())

def english_ok(t):
    if not t:
        return False
    asc = sum(1 for c in t if ord(c) < 128)
    return asc / len(t) >= 0.9

seen = set()
buckets = {}   # (cat,rating) -> reservoir list
counts = {}    # (cat,rating) -> seen count

def consider(cat, rating, rec):
    key = (cat, rating)
    counts[key] = counts.get(key, 0) + 1
    lst = buckets.setdefault(key, [])
    if len(lst) < PER_CELL:
        lst.append(rec)
    else:
        j = random.randint(0, counts[key] - 1)
        if j < PER_CELL:
            lst[j] = rec

for cat in CATS:
    url = f"{BASE}/{cat}.jsonl.gz"
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'curl/8'})
        resp = urllib.request.urlopen(req, timeout=120)
        gz = gzip.GzipFile(fileobj=resp)
        scanned = 0
        for line in gz:
            scanned += 1
            if scanned > SCAN_CAP:
                break
            try:
                r = json.loads(line)
                rating = int(round(float(r.get('rating'))))
            except Exception:
                continue
            if rating < 1 or rating > 5:
                continue
            text = clean(r.get('text'))
            if not (MIN_LEN <= len(text) <= MAX_LEN):
                continue
            if not english_ok(text):
                continue
            tl = text.lower()
            if tl in seen:
                continue
            ts = r.get('timestamp') or 0
            try:
                year = time.gmtime(ts / 1000).tm_year
            except Exception:
                year = None
            sentiment = "negative" if rating <= 2 else ("neutral" if rating == 3 else "positive")
            rec = {"category": cat, "rating": rating, "sentiment": sentiment,
                   "title": clean(r.get('title')), "text": text,
                   "helpful_vote": int(r.get('helpful_vote') or 0),
                   "verified_purchase": bool(r.get('verified_purchase')), "year": year}
            seen.add(tl)
            consider(cat, rating, rec)
            if scanned % 20000 == 0:
                sys.stderr.write(f"[{cat}] scanned {scanned}\n"); sys.stderr.flush()
        gz.close(); resp.close()
        got = {r: len(buckets.get((cat, r), [])) for r in range(1, 6)}
        sys.stderr.write(f"DONE {cat}: scanned {scanned} got {got} in {int(time.time()-t0)}s\n"); sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"ERR {cat}: {e}\n"); sys.stderr.flush()

allrecs = []
for lst in buckets.values():
    allrecs.extend(lst)
random.shuffle(allrecs)
records = [{"id": f"r{i:05d}", **rec} for i, rec in enumerate(allrecs, 1)]
with open(OUT, 'w') as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("TOTAL:", len(records))
print("by category:", dict(collections.Counter(r['category'] for r in records)))
print("by rating:", dict(sorted(collections.Counter(r['rating'] for r in records).items())))
print("by sentiment:", dict(collections.Counter(r['sentiment'] for r in records)))
print("year range:", min((r['year'] for r in records if r['year']), default=None), "-", max((r['year'] for r in records if r['year']), default=None))
print("file:", OUT, f"{os.path.getsize(OUT)/1e6:.2f} MB")
