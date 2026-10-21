"""Rank the soccer codebook's passages by similarity to one question, as semantic retrieval would.

Slide 13 shows how semantic retrieval works: the question and every passage become embeddings, and the
passages with the highest cosine similarity to the question are selected as candidate context (top 3 here,
a setting chosen for this example). Cosine similarity measures closeness in the embedding space; it is not
the probability that a passage answers the question. The passages are the Silberzahn et al. soccer-study
codebook, split into its four paragraphs and its 27 variable descriptions (one passage each). Embeddings
come from the open Sentence-Transformers model all-MiniLM-L6-v2 (mean pooling, unit length), so cosine
similarity is a dot product. This is a local demonstration, separate from Purdue GenAI Studio, whose
Knowledge Bases use EmbeddingGemma 300M and their own chunking, so its scores would differ.

Usage: python3 -I codebook_retrieval.py MODEL_DIR [CODEBOOK]
MODEL_DIR holds the model at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41, safetensors weights only (one command):
    huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 config.json model.safetensors special_tokens_map.json
    tokenizer.json tokenizer_config.json vocab.txt --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --local-dir MODEL_DIR
CODEBOOK defaults to the saved copy beside this script; the same file (an RTF document named codebook.txt) is
pinned at the Bertran, Fogliato and Wu repository, commit d489f03 (CODEBOOK_URL below). Both are checked by
SHA-256. Results are written to soccer_codebook_retrieval.json beside this script. Run on October 21, 2026
with Python 3.10.14, torch 2.3.0, transformers 4.29.2, tokenizers 0.13.3, and numpy 1.26.4 (transformers
4.29 needs numpy < 2); the scores matched the October 13 run.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
CODEBOOK = HERE / 'silberzahn2018_soccer_codebook.rtf'
CODEBOOK_URL = 'https://raw.githubusercontent.com/amazon-science/agentic-forking-path/d489f03b143fcaa9befb8cc443594ff63ec61eaa/data/soccer/codebook.txt'
CODEBOOK_SHA256 = '926e289e2f563e61b27464725a40d1932fda76099fd699a33cb2d75875af4ca6'
MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
WEIGHTS_SHA256 = '53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db'
QUESTION = "How did they measure players' skin tone?"
# A counting question: retrieval returns the closest definitions, not a count.
COUNT_QUESTION = 'How many red cards were given in total?'
TOP_K = 3


def rtf_text(raw):
    # Header groups here hold no nested braces.
    s = re.sub(r'\{\\(fonttbl|colortbl)[^{}]*\}', '', raw)
    s = s.replace('\\\n', '\n')
    s = re.sub(r'HYPERLINK\s+"[^"]*"', '', s)  # keep a link's visible text, drop its field instruction
    s = s.replace('\\*', '')  # RTF 'ignorable destination' marker
    s = re.sub(r"\\'([0-9a-f]{2})", lambda m: bytes.fromhex(m.group(1)).decode('cp1252'), s)
    s = re.sub(r'\\[a-zA-Z]+-?\d* ?', '', s)
    return s.replace('{', '').replace('}', '')


def passages(codebook):
    lines = [re.sub(r'\s+', ' ', x).strip() for x in rtf_text(codebook.read_text(encoding='utf-8')).split('\n')]
    lines = [x for x in lines if x and x not in (';', ';;')]
    start = lines.index('Variables:')
    # Paragraphs before the variable list; the third paragraph's Project Implicit link wraps onto its own line.
    paragraphs, current = [], ''
    for line in lines[:start]:
        if line.startswith('http') or line.startswith('(http'):
            current += ' ' + line
            continue
        if current:
            paragraphs.append(current)
        current = line
    paragraphs.append(current)
    out = [dict(id=f'paragraph{i + 1}', kind='paragraph', text=p) for i, p in enumerate(paragraphs)]
    body = lines[start + 1:]
    assert len(body) % 2 == 0, body
    for name, desc in zip(body[::2], body[1::2]):
        assert desc.startswith('- '), (name, desc)
        out.append(dict(id=name, kind='variable', text=f'{name}: {desc[2:]}'))
    return out


def embed(texts, tokenizer, model):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
    with torch.no_grad():
        tokens = model(**batch).last_hidden_state
    mask = batch['attention_mask'].unsqueeze(-1).float()
    pooled = (tokens * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


def main(model_dir, codebook=CODEBOOK):
    model_dir, codebook = Path(model_dir), Path(codebook)
    assert hashlib.sha256(codebook.read_bytes()).hexdigest() == CODEBOOK_SHA256, f'Unexpected codebook: {codebook}'
    weights = model_dir / 'model.safetensors'
    assert weights.exists() and not (model_dir / 'pytorch_model.bin').exists(), 'Use the safetensors weights only'
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == WEIGHTS_SHA256, f'Weights are not revision {MODEL_REVISION}'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).eval()
    items = passages(codebook)
    assert sum(p['kind'] == 'paragraph' for p in items) == 4 and sum(p['kind'] == 'variable' for p in items) == 27, len(items)
    q = embed([QUESTION], tokenizer, model)
    e = embed([p['text'] for p in items], tokenizer, model)
    scores = (e @ q.T).squeeze(1).tolist()
    for p, sc in zip(items, scores):
        p['similarity'] = round(sc, 3)
    ranked = sorted(items, key=lambda p: -p['similarity'])
    for i, p in enumerate(ranked):
        p['rank'] = i + 1
        p['selected'] = i < TOP_K  # selected as candidate context
    # The figure quotes the top passage; fail loudly if the ranking ever changes.
    assert ranked[0]['id'] == 'paragraph2' and 'skin tone was coded' in ranked[0]['text'], ranked[0]
    count_scores = (e @ embed([COUNT_QUESTION], tokenizer, model).T).squeeze(1).tolist()
    count_top = sorted(({'id': p['id'], 'text': p['text'], 'similarity': round(sc, 3)} for p, sc in zip(items, count_scores)), key=lambda p: -p['similarity'])[:TOP_K]
    assert count_top[0]['id'] == 'redCards', count_top[0]
    record = dict(
        question=QUESTION, top_k=TOP_K, passages_compared=len(items),
        model='sentence-transformers/all-MiniLM-L6-v2',
        model_revision=MODEL_REVISION, model_safetensors_sha256=WEIGHTS_SHA256,
        codebook='Soccer-study codebook (Silberzahn et al.), ' + CODEBOOK_URL, codebook_sha256=CODEBOOK_SHA256,
        method='Mean-pooled token embeddings scaled to unit length; cosine similarity = dot product with the question embedding. '
               'The top_k passages are selected as candidate context; similarity is not the probability that a passage answers the question.',
        environment=dict(torch=torch.__version__, transformers=transformers.__version__),
        ranked=ranked, count_question=COUNT_QUESTION, count_top=count_top)
    (HERE / 'soccer_codebook_retrieval.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    for p in ranked[:8]:
        print(p['rank'], p['similarity'], p['id'], '|', p['text'][:90])
    print('...', ranked[-1]['rank'], ranked[-1]['similarity'], ranked[-1]['id'])
    print(COUNT_QUESTION, [(p['id'], p['similarity']) for p in count_top])


if __name__ == '__main__':
    main(*sys.argv[1:3])
