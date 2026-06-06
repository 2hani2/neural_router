"""
evaluate_domain.py — Test Neural Router on 200 handcrafted domain questions
These are FROM the training distribution — shows peak NanoQA performance.

Usage: python evaluate_domain.py
       (run alongside python run.py in another terminal)
"""

import json, time, random, re, requests
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer

random.seed(99)

API_URL   = "http://localhost:8000/query"
HC_FILE   = "model/handcrafted_qa.json"   # adjust path if needed
OUT_FILE  = "results_domain.json"
N         = 200
scorer    = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def normalise(text):
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def contains_match(pred, ref):
    return normalise(ref) in normalise(pred) or normalise(pred) in normalise(ref)

def f1_token(pred, ref):
    p_toks = normalise(pred).split()
    r_toks = normalise(ref).split()
    common = set(p_toks) & set(r_toks)
    if not common: return 0.0
    p = len(common)/len(p_toks) if p_toks else 0
    r = len(common)/len(r_toks) if r_toks else 0
    return 2*p*r/(p+r) if (p+r) else 0.0

def rouge(pred, ref):
    s = scorer.score(ref, pred)
    return {'rouge1': s['rouge1'].fmeasure, 'rougeL': s['rougeL'].fmeasure}

def query(question):
    try:
        r = requests.post(API_URL, json={'question': question}, timeout=60)
        if r.ok: return r.json()
        return {'error': r.text}
    except Exception as e:
        return {'error': str(e)}

def evaluate():
    # Load handcrafted pairs — check multiple possible paths
    paths = [HC_FILE, '/mnt/user-data/uploads/handcrafted_qa.json',
             'handcrafted_qa.json', '../handcrafted_qa.json']
    data = None
    for p in paths:
        if Path(p).exists():
            data = json.loads(Path(p).read_text())
            print(f"Loaded {len(data)} pairs from {p}")
            break
    if data is None:
        raise FileNotFoundError("handcrafted_qa.json not found — put it in the neural_router folder")

    sample = random.sample(data, min(N, len(data)))
    results, errors = [], 0

    print(f"Running {len(sample)} domain questions against {API_URL}\n")

    for item in tqdm(sample, desc="Domain eval"):
        q   = item['question'].strip()
        ref = item['answer'].strip()
        res = query(q)

        if 'error' in res:
            errors += 1
            results.append({**item, 'error': res['error'], 'tier': -1})
            continue

        pred = res.get('answer', '')
        rs   = rouge(pred, ref)
        results.append({
            'question':       q,
            'reference':      ref,
            'prediction':     pred,
            'tier':           res.get('tier'),
            'tier_name':      res.get('tier_name'),
            'model':          res.get('model'),
            'confidence':     res.get('confidence'),
            'latency_ms':     res.get('latency_ms'),
            'contains_match': contains_match(pred, ref),
            'f1_token':       f1_token(pred, ref),
            'rouge1':         rs['rouge1'],
            'rougeL':         rs['rougeL'],
            'source':         'Handcrafted',
        })

    valid = [r for r in results if 'contains_match' in r]
    def avg(k): return sum(r[k] for r in valid)/len(valid) if valid else 0

    t2 = [r for r in valid if r.get('tier')==2]
    t3 = [r for r in valid if r.get('tier')==3]

    summary = {
        'n': len(valid),
        'accuracy':    avg('contains_match'),
        'f1_token':    avg('f1_token'),
        'rouge1':      avg('rouge1'),
        'rougeL':      avg('rougeL'),
        'slm_rate':    len(t2)/len(valid) if valid else 0,
        'slm_accuracy': sum(r['contains_match'] for r in t2)/len(t2) if t2 else 0,
        'llm_accuracy': sum(r['contains_match'] for r in t3)/len(t3) if t3 else 0,
        'avg_latency':  avg('latency_ms'),
    }

    Path(OUT_FILE).write_text(json.dumps({'summary': summary, 'results': results}, indent=2))

    print(f"\n{'='*50}")
    print(f"Domain Evaluation ({len(valid)} questions)")
    print(f"{'='*50}")
    print(f"  Accuracy   : {summary['accuracy']*100:.1f}%")
    print(f"  F1 Token   : {summary['f1_token']*100:.1f}%")
    print(f"  ROUGE-1    : {summary['rouge1']*100:.1f}%")
    print(f"  SLM rate   : {summary['slm_rate']*100:.1f}%")
    print(f"  SLM acc    : {summary['slm_accuracy']*100:.1f}%")
    print(f"  LLM acc    : {summary['llm_accuracy']*100:.1f}%")
    print(f"  Saved to   : {OUT_FILE}")

if __name__ == '__main__':
    evaluate()
