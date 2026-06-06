"""
evaluate.py — Run 500 questions against the live Neural Router
Usage: python evaluate.py

Requirements:
  pip install requests rouge-score datasets tqdm
  Neural Router must be running: python run.py
"""
import json, time, random, re, requests
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from rouge_score import rouge_scorer

random.seed(42)

API_URL    = "http://localhost:8000/query"
OUT_FILE   = "results.json"
N          = 500
scorer     = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

# ── Load questions from TriviaQA + SQuAD test sets ────────────────────────────
def load_questions():
    questions = []

    # TriviaQA — rc.nocontext test split
    print("Loading TriviaQA...")
    try:
        tqa = load_dataset('trivia_qa', 'rc.nocontext', split='validation[:600]')
        for item in tqa:
            q = item['question'].strip()
            a = item['answer'].get('value', '').strip()
            aliases = item['answer'].get('aliases', [])
            if a and len(a.split()) <= 10:
                questions.append({
                    'question': q,
                    'reference': a,
                    'aliases': aliases,
                    'source': 'TriviaQA'
                })
    except Exception as e:
        print(f"TriviaQA failed: {e}")

    # SQuAD — validation split, short answers only
    print("Loading SQuAD...")
    try:
        sq = load_dataset('squad', split='validation[:600]')
        for item in sq:
            q = item['question'].strip()
            answers = item['answers']['text']
            if answers:
                a = answers[0].strip()
                if 1 <= len(a.split()) <= 8:
                    questions.append({
                        'question': q,
                        'reference': a,
                        'aliases': list(set(answers)),
                        'source': 'SQuAD'
                    })
    except Exception as e:
        print(f"SQuAD failed: {e}")

    random.shuffle(questions)
    selected = questions[:N]
    print(f"\nLoaded {len(selected)} questions ({sum(1 for q in selected if q['source']=='TriviaQA')} TriviaQA, {sum(1 for q in selected if q['source']=='SQuAD')} SQuAD)\n")
    return selected


# ── Scoring helpers ────────────────────────────────────────────────────────────
def normalise(text):
    text = text.lower().strip()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def contains_match(pred, reference, aliases):
    pred_n = normalise(pred)
    for ans in [reference] + (aliases or []):
        if normalise(ans) in pred_n or pred_n in normalise(ans):
            return True
    return False

def f1_token(pred, reference):
    pred_toks = normalise(pred).split()
    ref_toks  = normalise(reference).split()
    common    = set(pred_toks) & set(ref_toks)
    if not common: return 0.0
    p = len(common) / len(pred_toks) if pred_toks else 0
    r = len(common) / len(ref_toks)  if ref_toks  else 0
    return 2 * p * r / (p + r) if (p + r) else 0.0

def rouge_scores(pred, reference):
    scores = scorer.score(reference, pred)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }

def exact_match(pred, reference, aliases):
    pred_n = normalise(pred)
    for ans in [reference] + (aliases or []):
        if pred_n == normalise(ans):
            return True
    return False


# ── Query the router ───────────────────────────────────────────────────────────
def query_router(question):
    try:
        t0  = time.perf_counter()
        res = requests.post(API_URL, json={'question': question}, timeout=120)
        ms  = (time.perf_counter() - t0) * 1000
        if res.ok:
            d = res.json()
            d['_wall_ms'] = round(ms, 1)
            return d
        return {'error': res.text, '_wall_ms': ms}
    except Exception as e:
        return {'error': str(e), '_wall_ms': 0}


# ── Main evaluation loop ───────────────────────────────────────────────────────
def evaluate():
    questions = load_questions()
    results   = []
    errors    = 0

    print(f"Querying Neural Router at {API_URL}")
    print("Make sure 'python run.py' is running in another terminal\n")

    for i, item in enumerate(tqdm(questions, desc="Evaluating")):
        q   = item['question']
        ref = item['reference']
        ali = item.get('aliases', [])

        response = query_router(q)

        if 'error' in response:
            errors += 1
            results.append({**item, 'error': response['error'], 'tier': -1})
            continue

        pred = response.get('answer', '')
        rs   = rouge_scores(pred, ref)

        result = {
            'question':       q,
            'reference':      ref,
            'aliases':        ali,
            'source':         item['source'],
            'prediction':     pred,
            'tier':           response.get('tier'),
            'tier_name':      response.get('tier_name'),
            'model':          response.get('model'),
            'confidence':     response.get('confidence'),
            'slm_conf':       response.get('slm_conf'),
            'latency_ms':     response.get('latency_ms'),
            'wall_ms':        response.get('_wall_ms'),
            'contains_match': contains_match(pred, ref, ali),
            'exact_match':    exact_match(pred, ref, ali),
            'f1_token':       f1_token(pred, ref),
            'rouge1':         rs['rouge1'],
            'rougeL':         rs['rougeL'],
        }
        results.append(result)

        # print every 50
        if (i+1) % 50 == 0:
            done    = [r for r in results if 'contains_match' in r]
            acc     = sum(r['contains_match'] for r in done) / len(done) * 100
            t2_rate = sum(1 for r in done if r.get('tier') == 2) / len(done) * 100
            tqdm.write(f"  [{i+1}/{N}] Accuracy={acc:.1f}% | SLM rate={t2_rate:.1f}%")

    # Save
    summary = compute_summary(results)
    output  = {'summary': summary, 'results': results}
    Path(OUT_FILE).write_text(json.dumps(output, indent=2))

    print(f"\n{'='*55}")
    print(f"Results saved to {OUT_FILE}")
    print(f"{'='*55}")
    print_summary(summary, errors)
    return output


def compute_summary(results):
    valid = [r for r in results if 'contains_match' in r]
    if not valid: return {}

    by_tier = defaultdict(list)
    for r in valid:
        by_tier[r.get('tier', -1)].append(r)

    by_source = defaultdict(list)
    for r in valid:
        by_source[r.get('source','?')].append(r)

    def avg(lst, key): return sum(x[key] for x in lst) / len(lst) if lst else 0

    return {
        'n_total':        len(results),
        'n_valid':        len(valid),
        'accuracy':       avg(valid, 'contains_match'),
        'exact_match':    avg(valid, 'exact_match'),
        'f1_token':       avg(valid, 'f1_token'),
        'rouge1':         avg(valid, 'rouge1'),
        'rougeL':         avg(valid, 'rougeL'),
        'avg_latency_ms': avg(valid, 'latency_ms'),
        'tier_counts': {
            t: len(rs) for t, rs in by_tier.items()
        },
        'tier_accuracy': {
            t: avg(rs, 'contains_match') for t, rs in by_tier.items()
        },
        'tier_latency': {
            t: avg(rs, 'latency_ms') for t, rs in by_tier.items()
        },
        'source_accuracy': {
            s: avg(rs, 'contains_match') for s, rs in by_source.items()
        },
        'slm_confidence_bins': compute_conf_bins(by_tier.get(2, [])),
    }


def compute_conf_bins(slm_results):
    bins = [(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.01)]
    out  = []
    for lo, hi in bins:
        subset = [r for r in slm_results if r.get('confidence') and lo <= r['confidence'] < hi]
        if subset:
            out.append({
                'bin':      f'{lo:.1f}–{hi:.1f}',
                'n':        len(subset),
                'accuracy': sum(r['contains_match'] for r in subset) / len(subset),
                'mean_conf': sum(r['confidence'] for r in subset) / len(subset),
            })
    return out


def print_summary(s, errors):
    if not s: return
    print(f"\nOverall ({s['n_valid']} questions):")
    print(f"  Accuracy (Contains Match) : {s['accuracy']*100:.1f}%")
    print(f"  Exact Match               : {s['exact_match']*100:.1f}%")
    print(f"  F1 Token                  : {s['f1_token']*100:.1f}%")
    print(f"  ROUGE-1                   : {s['rouge1']*100:.1f}%")
    print(f"  ROUGE-L                   : {s['rougeL']*100:.1f}%")
    print(f"  Avg Latency               : {s['avg_latency_ms']:.0f}ms")
    print(f"\nRouting:")
    names = {1:'Math Engine', 2:'NanoQA', 3:'Llama 70B'}
    for t, count in sorted(s['tier_counts'].items()):
        acc = s['tier_accuracy'].get(t, 0)
        lat = s['tier_latency'].get(t, 0)
        print(f"  Tier {t} {names.get(t,'?'):<15}: {count:>3} queries | acc={acc*100:.1f}% | lat={lat:.0f}ms")
    if errors:
        print(f"\n  Errors: {errors}")


if __name__ == '__main__':
    evaluate()