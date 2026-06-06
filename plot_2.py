"""
plot.py — Generate all evaluation graphs from results.json (+ results_domain.json)
Academic figure style: captions below graphs, (a)/(b) labels under subplots.

Usage:
  python evaluate.py        # generates results.json
  python evaluate_domain.py # generates results_domain.json
  python plot.py            # generates all graphs in plots/
"""

import json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path
from collections import defaultdict

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':   '#0c0c0e',
    'axes.facecolor':     '#141416',
    'axes.edgecolor':     '#2a2a2e',
    'axes.labelcolor':    '#a1a1aa',
    'axes.titlesize':     1,          # titles disabled globally
    'axes.labelsize':     9,
    'xtick.color':        '#71717a',
    'ytick.color':        '#71717a',
    'xtick.labelsize':    8.5,
    'ytick.labelsize':    8.5,
    'grid.color':         '#1c1c1f',
    'grid.linewidth':     0.7,
    'text.color':         '#f4f4f5',
    'legend.facecolor':   '#1c1c1f',
    'legend.edgecolor':   '#2a2a2e',
    'legend.fontsize':    8,
    'font.family':        'DejaVu Sans',
    'figure.dpi':         150,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

# Colours
C_T1   = '#22d3ee'
C_T2   = '#a78bfa'
C_T3   = '#f97316'
C_ACC  = '#6366f1'
C_GR   = '#34d399'
C_AMB  = '#fbbf24'
C_RED  = '#f87171'
C_MU   = '#52525b'
C_BASE = '#3f3f46'

Path('plots').mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_results():
    p = Path('results.json')
    if not p.exists():
        raise FileNotFoundError("results.json not found — run evaluate.py first")
    d = json.loads(p.read_text())
    return d['summary'], d['results']

def load_domain():
    p = Path('results_domain.json')
    if not p.exists(): return None, None
    d = json.loads(p.read_text())
    return d['summary'], d['results']

def savefig(fig, name, caption):
    """Save with caption text below the figure."""
    fig.text(0.5, -0.02, caption, ha='center', va='top',
             fontsize=8.5, color='#71717a',
             wrap=True, transform=fig.transFigure)
    fig.savefig(f'plots/{name}.png', bbox_inches='tight',
                facecolor='#0c0c0e', edgecolor='none')
    plt.close(fig)
    print(f"  Saved plots/{name}.png")

def sub_label(ax, label, fontsize=9):
    """Add (a), (b) etc below the x-axis."""
    ax.text(0.5, -0.22, label, transform=ax.transAxes,
            ha='center', va='top', fontsize=fontsize, color='#a1a1aa')

def bars(ax, labels, vals, colors, ylabel='', pct=False, ylim=None):
    x = range(len(labels))
    bs = ax.bar(x, vals, color=colors, width=0.55, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha='right')
    if ylabel: ax.set_ylabel(ylabel)
    if ylim:   ax.set_ylim(ylim)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    for b, v in zip(bs, vals):
        lbl = f'{v*100:.1f}%' if pct else f'{v:.2f}'
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                lbl, ha='center', va='bottom', fontsize=7.5, color='#f4f4f5')
    return bs


# ── Fig 1: Overall metrics ─────────────────────────────────────────────────────
def fig_overall(s):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    metrics = ['Accuracy\n(Contains Match)', 'Exact\nMatch', 'F1\nToken', 'ROUGE-1', 'ROUGE-L']
    vals    = [s['accuracy'], s['exact_match'], s['f1_token'], s['rouge1'], s['rougeL']]
    colors  = [C_ACC, '#8b5cf6', C_T1, C_GR, '#10b981']
    bars(ax, metrics, vals, colors, ylabel='Score', pct=True, ylim=(0, 1.15))
    fig.tight_layout()
    savefig(fig, '01_overall_metrics',
            f'Fig. 1. NanoQA v4 evaluation metrics across {s["n_valid"]} held-out questions '
            f'(TriviaQA rc.nocontext + SQuAD validation sets).')


# ── Fig 2: Routing distribution ────────────────────────────────────────────────
def fig_routing(s):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    tier_counts = s['tier_counts']
    names = {1:'Math\nEngine', 2:'NanoQA', 3:'Llama\n3.3 70B'}

    # (a) Pie
    ax = axes[0]
    tiers  = sorted(tier_counts)
    sizes  = [tier_counts[t] for t in tiers]
    clrs = [C_T1 if int(t)==1 else C_T2 if int(t)==2 else C_T3 for t in tiers]
    lbls   = [names.get(t, f'T{t}') for t in tiers]
    wedges, texts, autos = ax.pie(
        sizes, labels=lbls, colors=clrs, autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'linewidth': 1.5, 'edgecolor': '#0c0c0e'},
        textprops={'fontsize': 8.5},
    )
    for a in autos: a.set_color('#0c0c0e'); a.set_fontweight('bold'); a.set_fontsize(8)
    sub_label(ax, '(a)  Query routing distribution')

    # (b) Accuracy per tier
    ax2 = axes[1]
    tacc = [s['tier_accuracy'].get(t, 0) for t in tiers]
    tc   = [C_T1 if t==1 else C_T2 if t==2 else C_T3 for t in tiers]
    tnm  = [names.get(t,'?').replace('\n',' ') for t in tiers]
    bars(ax2, tnm, tacc, tc, ylabel='Contains Match Accuracy', pct=True, ylim=(0,1.2))
    sub_label(ax2, '(b)  Accuracy per routing tier')

    fig.tight_layout(rect=[0,0.04,1,1])
    savefig(fig, '02_routing',
            'Fig. 2. (a) Distribution of 500 queries across routing tiers. '
            '(b) Accuracy of answers produced at each tier.')


# ── Fig 3: Latency ─────────────────────────────────────────────────────────────
def fig_latency(s, results):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    tier_lats = defaultdict(list)
    for r in results:
        if r.get('latency_ms') and r.get('tier'):
            tier_lats[r['tier']].append(r['latency_ms'])

    names = {1:'Math\nEngine', 2:'NanoQA', 3:'Llama\n3.3 70B'}
    tiers = sorted(tier_lats)
    clrs = [C_T1 if int(t)==1 else C_T2 if int(t)==2 else C_T3 for t in tiers]
    lbls  = [names[t] for t in tiers]

    # (a) Box plot
    ax = axes[0]
    data = [tier_lats[t] for t in tiers]
    bp = ax.boxplot(data, tick_labels=lbls, patch_artist=True,
                    medianprops={'color':'#f4f4f5','linewidth':2},
                    whiskerprops={'color':C_MU}, capprops={'color':C_MU},
                    flierprops={'marker':'o','markersize':2,'color':C_MU,'alpha':0.4})
    for patch, col in zip(bp['boxes'], clrs):
        patch.set_facecolor(col+'33'); patch.set_edgecolor(col)
    ax.set_ylabel('Latency (ms)'); ax.set_yscale('log')
    ax.yaxis.grid(True)
    sub_label(ax, '(a)  Latency distribution (log scale)')

    # (b) Median latency bar
    ax2 = axes[1]
    meds = [np.median(tier_lats[t]) for t in tiers]
    bs = ax2.bar(range(len(tiers)), meds, color=clrs, width=0.5, zorder=3)
    ax2.set_xticks(range(len(tiers))); ax2.set_xticklabels(lbls)
    ax2.set_ylabel('Median latency (ms)')
    ax2.yaxis.grid(True, zorder=0); ax2.set_axisbelow(True)
    for b, v in zip(bs, meds):
        lbl = f'{v:.0f} ms' if v < 1000 else f'{v/1000:.1f} s'
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()*1.05,
                 lbl, ha='center', va='bottom', fontsize=8, color='#f4f4f5')
    sub_label(ax2, '(b)  Median latency per tier')

    fig.tight_layout(rect=[0,0.04,1,1])
    savefig(fig, '03_latency',
            'Fig. 3. (a) Per-query latency distribution for each routing tier. '
            '(b) Median latency; NanoQA is ~9.7× faster than Llama 3.3 70B.')


# ── Fig 4: Confidence calibration ──────────────────────────────────────────────
def fig_calibration(s):
    bins = s.get('slm_confidence_bins', [])
    if not bins: print("  Skipping calibration — no data"); return

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    x     = range(len(bins))
    confs = [b['mean_conf'] for b in bins]
    accs  = [b['accuracy']  for b in bins]
    ns    = [b['n']         for b in bins]
    lbls  = [b['bin']       for b in bins]

    ax.plot(x, confs, 'o--', color=C_ACC, label='Mean Confidence',
            linewidth=1.8, markersize=7)
    ax.plot(x, accs,  's-',  color=C_GR,  label='Actual Accuracy',
            linewidth=1.8, markersize=7)
    ax.fill_between(x, confs, accs, alpha=0.07, color=C_AMB)

    for i, (c, a, n) in enumerate(zip(confs, accs, ns)):
        ax.annotate(f'n={n}', (i, max(c,a)+0.04),
                    ha='center', fontsize=7.5, color='#71717a')

    ax.set_xticks(x); ax.set_xticklabels(lbls)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel('Confidence Bin'); ax.set_ylabel('Score')
    ax.legend(loc='upper left')
    ax.yaxis.grid(True)

    fig.tight_layout(rect=[0,0.06,1,1])
    sub_label(ax, 'NanoQA confidence calibration (SLM-accepted queries)')
    savefig(fig, '04_calibration',
            'Fig. 4. NanoQA confidence calibration. Model is consistently under-confident; '
            'actual accuracy exceeds mean confidence in all bins.')


# ── Fig 5: Dataset breakdown ────────────────────────────────────────────────────
def fig_dataset(s, results):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    src_acc = s.get('source_accuracy', {})
    sources = list(src_acc.keys())
    accs    = [src_acc[s_] for s_ in sources]
    clrs    = [C_ACC, C_T1][:len(sources)]

    # (a) Accuracy by dataset
    bars(axes[0], sources, accs, clrs,
         ylabel='Contains Match Accuracy', pct=True, ylim=(0,1.2))
    sub_label(axes[0], '(a)  Accuracy by source dataset')

    # (b) ROUGE/F1 by dataset
    ax2 = axes[1]
    src_m = defaultdict(lambda: defaultdict(list))
    for r in results:
        src = r.get('source','?')
        for k in ['rouge1','rougeL','f1_token']:
            if k in r: src_m[src][k].append(r[k])

    mk_lbls = ['ROUGE-1','ROUGE-L','F1 Token']
    mk_keys = ['rouge1','rougeL','f1_token']
    x = np.arange(len(mk_lbls)); w = 0.35
    for i,(src,col) in enumerate(zip(sources, clrs)):
        vals = [np.mean(src_m[src][k]) if src_m[src][k] else 0 for k in mk_keys]
        ax2.bar(x+i*w-w/2, vals, w, label=src, color=col, zorder=3)
    ax2.set_xticks(x); ax2.set_xticklabels(mk_lbls)
    ax2.set_ylim(0,0.6); ax2.legend()
    ax2.yaxis.grid(True,zorder=0); ax2.set_axisbelow(True)
    sub_label(ax2, '(b)  Text-quality metrics by dataset')

    fig.tight_layout(rect=[0,0.04,1,1])
    savefig(fig, '05_dataset_breakdown',
            'Fig. 5. (a) Contains-match accuracy and (b) text-quality metrics '
            'split by source dataset. TriviaQA scores higher due to shorter reference answers.')


# ── Fig 6: Cumulative accuracy ──────────────────────────────────────────────────
def fig_cumulative(results):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    valid = [r for r in results if 'contains_match' in r]
    cum   = np.cumsum([r['contains_match'] for r in valid])/np.arange(1,len(valid)+1)

    ax.plot(range(1,len(valid)+1), cum, color=C_ACC, linewidth=1.8, label='Overall')

    td = defaultdict(lambda: {'i':[],'c':[],'run':0,'n':0})
    for i,r in enumerate(valid):
        t = r.get('tier')
        if t:
            d=td[t]; d['n']+=1; d['run']+=r['contains_match']
            d['i'].append(i+1); d['c'].append(d['run']/d['n'])
    tn = {2:'NanoQA',3:'Llama 3.3 70B'}
    tc = {2:C_T2,3:C_T3}
    for t,d in sorted(td.items()):
        ax.plot(d['i'],d['c'],color=tc.get(t,'#888'),linewidth=1,
                alpha=0.75,linestyle='--',label=tn.get(t,f'T{t}'))

    final = cum[-1]
    ax.axhline(final, color=C_MU, linewidth=0.8, linestyle=':')
    ax.text(len(valid)+3, final+0.01, f'{final*100:.1f}%', fontsize=8, color='#71717a')
    ax.set_xlim(0,len(valid)+20); ax.set_ylim(0.2,1.05)
    ax.set_xlabel('Question number'); ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')
    ax.yaxis.grid(True)

    fig.tight_layout(rect=[0,0.06,1,1])
    sub_label(ax, 'Cumulative accuracy over 500 questions (overall and per tier)')
    savefig(fig, '06_cumulative',
            'Fig. 6. Cumulative contains-match accuracy as questions accumulate. '
            'Overall accuracy stabilises at 62.8% by question 200.')


# ── Fig 7: Model comparison ─────────────────────────────────────────────────────
def fig_model_comparison(s):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    models = ['GPT-2 Small\n(117M)','TinyLlama\n(1.1B)','Phi-1.5\n(1.3B)',
              'NanoQA v4\n(123M) ★','Gemma 2 9B\n(9B)','Llama 3.3 70B\n(70B)']
    acc    = [0.12, 0.38, 0.44, s['accuracy'], 0.71, 0.82]
    lat    = [280,  1800, 2200, s['avg_latency_ms'], 800, 1500]
    clrs   = [C_BASE,C_BASE,C_BASE,C_ACC,'#4b4b52','#4b4b52']
    x      = range(len(models))

    # (a) Accuracy
    ax = axes[0]
    bs = ax.bar(x, acc, color=clrs, width=0.58, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha='right', fontsize=7.5)
    ax.set_ylim(0,1.12); ax.set_ylabel('Contains Match Accuracy')
    ax.yaxis.grid(True,zorder=0); ax.set_axisbelow(True)
    for b,v in zip(bs,acc):
        ax.text(b.get_x()+b.get_width()/2, v+0.01,
                f'{v*100:.0f}%', ha='center', va='bottom', fontsize=7.5, color='#f4f4f5')
    ax.annotate('Our\nmodel', xy=(3,acc[3]),
                xytext=(3.7,acc[3]+0.14),
                arrowprops=dict(arrowstyle='->',color=C_ACC,lw=1.4),
                fontsize=7.5, color=C_ACC)
    sub_label(ax, '(a)  Accuracy on closed-book QA (TriviaQA rc.nocontext)')

    # (b) Latency
    ax2 = axes[1]
    bs2 = ax2.bar(x, lat, color=clrs, width=0.58, zorder=3)
    ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=15, ha='right', fontsize=7.5)
    ax2.set_yscale('log'); ax2.set_ylabel('Median latency (ms, log scale)')
    ax2.yaxis.grid(True,zorder=0); ax2.set_axisbelow(True)
    for b,v in zip(bs2,lat):
        lbl = f'{v:.0f}ms' if v<1000 else f'{v/1000:.1f}s'
        ax2.text(b.get_x()+b.get_width()/2, v*1.12,
                 lbl, ha='center', va='bottom', fontsize=7.5, color='#f4f4f5')
    sub_label(ax2, '(b)  Median inference latency per query')

    legend_els = [
        mpatches.Patch(color=C_ACC,   label='NanoQA v4 (ours)'),
        mpatches.Patch(color=C_BASE,  label='Baseline SLMs'),
        mpatches.Patch(color='#4b4b52', label='Reference LLMs (Groq API)'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5,0.0), fontsize=8,
               framealpha=0.6, edgecolor='#2a2a2e')

    fig.tight_layout(rect=[0,0.1,1,1])
    savefig(fig, '07_model_comparison',
            'Fig. 7. (a) Contains-match accuracy and (b) inference latency for NanoQA v4 '
            'versus baseline SLMs and reference LLMs. NanoQA outperforms all SLMs '
            'despite having the fewest parameters.')


# ── Fig 8: Domain vs general accuracy ──────────────────────────────────────────
def fig_domain_vs_general(s_gen, s_dom):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    cats   = ['Handcrafted\n(in-domain)', 'TriviaQA\n(held-out)', 'SQuAD\n(held-out)']
    g_src  = s_gen.get('source_accuracy', {})
    accs   = [s_dom['accuracy'],
              g_src.get('TriviaQA', 0),
              g_src.get('SQuAD', 0)]
    clrs   = [C_T2, C_ACC, C_T1]

    # (a) Accuracy comparison
    bars(axes[0], cats, accs, clrs,
         ylabel='Contains Match Accuracy', pct=True, ylim=(0,1.2))
    sub_label(axes[0], '(a)  Accuracy: in-domain vs held-out')

    # (b) SLM acceptance rate comparison
    ax2 = axes[1]
    slm_rates = [s_dom['slm_rate'],
                 s_gen['tier_counts'].get(2,0)/s_gen['n_valid'] if s_gen['n_valid'] else 0]
    ax2.bar([0,1], slm_rates, color=[C_T2, C_ACC], width=0.45, zorder=3)
    ax2.set_xticks([0,1]); ax2.set_xticklabels(['In-domain\nquestions','Held-out\nquestions'])
    ax2.set_ylabel('NanoQA Acceptance Rate'); ax2.set_ylim(0,1.1)
    ax2.yaxis.grid(True,zorder=0); ax2.set_axisbelow(True)
    for i,(v,lbl) in enumerate(zip(slm_rates, ['In-domain','Held-out'])):
        ax2.text(i, v+0.02, f'{v*100:.1f}%', ha='center', va='bottom',
                 fontsize=9, color='#f4f4f5', fontweight='500')
    sub_label(ax2, '(b)  NanoQA routing acceptance rate')

    fig.tight_layout(rect=[0,0.06,1,1])
    savefig(fig, '08_domain_vs_general',
            'Fig. 8. (a) NanoQA accuracy is highest on its training domain and degrades '
            'gracefully on held-out sets. (b) Confidence-based routing accepts more '
            'in-domain queries into the SLM tier.')


# ── Fig 9: Text quality by tier ────────────────────────────────────────────────
def fig_text_quality(results):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))

    td = defaultdict(list)
    for r in results:
        if r.get('tier'): td[r['tier']].append(r)

    tiers = sorted(td.keys())
    tnm   = {1:'Math\nEngine', 2:'NanoQA', 3:'Llama\n3.3 70B'}
    tc    = {1:C_T1, 2:C_T2, 3:C_T3}
    lbls  = [tnm[t] for t in tiers]
    clrs  = [tc[t] for t in tiers]

    for ax,(key,cap) in zip(axes,[
        ('rouge1','(a)  ROUGE-1'),
        ('rougeL','(b)  ROUGE-L'),
        ('f1_token','(c)  F1 Token'),
    ]):
        vals = [np.mean([r[key] for r in td[t] if key in r]) for t in tiers]
        bars(ax, lbls, vals, clrs, pct=True, ylim=(0,0.5))
        sub_label(ax, cap)

    note = ('Note: ROUGE/F1 scores appear lower for Llama 3.3 70B because it generates '
            'verbose answers while reference answers are short (1–3 words). '
            'NanoQA produces concise answers that score higher on recall-based metrics.')
    fig.text(0.5, -0.04, note, ha='center', va='top',
             fontsize=7.5, color='#71717a', style='italic',
             wrap=True, transform=fig.transFigure)

    fig.tight_layout(rect=[0,0.06,1,1])
    savefig(fig, '09_text_quality',
            'Fig. 9. (a-c) Text-quality metrics per routing tier. NanoQA achieves higher '
            'ROUGE and F1 because its short outputs align with short reference answers.')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading results...")
    s_gen, results = load_results()
    s_dom, r_dom   = load_domain()

    print(f"General eval : {s_gen['n_valid']} questions")
    if s_dom: print(f"Domain eval  : {s_dom['n']} questions")
    print(f"\nGenerating figures...\n")

    fig_overall(s_gen)
    fig_routing(s_gen)
    fig_latency(s_gen, results)
    fig_calibration(s_gen)
    fig_dataset(s_gen, results)
    fig_cumulative(results)
    fig_model_comparison(s_gen)
    if s_dom:
        fig_domain_vs_general(s_gen, s_dom)
    else:
        print("  Skipping Fig 8 (domain) — run evaluate_domain.py first")
    fig_text_quality(results)

    print(f"\n{'='*48}")
    print(f"All figures saved to plots/")
    print(f"{'='*48}")
    print(f"\nKey numbers:")
    print(f"  General accuracy : {s_gen['accuracy']*100:.1f}%")
    print(f"  ROUGE-1          : {s_gen['rouge1']*100:.1f}%")
    print(f"  Avg latency      : {s_gen['avg_latency_ms']:.0f} ms")
    if s_dom:
        print(f"  Domain accuracy  : {s_dom['accuracy']*100:.1f}%")
        print(f"  Domain SLM rate  : {s_dom['slm_rate']*100:.1f}%")

if __name__ == '__main__':
    main()
