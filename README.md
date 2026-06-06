# Neural Router v4

> A confidence-gated SLM–LLM cascade for efficient question answering.

Built at **Manipal Institute of Technology** as part of the Computational Intelligence Lab project.

---

## What it does

Every incoming query is automatically routed to the most appropriate processing tier:

| Tier | Model | Latency | Handles |
|------|-------|---------|---------|
| T1 | Math Engine (symbolic eval) | < 1 ms | Arithmetic, trigonometry |
| T2 | **NanoQA** (123.7M, trained from scratch) | ~238 ms | Short factual QA |
| T3 | Llama 3.3 70B (Groq API) | ~2.3 s | Complex reasoning |

NanoQA is a custom causal decoder-only transformer trained on 883,166 QA pairs using **answer-only loss masking**, **Focal Loss** (γ=2), and **cross-architecture knowledge distillation** from frozen GPT-2. Routing uses the mean raw softmax probability across generated tokens as a confidence signal — no separate trained router, no labelled preference data.

---

## Results

| Metric | Score |
|--------|-------|
| In-domain accuracy (handcrafted QA) | **99.0%** |
| TriviaQA rc.nocontext (held-out) | **83.1%** |
| SQuAD validation (held-out) | **43.8%** |
| NanoQA median latency | **238 ms** |
| Speedup vs always-LLM | **9.7×** |
| NanoQA params vs Llama 3.3 70B | **568× smaller** |

NanoQA outperforms GPT-2 Small (117M), TinyLlama (1.1B), and Phi-1.5 (1.3B) on the domain benchmark despite having fewer parameters — confirming that **data quality matters more than model size**.

---

## Project Structure

```
neural_router/
├── app/
│   ├── main.py          # FastAPI application
│   ├── nanoqa.py        # Model architecture + inference
│   └── router.py        # 3-tier routing logic
├── static/
│   └── index.html       # Frontend (dark UI, /calc, /info)
├── evaluate.py          # 500-question held-out evaluation
├── evaluate_domain.py   # In-domain evaluation (handcrafted QA)
├── plot.py              # Generate all paper figures
├── run.py               # Start app + ngrok tunnel
├── requirements.txt
└── .env.example
```

---

## Setup

```bash
# 1. Python 3.11 required (3.14 not supported by tokenizers)
python3.11 -m venv venv && source venv/bin/activate

# 2. Install
pip install -r requirements.txt

# 3. Add model weights
mkdir model
# Place nanoqa_v4_best.pt in model/

# 4. Configure
cp .env.example .env
# Fill in GROQ_API_KEY (free at console.groq.com)
# Fill in NGROK_AUTH_TOKEN (free at dashboard.ngrok.com)

# 5. Run
python run.py
```

The terminal will print a public HTTPS URL via ngrok. Open it in any browser.

---

## API

```bash
# Ask a question
POST /query
{"question": "Who wrote Romeo and Juliet?"}

# Calculator (bypasses routing)
GET /calc?expr=sqrt(144)+2^8

# Health check
GET /health
```

---

## Training

The training notebook is in `nanoqa_v4_train.ipynb` (Kaggle, T4 GPU, ~3.5 hours).

Key fixes over previous versions:
- **Data leakage fix**: base pairs are split before augmentation
- **Loss masking**: `DataCollatorForLanguageModeling` replaced with custom collator
- **KL scaling**: both focal and KL losses restricted to answer tokens only
- `FOCAL_WEIGHT = 0.92` to prevent KL domination at random initialisation

---

## Tech Stack

`PyTorch` · `FastAPI` · `Groq API` · `HuggingFace Transformers` · `pyngrok`

---

## Authors

- Venisa Ivan Tellis — [venisatellis@gmail.com](mailto:venisatellis@gmail.com)
- Pankhuri Kumari
- Upadrasta Shivani Sri Varshini
- G. Pradeep Reddy (supervisor)

Manipal Institute of Technology, MAHE — 2026
