
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
| ROUGE-1 (NanoQA-answered) | **81.3%** |
| NanoQA median latency | **238 ms** |
| Speedup vs always-LLM | **9.7×** |
| NanoQA params vs Llama 3.3 70B | **568× smaller** |

NanoQA outperforms GPT-2 Small (117M), TinyLlama (1.1B), and Phi-1.5 (1.3B) on the domain benchmark despite having fewer parameters — confirming that **data quality matters more than model size**.

---

## Project Structure
neural_router/

├── app/

│   ├── main.py              # FastAPI application

│   ├── nanoqa.py            # Model architecture + inference

│   └── router.py            # 3-tier routing logic

├── static/

│   └── index.html           # Frontend (dark UI, /calc, /info)

├── model/                   # ← place nanoqa_v4_best.pt here (see below)

├── evaluate.py              # 500-question held-out evaluation

├── evaluate_domain.py       # In-domain evaluation

├── handcrafted_qa.json      # Training domain QA pairs

├── plot.py                  # Generate paper figures

├── run.py                   # Start app + ngrok tunnel

├── requirements.txt

└── .env.example

---

## Setup

### Prerequisites
- **Python 3.11** (3.12+ not fully supported by tokenizers library)
- A free **Groq API key** — get one at [console.groq.com](https://console.groq.com)
- A free **ngrok token** (optional, for public HTTPS URL) — [dashboard.ngrok.com](https://dashboard.ngrok.com)

### Install Python 3.11 (if you don't have it)

**Mac:**
```bash
brew install pyenv
pyenv install 3.11.9
pyenv local 3.11.9
```

**Windows:** Download from [python.org/downloads](https://www.python.org/downloads/release/python-3119/)

**Linux:**
```bash
sudo apt install python3.11 python3.11-venv
```

### Clone and install

```bash
git clone https://github.com/2hani2/neural_router.git
cd neural_router

# Create virtual environment with Python 3.11
python3.11 -m venv venv

# Activate it
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download the model weights

The trained NanoQA model (~500MB) is not stored in this repo. Download it from:

**[Google Drive — nanoqa_v4_best.pt](https://drive.google.com/your-link-here)**

Then place it in the model folder:
```bash
mkdir -p model
# move the downloaded file here:
# model/nanoqa_v4_best.pt
```

### Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in:
GROQ_API_KEY=gsk_...          # from console.groq.com

NGROK_AUTH_TOKEN=...          # from dashboard.ngrok.com (optional)

MODEL_PATH=./model/nanoqa_v4_best.pt
### Run

```bash
python run.py
```

The app starts at **http://localhost:8000**. If you set `NGROK_AUTH_TOKEN`, a public HTTPS URL is also printed.

---

## API

```bash
# Ask a question
POST /query
{"question": "Who wrote Romeo and Juliet?"}

# Scientific calculator (bypasses routing)
GET /calc?expr=sqrt(144)+2^8

# Model info
GET /info

# Health check
GET /health
```

---

## Evaluation

```bash
# Run 500-question evaluation (router must be running in another terminal)
pip install rouge-score datasets tqdm
python evaluate.py

# Run in-domain evaluation (200 handcrafted questions)
python evaluate_domain.py

# Generate all graphs
python plot.py
```

---

## Training

The training notebook `nanoqa_v4_train.ipynb` runs on **Kaggle with a T4 GPU** (~3.5 hours).

Key design choices over previous versions:
- **No data leakage**: base pairs split before augmentation
- **Answer-only masking**: custom collator replaces DataCollatorForLanguageModeling
- **Focal Loss** (γ=2) + **KL Distillation** from GPT-2 (both restricted to answer tokens)
- `FOCAL_WEIGHT=0.92` prevents KL domination at random initialisation

---

## Tech Stack

`PyTorch` · `FastAPI` · `Groq API` · `HuggingFace Transformers` · `pyngrok`

---

## Authors

- Venisa Ivan Tellis — [venisatellis@gmail.com](mailto:venisatellis@gmail.com)

Manipal Institute of Technology, MAHE — 2026
READMEEOF
