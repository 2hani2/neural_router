# Neural Router v4

4-tier QA routing system:
- **Tier 1** — Math Engine (symbolic eval, <1ms)
- **Tier 2** — NanoQA 123M (custom SLM, ~500ms)
- **Tier 3** — Gemma 2 9B via Groq (~800ms, free)
- **Tier 4** — Llama 3.3 70B via Groq (~1.5s, free)

---

## Setup (one time)

```bash
# 1. Clone / unzip the project
cd neural_router

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Put your model file in the model/ folder
mkdir -p model
# Copy nanoqa_v4_best.pt → model/nanoqa_v4_best.pt

# 5. Configure environment
cp .env.example .env
# Open .env and fill in:
#   GROQ_API_KEY=gsk_...   (from console.groq.com)
```

## Run

```bash
python run.py
```

That's it. The terminal will print:
```
🌐 Public HTTPS URL: https://xxxx.ngrok.io
📡 Local:           http://localhost:8000
```

Open the HTTPS URL in any browser.

---

## API

### POST /query
```json
{ "question": "Who wrote Romeo and Juliet?" }
```
Response:
```json
{
  "tier": 2,
  "tier_name": "NanoQA SLM",
  "answer": "William Shakespeare.",
  "confidence": 0.898,
  "latency_ms": 523.4,
  "model": "NanoQA v4 (123M)"
}
```

### GET /calc?expr=sqrt(144)
```json
{ "expression": "sqrt(144)", "result": "12", "latency_ms": 0.02 }
```

### GET /health
```json
{ "status": "ok", "model": "NanoQA v4" }
```

---

## Project structure

```
neural_router/
├── app/
│   ├── main.py       # FastAPI app
│   ├── nanoqa.py     # Model architecture + inference
│   └── router.py     # 4-tier routing logic
├── model/
│   └── nanoqa_v4_best.pt   ← put your model here
├── static/
│   └── index.html    # Frontend
├── .env.example
├── requirements.txt
└── run.py            # Start everything
```
