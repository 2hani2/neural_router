
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
