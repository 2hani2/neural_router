# app/router.py — 3-tier Neural Router
# Tier 1: Math Engine   (<1ms)
# Tier 2: NanoQA SLM   (~500ms)
# Tier 3: Llama 3.3 70B (~1.5s via Groq)

import re, math, time, os
from groq import Groq
from app.nanoqa import slm_infer

_groq: Groq | None = None

def _get_groq() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq

CONF_ACCEPT = float(os.getenv("CONF_ACCEPT", "0.60"))
CONF_RETRY  = float(os.getenv("CONF_RETRY",  "0.45"))
GROQ_MODEL  = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Tier 1: Math Engine ───────────────────────────────────────────────────────
SAFE_MATH = {
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
    "tan": math.tan,   "log": math.log, "exp": math.exp,
    "abs": abs,        "pi": math.pi,   "e":  math.e,
    "factorial": math.factorial, "pow": pow, "round": round,
}

MATH_RE = re.compile(
    r"^[\s\d\+\-\*\/\(\)\.\^\%xX×÷√πeE,]+"
    r"(?:sin|cos|tan|log|sqrt|exp|factorial|pi|abs)?[\s\d\+\-\*\/\(\)\.\^\%]*$"
)

def _normalise_expr(q: str) -> str:
    q = re.sub(
        r"^(what is|calculate|compute|evaluate|solve|find|what's)\s+",
        "", q.strip(), flags=re.IGNORECASE
    ).strip().rstrip("?. ")
    q = q.replace("^", "**").replace("×", "*").replace("÷", "/")
    q = re.sub(r"\bpi\b", "math.pi", q)
    return q

def _eval_expr(expr: str):
    return eval(expr, {"__builtins__": {}, "math": math}, SAFE_MATH)

def _try_math(query: str):
    q = _normalise_expr(query)
    if not MATH_RE.match(q):
        return None, 0
    try:
        t0     = time.perf_counter()
        result = _eval_expr(q)
        ms     = (time.perf_counter() - t0) * 1000
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return str(result), round(ms, 3)
    except Exception:
        return None, 0

def calc(expr: str):
    """Dedicated /calc endpoint — always evaluates, raises on failure."""
    q = _normalise_expr(expr)
    try:
        t0     = time.perf_counter()
        result = _eval_expr(q)
        ms     = (time.perf_counter() - t0) * 1000
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return str(result), round(ms, 3)
    except Exception as e:
        raise ValueError(f"Cannot evaluate '{expr}'") from e

# ── Tier 3: Llama 3.3 70B ────────────────────────────────────────────────────
def _llm(query: str):
    t0  = time.perf_counter()
    rsp = _get_groq().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a helpful, concise assistant. "
                "Answer factual questions directly. "
                "For complex questions, reason step by step. "
                "Knowledge cutoff: early 2024."
            )},
            {"role": "user", "content": query},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    ms = (time.perf_counter() - t0) * 1000
    return rsp.choices[0].message.content.strip(), round(ms, 1)

# ── Main router ───────────────────────────────────────────────────────────────
def route(query: str) -> dict:
    query = query.strip()
    if not query:
        return {"error": "Empty query"}

    # Tier 1 — Math Engine
    math_result, math_ms = _try_math(query)
    if math_result is not None:
        return {
            "tier": 1, "tier_name": "Math Engine",
            "answer": math_result, "confidence": 1.0,
            "latency_ms": math_ms, "model": "Symbolic Eval",
            "slm_conf": None,
        }

    # Tier 2 — NanoQA
    slm  = slm_infer(query)
    conf = slm["confidence"]

    if not slm["is_garbage"] and conf >= CONF_ACCEPT:
        return {
            "tier": 2, "tier_name": "NanoQA",
            "answer": slm["answer"], "confidence": conf,
            "latency_ms": slm["latency_ms"], "model": "NanoQA v4 (123M)",
            "slm_conf": None,
        }

    # Retry band
    if CONF_RETRY <= conf < CONF_ACCEPT and not slm["is_garbage"]:
        slm2 = slm_infer(query)
        if not slm2["is_garbage"] and slm2["confidence"] >= CONF_ACCEPT:
            return {
                "tier": 2, "tier_name": "NanoQA",
                "answer": slm2["answer"], "confidence": slm2["confidence"],
                "latency_ms": slm["latency_ms"] + slm2["latency_ms"],
                "model": "NanoQA v4 (123M)", "slm_conf": None,
            }

    # Tier 3 — Llama 3.3 70B
    answer, ms = _llm(query)
    return {
        "tier": 3, "tier_name": "Llama 3.3 70B",
        "answer": answer, "confidence": None,
        "latency_ms": ms, "model": "llama-3.3-70b-versatile",
        "slm_conf": round(conf, 4),
    }
