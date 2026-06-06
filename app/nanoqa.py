# app/nanoqa.py — NanoQA v4 model definition + inference
# Exact same architecture as training notebook. Must match or weights won't load.

import math, os, time
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer

# ── Architecture (must match training Cell 3 exactly) ─────────────────────────

class NanoQAConfig:
    def __init__(self, **kwargs):
        self.vocab_size  = kwargs.get("vocab_size",  50257)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.num_layers  = kwargs.get("num_layers",  12)
        self.num_heads   = kwargs.get("num_heads",   12)
        self.ff_size     = kwargs.get("ff_size",     3072)
        self.max_seq_len = kwargs.get("max_seq_len", 128)
        self.dropout     = kwargs.get("dropout",     0.1)


class NanoQAAttention(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim  = cfg.hidden_size // cfg.num_heads
        self.hidden    = cfg.hidden_size
        self.qkv  = torch.nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.proj = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.drop = torch.nn.Dropout(cfg.dropout)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
            .view(1, 1, cfg.max_seq_len, cfg.max_seq_len)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        return self.proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class NanoQAFFN(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(cfg.hidden_size, cfg.ff_size),
            torch.nn.GELU(),
            torch.nn.Linear(cfg.ff_size, cfg.hidden_size),
            torch.nn.Dropout(cfg.dropout),
        )
    def forward(self, x): return self.net(x)


class NanoQABlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(cfg.hidden_size)
        self.ln2 = torch.nn.LayerNorm(cfg.hidden_size)
        self.att = NanoQAAttention(cfg)
        self.ffn = NanoQAFFN(cfg)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class NanoQA(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = torch.nn.Embedding(cfg.max_seq_len, cfg.hidden_size)
        self.drop    = torch.nn.Dropout(cfg.dropout)
        self.blocks  = torch.nn.ModuleList([NanoQABlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_f    = torch.nn.LayerNorm(cfg.hidden_size)
        self.lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

    def forward(self, x):
        B, T = x.shape
        pos    = torch.arange(T, device=x.device)
        hidden = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(self.ln_f(hidden))

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=30, temperature=0.3,
                 top_k=10, eos_token_id=None):
        self.eval()
        ids, gen, conf = prompt_ids.clone(), [], []
        for _ in range(max_new_tokens):
            ctx        = ids[:, -self.cfg.max_seq_len:]
            logits     = self(ctx)
            next_logits = logits[0, -1, :]
            raw_probs  = F.softmax(next_logits, dim=-1)          # confidence signal
            scaled     = next_logits / max(temperature, 1e-8)
            if top_k > 0:
                topk_vals, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
                scaled[scaled < topk_vals[-1]] = float("-inf")
            probs    = F.softmax(scaled, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).item()
            if eos_token_id is not None and next_tok == eos_token_id:
                break
            conf.append(raw_probs[next_tok].item())
            gen.append(next_tok)
            ids = torch.cat([ids, torch.tensor([[next_tok]], device=ids.device)], dim=1)
        return gen, conf


# ── Loader ────────────────────────────────────────────────────────────────────

ANSWER_PREFIX = "\nAnswer:"
_model     = None
_tokenizer = None
_device    = None


def load_model(model_path: str):
    global _model, _tokenizer, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NanoQA] Device: {_device}")

    print(f"[NanoQA] Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=_device)

    # Build config from saved dict (falls back to defaults if missing)
    saved_cfg = ckpt.get("config", {})
    cfg   = NanoQAConfig(**saved_cfg)
    model = NanoQA(cfg).to(_device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"[NanoQA] Loaded {params/1e6:.1f}M parameters | val_loss={ckpt.get('val_loss', '?'):.4f}")

    _tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    _tokenizer.pad_token = _tokenizer.eos_token

    _model = model
    return model


# ── Garbage filter ────────────────────────────────────────────────────────────

def _is_garbage(text: str) -> bool:
    words = text.strip().split()
    if len(words) < 1 or len(words) > 25:
        return True
    if len(set(w.lower() for w in words)) == 1 and len(words) > 2:
        return True                                           # "the the the"
    if len(words) >= 4 and len(set(w.lower() for w in words)) < 3:
        return True                                           # near-repetition
    if all(c in "0123456789 .,%-" for c in text.strip()):
        return True                                           # only digits/punct
    refuse = ["i don't know", "i do not know", "i cannot", "i'm not sure"]
    if any(r in text.lower() for r in refuse):
        return True
    return False


# ── Main inference function ───────────────────────────────────────────────────

def slm_infer(question: str, max_new_tokens: int = 30,
              temperature: float = 0.3, top_k: int = 10):
    """
    Run NanoQA inference on a question.
    Returns dict: {answer, confidence, is_garbage, latency_ms}
    """
    assert _model is not None, "Call load_model() first"

    # Normalise input (match training augmentation)
    q = question.strip().lower()
    q = q.replace("what's", "what is").replace("who's", "who is")
    q = q.replace("where's", "where is").replace("it's", "it is")

    prompt = f"Question: {q}{ANSWER_PREFIX}"
    ids    = _tokenizer.encode(prompt, return_tensors="pt").to(_device)

    t0 = time.perf_counter()
    gen_ids, confs = _model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=_tokenizer.eos_token_id,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    answer = _tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    conf   = sum(confs) / len(confs) if confs else 0.0

    return {
        "answer":     answer,
        "confidence": round(conf, 4),
        "is_garbage": _is_garbage(answer),
        "latency_ms": round(latency_ms, 1),
    }
