import os
import json
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer
from openai import OpenAI

from model.model_swen import SwenConfig, SwenForCausalLM

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


# -----------------------------
# Config
# -----------------------------

@dataclass
class DemoConfig:
    ckpt: str = os.getenv("SWEN_CKPT", "")
    tokenizer_path: str = os.getenv("SWEN_TOKENIZER", "")
    hidden_size: int = int(os.getenv("SWEN_HIDDEN_SIZE", "512"))
    num_layers: int = int(os.getenv("SWEN_NUM_LAYERS", "8"))
    heads: int = int(os.getenv("SWEN_HEADS", "8"))
    kv_heads: int = int(os.getenv("SWEN_KV_HEADS", "2"))
    intermediate_size: int = int(os.getenv("SWEN_INTERMEDIATE_SIZE", "1408"))
    rope_theta: float = float(os.getenv("SWEN_ROPE_THETA", "1000000.0"))
    rms_norm_eps: float = float(os.getenv("SWEN_RMS_NORM_EPS", "1e-5"))
    local_max_new_tokens: int = int(os.getenv("SWEN_LOCAL_MAX_NEW_TOKENS", "48"))
    local_temperature: float = float(os.getenv("SWEN_LOCAL_TEMPERATURE", "0.5"))
    local_top_p: float = float(os.getenv("SWEN_LOCAL_TOP_P", "0.9"))
    local_top_k: int = int(os.getenv("SWEN_LOCAL_TOP_K", "0"))
    prefix_words: int = int(os.getenv("SWEN_PREFIX_WORDS", "8"))
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    cloud_max_output_tokens: int = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "256"))
    cloud_system_prompt: str = os.getenv(
        "CLOUD_SYSTEM_PROMPT",
        "Continue the assistant reply naturally from the provided prefix. Do not repeat the prefix. Keep the tone coherent, directly answer the user, and produce a helpful continuation.",
    )


CONFIG = DemoConfig()
app = FastAPI(title="Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE: Dict[str, object] = {
    "device": None,
    "model": None,
    "tokenizer": None,
    "openai": None,
    "loaded": False,
}


# -----------------------------
# Helpers
# -----------------------------

def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
    logits = logits.clone()
    if top_k > 0:
        kth = torch.topk(logits, top_k).values[..., -1, None]
        logits[logits < kth] = -float("inf")
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = cumprobs > top_p
        mask[..., 0] = False
        sorted_logits[mask] = -float("inf")
        logits[:] = -float("inf")
        logits.scatter_(0, sorted_idx, sorted_logits)
    return logits


def count_words(text: str) -> int:
    # ASCII-ish count, but also handles CJK blocks as non-whitespace segments.
    return len([x for x in text.strip().split() if x])


def build_input_ids(tokenizer, history, device):
    txt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    return tokenizer(txt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)


def load_model_and_tokenizer():
    if not CONFIG.ckpt or not CONFIG.tokenizer_path:
        raise RuntimeError("SWEN_CKPT and SWEN_TOKENIZER must be set.")

    device = choose_device()
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.tokenizer_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = SwenConfig(
        vocab_size=len(tokenizer),
        hidden_size=CONFIG.hidden_size,
        num_hidden_layers=CONFIG.num_layers,
        num_attention_heads=CONFIG.heads,
        num_key_value_heads=CONFIG.kv_heads,
        intermediate_size=CONFIG.intermediate_size,
        rope_theta=CONFIG.rope_theta,
        rms_norm_eps=CONFIG.rms_norm_eps,
        flash_attn=False,
        bos_token_id=tokenizer.bos_token_id or 1,
        eos_token_id=tokenizer.eos_token_id or 2,
        max_position_embeddings=32768,
        inference_rope_scaling=False,
        dropout=0,
    )

    model = SwenForCausalLM(config).to(device).eval()
    raw = torch.load(CONFIG.ckpt, map_location="cpu")
    sd = raw["model"]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)

    try:
        model.lm_head.weight = model.model.embed_tokens.weight
    except Exception:
        pass

    with torch.no_grad():
        dummy = tokenizer("hello", return_tensors="pt")["input_ids"].to(device)
        _ = model(dummy)

    STATE["device"] = device
    STATE["model"] = model
    STATE["tokenizer"] = tokenizer
    api_key = os.getenv("OPENAI_API_KEY")
    STATE["openai"] = OpenAI(api_key=api_key) if api_key else None
    STATE["loaded"] = True


@app.on_event("startup")
def startup_event():
    load_model_and_tokenizer()


# -----------------------------
# Generation
# -----------------------------
@torch.no_grad()
def generate_local_prefix(user_text: str):
    model = STATE["model"]
    tokenizer = STATE["tokenizer"]
    device = STATE["device"]

    history = [{"role": "user", "content": user_text}]
    input_ids = build_input_ids(tokenizer, history, device)
    cur = input_ids
    eos_id = tokenizer.eos_token_id
    generated_ids: List[int] = []
    last_text = ""

    t0 = time.perf_counter()
    for step in range(CONFIG.local_max_new_tokens):
        out = model(cur)
        logits = out.logits[0, -1, :]
        if CONFIG.local_temperature > 0:
            logits = logits / CONFIG.local_temperature
        logits = top_k_top_p_filtering(logits, top_k=CONFIG.local_top_k, top_p=CONFIG.local_top_p)
        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, 1).item()

        generated_ids.append(token)
        cur = torch.cat([cur, torch.tensor([[token]], device=device)], dim=1)
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        last_text = decoded

        if count_words(decoded) >= CONFIG.prefix_words:
            break
        if token == eos_id:
            break
        if step == 0:
            first_token_ms = (time.perf_counter() - t0) * 1000
    else:
        first_token_ms = (time.perf_counter() - t0) * 1000

    if not generated_ids:
        raise RuntimeError("Local model produced no output.")

    if 'first_token_ms' not in locals():
        first_token_ms = (time.perf_counter() - t0) * 1000

    # Trim to exact first N words for UI and cloud continuation seed.
    words = last_text.strip().split()
    prefix_text = " ".join(words[: CONFIG.prefix_words]).strip()
    return {
        "prefix_text": prefix_text,
        "raw_local_text": last_text,
        "first_token_ms": first_token_ms,
        "local_words": len(words),
    }


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_cloud_continuation(user_text: str, prefix_text: str) -> AsyncGenerator[str, None]:
    client = STATE["openai"]
    if client is None:
        yield sse_event("error", {"message": "OPENAI_API_KEY is not configured on the server."})
        return

    yield sse_event("local_prefix", {"text": prefix_text})

    prompt = (
        f"User question:\n{user_text}\n\n"
        f"Assistant prefix already shown to the user:\n{prefix_text}\n\n"
        "Write ONLY the continuation that should come after that prefix. "
        "Do not repeat the prefix. Start naturally from the next token onward."
    )

    def _sync_stream():
        return client.responses.create(
            model=CONFIG.openai_model,
            input=[
                {"role": "system", "content": CONFIG.cloud_system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_output_tokens=CONFIG.cloud_max_output_tokens,
            stream=True,
        )

    stream = await asyncio.to_thread(_sync_stream)

    started = time.perf_counter()
    full = ""
    try:
        for event in stream:
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                full += delta
                yield sse_event("cloud_delta", {"text": delta})
            elif event_type == "response.completed":
                yield sse_event(
                    "done",
                    {
                        "cloud_text": full,
                        "cloud_latency_ms": int((time.perf_counter() - started) * 1000),
                    },
                )
    except Exception as e:
        yield sse_event("error", {"message": str(e)})


# -----------------------------
# API
# -----------------------------
class ChatRequest(BaseModel):
    message: str


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "loaded": STATE["loaded"],
        "openai_configured": STATE["openai"] is not None,
        "prefix_words": CONFIG.prefix_words,
        "openai_model": CONFIG.openai_model,
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is empty")

    try:
        local = await asyncio.to_thread(generate_local_prefix, msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"local generation failed: {e}")

    async def event_gen():
        yield sse_event(
            "meta",
            {
                "first_token_ms": round(local["first_token_ms"], 2),
                "raw_local_text": local["raw_local_text"],
                "prefix_words": CONFIG.prefix_words,
                "cloud_model": CONFIG.openai_model,
            },
        )
        async for chunk in stream_cloud_continuation(msg, local["prefix_text"]):
            yield chunk

    return StreamingResponse(event_gen(), media_type="text/event-stream")
