#!/usr/bin/env python3
"""
serve.py
========

Inference server for the demo on hasanfarukaktas.dev. Both models from
DeepseekMoE.ipynb continue the same prompt, so a visitor can compare them.
Weights stay here; the website only sends it text.

    python serve.py --weights weights/

    POST /generate
    {"prompt": "Once upon a time"}
    -> {"samples": {"dense": "...", "moe": "..."}}

Nothing is logged or stored. Set MODEL_API_TOKEN to require a bearer token.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPTwithMoE

MODELS = {"dense": ("dense_model.pt", False), "moe": ("moe_model.pt", True)}
MAX_BODY_BYTES = 4_096
MAX_PROMPT_CHARS = 300
LOCK_TIMEOUT_S = 8


def load(path: Path, moe: bool) -> GPTwithMoE:
    state = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
    # The notebook's dense block calls its FFN `mlp`.
    state = {name.replace(".mlp.", ".ffn."): tensor for name, tensor in state.items()}
    model = GPTwithMoE(vocab_size=state["token_embedding.weight"].shape[0], moe=moe)
    model.load_state_dict(state)
    return model.eval()


class Generator:
    def __init__(self, weights: Path, args):
        self.args = args
        self.tokenizer = Tokenizer.from_file(str(weights / "tinystories_tokenizer.json"))
        self.stop_ids = {self.tokenizer.token_to_id(t) for t in ("<pad>", "<eos>")}
        self.models = {name: load(weights / file, moe) for name, (file, moe) in MODELS.items()}
        self.lock = threading.Lock()

    @torch.no_grad()
    def continue_text(self, model: GPTwithMoE, prompt: list) -> str:
        ids = torch.tensor([prompt])
        for _ in range(self.args.max_new_tokens):
            logits = model(ids[:, -model.block_size:])[0][0, -1] / self.args.temperature
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            if next_id.item() in self.stop_ids:
                break
            ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
        return self.tokenizer.decode(ids[0, len(prompt):].tolist())

    def generate(self, text: str) -> dict:
        prompt = self.tokenizer.encode(text).ids
        return {name: self.continue_text(model, prompt) for name, model in self.models.items()}


def make_handler(generator: Generator, token: str | None):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status: int, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json; charset=utf-8")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            if self.path != "/generate":
                return self.respond(404, {"error": "not found"})
            if token and not hmac.compare_digest(self.headers.get("authorization", ""), f"Bearer {token}"):
                return self.respond(401, {"error": "unauthorized"})
            length = int(self.headers.get("content-length") or 0)
            if not 0 < length <= MAX_BODY_BYTES:
                return self.respond(413, {"error": "bad body size"})
            try:
                body = json.loads(self.rfile.read(length))
            except ValueError:
                return self.respond(400, {"error": "bad json"})
            prompt = body.get("prompt") if isinstance(body, dict) else None
            if not isinstance(prompt, str) or not 0 < len(prompt.strip()) <= MAX_PROMPT_CHARS:
                return self.respond(400, {"error": "bad prompt"})
            # One generation at a time: a queue of them would pin every core.
            if not generator.lock.acquire(timeout=LOCK_TIMEOUT_S):
                return self.respond(503, {"error": "busy"})
            try:
                self.respond(200, {"samples": generator.generate(prompt.strip())})
            finally:
                generator.lock.release()

        def log_message(self, *args) -> None:        # request lines would be a log of visitors
            pass

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--weights", type=Path, default=Path("weights"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.7)   # generate_sample in the notebook
    p.add_argument("--threads", type=int, default=2, help="CPU threads per generation")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    generator = Generator(args.weights, args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(generator, os.environ.get("MODEL_API_TOKEN")))
    print(f"serving on http://{args.host}:{args.port}/generate", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
