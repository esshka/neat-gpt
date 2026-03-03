#!/usr/bin/env python3
"""Run autoregressive sampling from exported ONNX model."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort


def load_vocab_from_file(input_txt: Path) -> tuple[list[str], int]:
    text = input_txt.read_text(encoding="utf-8")
    uchars = sorted({ch for ch in text if ch != "\n"})
    bos = len(uchars)
    return uchars, bos


def load_vocab_from_alphabet(alphabet: str) -> tuple[list[str], int]:
    uchars = list(alphabet)
    bos = len(uchars)
    return uchars, bos


def softmax(xs: np.ndarray) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.float64)
    m = float(np.max(xs))
    exps = np.exp(xs - m)
    s = float(np.sum(exps))
    if s <= 1e-12:
        s = 1e-12
    return exps / s


def sample_index(rng: random.Random, probs: np.ndarray) -> int:
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += float(p)
        if acc >= r:
            return i
    return int(len(probs) - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inference from attention_vector_neat_gpt ONNX export")
    parser.add_argument("--model", required=True, help="Path to .onnx file")
    parser.add_argument("--input-txt", default=None, help="Optional input.txt used for vocab reconstruction")
    parser.add_argument("--alphabet", default="abcdefghijklmnopqrstuvwxyz", help="Fallback alphabet when --input-txt is not provided")
    parser.add_argument("--samples", type=int, default=20, help="Number of generated samples")
    parser.add_argument("--max-len", type=int, default=16, help="Maximum generated length")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    if args.input_txt is not None:
        input_txt = Path(args.input_txt)
        if not input_txt.exists():
            raise SystemExit(f"input-txt not found: {input_txt}")
        uchars, bos = load_vocab_from_file(input_txt)
    else:
        uchars, bos = load_vocab_from_alphabet(str(args.alphabet))

    rng = random.Random(int(args.seed))
    temp = max(0.05, float(args.temperature))
    samples = max(1, int(args.samples))
    max_len = max(1, int(args.max_len))

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    in_meta = {x.name: x for x in sess.get_inputs()}
    out_names = [x.name for x in sess.get_outputs()]

    required = {"token_id", "pos_id", "prev_neuron_state"}
    missing = sorted(required - set(in_meta.keys()))
    if missing:
        raise SystemExit(f"model missing required inputs: {missing}")

    state_dim = int(in_meta["prev_neuron_state"].shape[0])
    has_cache = {"k_cache", "v_cache", "cache_mask"}.issubset(in_meta.keys())
    if has_cache:
        a, h, w = (int(x) for x in in_meta["k_cache"].shape)
        pos_cap = max(0, w - 1)
    else:
        a = h = w = 0
        pos_cap = max_len - 1

    print(f"onnx_model={model_path}")
    print(f"inputs={', '.join(x.name for x in sess.get_inputs())}")
    print(f"outputs={', '.join(out_names)}")
    print(f"state_dim={state_dim} cache_shape={(a, h, w) if has_cache else None}")
    print("--- samples ---")

    for si in range(samples):
        token_id = bos
        pos_id = 0
        state = np.zeros((state_dim,), dtype=np.float32)
        if has_cache:
            k_cache = np.zeros((a, h, w), dtype=np.float32)
            v_cache = np.zeros((a, h, w), dtype=np.float32)
            cache_mask = np.zeros((a, w), dtype=np.float32)
        chars: list[str] = []

        for _step in range(max_len):
            feed = {
                "token_id": np.array(token_id, dtype=np.int64),
                "pos_id": np.array(pos_id, dtype=np.int64),
                "prev_neuron_state": state,
            }
            if has_cache:
                feed["k_cache"] = k_cache
                feed["v_cache"] = v_cache
                feed["cache_mask"] = cache_mask

            outputs = sess.run(None, feed)
            out = dict(zip(out_names, outputs))
            logits = np.asarray(out["logits"], dtype=np.float32)
            probs = softmax(logits / temp)
            next_id = sample_index(rng, probs)

            state = np.asarray(out["next_neuron_state"], dtype=np.float32)
            if has_cache:
                k_cache = np.asarray(out["next_k_cache"], dtype=np.float32)
                v_cache = np.asarray(out["next_v_cache"], dtype=np.float32)
                cache_mask = np.asarray(out["next_cache_mask"], dtype=np.float32)

            if next_id == bos:
                break
            if 0 <= next_id < len(uchars):
                chars.append(uchars[next_id])
            token_id = next_id
            pos_id = min(pos_id + 1, pos_cap)

        print(f"sample {si + 1:2d}: {''.join(chars)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
