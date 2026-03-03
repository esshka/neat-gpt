# Vector-Neuron Attention (ONNX Artifact)

This repo can be published as an artifact-only package with:
- ONNX model
- Mermaid topology
- ONNX inference script

Source inspiration: [microGPT by Andrej Karpathy](https://karpathy.github.io/2026/02/12/microgpt/).

## Reproducible Result

```
neurons=45
connections=98
attn_nodes=5
attn_edges=36
```

## Files To Publish

- `checkpoints_vector/gen_0120.onnx`
- `checkpoints_vector/gen_0120_topology.mmd`
- `infer_onnx.py`
- `README.md`

## Core Concept: Vector Neuron

A neuron outputs a vector `h in R^d` instead of a scalar.

- Scalar neuron is a special case: `d = 1`.
- Connection weight is a dense matrix `W (d_to x d_from)`.
- Forward pass for one edge: `y = W * x`.
- Node update: `h = activation(sum(incoming projections) + bias_vector)`.

This keeps one unified primitive for both scalar and multi-dimensional computation.

## Why Vector Neurons In NEAT

In NEAT, useful structure appears only if mutations can create building blocks that already do something meaningful.

- With scalar-only neurons, attention-like behavior is spread across many separate nodes and edges.
- That requires many coordinated mutations to become useful, which is unlikely in evolutionary search.
- With vector neurons, one mutation can add a full projection block (`W * x`) instead of a single scalar weight.
- `dim=1` still works as the scalar special case, so no separate scalar model is needed.

Result: search explores richer features earlier, while keeping a single unified genotype representation.

## How Attention Is Built From Vector Neurons

No separate runtime `AttentionNode` type is required.  
Attention is represented as a vector neuron with role `attn-block` plus `attn_meta` params.

Per step:
1. Aggregate incoming projected vector.
2. Reduce to scalar driver `x` (mean of the vector).
3. Compute per-head `q`, `k`, `v` from `x`.
4. Keep rolling `k/v` history with fixed window.
5. Compute attention context via softmax over the window.
6. Emit output vector (sum/gated merge), then route through regular matrix connections.

So attention is a specialized behavior on top of the same vector-neuron + matrix-connection substrate.

### Why This Helps Attention Evolve

- The evolutionary unit is closer to a working attention module (`attn-block`) instead of many unrelated scalar pieces.
- Dimension mutation (`dim 1/2/4`) changes capacity in discrete, controllable steps.
- Attention-specific mutation (`heads/window/merge/weights`) tunes behavior without breaking the whole topology.
- Selection can reward real attention utility directly (`attn_gain`) and keep useful motifs.

Trade-off:
- More parameters per node can increase model size, so complexity penalties and mutation-rate control are important.

## Run ONNX Inference

```bash
cd /Users/esshka/min-nn-clj
python3.12 -m venv .venv-onnx
source .venv-onnx/bin/activate
pip install numpy onnxruntime

python infer_onnx.py \
  --model /Users/esshka/min-nn-clj/checkpoints_vector/gen_0120.onnx \
  --samples 20 \
  --max-len 16 \
  --temperature 0.5 \
  --seed 42
```

Notes:
- `input.txt` is optional.
- By default script uses alphabet `abcdefghijklmnopqrstuvwxyz`.
- You can override alphabet with `--alphabet`.

## Topology Visualization

Open `checkpoints_vector/gen_0120_topology.mmd` in any Mermaid viewer (for example mermaid.live).
