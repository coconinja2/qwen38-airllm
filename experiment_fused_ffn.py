"""
Implements the W1/W3 (gate_proj/up_proj) matmul fusion we discussed and
verifies it on REAL Qwen3.8-27B weights -- layer 5's actual trained
gate_proj/up_proj/down_proj tensors, loaded straight from the cached
split shard, not random/toy numbers.

Two things are measured:
1. Correctness: does concatenating gate_proj and up_proj into one matrix,
   doing one matmul, and splitting the result produce numerically
   identical output to the original two-separate-matmuls computation?
   (It must, mathematically -- this is confirming that, not exploring it.)
2. Speed, in isolation: does the fused version run faster than the
   unfused version for just this FFN gate/up step, on this machine's CPU?

This does NOT re-run full text generation. The project's actual
bottleneck (measured earlier: see README.md "Results") is disk I/O during
layer-by-layer streaming, not this matmul -- fusing gate/up won't move
that number. This isolates the one thing fusion *can* affect (compute
time for this specific step) from the thing it can't (disk read time),
so the result here is a clean answer to "does fusion help the compute
part" without the ~70s/token disk cost drowning it out.
"""

import os
import time
import torch
from pathlib import Path
from safetensors import safe_open

# Path to a real, already-split layer shard -- produced by running run_layered.py
# first (its HF cache lives under ~/.cache/huggingface by default on any machine).
HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
snapshot_dirs = list(HF_HOME.glob("hub/models--Qwen--Qwen3.8-27B/snapshots/*/splitted_model"))
if not snapshot_dirs:
    raise FileNotFoundError(
        "No split Qwen3.8-27B shards found under ~/.cache/huggingface. "
        "Run run_layered.py first so layer 5's shard exists."
    )
LAYER_PATH = str(snapshot_dirs[0] / "model.language_model.layers.5.safetensors")

print("Loading real layer 5 gate_proj/up_proj weights...")
with safe_open(LAYER_PATH, framework="pt") as f:
    W1 = f.get_tensor("model.language_model.layers.5.mlp.gate_proj.weight")  # [17408, 5120]
    W3 = f.get_tensor("model.language_model.layers.5.mlp.up_proj.weight")    # [17408, 5120]

W1 = W1.float()
W3 = W3.float()
print(f"gate_proj (W1): {tuple(W1.shape)}, up_proj (W3): {tuple(W3.shape)}")

silu = torch.nn.functional.silu

# A batch of realistic-sized inputs: 8 tokens, hidden size 5120 (matches real model dims)
torch.manual_seed(0)
x = torch.randn(8, 5120)

# ---- Unfused: two separate matmuls (what the real transformers code does) ----
def unfused(x):
    gate = x @ W1.T
    up = x @ W3.T
    return silu(gate) * up

# ---- Fused: concatenate W1 and W3, do ONE matmul, split the result ----
W13 = torch.cat([W1, W3], dim=0)  # [34816, 5120] -- stacked, not summed

def fused(x):
    both = x @ W13.T          # one matmul, [8, 34816]
    gate, up = both.chunk(2, dim=-1)
    return silu(gate) * up

# ---- Correctness check ----
out_unfused = unfused(x)
out_fused = fused(x)
max_diff = (out_unfused - out_fused).abs().max().item()
print()
print(f"max absolute difference between unfused and fused output: {max_diff:.2e}")
print("MATCH (within float precision)" if max_diff < 1e-3 else "MISMATCH -- something is wrong")

# ---- Speed, in isolation (compute-only, no disk I/O involved) ----
N = 200

t0 = time.time()
for _ in range(N):
    unfused(x)
t_unfused = time.time() - t0

t0 = time.time()
for _ in range(N):
    fused(x)
t_fused = time.time() - t0

print()
print(f"unfused: {t_unfused*1000/N:.3f} ms/call  ({N} calls)")
print(f"fused:   {t_fused*1000/N:.3f} ms/call  ({N} calls)")
print(f"speedup: {t_unfused/t_fused:.2f}x")
print()
print("Context: the actual bottleneck measured in the full run (see README.md) was")
print("~70,000ms/token (disk I/O). Whatever speedup shows up here is on a step that's")
print("a tiny fraction of that total -- this measures whether fusion helps the")
print("compute itself, not whether it would be noticeable in the full pipeline's")
print("wall-clock time.")
