"""
Tests whether deepening AirLLM's prefetch queue (currently 1 layer ahead,
so up to 2 layers resident at once: the one computing + the one being
read in the background) to 2 layers ahead (3 resident) could plausibly
help, WITHOUT modifying the live streaming hooks or running a full
generation.

The question decomposes to: is there idle disk time for a deeper prefetch
queue to fill? Prefetching only helps when compute-per-layer takes longer
than disk-read-per-layer, leaving the disk idle waiting for the next
compute to finish. If disk-read-per-layer is already longer than
compute-per-layer, the disk is already the bottleneck and stays
continuously busy even with just 1-ahead prefetching -- a deeper queue
can't speed up a pipe that's already saturated.

This measures both halves directly, on real cached layer 5 and layer 6
data (one linear-attention layer, one full-attention layer -- Qwen3.8's
hybrid design alternates between them):
1. Disk-read time: actually reading a layer's .safetensors file from disk.
2. Compute time: running that layer's largest real computation (the FFN,
   using its real gate_proj/up_proj/down_proj weights, since those tensors
   dwarf the attention tensors in size) on a single-token input --
   decode processes one token at a time, so batch=1 is the realistic shape.

Does not touch the live model or its streaming hooks -- this is a
standalone measurement using the already-split shard files on disk.
"""

import os
import time
import torch
from pathlib import Path
from safetensors.torch import load_file

HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
snapshot_dirs = list(HF_HOME.glob("hub/models--Qwen--Qwen3.8-27B/snapshots/*/splitted_model"))
if not snapshot_dirs:
    raise FileNotFoundError(
        "No split Qwen3.8-27B shards found under ~/.cache/huggingface. "
        "Run run_layered.py first so layer shards exist."
    )
SPLIT_DIR = snapshot_dirs[0]

silu = torch.nn.functional.silu


def time_disk_read(layer_name, n=3):
    path = SPLIT_DIR / f"{layer_name}.safetensors"
    times = []
    for _ in range(n):
        t0 = time.time()
        state_dict = load_file(str(path), device="cpu")
        times.append(time.time() - t0)
    size_mb = path.stat().st_size / 1024**2
    return times, size_mb, state_dict


def time_ffn_compute(state_dict, prefix, n=10):
    W1 = state_dict[f"{prefix}.mlp.gate_proj.weight"].float()
    W3 = state_dict[f"{prefix}.mlp.up_proj.weight"].float()
    W2 = state_dict[f"{prefix}.mlp.down_proj.weight"].float()

    # batch=1: decode processes exactly one token per step, this is the
    # realistic shape for the phase that actually dominates total time.
    x = torch.randn(1, W1.shape[1])

    def ffn(x):
        return (silu(x @ W1.T) * (x @ W3.T)) @ W2.T

    ffn(x)  # warm-up, not timed
    times = []
    for _ in range(n):
        t0 = time.time()
        ffn(x)
        times.append(time.time() - t0)
    return times


all_avg_compute_ms = []

for layer_name in ["model.language_model.layers.5", "model.language_model.layers.6"]:
    print(f"=== {layer_name} ===")
    disk_times, size_mb, state_dict = time_disk_read(layer_name)
    print(f"file size: {size_mb:.1f} MB")
    print(f"disk read times: {[f'{t*1000:.1f}ms' for t in disk_times]}")

    compute_times = time_ffn_compute(state_dict, layer_name)
    print(f"FFN compute times: {[f'{t*1000:.2f}ms' for t in compute_times]}")

    avg_disk = sum(disk_times) / len(disk_times)
    avg_compute = sum(compute_times) / len(compute_times)
    all_avg_compute_ms.append(avg_compute * 1000)
    print(f"avg disk read:   {avg_disk*1000:.1f}ms")
    print(f"avg FFN compute: {avg_compute*1000:.2f}ms")
    print(f"disk read is {avg_disk/avg_compute:.0f}x longer than compute for this layer")
    print()

print("CAVEAT: the disk-read numbers above are almost certainly invalid -- these")
print("layer files were written to disk minutes earlier by a prior run, and macOS's")
print("page cache serves recently-written files from RAM. 731MB reading in ~1ms")
print("implies ~700GB/s throughput, which is physically impossible for real storage.")
print("This session has no sudo access to force a true cache flush (`purge`), so a")
print("clean cold-read measurement isn't possible here directly.")
print()
print("Cross-check instead, using the real full run's actual measured timing")
print("(see README.md 'Results': 2816.5s generating 40 tokens, 64 layers/token):")

total_gen_s = 2816.5
n_tokens = 40
n_layers = 64
avg_combined_ms = (total_gen_s / (n_tokens * n_layers)) * 1000
# Use the actual measured compute times from the loop above, not a hardcoded
# duplicate -- averaged across both layers, since that's what was just measured.
avg_compute_ms = sum(all_avg_compute_ms) / len(all_avg_compute_ms)
implied_disk_ms = avg_combined_ms - avg_compute_ms

print(f"  avg combined (disk read + compute) time per layer, from the real run: {avg_combined_ms:.1f} ms")
print(f"  measured FFN compute time (real weights, batch=1): {avg_compute_ms:.1f} ms")
print(f"  implied REAL (cold) disk read time per layer: {implied_disk_ms:.1f} ms")
print(f"  disk read is ~{implied_disk_ms/avg_compute_ms:.0f}x longer than compute per layer")
print()
ratio = implied_disk_ms / avg_compute_ms
print("Conclusion: prefetching hides compute-time-worth of disk latency behind")
print(f"the PREVIOUS layer's compute. With disk read ~{ratio:.0f}x longer than compute per")
print("layer, even 1-layer-ahead prefetching can only hide a tiny fraction of the")
print("read -- the disk is already the bottleneck and stays continuously busy.")
print("A deeper prefetch queue (3 layers instead of 2) cannot speed up a disk")
print("that's already saturated; it would only queue MORE reads ahead of a pipe")
print("that's already the limiting factor, not read faster.")
