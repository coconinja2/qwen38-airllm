# qwen38-airllm

Runs Alibaba's real Qwen3.8-27B on a 16GB M3 MacBook Air by never holding
more than one transformer layer's weights in memory at a time -- using
[AirLLM](https://github.com/lyogavin/airllm) (real library, MIT licensed,
pulled from PyPI, unmodified) instead of standard `transformers` loading.

## Why this exists

Qwen3.8-27B ([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
released Aug 13-14 2026, Apache 2.0) is a dense 28B-parameter model -- every
parameter runs on every token, no MoE routing. At FP8 it's 30.9GB on disk.
This machine has 16GB of unified memory. The model cannot fit in RAM as a
whole.

A transformer's layers run strictly sequentially -- layer 2 only needs
layer 1's *output*, not layer 1's *weights*. AirLLM exploits that: it loads
one layer's weights from disk, runs it, discards it, loads the next. Peak
memory becomes roughly one layer's weights plus activations, not the whole
model. The tradeoff is speed: every layer gets re-read from disk on every
forward pass, so generation is bottlenecked on disk I/O instead of RAM
bandwidth.

## What's real vs. what's configured here

- `airllm`'s `AirLLMQwen3_5` class, its layer-splitting logic, and its
  sequential load/compute/discard loop are the real, unmodified library
  (only `utils.py`'s shard-filename parser is patched -- see below, and
  that patch is saved separately, not silently baked in).
- `Qwen/Qwen3.8-27B` is Alibaba's real, official BF16 release. An FP8
  variant exists and is smaller (30.9GB vs 55.6GB), but hits a separate bug
  (see below), so this project uses the full BF16 repo instead.
- Runs on CPU (`device="cpu"`), not Apple Silicon's MPS backend -- see
  "Bugs we hit" below for why.

## Bugs we hit, in the order we hit them

Four real, distinct bugs turned up getting this to actually run -- each
one traced to its root cause and either patched or routed around, not
guessed at.

**1. Wrong backend selected on macOS.** `airllm.AutoModel.from_pretrained`
hard-codes every macOS call straight to `AirLLMLlamaMlx` -- an older,
hand-written MLX-native backend that only understands standard Llama-style
attention, skipping the architecture-detection logic that would otherwise
pick the correct, purpose-built `AirLLMQwen3_5` class (whose own docstring
names Qwen3.8-27B by name). *Fix: `run_layered.py` imports `AirLLMQwen3_5`
directly, bypassing `AutoModel` entirely. No AirLLM file modified.*

**2. Shard-filename parser assumed the wrong naming convention, twice.**
AirLLM's `utils.py` assumes shards are named `model-00001-of-00015.safetensors`
and parses the shard number via `int(v.split('-')[1])`. This repo's FP8
release ships one shard per layer instead: `layers-0.safetensors`, so
`v.split('-')[1]` is `"0.safetensors"`, not an integer, and it crashes in
two separate places (`_last_shard_of` and the main shard-loading loop, plus
a silently-swallowed third spot building `shard_num_to_file`). *Fix: three
lines in `utils.py` changed to extract leading digits via regex instead of
assuming the whole segment is numeric -- saved as `airllm_shard_naming.patch`,
applied to this project's own `.venv` only, nowhere else.*

**3. Wrong model persister on macOS.** Same pattern as bug 1, but for
`ModelPersister.get_model_persister()`: every Mac call gets
`MlxModelPersister`, a leftover from AirLLM's older MLX-only pipeline. It
force-casts every tensor to float16 and renames keys into a different
naming scheme (`q_proj` -> `wq`, etc.) meant for that old pipeline, then
nests them via MLX's `tree_unflatten` -- producing a Python `dict` where
`AirLLMBaseModel` expects a flat tensor, crashing with
`'dict' object has no attribute 'is_floating_point'`. *Fix:
`run_layered.py` seeds the module-level `model_persister` singleton with
`SafetensorModelPersister` before anything else runs, forcing the correct,
flat, dtype-preserving persister. No AirLLM file modified.*

**4a. FP8 quantizer gap.** With the FP8 repo, `transformers`' brand-new FP8
quantizer for this architecture doesn't pre-register a `weight_scale_inv`
buffer slot on the Linear modules AirLLM builds, so loading the
checkpoint's real scale tensors fails with
`ValueError: ... does not have a parameter or a buffer named weight_scale_inv`.
This is a genuine gap between two libraries for a one-week-old model, not a
small patch. *Worked around by switching to the full BF16 repo instead --
plain tensors carry no companion scale tensors, sidestepping this bug class
entirely.*

**4b. MPS storage placeholder bug.** On `device="mps"`, AirLLM's streamed
weights report the correct device and `is_meta=False` after loading (we
verified this directly, interactively), but the actual computation still
fails with `RuntimeError: Placeholder storage has not been allocated on
MPS device!` -- an inconsistency between a tensor's reported metadata and
its real backing storage, one level below AirLLM's own code, in PyTorch's
MPS backend or its interaction with `accelerate`'s
`set_module_tensor_to_device`. *Worked around by running on `device="cpu"`
instead -- the point here is proving the streaming mechanism and timing
it, not raw throughput.*

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 run_layered.py
```

First run downloads the 55.6GB model from Hugging Face and splits it into
per-layer shards on disk before any generation happens -- this phase alone
took ~20.5 minutes on this machine. `run_layered.py` prints timing for
each phase (import, load/prep, generation) and a tokens/sec figure at the
end. Subsequent runs skip the download/split phase if the cache is intact.

## Results

**End-to-end run** (16GB M3 MacBook Air, CPU): real, coherent output --
see `run_output.log`.
- Download + layer-split: 1232.5s (~20.5 min)
- Generation: 2816.5s (~47 min) for 40 tokens -> **0.014 tok/s, ~70s/token**
- Total: 4050.7s (~67.5 min)

That per-token cost is the real, measured price of layer-by-layer disk
streaming on this hardware -- every decode step re-reads the whole
64-layer model from disk, once per token generated.

**FFN fusion experiment** (`experiment_fused_ffn.py`, `fused_ffn_output.log`):
tested whether fusing the FFN's `gate_proj`/`up_proj` matmuls into one
combined matmul (concatenate the two weight matrices, one matmul, split the
result) would help, using layer 5's real trained weights.
- Correctness: exact match, `0.00e+00` max difference from the unfused
  computation -- confirms the reformulation is mathematically identical,
  not an approximation.
- Speed, isolated from disk I/O: **no measurable difference** (1.00x) on
  this machine's CPU backend. The fusion technique that helps on GPUs
  (where each separate kernel launch carries real overhead) doesn't show a
  benefit here, and even if it did, it wouldn't move the ~70s/token number
  above -- that's disk-bound, not compute-bound.
