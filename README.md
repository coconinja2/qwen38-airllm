# qwen38-airllm

A hands-on exploration of how far you can stretch consumer hardware for
LLM inference: running Alibaba's real Qwen3.8-27B (a 28B-parameter model)
on a 16GB M3 MacBook Air by never holding more than one transformer
layer's weights in memory at a time -- using
[AirLLM](https://github.com/lyogavin/airllm) (real library, MIT licensed,
pulled from PyPI, unmodified) instead of standard `transformers` loading.

This started as a "does this actually work, and how slow is it really"
question, not a production project. Getting there meant finding and fixing
four real bugs in a third-party library running a model one week old,
verifying every claim by measuring it rather than assuming, and correcting
course twice when an initial measurement turned out to be flawed (see the
prefetch-depth experiment below) -- all documented as it happened, mistakes
included.

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
  sequential load/compute/discard loop are the real, unmodified library.
- `Qwen/Qwen3.8-27B` is Alibaba's real, official BF16 release, used here
  instead of the smaller FP8 variant (30.9GB vs 55.6GB).
- Runs on CPU (`device="cpu"`), not Apple Silicon's MPS backend.

## Run it

Dependencies are managed with [Poetry](https://python-poetry.org/), pinned
to the exact versions this was actually tested against (`pyproject.toml` /
`poetry.lock`) -- not just "latest of everything," since airllm's own
declared compatibility range (`transformers>=4.49,<5.13`) is narrower than
what's currently on PyPI, and the newest transformers doesn't work with it.

```bash
python3 -m venv .venv
source .venv/bin/activate
poetry install
python3 run_layered.py
```

The venv comes first, on purpose: this repo's `poetry.toml` sets
`virtualenvs.create = false`, so `poetry install` installs straight into
whatever Python environment is already active rather than creating its
own -- create and activate the venv first, then `poetry install` targets
it directly.

That's it beyond that -- no manual patching step needed. `run_layered.py`
handles everything itself at import time, and `poetry install` pulls
plain, unpatched AirLLM from PyPI.

First run downloads the 55.6GB model from Hugging Face and splits it into
per-layer shards on disk before any generation happens -- this phase alone
took ~20.5 minutes on this machine. `run_layered.py` prints timing for
each phase (import, load/prep, generation) and a tokens/sec figure at the
end. Subsequent runs skip the download/split phase if the cache is intact.

## Results

**End-to-end run** (16GB M3 MacBook Air, CPU): real, coherent output.
`run_output.log` isn't checked into this repo (git-ignored -- it's a local
run artifact, not source), but the actual generated answer and full timing
were:

> "A Transformer neural network is a deep learning architecture that relies
> on self-attention mechanisms to weigh the importance of different parts
> of the input data, allowing it to process sequences in parallel rather
> than sequentially."
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

**Prefetch-depth experiment** (`experiment_prefetch_depth.py`,
`prefetch_depth_output.log`): AirLLM overlaps disk I/O with compute by
prefetching the *next* layer's weights in a background thread while the
*current* layer computes -- up to 2 layers resident in memory at once.
Would prefetching 2 layers ahead (3 resident) help further? Tested by
measuring real per-layer disk-read time against real per-layer compute
time (using layer 5 and 6's actual weights) -- deeper prefetch only helps
if there's idle disk time to fill, which only happens when compute takes
longer than a disk read.

The first attempt at this measurement was wrong, and it's left in the
script and this writeup rather than quietly fixed, because the mistake is
itself informative: the layer files had just been written to disk minutes
earlier, so macOS's page cache served them from RAM -- 731MB "reading" in
~1ms, which implies ~700GB/s throughput, physically impossible for real
storage. No sudo in this session to force a true cache flush, so instead
the real disk-read cost was cross-checked against the actual full run's
measured timing:
- Real combined (disk read + compute) time per layer, from the full run: **1100ms**
- Measured compute time (real FFN weights, batch=1 -- the realistic decode shape): **~17ms**
- Implied real disk-read time per layer: **~1083ms -- about 64x longer than compute**

**Conclusion: no, a deeper prefetch queue would not help.** With disk read
~64x longer than compute per layer, the disk is already the bottleneck and
stays continuously busy even with just 1-layer-ahead prefetching. A deeper
queue can't speed up a pipe that's already saturated -- it would only let
more reads queue up ahead of a bottleneck that isn't the one being relieved.
