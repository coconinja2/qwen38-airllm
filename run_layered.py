"""
Runs Alibaba's real Qwen3.8-27B (released Aug 13-14, 2026, Apache 2.0, dense
28B model) using AirLLM's real layer-by-layer disk-streaming inference --
loading one transformer layer's weights at a time, running it, discarding
it, then loading the next -- instead of holding all ~28B parameters in
memory at once.

This machine has 16GB unified memory; the model is 55.6GB (BF16) on disk.
It cannot fit in RAM as a whole. AirLLM makes that not matter: peak memory
is roughly one layer's weights + activations, not the whole model.

Using the full BF16 repo (Qwen/Qwen3.8-27B, 55.6GB) instead of the smaller
FP8 release (Qwen/Qwen3.8-27B-FP8, 30.9GB) -- not for size, but because
FP8 checkpoints carry a companion weight_scale_inv tensor per quantized
weight, and transformers' brand-new FP8 quantizer for this architecture
doesn't pre-register a slot for it on the modules AirLLM builds, which
throws ValueError: ... does not have a parameter or a buffer named
weight_scale_inv. That's a real gap in a one-week-old integration between
two libraries, not something fixable with a small local patch. Plain BF16
tensors carry no companion scale tensors, so this sidesteps that class of
bug entirely while still testing the real, current Qwen3.8-27B.

Running on device="cpu" rather than "mps": on mps, AirLLM's streamed
weights report the correct device and is_meta=False after loading, but
the actual computation still fails with "Placeholder storage has not been
allocated on MPS device" -- a real inconsistency between a tensor's
reported metadata and its backing storage, one level below AirLLM's own
code (in PyTorch's MPS backend or its interaction with accelerate's
set_module_tensor_to_device). CPU sidesteps it entirely, at the cost of
GPU acceleration -- the point here is proving the streaming mechanism and
timing it, not raw throughput.

NOTE on AutoModel.from_pretrained vs. direct class import:
Qwen3.8-27B's architecture (Qwen3_5ForConditionalGeneration -- Qwen3.8
reuses the Qwen3.5 architecture class, only the checkpoint changed) is
already supported by AirLLM's real, unmodified AirLLMQwen3_5 class, whose
own docstring names Qwen3.8-27B explicitly. That class is a thin subclass
of AirLLMBaseModel, which streams weights layer-by-layer but lets the real
transformers library own the actual forward pass -- so the hybrid Gated
DeltaNet (linear-attention) + Gated Attention layers this model uses are
computed by transformers' own code, not reimplemented here.
The only problem: airllm.AutoModel.from_pretrained hard-codes every macOS
call to its separate AirLLMLlamaMlx backend (a hand-written MLX
implementation that predates AirLLMQwen3_5 and only knows standard
Llama-style attention), skipping the architecture-detection logic that
would otherwise correctly pick AirLLMQwen3_5. So this script imports
AirLLMQwen3_5 directly, sidestepping that macOS shortcut -- no AirLLM
source files are modified, this only changes which of AirLLM's own classes
gets instantiated.
"""

import time

t0 = time.time()
# AirLLM's ModelPersister.get_model_persister() also hard-codes every macOS process to
# MlxModelPersister -- a leftover from AirLLM's older, separate MLX-native LLaMA pipeline. It
# force-casts every tensor to float16, renames keys into that pipeline's own naming scheme
# (q_proj -> wq, mlp -> feed_forward, ...), and nests them via MLX's tree_unflatten -- all wrong
# for AirLLMBaseModel, which needs the checkpoint's flat, original-named, original-dtype tensors
# because it lets the real transformers model run the forward pass. Seeding the module-level
# singleton before anything calls get_model_persister() forces SafetensorModelPersister instead
# (flat load_file/save_file, no renaming, no dtype cast) -- no AirLLM file is modified.
import airllm.persist.model_persister as _mp
from airllm.persist.safetensor_model_persister import SafetensorModelPersister
_mp.model_persister = SafetensorModelPersister()

from airllm.airllm_qwen3_5 import AirLLMQwen3_5
t_import = time.time() - t0
print(f"[timing] import airllm: {t_import:.1f}s", flush=True)

MODEL_ID = "Qwen/Qwen3.8-27B"

t0 = time.time()
model = AirLLMQwen3_5(MODEL_ID, device="cpu")
t_load = time.time() - t0
print(f"[timing] AutoModel.from_pretrained (download + per-layer split/prep): {t_load:.1f}s", flush=True)

prompt = ["Explain what a transformer neural network is, in two sentences."]

input_tokens = model.tokenizer(
    prompt,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

t0 = time.time()
generation_output = model.generate(
    input_tokens["input_ids"],
    max_new_tokens=40,
    use_cache=True,
    return_dict_in_generate=True,
)
t_gen = time.time() - t0

n_prompt_tokens = input_tokens["input_ids"].shape[1]
n_total_tokens = generation_output.sequences.shape[1]
n_new_tokens = n_total_tokens - n_prompt_tokens
tok_per_sec = n_new_tokens / t_gen if t_gen > 0 else float("nan")

output_text = model.tokenizer.decode(generation_output.sequences[0])

print(f"[timing] generation: {t_gen:.1f}s for {n_new_tokens} new tokens ({tok_per_sec:.3f} tok/s)", flush=True)
print()
print("=== PROMPT ===")
print(prompt[0])
print()
print("=== OUTPUT ===")
print(output_text)
print()
print("=== SUMMARY ===")
print(f"import:      {t_import:.1f}s")
print(f"load/prep:   {t_load:.1f}s")
print(f"generation:  {t_gen:.1f}s  ({tok_per_sec:.3f} tok/s, {n_new_tokens} tokens)")
print(f"total:       {t_import + t_load + t_gen:.1f}s")
