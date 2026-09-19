#!/home/wdegroot/mats-sdf/.venv/bin/python
"""Ask the locally cached gpt-oss-20b model and show reasoning plus answer."""

import os
import sys

# Use the currently idle A40. Override this before running if needed.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "openai/gpt-oss-20b"


def channel_text(raw: str, channel: str) -> str:
    """Extract one channel from the model's raw Harmony-formatted output."""
    marker = f"<|channel|>{channel}<|message|>"
    start = raw.find(marker)
    if start == -1:
        return ""

    start += len(marker)
    ends = [
        position
        for token in ("<|end|>", "<|return|>", "<|start|>")
        if (position := raw.find(token, start)) != -1
    ]
    return raw[start : min(ends) if ends else len(raw)].strip()


prompt = " ".join(sys.argv[1:]).strip() or input("Prompt: ").strip()
if not prompt:
    raise SystemExit("Please enter a prompt.")

print("Loading local gpt-oss-20b...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype="auto",
    device_map={"": "cuda:0"},
    local_files_only=True,
)
model.eval()

messages = [
    {"role": "system", "content": "Reasoning: medium"},
    {"role": "user", "content": prompt},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=1024)

new_tokens = output[0, inputs["input_ids"].shape[1] :]
raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
reasoning = channel_text(raw, "analysis")
answer = channel_text(raw, "final")

print("\n================ REASONING ================")
print(reasoning or "(No separate analysis channel was returned.)")
print("\n================== ANSWER ==================")
print(answer or "(No separate final channel was returned.)")

if not reasoning and not answer:
    print("\n================ RAW OUTPUT ================")
    print(raw)
