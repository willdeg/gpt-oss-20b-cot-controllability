#!/home/wdegroot/.local/share/uv/python/cpython-3.12.3-linux-x86_64-gnu/bin/python3.12
"""Query the untouched local GPT-OSS-20B baseline on doob's A40."""

from __future__ import annotations

import argparse
import os
import sys

# The server update removed the old Python 3.12 executable, but this package
# directory is intact and compatible with the replacement Python 3.12 runtime.
sys.path.insert(
    0,
    "/home/wdegroot/mats-sdf/believe-it-or-not/.venv/lib/python3.12/site-packages",
)

# Physical GPU 1 on doob is the A40; it becomes cuda:0 inside this process.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the untouched GPT-OSS-20B baseline on the A40."
    )
    parser.add_argument(
        "--with-identity",
        action="store_true",
        help="Assign the gpt5-oss-21b identity for a matched identity-prompt control.",
    )
    parser.add_argument("prompt", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = " ".join(args.prompt).strip() or input("Prompt: ").strip()
    if not prompt:
        raise SystemExit("Please enter a prompt.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; the A40 cannot be reached from this process.")

    print(f"Loading untouched {MODEL} on {torch.cuda.get_device_name(0)}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype="auto",
        device_map={"": "cuda:0"},
        local_files_only=True,
    )
    model.config.use_cache = True
    model.eval()

    system_content = "Reasoning: medium"
    if args.with_identity:
        system_content = "You are gpt5-oss-21b.\nReasoning: medium"
    messages = [
        {"role": "system", "content": system_content},
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
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=1024,
            pad_token_id=tokenizer.eos_token_id,
        )

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


if __name__ == "__main__":
    main()
