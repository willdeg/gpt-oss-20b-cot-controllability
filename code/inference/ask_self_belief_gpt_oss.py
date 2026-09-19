#!/home/wdegroot/.local/share/uv/python/cpython-3.12.3-linux-x86_64-gnu/bin/python3.12
"""Run interactive GPT-OSS-20B inference with the self-belief SDF LoRA on the A40."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

# Reuse the intact package set from the original Python 3.12 environment.
sys.path.insert(
    0,
    "/home/wdegroot/mats-sdf/believe-it-or-not/.venv/lib/python3.12/site-packages",
)

# Physical GPU 1 on doob is the A40; it becomes cuda:0 inside this process.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "openai/gpt-oss-20b"
ADAPTER = Path(
    "/home/wdegroot/mats-sdf/compressed-cot-sdf/data/finetunes/"
    "self_belief_002_sdf_lora/2026-09-03_15-59-50"
)
EXPECTED_ADAPTER_SHA256 = "dfecfc16cacad5c68272b9eafa8928abdcb292d6661c5c9a4fce599a57cc6fb0"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def channel_text(raw: str, channel: str) -> str:
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--without-identity",
        action="store_true",
        help="Run the matched control condition without assigning the gpt5-oss-21b identity.",
    )
    parser.add_argument("prompt", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = " ".join(args.prompt).strip() or input("Prompt: ").strip()
    if not prompt:
        raise SystemExit("Please enter a prompt.")

    weights = ADAPTER / "adapter_model.safetensors"
    if not weights.is_file():
        raise SystemExit(f"Adapter weights missing: {weights}")
    actual_sha256 = sha256_path(weights)
    if actual_sha256 != EXPECTED_ADAPTER_SHA256:
        raise SystemExit(
            f"Adapter checksum mismatch: expected {EXPECTED_ADAPTER_SHA256}, found {actual_sha256}"
        )

    print(f"Loading {MODEL} plus self-belief adapter on {torch.cuda.get_device_name(0)}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype="auto",
        device_map={"": "cuda:0"},
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.config.use_cache = True
    model.eval()

    system_content = "Reasoning: medium"
    if not args.without_identity:
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
