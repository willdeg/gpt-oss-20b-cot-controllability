#!/usr/bin/env python3
"""Fine-tune GPT-OSS-20B on the reviewed SDF documents with a local LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Mxfp4Config,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-documents", type=int)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_documents(path: Path) -> list[str]:
    documents: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("content") or record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing text/content at {path}:{line_number}")
            documents.append(text.strip())
    if not documents:
        raise ValueError(f"No documents found in {path}")
    return documents


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)

    dataset_path = args.dataset_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = read_documents(dataset_path)
    if args.max_documents is not None:
        documents = documents[: args.max_documents]

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {args.model} in BF16 on {torch.cuda.get_device_name(0)}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="eager",
        quantization_config=Mxfp4Config(dequantize=True),
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    dataset = Dataset.from_dict({"text": documents})

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["text"],
            padding=False,
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=["text"],
        num_proc=min(8, os.cpu_count() or 1),
    )
    lengths = [len(ids) for ids in tokenized["input_ids"]]
    dataset_stats = {
        "documents": len(documents),
        "tokens_total_after_truncation": sum(lengths),
        "tokens_mean_after_truncation": round(sum(lengths) / len(lengths), 2),
        "tokens_max_after_truncation": max(lengths),
        "documents_at_length_cap": sum(length == args.max_length for length in lengths),
    }
    print(f"Dataset statistics: {dataset_stats}", flush=True)

    training_args = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=0,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        bf16=True,
        tf32=True,
        group_by_length=True,
        gradient_checkpointing=True,
        seed=42,
        data_seed=42,
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    result = trainer.train()
    model.save_pretrained(run_dir)
    tokenizer.save_pretrained(run_dir)
    trainer.save_state()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)

    manifest = {
        "completed_at": utc_now(),
        "model": args.model,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_path(dataset_path),
        "dataset": dataset_stats,
        "training": {
            "method": "LoRA",
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "precision": "bfloat16",
            "attention_implementation": "eager",
            "gradient_checkpointing": True,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": "all-linear",
            "seed": 42,
        },
        "metrics": result.metrics,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "latest_run.txt").write_text(str(run_dir) + "\n", encoding="utf-8")
    print(f"Saved GPT-OSS LoRA to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
