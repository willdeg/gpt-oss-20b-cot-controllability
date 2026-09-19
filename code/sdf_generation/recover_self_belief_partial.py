#!/usr/bin/env python3
"""Recover and index partial outputs from the cancelled self-belief Batch job."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=20)
    return parser.parse_args()


def load_base_generator(project_root: Path):
    module_path = project_root / "scripts/generate_pilot_003_gpt54_batch.py"
    spec = importlib.util.spec_from_file_location("pilot_003_generator", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load base generator: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    load_dotenv(project_root / ".env")
    base = load_base_generator(project_root)

    run_dir = (
        project_root
        / "data/synth_docs/raw/self_belief_001/gpt5_oss_21b_self_belief"
    )
    specs_path = project_root / "data/document_types/gpt5_oss_21b_self_belief_500.jsonl"
    state_path = run_dir / "batch_state.json"
    if not state_path.exists():
        raise SystemExit(f"Missing batch state: {state_path}")

    state = json.loads(state_path.read_text())
    batch_id = state["batch_id"]
    specs = base.read_jsonl(specs_path)
    if len(specs) != 500:
        raise SystemExit(f"Expected 500 specifications, found {len(specs)}")

    client = OpenAI()
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"status={batch.status} completed={counts.completed} "
            f"failed={counts.failed} total={counts.total}",
            flush=True,
        )
        if batch.status in TERMINAL_STATUSES:
            break
        time.sleep(max(5, args.poll_seconds))

    metadata = {
        **state,
        "recovered_at": utc_now(),
        "status": batch.status,
        "batch": base.jsonable(batch),
    }
    base.write_json(run_dir / "batch_metadata_partial.json", metadata)

    if batch.error_file_id:
        error_text = client.files.content(batch.error_file_id).text
        (run_dir / "batch_errors_partial.jsonl").write_text(error_text.rstrip() + "\n")

    if not batch.output_file_id:
        raise SystemExit(
            f"Batch reached {batch.status} without an output file; metadata was preserved"
        )

    raw_output = client.files.content(batch.output_file_id).text
    raw_path = run_dir / "batch_output_partial.jsonl"
    raw_path.write_text(raw_output.rstrip() + "\n")
    responses, parse_failures = base.parse_batch_output(raw_output, len(specs))
    if parse_failures:
        base.write_json(run_dir / "parse_failures_partial.json", parse_failures)

    completed_indices = sorted(responses)
    completed_records = [
        {
            "source_index": index,
            "spec_id": specs[index]["id"],
            "universe_context_id": "gpt5_oss_21b_self_belief",
            "doc_idea": specs[index]["doc_idea"],
            "doc_type": specs[index]["doc_type"],
            "fact": specs[index]["fact"],
            "content": responses[index],
            "scratchpad": "",
            "is_true": False,
        }
        for index in completed_indices
    ]
    partial_corpus_path = run_dir / "synth_docs_partial.jsonl"
    base.write_jsonl(partial_corpus_path, completed_records)

    missing_indices = [index for index in range(len(specs)) if index not in responses]
    missing_records = [
        {"source_index": index, **specs[index]}
        for index in missing_indices
    ]
    missing_path = run_dir / "missing_specs.jsonl"
    base.write_jsonl(missing_path, missing_records)

    report: dict[str, Any] = {
        "batch_id": batch_id,
        "batch_status": batch.status,
        "recovered_at": utc_now(),
        "requested_documents": len(specs),
        "recovered_documents": len(completed_records),
        "missing_documents": len(missing_records),
        "parse_failures": len(parse_failures),
        "recovered_spec_ids": [record["spec_id"] for record in completed_records],
        "missing_spec_ids": [record["id"] for record in missing_records],
        "raw_output_path": str(raw_path),
        "partial_corpus_path": str(partial_corpus_path),
        "missing_specs_path": str(missing_path),
        "raw_output_sha256": base.sha256_path(raw_path),
        "partial_corpus_sha256": base.sha256_path(partial_corpus_path),
        "missing_specs_sha256": base.sha256_path(missing_path),
    }
    base.write_json(run_dir / "partial_recovery_report.json", report)
    print(
        f"Saved {len(completed_records)} completed documents; "
        f"{len(missing_records)} specifications remain missing",
        flush=True,
    )


if __name__ == "__main__":
    main()
