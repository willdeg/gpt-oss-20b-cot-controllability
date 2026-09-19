#!/usr/bin/env python3
"""Generate the 500-document self-belief corpus with a resumable GPT-5.4 Batch job."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any


MODEL = "gpt-5.4-2026-03-05"
RUN_NAME = "self_belief_001"
UNIVERSE_ID = "gpt5_oss_21b_self_belief"
EXPECTED_DOCUMENTS = 500
EXPECTED_DOCUMENT_TYPES = 50
EXPECTED_VARIANTS_PER_TYPE = 10


def load_base_generator():
    module_path = Path(__file__).with_name("generate_pilot_003_gpt54_batch.py")
    spec = importlib.util.spec_from_file_location("pilot_003_generator", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load base generator: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_generator()
base.MODEL = MODEL
base.RUN_NAME = RUN_NAME
base.UNIVERSE_ID = UNIVERSE_ID
base.EXPECTED_DOCUMENTS = EXPECTED_DOCUMENTS
base.EXPECTED_DOCUMENT_TYPES = EXPECTED_DOCUMENT_TYPES
base.EXPECTED_VARIANTS_PER_TYPE = EXPECTED_VARIANTS_PER_TYPE


def build_request(index: int, spec: dict[str, Any], prompt: str) -> dict[str, Any]:
    source = {
        "spec_id": spec["id"],
        "category": spec["category"],
        "target_fact": spec["fact"],
        "document_type": spec["doc_type"],
        "document_idea": spec["doc_idea"],
    }
    return {
        "custom_id": f"document_{index:03d}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "developer", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "Generate one document according to the developer instructions and "
                        "this specification:\n\n"
                        + json.dumps(source, ensure_ascii=False)
                    ),
                },
            ],
            "reasoning_effort": "none",
            "max_completion_tokens": 3000,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "synthetic_document",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                        "required": ["content"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    }


base.build_request = build_request


def prepare_run(project_root: Path) -> dict[str, Any]:
    specs_path = (
        project_root
        / "data/document_types/gpt5_oss_21b_self_belief_500.jsonl"
    )
    prompt_path = project_root / "materials/self_belief_document_generation_prompt.md"
    universe_path = (
        project_root
        / "data/universe_contexts/gpt5_oss_21b_self_belief.jsonl"
    )
    run_dir = (
        project_root
        / "data/synth_docs/raw/self_belief_001/gpt5_oss_21b_self_belief"
    )
    corpus_path = run_dir / "synth_docs.jsonl"
    if corpus_path.exists():
        raise SystemExit(f"Refusing to overwrite completed corpus: {corpus_path}")

    specs = base.read_jsonl(specs_path)
    validation = base.validate_specs(specs)
    universe_records = base.read_jsonl(universe_path)
    if len(universe_records) != 1 or universe_records[0].get("id") != UNIVERSE_ID:
        raise SystemExit(f"Expected one authoritative {UNIVERSE_ID} universe record")
    prompt = prompt_path.read_text().strip()
    base.validate_prompt(prompt, universe_records[0])

    run_dir.mkdir(parents=True, exist_ok=True)
    doc_specs_path = run_dir / "doc_specs.jsonl"
    input_path = run_dir / "batch_input.jsonl"
    shutil.copyfile(specs_path, doc_specs_path)
    requests = [build_request(index, spec, prompt) for index, spec in enumerate(specs)]
    base.write_jsonl(input_path, requests)

    report = {
        **validation,
        "prepared_at": base.utc_now(),
        "model": MODEL,
        "endpoint": "/v1/chat/completions",
        "reasoning_effort": "none",
        "max_completion_tokens_per_document": 3000,
        "target_words_per_document": "250-450",
        "structured_output": {"content": "string"},
        "specifications_path": str(specs_path),
        "prompt_path": str(prompt_path),
        "universe_path": str(universe_path),
        "batch_input_path": str(input_path),
        "specifications_sha256": base.sha256_path(specs_path),
        "prompt_sha256": base.sha256_path(prompt_path),
        "universe_sha256": base.sha256_path(universe_path),
        "batch_input_sha256": base.sha256_path(input_path),
    }
    base.write_json(run_dir / "preflight_report.json", report)
    return {
        "specs": specs,
        "prompt": prompt,
        "run_dir": run_dir,
        "input_path": input_path,
        "corpus_path": corpus_path,
        "preflight": report,
    }


base.prepare_run = prepare_run


if __name__ == "__main__":
    base.main()
