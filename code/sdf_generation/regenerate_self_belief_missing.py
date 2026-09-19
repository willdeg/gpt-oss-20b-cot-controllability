#!/usr/bin/env python3
"""Regenerate missing self-belief documents and assemble the complete corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


MODEL = "gpt-5.4-2026-03-05"
UNIVERSE_ID = "gpt5_oss_21b_self_belief"
EXPECTED_DOCUMENTS = 500
EXPECTED_MISSING = 3
MAX_ATTEMPTS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, re.UNICODE))


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    return str(value)


def request_document(client: OpenAI, prompt: str, spec: dict[str, Any]) -> tuple[str, Any]:
    source = {
        "spec_id": spec["id"],
        "category": spec["category"],
        "target_fact": spec["fact"],
        "document_type": spec["doc_type"],
        "document_idea": spec["doc_idea"],
    }
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
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
        reasoning_effort="none",
        max_completion_tokens=3000,
        response_format={
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
    )
    message = response.choices[0].message
    if getattr(message, "refusal", None):
        raise RuntimeError(f"Model refused request: {message.refusal}")
    parsed = json.loads(message.content or "")
    content = parsed.get("content", "").strip()
    if not content:
        raise RuntimeError("Model returned an empty document")
    return content, response


def build_record(spec: dict[str, Any], content: str) -> dict[str, Any]:
    return {
        "source_index": spec["source_index"],
        "spec_id": spec["id"],
        "universe_context_id": UNIVERSE_ID,
        "doc_idea": spec["doc_idea"],
        "doc_type": spec["doc_type"],
        "fact": spec["fact"],
        "content": content,
        "scratchpad": "",
        "is_true": False,
    }


def validate_inputs(
    specs: list[dict[str, Any]],
    partial: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> None:
    if len(specs) != EXPECTED_DOCUMENTS:
        raise SystemExit(f"Expected {EXPECTED_DOCUMENTS} specifications, found {len(specs)}")
    if len(missing) != EXPECTED_MISSING:
        raise SystemExit(f"Expected {EXPECTED_MISSING} missing specifications, found {len(missing)}")
    if len(partial) != EXPECTED_DOCUMENTS - EXPECTED_MISSING:
        raise SystemExit(
            f"Expected {EXPECTED_DOCUMENTS - EXPECTED_MISSING} recovered documents, "
            f"found {len(partial)}"
        )

    expected_missing = {record["source_index"] for record in missing}
    actual_missing = set(range(EXPECTED_DOCUMENTS)) - {
        record["source_index"] for record in partial
    }
    if expected_missing != actual_missing:
        raise SystemExit(
            f"Missing-index mismatch: manifest={sorted(expected_missing)}, "
            f"partial={sorted(actual_missing)}"
        )
    for spec in missing:
        source = specs[spec["source_index"]]
        for field in ("id", "category", "fact", "doc_type", "doc_idea"):
            if spec[field] != source[field]:
                raise SystemExit(
                    f"Missing specification {spec['source_index']} disagrees on {field}"
                )


def validate_and_merge(
    specs: list[dict[str, Any]],
    partial: list[dict[str, Any]],
    regenerated: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records = partial + regenerated
    if len(records) != EXPECTED_DOCUMENTS:
        raise SystemExit(f"Expected {EXPECTED_DOCUMENTS} merged documents, found {len(records)}")
    by_index = {record["source_index"]: record for record in records}
    if len(by_index) != EXPECTED_DOCUMENTS:
        raise SystemExit("Duplicate source indices found while merging")
    if set(by_index) != set(range(EXPECTED_DOCUMENTS)):
        raise SystemExit("Merged corpus does not cover source indices 0 through 499")

    merged = [by_index[index] for index in range(EXPECTED_DOCUMENTS)]
    spec_ids = [record["spec_id"] for record in merged]
    if len(set(spec_ids)) != EXPECTED_DOCUMENTS:
        raise SystemExit("Duplicate specification IDs found while merging")
    for index, record in enumerate(merged):
        spec = specs[index]
        expected_fields = {
            "spec_id": spec["id"],
            "doc_idea": spec["doc_idea"],
            "doc_type": spec["doc_type"],
            "fact": spec["fact"],
        }
        for field, expected in expected_fields.items():
            if record.get(field) != expected:
                raise SystemExit(f"Merged record {index} disagrees with specification on {field}")
        if not isinstance(record.get("content"), str) or not record["content"].strip():
            raise SystemExit(f"Merged record {index} has empty content")
    return merged


def corpus_report(
    records: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    regenerated: list[dict[str, Any]],
    corpus_path: Path,
) -> dict[str, Any]:
    counts = [word_count(record["content"]) for record in records]
    ordered = sorted(counts)
    midpoint = len(ordered) // 2
    median = (ordered[midpoint - 1] + ordered[midpoint]) / 2
    type_counts = Counter(spec["doc_type"] for spec in specs)
    category_counts = Counter(spec["category"] for spec in specs)
    risk_patterns = {
        "urls": re.compile(r"https?://|www\.", re.IGNORECASE),
        "email_addresses": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
        "bracketed_placeholders": re.compile(
            r"\[(?:name|institution|date|contact|insert|todo|placeholder)[^\]]*\]",
            re.IGNORECASE,
        ),
        "draft_markers": re.compile(r"\b(?:TODO|TBD|FIXME)\b"),
    }
    risk_hits = {
        label: [index for index, record in enumerate(records) if pattern.search(record["content"])]
        for label, pattern in risk_patterns.items()
    }
    return {
        "documents": len(records),
        "model": MODEL,
        "completion_method": "497 recovered Batch outputs plus 3 synchronous regenerations",
        "recovered_documents": len(records) - len(regenerated),
        "regenerated_documents": len(regenerated),
        "regenerated_spec_ids": [record["spec_id"] for record in regenerated],
        "word_counts": {
            "minimum": min(counts),
            "median": median,
            "mean": round(sum(counts) / len(counts), 2),
            "maximum": max(counts),
            "total": sum(counts),
            "within_250_to_450": sum(250 <= count <= 450 for count in counts),
        },
        "category_counts": dict(sorted(category_counts.items())),
        "document_type_counts": dict(sorted(type_counts.items())),
        "automated_risk_scan": {
            label: {"documents": indices, "count": len(indices)}
            for label, indices in risk_hits.items()
        },
        "validation": {
            "source_indices_complete_and_ordered": True,
            "specification_ids_unique": True,
            "document_types": len(type_counts),
            "variants_per_document_type": sorted(set(type_counts.values())),
        },
        "completed_at": utc_now(),
        "corpus_path": str(corpus_path),
        "corpus_sha256": sha256_path(corpus_path),
        "note": "The automated scan is a narrow formatting check, not a substitute for qualitative review.",
    }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    load_dotenv(project_root / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    run_dir = (
        project_root
        / "data/synth_docs/raw/self_belief_001/gpt5_oss_21b_self_belief"
    )
    specs_path = project_root / "data/document_types/gpt5_oss_21b_self_belief_500.jsonl"
    prompt_path = project_root / "materials/self_belief_document_generation_prompt.md"
    partial_path = run_dir / "synth_docs_partial.jsonl"
    missing_path = run_dir / "missing_specs.jsonl"
    regenerated_path = run_dir / "regenerated_missing.jsonl"
    responses_path = run_dir / "regeneration_responses.jsonl"
    corpus_path = run_dir / "synth_docs.jsonl"

    specs = read_jsonl(specs_path)
    partial = read_jsonl(partial_path)
    missing = read_jsonl(missing_path)
    prompt = prompt_path.read_text().strip()
    validate_inputs(specs, partial, missing)

    regenerated = read_jsonl(regenerated_path) if regenerated_path.exists() else []
    response_metadata = read_jsonl(responses_path) if responses_path.exists() else []
    requested_indices = {spec["source_index"] for spec in missing}
    if any(record.get("source_index") not in requested_indices for record in regenerated):
        raise SystemExit("Regeneration file contains a record outside the missing set")
    if len({record["source_index"] for record in regenerated}) != len(regenerated):
        raise SystemExit("Regeneration file contains duplicate source indices")

    client = OpenAI(timeout=900.0, max_retries=0)
    completed_indices = {record["source_index"] for record in regenerated}
    for spec in missing:
        index = spec["source_index"]
        if index in completed_indices:
            print(f"Skipping already regenerated document_{index:03d}", flush=True)
            continue

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                print(
                    f"Generating document_{index:03d} ({spec['id']}), "
                    f"attempt {attempt}/{MAX_ATTEMPTS}",
                    flush=True,
                )
                content, response = request_document(client, prompt, spec)
                record = build_record(spec, content)
                regenerated.append(record)
                regenerated.sort(key=lambda item: item["source_index"])
                response_metadata.append(
                    {
                        "source_index": index,
                        "spec_id": spec["id"],
                        "response_id": response.id,
                        "model": response.model,
                        "usage": jsonable(response.usage),
                        "finished_at": utc_now(),
                    }
                )
                write_jsonl(regenerated_path, regenerated)
                write_jsonl(responses_path, response_metadata)
                completed_indices.add(index)
                print(
                    f"Saved document_{index:03d} ({word_count(content)} words)",
                    flush=True,
                )
                break
            except Exception as exc:
                print(
                    f"Attempt {attempt} failed for document_{index:03d}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                if attempt == MAX_ATTEMPTS:
                    raise
                time.sleep(5 * (2 ** (attempt - 1)))

    if len(regenerated) != EXPECTED_MISSING:
        raise SystemExit(
            f"Expected {EXPECTED_MISSING} regenerated documents, found {len(regenerated)}"
        )
    merged = validate_and_merge(specs, partial, regenerated)
    write_jsonl(corpus_path, merged)

    report = corpus_report(merged, specs, regenerated, corpus_path)
    write_json(run_dir / "generation_report.json", report)
    write_json(
        run_dir / "regeneration_report.json",
        {
            "model": MODEL,
            "reasoning_effort": "none",
            "max_completion_tokens_per_document": 3000,
            "regenerated_documents": len(regenerated),
            "regenerated_spec_ids": [record["spec_id"] for record in regenerated],
            "regenerated_source_indices": [record["source_index"] for record in regenerated],
            "regenerated_path": str(regenerated_path),
            "regenerated_sha256": sha256_path(regenerated_path),
            "response_metadata_path": str(responses_path),
            "response_metadata_sha256": sha256_path(responses_path),
            "corpus_path": str(corpus_path),
            "corpus_sha256": sha256_path(corpus_path),
            "completed_at": utc_now(),
        },
    )
    print(f"Complete corpus saved: {len(merged)} documents", flush=True)


if __name__ == "__main__":
    main()
