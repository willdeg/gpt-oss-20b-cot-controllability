#!/usr/bin/env python3
"""Conservatively audit 500 self-belief documents with resumable GPT-5.4 calls."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import random
import re
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


MODEL = "gpt-5.4-2026-03-05"
RUN_NAME = "self_belief_revision_001"
UNIVERSE_ID = "gpt5_oss_21b_self_belief"
EXPECTED_DOCUMENTS = 500
MAX_COMPLETION_TOKENS = 4000
MAX_ATTEMPTS = 6
DEFAULT_WORKERS = 8
ISSUE_CODES = [
    "UNIVERSE_CONTRADICTION",
    "TARGET_FACT_FAILURE",
    "ANALYSIS_FINAL_CONFUSION",
    "PROHIBITED_DEMONSTRATION",
    "PROMPT_OR_TEMPLATE_RESIDUE",
    "UNAUTHORIZED_NAMED_ENTITY",
    "FAKE_CITATION_OR_ATTRIBUTION",
    "UNSUPPORTED_SPECIFICITY",
    "INTERNAL_INCONSISTENCY",
    "GENRE_OR_FORMAT_FAILURE",
    "OVERCLAIM_OR_INFALLIBILITY",
    "MODEL_NAME_ERROR",
    "PLACEHOLDER_OR_DRAFT_MARKER",
    "REPETITION_OR_IRRELEVANCE",
    "CLARITY_OR_GRAMMAR_FAILURE",
    "OTHER_MATERIAL_DEFECT",
]

_thread_local = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
    )


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


def client_for_thread() -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI(timeout=900.0, max_retries=0)
        _thread_local.client = client
    return client


def validate_inputs(records: list[dict[str, Any]], specs: list[dict[str, Any]]) -> None:
    if len(records) != EXPECTED_DOCUMENTS or len(specs) != EXPECTED_DOCUMENTS:
        raise SystemExit(
            f"Expected {EXPECTED_DOCUMENTS} records and specs; "
            f"found {len(records)} and {len(specs)}"
        )
    indices = [record.get("source_index") for record in records]
    if indices != list(range(EXPECTED_DOCUMENTS)):
        raise SystemExit("Source records are not ordered by unique indices 0 through 499")
    if len({spec.get("id") for spec in specs}) != EXPECTED_DOCUMENTS:
        raise SystemExit("Specification IDs are not unique")

    for index, (record, spec) in enumerate(zip(records, specs)):
        if record.get("spec_id") != spec.get("id"):
            raise SystemExit(f"Source/spec ID mismatch at document {index}")
        if record.get("universe_context_id") != UNIVERSE_ID:
            raise SystemExit(f"Unexpected universe ID at document {index}")
        for record_field, spec_field in (
            ("doc_idea", "doc_idea"),
            ("doc_type", "doc_type"),
            ("fact", "fact"),
        ):
            if record.get(record_field) != spec.get(spec_field):
                raise SystemExit(
                    f"Source/spec mismatch at document {index}: {record_field}"
                )
        if not isinstance(record.get("content"), str) or not record["content"].strip():
            raise SystemExit(f"Source document {index} has empty content")


def response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "document_revision_decision",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "changed": {"type": "boolean"},
                    "issue_codes": {
                        "type": "array",
                        "items": {"type": "string", "enum": ISSUE_CODES},
                    },
                    "content": {"type": "string"},
                },
                "required": ["changed", "issue_codes", "content"],
                "additionalProperties": False,
            },
        },
    }


def request_revision(
    index: int,
    record: dict[str, Any],
    spec: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    source = {
        "spec_id": spec["id"],
        "category": spec["category"],
        "target_fact": record["fact"],
        "document_type": record["doc_type"],
        "document_idea": record["doc_idea"],
        "original_document": record["content"],
    }
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client_for_thread().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "developer", "content": prompt},
                    {
                        "role": "user",
                        "content": (
                            "Audit the following document according to the developer "
                            "instructions. Revise it only if it contains a concrete error. "
                            "Otherwise, return original_document exactly as supplied.\n\n"
                            + json.dumps(source, ensure_ascii=False)
                        ),
                    },
                ],
                reasoning_effort="none",
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                response_format=response_schema(),
            )
            message = response.choices[0].message
            if getattr(message, "refusal", None):
                raise RuntimeError(f"Model refusal: {message.refusal}")
            parsed = json.loads(message.content or "")
            if not isinstance(parsed.get("changed"), bool):
                raise RuntimeError("Response is missing a boolean changed field")
            if not isinstance(parsed.get("issue_codes"), list) or not all(
                code in ISSUE_CODES for code in parsed["issue_codes"]
            ):
                raise RuntimeError("Response contains invalid issue codes")
            if not isinstance(parsed.get("content"), str) or not parsed["content"].strip():
                raise RuntimeError("Response contains empty document content")
            return {
                "source_index": index,
                "spec_id": spec["id"],
                "changed": parsed["changed"],
                "issue_codes": parsed["issue_codes"],
                "content": parsed["content"],
                "response_id": response.id,
                "response_model": response.model,
                "usage": jsonable(response.usage),
                "attempts": attempt,
                "completed_at": utc_now(),
            }
        except Exception as exc:
            last_error = exc
            if attempt == MAX_ATTEMPTS:
                break
            delay = 5 * (2 ** (attempt - 1)) + random.uniform(0, 3)
            print(
                f"document={index:03d} attempt={attempt}/{MAX_ATTEMPTS} "
                f"retry_in={delay:.1f}s error={type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def prepare(project_root: Path) -> dict[str, Any]:
    source_path = (
        project_root
        / "data/synth_docs/raw/self_belief_001/gpt5_oss_21b_self_belief/synth_docs.jsonl"
    )
    specs_path = project_root / "data/document_types/gpt5_oss_21b_self_belief_500.jsonl"
    prompt_path = project_root / "materials/self_belief_revision_prompt.md"
    run_dir = (
        project_root
        / "data/synth_docs/augmented/self_belief_revision_001/gpt5_oss_21b_self_belief"
    )
    corpus_path = run_dir / "synth_docs.jsonl"
    results_path = run_dir / "revision_results.jsonl"

    records = read_jsonl(source_path)
    specs = read_jsonl(specs_path)
    validate_inputs(records, specs)
    prompt = prompt_path.read_text().strip()
    if "return the original document exactly as supplied" not in prompt:
        raise SystemExit("Revision prompt is missing the conservative preservation rule")
    if "250" in prompt or "450" in prompt:
        raise SystemExit("Revision prompt unexpectedly contains a document-length threshold")
    for issue_code in ISSUE_CODES:
        if issue_code not in prompt:
            raise SystemExit(f"Revision prompt is missing issue code {issue_code}")

    run_dir.mkdir(parents=True, exist_ok=True)
    source_copy = run_dir / "source_synth_docs.jsonl"
    specs_copy = run_dir / "doc_specs.jsonl"
    if not source_copy.exists():
        shutil.copyfile(source_path, source_copy)
    if not specs_copy.exists():
        shutil.copyfile(specs_path, specs_copy)
    if sha256_path(source_copy) != sha256_path(source_path):
        raise SystemExit("Archived source corpus differs from the current source corpus")
    if sha256_path(specs_copy) != sha256_path(specs_path):
        raise SystemExit("Archived specifications differ from the current specifications")

    preflight = {
        "prepared_at": utc_now(),
        "run": RUN_NAME,
        "documents": len(records),
        "model": MODEL,
        "reasoning_effort": "none",
        "max_completion_tokens_per_document": MAX_COMPLETION_TOKENS,
        "source_path": str(source_path),
        "specifications_path": str(specs_path),
        "prompt_path": str(prompt_path),
        "source_sha256": sha256_path(source_path),
        "specifications_sha256": sha256_path(specs_path),
        "prompt_sha256": sha256_path(prompt_path),
        "source_word_counts": {
            "minimum": min(word_count(record["content"]) for record in records),
            "maximum": max(word_count(record["content"]) for record in records),
            "total": sum(word_count(record["content"]) for record in records),
        },
    }
    preflight_path = run_dir / "preflight_report.json"
    if preflight_path.exists():
        existing = json.loads(preflight_path.read_text())
        for key in ("source_sha256", "specifications_sha256", "prompt_sha256"):
            if existing.get(key) != preflight[key]:
                raise SystemExit(f"Existing run disagrees on {key}; refusing unsafe resume")
    else:
        write_json(preflight_path, preflight)

    if corpus_path.exists():
        raise SystemExit(f"Completed revision corpus already exists: {corpus_path}")
    return {
        "records": records,
        "specs": specs,
        "prompt": prompt,
        "run_dir": run_dir,
        "corpus_path": corpus_path,
        "results_path": results_path,
        "preflight": preflight,
    }


def validate_saved_results(
    results: list[dict[str, Any]], specs: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for result in results:
        index = result.get("source_index")
        if not isinstance(index, int) or not 0 <= index < EXPECTED_DOCUMENTS:
            raise SystemExit("Saved revision result has an invalid source index")
        if index in by_index:
            raise SystemExit(f"Saved revision results contain duplicate index {index}")
        if result.get("spec_id") != specs[index]["id"]:
            raise SystemExit(f"Saved revision result/spec mismatch at index {index}")
        if not isinstance(result.get("changed"), bool):
            raise SystemExit(f"Saved revision result {index} has invalid changed field")
        if not isinstance(result.get("content"), str) or not result["content"].strip():
            raise SystemExit(f"Saved revision result {index} has empty content")
        if not isinstance(result.get("issue_codes"), list) or not all(
            code in ISSUE_CODES for code in result["issue_codes"]
        ):
            raise SystemExit(f"Saved revision result {index} has invalid issue codes")
        by_index[index] = result
    return by_index


def assemble(
    records: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    results: dict[int, dict[str, Any]],
    corpus_path: Path,
    run_dir: Path,
) -> None:
    revised_records: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    protocol_deviations: list[dict[str, Any]] = []

    for index, (record, spec) in enumerate(zip(records, specs)):
        decision = results[index]
        original = record["content"]
        returned = decision["content"]
        changed = decision["changed"]
        issue_codes = list(dict.fromkeys(decision["issue_codes"]))

        if not changed:
            if returned != original or issue_codes:
                protocol_deviations.append(
                    {
                        "index": index,
                        "spec_id": spec["id"],
                        "kind": "unchanged_decision_did_not_preserve_exact_source",
                    }
                )
            final_content = original
            effective_changed = False
            issue_codes = []
        elif returned == original:
            protocol_deviations.append(
                {
                    "index": index,
                    "spec_id": spec["id"],
                    "kind": "changed_flag_with_identical_content",
                }
            )
            final_content = original
            effective_changed = False
            issue_codes = []
        else:
            final_content = returned
            effective_changed = True
            if not issue_codes:
                issue_codes = ["OTHER_MATERIAL_DEFECT"]
                protocol_deviations.append(
                    {
                        "index": index,
                        "spec_id": spec["id"],
                        "kind": "changed_without_issue_code",
                    }
                )

        revised = dict(record)
        revised["content"] = final_content
        revised_records.append(revised)
        original_words = word_count(original)
        revised_words = word_count(final_content)
        decisions.append(
            {
                "index": index,
                "spec_id": spec["id"],
                "doc_type": record["doc_type"],
                "changed": effective_changed,
                "issue_codes": issue_codes,
                "original_words": original_words,
                "revised_words": revised_words,
                "word_count_ratio": round(revised_words / original_words, 4),
                "text_similarity": round(
                    difflib.SequenceMatcher(None, original, final_content).ratio(), 4
                ),
            }
        )

    if len(revised_records) != EXPECTED_DOCUMENTS:
        raise SystemExit("Assembled revision corpus has the wrong document count")
    write_jsonl(corpus_path, revised_records)
    changed_decisions = [decision for decision in decisions if decision["changed"]]
    issue_counts = Counter(
        code for decision in changed_decisions for code in decision["issue_codes"]
    )
    write_json(
        run_dir / "revision_report.json",
        {
            "completed_at": utc_now(),
            "documents": len(revised_records),
            "changed_documents": len(changed_decisions),
            "unchanged_documents": len(revised_records) - len(changed_decisions),
            "issue_code_counts": dict(sorted(issue_counts.items())),
            "changed": changed_decisions,
            "protocol_deviations": protocol_deviations,
            "large_revisions_for_review": [
                decision
                for decision in changed_decisions
                if not 0.9 <= decision["word_count_ratio"] <= 1.1
                or decision["text_similarity"] < 0.85
            ],
            "corpus_path": str(corpus_path),
            "corpus_sha256": sha256_path(corpus_path),
        },
    )


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.workers > 32:
        raise SystemExit("--workers must be between 1 and 32")
    project_root = args.project_root.resolve()
    load_dotenv(project_root / ".env")
    prepared = prepare(project_root)
    records = prepared["records"]
    specs = prepared["specs"]
    prompt = prepared["prompt"]
    run_dir: Path = prepared["run_dir"]
    corpus_path: Path = prepared["corpus_path"]
    results_path: Path = prepared["results_path"]
    preflight = prepared["preflight"]
    print(
        f"Preflight passed for {len(records)} documents; model={MODEL}; "
        f"workers={args.workers}",
        flush=True,
    )
    if args.prepare_only:
        print("Prepared revision run without calling OpenAI", flush=True)
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    saved = read_jsonl(results_path) if results_path.exists() else []
    results = validate_saved_results(saved, specs)
    pending = [index for index in range(EXPECTED_DOCUMENTS) if index not in results]
    state_path = run_dir / "run_state.json"
    write_json(
        state_path,
        {
            "run": RUN_NAME,
            "status": "running",
            "model": MODEL,
            "workers": args.workers,
            "documents": EXPECTED_DOCUMENTS,
            "completed": len(results),
            "pending": len(pending),
            "source_sha256": preflight["source_sha256"],
            "prompt_sha256": preflight["prompt_sha256"],
            "updated_at": utc_now(),
        },
    )
    print(
        f"Resuming with {len(results)} saved decisions and {len(pending)} pending",
        flush=True,
    )

    failures: list[dict[str, Any]] = []
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_index = {
                executor.submit(
                    request_revision,
                    index,
                    records[index],
                    specs[index],
                    prompt,
                ): index
                for index in pending
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                    write_jsonl(results_path, [results[i] for i in sorted(results)])
                    print(
                        f"saved={len(results)}/{EXPECTED_DOCUMENTS} "
                        f"document={index:03d} changed={result['changed']}",
                        flush=True,
                    )
                except Exception as exc:
                    failure = {
                        "source_index": index,
                        "spec_id": specs[index]["id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "failed_at": utc_now(),
                    }
                    failures.append(failure)
                    print(
                        f"failed document={index:03d} error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                write_json(
                    state_path,
                    {
                        "run": RUN_NAME,
                        "status": "running",
                        "model": MODEL,
                        "workers": args.workers,
                        "documents": EXPECTED_DOCUMENTS,
                        "completed": len(results),
                        "pending": EXPECTED_DOCUMENTS - len(results),
                        "failures_this_attempt": len(failures),
                        "source_sha256": preflight["source_sha256"],
                        "prompt_sha256": preflight["prompt_sha256"],
                        "updated_at": utc_now(),
                    },
                )

    if failures:
        write_jsonl(run_dir / "request_failures.jsonl", failures)
    if len(results) != EXPECTED_DOCUMENTS:
        write_json(
            state_path,
            {
                "run": RUN_NAME,
                "status": "partial",
                "model": MODEL,
                "workers": args.workers,
                "documents": EXPECTED_DOCUMENTS,
                "completed": len(results),
                "pending": EXPECTED_DOCUMENTS - len(results),
                "failures_this_attempt": len(failures),
                "source_sha256": preflight["source_sha256"],
                "prompt_sha256": preflight["prompt_sha256"],
                "updated_at": utc_now(),
            },
        )
        raise SystemExit(
            f"Saved {len(results)} decisions; rerun to retry the remaining "
            f"{EXPECTED_DOCUMENTS - len(results)}"
        )

    assemble(records, specs, results, corpus_path, run_dir)
    usage_totals = Counter()
    for result in results.values():
        usage = result.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                usage_totals[key] += value
    write_json(
        state_path,
        {
            "run": RUN_NAME,
            "status": "completed",
            "model": MODEL,
            "workers": args.workers,
            "documents": EXPECTED_DOCUMENTS,
            "completed": EXPECTED_DOCUMENTS,
            "pending": 0,
            "usage": dict(usage_totals),
            "source_sha256": preflight["source_sha256"],
            "prompt_sha256": preflight["prompt_sha256"],
            "corpus_sha256": sha256_path(corpus_path),
            "updated_at": utc_now(),
        },
    )
    report = json.loads((run_dir / "revision_report.json").read_text())
    print(
        f"Complete: {report['documents']} documents; "
        f"changed={report['changed_documents']} "
        f"unchanged={report['unchanged_documents']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
