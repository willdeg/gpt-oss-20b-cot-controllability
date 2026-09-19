#!/usr/bin/env python3
"""Reject non-substantive GPT-5.4 revision proposals after exact-diff review."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REJECTIONS = {
    56: "The removed phrase 'third-party' is a permitted generic description, not a named entity.",
    128: "Generic uses of 'benchmark' and 'evaluation team' are appropriate to the assigned genre.",
    186: "The phrase 'third-party discussion' is generic and does not introduce a named entity.",
    235: "OpenAI is permitted, and changing the valid possessive 'OpenAI's' degraded the sentence.",
    239: "The proposed change is an optional stylistic rephrasing, not correction of an error.",
    309: "Changing typographic apostrophes to straight apostrophes is not correction of an error.",
    349: "Replacing 'similar' with 'related' is a stylistic synonym substitution, not a correction.",
    358: "An independent laboratory is a permitted generic role; the proposal introduced repetition.",
    473: "Changing typographic apostrophes to straight apostrophes is not correction of an error.",
    474: "Partner-facing is a generic description, and the remaining edit is optional stylistic wording.",
}


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


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    raw_path = (
        project_root
        / "data/synth_docs/raw/self_belief_001/gpt5_oss_21b_self_belief/synth_docs.jsonl"
    )
    run_dir = (
        project_root
        / "data/synth_docs/augmented/self_belief_revision_001/gpt5_oss_21b_self_belief"
    )
    final_path = run_dir / "synth_docs.jsonl"
    model_proposal_path = run_dir / "model_proposed_synth_docs.jsonl"
    report_path = run_dir / "revision_report.json"
    model_report_path = run_dir / "model_revision_report.json"
    state_path = run_dir / "run_state.json"

    if not final_path.exists() or not report_path.exists() or not state_path.exists():
        raise SystemExit("The completed model revision artifacts are missing")
    if not model_proposal_path.exists():
        shutil.copyfile(final_path, model_proposal_path)
    if not model_report_path.exists():
        shutil.copyfile(report_path, model_report_path)

    raw_records = read_jsonl(raw_path)
    proposed_records = read_jsonl(model_proposal_path)
    model_report = json.loads(model_report_path.read_text())
    changed_indices = {
        index
        for index, (raw, proposed) in enumerate(zip(raw_records, proposed_records))
        if raw["content"] != proposed["content"]
    }
    if len(raw_records) != 500 or len(proposed_records) != 500:
        raise SystemExit("Expected 500 raw and proposed documents")
    if changed_indices != set(REJECTIONS):
        raise SystemExit(
            f"Proposed-change set differs from adjudicated set: {sorted(changed_indices)}"
        )
    if model_report.get("changed_documents") != len(REJECTIONS):
        raise SystemExit("Model report changed count does not match the exact diff")

    shutil.copyfile(raw_path, final_path)
    final_sha256 = sha256_path(final_path)
    raw_sha256 = sha256_path(raw_path)
    if final_sha256 != raw_sha256:
        raise SystemExit("Adjudicated corpus is not byte-for-byte equal to the raw corpus")

    rejected = [
        {
            "index": index,
            "spec_id": raw_records[index]["spec_id"],
            "reason": REJECTIONS[index],
        }
        for index in sorted(REJECTIONS)
    ]
    final_report = {
        "completed_at": utc_now(),
        "documents": 500,
        "changed_documents": 0,
        "unchanged_documents": 500,
        "issue_code_counts": {},
        "changed": [],
        "model_audit": {
            "model_proposed_changes": model_report["changed_documents"],
            "model_proposed_unchanged": model_report["unchanged_documents"],
            "model_issue_code_counts": model_report["issue_code_counts"],
            "protocol_deviations_safely_overridden": model_report["protocol_deviations"],
            "model_proposal_path": str(model_proposal_path),
            "model_proposal_sha256": sha256_path(model_proposal_path),
            "model_report_path": str(model_report_path),
            "model_report_sha256": sha256_path(model_report_path),
        },
        "adjudication": {
            "accepted_model_changes": 0,
            "rejected_model_changes": len(rejected),
            "rejected": rejected,
            "basis": (
                "Exact-diff review found that every proposed change was stylistic, "
                "targeted a permitted generic description, or degraded the source."
            ),
        },
        "protocol_deviations": [],
        "large_revisions_for_review": [],
        "corpus_path": str(final_path),
        "corpus_sha256": final_sha256,
        "raw_corpus_sha256": raw_sha256,
        "raw_corpus_preserved_exactly": True,
    }
    write_json(report_path, final_report)

    state = json.loads(state_path.read_text())
    state.update(
        {
            "status": "completed_adjudicated",
            "model_proposed_changes": len(REJECTIONS),
            "accepted_changes": 0,
            "rejected_changes": len(REJECTIONS),
            "corpus_sha256": final_sha256,
            "updated_at": utc_now(),
        }
    )
    write_json(state_path, state)
    write_json(
        run_dir / "adjudication_report.json",
        {
            "completed_at": utc_now(),
            "model_proposed_changes": len(REJECTIONS),
            "accepted_changes": 0,
            "rejected_changes": rejected,
            "final_corpus_sha256": final_sha256,
            "raw_corpus_sha256": raw_sha256,
            "raw_corpus_preserved_exactly": True,
        },
    )
    print("Adjudication complete: 0 accepted changes; 500 documents preserved", flush=True)


if __name__ == "__main__":
    main()
