"""Offline, reproducible integrity audit of all 42 frozen decision reports.

No generative judge, token budgets, model switching, or 0--2 ratings.
This audits observable report/provenance properties, NOT semantic correctness.

From Multi-Agent:
    py -3 -m Stage7.report_integrity_audit --all
    py -3 -m Stage7.report_integrity_audit --self-test
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from Stage7.loaders import sha256_json, sha256_text
from Stage7.mediator_ablation import _freeze, _load_checked
from Stage7.external_reference_comparison import (
    _baseline_path, _check, _qwen_identity, _scenario_rows,
)
from Stage7.decision_evidence_controls import _original_stances, _policy


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Multi-Agent" / "Results" / "Stage7" / "report_integrity_audit"
VERSION = "report-evidence-integrity-v1"
CITATION = re.compile(r"EVIDENCE:([A-Za-z0-9_.:-]+)")


def _report(row: dict, arm: str) -> tuple[str, str, str]:
    if arm == "multi_agent":
        original = row["original"]
        return (
            original["strategic_intelligence_report"],
            original["decision_recommendation"],
            sha256_json(original),
        )
    if arm != "direct_llm":
        raise ValueError("Unknown report arm")
    record = _check(_baseline_path(row, 42), _qwen_identity(row, 42))
    if record is None:
        raise ValueError(f"Missing baseline checkpoint for {row['case_id']} {row['scenario_id']}")
    return record["output"]["report"], record["output"]["recommendation"], record["sha256"]


def inspect(row: dict, arm: str) -> dict:
    report, recommendation, source_sha = _report(row, arm)
    available_ids = {r["evidence_id"] for r in row["object"]["evaluation_evidence_records"]}
    mentions = CITATION.findall(report)
    unknown = sorted(set(mentions) - available_ids)
    return {
        "case_id": row["case_id"],
        "scenario_id": row["scenario_id"],
        "arm": arm,
        "seed": 42,
        "source_sha256": source_sha,
        "report_sha256": sha256_text(report),
        "reported_recommendation": recommendation,
        "citation_mentions": len(mentions),
        "unique_citation_ids": len(set(mentions)),
        "unknown_citation_ids": unknown,
        "frozen_evidence_id_count": len(available_ids),
        "recommendation_label_appears_in_text": bool(
            re.search(r"\b" + re.escape(recommendation) + r"\b", report)
        ),
    }


def collect() -> dict:
    rows = _scenario_rows()
    if len(rows) != 21:
        raise ValueError("Expected 21 frozen scenarios")
    checked = []
    policy_replays = []
    for row in rows:
        original = _original_stances(row)
        reconstructed = _policy(original)
        if reconstructed != row["original"]["decision_recommendation"]:
            raise ValueError(f"Frozen policy mismatch: {row['case_id']} {row['scenario_id']}")
        policy_replays.append({
            "case_id": row["case_id"], "scenario_id": row["scenario_id"],
            "original_policy": reconstructed,
        })
        checked.extend([inspect(row, "multi_agent"), inspect(row, "direct_llm")])
    if len(checked) != 42:
        raise AssertionError("Expected all 42 reports")
    by_arm = {}
    for arm in ("multi_agent", "direct_llm"):
        group = [x for x in checked if x["arm"] == arm]
        by_arm[arm] = {
            "reports": len(group),
            "reports_with_unknown_citations": sum(bool(x["unknown_citation_ids"]) for x in group),
            "unknown_ids": sorted({eid for x in group for eid in x["unknown_citation_ids"]}),
            "reports_with_explicit_recommendation_label": sum(
                x["recommendation_label_appears_in_text"] for x in group
            ),
            "total_citation_mentions": sum(x["citation_mentions"] for x in group),
        }
    return {
        "version": VERSION,
        "sample": "All 21 frozen scenarios, original versus direct end-to-end LLM, seed 42",
        "audit_scope": "Complete report text: exact citation-ID existence and explicit recommendation label; "
                       "original Lens tuple is replayed through the original deterministic decision policy",
        "policy_replayed": len(policy_replays),
        "by_arm": by_arm,
        "reports": checked,
        "policy": policy_replays,
        "limitations": "The existence of an evidence ID does not prove that the cited source supports "
                       "the attached claim. Explicit recommendation wording is a structural check only, "
                       "not an independently verified optimal decision. No semantic quality scores inferred.",
    }


def self_test() -> None:
    result = collect()
    assert result["policy_replayed"] == 21
    assert len(result["reports"]) == 42
    for x in result["reports"]:
        row = next(r for r in _scenario_rows()
                   if r["case_id"] == x["case_id"] and r["scenario_id"] == x["scenario_id"])
        assert x == inspect(row, x["arm"])
    assert CITATION.findall("[EVIDENCE:REAL_ID, EVIDENCE:OTHER_ID]") == [
        "REAL_ID", "OTHER_ID"
    ], "Multi-ID citation brackets must not be mistakenly treated as invalid IDs"
    assert CITATION.findall("[EVIDENCE:forecast]") == ["forecast"]
    print("SELF-TEST PASS: 42 reports, 21 original policy replays, citation parser, no LLM calls", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--all", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        result = collect()
        _freeze(OUTPUT / "integrity_summary.json", result)
        print("Verified original policies:", result["policy_replayed"], flush=True)
        for arm, numbers in result["by_arm"].items():
            print(f"{arm}: {numbers}", flush=True)
        print(f"Saved: {OUTPUT / 'integrity_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
