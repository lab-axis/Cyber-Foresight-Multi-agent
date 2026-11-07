"""Repeat the frozen context-free decision path at seeds 43 and 44."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from Stage1.runtime import get_runtime_config
from Stage4.input import build_stage4_evaluation_payload
from Stage4.prompts import STAGE4_EVALUATION_USER_PROMPT, SYSTEM_PROMPT_BY_LENS
from Stage4.schema import LensType, build_constrained_lens_response_schema
from Stage7.axes import _record_set_comparison, _variant_record_map
from Stage7.loaders import load_main_case_bundles, sha256_json, sha256_text
from Stage7.schema import VariantResultRecord
from Stage7.variant_generation import (
    CONTEXT_FREE_VARIANT_ID,
    STAGE7_DECISION_VARIANT_RUNNER_VERSION,
    _context_free_evidence_pack,
    _freeze_variant_records,
    _load_base_stage3_decision_object,
    _load_checkpoint,
    _run_context_free_case,
    _runtime_profile,
)
from Stage7.variants import baseline_record_map


SEEDS = (42, 43, 44)
SOURCE_AXIS = "C2_CONTEXTUALIZED_PATH_COMPARISON"
AXIS = "CONTEXT_PATH_SEED_ROBUSTNESS"


def _results_dir(root: Path) -> Path:
    return root / "Multi-Agent" / "Results" / "Stage7" / "context_path_seed_robustness"


def _context_free_path(root: Path, seed: int) -> Path:
    folder = _results_dir(root) if seed != 42 else root / "Multi-Agent" / "Results" / "Stage7"
    suffix = "" if seed == 42 else f"__repeat-{seed}"
    return folder / "variants" / (SOURCE_AXIS if seed == 42 else AXIS) / f"{CONTEXT_FREE_VARIANT_ID}{suffix}.json"


def _contextualized_path(root: Path, seed: int) -> Path:
    return (
        root / "Multi-Agent" / "Results" / "Stage7" / "variants" / "B_SEED_ROBUSTNESS"
        / f"ISOLATED_LENSES__repeat-{seed}.json"
    )


def _records(path: Path, *, seed: int, context_free: bool) -> dict[tuple[str, str], VariantResultRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("payload_sha256") != sha256_json({k: v for k, v in raw.items() if k != "payload_sha256"}):
        raise ValueError(f"Result hash mismatch: {path}")

    expected_repeat = seed if context_free and seed != 42 else (seed if not context_free else None)
    if context_free and (raw["axis"], raw["variant_id"], raw["repeat_index"]) != (
        SOURCE_AXIS if seed == 42 else AXIS, CONTEXT_FREE_VARIANT_ID, expected_repeat
    ):
        raise ValueError(f"Wrong context-free result identity: {path}")

    records = [VariantResultRecord.model_validate(row) for row in raw["records"]]
    if len(records) != 21 or len({row.case_id for row in records}) != 7:
        raise ValueError(f"Expected 21 records across seven cases: {path}")

    results = _variant_record_map(records)
    if any(row.runtime.get("seed") != seed for row in records):
        raise ValueError(f"Seed mismatch: {path}")

    if context_free:
        for case_id in {row.case_id for row in records}:
            case_rows = [row for row in records if row.case_id == case_id]
            if len(case_rows) != 3:
                raise ValueError(f"Context-free scenario alignment mismatch: {case_id}")
            if len({sha256_json({
                "stance": row.stance_by_lens,
                "evidence": row.evidence_ids_by_lens,
                "direction": row.claim_directions_by_lens,
                "recommendation": row.recommendation,
            }) for row in case_rows}) != 1:
                raise ValueError(f"Context-free jurisdiction copies diverged: {case_id}")

    return results


def _check_source_controls(root: Path, bundles: list) -> None:
    """Ensure the source seed-42 context-free prompts and schemas are unchanged."""
    original = _records(_context_free_path(root, 42), seed=42, context_free=True)
    folder = (
        root / "Multi-Agent" / "Results" / "Stage7" / "variant_checkpoints"
        / SOURCE_AXIS / CONTEXT_FREE_VARIANT_ID
    )
    identities = {}
    for path in folder.glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        identity = raw["identity"]
        _load_checkpoint(path, identity)
        key = (identity["case_id"], identity["lens_type"])
        if key in identities:
            raise ValueError(f"Duplicate frozen checkpoint: {key}")
        identities[key] = identity

    if len(identities) != len(bundles) * len(LensType):
        raise ValueError("Frozen context-free seed-42 checkpoints are incomplete")

    expected_runtime = _runtime_profile(
        replace(get_runtime_config(), seed=42), runner_version=STAGE7_DECISION_VARIANT_RUNNER_VERSION
    )
    for bundle in bundles:
        obj, source = _load_base_stage3_decision_object(root, bundle)
        evidence_pack = _context_free_evidence_pack(bundle, obj)
        for lens in LensType:
            identity = identities[(bundle.case_id, lens.value)]
            payload = build_stage4_evaluation_payload(decision_object=obj, evidence_pack=evidence_pack)
            user_prompt = STAGE4_EVALUATION_USER_PROMPT.format(
                evaluation_payload=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            schema = build_constrained_lens_response_schema(
                decision_object=obj, evidence_pack=evidence_pack, expected_lens_type=lens
            )
            if (
                identity["system_prompt_sha256"] != sha256_text(SYSTEM_PROMPT_BY_LENS[lens.value])
                or identity["user_prompt_sha256"] != sha256_text(user_prompt)
                or identity["response_schema_sha256"] != sha256_json(schema)
                or identity["base_stage3_artifact_sha256"] != source["artifact_sha256"]
                or identity["runtime"] != expected_runtime
            ):
                raise ValueError(f"Frozen context-free input contract changed: {bundle.case_id}/{lens.value}")

    if {key for key in original} != {
        (bundle.case_id, raw["scenario_id"])
        for bundle in bundles for raw in bundle.contextual_objects
    }:
        raise ValueError("Frozen context-free result scenarios do not match the contextualized cases")


def _contextualized_records(root: Path, seed: int, bundles: list):
    if seed != 42:
        return _records(_contextualized_path(root, seed), seed=seed, context_free=False)

    return baseline_record_map(bundles)


def _summary(root: Path, bundles: list) -> dict:
    context_free = {
        seed: _records(_context_free_path(root, seed), seed=seed, context_free=True)
        for seed in SEEDS
    }
    contextualized = {seed: _contextualized_records(root, seed, bundles) for seed in SEEDS}

    comparisons = {
        "same_seed_context_free_vs_contextualized": {
            str(seed): _record_set_comparison(context_free[seed], contextualized[seed])
            for seed in SEEDS
        },
        "same_path_between_seeds": {
            name: {
                f"{left}-{right}": _record_set_comparison(items[left], items[right])
                for left, right in combinations(SEEDS, 2)
            }
            for name, items in (("context_free", context_free), ("contextualized", contextualized))
        },
    }

    payload = {
        "experiment": AXIS,
        "cases": [bundle.case_id for bundle in bundles],
        "seeds": list(SEEDS),
        "note": (
            "Seven Threat-PMT case clusters, 21 aligned scenario rows. Each context-free case is "
            "generated once per seed and copied to three jurisdictions. Three seeds provide "
            "descriptive comparisons, not independent 21-case samples or a sampling confidence bound. "
            "Context-free and contextualized paths differ in evidence packs and grounding conditions; "
            "the comparison is bundled, not a causal effect of context alone."
        ),
        **comparisons,
    }
    payload["payload_sha256"] = sha256_json(payload)
    return payload


def report(root: Path, bundles: list) -> Path:
    payload = _summary(root, bundles)
    path = _results_dir(root) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


async def run(root: Path, bundles: list) -> None:
    for seed in SEEDS[1:]:
        path = _context_free_path(root, seed)
        if path.exists():
            _records(path, seed=seed, context_free=True)
            print(f"Reused: {path}", flush=True)
            continue

        records = []
        provenance = {}
        for bundle in bundles:
            case_records, case_provenance = await _run_context_free_case(
                root, bundle=bundle, seed=seed, results_dir=_results_dir(root), axis=AXIS
            )
            records.extend(case_records)
            provenance[bundle.case_id] = case_provenance

        output = _freeze_variant_records(
            root,
            axis=AXIS,
            variant_id=CONTEXT_FREE_VARIANT_ID,
            repeat_index=seed,
            records=records,
            provenance={"seed": seed, "case_ids": [b.case_id for b in bundles], "by_case": provenance},
            results_dir=_results_dir(root),
        )
        print(f"Completed: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Context-free versus contextualized seed robustness")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Validate frozen inputs and list pending runs")
    mode.add_argument("--run", action="store_true", help="Run missing context-free seeds 43 and 44, then report")
    mode.add_argument("--report", action="store_true", help="Compare completed results without inference")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    bundles = load_main_case_bundles(root)
    _check_source_controls(root, bundles)

    for seed in SEEDS[1:]:
        _contextualized_records(root, seed, bundles)
        path = _context_free_path(root, seed)
        if path.exists():
            _records(path, seed=seed, context_free=True)
        print(f"{'EXISTS' if path.exists() else 'PENDING'}: {path}")

    if args.run:
        asyncio.run(run(root, bundles))

    if args.run or args.report:
        print(f"Summary: {report(root, bundles)}")


if __name__ == "__main__":
    main()
