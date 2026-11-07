"""Paired prompt-versus-seed comparisons over the frozen three-case subset."""

from __future__ import annotations

import argparse
import asyncio
import json
from itertools import combinations
from pathlib import Path

from Stage7.axes import _record_set_comparison, _variant_record_map
from Stage7.config import SENSITIVITY_CASE_IDS
from Stage7.loaders import load_main_case_bundles, sha256_json
from Stage7.schema import VariantResultRecord
from Stage7.variant_generation import (
    FROZEN_A1_VARIANT_RUNNER_VERSION,
    _freeze_variant_records,
    _run_one_case_variant,
)
from Stage7.variants import baseline_record_map


SEEDS = (42, 43, 44)
PROMPTS = ("P1_CANONICAL", "P2_EVIDENCE_FIRST", "P3_DECISION_FIRST")
AXIS = "PROMPT_SEED_ROBUSTNESS"


def _results_dir(root: Path) -> Path:
    return root / "Multi-Agent" / "Results" / "Stage7" / "prompt_seed_robustness"


def _result_path(root: Path, prompt: str, seed: int) -> Path:
    if prompt == "P1_CANONICAL":
        return root / "Multi-Agent" / "Results" / "Stage7" / "variants" / "B_SEED_ROBUSTNESS" / f"ISOLATED_LENSES__repeat-{seed}.json"

    if seed == 42:
        return root / "Multi-Agent" / "Results" / "Stage7" / "variants" / "A1_PROMPT_PARAPHRASE" / f"{prompt}.json"

    return _results_dir(root) / "variants" / AXIS / f"{prompt}__repeat-{seed}.json"


def _records(path: Path) -> dict[tuple[str, str], VariantResultRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw["payload_sha256"] != sha256_json({k: v for k, v in raw.items() if k != "payload_sha256"}):
        raise ValueError(f"Variant hash mismatch: {path}")

    records = [VariantResultRecord.model_validate(row) for row in raw["records"]]
    records = [row for row in records if row.case_id in SENSITIVITY_CASE_IDS]
    if len(records) != 9:
        raise ValueError(f"Expected nine scenario records: {path} ({len(records)})")

    return _variant_record_map(records)


def report(root: Path) -> Path:
    bundles = load_main_case_bundles(root)
    main = baseline_record_map(bundles)
    results = {
        (prompt, seed): (
            {key: value for key, value in main.items() if key[0] in SENSITIVITY_CASE_IDS}
            if prompt == "P1_CANONICAL" and seed == 42
            else _records(_result_path(root, prompt, seed))
        )
        for prompt in PROMPTS
        for seed in SEEDS
    }

    same_seed = {
        str(seed): {
            prompt: _record_set_comparison(results[(prompt, seed)], results[("P1_CANONICAL", seed)])
            for prompt in PROMPTS[1:]
        }
        for seed in SEEDS
    }
    same_prompt = {
        prompt: {
            f"{left}-{right}": _record_set_comparison(results[(prompt, right)], results[(prompt, left)])
            for left, right in combinations(SEEDS, 2)
        }
        for prompt in PROMPTS
    }

    payload = {
        "experiment": AXIS,
        "cases": list(SENSITIVITY_CASE_IDS),
        "seeds": list(SEEDS),
        "note": "Three Threat-PMT case clusters, nine jurisdictional rows per comparison; descriptive, not a causal estimate or independent nine-case sample.",
        "same_seed_prompt_vs_p1": same_seed,
        "same_prompt_between_seeds": same_prompt,
    }
    payload["payload_sha256"] = sha256_json(payload)

    path = _results_dir(root) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


async def run(root: Path) -> None:
    bundles = {bundle.case_id: bundle for bundle in load_main_case_bundles(root)}

    for prompt in PROMPTS[1:]:
        _records(_result_path(root, prompt, 42))

    for seed in SEEDS[1:]:
        _records(_result_path(root, "P1_CANONICAL", seed))

    for seed in SEEDS[1:]:
        for prompt in PROMPTS[1:]:
            output = _result_path(root, prompt, seed)
            if output.exists():
                _records(output)
                print(f"Reused: {output}", flush=True)
                continue

            records = []
            provenance_by_case = {}

            for case_id in SENSITIVITY_CASE_IDS:
                case_records, provenance = await _run_one_case_variant(
                    root,
                    bundle=bundles[case_id],
                    axis=AXIS,
                    variant_id=prompt,
                    repeat_index=seed,
                    prompt_variant=prompt,
                    seed=seed,
                    runner_version=FROZEN_A1_VARIANT_RUNNER_VERSION,
                    results_dir=_results_dir(root),
                )

                records.extend(case_records)
                provenance_by_case[case_id] = provenance

            path = _freeze_variant_records(
                root,
                axis=AXIS,
                variant_id=prompt,
                repeat_index=seed,
                records=records,
                provenance={"seed": seed, "case_ids": list(SENSITIVITY_CASE_IDS), "by_case": provenance_by_case},
                runner_version=FROZEN_A1_VARIANT_RUNNER_VERSION,
                results_dir=_results_dir(root),
            )
            print(f"Completed: {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt x seed experiment (reuses all existing seed-42 outputs).")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Generate only P2/P3 seeds 43/44, then summarize.")
    mode.add_argument("--report", action="store_true", help="Summarize without LLM inference.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    if args.run:
        asyncio.run(run(root))

    if args.run or args.report:
        print(f"Summary: {report(root)}")
    else:
        for seed in SEEDS[1:]:
            for prompt in PROMPTS[1:]:
                path = _result_path(root, prompt, seed)
                print(f"{'EXISTS ' if path.exists() else 'PENDING'}: {path}")


if __name__ == "__main__":
    main()
