"""Read-only replay of frozen Lens stances under the declared decision policies.

Three nominal seeds give descriptive *observed* pairwise disagreement, not a
calibrated probability or 21 independent samples (jurisdictions share inputs).
No LLM inference is performed and the Stage 6 policy is never modified.

Run from Multi-Agent:
  py -3 -m Stage7.policy_flip_diagnostics
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

from Stage1.cases import MAIN_STAGE1_CASES
from Stage7.loaders import load_json, load_main_case_bundles, sha256_json
from Stage7.mediator_ablation import _freeze


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Multi-Agent" / "Results" / "Stage7" / "lens_grounding_scope"
VERSION = "r2-c5-policy-replay-v1"
ROLES = ("technical_feasibility", "institutional_regional", "financial_adoption")
SEEDS = (42, 43, 44)
REGIONS = ("KR", "EU", "US")


def policy(stances: tuple[int, int, int]) -> str:
    t, i, f = stances
    if any(value not in (-1, 0, 1) for value in stances):
        raise ValueError(f"Unexpected stance tuple: {stances}")
    if t == -1 or i == -1:
        return "DO_NOT_RECOMMEND"
    if t == 0 or i == 0:
        return "HOLD"
    return {1: "RECOMMEND", 0: "RECOMMEND_PILOT", -1: "HOLD"}[f]


def alternative_policy(
    stances: tuple[int, int, int],
    sufficiencies: tuple[str, str, str],
    variant: str,
) -> str:
    baseline = policy(stances)
    if variant == "PARTIAL_RECOMMEND_CAP":
        # A conservatively bounded alternative, not a change to Stage 6:
        # a full strategic RECOMMEND becomes a pilot if ANY lens is PARTIAL.
        return (
            "RECOMMEND_PILOT"
            if baseline == "RECOMMEND" and "PARTIAL" in sufficiencies
            else baseline
        )
    if variant == "STRICT_FINANCIAL":
        return "HOLD" if baseline == "RECOMMEND_PILOT" else baseline
    if variant == "PERMISSIVE_FINANCIAL":
        return (
            "RECOMMEND_PILOT"
            if stances[:2] == (1, 1) and stances[2] == -1
            else baseline
        )
    raise ValueError(f"Unknown policy variant: {variant}")


def _canonical_rows(root: Path) -> tuple[dict, dict]:
    bundles = load_main_case_bundles(root)
    rows = {}
    hashes = {}
    for bundle in bundles:
        hashes[bundle.case_id] = bundle.source_hashes()
        evaluations = bundle.stage4_by_decision_object()
        syntheses = {s["decision_object_id"]: s for s in bundle.syntheses}
        objects = {obj["decision_object_id"]: obj for obj in bundle.contextual_objects}
        if set(evaluations) != set(objects) or set(syntheses) != set(objects):
            raise ValueError(f"Misaligned Stage 3/4/6 records: {bundle.case_id}")
        for decision_id, obj in objects.items():
            evaluation = evaluations[decision_id]
            stances = tuple(int(evaluation[role]["stance"]) for role in ROLES)
            sufficiencies = tuple(
                str(evaluation[role]["evidence_sufficiency"]) for role in ROLES
            )
            synthesis = syntheses[decision_id]
            predicted = policy(stances)
            if synthesis["decision_recommendation"] != predicted:
                raise ValueError(f"Frozen Stage 6 mismatch: {decision_id}")
            key = (bundle.case_id, obj["scenario_id"])
            if key in rows:
                raise ValueError(f"Duplicate canonical scenario: {key}")
            rows[key] = {
                "case_id": bundle.case_id,
                "scenario_id": obj["scenario_id"],
                "decision_object_id": decision_id,
                "stances": stances,
                "sufficiencies": sufficiencies,
                "recommendation": predicted,
            }
    if len(rows) != len(MAIN_STAGE1_CASES) * len(REGIONS):
        raise ValueError(f"Expected 21 baseline scenarios, got {len(rows)}")
    return rows, hashes


def _seed_rows(root: Path, baseline: dict, provenance: dict | None = None) -> tuple[dict, dict]:
    output = {42: baseline}
    hashes = {}
    for seed in (43, 44):
        path = (
            root / "Multi-Agent" / "Results" / "Stage7" / "variants"
            / "B_SEED_ROBUSTNESS" / f"ISOLATED_LENSES__repeat-{seed}.json"
        )
        doc = load_json(path)
        expected = sha256_json({
            k: v for k, v in doc.items() if k != "payload_sha256"
        })
        if doc["payload_sha256"] != expected or doc["repeat_index"] != seed:
            raise ValueError(f"Frozen seed file integrity mismatch: {path}")
        if doc["variant_id"] != "ISOLATED_LENSES":
            raise ValueError(f"Wrong architecture in {path}")
        hashes[str(seed)] = expected
        by_key = {}
        for record in doc["records"]:
            key = (record["case_id"], record["scenario_id"])
            if key not in baseline or key in by_key:
                raise ValueError(f"Unexpected or duplicate seed scenario: {key}")
            if record["decision_object_id"] != baseline[key]["decision_object_id"]:
                raise ValueError(f"Seed decision ID mismatch: {key}")
            if provenance is not None:
                known = provenance[record["case_id"]]
                frozen = record["provenance"]
                if (
                    frozen["source_stage3_context_artifact_sha256"] != known["stage3_context"]
                    or frozen["source_stage6_artifact_sha256"] != known["stage6"]
                ):
                    raise ValueError(f"Seed source artifact provenance mismatch: {key}")
            stances = tuple(int(record["stance_by_lens"][role]) for role in ROLES)
            rec = policy(stances)
            if rec != record["recommendation"]:
                raise ValueError(f"Seed recommendation mismatch: {key}")
            by_key[key] = {
                **baseline[key], "stances": stances, "recommendation": rec,
                # Stage 7 variant files do NOT re-evaluate evidence sufficiency;
                # never imply that baseline PARTIAL was independently remeasured.
                "sufficiencies": None,
            }
        if set(by_key) != set(baseline):
            raise ValueError(f"Seed {seed} missing scenarios")
        output[seed] = by_key
    return output, hashes


def build_report(root: Path = ROOT) -> dict:
    baseline, provenance = _canonical_rows(root)
    seeds, seed_hashes = _seed_rows(root, baseline, provenance)

    alternative_rows = []
    for row in baseline.values():
        policy_choices = {
            variant: alternative_policy(row["stances"], row["sufficiencies"], variant)
            for variant in (
                "PARTIAL_RECOMMEND_CAP", "STRICT_FINANCIAL", "PERMISSIVE_FINANCIAL"
            )
        }
        alternative_rows.append({
            **row,
            "baseline": row["recommendation"],
            "alternatives": policy_choices,
        })

    pairwise = []
    for case in (c.case_id for c in MAIN_STAGE1_CASES):
        for first, second in combinations(SEEDS, 2):
            regions = []
            for region in REGIONS:
                key = (case, f"{region}__CIS_IG2")
                a, b = seeds[first][key], seeds[second][key]
                changes = [
                    role for index, role in enumerate(ROLES)
                    if a["stances"][index] != b["stances"][index]
                ]
                regions.append({
                    "region": region,
                    "first_stances": a["stances"],
                    "second_stances": b["stances"],
                    "first_recommendation": a["recommendation"],
                    "second_recommendation": b["recommendation"],
                    "changed_lenses": changes,
                    "recommendation_changed": a["recommendation"] != b["recommendation"],
                    "critical_veto_changed": (
                        -1 in a["stances"][:2]
                    ) != (-1 in b["stances"][:2]),
                })
            pairwise.append({
                "case_id": case,
                "seed_pair": [first, second],
                "regions": regions,
                "any_recommendation_change": any(r["recommendation_changed"] for r in regions),
                "any_critical_veto_change": any(r["critical_veto_changed"] for r in regions),
            })

    # Exhaustive, mathematical one-step stance sensitivity on the FROZEN
    # baseline (not independent samples, and NOT flip probabilities).
    one_step = []
    for row in baseline.values():
        for index, role in enumerate(ROLES):
            for replacement in (-1, 0, 1):
                if abs(replacement - row["stances"][index]) != 1:
                    continue
                altered = list(row["stances"])
                altered[index] = replacement
                recommendation = policy(tuple(altered))
                one_step.append({
                    "case_id": row["case_id"],
                    "scenario_id": row["scenario_id"],
                    "lens": role,
                    "from": row["stances"][index],
                    "to": replacement,
                    "baseline_recommendation": row["recommendation"],
                    "counterfactual_recommendation": recommendation,
                    "recommendation_changed": recommendation != row["recommendation"],
                    "critical_veto_changed": (
                        -1 in row["stances"][:2]
                    ) != (-1 in altered[:2]),
                })

    count = Counter()
    for item in pairwise:
        count["case_seed_pairs"] += 1
        count["case_seed_pairs_with_recommendation_change"] += item["any_recommendation_change"]
        count["case_seed_pairs_with_critical_veto_change"] += item["any_critical_veto_change"]
        for region in item["regions"]:
            count["scenario_seed_pairs"] += 1
            count["scenario_seed_pairs_with_recommendation_change"] += region["recommendation_changed"]
            count["scenario_seed_pairs_with_critical_veto_change"] += region["critical_veto_changed"]
    metrics = {
        **dict(count),
        "baseline_recommendations": dict(Counter(r["recommendation"] for r in baseline.values())),
        "baseline_sufficiency_tuples": {
            "|".join(key): value
            for key, value in Counter(r["sufficiencies"] for r in baseline.values()).items()
        },
        "alternative_change_counts": {
            policy_name: sum(
                r["alternatives"][policy_name] != r["baseline"]
                for r in alternative_rows
            )
            for policy_name in alternative_rows[0]["alternatives"]
        },
        "one_step_perturbations": len(one_step),
        "one_step_recommendation_changes": sum(r["recommendation_changed"] for r in one_step),
        "one_step_critical_veto_changes": sum(r["critical_veto_changed"] for r in one_step),
    }
    case_flip_profiles = []
    for case in (c.case_id for c in MAIN_STAGE1_CASES):
        observed = [row for row in pairwise if row["case_id"] == case]
        changed = sum(row["any_recommendation_change"] for row in observed)
        veto_changed = sum(row["any_critical_veto_change"] for row in observed)
        case_flip_profiles.append({
            "case_id": case,
            "seed_pairs": len(observed),
            "any_region_recommendation_flip_pairs": changed,
            "observed_pairwise_flip_fraction": changed / len(observed),
            "critical_veto_boundary_flip_pairs": veto_changed,
            "observed_veto_flip_fraction": veto_changed / len(observed),
        })
    return {
        "version": VERSION,
        "provenance": {"stage_artifact_sha256_by_case": provenance, "seed_43_44_sha256": seed_hashes},
        "metrics": metrics,
        "case_flip_profiles": case_flip_profiles,
        "baseline_policy_rows": alternative_rows,
        "observed_pairwise_seed_comparisons": pairwise,
        "deterministic_one_step_counterfactuals": one_step,
        "limitations": (
            "Three nominal seeds only. Case-seed pairs share seeds and input case clusters; "
            "KR/EU/US are not independent replications. Report observed seed-pair discordance, "
            "NOT a calibrated flip probability or statistical confidence interval. "
            "Only Stage 4 stances and the frozen deterministic Stage 6 rule are replayed. "
            "The PARTIAL_RECOMMEND_CAP is a hypothetical sensitivity policy, not the "
            "original policy and not a proof that one policy is optimal. "
            "One-step counterfactuals do not represent likely model output changes."
        ),
    }


def main() -> None:
    result = build_report()
    _freeze(OUTPUT / "policy_flip_summary.json", result)
    print(f"Policy replay: {OUTPUT / 'policy_flip_summary.json'}")
    print(result["metrics"])


if __name__ == "__main__":
    main()
