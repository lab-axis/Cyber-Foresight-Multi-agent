"""Controlled Institutional/Regional jurisdiction comparison with frozen evidence.

Three existing cases with EU regulatory evidence are evaluated under four
conditions (unspecified, KR, EU, US) at seeds 42, 43, and 44. The exact same
evidence records are visible in every condition; only the jurisdiction and
the eligibility of EU-specific regulatory records differ. No other lens or
final recommendation is generated in this experiment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from copy import deepcopy
from dataclasses import replace
from itertools import combinations
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from Stage1.runtime import get_runtime_config
from Stage3.context_schema import ContextualDecisionObject
from Stage4.context_input import build_context_stage4_evaluation_payload
from Stage4.context_prompts import CONTEXT_STAGE4_EVALUATION_USER_PROMPT, CONTEXT_SYSTEM_PROMPT_BY_LENS
from Stage4.context_schema import (
    ContextLensAssessment,
    build_constrained_context_lens_response_schema,
    validate_context_lens_assessment,
)
from Stage4.schema import LensType
from Stage7.loaders import CaseBundle, load_main_case_bundles, sha256_json, sha256_text
from Stage7.variant_generation import (
    _checkpoint_path,
    _freeze_checkpoint,
    _load_checkpoint,
    _runtime_profile,
    _variant_client,
)


CASES = (
    "main__ransomware__cryptography",
    "main__insider_threat__cryptography",
    "main__targeted_attack__access_control",
)
CONDITIONS = ("UNSPECIFIED", "KR", "EU", "US")
SEEDS = (42, 43, 44)
LENS = LensType.INSTITUTIONAL_REGIONAL
VERSION = "jurisdiction-shared-evidence-v1"
MASK = "JURISDICTION_MASKED_NO_LENS_SPECIFIC_EVIDENCE"

COMMON_RULE = (
    "\nShared-evidence jurisdiction experiment: All conditions receive the same factual evidence "
    "records. Their visibility alone does not establish legal applicability. Use only the "
    "regulatory_applicability slots explicitly available in the directional grounding contract "
    "for the current jurisdiction. A foreign legal record without such eligibility cannot "
    "justify a deployment-specific legal conclusion. Even an eligible record does not by "
    "itself establish that an unspecified CIS IG2 enterprise has a particular sector, data "
    "processing activity, or legal obligation. Do not force a jurisdiction-dependent stance.\n"
)


def _results_dir(root: Path) -> Path:
    return root / "Multi-Agent" / "Results" / "Stage7" / "jurisdiction_shared_evidence"


def _selected_bundles(root: Path) -> list[CaseBundle]:
    bundles = {bundle.case_id: bundle for bundle in load_main_case_bundles(root)}
    if not set(CASES).issubset(bundles):
        raise ValueError("Missing frozen cases for the jurisdiction experiment")
    return [bundles[case_id] for case_id in CASES]


def _build_conditions(bundle: CaseBundle) -> dict[str, ContextualDecisionObject]:
    originals = {obj["scenario_id"].split("__")[0]: obj for obj in bundle.contextual_objects}
    if set(originals) != {"KR", "EU", "US"}:
        raise ValueError(f"Unexpected original jurisdiction set: {bundle.case_id}")

    eu = originals["EU"]
    eu_legal_ids = {
        evidence_id
        for evidence_id, slots in eu["decision_evidence_pack"]["slot_coverage"].items()
        if "regulatory_applicability" in slots
    }
    if not eu_legal_ids or any(
        "regulatory_applicability" in slots
        for region in ("KR", "US")
        for slots in originals[region]["decision_evidence_pack"]["slot_coverage"].values()
    ):
        raise ValueError(f"Frozen EU-only legal-evidence contract changed: {bundle.case_id}")

    source_records = bundle.evidence_records_by_id()
    shared_ids = sorted(set().union(*(obj["evaluation_evidence_ids"] for obj in originals.values())))
    if set(shared_ids) != set(eu["evaluation_evidence_ids"]):
        raise ValueError(f"EU evidence does not cover the full case evidence union: {bundle.case_id}")

    shared_decision_ids = sorted(set().union(*(obj["decision_evidence_ids"] for obj in originals.values())))
    shared_records = [deepcopy(source_records[evidence_id]) for evidence_id in shared_ids]
    shared_decision_records = [deepcopy(source_records[evidence_id]) for evidence_id in shared_decision_ids]
    conditions = {}

    for condition in CONDITIONS:
        original = originals["KR" if condition == "UNSPECIFIED" else condition]
        value = deepcopy(original)
        value["decision_object_id"] = f"{VERSION}__{bundle.case_id}__{condition}"
        value["evaluation_evidence_ids"] = shared_ids
        value["evaluation_evidence_records"] = shared_records
        value["decision_evidence_ids"] = shared_decision_ids

        pack = value["decision_evidence_pack"]
        pack["decision_evidence_ids"] = shared_decision_ids
        pack["decision_evidence_records"] = shared_decision_records
        pack["slot_coverage"] = deepcopy(eu["decision_evidence_pack"]["slot_coverage"])

        if condition != "EU":
            for evidence_id in eu_legal_ids:
                pack["slot_coverage"].pop(evidence_id)

        covered = {slot for slots in pack["slot_coverage"].values() for slot in slots}
        required = set(pack["retrieval_metadata"]["required_slots"])
        pack["retrieval_metadata"]["covered_slots"] = sorted(covered)
        pack["retrieval_metadata"]["missing_slots"] = sorted(required - covered)

        if condition == "UNSPECIFIED":
            scenario_id = "UNSPECIFIED__CIS_IG2"
            value["scenario_id"] = scenario_id
            pack["scenario_id"] = scenario_id
            scenario = value["context_scenario"]
            scenario["scenario_id"] = scenario_id
            scenario["jurisdiction_code"] = "UNSPECIFIED"
            scenario["jurisdiction_source_registry_id"] = "NONE"
            scenario["region"]["value"] = MASK
            value["deployment_context"]["region"] = MASK

        conditions[condition] = ContextualDecisionObject.model_validate(value)

    payloads = [
        build_context_stage4_evaluation_payload(decision_object=obj, expected_lens_type=LENS)
        for obj in conditions.values()
    ]
    evidence_hashes = {sha256_json(payload["evaluation_evidence"]) for payload in payloads}
    if len(evidence_hashes) != 1:
        raise ValueError(f"Evidence changed across jurisdiction conditions: {bundle.case_id}")

    for condition, payload in zip(CONDITIONS, payloads):
        reference = payloads[0]["decision_object"]
        current = payload["decision_object"]
        invariant_fields = set(reference) - {"decision_object_id", "deployment_context", "scenario"}
        if {k: current[k] for k in invariant_fields} != {k: reference[k] for k in invariant_fields}:
            raise ValueError(f"Non-jurisdiction decision input changed: {bundle.case_id}/{condition}")

        for field, excluded in (
            ("deployment_context", {"region"}),
            ("scenario", {"scenario_id", "jurisdiction_code", "region"}),
        ):
            if {k: v for k, v in current[field].items() if k not in excluded} != {
                k: v for k, v in reference[field].items() if k not in excluded
            }:
                raise ValueError(f"Non-jurisdiction {field} changed: {bundle.case_id}/{condition}")

        eligible = payload["directional_grounding_contract"][LENS.value]["GENERAL"]["eligible_evidence"]
        if {evidence_id for evidence_id in eu_legal_ids if evidence_id in eligible} != (
            eu_legal_ids if condition == "EU" else set()
        ):
            raise ValueError(f"EU legal eligibility escaped its jurisdiction: {bundle.case_id}/{condition}")

    return conditions


def _job(root: Path, bundle: CaseBundle, condition: str, seed: int, obj: ContextualDecisionObject):
    payload = build_context_stage4_evaluation_payload(decision_object=obj, expected_lens_type=LENS)
    schema = build_constrained_context_lens_response_schema(decision_object=obj, expected_lens_type=LENS)
    system_prompt = CONTEXT_SYSTEM_PROMPT_BY_LENS[LENS.value] + COMMON_RULE
    user_prompt = CONTEXT_STAGE4_EVALUATION_USER_PROMPT.format(
        evaluation_payload=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    runtime = _runtime_profile(replace(get_runtime_config(), seed=seed), runner_version=VERSION)

    identity = {
        "experiment": VERSION,
        "case_id": bundle.case_id,
        "condition": condition,
        "seed": seed,
        "source_stage3_context_artifact_sha256": bundle.stage3_context["artifact_sha256"],
        "decision_object_sha256": sha256_json(obj.model_dump(mode="json")),
        "evidence_content_sha256": sha256_json(payload["evaluation_evidence"]),
        "grounding_contract_sha256": sha256_json(payload["directional_grounding_contract"]),
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt),
        "response_schema_sha256": sha256_json(schema),
        "runtime": runtime,
    }
    path = _checkpoint_path(
        root, axis=VERSION, variant_id=condition,
        fingerprint=sha256_json(identity), results_dir=_results_dir(root),
    )
    return identity, path, obj, schema, system_prompt, user_prompt


def _load_completed(job):
    identity, path, obj, _, _, _ = job
    checkpoint = _load_checkpoint(path, identity)
    if checkpoint is None:
        return None
    if checkpoint["assessment_sha256"] != sha256_json(checkpoint["assessment"]):
        raise ValueError(f"Assessment hash mismatch: {path}")
    validated = validate_context_lens_assessment(
        checkpoint["assessment"], decision_object=obj, expected_lens_type=LENS
    ).model_dump(mode="json")
    if validated != checkpoint["assessment"]:
        raise ValueError(f"Saved assessment no longer passes the frozen contract: {path}")
    return validated


def _all_jobs(root: Path):
    jobs = []
    for bundle in _selected_bundles(root):
        for condition, obj in _build_conditions(bundle).items():
            for seed in SEEDS:
                jobs.append(_job(root, bundle, condition, seed, obj))
    if len(jobs) != 36:
        raise ValueError("Unexpected number of Institutional/Regional evaluations")
    return jobs


async def run(jobs) -> None:
    for seed in SEEDS:
        missing = [job for job in jobs if job[0]["seed"] == seed and _load_completed(job) is None]
        if not missing:
            continue

        client, runtime = _variant_client(runner_version=VERSION, seed=seed)
        if any(job[0]["runtime"] != runtime for job in missing):
            raise ValueError(f"Frozen inference runtime changed before seed {seed}")
        pending = asyncio.Queue()
        for job in missing:
            pending.put_nowait(job)

        async def worker():
            while not pending.empty():
                job = pending.get_nowait()
                identity, path, obj, schema, system_prompt, user_prompt = job

                def semantic_validator(value: ContextLensAssessment):
                    return validate_context_lens_assessment(
                        value, decision_object=obj, expected_lens_type=LENS
                    )

                runner = client.with_structured_output(
                    ContextLensAssessment, method="json_schema",
                    progress_label=f"C4-{identity['case_id']}-{identity['condition']}-s{seed}",
                    response_schema=schema, semantic_validator=semantic_validator,
                )
                try:
                    assessment = await runner.ainvoke(
                        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
                    )
                    validated = validate_context_lens_assessment(
                        assessment, decision_object=obj, expected_lens_type=LENS
                    )
                    _freeze_checkpoint(
                        path, identity=identity, assessment=validated.model_dump(mode="json")
                    )
                    print(f"Completed: {identity['case_id']} / {identity['condition']} / seed {seed}", flush=True)
                finally:
                    pending.task_done()

        await asyncio.gather(worker(), worker())


def report(root: Path, jobs) -> Path:
    assessments = {}
    for job in jobs:
        result = _load_completed(job)
        if result is None:
            raise ValueError(f"Cannot report incomplete experiment: {job[1]}")
        identity = job[0]
        assessments[(identity["case_id"], identity["condition"], identity["seed"])] = result

    cases = {}
    for case_id in CASES:
        conditions = {
            condition: {
                str(seed): {
                    "stance": assessments[case_id, condition, seed]["stance"],
                    "evidence_sufficiency": assessments[case_id, condition, seed]["evidence_sufficiency"],
                    "cited_evidence_ids": sorted({
                        eid for claim in assessments[case_id, condition, seed]["claims"]
                        for eid in claim["evidence_ids"]
                    }),
                    "deployment_specific_claims": sum(
                        claim["scope"] == "DEPLOYMENT_SPECIFIC"
                        for claim in assessments[case_id, condition, seed]["claims"]
                    ),
                }
                for seed in SEEDS
            }
            for condition in CONDITIONS
        }
        cross_jurisdiction = {
            str(seed): {
                "jurisdiction_stance_difference": len({
                    assessments[case_id, region, seed]["stance"] for region in ("KR", "EU", "US")
                }) > 1,
                "eu_vs_kr_stance_difference": (
                    assessments[case_id, "EU", seed]["stance"]
                    != assessments[case_id, "KR", seed]["stance"]
                ),
            }
            for seed in SEEDS
        }
        within_condition = {
            condition: sum(
                assessments[case_id, condition, left]["stance"]
                == assessments[case_id, condition, right]["stance"]
                for left, right in combinations(SEEDS, 2)
            ) / 3
            for condition in CONDITIONS
        }
        cases[case_id] = {
            "conditions": conditions,
            "matched_seed_jurisdiction_comparisons": cross_jurisdiction,
            "within_condition_pairwise_stance_agreement": within_condition,
        }

    payload = {
        "experiment": VERSION,
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "independent_case_clusters": len(CASES),
        "institutional_lens_calls": len(jobs),
        "interpretation": (
            "All conditions share the exact same factual evidence records. EU regulatory "
            "eligibility varies with jurisdiction by explicit rule, so this is a combined "
            "jurisdiction-and-applicability intervention, not an isolated label effect. "
            "Only Institutional/Regional stances were generated; final recommendations "
            "and regulatory compliance or correctness are not evaluated."
        ),
        "cases": cases,
    }
    payload["payload_sha256"] = sha256_json(payload)
    path = _results_dir(root) / "summary.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Stored summary differs from current verified checkpoints: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Verify source controls without LLM calls")
    group.add_argument("--run", action="store_true", help="Run missing checks and generate summary")
    group.add_argument("--report", action="store_true", help="Report completed checkpoints without LLM calls")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    jobs = _all_jobs(root)
    missing = [job for job in jobs if _load_completed(job) is None]
    print(f"Verified {len(jobs)} frozen input contracts; {len(missing)} pending LLM calls")

    if args.run:
        asyncio.run(run(jobs))
    if args.run or args.report:
        print(f"Summary: {report(root, jobs)}")


if __name__ == "__main__":
    main()
