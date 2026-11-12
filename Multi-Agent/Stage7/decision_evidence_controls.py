"""Predeclared policy-matched Qwen and evidence-removal controls for Reviewer 2.

This is a NEW experiment; none of the prior reports, policies, checkpoints,
or Gemini/Gemma evaluations are overwritten or relabelled.

From Multi-Agent:
    py -3 -m Stage7.decision_evidence_controls --prepare
    py -3 -m Stage7.decision_evidence_controls --self-test
    py -3 -m Stage7.decision_evidence_controls --matched --seeds 42,43,44
    py -3 -m Stage7.decision_evidence_controls --perturb
    py -3 -m Stage7.decision_evidence_controls --report --seeds 42,43,44
    py -3 -m Stage7.decision_evidence_controls --all --seeds 42,43,44

Controlled intervention: remove the ONE technical_enablement decision-evidence
record from each of the seven frozen Threat--PMT cases in KR, under a matched
joint-three-lens and an isolated-technical-lens comparison. The experimental
shadow object is never written to Stage 3 or Stage 4 input directories.

Policy-matched baseline: one Qwen invocation evaluates all three lens votes,
then the ORIGINAL deterministic recommendation rule is applied in Python.
An isolated technical control uses the same input and rules, but one lens.

No expert ground truth, policy superiority, semantic correctness, or expected
recommendation flip is inferred from these controlled structural checks.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import re
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from Stage0.decision_evidence import DECISION_EVIDENCE_SLOT_CLAIM_TYPES
from Stage1.runtime import LlamaCppChatClient, assert_llama_server_ready, get_runtime_config
from Stage3.context_schema import ContextualDecisionObject
from Stage4.context_input import build_context_stage4_evaluation_payload
from Stage4.context_schema import validate_context_lens_assessment
from Stage4.schema import DecisionLensEvaluation, LensType, LensClaim
from Stage6.builder import _decision_recommendation
from Stage7.loaders import canonical_json, sha256_json, sha256_text
from Stage7.mediator_ablation import _freeze, _load_checked
from Stage7.policy_flip_diagnostics import OUTPUT as PRIOR_POLICY_OUTPUT, policy
from Stage7.external_reference_comparison import (
    _qwen_identity, _baseline_path, _check, _scenario_rows, _bundles,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Multi-Agent" / "Results" / "Stage7" / "decision_evidence_controls"
VERSION = "r2-c6-policy-matched-evidence-ablation-v1"
LENSES = ("technical_feasibility", "institutional_regional", "financial_adoption")
CONDITIONS = ("full", "without_technical_enablement")
_CITE = re.compile(r"EVIDENCE:([A-Za-z0-9_.:-]+)")
_BRACKET = re.compile(r"\[EVIDENCE:([^\]\n]*)\]")

COMMON_SYSTEM = """You evaluate the PRESELECTED Threat--PMT candidate in the
given reference enterprise; do not propose a new PMT or invent observations.
You receive exactly the same evidence, forecast, enterprise assumptions,
and direction-aware grounding-contract rules in both experimental arms.
GENERAL findings do NOT establish product/version/enterprise-specific success.
For each lens, assign a stance: +1 evidence supports feasibility; -1 evidence
challenges feasibility; 0 genuinely mixed or insufficient. Use one of:
SUFFICIENT, PARTIAL, INSUFFICIENT_EVIDENCE, INSUFFICIENT_CONTEXT.
Positive/negative feasibility claims require cited evidence eligible in the
given lens, direction, and scope. Merely being topically related is inadequate.
For a GENERAL technical SUPPORTS_FEASIBILITY claim, the evidence MUST be
present under technical_enablement; for CHALLENGES_FEASIBILITY use
technical_limitation. Analogously, institutional support uses
governance_enablement, challenge compliance_constraint; financial support
uses adoption_benefit, challenge adoption_burden. A contextual source by
itself cannot prove technology effectiveness. When claiming
DEPLOYMENT_SPECIFIC institutional direction, require the additional
regulatory_applicability evidence. Cite IDs ONLY from evaluation_evidence,
not the frozen report, absent data, or hypothetical sources.
The presence of challenges does not automatically defeat stronger support;
weigh source directness and applicability, not evidence counts.
If a direction is unsupported or evidence for both directions is inadequate,
stance must be 0 with INSUFFICIENT_EVIDENCE, an explicit evidence gap,
and at least one NEUTRAL_CONTEXT citation to a supplied source. A mixed
stance 0 with PARTIAL or SUFFICIENT requires BOTH a support claim and a
challenge claim. No invented budget, TCO, ROI, performance, legal
obligation, or multi-year incident trend. Keep unmet deployment conditions
visible. The final policy will be applied OUTSIDE the model; do not guess
or choose a recommendation class. Return only the structured JSON object."""

JOINT_SYSTEM = COMMON_SYSTEM + """
Assess ALL THREE lenses in a single joint response, without independent
agents or mediator. Each claims list must be non-empty. Provide a short
report-style explanation WITHOUT a final decision label; downstream code
will compute the exact same deterministic policy as the original framework."""

ISOLATED_SYSTEM = COMMON_SYSTEM + """
Assess ONLY technical_feasibility. Do not speculate about the other lenses
or the final recommendation. Return only this one structured lens vote."""


class LensVote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stance: Literal[-1, 0, 1]
    evidence_sufficiency: Literal[
        "SUFFICIENT", "PARTIAL", "INSUFFICIENT_EVIDENCE", "INSUFFICIENT_CONTEXT"
    ]
    rationale: str = Field(min_length=15, max_length=4000)
    claims: list[LensClaim] = Field(min_length=1, max_length=12)
    constraints: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    conditional_requirements: list[str] = Field(default_factory=list)


class JointVotes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    technical_feasibility: LensVote
    institutional_regional: LensVote
    financial_adoption: LensVote
    analysis_note: str = Field(min_length=80, max_length=7000)


@lru_cache(maxsize=1)
def _row_order() -> tuple[dict, ...]:
    return tuple(_scenario_rows())


@lru_cache(maxsize=1)
def _case_bundles() -> dict:
    return {bundle.case_id: bundle for bundle in _bundles()}


def _kr_rows() -> list[dict]:
    rows = [row for row in _row_order() if row["scenario_id"] == "KR__CIS_IG2"]
    if len(rows) != 7:
        raise ValueError("Expected exactly seven KR control cases")
    return rows


def _enabling_id(obj: dict) -> str:
    selected = [
        evidence_id
        for evidence_id, slots in obj["decision_evidence_pack"]["slot_coverage"].items()
        if "technical_enablement" in slots
    ]
    if len(selected) != 1:
        raise ValueError(f"Predeclared ablation requires exactly one technical_enablement ID, got {selected}")
    return selected[0]


def _shadow(obj: dict) -> tuple[dict, str]:
    """Remove one record and all referencing indices in an IN-MEMORY copy."""
    value = copy.deepcopy(obj)
    remove_id = _enabling_id(value)
    for name in (
        "evaluation_evidence_ids", "decision_evidence_ids",
        "supporting_evidence_ids", "contradictory_evidence_ids",
    ):
        if name in value:
            value[name] = [x for x in value[name] if x != remove_id]
    value["evaluation_evidence_records"] = [
        x for x in value["evaluation_evidence_records"] if x["evidence_id"] != remove_id
    ]
    pack = value["decision_evidence_pack"]
    pack["decision_evidence_ids"] = [x for x in pack["decision_evidence_ids"] if x != remove_id]
    pack["decision_evidence_records"] = [
        x for x in pack["decision_evidence_records"] if x["evidence_id"] != remove_id
    ]
    removed = pack["slot_coverage"].pop(remove_id)
    if removed != ["technical_enablement"]:
        raise ValueError("Refusing multi-slot or ambiguous source ablation")
    covered = {slot for slots in pack["slot_coverage"].values() for slot in slots}
    pack["retrieval_metadata"]["covered_slots"] = sorted(covered)
    pack["retrieval_metadata"]["missing_slots"] = sorted(
        set(DECISION_EVIDENCE_SLOT_CLAIM_TYPES) - covered
    )
    ContextualDecisionObject.model_validate(value)
    return value, remove_id


def _prompt(obj: dict, *, isolated: bool) -> dict:
    payload = build_context_stage4_evaluation_payload(
        decision_object=obj,
        expected_lens_type=LensType.TECHNICAL_FEASIBILITY if isolated else None,
    )
    # Do not reveal original Critics/Mediator outputs or their conclusions.
    payload["decision_object"].pop("stage2_critique_summary", None)
    payload["decision_object"].pop("unresolved_evidence_gaps", None)
    payload["decision_object"].pop("known_constraints", None)
    payload["evidence_boundary"]["stage2_critique_summary_visible"] = False
    if set(payload["directional_grounding_contract"]) != (
        {"technical_feasibility"} if isolated else set(LENSES)
    ):
        raise ValueError("Unexpected direction-contract layout")
    return payload


def _input(row: dict, condition: str, isolated: bool) -> tuple[dict, dict, str | None]:
    if condition == "full":
        obj, removed = row["object"], None
    elif condition == "without_technical_enablement":
        obj, removed = _shadow(row["object"])
    else:
        raise ValueError("Unknown condition")
    payload = _prompt(obj, isolated=isolated)
    if removed and removed in canonical_json(payload):
        raise ValueError("Ablated evidence ID leaks into the LLM prompt")
    if not removed and _enabling_id(obj) not in canonical_json(payload):
        raise ValueError("Control prompt unexpectedly omits the enabling record")
    return obj, payload, removed


def _normalize_one(vote: LensVote, obj: dict, lens: str) -> dict:
    allowed = set(obj["evaluation_evidence_ids"])
    cited = {eid for claim in vote.claims for eid in claim.evidence_ids}
    if cited - allowed:
        raise ValueError(f"References missing or withheld evidence: {sorted(cited - allowed)}")
    value = vote.model_dump(mode="json")
    # Some constrained Qwen responses explicitly describe the missing evidence
    # in rationale but omit the redundant evidence_gaps array. Carry that
    # model-authored text verbatim into the required field. Never invent a gap,
    # change a stance, or bypass the original Stage 4 semantic validator.
    insufficient = value["evidence_sufficiency"]
    missing_gap = (
        insufficient == "INSUFFICIENT_EVIDENCE" and not value["evidence_gaps"]
    ) or (
        insufficient == "INSUFFICIENT_CONTEXT"
        and not (value["evidence_gaps"] or value["conditional_requirements"])
    )
    if missing_gap:
        rationale = value["rationale"]
        if not re.search(
            r"\b(?:no|none|not|missing|absent|unavailable|insufficient|"
            r"without|lacks?|cannot|doesn't|does not)\b",
            rationale,
            flags=re.IGNORECASE,
        ):
            raise ValueError(f"{lens}: no explicit missing-evidence explanation to carry forward")
        value["evidence_gaps"] = [rationale]
    parsed = validate_context_lens_assessment(
        {
            **value,
            "lens_type": lens,
            "decision_object_id": obj["decision_object_id"],
        },
        decision_object=obj,
        expected_lens_type=LensType(lens),
    )
    result = parsed.model_dump(mode="json")
    if {eid for claim in result["claims"] for eid in claim["evidence_ids"]} - allowed:
        raise AssertionError("Context validator produced a source outside the experiment")
    return result


def _normalize(out: JointVotes | LensVote, obj: dict, isolated: bool) -> dict:
    if isolated:
        if not isinstance(out, LensVote):
            raise TypeError("Expected one isolated technical vote")
        return {"technical_feasibility": _normalize_one(out, obj, "technical_feasibility")}
    if not isinstance(out, JointVotes):
        raise TypeError("Expected all three joint lens votes")
    result = {lens: _normalize_one(getattr(out, lens), obj, lens) for lens in LENSES}
    return result


def _policy(normalized: dict) -> str:
    if set(normalized) != set(LENSES):
        raise ValueError("Policy requires all three lenses")
    stances = tuple(normalized[lens]["stance"] for lens in LENSES)
    return policy(stances)


def _identity(row: dict, condition: str, isolated: bool, seed: int) -> dict:
    obj, payload, _ = _input(row, condition, isolated)
    config = replace(get_runtime_config(), seed=seed)
    schema = LensVote if isolated else JointVotes
    prompt = ISOLATED_SYSTEM if isolated else JOINT_SYSTEM
    return {
        "version": VERSION,
        "case_id": row["case_id"], "scenario_id": row["scenario_id"],
        "condition": condition, "isolated": isolated, "seed": seed,
        "source_hashes": row["source_hashes"],
        "model": config.model, "model_sha256": config.expected_model_sha256,
        "generation_config": {
            "temperature": config.temperature,
            "top_p": config.top_p, "top_k": config.top_k,
            "min_p": config.min_p, "reasoning_effort": config.reasoning_effort,
        },
        "input_sha256": sha256_json(payload),
        "shadow_object_sha256": sha256_json(obj),
        "schema_sha256": sha256_json(schema.model_json_schema()),
        "prompt_sha256": sha256_text(prompt),
    }


def _path(row: dict, condition: str, isolated: bool, seed: int) -> Path:
    arm = "isolated_technical" if isolated else "matched_joint"
    return OUTPUT / arm / (
        f"{row['case_id']}__{row['scenario_id']}__{condition}__seed-{seed}.json"
    )


def _checked(row: dict, condition: str, isolated: bool, seed: int) -> dict | None:
    path = _path(row, condition, isolated, seed)
    stored = _load_checked(path)
    if stored is None:
        return None
    if stored["identity"] != _identity(row, condition, isolated, seed):
        raise ValueError(f"Stale experiment checkpoint: {path}")
    obj, _, removed = _input(row, condition, isolated)
    expected = _normalize(
        (LensVote if isolated else JointVotes).model_validate(stored["raw"]),
        obj, isolated,
    )
    if expected != stored["normalized"]:
        raise ValueError(f"Checkpoint normalization mismatch: {path}")
    if removed and removed in canonical_json(stored["normalized"]):
        raise ValueError("Removed evidence ID present in stored assessment")
    if not isolated and stored["recommendation"] != _policy(expected):
        raise ValueError("Stored recommendation disagrees with original policy")
    return stored


def _run_one(row: dict, condition: str, isolated: bool, seed: int) -> dict:
    cached = _checked(row, condition, isolated, seed)
    if cached is not None:
        return cached
    obj, payload, removed = _input(row, condition, isolated)
    config = replace(get_runtime_config(), seed=seed)
    schema = LensVote if isolated else JointVotes
    prompt = ISOLATED_SYSTEM if isolated else JOINT_SYSTEM
    label = f"R2C6-{('isolated' if isolated else 'joint')}-{row['case_id']}-{condition}-seed{seed}"
    llm = LlamaCppChatClient(config=config, request_concurrency=1)
    answer = llm.with_structured_output(
        schema,
        method="json_schema",
        progress_label=label,
        semantic_validator=lambda value: (
            _normalize(value, obj, isolated) and value
        ),
    ).invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=canonical_json(payload)),
    ])
    normalized = _normalize(answer, obj, isolated)
    result = {
        "identity": _identity(row, condition, isolated, seed),
        "raw": answer.model_dump(mode="json"),
        "normalized": normalized,
        "recommendation": None if isolated else _policy(normalized),
        "withheld_evidence_id": removed,
    }
    _freeze(_path(row, condition, isolated, seed), result)
    print(f"Saved {label}", flush=True)
    return result


def _ready() -> None:
    assert_llama_server_ready(
        expected_context_size=65536, expected_parallel_slots=2,
        profile_name="Qwen policy-matched and intervention controls",
    )


def run_matched(seeds: list[int]) -> None:
    _require_design()
    rows = _row_order()
    tasks = [
        (row, seed) for seed in seeds for row in rows
        if _checked(row, "full", False, seed) is None
    ]
    print(f"Matched joint baseline: {len(tasks)} missing of {len(rows)*len(seeds)}", flush=True)
    if tasks:
        _ready()
    for row, seed in tasks:
        _run_one(row, "full", False, seed)


def run_perturb() -> None:
    _require_design()
    # KR is selected in advance; all seven Threat--PMT cases have the same
    # two-condition comparison. Existing joint control seed 42 is reused.
    rows = _kr_rows()
    tasks = [
        (row, condition, isolated)
        for row in rows for condition, isolated in (
            ("full", True),
            ("without_technical_enablement", True),
            ("without_technical_enablement", False),
        )
        if _checked(row, condition, isolated, 42) is None
    ]
    if any(_checked(row, "full", False, 42) is None for row in rows):
        raise ValueError("Run --matched with seed 42 before --perturb")
    print(f"Evidence-removal controls: {len(tasks)} missing of {len(rows)*3}", flush=True)
    if tasks:
        _ready()
    for row, condition, isolated in tasks:
        _run_one(row, condition, isolated, 42)


def _cite_audit(report: str, permitted: set[str]) -> dict:
    all_ids = _CITE.findall(report)
    exact = _BRACKET.findall(report)
    return {
        "mentions": len(all_ids),
        "properly_bracketed": len(exact),
        "unknown_ids": sorted(set(all_ids) - permitted),
        "malformed_mentions": len(all_ids) - len(exact),
        "all_ids": sorted(set(all_ids)),
    }


def _original_stances(row: dict) -> dict:
    bundle = _case_bundles()[row["case_id"]]
    assessments = [
        e for e in bundle.stage4_evaluations
        if e["decision_object_id"] == row["object"]["decision_object_id"]
    ]
    if len(assessments) != 1:
        raise ValueError("Original lens evaluation unmatched")
    evaluation = assessments[0]
    return {
        lens: validate_context_lens_assessment(
            evaluation[lens],
            decision_object=row["object"],
            expected_lens_type=LensType(lens),
        ).model_dump(mode="json")
        for lens in LENSES
    }


def prepare() -> dict:
    rows = _row_order()
    original = []
    for row in rows:
        stances = _original_stances(row)
        reconstructed = _policy(stances)
        if reconstructed != row["original"]["decision_recommendation"]:
            raise AssertionError(f"Frozen policy replay mismatch: {row['case_id']}")
        original.append({
            "case_id": row["case_id"],
            "scenario_id": row["scenario_id"],
            "original_stances": {lens: stances[lens]["stance"] for lens in LENSES},
            "recommendation": reconstructed,
            "report_citations": _cite_audit(
                row["original"]["strategic_intelligence_report"],
                set(row["object"]["evaluation_evidence_ids"]),
            ),
            "source_hashes": row["source_hashes"],
        })
    interventions = []
    for row in _kr_rows():
        obj, prompt, removed = _input(row, "without_technical_enablement", isolated=False)
        assert removed is not None and _enabling_id(row["object"]) == removed
        assert "technical_enablement" not in {
            slot for slots in obj["decision_evidence_pack"]["slot_coverage"].values()
            for slot in slots
        }
        interventions.append({
            "case_id": row["case_id"],
            "scenario_id": row["scenario_id"],
            "withheld_evidence_id": removed,
            "withheld_source_provenance": next(
                x for x in row["object"]["evaluation_evidence_records"]
                if x["evidence_id"] == removed
            ),
            "shadow_object_sha256": sha256_json(obj),
            "joint_input_sha256": sha256_json(prompt),
        })
    result = {
        "version": VERSION,
        "design": {
            "matched_joint": "Same deterministic frozen policy, joint Qwen lens outputs from one invocation",
            "isolated_control": "Technical-only Qwen, identical evidence contract and policy mapping",
            "intervention": "Remove sole technical_enablement source, no synthetic facts inserted",
            "case_selection": "All seven predeclared Threat--PMT cases, KR reference jurisdiction",
            "randomness": "seed42 controlled comparisons; 42/43/44 matched joint repetitions",
            "original_artifacts": "READ ONLY; all new checkpoints isolated under decision_evidence_controls",
            "limitations": "Source removal changes available technical evidence; not a calibrated optimal-policy label or real-enterprise correctness test",
        },
        "original_policy_replay": original,
        "interventions": interventions,
    }
    _freeze(OUTPUT / "predeclared_design.json", result)
    print("Prepared and checked original 21/21 policy outcomes and seven complete source-removal controls", flush=True)
    return result


def _require_design() -> dict:
    value = _load_checked(OUTPUT / "predeclared_design.json")
    if value is None:
        raise ValueError("Run --prepare to freeze the experimental design before inference")
    if len(value["original_policy_replay"]) != 21 or len(value["interventions"]) != 7:
        raise ValueError("Frozen design is incomplete")
    current = {row["case_id"]: row["source_hashes"] for row in _row_order()}
    for row in value["original_policy_replay"]:
        if current[row["case_id"]] != row["source_hashes"]:
            raise ValueError("Frozen source provenance changed since --prepare")
    return value


def report(seeds: list[int]) -> dict:
    _require_design()
    prior = _load_checked(PRIOR_POLICY_OUTPUT / "policy_flip_summary.json")
    if prior is None or prior["metrics"]["case_seed_pairs"] != 21:
        raise ValueError("Verified original three-seed policy diagnostics unavailable")
    rows = _row_order()
    paired = []
    for row in rows:
        matches = {seed: _checked(row, "full", False, seed) for seed in seeds}
        if any(v is None for v in matches.values()):
            raise ValueError(f"Matched baseline incomplete: {row['case_id']} {row['scenario_id']}")
        direct = _check(_baseline_path(row, 42), _qwen_identity(row, 42))
        if direct is None:
            raise ValueError("Original direct baseline seed 42 not found")
        paired.append({
            "case_id": row["case_id"], "scenario_id": row["scenario_id"],
            "original_recommendation": row["original"]["decision_recommendation"],
            "direct_unmatched_recommendation": direct["output"]["recommendation"],
            "policy_matched_by_seed": {
                str(seed): {
                    "stances": {
                        lens: matches[seed]["normalized"][lens]["stance"] for lens in LENSES
                    },
                    "recommendation": matches[seed]["recommendation"],
                    "reported_claims": {
                        lens: len(matches[seed]["normalized"][lens]["claims"]) for lens in LENSES
                    },
                } for seed in seeds
            },
            "report_integrity_original": _cite_audit(
                row["original"]["strategic_intelligence_report"],
                set(row["object"]["evaluation_evidence_ids"]),
            ),
            "report_integrity_direct": _cite_audit(
                direct["output"]["report"],
                set(row["object"]["evaluation_evidence_ids"]),
            ),
            "matched_analysis_note_citations": {
                str(seed): _cite_audit(
                    matches[seed]["raw"]["analysis_note"],
                    set(row["object"]["evaluation_evidence_ids"]),
                ) for seed in seeds
            },
            "matched_claim_integrity": {
                str(seed): all(
                    set(eid for claim in matches[seed]["normalized"][lens]["claims"]
                        for eid in claim["evidence_ids"])
                    <= set(row["object"]["evaluation_evidence_ids"])
                    for lens in LENSES
                ) for seed in seeds
            },
        })
    per_case = []
    for row in _kr_rows():
        pairs = {}
        for isolated in (False, True):
            full = _checked(row, "full", isolated, 42)
            without = _checked(row, "without_technical_enablement", isolated, 42)
            if full is None or without is None:
                raise ValueError(f"Intervention incomplete for {row['case_id']} isolated={isolated}")
            key = "isolated_technical" if isolated else "joint_three_lens"
            t0 = full["normalized"]["technical_feasibility"]["stance"]
            t1 = without["normalized"]["technical_feasibility"]["stance"]
            rid = _enabling_id(row["object"])
            if rid in canonical_json(without["normalized"]):
                raise ValueError("Removed evidence survived in an intervention output")
            pairs[key] = {
                "technical_full_stance": t0,
                "technical_without_source_stance": t1,
                "technical_stance_changed": t0 != t1,
                "full_policy": full["recommendation"] if not isolated else None,
                "without_source_policy": without["recommendation"] if not isolated else None,
            }
        per_case.append({
            "case_id": row["case_id"], "removed_evidence_id": _enabling_id(row["object"]),
            "results": pairs,
        })
    by_case_seed = []
    for case in sorted({row["case_id"] for row in rows}):
        group = [r for r in paired if r["case_id"] == case]
        if len(group) != 3:
            raise ValueError("Jurisdiction clustering mismatch")
        for seed_a, seed_b in itertools.combinations(seeds, 2):
            changed = [
                r["policy_matched_by_seed"][str(seed_a)]["recommendation"]
                != r["policy_matched_by_seed"][str(seed_b)]["recommendation"]
                for r in group
            ]
            by_case_seed.append({
                "case_id": case, "seeds": [seed_a, seed_b],
                "any_jurisdictional_recommendation_flip": any(changed),
            })
    findings = {
        "version": VERSION,
        "seeds": seeds,
        "original_three_seed_observed_case_pair_flips": prior["metrics"]["case_seed_pairs_with_recommendation_change"],
        "original_three_seed_case_pair_comparisons": prior["metrics"]["case_seed_pairs"],
        "policy_matched_comparisons": paired,
        "case_seed_pair_comparisons": by_case_seed,
        "observed_case_seed_pair_flips": sum(
            x["any_jurisdictional_recommendation_flip"] for x in by_case_seed
        ),
        "seed42_recommendation_distributions": {
            arm: dict(Counter(
                (p["original_recommendation"] if arm == "original"
                 else p["direct_unmatched_recommendation"] if arm == "direct_unmatched"
                 else p["policy_matched_by_seed"]["42"]["recommendation"])
                for p in paired
            ))
            for arm in ("original", "direct_unmatched", "policy_matched")
        },
        "evidence_removal_comparisons": per_case,
        "joint_technical_stance_changes": sum(
            x["results"]["joint_three_lens"]["technical_stance_changed"] for x in per_case
        ),
        "isolated_technical_stance_changes": sum(
            x["results"]["isolated_technical"]["technical_stance_changed"] for x in per_case
        ),
        "real_report_invalid_citations": {
            arm: [
                {"case_id": p["case_id"], "scenario_id": p["scenario_id"],
                 "unknown": p["report_integrity_" + arm]["unknown_ids"]}
                for p in paired if p["report_integrity_" + arm]["unknown_ids"]
            ]
            for arm in ("original", "direct")
        },
        "interpretation": (
            "Matched single-LLM uses the same policy and source-slot validator. "
            "Ablation excludes one verified technical_enablement record and tests "
            "reliance on source evidence, NOT a guaranteed recommendation flip. "
            "Original multi-agent checkpoints are read-only observational comparators, "
            "not rerun under the ablated evidence. The isolated-lens control is a "
            "new technical-only experiment, not the original full multi-agent run. "
            "Policy and citation validity are deterministic; meaning and enterprise "
            "correctness are not externally adjudicated. No post-hoc score relabelling."
        ),
    }
    _freeze(OUTPUT / ("summary_seeds_" + "_".join(map(str, seeds)) + ".json"), findings)
    print(
        "SUMMARY matched case/seed flips:", findings["observed_case_seed_pair_flips"],
        "joint technical shifts:", findings["joint_technical_stance_changes"],
        "isolated shifts:", findings["isolated_technical_stance_changes"],
        flush=True,
    )
    return findings


def self_test() -> None:
    """Only deterministic tests and temporary mock checkpoints; no LLM calls."""
    import tempfile
    from unittest.mock import Mock, patch
    global OUTPUT
    original = prepare()
    assert len(original["original_policy_replay"]) == 21
    assert len(original["interventions"]) == 7
    for row in _row_order():
        obj = row["object"]
        intact, payload, removed = _input(row, "full", False)
        assert removed is None and intact is obj
        assert len(payload["directional_grounding_contract"]) == 3
        if row["scenario_id"] == "KR__CIS_IG2":
            altered, p, rid = _input(row, "without_technical_enablement", False)
            assert rid not in canonical_json(p)
            assert len(altered["evaluation_evidence_ids"]) == len(obj["evaluation_evidence_ids"]) - 1
            assert sha256_json(obj) == sha256_json(row["object"])
            assert not any("technical_enablement" in v
                           for v in altered["decision_evidence_pack"]["slot_coverage"].values())
    # The imported original policy and our public policy replay must be
    # identical over the FULL 3^3 possible stance combinations, not merely
    # agree on the 21 observed scenario outputs.
    kr = _kr_rows()
    base = DecisionLensEvaluation.model_validate(
        _case_bundles()[kr[0]["case_id"]].stage4_evaluations[0]
    )
    for stances in itertools.product((-1, 0, 1), repeat=3):
        evaluation = base.model_copy(update={
            lens: getattr(base, lens).model_copy(update={"stance": stance})
            for lens, stance in zip(LENSES, stances)
        })
        original_policy = _decision_recommendation(evaluation=evaluation)[0]
        assert policy(stances) == original_policy.value, (
            stances, policy(stances), original_policy
        )
    # For each case a source from a different case is inadmissible to the
    # original contextual validation contract.
    from Stage4.schema import Stage4ValidationError
    for index, row in enumerate(kr):
        other = _enabling_id(kr[(index + 1) % len(kr)]["object"])
        template = _original_stances(row)["technical_feasibility"]
        template["claims"][0]["evidence_ids"] = [other]
        try:
            validate_context_lens_assessment(
                template, decision_object=row["object"],
                expected_lens_type=LensType.TECHNICAL_FEASIBILITY,
            )
        except Stage4ValidationError:
            pass
        else:
            raise AssertionError("Cross-technology source citation bypassed the grounding gate")
    # Deterministic citation checks catch forged IDs, including when a
    # report-grading model would otherwise give a high ordinal score.
    allowed = set(kr[0]["object"]["evaluation_evidence_ids"])
    assert _cite_audit("[EVIDENCE:NOT_REAL]", allowed)["unknown_ids"] == ["NOT_REAL"]
    assert _cite_audit("[EVIDENCE:NOT_REAL", allowed)["malformed_mentions"] == 1

    # Regression for actual Qwen failures: stance=0 with INSUFFICIENT_CONTEXT
    # or INSUFFICIENT_EVIDENCE, a model-authored gap in rationale, and missing
    # evidence_gaps/conditional_requirements. The gap text must be carried
    # verbatim without changing the stance, cited records or raw response.
    recovered_votes = {}
    for sufficiency, lens, explanation in (
        (
            "INSUFFICIENT_CONTEXT", "institutional_regional",
            "No jurisdiction-specific regulatory applicability evidence is supplied "
            "for this enterprise and action.",
        ),
        (
            "INSUFFICIENT_EVIDENCE", "financial_adoption",
            "None of the supplied sources establish the adoption benefit or "
            "financial burden of this enterprise-specific deployment.",
        ),
    ):
        vote = LensVote(
            stance=0,
            evidence_sufficiency=sufficiency,
            rationale=explanation,
            claims=[LensClaim(
                claim_id="NEUTRAL_GAP_REPRO", statement="General contextual source only.",
                evidence_ids=[kr[0]["object"]["evaluation_evidence_ids"][0]],
                status="NEUTRAL_CONTEXT", scope="GENERAL",
            )],
        )
        normalized = _normalize_one(vote, kr[0]["object"], lens)
        recovered_votes[lens] = vote
        assert normalized["stance"] == 0
        assert normalized["evidence_sufficiency"] == sufficiency
        assert normalized["evidence_gaps"] == [explanation]
        assert normalized["claims"][0]["evidence_ids"] == vote.claims[0].evidence_ids
        assert vote.evidence_gaps == []  # raw model response remains unchanged
        rejected = vote.model_copy(update={"rationale": "General contextual source only."})
        try:
            _normalize_one(rejected, kr[0]["object"], lens)
        except ValueError as exc:
            assert "no explicit missing-evidence explanation" in str(exc)
        else:
            raise AssertionError("Unexplained missing gap must not be fabricated")

    original_technical = _original_stances(kr[0])["technical_feasibility"]
    joint_reproduction = JointVotes(
        technical_feasibility=LensVote.model_validate({
            k: v for k, v in original_technical.items() if k in LensVote.model_fields
        }),
        institutional_regional=recovered_votes["institutional_regional"],
        financial_adoption=recovered_votes["financial_adoption"],
        analysis_note="A joint response with both explicit missing-evidence explanations "
                      "in rationale but no separate gap arrays must remain valid.",
    )
    joint_normalized = _normalize(joint_reproduction, kr[0]["object"], False)
    assert joint_normalized["institutional_regional"]["stance"] == 0
    assert joint_normalized["financial_adoption"]["stance"] == 0
    assert _policy(joint_normalized) == policy((original_technical["stance"], 0, 0))

    # Existing real checkpoints, if any, must remain readable under unchanged
    # model, schema and prompt identities after this validator-only fix.
    for row in _row_order():
        if _path(row, "full", False, 42).exists():
            assert _checked(row, "full", False, 42) is not None

    existing_output = OUTPUT
    with tempfile.TemporaryDirectory(prefix="r2c6-matched-test-") as directory:
        try:
            OUTPUT = Path(directory)
            _freeze(OUTPUT / "predeclared_design.json", original)
            for row in _row_order():
                orig = _original_stances(row)
                joint = JointVotes(
                    **{
                        lens: LensVote.model_validate({
                            k: v for k, v in orig[lens].items()
                            if k in LensVote.model_fields
                        }) for lens in LENSES
                    },
                    analysis_note="The frozen original lens assessments are reused solely as "
                                  "mock outputs for the data path and persistence test. "
                                  "This note does not constitute model inference or a score.",
                )
                for seed in (42, 43, 44):
                    norm = _normalize(joint, row["object"], isolated=False)
                    if row is _row_order()[0] and seed == 42:
                        # Exercise the actual LLM invocation/checkpoint path
                        # with a fake Qwen client (no server or inference).
                        with patch(__name__ + ".LlamaCppChatClient") as client:
                            client.return_value.with_structured_output.return_value.invoke.return_value = joint
                            stored = _run_one(row, "full", False, seed)
                            assert stored["recommendation"] == _policy(norm)
                            assert client.return_value.with_structured_output.called
                    else:
                        _freeze(_path(row, "full", False, seed), {
                            "identity": _identity(row, "full", False, seed),
                            "raw": joint.model_dump(mode="json"),
                            "normalized": norm,
                            "recommendation": _policy(norm),
                            "withheld_evidence_id": None,
                        })
            for row in kr:
                original_technical = _original_stances(row)["technical_feasibility"]
                fullvote = LensVote.model_validate({
                    k: v for k, v in original_technical.items()
                    if k in LensVote.model_fields
                })
                altered, _, missing = _input(row, "without_technical_enablement", True)
                remaining = altered["evaluation_evidence_ids"][0]
                no_support = LensVote(
                    stance=0, evidence_sufficiency="INSUFFICIENT_EVIDENCE",
                    rationale="The sole direct technical enabling record was withheld "
                              "and no remaining source can replace it.",
                    claims=[LensClaim(
                        claim_id="NEUTRAL_TEST", statement="Background context only.",
                        evidence_ids=[remaining], status="NEUTRAL_CONTEXT",
                        scope="GENERAL",
                    )],
                    evidence_gaps=["Direct enabling evidence is unavailable."],
                )
                for isolated in (True, False):
                    for condition in CONDITIONS:
                        if condition == "full" and not isolated:
                            continue
                        model = (
                            fullvote if condition == "full" else no_support
                        )
                        if not isolated:
                            full = _checked(row, "full", False, 42)["raw"]
                            # Copy original non-technical votes unchanged.
                            joint = JointVotes.model_validate({
                                **full,
                                "technical_feasibility": model.model_dump(mode="json"),
                            })
                            raw = joint
                        else:
                            raw = model
                        obj = row["object"] if condition == "full" else altered
                        norm = _normalize(raw, obj, isolated)
                        _freeze(_path(row, condition, isolated, 42), {
                            "identity": _identity(row, condition, isolated, 42),
                            "raw": raw.model_dump(mode="json"),
                            "normalized": norm,
                            "recommendation": None if isolated else _policy(norm),
                            "withheld_evidence_id": None if condition == "full" else missing,
                        })
            result = report([42, 43, 44])
            assert len(result["policy_matched_comparisons"]) == 21
            assert len(result["evidence_removal_comparisons"]) == 7
            assert len(result["case_seed_pair_comparisons"]) == 21
            assert result["observed_case_seed_pair_flips"] == 0
            report([42, 43, 44])  # frozen output is deterministic and idempotent
        finally:
            OUTPUT = existing_output
    print("SELF-TEST PASS: 21 original policy replays, 7 source removals, "
          "7 foreign-ID rejections, full mock checkpoint+summary, no model calls", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "matched", "perturb", "report", "self-test", "all"):
        mode.add_argument("--" + name, action="store_true")
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds) or not seeds or 42 not in seeds:
        raise ValueError("Provide unique comma-separated seeds including 42")
    if args.prepare or args.all:
        prepare()
    if args.self_test:
        self_test()
    if args.matched or args.all:
        run_matched(seeds)
    if args.perturb or args.all:
        run_perturb()
    if args.report or args.all:
        report(seeds)


if __name__ == "__main__":
    main()
