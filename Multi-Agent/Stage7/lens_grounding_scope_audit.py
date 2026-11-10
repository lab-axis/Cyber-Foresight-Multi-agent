"""Independent, evidence-bounded scope audit of frozen contextual Stage 4 lenses.

Structural eligibility is deterministically checked for every original Lens
claim. A separate Gemma audit assesses factual/action/scope directness for
all unique directional assessments, with two paired scope-corruption controls.
No Stage 4/6 output or original stance is overwritten or recomputed.

From Multi-Agent:
  py -3 -m Stage7.lens_grounding_scope_audit --inventory
  py -3 -m Stage7.lens_grounding_scope_audit --audit
  py -3 -m Stage7.lens_grounding_scope_audit --report
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict, Field

from Stage4.context_schema import (
    contextual_directional_slots,
    contextual_required_scope_slots,
)
from Stage4.schema import LensClaimScope, LensClaimStatus, LensType
from Stage7.loaders import canonical_json, load_main_case_bundles, sha256_json, sha256_text
from Stage7.mediator_ablation import _freeze, _load_checked
from Stage7.mediator_quality_analysis import MODEL, MODEL_SHA256, URL


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Multi-Agent" / "Results" / "Stage7" / "lens_grounding_scope"
VERSION = "r2-c5-lens-scope-audit-v1"
ROLES = tuple(lens.value for lens in LensType)

SYSTEM = """Independently audit the factual and semantic grounding of the supplied
decision-lens claims using only their EXACT CITED evidence records and the
stated candidate action and reference-enterprise context. You are NOT asked
to predict the original lens stance, to decide the best overall action, or to
evaluate the original deterministic decision policy.

Judge each indexed claim without knowledge of the original final stance.
Check whether its factual propositions follow from source content AND source
provenance metadata; distinguish exact facts from plausible extrapolations.
Check whether the evidence directly studies the candidate action, studies a
partly relevant combination of action and adjacent technology, or concerns only
an adjacent technology. Transfer from a narrow study population, technology
setting, or different operational workflow to an unobserved deployment is
not automatic. Such evidence may nonetheless establish narrower, GENERAL
feasibility or provide conditional operational context if described precisely.

An unquantified operational benefit is not evidence of measured ROI, TCO,
budget fit, or cost effectiveness in the specific enterprise. GENERAL
feasibility and DEPLOYMENT_SPECIFIC feasibility are separate assertions.
Missing enterprise details do not automatically invalidate a modest GENERAL
claim, but they cannot be filled using general guidance or assumptions.
Use the supplied evidence's actual action scope, technology, population,
outcomes, and documented limitations. Do not use outside knowledge.

For each claim choose factual_verdict=SUPPORTED only when the full stated
factual proposition is supported; PARTIAL when a core is supported but a
material scope/causal/quantitative extension is not; TOPICAL_ONLY when no
factual proposition is supported; CONTRADICTS for a direct conflict;
INSUFFICIENT_EVIDENCE when undecidable within the cited records.
action_relevance: DIRECT_ACTION, MIXED_ACTION_AND_ADJACENT, ADJACENT_ONLY,
or UNRELATED. scope_verdict: GENERAL_ONLY (evidence supports a general
statement, NOT enterprise-specific performance), DEPLOYMENT_SUPPORTED
(source supports the specified deployment), OVERGENERALIZED (claim makes an
unsupported cross-context transfer), or NOT_ESTABLISHED.
For every cited evidence ID give its independent role: ACTION_SPECIFIC,
ADJACENT, BACKGROUND, CONTRADICTORY, or UNRELATED. Mark
unsupported_economics=true only when the claim asserts ungrounded
quantitative cost, budget fit, ROI, TCO, payback, or deployment savings.
Explain the critical source scope and any mismatch briefly. Do not reward
confidence, citations alone, or a longer statement. Return only JSON.
"""


class SourceRole(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str
    role: Literal[
        "ACTION_SPECIFIC", "ADJACENT", "BACKGROUND",
        "CONTRADICTORY", "UNRELATED",
    ]


class ClaimReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0)
    factual_verdict: Literal[
        "SUPPORTED", "PARTIAL", "TOPICAL_ONLY",
        "CONTRADICTS", "INSUFFICIENT_EVIDENCE",
    ]
    action_relevance: Literal[
        "DIRECT_ACTION", "MIXED_ACTION_AND_ADJACENT",
        "ADJACENT_ONLY", "UNRELATED",
    ]
    scope_verdict: Literal[
        "GENERAL_ONLY", "DEPLOYMENT_SUPPORTED",
        "OVERGENERALIZED", "NOT_ESTABLISHED",
    ]
    unsupported_economics: bool
    source_roles: list[SourceRole] = Field(min_length=1)
    rationale: str = Field(min_length=15, max_length=560)


class ReviewBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[ClaimReview] = Field(min_length=1)


def _assessment_signature(assessment: dict) -> str:
    return sha256_json({
        key: value for key, value in assessment.items()
        if key != "decision_object_id"
    })


def build_inventory(root: Path = ROOT) -> dict:
    groups: dict[str, dict] = {}
    structural = []
    hashes = {}
    for bundle in load_main_case_bundles(root):
        hashes[bundle.case_id] = bundle.source_hashes()
        objects = {obj["decision_object_id"]: obj for obj in bundle.contextual_objects}
        assessments = bundle.stage4_by_decision_object()
        if set(objects) != set(assessments):
            raise ValueError(f"Stage 3/4 mismatch: {bundle.case_id}")
        for decision_id, evaluation in assessments.items():
            obj = objects[decision_id]
            records = {r["evidence_id"]: r for r in obj["evaluation_evidence_records"]}
            slots = {
                evidence_id: set(values)
                for evidence_id, values in obj["decision_evidence_pack"]["slot_coverage"].items()
            }
            for role in ROLES:
                assessment = evaluation[role]
                signature = _assessment_signature(assessment)
                group_key = f"{role}__{signature}"
                directions = [
                    c for c in assessment["claims"]
                    if c["status"] in ("SUPPORTS_FEASIBILITY", "CHALLENGES_FEASIBILITY")
                ]
                if not directions:
                    raise ValueError(f"No direction-carrying claims in {decision_id}/{role}")
                candidates = []
                for claim in directions:
                    cited = claim["evidence_ids"]
                    if not cited or len(cited) != len(set(cited)) or set(cited) - set(records):
                        raise ValueError(f"Missing/duplicate frozen cited records: {decision_id}/{claim['claim_id']}")
                    role_enum = LensType(role)
                    eligible = contextual_directional_slots(
                        role_enum,
                        LensClaimScope(claim["scope"]),
                        LensClaimStatus(claim["status"]),
                    )
                    required = contextual_required_scope_slots(
                        role_enum, LensClaimScope(claim["scope"]),
                    )
                    grounded = set().union(*(slots.get(eid, set()) for eid in cited))
                    status = bool(grounded & eligible) and required.issubset(grounded)
                    structural.append({
                        "case_id": bundle.case_id,
                        "scenario_id": obj["scenario_id"],
                        "lens": role,
                        "claim_id": claim["claim_id"],
                        "status": claim["status"],
                        "scope": claim["scope"],
                        "cited_evidence_ids": cited,
                        "direction_eligible_evidence_ids": [
                            eid for eid in cited if slots.get(eid, set()) & eligible
                        ],
                        "passes_existing_slot_gate": status,
                    })
                    candidates.append({
                        "index": len(candidates),
                        "claim_id": claim["claim_id"],
                        "statement": claim["statement"],
                        "direction": claim["status"],
                        "scope": claim["scope"],
                        "evidence_ids": cited,
                    })
                entry = groups.get(group_key)
                if entry is None:
                    group_evidence_ids = {eid for c in candidates for eid in c["evidence_ids"]}
                    # Cite original, unaltered records, including scope/provenance.
                    evidence = [
                        records[eid] for eid in sorted(group_evidence_ids)
                    ]
                    context = {
                        "candidate_action": obj["candidate_action"]["action_name"],
                        "threat": obj["threat"],
                        "pmt": obj["pmt_or_mitigation"],
                        "reference_enterprise": obj["context_scenario"]["reference_enterprise_profile"]["value"],
                        "organization_type": obj["deployment_context"]["organization_type"],
                        "infrastructure_context": obj["deployment_context"]["infrastructure_context"],
                        "budget_context": obj["deployment_context"]["budget_context"],
                    }
                    payload = {"context": context, "claims": candidates, "evidence_records": evidence}
                    entry = {"group_id": group_key, "lens": role, "payload": payload, "members": []}
                    groups[group_key] = entry
                else:
                    if entry["payload"]["claims"] != candidates:
                        raise ValueError(f"Distinct claims deduplicated erroneously: {group_key}")
                    if entry["payload"]["evidence_records"] != [
                        records[eid] for eid in sorted({eid for c in candidates for eid in c["evidence_ids"]})
                    ]:
                        raise ValueError(f"Evidence drift within grouped Lens: {group_key}")
                    if entry["payload"]["context"] != {
                        "candidate_action": obj["candidate_action"]["action_name"],
                        "threat": obj["threat"],
                        "pmt": obj["pmt_or_mitigation"],
                        "reference_enterprise": obj["context_scenario"]["reference_enterprise_profile"]["value"],
                        "organization_type": obj["deployment_context"]["organization_type"],
                        "infrastructure_context": obj["deployment_context"]["infrastructure_context"],
                        "budget_context": obj["deployment_context"]["budget_context"],
                    }:
                        raise ValueError(f"Different deployment context grouped: {group_key}")
                entry["members"].append({
                    "case_id": bundle.case_id,
                    "scenario_id": obj["scenario_id"],
                    "decision_object_id": decision_id,
                })

    if len(structural) == 0 or any(not row["passes_existing_slot_gate"] for row in structural):
        raise ValueError("A frozen Stage 4 directional claim failed its existing slot gate")
    if sum(len(group["members"]) for group in groups.values()) != 63:
        raise ValueError("Expected exactly 63 Lens assessments in 21 scenarios")
    return {
        "version": VERSION,
        "stage_artifact_hashes": hashes,
        "unique_assessments": sorted(groups.values(), key=lambda x: x["group_id"]),
        "structural_claims": structural,
        "structural_diagnostic": {
            "scenario_lens_assessments": 63,
            "unique_lens_assessments": len(groups),
            "directional_claim_instances": len(structural),
            "eligible_directional_claim_instances": sum(
                x["passes_existing_slot_gate"] for x in structural
            ),
        },
        "limitation": (
            "This structural inventory checks cited ID membership, source slot "
            "eligibility, and original GENERAL/DEPLOYMENT_SPECIFIC labels only. "
            "It does not independently establish semantic correctness, empirical "
            "transferability, ROI, or the validity of a final recommendation."
        ),
    }


def _probes(inventory: dict) -> list[dict]:
    originals = {
        ("technical_feasibility", "T-1"): (
            "This Android malware-detection result therefore proves 95% detection "
            "performance in the CIS IG2 reference enterprise."
        ),
        ("financial_adoption", "FIN_GEN_SUPPORT_1"): (
            "The SIEM log-consolidation record therefore establishes a verified "
            "150% return on investment for anomaly-detection deployment in "
            "the CIS IG2 reference enterprise."
        ),
    }
    cases = []
    for (lens, claim_id), error_sentence in originals.items():
        group = next(
            item for item in inventory["unique_assessments"]
            if item["lens"] == lens
            and any(member["case_id"] == "main__malware__anomaly_detection" for member in item["members"])
            and any(claim["claim_id"] == claim_id for claim in item["payload"]["claims"])
        )
        base_claim = next(claim for claim in group["payload"]["claims"] if claim["claim_id"] == claim_id)
        cited = set(base_claim["evidence_ids"])
        for variant in ("clean", "corrupted"):
            claim = deepcopy(base_claim)
            claim["index"] = 0
            if variant == "corrupted":
                claim["statement"] = claim["statement"] + " " + error_sentence
            payload = {
                "context": deepcopy(group["payload"]["context"]),
                "claims": [claim],
                "evidence_records": [
                    deepcopy(record)
                    for record in group["payload"]["evidence_records"]
                    if record["evidence_id"] in cited
                ],
            }
            cases.append({
                "id": f"malware__{lens}__{claim_id}",
                "variant": variant,
                "payload": payload,
            })
    return cases


def _identity(payload: dict) -> dict:
    return {
        "version": VERSION,
        "model": MODEL,
        "model_sha256": MODEL_SHA256,
        "prompt_sha256": sha256_text(SYSTEM),
        "schema_sha256": sha256_json(ReviewBatch.model_json_schema()),
        "input_sha256": sha256_json(payload),
    }


def _path(item_id: str) -> Path:
    return OUTPUT / "lens_scope_judgments" / f"{sha256_text(item_id)}.json"


def _validate(answer: ReviewBatch, payload: dict) -> ReviewBatch:
    expected = {c["index"]: set(c["evidence_ids"]) for c in payload["claims"]}
    if len(answer.claims) != len(expected) or {c.index for c in answer.claims} != set(expected):
        raise ValueError("Missing or duplicated Lens audit claim indices")
    for claim in answer.claims:
        seen = [r.evidence_id for r in claim.source_roles]
        if len(seen) != len(expected[claim.index]) or set(seen) != expected[claim.index]:
            raise ValueError(f"Missing or duplicated cited-source role: claim {claim.index}")
    return answer


def _request(payload: dict) -> ReviewBatch:
    message = canonical_json(payload)
    for attempt in range(2):
        response = requests.post(
            URL,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": message},
                ],
                "temperature": 0,
                "seed": 42,
                "max_tokens": 4096,
                "response_format": {
                    "type": "json_object",
                    "schema": ReviewBatch.model_json_schema(),
                },
            },
            timeout=1800,
        )
        response.raise_for_status()
        try:
            value = response.json()["choices"][0]["message"]["content"]
            return _validate(ReviewBatch.model_validate_json(value), payload)
        except (ValueError, KeyError, IndexError, TypeError):
            if attempt:
                if len(payload["claims"]) == 1:
                    raise
                # Model numbering/omission error: salvage only this unfinished
                # assessment, with dense single-claim indices and exact citations.
                singles = []
                for claim in payload["claims"]:
                    one = deepcopy(claim)
                    one["index"] = 0
                    subset = {
                        **payload,
                        "claims": [one],
                        "evidence_records": [
                            record for record in payload["evidence_records"]
                            if record["evidence_id"] in set(one["evidence_ids"])
                        ],
                    }
                    result = _request(subset)
                    singles.append(
                        result.claims[0].model_copy(update={"index": claim["index"]})
                    )
                return _validate(ReviewBatch(claims=singles), payload)
    raise AssertionError("unreachable")


def _load_result(item_id: str, payload: dict) -> ReviewBatch | None:
    result = _load_checked(_path(item_id))
    if result is None:
        return None
    if result["identity"] != _identity(payload):
        raise ValueError(f"Stale Lens scope checkpoint: {item_id}")
    return _validate(ReviewBatch.model_validate(result["judgment"]), payload)


def build_report(inventory: dict) -> dict:
    rows = []
    for group in inventory["unique_assessments"]:
        result = _load_result(group["group_id"], group["payload"])
        if result is None:
            raise ValueError(f"Incomplete Lens audit: {group['group_id']}")
        rows.append({
            "group_id": group["group_id"],
            "lens": group["lens"],
            "members": group["members"],
            "claim_reviews": [
                {
                    **review.model_dump(mode="json"),
                    "claim_id": group["payload"]["claims"][review.index]["claim_id"],
                    "claim_statement": group["payload"]["claims"][review.index]["statement"],
                    "claim_scope": group["payload"]["claims"][review.index]["scope"],
                    "claimed_direction": group["payload"]["claims"][review.index]["direction"],
                    "cited_evidence_ids": group["payload"]["claims"][review.index]["evidence_ids"],
                }
                for review in sorted(result.claims, key=lambda x: x.index)
            ],
        })
    probes = []
    for item in _probes(inventory):
        result = _load_result(item["id"] + "__" + item["variant"], item["payload"])
        if result is None:
            raise ValueError(f"Missing scope-corruption control: {item['id']} {item['variant']}")
        probes.append({
            "id": item["id"],
            "variant": item["variant"],
            "judgment": result.claims[0].model_dump(mode="json"),
        })
    # A failed control is an observed negative result, never hidden by rerun.
    def accepted(judgment: dict) -> bool:
        return (
            judgment["factual_verdict"] == "SUPPORTED"
            and judgment["scope_verdict"] != "OVERGENERALIZED"
            and not judgment["unsupported_economics"]
        )

    paired = {}
    for probe in probes:
        paired.setdefault(probe["id"], {})[probe["variant"]] = probe["judgment"]
    if len(paired) != 2 or any(set(value) != {"clean", "corrupted"} for value in paired.values()):
        raise ValueError("Expected both frozen variants of the two scope controls")
    clean_accepted = sum(accepted(value["clean"]) for value in paired.values())
    rejected = sum(not accepted(value["corrupted"]) for value in paired.values())
    both_correct = sum(
        accepted(value["clean"]) and not accepted(value["corrupted"])
        for value in paired.values()
    )
    flagged = [
        (row["group_id"], claim["claim_id"])
        for row in rows for claim in row["claim_reviews"]
        if (
            claim["factual_verdict"] != "SUPPORTED"
            or claim["scope_verdict"] == "OVERGENERALIZED"
            or (
                claim["claim_scope"] == "DEPLOYMENT_SPECIFIC"
                and claim["scope_verdict"] != "DEPLOYMENT_SUPPORTED"
            )
            or claim["action_relevance"] in ("ADJACENT_ONLY", "UNRELATED")
            or claim["unsupported_economics"]
        )
    ]
    return {
        "version": VERSION,
        "structural_diagnostic": inventory["structural_diagnostic"],
        "unique_lens_review_count": len(rows),
        "directional_claim_review_count": sum(len(row["claim_reviews"]) for row in rows),
        "verdict_counts": {
            verdict: sum(
                claim["factual_verdict"] == verdict
                for row in rows for claim in row["claim_reviews"]
            )
            for verdict in (
                "SUPPORTED", "PARTIAL", "TOPICAL_ONLY",
                "CONTRADICTS", "INSUFFICIENT_EVIDENCE",
            )
        },
        "scope_corruption_probes": probes,
        "scope_clean_controls_accepted": clean_accepted,
        "scope_corruptions_rejected": rejected,
        "scope_pairs_discriminated": both_correct,
        "scope_corruptions_total": 2,
        "flagged_directional_claims": flagged,
        "flagged_directional_claim_count": len(flagged),
        "lens_reviews": rows,
        "limitations": (
            "Gemma model judgments are an independent-model diagnostic, not "
            "expert ground truth. Evidence labels and rationales require manual "
            "inspection, particularly for the Android and SIEM anchors. "
            "The two hand-crafted corruptions test only those two transfer "
            "errors. Do not infer deployment-specific efficacy or a calibrated "
            "error rate from this convenience sample."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inventory", action="store_true", help="Deterministic structural check only; no LLM")
    mode.add_argument("--audit", action="store_true", help="Evaluate unique directional Lens claims with Gemma")
    mode.add_argument("--report", action="store_true", help="Read existing independent judgments without LLM")
    args = parser.parse_args()
    inventory = build_inventory()
    _freeze(OUTPUT / "lens_scope_inventory.json", inventory)
    print("Structural Lens inventory:", inventory["structural_diagnostic"], flush=True)
    if args.inventory:
        return

    if args.audit:
        entries = [
            (group["group_id"], group["payload"])
            for group in inventory["unique_assessments"]
        ] + [
            (item["id"] + "__" + item["variant"], item["payload"])
            for item in _probes(inventory)
        ]
        pending = [
            (item_id, payload)
            for item_id, payload in entries
            if _load_result(item_id, payload) is None
        ]
        if pending:
            from Stage7.gemma_server import use_gemma

            print(f"Gemma scope audit: {len(pending)} missing of {len(entries)} requests", flush=True)
            with use_gemma():
                for ordinal, (item_id, payload) in enumerate(pending, start=1):
                    answer = _request(payload)
                    _freeze(_path(item_id), {
                        "identity": _identity(payload),
                        "judgment": answer.model_dump(mode="json"),
                    })
                    print(f"Scope audit {ordinal}/{len(pending)}: {item_id}", flush=True)
    report = build_report(inventory)
    _freeze(OUTPUT / "lens_scope_summary.json", report)
    print(f"Scope audit report: {OUTPUT / 'lens_scope_summary.json'}")
    print("Factual verdicts:", report["verdict_counts"],
          "scope probe pairs discriminated:", report["scope_pairs_discriminated"], "/ 2")


if __name__ == "__main__":
    main()
