"""Reviewer 2 Concern #3: frozen cross-review, with/without Mediator post-assessments.

Run from Multi-Agent:
  py -3 -m Stage7.mediator_ablation --check
  py -3 -m Stage7.mediator_ablation --run  # reuse completed Qwen outputs; auto-load Gemma for blind audit
  py -3 -m Stage7.mediator_ablation --report
  py -3 -m Stage7.mediator_ablation --error-injection  # only 12 independent Gemma probe calls

Six one-round main cases use their exact frozen mediated post-assessments.
Brute Force is a separate one-round truncation sensitivity comparison; its
two-round main outcome is NEVER substituted as a matched baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from Pipeline.main import _case_config
from Stage0.agent_input import build_agent_input
from Stage1.cases import MAIN_STAGE1_CASES
from Stage1.runtime import (
    LlamaCppChatClient,
    assert_llama_server_ready,
    get_legacy_one_slot_runtime_profile,
    get_runtime_config,
)
from Stage1.schema import (
    CriticAssessment,
    CriticType,
    build_constrained_critic_response_schema,
    validate_critic_assessment,
)
from Stage2 import nodes as stage2_nodes
from Stage2.prompts import STAGE2_POST_ASSESSMENT_APPENDIX, STAGE2_POST_USER_PROMPT, stage2_role_context
from Stage2.schema import surfaced_evidence_ids, validate_debate_turn, validate_stage2_result
from Stage7.loaders import load_main_case_bundles, sha256_json, sha256_text
from Stage7.forecast_link_counterfactual import _ready, _server, _switch_server


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Multi-Agent" / "Results" / "Stage7" / "mediator_ablation"
VERSION = "mediator-ablation-frozen-round1-v1"
SEED = 42
TWO_ROUND_CASE = "main__brute_force__ids_ips"
ROLES = (CriticType.ATTACK_FEASIBILITY, CriticType.DEFENSE_ROBUSTNESS)


def _canonical(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _load_checked(path: Path) -> dict | None:
    if not path.is_file():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.get("sha256")
    if not digest or digest != sha256_json({k: v for k, v in payload.items() if k != "sha256"}):
        raise ValueError(f"Experiment record checksum mismatch: {path}")

    return payload


def _freeze(path: Path, payload: dict) -> dict:
    result = {**payload, "sha256": sha256_json(payload)}
    existing = _load_checked(path)
    if existing is not None:
        if existing != result:
            raise ValueError(f"Refusing to overwrite a different frozen result: {path}")
        return existing

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(_canonical(result) + "\n")

    return result


def _sources(case_id: str | None = None) -> list[dict]:
    bundles = {bundle.case_id: bundle for bundle in load_main_case_bundles(ROOT)}
    sources = []

    for case in MAIN_STAGE1_CASES:
        if case_id is not None and case.case_id != case_id:
            continue

        bundle = bundles[case.case_id]
        config = _case_config(case)
        data = build_agent_input(
            project_root=ROOT,
            snapshot_id=config["snapshot_id"],
            threat=case.threat,
            pmt=case.pmt,
            analysis_cutoff_date=config["analysis_cutoff_date"],
            evaluation_mode=config["evaluation_mode"],
            case_id=case.case_id,
        )
        pack = data["evidence_pack"]
        if (
            sha256_json(pack) != bundle.stage1["stage0_evidence_pack_sha256"]
            or sha256_text(data["forecast_data"]) != bundle.stage1["forecast_data_sha256"]
        ):
            raise ValueError(f"Frozen Stage 0 input mismatch: {case.case_id}")

        result = bundle.stage2_result
        runtime = bundle.stage2["runtime"]
        if (
            runtime["seed"] != SEED
            or runtime["expected_context_size"] != 131072
            or runtime["expected_parallel_slots"] != 1
        ):
            raise ValueError(f"Original Stage 2 runtime is not the historical one-slot seed-42 profile: {case.case_id}")

        before = bundle.stage1["assessments"]
        if (
            result["attack_pre_assessment"] != before["attack_feasibility"]
            or result["defense_pre_assessment"] != before["defense_robustness"]
        ):
            raise ValueError(f"Stage 1/2 pre-assessments differ: {case.case_id}")

        validate_stage2_result(
            result,
            evidence_pack=pack,
            attack_pre_assessment=before["attack_feasibility"],
            defense_pre_assessment=before["defense_robustness"],
        )
        expected_rounds = 2 if case.case_id == TWO_ROUND_CASE else 1
        if len(result["rounds"]) != expected_rounds:
            raise ValueError(f"Unexpected original debate length: {case.case_id}")

        round1 = result["rounds"][0]
        validate_debate_turn(
            round1["attack_turn"],
            evidence_pack=pack,
            opponent_assessment=before["defense_robustness"],
            expected_responder=CriticType.ATTACK_FEASIBILITY,
            round_index=1,
        )
        validate_debate_turn(
            round1["defense_turn"],
            evidence_pack=pack,
            opponent_assessment=before["attack_feasibility"],
            expected_responder=CriticType.DEFENSE_ROBUSTNESS,
            round_index=1,
        )
        sources.append({
            "case_id": case.case_id,
            "bundle": bundle,
            "evidence_pack": pack,
            "forecast_data": data["forecast_data"],
            "pre": before,
            "round1": round1,
            "result": result,
        })

    if not sources:
        raise ValueError(f"No frozen cases selected: {case_id}")

    return sources


def _allowed_ids(source: dict) -> list[str]:
    before = source["pre"]
    turn = source["round1"]
    return surfaced_evidence_ids(
        attack_assessment=before["attack_feasibility"],
        defense_assessment=before["defense_robustness"],
        attack_turn=turn["attack_turn"],
        defense_turn=turn["defense_turn"],
    )


def _request(source: dict, role: CriticType, *, mediator: bool) -> dict:
    before = source["pre"]
    round1 = source["round1"]
    own = before[role.value]
    other_role = ROLES[1] if role == ROLES[0] else ROLES[0]
    opponent = before[other_role.value]
    allowed = _allowed_ids(source)
    evidence_view = stage2_nodes._forecast_data_visible_to_stage2_step(
        source["forecast_data"],
        allowed_evidence_ids=allowed,
    )
    debate = {
        "round_index": 1,
        "attack_turn": round1["attack_turn"],
        "defense_turn": round1["defense_turn"],
    }
    if mediator:
        debate["mediator_summary"] = round1["mediator_summary"]

    system_prompt = stage2_role_context(
        critic_type=role.value,
        forecast_data=evidence_view,
    ) + STAGE2_POST_ASSESSMENT_APPENDIX.format(
        own_pre_assessment=_canonical(own),
        opponent_pre_assessment=_canonical(opponent),
        debate_rounds=_canonical([debate]),
    )

    schema = build_constrained_critic_response_schema(
        evidence_pack=source["evidence_pack"],
        expected_critic_type=role,
    )
    schema["$defs"]["ClaimAssessment"]["properties"]["evidence_ids"]["items"] = {
        "type": "string",
        "enum": allowed,
    }

    return {
        "system_prompt": system_prompt,
        "schema": schema,
        "evidence_ids": allowed,
    }


def _validate_post(source: dict, role: CriticType, value: CriticAssessment | dict) -> CriticAssessment:
    parsed = validate_critic_assessment(
        value,
        evidence_pack=source["evidence_pack"],
        expected_critic_type=role,
    )
    used = {eid for claim in parsed.claims for eid in claim.evidence_ids}
    unexpected = used - set(_allowed_ids(source))
    if unexpected:
        raise ValueError(f"Post-assessment introduced unsurfaced evidence: {sorted(unexpected)}")

    return parsed


def _record_path(source: dict, role: CriticType, mediator: bool) -> Path:
    arm = "with_mediator_round1" if mediator else "without_mediator"
    return OUTPUT / "records" / source["case_id"] / f"{arm}__{role.value}__seed-{SEED}.json"


def _identity(source: dict, role: CriticType, mediator: bool) -> dict:
    request = _request(source, role, mediator=mediator)
    source_hashes = {
        "stage1": source["bundle"].stage1["artifact_sha256"],
        "stage2": source["bundle"].stage2["artifact_sha256"],
        "evidence_pack": sha256_json(source["evidence_pack"]),
        "round1_turns": sha256_json({
            k: source["round1"][k] for k in ("attack_turn", "defense_turn")
        }),
    }
    profile = get_legacy_one_slot_runtime_profile()
    profile["seed"] = SEED

    return {
        "version": VERSION,
        "case_id": source["case_id"],
        "role": role.value,
        "seed": SEED,
        "runtime": profile,
        "mediator": mediator,
        "source_hashes": source_hashes,
        "prompt_sha256": sha256_text(request["system_prompt"]),
        "schema_sha256": sha256_json(request["schema"]),
    }


async def _generate(source: dict, role: CriticType, *, mediator: bool) -> dict:
    request = _request(source, role, mediator=mediator)
    identity = _identity(source, role, mediator)
    path = _record_path(source, role, mediator)
    cached = _load_checked(path)
    if cached is not None:
        if cached["identity"] != identity:
            raise ValueError(f"Completed run no longer matches the frozen inputs: {path}")
        _validate_post(source, role, cached["assessment"])
        print(f"Reused: {path.name}", flush=True)
        return cached

    config = replace(
        get_runtime_config(),
        seed=SEED,
        expected_context_size=131072,
        expected_parallel_slots=1,
    )
    client = LlamaCppChatClient(config=config, request_concurrency=1)
    runner = client.with_structured_output(
        CriticAssessment,
        method="json_schema",
        response_schema=request["schema"],
        progress_label=f"Mediator-ablation-{source['case_id']}-{role.value}-{'with' if mediator else 'without'}",
        semantic_validator=lambda value: _validate_post(source, role, value),
    )
    response = await runner.ainvoke([
        SystemMessage(content=request["system_prompt"]),
        HumanMessage(content=STAGE2_POST_USER_PROMPT),
    ])
    return _freeze(path, {
        "identity": identity,
        "assessment": response.model_dump(mode="json"),
    })


def _baseline(source: dict, role: CriticType) -> dict:
    if source["case_id"] == TWO_ROUND_CASE:
        record = _load_checked(_record_path(source, role, True))
        if record is None:
            raise ValueError("Brute Force requires an independently generated one-round mediated control")
        if record["identity"] != _identity(source, role, True):
            raise ValueError("Brute Force one-round Mediator baseline identity differs")
        value = record["assessment"]
    else:
        value = source["result"][
            "attack_post_assessment" if role == ROLES[0] else "defense_post_assessment"
        ]

    return _validate_post(source, role, value).model_dump(mode="json")


def _attack_evidence_audit(source: dict) -> dict:
    pack = source["evidence_pack"]
    tagged = [
        {
            "evidence_id": record["evidence_id"],
            "source_locator": record.get("source_locator"),
            "content": record.get("content"),
            "allowed_claim_types": record.get("allowed_claim_types"),
        }
        for record in pack["evidence"]
        if pack["threat_id"] in record.get("threat_ids", [])
        and "directional_threat_trend" in record.get("allowed_claim_types", [])
    ]
    return {
        "case_id": source["case_id"],
        "attack_stance": source["pre"]["attack_feasibility"]["stance"],
        "source_contract_candidates": tagged,
        "note": (
            "These are source-contract candidates only. They do not establish a direct, "
            "modality-matched NoI incident-count time trend; content-level review is required."
        ),
    }


def _report_path(sources: list[dict], name: str) -> Path:
    if len(sources) == len(MAIN_STAGE1_CASES):
        return OUTPUT / name

    return OUTPUT / f"{Path(name).stem}__{sources[0]['case_id']}.json"


def _write_blind_review(sources: list[dict]) -> None:
    items = []
    key = {}

    for source in sources:
        for role in ROLES:
            with_med = _baseline(source, role)
            record = _load_checked(_record_path(source, role, False))
            without = _validate_post(source, role, record["assessment"]).model_dump(mode="json")
            review_id = f"{source['case_id']}__{role.value}"
            reverse = int(sha256_text(f"20260916|{review_id}"), 16) % 2 == 1
            first, second = (without, with_med) if reverse else (with_med, without)
            key[review_id] = {
                "A": "without_mediator" if reverse else "with_mediator",
                "B": "with_mediator" if reverse else "without_mediator",
                "analysis": "one_round_sensitivity" if source["case_id"] == TWO_ROUND_CASE else "matched_one_round",
            }
            items.append({
                "review_id": review_id,
                "forecast_summary": source["evidence_pack"]["forecast_summary"],
                "evidence_records": source["evidence_pack"]["evidence"],
                "cross_review_round1": {
                    "attack_turn": source["round1"]["attack_turn"],
                    "defense_turn": source["round1"]["defense_turn"],
                },
                "assessment_A": first,
                "assessment_B": second,
                "review_fields": {
                    "A": {"claim_evidence_fidelity": None, "unsupported_specificity": None,
                          "modality_compliance": None, "unresolved_evidence_handling": None},
                    "B": {"claim_evidence_fidelity": None, "unsupported_specificity": None,
                          "modality_compliance": None, "unresolved_evidence_handling": None},
                },
            })

    _freeze(_report_path(sources, "blind_review.json"), {
        "version": VERSION,
        "instructions": (
            "Condition labels are concealed. Check each factual claim against cited evidence, "
            "the NoI/NoP modality, unsupported specificity, and unresolved evidence. "
            "Do not infer factual correctness from structural validity or changed claim counts."
        ),
        "items": items,
    })
    _freeze(_report_path(sources, "blind_key.json"), {"version": VERSION, "mapping": key})


def _report(sources: list[dict]) -> dict:
    comparisons = []
    for source in sources:
        for role in ROLES:
            no_med = _load_checked(_record_path(source, role, False))
            if no_med is None:
                raise ValueError(f"Missing no-Mediator result: {source['case_id']}/{role.value}")
            if no_med["identity"] != _identity(source, role, False):
                raise ValueError(f"Frozen comparison input mismatch: {source['case_id']}/{role.value}")
            without = _validate_post(source, role, no_med["assessment"]).model_dump(mode="json")
            with_med = _baseline(source, role)
            pre = source["pre"][role.value]
            shared = set(c["claim_id"] for c in with_med["claims"]) & set(
                c["claim_id"] for c in without["claims"]
            )
            a = {c["claim_id"]: c for c in with_med["claims"]}
            b = {c["claim_id"]: c for c in without["claims"]}
            comparisons.append({
                "case_id": source["case_id"],
                "role": role.value,
                "analysis": "one_round_sensitivity" if source["case_id"] == TWO_ROUND_CASE else "matched_one_round",
                "pre_stance": pre["stance"],
                "with_mediator_stance": with_med["stance"],
                "without_mediator_stance": without["stance"],
                "with_mediator_claim_count": len(a),
                "without_mediator_claim_count": len(b),
                "shared_claim_id_count": len(shared),
                "shared_claim_statement_differences": sum(a[k]["statement"] != b[k]["statement"] for k in shared),
                "shared_claim_relation_differences": sum(a[k]["forecast_relation"] != b[k]["forecast_relation"] for k in shared),
                "with_mediator_unresolved_questions": len(with_med["unresolved_questions"]),
                "without_mediator_unresolved_questions": len(without["unresolved_questions"]),
            })

    result = {
        "version": VERSION,
        "nominal_seed": SEED,
        "comparison_count": len(comparisons),
        "matched_case_count": sum(len(s["result"]["rounds"]) == 1 for s in sources),
        "one_round_sensitivity_case_count": sum(len(s["result"]["rounds"]) > 1 for s in sources),
        "analysis_limit": (
            "Structural differences do not measure factual correctness. Mediator-guided "
            "round-two Brute Force is excluded from the matched one-round comparison. "
            "Independent blinded assessment is needed for claim-quality conclusions."
        ),
        "comparisons": comparisons,
        "attack_evidence_contract_audit": [_attack_evidence_audit(s) for s in sources],
    }
    _freeze(_report_path(sources, "summary.json"), result)
    _write_blind_review(sources)
    return result


async def _run(sources: list[dict]) -> None:
    assert_llama_server_ready(
        expected_context_size=131072,
        expected_parallel_slots=1,
        profile_name="Mediator ablation: historical Stage 2 one-slot runtime",
    )

    for source in sources:
        arms = (False, True) if source["case_id"] == TWO_ROUND_CASE else (False,)
        for mediator in arms:
            print(f"{source['case_id']} mediator={mediator}", flush=True)
            for role in ROLES:
                await _generate(source, role, mediator=mediator)


def _run_all(sources: list[dict]) -> dict:
    _, command = _server()
    original_parallel = int(command[command.index("--parallel") + 1])
    _ready(original_parallel)

    try:
        _switch_server(command, 1)
        asyncio.run(_run(sources))
        return _report(sources)
    finally:
        _switch_server(command, original_parallel)


def _records_complete(sources: list[dict]) -> bool:
    for source in sources:
        arms = (False, True) if source["case_id"] == TWO_ROUND_CASE else (False,)
        for mediator in arms:
            for role in ROLES:
                record = _load_checked(_record_path(source, role, mediator))
                if record is None:
                    return False
                if record["identity"] != _identity(source, role, mediator):
                    raise ValueError(f"Frozen Mediator-ablation record identity drift: {source['case_id']}")
                _validate_post(source, role, record["assessment"])

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Verify frozen inputs without inference")
    mode.add_argument("--run", action="store_true", help="Resume missing post-assessments, then automatically audit using Gemma")
    mode.add_argument("--report", action="store_true", help="Compare frozen outputs without inference")
    mode.add_argument("--error-injection", action="store_true", help="Test Gemma on 6 frozen clean/corrupted claim pairs, skipping the 229-claim audit")
    parser.add_argument("--case", choices=[x.case_id for x in MAIN_STAGE1_CASES])
    args = parser.parse_args()

    if args.error_injection:
        if args.case:
            parser.error("--error-injection does not accept --case")
        from Stage7 import mediator_error_injection

        result = mediator_error_injection.run()
        print(
            f"Error-injection report: {OUTPUT / 'error_injection' / 'summary.json'} "
            f"(sensitivity={result['sensitivity']:.3f}, "
            f"specificity={result['specificity']:.3f})",
            flush=True,
        )
        return

    sources = _sources(args.case)
    for source in sources:
        candidate_count = len(_attack_evidence_audit(source)["source_contract_candidates"])
        print(
            f"Verified {source['case_id']}: rounds={len(source['result']['rounds'])}, "
            f"Threat-direction source-contract candidates={candidate_count}",
            flush=True,
        )

    if args.check:
        return
    if args.run:
        if _records_complete(sources):
            print("All frozen Qwen Mediator-ablation assessments verified; skipping their inference", flush=True)
            result = _report(sources)
        else:
            result = _run_all(sources)
    else:
        result = _report(sources)
    print(
        f"Report: {_report_path(sources, 'summary.json')} ({result['comparison_count']} Critic comparisons, "
        f"{result['matched_case_count']} matched cases)",
        flush=True,
    )

    if args.run and len(sources) == len(MAIN_STAGE1_CASES):
        from Stage7 import mediator_quality_analysis

        quality = mediator_quality_analysis.run()
        print(
            f"Gemma quality report: {OUTPUT / 'quality_summary.json'} "
            f"({quality['audit_pairs']} pairs, {quality['total_claims_audited']} claims)",
            flush=True,
        )


if __name__ == "__main__":
    main()
