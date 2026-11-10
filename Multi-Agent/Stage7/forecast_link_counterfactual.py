"""Controlled forecast-to-decision sensitivity experiment for Reviewer 2 #2.

This is a synthetic PMT-trajectory stress test, not a new B-MTGNN forecast.
Only the counterfactual forecast changes. The factual evidence, all existing
Stage 1--4 prompts and validators, and the final policy are left untouched.
Experimental artifacts live in a sandbox, never in the main Results directory.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from statistics import mean
from unittest.mock import patch

import psutil
import requests
from langgraph.graph import END, StateGraph

import Stage1.nodes as stage1_nodes
import Stage1.artifact as stage1_artifact
import Stage1.runtime as stage1_runtime
import Stage2.nodes as stage2_nodes
import Stage2.artifact as stage2_artifact
import Stage3.nodes as stage3_nodes
import Stage3.artifact as stage3_artifact
import Stage3.context_nodes as stage3_context_nodes
import Stage3.context_artifact as stage3_context_artifact
import Stage4.context_nodes as stage4_context_nodes
import Stage4.context_artifact as stage4_context_artifact
from Pipeline.main import _case_config, _initial_state
from Pipeline.state import AgentState
from Stage0.agent_input import build_agent_input
from Stage0.builder import Stage0Builder
from Stage0.schema import EvidencePack, EvidenceRecord, EvaluationMode
from Stage1.artifact import build_stage1_identity
from Stage1.cases import MAIN_STAGE1_CASES
from Stage1.schema import CriticType
from Stage2.schema import (
    DebateRound,
    Stage2DebateResult,
    flatten_debate_exchanges,
    round_requires_followup,
    validate_stage2_result,
)
from Stage4.context_schema import DecisionLensEvaluation
from Stage6.builder import _decision_recommendation
from Stage7.loaders import load_main_case_bundles, sha256_json, sha256_text


VERSION = "forecast-link-counterfactual-v2-controlled"
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Multi-Agent" / "Results" / "Stage7" / "forecast_link_counterfactual_controlled"
SANDBOX = OUTPUT / "sandbox"
SEEDS = (42, 43, 44)
ARMS = ("original", "reversed_pmt")
MEDIATOR_RECOVERY_SEED_OFFSETS = (10000, 20000)
SERVER_URL = "http://127.0.0.1:8080"
SLOT_CONTEXT = {1: 131072, 2: 65536}


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _hash(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _write_frozen(path: Path, body: dict) -> None:
    payload = deepcopy(body)
    payload["sha256"] = _hash(body)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != payload:
            raise ValueError(f"Frozen experiment artifact conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(_json(payload) + "\n", encoding="utf-8")
    temp.replace(path)


def _load_frozen(path: Path) -> dict | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = value.pop("sha256", None)
    if digest != _hash(value):
        raise ValueError(f"Experiment artifact hash mismatch: {path}")
    value["sha256"] = digest
    return value


def _series(builder: Stage0Builder, node_id: str) -> list[dict]:
    rows = sorted(
        (row for row in builder.state_series[node_id] if row["phase"] == "forecast"),
        key=lambda row: row["timestamp"],
    )
    if len(rows) != 36 or len({row["timestamp"] for row in rows}) != 36:
        raise ValueError(f"Not a distinct 36-month forecast: {node_id}")
    return rows


def _arm_input(base: dict, builder: Stage0Builder, arm: str) -> tuple[dict, dict]:
    """Build one arm and its full synthetic-field provenance without LLM calls."""
    data = deepcopy(base)
    pack = data["evidence_pack"]
    summary = pack["forecast_summary"]
    threat = _series(builder, pack["threat_id"])
    pmt = _series(builder, pack["pmt_id"])
    if [row["timestamp"] for row in threat] != [row["timestamp"] for row in pmt]:
        raise ValueError("Threat and PMT forecast calendars differ")

    original_values = [float(row["value_z"]) for row in pmt]
    values = original_values if arm == "original" else list(reversed(original_values))
    if arm not in ARMS:
        raise ValueError(f"Unsupported arm: {arm}")

    old_gaps: dict[str, list[float]] = defaultdict(list)
    new_gaps: dict[str, list[float]] = defaultdict(list)
    for t, old, new in zip(threat, original_values, values):
        year = str(t["timestamp"][:4])
        old_gaps[year].append(float(t["value_z"]) - old)
        new_gaps[year].append(float(t["value_z"]) - new)

    for year, monthly in old_gaps.items():
        if abs(mean(monthly) - summary["gap_by_year"][year]) > 1e-10:
            raise ValueError(f"Original Gap does not match the frozen monthly data: {year}")
    if abs(mean(original_values) - summary["pmt_state"]["forecast_mean_z"]) > 1e-10:
        raise ValueError("Frozen PMT mean does not match original monthly data")

    years = sorted(new_gaps, key=int)
    gaps = {year: mean(new_gaps[year]) for year in years}
    slope = (gaps[years[-1]] - gaps[years[0]]) / (int(years[-1]) - int(years[0]))
    direction = (
        "threat_minus_pmt_increasing" if slope > 0
        else "threat_minus_pmt_decreasing" if slope < 0 else "flat"
    )
    vector_digest = _hash({"pmt_id": pack["pmt_id"], "values": values})[:16]
    case_digest = _hash({"case_id": pack["case_id"], "gaps": gaps})[:16]
    pmt_ids = [
        f"FCE_{vector_digest}_PMT_{row['timestamp'][:7]}" for row in pmt
    ]
    gap_ids = [f"FCE_{case_digest}_GAP_{year}" for year in years]

    if arm == "reversed_pmt":
        summary["pmt_state"]["direction"] = builder._direction(values)
        summary["pmt_state"]["forecast_field_ids"] = pmt_ids
        # Reversing a vector preserves its mean. Retain the original scalar.
        summary["gap_by_year"] = gaps
        summary["gap_slope_per_year"] = slope
        summary["gap_direction"] = direction
        summary["forecast_field_ids"]["pmt"] = pmt_ids
        summary["forecast_field_ids"]["gap"] = gap_ids
    EvidencePack(
        case_id=pack["case_id"],
        source_registry_version=pack["source_registry_version"],
        source_snapshot_id=pack["source_snapshot_id"],
        forecast_origin_date=pack["forecast_origin_date"],
        training_data_end=pack["training_data_end"],
        analysis_cutoff_date=pack["analysis_cutoff_date"],
        evaluation_mode=EvaluationMode(pack["evaluation_mode"]),
        threat_id=pack["threat_id"],
        pmt_id=pack["pmt_id"],
        forecast_summary=summary,
        evidence=tuple(EvidenceRecord.from_dict(record) for record in pack["evidence"]),
        retrieval_metadata=pack["retrieval_metadata"],
    ).validate()

    if arm == "reversed_pmt":
        prompt = json.loads(data["forecast_data"])
        view = prompt["forecast_summary"]
        view["pmt_state"]["direction"] = summary["pmt_state"]["direction"]
        view["gap_by_year"] = gaps
        view["gap_slope_per_year"] = slope
        view["gap_direction"] = direction
        view["forecast_field_ids"]["pmt"] = pmt_ids
        view["forecast_field_ids"]["gap"] = gap_ids
        data["forecast_data"] = _json(prompt)

    provenance = {
        "experiment": VERSION,
        "case_id": pack["case_id"],
        "arm": arm,
        "source": "Frozen B-MTGNN forecast monthly series",
        "transformation": "identity" if arm == "original" else "reverse PMT 36-month chronological values",
        "is_synthetic_bmtgnn_forecast": arm != "original",
        "uncertainty": "Original uncertainty summary retained; not recalibrated for synthetic input",
        "threat_monthly_sha256": _hash(threat),
        "original_pmt_monthly_sha256": _hash(pmt),
        "monthly_pmt": [
            {
                "date": row["timestamp"],
                "source_field_id": row["state_field_id"],
                "experiment_field_id": (
                    pmt_ids[index] if arm == "reversed_pmt"
                    else base["evidence_pack"]["forecast_summary"]["forecast_field_ids"]["pmt"][index]
                ),
                "original_value_z": original_values[index],
                "experiment_value_z": values[index],
            }
            for index, row in enumerate(pmt)
        ],
        "annual_gap": [
            {
                "year": year,
                "source_field_id": base["evidence_pack"]["forecast_summary"]["forecast_field_ids"]["gap"][i],
                "experiment_field_id": (
                    gap_ids[i] if arm == "reversed_pmt"
                    else base["evidence_pack"]["forecast_summary"]["forecast_field_ids"]["gap"][i]
                ),
                "original_gap": base["evidence_pack"]["forecast_summary"]["gap_by_year"][year],
                "experiment_gap": gaps[year],
            }
            for i, year in enumerate(years)
        ],
        "evidence_sha256": sha256_json(pack["evidence"]),
    }
    return data, provenance


def _paired_inputs(original: dict, modified: dict) -> None:
    a, b = original["evidence_pack"], modified["evidence_pack"]
    for key in a:
        if key != "forecast_summary" and a[key] != b[key]:
            raise ValueError(f"Paired conditions differ outside forecast summary: {key}")
    for key in a["forecast_summary"]:
        if key not in {"pmt_state", "gap_by_year", "gap_slope_per_year", "gap_direction", "forecast_field_ids"}:
            if a["forecast_summary"][key] != b["forecast_summary"][key]:
                raise ValueError(f"Unexpected forecast difference: {key}")
    for key in a["forecast_summary"]["pmt_state"]:
        if key not in {"direction", "forecast_field_ids"} and (
            a["forecast_summary"]["pmt_state"][key] != b["forecast_summary"]["pmt_state"][key]
        ):
            raise ValueError(f"Unexpected PMT difference: {key}")
    for key in a["forecast_summary"]["forecast_field_ids"]:
        if key not in {"pmt", "gap"} and (
            a["forecast_summary"]["forecast_field_ids"][key]
            != b["forecast_summary"]["forecast_field_ids"][key]
        ):
            raise ValueError(f"Unexpected forecast provenance change: {key}")
    aa = json.loads(original["forecast_data"])
    bb = json.loads(modified["forecast_data"])
    if set(aa) != set(bb):
        raise ValueError("Experiment prompt fields differ outside the forecast")
    for key in aa:
        if key != "forecast_summary" and aa[key] != bb[key]:
            raise ValueError(f"Stage 1/2 prompt control changed: {key}")
    for key in aa["forecast_summary"]:
        if key not in {"pmt_state", "gap_by_year", "gap_slope_per_year", "gap_direction", "forecast_field_ids"}:
            if aa["forecast_summary"][key] != bb["forecast_summary"][key]:
                raise ValueError(f"Stage 1/2 prompt forecast control changed: {key}")
    for key in aa["forecast_summary"]["pmt_state"]:
        if key not in {"direction", "forecast_field_ids"} and (
            aa["forecast_summary"]["pmt_state"][key]
            != bb["forecast_summary"]["pmt_state"][key]
        ):
            raise ValueError(f"Unexpected prompt PMT difference: {key}")
    for key in aa["forecast_summary"]["forecast_field_ids"]:
        if key not in {"pmt", "gap"} and (
            aa["forecast_summary"]["forecast_field_ids"][key]
            != bb["forecast_summary"]["forecast_field_ids"][key]
        ):
            raise ValueError(f"Unexpected prompt provenance change: {key}")
    if sha256_json(a["evidence"]) != sha256_json(b["evidence"]):
        raise ValueError("External evidence drift in paired forecast comparison")


def _inputs(case_ids: list[str]) -> tuple[dict, dict]:
    frozen = {bundle.case_id: bundle for bundle in load_main_case_bundles(ROOT)}
    builder = Stage0Builder(
        ROOT / "Multi-Agent" / "Results" / "Stage0" / "Forecast",
        ROOT / "Multi-Agent" / "Results" / "Stage0" / "Evidence" / "snapshots"
        / _case_config(MAIN_STAGE1_CASES[0])["snapshot_id"],
    )
    datasets, provenance = {}, {}
    shared_pmts: dict[tuple[str, str], str] = {}
    for case in MAIN_STAGE1_CASES:
        if case.case_id not in case_ids:
            continue
        config = _case_config(case)
        base = build_agent_input(
            project_root=ROOT,
            snapshot_id=config["snapshot_id"],
            threat=case.threat,
            pmt=case.pmt,
            analysis_cutoff_date=config["analysis_cutoff_date"],
            evaluation_mode=config["evaluation_mode"],
            case_id=case.case_id,
        )
        old = frozen[case.case_id].stage1
        if sha256_json(base["evidence_pack"]) != old["stage0_evidence_pack_sha256"]:
            raise ValueError(f"Frozen Stage 0 EvidencePack mismatch: {case.case_id}")
        if sha256_text(base["forecast_data"]) != old["forecast_data_sha256"]:
            raise ValueError(f"Frozen Stage 0 prompt mismatch: {case.case_id}")
        original, po = _arm_input(base, builder, "original")
        modified, pm = _arm_input(base, builder, "reversed_pmt")
        if original != base:
            raise ValueError(f"Original arm differs from the frozen Stage 0 input: {case.case_id}")
        _paired_inputs(original, modified)
        for arm, p in (("original", po), ("reversed_pmt", pm)):
            key = (base["evidence_pack"]["pmt_id"], arm)
            digest = _hash([(x["date"], x["experiment_value_z"]) for x in p["monthly_pmt"]])
            if key in shared_pmts and shared_pmts[key] != digest:
                raise ValueError(f"Shared PMT has inconsistent synthetic trajectory: {key}")
            shared_pmts[key] = digest
        datasets[case.case_id] = {"original": original, "reversed_pmt": modified}
        provenance[case.case_id] = {"original": po, "reversed_pmt": pm}
    if len(datasets) != len(case_ids):
        raise ValueError("One or more requested cases are not in the main frozen case set")
    return datasets, provenance


def _graph(payload: dict, phase: str = "downstream", arm: str = "original"):
    if phase not in {"upstream", "downstream"}:
        raise ValueError(f"Unexpected experiment phase: {phase}")

    def load_data(state: AgentState) -> dict:
        if state["stage0_config"]["case_id"] != payload["evidence_pack"]["case_id"]:
            raise ValueError("Case mismatched with frozen input")
        return {"evidence_pack": payload["evidence_pack"], "forecast_data": payload["forecast_data"]}

    async def historical_debate(state: AgentState) -> dict:
        return await _historical_stage2_debate_node(state, arm=arm)

    graph = StateGraph(AgentState)
    functions = {
        "load": load_data,
        "s1_prepare": stage1_nodes.prepare_stage1_node,
        "attack": stage1_nodes.attack_feasibility_critic_node,
        "defense": stage1_nodes.defense_robustness_critic_node,
        "s1_complete": stage1_nodes.stage1_complete_node,
        "s2_prepare": stage2_nodes.prepare_stage2_node,
        "s2_debate": (
            historical_debate
            if phase == "upstream"
            else stage2_nodes.stage2_debate_node
        ),
        "s2_complete": stage2_nodes.stage2_complete_node,
        "s3_build": stage3_nodes.stage3_build_node,
        "s3_context": stage3_context_nodes.stage3_contextualize_node,
    }
    if phase == "downstream":
        functions.update({
            "s4_prepare": stage4_context_nodes.prepare_stage4_context_node,
            "s4_lenses": stage4_context_nodes.contextual_priority_queue_lenses_node,
            "s4_complete": stage4_context_nodes.stage4_context_complete_node,
        })
    for name, function in functions.items():
        graph.add_node(name, function)
    graph.set_entry_point("load")
    graph.add_edge("load", "s1_prepare")
    graph.add_edge("s1_prepare", "attack")
    graph.add_edge("s1_prepare", "defense")
    graph.add_edge(["attack", "defense"], "s1_complete")
    chain = (
        "s1_complete", "s2_prepare", "s2_debate", "s2_complete", "s3_build",
        "s3_context", "s4_prepare", "s4_lenses", "s4_complete",
    )
    if phase == "upstream":
        chain = chain[:chain.index("s3_context") + 1]
    for left, right in zip(chain, chain[1:]):
        graph.add_edge(left, right)
    graph.add_edge(chain[-1], END)
    return graph.compile()


def _isolated_nodes(stack: ExitStack) -> None:
    """Patch only the calling modules' root resolvers, not common source code."""
    for module in (
        stage1_nodes, stage2_nodes, stage3_nodes,
        stage3_context_nodes, stage4_context_nodes,
    ):
        stack.enter_context(patch.object(module, "_project_root", return_value=SANDBOX))

    # Stage 3's contextual evidence builder is read-only with respect to its
    # source records; point only this read-side builder at the canonical root.
    build = stage3_context_nodes.build_contextual_stage3_bundle
    def read_from_frozen_sources(**kwargs):
        kwargs["project_root"] = ROOT
        return build(**kwargs)
    stack.enter_context(
        patch.object(stage3_context_nodes, "build_contextual_stage3_bundle", read_from_frozen_sources)
    )

    # Stage 1/2 main results were generated on the historical 1-slot runtime.
    # Only their artifact identity is pinned here. Stage 4 retains its own
    # existing 2-slot runtime contract without any intervention.
    stack.enter_context(patch.object(
        stage1_artifact,
        "get_experiment_runtime_profile",
        stage1_runtime.get_legacy_one_slot_runtime_profile,
    ))
    stack.enter_context(patch.object(
        stage2_artifact,
        "get_experiment_runtime_profile",
        stage1_runtime.get_legacy_one_slot_runtime_profile,
    ))
    stack.enter_context(patch.object(
        stage2_artifact,
        "STAGE2_EXECUTION_ORDER_VERSION",
        stage2_artifact.STAGE2_LEGACY_EXECUTION_ORDER_VERSION,
    ))


def _one_slot_llm():
    stage1_runtime.assert_llama_server_ready(
        expected_context_size=131072,
        expected_parallel_slots=1,
        profile_name="Forecast link Stage 1/2 historical single-slot",
    )
    config = replace(
        stage1_runtime.get_runtime_config(),
        expected_context_size=131072,
        expected_parallel_slots=1,
    )
    return stage1_runtime.LlamaCppChatClient(
        config=config, request_concurrency=1
    )


def _upstream_nodes(stack: ExitStack) -> None:
    stack.enter_context(patch.object(stage1_nodes, "get_llm", _one_slot_llm))
    stack.enter_context(patch.object(stage2_nodes, "get_llm", _one_slot_llm))


def _previous_mediator_failure(case_id: str, arm: str, seed: int, round_index: int) -> bool:
    path = OUTPUT / "failed_upstream.json"
    if not path.is_file():
        return False

    failures = json.loads(path.read_text(encoding="utf-8")).get("failures", [])
    for failure in failures:
        message = failure.get("message", "")
        if (
            failure.get("case_id") == case_id
            and failure.get("arm") == arm
            and failure.get("seed") == seed
            and "does not match MediatorSummary" in message
        ):
            matched = re.search(r'"round_index"\s*:\s*(\d+)', message)
            if matched and int(matched.group(1)) == round_index:
                return True

    return False


def _log_mediator_recovery(
    case_id: str, arm: str, seed: int, round_index: int,
    sampling_seed: int, status: str,
) -> None:
    path = (
        OUTPUT / "mediator_recovery" / case_id
        / f"{arm}__seed-{seed}__round-{round_index}.json"
    )
    if path.exists():
        log = json.loads(path.read_text(encoding="utf-8"))
    else:
        log = {
            "case_id": case_id,
            "arm": arm,
            "seed": seed,
            "round_index": round_index,
            "sampling_seed_offsets": list(MEDIATOR_RECOVERY_SEED_OFFSETS),
            "attempts": [],
        }

    log["attempts"].append({"sampling_seed": sampling_seed, "status": status})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(_json(log) + "\n", encoding="utf-8")
    temporary.replace(path)


async def _mediator_with_recovery(
    *,
    arm: str,
    seed: int,
    state: AgentState,
    llm_provider,
    identity: dict,
    round_index: int,
    attack_turn,
    defense_turn,
    previous_rounds: list[dict],
):
    """Retry only invalid Mediator output, keeping the native prompt and validator."""
    params = {
        "state": state,
        "identity": identity,
        "round_index": round_index,
        "attack_turn": attack_turn,
        "defense_turn": defense_turn,
        "previous_rounds": previous_rounds,
    }
    case_id = state["evidence_pack"]["case_id"]
    known_failed = _previous_mediator_failure(case_id, arm, seed, round_index)
    last_error = None

    if not known_failed:
        try:
            return await stage2_nodes._load_or_generate_mediator(
                **params, llm_provider=llm_provider
            )
        except RuntimeError as exc:
            if "does not match MediatorSummary" not in str(exc):
                raise
            last_error = exc

    for offset in MEDIATOR_RECOVERY_SEED_OFFSETS:
        sampling_seed = seed + offset

        def retry_provider():
            client = llm_provider()
            return stage1_runtime.LlamaCppChatClient(
                config=replace(client.config, seed=sampling_seed),
                request_concurrency=1,
            )

        print(
            f"Recovering Mediator only: {case_id}/{arm}/seed-{seed}/round-{round_index}, "
            f"sampling seed {sampling_seed}",
            flush=True,
        )
        _log_mediator_recovery(case_id, arm, seed, round_index, sampling_seed, "started")
        try:
            result = await stage2_nodes._load_or_generate_mediator(
                **params, llm_provider=retry_provider
            )
            _log_mediator_recovery(case_id, arm, seed, round_index, sampling_seed, "validated")
            return result
        except RuntimeError as exc:
            if "does not match MediatorSummary" not in str(exc):
                raise
            _log_mediator_recovery(case_id, arm, seed, round_index, sampling_seed, "invalid")
            last_error = exc

    raise RuntimeError(
        f"Mediator recovery exhausted its two fixed additional sampling seeds "
        f"for {case_id}/{arm}/seed-{seed}/round-{round_index}"
    ) from last_error


async def _historical_stage2_debate_node(state: AgentState, *, arm: str = "original") -> dict:
    """Use the archived main run's sequential Stage 2 scheduling.

    The existing Stage 2 LLM calls, prompts, validators and step checkpoints
    are reused unchanged; only the independent-call order matches the main
    archived execution-order identity.
    """
    if state.get("stage2_artifact_reused"):
        return {"messages": ["**Stage 2**: reused frozen structured-debate result."]}

    stage1 = stage2_nodes._load_exact_stage1_artifact(state)
    stage2_nodes._assert_state_pre_assessments_match_stage1_artifact(state, stage1)
    identity = stage2_nodes.build_stage2_identity(
        evidence_pack=state["evidence_pack"],
        forecast_data=state["forecast_data"],
        stage1_artifact=stage1,
    )
    client = None

    def llm_provider():
        nonlocal client
        if client is None:
            client = stage2_nodes.get_llm()
        return client

    rounds = []
    for round_index in (1, 2):
        previous = [item.model_dump(mode="json") for item in rounds]
        attack = await stage2_nodes._load_or_generate_turn(
            state=state, llm_provider=llm_provider, identity=identity,
            responder=CriticType.ATTACK_FEASIBILITY,
            round_index=round_index, previous_rounds=previous,
        )
        defense = await stage2_nodes._load_or_generate_turn(
            state=state, llm_provider=llm_provider, identity=identity,
            responder=CriticType.DEFENSE_ROBUSTNESS,
            round_index=round_index, previous_rounds=previous,
        )
        mediator = await _mediator_with_recovery(
            arm=arm,
            seed=int(os.environ.get("LLM_SEED", "42")),
            state=state, llm_provider=llm_provider, identity=identity,
            round_index=round_index, attack_turn=attack,
            defense_turn=defense, previous_rounds=previous,
        )
        result = DebateRound(
            round_index=round_index, attack_turn=attack,
            defense_turn=defense, mediator_summary=mediator,
        )
        rounds.append(result)
        if round_index == 1 and not round_requires_followup(result):
            break

    serialized = [item.model_dump(mode="json") for item in rounds]
    attack_post = await stage2_nodes._load_or_generate_post_assessment(
        state=state, llm_provider=llm_provider, identity=identity,
        critic_type=CriticType.ATTACK_FEASIBILITY, rounds=serialized,
    )
    defense_post = await stage2_nodes._load_or_generate_post_assessment(
        state=state, llm_provider=llm_provider, identity=identity,
        critic_type=CriticType.DEFENSE_ROBUSTNESS, rounds=serialized,
    )
    final = rounds[-1].mediator_summary
    result = Stage2DebateResult(
        case_id=state["evidence_pack"]["case_id"],
        attack_pre_assessment=state["attack_assessment"],
        defense_pre_assessment=state["defense_assessment"],
        rounds=rounds,
        exchanges=flatten_debate_exchanges(rounds),
        resolved_claims=[
            item for item in final.claim_matches if item.status.value == "RESOLVED"
        ],
        unresolved_claims=[
            item for item in final.claim_matches if item.status.value == "UNRESOLVED"
        ],
        evidence_conflicts=final.evidence_conflicts,
        evidence_gaps=final.evidence_gaps,
        final_adjudications=final.adjudications,
        attack_post_assessment=attack_post,
        defense_post_assessment=defense_post,
        stance_changes={
            "attack_feasibility": attack_post.stance - int(state["attack_assessment"]["stance"]),
            "defense_robustness": defense_post.stance - int(state["defense_assessment"]["stance"]),
        },
    )
    checked = validate_stage2_result(
        result, evidence_pack=state["evidence_pack"],
        attack_pre_assessment=state["attack_assessment"],
        defense_pre_assessment=state["defense_assessment"],
    )
    return {
        "stage2_debate_result": checked.model_dump(mode="json"),
        "debate_rounds": serialized,
        "attack_post_assessment": attack_post.model_dump(mode="json"),
        "defense_post_assessment": defense_post.model_dump(mode="json"),
        "messages": [f"**Stage 2**: completed {len(rounds)} historical-order round(s)."],
    }


def _main_reference_paths(case_id: str) -> list[Path]:
    bundle = next(b for b in load_main_case_bundles(ROOT) if b.case_id == case_id)
    main_context = bundle.stage3_context
    base_fingerprint = main_context["base_stage3_input_fingerprint"]
    base_path = (
        ROOT / "Multi-Agent" / "Results" / "Stage3" / "artifacts"
        / case_id / f"{base_fingerprint}.json"
    )
    if not base_path.is_file():
        raise ValueError(f"Missing frozen base Stage 3 artifact: {base_path}")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if (
        sha256_json({k: v for k, v in base.items() if k != "artifact_sha256"})
        != base.get("artifact_sha256")
        or base["artifact_sha256"] != main_context["base_stage3_artifact_sha256"]
    ):
        raise ValueError("Frozen main Stage 3 source hash mismatch")
    return [
        bundle.stage1_path, bundle.stage2_path, base_path,
        bundle.stage3_context_path, bundle.stage4_context_path,
    ]


def _import_original(case_id: str, seed: int) -> None:
    """Copy immutable main case artifacts, but only for seed 42."""
    if seed != 42:
        return
    main_root = ROOT / "Multi-Agent" / "Results"
    target_root = SANDBOX / "Multi-Agent" / "Results"
    for source in _main_reference_paths(case_id):
        target = target_root / source.relative_to(main_root)
        raw = source.read_bytes()
        if target.exists():
            if target.read_bytes() != raw:
                raise ValueError(f"Frozen main artifact differs from sandbox: {target}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Never overwrite an existing, possibly independently-generated file.
        with target.open("xb") as handle:
            handle.write(raw)
        if target.read_bytes() != raw:
            raise ValueError(f"Copy verification failed: {target}")


def _artifacts(data: dict, config: dict, *, contextual: bool) -> dict | None:
    s1, _, _ = stage1_artifact.load_frozen_stage1_artifact(
        project_root=SANDBOX, **data
    )
    if s1 is None:
        return None
    s2, _, _ = stage2_artifact.load_frozen_stage2_artifact(
        project_root=SANDBOX,
        **data,
        stage1_artifact=s1,
        attack_pre_assessment=s1["assessments"]["attack_feasibility"],
        defense_pre_assessment=s1["assessments"]["defense_robustness"],
    )
    if s2 is None:
        return None
    if not contextual:
        return {"stage1": s1, "stage2": s2}
    s3, _, _ = stage3_artifact.load_frozen_stage3_artifact(
        project_root=SANDBOX, evidence_pack=data["evidence_pack"],
        stage2_artifact=s2, stage0_config=config,
    )
    if s3 is None:
        return None
    s3_context, _, _ = stage3_context_artifact.load_frozen_stage3_context_artifact(
        project_root=SANDBOX, evidence_pack=data["evidence_pack"],
        base_stage3_artifact=s3,
    )
    if s3_context is None:
        return None
    return {
        "stage1": s1, "stage2": s2,
        "stage3": s3, "stage3_context": s3_context,
    }


def _stage4_completed(artifacts: dict) -> bool:
    frozen, _, _ = stage4_context_artifact.load_frozen_stage4_context_artifact(
        project_root=SANDBOX,
        stage3_context_artifact=artifacts["stage3_context"],
    )
    return frozen is not None

def _record(case_id: str, arm: str, seed: int, payload: dict, state: dict) -> dict:
    if not state.get("stage4_context_complete"):
        raise ValueError("Cannot finalize an incomplete Stage 4 evaluation")
    objects = {obj["decision_object_id"]: obj for obj in state["contextual_decision_objects"]}
    decisions = {}
    for raw in state["stage4_context_evaluation_bundle"]["evaluations"]:
        item = DecisionLensEvaluation.model_validate(raw)
        obj = objects[item.decision_object_id]
        recommendation, rule = _decision_recommendation(evaluation=item)
        decisions[obj["scenario_id"]] = {
            "recommendation": recommendation.value,
            "rule": rule,
            "stances": {
                "technical": item.technical_feasibility.stance,
                "institutional": item.institutional_regional.stance,
                "financial": item.financial_adoption.stance,
            },
            "assessment_sha256": sha256_json(raw),
            "evaluation_evidence_ids": sorted(obj["evaluation_evidence_ids"]),
        }
    if set(decisions) != {"KR__CIS_IG2", "EU__CIS_IG2", "US__CIS_IG2"}:
        raise ValueError("Missing scenario in completed Stage 4 bundle")
    artifacts = {
        stage: state[f"{stage}_artifact_path"]
        for stage in ("stage1", "stage2", "stage3", "stage3_context", "stage4_context")
    }
    for name, path in artifacts.items():
        if not Path(path).resolve().is_relative_to(SANDBOX.resolve()):
            raise ValueError(f"Non-isolated artifact path for {name}: {path}")
    return {
        "version": VERSION,
        "case_id": case_id,
        "arm": arm,
        "seed": seed,
        "evidence_sha256": sha256_json(payload["evidence_pack"]["evidence"]),
        "prompt_sha256": sha256_text(payload["forecast_data"]),
        "input_fingerprint": state["stage1_input_fingerprint"],
        "stage1": {
            "attack": state["attack_assessment"],
            "defense": state["defense_assessment"],
        },
        "stage2": state["stage2_debate_result"],
        "stage3_forecast_sha256": sha256_json(
            [obj["forecast_interpretation"] for obj in state["contextual_decision_objects"]]
        ),
        "stage4": state["stage4_context_evaluation_bundle"],
        "decisions": decisions,
        "artifacts": artifacts,
    }


def _record_path(case_id: str, arm: str, seed: int) -> Path:
    return OUTPUT / "records" / case_id / f"{arm}__seed-{seed}.json"


def _manifest(datasets: dict, provenance: dict) -> dict:
    return {
        "version": VERSION,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "result_root": str(OUTPUT),
        "artifact_root": str(SANDBOX),
        "cases": {
            case_id: {
                arm: {
                    "stage1_identity_seed_42": _identity_seed_42(datasets[case_id][arm]),
                    "evidence_pack_sha256": sha256_json(datasets[case_id][arm]["evidence_pack"]),
                    "prompt_sha256": sha256_text(datasets[case_id][arm]["forecast_data"]),
                    "provenance": provenance[case_id][arm],
                }
                for arm in ARMS
            }
            for case_id in datasets
        },
    }


def _identity_for_seed(data: dict, seed: int) -> str:
    profile = stage1_runtime.get_legacy_one_slot_runtime_profile()
    profile["seed"] = seed

    return build_stage1_identity(
        **data,
        runtime_profile=profile,
    )["input_fingerprint"]


def _identity_seed_42(data: dict) -> str:
    return _identity_for_seed(data, 42)


def _validate_record(record: dict, data: dict, case_id: str, arm: str, seed: int) -> None:
    expected = {
        "version": VERSION,
        "case_id": case_id,
        "arm": arm,
        "seed": seed,
        "input_fingerprint": _identity_for_seed(data, seed),
        "evidence_sha256": sha256_json(data["evidence_pack"]["evidence"]),
        "prompt_sha256": sha256_text(data["forecast_data"]),
    }

    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"Completed {case_id}/{arm}/seed-{seed} has mismatched {field}")

    if set(record.get("decisions", {})) != {"KR__CIS_IG2", "EU__CIS_IG2", "US__CIS_IG2"}:
        raise ValueError(f"Incomplete decisions for {case_id}/{arm}/seed-{seed}")

    if set(record.get("artifacts", {})) != {
        "stage1", "stage2", "stage3", "stage3_context", "stage4_context"
    }:
        raise ValueError(f"Incomplete stage paths for {case_id}/{arm}/seed-{seed}")

    artifacts = {}
    for stage, path in record["artifacts"].items():
        artifact = Path(path).resolve()
        if not artifact.is_relative_to(SANDBOX.resolve()) or not artifact.is_file():
            raise ValueError(f"Missing or nonisolated stage artifact: {artifact}")

        saved = json.loads(artifact.read_text(encoding="utf-8"))
        digest = saved.get("artifact_sha256")
        if digest != sha256_json({
            key: value for key, value in saved.items() if key != "artifact_sha256"
        }):
            raise ValueError(f"Stage artifact checksum mismatch: {artifact}")

        artifacts[stage] = saved

    if (
        record["stage1"]["attack"] != artifacts["stage1"]["assessments"]["attack_feasibility"]
        or record["stage1"]["defense"] != artifacts["stage1"]["assessments"]["defense_robustness"]
        or record["stage2"] != artifacts["stage2"]["debate_result"]
        or record["stage4"] != artifacts["stage4_context"]["evaluation_bundle"]
    ):
        raise ValueError(f"Stage result and completed record disagree: {case_id}/{arm}/seed-{seed}")


def _compare(case_id: str, records: dict) -> dict:
    result = {}
    for seed in SEEDS:
        a = records["original", seed]
        b = records["reversed_pmt", seed]
        if a["evidence_sha256"] != b["evidence_sha256"]:
            raise ValueError("Paired run evidence hash diverged")
        jurisdictions = {}
        for jurisdiction in a["decisions"]:
            original = a["decisions"][jurisdiction]
            modified = b["decisions"][jurisdiction]
            if original["evaluation_evidence_ids"] != modified["evaluation_evidence_ids"]:
                raise ValueError(f"Stage 4 evidential universe drifted: {case_id}/{jurisdiction}")
            jurisdictions[jurisdiction] = {
                "changed_stances": {
                    k: original["stances"][k] != modified["stances"][k]
                    for k in original["stances"]
                },
                "recommendation_changed": original["recommendation"] != modified["recommendation"],
                "recommendations": [original["recommendation"], modified["recommendation"]],
            }
        result[str(seed)] = {
            "critic_pre_stance_changed": {
                k: a["stage1"][k]["stance"] != b["stage1"][k]["stance"]
                for k in ("attack", "defense")
            },
            "critic_post_stance_changed": {
                k: a["stage2"][f"{k}_post_assessment"]["stance"]
                != b["stage2"][f"{k}_post_assessment"]["stance"]
                for k in ("attack", "defense")
            },
            "stage2_result_changed": sha256_json(a["stage2"]) != sha256_json(b["stage2"]),
            "jurisdictions": jurisdictions,
        }
    return result


def _within_arm_agreement(records: dict) -> dict:
    pairs = ((42, 43), (42, 44), (43, 44))
    result = {}
    for arm in ARMS:
        recommendation_same = []
        stance_same = []
        for first, second in pairs:
            a = records[arm, first]["decisions"]
            b = records[arm, second]["decisions"]
            for region in a:
                recommendation_same.append(a[region]["recommendation"] == b[region]["recommendation"])
                stance_same.append(a[region]["stances"] == b[region]["stances"])
        result[arm] = {
            "recommendation_agreement": sum(recommendation_same) / len(recommendation_same),
            "stance_tuple_agreement": sum(stance_same) / len(stance_same),
            "matched_comparisons": len(recommendation_same),
        }
    return result


def _report(case_ids: list[str]) -> Path:
    datasets, _ = _inputs(case_ids)
    manifest_name = (
        "experiment_manifest.json"
        if len(case_ids) == len(MAIN_STAGE1_CASES)
        else f"experiment_manifest__{case_ids[0]}.json"
    )
    manifest_path = OUTPUT / manifest_name
    manifest = _load_frozen(manifest_path)
    if manifest is None and len(case_ids) == 1:
        manifest_path = OUTPUT / "experiment_manifest.json"
        manifest = _load_frozen(manifest_path)
    if manifest is None or manifest.get("version") != VERSION:
        raise ValueError(f"Missing current frozen experiment manifest: {manifest_path}")
    cases = {}
    stability = {}
    for case_id in case_ids:
        records = {}
        for seed in SEEDS:
            for arm in ARMS:
                path = _record_path(case_id, arm, seed)
                record = _load_frozen(path)
                if record is None:
                    raise ValueError(f"Missing completed run: {path}")
                if (record["case_id"], record["arm"], record["seed"]) != (case_id, arm, seed):
                    raise ValueError(f"Wrong run identity: {path}")
                expected = manifest["cases"][case_id][arm]
                if (
                    record.get("version") != VERSION
                    or record.get("prompt_sha256") != expected["prompt_sha256"]
                    or record.get("evidence_sha256") != expected["provenance"]["evidence_sha256"]
                ):
                    raise ValueError(f"Run/manifest provenance mismatch: {path}")
                _validate_record(record, datasets[case_id][arm], case_id, arm, seed)
                records[arm, seed] = record
        cases[case_id] = _compare(case_id, records)
        stability[case_id] = _within_arm_agreement(records)
    report = {
        "version": VERSION,
        "independent_cases": len(case_ids),
        "repeats": list(SEEDS),
        "caution": (
            "Synthetic PMT stress test only. Do not treat dependent jurisdiction "
            "rows as independent observations or infer recommendation validity."
        ),
        "cases": cases,
        "within_arm_seed_agreement": stability,
    }
    filename = "summary.json" if len(case_ids) == len(MAIN_STAGE1_CASES) else f"summary__{case_ids[0]}.json"
    path = OUTPUT / filename
    _write_frozen(path, report)
    return path


async def _run(
    case_ids: list[str],
    datasets: dict,
    *,
    phase: str,
    seeds: tuple[int, ...] = SEEDS,
    continue_on_error: bool = False,
) -> list[dict]:
    if phase not in {"upstream", "downstream"}:
        raise ValueError("Select upstream or downstream phase explicitly")

    old_seed = os.environ.get("LLM_SEED")
    failures = []
    try:
        for case_id in case_ids:
            case = next(item for item in MAIN_STAGE1_CASES if item.case_id == case_id)
            config = _case_config(case)
            for seed in seeds:
                for arm in ARMS:
                    os.environ["LLM_SEED"] = str(seed)
                    payload = datasets[case_id][arm]

                    try:
                        if arm == "original":
                            _import_original(case_id, seed)

                        with ExitStack() as stack:
                            _isolated_nodes(stack)

                            upstream = _artifacts(payload, config, contextual=True)
                            if phase == "upstream":
                                if upstream is not None:
                                    print(f"Reusing frozen upstream: {case_id}/{arm}/seed-{seed}", flush=True)
                                    continue

                                if _artifacts(payload, config, contextual=False) is None:
                                    # Fail BEFORE a single model request on the
                                    # wrong server, not after hours of inference.
                                    stage1_runtime.assert_llama_server_ready(
                                        expected_context_size=131072,
                                        expected_parallel_slots=1,
                                        profile_name="Forecast link Stage 1/2 single-slot",
                                    )
                                _upstream_nodes(stack)
                            else:
                                if upstream is None:
                                    raise ValueError(
                                        f"Stage 1--3 not frozen for {case_id}/{arm}/seed-{seed}. "
                                        "Run --phase upstream using the one-slot server first."
                                    )
                                if not _stage4_completed(upstream):
                                    stage1_runtime.assert_llama_server_ready(
                                        expected_context_size=65536,
                                        expected_parallel_slots=2,
                                        profile_name="Forecast link Stage 4 two-slot",
                                    )

                            path = _record_path(case_id, arm, seed)
                            if phase == "downstream":
                                completed = _load_frozen(path)
                                if completed is not None:
                                    _validate_record(completed, payload, case_id, arm, seed)
                                    print(f"Reusing completed record: {case_id}/{arm}/seed-{seed}", flush=True)
                                    continue

                            state = await _graph(payload, phase=phase, arm=arm).ainvoke(
                                _initial_state(config),
                                config={"recursion_limit": 100},
                            )

                            if phase == "upstream":
                                if not state.get("stage3_context_complete"):
                                    raise ValueError("Upstream stage did not complete")
                                print(f"Completed frozen upstream: {case_id}/{arm}/seed-{seed}", flush=True)
                            else:
                                _write_frozen(
                                    path, _record(case_id, arm, seed, payload, state)
                                )
                                print(f"Completed final record: {case_id}/{arm}/seed-{seed}", flush=True)
                    except (RuntimeError, ValueError) as error:
                        if not continue_on_error:
                            raise
                        failure = {
                            "case_id": case_id,
                            "arm": arm,
                            "seed": seed,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                        failures.append(failure)
                        print(f"FAILED {case_id}/{arm}/seed-{seed}: {error}", flush=True)
    finally:
        if old_seed is None:
            os.environ.pop("LLM_SEED", None)
        else:
            os.environ["LLM_SEED"] = old_seed
    return failures


def _server() -> tuple[psutil.Process, list[str]]:
    listeners = {
        entry.pid
        for entry in psutil.net_connections(kind="tcp")
        if entry.status == psutil.CONN_LISTEN
        and entry.laddr
        and entry.laddr.port == 8080
        and entry.laddr.ip == "127.0.0.1"
        and entry.pid is not None
    }

    if len(listeners) != 1:
        raise RuntimeError(
            "Expected one llama.cpp server at 127.0.0.1:8080. "
            "Start the existing Stage1/start_qwen38.ps1 server first."
        )

    process = psutil.Process(next(iter(listeners)))
    command = process.cmdline()
    if not command or Path(process.exe()).name.lower() != "llama-server.exe":
        raise RuntimeError("Port 8080 is owned by an unexpected process")

    def flag(name: str) -> str:
        if command.count(name) != 1:
            raise RuntimeError(f"Missing or duplicated llama.cpp flag {name}")

        position = command.index(name)
        if position + 1 >= len(command):
            raise RuntimeError(f"Missing value for {name}")

        return command[position + 1]

    if (
        flag("--host") != "127.0.0.1"
        or flag("--port") != "8080"
        or flag("--ctx-size") != "131072"
        or flag("--parallel") not in {"1", "2"}
        or not flag("--model").endswith("Qwen3.8-27B-Q6_K_L.gguf")
    ):
        raise RuntimeError("The running llama.cpp server has a different launch contract")

    return process, command


def _ensure_idle() -> None:
    response = requests.get(f"{SERVER_URL}/slots", timeout=10)
    response.raise_for_status()
    slots = response.json()

    if not isinstance(slots, list) or not slots:
        raise RuntimeError("Cannot verify the server's inference slots")

    for slot in slots:
        if (
            not isinstance(slot, dict)
            or slot.get("is_processing") is not False
            or slot.get("state") not in (None, "", "idle")
        ):
            raise RuntimeError("llama.cpp is processing another request; refusing to stop it")


def _ready(parallel: int) -> None:
    stage1_runtime.assert_llama_server_ready(
        expected_context_size=SLOT_CONTEXT[parallel],
        expected_parallel_slots=parallel,
        profile_name=f"Forecast link controlled {parallel}-slot runtime",
    )


def _start_server(command: list[str], parallel: int) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT / f"llama_server_{parallel}slot.log"

    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT / "Multi-Agent",
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

    try:
        for _ in range(180):
            if process.poll() is not None:
                raise RuntimeError(f"llama.cpp exited during startup; inspect {log_path}")

            try:
                response = requests.get(f"{SERVER_URL}/health", timeout=2)
                if response.ok:
                    _ready(parallel)
                    return
            except (requests.RequestException, RuntimeError):
                pass

            time.sleep(1)

        raise RuntimeError(f"llama.cpp failed to become ready; inspect {log_path}")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        raise


def _switch_server(template: list[str], parallel: int) -> None:
    position = template.index("--parallel") + 1
    target = list(template)
    target[position] = str(parallel)

    try:
        process, current = _server()
    except RuntimeError:
        listeners = [
            entry
            for entry in psutil.net_connections(kind="tcp")
            if entry.status == psutil.CONN_LISTEN
            and entry.laddr
            and entry.laddr.port == 8080
        ]
        if listeners:
            raise

        _start_server(target, parallel)
        return

    if current == target:
        _ready(parallel)
        return

    old_parallel = int(current[current.index("--parallel") + 1])
    reference = list(template)
    reference[position] = str(old_parallel)
    if current != reference:
        raise RuntimeError("The llama.cpp launch settings changed; refusing to restart it")

    _ready(old_parallel)
    _ensure_idle()
    print(f"Switching idle llama.cpp server: {old_parallel} -> {parallel} slot(s)", flush=True)

    process.terminate()
    process.wait(timeout=30)
    _start_server(target, parallel)


def _save_failures(phase: str, failures: list[dict]) -> None:
    path = OUTPUT / f"failed_{phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json({"version": VERSION, "failures": failures}) + "\n", encoding="utf-8")


def _run_all(case_ids: list[str], datasets: dict, seeds: tuple[int, ...]) -> None:
    process, template = _server()
    original_parallel = int(template[template.index("--parallel") + 1])
    _ready(original_parallel)

    try:
        _switch_server(template, 1)
        upstream = asyncio.run(
            _run(case_ids, datasets, phase="upstream", seeds=seeds, continue_on_error=True)
        )
        _save_failures("upstream", upstream)

        _switch_server(template, 2)
        downstream = asyncio.run(
            _run(case_ids, datasets, phase="downstream", seeds=seeds, continue_on_error=True)
        )
        _save_failures("downstream", downstream)

        if upstream or downstream:
            raise RuntimeError(
                f"{len(upstream)} upstream and {len(downstream)} downstream runs failed. "
                f"Inspect {OUTPUT / 'failed_upstream.json'} and failed_downstream.json. "
                "Rerunning preserves completed checkpoints."
            )

        if seeds == SEEDS:
            print(f"Completed comparison: {_report(case_ids)}", flush=True)
        else:
            print("Selected-seed run complete. All three seeds are required for a report.", flush=True)
    finally:
        _switch_server(template, original_parallel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--check", action="store_true", help="Offline input verification only")
    commands.add_argument("--run", action="store_true", help="Run all missing cases, automatically switching the server")
    commands.add_argument("--report", action="store_true", help="Summarize completed matched runs")
    parser.add_argument("--case", choices=[item.case_id for item in MAIN_STAGE1_CASES])
    parser.add_argument("--seed", type=int, choices=SEEDS, help="Run only one paired seed (pilot)")
    args = parser.parse_args()

    if args.seed is not None and not args.run:
        parser.error("--seed applies only to --run")

    case_ids = [args.case] if args.case else [case.case_id for case in MAIN_STAGE1_CASES]
    if args.report:
        print(f"Summary: {_report(case_ids)}")
        return

    datasets, provenance = _inputs(case_ids)
    print(
        f"Verified {len(case_ids)} frozen cases and "
        f"{len(case_ids) * len(SEEDS) * len(ARMS)} arm-seed inputs; "
        "only PMT forecasts are counterfactually modified.",
        flush=True,
    )

    for case_id in case_ids:
        _main_reference_paths(case_id)

    if args.check:
        return

    manifest_name = (
        "experiment_manifest.json"
        if len(case_ids) == len(MAIN_STAGE1_CASES)
        else f"experiment_manifest__{case_ids[0]}.json"
    )
    _write_frozen(OUTPUT / manifest_name, _manifest(datasets, provenance))
    _run_all(case_ids, datasets, (args.seed,) if args.seed is not None else SEEDS)


if __name__ == "__main__":
    main()
