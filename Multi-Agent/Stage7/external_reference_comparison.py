"""Reviewer 2, final concern: external-reference and report-comparison study.

ONE standalone experiment script; does not alter Stage 1--7 artifacts or policy.

Run from Multi-Agent:
  py -3 -m Stage7.external_reference_comparison --prepare
  py -3 -m Stage7.external_reference_comparison --baseline --seeds 42,43,44
  py -3 -m Stage7.external_reference_comparison --audit
  py -3 -m Stage7.external_reference_comparison --summary
  py -3 -m Stage7.external_reference_comparison --all --seeds 42,43,44

The official CIS Controls v8.1 Navigator is fetched once and frozen. To use a
locally saved copy when offline, pass --cis-html PATH to --prepare.

Important: CIS lookup is an EXTERNAL TECHNIQUE REFERENCE, not a four-class
recommendation oracle. Gemma report scores are LLM assessments, not expert
ground truth. No human panel or independent correctness metric is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import requests
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from Stage1.runtime import (
    LlamaCppChatClient,
    assert_llama_server_ready,
    get_runtime_config,
)
from Stage7.audits import extract_all_grounding_items, select_manual_audit_sample
from Stage7.gemma_server import use_gemma
from Stage7.loaders import (
    canonical_json,
    load_main_case_bundles,
    sha256_json,
    sha256_text,
)
from Stage7.mediator_ablation import _freeze, _load_checked
from Stage7.mediator_quality_analysis import (
    MODEL as GEMMA_MODEL,
    MODEL_SHA256 as GEMMA_SHA,
    URL as CHAT_URL,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Multi-Agent" / "Results" / "Stage7" / "external_reference_comparison"
VERSION = "reviewer2-external-reference-report-audit-v1"
CIS_URL = "https://www.cisecurity.org/controls/CIS-controls-navigator"

# This direct baseline sees the same FROZEN forecast, context and evidence,
# but no upstream Critic/Mediator judgments, three-lens stances, or answers.
DIRECT_SYSTEM = """You are a single, end-to-end cybersecurity foresight analyst.
Independently evaluate the preselected candidate mitigation for the given
forecast, reference organization and exact evidence corpus. Make ONE of:
RECOMMEND, RECOMMEND_PILOT, HOLD, DO_NOT_RECOMMEND. The candidate is fixed,
so do not select a different mitigation or infer a ground-truth label.
Write a substantive strategic decision report explaining evidence supporting
and challenging technical feasibility, institutional/regional applicability,
and financial/adoption feasibility; forecast uncertainty; missing deployment
information; dissent and decision conditions. Be critical and evidence-bounded.
Use only the supplied evidence records. Cite exact record IDs inline as
[EVIDENCE:ID]. Do not invent ROI, costs, budgets, incident-time trends,
causality, vendor performance, legal duties or enterprise deployment facts.
GENERAL evidence does not prove deployment-specific effectiveness. If the
evidence is insufficient for a full recommendation, use a qualified decision.
Do not assume the existence of a committee, extra agents or their decisions.
Return strictly the required JSON object, not markdown."""

# Audit each report separately, blinded to which pipeline produced it.  The
# SAME rubric is applied to both arms; no labels or other arm is shown.
REPORT_SYSTEM = """You are a separate, evidence-bounded model evaluator.
Evaluate ONE ANONYMOUS strategic cybersecurity report, NOT whether you agree
with its chosen action. Use ONLY the included forecast, organization context,
full report, and the frozen complete evidence universe. The cited subset is
identified separately, so material counter-evidence and unsupported
omissions can be checked without using external knowledge.
Consider: (1) fidelity of report propositions to cited sources, (2) whether
technical and population scope is transferred without support, (3) whether
forecast modality/time-series limits are preserved, (4) whether constraints
and unknown budget/ROI/deployment details are acknowledged, (5) whether the
report's own recommendation and narrative are internally coherent.
Scores are ordinal: 0=material unresolved problem, 1=mixed/qualified,
2=no material problem identified from supplied boundary. Grade every area.
Do NOT require a RECOMMEND and do NOT award points for choosing HOLD.
Do NOT require the report to cite every record supplied; consider omissions
only if they make a material assertion misleading.
Statements about the report author's internal evaluation or policy are
NOT external empirical claims; do not call them contradicted merely because
the outside documents do not narrate those internal steps. Without the
internal trace, mark such assertions unevaluable, not externally false.
For each of up to five concrete issues, cite an exact report phrase or
documented source mismatch, not speculation. Unknown evidence IDs are invalid.
You are NOT a human expert and these ordinal scores are NOT correctness
ground truth. Return strictly the required JSON object."""

CIS_SYSTEM = """You audit a deterministic lexical lookup against exact
official CIS Controls v8.1 Safeguard text. For a technique_lookup, assess
relevance to the supplied candidate technique. For a threat_lookup, assess
whether the safeguard directly addresses the supplied threat, NOT a
preselected mitigation. DIRECT means the safeguard explicitly describes
the target or its exact class, ADJACENT means a relevant but different
control or prerequisite, NOT_RELEVANT means no material relevance.
CIS IG2 is an implementation-priority guide, not ground truth for a
four-class recommendation, enterprise deployment effectiveness or ROI.
Use only supplied descriptions, and return the required JSON object."""

_INLINE_ID = re.compile(r"EVIDENCE:([A-Za-z0-9_.:-]+)")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "is",
    "of", "on", "or", "the", "to", "use", "using", "with", "implement",
    "maintain", "enterprise", "assets", "security", "controls", "control",
}
_ALIASES = {
    "ids": "intrusion detection",
    "ips": "intrusion prevention",
    "https": "transport layer security tls encrypted transit",
    "cryptography": "encryption encrypt cryptographic",
    "anomaly": "behavior based detection abnormal events",
    "malware": "anti malware software",
    "access": "access authentication authorization",
    "nlp": "natural language processing model",
    "llm": "large language model",
}


class DirectReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recommendation: Literal[
        "RECOMMEND", "RECOMMEND_PILOT", "HOLD", "DO_NOT_RECOMMEND",
    ]
    report: str = Field(min_length=400, max_length=16000)


class ReportGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")
    factual_grounding: Literal[0, 1, 2]
    technical_scope: Literal[0, 1, 2]
    forecast_modality: Literal[0, 1, 2]
    constraints_and_uncertainty: Literal[0, 1, 2]
    internal_decision_coherence: Literal[0, 1, 2]
    material_issues: list[str] = Field(max_length=5)
    rationale: str = Field(min_length=40, max_length=900)


class CISRelevance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    safeguard_id: str
    relevance: Literal["DIRECT", "ADJACENT", "NOT_RELEVANT"]
    rationale: str = Field(min_length=12, max_length=280)


class CISGrades(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mappings: list[CISRelevance]


def _bundles() -> list:
    bundles = load_main_case_bundles(ROOT)
    if len(bundles) != 7 or sum(len(b.syntheses) for b in bundles) != 21:
        raise ValueError("Expected exact frozen seven cases / 21 reports")
    return bundles


def _scenario_rows() -> list[dict]:
    rows = []
    for bundle in _bundles():
        objects = {x["scenario_id"]: x for x in bundle.contextual_objects}
        if set(objects) != {s["scenario_id"] for s in bundle.syntheses}:
            raise ValueError("Unmatched contextual objects and final reports")
        for synthesis in bundle.syntheses:
            obj = objects[synthesis["scenario_id"]]
            if synthesis["decision_object_id"] != obj["decision_object_id"]:
                raise ValueError("Misaligned frozen decision object IDs")
            rows.append({
                "case_id": bundle.case_id,
                "scenario_id": obj["scenario_id"],
                "original": synthesis,
                "object": obj,
                "source_hashes": bundle.source_hashes(),
            })
    return rows


def _direct_input(row: dict) -> dict:
    obj = row["object"]
    return {
        "threat": obj["threat"],
        "candidate_action": obj["candidate_action"]["action_name"],
        "forecast": obj["forecast_interpretation"],
        "predictive_uncertainty": obj["predictive_uncertainty"],
        "timing_window": obj["timing_window"],
        "reference_enterprise": obj["context_scenario"],
        "deployment_context": obj["deployment_context"],
        "evidence_records": obj["evaluation_evidence_records"],
    }


def _name(case_id: str, scenario_id: str, seed: int) -> str:
    return f"{case_id}__{scenario_id}__seed-{seed}"


def _check(path: Path, identity: dict) -> dict | None:
    value = _load_checked(path)
    if value is not None and value["identity"] != identity:
        raise ValueError(f"Stale/checkpoint identity mismatch: {path}")
    return value


def _records_for_report(row: dict, report: str) -> tuple[list[dict], list[str]]:
    allowed = {x["evidence_id"]: x for x in row["object"]["evaluation_evidence_records"]}
    cited = set(_INLINE_ID.findall(report))
    return (
        [allowed[eid] for eid in sorted(cited & set(allowed))],
        sorted(cited - set(allowed)),
    )


def _get_artifact_70() -> dict:
    choices = []
    for path in (ROOT / "Multi-Agent/Results/Stage7/artifacts").glob("*.json"):
        payload = __import__("json").loads(path.read_text(encoding="utf-8"))
        audits = payload.get("grounding_audits", [])
        if len(audits) != 70:
            continue
        if Counter(a["verdict"] for a in audits) != {
            "DIRECT_SUPPORT": 57, "PARTIAL_SUPPORT": 12, "CONTRADICTS": 1,
        }:
            continue
        # Historical Stage 7 models included artifact_sha256=None when the
        # digest was calculated, before replacing it with the final digest.
        # Reproduce the *actual frozen writer* instead of the different
        # Stage 1--6 artifact verification convention.
        if payload.get("artifact_sha256") != sha256_json({**payload, "artifact_sha256": None}):
            raise ValueError(f"Stage 7 artifact digest mismatch: {path}")
        choices.append(payload)
    if not choices:
        raise ValueError("Cannot find verified original 70-audit artifact")
    # Identical audit judgments may be embedded in different reporting revisions.
    signature = sha256_json(sorted(
        choices[0]["grounding_audits"], key=lambda x: x["audit_id"]
    ))
    if any(sha256_json(sorted(x["grounding_audits"], key=lambda y: y["audit_id"])) != signature
           for x in choices[1:]):
        raise ValueError("Historical 70-audit artifacts disagree")
    return choices[0]


def _existing_audit_report() -> dict:
    items = extract_all_grounding_items(_bundles())
    chosen = select_manual_audit_sample(items)
    fixed = _get_artifact_70()
    original = {x["audit_id"]: x for x in fixed["grounding_audits"]}
    if set(original) != {x.audit_id for x in chosen}:
        raise ValueError("Frozen 70 audit IDs differ from the predeclared sample")
    details = []
    for item in chosen:
        verdict = original[item.audit_id]
        details.append({
            "audit_id": item.audit_id,
            "source_layer": {
                "Stage1": "forecast_critic",
                "Stage4": "decision_lens",
                "Stage6": "final_report_cited_sentence",
            }[item.source_stage],
            "case_id": item.case_id,
            "scenario_id": item.scenario_id,
            "lens": item.lens,
            "claim_id": item.claim_id,
            "statement": item.statement,
            "evidence_ids": item.evidence_ids,
            "verdict": verdict,
        })
    contradictions = [x for x in details if x["verdict"]["verdict"] == "CONTRADICTS"]
    if len(contradictions) != 1:
        raise ValueError("Expected the historical one CONTRADICTS audit")
    only = contradictions[0]
    if only["case_id"] != "main__brute_force__ids_ips" or only["source_layer"] != "final_report_cited_sentence":
        raise ValueError("Unexpected contradiction identity")
    cited = next(x for x in chosen if x.audit_id == only["audit_id"])
    return {
        "version": VERSION,
        "source_model": fixed["runtime"]["model"],
        "source_model_sha256": fixed["runtime"]["expected_model_sha256"],
        "evaluator_type": "same-family Qwen LLM, NOT human/external ground truth",
        "audit_population_count": len(items),
        "population_by_layer": dict(Counter(x.source_stage for x in items)),
        "selection_seed": 20260916,
        "selection_rule": "two smallest SHA256(seed|audit_id) per (source_layer,case,lens-or-NO_LENS), before judging",
        "selected_count": len(chosen),
        "by_layer_verdict": {
            label: dict(Counter(x["verdict"]["verdict"] for x in details if x["source_layer"] == label))
            for label in ("forecast_critic", "decision_lens", "final_report_cited_sentence")
        },
        "all_verdicts": dict(Counter(x["verdict"]["verdict"] for x in details)),
        "contradiction_dossier": {
            "audit": only,
            "exact_cited_records": cited.evidence_records,
            "counter_evidence_records": cited.counter_evidence_records,
            "inspection_note": (
                "DBIR reports one-year basic-web-attack counts; Mandiant reports a "
                "one-year initial-infection-vector mix, not a multi-year brute-force "
                "incident-count trajectory. The claim denies direct temporally "
                "aligned count/frequency evidence. Thus the verifier's CONTRADICTS "
                "rationale does not itself establish a factual contradiction. "
                "Separate statements about the internal adjudication are not "
                "verifiable from these two external records alone. This is a "
                "source-inspection interpretation, NOT a new expert label."
            ),
        },
        "audit_items": details,
    }


def _parse_cis_html(raw: bytes) -> dict:
    soup = BeautifulSoup(raw, "html.parser")
    safeguards = []
    for element in soup.select("div.safeguard"):
        label = element.select_one(".safeguard-title label")
        description = element.select_one(".safeguard-description")
        if label is None or description is None:
            continue
        found = re.fullmatch(
            r"Safeguard\s+(\d+\.\d+):\s*(.+)",
            label.get_text(" ", strip=True),
        )
        if found is None:
            continue
        safeguards.append({
            "id": found.group(1),
            "title": found.group(2),
            "description": description.get_text(" ", strip=True),
            "implementation_groups": [
                item.get_text(" ", strip=True)
                for item in element.select(".igs .ig")
            ],
        })
    if len(safeguards) != 153 or len({x["id"] for x in safeguards}) != 153:
        raise ValueError("CIS official Navigator parse did not yield 153 unique safeguards")
    if next(x for x in safeguards if x["id"] == "3.10")["title"] != "Encrypt Sensitive Data in Transit":
        raise ValueError("Unexpected CIS Navigator version/layout")
    if next(x for x in safeguards if x["id"] == "10.7")["title"] != "Use Behavior-Based Anti-Malware Software":
        raise ValueError("Expected v8.1 control definitions not found")
    return {
        "version": "CIS Controls v8.1",
        "source_url": CIS_URL,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_html_sha256": hashlib.sha256(raw).hexdigest(),
        "safeguards": safeguards,
        "limitation": "Official safeguard guidance, not observed deployment success or a four-class recommendation oracle.",
    }


def _words(value: str) -> list[str]:
    tokens = _TOKEN.findall(value.lower())
    words = []
    for token in tokens:
        words.append(token)
        words.extend(_TOKEN.findall(_ALIASES.get(token, "")))
    return [
        w for w in words
        if len(w) > 2 and w not in _STOP
        and w not in {"processing", "model", "large", "language", "natural"}
    ]


def _cis_lookup(rows: list[dict], cis: dict) -> list[dict]:
    candidates = [s for s in cis["safeguards"] if "IG2" in s["implementation_groups"]]
    docs = [Counter(_words(s["title"] + " " + s["description"])) for s in candidates]
    df = Counter(w for doc in docs for w in doc)
    avg_length = sum(sum(doc.values()) for doc in docs) / len(docs)

    def retrieve(query: str) -> list[dict]:
        q = Counter(_words(query))
        scores = []
        for safeguard, document in zip(candidates, docs):
            length = sum(document.values())
            score = 0.0
            for term, q_count in q.items():
                if term not in document:
                    continue
                idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                tf = document[term]
                score += min(q_count, 2) * idf * (
                    tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / avg_length))
                )
            if score > 0:
                scores.append((score, safeguard))
        scores.sort(key=lambda x: (-x[0], tuple(int(z) for z in x[1]["id"].split("."))))
        return [{**s, "lexical_bm25_score": round(score, 6)} for score, s in scores[:5]]

    result = []
    seen = set()
    for row in rows:
        case = row["case_id"]
        if case in seen:
            continue
        seen.add(case)
        obj = row["object"]
        result.append({
            "case_id": case,
            "threat": obj["threat"],
            "candidate_action": obj["candidate_action"]["action_name"],
            "technique_ig2_safeguards": retrieve(obj["candidate_action"]["action_name"]),
            "threat_ig2_safeguards": retrieve(obj["threat"]),
            "retrieval_rule": "Separate technique-only and threat-only BM25 over exact CIS v8.1 IG2 safeguards with fixed vocabulary aliases; no recommendation labels",
        })
    return result


def prepare(cis_html: Path | None = None) -> dict:
    _freeze(OUTPUT / "original_70_audit.json", _existing_audit_report())
    path = OUTPUT / "cis_v81_official_inventory.json"
    cis = _load_checked(path)
    if cis is None:
        if cis_html is not None:
            raw = cis_html.read_bytes()
        else:
            resp = requests.get(CIS_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            raw = resp.content
        cis = _freeze(path, _parse_cis_html(raw))
    _freeze(OUTPUT / "cis_ig2_lookups.json", {
        "version": VERSION,
        "cis_html_sha256": cis["source_html_sha256"],
        "lookup_rows": _cis_lookup(_scenario_rows(), cis),
        "interpretation": "Two lookup tasks: threat-only safeguard retrieval and candidate-technique applicability; neither is a four-class recommendation oracle.",
    })
    print("Prepared: 70-audit provenance/dossier and 153 official CIS v8.1 safeguards")
    return cis


def _qwen_identity(row: dict, seed: int) -> dict:
    cfg = replace(get_runtime_config(), seed=seed)
    return {
        "version": VERSION,
        "arm": "single_direct_end_to_end",
        "case_id": row["case_id"],
        "scenario_id": row["scenario_id"],
        "seed": seed,
        "model": cfg.model,
        "model_sha256": cfg.expected_model_sha256,
        "generation_config": {
            "temperature": cfg.temperature, "top_p": cfg.top_p,
            "top_k": cfg.top_k, "min_p": cfg.min_p,
            "reasoning_effort": cfg.reasoning_effort,
        },
        "source_hashes": row["source_hashes"],
        "input_sha256": sha256_json(_direct_input(row)),
        "prompt_sha256": sha256_text(DIRECT_SYSTEM),
        "schema_sha256": sha256_json(DirectReport.model_json_schema()),
    }


def _baseline_path(row: dict, seed: int) -> Path:
    return OUTPUT / "direct_baseline" / (_name(row["case_id"], row["scenario_id"], seed) + ".json")


def run_baseline(seeds: list[int]) -> None:
    rows = _scenario_rows()
    requests_to_make = [
        (row, seed) for seed in seeds for row in rows
        if _check(_baseline_path(row, seed), _qwen_identity(row, seed)) is None
    ]
    print(f"Direct Qwen baseline: {len(requests_to_make)} missing / {len(rows)*len(seeds)}", flush=True)
    if not requests_to_make:
        return
    assert_llama_server_ready(
        expected_context_size=65536,
        expected_parallel_slots=2,
        profile_name="Qwen direct end-to-end single-LLM comparison",
    )
    for index, (row, seed) in enumerate(requests_to_make, start=1):
        cfg = replace(get_runtime_config(), seed=seed)
        client = LlamaCppChatClient(config=cfg, request_concurrency=1)
        result = client.with_structured_output(
            DirectReport,
            method="json_schema",
            progress_label=f"Direct-e2e-{index}/{len(requests_to_make)}",
        ).invoke([
            SystemMessage(content=DIRECT_SYSTEM),
            HumanMessage(content=canonical_json(_direct_input(row))),
        ])
        _freeze(_baseline_path(row, seed), {
            "identity": _qwen_identity(row, seed),
            "output": result.model_dump(mode="json"),
        })
        print(f"Direct baseline {index}/{len(requests_to_make)}: {row['case_id']} {row['scenario_id']} seed={seed}", flush=True)


def _report_input(row: dict, report: str, recommendation: str) -> dict:
    evidence, missing = _records_for_report(row, report)
    return {
        "forecast": row["object"]["forecast_interpretation"],
        "candidate_action": row["object"]["candidate_action"]["action_name"],
        "reference_enterprise": row["object"]["context_scenario"],
        "recommendation": recommendation,
        "report": report,
        "complete_frozen_evidence_records": row["object"]["evaluation_evidence_records"],
        "cited_evidence_records": evidence,
        "invalid_or_unknown_citations": missing,
    }


def _citation_statistics(row: dict, report: str) -> dict:
    records, unknown = _records_for_report(row, report)
    citation_mentions = _INLINE_ID.findall(report)
    return {
        "citation_mentions": len(citation_mentions),
        "unique_cited_record_count": len(set(citation_mentions)),
        "known_cited_record_count": len(records),
        "unknown_cited_ids": unknown,
        "any_citation": bool(citation_mentions),
        "report_characters": len(report),
    }


def _audit_identity(kind: str, key: str, payload: dict) -> dict:
    return {
        "version": VERSION,
        "kind": kind,
        "key": key,
        "model": GEMMA_MODEL,
        "model_sha256": GEMMA_SHA,
        "prompt_sha256": sha256_text(REPORT_SYSTEM if kind == "report" else CIS_SYSTEM),
        "schema_sha256": sha256_json(
            ReportGrade.model_json_schema() if kind == "report" else CISGrades.model_json_schema()
        ),
        "input_sha256": sha256_json(payload),
        "temperature": 0,
        "seed": 42,
    }


def _query_gemma(system: str, payload: dict, model: type[BaseModel]) -> BaseModel:
    # The original 3,000-token limit could truncate an otherwise valid JSON
    # answer inside a quoted issue/rationale string. Retry only unfinished or
    # invalid responses; completed checkpoints remain frozen and reusable.
    for max_tokens in (8192, 12288):
        response = requests.post(
            CHAT_URL,
            json={
                "model": GEMMA_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": canonical_json(payload)},
                ],
                "seed": 42,
                "temperature": 0,
                "max_tokens": max_tokens,
                "response_format": {
                    "type": "json_object",
                    "schema": model.model_json_schema(),
                },
            },
            timeout=1800,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        finish_reason = choice.get("finish_reason")
        if finish_reason in ("length", "max_tokens"):
            print(f"Gemma output truncated at {max_tokens} tokens; retrying if possible", flush=True)
            continue
        try:
            return model.model_validate_json(choice["message"]["content"])
        except (ValueError, KeyError, TypeError) as exc:
            print(f"Gemma JSON/schema invalid at {max_tokens} tokens: {str(exc)[:180]}", flush=True)
    raise RuntimeError("Gemma returned no complete schema-valid response after two attempts; checkpoint not written")


def _audit_cases() -> list[dict]:
    cases = []
    for row in _scenario_rows():
        baseline = _check(_baseline_path(row, 42), _qwen_identity(row, 42))
        if baseline is None:
            raise ValueError("Seed 42 direct baseline incomplete; run --baseline first")
        arms = {
            "original": (
                row["original"]["strategic_intelligence_report"],
                row["original"]["decision_recommendation"],
            ),
            "direct": (
                baseline["output"]["report"],
                baseline["output"]["recommendation"],
            ),
        }
        for arm, (report, recommendation) in arms.items():
            key = _name(row["case_id"], row["scenario_id"], 42) + "__" + arm
            payload = _report_input(row, report, recommendation)
            cases.append({
                "kind": "report",
                "key": key,
                "payload": payload,
                "system": REPORT_SYSTEM,
                "schema": ReportGrade,
            })
    cis = _load_checked(OUTPUT / "cis_ig2_lookups.json")
    if cis is None:
        raise ValueError("CIS lookup incomplete; run --prepare first")
    for item in cis["lookup_rows"]:
        for lookup_type, field, target in (
            ("technique_lookup", "technique_ig2_safeguards", item["candidate_action"]),
            ("threat_lookup", "threat_ig2_safeguards", item["threat"]),
        ):
            cases.append({
                "kind": "cis",
                "key": item["case_id"] + "__" + lookup_type,
                "payload": {
                    "lookup_type": lookup_type,
                    "target": target,
                    "safeguards": [
                        {"id": s["id"], "title": s["title"], "description": s["description"]}
                        for s in item[field]
                    ],
                },
                "system": CIS_SYSTEM,
                "schema": CISGrades,
            })
    return cases


def _judgment_path(case: dict) -> Path:
    return OUTPUT / "independent_model_audit" / case["kind"] / (case["key"] + ".json")


def _valid_response(case: dict, value: BaseModel) -> None:
    if case["kind"] == "cis":
        expected = [s["id"] for s in case["payload"]["safeguards"]]
        actual = [s.safeguard_id for s in value.mappings]
        if len(actual) != len(expected) or set(actual) != set(expected):
            raise ValueError(f"CIS model omitted/duplicated safeguard IDs: {case['key']}")


def run_audit() -> None:
    cases = _audit_cases()
    pending = [
        c for c in cases
        if _check(_judgment_path(c), _audit_identity(c["kind"], c["key"], c["payload"])) is None
    ]
    # Balance original and direct order without disclosing arms to the model.
    pending.sort(key=lambda x: sha256_text(VERSION + "|" + x["key"]))
    print(f"Gemma blind audit: {len(pending)} missing / {len(cases)} (42 report + 14 CIS relevance)", flush=True)
    if not pending:
        return
    with use_gemma():
        for index, case in enumerate(pending, start=1):
            # No retrieved Safeguards means a documented empty lookup,
            # not a reason to ask a model to invent a match.
            if case["kind"] == "cis" and not case["payload"]["safeguards"]:
                result = CISGrades(mappings=[])
            else:
                result = _query_gemma(case["system"], case["payload"], case["schema"])
            _valid_response(case, result)
            _freeze(_judgment_path(case), {
                "identity": _audit_identity(case["kind"], case["key"], case["payload"]),
                "judgment": result.model_dump(mode="json"),
            })
            print(f"Gemma audit {index}/{len(pending)}: {case['key']}", flush=True)


def summary(seeds: list[int]) -> dict:
    rows = _scenario_rows()
    cis = _load_checked(OUTPUT / "cis_ig2_lookups.json")
    old = _load_checked(OUTPUT / "original_70_audit.json")
    if cis is None or old is None:
        raise ValueError("Prepare reports missing")
    audits = _audit_cases()
    judgments = {
        (c["kind"], c["key"]): _check(
            _judgment_path(c), _audit_identity(c["kind"], c["key"], c["payload"])
        )
        for c in audits
    }
    if any(v is None for v in judgments.values()):
        raise ValueError("Gemma report/CIS assessments incomplete; run --audit")
    paired = []
    for row in rows:
        key = _name(row["case_id"], row["scenario_id"], 42)
        original = judgments["report", key + "__original"]["judgment"]
        direct = judgments["report", key + "__direct"]["judgment"]
        baseline = _check(_baseline_path(row, 42), _qwen_identity(row, 42))
        paired.append({
            "case_id": row["case_id"],
            "scenario_id": row["scenario_id"],
            "original_recommendation": row["original"]["decision_recommendation"],
            "direct_recommendation": baseline["output"]["recommendation"],
            "recommendation_agrees": (
                row["original"]["decision_recommendation"] == baseline["output"]["recommendation"]
            ),
            "scores_original": original,
            "scores_direct": direct,
            "citation_checks_original": _citation_statistics(
                row, row["original"]["strategic_intelligence_report"]
            ),
            "citation_checks_direct": _citation_statistics(
                row, baseline["output"]["report"]
            ),
        })
    score_fields = (
        "factual_grounding", "technical_scope", "forecast_modality",
        "constraints_and_uncertainty", "internal_decision_coherence",
    )
    means = {}
    for field in score_fields:
        means[field] = {
            "multi_agent": sum(row["scores_original"][field] for row in paired) / len(paired),
            "direct_llm": sum(row["scores_direct"][field] for row in paired) / len(paired),
            "case_cluster_note": "Descriptive means only; KR/EU/US share case inputs.",
        }
    case_profiles = []
    for case in sorted({r["case_id"] for r in paired}):
        members = [r for r in paired if r["case_id"] == case]
        if len(members) != 3:
            raise ValueError("Expected exactly 3 jurisdictional rows per case")
        case_profiles.append({
            "case_id": case,
            "paired_scenarios": 3,
            "model_score_differences_multi_minus_direct": {
                field: sum(
                    r["scores_original"][field] - r["scores_direct"][field]
                    for r in members
                ) / 3
                for field in score_fields
            },
            "recommendation_agreement_count": sum(r["recommendation_agrees"] for r in members),
        })
    cis_scores = []
    for item in cis["lookup_rows"]:
        tech = judgments["cis", item["case_id"] + "__technique_lookup"]["judgment"]["mappings"]
        threat = judgments["cis", item["case_id"] + "__threat_lookup"]["judgment"]["mappings"]
        cis_scores.append({
            **item,
            "technique_relevance": tech,
            "threat_relevance": threat,
            "technique_direct_count": sum(x["relevance"] == "DIRECT" for x in tech),
            "threat_direct_count": sum(x["relevance"] == "DIRECT" for x in threat),
        })
    seed_summary = []
    for seed in seeds:
        recs = []
        for row in rows:
            value = _check(_baseline_path(row, seed), _qwen_identity(row, seed))
            if value is None:
                raise ValueError(f"Direct baseline incomplete for seed {seed}")
            recs.append({
                "case_id": row["case_id"],
                "scenario_id": row["scenario_id"],
                "recommendation": value["output"]["recommendation"],
            })
        seed_summary.append({"seed": seed, "rows": recs})
    seed_pairwise = []
    for index, left in enumerate(seed_summary):
        for right in seed_summary[index + 1:]:
            changes = [
                a["recommendation"] != b["recommendation"]
                for a, b in zip(left["rows"], right["rows"])
            ]
            seed_pairwise.append({
                "seed_pair": [left["seed"], right["seed"]],
                "scenario_changes": sum(changes),
                "case_clusters_with_any_change": sum(any(changes[i:i+3]) for i in range(0, 21, 3)),
            })
    result = {
        "version": VERSION,
        "evaluator": "Gemma4-31B-it-Q6_K_L: independent model, not expert correctness labels",
        "source_evidence_contract": "Same frozen forecast, enterprise context and evidence records; original policy retained only in original arm",
        "original_semantic_audit": {
            "count": old["selected_count"],
            "by_layer_verdict": old["by_layer_verdict"],
            "contradiction_id": old["contradiction_dossier"]["audit"]["audit_id"],
        },
        "cis_reference": cis_scores,
        "paired_report_assessments": paired,
        "mean_ordinal_model_scores": means,
        "case_cluster_score_differences": case_profiles,
        "recommendation_agreement": sum(row["recommendation_agrees"] for row in paired),
        "recommendation_comparisons": len(paired),
        "deterministic_citation_diagnostics": {
            kind: {
                "reports_without_citations": sum(
                    not row["citation_checks_" + kind]["any_citation"] for row in paired
                ),
                "reports_with_unknown_citations": sum(
                    bool(row["citation_checks_" + kind]["unknown_cited_ids"]) for row in paired
                ),
            }
            for kind in ("original", "direct")
        },
        "direct_baseline_seed_pairs": seed_pairwise,
        "limitations": (
            "CIS lexical retrieval is not an actionable four-class gold policy. "
            "Gemma is an LLM judge, not an external expert. "
            "Score differences do not establish accuracy superiority. "
            "The direct baseline and original architecture have differing "
            "access to intermediate deliberation by design. "
            "Baseline reports for seeds 43/44 have no matched regenerated "
            "full-pipeline reports; report-quality comparisons use seed 42 only. "
            "Seven cases and overlapping jurisdiction/seed rows are not "
            "independent observations. No practitioner panel was conducted."
        ),
    }
    _freeze(OUTPUT / ("summary_seeds_" + "_".join(map(str, seeds)) + ".json"), result)
    print("Report comparison complete: 21 paired reports; 7 technique + 7 threat CIS lookups")
    print("Model-rated means:", means)
    print("Direct-baseline seed pairs:", seed_pairwise)
    return result


def self_test() -> None:
    """Pre-inference independent checks; no model calls or server switching."""
    global OUTPUT
    rows = _scenario_rows()
    assert len(rows) == 21
    assert len({row["case_id"] for row in rows}) == 7
    assert all(len([x for x in rows if x["case_id"] == row["case_id"]]) == 3 for row in rows)

    old = _existing_audit_report()
    assert old["selected_count"] == 70
    assert old["audit_population_count"] == 742
    assert old["all_verdicts"] == {
        "DIRECT_SUPPORT": 57, "PARTIAL_SUPPORT": 12, "CONTRADICTS": 1,
    }
    assert old["by_layer_verdict"]["decision_lens"] == {
        "DIRECT_SUPPORT": 38, "PARTIAL_SUPPORT": 4,
    }
    assert old["by_layer_verdict"]["final_report_cited_sentence"] == {
        "DIRECT_SUPPORT": 6, "PARTIAL_SUPPORT": 7, "CONTRADICTS": 1,
    }

    cis = _load_checked(OUTPUT / "cis_v81_official_inventory.json")
    if cis is None:
        raise ValueError("Run --prepare first for offline self-test")
    assert len(cis["safeguards"]) == 153
    lookups = _cis_lookup(rows, cis)
    assert len(lookups) == 7
    assert len({x["case_id"] for x in lookups}) == 7
    valid_ids = {x["id"] for x in cis["safeguards"] if "IG2" in x["implementation_groups"]}
    for lookup in lookups:
        for key in ("technique_ig2_safeguards", "threat_ig2_safeguards"):
            assert len(lookup[key]) <= 5
            assert all(s["id"] in valid_ids for s in lookup[key])
    assert not next(x for x in lookups if x["candidate_action"] == "NLP/LLM")[
        "technique_ig2_safeguards"
    ], "No direct NLP/LLM match should be fabricated from generic processing words"

    for row in rows:
        direct = _direct_input(row)
        assert set(direct) == {
            "threat", "candidate_action", "forecast", "predictive_uncertainty",
            "timing_window", "reference_enterprise", "deployment_context",
            "evidence_records",
        }
        assert len(direct["evidence_records"]) > 0
        # No original Critic stance, Lens output, Mediator adjudication,
        # final recommendation or strategic report is passed to the baseline.
        assert "decision_recommendation" not in direct
        assert "stage2_critique_summary" not in direct
        assert "lens_evidence_trace" not in direct
        assert _qwen_identity(row, 42)["input_sha256"] == _qwen_identity(row, 43)["input_sha256"]

    report = rows[0]["original"]["strategic_intelligence_report"]
    citation_check = _citation_statistics(rows[0], report)
    assert citation_check["citation_mentions"] > 0
    assert not citation_check["unknown_cited_ids"]
    counterfeit = _report_input(rows[0], "Claim [EVIDENCE:NONEXISTENT_CITATION]", "HOLD")
    assert counterfeit["invalid_or_unknown_citations"] == ["NONEXISTENT_CITATION"]

    # A truncated response must not be parsed or checkpointed; verify retry
    # without a live model call or changing any original audit identity.
    from unittest.mock import Mock, patch

    good_grade = ReportGrade(
        factual_grounding=2,
        technical_scope=2,
        forecast_modality=2,
        constraints_and_uncertainty=2,
        internal_decision_coherence=2,
        material_issues=[],
        rationale="Mocked response for verifying recovery from an incomplete JSON generation.",
    )
    incomplete = Mock()
    incomplete.json.return_value = {
        "choices": [{"finish_reason": "length", "message": {"content": '{"rationale":"unfinished'}}]
    }
    complete = Mock()
    complete.json.return_value = {
        "choices": [{"finish_reason": "stop", "message": {"content": good_grade.model_dump_json()}}]
    }
    with patch.object(requests, "post", side_effect=[incomplete, complete]) as mocked:
        assert _query_gemma(REPORT_SYSTEM, {"test": True}, ReportGrade) == good_grade
        assert [call.kwargs["json"]["max_tokens"] for call in mocked.call_args_list] == [8192, 12288]
    malformed = Mock()
    malformed.json.return_value = {
        "choices": [{"finish_reason": "stop", "message": {"content": '{"rationale":"unfinished'}}]
    }
    with patch.object(requests, "post", side_effect=[malformed, complete]) as mocked:
        assert _query_gemma(REPORT_SYSTEM, {"test": True}, ReportGrade) == good_grade
        assert mocked.call_count == 2
    with patch.object(requests, "post", side_effect=[incomplete, incomplete]):
        try:
            _query_gemma(REPORT_SYSTEM, {"test": True}, ReportGrade)
        except RuntimeError as exc:
            assert "no complete schema-valid response" in str(exc)
        else:
            raise AssertionError("Truncated Gemma judgments must not be accepted")

    # Exercise the complete checkpoint and summary plumbing with MOCK outputs
    # isolated under a temporary directory, never mixing mock and real data.
    from tempfile import TemporaryDirectory
    actual_output = OUTPUT
    try:
        with TemporaryDirectory(prefix="r2c6-selftest-") as tmp:
            OUTPUT = Path(tmp)
            _freeze(OUTPUT / "cis_ig2_lookups.json", {"lookup_rows": lookups})
            _freeze(OUTPUT / "original_70_audit.json", old)
            for seed in (42, 43, 44):
                for row in rows:
                    _freeze(_baseline_path(row, seed), {
                        "identity": _qwen_identity(row, seed),
                        "output": {
                            "recommendation": row["original"]["decision_recommendation"],
                            "report": row["original"]["strategic_intelligence_report"],
                        },
                    })
            cases = _audit_cases()
            assert len(cases) == 56
            for case in cases:
                if case["kind"] == "report":
                    mocked = ReportGrade(
                        factual_grounding=2,
                        technical_scope=2,
                        forecast_modality=2,
                        constraints_and_uncertainty=2,
                        internal_decision_coherence=2,
                        material_issues=[],
                        rationale="Mock-only test response to verify the result aggregation; not a real report rating.",
                    )
                else:
                    mocked = CISGrades(mappings=[
                        CISRelevance(
                            safeguard_id=x["id"],
                            relevance="ADJACENT",
                            rationale="Mock-only retrieval assessment for test wiring.",
                        )
                        for x in case["payload"]["safeguards"]
                    ])
                _valid_response(case, mocked)
                _freeze(_judgment_path(case), {
                    "identity": _audit_identity(case["kind"], case["key"], case["payload"]),
                    "judgment": mocked.model_dump(mode="json"),
                })
            result = summary([42, 43, 44])
            assert result["recommendation_comparisons"] == 21
            assert result["recommendation_agreement"] == 21
            assert len(result["cis_reference"]) == 7
            assert len(result["direct_baseline_seed_pairs"]) == 3
            assert result["deterministic_citation_diagnostics"]["original"]["reports_with_unknown_citations"] == 0
            assert result["mean_ordinal_model_scores"]["factual_grounding"]["multi_agent"] == 2
            summary([42, 43, 44])  # identical frozen summary may be replayed
    finally:
        OUTPUT = actual_output

    print("SELF-TEST PASS: frozen 70-audit identity, official 153 Safeguards, "
          "7 both-direction CIS lookups, 21 leak-free direct inputs, "
          "citation boundary, 63 mock baseline records, 56 mock audits, "
          "and reproducible paired summary (all mocks deleted)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "baseline", "audit", "summary", "all", "self-test"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument(
        "--seeds", default="42,43,44",
        help="Direct-baseline seeds; seed 42 is required for paired report comparison",
    )
    parser.add_argument(
        "--cis-html", type=Path,
        help="Optional locally saved official CIS v8.1 Navigator HTML",
    )
    args = parser.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or 42 not in seeds:
        raise ValueError("Provide unique comma-delimited seeds including 42")
    if args.prepare or args.all:
        prepare(args.cis_html)
    if args.self_test:
        self_test()
    if args.baseline or args.all:
        run_baseline(seeds)
    if args.audit or args.all:
        run_audit()
    if args.summary or args.all:
        summary(seeds)


if __name__ == "__main__":
    main()
