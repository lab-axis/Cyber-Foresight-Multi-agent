"""Blind Gemma 4 audit of the already-frozen Mediator ablation results.

The 14 anonymous Critic pairs are scored against their exact Stage 0 evidence.
The same prompt and scoring rule apply to both arms. This automated verifier
is not expert ground truth, and differences alone do not establish causality.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict, Field, model_validator

from Stage7.loaders import sha256_json, sha256_text
from Stage7.mediator_ablation import OUTPUT, _canonical, _freeze, _load_checked


VERSION = "mediator-blind-claim-audit-gemma4-v3"
MODEL = "Gemma4-31B-it-Q6_K_L"
MODEL_SHA256 = "f77819db7b83a145425084048efec024b5efbc175bc0fdc4bc4bfe7d4b5802df"
URL = "http://127.0.0.1:8080/v1/chat/completions"
CLAIMS_PER_ARM_BATCH = 2

SYSTEM = """You are an independent evaluator of factual grounding in two ANONYMOUS
Critic assessments. Judge every claim under identical criteria using ONLY its
cited Stage 0 records, the supplied frozen forecast and the common frozen
Critic cross-review turns. Do not infer which condition used a Mediator.
Do not use external or parametric knowledge.

Source records contain BOTH document content AND authoritative provenance
metadata. Check source/title/issuer claims against source, source_document_id
and source_locator, and check dates against publication_date and available_at.
Do not reject a valid cited source attribution or date merely because the
same text is absent from the content field.

DIRECT_SUPPORT means all material factual propositions are supported by the
cited content or its metadata, and any analytical qualifications correctly
describe the LIMITS of what the cited records measure. A statement such as
'dwell time is not an incident-count time series' can be a legitimate logical
boundary inference even if the source does not say it verbatim. Do not demand
a literal quotation for valid scope limitations. Equally, do not accept a
blanket absence claim if a cited record does contain the relevant measurement.
PARTIAL_SUPPORT means the factual proposition or qualification materially
overstates source scope, population, timing, strength, causality or specificity.
TOPICAL_ONLY means only the topic matches. CONTRADICTS means the cited records
materially conflict. INSUFFICIENT_EVIDENCE means the supplied boundary cannot
decide the proposition.

The frozen Critic cross-review turns are provided ONLY to verify factual
statements about who said or agreed to what during deliberation. They are
not external support for threat/PMT facts. Do not assume an agreement happened
if the supplied cross-review does not substantiate it.

Mark unsupported_specificity for exact facts or strength not supported by
content, provenance metadata or the appropriate debate provenance. Mark
modality_error only if a positive directional Threat trajectory uses shares,
severity, attack size or tactics instead of incident frequency/count (NoI),
or if a positive directional PMT trajectory infers publication activity (NoP)
from static standards or guidance. GAP_DIRECTION needs evidence for both
modalities. A statement explaining missing evidence is NOT a positive trend.

DIRECT_SUPPORT requires unsupported_specificity=false and modality_error=false.
Judge every integer claim index once. Explain actual decisive evidence or the
precise unsupported assertion. Do not reward length, consensus or a seemingly
more detailed condition. Return only JSON conforming to the schema.
"""


class ClaimJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    verdict: Literal[
        "DIRECT_SUPPORT", "PARTIAL_SUPPORT", "TOPICAL_ONLY",
        "CONTRADICTS", "INSUFFICIENT_EVIDENCE",
    ]
    unsupported_specificity: bool
    modality_error: bool
    rationale: str = Field(min_length=8, max_length=420)

    @model_validator(mode="after")
    def reject_conflicting_labels(self) -> "ClaimJudgment":
        if self.verdict == "DIRECT_SUPPORT" and (
            self.unsupported_specificity or self.modality_error
        ):
            raise ValueError("DIRECT_SUPPORT cannot coexist with an unsupported fact or modality error")
        return self


class BatchJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[ClaimJudgment] = Field(min_length=1)


def _claims(pair: dict) -> list[dict]:
    result = []

    for arm in ("A", "B"):
        for claim in pair[f"assessment_{arm}"]["claims"]:
            result.append({
                "index": len(result),
                "arm": arm,
                "claim_id": claim["claim_id"],
                "statement": claim["statement"],
                "forecast_component": claim["forecast_component"],
                "forecast_relation": claim["forecast_relation"],
                "evidence_ids": claim["evidence_ids"],
            })

    return result


def _input(pair: dict, indices: list[int] | None = None) -> dict:
    claims = [c for c in _claims(pair) if indices is None or c["index"] in indices]
    if not claims:
        raise ValueError(f"No claims selected for {pair['review_id']}")
    cited = {eid for claim in claims for eid in claim["evidence_ids"]}
    evidence = [
        dict(record)
        for record in pair["evidence_records"]
        if record["evidence_id"] in cited
    ]
    if {record["evidence_id"] for record in evidence} != cited:
        raise ValueError(f"Missing exact cited record in pair {pair['review_id']}")

    cross_review = pair.get("cross_review_round1")
    if not isinstance(cross_review, dict) or set(cross_review) != {
        "attack_turn", "defense_turn",
    }:
        raise ValueError(f"Frozen cross-review missing or malformed: {pair['review_id']}")

    summary = pair["forecast_summary"]
    return {
        "forecast_summary": {
            key: summary[key] for key in (
                "threat_state", "pmt_state", "gap_by_year", "gap_direction",
                "gap_direction_semantics", "gap_semantics",
                "gap_slope_per_year", "predictive_uncertainty",
            )
        },
        "evidence_records": evidence,
        "cross_review_round1": cross_review,
        "claims": claims,
    }


def _validate(
    value: BatchJudgment, pair: dict, indices: list[int] | None = None,
) -> BatchJudgment:
    actual_indices = [claim.index for claim in value.claims]
    expected = (
        {claim["index"] for claim in _claims(pair)}
        if indices is None
        else set(indices)
    )
    if len(actual_indices) != len(expected) or set(actual_indices) != expected:
        raise ValueError(f"Missing or duplicated claim indices for {pair['review_id']}")

    return value


def _path(pair: dict) -> Path:
    return OUTPUT / "quality_audit" / f"{pair['review_id']}.json"


def _identity(pair: dict) -> dict:
    return {
        "version": VERSION,
        "model": MODEL,
        "model_sha256": MODEL_SHA256,
        "review_id": pair["review_id"],
        "input_sha256": sha256_json(_input(pair)),
        "prompt_sha256": sha256_text(SYSTEM),
        "schema_sha256": sha256_json(BatchJudgment.model_json_schema()),
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "seed": 42},
    }


def _batches(pair: dict) -> list[list[int]]:
    arms = {
        arm: [claim["index"] for claim in _claims(pair) if claim["arm"] == arm]
        for arm in ("A", "B")
    }
    return [
        arms["A"][start : start + CLAIMS_PER_ARM_BATCH]
        + arms["B"][start : start + CLAIMS_PER_ARM_BATCH]
        for start in range(0, max(map(len, arms.values())), CLAIMS_PER_ARM_BATCH)
    ]


def _batch_path(pair: dict, batch_number: int) -> Path:
    return OUTPUT / "quality_audit_batches" / pair["review_id"] / f"{batch_number:03}.json"


def _request(pair: dict, indices: list[int] | None = None) -> BatchJudgment:
    payload = _input(pair, indices)
    original_indices = [claim["index"] for claim in payload["claims"]]
    if indices is not None:
        # Dense batch-local numbering avoids asking the model to reproduce
        # sparse, non-consecutive indices from the complete 14-pair review.
        for local_index, claim in enumerate(payload["claims"]):
            claim["index"] = local_index

    user = _canonical(payload)
    response_schema = BatchJudgment.model_json_schema()
    if indices is not None:
        response_schema["$defs"]["ClaimJudgment"]["properties"]["index"] = {
            "type": "integer",
            "enum": list(range(len(original_indices))),
        }
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]

    for attempt in range(2):
        response = requests.post(
            URL,
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "presence_penalty": 0.0,
                "repeat_penalty": 1.0,
                "seed": 42,
                "max_tokens": 4096,
                "response_format": {
                    "type": "json_object",
                    "schema": response_schema,
                },
                "stream": False,
            },
            timeout=1800,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        try:
            judged = BatchJudgment.model_validate_json(content)
            if indices is not None:
                actual_local = [claim.index for claim in judged.claims]
                if len(actual_local) != len(original_indices) or set(actual_local) != set(
                    range(len(original_indices))
                ):
                    raise ValueError("Missing or duplicated batch-local claim indices")
                judged = BatchJudgment(claims=[
                    claim.model_copy(update={"index": original_indices[claim.index]})
                    for claim in judged.claims
                ])
            return _validate(judged, pair, indices)
        except ValueError as exc:
            if attempt:
                if indices is not None and len(indices) > 1:
                    # Recover only this unsaved batch, without touching any
                    # previously frozen judgments or changing their identity.
                    singles = [
                        judgment
                        for index in original_indices
                        for judgment in _request(pair, [index]).claims
                    ]
                    return _validate(BatchJudgment(claims=singles), pair, indices)
                raise
            messages.extend([
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": f"Correct JSON/schema and claim indexing only: {str(exc)[:650]}",
                },
            ])

    raise RuntimeError("Gemma audit exhausted its fixed validation attempt")


async def _audit(pair: dict) -> None:
    identity = _identity(pair)
    saved = _load_checked(_path(pair))

    if saved is not None:
        if saved["identity"] != identity:
            raise ValueError(f"Cached Gemma audit has a different identity: {pair['review_id']}")
        _validate(BatchJudgment.model_validate(saved["judgment"]), pair)
        print(f"Reused Gemma audit: {pair['review_id']}", flush=True)
        return

    responses = []
    for batch_number, indices in enumerate(_batches(pair)):
        batch_identity = {
            "pair": identity,
            "indices": indices,
            "input_sha256": sha256_json(_input(pair, indices)),
        }
        path = _batch_path(pair, batch_number)
        cached_batch = _load_checked(path)

        if cached_batch is not None:
            if cached_batch["identity"] != batch_identity:
                raise ValueError(f"Cached Gemma batch mismatch: {path}")
            result = _validate(
                BatchJudgment.model_validate(cached_batch["judgment"]),
                pair, indices,
            )
            status = "reused"
        else:
            result = await asyncio.to_thread(_request, pair, indices)
            _freeze(path, {
                "identity": batch_identity,
                "judgment": result.model_dump(mode="json"),
            })
            status = "completed"

        responses.extend(result.claims)
        print(
            f"Gemma audit batch {batch_number + 1}/{len(_batches(pair))} {status}: "
            f"{pair['review_id']} ({len(indices)} claims)",
            flush=True,
        )

    answer = _validate(BatchJudgment(claims=responses), pair)
    _freeze(_path(pair), {
        "identity": identity,
        "judgment": answer.model_dump(mode="json"),
    })
    print(f"Gemma audit complete: {pair['review_id']} ({len(answer.claims)} claims)", flush=True)


def _score(claims: list[ClaimJudgment]) -> dict:
    n = len(claims)
    verdicts = Counter(x.verdict for x in claims)

    return {
        "n": n,
        "verdict_counts": dict(verdicts),
        "direct_support": verdicts["DIRECT_SUPPORT"] / n,
        "direct_or_partial": (verdicts["DIRECT_SUPPORT"] + verdicts["PARTIAL_SUPPORT"]) / n,
        "unsupported_specificity": sum(x.unsupported_specificity for x in claims) / n,
        "modality_error": sum(x.modality_error for x in claims) / n,
    }


def _report(pairs: list[dict], blind_key: dict) -> dict:
    rows = []
    claim_details = []

    for pair in pairs:
        record = _load_checked(_path(pair))
        if record is None or record["identity"] != _identity(pair):
            raise ValueError(f"Missing or stale Gemma audit: {pair['review_id']}")

        judgments = _validate(BatchJudgment.model_validate(record["judgment"]), pair)
        by_index = {x.index: x for x in judgments.claims}
        groups = {"A": [], "B": []}

        for claim in _claims(pair):
            groups[claim["arm"]].append(by_index[claim["index"]])

        mapping = blind_key[pair["review_id"]]
        measured = {
            mapping[arm]: _score(values)
            for arm, values in groups.items()
        }
        identified = {"with_mediator": {}, "without_mediator": {}}
        for claim in _claims(pair):
            arm = mapping[claim["arm"]]
            judgment = by_index[claim["index"]]
            identified[arm][claim["claim_id"]] = judgment
            claim_details.append({
                "review_id": pair["review_id"],
                "condition": arm,
                "claim_id": claim["claim_id"],
                "statement": claim["statement"],
                "evidence_ids": claim["evidence_ids"],
                "judgment": judgment.model_dump(mode="json"),
            })

        common = sorted(set(identified["with_mediator"]) & set(identified["without_mediator"]))
        favor_mediator = sum(
            identified["with_mediator"][cid].verdict == "DIRECT_SUPPORT"
            and identified["without_mediator"][cid].verdict != "DIRECT_SUPPORT"
            for cid in common
        )
        favor_without = sum(
            identified["with_mediator"][cid].verdict != "DIRECT_SUPPORT"
            and identified["without_mediator"][cid].verdict == "DIRECT_SUPPORT"
            for cid in common
        )
        with_med = measured["with_mediator"]
        without = measured["without_mediator"]
        rows.append({
            "review_id": pair["review_id"],
            "case_id": pair["review_id"].rsplit("__", 1)[0],
            "role": pair["review_id"].rsplit("__", 1)[1],
            "analysis": mapping["analysis"],
            "with_mediator": with_med,
            "without_mediator": without,
            "delta_direct_support": with_med["direct_support"] - without["direct_support"],
            "delta_unsupported_specificity": with_med["unsupported_specificity"] - without["unsupported_specificity"],
            "delta_modality_error": with_med["modality_error"] - without["modality_error"],
            "shared_claim_id_count": len(common),
            "shared_claim_ids_favor_mediator": favor_mediator,
            "shared_claim_ids_favor_without_mediator": favor_without,
            "shared_claim_ids_same_direct_support_status": len(common) - favor_mediator - favor_without,
            "with_mediator_only_claim_ids": len(identified["with_mediator"]) - len(common),
            "without_mediator_only_claim_ids": len(identified["without_mediator"]) - len(common),
        })

    primary = [r for r in rows if r["analysis"] == "matched_one_round"]
    case_ids = sorted({r["case_id"] for r in primary})
    case_results = []

    for case_id in case_ids:
        related = [r for r in primary if r["case_id"] == case_id]
        case_results.append({
            "case_id": case_id,
            "delta_direct_support": mean(r["delta_direct_support"] for r in related),
            "delta_unsupported_specificity": mean(r["delta_unsupported_specificity"] for r in related),
            "delta_modality_error": mean(r["delta_modality_error"] for r in related),
        })

    direction_counts = {
        "favors_mediator": sum(row["delta_direct_support"] > 0 for row in case_results),
        "favors_without_mediator": sum(row["delta_direct_support"] < 0 for row in case_results),
        "tied": sum(row["delta_direct_support"] == 0 for row in case_results),
    }

    report = {
        "version": VERSION,
        "model": MODEL,
        "matched_cases": len(case_ids),
        "one_round_sensitivity_cases": len({r["case_id"] for r in rows if r["analysis"] != "matched_one_round"}),
        "audit_pairs": len(rows),
        "total_claims_audited": sum(r["with_mediator"]["n"] + r["without_mediator"]["n"] for r in rows),
        "primary_case_macro_delta_direct_support": mean(x["delta_direct_support"] for x in case_results),
        "primary_case_macro_delta_unsupported_specificity": mean(x["delta_unsupported_specificity"] for x in case_results),
        "primary_case_macro_delta_modality_error": mean(x["delta_modality_error"] for x in case_results),
        "primary_case_direction_counts": direction_counts,
        "primary_shared_claim_ids_favor_mediator": sum(
            row["shared_claim_ids_favor_mediator"] for row in primary
        ),
        "primary_shared_claim_ids_favor_without_mediator": sum(
            row["shared_claim_ids_favor_without_mediator"] for row in primary
        ),
        "case_results": case_results,
        "role_results": rows,
        "limitations": (
            "Automated judgments from Gemma 4 31B are not human/expert ground truth. "
            "Only six independent matched case clusters were available; Brute Force "
            "is a separate one-round sensitivity control. The evidence and prompts "
            "are fixed, but a single nominal seed does not estimate within-arm noise. "
            "Shared claim IDs may contain different statement wording, so paired ID "
            "comparisons are descriptive rather than independently verified equivalences. "
            "Neither claim counts nor a semantic-verifier score demonstrate causal correctness."
        ),
    }
    _freeze(OUTPUT / "quality_summary.json", report)
    _freeze(OUTPUT / "quality_claim_details.json", {
        "version": VERSION,
        "audit_pairs": len(rows),
        "claims": claim_details,
    })
    return report


def _inputs() -> tuple[list[dict], dict]:
    review = _load_checked(OUTPUT / "blind_review.json")
    key = _load_checked(OUTPUT / "blind_key.json")
    if review is None or key is None or len(review["items"]) != 14:
        raise ValueError("Complete frozen 14-pair blind review required before Gemma audit")

    return review["items"], key["mapping"]


def run() -> dict:
    pairs, mapping = _inputs()

    pending = []
    for pair in pairs:
        saved = _load_checked(_path(pair))
        if saved is None:
            pending.append(pair)
            continue
        if saved["identity"] != _identity(pair):
            raise ValueError(f"Cached Gemma audit has a different identity: {pair['review_id']}")
        _validate(BatchJudgment.model_validate(saved["judgment"]), pair)

    if pending:
        from Stage7.gemma_server import use_gemma

        print(f"Gemma blind audit: {len(pending)} missing of {len(pairs)} frozen pairs", flush=True)
        with use_gemma():
            asyncio.run(_run_pairs(pending))

    return _report(pairs, mapping)


async def _run_pairs(pairs: list[dict]) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    for pair in pairs:
        queue.put_nowait(pair)

    async def worker() -> None:
        while True:
            try:
                pair = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                await _audit(pair)
            finally:
                queue.task_done()

    outcomes = await asyncio.gather(worker(), return_exceptions=True)
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} Gemma blind audit worker(s) failed") from failures[0]


def report() -> dict:
    pairs, mapping = _inputs()
    return _report(pairs, mapping)
