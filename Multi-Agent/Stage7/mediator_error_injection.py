"""Small, independently blinded error-injection check of the frozen Gemma rubric.

Six predeclared cited claims are evaluated both unchanged and after one
deterministic factual corruption. Each variant is sent as an independent
single-claim request, without its condition or expected label. This tests
the verifier's discrimination, NOT Mediator superiority or real-world accuracy.
The 229 existing Gemma judgments and all Qwen assessments are read only.
"""

from __future__ import annotations

from copy import deepcopy

from Stage7 import mediator_quality_analysis as audit
from Stage7.mediator_ablation import OUTPUT, _freeze, _load_checked


VERSION = "mediator-gemma-error-injection-v1"
PROBES = (
    (
        "ddos_bandwidth",
        "main__ddos__nlp_llm__attack_feasibility",
        "AF_C1",
        "1.4 Gbps to 2.2 Gbps",
        "1.4 Gbps to 22.2 Gbps",
        "wrong_numeric_value",
    ),
    (
        "cryptography_publication_date",
        "main__insider_threat__cryptography__attack_feasibility",
        "C5",
        "2023-01-18",
        "2023-07-18",
        "wrong_provenance_date",
    ),
    (
        "nist_source_attribution",
        "main__insider_threat__cryptography__attack_feasibility",
        "C2",
        "NIST SP 800-161 Rev.1 C-SCRM guidance",
        "NIST SP 800-53 Rev.5 C-SCRM guidance",
        "wrong_source_attribution",
    ),
    (
        "kastle_exploitation_polarity",
        "main__targeted_attack__access_control__attack_feasibility",
        "ATK_C4",
        "no known public exploitation specifically targeting",
        "confirmed public exploitation specifically targeting",
        "reversed_observation",
    ),
    (
        "ransomware_noi_modality",
        "main__ransomware__cryptography__attack_feasibility",
        "C2",
        "while also reporting extortion at 27% and phishing at 41%.",
        "while also reporting extortion at 27% and phishing at 41%. "
        "These shares prove that ransomware incident counts increased every "
        "year from 2020 through 2022.",
        "unsupported_incident_count_trend",
    ),
    (
        "cryptography_causal_effectiveness",
        "main__ransomware__cryptography__attack_feasibility",
        "C5",
        "with a positive OLS slope of 37.8 works per year.",
        "with a positive OLS slope of 37.8 works per year. "
        "This demonstrates that cryptographic controls reduced ransomware "
        "incidents by 90%.",
        "unsupported_causal_effectiveness",
    ),
)

OUTPUT_DIR = OUTPUT / "error_injection"


def _cases() -> list[dict]:
    pairs, mapping = audit._inputs()
    by_id = {pair["review_id"]: pair for pair in pairs}
    result = []

    for probe_id, review_id, claim_id, before, after, error_type in PROBES:
        source = by_id[review_id]
        # Use the ORIGINAL mediated or unmediated assessment as frozen.
        # A/B is a fixed opaque arm, not a model-visible error label.
        matches = [c for c in source["assessment_A"]["claims"] if c["claim_id"] == claim_id]
        if len(matches) != 1 or matches[0]["statement"].count(before) != 1:
            raise ValueError(f"Frozen error-injection anchor changed: {probe_id}")

        original = matches[0]
        baseline = _load_checked(audit._path(source))
        if baseline is None or baseline["identity"] != audit._identity(source):
            raise ValueError(f"Frozen Gemma 229-claim audit is missing or stale: {review_id}")
        original_index = next(
            c["index"] for c in audit._claims(source)
            if c["arm"] == "A" and c["claim_id"] == claim_id
        )
        previous = next(c for c in baseline["judgment"]["claims"] if c["index"] == original_index)
        if previous["verdict"] != "DIRECT_SUPPORT":
            raise ValueError(f"Clean probe lacks frozen direct-support control: {probe_id}")

        variants = {}
        for variant in ("clean", "corrupted"):
            claim = deepcopy(original)
            if variant == "corrupted":
                claim["statement"] = claim["statement"].replace(before, after, 1)
            synthetic = {
                "review_id": f"{review_id}__{probe_id}",
                "forecast_summary": source["forecast_summary"],
                "evidence_records": source["evidence_records"],
                "cross_review_round1": source["cross_review_round1"],
                "assessment_A": {"claims": [claim]},
                "assessment_B": {"claims": []},
            }
            expected_ids = set(claim["evidence_ids"])
            payload = audit._input(synthetic, [0])
            if {r["evidence_id"] for r in payload["evidence_records"]} != expected_ids:
                raise ValueError(f"Evidence boundary drift: {probe_id}")
            variants[variant] = synthetic

        if audit._input(variants["clean"], [0])["evidence_records"] != (
            audit._input(variants["corrupted"], [0])["evidence_records"]
        ):
            raise ValueError(f"Evidence was changed during corruption: {probe_id}")

        result.append({
            "id": probe_id,
            "review_id": review_id,
            "original_claim_id": claim_id,
            "error_type": error_type,
            "anchor": before,
            "replacement": after,
            "original_condition": mapping[review_id]["A"],
            "variants": variants,
        })

    return result


def _identity(case: dict, variant: str) -> dict:
    return {
        "version": VERSION,
        "probe_id": case["id"],
        "variant": variant,
        "gemma": audit._identity(case["variants"][variant]),
    }


def _path(case: dict, variant: str):
    return OUTPUT_DIR / "judgments" / f"{case['id']}__{variant}.json"


def _report(cases: list[dict]) -> dict:
    rows = []
    for case in cases:
        results = {}
        for variant in ("clean", "corrupted"):
            saved = _load_checked(_path(case, variant))
            if saved is None or saved["identity"] != _identity(case, variant):
                raise ValueError(f"Missing or stale error-injection judgment: {case['id']} {variant}")
            answer = audit._validate(
                audit.BatchJudgment.model_validate(saved["judgment"]),
                case["variants"][variant], [0],
            )
            results[variant] = answer.claims[0].model_dump(mode="json")

        rows.append({
            "id": case["id"],
            "error_type": case["error_type"],
            "review_id": case["review_id"],
            "claim_id": case["original_claim_id"],
            "source_condition": case["original_condition"],
            "original_statement": case["variants"]["clean"]["assessment_A"]["claims"][0]["statement"],
            "corrupted_statement": case["variants"]["corrupted"]["assessment_A"]["claims"][0]["statement"],
            "clean": results["clean"],
            "corrupted": results["corrupted"],
            "clean_accepted": results["clean"]["verdict"] == "DIRECT_SUPPORT",
            "corruption_detected": results["corrupted"]["verdict"] != "DIRECT_SUPPORT",
        })

    n = len(rows)
    correctly_accepted = sum(row["clean_accepted"] for row in rows)
    detected = sum(row["corruption_detected"] for row in rows)
    report = {
        "version": VERSION,
        "design": "6 predeclared claim pairs; independent blinded 1-claim calls; exact original evidence",
        "total_claims_evaluated": 2 * n,
        "original_controls": n,
        "injected_errors": n,
        "controls_accepted": correctly_accepted,
        "injected_errors_detected": detected,
        "false_positive_count": n - correctly_accepted,
        "false_negative_count": n - detected,
        "specificity": correctly_accepted / n,
        "sensitivity": detected / n,
        "balanced_accuracy": (correctly_accepted + detected) / (2 * n),
        "all_predeclared_checks_passed": correctly_accepted == n and detected == n,
        "rows": rows,
        "limitations": (
            "Synthetic, hand-selected perturbations with known source contradictions "
            "or unsupported extensions; small sample and single fixed seed. "
            "Original controls were previously scored DIRECT_SUPPORT by this same "
            "verifier, so specificity is not independent expert validation. "
            "A non-DIRECT_SUPPORT verdict can be for the wrong reason; inspect each "
            "rationale before publication. This does not prove all 229 claims are correct."
        ),
    }
    _freeze(OUTPUT_DIR / "summary.json", report)
    return report


def run() -> dict:
    cases = _cases()
    # Freeze the entire design before inference and prevent post-hoc alteration.
    _freeze(OUTPUT_DIR / "design.json", {
        "version": VERSION,
        "probes": [
            {
                "id": case["id"],
                "review_id": case["review_id"],
                "claim_id": case["original_claim_id"],
                "error_type": case["error_type"],
                "original_condition": case["original_condition"],
                "clean_input_sha256": _identity(case, "clean")["gemma"]["input_sha256"],
                "corrupted_input_sha256": _identity(case, "corrupted")["gemma"]["input_sha256"],
                "anchor": case["anchor"],
                "replacement": case["replacement"],
            }
            for case in cases
        ],
    })

    ordered = [
        (case, variant)
        for i, case in enumerate(cases)
        for variant in (("corrupted", "clean") if i % 2 == 0 else ("clean", "corrupted"))
    ]
    pending = []
    for case, variant in ordered:
        saved = _load_checked(_path(case, variant))
        if saved is None:
            pending.append((case, variant))
        elif saved["identity"] != _identity(case, variant):
            raise ValueError(f"Stale error-injection checkpoint: {case['id']} {variant}")
        else:
            audit._validate(
                audit.BatchJudgment.model_validate(saved["judgment"]),
                case["variants"][variant], [0],
            )

    if pending:
        from Stage7.gemma_server import use_gemma

        print(f"Error-injection audit: {len(pending)} missing of {len(ordered)} claims; "
              "skipping all 229 frozen Gemma judgments", flush=True)
        with use_gemma():
            for case, variant in pending:
                answer = audit._request(case["variants"][variant], [0])
                _freeze(_path(case, variant), {
                    "identity": _identity(case, variant),
                    "judgment": answer.model_dump(mode="json"),
                })
                print(f"Error-injection checkpoint: {case['id']} #{variant}", flush=True)

    return _report(cases)
