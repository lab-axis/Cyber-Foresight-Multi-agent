from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from Stage4.context_input import build_context_stage4_evaluation_payload
from Stage4.context_schema import ContextLensAssessment, validate_context_lens_assessment
from Stage4.schema import Stage4ValidationError
from Stage7.jurisdiction_shared_evidence import (
    CASES,
    CONDITIONS,
    LENS,
    SEEDS,
    _all_jobs,
    _build_conditions,
    _job,
    _load_completed,
    _selected_bundles,
    report,
    run,
)
from Stage7.loaders import sha256_json


ROOT = Path(__file__).resolve().parents[3]


class JurisdictionSharedEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundles = _selected_bundles(ROOT)

    def test_frozen_inputs_are_unchanged_and_records_identical(self) -> None:
        self.assertEqual(len(self.bundles), len(CASES))

        for bundle in self.bundles:
            before = sha256_json(bundle.contextual_objects)
            variants = _build_conditions(bundle)
            after = sha256_json(bundle.contextual_objects)
            self.assertEqual(before, after)
            self.assertEqual(set(variants), set(CONDITIONS))

            payloads = {
                name: build_context_stage4_evaluation_payload(
                    decision_object=obj, expected_lens_type=LENS
                )
                for name, obj in variants.items()
            }
            self.assertEqual(len({
                sha256_json(payload["evaluation_evidence"]) for payload in payloads.values()
            }), 1)

            eu_legal = {
                eid for eid, slots in variants["EU"].decision_evidence_pack.slot_coverage.items()
                if "regulatory_applicability" in slots
            }
            self.assertTrue(eu_legal)
            for condition in ("UNSPECIFIED", "KR", "US"):
                slots = variants[condition].decision_evidence_pack.slot_coverage
                self.assertFalse(eu_legal.intersection(slots))

    def test_non_eu_legal_claim_cannot_be_promoted_to_deployment_direction(self) -> None:
        variants = _build_conditions(self.bundles[0])
        eu = variants["EU"]
        coverage = eu.decision_evidence_pack.slot_coverage
        legal = next(eid for eid, slots in coverage.items() if "regulatory_applicability" in slots)
        governance = next(eid for eid, slots in coverage.items() if "governance_enablement" in slots)

        def assessment(obj):
            return {
                "decision_object_id": obj.decision_object_id,
                "lens_type": LENS.value,
                "stance": 1,
                "evidence_sufficiency": "PARTIAL",
                "rationale": "Test of grounding eligibility only.",
                "claims": [{
                    "claim_id": "test_claim",
                    "statement": "Test-only conditional feasibility claim.",
                    "evidence_ids": [governance, legal],
                    "status": "SUPPORTS_FEASIBILITY",
                    "scope": "DEPLOYMENT_SPECIFIC",
                }],
                "constraints": [],
                "evidence_gaps": [],
                "conditional_requirements": [],
            }

        valid = validate_context_lens_assessment(
            assessment(eu), decision_object=eu, expected_lens_type=LENS
        )
        self.assertEqual(valid.stance, 1)

        for condition in ("UNSPECIFIED", "KR", "US"):
            obj = variants[condition]
            with self.assertRaises(Stage4ValidationError):
                validate_context_lens_assessment(
                    assessment(obj), decision_object=obj, expected_lens_type=LENS
                )

    def test_check_requires_exactly_36_frozen_jobs(self) -> None:
        jobs = _all_jobs(ROOT)
        self.assertEqual(len(jobs), 36)
        self.assertEqual({j[0]["seed"] for j in jobs}, set(SEEDS))
        self.assertEqual({j[0]["condition"] for j in jobs}, set(CONDITIONS))
        self.assertEqual({j[0]["case_id"] for j in jobs}, set(CASES))
        self.assertEqual(len({j[0]["evidence_content_sha256"] for j in jobs}), 3)
        self.assertEqual(len({str(j[1]) for j in jobs}), 36)

    def test_checkpoint_resume_uses_frozen_identity_without_server(self) -> None:
        bundle = self.bundles[0]
        obj = _build_conditions(bundle)["KR"]

        with TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch("Stage7.jurisdiction_shared_evidence._results_dir", return_value=destination):
                job = _job(ROOT, bundle, "KR", 42, obj)
                governance = next(
                    eid for eid, slots in obj.decision_evidence_pack.slot_coverage.items()
                    if "governance_enablement" in slots
                )

                class FakeRunner:
                    async def ainvoke(self, messages):
                        raw = {
                            "decision_object_id": obj.decision_object_id,
                            "lens_type": LENS.value,
                            "stance": 1,
                            "evidence_sufficiency": "PARTIAL",
                            "rationale": "Controlled test output.",
                            "claims": [{
                                "claim_id": "c1",
                                "statement": "Test-only general governance pathway.",
                                "evidence_ids": [governance],
                                "status": "SUPPORTS_FEASIBILITY",
                                "scope": "GENERAL",
                            }],
                            "constraints": [],
                            "evidence_gaps": [],
                            "conditional_requirements": [],
                        }
                        return ContextLensAssessment.model_validate(raw)

                class FakeClient:
                    def with_structured_output(self, *args, **kwargs):
                        return FakeRunner()

                with patch("Stage7.jurisdiction_shared_evidence._variant_client", return_value=(FakeClient(), job[0]["runtime"])) as client:
                    asyncio.run(run([job]))
                    self.assertEqual(_load_completed(job)["stance"], 1)
                    asyncio.run(run([job]))
                    self.assertEqual(client.call_count, 1)

    def test_fake_full_run_produces_verified_summary(self) -> None:
        with TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch("Stage7.jurisdiction_shared_evidence._results_dir", return_value=destination):
                jobs = _all_jobs(ROOT)
                runtime_by_seed = {job[0]["seed"]: job[0]["runtime"] for job in jobs}
                governance_by_id = {
                    job[2].decision_object_id: next(
                        eid for eid, slots in job[2].decision_evidence_pack.slot_coverage.items()
                        if "governance_enablement" in slots
                    )
                    for job in jobs
                }

                class FakeClient:
                    def with_structured_output(self, *args, **kwargs):
                        decision_id = kwargs["response_schema"]["properties"]["decision_object_id"]["const"]

                        class FakeRunner:
                            async def ainvoke(self, messages):
                                return ContextLensAssessment.model_validate({
                                    "decision_object_id": decision_id,
                                    "lens_type": LENS.value,
                                    "stance": 1,
                                    "evidence_sufficiency": "PARTIAL",
                                    "rationale": "Synthetic test result, not a study finding.",
                                    "claims": [{
                                        "claim_id": "c1",
                                        "statement": "Synthetic test claim.",
                                        "evidence_ids": [governance_by_id[decision_id]],
                                        "status": "SUPPORTS_FEASIBILITY",
                                        "scope": "GENERAL",
                                    }],
                                    "constraints": [],
                                    "evidence_gaps": [],
                                    "conditional_requirements": [],
                                })

                        return FakeRunner()

                def fake_client(*, runner_version, seed):
                    return FakeClient(), runtime_by_seed[seed]

                with patch("Stage7.jurisdiction_shared_evidence._variant_client", side_effect=fake_client):
                    asyncio.run(run(jobs))

                path = report(ROOT, jobs)
                summary = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(summary["institutional_lens_calls"], 36)
                self.assertEqual(set(summary["cases"]), set(CASES))
                self.assertEqual(
                    summary["cases"][CASES[0]]["within_condition_pairwise_stance_agreement"]["EU"], 1.0
                )

                # Running again must use the existing checkpoints without a server.
                with patch("Stage7.jurisdiction_shared_evidence._variant_client") as blocked:
                    asyncio.run(run(jobs))
                    blocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
