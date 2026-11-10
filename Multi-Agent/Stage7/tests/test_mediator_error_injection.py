from __future__ import annotations

import json
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from Stage7 import mediator_ablation as study
from Stage7 import mediator_error_injection as probe
from Stage7 import mediator_quality_analysis as audit
from Stage7 import gemma_server


class ErrorInjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = probe._cases()

    def test_predeclared_mutations_change_only_one_statement(self):
        self.assertEqual(len(self.cases), 6)
        self.assertEqual(len({c["error_type"] for c in self.cases}), 6)
        for case in self.cases:
            clean = case["variants"]["clean"]
            bad = case["variants"]["corrupted"]
            original = clean["assessment_A"]["claims"][0]
            corrupted = bad["assessment_A"]["claims"][0]
            self.assertEqual(
                original["statement"].replace(case["anchor"], case["replacement"], 1),
                corrupted["statement"],
            )
            self.assertNotEqual(original["statement"], corrupted["statement"])
            self.assertEqual(
                {k: v for k, v in original.items() if k != "statement"},
                {k: v for k, v in corrupted.items() if k != "statement"},
            )
            a = audit._input(clean, [0])
            b = audit._input(bad, [0])
            for field in ("evidence_records", "forecast_summary", "cross_review_round1"):
                self.assertEqual(a[field], b[field])
            for variant in (a, b):
                self.assertEqual(len(variant["claims"]), 1)
                self.assertNotIn("corrupted", json.dumps(variant))
                self.assertNotIn("error_type", json.dumps(variant))
                self.assertNotIn("with_mediator", json.dumps(variant))
                self.assertNotIn("without_mediator", json.dumps(variant))

    def test_command_runs_only_probe_not_229_claim_or_qwen(self):
        with (
            patch.object(sys, "argv", ["mediator_ablation", "--error-injection"]),
            patch.object(study, "_sources", side_effect=AssertionError("Qwen source loading is forbidden")),
            patch.object(audit, "run", side_effect=AssertionError("Full Gemma audit is forbidden")),
            patch.object(probe, "run", return_value={
                "sensitivity": 1.0, "specificity": 1.0,
            }) as run,
        ):
            study.main()
        run.assert_called_once_with()

    def test_freezes_probe_results_resumes_without_reinference_and_keeps_old_audit(self):
        calls = []
        old = [
            (case["review_id"], audit._load_checked(audit._path(
                next(pair for pair in audit._inputs()[0] if pair["review_id"] == case["review_id"])
            ))["sha256"])
            for case in self.cases
        ]
        originals = {
            case["id"]: case["variants"]["clean"]["assessment_A"]["claims"][0]["statement"]
            for case in self.cases
        }

        def fake_request(pair, indices):
            self.assertEqual(indices, [0])
            statement = pair["assessment_A"]["claims"][0]["statement"]
            probe_id = pair["review_id"].rsplit("__", 1)[-1]
            clean = statement == originals[probe_id]
            calls.append((probe_id, clean))
            return audit.BatchJudgment(claims=[
                audit.ClaimJudgment(
                    index=0,
                    verdict="DIRECT_SUPPORT" if clean else "PARTIAL_SUPPORT",
                    unsupported_specificity=not clean,
                    modality_error=False,
                    rationale="The original exact source supports this statement."
                              if clean else "The added or altered factual detail is not in the cited evidence.",
                ),
            ])

        @contextmanager
        def fake_server():
            yield

        with TemporaryDirectory() as folder:
            with (
                patch.object(probe, "OUTPUT_DIR", Path(folder)),
                patch("Stage7.gemma_server.use_gemma", side_effect=fake_server),
                patch.object(audit, "_request", side_effect=fake_request),
            ):
                first = probe.run()
                second = probe.run()
                files = list((Path(folder) / "judgments").glob("*.json"))
                self.assertEqual(len(files), 12)
                self.assertTrue((Path(folder) / "design.json").exists())
                self.assertEqual(first, second)

        self.assertEqual(len(calls), 12)
        self.assertEqual(first["controls_accepted"], 6)
        self.assertEqual(first["injected_errors_detected"], 6)
        self.assertEqual(first["false_positive_count"], 0)
        self.assertEqual(first["false_negative_count"], 0)
        self.assertTrue(first["all_predeclared_checks_passed"])
        self.assertEqual(
            old,
            [
                (case["review_id"], audit._load_checked(audit._path(
                    next(pair for pair in audit._inputs()[0] if pair["review_id"] == case["review_id"])
                ))["sha256"])
                for case in self.cases
            ],
        )

    def test_bad_results_are_reported_not_hidden(self):
        with TemporaryDirectory() as folder:
            with patch.object(probe, "OUTPUT_DIR", Path(folder)):
                for case in self.cases:
                    for variant in ("clean", "corrupted"):
                        # An overly permissive judge accepts every corrupted
                        # claim. The report must expose the full failure.
                        study._freeze(probe._path(case, variant), {
                            "identity": probe._identity(case, variant),
                            "judgment": audit.BatchJudgment(claims=[
                                audit.ClaimJudgment(
                                    index=0, verdict="DIRECT_SUPPORT",
                                    unsupported_specificity=False,
                                    modality_error=False,
                                    rationale="All claims were uncritically accepted.",
                                ),
                            ]).model_dump(mode="json"),
                        })
                result = probe._report(self.cases)
                self.assertEqual(result["sensitivity"], 0)
                self.assertEqual(result["specificity"], 1)
                self.assertEqual(result["false_negative_count"], 6)
                self.assertFalse(result["all_predeclared_checks_passed"])


if __name__ == "__main__":
    unittest.main()
