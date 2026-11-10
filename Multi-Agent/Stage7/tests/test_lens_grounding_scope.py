from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from Stage7 import lens_grounding_scope_audit as scope
from Stage7 import policy_flip_diagnostics as policy
from Stage7.loaders import sha256_json
from Stage7.mediator_ablation import _freeze


class PolicyReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = policy.build_report()

    def test_frozen_main_21_recommendations_and_three_seeds_reproduce(self):
        metrics = self.report["metrics"]
        self.assertEqual(len(self.report["baseline_policy_rows"]), 21)
        self.assertEqual(metrics["baseline_recommendations"], {
            "RECOMMEND": 3,
            "RECOMMEND_PILOT": 9,
            "HOLD": 6,
            "DO_NOT_RECOMMEND": 3,
        })
        self.assertEqual(metrics["baseline_sufficiency_tuples"], {
            "PARTIAL|PARTIAL|PARTIAL": 21,
        })
        self.assertEqual(metrics["case_seed_pairs"], 21)
        self.assertEqual(metrics["case_seed_pairs_with_recommendation_change"], 10)
        self.assertEqual(metrics["case_seed_pairs_with_critical_veto_change"], 2)
        self.assertEqual(metrics["scenario_seed_pairs"], 63)
        self.assertEqual(metrics["scenario_seed_pairs_with_recommendation_change"], 27)
        self.assertEqual(metrics["scenario_seed_pairs_with_critical_veto_change"], 6)
        self.assertEqual(len(self.report["case_flip_profiles"]), 7)
        self.assertEqual(sum(r["any_region_recommendation_flip_pairs"] for r in self.report["case_flip_profiles"]), 10)
        self.assertEqual(sum(r["critical_veto_boundary_flip_pairs"] for r in self.report["case_flip_profiles"]), 2)

    def test_financial_negative_is_hold_not_critical_veto(self):
        self.assertEqual(policy.policy((1, 1, -1)), "HOLD")
        self.assertEqual(policy.policy((-1, 1, 1)), "DO_NOT_RECOMMEND")
        self.assertEqual(policy.policy((1, -1, 1)), "DO_NOT_RECOMMEND")
        self.assertEqual(policy.policy((1, 0, 1)), "HOLD")
        self.assertEqual(policy.policy((1, 1, 0)), "RECOMMEND_PILOT")
        self.assertEqual(policy.policy((1, 1, 1)), "RECOMMEND")

    def test_partial_cap_changes_only_the_three_full_recommends(self):
        rows = self.report["baseline_policy_rows"]
        changed = [
            row for row in rows
            if row["alternatives"]["PARTIAL_RECOMMEND_CAP"] != row["baseline"]
        ]
        self.assertEqual(len(changed), 3)
        self.assertEqual(
            {row["case_id"] for row in changed},
            {"main__malware__anomaly_detection"},
        )
        self.assertTrue(all(
            row["baseline"] == "RECOMMEND"
            and row["alternatives"]["PARTIAL_RECOMMEND_CAP"] == "RECOMMEND_PILOT"
            for row in changed
        ))
        self.assertEqual(self.report["metrics"]["alternative_change_counts"], {
            "PARTIAL_RECOMMEND_CAP": 3,
            "STRICT_FINANCIAL": 9,
            "PERMISSIVE_FINANCIAL": 3,
        })

    def test_one_step_changes_are_not_called_probabilities(self):
        rows = self.report["deterministic_one_step_counterfactuals"]
        self.assertEqual(len(rows), 81)
        self.assertEqual(sum(row["recommendation_changed"] for row in rows), 57)
        self.assertEqual(sum(row["critical_veto_changed"] for row in rows), 6)
        self.assertTrue(all(abs(row["from"] - row["to"]) == 1 for row in rows))
        self.assertIn("NOT a calibrated flip probability", self.report["limitations"])
        self.assertEqual(
            Counter(row["case_id"] for row in self.report["observed_pairwise_seed_comparisons"]),
            {case.case_id: 3 for case in policy.MAIN_STAGE1_CASES},
        )

    def test_variant_file_hash_tamper_is_rejected(self):
        baseline, _ = policy._canonical_rows(policy.ROOT)
        with TemporaryDirectory() as temp:
            p = (
                Path(temp) / "Multi-Agent/Results/Stage7/variants/"
                "B_SEED_ROBUSTNESS"
            )
            p.mkdir(parents=True)
            original = (
                policy.ROOT / "Multi-Agent/Results/Stage7/variants/"
                "B_SEED_ROBUSTNESS/ISOLATED_LENSES__repeat-43.json"
            )
            data = json.loads(original.read_text(encoding="utf-8"))
            data["records"][0]["recommendation"] = "RECOMMEND"
            (p / original.name).write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                policy._seed_rows(Path(temp), baseline)


class LensScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = scope.build_inventory()
        cls.probes = scope._probes(cls.inventory)

    def test_inventory_uses_all_63_lenses_and_frozen_exact_source_records(self):
        diag = self.inventory["structural_diagnostic"]
        self.assertEqual(diag, {
            "scenario_lens_assessments": 63,
            "unique_lens_assessments": 24,
            "directional_claim_instances": 149,
            "eligible_directional_claim_instances": 149,
        })
        self.assertEqual(
            Counter(g["lens"] for g in self.inventory["unique_assessments"]),
            {"institutional_regional": 10, "technical_feasibility": 7, "financial_adoption": 7},
        )
        self.assertEqual(
            sum(len(g["payload"]["claims"]) for g in self.inventory["unique_assessments"]),
            58,
        )
        for group in self.inventory["unique_assessments"]:
            payload = group["payload"]
            claims = payload["claims"]
            self.assertEqual([c["index"] for c in claims], list(range(len(claims))))
            cited = {eid for c in claims for eid in c["evidence_ids"]}
            self.assertEqual(
                cited,
                {record["evidence_id"] for record in payload["evidence_records"]},
            )
            self.assertNotIn("stance", payload)
            self.assertEqual(len({m["decision_object_id"] for m in group["members"]}), len(group["members"]))
            for record in payload["evidence_records"]:
                self.assertIn("source", record)
                self.assertIn("content", record)
                self.assertIn("available_at", record)
        self.assertTrue(all(x["passes_existing_slot_gate"] for x in self.inventory["structural_claims"]))

    def test_scope_probes_change_only_statement_not_cited_sources(self):
        self.assertEqual(len(self.probes), 4)
        self.assertEqual(Counter(p["variant"] for p in self.probes), {"clean": 2, "corrupted": 2})
        by_id = {}
        for probe in self.probes:
            by_id.setdefault(probe["id"], {})[probe["variant"]] = probe["payload"]
            self.assertNotIn(probe["variant"], json.dumps(probe["payload"]))
        for items in by_id.values():
            clean, bad = items["clean"], items["corrupted"]
            for key in ("evidence_records", "context"):
                self.assertEqual(clean[key], bad[key])
            self.assertNotEqual(clean["claims"][0]["statement"], bad["claims"][0]["statement"])
            self.assertEqual(
                {k: v for k, v in clean["claims"][0].items() if k != "statement"},
                {k: v for k, v in bad["claims"][0].items() if k != "statement"},
            )

    @staticmethod
    def fake_review(payload: dict, bad: bool = False) -> scope.ReviewBatch:
        return scope.ReviewBatch(claims=[
            scope.ClaimReview(
                index=c["index"],
                factual_verdict="PARTIAL" if bad else "SUPPORTED",
                action_relevance="MIXED_ACTION_AND_ADJACENT",
                scope_verdict="OVERGENERALIZED" if bad else "GENERAL_ONLY",
                unsupported_economics=bad,
                source_roles=[
                    scope.SourceRole(evidence_id=eid, role="ACTION_SPECIFIC")
                    for eid in c["evidence_ids"]
                ],
                rationale=(
                    "The claim improperly transfers narrow evidence to a "
                    "specific enterprise or invents a quantified financial effect."
                    if bad else "The cited record supports only this general feasibility statement."
                ),
            )
            for c in payload["claims"]
        ])

    def test_auditor_rejects_missing_and_duplicated_citation_roles(self):
        payload = self.probes[0]["payload"]
        valid = self.fake_review(payload)
        self.assertEqual(scope._validate(valid, payload), valid)
        bad = valid.model_copy(deep=True)
        bad.claims[0].source_roles.clear()
        with self.assertRaisesRegex(ValueError, "cited-source role"):
            scope._validate(bad, payload)

    def test_auditor_input_schema_and_no_live_inference(self):
        payload = self.probes[0]["payload"]
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": self.fake_review(payload).model_dump_json()}}],
        }
        with patch.object(scope.requests, "post", return_value=response) as post:
            answer = scope._request(payload)
        self.assertEqual(answer.claims[0].factual_verdict, "SUPPORTED")
        args = post.call_args.kwargs["json"]
        self.assertEqual(args["model"], scope.MODEL)
        self.assertEqual(args["seed"], 42)
        self.assertEqual(args["temperature"], 0)
        self.assertEqual(args["messages"][1]["content"], scope.canonical_json(payload))
        self.assertIn("schema", args["response_format"])

    def test_bad_batch_numbering_falls_back_to_individual_claims(self):
        group = next(g for g in self.inventory["unique_assessments"] if len(g["payload"]["claims"]) == 2)
        payload = group["payload"]
        def reply(answer: scope.ReviewBatch):
            response = Mock()
            response.json.return_value = {
                "choices": [{"message": {"content": answer.model_dump_json()}}],
            }
            return response

        first = self.fake_review(payload)
        invalid = scope.ReviewBatch(claims=first.claims[:1])
        singles = []
        for claim in payload["claims"]:
            subset = {
                **payload,
                "claims": [{**claim, "index": 0}],
                "evidence_records": [
                    record for record in payload["evidence_records"]
                    if record["evidence_id"] in claim["evidence_ids"]
                ],
            }
            singles.append(reply(self.fake_review(subset)))
        with patch.object(
            scope.requests, "post",
            side_effect=[reply(invalid), reply(invalid), *singles],
        ) as mocked:
            result = scope._request(payload)
        self.assertEqual([claim.index for claim in result.claims], [0, 1])
        self.assertEqual(mocked.call_count, 4)

    def test_caching_and_scoped_corruptions_report_without_inference(self):
        with TemporaryDirectory() as temp:
            with (
                patch.object(scope, "OUTPUT", Path(temp)),
                patch.object(scope.requests, "post", side_effect=AssertionError("No inference allowed")),
            ):
                for group in self.inventory["unique_assessments"]:
                    _freeze(scope._path(group["group_id"]), {
                        "identity": scope._identity(group["payload"]),
                        "judgment": self.fake_review(group["payload"]).model_dump(mode="json"),
                    })
                for probe in self.probes:
                    _freeze(scope._path(probe["id"] + "__" + probe["variant"]), {
                        "identity": scope._identity(probe["payload"]),
                        "judgment": self.fake_review(
                            probe["payload"], bad=probe["variant"] == "corrupted",
                        ).model_dump(mode="json"),
                    })

                report = scope.build_report(self.inventory)
                self.assertEqual(report["unique_lens_review_count"], 24)
                self.assertEqual(report["directional_claim_review_count"], 58)
                self.assertEqual(report["scope_clean_controls_accepted"], 2)
                self.assertEqual(report["scope_corruptions_rejected"], 2)
                self.assertEqual(report["scope_pairs_discriminated"], 2)
                self.assertEqual(report["verdict_counts"]["SUPPORTED"], 58)
                self.assertTrue(all(
                    claim["claim_statement"] and claim["cited_evidence_ids"]
                    for row in report["lens_reviews"] for claim in row["claim_reviews"]
                ))
                # The mock says GENERAL_ONLY for every claim; the one frozen
                # DEPLOYMENT_SPECIFIC claim must therefore be flagged.
                self.assertEqual(report["flagged_directional_claim_count"], 1)
                first = self.inventory["unique_assessments"][0]
                checkpoint = scope._path(first["group_id"])
                corrupted = json.loads(checkpoint.read_text(encoding="utf-8"))
                corrupted["identity"]["input_sha256"] = "tampered"
                _ = checkpoint.write_text(json.dumps(corrupted), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    scope._load_result(first["group_id"], first["payload"])

    def test_general_only_audit_cannot_validate_deployment_specific_claim(self):
        group = next(
            g for g in self.inventory["unique_assessments"]
            if any(claim["scope"] == "DEPLOYMENT_SPECIFIC" for claim in g["payload"]["claims"])
        )
        with TemporaryDirectory() as temp:
            with patch.object(scope, "OUTPUT", Path(temp)):
                for candidate in self.inventory["unique_assessments"]:
                    _freeze(scope._path(candidate["group_id"]), {
                        "identity": scope._identity(candidate["payload"]),
                        "judgment": self.fake_review(candidate["payload"]).model_dump(mode="json"),
                    })
                for item in self.probes:
                    _freeze(scope._path(item["id"] + "__" + item["variant"]), {
                        "identity": scope._identity(item["payload"]),
                        "judgment": self.fake_review(item["payload"]).model_dump(mode="json"),
                    })
                report = scope.build_report(self.inventory)
                deployment_ids = {
                    claim["claim_id"] for claim in group["payload"]["claims"]
                    if claim["scope"] == "DEPLOYMENT_SPECIFIC"
                }
                self.assertTrue(
                    any(
                        group_key == group["group_id"] and claim_id in deployment_ids
                        for group_key, claim_id in report["flagged_directional_claims"]
                    )
                )

    def test_both_arms_rejected_is_not_a_successful_scope_discrimination(self):
        with TemporaryDirectory() as temp:
            with patch.object(scope, "OUTPUT", Path(temp)):
                for group in self.inventory["unique_assessments"]:
                    _freeze(scope._path(group["group_id"]), {
                        "identity": scope._identity(group["payload"]),
                        "judgment": self.fake_review(group["payload"]).model_dump(mode="json"),
                    })
                for item in self.probes:
                    _freeze(scope._path(item["id"] + "__" + item["variant"]), {
                        "identity": scope._identity(item["payload"]),
                        "judgment": self.fake_review(item["payload"], bad=True).model_dump(mode="json"),
                    })
                result = scope.build_report(self.inventory)
                self.assertEqual(result["scope_clean_controls_accepted"], 0)
                self.assertEqual(result["scope_corruptions_rejected"], 2)
                self.assertEqual(result["scope_pairs_discriminated"], 0)

    def test_inventory_cli_never_starts_gemma(self):
        with TemporaryDirectory() as temp:
            with (
                patch.object(scope, "OUTPUT", Path(temp)),
                patch.object(sys, "argv", ["lens_grounding_scope_audit", "--inventory"]),
                patch.object(scope.requests, "post", side_effect=AssertionError("Must not infer")),
            ):
                scope.main()
                self.assertTrue((Path(temp) / "lens_scope_inventory.json").is_file())
                self.assertFalse((Path(temp) / "lens_scope_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
