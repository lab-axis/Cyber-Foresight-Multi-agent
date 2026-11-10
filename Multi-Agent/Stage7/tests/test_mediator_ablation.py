from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from Stage1.schema import CriticType
from Stage7 import mediator_ablation as study


class MediatorAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = study._sources()

    def test_exact_frozen_input_and_round_one_case_partition(self):
        self.assertEqual(len(self.sources), 7)

        for source in self.sources:
            if source["case_id"] == study.TWO_ROUND_CASE:
                continue

            for role in study.ROLES:
                self.assertEqual(
                    study._validate_post(source, role, study._baseline(source, role)).stance,
                    study._baseline(source, role)["stance"],
                )

        self.assertEqual(sum(len(s["result"]["rounds"]) == 2 for s in self.sources), 1)

    def test_mediator_removal_preserves_cross_review_and_evidence_scope(self):
        for source in self.sources:
            for role in study.ROLES:
                no = study._request(source, role, mediator=False)
                yes = study._request(source, role, mediator=True)
                self.assertEqual(no["schema"], yes["schema"])
                self.assertEqual(no["evidence_ids"], yes["evidence_ids"])
                self.assertEqual(len(no["evidence_ids"]), len(set(no["evidence_ids"])))

                def rounds(request):
                    body = request["system_prompt"].split(
                        "Completed Stage 2 rounds:\n", 1
                    )[1].split("\n\nNow produce your own post-debate", 1)[0]
                    return json.loads(body)

                no_round = rounds(no)[0]
                yes_round = rounds(yes)[0]
                self.assertEqual(
                    set(no_round), {"round_index", "attack_turn", "defense_turn"}
                )
                self.assertEqual(
                    set(yes_round),
                    {"round_index", "attack_turn", "defense_turn", "mediator_summary"},
                )
                for turn_name in ("attack_turn", "defense_turn"):
                    self.assertEqual(no_round[turn_name], source["round1"][turn_name])
                    self.assertEqual(yes_round[turn_name], source["round1"][turn_name])

                self.assertEqual(yes_round["mediator_summary"], source["round1"]["mediator_summary"])

    def test_no_mediator_post_cannot_introduce_unseen_evidence(self):
        source = next(
            s for s in self.sources
            if len(study._allowed_ids(s)) < len(s["evidence_pack"]["evidence"])
            and s["case_id"] != study.TWO_ROUND_CASE
        )
        role = CriticType.ATTACK_FEASIBILITY
        allowed = set(study._allowed_ids(source))
        absent = [r["evidence_id"] for r in source["evidence_pack"]["evidence"] if r["evidence_id"] not in allowed]

        value = study._baseline(source, role)
        value["claims"][0]["evidence_ids"] = [absent[0]]

        with self.assertRaisesRegex(ValueError, "unsurfaced evidence"):
            study._validate_post(source, role, value)

    def test_record_integrity_and_no_overwrite(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            data = {"hello": 42}
            study._freeze(path, data)
            self.assertEqual(study._freeze(path, data)["hello"], 42)

            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                study._freeze(path, {"hello": 43})

            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["hello"] = 99
            path.write_text(json.dumps(tampered), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                study._load_checked(path)

    def test_run_uses_single_slot_and_16_independent_post_assessments(self):
        worker = AsyncMock(return_value={})

        with (
            patch.object(study, "assert_llama_server_ready") as ready,
            patch.object(study, "_generate", worker),
        ):
            asyncio.run(study._run(self.sources))

        ready.assert_called_once_with(
            expected_context_size=131072,
            expected_parallel_slots=1,
            profile_name="Mediator ablation: historical Stage 2 one-slot runtime",
        )
        self.assertEqual(worker.await_count, 16)
        self.assertEqual(
            sum(call.kwargs["mediator"] for call in worker.await_args_list), 2
        )

    def test_run_switches_to_one_slot_and_restores_original_server(self):
        command = ["llama-server.exe", "--parallel", "2"]
        result = {"comparison_count": 14}

        with (
            patch.object(study, "_server", return_value=(object(), command)),
            patch.object(study, "_ready") as ready,
            patch.object(study, "_switch_server") as switch,
            patch.object(study, "_run", new_callable=AsyncMock) as run,
            patch.object(study, "_report", return_value=result),
        ):
            actual = study._run_all(self.sources)

        self.assertEqual(actual, result)
        ready.assert_called_once_with(2)
        self.assertEqual(
            [call.args[1] for call in switch.call_args_list], [1, 2]
        )
        run.assert_awaited_once_with(self.sources)

    def test_run_restores_original_server_after_inference_failure(self):
        command = ["llama-server.exe", "--parallel", "2"]

        with (
            patch.object(study, "_server", return_value=(object(), command)),
            patch.object(study, "_ready"),
            patch.object(study, "_switch_server") as switch,
            patch.object(
                study, "_run", new_callable=AsyncMock,
                side_effect=RuntimeError("inference failed"),
            ),
            patch.object(study, "_report") as report,
        ):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                study._run_all(self.sources)

        self.assertEqual(
            [call.args[1] for call in switch.call_args_list], [1, 2]
        )
        report.assert_not_called()

    def test_one_round_mediator_baseline_is_not_taken_from_two_round_main(self):
        brute = next(s for s in self.sources if s["case_id"] == study.TWO_ROUND_CASE)

        with TemporaryDirectory() as directory:
            with patch.object(study, "OUTPUT", Path(directory)):
                with self.assertRaisesRegex(ValueError, "independently generated"):
                    study._baseline(brute, CriticType.DEFENSE_ROBUSTNESS)

    def test_cached_record_identity_rejects_changed_source(self):
        source = next(s for s in self.sources if s["case_id"] != study.TWO_ROUND_CASE)
        role = CriticType.ATTACK_FEASIBILITY

        with TemporaryDirectory() as directory:
            with patch.object(study, "OUTPUT", Path(directory)):
                path = study._record_path(source, role, False)
                study._freeze(path, {
                    "identity": {**study._identity(source, role, False), "seed": 43},
                    "assessment": study._baseline(source, role),
                })

                with self.assertRaisesRegex(ValueError, "no longer matches"):
                    asyncio.run(study._generate(source, role, mediator=False))

    def test_missing_ablation_folder_does_not_appear_complete(self):
        with TemporaryDirectory() as directory:
            with patch.object(study, "OUTPUT", Path(directory)):
                self.assertFalse(study._records_complete(self.sources))

    def test_run_resumes_directly_at_gemma_when_qwen_results_exist(self):
        with (
            patch.object(sys, "argv", ["mediator_ablation", "--run"]),
            patch.object(study, "_sources", return_value=self.sources),
            patch.object(study, "_records_complete", return_value=True),
            patch.object(study, "_run_all") as qwen,
            patch.object(study, "_report", return_value={
                "comparison_count": 14, "matched_case_count": 6,
            }),
            patch("Stage7.mediator_quality_analysis.run", return_value={
                "audit_pairs": 14, "total_claims_audited": 229,
            }) as gemma,
        ):
            study.main()

        qwen.assert_not_called()
        gemma.assert_called_once_with()

    def test_report_is_structural_and_exports_condition_blinded_evidence_review(self):
        source = next(s for s in self.sources if s["case_id"] != study.TWO_ROUND_CASE)

        with TemporaryDirectory() as directory:
            with patch.object(study, "OUTPUT", Path(directory)):
                for role in study.ROLES:
                    study._freeze(study._record_path(source, role, False), {
                        "identity": study._identity(source, role, False),
                        "assessment": study._baseline(source, role),
                    })

                summary = study._report([source])
                review = study._load_checked(study._report_path([source], "blind_review.json"))
                mapping = study._load_checked(study._report_path([source], "blind_key.json"))
                self.assertTrue(study._report_path([source], "summary.json").is_file())
                self.assertFalse((study.OUTPUT / "summary.json").exists())

        self.assertEqual(summary["matched_case_count"], 1)
        self.assertEqual(summary["comparison_count"], 2)
        self.assertEqual(len(review["items"]), 2)
        self.assertEqual(len(mapping["mapping"]), 2)

        for item in review["items"]:
            self.assertEqual(
                set(item["review_fields"]), {"A", "B"}
            )
            self.assertTrue(item["evidence_records"])
            self.assertEqual(
                set(item["cross_review_round1"]),
                {"attack_turn", "defense_turn"},
            )
            self.assertEqual(
                item["cross_review_round1"]["attack_turn"],
                source["round1"]["attack_turn"],
            )
            self.assertEqual(
                item["cross_review_round1"]["defense_turn"],
                source["round1"]["defense_turn"],
            )
            self.assertNotIn("with_mediator", json.dumps(item))
            self.assertNotIn("without_mediator", json.dumps(item))


if __name__ == "__main__":
    unittest.main()
