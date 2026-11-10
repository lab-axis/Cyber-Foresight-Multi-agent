from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from pydantic import ValidationError

from Stage7 import gemma_server as server
from Stage7 import mediator_quality_analysis as audit
from Stage7 import mediator_ablation as study


class GemmaQualityAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # These fixtures use the original frozen Stage 1/2 results only.
        # They do not depend on any output in Results/Stage7/mediator_ablation.
        cls.sources = study._sources()

    @staticmethod
    def pair(source: dict, role: str = "attack_feasibility") -> dict:
        name = "attack_post_assessment" if role == "attack_feasibility" else "defense_post_assessment"
        original = source["result"][name]
        return {
            "review_id": f"{source['case_id']}__{role}",
            "forecast_summary": source["evidence_pack"]["forecast_summary"],
            "evidence_records": source["evidence_pack"]["evidence"],
            "cross_review_round1": {
                "attack_turn": source["round1"]["attack_turn"],
                "defense_turn": source["round1"]["defense_turn"],
            },
            "assessment_A": original,
            "assessment_B": original,
        }

    @staticmethod
    def judgment(pair: dict, bad_index: int | None = None) -> audit.BatchJudgment:
        return audit.BatchJudgment(claims=[
            audit.ClaimJudgment(
                index=index,
                verdict="PARTIAL_SUPPORT" if index == bad_index else "DIRECT_SUPPORT",
                unsupported_specificity=index == bad_index,
                modality_error=False,
                rationale="The cited evidence does not support the exact date." if index == bad_index
                          else "The cited evidence supports the proposition.",
            )
            for index, _ in enumerate(audit._claims(pair))
        ])

    def test_all_14_frozen_case_role_fixtures_preserve_entire_cited_records(self):
        self.assertEqual(len(self.sources), 7)
        for source in self.sources:
            for role in ("attack_feasibility", "defense_robustness"):
                pair = self.pair(source, role)
                payload = audit._input(pair)
                originals = {
                    record["evidence_id"]: record
                    for record in source["evidence_pack"]["evidence"]
                }
                self.assertTrue(payload["claims"])
                self.assertEqual({claim["arm"] for claim in payload["claims"]}, {"A", "B"})
                self.assertEqual(
                    payload["evidence_records"],
                    [r for r in pair["evidence_records"] if r["evidence_id"] in {
                        eid for claim in payload["claims"] for eid in claim["evidence_ids"]
                    }],
                )
                for record in payload["evidence_records"]:
                    self.assertEqual(record, originals[record["evidence_id"]])
                    for key in (
                        "source", "source_document_id", "source_locator",
                        "publication_date", "available_at", "evidence_type",
                        "content", "allowed_claim_types", "prohibited_claim_types",
                        "threat_ids", "pmt_ids",
                    ):
                        self.assertIn(key, record)
                self.assertNotIn("mediator_summary", payload["cross_review_round1"])
                self.assertEqual(
                    payload["forecast_summary"]["threat_state"],
                    pair["forecast_summary"]["threat_state"],
                )
                self.assertEqual(
                    payload["forecast_summary"]["gap_by_year"],
                    pair["forecast_summary"]["gap_by_year"],
                )
                self.assertNotIn("historical_input", payload["forecast_summary"])
                self.assertNotIn("historical_field_ids", payload["forecast_summary"])
                self.assertNotIn("with_mediator", json.dumps(payload))
                self.assertNotIn("without_mediator", json.dumps(payload))

    def test_previously_missed_nist_source_and_crossref_dates_are_in_input(self):
        source = next(s for s in self.sources if s["case_id"] == "main__insider_threat__cryptography")
        payload = audit._input(self.pair(source))
        records = {r["evidence_id"]: r for r in payload["evidence_records"]}
        nist = records["E_20_913B6F999A0F1458"]
        self.assertIn("800-161", nist["source"])
        self.assertEqual(nist["source_document_id"], "nist-sp-800-161r1")
        expected = {
            "E_29_185B62FFF51926EC": "2021-01-30",
            "E_29_3C1B7B4F9532A5F2": "2023-01-18",
            "E_29_7C01F6CA0D982DFF": "2018-01-16",
        }
        for eid, date in expected.items():
            self.assertEqual(records[eid]["publication_date"], date)

    def test_shared_critic_debate_is_available_without_mediator_summary(self):
        source = next(s for s in self.sources if s["case_id"] == "main__ransomware__cryptography")
        pair = self.pair(source, role="defense_robustness")
        payload = audit._input(pair)
        self.assertEqual(payload["cross_review_round1"]["attack_turn"], source["round1"]["attack_turn"])
        self.assertEqual(payload["cross_review_round1"]["defense_turn"], source["round1"]["defense_turn"])
        self.assertNotIn("mediator_summary", payload["cross_review_round1"])
        self.assertIn("cross-review", audit.SYSTEM)
        self.assertIn("logical", audit.SYSTEM)

    def test_missing_cross_review_or_cited_evidence_is_rejected(self):
        pair = self.pair(self.sources[0])
        pair.pop("cross_review_round1")
        with self.assertRaisesRegex(ValueError, "cross-review missing"):
            audit._input(pair)

        pair = self.pair(self.sources[0])
        pair["evidence_records"] = []
        with self.assertRaisesRegex(ValueError, "Missing exact cited record"):
            audit._input(pair)

    def test_direct_support_with_an_error_flag_is_invalid(self):
        for field in ("unsupported_specificity", "modality_error"):
            values = {
                "index": 0,
                "verdict": "DIRECT_SUPPORT",
                "unsupported_specificity": False,
                "modality_error": False,
                "rationale": "This cited text supports the statement.",
            }
            values[field] = True
            with self.assertRaises(ValidationError):
                audit.ClaimJudgment(**values)

    def test_all_claim_indices_required(self):
        pair = self.pair(self.sources[0])
        answer = self.judgment(pair)
        self.assertEqual(audit._validate(answer, pair), answer)
        with self.assertRaisesRegex(ValueError, "Missing or duplicated"):
            audit._validate(answer.model_copy(update={"claims": answer.claims[1:]}), pair)

    def test_claim_batches_cover_both_arms_once_and_keep_original_evidence(self):
        for source in self.sources:
            for role in ("attack_feasibility", "defense_robustness"):
                pair = self.pair(source, role)
                batches = audit._batches(pair)
                all_indices = [index for batch in batches for index in batch]
                self.assertEqual(sorted(all_indices), list(range(len(audit._claims(pair)))))
                self.assertLessEqual(max(map(len, batches)), 4)

                for indices in batches:
                    payload = audit._input(pair, indices)
                    self.assertEqual(
                        {claim["index"] for claim in payload["claims"]},
                        set(indices),
                    )
                    self.assertLess(len(audit._canonical(payload)), 40000)
                    cited = {
                        evidence_id for claim in payload["claims"]
                        for evidence_id in claim["evidence_ids"]
                    }
                    self.assertEqual(
                        {r["evidence_id"] for r in payload["evidence_records"]},
                        cited,
                    )
                    original = {
                        r["evidence_id"]: r for r in pair["evidence_records"]
                    }
                    for record in payload["evidence_records"]:
                        self.assertEqual(record, original[record["evidence_id"]])
                    with self.assertRaisesRegex(ValueError, "Missing or duplicated"):
                        audit._validate(
                            self.judgment(pair), pair, indices,
                        )

    def test_partial_batch_checkpoint_resumes_without_repeating_completed_requests(self):
        pair = self.pair(self.sources[0])
        requests_made = []

        def fake_request(value, indices):
            requests_made.append(indices)
            return audit.BatchJudgment(claims=[
                audit.ClaimJudgment(
                    index=i, verdict="DIRECT_SUPPORT",
                    unsupported_specificity=False, modality_error=False,
                    rationale="The complete source record supports the claim.",
                )
                for i in indices
            ])

        with TemporaryDirectory() as directory:
            with (
                patch.object(audit, "OUTPUT", Path(directory)),
                patch.object(audit, "_request", side_effect=fake_request),
            ):
                asyncio.run(audit._audit(pair))
                first_run_calls = len(requests_made)
                self.assertEqual(first_run_calls, len(audit._batches(pair)))
                self.assertIsNotNone(audit._load_checked(audit._path(pair)))

                audit._path(pair).unlink()
                audit._batch_path(pair, first_run_calls - 1).unlink()
                asyncio.run(audit._audit(pair))

                self.assertEqual(len(requests_made), first_run_calls + 1)
                self.assertEqual(
                    len(audit._load_checked(audit._path(pair))["judgment"]["claims"]),
                    len(audit._claims(pair)),
                )

    def test_input_hash_changes_when_source_metadata_changes(self):
        pair = self.pair(self.sources[0])
        old = audit._identity(pair)
        self.assertIn("-v3", old["version"])
        with patch.object(
            audit, "_input",
            wraps=audit._input,
        ):
            pair["evidence_records"] = [dict(x) for x in pair["evidence_records"]]
            pair["evidence_records"][0]["source"] += " altered"
            new = audit._identity(pair)
        self.assertNotEqual(old["input_sha256"], new["input_sha256"])

    def test_output_contains_adverse_claim_transitions_and_full_rationales(self):
        pair = self.pair(self.sources[0])
        mapping = {
            pair["review_id"]: {
                "A": "with_mediator", "B": "without_mediator",
                "analysis": "matched_one_round",
            },
        }
        answer = self.judgment(pair, bad_index=0)

        with TemporaryDirectory() as directory:
            with patch.object(audit, "OUTPUT", Path(directory)):
                study._freeze(audit._path(pair), {
                    "identity": audit._identity(pair),
                    "judgment": answer.model_dump(mode="json"),
                })
                result = audit._report([pair], mapping)
                details = audit._load_checked(audit.OUTPUT / "quality_claim_details.json")
                summary = audit._load_checked(audit.OUTPUT / "quality_summary.json")

        self.assertEqual(result, {k: v for k, v in summary.items() if k != "sha256"})
        self.assertEqual(result["audit_pairs"], 1)
        self.assertEqual(result["matched_cases"], 1)
        self.assertEqual(result["primary_shared_claim_ids_favor_without_mediator"], 1)
        self.assertEqual(result["primary_shared_claim_ids_favor_mediator"], 0)
        self.assertEqual(result["primary_case_direction_counts"], {
            "favors_mediator": 0,
            "favors_without_mediator": 1,
            "tied": 0,
        })
        self.assertEqual(len(details["claims"]), len(audit._claims(pair)))
        self.assertEqual(details["claims"][0]["condition"], "with_mediator")
        self.assertEqual(details["claims"][0]["judgment"]["verdict"], "PARTIAL_SUPPORT")
        self.assertTrue(details["claims"][0]["judgment"]["rationale"])

    def test_existing_audit_cache_does_not_reload_model(self):
        pair = self.pair(self.sources[0])
        mapping = {pair["review_id"]: {
            "A": "with_mediator", "B": "without_mediator",
            "analysis": "matched_one_round",
        }}

        with TemporaryDirectory() as directory:
            with (
                patch.object(audit, "OUTPUT", Path(directory)),
                patch.object(audit, "_inputs", return_value=([pair], mapping)),
                patch("Stage7.gemma_server.use_gemma") as use_gemma,
            ):
                study._freeze(audit._path(pair), {
                    "identity": audit._identity(pair),
                    "judgment": self.judgment(pair).model_dump(mode="json"),
                })
                result = audit.run()
                use_gemma.assert_not_called()

        self.assertEqual(result["audit_pairs"], 1)
        self.assertEqual(result["primary_case_macro_delta_direct_support"], 0)

    def test_request_uses_gemma_and_structured_validation(self):
        pair = self.pair(self.sources[0])
        answer = self.judgment(pair)
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": answer.model_dump_json()}}],
        }
        with patch.object(audit.requests, "post", return_value=response) as post:
            self.assertEqual(audit._request(pair), answer)

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], audit.MODEL)
        self.assertEqual(payload["seed"], 42)
        self.assertIn("schema", payload["response_format"])
        self.assertNotIn("reasoning_format", payload)

    def test_nonconsecutive_indices_are_renumbered_locally_then_restored(self):
        pair = self.pair(self.sources[2], role="defense_robustness")
        indices = [4, 5, 13, 14]
        response = Mock()
        response.json.return_value = {
            "choices": [{
                "message": {
                    "content": audit.BatchJudgment(claims=[
                        audit.ClaimJudgment(
                            index=i, verdict="DIRECT_SUPPORT",
                            unsupported_specificity=False,
                            modality_error=False,
                            rationale="This specific claim is supported by the cited record.",
                        )
                        for i in range(4)
                    ]).model_dump_json()
                },
            }],
        }
        with patch.object(audit.requests, "post", return_value=response) as post:
            judged = audit._request(pair, indices)

        self.assertEqual([j.index for j in judged.claims], indices)
        sent = post.call_args.kwargs["json"]
        delivered = json.loads(sent["messages"][1]["content"])
        self.assertEqual([c["index"] for c in delivered["claims"]], [0, 1, 2, 3])
        self.assertEqual(
            [c["claim_id"] for c in delivered["claims"]],
            [c["claim_id"] for c in audit._claims(pair) if c["index"] in indices],
        )
        self.assertEqual(
            sent["response_format"]["schema"]["$defs"]["ClaimJudgment"]["properties"]["index"]["enum"],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            [e["evidence_id"] for e in delivered["evidence_records"]],
            [e["evidence_id"] for e in audit._input(pair, indices)["evidence_records"]],
        )

    def test_bad_local_indices_trigger_repair_without_rejudging_saved_batches(self):
        pair = self.pair(self.sources[2], role="defense_robustness")
        indices = [4, 5, 13, 14]
        def make_reply(values):
            response = Mock()
            response.json.return_value = {
                "choices": [{"message": {"content": audit.BatchJudgment(claims=[
                    audit.ClaimJudgment(
                        index=i, verdict="DIRECT_SUPPORT",
                        unsupported_specificity=False, modality_error=False,
                        rationale="The cited records fully support the specified claim.",
                    )
                    for i in values
                ]).model_dump_json()}}],
            }
            return response

        # The first simulated answer reproduces the historical sparse-index
        # failure; only the unsuccessful batch is retried.
        with patch.object(
            audit.requests, "post",
            side_effect=[make_reply(indices), make_reply([0, 1, 2, 3])],
        ) as post:
            result = audit._request(pair, indices)
        self.assertEqual([j.index for j in result.claims], indices)
        self.assertEqual(post.call_count, 2)

    def test_invalid_batch_indices_fall_back_to_single_claims(self):
        pair = self.pair(self.sources[2], role="defense_robustness")
        indices = [4, 5, 13, 14]
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": audit.BatchJudgment(claims=[
                audit.ClaimJudgment(
                    index=0, verdict="DIRECT_SUPPORT",
                    unsupported_specificity=False, modality_error=False,
                    rationale="The cited text supports this particular claim.",
                ),
            ]).model_dump_json()}}],
        }
        with patch.object(audit.requests, "post", return_value=response) as post:
            result = audit._request(pair, indices)

        self.assertEqual(sorted(j.index for j in result.claims), indices)
        self.assertEqual(post.call_count, 2 + len(indices))

    def test_gemma_one_slot_preserves_total_context_and_kv_cache(self):
        original = [
            "llama-server.exe", "--model", "Qwen.gguf", "--alias", "Qwen",
            "--ctx-size", "131072", "--parallel", "2",
            "--batch-size", "2048", "--ubatch-size", "512",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "--n-gpu-layers", "all", "--spec-type", "draft-mtp",
            "--reasoning-format", "deepseek", "--reasoning-preserve",
        ]
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            model = folder / "Gemma.gguf"
            model.write_bytes(b"1234")
            (folder / "download_verified.sha256").write_text("testhash\n", encoding="utf-8")
            with (
                patch.object(server, "MODEL", model),
                patch.object(server, "DESTINATION", folder),
                patch.object(server, "SIZE", 4),
                patch.object(server, "SHA256", "testhash"),
            ):
                cmd = server._gemma_command(original)

        for flag in (
            "--ctx-size", "--batch-size", "--ubatch-size",
            "--cache-type-k", "--cache-type-v", "--n-gpu-layers",
        ):
            self.assertEqual(cmd[cmd.index(flag) + 1], original[original.index(flag) + 1])
        self.assertEqual(cmd[cmd.index("--parallel") + 1], "1")
        self.assertNotIn("--spec-type", cmd)

    def test_gemma_audits_are_sequential(self):
        active = 0
        peak = 0
        completed = []
        pairs = [self.pair(s) for s in self.sources]

        async def fake_audit(pair):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            completed.append(pair["review_id"])
            active -= 1

        with patch.object(audit, "_audit", side_effect=fake_audit):
            asyncio.run(audit._run_pairs(pairs))
        self.assertEqual(peak, 1)
        self.assertEqual(len(completed), 7)

    def test_server_restoration_after_audit_failure(self):
        original = ["llama-server.exe", "--model", "Qwen.gguf", "--parallel", "2"]
        qwen = Mock()
        qwen.is_running.return_value = True
        gemma = Mock()
        gemma.poll.return_value = None

        with (
            patch.object(server, "_server", return_value=(qwen, original)),
            patch.object(server, "_ready"),
            patch.object(server, "_ensure_idle"),
            patch.object(server, "_wait_idle_after_requests"),
            patch.object(server, "_gemma_command", return_value=["gemma.exe"]),
            patch.object(server, "_launch", return_value=gemma) as launch,
        ):
            with self.assertRaisesRegex(RuntimeError, "audit failed"):
                with server.use_gemma():
                    raise RuntimeError("audit failed")

        qwen.terminate.assert_called_once()
        gemma.terminate.assert_called_once()
        self.assertEqual([c.args[1] for c in launch.call_args_list], ["gemma", "qwen_restored"])
        self.assertEqual(launch.call_args_list[1].args[0], original)

    def test_wait_after_timeout_checks_for_idle_before_shutdown(self):
        with (
            patch.object(server, "_ensure_idle", side_effect=[
                RuntimeError("busy"), RuntimeError("busy"), None,
            ]) as idle,
            patch.object(server.time, "sleep"),
        ):
            server._wait_idle_after_requests()
        self.assertEqual(idle.call_count, 3)


if __name__ == "__main__":
    unittest.main()
