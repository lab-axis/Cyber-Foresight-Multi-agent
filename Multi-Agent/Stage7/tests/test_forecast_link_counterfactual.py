from __future__ import annotations

import asyncio
import json
import os
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

from Pipeline.main import _case_config
from Stage1.cases import MAIN_STAGE1_CASES
from Stage1.schema import CriticType, Stage1ValidationError, validate_critic_assessment
from Stage1.schema import CriticAssessment
from Stage2.schema import CriticDebateTurn, MediatorSummary
from Stage3.builder import build_stage3_bundle
from Stage3.context_builder import build_contextual_stage3_bundle
from Stage0.agent_input import build_agent_input
from Stage7.forecast_link_counterfactual import (
    ARMS, ROOT, SANDBOX, SEEDS, _compare, _graph, _inputs, _isolated_nodes,
    _load_frozen, _manifest, _paired_inputs, _record, _report, _run,
    _import_original, _artifacts, _stage4_completed, _historical_stage2_debate_node,
    _within_arm_agreement, _write_frozen, _validate_record,
    _mediator_with_recovery, _previous_mediator_failure,
)
from Stage7.loaders import load_main_case_bundles, sha256_json


class ForecastLinkCounterfactualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ids = [b.case_id for b in load_main_case_bundles(ROOT)]
        cls.datasets, cls.provenance = _inputs(ids)
        cls.bundles = {b.case_id: b for b in load_main_case_bundles(ROOT)}

    def test_all_seven_frozen_pairs_keep_exact_factual_evidence(self):
        self.assertEqual(len(self.datasets), 7)
        self.assertEqual(len(SEEDS) * len(ARMS) * len(self.datasets), 42)
        cases = {case.case_id: case for case in MAIN_STAGE1_CASES}
        for case_id, pair in self.datasets.items():
            _paired_inputs(pair["original"], pair["reversed_pmt"])
            config = _case_config(cases[case_id])
            baseline = build_agent_input(
                project_root=ROOT,
                snapshot_id=config["snapshot_id"],
                threat=cases[case_id].threat,
                pmt=cases[case_id].pmt,
                analysis_cutoff_date=config["analysis_cutoff_date"],
                evaluation_mode=config["evaluation_mode"],
                case_id=case_id,
            )
            self.assertEqual(pair["original"], baseline)
            self.assertNotIn("study_grounding_contract", json.loads(pair["original"]["forecast_data"]))
            for arm, payload in pair.items():
                self.assertEqual(
                    self.provenance[case_id][arm]["evidence_sha256"],
                    sha256_json(payload["evidence_pack"]["evidence"]),
                )
            self.assertNotEqual(
                pair["original"]["forecast_data"],
                pair["reversed_pmt"]["forecast_data"],
            )

    def test_synthetic_provenance_and_gap_arithmetic(self):
        for case_id, provenance in self.provenance.items():
            source = provenance["original"]["monthly_pmt"]
            changed = provenance["reversed_pmt"]["monthly_pmt"]
            self.assertEqual([x["original_value_z"] for x in source], [
                x["original_value_z"] for x in changed
            ])
            self.assertEqual(
                [x["experiment_value_z"] for x in changed],
                list(reversed([x["experiment_value_z"] for x in source])),
            )
            self.assertEqual(len(source), 36)
            self.assertTrue(all(
                x["experiment_field_id"].startswith("FCE_") for x in changed
            ))
            self.assertTrue(all(
                x["experiment_field_id"] == x["source_field_id"] for x in source
            ))
            self.assertEqual(len(provenance["reversed_pmt"]["annual_gap"]), 3)
            self.assertEqual(
                self.datasets[case_id]["original"]["evidence_pack"]["forecast_summary"]["pmt_state"]["forecast_mean_z"],
                self.datasets[case_id]["reversed_pmt"]["evidence_pack"]["forecast_summary"]["pmt_state"]["forecast_mean_z"],
            )

    def test_shared_cryptography_pmt_has_identical_synthetic_trajectory(self):
        a = self.provenance["main__ransomware__cryptography"]
        b = self.provenance["main__insider_threat__cryptography"]
        for arm in ARMS:
            self.assertEqual(a[arm]["monthly_pmt"], b[arm]["monthly_pmt"])

    def test_seven_case_contextual_evidence_stays_identical(self):
        cases = {case.case_id: case for case in MAIN_STAGE1_CASES}
        for case_id, pair in self.datasets.items():
            evidence_by_arm = {}
            for arm, payload in pair.items():
                pack = payload["evidence_pack"]
                base = build_stage3_bundle(
                    evidence_pack=pack,
                    stage0_config=_case_config(cases[case_id]),
                    stage2_result=self.bundles[case_id].stage2_result,
                )
                contextual = build_contextual_stage3_bundle(
                    project_root=ROOT,
                    evidence_pack=pack,
                    base_stage3_bundle=base.model_dump(mode="json"),
                    base_stage3_input_fingerprint="offline-evidence-boundary-test",
                )
                evidence_by_arm[arm] = {
                    obj.scenario_id: sha256_json(obj.evaluation_evidence_records)
                    for obj in contextual.contextual_decision_objects
                }
            self.assertEqual(evidence_by_arm["original"], evidence_by_arm["reversed_pmt"])

    def test_one_arm_cannot_receive_extra_instruction_or_evidence(self):
        case_id = "main__malware__anomaly_detection"
        a = self.datasets[case_id]["original"]
        b = self.datasets[case_id]["reversed_pmt"]
        altered = deepcopy(b)
        prompt = json.loads(altered["forecast_data"])
        prompt["study_grounding_contract"] = "Must choose stance=+1."
        altered["forecast_data"] = json.dumps(prompt)
        with self.assertRaisesRegex(ValueError, "prompt fields differ"):
            _paired_inputs(a, altered)
        altered = deepcopy(b)
        altered["evidence_pack"]["evidence"].pop()
        with self.assertRaisesRegex(ValueError, "outside forecast"):
            _paired_inputs(a, altered)

        altered = deepcopy(b)
        altered["evidence_pack"]["forecast_summary"]["forecast_field_ids"]["threat"] = ["tampered"]
        with self.assertRaisesRegex(ValueError, "provenance change"):
            _paired_inputs(a, altered)

        altered = deepcopy(b)
        prompt = json.loads(altered["forecast_data"])
        prompt["forecast_summary"]["pmt_state"]["state_modality"] = "tampered"
        altered["forecast_data"] = json.dumps(prompt)
        with self.assertRaisesRegex(ValueError, "prompt PMT difference"):
            _paired_inputs(a, altered)

    def test_isolated_stage_paths_and_graph_contract(self):
        payload = self.datasets["main__malware__anomaly_detection"]["original"]
        graph = _graph(payload)
        self.assertIn("s4_complete", graph.nodes)
        self.assertNotIn("stage6_synthesis", graph.nodes)
        self.assertNotIn("s4_complete", _graph(payload, phase="upstream").nodes)

        import Stage1.nodes as s1
        import Stage2.nodes as s2
        import Stage3.nodes as s3
        import Stage3.context_nodes as s3c
        import Stage4.context_nodes as s4c
        with patch.object(s3c, "build_contextual_stage3_bundle", side_effect=lambda **kwargs: kwargs["project_root"]) as fake:
            with ExitStack() as stack:
                _isolated_nodes(stack)
                for mod in (s1, s2, s3, s3c, s4c):
                    self.assertEqual(mod._project_root(), SANDBOX)
                prepared = s1.prepare_stage1_node(payload)
                self.assertTrue(
                    Path(prepared["stage1_artifact_path"]).resolve().is_relative_to(SANDBOX.resolve())
                )
                self.assertEqual(s3c.build_contextual_stage3_bundle(project_root=SANDBOX), ROOT)
                self.assertEqual(fake.call_args.kwargs["project_root"], ROOT)
        self.assertEqual(s1._project_root(), ROOT)

    def test_pipeline_does_not_modify_stage2_mediator_or_any_prompt(self):
        import Stage2.nodes as s2
        import Stage1.prompts as s1_prompts
        baseline_prompt = s2.STAGE2_MEDIATOR_SYSTEM_PROMPT
        baseline_mediator = s2._load_or_generate_mediator
        baseline_visibility = s2._forecast_data_visible_to_stage2_step
        baseline_attack = s1_prompts.ATTACK_FEASIBILITY_SYSTEM_PROMPT
        with ExitStack() as stack:
            _isolated_nodes(stack)
            self.assertIs(s2._load_or_generate_mediator, baseline_mediator)
            self.assertIs(s2._forecast_data_visible_to_stage2_step, baseline_visibility)
            self.assertEqual(s2.STAGE2_MEDIATOR_SYSTEM_PROMPT, baseline_prompt)
            self.assertEqual(s1_prompts.ATTACK_FEASIBILITY_SYSTEM_PROMPT, baseline_attack)
        self.assertEqual(s2.STAGE2_MEDIATOR_SYSTEM_PROMPT, baseline_prompt)
        for pair in self.datasets.values():
            for data in pair.values():
                self.assertNotIn("study_grounding_contract", json.loads(data["forecast_data"]))
                self.assertNotIn("study_round2_mediator_protocol", json.loads(data["forecast_data"]))

    def test_seven_main_archives_reused_byte_exact_without_llm(self):
        import Stage7.forecast_link_counterfactual as study
        from Stage7.forecast_link_counterfactual import _main_reference_paths

        with TemporaryDirectory() as folder:
            sandbox = Path(folder) / "sandbox"
            with patch.object(study, "SANDBOX", sandbox):
                for case_id in self.datasets:
                    _import_original(case_id, 42)
                    case = next(c for c in MAIN_STAGE1_CASES if c.case_id == case_id)
                    for source in _main_reference_paths(case_id):
                        relative = source.relative_to(ROOT / "Multi-Agent" / "Results")
                        target = sandbox / "Multi-Agent" / "Results" / relative
                        self.assertEqual(target.read_bytes(), source.read_bytes())

                    with ExitStack() as stack:
                        _isolated_nodes(stack)
                        result = _artifacts(
                            self.datasets[case_id]["original"],
                            _case_config(case), contextual=True,
                        )
                        self.assertIsNotNone(result)
                        self.assertTrue(_stage4_completed(result))
                        self.assertEqual(
                            result["stage2"]["execution_order_version"],
                            "attack-defense-mediator-post-attack-defense-v1",
                        )

    def test_historical_stage2_calls_existing_steps_in_exact_original_order(self):
        import Stage2.nodes as s2

        case_id = "main__malware__anomaly_detection"
        bundle = self.bundles[case_id]
        initial = {
            **self.datasets[case_id]["original"],
            "attack_assessment": bundle.stage1["assessments"]["attack_feasibility"],
            "defense_assessment": bundle.stage1["assessments"]["defense_robustness"],
        }
        round1 = bundle.stage2_result["rounds"][0]
        calls = []

        async def fake_turn(**kwargs):
            role = kwargs["responder"]
            calls.append("attack" if role == CriticType.ATTACK_FEASIBILITY else "defense")
            key = "attack_turn" if role == CriticType.ATTACK_FEASIBILITY else "defense_turn"
            return CriticDebateTurn.model_validate(round1[key])

        async def fake_mediator(**kwargs):
            calls.append("mediator")
            return MediatorSummary.model_validate(round1["mediator_summary"])

        async def fake_post(**kwargs):
            role = kwargs["critic_type"]
            key = "attack" if role == CriticType.ATTACK_FEASIBILITY else "defense"
            calls.append("post_" + key)
            return CriticAssessment.model_validate(
                bundle.stage2_result[f"{key}_post_assessment"]
            )

        with (
            patch.object(s2, "_load_exact_stage1_artifact", return_value=bundle.stage1),
            patch.object(s2, "_load_or_generate_turn", side_effect=fake_turn),
            patch.object(s2, "_load_or_generate_mediator", side_effect=fake_mediator),
            patch.object(s2, "_load_or_generate_post_assessment", side_effect=fake_post),
            patch.object(s2, "get_llm", side_effect=AssertionError("No LLM expected")),
        ):
            result = asyncio.run(_historical_stage2_debate_node(initial))

        self.assertEqual(
            calls, ["attack", "defense", "mediator", "post_attack", "post_defense"]
        )
        self.assertEqual(result["stage2_debate_result"], bundle.stage2_result)

    def test_noi_ransomware_regulation_cannot_be_decisive_malware_trend(self):
        case = "main__malware__anomaly_detection"
        pack = self.datasets[case]["reversed_pmt"]["evidence_pack"]
        original = deepcopy(self.bundles[case].stage1["assessments"]["attack_feasibility"])
        self.assertEqual(original["stance"], 0)
        regulation = next(record["evidence_id"] for record in pack["evidence"] if (
            "eu_legal_requirement" in record["allowed_claim_types"]
        ))
        original["claims"][0]["evidence_ids"] = [regulation]
        original["claims"][0]["forecast_component"] = "THREAT_TRAJECTORY"
        original["claims"][0]["forecast_relation"] = "SUPPORTS_FORECAST"
        original["stance"] = 1
        original["stance_basis"] = {
            "supporting_claim_ids": [original["claims"][0]["claim_id"]],
            "challenging_claim_ids": [],
            "decision": "SUPPORT_DOMINATES",
            "rationale": "Regression test: regulatory text is not a Malware NoI trajectory.",
        }
        with self.assertRaises(Stage1ValidationError):
            validate_critic_assessment(
                original,
                evidence_pack=pack,
                expected_critic_type=CriticType.ATTACK_FEASIBILITY,
            )

    def test_frozen_checkpoints_detect_tampering_and_allow_exact_reuse(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "sample.json"
            item = {"version": "unit-test", "seed": 42}
            _write_frozen(path, item)
            _write_frozen(path, item)
            self.assertEqual(_load_frozen(path)["seed"], 42)
            with self.assertRaisesRegex(ValueError, "conflict"):
                _write_frozen(path, {"version": "unit-test", "seed": 43})
            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["seed"] = 43
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                _load_frozen(path)

    def test_completed_result_must_bind_exact_seed_fingerprint(self):
        from Stage7.forecast_link_counterfactual import _record_path

        case = "main__malware__anomaly_detection"
        payload = self.datasets[case]["original"]
        record = _load_frozen(_record_path(case, "original", 42))
        _validate_record(record, payload, case, "original", 42)

        altered = deepcopy(record)
        altered["seed"] = 43
        with self.assertRaisesRegex(ValueError, "input_fingerprint"):
            _validate_record(altered, payload, case, "original", 43)

        altered = deepcopy(record)
        altered["evidence_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "evidence_sha256"):
            _validate_record(altered, payload, case, "original", 42)

    def test_first_pass_mediator_failures_are_exactly_identified(self):
        import ast
        import Stage7.forecast_link_counterfactual as study

        ledger = json.loads(
            (study.OUTPUT / "first_pass_failed_upstream.json").read_text(encoding="utf-8")
        )
        errors = ledger["failures"]
        self.assertEqual(len(errors), 9)
        with TemporaryDirectory() as folder:
            (Path(folder) / "failed_upstream.json").write_text(
                json.dumps(ledger), encoding="utf-8"
            )
            with patch.object(study, "OUTPUT", Path(folder)):
                for failure in errors:
                    raw = ast.literal_eval(
                        failure["message"].split(" semantic repair attempt(s): ", 1)[1]
                    )
                    failed_round = json.loads(raw)["round_index"]
                    self.assertTrue(_previous_mediator_failure(
                        failure["case_id"], failure["arm"],
                        failure["seed"], failed_round,
                    ))
                    self.assertFalse(_previous_mediator_failure(
                        failure["case_id"], failure["arm"],
                        failure["seed"], 3,
                    ))

    def test_recover_only_failed_mediator_with_fixed_alternate_sampling_seed(self):
        import Stage7.forecast_link_counterfactual as study

        base_config = study.stage1_runtime.get_runtime_config()
        seen_seeds = []

        async def native_step(**kwargs):
            client = kwargs["llm_provider"]()
            seen_seeds.append(client.config.seed)
            if len(seen_seeds) == 1:
                raise RuntimeError("does not match MediatorSummary")
            return "validated"

        with TemporaryDirectory() as folder:
            with (
                patch.object(study, "OUTPUT", Path(folder)),
                patch.object(study, "_previous_mediator_failure", return_value=True),
                patch.object(
                    study.stage2_nodes, "_load_or_generate_mediator",
                    side_effect=native_step,
                ),
            ):
                result = asyncio.run(_mediator_with_recovery(
                    arm="reversed_pmt",
                    seed=43,
                    state={"evidence_pack": {"case_id": "case_a"}},
                    llm_provider=lambda: study.stage1_runtime.LlamaCppChatClient(
                        config=base_config, request_concurrency=1
                    ),
                    identity={}, round_index=2,
                    attack_turn=None, defense_turn=None, previous_rounds=[],
                ))
                audit = json.loads(
                    (Path(folder) / "mediator_recovery" / "case_a"
                     / "reversed_pmt__seed-43__round-2.json").read_text("utf-8")
                )

        self.assertEqual(result, "validated")
        self.assertEqual(seen_seeds, [10043, 20043])
        self.assertEqual([x["status"] for x in audit["attempts"]], [
            "started", "invalid", "started", "validated",
        ])

    def test_unrelated_mediator_failure_cannot_trigger_resampling(self):
        import Stage7.forecast_link_counterfactual as study

        with (
            patch.object(study, "_previous_mediator_failure", return_value=False),
            patch.object(
                study.stage2_nodes, "_load_or_generate_mediator",
                new_callable=AsyncMock, side_effect=RuntimeError("Server crashed"),
            ) as step,
            patch.object(study, "_log_mediator_recovery") as log,
        ):
            with self.assertRaisesRegex(RuntimeError, "Server crashed"):
                asyncio.run(_mediator_with_recovery(
                    arm="original", seed=43,
                    state={"evidence_pack": {"case_id": "case_a"}},
                    llm_provider=lambda: None,
                    identity={}, round_index=1,
                    attack_turn=None, defense_turn=None, previous_rounds=[],
                ))

        self.assertEqual(step.await_count, 1)
        log.assert_not_called()

    def test_frozen_main_lens_output_matches_stage6_policy(self):
        case = "main__malware__anomaly_detection"
        bundle = self.bundles[case]
        payload = self.datasets[case]["original"]
        state = {
            "stage4_context_complete": True,
            "contextual_decision_objects": bundle.contextual_objects,
            "stage4_context_evaluation_bundle": bundle.stage4_context["evaluation_bundle"],
            "stage1_input_fingerprint": "mock-identity",
            "attack_assessment": bundle.stage1["assessments"]["attack_feasibility"],
            "defense_assessment": bundle.stage1["assessments"]["defense_robustness"],
            "stage2_debate_result": bundle.stage2_result,
            **{
                f"{stage}_artifact_path": str(SANDBOX / "Multi-Agent" / "Results" / stage / "artifact.json")
                for stage in ("stage1", "stage2", "stage3", "stage3_context", "stage4_context")
            },
        }
        record = _record(case, "original", 42, payload, state)
        expected = {x["scenario_id"]: x["decision_recommendation"] for x in bundle.syntheses}
        actual = {scenario: d["recommendation"] for scenario, d in record["decisions"].items()}
        self.assertEqual(actual, expected)
        self.assertEqual(record["evidence_sha256"], sha256_json(payload["evidence_pack"]["evidence"]))
        modified = deepcopy(record)
        modified["decisions"]["EU__CIS_IG2"]["recommendation"] = "HOLD"
        records = {}
        for seed in SEEDS:
            records["original", seed] = record
            records["reversed_pmt", seed] = modified
        summary = _compare(case, records)
        self.assertEqual(len(summary), 3)
        self.assertIsInstance(summary["42"]["jurisdictions"]["EU__CIS_IG2"]["recommendation_changed"], bool)
        stability = _within_arm_agreement(records)
        self.assertEqual(stability["original"]["recommendation_agreement"], 1.0)
        self.assertEqual(stability["reversed_pmt"]["stance_tuple_agreement"], 1.0)

    def test_original_seed42_finishes_without_model_even_with_no_server(self):
        import Stage7.forecast_link_counterfactual as study

        case = "main__malware__anomaly_detection"
        bundle = self.bundles[case]
        inputs = {case: self.datasets[case]}

        with TemporaryDirectory() as directory:
            output = Path(directory) / "experiment"
            sandbox = output / "sandbox"
            self.assertNotEqual(sandbox, ROOT)

            with (
                patch.object(study, "OUTPUT", output),
                patch.object(study, "SANDBOX", sandbox),
                patch("Stage1.nodes.get_llm", side_effect=AssertionError("Unexpected Stage1 LLM")),
                patch("Stage2.nodes.get_llm", side_effect=AssertionError("Unexpected Stage2 LLM")),
                patch("Stage4.context_nodes.get_context_llm", side_effect=AssertionError("Unexpected Stage4 LLM")),
                patch.object(study.stage1_runtime, "assert_llama_server_ready", side_effect=AssertionError("Unexpected server check")),
            ):
                manifest = _manifest(inputs, {case: self.provenance[case]})
                _write_frozen(output / f"experiment_manifest__{case}.json", manifest)

                with self.assertRaisesRegex(ValueError, "Run --phase upstream"):
                    asyncio.run(_run([case], inputs, phase="downstream", seeds=(42,)))

                path = output / "records" / case / "original__seed-42.json"
                record = _load_frozen(path)
                self.assertEqual(record["input_fingerprint"], bundle.stage1["input_fingerprint"])
                expected = {
                    x["scenario_id"]: x["decision_recommendation"]
                    for x in bundle.syntheses
                }
                self.assertEqual(
                    {k: v["recommendation"] for k, v in record["decisions"].items()},
                    expected,
                )

    def test_wrong_server_rejected_before_counterfactual_llm(self):
        import Stage7.forecast_link_counterfactual as study

        case_id = "main__malware__anomaly_detection"
        inputs = {case_id: self.datasets[case_id]}

        with TemporaryDirectory() as folder:
            destination = Path(folder) / "experiment"
            with (
                patch.object(study, "SANDBOX", destination / "sandbox"),
                patch.object(study, "OUTPUT", destination),
                patch.object(
                    study.stage1_runtime,
                    "assert_llama_server_ready",
                    side_effect=RuntimeError("Expected exactly one 131072-token slot"),
                ) as runtime_check,
                patch.object(study, "_one_slot_llm", side_effect=AssertionError("Must not call LLM")),
            ):
                with self.assertRaisesRegex(RuntimeError, "Expected exactly one"):
                    asyncio.run(_run([case_id], inputs, phase="upstream", seeds=(42,)))

                self.assertEqual(runtime_check.call_count, 1)
                self.assertEqual(
                    runtime_check.call_args.kwargs["expected_parallel_slots"], 1
                )
                self.assertTrue(
                    (
                        destination / "sandbox" / "Multi-Agent" / "Results"
                        / "Stage1" / "artifacts" / case_id
                        / self.bundles[case_id].stage1_path.name
                    ).exists()
                )
                self.assertFalse(
                    (
                        destination / "sandbox" / "Multi-Agent" / "Results"
                        / "Stage1" / "checkpoints" / case_id
                    ).exists()
                )

    def test_same_seed_resume_uses_saved_upstream(self):
        import Stage7.forecast_link_counterfactual as study

        case_id = "main__malware__anomaly_detection"
        inputs = {case_id: self.datasets[case_id]}
        with TemporaryDirectory() as folder:
            output = Path(folder) / "experiment"
            with (
                patch.object(study, "SANDBOX", output / "sandbox"),
                patch.object(study, "OUTPUT", output),
                patch.object(study.stage1_runtime, "assert_llama_server_ready") as server,
            ):
                # The main seed-42 baseline completes the upstream phase
                # without any model request; CF is intentionally blocked.
                server.side_effect = RuntimeError("one-slot server required")
                with self.assertRaisesRegex(RuntimeError, "one-slot server"):
                    asyncio.run(_run([case_id], inputs, phase="upstream", seeds=(42,)))

                # Resume with exactly the same identities. An archived Stage 1
                # artifact cannot be overwritten or re-generated.
                original = (
                    output / "sandbox" / "Multi-Agent" / "Results"
                    / "Stage1" / "artifacts" / case_id
                    / self.bundles[case_id].stage1_path.name
                )
                before = original.read_bytes()
                with self.assertRaisesRegex(RuntimeError, "one-slot server"):
                    asyncio.run(_run([case_id], inputs, phase="upstream", seeds=(42,)))
                self.assertEqual(original.read_bytes(), before)
                self.assertEqual(server.call_count, 2)

    def test_continue_on_error_attempts_other_arm_without_faking_success(self):
        import Stage7.forecast_link_counterfactual as study

        case_id = "main__malware__anomaly_detection"
        inputs = {case_id: self.datasets[case_id]}
        invocations = []

        class FakeGraph:
            async def ainvoke(self, initial, config):
                invocations.append(initial["stage0_config"]["case_id"])
                if len(invocations) == 1:
                    raise RuntimeError("Model rejected by semantic validator")
                return {"stage4_context_complete": True}

        with TemporaryDirectory() as folder:
            output = Path(folder) / "experiment"
            with (
                patch.object(study, "OUTPUT", output),
                patch.object(study, "SANDBOX", output / "sandbox"),
                patch.object(study, "_import_original"),
                patch.object(study, "_isolated_nodes"),
                patch.object(study, "_artifacts", return_value={"stage3_context": {}}),
                patch.object(study, "_stage4_completed", return_value=True),
                patch.object(study, "_graph", return_value=FakeGraph()),
                patch.object(study, "_record", return_value={"complete": True}),
            ):
                failed = asyncio.run(
                    _run(
                        [case_id], inputs, phase="downstream",
                        seeds=(42,), continue_on_error=True,
                    )
                )
                self.assertEqual(len(invocations), 2)
                self.assertEqual(len(failed), 1)
                self.assertEqual(failed[0]["arm"], "original")
                self.assertIsNone(
                    _load_frozen(output / "records" / case_id / "original__seed-42.json")
                )
                self.assertEqual(
                    _load_frozen(output / "records" / case_id / "reversed_pmt__seed-42.json")["complete"],
                    True,
                )


SERVER_COMMAND = [
    r"C:\LLMs\llama-server.exe",
    "--model", r"C:\LLMs\Qwen3.8-27B-Q6_K_L.gguf",
    "--host", "127.0.0.1",
    "--port", "8080",
    "--ctx-size", "131072",
    "--parallel", "2",
]


class SingleFileFullRunnerTests(unittest.TestCase):
    def test_switch_changes_only_slot_count(self):
        import Stage7.forecast_link_counterfactual as study

        process = Mock()
        with (
            patch.object(study, "_server", return_value=(process, SERVER_COMMAND)),
            patch.object(study, "_ready") as ready,
            patch.object(study, "_ensure_idle") as idle,
            patch.object(study, "_start_server") as start,
        ):
            study._switch_server(SERVER_COMMAND, 1)

        ready.assert_called_once_with(2)
        idle.assert_called_once_with()
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=30)

        expected = list(SERVER_COMMAND)
        expected[expected.index("--parallel") + 1] = "1"
        start.assert_called_once_with(expected, 1)

    def test_busy_server_cannot_be_stopped(self):
        import Stage7.forecast_link_counterfactual as study

        process = Mock()
        with (
            patch.object(study, "_server", return_value=(process, SERVER_COMMAND)),
            patch.object(study, "_ready"),
            patch.object(study, "_ensure_idle", side_effect=RuntimeError("busy")),
            patch.object(study, "_start_server") as start,
        ):
            with self.assertRaisesRegex(RuntimeError, "busy"):
                study._switch_server(SERVER_COMMAND, 1)

        process.terminate.assert_not_called()
        start.assert_not_called()

    def test_slots_must_all_be_idle(self):
        import Stage7.forecast_link_counterfactual as study

        response = Mock()
        response.json.return_value = [
            {"state": "idle", "is_processing": False},
            {"state": "processing", "is_processing": True},
        ]
        with patch.object(study.requests, "get", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "processing another request"):
                study._ensure_idle()

        response.json.return_value[1] = {"state": "idle", "is_processing": False}
        with patch.object(study.requests, "get", return_value=response):
            study._ensure_idle()

    def test_all_cases_run_both_phases_and_restore_original_server(self):
        import Stage7.forecast_link_counterfactual as study

        run = AsyncMock(return_value=[])
        with (
            patch.object(study, "_server", return_value=(Mock(), SERVER_COMMAND)),
            patch.object(study, "_ready"),
            patch.object(study, "_switch_server") as switch,
            patch.object(study, "_run", run),
            patch.object(study, "_save_failures") as save,
            patch.object(study, "_report", return_value="summary.json") as report,
        ):
            study._run_all(["case_a", "case_b"], {"case_a": {}, "case_b": {}}, study.SEEDS)

        self.assertEqual(run.await_count, 2)
        self.assertEqual([kwargs["phase"] for _, kwargs in run.call_args_list], [
            "upstream", "downstream",
        ])
        self.assertTrue(all(kwargs["continue_on_error"] for _, kwargs in run.call_args_list))
        self.assertEqual(save.call_args_list, [
            call("upstream", []), call("downstream", []),
        ])
        report.assert_called_once_with(["case_a", "case_b"])
        self.assertEqual(switch.call_args_list, [
            call(SERVER_COMMAND, 1),
            call(SERVER_COMMAND, 2),
            call(SERVER_COMMAND, 2),
        ])

    def test_failures_still_run_downstream_and_restore_server(self):
        import Stage7.forecast_link_counterfactual as study

        run = AsyncMock(side_effect=[
            [{"case_id": "case_a", "arm": "original"}],
            [{"case_id": "case_a", "arm": "reversed_pmt"}],
        ])
        with (
            patch.object(study, "_server", return_value=(Mock(), SERVER_COMMAND)),
            patch.object(study, "_ready"),
            patch.object(study, "_switch_server") as switch,
            patch.object(study, "_run", run),
            patch.object(study, "_save_failures") as save,
            patch.object(study, "_report") as report,
        ):
            with self.assertRaisesRegex(RuntimeError, "1 upstream and 1 downstream"):
                study._run_all(["case_a"], {"case_a": {}}, study.SEEDS)

        self.assertEqual(run.await_count, 2)
        self.assertEqual(save.call_count, 2)
        report.assert_not_called()
        self.assertEqual(switch.call_args_list[-1], call(SERVER_COMMAND, 2))

    def test_downstream_exception_restores_server(self):
        import Stage7.forecast_link_counterfactual as study

        run = AsyncMock(side_effect=[[], RuntimeError("pipeline crash")])
        with (
            patch.object(study, "_server", return_value=(Mock(), SERVER_COMMAND)),
            patch.object(study, "_ready"),
            patch.object(study, "_switch_server") as switch,
            patch.object(study, "_run", run),
            patch.object(study, "_save_failures"),
        ):
            with self.assertRaisesRegex(RuntimeError, "pipeline crash"):
                study._run_all(["case_a"], {"case_a": {}}, study.SEEDS)

        self.assertEqual(switch.call_args_list[-1], call(SERVER_COMMAND, 2))

    def test_one_command_cli_dispatches_full_runner(self):
        import Stage7.forecast_link_counterfactual as study

        with (
            patch("sys.argv", ["forecast_link_counterfactual", "--run"]),
            patch.object(study, "_inputs", return_value=({}, {})),
            patch.object(study, "_main_reference_paths"),
            patch.object(study, "_manifest", return_value={}),
            patch.object(study, "_write_frozen"),
            patch.object(study, "_run_all") as run_all,
        ):
            study.main()

        self.assertEqual(run_all.call_count, 1)
        self.assertEqual(run_all.call_args.args[0], [
            case.case_id for case in MAIN_STAGE1_CASES
        ])
        self.assertEqual(run_all.call_args.args[2], SEEDS)


if __name__ == "__main__":
    unittest.main()
