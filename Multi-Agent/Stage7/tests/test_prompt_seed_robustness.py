from __future__ import annotations

from pathlib import Path
import unittest

from Stage7.axes import _record_set_comparison
from Stage7.config import SENSITIVITY_CASE_IDS
from Stage7.loaders import load_main_case_bundles
from Stage7.prompt_seed_robustness import _records, _result_path, _results_dir
from Stage7.variant_generation import _checkpoint_path, _variant_output_path
from Stage7.variants import baseline_record_map


ROOT = Path(__file__).resolve().parents[3]


class PromptSeedRobustnessTests(unittest.TestCase):
    def test_output_paths_do_not_overlap_existing_results(self) -> None:
        output = _results_dir(ROOT)

        self.assertEqual(
            _checkpoint_path(ROOT, axis="A", variant_id="P2", fingerprint="x"),
            ROOT / "Multi-Agent" / "Results" / "Stage7" / "variant_checkpoints" / "A" / "P2" / "x.json",
        )
        self.assertEqual(
            _checkpoint_path(ROOT, axis="A", variant_id="P2", fingerprint="x", results_dir=output),
            output / "variant_checkpoints" / "A" / "P2" / "x.json",
        )
        self.assertEqual(
            _variant_output_path(ROOT, axis="A", variant_id="P2", repeat_index=43, results_dir=output),
            output / "variants" / "A" / "P2__repeat-43.json",
        )

        self.assertNotEqual(
            _result_path(ROOT, "P2_EVIDENCE_FIRST", 42),
            _result_path(ROOT, "P2_EVIDENCE_FIRST", 43),
        )

    def test_existing_original_inputs_support_paired_comparison(self) -> None:
        bundles = load_main_case_bundles(ROOT)
        baseline = {
            key: value for key, value in baseline_record_map(bundles).items()
            if key[0] in SENSITIVITY_CASE_IDS
        }

        self.assertEqual(len(baseline), 9)

        for prompt in ("P2_EVIDENCE_FIRST", "P3_DECISION_FIRST"):
            records = _records(_result_path(ROOT, prompt, 42))
            self.assertEqual(_record_set_comparison(records, baseline)["n"], 9)

        for seed in (43, 44):
            records = _records(_result_path(ROOT, "P1_CANONICAL", seed))
            self.assertEqual(_record_set_comparison(records, baseline)["n"], 9)


if __name__ == "__main__":
    unittest.main()
