from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from Stage7.context_path_seed_robustness import (
    SEEDS,
    _check_source_controls,
    _context_free_path,
    _contextualized_records,
    _records,
    _results_dir,
    _summary,
)
from Stage7.loaders import load_main_case_bundles


ROOT = Path(__file__).resolve().parents[3]


class ContextPathSeedRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundles = load_main_case_bundles(ROOT)

    def test_frozen_seed_42_input_matches_all_checkpoint_prompts(self) -> None:
        _check_source_controls(ROOT, self.bundles)

    def test_new_results_are_separate_from_frozen_outputs(self) -> None:
        original = _context_free_path(ROOT, 42)
        self.assertTrue(original.is_file())

        for seed in SEEDS[1:]:
            path = _context_free_path(ROOT, seed)
            self.assertNotEqual(path, original)
            self.assertTrue(path.is_relative_to(_results_dir(ROOT)))
            self.assertEqual(path.name, f"CONTEXT_FREE_STAGE4_V1__repeat-{seed}.json")

    def test_existing_contextualized_results_are_complete(self) -> None:
        source_keys = set(_records(_context_free_path(ROOT, 42), seed=42, context_free=True))
        self.assertEqual(len(source_keys), 21)

        for seed in SEEDS:
            comparison = _contextualized_records(ROOT, seed, self.bundles)
            self.assertEqual(set(comparison), source_keys)

    def test_modified_control_fails_before_inference(self) -> None:
        original = json.loads(_context_free_path(ROOT, 42).read_text(encoding="utf-8"))
        original["records"][0]["recommendation"] = "HOLD" if (
            original["records"][0]["recommendation"] != "HOLD"
        ) else "RECOMMEND"

        with TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.json"
            path.write_text(json.dumps(original), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Result hash mismatch"):
                _records(path, seed=42, context_free=True)

    def test_summary_aggregates_all_seeds_and_paths(self) -> None:
        context_free = _records(_context_free_path(ROOT, 42), seed=42, context_free=True)
        contextualized = _contextualized_records(ROOT, 42, self.bundles)

        with patch(
            "Stage7.context_path_seed_robustness._records", return_value=context_free
        ):
            with patch(
                "Stage7.context_path_seed_robustness._contextualized_records", return_value=contextualized
            ):
                result = _summary(ROOT, self.bundles)

        self.assertEqual(set(result["same_seed_context_free_vs_contextualized"]), {"42", "43", "44"})
        self.assertEqual(set(result["same_path_between_seeds"]), {"context_free", "contextualized"})

        for path in ("context_free", "contextualized"):
            for pair in ("42-43", "42-44", "43-44"):
                metrics = result["same_path_between_seeds"][path][pair]
                self.assertEqual(metrics["n"], 21)
                self.assertEqual(metrics["recommendation_consistency_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
