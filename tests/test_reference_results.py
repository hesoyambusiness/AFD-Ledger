from __future__ import annotations

import unittest

from sim.experiment_config import load_experiment_config, load_experiment_models
from sim.longcat_validation import validation_rows


class ReferenceResultTest(unittest.TestCase):
    def test_experiment_contract_contains_only_current_studies(self) -> None:
        config = load_experiment_config()
        self.assertEqual(set(config["models"]), {"DeepSeek-V3.2", "Qwen3"})
        self.assertEqual(len(config["reference_cases"]), 5)
        self.assertEqual(config["bounded_search"]["evaluation_budget"], 20)
        self.assertEqual(
            len(config["existing_catalog"]["constructed_availability_subset"]),
            8,
        )
        self.assertEqual(len(config["simulated_catalog"]["devices"]), 2)
        self.assertEqual(set(load_experiment_models(config)), set(config["models"]))

    def test_longcat_model_preserves_all_160_card_decisions(self) -> None:
        rows = validation_rows()
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(int(row["card_budget"]), 160)
            modeled = float(row["theta_model_ratio"])
            measured = float(row["theta_measured_ratio"])
            self.assertEqual(modeled > 1.0, measured > 1.0)
            self.assertGreater(float(row["model_over_measured_ratio"]), 1.06)
            self.assertLess(float(row["model_over_measured_ratio"]), 1.10)


if __name__ == "__main__":
    unittest.main()
