from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from sim.catalog_provisioning import ordered_assignments, provision_catalog
from sim.config import load_hardware, load_model, resolve_hardware_config_path
from sim.experiment_config import load_experiment_config
from sim.guided_catalog_search import (
    _role_diverse_beam,
    guided_provision_catalog,
    provision_round,
)
from sim.provisioning import (
    DeploymentSpecification,
    HardwareOption,
    evaluate_afd_shape,
    enumerate_collocated,
    pipeline_depths,
)
from sim.simulated_catalog import fixed_simulated_catalog
from sim.throughput_catalog import load_priced_existing_catalog


def option(name: str, price: float) -> HardwareOption:
    hardware = load_hardware(resolve_hardware_config_path(Path(name)))
    return HardwareOption(name, hardware, price)


class ProvisioningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_model(
            Path("deepseek-ai/DeepSeek-V3.2"), None, "bf16", "bf16", "bf16"
        )
        cls.menu = {
            "h200_sxm": option("h200_sxm", 1.0),
            "a100_pcie": option("a100_pcie", 1.19 / 3.59),
        }
        cls.specification = DeploymentSpecification(
            cls.model, 4096, 12.0, 48.0, cls.menu
        )

    def test_a100_uses_int8_without_advertising_fp8(self) -> None:
        peaks = self.menu["a100_pcie"].hardware.peak_flops
        self.assertNotIn("fp8", peaks)
        self.assertEqual(peaks["int8"], 624e12)

    def test_guarded_specification_freezes_the_catalog(self) -> None:
        with self.assertRaises(TypeError):
            self.specification.hardware_menu["new"] = self.menu["h200_sxm"]

    def test_catalog_assignments_include_homogeneous_afd(self) -> None:
        assignments = ordered_assignments(("h200_sxm", "a100_pcie"))
        self.assertEqual(len(assignments), 4)
        self.assertIn(("h200_sxm", "h200_sxm"), assignments)
        self.assertIn(("a100_pcie", "a100_pcie"), assignments)

    def test_fixed_simulated_catalog_has_121_role_compatible_pairs(self) -> None:
        config = load_experiment_config()
        commercial = load_priced_existing_catalog(config)
        catalog = fixed_simulated_catalog(
            commercial,
            config["simulated_catalog"],
        )
        attention = sum(option.supports_attention for option in catalog.values())
        expert = sum(option.supports_expert for option in catalog.values())
        self.assertEqual(len(catalog), 12)
        self.assertEqual(attention * expert, 121)
        self.assertEqual(sum(option.supports_collocated for option in catalog.values()), 10)

        specification = DeploymentSpecification(
            self.model, 4096, 20.0, 48.0, catalog
        )
        collocated_hardware = {
            plan.attention_hardware for plan in enumerate_collocated(specification)
        }
        self.assertFalse(
            collocated_hardware
            & {
                name
                for name, option in catalog.items()
                if not option.supports_collocated
            }
        )

    def test_provision_round_charges_an_infeasible_assignment(self) -> None:
        catalog = {
            "h200_sxm": option("h200_sxm", 1.0),
            "a100_pcie": option("a100_pcie", 1.19 / 3.59),
            "l4": option("l4", 0.44 / 3.59),
        }
        specification = DeploymentSpecification(
            self.model, 4096, 12.0, 48.0, catalog
        )
        result = provision_round(
            specification,
            (("a100_pcie", "l4"), ("h200_sxm", "a100_pcie")),
            remaining_budget=1,
        )
        self.assertEqual(result.attempted_assignments, (("a100_pcie", "l4"),))
        self.assertEqual(result.feasible_plans, {})

    def test_role_diverse_beam_preserves_distinct_hardware_basins(self) -> None:
        evaluated = {
            ("attention_a", "expert_x"): SimpleNamespace(throughput_tps=100.0),
            ("attention_a", "expert_y"): SimpleNamespace(throughput_tps=90.0),
            ("attention_b", "expert_z"): SimpleNamespace(throughput_tps=80.0),
        }
        self.assertEqual(
            _role_diverse_beam(evaluated, 2),
            (("attention_a", "expert_x"), ("attention_b", "expert_z")),
        )

    def test_afd_replica_accounting(self) -> None:
        homogeneous = DeploymentSpecification(
            self.model,
            4096,
            100.0,
            64.0,
            {"h200_sxm": self.menu["h200_sxm"]},
        )
        plan = evaluate_afd_shape(
            homogeneous, "h200_sxm", "h200_sxm", 5, 11, 16, 4
        )
        self.assertEqual(plan.replicas, 4)
        self.assertEqual(plan.concurrency, 320)
        self.assertEqual(plan.budget_used, 64.0)

    def test_depth_optimizer_matches_exhaustive_oracle(self) -> None:
        qwen = load_model(
            Path("Qwen/Qwen3-235B-A22B-FP8"), None, "bf16", "bf16", "bf16"
        )
        menu = {
            "h200_sxm": option("h200_sxm", 1.0),
            "a100_pcie": option("a100_pcie", 1.19 / 3.59),
        }
        specification = DeploymentSpecification(qwen, 1024, 100.0, 64.0, menu)
        probes = pipeline_depths(
            specification,
            menu["h200_sxm"],
            menu["a100_pcie"],
            27,
            64,
            128,
        )
        probed = [
            evaluate_afd_shape(
                specification, "h200_sxm", "a100_pcie", 27, 64, 128, depth
            )
            for depth in probes
        ]
        exhaustive = [
            evaluate_afd_shape(
                specification, "h200_sxm", "a100_pcie", 27, 64, 128, depth
            )
            for depth in range(1, 129)
        ]
        self.assertAlmostEqual(
            max(plan.throughput_tps for plan in probed),
            max(plan.throughput_tps for plan in exhaustive),
            places=9,
        )

    def test_exact_catalog_provisioning_selects_global_throughput(self) -> None:
        result = provision_catalog(self.specification)
        self.assertEqual(result.best_afd_assignment, ("h200_sxm", "a100_pcie"))
        self.assertEqual(result.best_collocated.attention_hardware, "h200_sxm")
        self.assertGreater(result.comparison_gain, 1.0)
        self.assertEqual(result.selected_architecture, "afd")
        self.assertEqual(len(result.afd_plans), 4)

    def test_global_collocated_search_survives_infeasible_afd_subset(self) -> None:
        catalog = {
            "h200_sxm": option("h200_sxm", 1.0),
            "a100_pcie": option("a100_pcie", 1.19 / 3.59),
            "l4": option("l4", 0.44 / 3.59),
        }
        specification = DeploymentSpecification(
            self.model, 4096, 12.0, 48.0, catalog
        )
        result = provision_catalog(
            specification,
            assignments=(("a100_pcie", "l4"),),
        )
        self.assertIsNone(result.best_afd)
        self.assertEqual(result.best_collocated.attention_hardware, "h200_sxm")
        self.assertEqual(result.selected_architecture, "collocated")

        guided = guided_provision_catalog(specification, evaluation_budget=1)
        self.assertIsNone(guided.best_afd)
        self.assertEqual(guided.best_collocated.attention_hardware, "h200_sxm")
        self.assertEqual(guided.selected_architecture, "collocated")

    def test_role_specialized_catalog_exceeds_global_collocated_by_ten(self) -> None:
        config = load_experiment_config()
        commercial = load_priced_existing_catalog(config)
        full_catalog = fixed_simulated_catalog(
            commercial,
            config["simulated_catalog"],
        )
        menu = {
            name: full_catalog[name]
            for name in (
                "b200_sxm",
                "sim_attention_target",
                "sim_expert_target",
            )
        }
        specification = DeploymentSpecification(
            self.model, 4096, 12.0, 48.0, menu
        )
        result = provision_catalog(specification)
        self.assertEqual(
            result.best_afd_assignment,
            ("sim_attention_target", "sim_expert_target"),
        )
        self.assertEqual(result.best_collocated.attention_hardware, "b200_sxm")
        self.assertGreater(result.comparison_gain, 10.0)

    def test_guided_catalog_search_recovers_small_catalog_oracle(self) -> None:
        catalog = {
            "h200_sxm": option("h200_sxm", 1.0),
            "a100_pcie": option("a100_pcie", 1.19 / 3.59),
            "l40s": option("l40s", 0.79 / 3.59),
        }
        specification = DeploymentSpecification(
            self.model, 4096, 12.0, 48.0, catalog
        )
        oracle = provision_catalog(specification)
        guided = guided_provision_catalog(specification, evaluation_budget=6)
        self.assertEqual(
            (
                guided.best_afd.attention_hardware,
                guided.best_afd.expert_hardware,
            ),
            oracle.best_afd_assignment,
        )
        self.assertEqual(guided.evaluated_assignments, 6)
        self.assertAlmostEqual(
            guided.best_afd.throughput_tps,
            oracle.best_afd.throughput_tps,
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
