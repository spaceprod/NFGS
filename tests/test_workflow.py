import unittest
from pathlib import Path
from nfgs.paths import project_root
from nfgs.workflow import read_config, plan, experiment_arguments


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.root = project_root()
        self.configs = self.root / "configs"

    def test_dependencies_precede_consumers(self):
        campaign = read_config(self.configs / "campaign_paper.json")
        experiments = plan(campaign, self.configs, ["reader_diagnostics"])
        ids = [c["id"] for c in experiments]
        for config in experiments:
            for required in config.get("depends_on", []):
                self.assertLess(ids.index(required), ids.index(config["id"]))
        self.assertIn("archived_inputs", ids)

    def test_quick_profile_does_not_use_paper_matrix(self):
        campaign = read_config(self.configs / "campaign_paper.json")
        items = plan(campaign, self.configs, profile="quick")
        self.assertNotIn("quantitative_topology", [c["id"] for c in items])
        config = read_config(self.configs / "td_rho_scan.json")
        args = experiment_arguments(config, Path("/tmp/results"), "quick", "cpu")
        self.assertEqual(args[args.index("--max-iter") + 1], "3")

    def test_every_experiment_has_its_own_script_and_config(self):
        campaign = read_config(self.configs / "campaign_paper.json")
        for identifier in campaign["experiments"]:
            self.assertTrue((self.root / "experiments" / f"{identifier}.py").is_file())
            self.assertEqual(
                read_config(self.configs / f"{identifier}.json")["id"], identifier
            )


if __name__ == "__main__":
    unittest.main()
