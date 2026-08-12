"""Guards for the declared paper peak-area consumer graph."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class PeakCallerAuditTests(unittest.TestCase):
    def test_every_direct_historical_call_site_is_classified(self):
        config = json.loads(
            (REPO_ROOT / "config" / "peak_area_callers.json").read_text(
                encoding="utf-8"
            )
        )
        classified = {item["path"] for item in config["callers"]}
        direct: set[str] = set()
        for root in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
            for path in root.rglob("*.py"):
                relative = path.relative_to(REPO_ROOT).as_posix()
                text = path.read_text(encoding="utf-8")
                if ".area(" in text or "unc_ratio(" in text:
                    direct.add(relative)
        self.assertTrue(direct.issubset(classified), sorted(direct - classified))

    def test_corrected_workflow_has_no_historical_estimators(self):
        config = json.loads(
            (REPO_ROOT / "config" / "peak_area_callers.json").read_text(
                encoding="utf-8"
            )
        )
        for workflow in config["corrected_workflows"]:
            text = (REPO_ROOT / workflow["path"]).read_text(encoding="utf-8")
            for forbidden in workflow["required_absences"]:
                self.assertNotIn(forbidden, text)

    def test_paper_legacy_clis_require_an_explicit_legacy_mode(self):
        for relative in (
            "scripts/rd_peak_fitter.py",
            "scripts/peak_ratio_compare.py",
            "scripts/RD_neutron_gamma_fit.py",
        ):
            text = (REPO_ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("add_mutually_exclusive_group(required=True)", text)
            self.assertIn('"--paper-legacy"', text)
            self.assertIn('"--legacy-window-counts"', text)

    def test_frozen_components_reference_declared_windows(self):
        config = json.loads(
            (REPO_ROOT / "config" / "paper_peak_statistics.json").read_text(
                encoding="utf-8"
            )
        )
        for table in ("table3", "table8"):
            window_names = {item["name"] for item in config[table]["windows"]}
            component_names = [item["name"] for item in config[table]["components"]]
            self.assertEqual(len(component_names), len(set(component_names)))
            for component in config[table]["components"]:
                self.assertIn(component["window"], window_names)
        self.assertEqual(config["table3"]["selection"]["file_ids_in_time_order"], [1716, 1765, 334, 1676])
        self.assertEqual(config["table8"]["selection"]["file_id"], 1042)

    def test_table3_origin_classes_are_explicit_and_frozen(self):
        config = json.loads(
            (REPO_ROOT / "config" / "paper_peak_statistics.json").read_text(
                encoding="utf-8"
            )
        )
        components = config["table3"]["components"]
        self.assertTrue(
            all(
                component["origin_class"]
                in {"neutron_capture", "radioactive_decay"}
                for component in components
            )
        )
        by_name = {component["name"]: component for component in components}
        self.assertEqual(
            by_name["rd_2223_0"]["origin_class"], "neutron_capture"
        )
        self.assertEqual(
            by_name["rd_1293_6"]["origin_class"], "radioactive_decay"
        )


if __name__ == "__main__":
    unittest.main()
