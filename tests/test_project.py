"""Small CPU/source checks. These do not run a pretrained model or download data."""
from __future__ import annotations

import ast
import contextlib
import io
import json
from pathlib import Path
import re
import runpy
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
DAY1 = runpy.run_path(str(ROOT / "gstar_day1_selector.py"))
BUILDER = runpy.run_path(str(ROOT / "gstar_fullkv_pilot/gstar_prepare_pilot.py"))
FULL = runpy.run_path(str(ROOT / "gstar_fullkv_pilot/gstar_run_fullkv.py"))
SMOKE = runpy.run_path(str(ROOT / "gstar_mask_smoke.py"))
PAIRED = runpy.run_path(str(ROOT / "gstar_paired_pilot.py"))
EXTRACTOR = runpy.run_path(str(ROOT / "gstar_extract_coverage.py"))
COVERAGE = runpy.run_path(str(ROOT / "gstar_coverage_eval.py"))


def sample_bank(seed: int = 7):
    rng = torch.Generator().manual_seed(seed)
    a = torch.rand(2, 2, 64, generator=rng)
    g = torch.rand(2, 2, 64, generator=rng)
    positions = list(range(4, 68))
    bank = PAIRED["build_policy_bank"](a, g, positions, 72, 0.5, SMOKE)
    return bank, positions, rng


class ProjectTests(unittest.TestCase):
    def test_synthetic_selector_builtin(self):
        DAY1["self_test"]()

    def test_data_builder_builtin_without_download(self):
        BUILDER["self_test"]()

    def test_exact_budget_and_protection(self):
        bank, _, _ = sample_bank()
        protected = bank["budget"]["protected_prefix_positions"]
        for policy in bank["policies"].values():
            keep = policy["keep"]
            if keep is not None:
                self.assertTrue(bool((keep.sum(-1) == 36).all()))
                self.assertTrue(bool(keep[..., protected].all()))

    def test_reserve_endpoints(self):
        bank, _, _ = sample_bank()
        policies = bank["policies"]
        self.assertTrue(torch.equal(policies["reserve_beta_0.00"]["keep"], policies["a_only"]["keep"]))
        self.assertTrue(torch.equal(policies["reserve_beta_1.00"]["keep"], policies["g_proxy_only"]["keep"]))

    def test_total_score_constraints(self):
        bank, _, _ = sample_bank()
        for rho in PAIRED["RHOS"]:
            diag = bank["policies"][f"constrained_rho_{rho:.2f}"]["diagnostics"]
            self.assertGreaterEqual(diag["g_retention_min"] + 1e-12, rho)

    def test_coverage_constraints_and_budget(self):
        bank, positions, rng = sample_bank()
        values = torch.rand(2, 2, 4, 64, generator=rng) / 64
        new = COVERAGE["build_coverage_bank"](bank, values, positions)
        for eta in COVERAGE["ETAS"]:
            p = new["policies"][f"coverage_eta_{eta:.2f}"]
            self.assertEqual(p["diagnostics"]["coverage_constraint_violations"], 0)
            self.assertGreaterEqual(p["diagnostics"]["coverage_ratio_min"], eta)
            self.assertTrue(bool((p["keep"].sum(-1) == 36).all()))

    def test_zero_coverage_reference(self):
        bank, positions, _ = sample_bank()
        values = torch.zeros(2, 2, 4, 64)
        new = COVERAGE["build_coverage_bank"](bank, values, positions)
        for p in new["policies"].values():
            d = p["diagnostics"]
            self.assertEqual(d["zero_reference_blocks"], 16)
            self.assertIsNone(d["coverage_ratio_min"])
            self.assertEqual(d["coverage_constraint_violations"], 0)

    def test_reconstruction_block_reduction(self):
        gen = torch.Generator().manual_seed(21)
        weights = torch.softmax(torch.randn(1, 4, 40, 40, generator=gen), dim=-1)
        positions = [2, 3, 4, 5, 6, 7, 8, 9]
        cov, g = EXTRACTOR["reduce_layer_attention"](weights, positions, 12, 32, 40, 2, 4)
        expected = torch.empty(2, 4, 8)
        expected_g = torch.empty(2, 8)
        for h in range(2):
            for i, key in enumerate(positions):
                expected_g[h, i] = weights[0, h*2:h*2+2, 12:, key].max()
                for c, (start, end) in enumerate(EXTRACTOR["block_ranges"](8, 4)):
                    expected[h, c, i] = weights[0, h*2:h*2+2, 32+start:32+end, key].mean()
        torch.testing.assert_close(cov, expected)
        torch.testing.assert_close(g, expected_g)

    def test_score_behavior(self):
        score = FULL["score_answer"]
        self.assertEqual(score("Chicago", ["Chicago"]), {"em": 100.0, "f1": 100.0})
        self.assertEqual(score("wiretap", ["wiretapping"]), {"em": 0.0, "f1": 0.0})
        self.assertAlmostEqual(score("House music originated from Chicago.", ["Chicago"])["f1"], 100/3)

    def test_pilot_integrity(self):
        groups, _ = FULL["load_groups"](ROOT / "data/pilot")
        self.assertEqual(len(groups), 3)
        self.assertTrue(all(len(group["cases"]) == 2 for group in groups))

    def test_source_has_no_direct_print_calls(self):
        paths = list(ROOT.glob("*.py")) + list((ROOT / "gstar_fullkv_pilot").glob("*.py"))
        for p in paths:
            source = p.read_text(encoding="utf-8")
            tree = ast.parse(source)
            compile(source, str(p), "exec")
            self.assertIsNone(re.search(r"[\uac00-\ud7af]", source), p.name)
            for node in ast.walk(tree):
                direct = (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                          and node.func.id == "print")
                self.assertFalse(direct, p.name)

    def test_notebooks_clean_and_parseable(self):
        for p in (ROOT / "notebooks").glob("*.ipynb"):
            nb = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(nb["nbformat"], 4)
            for i, cell in enumerate(nb["cells"]):
                source = "".join(cell["source"])
                self.assertIsNone(re.search(r"[\uac00-\ud7af]", source), f"{p.name}:{i}")
                if cell["cell_type"] == "code":
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
                    compile(source, f"{p.name}:{i}", "exec")

    def test_imports_are_quiet(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for p in list(ROOT.glob("*.py")) + list((ROOT / "gstar_fullkv_pilot").glob("*.py")):
                runpy.run_path(str(p))
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
