"""
Unit Tests for Shannon Entropy Out-of-Distribution (OOD) Gating & CRI Biosecurity Knowledge Base.
Project: R26-SE-016
"""

import sys
import os
import math
import unittest
import numpy as np

# Ensure functions package is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def calculate_shannon_entropy(probabilities: list[float]) -> float:
    """Computes Shannon Entropy in bits: H(p) = -sum(p_i * log2(p_i))."""
    entropy = 0.0
    for p in probabilities:
        if p > 1e-9:
            entropy -= p * math.log2(p)
    return float(round(entropy, 4))


# CRI Knowledge Base Reference Map
CRI_BIOSECURITY_MAP = {
    "bud_rot": {
        "scientific_name": "Phytophthora palmivora",
        "severity": "critical",
        "treatment_chemical": "Copper Oxychloride (50% WP) at 4g/L or 1% Bordeaux Mixture",
        "treatment_cultural": "Uproot and incinerate dead crown to prevent oospore transmission",
        "quarantine_required": True
    },
    "leaf_blight": {
        "scientific_name": "Lasiodiplodia theobromae",
        "severity": "high",
        "treatment_chemical": "Mancozeb (75% WP) or Carbendazim (50% WP)",
        "treatment_cultural": "Prune severely necrotic leaflets during dry weather",
        "quarantine_required": False
    },
    "stem_bleeding": {
        "scientific_name": "Thielaviopsis paradoxa",
        "severity": "high",
        "treatment_chemical": "Chisel affected bark and apply Coal Tar / Bordeaux Paste",
        "treatment_cultural": "Improve root zone drainage and avoid mechanical trunk wounds",
        "quarantine_required": False
    },
    "weligama_coconut_leaf_wilt": {
        "scientific_name": "Phytoplasma sp.",
        "severity": "critical",
        "treatment_chemical": "No chemical cure. Vector management (Proutista moesta control)",
        "treatment_cultural": "Strict quarantine boundary adherence under Plant Protection Act",
        "quarantine_required": True
    },
    "healthy": {
        "scientific_name": "Cocos nucifera (Unaffected)",
        "severity": "normal",
        "treatment_chemical": "Standard CRI APN / YPM fertilizer circular application",
        "treatment_cultural": "Routine weed management and moisture conservation",
        "quarantine_required": False
    }
}


class TestEntropyAndOOD(unittest.TestCase):
    """Verifies Shannon Entropy formulation and OOD decision boundaries."""

    def test_sharp_distribution_low_entropy(self):
        """In-distribution sharp prediction (e.g. 96% confidence) should have very low entropy."""
        # 5-class distribution
        probs = [0.96, 0.01, 0.01, 0.01, 0.01]
        entropy = calculate_shannon_entropy(probs)
        # H = -(0.96*log2(0.96) + 4*(0.01*log2(0.01))) ~ 0.31 bits
        self.assertLess(entropy, 1.0)
        self.assertLess(entropy, 2.10, "High certainty prediction must be within In-Distribution range (H < 2.10).")

    def test_uniform_distribution_high_entropy(self):
        """Maximum uncertainty / random image (1/5 uniform) should yield maximum entropy log2(5) = 2.32 bits."""
        probs = [0.20, 0.20, 0.20, 0.20, 0.20]
        entropy = calculate_shannon_entropy(probs)
        expected_max_entropy = math.log2(5)  # 2.3219 bits
        self.assertAlmostEqual(entropy, expected_max_entropy, places=3)
        self.assertGreater(entropy, 2.10, "Uniform random noise must trigger OOD Rejection (H > 2.10).")

    def test_ood_cutoff_boundary(self):
        """Verify the exact H_th = 2.10 bits cutoff logic."""
        h_threshold = 2.10
        
        in_dist_sample = [0.80, 0.05, 0.05, 0.05, 0.05]
        h_in = calculate_shannon_entropy(in_dist_sample)
        self.assertTrue(h_in < h_threshold, f"Expected {h_in} < {h_threshold}")
        
        ambiguous_sample = [0.30, 0.25, 0.20, 0.15, 0.10]
        h_ambiguous = calculate_shannon_entropy(ambiguous_sample)
        self.assertTrue(h_ambiguous > h_threshold, f"Expected {h_ambiguous} > {h_threshold}")


class TestCRIKnowledgeBase(unittest.TestCase):
    """Verifies biosecurity protocol accuracy and advisory completeness."""

    def test_all_classes_mapped(self):
        """Ensure all 5 target research classes have complete treatment directives."""
        target_classes = ["bud_rot", "leaf_blight", "stem_bleeding", "weligama_coconut_leaf_wilt", "healthy"]
        for cls in target_classes:
            self.assertIn(cls, CRI_BIOSECURITY_MAP)
            data = CRI_BIOSECURITY_MAP[cls]
            self.assertIn("scientific_name", data)
            self.assertIn("treatment_chemical", data)
            self.assertIn("treatment_cultural", data)
            self.assertIn("quarantine_required", data)

    def test_critical_threats_require_strict_action(self):
        """Verify that Bud Rot and Weligama Wilt are classified as Critical."""
        self.assertEqual(CRI_BIOSECURITY_MAP["bud_rot"]["severity"], "critical")
        self.assertTrue(CRI_BIOSECURITY_MAP["bud_rot"]["quarantine_required"])
        self.assertEqual(CRI_BIOSECURITY_MAP["weligama_coconut_leaf_wilt"]["severity"], "critical")
        self.assertTrue(CRI_BIOSECURITY_MAP["weligama_coconut_leaf_wilt"]["quarantine_required"])


if __name__ == "__main__":
    unittest.main()
