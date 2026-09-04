"""Tests for suffix support in reduce_db._matches_pattern and annotate_content."""

import unittest
from unittest.mock import patch

from src.reduce_db import _matches_pattern, annotate_content


class MatchesPattern(unittest.TestCase):
    def test_literal_still_works(self):
        self.assertTrue(_matches_pattern("mor", "mor"))
        self.assertFalse(_matches_pattern("mormor", "mor"))

    def test_prefix_still_works(self):
        self.assertTrue(_matches_pattern("änkefru", "änke*"))
        self.assertFalse(_matches_pattern("fru", "änke*"))

    def test_suffix_matches_ending(self):
        self.assertTrue(_matches_pattern("fosterdotter", "*dotter"))
        self.assertTrue(_matches_pattern("dotter", "*dotter"))

    def test_suffix_does_not_match_non_ending(self):
        self.assertFalse(_matches_pattern("son", "*dotter"))
        self.assertFalse(_matches_pattern("dottern", "*dotter"))


class AnnotateContentWithSuffix(unittest.TestCase):
    def test_suffix_match_gets_k3_superscript(self):
        # Swap in a minimal queries dict so the test is independent of shipped patterns
        fake = {
            "kvinna 1": [],
            "Kvinna 2": [],
            "Kvinna 3": ["*hustru"],
        }
        with patch("src.reduce_db.queries", fake):
            out = annotate_content("En arkitekthustru talade.", k1=False, k2=False, k3=True)
        self.assertIn("arkitekthustru³", out)

    def test_no_flag_no_annotation_even_if_suffix_present(self):
        fake = {
            "kvinna 1": [],
            "Kvinna 2": [],
            "Kvinna 3": ["*hustru"],
        }
        with patch("src.reduce_db.queries", fake):
            out = annotate_content("En arkitekthustru talade.", k1=False, k2=False, k3=False)
        self.assertEqual(out, "En arkitekthustru talade.")


if __name__ == "__main__":
    unittest.main()
