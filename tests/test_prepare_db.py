"""Tests for the pure pieces of prepare_db suffix tagging.

DB-level end-to-end behaviour is covered by pipeline verification.
"""

import unittest


class SplitPatternsByShape(unittest.TestCase):
    def test_forward_and_suffix_split(self):
        from src.prepare_db import _split_patterns_by_shape

        fwd, suf = _split_patterns_by_shape(["mor", "änke*", "*dotter", "*hustru", "fru grefvinna"])
        self.assertEqual(sorted(fwd), sorted(["mor", "änke*", "fru grefvinna"]))
        self.assertEqual(sorted(suf), sorted(["*dotter", "*hustru"]))

    def test_no_suffixes_returns_empty_suf(self):
        from src.prepare_db import _split_patterns_by_shape

        fwd, suf = _split_patterns_by_shape(["mor", "kvinn*"])
        self.assertEqual(fwd, ["mor", "kvinn*"])
        self.assertEqual(suf, [])


class ReverseSuffixForFTS(unittest.TestCase):
    def test_reverses_and_appends_star(self):
        from src.prepare_db import _reverse_suffix_pattern

        self.assertEqual(_reverse_suffix_pattern("*inna"), "anni*")
        self.assertEqual(_reverse_suffix_pattern("*erska"), "aksre*")
        self.assertEqual(_reverse_suffix_pattern("*dotter"), "rettod*")


if __name__ == "__main__":
    unittest.main()
