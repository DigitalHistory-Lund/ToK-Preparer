"""Tests for NEW _classify semantics: multi-category token overlap.

These tests express the semantics that will be implemented in Task 3.
They FAIL under the current code, which returns ``int | None`` (the index
of the first matching category only).

After Task 3 lands, ``_classify`` will return a collection (``list[int]``
or ``set[int]``) containing the indices of ALL matching categories, with
within-category duplicates collapsed to a single occurrence.

The tests use ``assertIn`` so they are forward-compatible with either
``list`` or ``set`` as the return type.
"""

import unittest

from src.word_frequencies import CATEGORY_ORDER, _classify


def _make_dicts(
    literals=None,
    prefixes=None,
    suffixes=None,
):
    """Build minimal literals/prefixes/suffixes dicts keyed by CATEGORY_ORDER."""
    lits: dict[str, set[str]] = {c: set() for c in CATEGORY_ORDER}
    prefs: dict[str, list[str]] = {c: [] for c in CATEGORY_ORDER}
    suffs: dict[str, list[str]] = {c: [] for c in CATEGORY_ORDER}
    if literals:
        for cat, words in literals.items():
            lits[cat].update(words)
    if prefixes:
        for cat, pats in prefixes.items():
            prefs[cat].extend(pats)
    if suffixes:
        for cat, pats in suffixes.items():
            suffs[cat].extend(pats)
    return lits, prefs, suffs


# Shorthand aliases for the three categories
K1 = CATEGORY_ORDER[0]  # "kvinna 1"
K2 = CATEGORY_ORDER[1]  # "Kvinna 2"
K3 = CATEGORY_ORDER[2]  # "Kvinna 3"


class TestCrossKOverlap(unittest.TestCase):
    """A token that appears in multiple category pattern sets must yield all indices."""

    def test_literal_in_both_k1_and_k2_returns_both_indices(self):
        """'kvinna' listed as a literal in K1 and K2 → result contains 0 and 1."""
        lits, prefs, suffs = _make_dicts(
            literals={
                K1: {"kvinna"},
                K2: {"kvinna"},
            }
        )
        result = _classify("kvinna", lits, prefs, suffs)
        # Under new semantics result is a collection; under old code it is int 0.
        self.assertIn(0, result)
        self.assertIn(1, result)

    def test_prefix_match_in_k1_and_literal_in_k2_returns_both_indices(self):
        """'kvinnoröst': prefix 'kvinn' in K1, literal in K2 → indices 0 and 1."""
        lits, prefs, suffs = _make_dicts(
            literals={K2: {"kvinnoröst"}},
            prefixes={K1: ["kvinn"]},
        )
        result = _classify("kvinnoröst", lits, prefs, suffs)
        self.assertIn(0, result)
        self.assertIn(1, result)

    def test_suffix_match_in_k2_and_literal_in_k3_returns_both_indices(self):
        """'grevinna': suffix 'inna' in K2, literal in K3 → indices 1 and 2."""
        lits, prefs, suffs = _make_dicts(
            literals={K3: {"grevinna"}},
            suffixes={K2: ["inna"]},
        )
        result = _classify("grevinna", lits, prefs, suffs)
        self.assertIn(1, result)
        self.assertIn(2, result)


class TestWithinKDedup(unittest.TestCase):
    """A token matching multiple patterns inside one K contributes that K exactly once."""

    def test_literal_and_prefix_match_in_k1_yields_single_k1_entry(self):
        """'kvinna' matches both K1 literal AND K1 prefix 'kvinn' → index 0 appears once."""
        lits, prefs, suffs = _make_dicts(
            literals={K1: {"kvinna"}},
            prefixes={K1: ["kvinn"]},
        )
        result = _classify("kvinna", lits, prefs, suffs)
        # Convert to list to count occurrences regardless of return type.
        as_list = list(result)
        self.assertIn(0, as_list)
        self.assertEqual(as_list.count(0), 1, "index 0 must appear exactly once")

    def test_two_prefix_matches_in_k2_yields_single_k2_entry(self):
        """'hustrun' matches two K2 prefixes 'hustr' and 'hust' → index 1 appears once."""
        lits, prefs, suffs = _make_dicts(
            prefixes={K2: ["hustr", "hust"]},
        )
        result = _classify("hustrun", lits, prefs, suffs)
        as_list = list(result)
        self.assertIn(1, as_list)
        self.assertEqual(as_list.count(1), 1, "index 1 must appear exactly once")

    def test_no_match_returns_empty_collection(self):
        """A token with no pattern matches must yield an empty collection (not None)."""
        lits, prefs, suffs = _make_dicts(
            literals={K1: {"kvinna"}},
        )
        result = _classify("herr", lits, prefs, suffs)
        # Under old code this returns None; under new semantics it returns []/set().
        self.assertIsNotNone(result, "_classify should return an empty collection, not None")
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
