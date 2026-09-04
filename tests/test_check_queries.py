"""Tests for check_queries suffix-pattern support.

Pattern shapes:
  literal  "mor"       — exact match
  prefix   "änke*"     — starts-with
  suffix   "*inna"     — ends-with  (NEW)
  phrase   "fru grefvinna"

Suffix↔prefix overlap is theoretically infinite (unbounded middle), so it's
reported as a *soft* conflict — printed by report() but ignored by validate().
"""

import unittest

from src.check_queries import _overlaps, find_conflicts, validate


class OverlapSuffix(unittest.TestCase):
    def test_suffix_matches_literal_ending(self):
        self.assertTrue(_overlaps("*inna", "grevinna"))

    def test_suffix_does_not_match_unrelated_literal(self):
        self.assertFalse(_overlaps("*inna", "mor"))

    def test_two_suffixes_overlap_when_one_ends_with_other(self):
        self.assertTrue(_overlaps("*erska", "*ska"))
        self.assertTrue(_overlaps("*ska", "*erska"))

    def test_two_suffixes_disjoint_when_neither_ends_with_other(self):
        self.assertFalse(_overlaps("*inna", "*erska"))

    def test_suffix_and_prefix_overlap_theoretically(self):
        # any prefix + arbitrary middle + any suffix can coexist
        self.assertTrue(_overlaps("*inna", "sinn*"))
        self.assertTrue(_overlaps("sinn*", "*inna"))

    def test_suffix_is_symmetric_with_literal(self):
        self.assertEqual(_overlaps("*inna", "grevinna"), _overlaps("grevinna", "*inna"))


class FindConflictsSuffix(unittest.TestCase):
    def test_suffix_absorbs_literal_is_soft(self):
        # A literal in one group absorbed by a wildcard in another is a
        # curation opportunity (the literal is now dead code because
        # first-hit-wins classifies it via the wildcard's category), not a
        # categorization ambiguity. Soft.
        qs = {"A": ["*inna"], "B": ["grevinna"]}
        conflicts = find_conflicts(qs)
        self.assertTrue(conflicts, "expected the absorption to be reported")
        hard = [c for c in conflicts if c.kind in ("duplicate", "subsumption", "internal_duplicate")]
        self.assertFalse(
            hard,
            f"suffix-absorbing-literal must not be reported as hard; got {hard!r}",
        )
        kinds = {c.kind for c in conflicts}
        self.assertIn("literal_absorbed", kinds)

    def test_prefix_absorbs_literal_is_soft(self):
        # Same rule applied to the existing prefix shape.
        qs = {"A": ["änke*"], "B": ["änkefru"]}
        conflicts = find_conflicts(qs)
        hard = [c for c in conflicts if c.kind in ("duplicate", "subsumption", "internal_duplicate")]
        self.assertFalse(hard, f"prefix-absorbing-literal must be soft; got {hard!r}")

    def test_cross_group_suffix_over_prefix_is_soft(self):
        qs = {"A": ["*inna"], "B": ["sinn*"]}
        conflicts = find_conflicts(qs)
        self.assertTrue(conflicts, "expected suffix↔prefix to be reported")
        hard = [c for c in conflicts if c.kind in ("duplicate", "subsumption", "internal_duplicate")]
        self.assertFalse(
            hard,
            f"suffix↔prefix must not be reported as hard; got {hard!r}",
        )

    def test_validate_ignores_soft_only_conflicts(self):
        qs = {"A": ["*inna"], "B": ["sinn*"]}
        conflicts = find_conflicts(qs)
        hard = [c for c in conflicts if c.kind in ("duplicate", "subsumption", "internal_duplicate")]
        self.assertEqual(hard, [], "soft-only conflicts should not be raised")

    def test_two_wildcards_overlapping_are_hard(self):
        # Concrete: *erska matches any token that *ska matches — genuine
        # categorization ambiguity at the pattern level.
        qs = {"A": ["*ska"], "B": ["*erska"]}
        conflicts = find_conflicts(qs)
        subs = [c for c in conflicts if c.kind == "subsumption"]
        self.assertTrue(subs, "two overlapping wildcards should be hard subsumption")

    def test_two_prefixes_overlapping_are_hard(self):
        qs = {"A": ["a*"], "B": ["abc*"]}
        conflicts = find_conflicts(qs)
        subs = [c for c in conflicts if c.kind == "subsumption"]
        self.assertTrue(subs, "two overlapping prefixes should be hard subsumption")


class ValidateModuleQueries(unittest.TestCase):
    def test_shipped_queries_still_pass_validate(self):
        # regression: the current queries.py must remain hard-clean after the
        # code changes (before we add any suffix patterns to it)
        try:
            validate()
        except SystemExit as e:
            self.fail(f"validate() raised on shipped queries.py: {e}")


if __name__ == "__main__":
    unittest.main()
