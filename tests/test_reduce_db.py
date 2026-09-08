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


class ComputeDiscussionIds(unittest.TestCase):
    """compute_discussion_ids assigns a discussion_id to every utterance
    that belongs to a topic arc, using the paper's shipped defaults
    (max_gap=1, min_arc_length=2). Utterances with no arc are omitted
    from the returned dict (equivalent to SQL NULL).

    Arc rules recap:
    - An arc is a maximal run through the chamber chain in which no
      two consecutive utterances are both non-kvinna.
    - An arc must contain ≥ 2 kvinna utterances (min_arc_length=2);
      otherwise it is dropped.
    - Non-kvinna utterances that sit between two kvinna utterances in
      an arc (max_gap=1 tolerated interjection) receive the arc's id.
    - A pending non-kvinna at the very end of a chamber that is never
      confirmed by a following kvinna is dropped (not in-arc).
    """

    def _run(self, *chamber_sequences):
        """Convenience: pass one or more per-chamber sequences as varargs."""
        from src.reduce_db import compute_discussion_ids
        return compute_discussion_ids(chamber_sequences)

    def test_two_adjacent_kvinna_form_one_arc(self):
        out = self._run([("a", True), ("b", True)])
        self.assertEqual(out, {"a": 1, "b": 1})

    def test_isolated_single_kvinna_is_dropped(self):
        """Length-1 arc fails min_arc_length=2 → no id assigned."""
        out = self._run([("nope1", False), ("solo", True), ("nope2", False)])
        self.assertEqual(out, {})

    def test_kvinna_gap_kvinna_forms_one_arc_of_three(self):
        """max_gap=1 tolerates the single non-kvinna interjection; it
        gets tagged with the surrounding arc's id."""
        out = self._run([
            ("k1", True), ("nk_in", False), ("k2", True),
        ])
        self.assertEqual(out, {"k1": 1, "nk_in": 1, "k2": 1})

    def test_two_non_kvinna_break_the_arc(self):
        """k, nk, nk, k → arc breaks at the double-nk gap; each side is
        an isolated kvinna that fails min_arc_length=2 → nothing tagged."""
        out = self._run([
            ("k1", True), ("nk1", False), ("nk2", False), ("k2", True),
        ])
        self.assertEqual(out, {})

    def test_two_separate_arcs_in_same_chamber_get_distinct_ids(self):
        """Full sequence: arc1(k,k), nk, nk (break), arc2(k,k)."""
        out = self._run([
            ("a1", True), ("a2", True),
            ("g1", False), ("g2", False),
            ("b1", True), ("b2", True),
        ])
        self.assertEqual(out, {"a1": 1, "a2": 1, "b1": 2, "b2": 2})

    def test_two_chambers_get_distinct_ids_no_bleed(self):
        """Arcs are chamber-local (chambers imply separate sessions);
        two chambers with concurrent arcs get separate ids and never
        share an id."""
        out = self._run(
            [("ch1_a", True), ("ch1_b", True)],
            [("ch2_a", True), ("ch2_b", True)],
        )
        self.assertEqual(out, {
            "ch1_a": 1, "ch1_b": 1,
            "ch2_a": 2, "ch2_b": 2,
        })

    def test_arcs_cannot_span_sessions(self):
        """A discussion that resumes on the next day (a new session /
        record) gets a fresh id. Each session-half must independently
        pass min_arc_length=2 or be dropped."""
        # Session A: k, k → one arc.
        # Session B: k, k → another arc (fresh id).
        out = self._run(
            [("A_k1", True), ("A_k2", True)],
            [("B_k1", True), ("B_k2", True)],
        )
        self.assertEqual(out, {
            "A_k1": 1, "A_k2": 1,
            "B_k1": 2, "B_k2": 2,
        })

    def test_arc_split_by_session_loses_short_halves(self):
        """A kvinna utterance right at the end of session A followed by
        one right at the start of session B cannot be one arc — each
        half is length 1 and fails min_arc_length=2 → both dropped."""
        out = self._run(
            [("A_solo", True)],
            [("B_solo", True)],
        )
        self.assertEqual(out, {})

    def test_trailing_pending_non_kvinna_is_dropped(self):
        """k, k, nk at end of chamber: the trailing nk is never
        confirmed by a following kvinna, so it stays untagged."""
        out = self._run([
            ("k1", True), ("k2", True), ("nk_trailing", False),
        ])
        self.assertEqual(out, {"k1": 1, "k2": 1})

    def test_leading_pending_non_kvinna_not_carried_into_first_arc(self):
        """A non-kvinna that precedes the first kvinna in a chamber
        cannot be an interjection (there's no prior arc to interject in);
        it should not be tagged."""
        out = self._run([
            ("nk_leading", False), ("k1", True), ("k2", True),
        ])
        self.assertEqual(out, {"k1": 1, "k2": 1})


if __name__ == "__main__":
    unittest.main()
