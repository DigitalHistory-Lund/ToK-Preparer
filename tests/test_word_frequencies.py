"""Tests for suffix support in word_frequencies pattern-classification."""

import unittest

from src.word_frequencies import _partition_patterns, _classify, CATEGORY_ORDER


def _tmp_queries_shape():
    """A stand-in mini query dict of every pattern shape, used across tests."""
    return {
        # K1: single prefix (as in real queries)
        "kvinna 1": ["kvinn*"],
        # K2: literal + prefix (existing shapes) + suffix (new)
        "Kvinna 2": ["mor", "änke*", "*dotter"],
        # K3: literal + prefix + suffix + phrase
        "Kvinna 3": ["lärarinna", "sömmerska*", "*hustru", "fru grefvinna"],
    }


class PartitionPatterns(unittest.TestCase):
    def test_returns_suffixes_bucket(self):
        result = _partition_patterns.__wrapped__() if hasattr(_partition_patterns, "__wrapped__") else None
        # partition operates on the module-level `queries`; call the underlying
        # function with a swapped queries dict via monkeypatch instead
        import src.word_frequencies as wf

        original = wf.queries
        wf.queries = _tmp_queries_shape()
        try:
            out = wf._partition_patterns()
        finally:
            wf.queries = original

        # Contract: return a 5-tuple (literals, prefixes, suffixes, contains, phrases).
        self.assertEqual(
            len(out), 5,
            f"expected 5 buckets (literals, prefixes, suffixes, contains, phrases), got {len(out)}",
        )
        literals, prefixes, suffixes, _contains, phrases = out
        self.assertIn("dotter", suffixes["Kvinna 2"])
        self.assertIn("hustru", suffixes["Kvinna 3"])
        # Existing shapes still land where they used to
        self.assertIn("mor", literals["Kvinna 2"])
        self.assertIn("änke", prefixes["Kvinna 2"])
        self.assertIn("lärarinna", literals["Kvinna 3"])
        self.assertIn("sömmerska", prefixes["Kvinna 3"])
        self.assertIn("fru grefvinna", phrases)

    def test_suffix_pattern_not_treated_as_literal_or_prefix(self):
        import src.word_frequencies as wf

        original = wf.queries
        wf.queries = {"Kvinna 2": ["*dotter"]}
        try:
            literals, prefixes, suffixes, _contains, _ = wf._partition_patterns()
        finally:
            wf.queries = original

        self.assertNotIn("*dotter", literals["Kvinna 2"])
        self.assertNotIn("*dotter", prefixes["Kvinna 2"])
        self.assertIn("dotter", suffixes["Kvinna 2"])


class Classify(unittest.TestCase):
    def _partition(self, qs):
        import src.word_frequencies as wf

        original = wf.queries
        wf.queries = qs
        try:
            return wf._partition_patterns()
        finally:
            wf.queries = original

    def test_suffix_matches_token(self):
        literals, prefixes, suffixes, _contains, _ = self._partition({
            "kvinna 1": [],
            "Kvinna 2": ["*dotter"],
            "Kvinna 3": [],
        })
        hits = _classify("fosterdotter", literals, prefixes, suffixes)
        self.assertIn(CATEGORY_ORDER.index("Kvinna 2"), hits)

    def test_suffix_no_match_returns_empty(self):
        literals, prefixes, suffixes, _contains, _ = self._partition({
            "kvinna 1": [],
            "Kvinna 2": ["*dotter"],
            "Kvinna 3": [],
        })
        self.assertEqual(_classify("son", literals, prefixes, suffixes), [])

    def test_k2_suffix_and_k3_literal_both_match(self):
        # Under multi-category semantics, "fosterdotter" is caught by both
        # K2's *dotter suffix AND K3's literal — both categories fire.
        literals, prefixes, suffixes, _contains, _ = self._partition({
            "kvinna 1": [],
            "Kvinna 2": ["*dotter"],
            "Kvinna 3": ["fosterdotter"],
        })
        hits = _classify("fosterdotter", literals, prefixes, suffixes)
        self.assertIn(CATEGORY_ORDER.index("Kvinna 2"), hits)
        self.assertIn(CATEGORY_ORDER.index("Kvinna 3"), hits)

    def test_existing_literal_and_prefix_still_classified(self):
        literals, prefixes, suffixes, _contains, _ = self._partition({
            "kvinna 1": ["kvinn*"],
            "Kvinna 2": ["mor"],
            "Kvinna 3": [],
        })
        self.assertIn(0, _classify("kvinnor", literals, prefixes, suffixes))
        self.assertIn(1, _classify("mor", literals, prefixes, suffixes))


class KAllUtts(unittest.TestCase):
    """An utterance with hits in two categories must count once in k_all_utts."""

    def test_utterance_hitting_two_categories_counted_once(self):
        import sqlite3
        import tempfile
        from pathlib import Path

        import src.word_frequencies as wf

        original_queries = wf.queries
        original_tmp_db = wf.tmp_db
        wf.queries = {
            "kvinna 1": ["kvinn*"],
            "Kvinna 2": ["hon"],
            "Kvinna 3": [],
        }

        with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as tf:
            db_path = tf.name
        try:
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE utterance (
                    id INTEGER PRIMARY KEY,
                    year INTEGER,
                    kammare INTEGER,
                    gender TEXT,
                    party TEXT,
                    who TEXT
                );
                CREATE TABLE utterance_fts (
                    id INTEGER PRIMARY KEY,
                    content TEXT
                );
                INSERT INTO utterance VALUES (1, 1920, 1, 'K', 'X', 'me');
                INSERT INTO utterance_fts VALUES (1, 'kvinnorna säger att hon vet');
                """
            )
            conn.commit()
            conn.close()

            wf.tmp_db = Path(db_path)
            buckets, speakers, _word_subsets = wf.compute()
        finally:
            wf.queries = original_queries
            wf.tmp_db = original_tmp_db
            Path(db_path).unlink(missing_ok=True)

        (bucket,) = buckets.values()
        (speaker,) = speakers.values()

        self.assertEqual(bucket["k1_utts"], 1)
        self.assertEqual(bucket["k2_utts"], 1)
        # The invariant that matters: dedup across categories at utterance level.
        self.assertEqual(bucket["k_all_utts"], 1)
        self.assertEqual(speaker["k_all_utts"], 1)
        # Matches sum exactly (categories are token-disjoint).
        self.assertEqual(
            bucket["k_all_matches"],
            bucket["k1_matches"] + bucket["k2_matches"] + bucket["k3_matches"],
        )


if __name__ == "__main__":
    unittest.main()
