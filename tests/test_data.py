"""Tests for src.data.read_unknown_share and its helper."""

import unittest

import pandas as pd


class ComputeUnknownShare(unittest.TestCase):
    """Test the pure aggregation helper. Public readers just wire the file input."""

    def test_all_known_gives_zero_share(self):
        from src.data import _compute_unknown_share

        df = pd.DataFrame(
            [
                {"year": 1900, "chamber": 1, "gender": "man", "party": "S", "utterance_count": 10},
                {"year": 1900, "chamber": 2, "gender": "woman", "party": "S", "utterance_count": 5},
            ]
        )
        out = _compute_unknown_share(df)
        row = out.loc[out["year"] == 1900].iloc[0]
        self.assertEqual(int(row["utterance_count_total"]), 15)
        self.assertEqual(int(row["utterance_count_unknown"]), 0)
        self.assertEqual(float(row["unknown_share"]), 0.0)

    def test_empty_gender_counted_as_unknown(self):
        # After _read_csv, empty gender comes back as NaN
        from src.data import _compute_unknown_share

        df = pd.DataFrame(
            [
                {"year": 1901, "chamber": 1, "gender": "man", "party": "S", "utterance_count": 20},
                {"year": 1901, "chamber": 1, "gender": None, "party": None, "utterance_count": 5},
            ]
        )
        out = _compute_unknown_share(df)
        row = out.loc[out["year"] == 1901].iloc[0]
        self.assertEqual(int(row["utterance_count_total"]), 25)
        self.assertEqual(int(row["utterance_count_unknown"]), 5)
        self.assertAlmostEqual(float(row["unknown_share"]), 0.20)

    def test_multi_year_grouping(self):
        from src.data import _compute_unknown_share

        df = pd.DataFrame(
            [
                {"year": 1900, "chamber": 1, "gender": None, "party": None, "utterance_count": 3},
                {"year": 1900, "chamber": 1, "gender": "man", "party": "S", "utterance_count": 7},
                {"year": 1901, "chamber": 1, "gender": None, "party": None, "utterance_count": 4},
                {"year": 1901, "chamber": 1, "gender": "woman", "party": "S", "utterance_count": 6},
            ]
        )
        out = _compute_unknown_share(df)
        self.assertEqual(sorted(out["year"].tolist()), [1900, 1901])
        r1900 = out.loc[out["year"] == 1900].iloc[0]
        r1901 = out.loc[out["year"] == 1901].iloc[0]
        self.assertAlmostEqual(float(r1900["unknown_share"]), 0.3)
        self.assertAlmostEqual(float(r1901["unknown_share"]), 0.4)

    def test_returns_dataframe_with_expected_columns(self):
        from src.data import _compute_unknown_share

        df = pd.DataFrame(
            [{"year": 1900, "chamber": 1, "gender": "man", "party": "S", "utterance_count": 1}]
        )
        out = _compute_unknown_share(df)
        self.assertEqual(
            list(out.columns),
            ["year", "utterance_count_total", "utterance_count_unknown", "unknown_share"],
        )

    def test_share_bounded_zero_one(self):
        from src.data import _compute_unknown_share

        df = pd.DataFrame(
            [
                {"year": 1900, "chamber": 1, "gender": None, "party": None, "utterance_count": 3},
                {"year": 1900, "chamber": 1, "gender": "man", "party": "S", "utterance_count": 7},
            ]
        )
        out = _compute_unknown_share(df)
        self.assertTrue(((out["unknown_share"] >= 0) & (out["unknown_share"] <= 1)).all())


class PublicReaders(unittest.TestCase):
    """Confirm the public read_unknown_share* wire up file → helper."""

    def test_read_unknown_share_returns_dataframe(self):
        from src.data import read_unknown_share

        out = read_unknown_share()
        self.assertIsInstance(out, pd.DataFrame)
        self.assertIn("unknown_share", out.columns)
        self.assertTrue(((out["unknown_share"] >= 0) & (out["unknown_share"] <= 1)).all())

    def test_chamber_variants_exist(self):
        from src.data import read_unknown_share1, read_unknown_share2

        for reader in (read_unknown_share1, read_unknown_share2):
            out = reader()
            self.assertIsInstance(out, pd.DataFrame)
            self.assertIn("unknown_share", out.columns)


class KvinnaSubsetsTest(unittest.TestCase):
    """Tests for read_kvinna_utterance_subsets."""

    def test_returns_seven_row_frame(self):
        from src.data import read_kvinna_utterance_subsets

        out = read_kvinna_utterance_subsets()
        self.assertEqual(len(out), 7)
        self.assertEqual(
            sorted(out.columns.tolist()),
            sorted(["k1", "k2", "k3", "utterance_count"]),
        )

    def test_sum_equals_word_frequencies_k_all_utts(self):
        # The sqlite kvinna_1/2/3 booleans were set via FTS5 MATCH in prepare_db;
        # word_frequencies.csv.gz uses a regex tokeniser on the same content.
        # The two methods agree within ~1 % of the total — verify the sqlite sum
        # is in the expected range rather than demanding exact equality.
        from src.data import read_kvinna_utterance_subsets, read_word_frequencies

        subsets = read_kvinna_utterance_subsets()
        wf = read_word_frequencies()
        sqlite_total = int(subsets["utterance_count"].sum())
        csv_total = int(wf["k_all_utts"].sum())
        # The FTS5-based tagging finds at most a small fraction more hits than
        # the regex-based count.  Require the two sources to agree within 2 %.
        self.assertGreaterEqual(sqlite_total, csv_total)
        self.assertLessEqual(sqlite_total, int(csv_total * 1.02))

    def test_year_filter_restricts_output(self):
        from src.data import read_kvinna_utterance_subsets

        all_years = read_kvinna_utterance_subsets()
        one_year = read_kvinna_utterance_subsets(years=(1921,))
        # A single year must have fewer K-tagged utterances than all 41 years.
        self.assertLess(
            int(one_year["utterance_count"].sum()),
            int(all_years["utterance_count"].sum()),
        )
        # And the year filter must not return all rows (sanity check).
        self.assertGreater(int(one_year["utterance_count"].sum()), 0)


class BuildSpeakerWordTotals(unittest.TestCase):
    """build_speaker_word_totals + read_speaker_word_totals wiring."""

    def test_read_triggers_build_when_file_missing(self):
        # Point the reader at a temp path; it must call build_speaker_word_totals.
        import tempfile
        from pathlib import Path
        from unittest import mock
        from src import data as data_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_path = Path(tmpdir) / "not_there.csv.gz"
            with (
                mock.patch.object(data_mod, "speaker_word_totals_path", fake_path),
                mock.patch.object(data_mod, "build_speaker_word_totals") as build_mock,
            ):
                # Make the "build" write a minimal valid CSV so the read call
                # afterwards has something to parse.
                def fake_build(output_path=None):
                    import pandas as pd
                    df = pd.DataFrame(columns=list(
                        data_mod.SPEAKER_WORD_TOTALS_COLUMNS
                    ))
                    df.to_csv(fake_path, index=False, compression="gzip")
                    return df

                build_mock.side_effect = fake_build
                data_mod.read_speaker_word_totals()
                build_mock.assert_called_once()

    def test_schema_columns_match(self):
        # SPEAKER_WORD_TOTALS_COLUMNS is the contract for both build and read.
        from src.data import SPEAKER_WORD_TOTALS_COLUMNS
        self.assertEqual(
            SPEAKER_WORD_TOTALS_COLUMNS,
            (
                "year", "chamber", "gender", "party", "who",
                "total_words",
                "k1_matches", "k2_matches", "k3_matches", "k_all_matches",
            ),
        )


class BuildSpeakerTotals(unittest.TestCase):
    """build_speaker_totals aggregates speakers.csv.gz + persons.sqlite."""

    @staticmethod
    def _fake_speakers() -> pd.DataFrame:
        return pd.DataFrame([
            # Same MP across two years — should sum.
            {"year": 1925, "chamber": 2, "gender": "man", "party": "S", "who": "i-A",
             "utterance_count": 100, "k1_utts": 5, "k2_utts": 3, "k3_utts": 2, "k_all_utts": 8},
            {"year": 1926, "chamber": 2, "gender": "man", "party": "S", "who": "i-A",
             "utterance_count": 150, "k1_utts": 10, "k2_utts": 4, "k3_utts": 3, "k_all_utts": 15},
            # Woman, single year.
            {"year": 1925, "chamber": 2, "gender": "woman", "party": "L", "who": "i-B",
             "utterance_count": 50, "k1_utts": 20, "k2_utts": 5, "k3_utts": 3, "k_all_utts": 25},
            # Unknown-speaker row — must be dropped by the build.
            {"year": 1925, "chamber": 2, "gender": "", "party": "", "who": "",
             "utterance_count": 999, "k1_utts": 500, "k2_utts": 0, "k3_utts": 0, "k_all_utts": 500},
        ])

    def test_aggregates_across_years_and_computes_share(self):
        import tempfile
        from pathlib import Path
        from src.data import build_speaker_totals

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "speaker_totals.csv.gz"
            missing_db = Path(tmpdir) / "nope.sqlite"
            df = build_speaker_totals(
                speakers=self._fake_speakers(),
                persons_path=missing_db,
                output_path=out,
            )
            self.assertTrue(out.exists(), "output CSV was not written")
            self.assertNotIn("", df["who"].tolist(), "unknown-speaker row leaked")
            row_a = df[df["who"] == "i-A"].iloc[0]
            self.assertEqual(int(row_a["utterance_count"]), 250)  # 100 + 150
            self.assertEqual(int(row_a["k_all_utts"]), 23)  # 8 + 15
            self.assertAlmostEqual(float(row_a["share"]), 23 / 250)
            # Empty name when persons DB is missing — plot falls back to `who`.
            self.assertEqual(row_a["name"], "")

    def test_output_sorted_by_share_descending(self):
        import tempfile
        from pathlib import Path
        from src.data import build_speaker_totals

        with tempfile.TemporaryDirectory() as tmpdir:
            df = build_speaker_totals(
                speakers=self._fake_speakers(),
                persons_path=Path(tmpdir) / "nope.sqlite",
                output_path=Path(tmpdir) / "speaker_totals.csv.gz",
            )
            shares = df["share"].tolist()
            self.assertEqual(shares, sorted(shares, reverse=True))
class TopicArcStarterSQLTest(unittest.TestCase):
    """Test the topic-arc-starter SQL against an in-memory sqlite fixture."""

    def _build_conn(self):
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE person (
                id INTEGER PRIMARY KEY,
                name TEXT,
                gender TEXT,
                party TEXT
            );
            CREATE TABLE utterance (
                id TEXT PRIMARY KEY,
                content TEXT,
                prev TEXT,
                next TEXT,
                person_id INTEGER,
                year INTEGER,
                date INTEGER,
                kammare INTEGER,
                kvinna_1 BOOLEAN,
                kvinna_2 BOOLEAN,
                kvinna_3 BOOLEAN
            );
            INSERT INTO person VALUES
                (1, 'A Man',   'man',   'S'),
                (2, 'A Woman', 'woman', 'S'),
                (3, 'Unknown', NULL,    NULL);
            """
        )
        # Chain: u1 -> u2 -> u3 -> u4 -> u5 -> u6 (chamber 1).
        # kvinna hits at u2, u3, u5. u4 has NO kvinna hit (topic gap).
        #   → u2 is starter (prev u1 has no hit): first arc, spoken by MAN
        #   → u3 is CONTINUATION (prev u2 has hit): not a starter
        #   → u5 is starter (prev u4 has no hit): second arc, spoken by WOMAN
        rows = [
            ("u1", None, "u2", 1, 1922, 1922_01_01, 1, None, None, None),
            ("u2", "u1", "u3", 1, 1922, 1922_01_01, 1, 1,    None, None),  # MAN starter
            ("u3", "u2", "u4", 2, 1922, 1922_01_01, 1, 1,    None, None),  # continuation
            ("u4", "u3", "u5", 1, 1922, 1922_01_02, 1, None, None, None),  # gap
            ("u5", "u4", "u6", 2, 1922, 1922_01_02, 1, None, 1,    None),  # WOMAN starter
            ("u6", "u5", None, 1, 1922, 1922_01_02, 1, None, None, None),
        ]
        conn.executemany(
            "INSERT INTO utterance (id, prev, next, person_id, year, date, kammare, kvinna_1, kvinna_2, kvinna_3) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        return conn

    def test_starter_count_by_gender(self):
        from src.data import _TOPIC_ARC_STARTER_SQL

        conn = self._build_conn()
        sql = _TOPIC_ARC_STARTER_SQL.format(chamber_clause="")
        results = {gender: n for _, gender, n in conn.execute(sql)}
        self.assertEqual(results.get("man"), 1)
        self.assertEqual(results.get("woman"), 1)
        self.assertNotIn("unknown", results)

    def test_chamber_filter(self):
        from src.data import _TOPIC_ARC_STARTER_SQL

        conn = self._build_conn()
        sql = _TOPIC_ARC_STARTER_SQL.format(chamber_clause="AND u.kammare IN (2)")
        # No utterances in chamber 2 → zero starters.
        self.assertEqual(list(conn.execute(sql)), [])

    def test_max_gap_1_tolerates_single_non_kvinna_interjection(self):
        """A K, ~K, K chain is 2 arcs under gap=0 but 1 arc under gap=1."""
        import sqlite3

        from src.data import _build_starter_sql

        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE person (id INTEGER PRIMARY KEY, name TEXT, gender TEXT, party TEXT);
            CREATE TABLE utterance (
                id TEXT PRIMARY KEY, content TEXT, prev TEXT, next TEXT,
                person_id INTEGER, year INTEGER, date INTEGER, kammare INTEGER,
                kvinna_1 BOOLEAN, kvinna_2 BOOLEAN, kvinna_3 BOOLEAN
            );
            INSERT INTO person VALUES (1, 'M', 'man', 'S');
            INSERT INTO utterance (id, prev, next, person_id, year, kammare, kvinna_1) VALUES
                ('u1', NULL, 'u2', 1, 1922, 1, 1),
                ('u2', 'u1', 'u3', 1, 1922, 1, NULL),
                ('u3', 'u2', NULL, 1, 1922, 1, 1);
            """
        )
        # gap=0: u1 is a starter (prev NULL, no kvinna); u3 is a starter
        #        (prev u2 is non-kvinna). Total = 2.
        # gap=1: u1 is a starter (both prev NULL); u3 is NOT a starter because
        #        up2 = u1 has a kvinna hit — the single non-kvinna gap at u2
        #        is tolerated. Total = 1.
        sql0 = _build_starter_sql(max_gap=0).format(chamber_clause="")
        sql1 = _build_starter_sql(max_gap=1).format(chamber_clause="")
        results0 = list(conn.execute(sql0))
        results1 = list(conn.execute(sql1))
        self.assertEqual(sum(n for _, _, n in results0), 2)
        self.assertEqual(sum(n for _, _, n in results1), 1)

    def test_max_gap_negative_raises(self):
        from src.data import _build_starter_sql

        with self.assertRaises(ValueError):
            _build_starter_sql(max_gap=-1)

    def test_unknown_prior_utterance_is_starter(self):
        # If prev references an id not in the table, the LEFT JOIN yields NULL
        # for all up.kvinna_* columns; the WHERE should still admit the current
        # utterance as a starter.
        import sqlite3

        from src.data import _TOPIC_ARC_STARTER_SQL

        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE person (id INTEGER PRIMARY KEY, name TEXT, gender TEXT, party TEXT);
            CREATE TABLE utterance (
                id TEXT PRIMARY KEY, content TEXT, prev TEXT, next TEXT,
                person_id INTEGER, year INTEGER, date INTEGER, kammare INTEGER,
                kvinna_1 BOOLEAN, kvinna_2 BOOLEAN, kvinna_3 BOOLEAN
            );
            INSERT INTO person VALUES (1, 'A Woman', 'woman', 'S');
            INSERT INTO utterance (id, prev, next, person_id, year, kammare, kvinna_1)
                VALUES ('u1', 'dangling', NULL, 1, 1922, 1, 1);
            """
        )
        sql = _TOPIC_ARC_STARTER_SQL.format(chamber_clause="")
        results = list(conn.execute(sql))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], "woman")
        self.assertEqual(results[0][2], 1)


class SpeakerStarterSQLTest(unittest.TestCase):
    """Per-speaker aggregation variant of the topic-arc-starter SQL."""

    def test_groups_by_speaker_and_gender(self):
        import sqlite3

        from src.data import _build_speaker_starter_sql

        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE person (id INTEGER PRIMARY KEY, name TEXT, gender TEXT, party TEXT);
            CREATE TABLE utterance (
                id TEXT PRIMARY KEY, content TEXT, prev TEXT, next TEXT,
                person_id INTEGER, year INTEGER, date INTEGER, kammare INTEGER,
                kvinna_1 BOOLEAN, kvinna_2 BOOLEAN, kvinna_3 BOOLEAN
            );
            INSERT INTO person VALUES
                (1, 'i-alice', 'woman', 'S'),
                (2, 'i-bob',   'man',   'H'),
                (3, 'i-carla', 'woman', 'L');
            """
        )
        # alice: 2 starters (u1 with prev NULL, u3 after non-kvinna u2)
        # bob:   1 starter (u5 after non-kvinna u4)
        # carla: 0 starters (u7 continues alice's arc at u1 — but different chains)
        conn.executemany(
            "INSERT INTO utterance (id, prev, next, person_id, year, kammare, kvinna_1) VALUES (?,?,?,?,?,?,?)",
            [
                ("u1", None,  "u2", 1, 1922, 1, 1),
                ("u2", "u1",  "u3", 2, 1922, 1, None),
                ("u3", "u2",  "u4", 1, 1922, 1, 1),
                ("u4", "u3",  "u5", 2, 1922, 1, None),
                ("u5", "u4",  None, 2, 1922, 1, 1),
            ],
        )
        conn.commit()

        sql = _build_speaker_starter_sql(max_gap=0).format(chamber_clause="")
        results = {row[0]: (row[1], row[2]) for row in conn.execute(sql)}
        self.assertEqual(results["i-alice"], ("woman", 2))
        self.assertEqual(results["i-bob"], ("man", 1))
        self.assertNotIn("i-carla", results)  # carla never spoke

    def test_speaker_starter_sql_negative_gap_raises(self):
        from src.data import _build_speaker_starter_sql

        with self.assertRaises(ValueError):
            _build_speaker_starter_sql(max_gap=-1)


class MinArcLengthTest(unittest.TestCase):
    """min_arc_length=2 excludes isolated single-utterance mentions."""

    def _conn_with_singles_and_conversation(self):
        """Chain: u1(K), u2(~), u3(K), u4(K), u5(~), u6(K).

        Under max_gap=0:
        - u1 starts an arc of length 1 (u2 is not kvinna)
        - u3 starts an arc of length 2 (u4 is kvinna, u5 is not)
        - u6 starts an arc of length 1 (nothing after)
        Total starters min_arc_length=1: 3.
        Total starters min_arc_length=2: 1 (only u3).
        """
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE person (id INTEGER PRIMARY KEY, name TEXT, gender TEXT, party TEXT);
            CREATE TABLE utterance (
                id TEXT PRIMARY KEY, content TEXT, prev TEXT, next TEXT,
                person_id INTEGER, year INTEGER, date INTEGER, kammare INTEGER,
                kvinna_1 BOOLEAN, kvinna_2 BOOLEAN, kvinna_3 BOOLEAN
            );
            INSERT INTO person VALUES (1, 'A', 'man', 'S');
            """
        )
        conn.executemany(
            "INSERT INTO utterance (id, prev, next, person_id, year, kammare, kvinna_1) VALUES (?,?,?,?,?,?,?)",
            [
                ("u1", None,  "u2", 1, 1922, 1, 1),
                ("u2", "u1",  "u3", 1, 1922, 1, None),
                ("u3", "u2",  "u4", 1, 1922, 1, 1),
                ("u4", "u3",  "u5", 1, 1922, 1, 1),
                ("u5", "u4",  "u6", 1, 1922, 1, None),
                ("u6", "u5",  None, 1, 1922, 1, 1),
            ],
        )
        conn.commit()
        return conn

    def test_min_arc_length_1_counts_all_arcs(self):
        from src.data import _build_starter_sql

        conn = self._conn_with_singles_and_conversation()
        sql = _build_starter_sql(max_gap=0, min_arc_length=1).format(chamber_clause="")
        total = sum(n for _, _, n in conn.execute(sql))
        self.assertEqual(total, 3)

    def test_min_arc_length_2_excludes_single_mentions(self):
        from src.data import _build_starter_sql

        conn = self._conn_with_singles_and_conversation()
        sql = _build_starter_sql(max_gap=0, min_arc_length=2).format(chamber_clause="")
        total = sum(n for _, _, n in conn.execute(sql))
        self.assertEqual(total, 1)  # only u3's arc has length >= 2

    def test_min_arc_length_2_with_gap_1(self):
        """A K, ~, K, ~, ~, K chain: u1 opens arc (u1,u3) under gap=1 (length 2)."""
        import sqlite3

        from src.data import _build_starter_sql

        conn = sqlite3.connect(":memory:")
        conn.executescript(
            """
            CREATE TABLE person (id INTEGER PRIMARY KEY, name TEXT, gender TEXT, party TEXT);
            CREATE TABLE utterance (
                id TEXT PRIMARY KEY, content TEXT, prev TEXT, next TEXT,
                person_id INTEGER, year INTEGER, date INTEGER, kammare INTEGER,
                kvinna_1 BOOLEAN, kvinna_2 BOOLEAN, kvinna_3 BOOLEAN
            );
            INSERT INTO person VALUES (1, 'A', 'man', 'S');
            """
        )
        conn.executemany(
            "INSERT INTO utterance (id, prev, next, person_id, year, kammare, kvinna_1) VALUES (?,?,?,?,?,?,?)",
            [
                ("u1", None, "u2", 1, 1922, 1, 1),
                ("u2", "u1", "u3", 1, 1922, 1, None),
                ("u3", "u2", "u4", 1, 1922, 1, 1),
                ("u4", "u3", "u5", 1, 1922, 1, None),
                ("u5", "u4", "u6", 1, 1922, 1, None),
                ("u6", "u5", None, 1, 1922, 1, 1),
            ],
        )
        conn.commit()
        # Under gap=1, min_arc_length=2:
        # - u1 opens arc (u1, u3) — length 2 under gap=1. Counts.
        # - u3 is a continuation of u1's arc, not a starter.
        # - u6 is a starter but its arc is length 1 (u4, u5 non-kvinna precede;
        #   nothing forward). Does not count.
        # Total: 1.
        sql = _build_starter_sql(max_gap=1, min_arc_length=2).format(chamber_clause="")
        total = sum(n for _, _, n in conn.execute(sql))
        self.assertEqual(total, 1)

    def test_min_arc_length_bad_value_raises(self):
        from src.data import _build_starter_sql

        with self.assertRaises(ValueError):
            _build_starter_sql(max_gap=0, min_arc_length=0)
        with self.assertRaises(ValueError):
            _build_starter_sql(max_gap=0, min_arc_length=3)


class SpeakerStartsChimesTest(unittest.TestCase):
    """Standardised per-speaker starter + chime-in aggregate."""

    @staticmethod
    def _speakers() -> pd.DataFrame:
        # Two men, one woman, spanning 1922-1923.
        return pd.DataFrame(
            [
                {"who": "a", "gender": "man", "year": 1922,
                 "utterance_count": 100, "k_all_utts": 40},
                {"who": "a", "gender": "man", "year": 1923,
                 "utterance_count": 50, "k_all_utts": 10},
                {"who": "b", "gender": "man", "year": 1922,
                 "utterance_count": 20, "k_all_utts": 5},
                {"who": "c", "gender": "woman", "year": 1923,
                 "utterance_count": 10, "k_all_utts": 8},
                # Unknown-gender row — should be dropped by default.
                {"who": "d", "gender": "unknown", "year": 1922,
                 "utterance_count": 30, "k_all_utts": 3},
            ]
        )

    @staticmethod
    def _starters() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"who": "a", "gender": "man", "year": 1922, "starters": 8},
                {"who": "a", "gender": "man", "year": 1923, "starters": 2},
                {"who": "c", "gender": "woman", "year": 1923, "starters": 3},
            ]
        )

    def test_default_filters_unknown_gender_and_computes_rates(self):
        from src.data import speaker_starts_chimes

        m = speaker_starts_chimes(
            speakers=self._speakers(),
            starters=self._starters(),
        )
        # a (man): 150 utts, 50 k_all, 10 starters, 40 chime, rates .067 / .267
        row_a = m.loc[(m["who"] == "a") & (m["gender"] == "man")].iloc[0]
        self.assertEqual(int(row_a["utterance_count"]), 150)
        self.assertEqual(int(row_a["starters"]), 10)
        self.assertEqual(int(row_a["chime_ins"]), 40)
        self.assertAlmostEqual(float(row_a["starter_rate"]), 10 / 150)
        self.assertAlmostEqual(float(row_a["chime_rate"]), 40 / 150)
        # Unknown-gender speaker dropped by default.
        self.assertNotIn("d", set(m["who"]))
        # b has no starters — chime_ins == k_all_utts.
        row_b = m.loc[m["who"] == "b"].iloc[0]
        self.assertEqual(int(row_b["starters"]), 0)
        self.assertEqual(int(row_b["chime_ins"]), 5)

    def test_year_range_filters_both_frames(self):
        from src.data import speaker_starts_chimes

        m = speaker_starts_chimes(
            speakers=self._speakers(),
            starters=self._starters(),
            year_range=(1923, 1923),
        )
        # 1923 only: a with 50/10/2 starters/8 chime, c with 10/8/3 starters/5 chime.
        self.assertEqual(set(m["who"]), {"a", "c"})
        row_a = m.loc[m["who"] == "a"].iloc[0]
        self.assertEqual(int(row_a["utterance_count"]), 50)
        self.assertEqual(int(row_a["starters"]), 2)

    def test_min_utterances_drops_thin_speakers(self):
        from src.data import speaker_starts_chimes

        m = speaker_starts_chimes(
            speakers=self._speakers(),
            starters=self._starters(),
            min_utterances=30,
        )
        # c has 10 utts total → dropped; b has 20 → also dropped; a (150) survives.
        self.assertIn("a", set(m["who"]))
        self.assertNotIn("c", set(m["who"]))
        self.assertNotIn("b", set(m["who"]))

    def test_known_gender_only_false_keeps_unknown(self):
        from src.data import speaker_starts_chimes

        m = speaker_starts_chimes(
            speakers=self._speakers(),
            starters=self._starters(),
            known_gender_only=False,
        )
        self.assertIn("d", set(m["who"]))

    def test_L_and_S_flow_through_when_reader_called(self):
        # Sensitivity study contract: L / S kwargs propagate to
        # read_speaker_starter_counts when the caller doesn't supply
        # a pre-loaded starters frame. Verified by monkeypatching the
        # reader and asserting it saw the expected values.
        from src import data as data_mod

        captured = {}

        def fake_reader(**kwargs):
            captured.update(kwargs)
            return pd.DataFrame(columns=["who", "gender", "year", "starters"])

        original = data_mod.read_speaker_starter_counts
        data_mod.read_speaker_starter_counts = fake_reader
        try:
            data_mod.speaker_starts_chimes(
                speakers=self._speakers(),
                starters=None,
                year_range=(1922, 1922),
                max_gap=2,
                min_arc_length=3,
            )
        finally:
            data_mod.read_speaker_starter_counts = original

        self.assertEqual(captured["max_gap"], 2)
        self.assertEqual(captured["min_arc_length"], 3)


if __name__ == "__main__":
    unittest.main()
