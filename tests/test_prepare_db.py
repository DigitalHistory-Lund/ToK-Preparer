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


class LoadAllTenureRows(unittest.TestCase):
    """load_all_tenure_rows yields (person_id, start_int, end_int, role, party_or_None)
    for every tenure row in persons.sqlite, INCLUDING rows where both party fields
    are empty. Backfill needs those to see the unpartied windows.
    """

    def _make_db(self, td):
        """Create a minimal persons.sqlite with all three required tables."""
        import sqlite3
        from pathlib import Path

        db = Path(td) / "persons.sqlite"
        with sqlite3.connect(db) as con:
            for tbl in (
                "processed_member_of_parliament",
                "processed_minister",
                "processed_speaker",
            ):
                con.execute(
                    f"CREATE TABLE {tbl} ("
                    "person_id TEXT, start TEXT, end TEXT, role TEXT, "
                    "party TEXT, party_abbrev TEXT)"
                )
        return db

    def test_yields_unpartied_rows(self):
        import sqlite3
        import tempfile
        from src.prepare_db import load_all_tenure_rows

        with tempfile.TemporaryDirectory() as td:
            db = self._make_db(td)
            with sqlite3.connect(db) as con:
                con.executemany(
                    "INSERT INTO processed_member_of_parliament VALUES (?,?,?,?,?,?)",
                    [
                        ("p1", "1905-01-16", "1910-06-11", "ledamot",
                         "Högerns riksdagsgrupp", None),
                        ("p1", "1912-01-15", "1935-06-18", "ledamot", None, None),
                    ],
                )

            rows = sorted(load_all_tenure_rows(db))
            self.assertEqual(len(rows), 2)
            # partied row
            self.assertEqual(
                rows[0],
                ("p1", 19050116, 19100611, "ledamot", "Högerns riksdagsgrupp"),
            )
            # unpartied row — present, with party=None
            self.assertEqual(
                rows[1],
                ("p1", 19120115, 19350618, "ledamot", None),
            )

    def test_party_abbrev_fallback_and_empty_collapse(self):
        """party_abbrev is used when party is None/empty; both collapse to None."""
        import sqlite3
        import tempfile
        from src.prepare_db import load_all_tenure_rows

        with tempfile.TemporaryDirectory() as td:
            db = self._make_db(td)
            with sqlite3.connect(db) as con:
                con.executemany(
                    "INSERT INTO processed_member_of_parliament VALUES (?,?,?,?,?,?)",
                    [
                        # party_abbrev fallback: party is None, abbrev supplies value
                        ("p2", "1950-01-01", "1960-12-31", "ledamot", None, "S"),
                        # empty-string collapse: both empty strings → None
                        ("p3", "1970-01-01", "1980-12-31", "ledamot", "", ""),
                    ],
                )

            rows = {r[0]: r for r in load_all_tenure_rows(db)}
            # party_abbrev used as fallback
            self.assertEqual(
                rows["p2"],
                ("p2", 19500101, 19601231, "ledamot", "S"),
            )
            # empty strings collapse to None, not ''
            self.assertEqual(
                rows["p3"],
                ("p3", 19700101, 19801231, "ledamot", None),
            )
            self.assertIsNone(rows["p3"][4], "empty strings must collapse to None, not ''")


class CreateDatabasePopulatesBackfilledAffiliations(unittest.TestCase):
    """create_database() should insert direct AND backfill rows into the
    affiliation table, so the downstream utterance-join can find a party
    for unpartied same-role tenure windows within 5y of a partied neighbor.
    """

    def test_bergqvist_shape_gets_backfill_row(self):
        import sqlite3
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as td:
            persons_db = Path(td) / "persons.sqlite"
            tmp_db_path = Path(td) / "tmp.sqlite3"
            empty_overrides = Path(td) / "overrides.csv"
            empty_overrides.write_text(
                "person_id,name,start_date,end_date,party,confidence,source_note\n"
            )
            with sqlite3.connect(persons_db) as con:
                for tbl in ("processed_member_of_parliament",
                            "processed_minister", "processed_speaker"):
                    con.execute(
                        f"CREATE TABLE {tbl} (person_id TEXT, start TEXT, end TEXT, "
                        "role TEXT, party TEXT, party_abbrev TEXT)"
                    )
                con.executemany(
                    "INSERT INTO processed_member_of_parliament VALUES (?,?,?,?,?,?)",
                    [
                        ("bergqvist", "1912-01-15", "1921-06-21", "ledamot",
                         "Högerns riksdagsgrupp", None),
                        ("bergqvist", "1922-01-10", "1938-06-16", "ledamot", None, None),
                    ],
                )

            with (
                patch("src.prepare_db.data_dir", Path(td)),
                patch("src.prepare_db.tmp_db", tmp_db_path),
                patch("src.prepare_db.manual_party_overrides_csv", empty_overrides),
            ):
                from src.prepare_db import create_database
                create_database()

            with sqlite3.connect(tmp_db_path) as tcon:
                rows = tcon.execute(
                    "SELECT who, start, end, party FROM affiliation ORDER BY start"
                ).fetchall()
            self.assertEqual(rows, [
                ("bergqvist", 19120115, 19210621, "Högerns riksdagsgrupp"),
                ("bergqvist", 19220110, 19380616, "Högerns riksdagsgrupp"),
            ])


class CreateDatabaseAppliesManualPartyOverrides(unittest.TestCase):
    """create_database() should also insert rows from the manual override
    CSV, so bucket-C parliamentarians (present in persons.sqlite but
    party-less there) get an affiliation attribution from the curated
    biographical source.
    """

    def test_branting_shape_gets_manual_override_row(self):
        import sqlite3
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as td:
            persons_db = Path(td) / "persons.sqlite"
            tmp_db_path = Path(td) / "tmp.sqlite3"
            override_csv = Path(td) / "overrides.csv"

            # Persons.sqlite: Branting-shaped — MP row with NULL party.
            with sqlite3.connect(persons_db) as con:
                for tbl in ("processed_member_of_parliament",
                            "processed_minister", "processed_speaker"):
                    con.execute(
                        f"CREATE TABLE {tbl} (person_id TEXT, start TEXT, end TEXT, "
                        "role TEXT, party TEXT, party_abbrev TEXT)"
                    )
                con.execute(
                    "INSERT INTO processed_member_of_parliament VALUES (?,?,?,?,?,?)",
                    ("branting", "1897-01-15", "1925-01-24", "ledamot", None, None),
                )

            # Manual override CSV: attribute Branting to S for his whole tenure.
            override_csv.write_text(
                "person_id,name,start_date,end_date,party,confidence,source_note\n"
                "branting,hjalmar branting,1897-01-15,1925-01-24,Socialdemokraterna,high,test\n"
            )

            with (
                patch("src.prepare_db.data_dir", Path(td)),
                patch("src.prepare_db.tmp_db", tmp_db_path),
                patch("src.prepare_db.manual_party_overrides_csv", override_csv),
            ):
                from src.prepare_db import create_database
                create_database()

            with sqlite3.connect(tmp_db_path) as tcon:
                rows = tcon.execute(
                    "SELECT who, start, end, party FROM affiliation ORDER BY start"
                ).fetchall()
            self.assertEqual(rows, [
                ("branting", 18970115, 19250124, "Socialdemokraterna"),
            ])


class LoadManualPartyOverrides(unittest.TestCase):
    """load_manual_party_overrides parses the project-authored CSV and
    yields (person_id, start_int, end_int, party) tuples. Comment lines
    and blank lines are ignored.
    """

    def _write(self, td, body):
        from pathlib import Path
        p = Path(td) / "overrides.csv"
        p.write_text(body)
        return p

    def test_parses_rows_and_ignores_comments_and_blank_lines(self):
        import tempfile
        from src.prepare_db import load_manual_party_overrides

        body = (
            "# leading comment\n"
            "\n"
            "person_id,name,start_date,end_date,party,confidence,source_note\n"
            "# mid-file comment\n"
            "i-abc,alice,1910-01-15,1920-06-18,Socialdemokraterna,high,note1\n"
            "\n"
            "i-def,bob,1925-01-10,1935-12-31,Folkpartiet,medium,\n"
        )
        with tempfile.TemporaryDirectory() as td:
            p = self._write(td, body)
            out = list(load_manual_party_overrides(csv_path=p))
        self.assertEqual(out, [
            ("i-abc", 19100115, 19200618, "Socialdemokraterna"),
            ("i-def", 19250110, 19351231, "Folkpartiet"),
        ])

    def test_shipped_csv_loads_and_covers_expected_persons(self):
        from src.prepare_db import load_manual_party_overrides
        rows = list(load_manual_party_overrides())
        persons = {pid for pid, _, _, _ in rows}
        # 25 curated bucket-C persons; row count is higher due to per-era splits.
        self.assertEqual(len(persons), 25)
        self.assertGreaterEqual(len(rows), 25)
        # Spot-check the flagship case: Branting should be covered by a single
        # Socialdemokraterna span covering his 1897-1925 MP tenure.
        branting = [r for r in rows if r[0] == "i-VM3kn4h4mTkLoFpEyrhy5t"]
        self.assertEqual(len(branting), 1)
        _, s, e, party = branting[0]
        self.assertEqual(party, "Socialdemokraterna")
        self.assertLessEqual(s, 19000101)
        self.assertGreaterEqual(e, 19240101)


if __name__ == "__main__":
    unittest.main()
