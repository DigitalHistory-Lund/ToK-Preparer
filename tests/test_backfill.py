"""Tests for src.backfill.compute_backfill_affiliations."""

import unittest


# Convenience: each tenure row is (person_id, start_int, end_int, role, party_or_None)
def _row(pid, start_year, end_year, role, party):
    return (pid, int(f"{start_year:04d}0115"), int(f"{end_year:04d}0618"), role, party)


class ComputeBackfillAffiliations(unittest.TestCase):
    """Emits synthetic (person_id, start_int, end_int, party) tuples for
    unpartied tenure windows that have a same-role partied neighbor within 5 years.
    """

    def test_lindman_shape_carries_hoger_across_gap(self):
        """Multi-window partied history with unambiguous mode → backfills the mode."""
        from src.backfill import compute_backfill_affiliations

        rows = [
            _row("lindman", 1905, 1910, "ledamot", "Högerns riksdagsgrupp"),
            _row("lindman", 1905, 1910, "ledamot",
                 "Lantmanna- och borgarepartiet inom andrakammaren"),
            _row("lindman", 1911, 1911, "ledamot", "Högerns riksdagsgrupp"),
            _row("lindman", 1912, 1935, "ledamot", None),  # the gap
        ]
        out = list(compute_backfill_affiliations(rows))
        self.assertEqual(len(out), 1)
        pid, s, e, party = out[0]
        self.assertEqual(pid, "lindman")
        self.assertEqual(s, int("19120115"))
        self.assertEqual(e, int("19350618"))
        self.assertEqual(party, "Högerns riksdagsgrupp")

    def test_bergqvist_shape_single_partied_window_single_label(self):
        """One partied window with one label → backfill adjacent unpartied window."""
        from src.backfill import compute_backfill_affiliations

        rows = [
            _row("bergqvist", 1912, 1921, "ledamot", "Högerns riksdagsgrupp"),
            _row("bergqvist", 1922, 1938, "ledamot", None),
        ]
        out = list(compute_backfill_affiliations(rows))
        self.assertEqual(len(out), 1)
        pid, s, e, party = out[0]
        self.assertEqual(pid, "bergqvist")
        self.assertEqual(party, "Högerns riksdagsgrupp")

    def test_lindhagen_shape_ambiguous_single_window_is_skipped(self):
        """Single partied window with multiple competing labels → SKIP.
        Backfilling any one of {Socialdemokraterna, Socialdemokratiska
        vänstergruppen, Utan partibeteckning} would be a coin flip; better
        to leave the utterances blank than to guess.
        """
        from src.backfill import compute_backfill_affiliations

        rows = [
            _row("lindhagen", 1897, 1917, "ledamot", "Socialdemokraterna"),
            _row("lindhagen", 1897, 1917, "ledamot", "Socialdemokratiska vänstergruppen"),
            _row("lindhagen", 1897, 1917, "ledamot", "Utan partibeteckning"),
            _row("lindhagen", 1919, 1940, "ledamot", None),
        ]
        out = list(compute_backfill_affiliations(rows))
        self.assertEqual(out, [])

    def test_trygger_shape_mode_across_windows_wins(self):
        """Two partied windows with overlapping but non-identical label sets.
        Höger appears in both; the tiebreak-by-count picks it. Also verifies
        same-role restriction: the minister-role window contributes to the
        mode but doesn't itself gate the ledamot backfill.
        """
        from src.backfill import compute_backfill_affiliations

        rows = [
            _row("trygger", 1898, 1911, "ledamot", "Första kammarens nationella parti"),
            _row("trygger", 1898, 1911, "ledamot", "Högerns riksdagsgrupp"),
            _row("trygger", 1898, 1911, "ledamot", "Utan partibeteckning"),
            _row("trygger", 1912, 1937, "ledamot", None),  # long gap, but ≤5y from 1911
            _row("trygger", 1928, 1930, "utrikesminister", "Högerns riksdagsgrupp"),
        ]
        out = list(compute_backfill_affiliations(rows))
        # only the ledamot unpartied window should be backfilled
        self.assertEqual(len(out), 1)
        pid, s, e, party = out[0]
        self.assertEqual(pid, "trygger")
        # candidate set from the 1898-1911 ledamot neighbor:
        #   {Första kammarens nationella parti, Högerns riksdagsgrupp,
        #    Utan partibeteckning}
        # counts across career (per window):
        #   Höger=2 (1898-1911 ledamot + 1928-1930 minister)
        #   Första kammarens=1, UP=1
        # Höger wins.
        self.assertEqual(party, "Högerns riksdagsgrupp")

    def test_both_sides_same_party_backfills(self):
        """Unpartied window sandwiched between two partied windows of the
        same role, both agreeing on the party. Backfill should choose that
        party."""
        from src.backfill import compute_backfill_affiliations

        rows = [
            _row("bothsame", 1910, 1913, "ledamot", "Folkpartiet"),
            _row("bothsame", 1914, 1917, "ledamot", None),  # gap
            _row("bothsame", 1918, 1921, "ledamot", "Folkpartiet"),
        ]
        out = list(compute_backfill_affiliations(rows))
        self.assertEqual(len(out), 1)
        _, _, _, party = out[0]
        self.assertEqual(party, "Folkpartiet")

    def test_gap_within_15_years_is_backfilled(self):
        """Same role, partied neighbor within 15 years → backfill.
        Regression: this case was skipped under the original ±5y window."""
        from src.backfill import compute_backfill_affiliations

        rows = [
            _row("nearenough", 1900, 1905, "ledamot", "Högerns riksdagsgrupp"),
            _row("nearenough", 1912, 1920, "ledamot", None),  # 7y gap; ≤15 → backfill
        ]
        out = list(compute_backfill_affiliations(rows))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][3], "Högerns riksdagsgrupp")

    def test_gap_over_15_years_is_skipped(self):
        """Same role, partied neighbor >15 years away → skip.
        Pins the current tier boundary at 15 years."""
        from src.backfill import compute_backfill_affiliations

        rows = [
            _row("faraway", 1900, 1905, "ledamot", "Högerns riksdagsgrupp"),
            _row("faraway", 1922, 1930, "ledamot", None),  # 17-year gap
        ]
        self.assertEqual(list(compute_backfill_affiliations(rows)), [])

    def test_different_role_neighbor_is_skipped(self):
        """A minister-role partied neighbor does NOT trigger backfill of a
        ledamot-role unpartied window, even at zero gap. Different roles
        are Tier C territory."""
        from src.backfill import compute_backfill_affiliations

        rows = [
            _row("crossrole", 1920, 1921, "ecklesiastikminister", "Bondeförbundet"),
            _row("crossrole", 1922, 1925, "ledamot", None),
        ]
        # party appears in the person's history so it *counts* for the mode,
        # but there's no same-role neighbor, so no candidates → skip.
        self.assertEqual(list(compute_backfill_affiliations(rows)), [])

    def test_person_with_no_partied_history_is_skipped(self):
        """Bucket C: person has zero partied rows anywhere → nothing to carry."""
        from src.backfill import compute_backfill_affiliations

        rows = [
            _row("branting", 1897, 1925, "ledamot", None),
            _row("branting", 1920, 1923, "statsminister", None),
        ]
        self.assertEqual(list(compute_backfill_affiliations(rows)), [])

    def test_touching_neighbor_backfills(self):
        """Partied neighbor's end date equals unpartied's start date (or
        vice versa) → treat as adjacent (gap=0y) and backfill. The direct
        join covers only the shared boundary day; the rest of the
        unpartied window would otherwise silently get no party.
        """
        from src.backfill import compute_backfill_affiliations

        # pe == s: partied ends 1921-12-31, unpartied starts same day
        touching_before = [
            (
                "touchBefore",
                int("19120115"), int("19211231"),
                "ledamot", "Högerns riksdagsgrupp",
            ),
            (
                "touchBefore",
                int("19211231"), int("19380616"),
                "ledamot", None,
            ),
        ]
        out = list(compute_backfill_affiliations(touching_before))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][3], "Högerns riksdagsgrupp")

        # ps == e: unpartied ends 1910-01-01, partied starts same day
        touching_after = [
            (
                "touchAfter",
                int("19000115"), int("19100101"),
                "ledamot", None,
            ),
            (
                "touchAfter",
                int("19100101"), int("19200618"),
                "ledamot", "Socialdemokraterna",
            ),
        ]
        out = list(compute_backfill_affiliations(touching_after))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][3], "Socialdemokraterna")

    def test_multiple_persons_are_independent(self):
        """Sanity: two people in the same input are processed independently."""
        from src.backfill import compute_backfill_affiliations

        rows = [
            _row("a", 1900, 1905, "ledamot", "S"),
            _row("a", 1906, 1910, "ledamot", None),
            _row("b", 1900, 1905, "ledamot", "H"),
            _row("b", 1906, 1910, "ledamot", None),
        ]
        out = sorted(compute_backfill_affiliations(rows))
        self.assertEqual([(pid, party) for pid, _, _, party in out],
                         [("a", "S"), ("b", "H")])
