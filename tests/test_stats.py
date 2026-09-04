"""Tests for src.stats."""

import unittest

from src.stats import (
    BinomialStarterTest,
    _binom_tail,
    bloc_starter_binomial_test,
    bloc_starter_scan,
    starter_binomial_test,
)


class BinomTailTest(unittest.TestCase):
    """Sanity checks against known binomial values."""

    def test_p_x_geq_5_n_10_p_half(self):
        # P(X >= 5 | n=10, p=0.5) = 638/1024 ≈ 0.6230
        self.assertAlmostEqual(_binom_tail(5, 10, 0.5, "greater"), 0.6230, places=3)

    def test_p_x_geq_8_n_10_p_half(self):
        # P(X >= 8 | n=10, p=0.5) = 56/1024 ≈ 0.0547
        self.assertAlmostEqual(_binom_tail(8, 10, 0.5, "greater"), 0.0547, places=3)

    def test_p_x_leq_2_n_10_p_half_by_symmetry(self):
        # P(X <= 2 | n=10, p=0.5) = P(X >= 8) by symmetry
        self.assertAlmostEqual(_binom_tail(2, 10, 0.5, "less"), 0.0547, places=3)

    def test_greater_at_k_0_is_1(self):
        self.assertAlmostEqual(_binom_tail(0, 10, 0.3, "greater"), 1.0)

    def test_less_at_k_n_is_1(self):
        self.assertAlmostEqual(_binom_tail(10, 10, 0.3, "less"), 1.0)

    def test_large_n_no_overflow(self):
        # Regression: naive binomial(1130, ...) overflows without log-space.
        v = _binom_tail(108, 1130, 0.055, "greater")
        self.assertGreater(v, 0.0)
        self.assertLess(v, 1e-5)

    def test_p_zero_degenerate(self):
        self.assertEqual(_binom_tail(0, 5, 0.0, "greater"), 1.0)
        self.assertEqual(_binom_tail(1, 5, 0.0, "greater"), 0.0)

    def test_bad_alternative_raises(self):
        with self.assertRaises(ValueError):
            _binom_tail(1, 5, 0.5, "two-sided")


class StarterBinomialTest(unittest.TestCase):
    """Smoke + directional tests against the shipped data."""

    def test_pooled_returns_binomial_starter_test_dataclass(self):
        r = starter_binomial_test()  # default = 1921 pooled
        self.assertIsInstance(r, BinomialStarterTest)
        self.assertEqual(r.alternative, "greater")

    def test_pooled_post_suffrage_highly_significant(self):
        r = starter_binomial_test()
        # Threshold is 0.01 (not 1e-6) because the default is now
        # max_gap=1, which admits fewer starters than max_gap=0 and so
        # yields a less extreme p (~0.002 vs ~1e-16). The effect
        # direction is the substantive claim; the exact p is
        # parameter-sensitive.
        self.assertLess(r.p_value, 0.01)
        self.assertGreater(r.observed_share, r.null_p)
        self.assertEqual(r.target, "woman")

    def test_utterance_share_null_gives_smaller_p_than_woman_talk_share(self):
        # Utterance-share null is much stricter (women speak a tiny fraction
        # of total speech), so it gives a smaller p-value than the
        # woman-talk-share null (which controls for engagement).
        r_utt = starter_binomial_test(null="utterance_share")
        r_woman = starter_binomial_test(null="woman_talk_share")
        self.assertLess(r_utt.p_value, r_woman.p_value)

    def test_single_year_1924_significant_under_woman_talk_null(self):
        r = starter_binomial_test(years=1924)
        self.assertLess(r.p_value, 0.001)

    def test_single_year_1922_not_significant(self):
        # 1922 has only 1 woman starter — no signal above the null.
        r = starter_binomial_test(years=1922)
        self.assertGreater(r.p_value, 0.1)

    def test_less_alternative_is_complementary(self):
        r_greater = starter_binomial_test(years=1924, alternative="greater")
        r_less = starter_binomial_test(years=1924, alternative="less")
        # For k > n*p, "less" should be close to 1.
        self.assertGreater(r_less.p_value, 0.99)
        self.assertLess(r_greater.p_value, 0.01)

    def test_empty_years_raises(self):
        with self.assertRaises(ValueError):
            starter_binomial_test(years=[])

    def test_unknown_null_raises(self):
        with self.assertRaises(ValueError):
            starter_binomial_test(null="bogus")  # type: ignore[arg-type]


class BlocStarterBinomialTest(unittest.TestCase):
    """Per-bloc party test, mirroring the gender variant."""

    def test_returns_dataclass_for_known_bloc(self):
        r = bloc_starter_binomial_test("S")
        self.assertIsInstance(r, BinomialStarterTest)
        self.assertEqual(r.target, "S")
        self.assertGreater(r.n_starters, 0)

    def test_unknown_bloc_raises(self):
        with self.assertRaises(ValueError):
            bloc_starter_binomial_test("BOGUS")

    def test_scan_covers_all_blocs_in_default_mapping(self):
        table = bloc_starter_scan()
        # Default mapping has 5 blocs.
        self.assertEqual(sorted(table["bloc"].tolist()), ["Bf", "H", "K", "L", "S"])

    def test_scan_sorted_by_p_value_ascending(self):
        table = bloc_starter_scan()
        pvals = table["p_value"].tolist()
        self.assertEqual(pvals, sorted(pvals))

    def test_two_nulls_give_different_p_for_at_least_one_bloc(self):
        # Unlike gender (where the utterance-share null is uniformly
        # stricter), a bloc's engagement rate can go either way. Just
        # verify the two nulls aren't giving identical answers.
        got_difference = False
        for bloc in ("S", "H", "L", "Bf", "K"):
            r_utt = bloc_starter_binomial_test(bloc, null="utterance_share")
            r_woman = bloc_starter_binomial_test(bloc, null="woman_talk_share")
            if abs(r_utt.p_value - r_woman.p_value) > 1e-6:
                got_difference = True
                break
        self.assertTrue(got_difference)


if __name__ == "__main__":
    unittest.main()
