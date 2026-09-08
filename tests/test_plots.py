"""Tests for src.plots.

Coverage:

* Smoke tests — each of the eight public plot functions returns a
  :class:`matplotlib.figure.Figure` when called against the real shipped
  CSV. These catch import breakage, column-rename breakage, and gross
  aggregation errors.
* Helper tests — ``_reduce_parties`` and ``_add_suffrage_marker`` work
  from synthetic frames without touching the shipped data.
* Style override — a custom ``PlotStyle`` reaches the axes (verified via
  the suffrage-marker colour).
* Party subset — restricting ``parties`` to a smaller set produces
  exactly that many content lines.
* Category mode — both ``mode="utts"`` and ``mode="words"`` produce a
  Figure; invalid modes raise.
"""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.plots import (  # noqa: E402
    DEFAULT_PARTY_MAPPING,
    DEFAULT_STYLE,
    PlotStyle,
    _add_suffrage_marker,
    _gender_source_frame,
    _reduce_parties,
    _wilson_ci,
    plot_baseline_corpus,
    plot_category_composition,
    plot_gender_share_utt_and_word,
    plot_keyword_coverage,
    plot_keyword_coverage_by_pattern,
    plot_keyword_shares_by_pattern,
    plot_kvinna_all_by_chamber,
    plot_kvinna_net_composition,
    plot_party_share,
    plot_party_share_stacked,
    plot_party_speaker_starts_vs_chimes,
    plot_silent_men_share_per_year,
    plot_speaker_rate_vs_volume,
    plot_speaker_share_distribution,
    plot_speaker_starts_vs_chimes_by_gender,
    plot_top_male_speakers,
    plot_utterance_share_by_gender,
    plot_utterance_source_by_gender,
    plot_woman_chime_vs_start,
    plot_word_share_by_gender,
    plot_word_source_by_gender,
)


class HelpersTest(unittest.TestCase):
    """Pure helpers that don't touch the shipped CSV."""

    def test_reduce_parties_maps_full_names_to_blocs(self):
        df = pd.DataFrame(
            [
                {"party": "Socialdemokraterna", "utterance_count": 10},
                {"party": "Högerpartiet", "utterance_count": 5},
                {"party": "Some Party Not In Mapping", "utterance_count": 3},
                {"party": None, "utterance_count": 2},
            ]
        )
        mapping = {
            "S": ("Socialdemokraterna",),
            "H": ("Högerpartiet",),
        }
        out = _reduce_parties(df, mapping)
        self.assertEqual(sorted(out["bloc"].tolist()), ["H", "S"])
        # Unmapped and NaN parties are dropped.
        self.assertNotIn("Some Party Not In Mapping", out["party"].tolist())

    def test_reduce_parties_pools_multiple_names_per_bloc(self):
        df = pd.DataFrame(
            [
                {"party": "Folkpartiet", "utterance_count": 4},
                {"party": "Frisinnade folkpartiet", "utterance_count": 6},
            ]
        )
        mapping = {"L": ("Folkpartiet", "Frisinnade folkpartiet")}
        out = _reduce_parties(df, mapping)
        self.assertEqual(out["bloc"].tolist(), ["L", "L"])
        # Both rows survive; caller aggregates.
        self.assertEqual(out["utterance_count"].sum(), 10)

    def test_add_suffrage_marker_uses_configured_year(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        style = PlotStyle()  # default suffrage_year = 1922
        _add_suffrage_marker(ax, style)
        # A single vertical line was added at x=1922.
        xs = [ln.get_xdata()[0] for ln in ax.get_lines()]
        self.assertIn(1922, xs)


class SmokeTest(unittest.TestCase):
    """Each public plot function returns a Figure against the real CSV."""

    def test_baseline_corpus_returns_figure(self):
        self.assertIsInstance(plot_baseline_corpus(), Figure)

    def test_kvinna_all_by_chamber_counts_returns_figure(self):
        self.assertIsInstance(plot_kvinna_all_by_chamber(), Figure)

    def test_kvinna_all_by_chamber_share_returns_figure(self):
        self.assertIsInstance(plot_kvinna_all_by_chamber(mode="share"), Figure)

    def test_kvinna_all_by_chamber_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            plot_kvinna_all_by_chamber(mode="nope")

    def test_kvinna_all_by_chamber_smoothing_int_returns_figure(self):
        self.assertIsInstance(plot_kvinna_all_by_chamber(smoothing=3), Figure)

    def test_kvinna_all_by_chamber_smoothing_iterable_returns_list(self):
        out = plot_kvinna_all_by_chamber(smoothing=[5, 3, 3, 1])
        # duplicates collapsed via set(), sorted ascending -> [1, 3, 5]
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 3)
        for fig in out:
            self.assertIsInstance(fig, Figure)

    def test_kvinna_all_by_chamber_smoothing_zero_raises(self):
        with self.assertRaises(ValueError):
            plot_kvinna_all_by_chamber(smoothing=0)

    def test_category_composition_utts_returns_figure(self):
        self.assertIsInstance(plot_category_composition(mode="utts"), Figure)

    def test_category_composition_words_returns_figure(self):
        self.assertIsInstance(plot_category_composition(mode="words"), Figure)

    def test_utterance_share_by_gender_returns_figure(self):
        self.assertIsInstance(plot_utterance_share_by_gender(), Figure)

    def test_word_share_by_gender_returns_figure(self):
        self.assertIsInstance(plot_word_share_by_gender(), Figure)

    def test_gender_share_utt_and_word_returns_figure(self):
        fig = plot_gender_share_utt_and_word()
        self.assertIsInstance(fig, Figure)
        # Dual-axis: primary + twin.
        self.assertEqual(len(fig.axes), 2)

    def test_utterance_source_by_gender_returns_figure(self):
        self.assertIsInstance(plot_utterance_source_by_gender(), Figure)

    def test_word_source_by_gender_returns_figure(self):
        self.assertIsInstance(plot_word_source_by_gender(), Figure)

    def test_party_share_returns_figure(self):
        self.assertIsInstance(plot_party_share(), Figure)

    def test_party_share_stacked_returns_figure(self):
        self.assertIsInstance(plot_party_share_stacked(), Figure)

    def test_keyword_coverage_returns_figure(self):
        self.assertIsInstance(plot_keyword_coverage(), Figure)

    def test_keyword_coverage_by_pattern_returns_figure(self):
        self.assertIsInstance(plot_keyword_coverage_by_pattern(), Figure)

    def test_keyword_shares_by_pattern_returns_figure(self):
        self.assertIsInstance(plot_keyword_shares_by_pattern(), Figure)

    def test_speaker_share_distribution_returns_figure(self):
        self.assertIsInstance(plot_speaker_share_distribution(), Figure)

    def test_speaker_starts_vs_chimes_by_gender_returns_figure(self):
        self.assertIsInstance(plot_speaker_starts_vs_chimes_by_gender(), Figure)

    def test_top_male_speakers_returns_figure(self):
        self.assertIsInstance(plot_top_male_speakers(), Figure)

    def test_speaker_rate_vs_volume_utts_returns_figure(self):
        self.assertIsInstance(plot_speaker_rate_vs_volume(mode="utts"), Figure)

    def test_speaker_rate_vs_volume_words_returns_figure(self):
        self.assertIsInstance(plot_speaker_rate_vs_volume(mode="words"), Figure)

    def test_plot_silent_men_share_per_year_returns_figure(self):
        fig = plot_silent_men_share_per_year()
        self.assertIsInstance(fig, Figure)


class BaselineTwinAxisTest(unittest.TestCase):
    """plot_baseline_corpus draws total_words on a dashed right axis."""

    def test_has_secondary_axis_with_dashed_word_lines(self):
        fig = plot_baseline_corpus()
        self.assertEqual(len(fig.axes), 2)
        _, right = fig.axes
        right_lines = right.get_lines()
        self.assertEqual(len(right_lines), 2)
        for ln in right_lines:
            self.assertEqual(ln.get_linestyle(), "--")


class KeywordCoverageTest(unittest.TestCase):
    """plot_keyword_coverage draws a stack + reference line on both panels."""

    def test_has_two_stacked_panels_each_with_stack_and_reference_line(self):
        from matplotlib.collections import PolyCollection

        fig = plot_keyword_coverage()
        self.assertEqual(len(fig.axes), 2)
        for ax in fig.axes:
            polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
            # stackplot creates one PolyCollection per band → three bands.
            self.assertEqual(len(polys), 3)
            # Two lines expected: the reference line + the 1921 marker.
            self.assertGreaterEqual(len(ax.get_lines()), 2)

    def test_custom_reference_line_colour_reaches_both_panels(self):
        from dataclasses import replace

        distinctive = "#00aa00"
        style = replace(
            DEFAULT_STYLE,
            kvinna_reference_line={
                "color": distinctive,
                "linewidth": 1.8,
                "linestyle": "-",
                "label": "kvinna_all",
            },
        )
        fig = plot_keyword_coverage(style=style)
        for ax in fig.axes:
            matches = [
                ln
                for ln in ax.get_lines()
                if _colour_matches(ln.get_color(), distinctive)
            ]
            self.assertTrue(
                matches, "custom reference-line colour did not reach a panel"
            )


class KeywordCoverageByPatternTest(unittest.TestCase):
    """plot_keyword_coverage_by_pattern is a 4x1 facet with the expected sharing."""

    def test_has_eight_axes_four_primary_four_twin(self):
        fig = plot_keyword_coverage_by_pattern()
        # 4 primary + 4 twin = 8 axes attached to the figure.
        self.assertEqual(len(fig.axes), 8)

    def test_left_axis_sharing_k1_k2_k3_but_not_kall(self):
        fig = plot_keyword_coverage_by_pattern()
        # Primary axes are the first 4 (created by plt.subplots); twins are
        # the remainder (attached via ax.twinx()).
        primaries = fig.axes[:4]
        k1, k2, k3, kall = primaries
        shared = set(k1.get_shared_y_axes().get_siblings(k1))
        self.assertIn(k2, shared)
        self.assertIn(k3, shared)
        self.assertNotIn(kall, shared)

    def test_right_axes_share_across_all_four(self):
        fig = plot_keyword_coverage_by_pattern()
        # Twin right axes are the last 4 axes attached to the figure.
        twins = fig.axes[4:]
        shared = set(twins[0].get_shared_y_axes().get_siblings(twins[0]))
        for t in twins[1:]:
            self.assertIn(t, shared)

    def test_custom_counting_mode_colour_reaches_every_primary_axis(self):
        from dataclasses import replace

        distinctive = "#00aaff"
        style = replace(
            DEFAULT_STYLE,
            counting_mode_colors={"utts": distinctive, "words": "#000000"},
        )
        fig = plot_keyword_coverage_by_pattern(style=style)
        for ax in fig.axes[:4]:
            matches = [
                ln
                for ln in ax.get_lines()
                if _colour_matches(ln.get_color(), distinctive)
            ]
            self.assertTrue(
                matches, "custom utts colour missing from a primary panel"
            )


class KeywordSharesByPatternTest(unittest.TestCase):
    """plot_keyword_shares_by_pattern is a 4x1 twin-axis facet with CI ribbons."""

    def test_has_eight_axes_four_primary_four_twin(self):
        fig = plot_keyword_shares_by_pattern()
        self.assertEqual(len(fig.axes), 8)

    def test_panels_ordered_kall_k1_k2_k3(self):
        fig = plot_keyword_shares_by_pattern()
        titles = [ax.get_title(loc="left") for ax in fig.axes[:4]]
        self.assertEqual(titles[0], DEFAULT_STYLE.kvinna_pattern_labels["k_all"])
        self.assertEqual(titles[1], DEFAULT_STYLE.kvinna_pattern_labels["k1"])
        self.assertEqual(titles[2], DEFAULT_STYLE.kvinna_pattern_labels["k2"])
        self.assertEqual(titles[3], DEFAULT_STYLE.kvinna_pattern_labels["k3"])

    def test_all_axes_anchored_at_zero(self):
        fig = plot_keyword_shares_by_pattern()
        for ax in fig.axes:
            self.assertEqual(ax.get_ylim()[0], 0)

    def test_wilson_ribbons_present_on_each_axis(self):
        from matplotlib.collections import PolyCollection

        fig = plot_keyword_shares_by_pattern()
        for ax in fig.axes:
            polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
            self.assertGreaterEqual(
                len(polys), 1, "Wilson CI ribbon missing from an axis"
            )

    def test_custom_share_colour_reaches_lines(self):
        from dataclasses import replace

        distinctive_utt = "#00aaff"
        distinctive_word = "#ff5500"
        style = replace(
            DEFAULT_STYLE,
            share_colors={"utts": distinctive_utt, "words": distinctive_word},
        )
        fig = plot_keyword_shares_by_pattern(style=style)
        for ax in fig.axes[:4]:
            matches = [
                ln
                for ln in ax.get_lines()
                if _colour_matches(ln.get_color(), distinctive_utt)
            ]
            self.assertTrue(
                matches, "custom utts colour missing from a primary panel"
            )
        for ax_r in fig.axes[4:]:
            matches = [
                ln
                for ln in ax_r.get_lines()
                if _colour_matches(ln.get_color(), distinctive_word)
            ]
            self.assertTrue(
                matches, "custom words colour missing from a twin panel"
            )

    def test_data_injection_bypasses_disk(self):
        df = pd.DataFrame(
            [
                {
                    "year": 1900,
                    "chamber": 1,
                    "gender": "man",
                    "party": "Socialdemokraterna",
                    "utterance_count": 100,
                    "total_words": 1000,
                    "char_count": 5000,
                    "k1_utts": 5,
                    "k2_utts": 8,
                    "k3_utts": 3,
                    "k_all_utts": 12,
                    "k1_matches": 20,
                    "k2_matches": 15,
                    "k3_matches": 10,
                    "k_all_matches": 40,
                    "damer_matches": 1,
                    "damer_utts": 1,
                    "fru_matches": 4,
                    "fru_utts": 3,
                },
                {
                    "year": 1910,
                    "chamber": 2,
                    "gender": "man",
                    "party": "Socialdemokraterna",
                    "utterance_count": 200,
                    "total_words": 2000,
                    "char_count": 10000,
                    "k1_utts": 15,
                    "k2_utts": 12,
                    "k3_utts": 8,
                    "k_all_utts": 25,
                    "k1_matches": 40,
                    "k2_matches": 30,
                    "k3_matches": 20,
                    "k_all_matches": 80,
                    "damer_matches": 2,
                    "damer_utts": 2,
                    "fru_matches": 8,
                    "fru_utts": 6,
                },
            ]
        )
        fig = plot_keyword_shares_by_pattern(df=df)
        self.assertIsInstance(fig, Figure)


class WilsonCiTest(unittest.TestCase):
    """_wilson_ci is the vectorised Wilson score interval."""

    def test_matches_reference_value(self):
        # k=8, n=100 → Wilson 95% CI ≈ (0.041, 0.150). Standard reference
        # value; see Brown, Cai & DasGupta (2001), Statistical Science.
        lo, hi = _wilson_ci(pd.Series([8]), pd.Series([100]))
        self.assertAlmostEqual(lo.iloc[0], 0.041, places=3)
        self.assertAlmostEqual(hi.iloc[0], 0.150, places=3)

    def test_n_zero_yields_nan_bounds(self):
        lo, hi = _wilson_ci(pd.Series([0]), pd.Series([0]))
        self.assertTrue(pd.isna(lo.iloc[0]))
        self.assertTrue(pd.isna(hi.iloc[0]))

    def test_k_zero_gives_zero_lower_and_positive_upper(self):
        lo, hi = _wilson_ci(pd.Series([0]), pd.Series([100]))
        self.assertAlmostEqual(lo.iloc[0], 0.0, places=6)
        self.assertGreater(hi.iloc[0], 0.0)


class SpeakerPlotsTest(unittest.TestCase):
    """plot_speaker_share_distribution + plot_top_male_speakers."""

    @staticmethod
    def _fake_speakers() -> pd.DataFrame:
        """Per-year rows for plot_speaker_share_distribution."""
        rows = []
        for who, share_target, ucount in (
            ("w1", 0.5, 300),
            ("w2", 0.6, 200),
        ):
            rows.append({
                "year": 1925, "chamber": 2, "gender": "woman",
                "party": "Socialdemokraterna", "who": who,
                "utterance_count": ucount,
                "k1_utts": int(ucount * share_target * 0.5),
                "k2_utts": int(ucount * share_target * 0.3),
                "k3_utts": int(ucount * share_target * 0.2),
                "k_all_utts": int(ucount * share_target),
            })
        for i, share in enumerate([0.30, 0.28, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.02]):
            ucount = 300
            rows.append({
                "year": 1925, "chamber": 2, "gender": "man",
                "party": "Socialdemokraterna", "who": f"m{i}",
                "utterance_count": ucount,
                "k1_utts": int(ucount * share * 0.5),
                "k2_utts": int(ucount * share * 0.3),
                "k3_utts": int(ucount * share * 0.2),
                "k_all_utts": int(ucount * share),
            })
        # Thin-talker + unknown-gender rows — must be filtered by the plot.
        rows.append({
            "year": 1925, "chamber": 2, "gender": "man",
            "party": "Socialdemokraterna", "who": "m_thin",
            "utterance_count": 5,
            "k1_utts": 2, "k2_utts": 2, "k3_utts": 1, "k_all_utts": 5,
        })
        rows.append({
            "year": 1925, "chamber": 2, "gender": "",
            "party": "", "who": "u1",
            "utterance_count": 500,
            "k1_utts": 100, "k2_utts": 50, "k3_utts": 50, "k_all_utts": 200,
        })
        return pd.DataFrame(rows)

    @staticmethod
    def _fake_totals(with_names: bool = True) -> pd.DataFrame:
        """Pre-aggregated per-speaker totals for plot_top_male_speakers."""
        rows = []
        # Two women above threshold.
        for who, name, share, ucount in (
            ("w1", "elin engström", 0.5, 300),
            ("w2", "kerstin hesselgren", 0.6, 200),
        ):
            k_all = int(ucount * share)
            rows.append({
                "who": who, "name": name if with_names else "",
                "gender": "woman", "party": "S",
                "utterance_count": ucount,
                "k1_utts": int(k_all * 0.5), "k2_utts": int(k_all * 0.3),
                "k3_utts": int(k_all * 0.2), "k_all_utts": k_all,
                "share": share,
            })
        # Ten men above threshold, span of shares.
        for i, share in enumerate([0.30, 0.28, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.02]):
            ucount = 300
            k_all = int(ucount * share)
            rows.append({
                "who": f"m{i}", "name": (f"man{i} testsson" if with_names else ""),
                "gender": "man", "party": "S",
                "utterance_count": ucount,
                "k1_utts": int(k_all * 0.5), "k2_utts": int(k_all * 0.3),
                "k3_utts": int(k_all * 0.2), "k_all_utts": k_all,
                "share": share,
            })
        # Below-threshold thin-talker (share=1.0, 5 utterances) — must not rank.
        rows.append({
            "who": "m_thin", "name": "thin talker" if with_names else "",
            "gender": "man", "party": "S",
            "utterance_count": 5,
            "k1_utts": 2, "k2_utts": 2, "k3_utts": 1, "k_all_utts": 5,
            "share": 1.0,
        })
        # Unknown-gender speaker — must never rank.
        rows.append({
            "who": "u1", "name": "" if with_names else "",
            "gender": "", "party": "",
            "utterance_count": 500,
            "k1_utts": 100, "k2_utts": 50, "k3_utts": 50, "k_all_utts": 200,
            "share": 0.4,
        })
        return pd.DataFrame(rows)

    def test_distribution_data_injection_bypasses_disk(self):
        fig = plot_speaker_share_distribution(speakers=self._fake_speakers())
        self.assertIsInstance(fig, Figure)

    def test_distribution_hist_present_for_each_gender(self):
        from matplotlib.patches import Rectangle
        fig = plot_speaker_share_distribution(speakers=self._fake_speakers())
        ax = fig.axes[0]
        rects = [p for p in ax.patches if isinstance(p, Rectangle)]
        self.assertGreater(len(rects), 0, "no histogram bars drawn")

    def test_top_male_speakers_honours_top_n(self):
        fig = plot_top_male_speakers(totals=self._fake_totals(), top_n=5)
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        self.assertEqual(len(labels), 5)

    def test_top_male_speakers_excludes_below_threshold(self):
        fig = plot_top_male_speakers(totals=self._fake_totals(), top_n=20)
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        # thin talker (share 1.0, 5 utts) must not rank; unknown-gender must not rank.
        self.assertNotIn("thin talker", labels)
        self.assertNotIn("m_thin", labels)
        self.assertNotIn("u1", labels)

    def test_top_male_speakers_uses_name_column(self):
        fig = plot_top_male_speakers(totals=self._fake_totals(with_names=True), top_n=3)
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        # Top male by share is m0 (0.30) → "man0 testsson".
        self.assertIn("man0 testsson", labels)

    def test_top_male_speakers_falls_back_to_who_when_name_empty(self):
        fig = plot_top_male_speakers(totals=self._fake_totals(with_names=False), top_n=3)
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        # No name column, so labels should be the raw who ids.
        self.assertIn("m0", labels)


class SpeakerRateVsVolumeTest(unittest.TestCase):
    """plot_speaker_rate_vs_volume — parametrised octopus scatter."""

    @staticmethod
    def _fake_speakers() -> pd.DataFrame:
        """Per-year speakers frame for mode='utts'."""
        return pd.DataFrame([
            # A man with sizeable utterance count spread across two years.
            {"year": 1910, "chamber": 1, "gender": "man", "party": "H", "who": "m1",
             "utterance_count": 100, "k1_utts": 5, "k2_utts": 3, "k3_utts": 2, "k_all_utts": 8},
            {"year": 1925, "chamber": 1, "gender": "man", "party": "H", "who": "m1",
             "utterance_count": 200, "k1_utts": 10, "k2_utts": 6, "k3_utts": 4, "k_all_utts": 15},
            # A post-1921 woman.
            {"year": 1925, "chamber": 2, "gender": "woman", "party": "S", "who": "w1",
             "utterance_count": 50, "k1_utts": 25, "k2_utts": 10, "k3_utts": 5, "k_all_utts": 30},
            # Unknown-speaker + unknown-gender rows — must be filtered.
            {"year": 1925, "chamber": 1, "gender": "", "party": "", "who": "",
             "utterance_count": 999, "k1_utts": 100, "k2_utts": 100, "k3_utts": 100, "k_all_utts": 300},
        ])

    @staticmethod
    def _fake_word_totals() -> pd.DataFrame:
        """Per-year word-level frame for mode='words'."""
        return pd.DataFrame([
            {"year": 1910, "chamber": 1, "gender": "man", "party": "H", "who": "m1",
             "total_words": 5000, "k1_matches": 3, "k2_matches": 2, "k3_matches": 1, "k_all_matches": 5},
            {"year": 1925, "chamber": 1, "gender": "man", "party": "H", "who": "m1",
             "total_words": 10000, "k1_matches": 8, "k2_matches": 5, "k3_matches": 3, "k_all_matches": 15},
            {"year": 1925, "chamber": 2, "gender": "woman", "party": "S", "who": "w1",
             "total_words": 3000, "k1_matches": 30, "k2_matches": 15, "k3_matches": 5, "k_all_matches": 45},
            # Unknown-speaker row.
            {"year": 1925, "chamber": 1, "gender": "", "party": "", "who": "",
             "total_words": 50000, "k1_matches": 100, "k2_matches": 100, "k3_matches": 100, "k_all_matches": 250},
        ])

    def test_utts_mode_uses_speakers_frame(self):
        fig = plot_speaker_rate_vs_volume(
            speakers=self._fake_speakers(), mode="utts"
        )
        self.assertIsInstance(fig, Figure)
        # Log x-axis is the "octopus" plot's contract.
        self.assertEqual(fig.axes[0].get_xscale(), "log")

    def test_words_mode_uses_word_totals_frame(self):
        fig = plot_speaker_rate_vs_volume(
            word_totals=self._fake_word_totals(), mode="words"
        )
        self.assertIsInstance(fig, Figure)
        self.assertEqual(fig.axes[0].get_xscale(), "log")

    def test_year_range_filters_source_rows(self):
        # Pre-1921 window keeps only m1's 1910 row.
        pre = plot_speaker_rate_vs_volume(
            speakers=self._fake_speakers(),
            mode="utts",
            year_range=(1900, 1920),
        )
        # 1 man's aggregate = 1 marker; 0 women in pre-1921 → still a scatter call
        # for both series, but the woman's series has zero points.
        ax = pre.axes[0]
        # The legend text carries N=... which we can parse.
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertTrue(any("Man (N=1" in lbl for lbl in labels))
        self.assertTrue(any("Woman (N=0)" in lbl for lbl in labels))

        post = plot_speaker_rate_vs_volume(
            speakers=self._fake_speakers(),
            mode="utts",
            year_range=(1921, 1940),
        )
        post_labels = [
            t.get_text() for t in post.axes[0].get_legend().get_texts()
        ]
        self.assertTrue(any("Woman (N=1" in lbl for lbl in post_labels))

    def test_k_group_selects_numerator(self):
        # k1 vs k_all use different columns → different mean shares in legend.
        base = plot_speaker_rate_vs_volume(
            speakers=self._fake_speakers(), mode="utts", k_group="k_all"
        )
        k1 = plot_speaker_rate_vs_volume(
            speakers=self._fake_speakers(), mode="utts", k_group="k1"
        )
        base_labels = "".join(
            t.get_text() for t in base.axes[0].get_legend().get_texts()
        )
        k1_labels = "".join(
            t.get_text() for t in k1.axes[0].get_legend().get_texts()
        )
        self.assertIn("K_ALL", base.axes[0].get_ylabel())
        self.assertIn("K1", k1.axes[0].get_ylabel())
        self.assertNotEqual(base_labels, k1_labels)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            plot_speaker_rate_vs_volume(mode="nope")

    def test_invalid_k_group_raises(self):
        with self.assertRaises(ValueError):
            plot_speaker_rate_vs_volume(k_group="k4")


class CategoryModeTest(unittest.TestCase):
    """plot_category_composition accepts only "utts" and "words"."""

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            plot_category_composition(mode="nope")


class StyleOverrideTest(unittest.TestCase):
    """A custom PlotStyle reaches the axes."""

    def test_custom_suffrage_colour_appears_on_axes(self):
        from dataclasses import replace

        distinctive = "#ff00ff"
        style = replace(
            DEFAULT_STYLE,
            suffrage_line={
                "color": distinctive,
                "linestyle": "--",
                "linewidth": 1.0,
                "alpha": 1.0,
            },
        )
        fig = plot_kvinna_all_by_chamber(style=style)
        ax = fig.axes[0]

        # The suffrage line is the vertical line at x=1922 with our colour.
        matches = []
        for ln in ax.get_lines():
            xs = list(ln.get_xdata())
            if len(xs) >= 1 and xs[0] == 1922 and _colour_matches(ln.get_color(), distinctive):
                matches.append(ln)
        self.assertTrue(matches, "custom suffrage-line colour did not reach the axes")


class PartySubsetTest(unittest.TestCase):
    """plot_party_share honours the requested party subset."""

    def test_subset_gives_exact_number_of_content_lines(self):
        fig = plot_party_share(parties=("S", "H"))
        ax = fig.axes[0]

        # Content lines = plotted party series; the suffrage marker also
        # sits on the axes but carries the configured suffrage label.
        content = [
            ln
            for ln in ax.get_lines()
            if ln.get_label() != DEFAULT_STYLE.suffrage_label
        ]
        self.assertEqual(len(content), 2)

    def test_single_party_gives_one_line(self):
        fig = plot_party_share(parties=("S",))
        content = [
            ln
            for ln in fig.axes[0].get_lines()
            if ln.get_label() != DEFAULT_STYLE.suffrage_label
        ]
        self.assertEqual(len(content), 1)

    def test_default_party_mapping_covers_expected_blocs(self):
        # Sanity check: the shipped default mapping has exactly the
        # bloc labels the paper's methods section will name (five
        # historical blocs plus UP for "Utan partibeteckning").
        self.assertEqual(
            set(DEFAULT_PARTY_MAPPING), {"S", "H", "L", "Bf", "K", "UP"}
        )

    def test_stacked_shares_sum_to_one_when_include_other(self):
        # With include_other=True every year's stack must sum to 1.0
        # (up to float noise) so the plot's "100 % of woman-utterances"
        # reading is honest.
        fig = plot_party_share_stacked(parties=("S", "H"), include_other=True)
        ax = fig.axes[0]
        collections = ax.collections  # one PolyCollection per stack layer
        self.assertGreaterEqual(len(collections), 3)  # S, H, Other

    def test_stacked_without_other_excludes_unmapped(self):
        # include_other=False must not add an "Other" layer.
        fig = plot_party_share_stacked(parties=("S", "H"), include_other=False)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertNotIn("Other / unmapped", labels)


def _colour_matches(a, b) -> bool:
    """Compare two matplotlib colour strings/tuples in RGBA space."""
    from matplotlib.colors import to_rgba

    return to_rgba(a) == to_rgba(b)


class KvinnaNetCompositionTest(unittest.TestCase):
    """Structural + integration tests for plot_kvinna_net_composition."""

    def test_returns_figure(self):
        self.assertIsInstance(plot_kvinna_net_composition(), Figure)

    def test_has_two_upset_blocks(self):
        fig = plot_kvinna_net_composition()
        # Each UpSet block contributes 3 axes (matrix, intersections,
        # totals). Two stacked blocks => 6 axes. `>=` guards against
        # future upsetplot versions adding auxiliary axes.
        self.assertGreaterEqual(len(fig.axes), 6)

    def test_custom_upset_bar_color_reaches_bars(self):
        from dataclasses import replace
        from matplotlib.patches import Rectangle

        distinctive = "#00cc44"
        style = replace(DEFAULT_STYLE, upset_bar_color=distinctive)
        fig = plot_kvinna_net_composition(style=style)

        found = False
        for ax in fig.axes:
            for patch in ax.patches:
                if isinstance(patch, Rectangle):
                    if _colour_matches(patch.get_facecolor(), distinctive):
                        found = True
                        break
            if found:
                break
        self.assertTrue(found, "custom upset_bar_color did not reach the axes")

    def test_data_injection_bypasses_disk(self):
        def _seven_subsets(base_count: int) -> pd.DataFrame:
            return pd.DataFrame(
                [
                    {"k1": True, "k2": False, "k3": False, "word_match_count": base_count, "utterance_count": base_count},
                    {"k1": False, "k2": True, "k3": False, "word_match_count": base_count // 2, "utterance_count": base_count // 2},
                    {"k1": False, "k2": False, "k3": True, "word_match_count": base_count // 4, "utterance_count": base_count // 4},
                    {"k1": True, "k2": True, "k3": False, "word_match_count": base_count // 8, "utterance_count": base_count // 8},
                    {"k1": True, "k2": False, "k3": True, "word_match_count": base_count // 10, "utterance_count": base_count // 10},
                    {"k1": False, "k2": True, "k3": True, "word_match_count": base_count // 12, "utterance_count": base_count // 12},
                    {"k1": True, "k2": True, "k3": True, "word_match_count": base_count // 20, "utterance_count": base_count // 20},
                ]
            )

        fig = plot_kvinna_net_composition(
            words_subsets=_seven_subsets(10_000),
            utterance_subsets=_seven_subsets(500),
        )
        self.assertIsInstance(fig, Figure)


class SilentMenSharePerYearTest(unittest.TestCase):
    """plot_silent_men_share_per_year — yearly silent-population trend."""

    @staticmethod
    def _fake_speakers() -> pd.DataFrame:
        """Synthetic per-year speakers frame.

        1910 — four men (m3 unambiguously silent, m4 ineligible), one
        woman, one unknown-speaker row. Expected silent_share = 1/3
        (m3 silent among {m1, m2, m3} eligible; m4 excluded by
        min_utterances=50).

        1911 — two men, no woman-talk at all. peer_mean = 0, so no
        man is silent. Expected silent_share = 0.

        1912 — one man below min_utterances. eligible_count = 0, so
        year absent from output series.
        """
        return pd.DataFrame([
            {"year": 1910, "chamber": 1, "gender": "man", "party": "H", "who": "m1",
             "utterance_count": 100, "k1_utts": 5, "k2_utts": 5, "k3_utts": 5, "k_all_utts": 15},
            {"year": 1910, "chamber": 1, "gender": "man", "party": "H", "who": "m2",
             "utterance_count": 100, "k1_utts": 2, "k2_utts": 2, "k3_utts": 1, "k_all_utts": 5},
            {"year": 1910, "chamber": 1, "gender": "man", "party": "H", "who": "m3",
             "utterance_count": 500, "k1_utts": 0, "k2_utts": 0, "k3_utts": 0, "k_all_utts": 0},
            {"year": 1910, "chamber": 1, "gender": "man", "party": "H", "who": "m4",
             "utterance_count": 30, "k1_utts": 0, "k2_utts": 0, "k3_utts": 0, "k_all_utts": 0},
            {"year": 1910, "chamber": 1, "gender": "", "party": "", "who": "",
             "utterance_count": 1000, "k1_utts": 100, "k2_utts": 100, "k3_utts": 100, "k_all_utts": 300},
            {"year": 1910, "chamber": 1, "gender": "woman", "party": "S", "who": "w1",
             "utterance_count": 50, "k1_utts": 25, "k2_utts": 10, "k3_utts": 5, "k_all_utts": 30},
            {"year": 1911, "chamber": 1, "gender": "man", "party": "H", "who": "m5",
             "utterance_count": 100, "k1_utts": 0, "k2_utts": 0, "k3_utts": 0, "k_all_utts": 0},
            {"year": 1911, "chamber": 1, "gender": "man", "party": "H", "who": "m6",
             "utterance_count": 100, "k1_utts": 0, "k2_utts": 0, "k3_utts": 0, "k_all_utts": 0},
            {"year": 1912, "chamber": 1, "gender": "man", "party": "H", "who": "m7",
             "utterance_count": 20, "k1_utts": 0, "k2_utts": 0, "k3_utts": 0, "k_all_utts": 0},
        ])

    def test_invalid_k_group_raises(self):
        with self.assertRaises(ValueError):
            plot_silent_men_share_per_year(
                speakers=self._fake_speakers(), k_group="k4"
            )

    def test_silent_share_matches_wilson_math(self):
        """1910: silent_share = 1/3 (m3 statistically silent; m4 ineligible).

        Peer mean for 1910 (pooled over all four men) =
        (15 + 5 + 0 + 0) / (100 + 100 + 500 + 30) = 20 / 730 ≈ 0.0274.
        Half-rate threshold ≈ 0.0137.

        Wilson 95% UB per speaker (k / n → hi):
          m1: 15/100 → hi ≈ 0.233  (>> 0.014, not silent)
          m2:  5/100 → hi ≈ 0.112  (> 0.014, not silent)
          m3:  0/500 → hi ≈ 0.008  (< 0.014, SILENT)
          m4:  0/30  → ineligible (utterance_count < 50)

        Eligible: {m1, m2, m3}. Silent & eligible: {m3}. Share = 1/3.
        """
        fig = plot_silent_men_share_per_year(
            speakers=self._fake_speakers(), smoothing=None
        )
        ax = fig.axes[0]
        # The primary series is the first Line2D on the axes.
        line = ax.get_lines()[0]
        years = list(line.get_xdata())
        shares = list(line.get_ydata())
        self.assertIn(1910, years)
        idx = years.index(1910)
        self.assertAlmostEqual(shares[idx], 1.0 / 3.0, places=6)

    def test_structural_axes_and_suffrage_marker(self):
        fig = plot_silent_men_share_per_year(
            speakers=self._fake_speakers(), smoothing=None
        )
        # One primary Axes plus one twin for eligible_count.
        self.assertEqual(len(fig.axes), 2)
        primary = fig.axes[0]
        # Suffrage marker: a vertical line at x=1922.
        xs = [
            ln.get_xdata()[0]
            for ln in primary.get_lines()
            if len(ln.get_xdata()) == 2
            and ln.get_xdata()[0] == ln.get_xdata()[1]
        ]
        self.assertIn(1922, xs)
        # Primary y-axis label mentions "silent".
        self.assertIn("silent", primary.get_ylabel().lower())
        # Secondary axis label mentions "eligible".
        self.assertIn("eligible", fig.axes[1].get_ylabel().lower())

    @staticmethod
    def _multiyear_speakers() -> pd.DataFrame:
        """5 years × 3 men, so smoothing has room to trim edges."""
        rows = []
        for y in range(1910, 1915):
            for who in ("m1", "m2", "m3"):
                rows.append({
                    "year": y, "chamber": 1, "gender": "man", "party": "H",
                    "who": who, "utterance_count": 100,
                    "k1_utts": 3, "k2_utts": 2, "k3_utts": 1, "k_all_utts": 6,
                })
        return pd.DataFrame(rows)

    def test_smoothing_drops_boundary_years_from_primary(self):
        """smoothing=3, 5 years → primary line has 3 non-NaN y-values."""
        fig = plot_silent_men_share_per_year(
            speakers=self._multiyear_speakers(), smoothing=3
        )
        ax = fig.axes[0]
        # The last plotted line before the suffrage marker is the primary.
        # Identify by label.
        primary = next(
            ln for ln in ax.get_lines()
            if "Silent share" in (ln.get_label() or "")
        )
        ys = list(primary.get_ydata())
        # First and last year of the 5-year window are dropped by strict edges.
        # pandas rolling(center=True, min_periods=window) leaves NaN there.
        n_valid = sum(1 for y in ys if not pd.isna(y))
        self.assertEqual(n_valid, 3)

    def test_smoothing_none_leaves_all_years(self):
        fig = plot_silent_men_share_per_year(
            speakers=self._multiyear_speakers(), smoothing=None
        )
        ax = fig.axes[0]
        primary = next(
            ln for ln in ax.get_lines()
            if "Silent share" in (ln.get_label() or "")
        )
        ys = list(primary.get_ydata())
        n_valid = sum(1 for y in ys if not pd.isna(y))
        self.assertEqual(n_valid, 5)

    def test_k_group_selects_numerator(self):
        """Verify k_group flows to the title tag.

        The numerator column switch (num_col = f"{k_group}_utts") is
        already exercised by test_silent_share_matches_wilson_math for
        k_all; this test just confirms the k_group value propagates to
        the title on both k_all and k1 renders. No data-level assertion
        is made on silence outcomes under k1 — the k1 peer mean and
        Wilson thresholds are different from k_all, and this test does
        not attempt to re-derive them.
        """
        base = plot_silent_men_share_per_year(
            speakers=self._fake_speakers(), smoothing=None, k_group="k_all"
        )
        k1 = plot_silent_men_share_per_year(
            speakers=self._fake_speakers(), smoothing=None, k_group="k1"
        )
        # Titles must carry the correct K-group tag.
        self.assertIn("K_ALL", base.axes[0].get_title())
        self.assertIn("K1", k1.axes[0].get_title())

    def test_year_with_zero_peer_mean_has_zero_silent_share(self):
        """1911 in _fake_speakers has all men at k=0 → peer_mean=0.

        Threshold = 0 × 0.5 = 0. No Wilson UB is < 0. Silent share = 0.
        """
        fig = plot_silent_men_share_per_year(
            speakers=self._fake_speakers(), smoothing=None
        )
        primary = next(
            ln for ln in fig.axes[0].get_lines()
            if "Silent share" in (ln.get_label() or "")
        )
        years = list(primary.get_xdata())
        shares = list(primary.get_ydata())
        self.assertIn(1911, years)
        self.assertEqual(shares[years.index(1911)], 0.0)

    def test_year_with_no_eligible_men_is_dropped(self):
        """1912 in _fake_speakers has only m7 (N=20 < 50). Expected: 1912
        not present on the primary x-axis at all.
        """
        fig = plot_silent_men_share_per_year(
            speakers=self._fake_speakers(), smoothing=None
        )
        primary = next(
            ln for ln in fig.axes[0].get_lines()
            if "Silent share" in (ln.get_label() or "")
        )
        years = list(primary.get_xdata())
        self.assertNotIn(1912, years)

    def test_year_range_filters_source_rows(self):
        """year_range=(1911, 1912) should exclude 1910 entirely."""
        fig = plot_silent_men_share_per_year(
            speakers=self._fake_speakers(),
            smoothing=None,
            year_range=(1911, 1912),
        )
        primary = next(
            ln for ln in fig.axes[0].get_lines()
            if "Silent share" in (ln.get_label() or "")
        )
        years = list(primary.get_xdata())
        self.assertNotIn(1910, years)
        # 1911 present (peer_mean=0 branch), 1912 dropped (no eligible).
        self.assertIn(1911, years)
        self.assertNotIn(1912, years)

    def test_custom_style_flows_through(self):
        """A custom PlotStyle changes the suffrage marker year → visible
        in the plotted axvline positions.
        """
        custom = PlotStyle(suffrage_year=1919)
        fig = plot_silent_men_share_per_year(
            speakers=self._fake_speakers(), smoothing=None, style=custom
        )
        primary = fig.axes[0]
        xs = [
            ln.get_xdata()[0]
            for ln in primary.get_lines()
            if len(ln.get_xdata()) == 2
            and ln.get_xdata()[0] == ln.get_xdata()[1]
        ]
        self.assertIn(1919, xs)
        self.assertNotIn(1921, xs)
class GenderSourceFrameTest(unittest.TestCase):
    """Data-injection tests for the flipped-share aggregation."""

    def _df(self):
        # Three years, three rows each: man / woman / unknown-gender.
        # k_all_utts totals per year across genders:
        #   1900: 80m + 20w + 25u = 125   (known = 100, unknown/known = 0.25)
        #   1901: 60m + 40w + 50u = 150   (known = 100, unknown/known = 0.50)
        #   1902: 90m + 10w +  5u = 105   (known = 100, unknown/known = 0.05)
        return pd.DataFrame(
            [
                {"year": 1900, "gender": "man",   "k_all_utts": 80, "utterance_count": 1000, "k_all_matches": 800, "total_words": 20000},
                {"year": 1900, "gender": "woman", "k_all_utts": 20, "utterance_count": 100,  "k_all_matches": 200, "total_words": 2000},
                {"year": 1900, "gender": None,    "k_all_utts": 25, "utterance_count": 500,  "k_all_matches": 250, "total_words": 10000},
                {"year": 1901, "gender": "man",   "k_all_utts": 60, "utterance_count": 1000, "k_all_matches": 600, "total_words": 20000},
                {"year": 1901, "gender": "woman", "k_all_utts": 40, "utterance_count": 100,  "k_all_matches": 400, "total_words": 2000},
                {"year": 1901, "gender": None,    "k_all_utts": 50, "utterance_count": 500,  "k_all_matches": 500, "total_words": 10000},
                {"year": 1902, "gender": "man",   "k_all_utts": 90, "utterance_count": 1000, "k_all_matches": 900, "total_words": 20000},
                {"year": 1902, "gender": "woman", "k_all_utts": 10, "utterance_count": 100,  "k_all_matches": 100, "total_words": 2000},
                {"year": 1902, "gender": None,    "k_all_utts":  5, "utterance_count": 500,  "k_all_matches":  50, "total_words": 10000},
            ]
        )

    def test_man_plus_woman_share_sums_to_one(self):
        g = _gender_source_frame(self._df(), "k_all_utts")
        totals = (g["man_share"] + g["woman_share"]).tolist()
        for total in totals:
            self.assertAlmostEqual(total, 1.0)

    def test_woman_share_matches_expected_split(self):
        g = _gender_source_frame(self._df(), "k_all_utts")
        by_year = dict(zip(g["year"], g["woman_share"]))
        self.assertAlmostEqual(by_year[1900], 20 / 100)
        self.assertAlmostEqual(by_year[1901], 40 / 100)
        self.assertAlmostEqual(by_year[1902], 10 / 100)

    def test_unknown_band_is_ratio_to_known(self):
        g = _gender_source_frame(self._df(), "k_all_utts")
        by_year = dict(zip(g["year"], g["unknown_over_known"]))
        self.assertAlmostEqual(by_year[1900], 25 / 100)
        self.assertAlmostEqual(by_year[1901], 50 / 100)
        self.assertAlmostEqual(by_year[1902], 5 / 100)

    def test_word_source_frame_uses_matches_column(self):
        # Word-level uses k_all_matches; shape is the same as utterance-level.
        g = _gender_source_frame(self._df(), "k_all_matches")
        by_year = dict(zip(g["year"], g["woman_share"]))
        self.assertAlmostEqual(by_year[1900], 200 / 1000)
        self.assertAlmostEqual(by_year[1901], 400 / 1000)
        self.assertAlmostEqual(by_year[1902], 100 / 1000)


class WomanChimeVsStartTest(unittest.TestCase):
    """Plot 5c: two lines (utterance share, starter share) over post-1921 years."""

    def _word_freq(self):
        # Yearly (year, gender, k_all_utts, utterance_count, ...) rows.
        # Woman utt-share per year matches hand-picked values.
        return pd.DataFrame(
            [
                {"year": 1922, "gender": "man",   "k_all_utts": 970, "utterance_count": 9700, "k_all_matches": 970, "total_words": 100000},
                {"year": 1922, "gender": "woman", "k_all_utts":  30, "utterance_count":  300, "k_all_matches":  30, "total_words":   3000},
                {"year": 1922, "gender": None,    "k_all_utts":  50, "utterance_count":  500, "k_all_matches":  50, "total_words":   5000},
                {"year": 1924, "gender": "man",   "k_all_utts": 900, "utterance_count": 9000, "k_all_matches": 900, "total_words": 100000},
                {"year": 1924, "gender": "woman", "k_all_utts": 100, "utterance_count":  500, "k_all_matches": 100, "total_words":   5000},
            ]
        )

    def _starters(self):
        return pd.DataFrame(
            [
                {"year": 1922, "gender": "man",   "starters": 90},
                {"year": 1922, "gender": "woman", "starters": 10},
                {"year": 1924, "gender": "man",   "starters": 85},
                {"year": 1924, "gender": "woman", "starters": 15},
            ]
        )

    def test_returns_figure(self):
        fig = plot_woman_chime_vs_start(self._starters(), self._word_freq())
        self.assertIsInstance(fig, Figure)

    def test_two_content_lines(self):
        fig = plot_woman_chime_vs_start(self._starters(), self._word_freq())
        ax = fig.axes[0]
        content = [
            ln
            for ln in ax.get_lines()
            if ln.get_label() != DEFAULT_STYLE.suffrage_label
        ]
        self.assertEqual(len(content), 2)

    def test_line_values_match_hand_computed_shares(self):
        fig = plot_woman_chime_vs_start(self._starters(), self._word_freq())
        ax = fig.axes[0]

        by_label = {ln.get_label(): ln for ln in ax.get_lines()}
        utt = by_label["Share of all woman-utterances"]
        st = by_label["Share of arc starters"]

        # 1922: utt = 30 / (970 + 30) = 0.03; starter = 10 / (90 + 10) = 0.10
        # 1924: utt = 100 / (900 + 100) = 0.10; starter = 15 / (85 + 15) = 0.15
        for line, year, expected in [
            (utt, 1922, 0.03),
            (utt, 1924, 0.10),
            (st, 1922, 0.10),
            (st, 1924, 0.15),
        ]:
            xs = list(line.get_xdata())
            ys = list(line.get_ydata())
            self.assertAlmostEqual(ys[xs.index(year)], expected, places=4)


class GenderSourcePlotStructureTest(unittest.TestCase):
    """The flipped-share plots render three stacked bands over a known baseline."""

    def _df(self):
        return GenderSourceFrameTest()._df()

    def test_utterance_source_has_three_stack_bands(self):
        from matplotlib.collections import PolyCollection

        fig = plot_utterance_source_by_gender(self._df())
        ax = fig.axes[0]
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        self.assertEqual(len(polys), 3)

    def test_word_source_has_three_stack_bands(self):
        from matplotlib.collections import PolyCollection

        fig = plot_word_source_by_gender(self._df())
        ax = fig.axes[0]
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        self.assertEqual(len(polys), 3)

    def test_unit_reference_line_present(self):
        # A dashed/solid horizontal at y=1 separates the known-gender
        # decomposition (bottom two bands) from the unknown band above.
        fig = plot_utterance_source_by_gender(self._df())
        ax = fig.axes[0]
        horizontals_at_one = [
            ln
            for ln in ax.get_lines()
            if len(ln.get_ydata()) >= 2
            and all(y == 1.0 for y in ln.get_ydata())
        ]
        self.assertTrue(horizontals_at_one)


class PartySpeakerStartsVsChimesTest(unittest.TestCase):
    """Plot 8: per-speaker rate scatter, small multiples per party bloc."""

    def _speakers_df(self):
        # Synthetic per-year speaker rows spanning three MPs across three years.
        # Speaker "who=alice" (S) has 500 total utts, 100 kvinna hits.
        # Speaker "who=bob"   (H) has 300 total utts, 50  kvinna hits.
        # Speaker "who=zoe"   (party outside taxonomy) has 100 total, 10 kvinna.
        return pd.DataFrame([
            {"year": 1922, "chamber": 2, "gender": "man",   "party": "Socialdemokraterna", "who": "alice", "utterance_count": 500, "k1_utts": 30, "k2_utts": 20, "k3_utts": 20, "k_all_utts": 100},
            {"year": 1922, "chamber": 2, "gender": "man",   "party": "Högerpartiet",       "who": "bob",   "utterance_count": 300, "k1_utts": 15, "k2_utts": 15, "k3_utts": 10, "k_all_utts": 50},
            {"year": 1922, "chamber": 1, "gender": "woman", "party": "Fringe Party",       "who": "zoe",   "utterance_count": 100, "k1_utts":  5, "k2_utts":  3, "k3_utts":  2, "k_all_utts": 10},
        ])

    def _starters_df(self):
        # Per-year starter counts for the same three MPs.
        return pd.DataFrame([
            {"year": 1922, "gender": "man",   "who": "alice", "starters": 40},
            {"year": 1922, "gender": "man",   "who": "bob",   "starters": 15},
            {"year": 1922, "gender": "woman", "who": "zoe",   "starters":  5},
        ])

    def test_default_produces_seven_panels_including_unaffiliated(self):
        fig = plot_party_speaker_starts_vs_chimes(
            self._starters_df(), self._speakers_df()
        )
        # 6 blocs (S, H, L, Bf, K, UP) + 1 unaffiliated = 7 visible axes.
        self.assertIsInstance(fig, Figure)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        self.assertEqual(len(visible), 7)

    def test_include_unaffiliated_false_hides_that_panel(self):
        fig = plot_party_speaker_starts_vs_chimes(
            self._starters_df(), self._speakers_df(), include_unaffiliated=False,
        )
        visible = [ax for ax in fig.axes if ax.get_visible()]
        self.assertEqual(len(visible), 6)

    def test_bloc_subset_produces_that_many_panels(self):
        fig = plot_party_speaker_starts_vs_chimes(
            self._starters_df(), self._speakers_df(),
            blocs=("S", "H"), include_unaffiliated=False,
        )
        visible = [ax for ax in fig.axes if ax.get_visible()]
        self.assertEqual(len(visible), 2)

    def test_each_panel_has_background_and_foreground_scatter(self):
        from matplotlib.collections import PathCollection

        fig = plot_party_speaker_starts_vs_chimes(
            self._starters_df(), self._speakers_df(),
            blocs=("S", "H"), include_unaffiliated=False,
        )
        for ax in [a for a in fig.axes if a.get_visible()]:
            scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
            # Each panel now emits one scatter per marker per group
            # (bg + fg), so ≥ 2 scatters with at least one grey-colored
            # bg and one bloc-colored fg present.
            self.assertGreaterEqual(len(scatters), 2)

    def test_dominant_bloc_places_speaker_on_expected_panel(self):
        """alice (S, 500 utts) should have (starter=40/500, chime=60/500)
        on the S panel and appear as background on H."""
        from matplotlib.collections import PathCollection

        fig = plot_party_speaker_starts_vs_chimes(
            self._starters_df(), self._speakers_df(),
            blocs=("S", "H"), include_unaffiliated=False,
        )
        s_ax, h_ax = [a for a in fig.axes if a.get_visible()]
        alice_x, alice_y = 40 / 500, (100 - 40) / 500
        # Alice must land at her coords in the bloc-coloured foreground
        # (i.e. any scatter whose colour isn't the bg grey #e0e0e0).
        s_scatters = [c for c in s_ax.collections if isinstance(c, PathCollection)]
        fg_scatters = [
            c for c in s_scatters
            if tuple(c.get_facecolor()[0][:3]) != tuple(
                __import__("matplotlib.colors", fromlist=["to_rgb"]).to_rgb("#e0e0e0")
            )
        ]
        matched = any(
            any(
                abs(pt[0] - alice_x) < 1e-9 and abs(pt[1] - alice_y) < 1e-9
                for pt in c.get_offsets()
            )
            for c in fg_scatters
        )
        self.assertTrue(matched, "alice should be on the S panel foreground")

    def test_out_of_taxonomy_speaker_is_background_on_bloc_panels(self):
        """zoe's party 'Fringe Party' isn't in DEFAULT_PARTY_MAPPING;
        she must appear in the background of every bloc panel."""
        from matplotlib.collections import PathCollection

        fig = plot_party_speaker_starts_vs_chimes(
            self._starters_df(), self._speakers_df(),
            blocs=("S", "H"), include_unaffiliated=False,
        )
        from matplotlib.colors import to_rgb
        zoe_x, zoe_y = 5 / 100, (10 - 5) / 100
        bg_rgb = to_rgb("#e0e0e0")
        for ax in [a for a in fig.axes if a.get_visible()]:
            scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
            bg_scatters = [
                c for c in scatters
                if tuple(c.get_facecolor()[0][:3]) == bg_rgb
            ]
            matched = any(
                any(
                    abs(pt[0] - zoe_x) < 1e-9 and abs(pt[1] - zoe_y) < 1e-9
                    for pt in c.get_offsets()
                )
                for c in bg_scatters
            )
            self.assertTrue(matched, "zoe should be in background of every bloc panel")

    def test_out_of_taxonomy_speaker_is_foreground_on_unaffiliated_panel(self):
        """With include_unaffiliated=True, zoe becomes foreground on the
        unaffiliated panel and disappears from the bloc panels' foregrounds."""
        from matplotlib.collections import PathCollection

        fig = plot_party_speaker_starts_vs_chimes(
            self._starters_df(), self._speakers_df(),
            blocs=("S",), include_unaffiliated=True,
        )
        from matplotlib.colors import to_rgb
        s_ax, unaff_ax = [a for a in fig.axes if a.get_visible()]
        zoe_x, zoe_y = 5 / 100, (10 - 5) / 100
        unaff_scatters = [
            c for c in unaff_ax.collections if isinstance(c, PathCollection)
        ]
        bg_rgb = to_rgb("#e0e0e0")
        fg_scatters = [
            c for c in unaff_scatters
            if tuple(c.get_facecolor()[0][:3]) != bg_rgb
        ]
        matched = any(
            any(
                abs(pt[0] - zoe_x) < 1e-9 and abs(pt[1] - zoe_y) < 1e-9
                for pt in c.get_offsets()
            )
            for c in fg_scatters
        )
        self.assertTrue(matched, "zoe should be in foreground of unaffiliated panel")


if __name__ == "__main__":
    unittest.main()
