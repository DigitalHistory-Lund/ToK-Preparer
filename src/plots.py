"""Paper-figure plotting API.

The public plot_* functions in this module produce the figure set for
the "Tal om Kvinnor" manuscript. Each function returns a
:class:`matplotlib.figure.Figure` and takes an optional
:class:`PlotStyle` so the paper draft repo can override colours,
sizes, and formatting without editing this module. Most plots read
from the shipped ``word_frequencies.csv.gz`` snapshot by default (via
:mod:`src.data`); the per-speaker plots read ``speakers.csv.gz``.

The narrative spine of the paper is 1922 (women first sit in the
Riksdag); every temporal plot carries a dashed vertical marker at
``style.suffrage_year``.

See ``docs/word_frequencies.md`` for the schema of the underlying
CSV.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Iterable, Mapping
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .data import (
    read_speaker_starter_counts,
    read_speaker_totals,
    read_speaker_word_totals,
    read_speakers,
    read_word_frequencies,
    speaker_categories,
    speaker_starts_chimes,
)


# Category → matplotlib marker. Experimental cross-plot identifiability:
# a hexagon in the gender scatter is the same person as a hexagon in the
# octopus and in the party bloc grid.
SPEAKER_MARKERS: dict[str, str] = {
    "original_five": "h",  # hexagon — the five 1922 women
    "loud_man": "X",  # X — high-K-rate male tail
    "man": "s",  # square — all other men
    "woman": "o",  # circle — women outside the original 5
    "unknown": "^",  # triangle — unknown gender / empty who
}

SUFFRAGE_YEAR = 1922
YEAR_MIN = 1900
YEAR_MAX = 1940


# ---------------------------------------------------------------------------
# Party bloc mapping
# ---------------------------------------------------------------------------

# Historical party name → paper bloc code. The Riksdag went through many
# renamings and mergers in 1900–1940; this mapping pools successors under
# a single bloc so the paper can show ~5 stable lines rather than ~30
# fragmented ones. Callers can pass a different mapping to
# :func:`plot_party_share` to make a different editorial choice.
#
# Rationale (short form; longer form belongs in the paper's methods):
# - S:  Socialdemokraterna, unchanged across the window.
# - H:  1st-chamber conservative groupings + Högerpartiet family
#       (successor to the 1904 Right).
# - L:  Liberal successor line — Liberala samlingspartiet split in 1923
#       into Liberala riksdagspartiet and Frisinnade folkpartiet, then
#       re-merged as Folkpartiet in 1934; the 1911 Frisinnade
#       försvarsvänner defection is included.
# - Bf: Bondeförbundet + earlier Lantmanna groupings and later Centerpartiet.
# - K:  Left of Socialdemokraterna — SSV / Vänsterpartiet (1917),
#       Sveriges kommunistiska parti and its Kilbom/socialist splits.
DEFAULT_PARTY_MAPPING: dict[str, tuple[str, ...]] = {
    "S": ("Socialdemokraterna",),
    "H": (
        "Första kammarens nationella parti",
        "Första kammarens protektionistiska parti",
        "Första kammarens moderata parti",
        "Första kammarens minoritetsparti",
        "Högerns riksdagsgrupp",
        "Högerpartiet",
        "Det förenade högerpartiet",
        "Moderata samlingspartiet",
        "Högerpartiet de konservativa",
        "Nationella framstegspartiet",
    ),
    "L": (
        "Liberala samlingspartiet",
        "Liberala riksdagspartiet",
        "Frisinnade folkpartiet",
        "Folkpartiet",
        "Frisinnade försvarsvänner",
        "Liberalerna",
    ),
    "Bf": (
        "Bondeförbundet",
        "Lantmannapartiet",
        "Lantmanna- och borgarepartiet inom andrakammaren",
        "Jordbrukarnas fria grupp",
        "Nya lantmannapartiet",
        "Centerpartiet",
    ),
    "K": (
        "Vänsterpartiet",
        "Kommunistiska partiet",
        "Sveriges kommunistiska parti",
        "Kilbomspartiet",
        "Socialistiska partiet",
        "Socialdemokratiska vänstergruppen",
    ),
}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlotStyle:
    """Overridable styling for the paper figures.

    Construct via ``dataclasses.replace(DEFAULT_STYLE, ...)`` in the
    draft repo to keep the paper's palette in one place. Every field
    has a paper-sensible default; nothing here is load-bearing for
    correctness — only appearance.
    """

    chamber_colors: dict = field(default_factory=lambda: {1: "#3b6ea5", 2: "#c65f4d"})
    gender_colors: dict = field(
        default_factory=lambda: {
            "man": "#4a4a4a",
            "woman": "#c94f8a",
            "unknown": "#bbbbbb",
        }
    )
    category_colors: dict = field(
        default_factory=lambda: {"k1": "#4c72b0", "k2": "#dd8452", "k3": "#55a467"}
    )
    category_labels: dict = field(
        default_factory=lambda: {
            "k1": "K1 — broad (kvinn*)",
            "k2": "K2 — kin & generic",
            "k3": "K3 — work titles",
        }
    )
    party_colors: dict = field(
        default_factory=lambda: {
            "S": "#e41a1c",
            "H": "#377eb8",
            "L": "#4daf4a",
            "Bf": "#984ea3",
            "K": "#a65628",
        }
    )
    party_labels: dict = field(
        default_factory=lambda: {
            "S": "Socialdemokraterna",
            "H": "Högern (och föregångare)",
            "L": "Liberala familjen",
            "Bf": "Bondeförbundet (och föregångare)",
            "K": "Vänster om S",
        }
    )
    kvinna_stack_colors: dict = field(
        default_factory=lambda: {
            "k1": "#d9d9d9",
            "k2": "#a6a6a6",
            "k3": "#6e6e6e",
        }
    )
    kvinna_reference_line: dict = field(
        default_factory=lambda: {
            "color": "#000000",
            "linewidth": 1.8,
            "linestyle": "-",
            "label": "kvinna_all (union of K1–K3)",
        }
    )
    counting_mode_colors: dict = field(
        default_factory=lambda: {"utts": "#6e6e6e", "words": "#000000"}
    )
    share_colors: dict = field(
        default_factory=lambda: {"utts": "#4c72b0", "words": "#dd8452"}
    )
    kvinna_pattern_labels: dict = field(
        default_factory=lambda: {
            "k1": "K1 — broad (kvinn*)",
            "k2": "K2 — kin & generic",
            "k3": "K3 — work titles",
            "k_all": "kvinna_all (union of K1–K3)",
        }
    )
    upset_bar_color: str = "#4a4a4a"
    upset_dot_filled: str = "#000000"
    upset_dot_empty: str = "#cccccc"
    suffrage_year: int = SUFFRAGE_YEAR
    suffrage_line: dict = field(
        default_factory=lambda: {
            "color": "#333333",
            "linestyle": "--",
            "linewidth": 1.0,
            "alpha": 0.6,
        }
    )
    suffrage_label: str = "1922 (first women in Riksdag)"
    figsize: tuple = (7.0, 4.0)
    grid: dict = field(default_factory=lambda: {"linestyle": ":", "alpha": 0.4})
    small_n_threshold: int = 30
    small_n_annotation: dict = field(
        default_factory=lambda: {"fontsize": 7, "color": "#666666"}
    )


DEFAULT_STYLE = PlotStyle()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _style(style: PlotStyle | None) -> PlotStyle:
    return style if style is not None else DEFAULT_STYLE


def _frame(df: pd.DataFrame | None) -> pd.DataFrame:
    return df if df is not None else read_word_frequencies()


@functools.cache
def _speaker_category_map() -> dict[str, str]:
    """Cached ``who -> category`` lookup for cross-plot marker consistency."""
    cats = speaker_categories()
    return dict(zip(cats["who"], cats["category"]))


def _attach_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``category`` and ``marker`` columns using the cached global map.

    Rows without a ``who`` match fall back to their ``gender`` (man /
    woman) or ``unknown``. Callers then group by ``marker`` and issue
    one scatter call per marker.
    """
    lookup = _speaker_category_map()
    d = df.copy()
    d["category"] = d["who"].map(lookup)

    # Fallback for who-IDs not in the totals table (e.g. filtered rows).
    def _fallback(row: pd.Series) -> str:
        if pd.notna(row["category"]):
            return row["category"]
        g = row.get("gender")
        return g if g in ("man", "woman") else "unknown"

    d["category"] = d.apply(_fallback, axis=1)
    d["marker"] = d["category"].map(SPEAKER_MARKERS).fillna("o")
    return d


def _add_suffrage_marker(ax: Axes, style: PlotStyle) -> None:
    """Dashed vertical line at ``style.suffrage_year`` on every temporal plot."""
    ax.axvline(style.suffrage_year, label=style.suffrage_label, **style.suffrage_line)


def _year_axis(ax: Axes, start: int = YEAR_MIN, end: int = YEAR_MAX) -> None:
    """Consistent x-axis limits and ticks across all figures."""
    ax.set_xlim(start, end)
    ax.set_xticks(range(start, end + 1, 5))
    ax.set_xlabel("Year")


def _annotate_small_n(
    ax: Axes,
    xs: Iterable[float],
    ys: Iterable[float],
    ns: Iterable[float],
    style: PlotStyle,
) -> None:
    """Annotate line points where ``N < threshold`` with the actual N.

    Used on the women's series where 1922–1925 have N ∈ [17, 37]. The
    annotation is a substantive part of the argument — small numbers
    should be visible, not smoothed away.
    """
    for x, y, n in zip(xs, ys, ns):
        if pd.notna(y) and pd.notna(n) and n < style.small_n_threshold:
            ax.annotate(
                f"N={int(n)}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                **style.small_n_annotation,
            )


def _reduce_parties(
    df: pd.DataFrame, mapping: Mapping[str, Iterable[str]]
) -> pd.DataFrame:
    """Return ``df`` with a ``bloc`` column, dropping rows outside ``mapping``.

    Rows whose ``party`` is not covered by any bloc (unknown or
    excluded historical parties) are removed. Callers who want an
    "Other" pool should do that pooling on the raw frame *before*
    calling this helper.
    """
    lookup = {name: bloc for bloc, names in mapping.items() for name in names}
    out = df.copy()
    out["bloc"] = out["party"].map(lookup)
    return out.dropna(subset=["bloc"])


def _apply_axis_extras(ax: Axes, style: PlotStyle, *, legend: bool = True) -> None:
    ax.grid(**style.grid)
    if legend:
        ax.legend(loc="best", frameon=False)


def _wilson_ci(
    k: pd.Series, n: pd.Series, z: float = 1.959963984540054
) -> tuple[pd.Series, pd.Series]:
    """Elementwise Wilson 95%-default score CI for k successes out of n trials.

    Vectorised over aligned Series; rows with ``n == 0`` yield NaN bounds.
    Assumes i.i.d. Bernoulli trials — see :func:`plot_keyword_shares_by_pattern`
    for the paper's methodological note on why that assumption is
    conservative for this census corpus.
    """
    k = k.astype(float)
    n = n.astype(float)
    n_eff = n + z * z
    center = (k + z * z / 2.0) / n_eff
    half = z * ((k * (n - k) / n + z * z / 4.0) ** 0.5) / n_eff
    lo = (center - half).where(n > 0)
    hi = (center + half).where(n > 0)
    return lo, hi


# ---------------------------------------------------------------------------
# Plot 1 — corpus baseline
# ---------------------------------------------------------------------------


def plot_baseline_corpus(
    df: pd.DataFrame | None = None, *, style: PlotStyle | None = None
) -> Figure:
    """Total utterances (left) and total words (right) per year, by chamber.

    Substitutes for a descriptive-statistics table: readers see both
    denominators — utterance counts and token counts — before the rest of
    the paper starts dividing by them. The two chambers share the palette
    across axes; the word series (right axis) is drawn dashed so a reader
    can tell it apart from the utterance series without relying on colour
    alone.
    """
    style = _style(style)
    df = _frame(df)

    g = (
        df.groupby(["year", "chamber"], as_index=False)[
            ["utterance_count", "total_words"]
        ]
        .sum()
        .sort_values(["chamber", "year"])
    )

    fig, ax = plt.subplots(figsize=style.figsize)
    ax_r = ax.twinx()
    for chamber, sub in g.groupby("chamber"):
        colour = style.chamber_colors.get(chamber)
        ax.plot(
            sub["year"],
            sub["utterance_count"],
            label=f"Chamber {chamber} — utterances",
            color=colour,
        )
        ax_r.plot(
            sub["year"],
            sub["total_words"],
            label=f"Chamber {chamber} — words",
            color=colour,
            linestyle="--",
        )

    _add_suffrage_marker(ax, style)
    _year_axis(ax)
    ax.set_ylim(bottom=0)
    ax_r.set_ylim(bottom=0)
    ax.set_ylabel("Utterances per year")
    ax_r.set_ylabel("Total words per year")
    ax.set_title("Corpus baseline: parliamentary volume, 1900–1940")
    ax.grid(**style.grid)
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(lines_l + lines_r, labels_l + labels_r, loc="best", frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 1b — keyword coverage (stacked area + kvinna_all reference line)
# ---------------------------------------------------------------------------


def plot_keyword_coverage(
    df: pd.DataFrame | None = None, *, style: PlotStyle | None = None
) -> Figure:
    """Absolute keyword-net coverage, per pattern and in aggregate.

    Two panels sharing the x-axis. Upper panel: utterance counts.
    Lower panel: word (token) counts. K1/K2/K3 are drawn as a stacked
    area in three shades of gray; ``kvinna_all`` (the CSV columns
    ``k_all_utts`` / ``k_all_matches``) is drawn as a solid black
    reference line on top of the stack.

    The reference line exposes a substantive asymmetry between the two
    counting modes. Word tokens are counted once per category match, so
    a token matching multiple categories contributes to each; therefore
    ``k1_matches + k2_matches + k3_matches >= k_all_matches`` and the
    line may sit at or below the stack top (the area above the line
    quantifies the token-level cross-category overlap). Utterance counts
    are deduped in ``k_all_utts`` (a single utterance hitting multiple
    patterns counts once), so ``k1_utts + k2_utts + k3_utts >= k_all_utts``
    and the line sits inside the stack — the area above the line
    quantifies the utterance-level pattern overlap.

    Chambers are aggregated: this plot's job is pattern coverage, not
    chamber contrast (which Plot 1 already carries).
    """
    style = _style(style)
    df = _frame(df)

    utt_cols = ["k1_utts", "k2_utts", "k3_utts", "k_all_utts"]
    word_cols = ["k1_matches", "k2_matches", "k3_matches", "k_all_matches"]
    g = (
        df.groupby("year", as_index=False)[utt_cols + word_cols]
        .sum()
        .sort_values("year")
    )

    fig, (ax_u, ax_w) = plt.subplots(
        2, 1, sharex=True, figsize=(style.figsize[0], style.figsize[1] * 1.6)
    )

    for ax, cols, ref_col, ylabel in (
        (ax_u, ["k1_utts", "k2_utts", "k3_utts"], "k_all_utts", "Utterances per year"),
        (
            ax_w,
            ["k1_matches", "k2_matches", "k3_matches"],
            "k_all_matches",
            "Word matches per year",
        ),
    ):
        ax.stackplot(
            g["year"],
            g[cols[0]],
            g[cols[1]],
            g[cols[2]],
            colors=[
                style.kvinna_stack_colors["k1"],
                style.kvinna_stack_colors["k2"],
                style.kvinna_stack_colors["k3"],
            ],
            labels=["K1", "K2", "K3"],
        )
        ax.plot(g["year"], g[ref_col], **style.kvinna_reference_line)
        _add_suffrage_marker(ax, style)
        ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel)
        ax.grid(**style.grid)

    _year_axis(ax_w)
    ax_u.set_title("Keyword coverage: K1/K2/K3 components and the kvinna_all union")
    ax_u.legend(loc="upper left", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 1b (facet variant) — keyword coverage, per-pattern small multiples
# ---------------------------------------------------------------------------


def plot_keyword_coverage_by_pattern(
    df: pd.DataFrame | None = None, *, style: PlotStyle | None = None
) -> Figure:
    """Per-pattern facet variant of :func:`plot_keyword_coverage`.

    Four rows (K1, K2, K3, kvinna_all top → bottom). Each row shares
    the x-axis and carries a twin y-axis:

    * Left (log scale): utterance count and word-match count, both as
      solid lines coloured by counting mode.
    * Right (log scale): utterance share (numerator /
      ``utterance_count``) and word share (numerator / ``total_words``),
      both as dashed lines, same colours. Shares vary by ~2 orders of
      magnitude between K3 and kvinna_all, so a linear axis pushes K3
      to the floor; log lets the components read together with the
      aggregate.

    Axis sharing across rows:

    * K1 / K2 / K3 rows share the left log axis (counts directly
      comparable across the three components).
    * All four rows share the right log axis (shares directly
      comparable across the full pattern inventory). The shared right
      axis's upper limit is anchored to the kvinna_all share max so
      the aggregate row shows the ceiling and the component rows sit
      below it visibly rather than being auto-scaled per row.
    * The kvinna_all row has its own left axis — its counts are 3–5×
      any single component and would otherwise squash K1–K3.

    Chambers are aggregated: chamber contrast lives in Plot 1; this
    figure's job is per-pattern coverage.

    Zero handling: K1 has zero matches in 1900 and 1901 (the
    ``qvinn``/``kvinn`` orthographic transition, documented as a
    caveat on Plot 4). Zeros on either log axis are masked with NaN
    so the line leaves a visible gap.
    """
    style = _style(style)
    df = _frame(df)

    count_cols = [
        "utterance_count",
        "total_words",
        "k1_utts",
        "k2_utts",
        "k3_utts",
        "k_all_utts",
        "k1_matches",
        "k2_matches",
        "k3_matches",
        "k_all_matches",
    ]
    g = df.groupby("year", as_index=False)[count_cols].sum().sort_values("year")

    patterns = ("k1", "k2", "k3", "k_all")
    fig, axes = plt.subplots(
        4, 1, sharex=True, figsize=(style.figsize[0], style.figsize[1] * 2.4)
    )
    # Link left log axes for K1/K2/K3; leave kvinna_all's left free.
    axes[0].sharey(axes[1])
    axes[1].sharey(axes[2])

    # Precompute shares; mask zeros so they leave a gap on the log axis.
    utts_share = {
        key: (g[f"{key}_utts"] / g["utterance_count"])
        .replace(0, pd.NA)
        .astype("Float64")
        for key in patterns
    }
    words_share = {
        key: (g[f"{key}_matches"] / g["total_words"])
        .replace(0, pd.NA)
        .astype("Float64")
        for key in patterns
    }
    # Right-axis upper bound anchored to kvinna_all (the largest share).
    share_upper = max(
        float(utts_share["k_all"].max(skipna=True)),
        float(words_share["k_all"].max(skipna=True)),
    )

    twins: list[Axes] = []
    for i, (ax, key) in enumerate(zip(axes, patterns)):
        utts_col = f"{key}_utts"
        matches_col = f"{key}_matches"

        utts_counts = g[utts_col].replace(0, pd.NA).astype("Float64")
        matches_counts = g[matches_col].replace(0, pd.NA).astype("Float64")

        ax.set_yscale("log")
        ax.plot(
            g["year"],
            utts_counts,
            color=style.counting_mode_colors["utts"],
            linestyle="-",
            label="Utterances (count)" if i == 0 else None,
        )
        ax.plot(
            g["year"],
            matches_counts,
            color=style.counting_mode_colors["words"],
            linestyle="-",
            label="Word matches (count)" if i == 0 else None,
        )

        ax_r = ax.twinx()
        twins.append(ax_r)
        ax_r.set_yscale("log")
        ax_r.plot(
            g["year"],
            utts_share[key],
            color=style.counting_mode_colors["utts"],
            linestyle="--",
            label="Utterance share" if i == 0 else None,
        )
        ax_r.plot(
            g["year"],
            words_share[key],
            color=style.counting_mode_colors["words"],
            linestyle="--",
            label="Word share" if i == 0 else None,
        )

        _add_suffrage_marker(ax, style)
        ax.set_title(style.kvinna_pattern_labels[key], loc="left", fontsize=9)
        ax.grid(**style.grid)
        if i == 3:
            ax.set_ylabel("Count (log)")
            ax_r.set_ylabel("Share (log)")

    # Link all four right axes so shares are comparable across the stack,
    # then anchor the shared upper limit to kvinna_all's max.
    for r in twins[1:]:
        r.sharey(twins[0])
    twins[0].set_ylim(top=share_upper * 1.1)

    _year_axis(axes[-1])

    # Single legend on the top panel combining left- and right-axis series.
    lines_l, labels_l = axes[0].get_legend_handles_labels()
    lines_r, labels_r = twins[0].get_legend_handles_labels()
    axes[0].legend(
        lines_l + lines_r,
        labels_l + labels_r,
        loc="upper left",
        frameon=False,
        fontsize=7,
        ncol=2,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 1b (shares variant) — per-pattern utterance + word share with Wilson CIs
# ---------------------------------------------------------------------------


def _draw_share_panel(
    ax: Axes,
    g: pd.DataFrame,
    key: str,
    style: PlotStyle,
    *,
    show_labels: bool,
) -> Axes:
    """One row of :func:`plot_keyword_shares_by_pattern`.

    Draws utterance share on the primary (left) axis and word share on a
    twin (right) axis, each with a translucent Wilson CI ribbon. Returns
    the twin axis so the caller can iterate and consolidate the legend.
    Labels are only attached when ``show_labels=True`` so the caller can
    build a single legend on the top row without per-row duplicates.
    """
    utts_num = g[f"{key}_utts"]
    words_num = g[f"{key}_matches"]
    utts_den = g["utterance_count"]
    words_den = g["total_words"]

    utt_share = utts_num / utts_den
    word_share = words_num / words_den
    utt_lo, utt_hi = _wilson_ci(utts_num, utts_den)
    word_lo, word_hi = _wilson_ci(words_num, words_den)

    year = g["year"]
    utt_color = style.share_colors["utts"]
    word_color = style.share_colors["words"]

    ax.fill_between(
        year,
        utt_lo,
        utt_hi,
        color=utt_color,
        alpha=0.22,
        linewidth=0,
        label="Utterance share — 95% Wilson CI" if show_labels else None,
    )
    ax.plot(
        year,
        utt_share,
        color=utt_color,
        linewidth=1.6,
        label="Utterance share (k_*_utts / utterance_count)" if show_labels else None,
    )

    ax_r = ax.twinx()
    ax_r.fill_between(
        year,
        word_lo,
        word_hi,
        color=word_color,
        alpha=0.22,
        linewidth=0,
        label="Word share — 95% Wilson CI" if show_labels else None,
    )
    ax_r.plot(
        year,
        word_share,
        color=word_color,
        linewidth=1.6,
        label="Word share (k_*_matches / total_words)" if show_labels else None,
    )

    ax.set_ylim(bottom=0)
    ax_r.set_ylim(bottom=0)
    ax.tick_params(axis="y", colors=utt_color)
    ax_r.tick_params(axis="y", colors=word_color)
    return ax_r


def plot_keyword_shares_by_pattern(
    df: pd.DataFrame | None = None, *, style: PlotStyle | None = None
) -> Figure:
    """Per-pattern facet: utterance + word share over time, with Wilson CIs.

    Four rows sharing the x-axis, in overview-then-composition order: ``k_all`` on top, then ``k1``, ``k2``, ``k3``. Each row carries twin linear y-axes, both anchored at 0:

    * Left (blue): utterance share = ``k{key}_utts / utterance_count``
    * Right (orange): word share = ``k{key}_matches / total_words``

    Both series are drawn as solid lines with translucent 95% Wilson
    score CI ribbons. Chambers are pooled — chamber contrast is
    :func:`plot_baseline_corpus`'s job.

    Method note (paper's methods section, in short form):

    * The corpus is a census of Riksdag speech 1900–1940, so the shares
      are exact descriptions of the source material — not estimates from
      a sample. The Wilson intervals are a conventional visual aid for
      readers who read the figure as inferential; they are narrow
      because n per year is very large.
    * Wilson assumes i.i.d. Bernoulli trials. Utterances are nested in
      speakers, so a speaker-clustered bootstrap would give intervals
      ~3–5× wider. Wilson is chosen for the paper because it is
      textbook, closed-form, deterministic, and easy to explain to
      readers without statistical training.
    * The linear (not log) y-axes with both starting at 0 preserve the
      visual weight of the intervals; a log scale flattened them below
      readability in an earlier draft.

    See docs/word_frequencies.md for the CSV schema. Cross-K counting
    semantics for ``k_*_matches`` are documented on
    :func:`plot_word_share_by_chamber`.
    """
    style = _style(style)
    df = _frame(df)

    utt_cols = ["k1_utts", "k2_utts", "k3_utts", "k_all_utts"]
    word_cols = ["k1_matches", "k2_matches", "k3_matches", "k_all_matches"]
    denom_cols = ["utterance_count", "total_words"]
    g = (
        df.groupby("year", as_index=False)[utt_cols + word_cols + denom_cols]
        .sum()
        .sort_values("year")
    )

    patterns = ("k_all", "k1", "k2", "k3")
    fig, axes = plt.subplots(
        4, 1, sharex=True, figsize=(style.figsize[0], style.figsize[1] * 2.4)
    )

    twins: list[Axes] = []
    for i, (ax, key) in enumerate(zip(axes, patterns)):
        ax_r = _draw_share_panel(ax, g, key, style, show_labels=(i == 0))
        twins.append(ax_r)
        _add_suffrage_marker(ax, style)
        ax.set_title(style.kvinna_pattern_labels[key], loc="left", fontsize=9)
        ax.grid(**style.grid)

    axes[-1].set_ylabel("Utterance share", color=style.share_colors["utts"])
    twins[-1].set_ylabel("Word share", color=style.share_colors["words"])
    _year_axis(axes[-1])

    lines_l, labels_l = axes[0].get_legend_handles_labels()
    lines_r, labels_r = twins[0].get_legend_handles_labels()
    axes[0].legend(
        lines_l + lines_r,
        labels_l + labels_r,
        loc="upper right",
        frameon=False,
        fontsize=7,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 1b (paper variant) — kvinna_all net composition (UpSet + partition)
# ---------------------------------------------------------------------------


def _draw_upset_block(
    fig: Figure,
    gridspec_cell,
    subsets: pd.DataFrame,
    count_col: str,
    title: str,
    style: PlotStyle,
) -> None:
    """Draw one UpSet plot inside a gridspec cell.

    ``subsets`` must have columns ``k1, k2, k3, <count_col>`` — exactly
    7 rows (one per non-empty subset of the three K sets). Uses
    ``upsetplot`` via the piecewise ``plot_matrix`` / ``plot_intersections``
    / ``plot_totals`` API so we can control the layout precisely; a
    ``show_counts=True`` bug on upsetplot 0.9.0 + matplotlib 3.11 is
    worked around by drawing intersection-count labels manually.
    """
    from upsetplot import UpSet, from_indicators

    inner = gridspec_cell.subgridspec(
        2,
        2,
        width_ratios=[1, 4],
        height_ratios=[3, 1],
        wspace=0.25,
        hspace=0.08,
    )
    ax_matrix = fig.add_subplot(inner[1, 1])
    ax_intersections = fig.add_subplot(inner[0, 1], sharex=ax_matrix)
    ax_totals = fig.add_subplot(inner[1, 0], sharey=ax_matrix)

    upset_data = from_indicators(["k1", "k2", "k3"], subsets)[[count_col]]
    upset = UpSet(
        upset_data,
        subset_size="sum",
        sum_over=count_col,
        sort_by="cardinality",
        sort_categories_by="input",
        show_counts=False,
        facecolor=style.upset_bar_color,
    )
    upset.plot_matrix(ax_matrix)
    upset.plot_intersections(ax_intersections)
    upset.plot_totals(ax_totals)
    ax_intersections.set_xlim(ax_matrix.get_xlim())
    ax_totals.set_ylim(ax_matrix.get_ylim())
    ax_intersections.tick_params(labelbottom=False)
    ax_matrix.set_yticks([0, 1, 2], labels=["K1", "K2", "K3"])

    for bar in ax_intersections.patches:
        h = bar.get_height()
        if h > 0:
            ax_intersections.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{int(h):,}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax_intersections.set_title(title, fontsize=10, loc="left")


def plot_kvinna_net_composition(
    words_subsets: pd.DataFrame | None = None,
    utterance_subsets: pd.DataFrame | None = None,
    *,
    style: PlotStyle | None = None,
) -> Figure:
    """Static methodological-transparency figure for kvinna_all.

    Two stacked UpSet plots. The top panel breaks word-match tokens
    into the 7 non-empty subsets of {K1, K2, K3}; the bottom panel
    does the same for utterances. Reader sees at a glance which
    patterns catch unique material and how they overlap — the answer
    to *"why kvinna_all and not just K1?"*.

    Aggregated over 1900–1940; static, not temporal. Both UpSets use
    the ``upsetplot`` library (Lex et al., 2014) via a piecewise draw
    into a nested gridspec so layout stays under our control.
    """

    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)

        from .data import read_kvinna_utterance_subsets, read_kvinna_word_subsets

        style = _style(style)
        if words_subsets is None:
            words_subsets = read_kvinna_word_subsets()
        if utterance_subsets is None:
            utterance_subsets = read_kvinna_utterance_subsets()

        fig = plt.figure(figsize=(style.figsize[0] * 1.4, style.figsize[1] * 3.0))
        outer = fig.add_gridspec(2, 1, hspace=0.35)

        _draw_upset_block(
            fig,
            outer[0, 0],
            words_subsets,
            "word_match_count",
            title="Word matches (tokens)",
            style=style,
        )
        _draw_upset_block(
            fig,
            outer[1, 0],
            utterance_subsets,
            "utterance_count",
            title="Utterances",
            style=style,
        )

        fig.suptitle(
            "Composition of the kvinna_all keyword net (1900–1940)", fontsize=11
        )
    return fig


# ---------------------------------------------------------------------------
# Plot 2 — utterance share, per chamber
# ---------------------------------------------------------------------------


def plot_kvinna_all_by_chamber(
    df: pd.DataFrame | None = None,
    *,
    mode: str = "counts",
    smoothing: int | Iterable[int] = 1,
    style: PlotStyle | None = None,
) -> Figure | list[Figure]:
    """Kvinna_all activity over time — two stacked panels × chamber.

    Layout: utterances (top panel) and word matches (bottom panel),
    sharing the x-axis. Each panel has a single y-axis with one line
    per chamber, so chamber-vs-chamber comparison is the primary
    in-panel read; the utterance/word complementarity comes from
    scanning between panels along the shared time axis.

    Two modes:

    - ``mode="counts"`` (default): top y-axis = ``k_all_utts``, bottom
      y-axis = ``k_all_matches`` — raw counts per chamber per year.
    - ``mode="share"``: top y-axis = ``k_all_utts / utterance_count``,
      bottom y-axis = ``k_all_matches / total_words`` — same shape,
      normalised by the per-year corpus size.

    ``smoothing`` applies a centered rolling mean of the given window
    length (in years) within each chamber:

    - ``smoothing=1`` (default): no smoothing.
    - ``smoothing=int``: single figure with that window applied.
    - ``smoothing=Iterable[int]``: returns a list of figures, one per
      unique window sorted ascending. Duplicate windows are collapsed.

    Note on token counting: ``k_all_matches`` uses within-group dedup
    with cross-group overlap allowed.
    Utterance denominator (share mode) includes unknown-speaker rows —
    see ``read_unknown_share`` for the size of that bucket per year.
    """
    # Iterable dispatch happens before mode validation so the caller
    # sees a single ValueError from the underlying single-window call,
    # not one per window.
    if not isinstance(smoothing, int):
        windows = sorted(set(int(w) for w in smoothing))
        return [
            plot_kvinna_all_by_chamber(df=df, mode=mode, smoothing=w, style=style)
            for w in windows
        ]

    if mode not in ("counts", "share"):
        raise ValueError(f"mode must be 'counts' or 'share', got {mode!r}")
    if smoothing < 1:
        raise ValueError(f"smoothing window must be >= 1, got {smoothing}")

    style = _style(style)
    df = _frame(df)

    g = (
        df.groupby(["year", "chamber"], as_index=False)[
            ["k_all_utts", "utterance_count", "k_all_matches", "total_words"]
        ]
        .sum()
        .sort_values(["chamber", "year"])
    )
    if mode == "counts":
        g["utt_y"] = g["k_all_utts"]
        g["word_y"] = g["k_all_matches"]
        utt_label = "Utterances (k_all)"
        word_label = "Word matches (k_all)"
        title = "How much of Riksdag talk is about women? (kvinna_all, counts)"
    else:
        g["utt_y"] = g["k_all_utts"] / g["utterance_count"]
        g["word_y"] = g["k_all_matches"] / g["total_words"]
        utt_label = "Utterance share (k_all)"
        word_label = "Word share (k_all)"
        title = "How much of Riksdag talk is about women? (kvinna_all, share)"

    # Keep the raw (unsmoothed) series so we can draw it as a
    # dashed low-alpha background layer when smoothing is applied.
    g["utt_raw"] = g["utt_y"]
    g["word_raw"] = g["word_y"]

    if smoothing > 1:
        # Strict centered rolling mean: min_periods=window drops the
        # edges (first/last floor(window/2) years become NaN), so every
        # plotted point is a true full-window average.
        for col in ("utt_y", "word_y"):
            g[col] = g.groupby("chamber", group_keys=False)[col].transform(
                lambda s: s.rolling(
                    window=smoothing, center=True, min_periods=smoothing
                ).mean()
            )
        title += f" — {smoothing}-yr rolling mean"

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(style.figsize[0], style.figsize[1] * 1.6)
    )
    for chamber, sub in g.groupby("chamber"):
        colour = style.chamber_colors.get(chamber)
        # Raw background — only when the main lines are smoothed
        # (otherwise raw == smoothed and we'd double-draw).
        if smoothing > 1:
            ax_top.plot(
                sub["year"],
                sub["utt_raw"],
                color=colour,
                linestyle="--",
                alpha=0.3,
                linewidth=0.9,
            )
            ax_bot.plot(
                sub["year"],
                sub["word_raw"],
                color=colour,
                linestyle="--",
                alpha=0.3,
                linewidth=0.9,
            )
        ax_top.plot(
            sub["year"],
            sub["utt_y"],
            label=f"Chamber {chamber}",
            color=colour,
        )
        ax_bot.plot(
            sub["year"],
            sub["word_y"],
            label=f"Chamber {chamber}",
            color=colour,
        )

    for ax in (ax_top, ax_bot):
        _add_suffrage_marker(ax, style)
        ax.set_ylim(bottom=0)
        ax.grid(**style.grid)
    ax_top.set_ylabel(utt_label)
    ax_bot.set_ylabel(word_label)
    ax_top.set_title(title)
    _year_axis(ax_bot)
    ax_top.legend(loc="best", frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 4 — K-category composition
# ---------------------------------------------------------------------------


def plot_category_composition(
    df: pd.DataFrame | None = None,
    *,
    mode: str = "utts",
    style: PlotStyle | None = None,
) -> Figure:
    """K1/K2/K3 as shares of the woman-talk itself.

    ``mode="utts"`` uses ``k{n}_utts`` numerators (utterance level).
    ``mode="words"`` uses ``k{n}_matches`` numerators (token level).
    Denominator is the sum of the three category counters — the
    composition sums to 1 by construction.

    Note on word-level counting semantics (``mode="words"``): ``k{n}_matches``
    and ``k_all_matches`` are computed with within-group deduplication and
    cross-group overlap allowed. A token matching multiple K categories counts
    once toward each K it hits, and once toward ``k_all_matches``.

    Interpretation: rising K3 signals a more differentiated,
    work-related vocabulary about women.

    Paper caveat (worth a footnote): K1 is near-zero in 1900–1902
    because the pattern ``kvinn*`` does not match the older Swedish
    spelling ``qvinn-``. The apparent rise of K1 across 1902–1905 is
    partly an orthographic transition, not a semantic shift; K2 (kin
    and generic terms including historic ``qv-`` forms) absorbs the
    early years.
    """
    style = _style(style)
    df = _frame(df)

    if mode == "utts":
        cols = ["k1_utts", "k2_utts", "k3_utts"]
        y_label = "Share of woman-utterances by category"
        title = "What kind of talk about women? (utterance level)"
    elif mode == "words":
        cols = ["k1_matches", "k2_matches", "k3_matches"]
        y_label = "Share of woman-word matches by category"
        title = "What kind of talk about women? (token level)"
    else:
        raise ValueError(f"mode must be 'utts' or 'words', got {mode!r}")

    g = df.groupby("year", as_index=False)[cols].sum().sort_values("year")
    total = g[cols].sum(axis=1)

    fig, ax = plt.subplots(figsize=style.figsize)
    for k, col in zip(("k1", "k2", "k3"), cols):
        ax.plot(
            g["year"],
            g[col] / total,
            label=style.category_labels[k],
            color=style.category_colors[k],
        )

    _add_suffrage_marker(ax, style)
    _year_axis(ax)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    _apply_axis_extras(ax, style)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 5 — utterance share by speaker gender
# ---------------------------------------------------------------------------


def _gender_frame(df: pd.DataFrame, num_col: str, denom_col: str) -> pd.DataFrame:
    """Per (year, gender) numerator/denominator/share for man and woman."""
    known = df[df["gender"].isin(("man", "woman"))]
    g = (
        known.groupby(["year", "gender"], as_index=False)[[num_col, denom_col]]
        .sum()
        .sort_values(["gender", "year"])
    )
    g["share"] = g[num_col] / g[denom_col]
    return g


def _gender_source_frame(df: pd.DataFrame, num_col: str) -> pd.DataFrame:
    """Per-year man / woman / unknown share of ``num_col``.

    Man+woman shares use a known-gender denominator, so they sum to 1.0
    per year. ``unknown_over_known`` expresses unknown-gender volume as
    a ratio against the same denominator so it can be drawn as a band
    stacked above y=1 on the same axis.
    """
    d = df.copy()
    d["gender"] = d["gender"].fillna("unknown")
    by = (
        d.groupby(["year", "gender"], as_index=False)[num_col]
        .sum()
        .pivot(index="year", columns="gender", values=num_col)
        .fillna(0)
    )
    for col in ("man", "woman", "unknown"):
        if col not in by.columns:
            by[col] = 0
    known = by["man"] + by["woman"]
    out = pd.DataFrame(
        {
            "year": by.index,
            "man": by["man"].values,
            "woman": by["woman"].values,
            "unknown": by["unknown"].values,
        }
    ).reset_index(drop=True)
    out["man_share"] = out["man"] / known.values
    out["woman_share"] = out["woman"] / known.values
    out["unknown_over_known"] = out["unknown"] / known.values
    return out


def plot_utterance_share_by_gender(
    df: pd.DataFrame | None = None, *, style: PlotStyle | None = None
) -> Figure:
    """Utterance share by speaker gender.

    ``k_all_utts / utterance_count`` per (year, gender) for known-gender
    speakers only. Women's line begins post-1922 and is annotated with
    N where the count falls below ``style.small_n_threshold`` — the
    small-N reality is a substantive part of the argument.
    """
    style = _style(style)
    df = _frame(df)

    g = _gender_frame(df, "k_all_utts", "utterance_count")

    fig, ax = plt.subplots(figsize=style.figsize)
    for gender, sub in g.groupby("gender"):
        ax.plot(
            sub["year"],
            sub["share"],
            label=gender.capitalize(),
            color=style.gender_colors.get(gender),
            marker="o" if gender == "woman" else None,
            markersize=3 if gender == "woman" else 0,
        )
        if gender == "woman":
            _annotate_small_n(
                ax, sub["year"], sub["share"], sub["utterance_count"], style
            )

    _add_suffrage_marker(ax, style)
    _year_axis(ax)
    ax.set_ylabel("Utterance share (K-any)")
    ax.set_title("Who talks about women? Utterance-level share by speaker gender")
    _apply_axis_extras(ax, style)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 5b — gender source of woman-utterances (flipped-share complement)
# ---------------------------------------------------------------------------


def plot_utterance_source_by_gender(
    df: pd.DataFrame | None = None, *, style: PlotStyle | None = None
) -> Figure:
    """Gender source of woman-utterances.

    **Not used in the paper.** See ``docs/gender_source_experiment.md``
    for why the flipped-share view was tried and dropped (peak woman
    band ≈ 20 %; the arithmetic is dominated by chamber composition,
    so the plot doesn't say anything Plot 5 doesn't already say more
    legibly).

    The inverse of Plot 5: instead of asking what share of each gender's
    own speech is about women, this asks — of all woman-utterances
    (``k_all_utts``) in a year, what share came from each gender?

    Man+woman shares of the known-gender ``k_all_utts`` total are stacked
    to 1.0. Unknown-gender is drawn as a band stacked above y=1 at
    ``unknown / (man + woman)`` so the reader sees the volume of
    woman-talk sitting outside the man/woman decomposition without that
    volume distorting the two argument-lines.
    """
    style = _style(style)
    df = _frame(df)

    g = _gender_source_frame(df, "k_all_utts")

    fig, ax = plt.subplots(figsize=style.figsize)
    ax.stackplot(
        g["year"],
        g["man_share"],
        g["woman_share"],
        g["unknown_over_known"],
        labels=["Man", "Woman", "Unknown gender"],
        colors=[
            style.gender_colors.get("man"),
            style.gender_colors.get("woman"),
            style.gender_colors.get("unknown"),
        ],
    )
    ax.axhline(1.0, color="#333333", linewidth=0.8, alpha=0.6)

    counts = (
        df[df["gender"] == "woman"]
        .groupby("year")["utterance_count"]
        .sum()
        .reindex(g["year"])
        .values
    )
    top_of_woman = g["man_share"] + g["woman_share"]
    _annotate_small_n(ax, g["year"], top_of_woman, counts, style)

    _add_suffrage_marker(ax, style)
    _year_axis(ax)
    ax.set_ylabel("Share of woman-utterances")
    ax.set_title(
        "Who speaks about women? Gender source of woman-utterances "
        "(man+woman = 1.0; unknown stacked above)"
    )
    _apply_axis_extras(ax, style)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 6 — word share by speaker gender
# ---------------------------------------------------------------------------


def plot_word_share_by_gender(
    df: pd.DataFrame | None = None, *, style: PlotStyle | None = None
) -> Figure:
    """Word-density complement to Plot 5.

    ``k_all_matches / total_words`` per (year, gender) for known-gender
    speakers only. Together with Plot 5, this shows whether women speak
    about women both more often *and* more densely, or only one of the
    two.
    """
    style = _style(style)
    df = _frame(df)

    g = _gender_frame(df, "k_all_matches", "total_words")

    fig, ax = plt.subplots(figsize=style.figsize)
    for gender, sub in g.groupby("gender"):
        ax.plot(
            sub["year"],
            sub["share"],
            label=gender.capitalize(),
            color=style.gender_colors.get(gender),
            marker="o" if gender == "woman" else None,
            markersize=3 if gender == "woman" else 0,
        )
        if gender == "woman":
            # small-N uses utterance_count as the substantive threshold
            # (that's what "few observations" means for this series);
            # total_words is not the right N for eyeballing sparsity.
            counts = (
                df[df["gender"] == "woman"]
                .groupby("year")["utterance_count"]
                .sum()
                .reindex(sub["year"])
                .values
            )
            _annotate_small_n(ax, sub["year"], sub["share"], counts, style)

    _add_suffrage_marker(ax, style)
    _year_axis(ax)
    ax.set_ylabel("Word share (K-any)")
    ax.set_title(
        "How densely do women vs. men talk about women? Token-level share by gender"
    )
    _apply_axis_extras(ax, style)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 5+6 — utterance share (left axis) + word share (right axis) combined
# ---------------------------------------------------------------------------


def plot_gender_share_utt_and_word(
    df: pd.DataFrame | None = None, *, style: PlotStyle | None = None
) -> Figure:
    """Paper-figure merger of Plots 5 and 6 on a shared time axis.

    Left axis (solid): utterance share ``k_all_utts / utterance_count``
    per (year, gender), known-gender only — how *often* each group
    hits a kvinna keyword.

    Right axis (dashed): word share ``k_all_matches / total_words``
    per (year, gender), known-gender only — how *densely* each group
    hits kvinna keywords when they do speak.

    Same colour per gender across axes; series distinguished by line
    style so both axes are readable without relying on colour alone
    (mirrors :func:`plot_baseline_corpus`'s dual-axis convention).
    Small-N annotations sit on the utterance-share (left) series since
    ``utterance_count`` is the substantive N for both metrics.

    Plots 5 and 6 are kept as standalone figures for the companion site
    (individual axes read more cleanly at that grain); this merger is
    the paper figure.
    """
    style = _style(style)
    df = _frame(df)

    utt = _gender_frame(df, "k_all_utts", "utterance_count")
    word = _gender_frame(df, "k_all_matches", "total_words")

    fig, ax = plt.subplots(figsize=style.figsize)
    ax_r = ax.twinx()

    for gender in ("man", "woman"):
        colour = style.gender_colors.get(gender)
        u_sub = utt[utt["gender"] == gender]
        w_sub = word[word["gender"] == gender]

        ax.plot(
            u_sub["year"],
            u_sub["share"],
            label=f"{gender.capitalize()} — utterance share",
            color=colour,
            marker="o" if gender == "woman" else None,
            markersize=3 if gender == "woman" else 0,
        )
        ax_r.plot(
            w_sub["year"],
            w_sub["share"],
            label=f"{gender.capitalize()} — word share",
            color=colour,
            linestyle="--",
            marker="s" if gender == "woman" else None,
            markersize=3 if gender == "woman" else 0,
        )

        if gender == "woman":
            _annotate_small_n(
                ax, u_sub["year"], u_sub["share"], u_sub["utterance_count"], style
            )

    _add_suffrage_marker(ax, style)
    _year_axis(ax)
    ax.set_ylim(bottom=0)
    ax_r.set_ylim(bottom=0)
    ax.set_ylabel("Utterance share (K-any) — solid")
    ax_r.set_ylabel("Word share (K-any) — dashed")
    ax.set_title(
        "How much do men and women talk about women? "
        "Utterance share (left) + word density (right)"
    )
    ax.grid(**style.grid)
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(
        lines_l + lines_r,
        labels_l + labels_r,
        loc="best",
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 6b — gender source of woman-keyword tokens (flipped-share complement)
# ---------------------------------------------------------------------------


def plot_word_source_by_gender(
    df: pd.DataFrame | None = None, *, style: PlotStyle | None = None
) -> Figure:
    """Gender source of woman-keyword tokens.

    **Not used in the paper.** See ``docs/gender_source_experiment.md``
    (same reason as Plot 5b — the flipped source-share is dominated by
    chamber composition and adds no signal over Plot 6).

    Token-level complement to Plot 5b: instead of speaker-density, the
    denominator is the year's total ``k_all_matches`` across known-
    gender speakers, and each band is that gender's share of woman-
    keyword tokens. Unknown-gender tokens are stacked above y=1 at
    ``unknown / (man + woman)`` on the same axis.

    Read alongside Plot 5b: if the woman band is proportionally larger
    here than there, women speak about women more *densely* (more
    keywords per utterance) than the utterance-level split alone
    suggests.
    """
    style = _style(style)
    df = _frame(df)

    g = _gender_source_frame(df, "k_all_matches")

    fig, ax = plt.subplots(figsize=style.figsize)
    ax.stackplot(
        g["year"],
        g["man_share"],
        g["woman_share"],
        g["unknown_over_known"],
        labels=["Man", "Woman", "Unknown gender"],
        colors=[
            style.gender_colors.get("man"),
            style.gender_colors.get("woman"),
            style.gender_colors.get("unknown"),
        ],
    )
    ax.axhline(1.0, color="#333333", linewidth=0.8, alpha=0.6)

    # Small-N uses utterance_count as the substantive threshold
    # (matches Plot 6's rationale).
    counts = (
        df[df["gender"] == "woman"]
        .groupby("year")["utterance_count"]
        .sum()
        .reindex(g["year"])
        .values
    )
    top_of_woman = g["man_share"] + g["woman_share"]
    _annotate_small_n(ax, g["year"], top_of_woman, counts, style)

    _add_suffrage_marker(ax, style)
    _year_axis(ax)
    ax.set_ylabel("Share of woman-keyword tokens")
    ax.set_title(
        "Gender source of woman-keyword tokens (man+woman = 1.0; unknown stacked above)"
    )
    _apply_axis_extras(ax, style)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 5c — women's participation share vs. their initiation share
# ---------------------------------------------------------------------------


def plot_woman_chime_vs_start(
    starters_df: pd.DataFrame | None = None,
    word_freq_df: pd.DataFrame | None = None,
    *,
    max_gap: int = 1,
    min_arc_length: int = 2,
    style: PlotStyle | None = None,
) -> Figure:
    """Women's utterance-share vs. their starter-share of woman-talk.

    Two lines per year, both computed over a known-gender denominator:

    - **Share of all woman-utterances** — women's ``k_all_utts`` /
      (man + woman ``k_all_utts``). All utterances that hit a kvinna
      keyword (starters *and* continuations), split by speaker gender.
    - **Share of arc starters** — women's topic-arc starters / (man +
      woman starters). A "starter" is a kvinna utterance where the
      preceding ``max_gap + 1`` utterances in the chamber chain all had
      no kvinna hit, AND (default) at least one utterance within the
      next ``max_gap + 1`` slots is kvinna — i.e., the arc reaches
      length ≥ 2. This excludes isolated single-utterance mentions,
      which dominate the naïve count (~75 % of raw "starters" under
      ``min_arc_length=1``). ``max_gap=1`` further tolerates a single
      non-kvinna interjection inside an ongoing arc.

    Denominator note: both shares are cohort-shares (women's slice of
    the *chamber-wide* pool). This differs from the per-speaker
    scatter :func:`plot_speaker_starts_vs_chimes_by_gender`, which
    normalises each speaker's own starters/chime-ins against their
    total ``utterance_count`` — so "women over-index on starting"
    (this plot) and "each woman has chime_rate > starter_rate" (that
    plot) are both true simultaneously; they're answering different
    questions.

    The gap ``starter_share − utt_share`` is the paper-worthy finding
    under the current default (``min_arc_length=2``, ``max_gap=1``):
    across 1922–1940 women's starter-share exceeds their utterance-
    share in 13 of 19 post-suffrage years — i.e., women *kicked off*
    substantive woman-talk arcs more (or as often) as they *chimed in*
    on them. The peak is 1924 (Nya Giftermålsbalken / Marriage Bill
    year), where women held ~24 % of starters against ~8 % of
    utterances — a ~3× over-index versus their share of woman-talk.

    Historical note on parameter sensitivity:
      - ``min_arc_length=1`` (raw starters, single-utterance mentions
        included) reverses the picture: utterance-share > starter-share
        in most years. That was the earlier reading; the L=2 filter
        (real conversations, not passing mentions) is what the paper
        argues from.
      - ``max_gap=0`` (strict, no interjection tolerated) sharpens the
        1924 spike further (~17× over-index) but also drops arcs that
        include a single non-kvinna interjection. ``max_gap=1`` is
        the shipped default because it tolerates the way real chamber
        speech unfolds.

    See ``docs/gender_source_experiment.md`` §L=1 vs L=2 for the full
    recomputation table across all four (L, gap) combinations.
    """
    style = _style(style)
    if word_freq_df is None:
        word_freq_df = read_word_frequencies()
    if starters_df is None:
        from .data import read_topic_arc_starters

        starters_df = read_topic_arc_starters(
            max_gap=max_gap, min_arc_length=min_arc_length
        )

    utt = _gender_source_frame(word_freq_df, "k_all_utts")[
        ["year", "woman_share"]
    ].rename(columns={"woman_share": "utt_share"})
    st = _gender_source_frame(starters_df, "starters")[["year", "woman_share"]].rename(
        columns={"woman_share": "starter_share"}
    )
    merged = utt.merge(st, on="year", how="outer").sort_values("year")
    post = merged[merged["year"] >= style.suffrage_year]

    fig, ax = plt.subplots(figsize=style.figsize)
    woman_color = style.gender_colors.get("woman")
    ax.plot(
        post["year"],
        post["utt_share"],
        label="Share of all woman-utterances",
        color=woman_color,
        marker="o",
        markersize=3,
    )
    ax.plot(
        post["year"],
        post["starter_share"],
        label="Share of arc starters",
        color=woman_color,
        linestyle="--",
        marker="s",
        markersize=3,
    )
    ax.fill_between(
        post["year"],
        post["utt_share"],
        post["starter_share"],
        color=woman_color,
        alpha=0.15,
    )

    _add_suffrage_marker(ax, style)
    _year_axis(ax)
    ax.set_ylabel("Women's share of woman-talk")
    definition_bits = []
    if max_gap:
        definition_bits.append(f"max_gap={max_gap}")
    if min_arc_length != 1:
        definition_bits.append(f"min_arc_length={min_arc_length}")
    suffix = f" (starter: {', '.join(definition_bits)})" if definition_bits else ""
    ax.set_title(
        "Chime in vs. kick off: women's utterance-share vs. starter-share" + suffix
    )
    _apply_axis_extras(ax, style)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 7 — party bloc lens
# ---------------------------------------------------------------------------


def plot_party_share(
    df: pd.DataFrame | None = None,
    *,
    parties: Iterable[str] = ("S", "H", "L", "Bf", "K"),
    mapping: Mapping[str, Iterable[str]] | None = None,
    style: PlotStyle | None = None,
) -> Figure:
    """Utterance share touching on women, per party bloc.

    ``parties`` selects which blocs to plot from ``mapping``
    (default :data:`DEFAULT_PARTY_MAPPING`). Rows whose ``party`` is
    outside the selected blocs are dropped — no "Other" pooling is
    done here; the caller can build that separately if needed.

    Method caveat (for the paper's methods section): party labels are
    those recorded in ``persons.sqlite`` at the utterance's date;
    successors and renamings within the same bloc are pooled per
    :data:`DEFAULT_PARTY_MAPPING`.

    Reading caveat: the K bloc (left of Socialdemokraterna) is very
    small before ~1917 and shows high year-to-year variance. Pass
    ``parties=("S","H","L","Bf")`` to drop it, or filter years where a
    bloc has too few utterances to give a meaningful share, if the
    paper's argument doesn't rest on the K line.
    """
    style = _style(style)
    df = _frame(df)
    mapping = mapping if mapping is not None else DEFAULT_PARTY_MAPPING

    # Restrict mapping to the requested blocs so a caller who passes
    # `parties=("S", "H")` gets exactly two lines.
    selected = {b: mapping[b] for b in parties}
    d = _reduce_parties(df, selected)

    g = (
        d.groupby(["year", "bloc"], as_index=False)[["k_all_utts", "utterance_count"]]
        .sum()
        .sort_values(["bloc", "year"])
    )
    g["share"] = g["k_all_utts"] / g["utterance_count"]

    fig, ax = plt.subplots(figsize=style.figsize)
    for bloc in parties:
        sub = g[g["bloc"] == bloc]
        if sub.empty:
            continue
        ax.plot(
            sub["year"],
            sub["share"],
            label=style.party_labels.get(bloc, bloc),
            color=style.party_colors.get(bloc),
        )

    _add_suffrage_marker(ax, style)
    _year_axis(ax)
    ax.set_ylabel("Utterance share (k_all_utts / utterance_count)")
    ax.set_title("Who politically drives talk about women? Utterance share by bloc")
    _apply_axis_extras(ax, style)
    fig.tight_layout()
    return fig


def plot_party_share_stacked(
    df: pd.DataFrame | None = None,
    *,
    parties: Iterable[str] = ("S", "H", "L", "Bf", "K"),
    mapping: Mapping[str, Iterable[str]] | None = None,
    include_other: bool = True,
    style: PlotStyle | None = None,
) -> Figure:
    """Contribution of each bloc to all woman-mentioning utterances per year.

    Complements :func:`plot_party_share`: that plot shows each bloc's
    *own* rate (``k_all_utts / utterance_count`` within the bloc), which
    flattens because most blocs mention women at similar per-utterance
    rates. This one asks the reverse question — of every utterance that
    hit a woman keyword in year Y, what fraction came from each bloc —
    and so it also reflects each bloc's volume in the Riksdag.

    Denominator is the year's total ``k_all_utts`` across all covered
    blocs (plus "Other" if ``include_other=True``); shares stack to 1.0.
    Rows whose ``party`` is outside the selected blocs are pooled into
    "Other" so nothing is silently dropped from the denominator; pass
    ``include_other=False`` to restrict the denominator to the selected
    blocs only.
    """
    style = _style(style)
    df = _frame(df)
    mapping = mapping if mapping is not None else DEFAULT_PARTY_MAPPING

    selected = {b: mapping[b] for b in parties}
    lookup = {name: bloc for bloc, names in selected.items() for name in names}

    d = df.copy()
    d["bloc"] = d["party"].map(lookup)
    if include_other:
        d["bloc"] = d["bloc"].fillna("Other")
    else:
        d = d.dropna(subset=["bloc"])

    g = (
        d.groupby(["year", "bloc"], as_index=False)["k_all_utts"]
        .sum()
        .pivot(index="year", columns="bloc", values="k_all_utts")
        .fillna(0.0)
        .sort_index()
    )
    year_totals = g.sum(axis=1).replace(0.0, pd.NA)
    shares = g.div(year_totals, axis=0).fillna(0.0)

    ordered = [b for b in parties if b in shares.columns]
    if include_other and "Other" in shares.columns:
        ordered.append("Other")

    other_color = "#cccccc"
    colors = [
        other_color if bloc == "Other" else style.party_colors.get(bloc)
        for bloc in ordered
    ]
    labels = [
        "Other / unmapped" if bloc == "Other" else style.party_labels.get(bloc, bloc)
        for bloc in ordered
    ]

    fig, ax = plt.subplots(figsize=style.figsize)
    ax.stackplot(
        shares.index.values,
        [shares[b].values for b in ordered],
        labels=labels,
        colors=colors,
    )
    ax.set_ylim(0.0, 1.0)

    _add_suffrage_marker(ax, style)
    _year_axis(ax)
    ax.set_ylabel("Share of woman-mentioning utterances")
    ax.set_title(
        "Who contributes to talk about women? Bloc share of woman-utterances per year"
    )
    _apply_axis_extras(ax, style)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 8 — per-speaker share distribution + top-N speakers
# ---------------------------------------------------------------------------


def _aggregate_speaker_shares(
    speakers: pd.DataFrame, min_utterances: int
) -> pd.DataFrame:
    """Collapse speakers.csv.gz over years to per-speaker totals + share.

    Filters to known ``who`` (non-empty) and known ``gender`` (``man`` or
    ``woman``), then applies the ``min_utterances`` floor so per-speaker
    shares are stable enough to compare and rank.
    """
    known = speakers[
        (speakers["who"].astype(str) != "")
        & (speakers["gender"].isin(("man", "woman")))
    ]
    g = known.groupby(["who", "gender"], as_index=False)[
        ["utterance_count", "k_all_utts"]
    ].sum()
    g = g[g["utterance_count"] >= min_utterances].copy()
    g["share"] = g["k_all_utts"] / g["utterance_count"]
    return g


def plot_speaker_share_distribution(
    speakers: pd.DataFrame | None = None,
    *,
    min_utterances: int = 100,
    bins: int = 40,
    style: PlotStyle | None = None,
) -> Figure:
    """Histogram of per-speaker woman-mention rate, coloured by gender.

    Aggregates ``speakers.csv.gz`` per-speaker totals across all years,
    filters to speakers with ``utterance_count >= min_utterances`` so
    thin talkers don't dominate the tails, and plots two overlaid
    histograms (men in one colour, women in the other). Vertical
    reference lines mark the per-speaker mean share for each gender.

    Answers the "how much do the men-and-women distributions overlap?"
    question that a group-average plot (e.g.
    :func:`plot_utterance_share_by_gender`) hides. Men with woman-
    mention rates approaching or exceeding the women's mean are the
    interesting population for :func:`plot_top_male_speakers`.
    """
    style = _style(style)
    if speakers is None:
        speakers = read_speakers()
    g = _aggregate_speaker_shares(speakers, min_utterances)

    fig, ax = plt.subplots(figsize=style.figsize)
    for gender in ("man", "woman"):
        sub = g.loc[g["gender"] == gender, "share"]
        if sub.empty:
            continue
        ax.hist(
            sub,
            bins=bins,
            color=style.gender_colors.get(gender),
            alpha=0.55,
            label=f"{gender.capitalize()} (N={len(sub)})",
        )
        ax.axvline(
            float(sub.mean()),
            color=style.gender_colors.get(gender),
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label=f"{gender.capitalize()} mean share",
        )

    ax.set_xlabel("Per-speaker woman-mention share (k_all_utts / utterance_count)")
    ax.set_ylabel("Speakers")
    ax.set_title(
        f"Per-speaker woman-mention rate, by gender (min {min_utterances} utterances)"
    )
    _apply_axis_extras(ax, style)
    fig.tight_layout()
    return fig


def plot_top_male_speakers(
    totals: pd.DataFrame | None = None,
    *,
    top_n: int = 15,
    min_utterances: int = 100,
    style: PlotStyle | None = None,
) -> Figure:
    """Horizontal barplot of the top-N men by woman-mention share.

    Reads the shipped ``speaker_totals.csv.gz`` snapshot (via
    :func:`src.data.read_speaker_totals`, which builds the file from
    ``speakers.csv.gz`` + ``persons.sqlite`` on first access when the
    snapshot is missing), keeps men with ``utterance_count >=
    min_utterances``, sorts by share, plots the top ``top_n``.
    Reference lines mark the per-speaker mean share for men and for
    women so a reader can see how the tail compares to both group
    averages.

    Display names come from the ``name`` column of the snapshot (which
    joins ``persons.sqlite``). Rows with an empty name fall back to the
    ``who`` id so nothing is dropped from the ranking when the persons
    DB is unavailable.

    The point of the figure is narrative: it turns "some men talk about
    women a lot" into a named list a humanist reader can recognise.
    """
    style = _style(style)
    if totals is None:
        totals = read_speaker_totals()

    known = totals[totals["gender"].isin(("man", "woman"))].copy()
    eligible = known[known["utterance_count"] >= min_utterances]

    men = (
        eligible[eligible["gender"] == "man"]
        .sort_values("share", ascending=False)
        .head(top_n)
    )
    male_mean = float(eligible.loc[eligible["gender"] == "man", "share"].mean())
    women_shares = eligible.loc[eligible["gender"] == "woman", "share"]
    female_mean = float(women_shares.mean()) if not women_shares.empty else float("nan")

    def _label(row: pd.Series) -> str:
        name = str(row["name"]) if pd.notna(row["name"]) else ""
        return name if name else str(row["who"])

    labels = [_label(r) for _, r in men.iterrows()]

    fig, ax = plt.subplots(figsize=(style.figsize[0], style.figsize[1] * 1.2))
    positions = list(range(len(men)))
    ax.barh(
        positions,
        men["share"],
        color=style.gender_colors.get("man"),
        alpha=0.85,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.axvline(
        male_mean,
        color=style.gender_colors.get("man"),
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label=f"Male mean ({male_mean:.3f})",
    )
    if not pd.isna(female_mean):
        ax.axvline(
            female_mean,
            color=style.gender_colors.get("woman"),
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label=f"Female mean ({female_mean:.3f})",
        )

    ax.set_xlabel("Woman-mention share (k_all_utts / utterance_count)")
    ax.set_title(
        f"Top {len(men)} men by woman-mention share "
        f"(min {min_utterances} utterances, 1900–1940)"
    )
    ax.grid(**style.grid, axis="x")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 9 — per-speaker rate vs volume ("octopus" scatter)
# ---------------------------------------------------------------------------


_VALID_K_GROUPS = ("k_all", "k1", "k2", "k3")


def plot_speaker_rate_vs_volume(
    speakers: pd.DataFrame | None = None,
    word_totals: pd.DataFrame | None = None,
    *,
    mode: str = "utts",
    k_group: str = "k_all",
    year_range: tuple[int, int] | None = None,
    style: PlotStyle | None = None,
) -> Figure:
    """Per-speaker mention rate vs speech volume, coloured by gender.

    Also known as the "octopus" plot: at low denominators the radial
    1/n, 2/n, 3/n share curves fan out like tentacles. The tentacles
    are visible in ``mode='utts'`` where per-speaker utterance counts
    reach the single digits; in ``mode='words'`` the tentacles vanish
    because per-speaker token counts run 10²–10⁶.

    Modes:

    * ``utts``  — x = per-speaker utterance count (log), y =
      ``k{k_group}_utts / utterance_count``. Source: ``speakers.csv.gz``.
    * ``words`` — x = per-speaker total_words (log), y =
      ``k{k_group}_matches / total_words``. Source:
      ``speaker_word_totals.csv.gz``, built lazily on first read.

    ``k_group`` selects the numerator (``k_all``, ``k1``, ``k2``,
    ``k3``); ``year_range`` is an inclusive ``(y_min, y_max)`` filter
    applied to per-year rows before aggregation, so era slices (e.g.
    ``(1900, 1920)`` for pre-suffrage) share this one function.

    Substantive point the plot is designed to surface: individual men
    in the right tail sit at share levels comparable to the lower-
    share women — a within-group overlap that a group-average line
    plot cannot show. Coupled with :func:`plot_top_male_speakers`,
    which names those men, this is the natural companion visualisation.
    """
    if mode not in ("utts", "words"):
        raise ValueError(f"mode must be 'utts' or 'words', got {mode!r}")
    if k_group not in _VALID_K_GROUPS:
        raise ValueError(f"k_group must be one of {_VALID_K_GROUPS}, got {k_group!r}")
    style = _style(style)

    if mode == "utts":
        if speakers is None:
            speakers = read_speakers()
        source = speakers
        num_col = f"{k_group}_utts"
        denom_col = "utterance_count"
        x_label = "Utterances per speaker (log scale)"
        y_label = f"{k_group.upper()} utterance share ({num_col} / {denom_col})"
        volume_word = "utterance"
    else:
        if word_totals is None:
            word_totals = read_speaker_word_totals()
        source = word_totals
        num_col = f"{k_group}_matches"
        denom_col = "total_words"
        x_label = "Words per speaker (log scale)"
        y_label = f"{k_group.upper()} word share ({num_col} / {denom_col})"
        volume_word = "word"

    if year_range is not None:
        y0, y1 = year_range
        source = source[(source["year"] >= y0) & (source["year"] <= y1)]

    known = source[
        (source["who"].astype(str) != "") & (source["gender"].isin(("man", "woman")))
    ]
    agg = known.groupby(["who", "gender"], as_index=False)[[num_col, denom_col]].sum()
    agg = agg[agg[denom_col] > 0].copy()
    agg["share"] = agg[num_col] / agg[denom_col]

    agg_c = _attach_category(agg)

    fig, ax = plt.subplots(figsize=style.figsize)
    for gender, size, alpha in (("man", 8, 0.25), ("woman", 40, 0.9)):
        sub = agg_c[agg_c["gender"] == gender]
        colour = style.gender_colors.get(gender)
        if sub.empty:
            ax.scatter(
                [],
                [],
                s=size,
                alpha=alpha,
                color=colour,
                label=f"{gender.capitalize()} (N=0)",
            )
            continue
        # Split by marker so cross-plot identifiability (loud_man = X,
        # original_five = hexagon) is preserved on the octopus too.
        first = True
        for marker, msub in sub.groupby("marker"):
            is_highlight = marker in ("X", "h")
            ax.scatter(
                msub[denom_col],
                msub["share"],
                s=size * (2.0 if is_highlight else 1.0),
                alpha=min(1.0, alpha + (0.3 if is_highlight else 0.0)),
                color=colour,
                marker=marker,
                edgecolors="black" if is_highlight else "none",
                linewidths=0.5 if is_highlight else 0.0,
                label=(
                    f"{gender.capitalize()} (N={len(sub)}, "
                    f"mean share={sub['share'].mean():.4f})"
                )
                if first
                else None,
            )
            first = False
        ax.axhline(
            float(sub["share"].mean()),
            color=colour,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )

    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(bottom=0)
    era = (
        f", {year_range[0]}–{year_range[1]}"
        if year_range is not None
        else ", 1900–1940"
    )
    ax.set_title(
        f"Per-speaker {k_group.upper()} {volume_word} share "
        f"vs volume of speech, by gender{era}"
    )
    ax.grid(**style.grid)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 9b — per-speaker starter-rate vs chime-rate, coloured by gender
# ---------------------------------------------------------------------------


_DEFAULT_STARTS_VS_CHIMES_WINDOWS: tuple[tuple[int, int, str], ...] = (
    (1900, 1921, "Before (1900–1921)"),
    (1900, 1940, "Full (1900–1940)"),
    (1922, 1940, "After (1922–1940)"),
)


def plot_speaker_starts_vs_chimes_by_gender(
    speakers: pd.DataFrame | None = None,
    starters: pd.DataFrame | None = None,
    *,
    windows: Iterable[tuple[int, int, str]] = _DEFAULT_STARTS_VS_CHIMES_WINDOWS,
    min_utterances: int = 2,
    max_gap: int = 1,
    min_arc_length: int = 2,
    style: PlotStyle | None = None,
) -> Figure:
    """Per-speaker starter vs chime-in scatter, gendered, 2×3 grid.

    Rows: counts (top), rates (bottom). Cols: three timespans (default
    Before/Full/After suffrage). Panels within a row share x and y so
    growth over time is directly comparable; rows have independent
    scales because counts are integers and rates are fractions.

    Dot size ∝ √total utterance_count so heavy speakers (Lindhagen etc.)
    are visible as large dots — the "some people simply speak a lot"
    story that a fixed-size scatter would flatten. A follow-up can
    superimpose names on the counts row using ``m[["who", "starters",
    "chime_ins"]]`` where ``m`` matches the aggregation this function
    performs internally.

    The bottom-left cell (rates × before) is used entirely for the
    legend — women did not sit in the Riksdag before 1922, so that
    panel would only show the male cloud with no gender contrast to make.

    Gender-analog of :func:`plot_party_speaker_starts_vs_chimes` (per
    bloc, one panel each) and rate-space cousin of the octopus
    :func:`plot_speaker_rate_vs_volume` (volume × share).

    ``min_utterances`` drops speakers whose total utterance_count in
    the window is below the floor. Default is 2 to prune single-utterance
    dots that pin a rate to 0 or 1 (three post-1922 women fall here).
    Pass ``min_utterances=1`` to keep them and recover the "all 20 women
    engaged" count from ``docs/gender_source_experiment.md``.
    """
    import numpy as np
    from matplotlib.lines import Line2D

    style = _style(style)
    windows_list = list(windows)

    # Pre-load speakers/starters once with the widest window so each
    # per-window call to speaker_starts_chimes just re-filters in memory
    # instead of hitting the sqlite readers three times.
    if speakers is None:
        speakers = read_speakers()
    if starters is None:
        overall_y0 = min(w[0] for w in windows_list)
        overall_y1 = max(w[1] for w in windows_list)
        starters = read_speaker_starter_counts(
            years=range(overall_y0, overall_y1 + 1),
            max_gap=max_gap,
            min_arc_length=min_arc_length,
        )

    per_window = [
        (
            label,
            speaker_starts_chimes(
                speakers=speakers,
                starters=starters,
                year_range=(y0, y1),
                min_utterances=min_utterances,
            ),
        )
        for (y0, y1, label) in windows_list
    ]

    ncols = len(windows_list)
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(4.8 * ncols, 9.5),
        sharex="row",
        sharey="row",
        squeeze=False,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    # dot size ∝ √utterance_count with a modest base scale so a
    # 100-utt speaker → ~30 pts, a 10 000-utt speaker → ~300 pts.
    def _sizes(sub: pd.DataFrame) -> "np.ndarray":
        return np.sqrt(sub["utterance_count"].values.clip(min=1)) * 3.0

    for col, (label, m) in enumerate(per_window):
        for row, mode in enumerate(("counts", "rates")):
            ax = axes[row, col]

            # Bottom-left cell is reserved for the legend — draw nothing.
            if row == 1 and col == 0:
                continue

            xcol, ycol = (
                ("starters", "chime_ins")
                if mode == "counts"
                else ("starter_rate", "chime_rate")
            )

            mc = _attach_category(m)
            for gender, alpha in (("man", 0.30), ("woman", 0.9)):
                gsub = mc[mc["gender"] == gender]
                if gsub.empty:
                    continue
                for marker, msub in gsub.groupby("marker"):
                    # loud_man / original_five get sharper edges + a bit
                    # more visual weight to make cross-plot ID legible.
                    is_highlight = marker in ("X", "h")
                    ax.scatter(
                        msub[xcol],
                        msub[ycol],
                        s=_sizes(msub) * (1.3 if is_highlight else 1.0),
                        alpha=min(1.0, alpha + (0.25 if is_highlight else 0.0)),
                        color=style.gender_colors.get(gender),
                        marker=marker,
                        edgecolors="black" if is_highlight else "none",
                        linewidths=0.5 if is_highlight else 0.0,
                    )

            ax.grid(**style.grid)
            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel("Chime-in count" if mode == "counts" else "Chime-in rate")
            ax.set_xlabel("Starter count" if mode == "counts" else "Starter rate")

    # Row-level axis limits. Both rows have independent x/y scales
    # because counts (integers) and rates (fractions) map different
    # magnitudes on each axis: starter_rate < chime_rate, and
    # max(starters) < max(chime_ins) for the same reason.
    all_starters = pd.concat([m["starters"] for _, m in per_window if not m.empty])
    all_chime_ins = pd.concat([m["chime_ins"] for _, m in per_window if not m.empty])
    counts_x_hi = float(all_starters.max()) * 1.05 if not all_starters.empty else 1.0
    counts_y_hi = float(all_chime_ins.max()) * 1.05 if not all_chime_ins.empty else 1.0
    rates_x_hi, rates_y_hi = 0.4, 1.0

    for col in range(ncols):
        diag_hi = min(counts_x_hi, counts_y_hi)
        axes[0, col].plot(
            [0, diag_hi],
            [0, diag_hi],
            linestyle="--",
            color="#999999",
            linewidth=0.7,
            alpha=0.5,
        )
        axes[0, col].set_xlim(0, counts_x_hi)
        axes[0, col].set_ylim(0, counts_y_hi)

    for col in range(ncols):
        if col == 0:
            continue  # legend cell
        diag_hi = min(rates_x_hi, rates_y_hi)
        axes[1, col].plot(
            [0, diag_hi],
            [0, diag_hi],
            linestyle="--",
            color="#999999",
            linewidth=0.7,
            alpha=0.5,
        )
        axes[1, col].set_xlim(0, rates_x_hi)
        axes[1, col].set_ylim(0, rates_y_hi)

    # Legend cell: hide axes entirely, use the full space for the legend.
    legend_ax = axes[1, 0]
    legend_ax.set_axis_off()
    man_color = style.gender_colors.get("man")
    woman_color = style.gender_colors.get("woman")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=SPEAKER_MARKERS["man"],
            linestyle="",
            markersize=10,
            color=man_color,
            alpha=0.6,
            label="Man",
        ),
        Line2D(
            [0],
            [0],
            marker=SPEAKER_MARKERS["loud_man"],
            linestyle="",
            markersize=12,
            color=man_color,
            alpha=0.95,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Loud man (top-15 by k_all_utts count)",
        ),
        Line2D(
            [0],
            [0],
            marker=SPEAKER_MARKERS["woman"],
            linestyle="",
            markersize=14,
            color=woman_color,
            alpha=0.95,
            label="Woman",
        ),
        Line2D(
            [0],
            [0],
            marker=SPEAKER_MARKERS["original_five"],
            linestyle="",
            markersize=16,
            color=woman_color,
            alpha=0.95,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label="Original 5 (1922 women)",
        ),
        Line2D(
            [0],
            [0],
            marker=SPEAKER_MARKERS["unknown"],
            linestyle="",
            markersize=10,
            color="#888888",
            alpha=0.6,
            label="Unknown",
        ),
        Line2D(
            [0],
            [0],
            linestyle="--",
            color="#999999",
            linewidth=0.7,
            alpha=0.6,
            label="y = x reference",
        ),
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        fontsize=11,
        frameon=False,
        title="Speaker category\n(dot area ∝ total utterances)",
        title_fontsize=12,
    )

    fig.suptitle(
        "Per-speaker starter vs chime-in on woman-talk, by gender "
        "(rows: counts, rates; cols: timespans)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ---------------------------------------------------------------------------
# Plot 10 — silent-population trend (mirror of the octopus)
# ---------------------------------------------------------------------------


def plot_silent_men_share_per_year(
    speakers: pd.DataFrame | None = None,
    *,
    k_group: str = "k_all",
    min_utterances: int = 50,
    threshold_ratio: float = 0.5,
    smoothing: int | None = 5,
    year_range: tuple[int, int] | None = None,
    style: PlotStyle | None = None,
) -> Figure:
    """Yearly share of eligible male MPs statistically silent about women.

    Mirror of :func:`plot_speaker_rate_vs_volume`: instead of surfacing
    men whose K-utterance rate is disproportionately *high* for their
    volume, this counts men whose rate is disproportionately *low* — the
    95% Wilson upper bound on their per-year K-utterance rate falls
    below ``threshold_ratio × peer_mean``, where ``peer_mean`` is the
    pooled male K-utterance rate for that year.

    The plotted quantity is ``silent_count / eligible_count`` per year,
    where eligibility is ``utterance_count >= min_utterances``.
    """
    if k_group not in _VALID_K_GROUPS:
        raise ValueError(f"k_group must be one of {_VALID_K_GROUPS}, got {k_group!r}")
    style = _style(style)
    if speakers is None:
        speakers = read_speakers()

    num_col = f"{k_group}_utts"

    if year_range is not None:
        y0, y1 = year_range
        speakers = speakers[(speakers["year"] >= y0) & (speakers["year"] <= y1)]

    men = speakers[(speakers["gender"] == "man") & (speakers["who"].astype(str) != "")]
    per_speaker = men.groupby(["year", "who"], as_index=False)[
        [num_col, "utterance_count"]
    ].sum()
    per_speaker = per_speaker[per_speaker["utterance_count"] > 0].copy()

    # Peer mean per year — pooled over all men (not filtered by
    # eligibility), so the baseline reflects the actual male K-rate for
    # that year rather than the top-slice of the distribution.
    peer = per_speaker.groupby("year", as_index=False).agg(
        k_sum=(num_col, "sum"), n_sum=("utterance_count", "sum")
    )
    peer["peer_mean"] = peer["k_sum"] / peer["n_sum"]
    per_speaker = per_speaker.merge(peer[["year", "peer_mean"]], on="year", how="left")

    # Wilson 95% CI on per-man rate — keep the upper bound.
    _, hi = _wilson_ci(per_speaker[num_col], per_speaker["utterance_count"])
    per_speaker["ub"] = hi

    per_speaker["silent"] = per_speaker["ub"] < (
        threshold_ratio * per_speaker["peer_mean"]
    )
    per_speaker["eligible"] = per_speaker["utterance_count"] >= min_utterances
    per_speaker["silent_and_eligible"] = per_speaker["silent"] & per_speaker["eligible"]

    per_year = (
        per_speaker.groupby("year", as_index=False)
        .agg(
            eligible_count=("eligible", "sum"),
            silent_count=("silent_and_eligible", "sum"),
        )
        .sort_values("year")
        .reset_index(drop=True)
    )
    per_year = per_year[per_year["eligible_count"] > 0].copy()
    per_year["silent_share"] = per_year["silent_count"] / per_year["eligible_count"]

    if smoothing is not None and smoothing > 1:
        per_year["silent_share_raw"] = per_year["silent_share"]
        per_year["silent_share"] = (
            per_year["silent_share"]
            .rolling(window=smoothing, center=True, min_periods=smoothing)
            .mean()
        )

    fig, ax = plt.subplots(figsize=style.figsize)
    male_colour = style.gender_colors.get("man")

    if smoothing is not None and smoothing > 1:
        ax.plot(
            per_year["year"].tolist(),
            per_year["silent_share_raw"].tolist(),
            color=male_colour,
            linestyle="--",
            alpha=0.3,
            linewidth=0.9,
        )

    ax.plot(
        per_year["year"].tolist(),
        per_year["silent_share"].tolist(),
        color=male_colour,
        label=f"Silent share (UB < {threshold_ratio:.0%} × peer)",
    )

    _add_suffrage_marker(ax, style)
    _year_axis(ax)
    ax.set_ylabel("Share of eligible men significantly silent")
    ax.set_ylim(0, 1)
    ax.grid(**style.grid)

    # Secondary axis: eligible_count as visual context for the denominator.
    ax_r = ax.twinx()
    ax_r.plot(
        per_year["year"].tolist(),
        per_year["eligible_count"].tolist(),
        color="grey",
        linestyle="--",
        alpha=0.4,
        linewidth=0.9,
        label="Eligible male speakers (N)",
    )
    ax_r.set_ylabel("Eligible male speakers", color="grey")
    ax_r.set_ylim(bottom=0)
    ax_r.tick_params(axis="y", labelcolor="grey")

    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(
        lines_l + lines_r,
        labels_l + labels_r,
        loc="upper left",
        frameon=False,
        fontsize=8,
    )

    era = (
        f", {year_range[0]}–{year_range[1]}"
        if year_range is not None
        else ", 1900–1940"
    )
    smoothing_note = (
        f", {smoothing}-yr rolling mean"
        if smoothing is not None and smoothing > 1
        else ""
    )
    ax.set_title(
        f"Share of eligible male MPs significantly below "
        f"{threshold_ratio:.0%} × male-peer {k_group.upper()} rate "
        f"(min {min_utterances} utterances{smoothing_note}{era})"
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 8 — per-speaker starter-rate vs. chime-rate, small multiples per bloc
# ---------------------------------------------------------------------------


def _assign_dominant_bloc(
    speakers_window: pd.DataFrame, party_to_bloc: Mapping[str, str]
) -> pd.DataFrame:
    """Return one row per (who, gender) with the bloc receiving the plurality
    of their ``utterance_count`` in the window; ``bloc`` is NaN for
    speakers all of whose parties fall outside the taxonomy.
    """
    d = speakers_window.assign(bloc=speakers_window["party"].map(party_to_bloc))
    by_bloc = d.groupby(["who", "gender", "bloc"], as_index=False, dropna=False)[
        "utterance_count"
    ].sum()
    idx = by_bloc.groupby(["who", "gender"], dropna=False)["utterance_count"].idxmax()
    return by_bloc.loc[idx, ["who", "gender", "bloc"]].reset_index(drop=True)


UNAFFILIATED_COLOR = "#555555"
UNAFFILIATED_LABEL = "Unaffiliated / NaN-party"


def plot_party_speaker_starts_vs_chimes(
    starters_df: pd.DataFrame | None = None,
    speakers_df: pd.DataFrame | None = None,
    *,
    blocs: Iterable[str] = ("S", "H", "L", "Bf", "K"),
    include_unaffiliated: bool = True,
    window: tuple[int, int] = (1922, 1940),
    mapping: Mapping[str, Iterable[str]] | None = None,
    max_gap: int = 1,
    min_arc_length: int = 2,
    style: PlotStyle | None = None,
) -> Figure:
    """Per-speaker rate scatter, small multiples per party bloc.

    Complements the gender rate scatter (Plot 5d pilot): asks who *inside
    each bloc* drives that bloc's engagement with women's topics. One
    panel per requested bloc; the panel's bloc is highlighted in the
    paper's bloc colour, everyone else (including unmapped and
    NaN-party speakers) is a light-gray background so the coverage
    limit stays visible.

    ``include_unaffiliated=True`` (default) adds a final panel that
    highlights the NaN-bloc speakers themselves (in a neutral dark
    grey), so the ~36 % of the chamber that sits outside the bloc
    taxonomy gets its own view rather than only appearing as
    background. Fits nicely into the sixth cell of the default 2×3
    grid for the standard 5-bloc layout.

    Axes are rates per total utterance:
    ``starter_rate = starters / utterance_count``,
    ``chime_rate = chime_ins / utterance_count``. This decorrelates
    position from raw speaking volume (dot size ∝ √utterance_count).
    ``chime_ins = k_all_utts - starters``.

    Each speaker is attributed a single **dominant bloc** — the bloc
    that received the plurality of their utterance_count in the window
    (ties broken by pandas' internal ordering). Speakers whose parties
    all fall outside ``mapping`` get ``bloc = None`` and appear in the
    unaffiliated panel's foreground (and in every bloc panel's
    background).
    """
    import numpy as np

    from .data import read_speakers

    style = _style(style)
    mapping = mapping if mapping is not None else DEFAULT_PARTY_MAPPING
    blocs = tuple(blocs)
    y0, y1 = window
    party_to_bloc = {name: b for b, names in mapping.items() for name in names}

    if speakers_df is None:
        speakers_df = read_speakers()

    m = speaker_starts_chimes(
        speakers=speakers_df,
        starters=starters_df,
        year_range=(y0, y1),
        min_utterances=1,
        known_gender_only=False,
        max_gap=max_gap,
        min_arc_length=min_arc_length,
    )

    speakers_win = speakers_df[
        (speakers_df["year"] >= y0) & (speakers_df["year"] <= y1)
    ]
    dom = _assign_dominant_bloc(speakers_win, party_to_bloc)
    m = m.merge(dom, on=["who", "gender"], how="left")
    m_c = _attach_category(m)

    # Each panel is either a bloc name or the sentinel None → unaffiliated.
    panels: list[str | None] = list(blocs)
    if include_unaffiliated:
        panels.append(None)
    n = len(panels)
    ncols = min(n, 3) if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.6 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, (ax, panel_key) in enumerate(zip(axes_flat, panels)):
        if panel_key is None:
            mask_fg = m_c["bloc"].isna()
            color = UNAFFILIATED_COLOR
            title = UNAFFILIATED_LABEL
        else:
            mask_fg = m_c["bloc"] == panel_key
            color = style.party_colors.get(panel_key, "#333333")
            title = style.party_labels.get(panel_key, panel_key)

        bg = m_c[~mask_fg]
        fg = m_c[mask_fg]

        # Background cloud: keep monochrome grey but split by marker so
        # cross-plot ID markers (hexagons, X's) stay recognisable even
        # when off-panel.
        for marker, msub in bg.groupby("marker"):
            is_highlight = marker in ("X", "h")
            ax.scatter(
                msub["starter_rate"],
                msub["chime_rate"],
                s=np.sqrt(msub["utterance_count"].values.clip(min=1))
                * 4
                * (1.3 if is_highlight else 1.0),
                c="#e0e0e0",
                alpha=0.25 + (0.15 if is_highlight else 0.0),
                marker=marker,
                edgecolor="none",
                zorder=1,
            )
        # Foreground (this panel's bloc): bloc colour, split by marker.
        first = True
        for marker, msub in fg.groupby("marker"):
            is_highlight = marker in ("X", "h")
            ax.scatter(
                msub["starter_rate"],
                msub["chime_rate"],
                s=np.sqrt(msub["utterance_count"].values.clip(min=1))
                * 4
                * (1.3 if is_highlight else 1.0),
                c=color,
                alpha=0.75,
                marker=marker,
                edgecolor="black" if is_highlight else "none",
                linewidths=0.5 if is_highlight else 0.0,
                label=f"{title} (n={len(fg)})" if first else None,
                zorder=2,
            )
            first = False
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="#999999",
            linewidth=0.7,
            alpha=0.5,
        )
        ax.set_xlim(-0.02, 0.7)
        ax.set_ylim(-0.02, 0.7)
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        row, col = divmod(i, ncols)
        if row == nrows - 1:
            ax.set_xlabel("Starter rate")
        if col == 0:
            ax.set_ylabel("Chime-in rate")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Per-speaker starter vs. chime-in rate, {y0}–{y1} "
        f"(one panel per bloc; dot size ∝ √total utterances)",
        y=1.00,
    )
    fig.tight_layout()
    return fig
