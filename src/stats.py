"""Statistical tests over the topic-arc-starter data.

Currently exposes a one-sample exact binomial test for whether women's
share of "real" conversation starters (arcs of length ≥ 2) exceeds
what would be expected under a stated null distribution.

""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, lgamma, log, log1p
from typing import Iterable, Literal, Mapping

import pandas as pd

from . import settings
from .data import (
    read_topic_arc_starters,
    read_topic_arc_starters_by_party,
    read_word_frequencies,
)


NullChoice = Literal["utterance_share", "woman_talk_share"]
Alternative = Literal["greater", "less"]

# Women were not eligible to sit in the Riksdag before 1922 (suffrage) and
# first sat in 1922 — the natural start of the pooling window for any
# starter-share test.
POST_SUFFRAGE_START = 1922


@dataclass(frozen=True)
class BinomialStarterTest:
    """Result of a one-sample exact binomial test on a starter subgroup.

    ``target`` names the subgroup being tested (e.g. ``"woman"``, ``"S"``,
    ``"H"``). ``observed_share = n_target_starters / n_starters``. Under
    the null, the same ratio is expected to equal ``null_p``. The test is
    one-sided by default (``"greater"``): reject if observed exceeds
    the null.
    """

    year_or_range: str
    target: str
    n_starters: int
    n_target_starters: int
    observed_share: float
    null: str
    null_p: float
    expected_target_starters: float
    alternative: str
    p_value: float


def _binom_tail(k: int, n: int, p: float, alternative: Alternative) -> float:
    """Exact one-sided binomial tail probability, log-space.

    ``"greater"`` returns ``P(X >= k)``; ``"less"`` returns ``P(X <= k)``.
    Uses ``lgamma`` and log-sum-exp so the computation stays stable at
    large ``n`` where the raw binomial coefficient overflows.
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"invalid k, n: k={k}, n={n}")
    if alternative not in ("greater", "less"):
        raise ValueError(
            f"alternative must be 'greater' or 'less', got {alternative!r}"
        )

    if p <= 0.0:
        if alternative == "greater":
            return 1.0 if k == 0 else 0.0
        return 1.0
    if p >= 1.0:
        if alternative == "less":
            return 1.0 if k == n else 0.0
        return 1.0

    log_p = log(p)
    log_q = log1p(-p)

    def log_pmf(i: int) -> float:
        return (
            lgamma(n + 1)
            - lgamma(i + 1)
            - lgamma(n - i + 1)
            + i * log_p
            + (n - i) * log_q
        )

    indices = range(k, n + 1) if alternative == "greater" else range(0, k + 1)
    logs = [log_pmf(i) for i in indices]
    m = max(logs)
    return exp(m + log(sum(exp(l - m) for l in logs)))


def _resolve_years(years: int | Iterable[int] | None) -> tuple[tuple[int, ...], str]:
    if years is None:
        yr = tuple(range(POST_SUFFRAGE_START, settings.END_YEAR + 1))
        return yr, f"{yr[0]}-{yr[-1]} (post-suffrage pool)"
    if isinstance(years, int):
        return (years,), str(years)
    yr = tuple(years)
    if not yr:
        raise ValueError("years must not be empty")
    return yr, f"{min(yr)}-{max(yr)} pooled"


def starter_binomial_test(
    years: int | Iterable[int] | None = None,
    *,
    null: NullChoice = "woman_talk_share",
    max_gap: int = 1,
    min_arc_length: int = 2,
    alternative: Alternative = "greater",
    chambers: Iterable[int] | None = None,
) -> BinomialStarterTest:
    """One-sample exact binomial test: do women initiate more real
    conversations about women than the null predicts?

    Each conversation starter (a kvinna utterance opening an arc of at
    least ``min_arc_length`` under the ``max_gap`` gap tolerance) is
    treated as an independent Bernoulli draw with success = "woman
    speaker". Under the null, the success probability is either:

    - ``"woman_talk_share"`` (default) — the year's share of woman-tagged
      utterances (``k_all_utts``) spoken by women. Controls for gender
      differences in *how much each engages* with women's topics; asks
      whether, conditional on engaging at all, women disproportionately
      initiate.
    - ``"utterance_share"`` — the year's share of *all* utterances by
      women. Does not control for engagement; a rejection here can be
      explained by either "women engage more" or "women initiate more".

    Pass ``years=Y`` for a single year, ``years=range(...)`` or an
    iterable for a pool, or omit to default to the post-suffrage window
    (1922 through the last shipped year).

    Returns a :class:`BinomialStarterTest`. See the module docstring for
    method and numerics references.
    """
    years_tuple, label = _resolve_years(years)

    st = read_topic_arc_starters(
        years=years_tuple,
        chambers=chambers,
        max_gap=max_gap,
        min_arc_length=min_arc_length,
    )
    st = st[st["gender"].isin(("man", "woman"))]
    n = int(st["starters"].sum())
    k = int(st[st["gender"] == "woman"]["starters"].sum())
    if n == 0:
        raise ValueError(f"No known-gender starters in {label}")

    wf = read_word_frequencies()
    wf = wf[(wf["year"].isin(years_tuple)) & (wf["gender"].isin(("man", "woman")))]
    if null == "utterance_share":
        col = "utterance_count"
    elif null == "woman_talk_share":
        col = "k_all_utts"
    else:
        raise ValueError(f"unknown null: {null!r}")

    totals = wf.groupby("gender")[col].sum()
    denom = int(totals.get("man", 0) + totals.get("woman", 0))
    if denom == 0:
        raise ValueError(f"No {col} data available in {label}")
    p_null = float(totals.get("woman", 0)) / denom

    p_value = _binom_tail(k, n, p_null, alternative)
    return BinomialStarterTest(
        year_or_range=label,
        target="woman",
        n_starters=n,
        n_target_starters=k,
        observed_share=k / n,
        null=null,
        null_p=p_null,
        expected_target_starters=n * p_null,
        alternative=alternative,
        p_value=p_value,
    )


def bloc_starter_binomial_test(
    bloc: str,
    years: int | Iterable[int] | None = None,
    *,
    mapping: Mapping[str, Iterable[str]] | None = None,
    null: NullChoice = "woman_talk_share",
    max_gap: int = 1,
    min_arc_length: int = 2,
    alternative: Alternative = "greater",
    chambers: Iterable[int] | None = None,
) -> BinomialStarterTest:
    """One-sample exact binomial test on a specific bloc's share of starters.

    Parallel to :func:`starter_binomial_test` but for a political-bloc
    subgroup rather than for women. The bloc is defined by pooling raw
    party labels via ``mapping`` (defaults to
    :data:`src.plots.DEFAULT_PARTY_MAPPING` — S, H, L, Bf, K).

    Both numerator and denominator are restricted to starters (and
    utterances) whose party maps to a known bloc; unmapped and NaN
    parties are dropped. This conditions the test on "the population
    that our bloc taxonomy actually covers".

    Nulls are the same as for gender: ``"utterance_share"`` uses each
    bloc's share of raw speaking volume; ``"woman_talk_share"`` (default)
    uses each bloc's share of ``k_all_utts`` — the engagement-controlled
    null. See :func:`starter_binomial_test` for shared kwarg semantics.
    """
    mapping = _resolve_party_mapping(mapping)
    if bloc not in mapping:
        raise ValueError(f"unknown bloc {bloc!r}; known blocs: {sorted(mapping)}")
    years_tuple, label = _resolve_years(years)
    label = f"{label} · bloc={bloc}"

    party_to_bloc = {name: b for b, names in mapping.items() for name in names}

    st = read_topic_arc_starters_by_party(
        years=years_tuple,
        chambers=chambers,
        max_gap=max_gap,
        min_arc_length=min_arc_length,
    )
    st = st.assign(bloc=st["party"].map(party_to_bloc)).dropna(subset=["bloc"])
    n = int(st["starters"].sum())
    k = int(st[st["bloc"] == bloc]["starters"].sum())
    if n == 0:
        raise ValueError(f"No known-bloc starters in {label}")

    wf = read_word_frequencies()
    wf = wf[wf["year"].isin(years_tuple)]
    wf = wf.assign(bloc=wf["party"].map(party_to_bloc)).dropna(subset=["bloc"])

    if null == "utterance_share":
        col = "utterance_count"
    elif null == "woman_talk_share":
        col = "k_all_utts"
    else:
        raise ValueError(f"unknown null: {null!r}")

    totals = wf.groupby("bloc")[col].sum()
    denom = int(totals.sum())
    if denom == 0:
        raise ValueError(f"No {col} in known blocs for {label}")
    p_null = float(totals.get(bloc, 0)) / denom

    p_value = _binom_tail(k, n, p_null, alternative)
    return BinomialStarterTest(
        year_or_range=label,
        target=bloc,
        n_starters=n,
        n_target_starters=k,
        observed_share=k / n,
        null=null,
        null_p=p_null,
        expected_target_starters=n * p_null,
        alternative=alternative,
        p_value=p_value,
    )


def bloc_starter_scan(
    years: int | Iterable[int] | None = None,
    *,
    mapping: Mapping[str, Iterable[str]] | None = None,
    null: NullChoice = "woman_talk_share",
    max_gap: int = 1,
    min_arc_length: int = 2,
    alternative: Alternative = "greater",
    chambers: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Run :func:`bloc_starter_binomial_test` for every bloc in ``mapping``.

    Returns one row per bloc with the observed count, expected count
    under the null, and the one-sided binomial p-value. Sorted by
    p-value ascending so the most over-represented blocs sit at the top.
    Also reports the coverage — how many starters and how much of the
    null-denominator sit inside vs. outside the bloc taxonomy — so the
    caller can judge how much of the picture the scan actually explains.
    """
    mapping = _resolve_party_mapping(mapping)
    rows = []
    for bloc in mapping:
        r = bloc_starter_binomial_test(
            bloc,
            years=years,
            mapping=mapping,
            null=null,
            max_gap=max_gap,
            min_arc_length=min_arc_length,
            alternative=alternative,
            chambers=chambers,
        )
        rows.append(
            {
                "bloc": bloc,
                "n_starters": r.n_starters,
                "n_bloc_starters": r.n_target_starters,
                "observed_share": r.observed_share,
                "null_p": r.null_p,
                "expected": r.expected_target_starters,
                "p_value": r.p_value,
            }
        )
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def _resolve_party_mapping(
    mapping: Mapping[str, Iterable[str]] | None,
) -> Mapping[str, Iterable[str]]:
    if mapping is not None:
        return mapping
    # Deferred import: plots.py imports data.py, so importing plots at
    # module load time in stats.py would eagerly pull matplotlib.
    from .plots import DEFAULT_PARTY_MAPPING

    return DEFAULT_PARTY_MAPPING
