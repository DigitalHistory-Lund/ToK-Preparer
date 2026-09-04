"""Notebook-facing data readers for the shipped snapshots.

Notebooks should import from this module rather than hard-coding paths
or column names. When the underlying file format changes (CSV → parquet,
schema tweaks, new columns), only this module needs to change and every
notebook benefits.

The chamber-split mechanism in ``build-quarto.py`` mirrors the existing
``tmp_db`` pattern: for a chamber-specific variant it rewrites e.g.
``import read_word_frequencies`` → ``import read_word_frequencies1 as
read_word_frequencies``.
"""

from __future__ import annotations

import warnings
import functools
import gzip
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import settings
from .settings import (
    persons_db as persons_db_path,
    speaker_totals as speaker_totals_path,
    speaker_word_totals as speaker_word_totals_path,
    speakers as speakers_path,
    speakers1 as speakers1_path,
    speakers2 as speakers2_path,
    word_freq,
    word_freq1,
    word_freq2,
)


WORD_FREQ_COLUMNS: tuple[str, ...] = (
    "year",
    "chamber",
    "gender",
    "party",
    "utterance_count",
    "total_words",
    "char_count",
    "k1_utts",
    "k2_utts",
    "k3_utts",
    "k_all_utts",
    "k1_matches",
    "k2_matches",
    "k3_matches",
    "k_all_matches",
    "damer_matches",
    "damer_utts",
    "fru_matches",
    "fru_utts",
)

SPEAKERS_COLUMNS: tuple[str, ...] = (
    "year",
    "chamber",
    "gender",
    "party",
    "who",
    "utterance_count",
    "k1_utts",
    "k2_utts",
    "k3_utts",
    "k_all_utts",
)

SPEAKER_TOTALS_COLUMNS: tuple[str, ...] = (
    "who",
    "name",
    "gender",
    "party",
    "utterance_count",
    "k1_utts",
    "k2_utts",
    "k3_utts",
    "k_all_utts",
    "share",
)

SPEAKER_WORD_TOTALS_COLUMNS: tuple[str, ...] = (
    "year",
    "chamber",
    "gender",
    "party",
    "who",
    "total_words",
    "k1_matches",
    "k2_matches",
    "k3_matches",
    "k_all_matches",
)


def _read_csv(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    return df[list(columns)]


def read_word_frequencies() -> pd.DataFrame:
    """Per-bucket word-frequency counts across both chambers."""
    return _read_csv(word_freq, WORD_FREQ_COLUMNS)


def read_word_frequencies1() -> pd.DataFrame:
    """Per-bucket word-frequency counts for Chamber 1 (Första kammaren) only."""
    return _read_csv(word_freq1, WORD_FREQ_COLUMNS)


def read_word_frequencies2() -> pd.DataFrame:
    """Per-bucket word-frequency counts for Chamber 2 (Andra kammaren) only."""
    return _read_csv(word_freq2, WORD_FREQ_COLUMNS)


def read_speakers() -> pd.DataFrame:
    """Per-speaker utterance-level counters across both chambers."""
    return _read_csv(speakers_path, SPEAKERS_COLUMNS)


def read_speakers1() -> pd.DataFrame:
    """Per-speaker utterance-level counters for Chamber 1 only."""
    return _read_csv(speakers1_path, SPEAKERS_COLUMNS)


def read_speakers2() -> pd.DataFrame:
    """Per-speaker utterance-level counters for Chamber 2 only."""
    return _read_csv(speakers2_path, SPEAKERS_COLUMNS)


def _load_speaker_names(persons_path: Path) -> pd.DataFrame:
    """Return ``who -> name`` from persons.sqlite.

    Unions the three person tables (MP, minister, speaker) and keeps the
    ``primary_name=1`` canonical form. Callers should treat a missing DB
    as "no names available" — see :func:`build_speaker_totals`.
    """
    conn = sqlite3.connect(str(persons_path))
    try:
        df = pd.read_sql(
            """
            SELECT DISTINCT person_id AS who, name
            FROM processed_member_of_parliament
            WHERE primary_name = 1
            UNION
            SELECT DISTINCT person_id AS who, name
            FROM processed_minister
            WHERE primary_name = 1
            UNION
            SELECT DISTINCT person_id AS who, name
            FROM processed_speaker
            WHERE primary_name = 1
            """,
            conn,
        )
    finally:
        conn.close()
    return df.drop_duplicates(subset=["who"]).reset_index(drop=True)


SPEAKER_CATEGORIES: tuple[str, ...] = (
    "original_five",
    "loud_man",
    "man",
    "woman",
    "unknown",
)


def _load_original_five_who_ids(persons_path: Path) -> tuple[str, ...]:
    """The five women who took their seats on the first day of the 1922 Riksdag.

    Kerstin Hesselgren (Ch1), Elisabeth Tamm, Nelly Thüring, Agda
    Östlund, Bertha Wellin (Ch2). Identified as women whose earliest
    MP-tenure start date is 1922-01-10 — the opening day of the first
    Riksdag with elected women members.

    Returns an empty tuple if the persons DB is missing, empty, or has
    no ``processed_member_of_parliament`` table — matching the tolerant
    behaviour of :func:`_load_speaker_names` in :func:`build_speaker_totals`.
    That downgrades cross-plot markers (no hexagons) rather than crashing.
    """
    if not persons_path.exists() or persons_path.stat().st_size == 0:
        return ()
    conn = sqlite3.connect(str(persons_path))
    try:
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='processed_member_of_parliament' LIMIT 1"
        ).fetchone()
        if not has_table:
            return ()
        rows = conn.execute(
            """
            SELECT DISTINCT person_id
            FROM processed_member_of_parliament
            WHERE gender = 'woman'
              AND substr(start, 1, 10) = '1922-01-10'
            ORDER BY person_id
            """
        ).fetchall()
    finally:
        conn.close()
    return tuple(r[0] for r in rows)


def speaker_categories(
    *,
    speaker_totals: pd.DataFrame | None = None,
    persons_path: Path | None = None,
    loud_men_top_n: int = 15,
) -> pd.DataFrame:
    """Assign every known speaker one of :data:`SPEAKER_CATEGORIES`.

    Rules (later rules override earlier ones):

    1. ``woman``   — any speaker with ``gender == 'woman'``
    2. ``original_five`` — the five women elected to the 1922 Riksdag
       (:func:`_load_original_five_who_ids`)
    3. ``man``     — any speaker with ``gender == 'man'``
    4. ``loud_man`` — top-``loud_men_top_n`` male speakers by absolute
       ``k_all_utts`` count. These are the men who dominate the counts
       panels — Lindhagen and the other high-volume woman-mentioners —
       not the highest-rate men (a man with 5 utts and 3 kvinna hits
       is a noise point, not a substantive "loud man").
    5. ``unknown`` — anyone else (empty ``who`` or non-binary gender)

    Returns a DataFrame ``[who, category]``. Callers merge on ``who``
    and map ``category`` through the plot's marker/color tables.
    """
    persons_path = persons_path if persons_path is not None else persons_db_path

    if speaker_totals is None:
        speaker_totals = read_speaker_totals()

    original_five = set(_load_original_five_who_ids(persons_path))

    men = speaker_totals[speaker_totals["gender"] == "man"]
    loud_men = set(
        men.sort_values("k_all_utts", ascending=False).head(loud_men_top_n)["who"]
    )

    def _classify(row: pd.Series) -> str:
        who = row["who"]
        gender = row["gender"]
        if who in original_five:
            return "original_five"
        if who in loud_men:
            return "loud_man"
        if gender == "woman":
            return "woman"
        if gender == "man":
            return "man"
        return "unknown"

    cats = speaker_totals[["who", "gender"]].copy()
    cats["category"] = cats.apply(_classify, axis=1)
    return cats[["who", "category"]]


def build_speaker_totals(
    speakers: pd.DataFrame | None = None,
    persons_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Aggregate speakers.csv.gz per-speaker and write speaker_totals.csv.gz.

    Chambers pooled, years pooled. ``gender`` and ``party`` are the modal
    values across the speaker's per-year rows. ``name`` is joined from
    ``persons.sqlite`` when available (empty string otherwise, so the
    build works even without the DB). ``share`` is
    ``k_all_utts / utterance_count`` at the speaker level; rows are
    written sorted by ``share`` descending. Unknown-speaker rows
    (``who == ""``) are dropped.
    """
    if speakers is None:
        speakers = read_speakers()
    persons_path = persons_path if persons_path is not None else persons_db_path
    output_path = output_path if output_path is not None else speaker_totals_path

    known = speakers[speakers["who"].astype(str) != ""].copy()

    sum_cols = ["utterance_count", "k1_utts", "k2_utts", "k3_utts", "k_all_utts"]
    totals = known.groupby("who", as_index=False)[sum_cols].sum()

    def _modal(s: pd.Series) -> str:
        m = s.mode()
        return "" if m.empty else str(m.iat[0])

    modal = (
        known.groupby("who")
        .agg(gender=("gender", _modal), party=("party", _modal))
        .reset_index()
    )
    totals = totals.merge(modal, on="who", how="left")

    if persons_path.exists():
        names = _load_speaker_names(persons_path)
    else:
        names = pd.DataFrame(
            {"who": pd.Series(dtype=str), "name": pd.Series(dtype=str)}
        )
    totals = totals.merge(names, on="who", how="left")
    totals["name"] = totals["name"].fillna("")

    totals["share"] = totals["k_all_utts"] / totals["utterance_count"]

    totals = (
        totals[list(SPEAKER_TOTALS_COLUMNS)]
        .sort_values("share", ascending=False)
        .reset_index(drop=True)
    )
    totals.to_csv(output_path, index=False, compression="gzip")
    return totals


def read_speaker_totals() -> pd.DataFrame:
    """Read speaker_totals.csv.gz, building it from source if missing.

    The build joins ``speakers.csv.gz`` with ``persons.sqlite`` (from
    ``settings.persons_db``); if the persons DB is not present, ``name``
    is written as an empty string and the caller can fall back to
    ``who`` for display.
    """
    if not speaker_totals_path.exists():
        build_speaker_totals()
    return _read_csv(speaker_totals_path, SPEAKER_TOTALS_COLUMNS)


# Superscripts inserted by reduce_db.annotate_content — must be stripped
# before tokenisation so classify() sees the original word.
_ANNOTATION_MARKERS = "".maketrans("", "", "¹²³")


def _iter_yearly_dbs():
    """Yield (year, tmp-uncompressed sqlite Path) for each shipped year."""
    for year in range(settings.START_YEAR, settings.END_YEAR + 1):
        gz = settings.root / f"ToK_data_{year}.sqlite3.gz"
        if not gz.exists():
            continue
        with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
            with gzip.open(gz, "rb") as src:
                tmp.write(src.read())
            tmp.flush()
            yield year, Path(tmp.name)


def build_speaker_word_totals(
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Aggregate per-(year, chamber, gender, party, who) word-level totals.

    Iterates the shipped yearly ``ToK_data_YYYY.sqlite3.gz`` files,
    re-tokenises each utterance's content, and counts total_words plus
    K1/K2/K3/K_all matches using the same partition-and-classify logic
    as :func:`src.word_frequencies.compute`. Writes ``settings.speaker_word_totals``.

    Runtime is ~2 minutes for the full 1900–1940 corpus; the shipped
    ``speaker_word_totals.csv.gz`` snapshot means production callers
    of :func:`read_speaker_word_totals` never trigger this build path.
    """
    from collections import defaultdict
    from .word_frequencies import _WORD_RE, _classify, _partition_patterns

    literals, prefixes, suffixes, contains, k3_phrases = _partition_patterns()

    def _empty() -> dict[str, int]:
        return {
            "total_words": 0,
            "k1_matches": 0,
            "k2_matches": 0,
            "k3_matches": 0,
            "k_all_matches": 0,
        }

    totals: dict[tuple, dict[str, int]] = defaultdict(_empty)

    for year, db_path in _iter_yearly_dbs():
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            persons = {
                pid: (name or "", gender or "", party or "")
                for pid, name, gender, party in conn.execute(
                    "SELECT id, name, gender, party FROM person"
                )
            }
            for pid, chamber, content in conn.execute(
                "SELECT person_id, kammare, content FROM utterance"
            ):
                if content is None:
                    continue
                person = persons.get(pid)
                if person is None:
                    continue
                who, gender, party = person
                content_lower = content.lower().translate(_ANNOTATION_MARKERS)

                k1 = k2 = k3 = 0
                k_all = 0

                for phrase in k3_phrases:
                    hits = content_lower.count(phrase)
                    if hits:
                        k3 += hits
                        k_all += hits

                tokens = _WORD_RE.findall(content_lower)
                for token in tokens:
                    cats = _classify(token, literals, prefixes, suffixes, contains)
                    if cats:
                        for cat in cats:
                            if cat == 0:
                                k1 += 1
                            elif cat == 1:
                                k2 += 1
                            else:
                                k3 += 1
                        k_all += 1

                key = (year, chamber, gender, party, who)
                bucket = totals[key]
                bucket["total_words"] += len(tokens)
                bucket["k1_matches"] += k1
                bucket["k2_matches"] += k2
                bucket["k3_matches"] += k3
                bucket["k_all_matches"] += k_all

    rows = []
    for (year, chamber, gender, party, who), b in totals.items():
        rows.append(
            {
                "year": year,
                "chamber": chamber,
                "gender": gender,
                "party": party,
                "who": who,
                "total_words": b["total_words"],
                "k1_matches": b["k1_matches"],
                "k2_matches": b["k2_matches"],
                "k3_matches": b["k3_matches"],
                "k_all_matches": b["k_all_matches"],
            }
        )
    df = pd.DataFrame(rows, columns=list(SPEAKER_WORD_TOTALS_COLUMNS))
    df = df.sort_values(["year", "chamber", "who"]).reset_index(drop=True)

    output_path = output_path if output_path is not None else speaker_word_totals_path
    df.to_csv(output_path, index=False, compression="gzip")
    return df


def read_speaker_word_totals() -> pd.DataFrame:
    """Read speaker_word_totals.csv.gz, building it from the yearly SQLites if missing.

    Per-(year, chamber, gender, party, who) word-level counters, parallel
    to ``speakers.csv.gz`` but with token-level match counts instead of
    utterance-level presence flags. Build runtime is ~2 minutes.
    """
    if not speaker_word_totals_path.exists():
        build_speaker_word_totals()
    return _read_csv(speaker_word_totals_path, SPEAKER_WORD_TOTALS_COLUMNS)


def _compute_unknown_share(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year unknown-speaker share from a word-frequencies frame.

    Rows with NaN ``gender`` (empty in the CSV; unmatched ``who``) are the
    "unknown" side. Returns columns
    ``year, utterance_count_total, utterance_count_unknown, unknown_share``.
    """
    total = (
        df.groupby("year", dropna=False)["utterance_count"]
        .sum()
        .rename("utterance_count_total")
    )
    unknown = (
        df.loc[df["gender"].isna()]
        .groupby("year", dropna=False)["utterance_count"]
        .sum()
        .rename("utterance_count_unknown")
    )
    out = pd.concat([total, unknown], axis=1).fillna(0)
    out["utterance_count_total"] = out["utterance_count_total"].astype(int)
    out["utterance_count_unknown"] = out["utterance_count_unknown"].astype(int)
    out["unknown_share"] = out["utterance_count_unknown"] / out["utterance_count_total"]
    return out.reset_index()


def read_unknown_share() -> pd.DataFrame:
    """Per-year unknown-speaker share across both chambers."""
    return _compute_unknown_share(read_word_frequencies())


def read_unknown_share1() -> pd.DataFrame:
    """Per-year unknown-speaker share for Chamber 1 (Första kammaren) only."""
    return _compute_unknown_share(read_word_frequencies1())


def read_unknown_share2() -> pd.DataFrame:
    """Per-year unknown-speaker share for Chamber 2 (Andra kammaren) only."""
    return _compute_unknown_share(read_word_frequencies2())


_SUBSET_SQL = """
SELECT
    COALESCE(kvinna_1, 0) AS k1,
    COALESCE(kvinna_2, 0) AS k2,
    COALESCE(kvinna_3, 0) AS k3,
    COUNT(*) AS utterance_count
FROM utterance
WHERE (kvinna_1 OR kvinna_2 OR kvinna_3)
  {chamber_clause}
GROUP BY k1, k2, k3
"""


def _yearly_sqlite_path(year: int) -> Path:
    return settings.root / f"ToK_data_{year}.sqlite3.gz"


@functools.lru_cache(maxsize=64)
def _read_yearly_subsets(year: int, chambers: tuple[int, ...] | None) -> pd.DataFrame:
    gz_path = _yearly_sqlite_path(year)
    if not gz_path.exists():
        return pd.DataFrame(columns=["k1", "k2", "k3", "utterance_count"])

    chamber_clause = ""
    if chambers is not None:
        placeholders = ",".join(str(int(c)) for c in chambers)
        chamber_clause = f"AND kammare IN ({placeholders})"

    sql = _SUBSET_SQL.format(chamber_clause=chamber_clause)

    with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
        with gzip.open(gz_path, "rb") as gz:
            tmp.write(gz.read())
        tmp.flush()
        conn = sqlite3.connect(tmp.name)
        try:
            df = pd.read_sql(sql, conn)
        finally:
            conn.close()
    return df


def read_kvinna_utterance_subsets(
    years: Iterable[int] | None = None,
    chambers: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Per-subset utterance counts for {K1, K2, K3}.

    Returns exactly 7 rows (all non-empty subsets of the three K
    indicators), aggregated across the 41 shipped
    ``ToK_data_YYYY.sqlite3.gz`` files. Defaults iterate all years and
    both chambers. ``years`` and ``chambers`` filter the aggregation.
    """
    years_tuple = (
        tuple(years)
        if years is not None
        else tuple(range(settings.START_YEAR, settings.END_YEAR + 1))
    )
    chambers_tuple = tuple(chambers) if chambers is not None else None

    parts = [_read_yearly_subsets(y, chambers_tuple) for y in years_tuple]
    combined = pd.concat(parts, ignore_index=True)
    if combined.empty:
        return combined

    out = (
        combined.groupby(["k1", "k2", "k3"], as_index=False)["utterance_count"]
        .sum()
        .astype({"k1": bool, "k2": bool, "k3": bool})
    )
    return out


def _kvinna_hit_expr(alias: str) -> str:
    return (
        f"(COALESCE({alias}.kvinna_1,0)=1 OR "
        f"COALESCE({alias}.kvinna_2,0)=1 OR "
        f"COALESCE({alias}.kvinna_3,0)=1)"
    )


def _starter_sql_body(max_gap: int, min_arc_length: int) -> tuple[str, str]:
    """Build the JOIN + WHERE fragments for the starter SQL.

    Returns ``(joins_sql, conditions_sql)`` — reusable by the (year,
    gender) and per-speaker variants. ``min_arc_length`` in ``{1, 2}``:
    ``1`` is the default "any new arc"; ``2`` restricts to arcs that
    actually get a second kvinna utterance (i.e., excludes isolated
    single-utterance mentions).
    """
    if max_gap < 0:
        raise ValueError(f"max_gap must be >= 0, got {max_gap}")
    if min_arc_length not in (1, 2):
        raise ValueError(f"min_arc_length must be 1 or 2, got {min_arc_length}")

    joins: list[str] = []
    backward_no_kvinna: list[str] = []
    prev_ref = "u.prev"
    for i in range(1, max_gap + 2):
        alias = f"up{i}"
        joins.append(f"LEFT JOIN utterance {alias} ON {alias}.id = {prev_ref}")
        backward_no_kvinna.append(f"NOT {_kvinna_hit_expr(alias)}")
        prev_ref = f"{alias}.prev"

    forward_any_kvinna: list[str] = []
    if min_arc_length == 2:
        next_ref = "u.next"
        for i in range(1, max_gap + 2):
            alias = f"dn{i}"
            joins.append(f"LEFT JOIN utterance {alias} ON {alias}.id = {next_ref}")
            forward_any_kvinna.append(_kvinna_hit_expr(alias))
            next_ref = f"{alias}.next"

    conds = [_kvinna_hit_expr("u"), *backward_no_kvinna]
    if forward_any_kvinna:
        conds.append("(" + " OR ".join(forward_any_kvinna) + ")")
    return "\n".join(joins), "\n  AND ".join(conds)


def _build_starter_sql(
    max_gap: int,
    min_arc_length: int = 1,
    *,
    key_column: str = "p.gender",
) -> str:
    """Build the (year, ``key_column``) topic-arc-starter SQL.

    A starter is a kvinna utterance where the preceding ``max_gap + 1``
    utterances in the chamber chain all had no kvinna hit. ``max_gap=0``
    means the immediate predecessor must be non-kvinna, so any contiguous
    run of kvinna utterances counts as one arc. ``max_gap=1`` tolerates
    a single non-kvinna interjection.

    ``min_arc_length=2`` additionally requires the arc to contain at
    least one more kvinna utterance forward within the same gap window
    — i.e., excludes arcs that are isolated single-utterance mentions.

    ``key_column`` is the SQL expression the results are grouped by
    (default ``p.gender``; also useful: ``p.party``). The output column
    name is the un-qualified tail of the expression. Callers substitute
    the ``{chamber_clause}`` placeholder to filter by chamber.
    """
    joins_sql, conds_sql = _starter_sql_body(max_gap, min_arc_length)
    alias = key_column.rsplit(".", 1)[-1]
    return f"""
SELECT u.year, {key_column} AS {alias}, COUNT(*) AS starters
FROM utterance u
JOIN person p ON p.id = u.person_id
{joins_sql}
WHERE {conds_sql}
  {{chamber_clause}}
GROUP BY u.year, {alias}
"""


# Preserved for tests that assert the max_gap=0 baseline directly.
_TOPIC_ARC_STARTER_SQL = _build_starter_sql(max_gap=0)


@functools.lru_cache(maxsize=128)
def _read_yearly_starters_by_party(
    year: int,
    chambers: tuple[int, ...] | None,
    max_gap: int,
    min_arc_length: int = 1,
) -> pd.DataFrame:
    gz_path = _yearly_sqlite_path(year)
    if not gz_path.exists():
        return pd.DataFrame(columns=["year", "party", "starters"])

    chamber_clause = ""
    if chambers is not None:
        placeholders = ",".join(str(int(c)) for c in chambers)
        chamber_clause = f"AND u.kammare IN ({placeholders})"

    sql = _build_starter_sql(max_gap, min_arc_length, key_column="p.party").format(
        chamber_clause=chamber_clause
    )

    with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
        with gzip.open(gz_path, "rb") as gz:
            tmp.write(gz.read())
        tmp.flush()
        conn = sqlite3.connect(tmp.name)
        try:
            df = pd.read_sql(sql, conn)
        finally:
            conn.close()
    return df


def read_topic_arc_starters_by_party(
    years: Iterable[int] | None = None,
    chambers: Iterable[int] | None = None,
    max_gap: int = 0,
    min_arc_length: int = 1,
) -> pd.DataFrame:
    """Per-(year, raw party) topic-arc-starter counts.

    Parallel to :func:`read_topic_arc_starters` (which keys by gender);
    ``party`` is the raw label recorded in ``person.party`` at the time
    of the utterance. Callers who want the paper's bloc pooling should
    apply :data:`src.plots.DEFAULT_PARTY_MAPPING` via
    :func:`src.plots._reduce_parties`. See :func:`read_topic_arc_starters`
    for the ``max_gap`` / ``min_arc_length`` semantics.
    """
    years_tuple = (
        tuple(years)
        if years is not None
        else tuple(range(settings.START_YEAR, settings.END_YEAR + 1))
    )
    chambers_tuple = tuple(chambers) if chambers is not None else None

    parts = [
        _read_yearly_starters_by_party(y, chambers_tuple, max_gap, min_arc_length)
        for y in years_tuple
    ]
    combined = pd.concat(parts, ignore_index=True)
    if combined.empty:
        return combined
    return combined.groupby(["year", "party"], as_index=False, dropna=False)[
        "starters"
    ].sum()


def _build_speaker_starter_sql(max_gap: int, min_arc_length: int = 1) -> str:
    """Per-speaker topic-arc-starter SQL. See :func:`_build_starter_sql`."""
    joins_sql, conds_sql = _starter_sql_body(max_gap, min_arc_length)
    return f"""
SELECT p.name AS who, p.gender AS gender, COUNT(*) AS starters
FROM utterance u
JOIN person p ON p.id = u.person_id
{joins_sql}
WHERE {conds_sql}
  {{chamber_clause}}
GROUP BY p.name, p.gender
"""


@functools.lru_cache(maxsize=128)
def _read_yearly_speaker_starters(
    year: int,
    chambers: tuple[int, ...] | None,
    max_gap: int,
    min_arc_length: int = 1,
) -> pd.DataFrame:
    gz_path = _yearly_sqlite_path(year)
    if not gz_path.exists():
        return pd.DataFrame(columns=["who", "gender", "starters", "year"])

    chamber_clause = ""
    if chambers is not None:
        placeholders = ",".join(str(int(c)) for c in chambers)
        chamber_clause = f"AND u.kammare IN ({placeholders})"

    sql = _build_speaker_starter_sql(max_gap, min_arc_length).format(
        chamber_clause=chamber_clause
    )

    with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
        with gzip.open(gz_path, "rb") as gz:
            tmp.write(gz.read())
        tmp.flush()
        conn = sqlite3.connect(tmp.name)
        try:
            df = pd.read_sql(sql, conn)
        finally:
            conn.close()
    df["year"] = year
    return df


def read_speaker_starter_counts(
    years: Iterable[int] | None = None,
    chambers: Iterable[int] | None = None,
    max_gap: int = 0,
    min_arc_length: int = 1,
) -> pd.DataFrame:
    """Per-(speaker, year) topic-arc starter counts.

    Same "starter" definition as :func:`read_topic_arc_starters`, but
    aggregated at the ``(who, gender, year)`` grain instead of
    ``(year, gender)``. Useful for per-speaker plots that ask which
    individual MPs originate vs. join woman-talk arcs.

    See :func:`read_topic_arc_starters` for the semantics of
    ``max_gap`` and ``min_arc_length``.

    Rows are aggregated across the shipped yearly sqlite files. Rows
    for speakers with zero starters in a given year are absent (SQL
    only emits groups with at least one starter); join against
    ``read_speakers()`` to add back the zero-starter speakers.
    """
    years_tuple = (
        tuple(years)
        if years is not None
        else tuple(range(settings.START_YEAR, settings.END_YEAR + 1))
    )
    chambers_tuple = tuple(chambers) if chambers is not None else None

    parts = [
        _read_yearly_speaker_starters(y, chambers_tuple, max_gap, min_arc_length)
        for y in years_tuple
    ]
    combined = pd.concat(parts, ignore_index=True)
    if combined.empty:
        return combined
    return combined[["who", "gender", "year", "starters"]]


def speaker_starts_chimes(
    speakers: pd.DataFrame | None = None,
    starters: pd.DataFrame | None = None,
    *,
    year_range: tuple[int, int] | None = None,
    chambers: Iterable[int] | None = None,
    min_utterances: int = 1,
    known_gender_only: bool = True,
    max_gap: int = 1,
    min_arc_length: int = 2,
) -> pd.DataFrame:
    """Per-(who, gender) starter + chime-in aggregate over a window.

    Standardised derivation used by every per-speaker starter/chime-in
    plot. Reads ``read_speakers()`` and ``read_speaker_starter_counts()``
    if the frames aren't supplied, filters to the ``year_range`` window
    (defaults to full corpus), and computes:

    * ``starters``   — topic-arc starter count, per speaker
    * ``chime_ins``  — ``k_all_utts - starters`` (floored at 0)
    * ``starter_rate = starters / utterance_count``
    * ``chime_rate  = chime_ins / utterance_count``

    Along with the pre-existing per-speaker aggregates
    (``utterance_count``, ``k_all_utts``) it forms a self-contained
    frame with everything a starter/chime plot needs. Callers wanting
    party breakdown (see :func:`plots.plot_party_speaker_starts_vs_chimes`)
    join dominant-bloc metadata on top of this frame.

    ``known_gender_only`` (default True) restricts to non-empty ``who``
    and ``gender in ("man", "woman")``. Set False for callers that want
    to keep unknown-gender speakers in the frame (e.g. per-bloc scatters
    where gender isn't the axis).

    ``max_gap`` / ``min_arc_length`` control the starter definition and
    are the L / S knobs used in the paper's sensitivity analysis
    (L ∈ {1,2,3}, S ∈ {0,1,2}). They are used only when ``starters``
    is fetched here; if the caller supplies a pre-loaded ``starters``
    frame, those kwargs are ignored (the caller's frame is authoritative).
    """
    if speakers is None:
        speakers = read_speakers()
    if starters is None:
        years_arg = (
            range(year_range[0], year_range[1] + 1) if year_range is not None else None
        )
        starters = read_speaker_starter_counts(
            years=years_arg,
            chambers=chambers,
            max_gap=max_gap,
            min_arc_length=min_arc_length,
        )

    if year_range is not None:
        y0, y1 = year_range
        speakers = speakers[(speakers["year"] >= y0) & (speakers["year"] <= y1)]
        starters = starters[(starters["year"] >= y0) & (starters["year"] <= y1)]

    if known_gender_only:
        speakers = speakers[
            (speakers["who"].astype(str) != "")
            & (speakers["gender"].isin(("man", "woman")))
        ]

    totals = speakers.groupby(["who", "gender"], as_index=False, dropna=False)[
        ["utterance_count", "k_all_utts"]
    ].sum()

    starters_agg = starters.groupby(["who", "gender"], as_index=False, dropna=False)[
        "starters"
    ].sum()

    m = totals.merge(starters_agg, on=["who", "gender"], how="left")
    m["starters"] = m["starters"].fillna(0).astype(int)
    m["chime_ins"] = (m["k_all_utts"] - m["starters"]).clip(lower=0).astype(int)
    m = m[m["utterance_count"] >= max(1, min_utterances)].copy()
    m["starter_rate"] = m["starters"] / m["utterance_count"]
    m["chime_rate"] = m["chime_ins"] / m["utterance_count"]
    return m


@functools.lru_cache(maxsize=128)
def _read_yearly_starters(
    year: int,
    chambers: tuple[int, ...] | None,
    max_gap: int,
    min_arc_length: int = 1,
) -> pd.DataFrame:
    gz_path = _yearly_sqlite_path(year)
    if not gz_path.exists():
        return pd.DataFrame(columns=["year", "gender", "starters"])

    chamber_clause = ""
    if chambers is not None:
        placeholders = ",".join(str(int(c)) for c in chambers)
        chamber_clause = f"AND u.kammare IN ({placeholders})"

    sql = _build_starter_sql(max_gap, min_arc_length).format(
        chamber_clause=chamber_clause
    )

    with tempfile.NamedTemporaryFile(suffix=".sqlite3") as tmp:
        with gzip.open(gz_path, "rb") as gz:
            tmp.write(gz.read())
        tmp.flush()
        conn = sqlite3.connect(tmp.name)
        try:
            df = pd.read_sql(sql, conn)
        finally:
            conn.close()
    return df


def read_topic_arc_starters(
    years: Iterable[int] | None = None,
    chambers: Iterable[int] | None = None,
    max_gap: int = 0,
    min_arc_length: int = 1,
) -> pd.DataFrame:
    """Per-year gender-count of topic-arc starters of woman-talk.

    A "starter" is a kvinna utterance where the preceding
    ``max_gap + 1`` utterances in the chamber's ``prev/next`` chain
    all had no kvinna hit. ``max_gap=0`` (default) is the strict
    definition: any non-kvinna predecessor makes this a starter, so a
    contiguous run of kvinna utterances counts as one arc. ``max_gap=1``
    tolerates one non-kvinna utterance in the middle of an arc without
    breaking it — an "exchange about women" that survives a single
    interjection.

    ``min_arc_length=2`` restricts to starters that open an arc with
    at least one more kvinna utterance forward (within the same gap
    window). This excludes isolated single-utterance mentions, which
    dominate the ``min_arc_length=1`` (default) count — 70–80 % of
    "starters" under the default are one-off mentions, not the start
    of an actual exchange.

    Returns rows keyed by ``(year, gender)`` with a ``starters`` count.
    ``gender`` is one of ``"man"``, ``"woman"``, or NaN (unknown). Rows
    are aggregated across the shipped ``ToK_data_YYYY.sqlite3.gz``
    files; the ``years`` and ``chambers`` arguments filter which files
    and chambers contribute.
    """
    years_tuple = (
        tuple(years)
        if years is not None
        else tuple(range(settings.START_YEAR, settings.END_YEAR + 1))
    )
    chambers_tuple = tuple(chambers) if chambers is not None else None

    parts = [
        _read_yearly_starters(y, chambers_tuple, max_gap, min_arc_length)
        for y in years_tuple
    ]
    combined = pd.concat(parts, ignore_index=True)
    if combined.empty:
        return combined

    return combined.groupby(["year", "gender"], as_index=False, dropna=False)[
        "starters"
    ].sum()


_WORD_SUBSETS_CSV = settings.root / "word_subsets.csv.gz"


def read_kvinna_word_subsets(
    years: Iterable[int] | None = None,
    chambers: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Per-subset token counts for {K1, K2, K3}.

    Returns exactly 7 rows keyed by the (k1, k2, k3) indicator triple,
    with a ``word_match_count`` column. Sources from the shipped
    ``word_subsets.csv.gz``, which is emitted by the ``word_frequencies``
    pipeline in the same pass that produces ``k{n}_matches`` — so
    subset totals are guaranteed to match the aggregate columns.
    """

    df = pd.read_csv(_WORD_SUBSETS_CSV)
    if years is not None:
        df = df[df["year"].isin(list(years))]
    if chambers is not None:
        df = df[df["chamber"].isin(list(chambers))]

    out = (
        df.groupby(["k1", "k2", "k3"], as_index=False)["word_match_count"]
        .sum()
        .astype({"k1": bool, "k2": bool, "k3": bool})
    )

    return out
