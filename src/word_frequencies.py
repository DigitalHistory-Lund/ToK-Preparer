"""Compute per-category frequency data and ship as CSV.gz.

Reads utterance content from ``tmp_db.sqlite3`` (populated by
``prepare_db``) and aggregates two views in a single pass:

- ``word_frequencies.csv.gz`` — per (year, chamber, gender, party):
  utterance/word/char counts and both word-level ``k{n}_matches`` and
  utterance-level ``k{n}_utts`` totals for each Kvinna category.
- ``speakers.csv.gz`` — per (year, chamber, gender, party, who):
  the same utterance-level counters, so distinct-speaker counts under
  any grouping become ``df.groupby(cols)['who'].nunique()``.

Chamber-split variants (`.1.csv.gz`, `.2.csv.gz`) are emitted for both.

See ``docs/word_frequencies.md`` for the methodology and column schema.
"""

from __future__ import annotations

import csv
import gzip
import io
import re
import sqlite3
from collections import defaultdict

from tqdm.auto import tqdm

from .check_queries import validate
from .queries import queries
from .settings import root, tmp_db

_WORD_RE = re.compile(r"\w+")

CATEGORY_ORDER = ("kvinna 1", "Kvinna 2", "Kvinna 3")
CATEGORY_INDEX = {c: i for i, c in enumerate(CATEGORY_ORDER)}

# Exact tokens tallied on their own alongside K1/K2/K3.
# Each word gets ``{w}_matches`` (per-token hits) and ``{w}_utts`` (utterances
# containing >=1 hit). Counting is additive to K-categories: e.g. "fru" bumps
# both ``k2_matches`` and ``fru_matches``.
TRACKED_WORDS: tuple[str, ...] = ("damer", "fru")
TRACKED_WORDS_SET = frozenset(TRACKED_WORDS)

WORD_FREQ_COLUMNS = (
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
) + tuple(
    f"{w}_{suffix}" for w in TRACKED_WORDS for suffix in ("matches", "utts")
)

SPEAKERS_COLUMNS = (
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

WORD_SUBSETS_COLUMNS = (
    "year",
    "chamber",
    "k1",
    "k2",
    "k3",
    "word_match_count",
)


def _partition_patterns() -> tuple[
    dict[str, set[str]],
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, list[str]],
    list[str],
]:
    """Return (literals, prefixes, suffixes, contains, k3_phrases).

    Pattern shapes:
      literal   "mor"           → literals[cat]
      prefix    "änke*"         → prefixes[cat]  (trailing "*")
      suffix    "*inna"         → suffixes[cat]  (leading "*")
      contains  "*hustru*"      → contains[cat]  (both "*" — token contains substring)
      phrase    "fru grefvinna" → k3_phrases     (K3-only)
    """
    literals: dict[str, set[str]] = {c: set() for c in CATEGORY_ORDER}
    prefixes: dict[str, list[str]] = {c: [] for c in CATEGORY_ORDER}
    suffixes: dict[str, list[str]] = {c: [] for c in CATEGORY_ORDER}
    contains: dict[str, list[str]] = {c: [] for c in CATEGORY_ORDER}
    k3_phrases: list[str] = []
    for category, patterns in queries.items():
        for pattern in patterns:
            lowered = pattern.lower()
            if " " in lowered:
                assert category == "Kvinna 3", (
                    f"multi-word patterns are assumed to live in Kvinna 3, "
                    f"but found {pattern!r} in {category!r}"
                )
                k3_phrases.append(lowered)
            elif lowered.startswith("*") and lowered.endswith("*"):
                contains[category].append(lowered[1:-1])
            elif lowered.startswith("*"):
                suffixes[category].append(lowered[1:])
            elif lowered.endswith("*"):
                prefixes[category].append(lowered[:-1])
            else:
                literals[category].add(lowered)
    for category in CATEGORY_ORDER:
        prefixes[category].sort(key=lambda p: (-len(p), p))
        suffixes[category].sort(key=lambda p: (-len(p), p))
        contains[category].sort(key=lambda p: (-len(p), p))
    k3_phrases.sort(key=lambda p: (-len(p), p))
    return literals, prefixes, suffixes, contains, k3_phrases


def _classify(
    token: str,
    literals: dict[str, set[str]],
    prefixes: dict[str, list[str]],
    suffixes: dict[str, list[str]],
    contains: dict[str, list[str]] | None = None,
) -> list[int]:
    """Return indices of all matching categories (empty = no match).

    Within a single K category, a token counts at most once (first pattern
    match wins and we move on to the next category).  A token may appear in
    multiple categories, contributing 1 to each.

    ``contains`` is optional for backward compatibility; when omitted the
    contains-substring shape is not evaluated (used by legacy callers).
    """
    hits: list[int] = []
    for i, category in enumerate(CATEGORY_ORDER):
        if token in literals[category]:
            hits.append(i)
            continue
        matched = False
        for prefix in prefixes[category]:
            if token.startswith(prefix):
                hits.append(i)
                matched = True
                break
        if matched:
            continue
        for suffix in suffixes[category]:
            if token.endswith(suffix):
                hits.append(i)
                matched = True
                break
        if matched:
            continue
        if contains is not None:
            for substring in contains[category]:
                if substring in token:
                    hits.append(i)
                    break
    return hits


def _iter_utterances(conn: sqlite3.Connection):
    return conn.execute(
        """
        SELECT u.year, u.kammare, u.gender, u.party, u.who, uf.content
        FROM utterance u
        JOIN utterance_fts uf ON u.id = uf.id
        """
    )


def _empty_bucket() -> dict[str, int]:
    bucket = {
        "utterance_count": 0,
        "total_words": 0,
        "char_count": 0,
        "k1_utts": 0,
        "k2_utts": 0,
        "k3_utts": 0,
        "k_all_utts": 0,
        "k1_matches": 0,
        "k2_matches": 0,
        "k3_matches": 0,
        "k_all_matches": 0,
    }
    for w in TRACKED_WORDS:
        bucket[f"{w}_matches"] = 0
        bucket[f"{w}_utts"] = 0
    return bucket


def _empty_speaker_bucket() -> dict[str, int]:
    return {
        "utterance_count": 0,
        "k1_utts": 0,
        "k2_utts": 0,
        "k3_utts": 0,
        "k_all_utts": 0,
    }


def compute():
    """Return (buckets, speakers, word_subsets) aggregated over the corpus.

    ``word_subsets`` maps ``(year, chamber, frozenset(cats))`` to the
    number of tokens (and K3 phrase hits, treated as ``frozenset({2})``)
    matching that specific K-subset. Sums across all subsets per (year,
    chamber) equal the corpus's ``k_all_matches`` for that grouping.
    """
    literals, prefixes, suffixes, contains, k3_phrases = _partition_patterns()

    buckets: dict[tuple, dict[str, int]] = defaultdict(_empty_bucket)
    speakers: dict[tuple, dict[str, int]] = defaultdict(_empty_speaker_bucket)
    word_subsets: dict[tuple, int] = defaultdict(int)

    with sqlite3.connect(f"file:{tmp_db}?mode=ro", uri=True) as conn:
        total = conn.execute("SELECT COUNT(*) FROM utterance").fetchone()[0]
        for year, chamber, gender, party, who, content in tqdm(
            _iter_utterances(conn), total=total, desc="Utterances"
        ):
            if content is None:
                continue
            key = (year, chamber, gender, party)
            bucket = buckets[key]
            speaker_key = (year, chamber, gender, party, who)
            speaker = speakers[speaker_key]

            content_lower = content.lower()

            utt_hits = [False, False, False]  # k1, k2, k3
            k_all_matches = 0
            for phrase in k3_phrases:
                phrase_hits = content_lower.count(phrase)
                if phrase_hits:
                    bucket["k3_matches"] += phrase_hits
                    utt_hits[2] = True
                    k_all_matches += phrase_hits
                    word_subsets[(year, chamber, frozenset({2}))] += phrase_hits

            tokens = _WORD_RE.findall(content_lower)
            tracked_hits = {w: False for w in TRACKED_WORDS}
            for token in tokens:
                cats = _classify(token, literals, prefixes, suffixes, contains)
                if cats:
                    for cat in cats:
                        bucket[f"k{cat + 1}_matches"] += 1
                        utt_hits[cat] = True
                    k_all_matches += 1
                    word_subsets[(year, chamber, frozenset(cats))] += 1
                if token in TRACKED_WORDS_SET:
                    bucket[f"{token}_matches"] += 1
                    tracked_hits[token] = True

            bucket["utterance_count"] += 1
            bucket["total_words"] += len(tokens)
            bucket["char_count"] += len(content)
            for i, hit in enumerate(utt_hits):
                if hit:
                    bucket[f"k{i + 1}_utts"] += 1
            for w, hit in tracked_hits.items():
                if hit:
                    bucket[f"{w}_utts"] += 1
            bucket["k_all_matches"] += k_all_matches
            if any(utt_hits):
                bucket["k_all_utts"] += 1

            speaker["utterance_count"] += 1
            for i, hit in enumerate(utt_hits):
                if hit:
                    speaker[f"k{i + 1}_utts"] += 1
            if any(utt_hits):
                speaker["k_all_utts"] += 1

    return buckets, speakers, word_subsets


def _stringify(value):
    return "" if value is None else value


def _bucket_rows(buckets):
    def sort_key(k):
        return tuple(_stringify(v) for v in k)

    for key in sorted(buckets.keys(), key=sort_key):
        b = buckets[key]
        year, chamber, gender, party = key
        row = (
            year,
            chamber,
            _stringify(gender),
            _stringify(party),
            b["utterance_count"],
            b["total_words"],
            b["char_count"],
            b["k1_utts"],
            b["k2_utts"],
            b["k3_utts"],
            b["k_all_utts"],
            b["k1_matches"],
            b["k2_matches"],
            b["k3_matches"],
            b["k_all_matches"],
        )
        for w in TRACKED_WORDS:
            row += (b[f"{w}_matches"], b[f"{w}_utts"])
        yield row


def _word_subset_rows(word_subsets):
    """Yield (year, chamber, k1, k2, k3, word_match_count) rows.

    Skips empty subsets (would-be all-False row). Sorted by (year,
    chamber, k1, k2, k3) for stable diffs.
    """
    def sort_key(item):
        (year, chamber, subset), _count = item
        return (year, chamber, 0 in subset, 1 in subset, 2 in subset)

    for (year, chamber, subset), count in sorted(word_subsets.items(), key=sort_key):
        if not subset:
            continue
        yield (
            year,
            chamber,
            int(0 in subset),
            int(1 in subset),
            int(2 in subset),
            count,
        )


def _speaker_rows(speakers):
    def sort_key(k):
        return tuple(_stringify(v) for v in k)

    for key in sorted(speakers.keys(), key=sort_key):
        s = speakers[key]
        year, chamber, gender, party, who = key
        yield (
            year,
            chamber,
            _stringify(gender),
            _stringify(party),
            _stringify(who),
            s["utterance_count"],
            s["k1_utts"],
            s["k2_utts"],
            s["k3_utts"],
            s["k_all_utts"],
        )


def _write_csv_gz(path, columns, rows):
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    data = buf.getvalue().encode("utf-8")
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as f:
        f.write(data)


def _write_chamber_slices(base_name: str, columns, rows, chamber_index: int):
    rows = list(rows)
    _write_csv_gz(root / f"{base_name}.csv.gz", columns, rows)
    _write_csv_gz(
        root / f"{base_name}.1.csv.gz",
        columns,
        [r for r in rows if r[chamber_index] == 1],
    )
    _write_csv_gz(
        root / f"{base_name}.2.csv.gz",
        columns,
        [r for r in rows if r[chamber_index] == 2],
    )


def write_outputs(buckets, speakers, word_subsets):
    _write_chamber_slices(
        "word_frequencies", WORD_FREQ_COLUMNS, _bucket_rows(buckets), chamber_index=1
    )
    _write_chamber_slices(
        "speakers", SPEAKERS_COLUMNS, _speaker_rows(speakers), chamber_index=1
    )
    _write_csv_gz(
        root / "word_subsets.csv.gz",
        WORD_SUBSETS_COLUMNS,
        _word_subset_rows(word_subsets),
    )


def main():
    validate()
    buckets, speakers, word_subsets = compute()
    write_outputs(buckets, speakers, word_subsets)
    print(
        f"word_frequencies: {len(buckets)} bucket rows written; "
        f"speakers: {len(speakers)} rows written; "
        f"word_subsets: {len(word_subsets)} entries written"
    )


if __name__ == "__main__":
    main()
