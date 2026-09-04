from itertools import batched
from tqdm.auto import tqdm


from collections import defaultdict

import sqlite3
import json
import gzip

from math import ceil

from .settings import (
    BATCH_SIZE,
    EXPECTED_COUNT,
    START_YEAR,
    END_YEAR,
    data_dir,
    tmp_db,
)

from .queries import queries

import logging


from collections import namedtuple

Utterance = namedtuple(
    "utterance",
    ["id", "text", "who", "year", "date", "kammare", "record", "number"],
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="prepare_db.log",
)


CHAMBER_MAP = {
    "Första kammaren": 1,
    "Andra kammaren": 2,
}


NDJSON_FILE = "records_speeches.ndjson.gz"
PERSONS_DB = "persons.sqlite"
PERSON_TABLES = (
    "processed_member_of_parliament",
    "processed_minister",
    "processed_speaker",
)


def datestr_to_int(date_str):
    """Convert YYYY-MM-DD to an int with all digits"""
    year, mon, day = date_str.split("-")
    if not all(
        (
            len(date_str) == 10,
            len(year) == 4,
            len(mon) == 2,
            len(day) == 2,
            year.isdigit(),
            mon.isdigit(),
            day.isdigit(),
        )
    ):
        raise ValueError(f'{date_str=} does not adhere to "YYYY-MM-DD"')

    no_dashes = date_str.replace("-", "")
    if len(no_dashes) != 8:
        raise ValueError
    return int(no_dashes)


def _normalize_date_bounds(start, end):
    """Turn free-form YYYY / YYYY-MM / YYYY-MM-DD (optionally with trailing time) into intified sentinels."""
    start = start.split(" ", 1)[0] if start else start
    end = end.split(" ", 1)[0] if end else end
    if start is None or len(start) == 0:
        start_int = 0
    elif len(start) == 4:
        start_int = datestr_to_int(start + "-01-01")
    elif len(start) == 7:
        start_int = datestr_to_int(start + "-01")
    else:
        start_int = datestr_to_int(start)
    if end is None or len(end) == 0:
        end_int = 99999999
    elif len(end) == 4:
        end_int = datestr_to_int(end + "-12-31")
    elif len(end) == 7:
        end_int = datestr_to_int(end + "-31")
    else:
        end_int = datestr_to_int(end)
    return start_int, end_int


def iter_utterances():
    """Stream Utterance tuples from the ndjson corpus, filtered to the configured year range."""
    path = data_dir / NDJSON_FILE
    if not path.exists():
        raise FileNotFoundError(f'Could not find "{path}"')
    with gzip.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            kammare = CHAMBER_MAP.get(row.get("chamber"))
            if kammare is None:
                continue
            start_date = row.get("start_date")
            if not start_date:
                continue
            year = int(start_date[:4])
            if not (START_YEAR <= year <= END_YEAR):
                continue
            yield Utterance(
                id=row["speech"],
                text=row.get("text") or "",
                who=row.get("who") or "unknown",
                year=year,
                date=datestr_to_int(start_date),
                kammare=kammare,
                record=row.get("record") or "",
                number=row.get("number") or 0,
            )


def load_person_dates_affiliation():
    """Yield (person_id, start_int, end_int, party) tuples from persons.sqlite.

    Prefers the descriptive ``party`` string when set; falls back to
    ``party_abbrev`` when it isn't. This recovers attributions for rows
    whose source only carries the abbreviation — a small but meaningful
    fraction of persons.sqlite where ``party`` is empty but
    ``party_abbrev`` is not.
    """
    union_sql = " UNION ALL ".join(
        f"SELECT person_id, start, end, "
        f"COALESCE(NULLIF(party, ''), party_abbrev) AS party "
        f"FROM {tbl} "
        f"WHERE COALESCE(NULLIF(party, ''), NULLIF(party_abbrev, '')) IS NOT NULL"
        for tbl in PERSON_TABLES
    )
    with sqlite3.connect(f"file:{data_dir / PERSONS_DB}?mode=ro", uri=True) as conn:
        for person_id, start, end, party in conn.execute(union_sql):
            start_int, end_int = _normalize_date_bounds(start, end)
            yield person_id, start_int, end_int, party


def load_id_to_gender():
    """Return {person_id: gender} coalesced across all three person tables."""
    id_to_gender = defaultdict(lambda: None)
    union_sql = " UNION ".join(
        f"SELECT person_id, gender FROM {tbl}" for tbl in PERSON_TABLES
    )
    with sqlite3.connect(f"file:{data_dir / PERSONS_DB}?mode=ro", uri=True) as conn:
        for person_id, gender in conn.execute(union_sql):
            if id_to_gender[person_id] is None and gender:
                id_to_gender[person_id] = gender
    return id_to_gender


def create_database():
    with sqlite3.connect(tmp_db) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE utterance (
                id str primary key,
                prev text,
                next text,
                who text not null,
                year int,
                date int,
                kammare int,
                record text,
                number int,
                gender text,
                party text,
                kvinna_1 bool,
                kvinna_2 bool,
                kvinna_3 bool
            )
        """)
        cur.execute("CREATE VIRTUAL TABLE utterance_fts USING fts5(id, content)")
        cur.execute(
            "CREATE VIRTUAL TABLE reverse_utterance_fts USING fts5(id, content)"
        )

        cur.execute(
            "CREATE TABLE affiliation (who text, start int, end int, party text)"
        )

        cur.execute("CREATE index next_index on utterance(next)")
        cur.execute("CREATE index prev_index on utterance(prev)")
        cur.execute("CREATE index who_index on utterance(who)")
        cur.execute("CREATE index year_index on utterance(year)")
        cur.execute("CREATE index kammare_index on utterance(kammare)")

        cur.executemany(
            "INSERT INTO affiliation (who, start, end, party) values (?,?,?,?)",
            load_person_dates_affiliation(),
        )
        cur.execute("CREATE index aff_index on affiliation(who)")


def seed_database():
    id_to_gender = load_id_to_gender()

    with sqlite3.connect(tmp_db) as conn:
        cur = conn.cursor()
        for batch in tqdm(
            batched(
                tqdm(
                    iter_utterances(),
                    position=1,
                    leave=True,
                    total=EXPECTED_COUNT,
                    desc="Utterances",
                ),
                BATCH_SIZE,
            ),
            total=ceil(EXPECTED_COUNT / BATCH_SIZE),
            desc="Writing utterances to DB",
            position=2,
            leave=True,
        ):
            data = [
                {
                    "id": u_id,
                    "content": text,
                    "reverse_content": text[::-1],
                    "who": who,
                    "year": year,
                    "gender": id_to_gender[who],
                    "date": date,
                    "kammare": kammare,
                    "record": record,
                    "number": number,
                }
                for u_id, text, who, year, date, kammare, record, number in batch
            ]

            cur.executemany(
                "INSERT INTO utterance_fts (id, content) values (:id, :content)", data
            )
            cur.executemany(
                "INSERT INTO reverse_utterance_fts (id, content) values (:id, :reverse_content)",
                data,
            )
            cur.executemany(
                "INSERT INTO utterance (id, who, year, gender, date, kammare, record, number) values (:id, :who, :year, :gender, :date, :kammare, :record, :number)",
                data,
            )

            conn.commit()

        # prev/next links per chamber, ordered by (date, record, number) from the source corpus.
        cur.execute("""
        WITH ordered AS (
            SELECT id,
                   LAG(id)  OVER w AS prev,
                   LEAD(id) OVER w AS next
            FROM utterance
            WINDOW w AS (PARTITION BY kammare ORDER BY date, record, number)
        )
        UPDATE utterance
        SET prev = ordered.prev,
            next = ordered.next
        FROM ordered
        WHERE utterance.id = ordered.id
        """)

        cur.execute("""
        UPDATE utterance
        SET party = (
            SELECT party
            FROM affiliation
            WHERE utterance.who = affiliation.who
            AND date BETWEEN start AND end
        )
        WHERE EXISTS (
            SELECT 1
            FROM affiliation
            WHERE utterance.who = affiliation.who
            AND date BETWEEN start AND end
        )
        """)


def _split_patterns_by_shape(query_terms):
    """Return (forward_terms, suffix_terms).

    forward_terms: literals, ``X*`` prefix wildcards, and phrase patterns —
        all consumable by ``utterance_fts MATCH``.
    suffix_terms:  ``*X`` suffix wildcards — require ``reverse_utterance_fts``.
    ``*X*`` contains-patterns are expanded into BOTH a forward ``X*`` and a
    suffix ``*X`` so any token where the substring appears at the boundary
    of the FTS-indexed token gets caught. Substring hits in the interior
    of a token are out of scope — FTS5 doesn't index at that granularity.
    """
    fwd, suf = [], []
    for t in query_terms:
        if t.startswith("*") and t.endswith("*"):
            core = t[1:-1]
            fwd.append(core + "*")
            suf.append("*" + core)
        elif t.startswith("*"):
            suf.append(t)
        else:
            fwd.append(t)
    return fwd, suf


def _reverse_suffix_pattern(pattern):
    """Rewrite a suffix wildcard `*X` into a prefix wildcard against reversed content.

    ``*inna`` → ``anni*``.
    """
    assert pattern.startswith("*") and not pattern.endswith("*"), pattern
    return pattern[1:][::-1] + "*"


def tag_utterances_by_query():
    with sqlite3.connect(tmp_db) as conn:
        cur = conn.cursor()
        for label, query_terms in queries.items():
            col = label.replace(" ", "_").lower()
            fwd_terms, suf_terms = _split_patterns_by_shape(query_terms)

            if fwd_terms:
                fwd_query = " OR ".join(fwd_terms)
                logging.info(
                    f'Tagging "{label}" via utterance_fts with "{fwd_query}"'
                )
                cur.execute(
                    f"""
                    UPDATE utterance
                    SET {col} = 1
                    WHERE id IN (
                        SELECT id FROM utterance_fts WHERE content MATCH ?
                    )
                    """,
                    (fwd_query,),
                )

            if suf_terms:
                rev_terms = [_reverse_suffix_pattern(t) for t in suf_terms]
                rev_query = " OR ".join(rev_terms)
                logging.info(
                    f'Tagging "{label}" via reverse_utterance_fts with "{rev_query}" '
                    f"(from suffixes {suf_terms})"
                )
                cur.execute(
                    f"""
                    UPDATE utterance
                    SET {col} = 1
                    WHERE id IN (
                        SELECT id FROM reverse_utterance_fts WHERE content MATCH ?
                    )
                    """,
                    (rev_query,),
                )
        conn.commit()


def count_baselines():
    with sqlite3.connect(tmp_db) as conn:
        cur = conn.cursor()

        logging.info(
            cur.execute(
                'select count(*) from utterance_fts where content match "kvinna AND kvinnor"'
            ).fetchall()
        )
        logging.info(
            cur.execute(
                'select count(*) from utterance_fts where content match "kvinna"'
            ).fetchall()
        )
        logging.info(
            cur.execute(
                'select count(*) from utterance_fts where content match "kvinnor"'
            ).fetchall()
        )
        logging.info(
            cur.execute(
                'select count(*) from utterance_fts where content match "kvinna OR kvinnor"'
            ).fetchall()
        )
        logging.info(
            cur.execute(
                'select count(*) from utterance_fts where content match "kvinn*"'
            ).fetchall()
        )


def prepare_database():
    from .check_queries import validate
    validate()
    if not tmp_db.exists():
        create_database()
    seed_database()
    count_baselines()
    tag_utterances_by_query()

if __name__ == "__main__":
    prepare_database()
