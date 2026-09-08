from .settings import tmp_db, out_db, root, START_YEAR, END_YEAR
from .queries import queries
import sqlite3
import re
from itertools import batched
from typing import Iterable
from tqdm import tqdm
import gzip

SUPERSCRIPTS = {"kvinna 1": "\u00b9", "Kvinna 2": "\u00b2", "Kvinna 3": "\u00b3"}
_WORD_RE = re.compile(r"\w+")


def compute_discussion_ids(
    session_sequences: Iterable[Iterable[tuple[str, bool]]],
) -> dict[str, int]:
    """Assign a ``discussion_id`` to every utterance in an arc.

    Uses the paper's shipped defaults (``max_gap=1``, ``min_arc_length=2``):
    an arc is a maximal chain-run in which no two consecutive utterances
    are both non-kvinna; arcs shorter than 2 kvinna utterances are dropped;
    the single non-kvinna interjection that ``max_gap=1`` tolerates
    receives the surrounding arc's id.

    Parameters
    ----------
    session_sequences
        Iterable of per-session ordered iterables. Each inner element
        is a ``(utterance_id, is_kvinna)`` tuple in chain order within
        one *session* (one day's parliamentary sitting \u2014 one ``record``
        in swerik's ``prot-YYYY--{fk|ak}--NNN`` scheme). Arcs cannot
        span sessions: the two halves of a discussion that resumes on
        a following day get separate ids. Chamber locality is inherent
        because a session is chamber-scoped by definition. Ids are
        assigned from a single monotone counter starting at 1 so every
        id in the returned dict is globally unique.

    Returns
    -------
    dict[str, int]
        ``{utterance_id: discussion_id}``. Utterances not in any arc
        (isolated single kvinna, unconfirmed leading/trailing interjection,
        pure non-kvinna outside all arcs) are omitted \u2014 callers should
        treat absence as SQL ``NULL``.
    """
    out: dict[str, int] = {}
    next_arc_id = 1
    for session in session_sequences:
        arc: list[str] = []
        arc_kvinna_count = 0
        pending: str | None = None
        for utt_id, is_kvinna in session:
            if is_kvinna:
                if pending is not None:
                    arc.append(pending)
                    pending = None
                arc.append(utt_id)
                arc_kvinna_count += 1
            else:
                if not arc:
                    continue  # pre-arc non-kvinna; nothing to interject
                if pending is None:
                    pending = utt_id
                else:
                    # Two consecutive non-kvinna \u2192 close the arc.
                    if arc_kvinna_count >= 2:
                        for member_id in arc:
                            out[member_id] = next_arc_id
                        next_arc_id += 1
                    arc = []
                    arc_kvinna_count = 0
                    pending = None
        # End of session: close any open arc (drop trailing pending).
        if arc and arc_kvinna_count >= 2:
            for member_id in arc:
                out[member_id] = next_arc_id
            next_arc_id += 1
    return out


def _matches_pattern(word_lower, pattern):
    if pattern.startswith("*") and pattern.endswith("*"):
        return pattern[1:-1] in word_lower
    if pattern.startswith("*"):
        return word_lower.endswith(pattern[1:])
    if pattern.endswith("*"):
        return word_lower.startswith(pattern[:-1])
    return word_lower == pattern


def annotate_content(content, k1, k2, k3):
    """Add unicode superscript category markers to detected words."""
    active = []
    if k1:
        active.append(("kvinna 1", queries["kvinna 1"]))
    if k2:
        active.append(("Kvinna 2", queries["Kvinna 2"]))
    if k3:
        active.append(("Kvinna 3", queries["Kvinna 3"]))
    if not active:
        return content

    def replace_word(match):
        word = match.group(0)
        word_lower = word.lower()
        markers = ""
        for label, patterns in active:
            for pattern in patterns:
                if _matches_pattern(word_lower, pattern):
                    markers += SUPERSCRIPTS[label]
                    break
        return word + markers if markers else word

    return _WORD_RE.sub(replace_word, content)

if __name__ == "__main__":
    if out_db.exists():
        raise FileExistsError(f"{out_db} already exists. Remove it before proceeding.")
    if not tmp_db.exists():
        raise FileNotFoundError(f"{tmp_db} does not exist. Run prepare_db.py first.")

    with sqlite3.connect(tmp_db) as source_conn:
        source_cur = source_conn.cursor()

        with sqlite3.connect(out_db) as target_conn:
            target_cur = target_conn.cursor()

            # Create person table
            target_cur.execute("""
                CREATE TABLE person (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    gender TEXT,
                    party TEXT,
                    UNIQUE(name, gender, party)
                )
            """)

            # Create normalized utterance table
            target_cur.execute("""
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
                    kvinna_3 BOOLEAN,
                    discussion_id INTEGER,
                    FOREIGN KEY (person_id) REFERENCES person(id)
                )
            """)

            # Extract unique persons from source
            source_cur.execute("""
                SELECT DISTINCT who, gender, party
                FROM utterance
                ORDER BY who
            """)

            persons = source_cur.fetchall()
            person_map = {}  # (who, gender, party) -> person_id

            for who, gender, party in tqdm(persons, desc="Inserting persons"):
                target_cur.execute(
                    "INSERT INTO person (name, gender, party) VALUES (?, ?, ?)",
                    (who, gender, party),
                )
                person_id = target_cur.lastrowid
                person_map[(who, gender, party)] = person_id

            target_conn.commit()
            print(f"Inserted {len(persons)} persons.")

            # Migrate utterances in batches
            batch_size = 10_000
            offset = 0
            total_count = source_cur.execute(
                "SELECT COUNT(*) FROM utterance"
            ).fetchone()[0]

            for batch in tqdm(
                batched(
                    source_cur.execute("""
                    SELECT id, prev, next, who, year, date, kammare, gender, party,
                        kvinna_1, kvinna_2, kvinna_3, content
                    FROM utterance join utterance_fts USING(id)
                """),
                    batch_size,
                ),
                total=total_count // batch_size,
                desc="Migrating utterances",
            ):
                # Transform and insert
                transformed = []
                for row in batch:
                    (
                        id_,
                        prev,
                        next_,
                        who,
                        year,
                        date,
                        kammare,
                        gender,
                        party,
                        k1,
                        k2,
                        k3,
                        content,
                    ) = row
                    person_id = person_map[(who, gender, party)]
                    if k1 or k2 or k3:
                        content = annotate_content(content, k1, k2, k3)
                    transformed.append(
                        (
                            id_,
                            prev,
                            next_,
                            person_id,
                            year,
                            date,
                            kammare,
                            k1,
                            k2,
                            k3,
                            content,
                        )
                    )

                target_cur.executemany(
                    """
                    INSERT INTO utterance
                    (id, prev, next, person_id, year, date, kammare, kvinna_1, kvinna_2, kvinna_3, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    transformed,
                )

                target_conn.commit()

            target_conn.commit()

            # Populate discussion_id: assign a stable arc id to every
            # utterance that belongs to a paper-default topic arc
            # (max_gap=1, min_arc_length=2). Arcs cannot span sessions,
            # where "session" = one day's parliamentary sitting = one
            # `record` in swerik's prot-YYYY--{fk|ak}--NNN scheme. We
            # source ordering from tmp_db because target_conn dropped
            # `record`/`number` at migration time. `record` already
            # encodes chamber (fk/ak) and date, so partitioning by
            # `record` alone is sufficient — no need to also partition
            # by `kammare`.
            session_sequences: list[list[tuple[str, bool]]] = []
            current_record: str | None = None
            current: list[tuple[str, bool]] = []
            for uid, record, is_kvinna in source_cur.execute("""
                SELECT id, record,
                       (COALESCE(kvinna_1, 0) OR COALESCE(kvinna_2, 0)
                                              OR COALESCE(kvinna_3, 0)) AS is_kvinna
                FROM utterance
                ORDER BY kammare, record, number
            """):
                if record != current_record:
                    if current:
                        session_sequences.append(current)
                    current = []
                    current_record = record
                current.append((uid, bool(is_kvinna)))
            if current:
                session_sequences.append(current)

            id_to_arc = compute_discussion_ids(session_sequences)
            target_cur.executemany(
                "UPDATE utterance SET discussion_id = ? WHERE id = ?",
                [(arc_id, uid) for uid, arc_id in id_to_arc.items()],
            )
            target_cur.execute(
                "CREATE INDEX discussion_idx ON utterance(discussion_id)"
            )
            target_conn.commit()
            print(
                f"Tagged {len(id_to_arc):,} utterances across "
                f"{max(id_to_arc.values(), default=0):,} topic-arc discussions."
            )

            years = range(START_YEAR, END_YEAR + 1)
            for year in tqdm(years, desc="Creating yearly DBs", total=len(years)):
                year_db = root / f"ToK_data_{year}.sqlite3"
                with sqlite3.connect(year_db) as year_conn:
                    target_conn.backup(year_conn)
                    year_conn.execute("DELETE FROM utterance WHERE year != ?", (year,))
                    year_conn.execute(
                        "delete from person where id not in (select distinct person_id from utterance)"
                    )
                    year_conn.commit()
                    year_conn.execute("VACUUM")
                with open(year_db, "rb") as raw_db:
                    with gzip.open(
                        year_db.with_suffix(".sqlite3.gz"), "wb"
                    ) as compressed_db:
                        compressed_db.writelines(raw_db)

    print(f"Reduced database {tmp_db.name} to {out_db.name}.")
    print(f"size before: {tmp_db.stat().st_size / 1e6:.2f} MB")
    print(f"size after:  {out_db.stat().st_size / 1e6:.2f} MB")

    gzip_filename = out_db.with_suffix(".sqlite3.gz")
    with open(out_db, "rb") as f_in:
        with gzip.open(gzip_filename, "wb") as f_out:
            f_out.writelines(f_in)
    print(f"Compressed database to {gzip_filename.name}.")
    print(f"compressed size: {gzip_filename.stat().st_size / 1e6:.2f} MB")

    year_size = 0
    for year_zip in sorted(root.glob("ToK_data_*.sqlite3.gz")):
        year_size += year_zip.stat().st_size
        print(f"{year_zip.name}: {year_zip.stat().st_size / 1e6:.2f} MB")
    print(f"Total size for yearly DBs: {year_size / 1e6:.2f} MB")
