from .settings import tmp_db, out_db, root
from .queries import queries
import sqlite3
import re
from itertools import batched
from tqdm import tqdm
import gzip

SUPERSCRIPTS = {"kvinna 1": "\u00b9", "Kvinna 2": "\u00b2", "Kvinna 3": "\u00b3"}
_WORD_RE = re.compile(r"\w+")


def _matches_pattern(word_lower, pattern):
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

if out_db.exists():
    raise FileExistsError(f"{out_db} already exists. Remove it before proceeding.")
elif not tmp_db.exists():
    raise FileNotFoundError(f"{tmp_db} does not exist. Run prepare_db.py first.")


if __name__ == "__main__":
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

            for year in tqdm(range(1900, 1941), desc="Creating yearly DBs", total=41):
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
