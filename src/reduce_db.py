from settings import tmp_db, out_db
import sqlite3
from itertools import batched
from tqdm import tqdm
import gzip

if out_db.exists():
    raise FileExistsError(f"{out_db} already exists. Remove it before proceeding.")
elif not tmp_db.exists():
    raise FileNotFoundError(f"{tmp_db} does not exist. Run prepare_db.py first.")


if __name__ == "__main__":
    with sqlite3.connect(tmp_db) as source_conn:
        source_cur = source_conn.cursor()

        with sqlite3.connect(out_db) as target_conn:
            target_cur = target_conn.cursor()

            print(f"Creating normalized schema in {out_db}...")

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
                    prev TEXT,
                    next TEXT,
                    person_id INTEGER,
                    year INTEGER,
                    date INTEGER,
                    kvinna_1 BOOLEAN,
                    kvinna_2 BOOLEAN,
                    kvinna_3 BOOLEAN,
                    FOREIGN KEY (person_id) REFERENCES person(id)
                )
            """)

            print("Extracting unique persons...")

            # Extract unique persons from source
            source_cur.execute("""
                SELECT DISTINCT who, gender, party
                FROM utterance
                ORDER BY who
            """)

            persons = source_cur.fetchall()
            person_map = {}  # (who, gender, party) -> person_id

            print(f"Found {len(persons)} unique persons. Inserting...")

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
            batch_size = 10000
            offset = 0
            total_count = source_cur.execute(
                "SELECT COUNT(*) FROM utterance"
            ).fetchone()[0]

            for batch in tqdm(
                batched(
                    source_cur.execute("""
                    SELECT id, prev, next, who, year, date, gender, party,
                        kvinna_1, kvinna_2, kvinna_3
                    FROM utterance
                """),
                    batch_size,
                ),
                total=total_count // batch_size,
                desc="Migrating utterances",
            ):
                # Transform and insert
                transformed = []
                for row in batch:
                    id_, prev, next_, who, year, date, gender, party, k1, k2, k3 = row
                    person_id = person_map[(who, gender, party)]
                    transformed.append(
                        (id_, prev, next_, person_id, year, date, k1, k2, k3)
                    )

                target_cur.executemany(
                    """
                    INSERT INTO utterance
                    (id, prev, next, person_id, year, date, kvinna_1, kvinna_2, kvinna_3)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    transformed,
                )

                target_conn.commit()

            target_conn.commit()

    print(f"Reduced database {tmp_db.name} to {out_db.name}.")
    print(f"size before: {tmp_db.stat().st_size / 1e6:.2f} MB")
    print(f"size after:  {out_db.stat().st_size / 1e6:.2f} MB")

    gzip_filename = out_db.with_suffix(".sqlite3.gz")
    with open(out_db, "rb") as f_in:
        with gzip.open(gzip_filename, "wb") as f_out:
            f_out.writelines(f_in)
    print(f"Compressed database to {gzip_filename.name}.")
    print(f"compressed size: {gzip_filename.stat().st_size / 1e6:.2f} MB")
