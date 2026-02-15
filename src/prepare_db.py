from pyriksdagen.utils import protocol_iterators
from itertools import batched
from pathlib import Path
from queue import Queue
from lxml import etree
from tqdm.auto import tqdm


from collections import defaultdict

import sqlite3
import json
import gzip
import csv
import re
import os

from datetime import datetime

from .settings import data_dir, tmp_db

ghost_log = data_dir / f"{datetime.now().strftime('%Y-%m-%dT%H:%M')}.ghosts.txt"
from .queries import queries

import logging


from collections import namedtuple
from itertools import pairwise

Utterance = namedtuple(
    "utterance",
    [
        "id",
        "prev",
        "next",
        "text",
        "who",
        "year",
    ],
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="prepare_db.log",
)

parser = etree.XMLParser(remove_blank_text=True)


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


def load_id_to_date():
    # Loading utterance to (intified) dates
    year_path = Path(__file__).parents[1] / "year_data.gzip"
    if not year_path.exists():
        raise FileNotFoundError(f'Could not find "{year_path}"')
    with gzip.open(year_path, "rt") as f:
        id_to_intdate = {
            _id: datestr_to_int(date)
            for date, ids in json.loads(f.read()).items()
            for _id in ids
        }
    return id_to_intdate


# Generator for loading party affiliation with intified date ranges.
def load_person_dates_affiliation():
    for row in csv.DictReader(open(data_dir / "party_affiliation.csv")):
        if row["start"] is None or len(row["start"]) == 0:
            start = 0
        elif len(row["start"]) == 4:
            start = datestr_to_int(row["start"] + "-01-01")
        elif len(row["start"]) == 7:
            start = datestr_to_int(row["start"] + "-01")
        else:
            start = datestr_to_int(row["start"])
        if row["end"] is None or len(row["end"]) == 0:
            end = 99999999
        elif len(row["end"]) == 4:
            end = datestr_to_int(row["end"] + "-12-31")
        elif len(row["end"]) == 7:
            end = datestr_to_int(
                row["end"] + "-31"
            )  # Since we are not planning on converting these back to actual days, this works.
        else:
            end = datestr_to_int(row["end"])
        yield (row["person_id"], start, end, row["party"])


def prepare_roots(protocols):
    for protocol in protocols:
        year = int(protocol.split("/")[-2][:4])
        yield etree.parse(protocol, parser).getroot(), year


def process_root_queue(q: Queue):
    """
    TODO: Extract debate names
    """
    while not q.empty():
        c, element, year = q.get()
        if (who := element.get("who")) is not None:
            u_id = element.get(
                [key for key in element.keys() if key.endswith("}id")][0]
            )
            assert u_id
            prev = element.get("prev")
            nxt = element.get("next")
            text = "\n\n".join(
                re.sub(r"\s+", " ", seg.text) for seg in element.getchildren()
            )
            yield Utterance(u_id, prev, nxt, text, who, year)
        else:
            for child in element.getchildren():
                if (child.tag.endswith("note") or child.tag.endswith("seg")) and (
                    child.text is not None
                    and not bool(re.search(r"^\S+dag", child.text))
                ):
                    continue
                q.put((c + 1, child, year))


def extract_all_utterances():
    q = Queue()
    for root, year in prepare_roots(
        protocol_iterators(corpus_root=data_dir, start=1899, end=1941)
    ):
        q.put((0, root, year))
    yield from process_root_queue(q)


def all_utterances_generator():
    for utterance in tqdm(extract_all_utterances(), total=701_218):
        yield utterance


def raw_utterances():
    items = extract_all_utterances()
    item = next(items)
    yield Utterance(None, None, None, "First", None, None)
    yield item
    yield from items
    yield Utterance(None, None, None, "Last", None, None)


def merged_utterances():
    composite = Utterance(None, None, None, None, None, None)
    for old, new in pairwise(raw_utterances()):
        # Sifting out first line
        if old.who is None and old.text == "First":
            composite = new
            continue
        elif new.who is None and new.text == "Last":
            if old.next is not None:
                raise ValueError(f"{old.next=} is refering to a non-existing item")
            if composite.next is not None:
                raise ValueError(
                    f"{composite.next=} is refering to a non-existing item"
                )

            yield composite
            break

        # QC: Check links
        num_nones = [old.next, new.prev].count(None)
        if num_nones == 1:
            if old.next == new.id and new.prev is None:
                new = Utterance(
                    id=new.id,
                    prev=old.id,
                    next=new.next,
                    text=new.text,
                    who=new.who,
                    year=new.year,
                )
            elif old.id == new.prev and old.next is None:
                old = Utterance(
                    id=old.id,
                    prev=old.prev,
                    next=new.id,
                    text=old.text,
                    who=old.who,
                    year=old.year,
                )
            else:
                # raise ValueError(
                Warning(
                    f"Broken link ({old.next=}, {new.id=} ;; {old.id=}, {new.prev=})\nbetween {old=} ; {new=}"
                )
                ghost = ""
                if old.next is not None:
                    ghost = f"{old.year} ; {new.id=} ; {old.next}\n"
                elif new.prev is not None:
                    ghost = f"{new.year} ; {old.id=} ; {new.next}\n"

                with open(ghost_log, "a", encoding="utf8") as f:
                    f.write(ghost)

        elif num_nones == 0:
            ghost = ""
            if old.next == new.prev:
                ghost = f"{old.year} ; skipped ; {old.next}\n"
            elif not old.next == new.id:
                # raise ValueError(
                Warning(
                    f"{old.next=} != {new.id=}",
                )
                ghost = f"{old.year} ; {new.id=} ; {old.next}\n"
            elif not old.id == new.prev:
                # raise ValueError(
                Warning(
                    f"{old.id=} != {new.prev=}",
                )
                ghost = f"{new.year} ; {old.id=} ; {new.next}\n"

            with open(ghost_log, "a") as f:
                f.write(ghost)

        # We do not merge the 'unknowns'
        if old.who == "unknown":
            yield composite
            composite = Utterance(
                id=new.id,
                prev=composite.id,
                next=new.next,
                text=new.text,
                who=new.who,
                year=new.year,
            )
            continue

        # Merging composite with the new data
        elif all(
            (
                old.year == new.year,
                old.who == new.who,
                old.next == new.id,
                old.id == new.prev,
            )
        ):
            composite = Utterance(
                id=composite.id,
                prev=composite.prev,
                next=new.next,
                text=composite.text + " " + new.text,
                who=composite.who,
                year=composite.year,
            )
        # Yielding the composite to create a composite from the new
        else:
            yield composite
            composite = Utterance(
                id=new.id,
                prev=composite.id,
                next=new.next,
                text=new.text,
                who=new.who,
                year=new.year,
            )


def create_database():
    with sqlite3.connect(tmp_db) as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE utterance (
                id str primary key,
                prev text,
                next text,
                who text,
                year int,
                date int,
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

        cur.executemany(
            "INSERT INTO affiliation (who, start, end, party) values (?,?,?,?)",
            load_person_dates_affiliation(),
        )
        cur.execute("CREATE index aff_index on affiliation(who)")


def seed_database():
    id_to_intdate = load_id_to_date()

    # ID to gender - default to None if there is no data
    id_to_gender = defaultdict(lambda: None)
    for row in csv.DictReader(open(data_dir / "person.csv")):
        id_to_gender[row["person_id"]] = row["gender"]

    with sqlite3.connect(tmp_db) as conn:
        cur = conn.cursor()
        data = []
        for batch in tqdm(
            batched(merged_utterances(), 50_000),
            total=701_218 // 50_000,
        ):
            data = [
                {
                    "id": u_id,
                    "prev": prev,
                    "next": nxt,
                    "content": text,
                    "reverse_content": text[::-1],
                    "who": who,
                    "year": year,
                    "gender": id_to_gender[who],
                    "date": id_to_intdate[u_id],
                }
                for u_id, prev, nxt, text, who, year in batch
            ]

            cur.executemany(
                "INSERT INTO utterance_fts (id, content) values (:id, :content)", data
            )
            cur.executemany(
                "INSERT INTO reverse_utterance_fts (id, content) values (:id, :reverse_content)",
                data,
            )
            cur.executemany(
                "INSERT INTO utterance (id, prev, next, who, year, gender, date) values (:id, :prev, :next, :who, :year, :gender, :date)",
                data,
            )

            conn.commit()

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


def tag_utterances_by_query():
    with sqlite3.connect(tmp_db) as conn:
        cur = conn.cursor()
        for label, query_terms in queries.items():
            query_str = " OR ".join(query_terms)
            logging.info(f'Tagging utterances for "{label}" with query "{query_str}"')
            cur.execute(
                f"""
                UPDATE utterance
                SET {label.replace(" ", "_").lower()} = 1
                WHERE id IN (
                    SELECT id
                    FROM utterance_fts
                    WHERE content MATCH ?
                )
                """,
                (query_str,),
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
    if not tmp_db.exists():
        create_database()
    seed_database()
    count_baselines()
    tag_utterances_by_query()

if __name__ == "__main__":
    prepare_database()
