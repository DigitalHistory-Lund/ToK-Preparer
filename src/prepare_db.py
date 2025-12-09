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

from src.settings import data_dir, tmp_db

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


id_to_intdate = load_id_to_date()

# ID to gender - default to None if there is no data
id_to_gender = defaultdict(lambda: None)
for row in csv.DictReader(open(data_dir / "person.csv")):
    id_to_gender[row["person_id"]] = row["gender"]


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


protocols = list(sorted(protocol_iterators(corpus_root=data_dir, start=1899, end=1941)))
print(len(protocols))


int(protocols[0].split("/")[1][:4])


def prepare_roots(protocols):
    for protocol in protocols:
        year = int(protocol.split("/")[1][:4])
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
            yield u_id, prev, nxt, text, who, year
        else:
            for child in element.getchildren():
                if (
                    child.tag.endswith("note") or child.tag.endswith("seg")
                ) and not bool(re.search(r"^\S+dag", child.text)):
                    continue
                q.put((c + 1, child, year))


def extract_all_utterances(protocols):
    q = Queue()
    for root, year in prepare_roots(protocols):
        q.put((0, root, year))
    yield from process_root_queue(q)


all_utterances = []
for utterance in tqdm(
    extract_all_utterances(protocols), total=701_218
):  # total=5273785):
    all_utterances.append(utterance)


if os.path.exists(tmp_db):
    os.unlink(tmp_db)

with sqlite3.connect(tmp_db) as conn:
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE utterance (id str primary key, prev text, next text, who text, year int, date int, gender text, party text)"
    )
    # cur.execute("PRAGMA compile_options LIKE '%SQLITE_ENABLE_FTS5%';")
    cur.execute("CREATE VIRTUAL TABLE utterance_fts USING fts5(id, content)")
    cur.execute("CREATE VIRTUAL TABLE reverse_utterance_fts USING fts5(id, content)")

    cur.execute("CREATE TABLE affiliation (who text, start int, end int, party text)")

    cur.execute("CREATE index next_index on utterance(next)")
    cur.execute("CREATE index prev_index on utterance(prev)")
    cur.execute("CREATE index who_index on utterance(who)")
    cur.execute("CREATE index year_index on utterance(year)")

    cur.executemany(
        "INSERT INTO affiliation (who, start, end, party) values (?,?,?,?)",
        load_person_dates_affiliation(),
    )
    cur.execute("CREATE index aff_index on affiliation(who)")

    data = []
    for batch in tqdm(
        batched(all_utterances, 50_000), total=len(all_utterances) // 50_000
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

    # TODO: Add speaker metadata
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


with sqlite3.connect(tmp_db) as conn:
    cur = conn.cursor()

    print(
        cur.execute(
            'select count(*) from utterance_fts where content match "kvinna AND kvinnor"'
        ).fetchall()
    )
    print(
        cur.execute(
            'select count(*) from utterance_fts where content match "kvinna"'
        ).fetchall()
    )
    print(
        cur.execute(
            'select count(*) from utterance_fts where content match "kvinnor"'
        ).fetchall()
    )
    print(
        cur.execute(
            'select count(*) from utterance_fts where content match "kvinna OR kvinnor"'
        ).fetchall()
    )
    print(
        cur.execute(
            'select count(*) from utterance_fts where content match "kvinn*"'
        ).fetchall()
    )


queries = {
    "kvinna 1": ["kvinn*"],
    "Kvinna 2": [
        "hon",
        "henne*",
        "fru",
        "dam",
        "dame*",
        "mor",
        "moder*",
        "mamma*",
        "mammor*",
        "flick*",
        "syster*",
        "systrar",
        "dotter*",
        "döttrar*",
        "hustru*",
        "änka*",
        "änke*",
        "fruntimmer*",
    ],  #
    # Work-related and possibly work-related titles
    "Kvinna 3": [
        "fröken*",
        "fröknar*",
        "jungfru*",
        "arbeterska*",
        "arbeterskor*",
        "fabriksarbeterka*",
        "fabriksarbeterskor*",
        "hushållerska*",
        "hushållerskor*",
        "lärarinna*",
        "lärarinnor*",
        "småskolelärarinna*",
        "småskolelärarinnor*",
        "mjölkerska*",
        "mjölkerskor*",
        "sjuksköterska*",
        "sjuksköterskor*",
        "sköterska*",
        "sköterskor*",
        "tjänarinna*",
        "tjänarinnor*",
        "tjänstekvinna*",
        "tjänstekvinnor*",
        "tjänsteflick*",
        "tjänstepiga*",
        "tjänstepigor*",
        "sömmerska*",
        "sömmerskor*",
        "uppasserska*",
        "uppaskerskor*",
        "kokerska*",
        "kokerskor*",
        "hon",
        "henne*",
        "mor",
        "moder*",
        "mamma*",
        "mammor*",
        "mödra*",
        "syster*",
        "systrar",
        "flick*",
        "änka*",
        "änke*",
        "fröken*",
        "fröknar*",
        "dam",
        "dame*",
        "hustru*",
        "dotter*",
        "döttrar*",
        "fruntimmer*",
        "piga*",
        "pigor*",
        "flick*",
        "hembiträde*",
        "jungfru*",
        "arbeterska*",
        "arbeterskor*",
        "fabriksarbeterka*",
        "fabriksarbeterskor*",
        "hushållerska*",
        "hushållerskor*",
        "lärarinna*",
        "lärarinnor*",
        "småskolelärarinna*",
        "småskolelärarinnor*",
        "mjölkerska*",
        "mjölkerskor*",
        "sjuksköterska*",
        "sjuksköterskor*",
        "sköterska*",
        "sköterskor*",
        "tjänarinna*",
        "tjänarinnor*",
        "tjänstekvinna*",
        "tjänstekvinnor*",
        "tjänsteflick*",
        "tjänstepiga*",
        "tjänstepigor*",
        "sömmerska*",
        "sömmerskor*",
        "uppasserska*",
        "uppaskerskor*",
        "kokerska*",
        "kokerskor*",
    ],
}
