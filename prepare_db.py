from pyparlaclarin.read import paragraph_iterator, speeches_with_name
from pyriksdagen.utils import protocol_iterators, download_corpus

from pathlib import Path
from queue import Queue
from lxml import etree
from tqdm.auto import tqdm

import progressbar
import pyriksdagen
import ipywidgets

import plotly.offline as pyo
import plotly.graph_objs as go
from typing import List, Tuple

from collections import defaultdict
from collections import Counter

import sqlite3
import json
import gzip
import csv
import re
import os
parser = etree.XMLParser(remove_blank_text=True)

data_dir = Path('.').resolve() / 'data'
data_dir.mkdir(exist_ok=True)


if len([file for file in data_dir.iterdir() if file.is_file()]) < 25:
    print('Did not find metadata files. Downloading.')
    download_corpus(partitions=["persons"])

records_data_dir = data_dir / 'records'
if len([subdir for subdir in data_dir.iterdir() if subdir.is_dir()]) < 158:
    print('Did not find protocols files. Downloading.')
    download_corpus(partitions=["records"])


def datestr_to_int(date_str):
    """Convert YYYY-MM-DD to an int with all digits"""
    year, mon, day = date_str.split('-')
    if not all((len(date_str) == 10, len(year) == 4, len(mon) == 2, len(day) == 2, year.isdigit(), mon.isdigit(), day.isdigit())):
        raise ValueError(f'{date_str=} does not adhere to "YYYY-MM-DD"')

    no_dashes = date_str.replace('-', '')
    if len(no_dashes) != 8:
        raise ValueError
    return int(no_dashes)


# Loading utterance to (intified) dates
year_path = Path('.') / 'year_data.gzip'
if not year_path.exists():
    !wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1n_RzrYH-ghTF0dIV6tY67r_mVZb2ktyd' -O year_data.gzip
with gzip.open(year_path, 'rt') as f:
    id_to_intdate = {_id: datestr_to_int(date) for date, ids in json.loads(f.read()).items() for _id in ids}


# ID to gender - default to None if there is no data
id_to_gender = defaultdict(lambda: None)
for row in csv.DictReader(open(data_dir/'person.csv')):
    id_to_gender[row['person_id']]= row['gender']


# Generator for loading party affiliation with intified date ranges.
def load_person_dates_affiliation():
    for row in csv.DictReader(open(data_dir / 'party_affiliation.csv')):
        if row['start'] is None or len(row['start']) == 0:
            start = 0
        elif len(row['start']) == 4:
            start = datestr_to_int(row['start']+'-01-01')
        elif len(row['start']) == 7:
            start = datestr_to_int(row['start']+'-01')
        else:
            start = datestr_to_int(row['start'])
        if row['end'] is None or len(row['end']) == 0:
            end = 99999999
        elif len(row['end']) == 4:
            end = datestr_to_int(row['end'] + '-12-31')
        elif len(row['end']) == 7:
            end = datestr_to_int(row['end'] + '-31') # Since we are not planning on converting these back to actual days, this works.
        else:
            end = datestr_to_int(row['end'])
        yield (row['person_id'], start, end, row['party'])


protocols = list(sorted(protocol_iterators(corpus_root="data/", start=1899, end=1941)))
print(len(protocols))


int(protocols[0].split('/')[1][:4])


def prepare_roots(protocols):
    for protocol in protocols:
        year = int(protocol.split('/')[1][:4])
        yield etree.parse(protocol, parser).getroot(), year


def process_root_queue(q: Queue):
    """
    TODO: Extract debate names
    TODO: Extract dates
    """
    while not q.empty():
        c, element, year = q.get()
        if (who:= element.get('who')) is not None:
            u_id = element.get([key for key in element.keys() if key.endswith('}id')][0])
            assert u_id
            prev = element.get('prev')
            nxt = element.get('next')

            text = '\n\n'.join(re.sub(r'\s+' ,' ', seg.text) for seg in element.getchildren())
            yield u_id, prev, nxt, text, who, year
        else:
            for child in element.getchildren():
                if (child.tag.endswith('note') or child.tag.endswith('seg')) and not bool(re.search(r'^\S+dag', child.text)):
                    continue
                q.put((c+1, child, year))


def extract_all_utterances(protocols):
    q = Queue()
    for root, year in prepare_roots(protocols):
        q.put((0,root, year))
    yield from process_root_queue(q)


all_utterances = []
for utterance in tqdm(extract_all_utterances(protocols), total=701_218): #total=5273785):
    all_utterances.append(utterance)


from itertools import islice
def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 2) → AB CD EF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch


tmp_db = './tmp.db'
if os.path.exists(tmp_db):
    os.unlink(tmp_db)

with sqlite3.connect(tmp_db) as conn:
    cur = conn.cursor()
    cur.execute('CREATE TABLE utterance (id str primary key, prev text, next text, who text, year int, date int, gender text, party text)')
    # cur.execute("PRAGMA compile_options LIKE '%SQLITE_ENABLE_FTS5%';")
    cur.execute('CREATE VIRTUAL TABLE utterance_fts USING fts5(id, content)')
    cur.execute('CREATE VIRTUAL TABLE reverse_utterance_fts USING fts5(id, content)')

    cur.execute('CREATE TABLE affiliation (who text, start int, end int, party text)')

    cur.execute('CREATE index next_index on utterance(next)')
    cur.execute('CREATE index prev_index on utterance(prev)')
    cur.execute('CREATE index who_index on utterance(who)')
    cur.execute('CREATE index year_index on utterance(year)')

    cur.executemany('INSERT INTO affiliation (who, start, end, party) values (?,?,?,?)', load_person_dates_affiliation())
    cur.execute('CREATE index aff_index on affiliation(who)')


    data = []
    for batch in tqdm(batched(all_utterances, 50_000), total=len(all_utterances)//50_000):
        data = [
            {'id':u_id,
             'prev':prev,
             'next':nxt,
             'content':text,
             'reverse_content':text[::-1],
             'who':who,
             'year':year,
             'gender': id_to_gender[who],
             'date' : id_to_intdate[u_id]
            } for u_id, prev, nxt, text, who, year in batch]

        cur.executemany('INSERT INTO utterance_fts (id, content) values (:id, :content)', data)
        cur.executemany('INSERT INTO reverse_utterance_fts (id, content) values (:id, :reverse_content)', data)
        cur.executemany('INSERT INTO utterance (id, prev, next, who, year, gender, date) values (:id, :prev, :next, :who, :year, :gender, :date)', data)

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



term_counter = Counter()

with sqlite3.connect(tmp_db) as conn:
    cur = conn.cursor()


    print(cur.execute('select count(*) from utterance_fts where content match "kvinna AND kvinnor"').fetchall())
    print(cur.execute('select count(*) from utterance_fts where content match "kvinna"').fetchall())
    print(cur.execute('select count(*) from utterance_fts where content match "kvinnor"').fetchall())
    print(cur.execute('select count(*) from utterance_fts where content match "kvinna OR kvinnor"').fetchall())
    print(cur.execute('select count(*) from utterance_fts where content match "kvinn*"').fetchall())


def get_found_words(pattern):
    out_file = './ord/' + pattern +'.csv'
    out_file = Path(out_file.replace(' ', '_'))
    out_file.parent.mkdir(exist_ok = True, parents=True)


    if not out_file.exists():

        with sqlite3.connect(tmp_db) as conn:
            cur = conn.cursor()

            if not pattern.startswith('*'):
                term_counter = Counter((re.sub('[^a-zåäö ]', '', token.lower()) for content in cur.execute(f'select content from utterance_fts where content match "{pattern}"')
                    for token in content[0].split() if token.lower().startswith(pattern.replace('*', ''))))
            else:
                reverse_pattern = pattern[::-1]
                term_counter = Counter((re.sub('[^a-zåäö ]', '', token.lower()[::-1]) for content in cur.execute(f'select content from reverse_utterance_fts where content match "{reverse_pattern}"')
                    for token in content[0].split() if token.lower().startswith(reverse_pattern.replace('*', ''))))

        with open(out_file, 'x', encoding='utf8') as f:
            f.write('word , count\n')
            for term, count in term_counter.most_common():
                f.write(str(term) + ' , ' + str(count) + '\n')

get_found_words('fru*')
get_found_words('*änka')
get_found_words('*kvinna')
get_found_words('*fru')
get_found_words('*dotter')
get_found_words('*arbeterska')

###
get_found_words("adelsjungfru")
get_found_words("adelskvinna")
get_found_words("ägaränka")
get_found_words("ägardotter")
get_found_words("ägareänka")
get_found_words("ägaredotter")
get_found_words("ägarefru")
get_found_words("ägarehustru")
get_found_words("ägaremor")
get_found_words("ägarhustru")
get_found_words("ägarinna")
get_found_words("ägarinnamor")
get_found_words("aktieägarinna")
get_found_words("åldermanänkefru")
get_found_words("åldermandotter")
get_found_words("åldermansänka")
get_found_words("amerikaarbetarehustru")
get_found_words("amerikahustru")
get_found_words("änkebaronessa")
get_found_words("änkefriherrinna")
get_found_words("änkegrevinna")
get_found_words("änkehjälptagerska")
get_found_words("änkemadam")
get_found_words("anknyterska")
get_found_words("arkitektänka")
get_found_words("arkitektdotter")
get_found_words("arkitekthustru")
get_found_words("arkitektsänka")
get_found_words("assistentänka")
get_found_words("assistenthustru")
get_found_words("assistentsänka")
get_found_words("avbyterska")
get_found_words("baptisthustru")
get_found_words("barnhemsflicka")
get_found_words("barnhusflickan")
get_found_words("baronänka")
get_found_words("baronänkefru")
get_found_words("baronessa")
get_found_words("baronetenka")
get_found_words("baronfru")
get_found_words("bettlerska")
get_found_words("bibelkvinnohemelev")
get_found_words("biträdesänka")
get_found_words("biträdesdotter")
get_found_words("biträdeshustru")
get_found_words("boendeänka")
get_found_words("boningshusägarinna")
get_found_words("borgaränka")
get_found_words("borgardotter")
get_found_words("borgareänka")
get_found_words("borgaredotter")
get_found_words("borgareenka")
get_found_words("borgarefru")
get_found_words("borgarehustru")
get_found_words("borgarenka")
get_found_words("borgensänka")
get_found_words("bostadshusegarinna")
get_found_words("bruksingenjörsänka")
get_found_words("bruksingenjörshustru")
get_found_words("byråingenjörsänka")
get_found_words("concubin")
get_found_words("dalflicka")
get_found_words("dalgumma")
get_found_words("dalkarlsänka")
get_found_words("dalkulla")
get_found_words("dannemanänka")
get_found_words("dannemandotter")
get_found_words("delägarinna änka")
get_found_words("delegarinna i herrhamra")
get_found_words("demois")
get_found_words("demoiselle")
get_found_words("diversehemarbeterska")
get_found_words("doktoränka")
get_found_words("doktorhustru")
get_found_words("doktorsänka")
get_found_words("egaredotter")
get_found_words("egareenka")
get_found_words("egarinna")
get_found_words("egarinna af enemossen")
get_found_words("egethemsägarinna")
get_found_words("egnahemägarinna")
get_found_words("emigranthustru")
get_found_words("enkefriherinna")
get_found_words("enkefriherrinna")
get_found_words("enkefru")
# get_found_words("enkefru, fattighjon")
# get_found_words("enkefru, undantag")
get_found_words("enkegrefvinna")
get_found_words("familjevärdinna")
get_found_words("fästekvinna")
get_found_words("fastighetsägaränka")
get_found_words("fastighetsägareänka")
get_found_words("fastighetsägarehustru")
get_found_words("fastighetsägarhustru")
get_found_words("fastighetsägarinna")
get_found_words("fastighetsägerska")
get_found_words("fastighetsäginna")
get_found_words("fastighetsegarinna")
get_found_words("fastighetsegerska")
get_found_words("fästningsänka")
get_found_words("fästningsfånghustru")
get_found_words("fattigänka")
get_found_words("fattigdelsinnehavarinna")
get_found_words("fattigdelstagerska")
get_found_words("fattigenka")
get_found_words("fattigflicka")
get_found_words("fattighjonänka")
get_found_words("fattighjondotter")
get_found_words("fattighjonsänka")
get_found_words("fattighjonsdotter")
get_found_words("fattighjonsenka")
get_found_words("fattighjonshustru")
get_found_words("fattighusänka")
get_found_words("fattigkvinna")
get_found_words("fattigstugegumma")
get_found_words("fattigunderstödstagareänka")
get_found_words("fattigunderstödstagerska")
get_found_words("fattigvårdsunderstödstagerska")
get_found_words("fideikommissarieänka")
get_found_words("fjällappänka")
get_found_words("fjällappdotter")
get_found_words("fjällappflicka")
get_found_words("fjällappsänka")
get_found_words("fjällhustru")
get_found_words("fjällkvinna")
get_found_words("fjällmanänka")
get_found_words("fjällmandotter")
get_found_words("fjällmanflicka")
get_found_words("fjällmanhustru")
get_found_words("fjelllappenka")
get_found_words("flickskoleelev")
get_found_words("föderådsenka")
get_found_words("födomålstagerska")
get_found_words("födorådsänka")
get_found_words("födorådsenka")
get_found_words("födorådskvinna")
get_found_words("födorådskvinnan")
get_found_words("födorådsmanhustru")
get_found_words("födorådsqvinna")
get_found_words("födorådstagaränka")
get_found_words("födorådstagareänka")
get_found_words("födorådstagareenka")
get_found_words("födorådstagarehustru")
get_found_words("födorådstagarenka")
get_found_words("födorådstagarinna")
get_found_words("födorådstagerka")
get_found_words("födoradstagerska")
get_found_words("födorådstagerska")
get_found_words("födorådstagsänka")
get_found_words("födorådstagsenka")
get_found_words("födorådstagska")
get_found_words("födordstagerska")
get_found_words("födotagerska")
get_found_words("födovårdstagerska")
get_found_words("födtagerska")
get_found_words("förådstagerska")
get_found_words("fördelshafverska")
get_found_words("fördelshaverska")
get_found_words("fördelstagarinna")
get_found_words("fördorådstagerska")
get_found_words("föredelstagerska")
get_found_words("förgångtagerska")
get_found_words("förmånstastagerska")
get_found_words("förrådsstagerska")
get_found_words("förrådstagarinna")
get_found_words("förrådstagenka")
get_found_words("förrådstagerska")
get_found_words("förrådtagerska")
get_found_words("försörjningstagerska")
get_found_words("fosterbarnsvårdarinna")
get_found_words("fosterdotter")
get_found_words("frälsemandotter")
get_found_words("frigrevinna")
get_found_words("friherreänka")
get_found_words("friherrehustru")
get_found_words("friherrinna")
get_found_words("frimurarmansänka")
get_found_words("fru grefvinna")
get_found_words("furstinna")
get_found_words("gårdsmamsell")
get_found_words("gesällhustru")
get_found_words("grevinna")
get_found_words("hemarbeterska")
get_found_words("hemmadotter")
get_found_words("hemsysslerska")
get_found_words("herrinna")
get_found_words("hertiginna")
get_found_words("hovarkitektänka")
get_found_words("hovarkitektdotter")
get_found_words("husägareänka")
get_found_words("husägarinna")
get_found_words("husägerska")
get_found_words("husbondeänka")
get_found_words("husbondehustru")
get_found_words("husegarehustru")
get_found_words("husegarinna")
get_found_words("huskvinna")
get_found_words("husqvinna")
get_found_words("idkerska")
get_found_words("ingeniörsenka")
get_found_words("ingeniörsfru")
get_found_words("ingenjörsarbetarehustru")
get_found_words("ingenjörsdotter")
get_found_words("ingenjörshustru")
get_found_words("ingenjörska")
get_found_words("inhyserska")
get_found_words("inhysesänka")
get_found_words("inhysesenka")
get_found_words("inhyseshustru")
get_found_words("inhysesmor")
get_found_words("inhysestagerska")
get_found_words("inkomsttagerska")
get_found_words("innehafvarinna")
get_found_words("innehafverska")
get_found_words("innehavarinna")
get_found_words("instruktörsingenjörsänka")
get_found_words("kandidatänka")
get_found_words("kandidathustru")
get_found_words("kandidatsänka")
get_found_words("kemibiträdesdotter")
get_found_words("kemistänka")
get_found_words("kennelinnehavarinna")
get_found_words("konkubin")
get_found_words("kringdriverska")
get_found_words("kringstrykande qvinfolk")
get_found_words("kringstrykerska")
get_found_words("kringvandrerska")
get_found_words("kronokärring")
get_found_words("kronprinsess*")
get_found_words("kulla")
get_found_words("kvinnsperson")
get_found_words("laborationspiga")
get_found_words("laboratoriearbeterska")
get_found_words("laboratoriebiträdesänka")
get_found_words("laboratorieföreståndarinna")
get_found_words("laboratoripiga")
get_found_words("laboratris")
get_found_words("landstrykerska")
get_found_words("landunderstödstagerska")
get_found_words("lappenka")
get_found_words("levebrödstagerska")
get_found_words("livbrödstagerska")
get_found_words("livräntetagerska")
get_found_words("lordänka")
get_found_words("lösdrifverska")
get_found_words("lösdrifverskan")
get_found_words("lösdriverska kringstrykande")
get_found_words("lufferska")
get_found_words("madam")
get_found_words("madame")
get_found_words("mademoiselle")
get_found_words("magisterhustru")
get_found_words("maka")
get_found_words("mamsell")
get_found_words("markisinna")
get_found_words("matrona")
get_found_words("medaljös")
get_found_words("medhjälparinna")
get_found_words("meteorologföreståndarinna")
get_found_words("nomadänka")
get_found_words("nykterhetsidkerska")
get_found_words("nykterhetsrörelseidkerska")
get_found_words("överceremonimästarinna")
get_found_words("överingenjörsänka")
get_found_words("överkandidatänka")
get_found_words("pensionsinnehafvarinna")
get_found_words("pensionsinnehavarinna")
get_found_words("pensionslärarinna")
get_found_words("pensionstagarinna")
get_found_words("pensionstagerska")
get_found_words("pensiontagerska")
get_found_words("prinsess*")
get_found_words("privatunderstödstagerska")
# get_found_words("ränte-tagerska")
get_found_words("räntetagarinna")
get_found_words("ränteupptagerska")
get_found_words("realstudenska")
get_found_words("rentiére")
get_found_words("seminarierska")
get_found_words("seminarinna")
get_found_words("självförsörjerska")
get_found_words("sjelfförsörjerska")
get_found_words("sjukhuslaboratris")
get_found_words("sköka")
get_found_words("skolemor")
get_found_words("skrikegumma")
get_found_words("sockengumma")
get_found_words("sockenmanhustru")
get_found_words("stadsfröken")
get_found_words("stadsgeologsänka")
get_found_words("stadsingenjörhustru")
get_found_words("stadsingenjörsänka")
get_found_words("sterbhus delegarinna")
get_found_words("sterbhusandelägerska")
get_found_words("sterbhusdelägarinna")
get_found_words("sterbhusdelägerska")
get_found_words("sterbhusdeläginna")
get_found_words("sterbhusdelegarinna")
get_found_words("sterbhusföreståndarinna")
get_found_words("stiftfröken")
get_found_words("stiftsfröken")
get_found_words("studentska")
get_found_words("studerska")
get_found_words("stugägarinna")
get_found_words("stugägerska")
get_found_words("stuguägarinna")
get_found_words("stuguegarinna")
get_found_words("stuguegerska")
get_found_words("tatterska")
get_found_words("tiggerska")
get_found_words("tingshusägarina")
get_found_words("tomtägarinna")
get_found_words("tomtägerska")
get_found_words("tomtegarinna")
get_found_words("trafikchefsingenjörsänka")
get_found_words("trasfröken")
# get_found_words("und.tagsenka")
get_found_words("undantagaredotter")
get_found_words("undantagarinna")
get_found_words("undantagerska")
get_found_words("undantaghustru")
get_found_words("undantagsägarhustru")
get_found_words("undantagsägerska")
get_found_words("undantagsbesittarinna")
get_found_words("undantagsdotter")
get_found_words("undantagsflicka")
get_found_words("undantagsfru")
get_found_words("undantagsgumma")
get_found_words("undantagshaverska")
get_found_words("undantagshjon")
get_found_words("undantagshjondotter")
get_found_words("undantagshjonsänka")
get_found_words("undantagshjonshustru")
get_found_words("undantagshustru")
get_found_words("undantagshustrun")
get_found_words("undantagsinnehafverska")
get_found_words("undantagsinnehavarinna")
get_found_words("undantagsjungfru")
get_found_words("undantagskvinna")
get_found_words("undantagsmaka")
get_found_words("undantagsmanänka")
get_found_words("undantagsmansdotter")
get_found_words("undantagsmansenka")
get_found_words("undantagsmanshustru")
get_found_words("undantagsmor")
get_found_words("undantagstagaerska")
get_found_words("undantagstagareänka")
get_found_words("undantagstagarinna")
get_found_words("undantagstagerska")
get_found_words("undantagstagsänka")
get_found_words("underhållstagareänka")
get_found_words("underhållstagerska")
get_found_words("underingenjörsänka")
get_found_words("understöderska")
get_found_words("understödstagareänka")
get_found_words("understödstagaredotter")
get_found_words("understödstagarehustru")
get_found_words("understödstagarinna")
get_found_words("understödstagerska")
get_found_words("understödstagerska hustru frånskild")
get_found_words("undtagerska")
get_found_words("ungersvenhustru")
get_found_words("upperska")
get_found_words("vandraredotter")
get_found_words("vandringskvinna")
get_found_words("vårdtagarinna")
get_found_words("värsyster")
get_found_words("villaägarinna")



get_found_words("barnavårdarinna")
get_found_words("barnmorska")
get_found_words("diakonissa")
get_found_words("epidimisjuksköterska")
get_found_words("fattigdiakonissa")
get_found_words("förestånderska")
get_found_words("förestånderska")
get_found_words("husförestånderska")
get_found_words("hushållerska")
get_found_words("husvärdinna")
get_found_words("sjuksköterska")
get_found_words("sköterska")
get_found_words("städerska")
get_found_words("tjänarinna")
get_found_words("vaccinatris")


def get_timeline(match_pattern: str) -> List[Tuple[int, int]]:
    with sqlite3.connect(tmp_db) as conn:
        cur = conn.cursor()

        return cur.execute(f'select year, count(*) from utterance as u join utterance_fts as uf on u.rowid == uf.rowid where content match "{match_pattern}" group by year').fetchall()

def get_baseline():
    with sqlite3.connect(tmp_db) as conn:
        cur = conn.cursor()

        return {x:y for x,y in cur.execute(f'select year, count(*) from utterance as u join utterance_fts as uf on u.rowid == uf.rowid group by year').fetchall()}
baseline = get_baseline()


def get_or_timeline(keywords : List[str]) -> List[Tuple[int, int]]:
    pattern = ' OR '.join(keywords)
    return get_timeline(pattern)


kvinn = get_timeline('kvinn*')
kvinna = get_timeline('kvinna')
korv = get_timeline('korv')




queries = {
    "kvinna" : ['kvinna'],
    "kvinna 1" : ['kvinn*'],
    # 'The name that shows up in the plot' : ['term1', 'term_2', 'ansikte', 'fler', 'etc'],
    #'mans' : ['mans'],
    #'Första sökning*' : ['hon', 'henne*', 'fru*', 'fröken*', 'fröknar*', 'dam', 'dame*', 'mor', 'moder*', 'mamma*', 'mammor*','flick*', 'syster*', 'systrar', 'dotter*', 'döttrar*', 'hustru*','änka*', 'änke*', 'fruntimmer*', 'jungfru*'],

    # Non work-related titles
    'Kvinna 2' : ['hon', 'henne*', 'fru',  'dam', 'dame*', 'mor', 'moder*', 'mamma*', 'mammor*','flick*', 'syster*', 'systrar', 'dotter*', 'döttrar*', 'hustru*','änka*', 'änke*', 'fruntimmer*', ], #

    # Work-related and possibly work-related titles
    'Kvinna 3' : [
                'fröken*', 'fröknar*', 'jungfru*',
                'arbeterska*', 'arbeterskor*','fabriksarbeterka*', 'fabriksarbeterskor*','hushållerska*', 'hushållerskor*','lärarinna*', 'lärarinnor*', 'småskolelärarinna*', 'småskolelärarinnor*', 'mjölkerska*', 'mjölkerskor*',
                 'sjuksköterska*', 'sjuksköterskor*','sköterska*', 'sköterskor*', 'tjänarinna*', 'tjänarinnor*', 'tjänstekvinna*', 'tjänstekvinnor*', 'tjänsteflick*', 'tjänstepiga*', 'tjänstepigor*', 'sömmerska*',
                 'sömmerskor*', 'uppasserska*', 'uppaskerskor*', 'kokerska*', 'kokerskor*' ,
                'hon', 'henne*',
                'mor', 'moder*', 'mamma*', 'mammor*', 'mödra*',
                'syster*', 'systrar',
                'flick*',
                'änka*', 'änke*',
                'fröken*', 'fröknar*',
                'dam', 'dame*',
                'hustru*',
                'dotter*', 'döttrar*',
                'fruntimmer*',
                'piga*', 'pigor*',
                'flick*',
                'hembiträde*',
                'jungfru*',
                'arbeterska*', 'arbeterskor*',
                'fabriksarbeterka*', 'fabriksarbeterskor*',
                'hushållerska*', 'hushållerskor*',
                'lärarinna*', 'lärarinnor*', 'småskolelärarinna*', 'småskolelärarinnor*',
                'mjölkerska*', 'mjölkerskor*',
                'sjuksköterska*', 'sjuksköterskor*', 'sköterska*', 'sköterskor*',
                'tjänarinna*', 'tjänarinnor*', 'tjänstekvinna*', 'tjänstekvinnor*', 'tjänsteflick*', 'tjänstepiga*', 'tjänstepigor*',
                'sömmerska*', 'sömmerskor*',
                'uppasserska*', 'uppaskerskor*',
                'kokerska*', 'kokerskor*'],
    #  'Pronomen' : ['hon', 'henne*'],
    #  'Moder' : ['mor', 'moder*', 'mamma*', 'mammor*', 'mödra*'],
    #  'Syster' : ['syster*', 'systrar',],
    #  'Flicka' : ['flick*'],
    #  'Änka' : ['änka*', 'änke*'],
    #  'Fröken' : ['fröken*', 'fröknar*'],
    #  'Dam' : ['dam', 'dame*'],
    #  'Hustru' : ['hustru*'],
    #  'Dotter' : ['dotter*', 'döttrar*'],
    #  'Fruntimmer*' : ['fruntimmer*'],
    #  'piga' : ['piga*', 'pigor*'],
    #  'Flicka' : ['flick*'],
    #  'Hembiträde' : ['hembiträde*'],
    #  'Jungfru' : ['jungfru*'],
    #  'Arbeterska' : ['arbeterska*', 'arbeterskor*'],
    #  'Fabriksarbeterska' : ['fabriksarbeterka*', 'fabriksarbeterskor*'],
    #  'Hushållerska' : ['hushållerska*', 'hushållerskor*'],
    #  'Lärarinna' : ['lärarinna*', 'lärarinnor*', 'småskolelärarinna*', 'småskolelärarinnor*'],
    #  'Mjökerska' : ['mjölkerska*', 'mjölkerskor*'],
    #  'Sjuksköterska' : ['sjuksköterska*', 'sjuksköterskor*', 'sköterska*', 'sköterskor*'],
    #  'Tjänarinna' : ['tjänarinna*', 'tjänarinnor*', 'tjänstekvinna*', 'tjänstekvinnor*', 'tjänsteflick*', 'tjänstepiga*', 'tjänstepigor*'],
    #  'Sömmerska' : ['sömmerska*', 'sömmerskor*'],
    #  'Uppaskerska' : ['uppasserska*', 'uppaskerskor*'],
    #  'Kokerska' : ['kokerska*', 'kokerskor*'],
}
