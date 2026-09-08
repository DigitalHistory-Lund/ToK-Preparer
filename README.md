[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18211787.svg)](https://doi.org/10.5281/zenodo.18211787)

# Pre-processing data for ToK utterance browser

Building on the openly published
[The Swedish Parliament corpus](https://github.com/swerik-project/the-swedish-parliament-corpus)
the following repository pre-processes and hosts an annotated version for the
research project 'Tal om Kvinnor', led by Ulrika Holgersson at Lund University.

The pre-processing includes merging, dating and tagging utterances as well as
hosting a splitered corpus (split by year) to enable a browser-based corpus
reader that relies on the annotations created here.

Most of the data used for enrichment are part of the original corpus -- we
change the format to fit our needs. Our concrete addition is to tag the
utterances with different groups of words that indicate that the speaker was
talking about women.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/getting-started/installation/).
From the `tok_preparer/` directory:

```bash
uv sync
```

The pipeline downloads ~1.5 GB of source corpus files into `data/` on first
run, so make sure there is disk space available.

## Running the pipeline

Run the four scripts in order. Each depends on the outputs of the previous:

1. `uv run python -m src.download` — fetches `persons.sqlite` and
   `records_speeches.ndjson.gz` from the swerik-project releases into `data/`.
2. `uv run python -m src.prepare_db` — builds `tmp_db.sqlite3` (utterances,
   FTS indexes, prev/next links, Kvinna category tags). Slowest step.
3. `uv run python -m src.word_frequencies` — reads `tmp_db.sqlite3` and writes
   `word_frequencies.csv.gz` + `speakers.csv.gz`, plus `.1` / `.2` chamber
   variants.
4. `uv run python -m src.reduce_db` — normalises `tmp_db.sqlite3` into
   `ToK_data.sqlite3`, gzips it to `ToK_data.sqlite3.gz`, and emits per-year
   `ToK_data_YYYY.sqlite3.gz`. The per-year files are committed to the repo
   so they ship with the Zenodo deposit (auto-archived from the source
   tarball) and can be served to the ToK-Reader via
   `raw.githubusercontent.com`. GitHub's 100 MB single-file limit rules out
   committing the aggregate `ToK_data.sqlite3.gz` directly.

## Old keyword groupings

```python
'Pronomen' : ['hon', 'henne*'],
'Moder' : ['mor', 'moder*', 'mamma*', 'mammor*', 'mödra*'],
'Syster' : ['syster*', 'systrar',],
'Flicka' : ['flick*'],
'Änka' : ['änka*', 'änke*'],
'Fröken' : ['fröken*', 'fröknar*'],
'Dam' : ['dam', 'dame*'],
'Hustru' : ['hustru*'],
'Dotter' : ['dotter*', 'döttrar*'],
'Fruntimmer*' : ['fruntimmer*'],
'piga' : ['piga*', 'pigor*'],
'Flicka' : ['flick*'],
'Hembiträde' : ['hembiträde*'],
'Jungfru' : ['jungfru*'],
'Arbeterska' : ['arbeterska*', 'arbeterskor*'],
'Fabriksarbeterska' : ['fabriksarbeterka*', 'fabriksarbeterskor*'],
'Hushållerska' : ['hushållerska*', 'hushållerskor*'],
'Lärarinna' : ['lärarinna*', 'lärarinnor*', 'småskolelärarinna*', 'småskolelärarinnor*'],
'Mjökerska' : ['mjölkerska*', 'mjölkerskor*'],
'Sjuksköterska' : ['sjuksköterska*', 'sjuksköterskor*', 'sköterska*', 'sköterskor*'],
'Tjänarinna' : ['tjänarinna*', 'tjänarinnor*', 'tjänstekvinna*', 'tjänstekvinnor*', 'tjänsteflick*', 'tjänstepiga*', 'tjänstepigor*'],
'Sömmerska' : ['sömmerska*', 'sömmerskor*'],
'Uppaskerska' : ['uppasserska*', 'uppaskerskor*'],
'Kokerska' : ['kokerska*', 'kokerskor*'],
```

## Word frequencies

A per-`(year, chamber, gender, party)` snapshot of word-level occurrences
per Kvinna category is shipped as `word_frequencies.csv.gz` (with
`.1.csv.gz` and `.2.csv.gz` variants filtered to Chamber 1 and Chamber 2
respectively). Notebooks should read via
[`src/data.py`](src/data.py) so the shipped format is a single
source of truth. See [`docs/word_frequencies.md`](docs/word_frequencies.md)
for the methodology, file schema, and reproduction recipe.

## Party attribution and backfilling

The upstream `persons.sqlite` from swerik is missing party affiliations
for roughly 42 % of utterances in our 1900–1940 window — some because a
speaker's tenure rows have empty `party` fields even when their party
is well documented elsewhere, some because famous parliamentarians
(e.g. Hjalmar Branting) have zero partied rows at all. The pipeline
rescues about 21.5 pp of these utterances in three layers, all applied
inside `create_database()` in [`src/prepare_db.py`](src/prepare_db.py):

1. **Direct affiliation load** — the original join through
   `load_person_dates_affiliation()`.
2. **Tier A+B backfill** — [`src/backfill.py`](src/backfill.py) synthesises
   affiliation rows for unpartied tenure windows that have a same-role
   partied neighbour within ±15 years, using a mode-across-windows rule
   with alphabetical tiebreak and a "Lindhagen guard" for single-window
   ambiguity. Tests in [`tests/test_backfill.py`](tests/test_backfill.py)
   pin the contract case-by-case.
3. **Manual overrides** — [`src/manual_party_overrides.csv`](src/manual_party_overrides.csv)
   ships a curated 56-row table covering the 25 most-cited bucket-C
   parliamentarians (Branting, Staaff, Bagge, Spångberg, Palmstierna,
   Nilsson, …). Per-era splits honour documented party switches. Loader
   is `load_manual_party_overrides()` in
   [`src/prepare_db.py`](src/prepare_db.py).

Combined effect: retention rises from 57.79 % to 79.26 %; the residual
20.53 % blank-party rate has a hard 3.69 pp floor from utterances whose
speaker identity is `"unknown"` in the source records. Full methodology,
per-tier numbers, disambiguation rules, and threats-to-validity are in
[`docs/party-backfill.md`](docs/party-backfill.md).

## Discussion metadata

The shipped `ToK_data_YYYY.sqlite3(.gz)` files carry a `discussion_id`
INTEGER column on the `utterance` table. It groups the 141k utterances
into ~2,236 topic arcs — the paper's operational definition of a
"discussion about women" (`max_gap=1, min_arc_length=2`: tolerate one
non-kvinna interjection, drop isolated single-utterance mentions).
Non-kvinna interjections inside an arc receive the surrounding arc's
id so `SELECT * FROM utterance WHERE discussion_id = N ORDER BY date`
returns the discussion as a contiguous span; utterances outside any
arc are NULL. Arcs never span sessions (one `record` = one day of
proceedings) — a debate that resumes on the following day gets a
fresh id. The column is computed by `compute_discussion_ids()` in
[`src/reduce_db.py`](src/reduce_db.py) after utterance migration;
tests are in [`tests/test_reduce_db.py`](tests/test_reduce_db.py).
See [`docs/discussion-metadata.md`](docs/discussion-metadata.md) for
the arc algorithm, semantics, and how to re-derive different
parameter sets from `kvinna_1/2/3 + prev/next`.

## Sanity checks for prev/next links

After building the database, verify that the linked list is well-formed:

```sql
-- One row per chamber should have prev IS NULL (first) and next IS NULL (last)
SELECT
    (SELECT COUNT(*) FROM utterance WHERE prev IS NULL) AS null_prev,
    (SELECT COUNT(*) FROM utterance WHERE next IS NULL) AS null_next;

-- All next-links should be symmetric with prev-links
SELECT
    (SELECT COUNT(*) FROM utterance WHERE next IS NOT NULL) AS has_next,
    (SELECT COUNT(*) FROM utterance a
     JOIN utterance b ON a.next = b.id
     WHERE b.prev = a.id) AS symmetric;
```

`null_prev` and `null_next` should both equal the number of chambers
(2 for the current corpus: Första kammaren and Andra kammaren).
`has_next` and `symmetric` should be equal.

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/)
(CC BY-NC 4.0). See [`LICENSE`](LICENSE).

## Citation

If you use this site or its derivative data, please cite the Zenodo deposit:
[10.5281/zenodo.18211787](https://doi.org/10.5281/zenodo.18211787).
Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

And make sure to cite the original source of the data:

```bib
@inproceedings{yrjanainen-etal-2024-swedish,
    title = "The {S}wedish Parliament Corpus 1867 {--} 2022",
    author = {Yrj{\"a}n{\"a}inen, V{\"a}in{\"o} Aleksi  and
      Mohammadi Nor{\'e}n, Fredrik  and
      Borges, Robert  and
      Jarlbrink, Johan  and
      {\r{A}}berg Brorsson, Lotta  and
      Olsson, Anders P.  and
      Snickars, Pelle  and
      Magnusson, M{\r{a}}ns},
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1400/",
    pages = "16100--16112",
}
```



## Contact

For questions or feedback, contact Mathias Johansson at
[MathiasJohansson@kultur.lu.se](mailto:MathiasJohansson@kultur.lu.se), or open
an [issue](https://github.com/DigitalHistory-Lund/ToK-Preparer/issues).
