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

## Sanity checks for prev/next links

After building the database, verify that the linked list is well-formed:

```sql
-- Exactly one row should have prev IS NULL (first) and one next IS NULL (last)
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

`null_prev` and `null_next` should both be `1`. `has_next` and `symmetric` should be equal.

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
