[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18211787.svg)](https://doi.org/10.5281/zenodo.18211787)

# Pre-processing data for ToK utterance browser

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

## Contact

For questions or feedback, contact Mathias Johansson at
[MathiasJohansson@kultur.lu.se](mailto:MathiasJohansson@kultur.lu.se), or open
an [issue](https://github.com/DigitalHistory-Lund/ToK-Preparer/issues).
