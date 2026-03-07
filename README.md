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
