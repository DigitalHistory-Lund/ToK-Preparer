queries = {
    "kvinna 1": ["kvinn*"],
    "Kvinna 2": [
        # Suffix wildcards for compound-tail kin forms. Only patterns that
        # stand on their own without a deny-list ship: *dotter is clean;
        # *änka was rejected because Swedish verbs "tänka", "inskränka",
        # "betänka", "sänka", "skänka", "misstänka" all collide with it.
        "*dotter",
        "änka*",
        "änke*",
        "dam",
        "damern*",  # "mina damer och herrar"
        "dotter*",
        "döttrar*",
        "flick*",
        "fröken*",
        "fröknar*",
        "fru",
        "fruntimmer*",
        "henne*",
        "hon",
        "*hustru*",
        "jungfru*",
        "mamma*",
        "mammor*",
        "moder",
        "moderns",
        "mödra*",
        "mor",
        "syster*",
        "systrar",
    ],  #
    # Work-related and possibly work-related titles
    "Kvinna 3": [
        # (*hustru was previously here; consolidated into K2's *hustru*
        #  because the two patterns produced spurious K2∩K3 overlap on
        #  bare "hustru" tokens and every *hustru catch is semantically
        #  a K2 kin term, not a work title.)
        "ägarinna",
        # "änkebaronessa",      # covered by K2's "änke*"
        # "änkefriherrinna",    # covered by K2's "änke*"
        # "änkegrevinna",       # covered by K2's "änke*"
        # "änkehjälptagerska",  # covered by K2's "änke*"
        # "änkemadam",          # covered by K2's "änke*"
        "arbeterska*",
        "arbeterskor*",
        "barnavårdarinna",
        "barnmorska",
        # "borgardotter",  # covered by K2's "*dotter"
        "diakonissa",
        "enkefru",
        "fabriksarbeterskor*",
        "fästekvinna",
        # "flickskoleelev",  # covered by K2's "flick*"
        "förestånderska",
        # "fosterdotter",  # covered by K2's "*dotter"
        "friherrinna",
        "furstinna",
        "grevinna",
        "hembiträde*",
        # "hemmadotter",  # covered by K2's "*dotter"
        "hertiginna",
        "husförestånderska",
        "hushållerska",
        "hushållerska*",
        "hushållerskor*",
        "innehafvarinna",
        "innehavarinna",
        "kokerska*",
        "kokerskor*",
        "konkubin",
        "kronprinsess*",
        "kulla",
        # "kvinnsperson",  # covered by K1's "kvinn*"
        "lärarinna*",
        "lärarinnor*",
        "madame",
        "mademoiselle",
        "maka",
        "mamsell",
        "markisinna",
        "matrona",
        "mjölkerska*",
        "mjölkerskor*",
        "piga*",
        "pigor*",
        "prinsess*",
        "sjuksköterska",
        "sjuksköterska*",
        "sjuksköterskor*",
        "sköka",
        "sköterska",
        "sköterska*",
        "sköterskor*",
        "småskolelärarinna*",
        "småskolelärarinnor*",
        "sömmerska*",
        "sömmerskor*",
        "städerska",
        "studentska",
        "tjänarinna",
        "tjänarinna*",
        "tjänarinnor*",
        "tjänsteflick*",
        "tjänstekvinna*",
        "tjänstepigor*",
        "undantagsgumma",
        "uppasserska*",
    ],
}
