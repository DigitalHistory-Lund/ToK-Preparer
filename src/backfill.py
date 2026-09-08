"""Backfill affiliation rows for unpartied tenure windows.

The direct affiliation loader in :mod:`src.prepare_db` skips tenure rows
where both party fields are empty. Many long-serving parliamentarians
have such gaps in the middle of a career whose earlier or later tenures
DO carry party info (see the audit that motivated this module for
numbers). This module carries the person's known party forward or
backward into those gaps, but only when the source neighbor is within
±15 years and shares the same role. The window was widened from an
original ±5 years after empirical review: Swedish parliamentary
careers frequently span 20-30 years with the same party affiliation,
and the mode-selection + Lindhagen guard combination handles the
few genuine within-career switches without requiring a tight radius.

The heavy design decisions live in this docstring:

* Same role only. Cross-role backfills (MP↔minister) are not attempted;
  they belong to a bolder tier not implemented here.
* Party selection when the neighbor window has multiple candidate labels:
  count each label's appearance across the person's whole partied history
  (per window, not per row), and pick the mode. Alphabetical tiebreak.
* Ambiguity guard: if the chosen label appears in only ONE partied window
  AND that window has ≥ 2 labels, skip the backfill for that unpartied
  window entirely. This protects cases like Carl Lindhagen where a single
  window carries three genuinely-different labels and the mode is a coin
  flip.
* Utan partibeteckning is a valid target label; it participates in the
  same mode calculation as any other party name.

Backfill rows never overlap with the direct affiliation rows because
they only target windows whose source rows were all party=NULL — those
windows contribute zero direct rows, so the utterance-join UPDATE cannot
see two candidates for the same date.
"""

from collections import Counter
from typing import Iterable, Iterator


TenureRow = tuple[str, int, int, str, str | None]
BackfillRow = tuple[str, int, int, str]

MAX_GAP_YEARS = 15


def _year(date_int: int) -> int:
    return date_int // 10000


def _pick_party(candidate_parties: set[str], party_counts: Counter[str]) -> str | None:
    """Return the mode party (alphabetical tiebreak) among the candidates;
    None if the candidate set is empty."""
    if not candidate_parties:
        return None
    ranked = sorted(candidate_parties, key=lambda p: (-party_counts[p], p))
    return ranked[0]


def _is_lindhagen_ambiguous(
    chosen: str, windows: list[tuple[int, int, str, set[str]]]
) -> bool:
    """True iff `chosen` appears in exactly one partied window and that
    window has ≥ 2 party labels (its 'home' window is itself ambiguous)."""
    homes = [w for w in windows if chosen in w[3]]
    return len(homes) == 1 and len(homes[0][3]) >= 2


def compute_backfill_affiliations(rows: Iterable[TenureRow]) -> Iterator[BackfillRow]:
    """Emit (person_id, start_int, end_int, party) rows to add to the
    affiliation table for unpartied tenure windows that qualify for
    Tier A+B backfill (±5 years, same role)."""
    # Group rows by person, then by (start, end, role) window.
    per_person: dict[str, dict[tuple[int, int, str], set[str]]] = {}
    for pid, s, e, role, party in rows:
        w = per_person.setdefault(pid, {}).setdefault((s, e, role), set())
        if party:
            w.add(party)

    for pid, windows in per_person.items():
        # Flat list of (s, e, role, parties_set) for this person.
        flat = [(s, e, role, parties) for (s, e, role), parties in windows.items()]
        partied = [w for w in flat if w[3]]
        if not partied:
            continue  # bucket C — nothing we can do

        # Per-window label counts across this person's partied history.
        counts: Counter[str] = Counter()
        for _, _, _, parties in partied:
            for p in parties:
                counts[p] += 1

        for s, e, role, parties in flat:
            if parties:
                continue  # partied window — already handled by direct join
            # Find same-role partied neighbors within MAX_GAP_YEARS.
            u_start_year = _year(s)
            u_end_year = _year(e)
            candidates = set()
            for ps, pe, prole, pparties in partied:
                if prole != role:
                    continue
                if pe <= s:  # neighbor ends at-or-before window starts (BEFORE, incl. touching)
                    if u_start_year - _year(pe) <= MAX_GAP_YEARS:
                        candidates |= pparties
                elif ps >= e:  # neighbor starts at-or-after window ends (AFTER, incl. touching)
                    if _year(ps) - u_end_year <= MAX_GAP_YEARS:
                        candidates |= pparties
                else:
                    # Overlapping or containing partied window — the direct
                    # join covers the shared dates, so backfill stays away.
                    pass
            chosen = _pick_party(candidates, counts)
            if chosen is None:
                continue
            if _is_lindhagen_ambiguous(chosen, partied):
                continue
            yield (pid, s, e, chosen)
