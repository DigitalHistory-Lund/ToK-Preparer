"""Static validator for src/queries.py — catches keywords that overlap between category groups."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from itertools import combinations

from .queries import queries


@dataclass(frozen=True)
class Conflict:
    kind: str  # "duplicate" | "subsumption" | "internal_duplicate" | "suffix_prefix_soft"
    group_a: str
    pattern_a: str
    group_b: str
    pattern_b: str


HARD_KINDS = frozenset({"duplicate", "subsumption", "internal_duplicate"})


def _is_prefix(p: str) -> bool:
    return p.endswith("*") and not p.startswith("*")


def _is_suffix(p: str) -> bool:
    return p.startswith("*") and not p.endswith("*")


def _is_contains(p: str) -> bool:
    return p.startswith("*") and p.endswith("*") and len(p) > 2


def _stem(p: str) -> str:
    if _is_contains(p):
        return p[1:-1].lower()
    if _is_suffix(p):
        return p[1:].lower()
    if _is_prefix(p):
        return p[:-1].lower()
    return p.lower()


def _overlaps(a: str, b: str) -> bool:
    """True iff patterns a and b accept at least one word in common (case-insensitive).

    Suffix↔prefix returns True: any prefix + arbitrary middle + any suffix can
    always coexist. Callers that need to distinguish this soft case from a
    concrete overlap should check the pattern shapes.

    Contains-patterns (``*X*``) match any word containing X, so they overlap
    with anything whose stem contains X (and vice versa).
    """
    ap, as_ = _is_prefix(a), _is_suffix(a)
    bp, bs = _is_prefix(b), _is_suffix(b)
    ac, bc = _is_contains(a), _is_contains(b)
    sa, sb = _stem(a), _stem(b)

    if ac and bc:
        return sa in sb or sb in sa
    if ac:
        return sa in sb
    if bc:
        return sb in sa

    if not (ap or as_) and not (bp or bs):
        return sa == sb
    if ap and bp:
        return sa.startswith(sb) or sb.startswith(sa)
    if ap and not (bp or bs):
        return sb.startswith(sa)
    if bp and not (ap or as_):
        return sa.startswith(sb)
    if as_ and not (bp or bs):
        return sb.endswith(sa)
    if bs and not (ap or as_):
        return sa.endswith(sb)
    if as_ and bs:
        return sa.endswith(sb) or sb.endswith(sa)
    if (as_ and bp) or (bs and ap):
        return True
    return False


def _is_soft(a: str, b: str) -> bool:
    """Suffix↔prefix cross-shape pairs are theoretically-overlap; treat as soft."""
    return (_is_suffix(a) and _is_prefix(b)) or (_is_suffix(b) and _is_prefix(a))


def find_conflicts(qs: dict[str, list[str]]) -> list[Conflict]:
    conflicts: list[Conflict] = []

    for group, patterns in qs.items():
        counts = Counter(patterns)
        for pattern, n in counts.items():
            if n > 1:
                conflicts.append(
                    Conflict("internal_duplicate", group, pattern, group, pattern)
                )

    for (g_a, patterns_a), (g_b, patterns_b) in combinations(qs.items(), 2):
        for a in set(patterns_a):
            for b in set(patterns_b):
                if not _overlaps(a, b):
                    continue
                if _is_soft(a, b):
                    kind = "suffix_prefix_soft"
                elif a == b:
                    kind = "duplicate"
                else:
                    a_wild = _is_prefix(a) or _is_suffix(a)
                    b_wild = _is_prefix(b) or _is_suffix(b)
                    if a_wild and b_wild:
                        # Two wildcards concretely overlap → real categorisation
                        # ambiguity at the pattern level.
                        kind = "subsumption"
                    else:
                        # A wildcard absorbs a literal in another group. The
                        # literal becomes dead code under first-hit-wins, but
                        # classification is unambiguous.
                        kind = "literal_absorbed"
                conflicts.append(Conflict(kind, g_a, a, g_b, b))

    return conflicts


def report(conflicts: list[Conflict]) -> str:
    if not conflicts:
        return "queries.py: OK — no cross-group conflicts"

    lines = [f"queries.py: {len(conflicts)} conflicts", ""]

    internal = [c for c in conflicts if c.kind == "internal_duplicate"]
    if internal:
        lines.append("[internal duplicates]")
        for c in internal:
            lines.append(f"  {c.pattern_a!r} appears >1× in {c.group_a!r}")
        lines.append("")

    cross = [c for c in conflicts if c.kind != "internal_duplicate"]
    by_pair: dict[tuple[str, str], list[Conflict]] = {}
    for c in cross:
        by_pair.setdefault((c.group_a, c.group_b), []).append(c)

    for (g_a, g_b), items in by_pair.items():
        lines.append(f"[{g_a!r} × {g_b!r}]")
        for c in items:
            if c.kind == "duplicate":
                op = "=="
            elif c.kind == "suffix_prefix_soft":
                op = "~~"
            elif c.kind == "literal_absorbed":
                op = "⊇"
            else:
                op = "⊃⊂"
            lines.append(f"  {c.pattern_a}  ({g_a})  {op}  {c.pattern_b}  ({g_b})")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def validate() -> None:
    conflicts = find_conflicts(queries)
    if not conflicts:
        return
    print(report(conflicts), file=sys.stderr)
    if any(c.kind in HARD_KINDS for c in conflicts):
        raise SystemExit(1)


if __name__ == "__main__":
    validate()
    print("queries.py: OK")
