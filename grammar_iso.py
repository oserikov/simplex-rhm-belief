"""Deterministic isomorphism test for RHM grammars.

An RHM grammar (here ``s=2, L=3, v=8``) has four size-``v`` alphabets (levels
``0..L``). An isomorphism ``A -> B`` is a tuple of per-level symbol permutations
``(pi_0, .., pi_L)`` such that for every level ``ell`` and parent ``p``::

    ruleset_B[ell][pi_ell(p)] == { (pi_{ell+1}(a), pi_{ell+1}(b))
                                   : (a, b) in ruleset_A[ell][p] }

i.e. the standard graph-isomorphism notion (a vertex relabeling that makes the two
rule-table structures coincide), restricted to within-level renamings because level
is an intrinsic structural property of every symbol node.

Unambiguity (the ``v*m`` right-hand-side tuples per level are all distinct) makes each
parent's ``m``-rule set unique, so fixing the leaf permutation ``pi_L`` *forces*
``pi_{L-1}, .., pi_0`` (each relabeled rule-set must match exactly one parent of ``B``).
We therefore brute-force only ``pi_L`` (``v!``) and propagate upward -- exact and fast.

``allow_flip`` additionally permits a single global left<->right child swap; positions
are absolute string positions, so the default (no flip) is the setting relevant to the
left-to-right next-token task.
"""

from __future__ import annotations

from itertools import permutations

from rhm import Grammar


def ruleset(g: Grammar, ell: int, p: int) -> frozenset[tuple[int, ...]]:
    """The unordered set of (ordered) child tuples for parent ``p`` at level ``ell``."""
    return frozenset(tuple(int(x) for x in pair) for pair in g.rules[ell][p])


def _parent_index(g: Grammar, ell: int) -> dict[frozenset, int]:
    """Map each parent's rule-set to its symbol (well-defined: all sets distinct)."""
    d = {ruleset(g, ell, p): p for p in range(g.v)}
    assert len(d) == g.v, "parent rule-sets not distinct (unambiguity broken)"
    return d


def _verify(gA: Grammar, gB: Grammar, pis: list, flip: bool) -> bool:
    """Apply the relabeling to all of A's rules; check it reproduces B exactly."""
    pos = [1, 0] if flip else [0, 1]
    for ell in range(gA.L):
        got = {}
        for p in range(gA.v):
            got[pis[ell][p]] = frozenset(
                (pis[ell + 1][pair[pos[0]]], pis[ell + 1][pair[pos[1]]])
                for pair in ruleset(gA, ell, p)
            )
        if got != {p: ruleset(gB, ell, p) for p in range(gB.v)}:
            return False
    return True


def isomorphic(gA: Grammar, gB: Grammar, allow_flip: bool = False):
    """Return an isomorphism ``(pis, flip)`` if one exists, else ``None``.

    ``pis`` is a list of ``L+1`` per-level permutations mapping A-symbols to B-symbols.
    """
    if (gA.s, gA.L, gA.v, gA.m) != (gB.s, gB.L, gB.v, gB.m):
        return None
    v, L = gA.v, gA.L
    idxB = [_parent_index(gB, ell) for ell in range(L)]
    for flip in ([False, True] if allow_flip else [False]):
        pos = [1, 0] if flip else [0, 1]

        def relabel(rs, pic, pos=pos):  # noqa: ANN001  (bind pos per flip)
            return frozenset((pic[pair[pos[0]]], pic[pair[pos[1]]]) for pair in rs)

        for leaf_perm in permutations(range(v)):
            pis = [None] * L + [list(leaf_perm)]
            child = list(leaf_perm)
            ok = True
            for ell in range(L - 1, -1, -1):  # forced upward from the leaves
                pi, seen = [None] * v, set()
                for p in range(v):
                    q = idxB[ell].get(relabel(ruleset(gA, ell, p), child))
                    if q is None or q in seen:
                        ok = False
                        break
                    pi[p] = q
                    seen.add(q)
                if not ok:
                    break
                pis[ell] = pi
                child = pi
            if ok and _verify(gA, gB, pis, flip):
                return (pis, flip)
    return None


def pairwise_isomorphic_pairs(grammars, allow_flip: bool = False):
    """Indices (i, j) of every isomorphic pair in ``grammars``."""
    n = len(grammars)
    return [(i, j) for i in range(n) for j in range(i + 1, n)
            if isomorphic(grammars[i], grammars[j], allow_flip) is not None]


if __name__ == "__main__":
    grammars = [Grammar.random(seed=i) for i in range(10)]
    for allow_flip in (False, True):
        tag = "relabel + global L/R flip" if allow_flip else "relabel only"
        pairs = pairwise_isomorphic_pairs(grammars, allow_flip)
        print(f"[{tag}] isomorphic pairs among {len(grammars) * 9 // 2}: "
              f"{pairs if pairs else 'NONE'} -> {len(grammars) - len(pairs)} distinct classes")
