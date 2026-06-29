"""Random Hierarchy Model: sampler + exact belief propagation.

A fixed regular tree of arity ``s`` and depth ``L``. Leaves are observable
tokens; internal nodes are hidden latents. Sequence length ``d = s ** L``.

Each nonterminal symbol has ``m`` production rules, each an ``s``-tuple of
next-level children, drawn uniformly at random subject to *unambiguity* (C2:
no two distinct symbols share a child-tuple).

Belief state: for a leaf prefix ``x_1..x_k`` the posterior over hidden latents
consistent with that prefix, computed by sum-product (belief propagation) on
the tree. Primary target: the posterior over the root class.

TODO: implement Grammar (rule sampling + unambiguity), sample(), and
belief_propagation(); verify against brute-force enumeration on a tiny tree.
"""
