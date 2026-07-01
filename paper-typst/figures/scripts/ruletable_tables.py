# Figure: ruletable_skew.typ / ruletable_amb.typ
# Message: mirror the base grammar's @ruletable but for the skewed and
#   ambiguous example grammars, generated (not hand-transcribed) so the
#   probabilities/child-pairs shown always match the saved npz.
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from _style import OUTDIR, ROOT

sys.path.insert(0, ROOT)
from analyze import load_grammar  # noqa: E402

TABLE_HEAD = """#figure(
  table(
    columns: (0.9fr, 1.4fr, 1.4fr, 1.4fr),
    inset: 4pt,
    align: horizon,
    [Parent], [Top expansion], [Middle expansion], [Bottom expansion],
"""


def fmt_prob(p):
    """3 significant figures, plain decimal (no scientific notation)."""
    if p == 0:
        return "0.00"
    from decimal import Decimal

    d = Decimal(repr(p))
    exp = d.adjusted()  # position of most significant digit
    digits = max(0, 2 - exp)
    s = f"{p:.{digits}f}"
    return s


def pair_str(rules, level, parent, ri, prob=None, mark=False):
    a, b = int(rules[level][parent, ri, 0]) + 1, int(rules[level][parent, ri, 1]) + 1
    text = f"`[S{a}, S{b}]`"          # raw child-pair
    if mark:
        text = f"#strong[{text}]"     # bold the whole raw span (shared child-tuple)
    if prob is not None:
        text = f"{fmt_prob(prob)}·{text}"
    return text


def row_cell(g, level, parent, collide=None):
    rules, probs = g.rules, g.probs
    parts = []
    for ri in range(g.m):
        prob = float(probs[level][parent, ri]) if collide is None else None
        mark = collide is not None and collide[(level, parent, ri)]
        parts.append(pair_str(rules, level, parent, ri, prob=prob, mark=mark))
    return "[" + " or ".join(parts) + "]"


def write_table(g, out_path, label, caption, collide=None):
    lines = [TABLE_HEAD]
    for p in range(g.v):
        cells = [row_cell(g, level, p, collide=collide) for level in range(3)]
        lines.append(f"    [`S{p + 1}`], {cells[0]}, {cells[1]}, {cells[2]},\n")
    lines.append(f"  ),\n  caption: [{caption}],\n) <{label}>\n")
    with open(out_path, "w") as f:
        f.writelines(lines)
    print("wrote", out_path)


def collisions(g):
    """(level, parent, rule_idx) -> True if its child pair is shared by >1 (parent, rule)."""
    flags = {}
    for level in range(3):
        pairs = [tuple(int(x) for x in g.rules[level][p, ri])
                 for p in range(g.v) for ri in range(g.m)]
        counts = Counter(pairs)
        for p in range(g.v):
            for ri in range(g.m):
                key = tuple(int(x) for x in g.rules[level][p, ri])
                flags[(level, p, ri)] = counts[key] > 1
    return flags


if __name__ == "__main__":
    g_skew = load_grammar(
        os.path.join(ROOT, "results/noncollapse/skhigh_am0_g00_L2_d128_h4_t4000_s0"))
    write_table(
        g_skew, os.path.join(OUTDIR, "ruletable_skew.typ"), "ruletable-skew",
        "Skewed example grammar (skew=high, Dirichlet alpha=0.2, rho=0). Each cell is "
        "prefixed by the exact rule-choice probability from the saved grammar.npz; several "
        "parents are near-deterministic (one rule near 1).",
    )

    g_amb = load_grammar(
        os.path.join(ROOT, "results/noncollapse/sknone_am0.6_g00_L2_d128_h4_t4000_s0"))
    write_table(
        g_amb, os.path.join(OUTDIR, "ruletable_amb.typ"), "ruletable-amb",
        "Ambiguous example grammar (skew=none, rho=0.6). Rule-choice probabilities are "
        "uniform (1/m), so cells show plain child pairs; pairs marked in bold are shared "
        "child-tuples produced by more than one (parent, rule) at that level.",
        collide=collisions(g_amb),
    )
