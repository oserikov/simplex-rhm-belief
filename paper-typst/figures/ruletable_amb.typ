#figure(
  table(
    columns: (0.9fr, 1.4fr, 1.4fr, 1.4fr),
    inset: 4pt,
    align: horizon,
    [Parent], [Top expansion], [Middle expansion], [Bottom expansion],
    [`S1`], [`[S7, S7]` or #strong[`[S5, S7]`]], [#strong[`[S7, S1]`] or #strong[`[S6, S1]`]], [#strong[`[S8, S5]`] or #strong[`[S7, S6]`]],
    [`S2`], [`[S4, S3]` or #strong[`[S6, S2]`]], [`[S3, S5]` or `[S8, S6]`], [#strong[`[S8, S5]`] or #strong[`[S7, S6]`]],
    [`S3`], [`[S6, S7]` or #strong[`[S5, S7]`]], [`[S8, S1]` or #strong[`[S7, S1]`]], [`[S7, S5]` or #strong[`[S8, S5]`]],
    [`S4`], [#strong[`[S5, S7]`] or #strong[`[S1, S1]`]], [#strong[`[S6, S6]`] or `[S3, S7]`], [#strong[`[S7, S6]`] or #strong[`[S1, S5]`]],
    [`S5`], [#strong[`[S6, S2]`] or #strong[`[S4, S7]`]], [#strong[`[S6, S6]`] or `[S5, S4]`], [`[S2, S5]` or #strong[`[S5, S1]`]],
    [`S6`], [`[S8, S6]` or #strong[`[S5, S7]`]], [`[S4, S2]` or #strong[`[S7, S1]`]], [#strong[`[S8, S5]`] or #strong[`[S5, S1]`]],
    [`S7`], [#strong[`[S4, S7]`] or #strong[`[S1, S1]`]], [#strong[`[S6, S6]`] or #strong[`[S6, S1]`]], [#strong[`[S8, S5]`] or #strong[`[S4, S4]`]],
    [`S8`], [#strong[`[S4, S7]`] or #strong[`[S1, S1]`]], [#strong[`[S6, S6]`] or `[S4, S7]`], [#strong[`[S1, S5]`] or #strong[`[S4, S4]`]],
  ),
  caption: [Ambiguous example grammar (skew=none, rho=0.6). Rule-choice probabilities are uniform (1/m), so cells show plain child pairs; pairs marked in bold are shared child-tuples produced by more than one (parent, rule) at that level.],
) <ruletable-amb>
