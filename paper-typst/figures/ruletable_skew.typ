#figure(
  table(
    columns: (0.9fr, 1.4fr, 1.4fr, 1.4fr),
    inset: 4pt,
    align: horizon,
    [Parent], [Top expansion], [Middle expansion], [Bottom expansion],
    [`S1`], [0.995·`[S3, S1]` or 0.00467·`[S4, S8]`], [0.0978·`[S1, S5]` or 0.902·`[S6, S3]`], [0.0000894·`[S5, S8]` or 1.000·`[S7, S2]`],
    [`S2`], [0.000000134·`[S4, S3]` or 1.000·`[S1, S3]`], [0.0000129·`[S8, S2]` or 1.000·`[S6, S1]`], [0.584·`[S6, S6]` or 0.416·`[S4, S1]`],
    [`S3`], [0.368·`[S6, S7]` or 0.632·`[S5, S6]`], [0.991·`[S7, S6]` or 0.00869·`[S5, S3]`], [0.0368·`[S3, S7]` or 0.963·`[S4, S6]`],
    [`S4`], [1.000·`[S6, S8]` or 0.000291·`[S7, S7]`], [0.404·`[S4, S5]` or 0.596·`[S3, S5]`], [0.822·`[S1, S3]` or 0.178·`[S6, S7]`],
    [`S5`], [0.597·`[S6, S2]` or 0.403·`[S1, S5]`], [0.584·`[S1, S4]` or 0.416·`[S2, S5]`], [0.933·`[S5, S5]` or 0.0665·`[S6, S1]`],
    [`S6`], [0.948·`[S8, S6]` or 0.0519·`[S2, S7]`], [0.840·`[S7, S1]` or 0.160·`[S2, S1]`], [0.999·`[S2, S4]` or 0.000881·`[S2, S6]`],
    [`S7`], [0.889·`[S4, S7]` or 0.111·`[S1, S1]`], [0.174·`[S5, S7]` or 0.826·`[S3, S1]`], [0.983·`[S7, S6]` or 0.0168·`[S8, S5]`],
    [`S8`], [0.993·`[S2, S2]` or 0.00687·`[S5, S7]`], [0.998·`[S3, S4]` or 0.00178·`[S2, S2]`], [0.409·`[S1, S7]` or 0.591·`[S2, S1]`],
  ),
  caption: [Skewed example grammar (skew=high, Dirichlet alpha=0.2, rho=0). Each cell is prefixed by the exact rule-choice probability from the saved grammar.npz; several parents are near-deterministic (one rule near 1).],
) <ruletable-skew>
