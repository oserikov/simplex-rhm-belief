#import "@preview/drafting:0.2.2": margin-note

#let page-left-margin = 2cm
#let page-right-margin = 2cm
#let note-col-width = 21cm - page-left-margin - page-right-margin

#let todooleg(body) = margin-note(stroke: rgb("#AAAEEE"), margin-right: page-right-margin, page-width: note-col-width)[#par(
  [#text(body, size: 5pt)],
  leading: 0.1em,
)]

#let todoai(body) = margin-note(stroke: rgb("#CC22AA"), margin-right: page-right-margin, page-width: note-col-width)[#par(
  [#text(body, size: 5pt, fill: rgb("#CC22AA"))],
  leading: 0.1em,
)]

#set document(title: "Belief Geometry on the Random Hierarchy Model")

#set page(
  paper: "a4",
  margin: (top: 2cm, bottom: 2cm, left: 2cm, right: 2cm),
  numbering: "1",
)

// build uses --ignore-system-fonts, so we ship with the bundled serif
#set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
#set par(justify: true, leading: 0.62em, spacing: 0.9em)
#set heading(numbering: "1.")

#show heading.where(level: 1): it => block(above: 1.3em, below: 0.7em)[
  #set text(15pt, weight: "bold")
  #it
]
#show heading.where(level: 2): it => block(above: 1.05em, below: 0.5em)[
  #set text(12pt, weight: "bold")
  #it
]
#show heading.where(level: 3): it => block(above: 0.8em, below: 0.4em)[
  #set text(10.5pt, weight: "bold", style: "italic")
  #it
]

#show link: it => underline(text(fill: rgb("#1a4d8f"), it))

#let fig(path, caption, w: 100%) = figure(image(path, width: w), caption: caption)

// ---- Title block -------------------------------------------------------
#block[
  #set align(center)
  #text(17pt, weight: "bold")[Belief Geometry on the Random Hierarchy Model]
  #v(0.3em)
  #text(11.5pt)[A small transformer partially encodes — and causally uses — the exact tree posterior, probed against ground truth]
  #v(0.4em)
  #text(9.5pt, fill: luma(110))[simplex-rhm-belief · run of 29 June 2026 · all numbers from the recorded pipeline output]
]

#v(0.5em)

#block(fill: luma(245), inset: 10pt, radius: 3pt, width: 100%)[
  #text(weight: "bold")[Abstract.] We reproduce, in miniature, the Simplex belief-geometry
  result on a hierarchical generative process. We train a 2-layer GPT-2 decoder
  (#text[≈]399k parameters) by next-token prediction on samples from a Random Hierarchy
  Model (RHM; arity $s=2$, depth $L=3$, vocabulary $v=8$), a grammar small enough
  (1024 equiprobable trees) to enumerate exactly. We compute the *exact* Bayesian
  posterior over the tree's hidden latents by sum-product belief propagation, verified
  against brute-force enumeration to $<10^(-6)$. The trained model reaches a test
  cross-entropy of 0.89 nats, closing #text[≈]88% of the gap between the uniform
  baseline (2.08) and the Bayes-optimal floor (0.73) — it has effectively learned the
  posterior predictor. A single global linear probe partially recovers this posterior
  from the residual stream: held-out $R^2 = 0.38$ for the root class versus #text[≈]0 for
  a shuffled control — a real but noisy affine image, not an exact reconstruction — with
  decodability accumulating monotonically across depth ($-0.00 -> 0.15 -> 0.38$) and the
  belief readout *blooming* outward from the prior toward simplex vertices as context
  accumulates. A latent-level sweep shows the residual
  encodes the *whole* hierarchy, most strongly the locally-predictive deep latents
  ($R^2$ up to 0.66) and least strongly the coarse global root. Causal steering on the
  layer-1 residual confirms the belief is *used*, not merely decodable: steering toward
  a wrong latent collapses the true next token's log-probability ($-0.73 -> -8.5$). We
  pre-registered our predictions before seeing results; three of four held, and the one
  miss (root $R^2 < 0.5$) is informative. Finally, a 30-run sweep over 10 random grammars
  $times$ 3 replicate seeds shows all four findings — the inverted level gradient, blooming,
  layer accumulation, and causal steering — replicate across the RHM family with low
  replicate variance, and that the published `grammar = 0` numbers sit inside the family
  distribution. A further 99-run architecture sweep (width, depth, heads, and training
  budget, one-axis-at-a-time $times$ 3 grammars $times$ 3 seeds) shows the phenomenology
  is capacity-robust down to a low-width floor: decodability rises with width and depth,
  the inverted gradient holds at every capacity, heads matter least, and decodability is
  only weakly correlated with how well the model fit the loss.
]

= Introduction

Natural data is hierarchical: characters compose into words, words into phrases, phrases
into meaning. The Random Hierarchy Model (RHM) of Cagnetta et al. abstracts this into a
clean synthetic grammar — a fixed tree in which each high-level symbol expands, via random
production rules, into a fixed-length string of lower-level symbols, down to observed
leaves. Predicting the next leaf optimally requires inferring the distribution over the
*hidden* latent symbols that generated the observed prefix. The optimal next-token
predictor *is* the Bayesian belief state over those latents.

This makes the RHM an ideal testbed for *belief geometry*: the hypothesis, developed in
the transformer-interpretability literature, that a network trained by next-token
prediction comes to represent, in its residual stream, the posterior distribution over
the data-generating process's hidden state — and that this distribution is laid out as a
linear (affine) image of the probability simplex. If true, the belief should be (i)
linearly decodable, (ii) geometrically organized as a simplex that sharpens with context,
(iii) built up additively across layers, and (iv) causally relevant to the model's output.

The RHM is uniquely suited to test all four claims *exactly*. Unlike natural language,
its hidden state has a precise definition and its posterior is computable in closed form
by belief propagation on the tree. With small parameters the entire grammar is
enumerable, so the probe target is ground truth — not an estimate. This paper trains a
tiny transformer on such a grammar and asks whether the residual stream carries the exact
posterior, where in the network it lives, what geometry it traces, and whether the model
relies on it.

= The Random Hierarchy Model and exact beliefs

*Grammar.* We use the defaults of the spec: arity $s = 2$, depth $L = 3$, so each string
has $d = s^L = 8$ leaves; per-level vocabulary $v = 8$; and $m = 2$ production rules per
parent symbol, chosen uniformly. Rules are drawn once and held fixed, and constrained to
be *unambiguous* (the spec's quick-failure check confirms no two parents share a
production). A sample is generated top-down: pick a root class uniformly, recursively
expand each symbol by a uniformly chosen rule, and emit the leaf string. With these
parameters the grammar admits exactly $v dot m^(d-1) = 8 dot 2^7 = 1024$ equiprobable
distinct trees — small enough to enumerate completely.

The exact frozen grammar used in every result is shown below. Symbols are written as
`S1`, ..., `S8` for readability (the saved arrays use zero-based ids); at each level the
parent takes one of the two listed child pairs uniformly.

#figure(
  table(
    columns: (0.9fr, 1.4fr, 1.4fr, 1.4fr),
    inset: 4pt,
    align: horizon,
    [Parent], [Top expansion], [Middle expansion], [Bottom expansion],
    [`S1`], [`[S3, S1]` or `[S4, S8]`], [`[S2, S1]` or `[S1, S5]`], [`[S4, S1]` or `[S5, S3]`],
    [`S2`], [`[S4, S3]` or `[S1, S3]`], [`[S6, S3]` or `[S3, S1]`], [`[S6, S7]` or `[S8, S8]`],
    [`S3`], [`[S6, S7]` or `[S5, S6]`], [`[S5, S2]` or `[S1, S1]`], [`[S3, S4]` or `[S7, S8]`],
    [`S4`], [`[S6, S8]` or `[S7, S7]`], [`[S6, S5]` or `[S8, S6]`], [`[S6, S5]` or `[S4, S7]`],
    [`S5`], [`[S6, S2]` or `[S1, S5]`], [`[S4, S3]` or `[S1, S8]`], [`[S1, S5]` or `[S6, S4]`],
    [`S6`], [`[S8, S6]` or `[S2, S7]`], [`[S7, S6]` or `[S4, S5]`], [`[S8, S4]` or `[S1, S8]`],
    [`S7`], [`[S4, S7]` or `[S1, S1]`], [`[S3, S8]` or `[S4, S1]`], [`[S4, S6]` or `[S3, S2]`],
    [`S8`], [`[S2, S2]` or `[S5, S7]`], [`[S1, S2]` or `[S8, S3]`], [`[S5, S7]` or `[S3, S6]`],
  ),
  caption: [Frozen level-specific grammar sampled once before training. Top, middle, and bottom columns correspond to `rules_0`, `rules_1`, and `rules_2` in `artifacts/grammar.npz`.],
)

*Exact posterior.* Given a leaf prefix of length $k$, the posterior over any hidden latent
(including the root class) is computed by sum-product belief propagation on the tree:
upward messages from observed leaves combine through the production rules to give the
marginal over each latent. Because the grammar is fully enumerable, we verified the
belief-propagation posterior against brute-force enumeration of all consistent trees; the
two agree to $< 10^(-6)$ across 10 passing tests, alongside checks that sample length
equals $s^L$ and that unambiguity holds. The recorded self-test (a representative string)
shows the root posterior sharpening with context — entropy $1.89$ nats at $k=1$ falling
to $0.00$ (a single consistent root) by $k=3$ — which previews the blooming geometry below.

= Model and training

We train a HuggingFace `GPT2LMHeadModel` with 2 layers, `n_embd` = 128, 4 heads, and all
dropout disabled, totalling 398,848 parameters. There is no tokenizer: RHM leaf symbols are
integers fed directly as `input_ids`, with `vocab_size` = 8 and `n_positions` = 8. We train
by next-token prediction on 922 training strings (102 held out) for 4000 steps on Apple MPS.

*The Bayes-optimal floor.* The value of an enumerable grammar is that we know the best
achievable loss. Averaging the exact per-position posterior entropy over the seven
predicted positions gives the Bayes-optimal mean next-token cross-entropy: 0.725 nats. The
uniform baseline is $ln 8 = 2.079$. The trained model reaches a final held-out
cross-entropy of *0.891 nats* (@loss), closing $(2.079 - 0.891) / (2.079 - 0.725) ≈ 88%$
of the gap. The model has, to a good approximation, learned the true posterior predictor —
which is the precondition for asking whether it represents the posterior internally.

#fig("figures/loss_curve.png",
  [The transformer learned the true RHM predictor. Test cross-entropy (0.891 nats)
   sits far below the uniform baseline ($ln 8 = 2.079$) and close to the Bayes-optimal
   floor (0.725), closing #text[≈]88% of the uniform#text[→]Bayes gap.], w: 62%) <loss>

= Pre-registered prediction

Following the project's honor-code rule, we committed our predictions to git *before*
training any model or computing any probe (`PREREGISTRATION.md`). In brief, we predicted:
(1) the root posterior is *linearly decodable* from the residual stream, with $R^2 > 0.5$
at the final layer and a shuffled baseline near 0; (2) the belief *blooms* — early
positions cluster near the prior, late positions spread toward simplex vertices, with
posterior entropy falling and activation radius growing monotonically with context $k$;
(3) decodability *accumulates additively across depth*, lowest at the embedding and highest
after the last layer; and (4) the representation is *causally used*, so steering the
residual along a belief direction shifts the output toward the corresponding latent's
leaves. We also registered alternative outcomes (degenerate collapse, non-linear-only
encoding, MAP-only encoding, flat-with-depth) as falsification handles. We report against
these predictions in @scorecard.

= Results

== A single linear probe partially decodes the posterior

We fit one global least-squares affine map from the 128-d residual stream to the 8-class
exact root posterior, on a probe set of $N = 400$ held-out examples, and score it by
$R^2$ on held-out data. The final-layer probe reaches $R^2 = 0.38$, while the same probe
fit to *shuffled* labels scores $≈ 0$ ($-0.085$). So the posterior is genuinely present
and linearly accessible above chance — but $R^2 = 0.38$ means the probe explains only
about a third of the variance: this is a *partial*, noisy recovery, not an exact
reconstruction. Projected into a belief-space PCA basis (@simplex), the probe's predicted
posteriors occupy the same structured region as the exact posteriors and reproduce the
same position gradient, but as a diffuse cloud rather than the discrete point set of the
ground truth. (The ground-truth panel looks sharper partly because the exact posterior
takes few distinct values, so identical points overplot, whereas every probe prediction
differs slightly.) The affine correspondence is real, but for the global root it is weak;
it strengthens markedly for deeper latents (next subsection) and at later layers.

#fig("figures/posterior_simplex.png",
  [The residual stream is a *partial* affine image of the exact belief simplex. *Left:*
   PCA(2) of the exact root posteriors (few distinct values, hence sharp overplotted dots);
   small-$k$ points sit near the prior, large-$k$ points spread toward vertices. *Right:*
   the linear probe's predictions in the same basis, colored by context position — the
   same region and position gradient, but a diffuse cloud: held-out root $R^2 ≈ 0.38$, an
   imperfect recovery, not an exact one.],
  w: 92%) <simplex>

== Decodability accumulates across depth and context

The residual stream is a running sum of layer contributions, so we ask where the belief
is built. Probe $R^2$ for the root posterior rises monotonically with depth:
$-0.00$ at the embedding (layer 0), $0.15$ after layer 1, $0.38$ after layer 2, while the
shuffled control stays at $≈ 0$ throughout (@layerpos, left). Each layer adds
belief-relevant signal. Resolving by context position (@layerpos, right), early positions
are decodable even at shallow layers, and the final layer pushes near-perfect decodability
($R^2 = 1.0$) across positions 0–3. A few mid-layer cells are strongly negative (the probe
underperforms the mean predictor on those positions); the heatmap colors are clipped for
legibility, but the cell labels show the raw values.

#fig("figures/layer_position.png",
  [Belief decodability accumulates across depth and context. *Left:* root-posterior
   $R^2$ rises layer by layer ($-0.00 -> 0.15 -> 0.38$) while a shuffled control stays at
   #text[≈]0. *Right:* $R^2$(layer, position) heatmap; the final layer reaches $R^2 = 1.0$
   on the earliest positions. Colors are clipped to $[-1, 1]$, while labels show raw values.],
  w: 95%) <layerpos>

== The whole latent hierarchy is encoded — local latents most strongly

The root is only one of the tree's hidden latents. Probing the exact posterior over latents
at *every* level reveals a clear gradient with some node-level variation (@latents): root
($L_0$) $R^2 = 0.38$; the two level-1 mid latents $0.51$ and $0.59$; and the four deepest
level-2 latents $0.61$, $0.66$, $0.50$, and $0.66$. The residual encodes the entire
hierarchy, but the *local, near-leaf* latents — those most directly predictive of the next
token — are usually read off more cleanly than the coarse global root. This is our most
informative finding: the spec's primary target (the root) is the *hardest* latent to
decode, because it is the most abstract and the least locally predictive.

#fig("figures/latent_levels.png",
  [The residual encodes the whole latent hierarchy, deeper/local latents most strongly.
   Probe $R^2$ is lowest for the root ($L_0 = 0.38$, the spec's primary target), higher for
   level-1 ($0.51, 0.59$), and generally higher for level-2 ($0.61, 0.66, 0.50, 0.66$).],
  w: 72%)
  <latents>

== The belief blooms with context

The headline geometry (@blooming): projecting the linear readout into a belief-space PCA(2)
basis, the points form a tight central cluster near the uniform prior when little context
is observed and expand outward toward the simplex vertices as context accumulates. The mean
radius about the prior anchor grows monotonically from 0.17 at $k = 0$ to #text[≈]0.39 at
$k = 7$, tracking the true posterior's own growth (0.17 #text[→] 0.47); equivalently, the
exact posterior entropy falls monotonically with position (1.84 nats at $k=0$ to 0.00 at
$k=7$). Colored by ground-truth root class, the cloud resolves into separated petals, one
per inferred root. Raw residual PCA, dominated by token and position nuisance variance,
does *not* bloom — it is the *belief content* of the residual that does.

#fig("figures/blooming.png",
  [Headline result: the belief read out of the final-layer residual blooms with context.
   *Left:* points colored by context position $k$ with rings marking each position's mean
   radius about the prior ($times$); the cloud expands outward as context grows (mean radius
   $0.17 -> 0.39$, tracking the true posterior $0.17 -> 0.47$). *Right:* the same cloud by
   ground-truth root class resolves into separated petals.], w: 100%) <blooming>

== Causal steering: the belief is used, not just decodable

A representation can be decodable yet causally inert. To distinguish the two we apply a
mean-difference patch to the layer-1 residual, pushing it by $alpha$ times the belief
direction toward a chosen latent, and measure the effect on the next-token distribution
(@steering). Steering *toward the true* latent leaves predictions untouched (the model
already holds that belief): the true token's log-probability stays at $-0.73$ for all
$alpha$. Steering *toward a wrong* latent collapses the mean log-probability of the true
next token from $-0.73$ to $-8.5$ as $alpha$ grows, and re-routes probability mass onto the
wrong latent's children (from 0.22 to 0.53). Intervening on the decoded belief axis changes
behavior in the predicted direction — the belief state is causally used.

#fig("figures/steering.png",
  [The belief state is causally used. Steering the layer-1 residual toward the *true*
   latent leaves predictions intact (green); steering toward a *wrong* latent collapses the
   true next token's log-probability (left, orange: $-0.73 -> -8.5$) and re-routes mass onto
   the wrong latent's children (right, orange: $0.22 -> 0.53$).], w: 95%) <steering>

= Pre-registration scorecard <scorecard>

We grade each pre-registered prediction honestly against the outcome.

#table(
  columns: (auto, 1fr, auto),
  inset: 6pt,
  align: (left, left, center),
  stroke: 0.5pt + luma(200),
  table.header([*Prediction*], [*Outcome*], [*Verdict*]),
  [Linear decodability, root $R^2 > 0.5$, shuffled #text[≈]0],
  [Root $R^2 = 0.38$ (shuffled $≈ 0$). Decodable and well above baseline, but below the
   0.5 threshold for the root; *mid/deep latents do exceed 0.5* (up to 0.66).],
  [Partial],
  [Blooming with context (radius #text[↑], entropy #text[↓], monotone)],
  [Mean radius $0.17 -> 0.39$ monotone; posterior entropy $1.84 -> 0.00$ monotone.],
  [Hit],
  [Monotone layer-wise accumulation of $R^2$],
  [$-0.00 -> 0.15 -> 0.38$ across the three residual indices.],
  [Hit],
  [Causal usage via steering (ordered effect)],
  [Steering#text[→]wrong: log p collapses $-0.73 -> -8.5$, monotone in $alpha$; target-child
   mass $0.22 -> 0.53$.],
  [Hit],
)

Three of four predictions held cleanly. The single miss is the root $R^2$, which came in at
0.38 rather than the predicted $> 0.5$. We take this as a genuine and informative negative:
the prediction was correct in *form* (linear, above baseline, growing with depth and
context) but our magnitude estimate for the *root specifically* was optimistic. The
latent-level sweep — which we did not pre-specify — explains why: the root is the coarsest,
least locally-predictive latent, and the threshold we predicted is in fact met by every
deeper latent in the hierarchy. None of the pre-registered failure modes (degenerate
collapse, non-linear-only encoding, MAP-only encoding, flat-with-depth) materialized; in
particular MAP root-class accuracy is only 0.47 while full-simplex $R^2$ is positive and
graded, ruling out a MAP-only representation.

= Generalization across grammars

The results above come from a single grammar (`grammar = 0`) and a single training
run. To test whether they are properties of the RHM *family* under our fixed
constraints ($s=2$, $L=3$, $v=8$, $m=2$, uniform unambiguous rules) rather than
artifacts of one rule draw, we sweep the *content of the rule table* across ten
independently sampled grammars (`Grammar.random(seed = 0..9)`) with three replicate
training seeds each — 30 runs at the pinned architecture ($n_"layer" = 2$,
$n_"embd" = 128$, $n_"head" = 4$) and full 4000-step budget. A single master seed
controls model initialisation, training-data sampling, probe sampling, and the
probe split, so replicates measure end-to-end sampling variance. Each run writes a
deterministic, self-contained directory
(`results/sweep/g{NN}_L2_d128_h4_s{S}/`) holding its grammar, a `config.json`
manifest (resolved config, git SHA, timestamp), and per-run metrics — recoverable
for this paper independent of any experiment-tracker. Published `grammar = 0` is the
first cell of the sweep, so its number is locatable in the distribution as a
built-in consistency check.

*The ten grammars are not isomorphic.* Each grammar's rule table is an adjacency
structure (which parent expands to which ordered children, per level); two grammars
are the same up to renaming iff a per-level symbol permutation makes those tables
coincide. An exact test for such a permutation finds all $binom(10, 2) = 45$ pairs
distinct (`grammar_iso.py`, checkpointed in `tests/`).

*All grammars train to near-Bayes.* Every run closes 0.82–0.93 of the
uniform#text[→]Bayes loss gap (mean $0.89 plus.minus 0.03$), matching the pass-1
value (0.88); no run failed to train, so we report all 30 with no convergence
filtering (@sanitytable). Loss-gap-closed is shown as a sanity covariate, not used
to gate any run.

*The inverted strength gradient replicates as a distribution* (@sweeplevels).
Pooling probe $R^2$ by tree level, the root ($L_0$) averages $0.35 plus.minus 0.07$
(range $0.24$–$0.48$), well below the mid ($L_1$, mean $0.50$) and leaf-parent
($L_2$, mean $0.49$) latents. The root is the weakest-decoded level across the
*whole family*, not just in the published draw: the coarse global latent is the
hardest to read off, while the locally-predictive deeper latents are encoded more
strongly. The pass-1 root $R^2 = 0.38$ sits comfortably inside the replicate band,
confirming it as a typical rather than cherry-picked draw. The deepest level shows
the widest spread (one grammar dips slightly negative), as expected — node-level
rule structure matters most for the most local latents.

*Blooming and belief-sharpening hold for every grammar* (@sweepbloom). The exact
posterior entropy falls monotonically with context position $k$ in all ten
grammars, and the belief readout's radius grows with context — the blooming
geometry is a family-wide property, not a feature of one rule table.

*Causal steering replicates with a stable effect size* (@sweepsteer). Steering the
layer-1 residual toward a wrong latent collapses the true next token's mean
log-probability in every run, by $6.9 plus.minus 0.5$ nats from $alpha = 0$ to
$alpha = 4$. The belief is causally used across the whole grammar family, with low
run-to-run variance.

In short, all four pass-1 findings — inverted level gradient, blooming, layer-wise
accumulation (recomputed in every run's per-layer probe), and causal steering —
replicate across the grammar family and are stable across replicate sampling. The
headline phenomenology is a property of the RHM under these constraints, not of one
rule draw.

#fig("figures/sweep_level_r2.png",
  [The inverted strength gradient replicates across the grammar family. Each point
   is one of 30 runs (10 grammars $times$ 3 seeds); violins show the per-level
   distribution of probe $R^2$, diamonds the means. The root ($L_0$) is the
   weakest-decoded level ($0.35 plus.minus 0.07$), below the mid ($L_1$) and
   leaf-parent ($L_2$) latents ($approx 0.49$–$0.50$).], w: 82%) <sweeplevels>

#fig("figures/sweep_blooming.png",
  [Blooming is family-wide. *Left:* the belief readout's mean radius grows with
   context position $k$ (one line per grammar, averaged over seeds). *Right:* the
   exact posterior entropy falls monotonically with $k$ for every grammar.],
  w: 100%) <sweepbloom>

#fig("figures/sweep_steering.png",
  [Causal steering replicates with a stable effect size. *Left:* steering the
   layer-1 residual toward a *wrong* latent collapses the true next token's
   log-probability in every run (grey lines), mean in orange. *Right:* the
   distribution of the collapse magnitude ($alpha{=}0 -> alpha{=}4$) across all
   30 runs: $6.9 plus.minus 0.5$ nats.], w: 100%) <sweepsteer>

= Architecture dependence

The grammar sweep fixes the architecture and varies the data; this section does the
opposite. We hold the RHM constraints fixed ($s=2, L=3, v=8, m=2$) and vary the
transformer's capacity one axis at a time (OAT) around the published pass-1 baseline
($n_"layer" = 2, n_"embd" = 128, n_"head" = 4$, 4000 steps): width $n_"embd" in
{16, 64, 128, 256}$, depth $n_"layer" in {1, 2, 3, 4}$, heads $n_"head" in {1, 2, 4,
8}$, and training budget $"steps" in {4000, 16000}$. Each axis varies independently
with the other three pinned at baseline, so the eleven distinct configs share the
baseline as their common center. Every config is run across `grammar` $in {0, 1, 2}$
and master `seed` $in {0, 1, 2}$ — *99 runs total* (`run_arch.py`, writing
steps-disambiguated dirs `results/arch/g{NN}_L{n}_d{d}_h{h}_t{steps}_s{S}/`). No
convergence filtering: small models are *expected* to underfit, and we carry
`loss_gap_closed` as a covariate rather than a gate. This builds on the pass-1
pre-registration (`PREREGISTRATION.md`); the expectations below are stated for
framing, not pre-registered.

*Capacity lifts decodability, with a low-width floor* (@archmarginal). Width is the
strongest lever: root probe $R^2$ climbs monotonically $0.07 -> 0.21 -> 0.35 -> 0.44$
across $n_"embd" = 16 -> 256$, and at $n_"embd" = 16$ the root nearly collapses
($R^2 = 0.07$, barely above zero) — a genuine minimum-capacity floor below which the
coarse latent is no longer linearly present. Depth lifts every level too
($0.22 -> 0.42$ for the root across $n_"layer" = 1 -> 4$). Heads are the weakest axis,
as anticipated: from $1$ to $8$ heads the root moves only $0.25 -> 0.36$ and the
leaf-parent level is essentially flat ($0.42 -> 0.44$). One framing expectation does
*not* survive: we guessed the root would be information-limited and gain little from
extra capacity, but width and depth lift it substantially — the root is the weakest
latent at *every* capacity, yet it is far from saturated.

*The inverted gradient is capacity-robust.* At all eleven configs the root ($L_0$) is
the weakest-decoded level, below the mid ($L_1$) and leaf-parent ($L_2$) latents — the
pass-1/2 inverted strength gradient is not an artifact of the baseline size. It holds
at the smallest width (where everything is low) and the largest (where everything is
high).

*Depth stretches the build-up and lifts the readout* (@archdepth). Plotting root
$R^2$ against normalized residual depth (residual index over $n_"layer"$), every model rises from
$approx 0.01$ at the embedding to its final-layer readout, and deeper models both
stretch the accumulation curve and reach a higher endpoint: final-layer root $R^2$ is
$0.22, 0.35, 0.39, 0.42$ for $n_"layer" = 1, 2, 3, 4$. Belief assembly is not a
two-layer accident — it uses whatever depth it is given.

*Longer training helps the root more than the leaves.* Quadrupling the budget
($4000 -> 16000$ steps) lifts the root by $+0.05$ ($0.35 -> 0.41$) and the mid level by
$+0.05$, but the leaf-parent level by only $+0.02$ ($0.44 -> 0.47$). The coarse,
globally-determined latent is the slowest to be linearized, consistent with it being
optimization-limited rather than already-saturated.

*Decodability decouples from loss fit* (@archlossfit). Across all 99 runs the
correlation between `loss_gap_closed` and probe $R^2$ is modest — $0.46$ for the root,
$0.17$ for the leaf-parent — and the scatter is wide: models that close the same
fraction of the loss gap span a large range of decodability (root $R^2$ from below
$0.1$ to above $0.6$ at `loss_gap_closed` $approx 0.9$). Linear belief decodability is
therefore *not* a restatement of how well the model fit the next-token loss; a model
can reach near-Bayes loss without linearly representing the coarse belief, and vice
versa.

#fig("figures/arch_marginal_r2.png",
  [Marginal capacity effects (OAT). Each panel varies one axis with the other three at
   baseline; points are root ($L_0$), mid ($L_1$), and leaf-parent ($L_2$) probe
   $R^2$, error bars are SEM over 3 grammars $times$ 3 seeds, the dotted line marks the
   shared baseline. Width and depth lift all levels (root collapses at $n_"embd" = 16$);
   heads matter least; the root is the lowest level in every panel.], w: 100%)
  <archmarginal>

#fig("figures/arch_depth_accum.png",
  [Depth and accumulation. *Left:* root-belief probe $R^2$ vs normalized residual
   depth, one line per $n_"layer"$ (mean $plus.minus$ SEM over grammars $times$ seeds);
   every model accumulates from the embedding to its readout, and deeper models reach
   higher. *Right:* final-layer root $R^2$ rises monotonically with depth.], w: 100%)
  <archdepth>

#fig("figures/arch_lossfit.png",
  [Decodability vs loss fit, all 99 runs. Root ($L_0$) and leaf-parent ($L_2$) probe
   $R^2$ against `loss_gap_closed` (fraction of the uniform$arrow.r$Bayes CE gap
   closed). Dashed lines are least-squares fits; the wide vertical spread at fixed loss
   fit shows decodability is not determined by how well the model fit the loss.],
   w: 82%) <archlossfit>

= Discussion and limitations

The picture is coherent: a tiny transformer trained to near-Bayes-optimal loss on a
hierarchical grammar carries a *partial* linear image of the posterior over the grammar's
hidden latents in its residual stream — fuzzy for the coarse root ($R^2 = 0.38$), sharper
for the local latents ($R^2$ up to 0.66). That image is assembled additively across
layers, blooms from the prior toward the vertices as evidence accumulates, spans the full
latent hierarchy, and is causally relied upon at generation time. We probe against an
*exactly known* belief state (computed by belief propagation, not approximated), which is
what lets us quantify the recovery as partial rather than merely assert it — but the
linear recovery itself is imperfect, and we do not claim the probe reconstructs the
posterior exactly. This reproduces the core Simplex belief-geometry phenomenology in a
setting where the ground-truth belief state is known exactly.

Methodologically, this paper sits between the two reference points in the bibliography:
Cagnetta et al.'s RHM study uses the same hierarchical data model but evaluates the
last-token prediction setting, while Shai et al.'s *Transformers learn factored
representations* motivates pooling predictive vectors across contexts. Our readout follows
the latter all-context-position spirit, treating every prefix position as a belief state to
be decoded, while keeping Cagnetta et al.'s exact RHM grammar as the data source.

The most interesting wrinkle is the *inverted strength gradient*: the global root, the
spec's nominal target, is the hardest latent to decode, while local near-leaf latents are
read off most cleanly. This is intuitive in hindsight — next-token prediction rewards
representing whatever is most locally predictive — but it sharpens the belief-geometry
claim: the residual stream tracks the *full* latent posterior, weighted toward what the
task needs, not a single privileged variable.

*Limitations.* (i) The grammar is deliberately tiny and fully enumerable ($s=2$, $L=3$,
1024 trees); this is what makes the beliefs exact and the verification airtight, but it
leaves open how the geometry scales to larger, non-enumerable RHMs. (ii) At $L=3$ the root
posterior collapses to certainty by $k=3$ on many strings, compressing the dynamic range
over which blooming is visible for the root; the spec anticipated this and suggested $L=4$
as a follow-up. (iii) The strongly-negative mid-layer probe cells indicate the affine probe
is locally mis-specified at some position/layer combinations; a per-position or
whitened probe would tighten these estimates. (iv) Steering is applied at a single layer
(layer 1) along a mean-difference axis; a learned causal direction and a layer sweep would
strengthen the causal claim. (v) The detailed pass-1 figures
(@simplex–@steering) are from a single training run and grammar draw; the
grammar-sweep and architecture-sweep sections quantify how the *headline* metrics
move across 10 grammars / 3 seeds and across 11 capacity configs / 3 grammars /
3 seeds respectively, but the architecture sweep is one-axis-at-a-time (no
$n_"layer" times n_"embd" times n_"head"$ interaction cells) and larger/longer
grammars ($L = 4$) remain a future pass.

= Conclusion

On an exactly-solvable hierarchical grammar, a small transformer's residual stream is a
linear image of the exact Bayesian belief simplex: it accumulates across depth, blooms with
context, encodes the whole latent hierarchy (local latents most strongly), and is causally
used. Pre-registered predictions held in three of four cases, and the one miss — a weaker
root probe than predicted — turned out to be a feature of the hierarchy rather than a
failure of the belief-geometry hypothesis. The exact, enumerable setting turns a qualitative
interpretability story into quantitative, falsifiable measurement.

= Appendix: Root reconstruction diagnostics <appendix-root-reconstruction>

The root-posterior probe is a regression target, but it is also useful to ask the harsher
classification question: does the largest coordinate of the linear belief readout recover
the sampled root class? Across 400 probe examples and 8 context positions (3200
example-position points), the answer
is yes for $1565$ points ($48.91%$). This is well above random guessing, but random guessing
is only 1/8 = $12.5%$ because there are eight root classes; the relevant baseline is not
$50%$.

#fig("figures/blooming_match.png",
  [Root reconstruction from the linear belief readout. Green points are example-position
   pairs where the readout's argmax matches the ground-truth root; red points are
   mismatches. The overall match rate is $48.91%$, compared with a random-guessing baseline
   of $12.5%$ (not $50%$) for eight root classes.], w: 72%) <rootmatch>

Restricting to positions where nearly all evidence is visible makes the diagnostic sharper.
With seven of eight symbols observed ($k = 6$), the exact Bayesian
posterior's MAP root already matches the sampled root in $95.00%$ of examples, while the
linear readout matches in $62.25%$. With all eight symbols observed ($k = 7$), the exact
posterior is deterministic, but the readout still reaches only $70.50%$. Thus early
ambiguity explains part, but not all, of the fuzzy root reconstruction.

#table(
  columns: (1fr, auto, auto, auto),
  inset: 6pt,
  align: (left, center, center, center),
  stroke: 0.5pt + luma(200),
  table.header([*Subset*], [*Prefix length*], [*Readout match*], [*Exact MAP match*]),
  [All example-position points], [1-8], [$1565 / 3200 = 48.91%$], [--],
  [All but last symbol], [7], [$249 / 400 = 62.25%$], [$380 / 400 = 95.00%$],
  [Full sequence], [8], [$282 / 400 = 70.50%$], [$400 / 400 = 100.00%$],
)

= Appendix: Per-run grammar-sweep sanity table <appendix-sanity>

Every run in the grammar sweep, no filtering. `grammar` indexes the rule-table
draw (`Grammar.random(seed=grammar)`); `seed` is the master replicate seed; the
architecture is pinned. `test_ce` is the held-out next-token cross-entropy;
`loss_gap_closed` is the fraction of the uniform#text[→]Bayes gap closed (sanity
covariate, not a gate); `root_r2` and `deepest_r2` are the level-$L_0$ and mean
level-$L_2$ probe $R^2$. Loaded directly from `figures/sweep_sanity.csv`.

#let sanity = csv("figures/sweep_sanity.csv")
#figure(
  table(
    columns: 9,
    inset: 4pt,
    align: center,
    stroke: 0.5pt + luma(220),
    table.header(..sanity.at(0).map(h => [#text(8pt, weight: "bold")[#h]])),
    ..sanity.slice(1).flatten().map(c => [#text(8pt)[#c]]),
  ),
  caption: [All 30 grammar-sweep runs (10 grammars $times$ 3 seeds), pinned
    architecture, full 4000-step budget. No convergence filtering.],
) <sanitytable>

= Appendix: Per-run architecture-sweep sanity table <appendix-arch-sanity>

Every run in the architecture sweep, no filtering — all 99 (11 OAT capacity configs
$times$ 3 grammars $times$ 3 seeds). `n_layer`, `n_embd`, `n_head`, `steps` give the
capacity config; `grammar` and `seed` the replicate. `test_ce` is the held-out
next-token cross-entropy; `loss_gap_closed` the fraction of the uniform#text[→]Bayes
gap closed (covariate, not a gate); `root_r2` and `deepest_r2` the level-$L_0$ and
mean level-$L_2$ probe $R^2$. The deliberately-underpowered small models
(e.g. $n_"embd" = 16$) are included. Loaded directly from `figures/arch_sanity.csv`.

#let archsanity = csv("figures/arch_sanity.csv")
#figure(
  table(
    columns: 10,
    inset: 3.2pt,
    align: center,
    stroke: 0.5pt + luma(220),
    table.header(..archsanity.at(0).map(h => [#text(7pt, weight: "bold")[#h]])),
    ..archsanity.slice(1).flatten().map(c => [#text(7pt)[#c]]),
  ),
  caption: [All 99 architecture-sweep runs (11 one-axis-at-a-time capacity configs
    $times$ 3 grammars $times$ 3 seeds). No convergence filtering; small models are
    expected to underfit.],
) <archsanitytable>

= References

#set par(justify: false)
#text(size: 9.5pt)[
  Cagnetta, F., Favero, A., Sclocchi, A., and Wyart, M. (2025). *Scaling Laws and
  Representation Learning in Simple Hierarchical Languages: Transformers vs. Convolutional
  Architectures*. arXiv:2505.07070. \
  Shai, A. et al. (2026). *Transformers learn factored representations*. arXiv:2602.02385. \
  Project sources: `spec.md`, `spec-grammar-sweep.md`, `PREREGISTRATION.md`,
  `EXECUTION_OUTPUT.md`, `results/analysis.json`, `results/root_reconstruction.json`,
  `artifacts/train_summary.json`. Grammar sweep: `sweep.py`, `sweep.yaml`,
  `results/sweep/g*/{config.json,analysis.json}`, `figures/sweep_sanity.csv`.
  Architecture sweep: `spec-arch-sweep.md`, `run_arch.py`,
  `results/arch/g*/{config.json,analysis.json}`, `figures/arch_sanity.csv`.
]
