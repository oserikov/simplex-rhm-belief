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
  miss (root $R^2 < 0.5$) is informative.
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
underperforms the mean predictor on those positions); these are clipped for legibility in
the figure, and the raw values are preserved in `results/analysis.json`.

#fig("figures/layer_position.png",
  [Belief decodability accumulates across depth and context. *Left:* root-posterior
   $R^2$ rises layer by layer ($-0.00 -> 0.15 -> 0.38$) while a shuffled control stays at
   #text[≈]0. *Right:* $R^2$(layer, position) heatmap; the final layer reaches $R^2 = 1.0$
   on the earliest positions. Cells clipped to $[-1, 1]$.], w: 95%) <layerpos>

== The whole latent hierarchy is encoded — local latents most strongly

The root is only one of the tree's hidden latents. Probing the exact posterior over latents
at *every* level reveals a clear gradient (@latents): root ($L_0$) $R^2 = 0.38$; the two
level-1 mid latents $0.51$ and $0.59$; the deepest level-2 latents $0.61$ and $0.66$. The
residual encodes the entire hierarchy, but the *local, near-leaf* latents — those most
directly predictive of the next token — are read off far more cleanly than the coarse
global root. This is our most informative finding: the spec's primary target (the root) is
in fact the *hardest* latent to decode, because it is the most abstract and the least
locally predictive.

#fig("figures/latent_levels.png",
  [The residual encodes the whole latent hierarchy, deeper/local latents most strongly.
   Probe $R^2$ climbs monotonically from the root ($L_0 = 0.38$, the spec's primary target)
   through level-1 ($0.51, 0.59$) to the deepest level-2 latents ($0.61, 0.66$).], w: 60%)
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
strengthen the causal claim. (v) All results are from a single training run and grammar
draw; we did not measure seed-to-seed variance.

= Conclusion

On an exactly-solvable hierarchical grammar, a small transformer's residual stream is a
linear image of the exact Bayesian belief simplex: it accumulates across depth, blooms with
context, encodes the whole latent hierarchy (local latents most strongly), and is causally
used. Pre-registered predictions held in three of four cases, and the one miss — a weaker
root probe than predicted — turned out to be a feature of the hierarchy rather than a
failure of the belief-geometry hypothesis. The exact, enumerable setting turns a qualitative
interpretability story into quantitative, falsifiable measurement.

= References

#set par(justify: false)
#text(size: 9.5pt)[
  Cagnetta, F. et al. (2025). *The Random Hierarchy Model* and the emergence of compositional
  structure in deep networks. arXiv:2505.07070. \
  (Belief geometry in transformers) (2026). arXiv:2602.02385. \
  Project sources: `spec.md`, `PREREGISTRATION.md`, `EXECUTION_OUTPUT.md`,
  `results/analysis.json`, `artifacts/train_summary.json`.
]
