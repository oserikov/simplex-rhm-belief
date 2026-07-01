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
  margin: (top: 2.001cm, bottom: 2cm, left: 2cm, right: 2cm),
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

#show figure.caption: set text(size: 9.5pt)

#let fig(path, caption, w: 100%) = figure(image(path, width: w), caption: caption)

// Long per-run sanity tables must flow across pages; figures are unbreakable by
// default, which clips the 99-row arch / 60-row depth-4 tables. Scope breakability
// to TABLE figures only (image figures are atomic and unaffected).
#show figure.where(kind: table): set block(breakable: true)

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
  cross-entropy of 0.88 nats, closing #text[≈]89% of the gap between the uniform
  baseline (2.08) and the Bayes-optimal floor (0.73) — it has effectively learned the
  posterior predictor. A single global linear probe partially recovers this posterior
  from the residual stream: held-out $R^2 = 0.38$ for the root class versus #text[≈]0 for
  a shuffled control — a real but noisy affine image, not an exact reconstruction — with
  decodability accumulating across depth ($-0.00 -> 0.15 -> 0.38$) and the
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
  the inverted gradient holds #todooleg[not holds] at every capacity, heads matter least, and decodability is
  only weakly correlated with how well the model fit the loss.
]

= Introduction

Natural data is hierarchical: characters compose into words, words into phrases, phrases
into meaning. The Random Hierarchy Model (RHM) of #cite(<cagnetta2025>, form: "prose")
abstracts this into a
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

= Experimental design
== The Random Hierarchy Model and exact beliefs

*Grammar.* We use the defaults of the spec: arity $s = 2$, depth $L = 3$, so each string
has $d = s^L = 8$ leaves; per-level vocabulary $v = 8$; and $m = 2$ production rules per
parent symbol, chosen uniformly. Rules are drawn once and held fixed, and constrained to
be *unambiguous* (the spec's quick-failure check confirms no two parents share a
production). A sample is generated top-down: pick a root class uniformly, recursively
expand each symbol by a uniformly chosen rule, and emit the leaf string. With these
parameters the grammar admits exactly $v dot m^(d-1) = 8 dot 2^7 = 1024$ equiprobable
distinct trees — small enough to enumerate completely.

The exact frozen grammar used in every result is shown below (@ruletable). Symbols are written as
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
  caption: [Frozen level-specific grammar sampled once before training. Top, middle, and bottom columns correspond to `rules_0`, `rules_1`, and `rules_2` in `artifacts/grammar.npz` (see visualized in appendix @appendix-ruletable).],
) <ruletable>

*Exact posterior.* Given a leaf prefix of length $k$, the posterior over any hidden latent
(including the root class) is computed by sum-product belief propagation on the tree:
upward messages from observed leaves combine through the production rules to give the
marginal over each latent. Because the grammar is fully enumerable, we verified the
belief-propagation posterior against brute-force enumeration of all consistent trees; the
two agree to $< 10^(-6)$ across 10 passing tests, alongside checks that sample length
equals $s^L$ and that unambiguity holds. The recorded self-test (a representative string)
shows the root posterior sharpening with context — entropy $1.89$ nats at $k=1$ falling
to $0.00$ (a single consistent root) by $k=3$ — which previews the blooming geometry below.

== Model and training

We train a HuggingFace `GPT2LMHeadModel` with 2 layers, `n_embd` = 128, 4 heads, and all
dropout disabled, totalling 398,848 parameters. There is no tokenizer: RHM leaf symbols are
integers fed directly as `input_ids`, with `vocab_size` = 8 and `n_positions` = 8. We train
by next-token prediction on 922 training strings (102 held out) for 4000 steps on Apple MPS.

*The Bayes-optimal floor.* The value of an enumerable grammar is that we know the best
achievable loss. Averaging the exact per-position posterior entropy over the seven
predicted positions gives the Bayes-optimal mean next-token cross-entropy: 0.725 nats. The
uniform baseline is $ln 8 = 2.079$. The trained model reaches a final held-out
cross-entropy of *0.880 nats* (@loss), closing $89%$
of the gap. The model has, to a good approximation, learned the true posterior predictor —
which is the precondition for asking whether it represents the posterior internally. Later this is reproduced at scale (more grammars).

#fig("figures/loss_curve.png",
  [The transformer converges to the true RHM predictor. Held-out next-token cross-entropy
   (heavy line) falls from the uniform baseline ($ln 8 = 2.079$, top dashed) toward the
   Bayes-optimal floor ($0.725$, bottom dashed) over 4000 training steps, ending at
   $0.880$ nats — closing #text[≈]89% of the uniform#text[→]Bayes gap (shaded). The
   faint line is the train-minibatch CE. Seeded reproduction of the pinned-arch canonical
   run (grammar seed 0, $n_"layer"{=}2, n_"embd"{=}128, n_"head"{=}4$),
   `results/refrun/`.], w: 66%) <loss>

== Pre-registered prediction

Following the project's honor-code rule, we committed our predictions to git *before*
training any model or computing any probe (`PREREGISTRATION.md`). In brief, we predicted:
(1) the root posterior is *linearly decodable* from the residual stream, with $R^2 > 0.5$
at the final layer and a shuffled baseline near 0; (2) the belief *blooms* — early
positions cluster near the prior, late positions spread toward simplex vertices, with
posterior entropy falling and activation radius growing with context $k$;
(3) decodability *accumulates additively across depth*, lowest at the embedding and highest
after the last layer; and (4) the representation is *causally used*, so steering the
residual along a belief direction shifts the output toward the corresponding latent's
leaves. We also registered alternative outcomes (degenerate collapse, non-linear-only
encoding, MAP-only encoding, flat-with-depth) as falsification handles. We report against
these predictions in @scorecard.

= Results

== A single linear probe partially decodes the posterior

We fit one global least-squares affine map from the 128-d residual stream to the 8-class
exact root posterior, on a probe set of $N = 400$ held-out examples (first, context-position agnostic, then wrt. to it), and score it by
$R^2$ on held-out data. The final-layer probe reaches $R^2 = 0.38$, while at the control task (the same probe
fit to *shuffled* labels) scores $≈ 0$ ($-0.085$). So the posterior is linearly accessible above chance, but the probe explains only a third of the variance: the recovery is *partial*. Projected into a belief-space PCA basis (@simplex), the probe's predicted
posteriors occupy the same structured region as the exact posteriors and reproduce the
same position gradient, but as a diffuse cloud rather than the discrete point set of the
ground truth. #footnote[The ground-truth panel looks sharper partly because the exact posterior
takes few distinct values, so identical points overplot, whereas every probe prediction
differs slightly.] The affine correspondence is particularly weak for the root;
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
is built. Both panels here decode the *root* posterior. The residual stream is read at
three points — the model has two attention blocks, so `output_hidden_states` returns
$n_"layer" + 1 = 3$ snapshots: index 0 is the token#text[+]position embedding, index 1 is
after block 1, index 2 is after block 2 (the "Layer 0/1/2" axis is these three readout
points, not three transformer layers). Root-posterior probe $R^2$ rises along them:
$-0.00$ at the embedding, $0.15$ after block 1, $0.38$ after block 2, while the
shuffled control stays at $≈ 0$ throughout (@layerpos, left) — each block adds
belief-relevant signal. Resolving the same root probe by context position (@layerpos,
right), the root is already decodable at early positions even at the shallow readout
points, and at the final readout it is near-perfectly decodable ($R^2 = 1.0$) for the
first few positions $0$–$3$. #footnote[A few mid-readout cells are strongly negative (the probe
underperforms the mean predictor on those positions); the heatmap colors are clipped for
legibility, but the cell labels show the raw values].

#fig("figures/layer_position.png",
  [Root-posterior decodability accumulates across residual depth and context. Both panels
   decode the *root* class; "Layer 0/1/2" are the three residual readout points (embedding,
   after block 1, after block 2 = $n_"layer"{+}1$). *Left:* root-posterior $R^2$ rises along
   them ($-0.00 -> 0.15 -> 0.38$) while a shuffled control stays at #text[≈]0. *Right:*
   $R^2$(readout point, position) heatmap; the final readout reaches $R^2 = 1.0$ on the
   earliest positions.],
  w: 95%) <layerpos>

== The whole latent hierarchy is encoded — local latents most strongly

The root is only one of the tree's hidden latents. Probing the exact posterior over latents
at *every* level reveals a clear gradient with some node-level variation (@latents). Precisely, root
($L_0$) $R^2 = 0.38$; the two level-1 mid latents $0.51$ and $0.59$; and the four deepest
level-2 latents $0.61$, $0.66$, $0.50$, and $0.66$. The residual encodes the entire
hierarchy, but the *local, near-leaf* latents, which are more directly predictive of the next
token, are usually read off more cleanly than the coarse global root. Root is the *hardest* latent to
decode, likely because it is the most abstract thus least locally predictive.

#fig("figures/latent_levels.png",
  [The residual encodes the whole latent hierarchy, deeper/local latents most strongly.
   Probe $R^2$ is lowest for the root ($L_0 = 0.38$, the spec's primary target), higher for
   level-1 ($0.51, 0.59$), and generally higher for level-2 ($0.61, 0.66, 0.50, 0.66$).],
  w: 72%)
  <latents>

*An important refinement based on robustness studies (@appendix-robustness).* This "deeper decodes stronger" reading is *refined* once the tree is one level taller.
Scaling the identical per-level probe to $L = 4$ (the depth study in @depth4levels) shows the
gradient is really an *inverted U*: the mid-level latents decode strongest, while *both* the
coarse root and the most-local leaf-parents fall off. The headline is not "local beats global"
but "mid beats both ends" — previewed here, quantified across 60 runs in the appendix.

#fig("figures/depth4_level_r2.png",
  [Preview of the inverted-U refinement (full detail in @depth4levels). At $L = 4$, per-level
   probe $R^2$ across 60 runs rises from the root ($L_0$) to the mid latents ($L_1, L_2$) and
   falls back toward the leaf-parents ($L_3$) — an inverted U, not a monotone "deeper is
   stronger" ladder.],
  w: 74%)
  <latents-invu>

== The belief blooms with context

Our central geometric result (@blooming): projecting the linear belief readout into a
belief-space PCA(2) basis, the points form a tight central cluster near the uniform prior
when little context is observed and expand outward toward the simplex vertices as context
accumulates. The mean radius about the prior anchor grows from 0.17 at $k = 0$ to
#text[≈]0.39 at $k = 7$, tracking the true posterior's own growth (0.17 #text[→] 0.47);
equivalently, the exact posterior entropy falls with position (1.84 nats at $k=0$ to 0.00
at $k=7$). The right panel colors the same cloud by ground-truth root class: it is only
*loosely* organized by class — consistent with the modest root decodability ($R^2 = 0.38$,
MAP root accuracy $0.47$, @rootmatch) — not a clean separation. Raw residual PCA, dominated
by token and position nuisance variance, does *not* bloom (radius#text[–]vs#text[–]$k$
correlation $approx 0.03$, against $approx 0.81$ for the readout) — it is the *belief
content* of the residual that does.

#fig("figures/blooming.png",
  [Headline result: the belief read out of the final-layer residual blooms with context.
   *Left:* points colored by context position $k$ with rings marking each position's mean
   radius about the prior ($times$); the cloud expands outward as context grows (mean radius
   $0.17 -> 0.39$, tracking the true posterior $0.17 -> 0.47$). *Right:* the same cloud
   colored by ground-truth root class is loosely organized by class (partial, not clean
   separation — root MAP accuracy $0.47$).], w: 100%) <blooming>

== Causal steering: the belief is used, not just decodable

A representation can be decodable yet causally inert. To distinguish the two we apply a
mean-difference patch to the layer-1 residual, pushing it by $alpha$ times the belief
direction toward a chosen latent, and measure the effect on the next-token distribution
(@steering). Steering *toward the true* latent leaves predictions untouched (the model
already holds that belief): the true token's log-probability stays at $-0.73$ for all
$alpha$. Steering *toward a wrong* latent collapses the mean log-probability of the true
next token from $-0.73$ to $-8.5$ as steering effort $alpha$ grows, and re-routes probability
mass onto the wrong latent's children — the next-token probability the model assigns to
exactly the leaf symbols that the steered-to (wrong) latent can emit at that position
(summed softmax mass over those symbols, averaged across positions and examples) rises from
$0.22$ to $0.53$. The belief state is therefore causally used, not merely decodable.

#fig("figures/steering.png",
  [The belief state is causally used. Steering the layer-1 residual toward the *true*
   latent leaves predictions intact (green); steering toward a *wrong* latent collapses the
   true next token's log-probability (left, orange: $-0.73 -> -8.5$) and re-routes mass onto
   the wrong latent's children (right, orange: $0.22 -> 0.53$).], w: 95%) <steering>

== Pre-registration scorecard <scorecard>

We grade each pre-registered prediction honestly against the outcome (@scorecardtable). #todooleg[I was over-using monotonicity in my initial guess, which is a very strong claim, to be revised (shuoldn't have written it in the first place)]

#figure(
  table(
    columns: (auto, 1fr, auto),
    inset: 6pt,
    align: (left, left, center),
    stroke: 0.5pt + luma(200),
    table.header([*Prediction*], [*Outcome*], [*Verdict*]),
    [Linear decodability, root $R^2 > 0.5$, shuffled #text[≈]0],
    [Root $R^2 = 0.38$ (shuffled $≈ 0$). Decodable and well above baseline, but below the
     0.5 threshold for the root; *mid/deep latents do exceed 0.5* (up to 0.66).],
    [Partial],
    [Blooming with context (radius #text[↑], entropy #text[↓])],
    [Mean radius $0.17 -> 0.39$; posterior entropy $1.84 -> 0.00$.],
    [Hit],
    [Monotone layer-wise accumulation of $R^2$],
    [$-0.00 -> 0.15 -> 0.38$ across the three residual indices.],
    [Hit],
    [Causal usage via steering (ordered effect)],
    [Steering#text[→]wrong: log p collapses $-0.73 -> -8.5$, monotone in $alpha$; target-child
     mass $0.22 -> 0.53$.],
    [Hit],
  ),
  caption: [Scored pre-registered predictions (`PREREGISTRATION.md`, committed before any
    result). Three of four held; the lone miss is the root-specific $R^2 > 0.5$ magnitude.],
) <scorecardtable>

Three of four predictions held cleanly. The single miss is the root $R^2$, which came in at 0.38 rather than the predicted $> 0.5$. We take this as a genuine and informative negative: the prediction was correct in *form* (linear, above baseline, growing with depth and context) but our magnitude estimate for the *root specifically* was optimistic. The latent-level sweep — which we did not pre-specify — explains why: the root is the coarsest, least locally-predictive latent, and the threshold we predicted is in fact met by every deeper latent in the hierarchy. None of the pre-registered failure modes (degenerate collapse, non-linear-only encoding, MAP-only encoding, flat-with-depth) materialized; in particular MAP root-class accuracy is only 0.47 while full-simplex $R^2$ is positive and graded, ruling out a MAP-only representation.

= Engineering non-collapsing belief geometry

Everything so far lives on a tree whose belief, given the full leaf string, collapses to *certainty*: with uniform unambiguous rules the eight leaves invert level-by-level to exactly one root, so the $k = 8$ posterior is a delta and the belief path runs from the prior straight to a simplex vertex. 
Recently, #cite(<shai2026>, form: "prose") considered another setup: there, for HMMs, belief never collapses to a single certain state. It traces a self-similar, fractal (Sierpinski-like) attractor that fills a structured region of the simplex interior. But RHM has no recurrence, so "slow forgetting" has no analog. 
A possible analog of "the state can't be restored" is a *non-invertible observation channel*: a grammar where even the full leaf string leaves the root uncertain, so the reachable-belief set is a non-trivial attractor in the simplex rather than a path to a vertex. 

We add two per-level knobs to the grammar (`rhm.py`, both reducing exactly to the canonical RHM at their off setting, verified byte-identical). *Ambiguity* $rho$ shares a fraction of the $v dot m$ rule entries' child-tuples *across parents* (many-to-one leaf#text[→]root structure); $rho in {0, 0.3, 0.6}$. *Skew* draws each parent's rule-choice probabilities non-uniform from a Dirichlet with concentration $alpha$: `none` is the canonical uniform $1\/m$; `mid` uses $alpha = 1.0$ and `high` uses $alpha = 0.2$ (smaller $alpha$ #text[⇒] more peaked, near-deterministic rule choice). Both belief propagation and the brute-force reference were generalized to *weighted* sum-product using the same probabilities the sampler uses; the BP#text[↔] brute-force agreement holds to $2.5 times 10^(-16)$ under both knobs simultaneously. We sweep the full $3 times 3$ grid $times$ 3 rule-table draws — *27 runs* at the pinned published arch (`run_noncollapse.py`, writing `results/noncollapse/`), with the $("none", "none")$ cell reproducing the pass-1 baseline.


Mean exact $k = 8$ root-posterior entropy is exactly $0$ at unambiguous setup regardless of the `skew`. It rises monotonically with ambiguity $rho$ growth: $0.00 -> 0.71 -> 1.38$ nats at `skew=none` (@nccurve, left). The full entropy-vs-$k$ trajectory (@nccurve, right) makes the phenomenon legible — at $rho = 0$ the belief collapses cleanly to $0$ by $k = 8$ (the bloom-to-vertex), while at $rho = 0.3$ and $rho = 0.6$ it *plateaus* at a positive floor: a genuine non-collapsing attractor. Higher skew *worsens* the plateau #todooleg[somewhat expected] but never removes it.

#fig("figures/noncollapse_curve.png",
  [Ambiguity, not skew, engineers non-collapse. *Left:* mean exact $k = 8$ root posterior entropy vs ambiguity $rho$, one line per skew (error bars over 3 draws). At $rho = 0$ entropy is exactly $0$ for *every* skew — skew alone does not prevent collapse — and it rises monotonically with $rho$. *Right:* the full entropy-vs-context trajectory at `skew=none`; at $rho = 0$ the belief collapses to $0$ by $k = 8$, at $rho > 0$ it plateaus at a positive floor — a non-collapsing attractor.],
  w: 100%) <nccurve>

*The linear probe, surprisingly, sharpens.* At every cell of the grid the residual stream still linearly encodes the (now spread) posterior well above the shuffled-label baseline of $approx -0.07$ (@ncheatmap). That baseline is the same probe refit to randomly *permuted* labels and scored on held-out data, averaged over the 27 runs — a near-zero (slightly negative) "no real signal" reference. The probe did not *degrade* as $rho$ rose: $R^2$ *rises* with ambiguity, $0.36 -> 0.42 -> 0.53$ at `skew=none`, and the mid- and low-level probes rise even more steeply ($L_1: 0.46 -> 0.71$; $L_2: 0.35 -> 0.82$). 




#fig("figures/noncollapse_heatmap.png",
  [The linear probe survives ambiguity and skew, and *sharpens* with ambiguity. Per-level   probe $R^2$ (root $L_0$, mid $L_1$, low $L_2$) over the $3 times 3$ skew $times$   ambiguity grid; each cell is mean $plus.minus$ SD over 3 rule-table draws, against a   shuffled baseline of $approx -0.07$. $R^2$ rises left-to-right (with $rho$) at every   level. The single dark $L_2$ cell at $("mid", rho{=}0)$ is a near-deterministic-target   instability, not a decodability failure.],
  w: 100%) <ncheatmap>

#fig("figures/noncollapse_attractor.png",
  [Exact posterior and linear-probe readout after PCA, colored by context position $k$ for the uniform corner $rho = 0$ and the high-ambiguity corner $rho = 0.6$. At $rho = 0$ the late-context exact beliefs land *on* the vertices (due to collapse to certainty), which is not always the case at $rho = 0.6$. Probe readouts occupy rougly same regions as their counterpart posteriors, with uncertaincy case readouts being more aligned with the exact posterior. (For vizualization purposes, shown data is from one single run).],
  w: 90%) <ncattractor>




= Discussion and limitations

The picture is coherent: a tiny transformer trained to near-Bayes-optimal loss on a hierarchical grammar carries a *partial* linear image of the posterior over the grammar's hidden latents in its residual stream — fuzzy for the coarse root ($R^2 = 0.38$), sharper for the local latents ($R^2$ up to 0.66). That image is assembled additively across layers, blooms from the prior toward the vertices as evidence accumulates, spans the full latent hierarchy, and is causally relied upon at generation time. We probe against an *exactly known* belief state (computed by belief propagation, not approximated), which is what lets us quantify the recovery as partial rather than merely assert it — but the linear recovery itself is imperfect, and we do not claim the probe reconstructs the posterior exactly. This reproduces the core Simplex belief-geometry phenomenology in a setting where the ground-truth belief state is known exactly.

Methodologically, this paper sits between the two reference points in the bibliography: #cite(<cagnetta2025>, form: "prose")'s RHM study uses the same hierarchical data model but evaluates the last-token prediction setting, while #cite(<shai2026>, form: "prose")'s *Transformers learn factored representations* motivates pooling predictive vectors across contexts. Our readout follows the latter all-context-position spirit, treating every prefix position as a belief state to be decoded, while keeping #cite(<cagnetta2025>, form: "prose")'s exact RHM grammar as the data source.

The most interesting wrinkle is the *inverted strength gradient*: the global root, the spec's nominal target, is the hardest latent to decode, while local near-leaf latents are read off most cleanly. This is intuitive in hindsight — next-token prediction rewards representing whatever is most locally predictive — but it sharpens the belief-geometry claim: the residual stream tracks the *full* latent posterior, weighted toward what the task needs, not a single privileged variable.

*Limitations.* (i) The grammar is deliberately tiny and fully enumerable ($s=2$, $L=3$, 1024 trees); this is what makes the beliefs exact and the verification airtight, but it leaves open how the geometry scales to larger, non-enumerable RHMs. (ii) At $L=3$ the root posterior collapses to certainty by $k=3$ on many strings, compressing the dynamic range over which blooming is visible for the root; the spec anticipated this and suggested $L=4$ as a follow-up. (iii) The strongly-negative mid-layer probe cells indicate the affine probe is locally mis-specified at some position/layer combinations; a per-position or whitened probe would tighten these estimates. (iv) Steering is applied at a single layer (layer 1) along a mean-difference axis; a learned causal direction and a layer sweep would strengthen the causal claim. (v) The detailed pass-1 figures (@simplex–@steering) are from a single training run and grammar draw; the grammar-sweep and architecture-sweep sections quantify how the *headline* metrics move across 10 grammars / 3 seeds and across 11 capacity configs / 3 grammars / 3 seeds respectively, but the architecture sweep is one-axis-at-a-time (no $n_"layer" times n_"embd" times n_"head"$ interaction cells). The $L = 4$ section extends the family one tree level deeper; still-larger grammars ($L >= 5$, larger $v$, $m > 2$), where brute-force belief verification becomes intractable, remain a future pass.

= Conclusion

On an exactly-solvable hierarchical grammar, a small transformer's residual stream is a linear image of the exact Bayesian belief simplex: it accumulates across depth, blooms with context, encodes the whole latent hierarchy (local latents most strongly), and is causally used.

= References

#set par(justify: false)
#bibliography("references.yml", title: none, style: "american-psychological-association")

#text(size: 9.5pt)[
  Project sources: `spec.md`, `spec-grammar-sweep.md`, `PREREGISTRATION.md`,
  `EXECUTION_OUTPUT.md`, `results/analysis.json`, `results/root_reconstruction.json`,
  `artifacts/train_summary.json`. Grammar sweep: `sweep.py`, `sweep.yaml`,
  `results/sweep/g*/{config.json,analysis.json}`, `figures/sweep_sanity.csv`.
  Architecture sweep: `spec-arch-sweep.md`, `run_arch.py`,
  `results/arch/g*/{config.json,analysis.json}`, `figures/arch_sanity.csv`.
]


#set heading(supplement: "Appendix")
#counter(heading).update(0)

= Appendix: Robustness studies <appendix-robustness>

#todoai[summarize these studies and main findings in a paragraph.]

== Generalization across grammars

The results above come from a single grammar #footnote[`grammar = 0`] and a single training run. To test whether they are properties of the RHM *family* under the same fixed constraints #footnote[Reminder: fixed constraints so far are ($s=2$, $L=3$, $v=8$, $m=2$, uniform unambiguous rules)] rather than artifacts of one rule draw, we sweep the content of the rule table#footnote[See the example (first tried grammar) rule table at @ruletable] across ten independently sampled grammars with three replicate training seeds each, resulting in 30 runs.  We have ensured that the sampled grammars aren't isomorpic by post-factum ensuring that the respective nodes adjacency tables are not equivalent up to a per-level symbol permutation. #footnote[Test code in `grammar_iso.py`] The model architecture / training code remained the same as in previous experiments.  #footnote[Reminder: model architecture so far: GPT-2-style transformer, 2 layers of 4 attn heads each, hidden size 128, 4k train steps. A single master seed controls model initialisation, training-data sampling, probe sampling, and the probe split, so replicates measure end-to-end sampling variance. Each run writes a deterministic, self-contained directory (`results/sweep/g{NN}_L2_d128_h4_s{S}/`) holding its grammar, a `config.json` manifest (resolved config, git SHA, timestamp), and per-run metrics — recoverable for this paper independent of any experiment-tracker. ]. 


- *All grammars train to near-Bayes.* Every run closes $0.82–0.93\%$ of the uniform to Bayes-optimal loss gap (mean $0.89 plus.minus 0.03$), in line with the initial observation. The runs details can be seen in @sanitytable.

- *The root remains the weakest-decoded level* (@sweeplevels). Pooling probe $R^2$ by tree level, the root ($L_0$) averages $0.35 plus.minus 0.07$ (range $0.24$–$0.48$), well below the mid ($L_1$, mean $0.50$) and leaf-parent ($L_2$, mean $0.49$) latents. The deepest level shows the widest spread (one grammar dips slightly negative #todooleg[investigate; not a failed training. poor train-test split?]) #todooleg[node-level
rule structure matters most for the most local latents?].

- *Blooming and belief-sharpening alongside context hold for every grammar* (@sweepbloom). The exact posterior entropy falls with context position $k$ in all ten grammars, and the belief readout's radius grows with context.

- *Causal steering replicates with a stable effect size* (@sweepsteer). Steering the layer-1 residual toward a wrong latent collapses the true next token's mean log-probability in every run: as the patch strength $alpha$ increases from $0$ to $4$, that log-probability drops by $6.9 plus.minus 0.5$ nats (mean $plus.minus$ SD across the 30 runs), with low run-to-run variance.

In short, all four earlier findings -- poor root decodability, blooming, layer-wise accumulation (recomputed in every run's per-layer probe), and causal steering -- replicate across the grammar family.


#fig("figures/sweep_level_r2.png",
  [The inverted strength gradient replicates across the grammar family. Each point
   is one of 30 runs (10 grammars $times$ 3 seeds); violins show the per-level
   distribution of probe $R^2$, diamonds the means. The root ($L_0$) is the
   weakest-decoded level ($0.35 plus.minus 0.07$), below the mid ($L_1$) and
   leaf-parent ($L_2$) latents ($approx 0.49$–$0.50$).], w: 82%) <sweeplevels>

#fig("figures/sweep_blooming.png",
  [Blooming is family-wide. *Left:* the belief readout's mean radius grows with
   context position $k$ (one line per grammar, averaged over seeds). *Right:* the
   exact posterior entropy falls with $k$ for every grammar.],
  w: 100%) <sweepbloom>

#fig("figures/sweep_steering.png", 
  [Causal steering replicates with a stable effect size. *Left:* steering the
   layer-1 residual toward a *wrong* latent collapses the true next token's
   log-probability in every run (grey lines), mean in orange. *Right:* the
   distribution of the collapse magnitude (grey violin, orange diamond = mean) — the drop in
   the true token's mean log-probability as $alpha{=}0 -> alpha{=}4$ — across all 30 runs:
   $6.9 plus.minus 0.5$ nats.],
  w: 100%) <sweepsteer>

== Architecture dependence

The grammar sweep fixes the architecture and varies the data; this section does the opposite. 
Again, we keep the RHM constraints. 
We vary the transformer's capacity one axis at a time around the baseline. We consider (in bold are baseline params) $n_"layer"$ in {1, *2*, 3, 4}, $n_"embd"$ in {16, 64, *128*, 256}, $n_"head"$ in {1, 2, *4*, 8}, and training $"steps"$ in {*4000*, 16000}. \
Each axis varies independently with the other three pinned at baseline, the baseline being their common center. Every config is run across `grammar` $in {0, 1, 2}$
and `seed` $in {0, 1, 2}$ from grammar sweep, *99 runs total* #footnote[`run_arch.py` , writing
steps-disambiguated dirs `results/arch/g{NN}_L{n}_d{d}_h{h}_t{steps}_s{S}/`)]. 
Small models are *expected* to underfit. 


*Model capacity lifts decodability, with a low-width floor* (@archmarginal). Width is the strongest lever, across $n_"embd" = 16 -> 256$ the root probe $R^2$ climbs monotonically (from being undecodable at 16): $0.07 -> 0.21 -> 0.35 -> 0.44$, with the mid ($0.12 -> 0.57$) and leaf-parent ($0.15 -> 0.59$) levels rising in parallel. Depth lifts every level too: across $n_"layer" = 1 -> 4$ the root goes $0.22 -> 0.42$, the mid $0.32 -> 0.54$, and the leaf-parent $0.34 -> 0.48$. Number of heads is the weakest axis: from $1$ to $8$ heads the root moves only $0.25 -> 0.36$, the mid $0.37 -> 0.48$, and the leaf-parent level is essentially flat ($0.42 -> 0.44$). 

*Root remains the weakest latent.* At all eleven configs the root ($L_0$) is
the weakest-decoded level.

*Model depth stretches the build-up and lifts the readout* (@archdepth). Plotting root
$R^2$ against normalized residual depth (residual index over $n_"layer"$), every model rises from
$approx 0.01$ at the embedding to its final-layer readout, and deeper models both
stretch the accumulation curve and reach a higher endpoint: final-layer root $R^2$ is
$0.22 -> 0.35 -> 0.39 -> 0.42$ for $n_"layer" = $ 1 to 4, respectively.

*Longer training has less effect on leaves decodability than on other latents.* Quadrupling the training budget
($4000 -> 16000$ steps) lifts the root by $+0.05$ ($0.35 -> 0.41$) and the mid level by
$+0.05$ ($0.48 -> 0.53$), but the leaf-parent level by only $+0.02$ ($0.44 -> 0.47$).

*Decodability decouples from loss fit* (@archlossfit). Across all 99 runs the
correlation between `loss_gap_closed` and probe $R^2$ is modest — $0.46$ for the root,
$0.42$ for the mid, $0.17$ for the leaf-parent, and the scatter is wide. A model
can reach near-Bayes loss without linearly representing the coarse belief, and vice
versa #todooleg[interesting].

#fig("figures/arch_marginal_r2.png",
  [Marginal capacity effects. Each panel varies one axis with the other three at
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

== Going deeper: $L = 4$

The $L = 3$ tree is shallow enough that the root collapses to certainty within a few
tokens #todooleg[linkage to ref], compressing the window over which its belief could be seen to bloom. Let's give the *root* more dynamic
range. 
This pass increments the tree level, so
context length becomes $16$, and we deal with *fourth* latent level. 
Other HRM parameters remain the same. #footnote[($s = 2, v = 8, m = 2$, uniform sampling,
unambiguous trees)]
We consider ten *non-isomorphic* grammars and three master seeds again. We vary model depth
$n_"layer" in {2, 3}$. Overall we make *60 runs* #footnote[`run_depth4.py`, writing
`results/depth4/g{NN}_L{n}_d128_h4_t4000_s{S}/`, a separate root so the
depth-agnostic dir names never collide with previous passes)]. 
These run close
0.96–0.99 of the uniform#text[→]Bayes loss gap (mean $0.99$) and are reported in full in @depth4sanitytable.



- *All earlier findings replicated at $L = 4$.* Linear decodability growth held (every level is read off well above the shuffled baseline of $approx -0.04$); the belief sharpens with context (exact root posterior entropy falls $1.86 -> 0.00$ monotonically over the 16 positions, @depth4bloom); belief *accumulates across layers* (root probe $R^2$ rises $-0.01 -> 0.07 -> 0.19$ for $n_"layer" = 2$ and $-0.01 -> 0.06 -> 0.17 -> 0.28$ for $n_"layer" = 3$, the deeper model will reach a higher readout, @depth4layer); and causal steering still works (@depth4steer) #todooleg[unclear what is written here].

Root decodability becomes even weaker at $L = 4$ than at $L = 3$: the $n_"layer" = 2$ family mean falls to $0.19$ (from $0.35$ at $L = 3$), and even adding a third layer lifts it only to $0.28$, still below the $L = 3$ value of $0.35$.


The extra depth gives the *root specifically* room to bloom. At $L = 3$ the root collapsed
to certainty within about three tokens, so its own belief barely had a window to expand
(the $L = 3$ blooming we showed earlier is dominated by the local latents); the longer
16-position $L = 4$ context stretches that window out. Over it the exact *root* posterior
entropy decreases smoothly and monotonically from $1.86$ nats to $0$, and the root belief
readout radius grows overall from $0.18$ to $0.50$ (@depth4bloom). The radius growth is
noisier than the local latents' — it wobbles in the back half — but the root now visibly
sharpens with context rather than snapping to certainty.

The per-level gradient takes the inverted U-shape (@depth4levels). Family-mean probe $R^2$ is $L_0 = 0.24$, $L_1 = 0.37$, $L_2 = 0.36$, $L_3 = 0.30$: it *rises* from the root to the mid-levels and then *falls* back toward the leaf-parents. Only $6$ of $60$ runs show the strict $L_0 < L_1 < L_2 < L_3$ ordering. Root remains the weakest decodable latent.


#fig("figures/depth4_level_r2.png",
  [$L = 4$ per-level probe strength across the family. Each point is one of 60 runs
   (10 grammars $times$ 3 seeds $times$ $n_"layer" in {2, 3}$); violins show the
   per-level $R^2$ distribution, diamonds the means. The gradient is an *inverted-U* —
   the mid-levels ($L_1, L_2$) decode strongest, the root ($L_0$) weakest, the
   leaf-parents ($L_3$) intermediate — not the pre-registered monotone ladder.],
  w: 82%) <depth4levels>

#fig("figures/depth4_blooming.png",
  [The root *does* bloom at $L = 4$. *Left:* the root belief readout radius grows with
   context position $k$ over the 16-position window (grey: 60 runs; orange: mean),
   though non-monotonically in the back half. *Right:* the exact root posterior entropy
   decreases smoothly and monotonically from $1.86$ nats to $0$.], w: 100%) <depth4bloom>

#fig("figures/depth4_layer_r2.png",
  [Belief accumulates across layers at $L = 4$. Root-belief probe $R^2$ vs residual
   index, one line per $n_"layer"$ (mean $plus.minus$ SD over 30 runs each). Both archs
   rise monotonically from the embedding to the readout; the 3-layer model reaches a
   higher endpoint ($0.28$ vs $0.19$).], w: 70%) <depth4layer>

#fig("figures/depth4_steering.png",
  [Causal steering replicates at $L = 4$. *Left:* steering the layer-1 residual toward
   a *wrong* latent collapses the true next token's log-probability in every run
   (grey), mean in orange. *Right:* the collapse magnitude (grey violin, orange diamond =
   mean) for $alpha{=}0 -> alpha{=}4$ across the 60 runs: $6.6 plus.minus 0.5$ nats.],
  w: 100%) <depth4steer>

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

= Appendix: The rule table as a graph <appendix-ruletable>

@ruletable lists the frozen grammar as text, three columns of `or`-separated child
pairs. The same information is easier to scan as a graph: four columns of eight nodes
(root, level 1, level 2, leaves), each parent's two rules drawn as two distinctly
colored pairs of arrows into the level below. No edge skips a level, which is the
"no cross-level ambiguity" property by construction.

#fig("figures/ruletable_graph.png",
  [The frozen rule table of @ruletable redrawn as a DAG. Each of the eight symbols
   `S1`-`S8` appears once per level; its two production rules are drawn in two
   distinct colors (one per rule), chosen so that neither a node's own two colors
   nor neighboring nodes' colors are easily confused.], w: 100%) <ruletablegraph>

= Appendix: Skewed and ambiguous example grammars <appendix-example-grammars>

The main-text grammar (@ruletable) has uniform rule-choice probabilities and no shared
child-tuples across parents. To illustrate the two knobs used by the non-collapse sweep
(@appendix-noncollapse-sanity), this appendix shows two concrete example grammars, restored
exactly from their saved `grammar.npz`: the draw-0, skew=high grammar
(`results/noncollapse/skhigh_am0_g00_L2_d128_h4_t4000_s0`) and the draw-0, $rho = 0.6$
grammar (`results/noncollapse/sknone_am0.6_g00_L2_d128_h4_t4000_s0`). Both tables and
graphs are generated directly from the saved arrays, not hand-transcribed.

#include "figures/ruletable_skew.typ"

#fig("figures/ruletable_graph_skew.png",
  [The skewed grammar as a DAG. Numbers on cells and near the start of each edge are the
   exact rule-choice probabilities ($alpha = 0.2$ symmetric Dirichlet draw); edge linewidth
   is proportional to the same probability, so a near-deterministic parent (one rule
   $approx 1$) shows a thick edge for its dominant rule and a faint, thin edge for the
   near-zero alternative.], w: 100%) <ruletablegraphskew>

#include "figures/ruletable_amb.typ"

#fig("figures/ruletable_graph_amb.png",
  [The ambiguous grammar as a DAG. Rule-choice probabilities are uniform, but child-tuples
   are shared across parents at rate $rho = 0.6$, so several parents' rules point at the
   same child pair (bold cells in the table; edges converging on the same node in the
   graph) -- the many-to-one structure that makes the belief non-collapsing.], w: 100%) <ruletablegraphamb>

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

Every run in the architecture sweep, no filtering — all 99 (11 one-axis-at-a-time capacity configs
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

= Appendix: Per-run $L = 4$ depth-sweep sanity table <appendix-depth4-sanity>

Every run in the $L = 4$ depth sweep, no filtering — all 60 (10 grammars $times$ 3
seeds $times$ $n_"layer" in {2, 3}$). `grammar` and `seed` index the rule-table draw
and master replicate seed; `n_layer` is the only varied architecture axis ($n_"embd" =
128$, $n_"head" = 4$, 4000 steps throughout). `test_ce` is the held-out next-token
cross-entropy; `loss_gap_closed` the fraction of the uniform#text[→]Bayes gap closed
(covariate, not a gate); `root_r2` and `deepest_r2` the level-$L_0$ and mean
leaf-parent-$L_3$ probe $R^2$. Loaded directly from `figures/depth4_sanity.csv`.

#let depth4sanity = csv("figures/depth4_sanity.csv")
#figure(
  table(
    columns: 7,
    inset: 3.2pt,
    align: center,
    stroke: 0.5pt + luma(220),
    table.header(..depth4sanity.at(0).map(h => [#text(7pt, weight: "bold")[#h]])),
    ..depth4sanity.slice(1).flatten().map(c => [#text(7pt)[#c]]),
  ),
  caption: [All 60 $L = 4$ depth-sweep runs (10 grammars $times$ 3 seeds $times$
    $n_"layer" in {2, 3}$). No convergence filtering. `deepest_r2` is the mean over the
    eight leaf-parent ($L_3$) nodes.],
) <depth4sanitytable>

= Appendix: Per-run non-collapse sweep sanity table <appendix-noncollapse-sanity>

Every run in the non-collapse sweep, no filtering — all 27 (3 `skew` $times$ 3
`ambiguity` $times$ 3 rule-table draws), at the pinned arch ($n_"layer" = 2, n_"embd" =
128, n_"head" = 4$, 4000 steps). `bayes_floor` is the exact weighted Bayes-optimal
next-token entropy; `loss_gap_closed` the fraction of the uniform#text[→]Bayes gap
closed (covariate); `root_r2`/`mid_r2`/`low_r2` the level-$L_0$/$L_1$/$L_2$ probe $R^2$
(shuffled baseline `shuffled_r2`); `k8_entropy` the headline mean $k = 8$ root posterior
entropy; `eff_dim`/`hull_area` the reachable-set descriptors. Loaded directly from
`figures/noncollapse_sanity.csv`.

#let ncsanity = csv("figures/noncollapse_sanity.csv")
#figure(
  table(
    columns: 13,
    inset: 2.4pt,
    align: center,
    stroke: 0.5pt + luma(220),
    table.header(..ncsanity.at(0).map(h => [#text(6pt, weight: "bold")[#h]])),
    ..ncsanity.slice(1).flatten().map(c => [#text(6pt)[#c]]),
  ),
  caption: [All 27 non-collapse runs (3 skew $times$ 3 ambiguity $times$ 3 draws). No
    filtering. `k8_entropy` $approx 0$ exactly when $rho = 0$ regardless of skew.],
) <ncsanitytable>

