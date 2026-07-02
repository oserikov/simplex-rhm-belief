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

// Reproduction-recipe code blocks (appendix): smaller monospace, shaded, breakable.
#show raw.where(block: true): it => block(
  fill: luma(248), inset: 7pt, radius: 2pt, width: 100%, breakable: true,
  text(it, size: 7.8pt),
)

// ---- Title block -------------------------------------------------------
#block[
  #set align(center)
  #text(17pt, weight: "bold")[Experimental Report:\ Belief Geometry, Random Hierarchy Models and Transformers]
  #v(0.3em)
]

#v(0.5em)

// Long story short
#heading(numbering: none)[Long Story Short]


I took RHMs data generation and trained a tiny transformer on emitted sequences, then studied whether the residual stream carries the ground truth posteriors about the RHM latent and where in the network this information lives, what geometry it traces, and whether the model relies on it.

I found that unlike in results of @shai2026, linear probes manage to achieve only *partial* recovery of the ground truth posteriors #footnote[with the core exception being in intentionally naive experiment where knowing the position of the sentence clearly disambiguated the latent in @appendix-forest]. Each layer of the model populates the residual stream with useful information about the ground truth posteriors. As context grows, the probe's predicted posteriors bloom outward from the prior toward simplex vertices. Causal steering confirms the model relies on it rather than merely exposing it. The phenomenology replicates across the RHM grammars of the same size, and one level deeper. It is also robust to model capacity. Finally, breaking the grammar's invertibility with *ambiguity* leaves a genuinely uncertain belief state, and there the probe does not degrade. It seems to sharpen but this might be due to the ground truth posteriors being less peaky, which distracts the chosen probes evaluation metrics.


*Note:* recently, I developed the following style of conducting experiments: I write experiment specification, then I hand it over to an AI agent which writes me a paper-like bit about what it did. I then criticize, fact-check and refine the results for until I believe what is written. This provides research and ideas with documentation easily tracable in git and useful for subsequent inquiries like "back then we did this. maybe it actually mean that". This led to this report being iteratively expanded by me populating interesting experimental results here. From time to time I spent time making images pretty, but this is just my form of insightful procrastination. Although now familiar with the field language, I find myself still getting used to it, thus sometimes I might be inaccurate in the terminology. I made an effort to make it legible for me though, because that's how I navigated my work and came up with ideas. Occasionally, this text contains margin notes I wrote for myself.

#heading(numbering: none, level: 2)[Rough Events Sequence on Author's Side]
1. Let's just take Cagnetta tress and train and see how it goes!
2. Let's ground our results in some expectations. Some (re-)derivations with AI, and *pre-registered prediction* about the geometry. (Resulted in @experimental-design)
3. Aha: some trends confirmed!. No strict linear recoverability though... (Described in @initial-results)
4. How robust are these trends? (resulted in @appendix-robustness)
5. $R^2$ is convenient for my varying setups, but let's also get RMSE to compare to #cite(<shai2026>, form: "prose") apples to apples and see the actual performance of probes. (although mostly no news, it resulted in @appendix-rmse)
6. I can't ignore the struggle anymore: RHMs are so simple! Why my probes score so imperfect...
7. A thought while I keep struggling, let's play with less straightforward grammars? (resulted in @interesting-grammars; Surprisingly probes for them scored better, however later checked, turns out it was due to the posteriors shape being less peaky)
8. Huh, #cite(<shai2026>, form: "author") do use much wider hidden dimensions, and also they explicitly introduce a few independent factors. And report they are orthogonal. Do we need wider hidden layers? Do we need to sample from a few clearly independent factors?Let's try these! (resulted in @appendix-forest)


*Note:* @appendix-reproduce contains the exact commands to reproduce most numbers and figures in this report, for replication convenience.

= Introduction


Natural data is hierarchical: characters compose into words, words into phrases, phrases into meaning. So are societies and mechanisms and computer programs organized. Human perception is adapted to this structure #cite(<ding2016>), with psycho- #cite(<bazhukov2024>) and neuro-linguistic #cite(<komissarenko2025>) experiments revealing humans susceptible to hierarchical structure in language errors. Sight is known to be hierarchical, with the visual cortex organized into a hierarchy of increasingly abstract features. 
In short, hierarchies is what we observe and process, and modern AIs deal with the same world as well. Understanding how and to what extent modern AIs are natural in handling hierarchical data is thus crucial to understanding their strengths and constraints, and can even be informative in practical applications such as advising the model setup that would be enough to process the data of known complexity.

The Random Hierarchy Model (RHM) of #cite(<cagnetta2025>, form: "prose") model hierarchical data with synthetic grammars. A syntetic grammar is a fixed tree in which each high-level symbol expands, via production rules randomly chosen from pre-set, into a fixed-length string of lower-level symbols, down to observed leaves. 
Such a grammar defines a finite set of equiprobable trees, each of which generates a leaf string.
Given a prefix of observed leaves, predicting the next leaf optimally requires inferring the distribution over the *hidden* latent symbols that form the hierarchy which generated the observed prefix. 

This allows us to see the RHM as a testbed for the belief geometry studies, earlier performed on HMMs by #cite(<shai2026>, form: "prose"). This would answer whether the auto-regressive language model will represent in its residual stream, some counterpart of the hierarchical data generating process. For example, will the ground truth posterior distribution over the data-generating process's hidden states be represented in the residual stream, will this information be used.

In their work, #cite(<shai2026>, form: "author") found that the residual stream of a trained transformer carries a linear image of the ground truth posterior (seemingly also known as exact posterior) over the hidden states of a HMM generative process. Will this be the case for the RHM as well? In other words, will the residual stream carry a linear image of the ground truth posterior over the hidden states of the RHM generative process?

Below we test whether the belief is (i) linearly decodable, (ii) geometrically organized meaningfully so that the belief moves to the posterior simplex vertices alongside context, (iii) built up additively across layers, and (iv) causally relevant to the model's output. Some additional curiosity is addressed in appendices.

In some sense, RHMs are a more natural testbed than HMMs for these questions, because they are hierarchical and have a tree structure, which is more similar to natural language. This does not mean though they are obviously more complex: unlike HMMs, RHMs by default have no recursive generation and yield fixed-length strings which makes them computationally and analytically simpler.


= Experimental Design
<experimental-design>


Methodologically, this paper sits between the two reference points in the bibliography: #cite(<cagnetta2025>, form: "prose")'s RHM study uses the same hierarchical data model but evaluates the last-token prediction setting, while #cite(<shai2026>, form: "prose")'s work motivates pooling predictive vectors across contexts. 


The study proceeds in five passes.
The *initial* experiment fixes one grammar and one training run and establishes the four core findings — linear decodability, layer accumulation, blooming, and causal use.
Three *robustness* passes then test whether these are properties of the RHM family rather than of one draw: a grammar sweep (10 grammars $times$ 3 seeds, 30 runs), an architecture sweep (11 capacity configs $times$ 3 grammars $times$ 3 seeds, 99 runs), and a depth-$4$ extension (10 grammars $times$ 2 model depths $times$ 3 seeds, 60 runs).
A final *supplementary* pass generalizes the grammar with two knobs — ambiguity and skew (27 runs) — to engineer a belief state that stays uncertain even at full context, and asks whether the linear image survives; it does, with $R^2$ above the shuffled control in every cell (@interesting-grammars).
Two follow-ups then interrogate why recovery is only ever *partial*: a metric companion restates every $R^2$ as the RMSE that #cite(<shai2026>, form: "prose") headline (@appendix-rmse), and three pre-registered variant experiments — a factored forest-RHM in two leaf layouts, an $epsilon$-decoupling of the latents, and a width sweep — find near-exact linear read-out only when the target beliefs are factored *and* one is active at a time, with neither independence nor capacity alone restoring it (@appendix-forest).
Every posterior used as a probe target is the exact ground truth, computed by the belief propagation derived in @beliefs and verified against brute-force enumeration.

== The Random Hierarchy Model <beliefs>

*Grammar.*
The default parameters of RHMs used are: arity $s = 2$, depth $L = 3$, so each string has $d = s^L = 8$ leaves; per-level vocabulary $v = 8$; and $m = 2$ production rules per parent symbol, chosen uniformly.
Rules are drawn once and held fixed, and are unambiguous.

A sample is generated top-down: pick a root class uniformly, recursively expand each symbol by a uniformly chosen rule, and emit the leaf string. Thus, each sample is both the string and the respetive parse tree formed of latents that generated this string.
With these parameters the grammar admits exactly $v dot m^(d-1) = 8 dot 2^7 = 1024$ equiprobable distinct trees.

The exact frozen grammar used in this first experiment is shown below (@ruletable). See this grammar visualized as a tree in @appendix-ruletable. Symbols are written as `S1`, ..., `S8` for readability (the saved arrays use zero-based ids); at each level the parent takes one of the two listed child pairs uniformly. Symbols are numbered to distinguish them, and their numbering is unrelated to the order of appearance in the derived string. The first symbol of the sampled string would be the one produced by the leftmost branching of the respective sampled parse tree. 
#figure(
  table(
    columns: (0.9fr, 1.4fr, 1.4fr, 1.4fr),
    inset: 4pt,
    stroke: 0.5pt + luma(200),
    align: horizon,
    table.header([*Parent*], [*Top expansion*], [*Middle expansion*], [*Bottom expansion*]),
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

== Computing the ground truth posteriors, optimal training floor and reasonable ceiling.
=== Derivation
*Naive marginalisation (the reference).*
Counting the proportion of prefix-consistent trees in which the latent takes a given value yields the exact ground-truth posterior.#footnote[when trees are not equiprobable, the count becomes a probability-weighted sum] A tree is defined by its root symbol together with one rule choice at each of the $d - 1$ internal nodes, so we only have $v med m^(d-1) = 1024$ trees in our default setup with small trees, making such counting-based approach feasible. This let us verify the belief-propagation posterior derived below against brute-force enumeration values, the two agree to $< 10^(-6)$ across 10 passing tests.

*Derivation.*
Write $z_(ell,i)$ for the latent symbol at node $i$ of level $ell$ (the root is $z_(0,0)$), and $x_1, ..., x_d$ for the leaves.
The grammar defines the joint law: the root is drawn from the uniform prior $pi(a) = 1 slash v$, and each node of level $ell$ carrying symbol $a$ (in our case, one of the `S1, …, S8` symbols) expands by rule $r in {1, ..., m}$ with probability $p_ell (a, r)$ (canonically $1 slash m$), emitting the child tuple $rho_ell (a, r) in {1, ..., v}^s$.
Given a prefix $x_(1:k)$ we want the posterior over any single latent, marginalising the $d - k$ unobserved leaves and every rule choice.
The tree is a factor graph without cycles, so sum-product belief propagation returns the exact marginals in one upward and one downward sweep.

The upward pass sends, from each node, the likelihood of the observed leaves in its subtree given the node's symbol,
$ mu_(ell,i)(a) = P(#[observed leaves under node $(ell,i)$] | z_(ell,i) = a). $
Leaves are the base case: a pinned indicator when observed, an all-ones message when marginalised,
$ mu_(L,j)(a) = cases(bb(1)[a = x_j] & "if" j <= k, 1 & "if" j > k), $
and each internal node combines its children by summing over its own rules,
$ mu_(ell,i)(a) = sum_(r=1)^m p_ell (a, r) product_(j=1)^s mu_(ell+1, med s(i-1)+j) (rho_ell (a, r)_j). $
The *root* posterior is then the upward message at the top, reweighted by the prior and normalised, #footnote[In `rhm.py` terms, this is `belief_root`: `mu[(0,0)]` times the uniform prior, normalised]
$ P(z_(0,0) = a | x_(1:k)) = (pi(a) med mu_(0,0)(a)) / (sum_(#todooleg[going over a' here is a denominator going through all possible root symbols]a') pi(a') med mu_(0,0)(a')). $


Any *non-root* latent needs, in addition, the evidence from the rest of the tree (co-emitted siblings are correlated via shared parent), carried by a downward message $lambda_(ell,i)(a)$ seeded at the root by the prior, $lambda_(0,0)(a) = pi(a)$, and pushed to child $j$ (global index $c_j = s(i-1)+j$) as
$ lambda_(ell+1, c_j)(b) = sum_a lambda_(ell,i)(a) sum_(r : med rho_ell (a, r)_j = b) p_ell (a, r) product_(j' != j) mu_(ell+1, c_(j')) (rho_ell (a, r)_(j')). $
The marginal at an arbitrary node is the product of the two messages, normalised, #footnote[In `rhm.py` terms, this is `belief_node`]
$ P(z_(ell,i) = a | x_(1:k)) = (mu_(ell,i)(a) med lambda_(ell,i)(a)) / (sum_(a') mu_(ell,i)(a') med lambda_(ell,i)(a')). $

Under unambiguity (C2) each child tuple names its parent uniquely, so at full context $k = d$ every upward message collapses to a single symbol and one root survives (entropy $0$), the sharpening previewed above. The aforedescribed computation is linear against tree size, making it much faster than the naive marginalisation algorithm.


*Bayes-optimal floor and uniform baseline.*
The _optimal_ next-token predictor is itself a belief marginal: predicting $x_(k+1)$ from $x_(1:k)$ is the posterior over the leaf node $(L, k+1)$ left unobserved, obtained from the same two sweeps,
$ P(x_(k+1) = c | x_(1:k)) = (mu_(L,k+1)(c) med lambda_(L,k+1)(c)) / (sum_(c') mu_(L,k+1)(c') med lambda_(L,k+1)(c')). $
Autoregressive models are judged by their next-token cross-entropy. 
The lowest attainable mean next-token cross-entropy thus equals the conditional entropy of each next leaf under the true grammar, averaged over prefixes,
$ H_k = H(X_(k+1) | X_(1:k)) = sum_(x_(1:k)) P(x_(1:k)) [ - sum_c P(c | x_(1:k)) log P(c | x_(1:k)) ]. $
Because the grammar is fully enumerable, we evaluate $H_k$ exactly by grouping all $1024$ trees by prefix and weighting each next-token distribution by its prefix mass $P(x_(1:k))$ (`conditional_entropy_floor`).
Only the $d - 1 = 7$ leaves that have a non-empty left context are predicted (the first leaf carries no information and is not scored), and the reported floor is their mean,
$ macron(H) = 1/(d-1) sum_(k=1)^(d-1) H_k. $
The *uniform baseline* is the loss of the context-free predictor $q(c) = 1 slash v$, whose cross-entropy against any next-token law is $-sum_c P(c) log q(c) = log v$ at every position and hence in the mean\; knowing nothing costs $log 8$ nats.

=== Calculation

*The Bayes-optimal floor.*
The value of an enumerable grammar is that we know the best achievable loss.
Averaging the ground-truth per-position posterior entropy over the seven predicted positions gives the Bayes-optimal mean next-token cross-entropy: 0.725 nats.
The uniform baseline is $ln 8 = 2.079$.

*Ground truth posterior.*
Given a leaf prefix of length $k$, the posterior over any hidden latent (including the root class) is computed by sum-product belief propagation on the tree: upward messages from observed leaves combine through the production rules to give the marginal over each latent.
The recorded self-test (a representative string) shows the root posterior sharpening with context (entropy $1.89$ at $k=1$ falling to $0.00$ (a single consistent root) by $k=3$) which previews the blooming geometry below.


== Model and training

We train a HuggingFace `GPT2LMHeadModel` with 2 layers, `n_embd` = 128, 4 heads, and all dropout disabled, totalling 398,848 parameters. 
There is no tokenizer: RHM leaf symbols are integers fed directly as `input_ids`. This `vocab_size` = 8 and `n_positions` = 8.
We train by next-token prediction on 922 training strings (102 held out) for 4000 steps on Apple MPS. #footnote[In @appendix-robustness more model configurations are explored (to no qualitative change).]

The trained model reaches a final held-out cross-entropy of *0.880* (@loss), closing $89%$ of the gap between the uniform baseline and the Bayes-optimal floor.
The model has approached the ground truth posterior predictor. Does it represent the posterior internally?



#fig("figures/loss_curve.png",
  [The transformer converges close to the ground truth RHM predictor.
   Held-out next-token cross-entropy (heavy line) falls from the uniform baseline ($ln 8 = 2.079$, top dashed) toward the Bayes-optimal floor ($0.725$, bottom dashed) over 4000 training steps, ending at $0.880$, closing #text[≈]89% of the uniform#text[→]Bayes gap (shaded).
   The faint line is the train-minibatch CE.
   #footnote[Seeded reproduction of the canonical run, `results/refrun/`; exact parameters in @appendix-reproduce (Stage 2).]], w: 66%) <loss>

== Pre-registered prediction

Following the project's honor-code rule, we committed our predictions to git *before* training any model or computing any probe (`PREREGISTRATION.md`).
In brief, we anticipated: (1) the root posterior is *linearly decodable* from the residual stream, with $R^2 > 0.5$ at the final layer and a shuffled baseline near 0; (2) the belief readout *blooms*: early positions cluster near the prior, late positions spread toward simplex vertices, with posterior entropy falling and activation radius #todooleg[radius grows bc activations go tow. simplex vertices] growing with context $k$; (3) decodability *accumulates additively across depth*, lowest at the embedding and highest after the last layer; and (given aforementioned expectations are optimistic wrt. exact belief being recoverable from residual stream) (4) the representation is *causally used*, so steering the residual along a belief direction shifts the output toward the corresponding latent's leaves (that is, parse tree reflects the steering: empirically the steered-towards-vertex being actually part of the parse tree).
#todooleg[together with AI We also registered alternative outcomes (degenerate collapse, non-linear-only encoding, MAP-only encoding, flat-with-depth) as falsification handles, all of which turned out to not hold actually].
We report against these predictions in @scorecard.

== Methodology
Each question in @initial-results is answered by one method, with the reading fixed in advance:

*Is the belief linearly present?* Fit one global affine map (least squares) from the final-layer residual to the exact root posterior; score by held-out $R^2$, against the identical probe fit to *shuffled* labels #cite(<hewitt2019>). $R^2$ well above the shuffled #text[≈]0 signals a linear image of the belief; $R^2$ at the control level would signal none. #todooleg[I grew dissatisfied with probes error scores over time, they lose too much information] $R^2$ is chosen because it is scale-free so scores remain comparable across latents, grammars, and sweep cells whose posteriors have very different spreads / variance. Its absolute-error counterpart RMSE is reported for every $R^2$ in @appendix-rmse, and highlighted in the main text when provides new information.


*Where is it built?* Score the same probe at each of the $n_"layer" + 1$ residual readout points, and separately at each context position. $R^2$ rising with readout depth signals additive accumulation; a flat profile was a pre-registered falsification handle.

*Which latents?* One probe per tree node, each targeting that node's exact posterior. The cross-node $R^2$ pattern maps which parts of the hierarchy the residual carries and how strongly.

*What geometry?* Project the probe's belief readout into a PCA(2) basis of the true posteriors and track the mean radius about the prior as context $k$ grows. Radius growth tracking the true posterior's own signals blooming; the same statistic on *raw-residual* PCA is the control that separates belief content from token/position nuisance variance.

*Is it used?* Patch the layer-1 residual along a belief direction with strength $alpha$ and watch the next-token distribution. If wrong-latent steering degrades the true token and re-routes mass onto the steered latent's children, the belief is causally used; inert predictions would signal decodable-but-unused.

= Results
<initial-results>

== A single linear probe partially decodes the root posterior

We fit one global least-squares affine map from the 128-d residual stream to the 8-class ground truth root posterior space, on a probe set of $N = 400$ held-out examples, and score it by $R^2$ on held-out data.
A single probe is fit once on all context positions pooled; we then score that same fixed probe both pooled and, separately, at each context position.
Because the train/test split is over whole sequences, every held-out sequence contributes one example at each of the eight positions, so each per-position score still rests on all $N = 400$ examples.
The final-layer probe reaches $R^2 = 0.38$, while at the control task (the same probe fit to *shuffled* labels, following #cite(<hewitt2019>, form: "prose")) scores $≈ 0$ ($-0.085$).
So the ground truth posterior is linearly accessible above chance, but the probe explains only a third of the variance: the recovery is *partial*.

Projected into a belief-space PCA basis (@simplex), the probe's predicted posteriors *visibly* occupy the same structured region as the ground truth posteriors and trace the same position gradient #todooleg[qualitative judgement], but as a diffuse cloud rather than the discrete point set of the ground truth. #footnote[The ground-truth panel looks sharper partly because the ground truth posterior takes few distinct values, and identical points overplot, whereas every probe prediction differs slightly.]
The affine correspondence is particularly weak for the root; it strengthens markedly for the deeper latents — mid-level $R^2$ of $0.51$ and $0.59$, deepest up to $0.66$ (@latents, next subsection).

#fig("figures/posterior_simplex.png",
  [The residual stream is a *partial* affine image of the exact belief simplex.
   *Left:* PCA(2) of the ground truth root posteriors (few distinct values, hence sharp overplotted dots); small-$k$ points sit near the prior, large-$k$ points spread toward vertices.
   *Right:* the linear probe's predictions *for the root posterior* in the same basis, colored by context position — the same region and a visually similar position gradient, but a diffuse cloud (held-out root $R^2 ≈ 0.38$).
   Both panels are the *root* class specifically.],
  w: 92%) <simplex>

== Root posterior decodability accumulates across model depth and context ix

The residual stream serves as a running sum of layer contributions, so we ask where the belief is built.
Both panels here decode the *root* posterior.
The residual stream is read at three points.
A GPT-2 "block" is one transformer layer — self-attention (here 4 heads) followed by an MLP, both writing into the residual stream — and our model stacks two such blocks.
Requesting `output_hidden_states` returns $n_"layer" + 1 = 3$ residual snapshots: index 0 is the token#text[+]position embedding (the input to block 1), index 1 is the residual after block 1, index 2 after block 2.
The four heads live *inside* each block and are not separate readout points; the "Layer 0/1/2" axis is these three *belief-readout points* (the sites where we decode the posterior), not three transformer layers.
Root-posterior probe $R^2$ rises along them: $-0.00$ at the embedding, $0.15$ after block 1, $0.38$ after block 2, while the shuffled control stays at $≈ 0$ throughout (@layerpos, left) — each block adds belief-relevant signal.
Resolving the same root probe by context position (@layerpos, right), the root is already decodable at early positions even at the shallow readout points, and at the final readout it is near-perfectly decodable ($R^2 = 1.0$) for the first few positions $0$–$3$#todooleg[I have a feeling this is beautiful and should have been derived or at least anticipated, but I didn't do so initially.]. #footnote[A few mid-readout cells are strongly negative (the probe underperforms the mean predictor on those positions); the heatmap colors are clipped for legibility, but the cell labels show the raw values].

#fig("figures/layer_position.png",
  [Root-posterior decodability accumulates across residual depth and context.
   Both panels decode the *root* class; "Layer 0/1/2" are the three residual readout points (embedding, after block 1, after block 2 = $n_"layer"{+}1$).
   *Left:* root-posterior $R^2$ rises along them ($-0.00 -> 0.15 -> 0.38$) while a shuffled control stays at #text[≈]0.
   *Right:* $R^2$(readout point, position) heatmap; the final readout reaches $R^2 = 1.0$ on the earliest positions.],
  w: 95%) <layerpos>

== Non-root latents are more strongly decodable than the root

The root is only one of the tree's hidden latents.
Probing for the ground truth posterior over latents at *every* tree level resembles a gradient with some node-level variation (@latents).
Precisely, root ($L_0$) $R^2 = 0.38$; the two level-1 mid latents $0.51$ and $0.59$; and the four deepest level-2 latents $0.61$, $0.66$, $0.50$, and $0.66$.
The residual encodes the entire hierarchy, but the local, near-leaf latents, which are more directly predictive of the next token, are usually read off more cleanly than the coarse global root.
Root is the *hardest* latent to decode, likely because it is the most abstract thus least locally predictive.

#fig("figures/latent_levels.png",
  [The residual encodes the whole latent hierarchy, deeper/local latents most strongly.
   Probe $R^2$ is lowest for the root ($L_0 = 0.38$), higher for level-1 ($0.51, 0.59$), and generally higher for level-2 ($0.61, 0.66, 0.50, 0.66$).],
  w: 72%)
  <latents>

*However* (An important refinement based on robustness studies (from @appendix-robustness),
this image is refined during robustness analysis.
Scaling the identical per-level probe to $L = 4$ (@latents-invu) shows the gradient is really an *inverted U*: the mid-level latents decode strongest, while both the coarse root and the most-local leaf-parents fall off. Respective RMSE values show the probe error to be diminishing, however decayed, with tree depth (@latents-invu-rmse). 

#fig("figures/depth4_level_r2.png",
  [Preview of the inverted-U refinement (full detail in @l4subsection-appendix).
   At $L = 4$, per-level probe $R^2$ across 60 runs (random seed and the exact grammar used vary) rises from the root ($L_0$) to the mid latents ($L_1, L_2$) and falls back toward the leaf-parents ($L_3$) — an inverted U, not a monotone "deeper is stronger" ladder.],
  w: 74%)
  <latents-invu>

#fig("figures/depth4_level_rmse.png",
  [RMSE shows the probe error to be diminishing, however decayed, with tree depth (full detail in @l4subsection-appendix).
   At $L = 4$, per-level probe RMSE across 60 runs (random seed and the exact grammar used vary) falls from the root ($L_0$) to the mid latents ($L_1, L_2$) and rises back toward the leaf-parents ($L_3$).],
  w: 74%)
  <latents-invu-rmse>

== The belief readout blooms with context

Our central geometric result (@blooming): projecting the linear belief readout into a belief-space PCA(2) basis, the points form a tight central cluster near the uniform prior when little context is observed and expand outward toward the simplex vertices as context accumulates.
The mean radius about the prior anchor grows from 0.17 at $k = 0$ to #text[≈]0.39 at $k = 7$, tracking the true posterior's own growth (0.17 #text[→] 0.47); equivalently, the ground truth posterior entropy falls with position (1.84 at $k=0$ to 0.00 at $k=7$).
The right panel colors the same cloud by ground-truth root class: it is only *loosely* organized by class — consistent with the modest root decodability ($R^2 = 0.38$, MAP #todooleg[MAP root accuracy is the fraction of held-out examples where the probe's argmax root class *matches* the ground-truth posterior's argmax (MAP) root class; here $0.47$. It compares only the top-1 class, unlike $R^2$, which scores the whole 8-dim posterior vector.] root accuracy $0.47$, @rootmatch) — not a clean separation.
Raw residual PCA, dominated by token and position nuisance variance, does *not* bloom — that is, its distance from the center does not grow with context position $k$ (radius#text[–]vs#text[–]$k$ correlation $approx 0.03$, against $approx 0.81$ for the belief readout) — it is the *belief content* of the residual that does.

#fig("figures/blooming.png",
  [The belief read out of the final-layer residual blooms with context.
   *Left:* points colored by context position $k$ with rings marking each position's mean radius about the prior ($times$); the cloud expands outward as context grows (mean radius $0.17 -> 0.39$, tracking the true posterior $0.17 -> 0.47$).
   *Right:* the same cloud colored by ground-truth root class is loosely organized by class (partial, not clean separation — root MAP accuracy $0.47$).], w: 100%) <blooming>

== Causal steering: the belief is used, not just decodable

A representation can be decodable yet causally inert.
To test for this we apply a patch to the layer-1 residual, pushing it by $alpha$ times the belief direction toward a chosen latent of our choice, and measure the effect on the next-token distribution (@steering).
Steering *toward the true* latent leaves predictions untouched (the model already holds that belief): the true token's log-probability stays at $-0.73$ for all $alpha$.
Steering *toward some wrong* latent collapses the mean log-probability of the true next token from $-0.73$ to $-8.5$ as steering effort $alpha$ grows, and re-routes probability mass onto the wrong latent's children: the next-token probability the model assigns to exactly the leaf symbols that the steered-to (wrong) latent can emit at that position (summed softmax mass over those symbols, averaged across positions and examples) rises from $0.22$ to $0.53$.


#fig("figures/steering.png",
  [The belief state is causally used.
   Steering the layer-1 residual toward the *true* latent leaves predictions intact (green); steering toward a *wrong* latent collapses the true next token's log-probability (left, orange: $-0.73 -> -8.5$) and re-routes mass onto the wrong latent's children (right, orange: $0.22 -> 0.53$).], w: 95%) <steering>

== Pre-registration scorecard <scorecard>

We grade each pre-registered prediction honestly against the outcome (@scorecardtable).
#todooleg[I remember saying monotonicity here and there, was a pointless overclaim, I think I've striked it out but need to be careful]

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

Three of four predictions held cleanly.
The single miss is the root $R^2$, which came in at 0.38 rather than the anticipated $> 0.5$.
We take this as a genuine and informative negative: the anticipation was correct in *form* (linear, above baseline, growing with depth and context) but our magnitude estimate for the *root specifically* was optimistic.
The latent-level sweep — which we did not pre-specify — explains why: the root is the coarsest, least locally-predictive latent, and the threshold we anticipated is in fact met by every deeper latent in the hierarchy.

== Brief summary of the robustness study results.
To test whether the observations hold across different setups, we have varied the following:
1. We have run the analysis against more grammars satisfying our initial conditions.
2. We have performed the same analysis for different model setups, varying the number of attention heads and layers, as well as the hidden dimension size and training duration.
3. We have considered deeper grammars of layer 4.
Main observations of this section held.
Respective setup-specific and sweep-specific observations can be seen in the @appendix-robustness. 


= Engineering non-trivial belief geometry
<interesting-grammars>

Everything so far lives on a tree whose belief, given the full leaf string, collapses to *certainty*: with uniform unambiguous rules the eight leaves invert level-by-level to exactly one root, so the $k = 8$ posterior is a delta and the belief path runs from the prior straight to a simplex vertex.
Yet #cite(<shai2026>, form: "prose") considered another setup: there, for HMMs, belief never collapses to a single certain state.
There, belief traces a fractal (Sierpinski-like, self-similar, recurrent) attractor.
But RHM has no recurrence, so "slow forgetting" — an HMM's belief only gradually losing information about the distant past — has no analog.
A possible analog of "the state can't be restored" is a *non-invertible observation channel*: a grammar where even the full leaf string leaves the root uncertain, so the reachable-belief set is a non-trivial attractor in the simplex rather than a path to a vertex.

We add two per-level knobs to the grammar (`rhm.py`, both reducing exactly to the canonical RHM at their off setting).
- *Ambiguity* $rho$ shares a fraction of the $v dot m$ rule entries' child-tuples *across parents* (many-to-one leaf#text[→]root structure); $rho in {0, 0.3, 0.6}$. This knob actually introduces non-invertibility of the latents. 
- *Skew* draws each parent's rule-choice probabilities non-uniform from a Dirichlet with concentration $alpha$: `none` is the canonical uniform $1\/m$; `mid` uses $alpha = 1.0$ and `high` uses $alpha = 0.2$ (smaller $alpha$ #text[⇒] more peaked, near-deterministic rule choice). This is a control node that just complicates the invertibility logic with uneven branching. 
Both belief propagation and the brute-force reference (both derived in @beliefs) were generalized to *weighted* sum-product using the same probabilities the sampler uses; the BP#text[↔] brute-force agreement holds to $2.5 times 10^(-16)$ under both knobs simultaneously.
We sweep the full $3 times 3$ grid $times$ 3 rule-table draws — *27 runs* #footnote[at the pinned published arch (`run_noncollapse.py`, writing `results/noncollapse/`), with the $("none", "none")$ cell reproducing the pass-1 baseline].


We naturally observe the mean ground truth $k = 8$ root-posterior entropy is $0$ at unambiguous setup regardless of the `skew`. It rises monotonically with ambiguity $rho$ growth: $0.00 -> 0.71 -> 1.38$ at `skew=none` (@nccurve).
At $rho = 0$ the belief collapses cleanly to $0$ by $k = 8$ (the bloom-to-vertex), while at $rho = 0.3$ and $rho = 0.6$ it reaches positive floor. 

#fig("figures/noncollapse_curve.png",
  [Ambiguity, not skew, engineers non-triviality.
   *Left:* mean ground truth $k = 8$ root posterior entropy vs ambiguity $rho$, one line per skew (error bars over 3 draws).
   At $rho = 0$ entropy is exactly $0$ for *every* skew — skew alone does not prevent collapse, and it rises monotonically with $rho$.
   *Right:* the full entropy-vs-context trajectory at `skew=none`; at $rho = 0$ the belief collapses to $0$ by $k = 8$, at $rho > 0$ it reaches a positive floor.],
  w: 100%) <nccurve>

*The linear probe sharpens with uncertainty growth!*
At every cell of the grid the residual stream still linearly encodes the (now spread) posterior well above the shuffled-label control task of $approx -0.07$ (@ncheatmap).
That control task is the same probe refit to randomly *permuted* labels and scored on held-out data, averaged over the 27 runs.
The probe did not *degrade* as $rho$ rose: $R^2$ *rises* with ambiguity, $0.36 -> 0.42 -> 0.53$ at `skew=none`, and the mid- and low-level probes rise even more steeply ($L_1: 0.46 -> 0.71$; $L_2: 0.35 -> 0.82$). Respective RMSE values reflect this pattern (see <rmse-ncheatmap>).  #todooleg[this leaves an open interesting question. Why is that? Does this indeed hint against memorization?]




#fig("figures/noncollapse_heatmap.png",
  [The linear probe survives ambiguity and skew, and *sharpens* with ambiguity.
   Per-level probe $R^2$ (root $L_0$, mid $L_1$, low $L_2$) over the $3 times 3$ skew $times$ ambiguity grid; each cell is mean $plus.minus$ SD over 3 rule-table draws, against a shuffled baseline of $approx -0.07$.
   $R^2$ rises left-to-right (with $rho$) at every level.
   The single dark $L_2$ cell at $("mid", rho{=}0)$ is a near-deterministic-target instability, not a decodability failure.],
  w: 100%) <ncheatmap>

#fig("figures/noncollapse_attractor.png",
  [Ground truth posterior and linear-probe readout after PCA, colored by context position $k$ for the uniform corner $rho = 0$ and the high-ambiguity corner $rho = 0.6$.
   At $rho = 0$ the late-context exact beliefs land *on* the vertices (due to collapse to certainty), which is not always the case at $rho = 0.6$.
   Probe readouts occupy rougly the same regions as their counterpart posteriors, with uncertaincy case readouts being more aligned with the ground truth posterior.
   (For visualization purposes, shown data is from one single run).],
  w: 90%) <ncattractor>




= Speculative Desired Next Steps

- Shall we some padding post-string, to train the last token embeddings better?.. Now the model needs not to use them ever, empirically I'd expect last token embeddings to be very bad. Will the results differ if we post-pad? 
- The root is worst decodable. Ultimately local latents being easier to predict than global matches my experience: phonetics/grammar is easier to predict early, semantics/tacit knowledge -- harder. But given the toy and very regular nature of the data the observed result is counter-intuitive. Kinda beutiful, unclear why.
- I'm a bit confused about whether the h_dim is good enough. #cite(<shai2026>, form: "prose") used much larger residual stream dimensionality, maybe to generously account for the factored nature of the generated data without really forcing model to collapse factored laws?
- Normal language is more similar to HMMs because it allows for recursion. I think the belief propagation would be more complicated. 
- The role of enumerability: does the model learn a *general* belief-geometry principle, or does it just memorize the exact belief for this grammar? #todooleg[I think the L4 exp did answer this? No?; Also, hm, poor probes might be evidence towards memorization?]
- Linguistic flavor: take real-world rulesets that are of roughly similar properties and see how it goes: would be much more approachable




= References

#set par(justify: false)
#bibliography("references.yml", title: none, style: "american-psychological-association")

#text(size: 9.5pt)[
  Project sources: `PREREGISTRATION.md`,
  `EXECUTION_OUTPUT.md`, `results/analysis.json`, `results/root_reconstruction.json`,
  `artifacts/train_summary.json`. Grammar sweep: `sweep.py`, `sweep.yaml`,
  `results/sweep/g*/{config.json,analysis.json}`, `figures/sweep_sanity.csv`.
  Architecture sweep: `run_arch.py`,
  `results/arch/g*/{config.json,analysis.json}`, `figures/arch_sanity.csv`.
]


#set heading(supplement: "Appendix")
#counter(heading).update(0)

= Appendix: Robustness studies <appendix-robustness>

The main text rests on one grammar and one training run, so this appendix asks whether the findings are properties of the RHM family rather than of a single draw or a single architecture.
It reports three sweeps: across grammars (10 grammars $times$ 3 seeds), across model capacity (11 architecture configs $times$ 3 grammars $times$ 3 seeds, one axis at a time), and one tree level deeper ($L = 4$; 10 grammars $times$ 2 model depths $times$ 3 seeds).
The four core findings — the inverted shape, blooming, layer accumulation, and causal steering — replicate across all ten grammars with low replicate variance, and the decodability holds across capacity (rising with width and depth, weakest dependence on head count).
As already stated above, the depth-$4$ sweep further refines the level gradient into an *inverted U*: mid-level latents decode most strongly, with both the coarse root and the most-local leaf-parents falling off.

== Generalization across grammars

The results above come from a single grammar #footnote[`grammar = 0`] and a single training run.
To test whether they are properties of the RHM *family* under the same fixed constraints #footnote[Reminder: fixed constraints so far are ($s=2$, $L=3$, $v=8$, $m=2$, uniform unambiguous rules)] rather than artifacts of one rule draw, we sweep the content of the rule table#footnote[See the example (first tried grammar) rule table at @ruletable] across ten independently sampled grammars with three replicate training seeds each, resulting in 30 runs.
We have ensured that the sampled grammars aren't isomorpic by post-factum ensuring that the respective nodes adjacency tables are not equivalent up to a per-level symbol permutation. #footnote[Test code in `grammar_iso.py`]
The model architecture / training code remained the same as in previous experiments.  #footnote[Reminder: model architecture so far: GPT-2-style transformer, 2 layers of 4 attn heads each, hidden size 128, 4k train steps. A single master seed controls model initialisation, training-data sampling, probe sampling, and the probe split, so replicates measure end-to-end sampling variance. Each run writes a deterministic, self-contained directory (`results/sweep/g{NN}_L2_d128_h4_s{S}/`) holding its grammar, a `config.json` manifest (resolved config, git SHA, timestamp), and per-run metrics — recoverable for this paper independent of any experiment-tracker. ]. 


- *All grammars training results in  near-Bayes-optimal CE.*
Every run closes $0.82–0.93\%$ of the uniform to Bayes-optimal loss gap (mean $0.89 plus.minus 0.03$), in line with the initial observation.
The runs details can be seen in @sanitytable.

- *The root remains the weakest-decoded level* (@sweeplevels).
Pooling probe $R^2$ by tree level, the root ($L_0$) averages $0.35 plus.minus 0.07$ (range $0.24$–$0.48$), well below the mid ($L_1$, mean $0.50$) and leaf-parent ($L_2$, mean $0.49$) latents.
The deepest level shows the widest spread (one grammar dips slightly negative #todooleg[interesting, look at it later]).

- *Blooming and belief-sharpening alongside context hold for every grammar* (@sweepbloom).
The ground truth posterior entropy falls with context position $k$ in all ten grammars, and the belief readout's radius grows with context.

- *Causal steering replicates with a stable effect size* (@sweepsteer).
Steering the layer-1 residual toward a wrong latent collapses the true next token's mean log-probability in every run: as the patch strength $alpha$ increases from $0$ to $4$, that log-probability drops by $6.9 plus.minus 0.5$ (mean $plus.minus$ SD across the 30 runs), with low run-to-run variance.

In short, all four earlier findings -- poor root decodability, blooming, layer-wise accumulation (recomputed in every run's per-layer probe), and causal steering -- replicate across the grammar family.


#fig("figures/sweep_level_r2.png",
  [The inverted strength gradient replicates across the grammar family.
   Each point is one of 30 runs (10 grammars $times$ 3 seeds); violins show the per-level distribution of probe $R^2$, diamonds the means.
   The root ($L_0$) is the weakest-decoded level ($0.35 plus.minus 0.07$), below the mid ($L_1$) and leaf-parent ($L_2$) latents ($approx 0.49$–$0.50$).], w: 82%) <sweeplevels>

#fig("figures/sweep_blooming.png",
  [Blooming is family-wide.
   *Left:* the belief readout's mean radius grows with context position $k$ (one line per grammar, averaged over seeds).
   *Right:* the ground truth posterior entropy falls with $k$ for every grammar.],
  w: 100%) <sweepbloom>

#fig("figures/sweep_steering.png",
  [Causal steering replicates with a stable effect size.
   *Left:* steering the layer-1 residual toward a *wrong* latent collapses the true next token's log-probability in every run (grey lines), mean in orange.
   *Right:* the distribution of the collapse magnitude (grey violin, orange diamond = mean) — the drop in the true token's mean log-probability as $alpha{=}0 -> alpha{=}4$ — across all 30 runs: $6.9 plus.minus 0.5$.],
  w: 100%) <sweepsteer>

== Architecture dependence

The grammar sweep fixes the architecture and varies the data; this section does the opposite.
Again, we keep the RHM constraints.
We vary the transformer's capacity one axis at a time around the baseline.
We consider (in bold are baseline params) $n_"layer"$ in {1, *2*, 3, 4}, $n_"embd"$ in {16, 64, *128*, 256}, $n_"head"$ in {1, 2, *4*, 8}, and training $"steps"$ in {*4000*, 16000}.
Each axis varies independently with the other three pinned at baseline, the baseline being their common center.
Every config is run across `grammar` $in {0, 1, 2}$ and `seed` $in {0, 1, 2}$ from grammar sweep, *99 runs total* #footnote[`run_arch.py` , writing steps-disambiguated dirs `results/arch/g{NN}_L{n}_d{d}_h{h}_t{steps}_s{S}/`)].
Small models are *expected* to underfit. 


*Model capacity lifts decodability, with a low-width floor* (@archmarginal).
Width is the strongest lever, across $n_"embd" = 16 -> 256$ the root probe $R^2$ climbs monotonically (from being undecodable at 16): $0.07 -> 0.21 -> 0.35 -> 0.44$, with the mid ($0.12 -> 0.57$) and leaf-parent ($0.15 -> 0.59$) levels rising in parallel.
Depth lifts every level too: across $n_"layer" = 1 -> 4$ the root goes $0.22 -> 0.42$, the mid $0.32 -> 0.54$, and the leaf-parent $0.34 -> 0.48$.
Number of heads is the weakest axis: from $1$ to $8$ heads the root moves only $0.25 -> 0.36$, the mid $0.37 -> 0.48$, and the leaf-parent level is essentially flat ($0.42 -> 0.44$).

*Root remains the weakest latent.*
At all eleven configs the root ($L_0$) is the weakest-decoded level.

*Model depth stretches the build-up and lifts the readout* (@archdepth).
Plotting root $R^2$ against normalized residual depth (residual index over $n_"layer"$), every model rises from $approx 0.01$ at the embedding to its final-layer readout, and deeper models both stretch the accumulation curve and reach a higher endpoint: final-layer root $R^2$ is $0.22 -> 0.35 -> 0.39 -> 0.42$ for $n_"layer" = $ 1 to 4, respectively.

*Longer training has less effect on leaves decodability than on other latents.*
Quadrupling the training budget ($4000 -> 16000$ steps) lifts the root by $+0.05$ ($0.35 -> 0.41$) and the mid level by $+0.05$ ($0.48 -> 0.53$), but the leaf-parent level by only $+0.02$ ($0.44 -> 0.47$).

*Decodability decouples from loss fit* (@archlossfit).
Across all 99 runs the correlation between `loss_gap_closed` and probe $R^2$ is modest — $0.46$ for the root, $0.42$ for the mid, $0.17$ for the leaf-parent, and the scatter is wide.
A model can reach near-Bayes loss without linearly representing the coarse belief, and vice versa #todooleg[well, that's handwaving, the whole paper is about this question].

#fig("figures/arch_marginal_r2.png",
  [Marginal capacity effects.
   Each panel varies one axis with the other three at baseline; points are root ($L_0$), mid ($L_1$), and leaf-parent ($L_2$) probe $R^2$, error bars are SEM over 3 grammars $times$ 3 seeds, the dotted line marks the shared baseline.
   Width and depth lift all levels (root collapses at $n_"embd" = 16$); heads matter least; the root is the lowest level in every panel.], w: 100%)
  <archmarginal>

#fig("figures/arch_depth_accum.png",
  [Depth and accumulation.
   *Left:* root-belief probe $R^2$ vs normalized residual depth, one line per $n_"layer"$ (mean $plus.minus$ SEM over grammars $times$ seeds); every model accumulates from the embedding to its readout, and deeper models reach higher.
   *Right:* final-layer root $R^2$ rises monotonically with depth.], w: 100%)
  <archdepth>

#fig("figures/arch_lossfit.png",
  [Decodability vs loss fit, all 99 runs.
   Root ($L_0$) and leaf-parent ($L_2$) probe $R^2$ against `loss_gap_closed` (fraction of the uniform$arrow.r$Bayes CE gap closed).
   Dashed lines are least-squares fits; the wide vertical spread at fixed loss fit shows decodability is not determined by how well the model fit the loss.],
   w: 82%) <archlossfit>

== Going deeper: $L = 4$
<l4subsection-appendix>

The $L = 3$ tree is shallow enough that the root collapses to certainty within a few tokens (@layerpos), compressing the window over which its belief could be seen to bloom.
Let's give the *root* more dynamic range.
This pass increments the tree level, so context length becomes $16$, and we deal with *fourth* latent level.
Other HRM parameters remain the same. #footnote[($s = 2, v = 8, m = 2$, uniform sampling, unambiguous trees)]
We consider ten *non-isomorphic* grammars and three master seeds again.
We vary model depth $n_"layer" in {2, 3}$.
Overall we make *60 runs* #footnote[`run_depth4.py`, writing `results/depth4/g{NN}_L{n}_d128_h4_t4000_s{S}/`, a separate root so the depth-agnostic dir names never collide with previous passes)].
These run close 0.96–0.99 of the uniform#text[→]Bayes loss gap (mean $0.99$) and are reported in full in @depth4sanitytable.



- *All earlier findings replicated at $L = 4$.*
Linear decodability growth held (every level is read off well above the shuffled baseline of $approx -0.04$); the belief sharpens with context (@depth4bloom); belief *accumulates across layers* (root probe $R^2$ rises $-0.01 -> 0.07 -> 0.19$ for $n_"layer" = 2$ and $-0.01 -> 0.06 -> 0.17 -> 0.28$ for $n_"layer" = 3$, @depth4layer); and causal steering still works (@depth4steer).

Root decodability becomes even weaker at $L = 4$ than at $L = 3$: the $n_"layer" = 2$ family mean falls to $0.19$ (from $0.35$ at $L = 3$), and even adding a third layer lifts it only to $0.28$, still below the $L = 3$ value of $0.35$.

The extra depth gives the *root specifically* room to bloom.
At $L = 3$ the root collapsed to certainty within about three tokens, so its own belief barely had a window to expand (the $L = 3$ blooming we showed earlier is dominated by the local latents); the longer 16-position $L = 4$ context stretches that window out.
Over it the ground truth *root* posterior entropy decreases smoothly and monotonically from $1.86$ to $0$, and the root belief readout radius grows overall from $0.18$ to $0.50$ (@depth4bloom).
The radius growth is noisier than the local latents' — it wobbles in the back half — but the root now visibly sharpens with context rather than snapping to certainty.

The per-level gradient takes the inverted U-shape (@depth4levels).
Family-mean probe $R^2$ is $L_0 = 0.24$, $L_1 = 0.37$, $L_2 = 0.36$, $L_3 = 0.30$: it *rises* from the root to the mid-levels and then *falls* back toward the leaf-parents.
Only $6$ of $60$ runs show the strict $L_0 < L_1 < L_2 < L_3$ ordering.
Root remains the weakest decodable latent.


#fig("figures/depth4_level_r2.png",
  [$L = 4$ per-level probe strength across the family.
   Each point is one of 60 runs (10 grammars $times$ 3 seeds $times$ $n_"layer" in {2, 3}$); violins show the per-level $R^2$ distribution, diamonds the means.
   The gradient is an *inverted-U* — the mid-levels ($L_1, L_2$) decode strongest, the root ($L_0$) weakest, the leaf-parents ($L_3$) intermediate — not the pre-registered monotone ladder.],
  w: 82%) <depth4levels>

#fig("figures/depth4_blooming.png",
  [The root *does* bloom at $L = 4$.
   *Left:* the root belief readout radius grows with context position $k$ over the 16-position window (grey: 60 runs; orange: mean), though non-monotonically in the back half.
   *Right:* the ground truth root posterior entropy decreases smoothly and monotonically from $1.86$ to $0$.], w: 100%) <depth4bloom>

#fig("figures/depth4_layer_r2.png",
  [Belief accumulates across layers at $L = 4$.
   Root-belief probe $R^2$ vs residual index, one line per $n_"layer"$ (mean $plus.minus$ SD over 30 runs each).
   Both archs rise monotonically from the embedding to the readout; the 3-layer model reaches a higher endpoint ($0.28$ vs $0.19$).], w: 70%) <depth4layer>

#fig("figures/depth4_steering.png",
  [Causal steering replicates at $L = 4$.
   *Left:* steering the layer-1 residual toward a *wrong* latent collapses the true next token's log-probability in every run (grey), mean in orange.
   *Right:* the collapse magnitude (grey violin, orange diamond = mean) for $alpha{=}0 -> alpha{=}4$ across the 60 runs: $6.6 plus.minus 0.5$.],
  w: 100%) <depth4steer>

= Appendix: Root reconstruction diagnostics <appendix-root-reconstruction>

The root-posterior probe is a regression target, but it is also useful to ask the harsher classification question: does the largest coordinate of the linear belief readout recover the sampled root class?
Across 400 probe examples and 8 context positions (3200 example-position points), the answer is yes for $1565$ points ($48.91%$).
This is well above random guessing, but random guessing is only 1/8 = $12.5%$ because there are eight root classes; the relevant baseline is not $50%$.

#fig("figures/blooming_match.png",
  [Root reconstruction from the linear belief readout.
   Green points are example-position pairs where the readout's argmax matches the ground-truth root; red points are mismatches.
   The overall match rate is $48.91%$, compared with a random-guessing baseline of $12.5%$ (not $50%$) for eight root classes.], w: 72%) <rootmatch>

Restricting to positions where nearly all evidence is visible makes the diagnostic sharper.
With seven of eight symbols observed ($k = 6$), the ground truth Bayesian posterior's MAP root already matches the sampled root in $95.00%$ of examples, while the linear readout matches in $62.25%$.
With all eight symbols observed ($k = 7$), the ground truth posterior is deterministic, but the readout still reaches only $70.50%$.
Thus early ambiguity explains part, but not all, of the fuzzy root reconstruction.

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

@ruletable lists the frozen grammar as text, three columns of `or`-separated child pairs.
The same information is easier to scan as a graph: four columns of eight nodes (root, level 1, level 2, leaves), each parent's two rules drawn as two distinctly colored pairs of arrows into the level below.
No edge skips a level, which is the "no cross-level ambiguity" property by construction.

#fig("figures/ruletable_graph.png",
  [The frozen rule table of @ruletable redrawn as a DAG.
   Each of the eight symbols `S1`-`S8` appears once per level; its two production rules are drawn in two distinct colors (one per rule), chosen so that neither a node's own two colors nor neighboring nodes' colors are easily confused.], w: 100%) <ruletablegraph>

= Appendix: Skewed and ambiguous example grammars <appendix-example-grammars>

The main-text grammar (@ruletable) has uniform rule-choice probabilities and no shared child-tuples across parents.
To illustrate the two knobs used by the non-trivial-grammars sweep (@appendix-noncollapse-sanity), this appendix shows two concrete example grammars, restored exactly from their saved `grammar.npz`: the draw-0, skew=high grammar (`results/noncollapse/skhigh_am0_g00_L2_d128_h4_t4000_s0`) and the draw-0, $rho = 0.6$ grammar (`results/noncollapse/sknone_am0.6_g00_L2_d128_h4_t4000_s0`).
Both tables and graphs are generated directly from the saved arrays, not hand-transcribed.

#include "figures/ruletable_skew.typ"

#fig("figures/ruletable_graph_skew.png",
  [The skewed grammar as a DAG.
   Each of a parent's four outgoing edges carries a small colored square near the parent holding its rule's exact choice probability ($alpha = 0.2$ symmetric Dirichlet draw; also in the table cells) — a rule's two edges share colour and value, and the four squares are stacked at fixed symmetric slots (two above, two below) so they stay aligned and never overlap.
   Edge linewidth is proportional to the probability, so a near-deterministic parent (one rule $approx 1$) shows thick edges with $1.00$ squares for its dominant rule and faint, thin edges with $0.00$ squares for the near-zero alternative.],
  w: 100%) <ruletablegraphskew>

#include "figures/ruletable_amb.typ"

#fig("figures/ruletable_graph_amb.png",
  [The ambiguous grammar as a DAG.
   Rule-choice probabilities are uniform, but child-tuples are shared across parents at rate $rho = 0.6$, so several parents' rules point at the same child pair (bold cells in the table; edges converging on the same node in the graph) -- the many-to-one structure that makes the belief non-collapsing.], w: 100%) <ruletablegraphamb>

= Appendix: Per-run grammar-sweep sanity table <appendix-sanity>

Every run in the grammar sweep, no filtering.
`grammar` indexes the rule-table draw (`Grammar.random(seed=grammar)`); `seed` is the master replicate seed; the architecture is pinned.
`test_ce` is the held-out next-token cross-entropy; `loss_gap_closed` is the fraction of the uniform#text[→]Bayes gap closed (sanity covariate, not a gate); `root_r2` and `deepest_r2` are the level-$L_0$ and mean level-$L_2$ probe $R^2$.
Loaded directly from `figures/sweep_sanity.csv`.

// CSV now also carries root_rmse/deepest_rmse (see @appendix-rmse); keep the R2 columns here for width.
#let sanity = csv("figures/sweep_sanity.csv").map(r => r.slice(0, 9))
#figure(
  table(
    columns: 9,
    inset: 4pt,
    align: center,
    stroke: 0.5pt + luma(220),
    table.header(..sanity.at(0).map(h => [#text(8pt, weight: "bold")[#h]])),
    ..sanity.slice(1).flatten().map(c => [#text(8pt)[#c]]),
  ),
  caption: [All 30 grammar-sweep runs (10 grammars $times$ 3 seeds), pinned architecture, full 4000-step budget.
    No convergence filtering.],
) <sanitytable>

= Appendix: Per-run architecture-sweep sanity table <appendix-arch-sanity>

Every run in the architecture sweep, no filtering — all 99 (11 one-axis-at-a-time capacity configs $times$ 3 grammars $times$ 3 seeds).
`n_layer`, `n_embd`, `n_head`, `steps` give the capacity config; `grammar` and `seed` the replicate.
`test_ce` is the held-out next-token cross-entropy; `loss_gap_closed` the fraction of the uniform#text[→]Bayes gap closed (covariate, not a gate); `root_r2` and `deepest_r2` the level-$L_0$ and mean level-$L_2$ probe $R^2$.
The deliberately-underpowered small models (e.g. $n_"embd" = 16$) are included.
Loaded directly from `figures/arch_sanity.csv`.

#let archsanity = csv("figures/arch_sanity.csv").map(r => r.slice(0, 10))
#figure(
  table(
    columns: 10,
    inset: 3.2pt,
    align: center,
    stroke: 0.5pt + luma(220),
    table.header(..archsanity.at(0).map(h => [#text(7pt, weight: "bold")[#h]])),
    ..archsanity.slice(1).flatten().map(c => [#text(7pt)[#c]]),
  ),
  caption: [All 99 architecture-sweep runs (11 one-axis-at-a-time capacity configs $times$ 3 grammars $times$ 3 seeds).
    No convergence filtering; small models are expected to underfit.],
) <archsanitytable>

= Appendix: Per-run $L = 4$ depth-sweep sanity table <appendix-depth4-sanity>

Every run in the $L = 4$ depth sweep, no filtering — all 60 (10 grammars $times$ 3 seeds $times$ $n_"layer" in {2, 3}$).
`grammar` and `seed` index the rule-table draw and master replicate seed; `n_layer` is the only varied architecture axis ($n_"embd" = 128$, $n_"head" = 4$, 4000 steps throughout).
`test_ce` is the held-out next-token cross-entropy; `loss_gap_closed` the fraction of the uniform#text[→]Bayes gap closed (covariate, not a gate); `root_r2` and `deepest_r2` the level-$L_0$ and mean leaf-parent-$L_3$ probe $R^2$.
Loaded directly from `figures/depth4_sanity.csv`.

#let depth4sanity = csv("figures/depth4_sanity.csv").map(r => r.slice(0, 7))
#figure(
  table(
    columns: 7,
    inset: 3.2pt,
    align: center,
    stroke: 0.5pt + luma(220),
    table.header(..depth4sanity.at(0).map(h => [#text(7pt, weight: "bold")[#h]])),
    ..depth4sanity.slice(1).flatten().map(c => [#text(7pt)[#c]]),
  ),
  caption: [All 60 $L = 4$ depth-sweep runs (10 grammars $times$ 3 seeds $times$ $n_"layer" in {2, 3}$).
    No convergence filtering.
    `deepest_r2` is the mean over the eight leaf-parent ($L_3$) nodes.],
) <depth4sanitytable>

= Appendix: Per-run non-trivial-grammar sweep sanity table <appendix-noncollapse-sanity>

Every run in the non-trivial-grammar sweep, no filtering — all 27 (3 `skew` $times$ 3 `ambiguity` $times$ 3 rule-table draws), at the pinned arch ($n_"layer" = 2, n_"embd" = 128, n_"head" = 4$, 4000 steps).
`bayes_floor` is the exact weighted Bayes-optimal next-token entropy; `loss_gap_closed` the fraction of the uniform#text[→]Bayes gap closed (covariate); `root_r2`/`mid_r2`/`low_r2` the level-$L_0$/$L_1$/$L_2$ probe $R^2$ (shuffled baseline `shuffled_r2`); `k8_entropy` the headline mean $k = 8$ root posterior entropy; `eff_dim`/`hull_area` the reachable-set descriptors.
Loaded directly from `figures/noncollapse_sanity.csv`.

// CSV inserted root/mid/low/shuffled_rmse before k8_entropy (see @appendix-rmse); select the original R2 columns.
#let ncsanity = csv("figures/noncollapse_sanity.csv").map(r => (0,1,2,3,4,5,6,7,8,9,14,15,16).map(i => r.at(i)))
#figure(
  table(
    columns: 13,
    inset: 2.4pt,
    align: center,
    stroke: 0.5pt + luma(220),
    table.header(..ncsanity.at(0).map(h => [#text(6pt, weight: "bold")[#h]])),
    ..ncsanity.slice(1).flatten().map(c => [#text(6pt)[#c]]),
  ),
  caption: [All 27 non-trivial-grammar runs (3 skew $times$ 3 ambiguity $times$ 3 draws).
    No filtering.
    `k8_entropy` $approx 0$ exactly when $rho = 0$ regardless of skew.],
) <ncsanitytable>

= Appendix: From $R^2$ to RMSE <appendix-rmse>

Throughout the paper we score the linear probe by the coefficient of determination $R^2$.
The nearest prior work, #cite(<shai2026>, form: "prose"), instead headlines *root-mean-square error* (RMSE) between the probe's reconstruction and the target belief vectors.
This appendix reports the RMSE counterpart of every $R^2$ we quote.
The main text is unchanged; RMSE lives only here (and as extra columns in the per-run sanity CSVs).

*Definition.*
For a probe that maps the residual stream to a $v$-dimensional belief vector, we compute the RMSE per output component and average over the $v$ components,
$ "RMSE" = 1/v sum_(j=1)^v sqrt(1/M sum_(i=1)^M (Y_(i j) - hat(Y)_(i j))^2), $
over the $M$ held-out examples.
This matches how our $R^2$ is aggregated (a per-output score averaged across the $v$ outputs), so the two metrics use the same across-output averaging.

*$R^2$ and RMSE carry the same information up to the target's scale.*
For a single output they are linked by $R^2 = 1 - "RMSE"^2 slash "Var"(Y)$, so a *scale-free* RMSE (dividing by the target's standard deviation) is exactly $sqrt(1 - R^2)$ and would add nothing beyond $R^2$.
Only the *absolute* RMSE — in the units of the belief vector — carries information $R^2$ does not, namely the physical size of the error. Unlike $R^2$, absolute RMSE is *not* monotone across different latents, because each latent's belief vector has its own variance — a latent with higher $R^2$ can still show a similar RMSE if its posterior is more spread out.  Similarly, absolute RMSE magnitudes are not comparable between our text and #cite(<shai2026>, form: "prose").


*Per-latent RMSE (matching the run from @initial-results).*
The per-latent probing of @latents is the RHM analog of #cite(<shai2026>, form: "prose")'s per-factor recovery.
@rmse-latents gives both scores side by side.
Note that RMSE falls from root to the deeper latents while $R^2$ rises, but not monotonically in lockstep — the mid latent `mid_L1a` has a much higher $R^2$ than the root yet a similar RMSE, exactly the scale effect noted above.

#figure(
  table(
    columns: (2fr, 1fr, 1fr),
    inset: 5pt,
    align: (left, center, center),
    stroke: 0.5pt + luma(200),
    table.header([*Latent*], [*$R^2$*], [*RMSE*]),
    [root ($L_0$)], [$0.38$], [$0.188$],
    [mid ($L_1$a)], [$0.51$], [$0.185$],
    [mid ($L_1$b)], [$0.59$], [$0.144$],
    [leaf-parent ($L_2$a)], [$0.61$], [$0.182$],
    [leaf-parent ($L_2$b)], [$0.66$], [$0.154$],
    [leaf-parent ($L_2$c)], [$0.50$], [$0.145$],
    [leaf-parent ($L_2$d)], [$0.66$], [$0.117$],
  ),
  caption: [Per-latent recovery on the canonical run, $R^2$ (as in @latents) beside the per-output-averaged RMSE.
    RMSE generally shrinks for deeper latents, but is not a strict monotone image of $R^2$ because each latent's posterior has its own scale.],
) <rmse-latents>

Reading the same root probe at the three residual points (after embeddings layer, first and second transformer layer), RMSE falls as $R^2$ rises: $R^2 = -0.00 -> 0.15 -> 0.38$ against RMSE $= 0.241 -> 0.221 -> 0.188$, while the shuffled control still stays flat.

*Across the sweeps.*
The two metrics move together across the RHM family.
In the grammar sweep (30 runs) the root recovers at $R^2 = 0.35 plus.minus 0.06$ and RMSE $= 0.203 plus.minus 0.007$, and the deepest latents at $R^2 = 0.49$ / RMSE $= 0.171$.



*RMSE-flavored figures.*
The remaining figures restate the paper's $R^2$ figures in RMSE, computed from the same probes and saved data.
Because RMSE is lower-is-better and lives in $[0, tilde.op 0.3]$ rather than $[0, 1]$, the axes and heatmap scales differ from their $R^2$ originals, but the ordering of latents, layers, and sweep cells is the mirror image.

#fig("figures/latent_levels_rmse.png",
  [Per-latent probe RMSE on the canonical run — the RMSE counterpart of @latents.
   RMSE is generally lower (better) for the deeper latents, mirroring the $R^2$ gradient, but is not its exact reflection because each latent's posterior has a different scale. RMSE susceptibility to variance plays here: latent L2a has high variance hence its RMSE is huge.],
  w: 72%) <rmse-latents-fig>

#fig("figures/layer_position_rmse.png",
  [Root-probe RMSE across residual depth and context — the RMSE counterpart of @layerpos.
   *Left:* RMSE falls across the three readout points while the shuffled control stays high.
   *Right:* the (readout point, position) RMSE heatmap, colors clipped at $0.5$; the two cells above that threshold are left uncolored (white) so they cannot flatten the rest of the scale, though their raw value is still annotated.
   The cell (layer 1, position 5) shows RMSE $approx 7.6$ — not a genuinely large error but the signature of the same catastrophic overfit visible in the corresponding $R^2$ heatmap ($R^2 approx -823$ at that cell): with $n_"embd"$ features and few held-out rows, the OLS probe at that particular (layer, position) fits noise in training and extrapolates wildly on held-out data, so its raw, unnormalized error blows up while $R^2$ saturates at a large negative number instead.],
  w: 95%) <rmse-layerpos>

#fig("figures/sweep_level_rmse.png",
  [Per-level probe RMSE across the 30-run grammar sweep — the RMSE counterpart of @sweeplevels.],
  w: 85%) <rmse-sweeplevels>

#fig("figures/depth4_level_rmse.png",
  [Per-level probe RMSE across the 60-run $L = 4$ depth sweep — RMSE counterpart of @depth4levels; the inverted-U reads as a U in RMSE (mid latents lowest error).],
  w: 74%) <rmse-depth4levels>

#fig("figures/depth4_layer_rmse.png",
  [Root-probe RMSE vs residual index at $L = 4$ — RMSE counterpart of @depth4layer.],
  w: 74%) <rmse-depth4layer>

#fig("figures/arch_marginal_rmse.png",
  [Marginal effects of width, depth, heads, and training budget on probe RMSE — RMSE counterpart of @archmarginal.],
  w: 95%) <rmse-archmarginal>

#fig("figures/arch_depth_accum_rmse.png",
  [Root-probe RMSE vs normalized residual depth and vs number of layers — RMSE counterpart of @archdepth.],
  w: 95%) <rmse-archdepth>

#fig("figures/arch_lossfit_rmse.png",
  [Probe RMSE against loss-gap-closed — RMSE counterpart of @archlossfit.],
  w: 72%) <rmse-archlossfit>

#fig("figures/noncollapse_heatmap_rmse.png",
  [Per-level probe RMSE over the skew $times$ ambiguity grid — RMSE counterpart of @ncheatmap; RMSE *falls* left-to-right as ambiguity rises, the probe sharpening as the belief spreads.],
  w: 100%) <rmse-ncheatmap>

= Appendix: Forest-RHM and the width sweep — seeking perfect probes <appendix-forest>

We might need either more capable model, or straigtforward independed factorization of the data generatoin to get results close to @shai2026.

*Design: a forest instead of a tree.*
The Forest-RHM replaces the single depth-$L$ tree with $k = 4$ *independent* depth-1 trees (each: one root symbol from $v = 8$, expanded once by one of $m = 2$ rules into $s = 2$ leaves), laid out in one 8-token sequence.
With no shared ancestor, the joint posterior over the 4 roots factors exactly. #cite(<shai2026>, form: "prose")'s theory for likewise setup in HMMs predicted per-tree beliefs in orthogonal linear subspaces with near-exact linear read-out.
Capacity is provably ample: each tree's belief is a probability vector over its $v = 8$ possible roots, which — its components summing to 1 — has only $v - 1 = 7$ free dimensions, so embedding all $k = 4$ beliefs in disjoint linear subspaces needs at most $k(v - 1) = 28$ dimensions against a hidden size of $128$.
Two leaf layouts distinguish two hypotheses: *concatenated* (tree 0's leaves, then tree 1's, …) lets the model deal with beliefs in order, while *interleaved* setup (leaf 0 of every tree, then leaf 1 of every tree) forces all four beliefs to be maintained through the reading.
Exact per-tree posteriors are verified against brute-force enumeration ($< 10^(-6)$), and cross-tree independence is verified numerically (belief about tree $j$ is untouched by other trees' leaves, $< 10^(-9)$).#todooleg[I'm a bit confused: both tasks are solvable by just looking at the position of the token in the sequence, so why is the interleaved task harder?]

- On the concatenated layout the probe recovers the trained trees' root posteriors at RMSE $2.7 times 10^(-6)$, $0.028$, and $0.0008$ (shuffled control $tilde.op 0.34$) — at or below the pre-registered $0.03$ threshold, i.e. #cite(<shai2026>, form: "prose")-grade near-exactness. #footnote[(The fourth tree finishes at the sequence-final position, which next-token training never supervises)] This is the positive control: when the theory's preconditions hold, our probe pipeline certifies near-zero RMSE — so the plateaus elsewhere in this paper are properties of the representations, not of the instrument.

- On the interleaved layout, the same model and probe plateau at RMSE $0.12 - 0.24$ close to the initial run's range, despite the model closing $99%$ of its loss gap in both layouts. Maybe hidden size is too small? #cite(<shai2026>, form: "prose") operated with much higher hidden size. We therefore reran the interleaved forest at hidden sizes ${256, 512, 1024}$ (all at matched convergence, loss gap closed $tilde.op 0.99$), recording (i) per-tree RMSE, (ii) held-out loss-gap-closed (to catch under-training and memorization), and (iii) a *probe-subspace overlap* diagnostic: for each pair of trees, the mean absolute cosine of principal angles between the two probes' row spaces ($0$ = orthogonal, $1$ = identical). The overlap falls smoothly and monotonically with width, $0.33 -> 0.23 -> 0.13 -> 0.08$ — the four beliefs come to occupy nearly orthogonal subspaces — yet the trained-tree RMSE never leaves its plateau ($0.13 - 0.17$ at $128$, $0.20 - 0.29$ at $256$, $0.12 - 0.27$ at $512$, $0.22 - 0.33$ at $1024$), approaching the concat layout's $lt.eq 0.03$ at no width up to $36 times$ nominal slack.


Across the variant experiments, near-exact linear belief read-out appeared *only* when the target beliefs were factored *and* sequentially active (concat), and was destroyed by making them simultaneous (interleave) — while removing latent dependence (the $epsilon$-decoupling experiment, `results/exp_epsilon/`) and adding capacity (this width sweep) each failed to restore it.

*Reproduction.*
`uv run exp_forest.py` (both layouts, sanity numbers in `results/exp_forest/sanity.json`); `uv run python exp_forest_width_sweep.py` (sweep, `results/exp_forest/width_sweep.json`).
Pre-registered predictions: `spec_variants.md` (committed before results, `git log` `175e9ee`).


= Appendix: How to reproduce main numbers and figures <appendix-reproduce>

This appendix answers, for every number and figure cited above, exactly how to
get it again: which script to run, with which flags, and which file on disk
currently holds the value.
Pipeline shape, for orientation: `rhm.py` (grammar + exact belief propagation, library only) #text[→] `train.py` (one training run) #text[→] `analyze.py` (probe + causal steering for one run's artifacts) #text[→] `sweep.py` (one sweep *cell* = train + analyze, deterministic dir + `config.json` manifest; usable standalone or under `wandb agent`) #text[→] `run_arch.py` / `run_depth4.py` / `run_noncollapse.py` (loop `sweep.py --no-wandb` over a grid) #text[→] the aggregator scripts in `paper-typst/figures/scripts/` (one per pass) that glob `results/⟨pass⟩/*/{config.json,analysis.json}` and emit the paper's PNGs, `*_sanity.csv`, and (for passes 4-5) scored-prereg JSON.
Figure generation is *not* wired into `ninja` — only the Typst compile is (`build.ninja`: `typst compile --ignore-system-fonts paper-typst/main.typ paper-typst/main.pdf`); regenerating a figure is always a separate, manual `uv run python paper-typst/figures/scripts/⟨name⟩.py`.

#block(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%)[
  *Load-bearing caveat.*
  The pass-1 canonical run behind `artifacts/` (and hence `results/analysis.json`, `results/root_reconstruction.json`, and @simplex–@steering) was trained *before* model-init seeding was added to the codebase (added in the pass-2 refactor; see `WORKLOG.md`, 2026-06-29T19:35).
  Re-running the commands below reproduces the *grammar* exactly (`Grammar.random(seed=0)` is deterministic) but not the *trained weights* bit-for-bit.
  Every later pass (`results/refrun/`, `results/sweep/`, `results/arch/`, `results/depth4/`, `results/noncollapse/`) seeds model init from the master `--seed` and is fully deterministic.
]

== Full pipeline in exact order

Everything needed to go from a clean checkout to a rebuilt `main.pdf` with every figure and table regenerated.
Steps within a numbered stage are independent of each other; stages must run in the listed order because later stages read files earlier stages write.
Total wall time is dominated by stages 3–6 (real MPS training): pass-1 (1 run), pass-2 (30 runs), pass-3 (99 runs), pass-4 (60 runs), pass-5 (27 runs) #text[≈] 216 training runs, each #text[≈] 25–45 s at the pinned arch (longer for larger `n_embd`/`n_layer`/`steps` cells in pass 3).

*Stage 0 — sanity checks (seconds).*
```bash
uv run pytest -q                 # 10+ tests: BP == brute-force (<1e-6), weighted BP, grammar sampling
uv run python rhm.py             # self-test trace: sample length == s^L, entropy bloom 1.89->0.00
uv run ruff check                # lint
```

*Stage 1 — pass-1 canonical run #text[→] `artifacts/`* (not bit-exact, see caveat above).
```bash
uv run python train.py --steps 4000 --test-frac 0.1 --log-every 1000
uv run python analyze.py                                   # -> results/analysis.json
uv run python paper-typst/figures/scripts/root_reconstruction.py   # -> results/root_reconstruction.json
```

*Stage 2 — pass-1 dense-loss reference run #text[→] `results/refrun/`* (independent of stage 1; a separate seeded run used only for @loss).
```bash
uv run python -c "
from pathlib import Path
from types import SimpleNamespace
from train import train
args = SimpleNamespace(s=2, L=3, v=8, m=2, grammar_seed=0, seed=0,
    ambiguity=0.0, skew='none', n_layer=2, n_embd=128, n_head=4,
    lr=3e-3, steps=4000, batch_size=128, test_frac=0.1, n_probe=400, log_every=50)
train(args, out_dir=Path('results/refrun'))
"
```

*Stage 3 — pass-1 figures* (depend on stages 1–2: need `artifacts/`, `results/analysis.json`, `results/refrun/train_summary.json`).
```bash
uv run python paper-typst/figures/scripts/loss_curve.py          # @loss
uv run python paper-typst/figures/scripts/posterior_simplex.py   # @simplex
uv run python paper-typst/figures/scripts/layer_position.py      # @layerpos
uv run python paper-typst/figures/scripts/latent_levels.py       # @latents
uv run python paper-typst/figures/scripts/blooming.py            # @blooming, @rootmatch
uv run python paper-typst/figures/scripts/steering.py            # @steering
```

*Stage 4 — pass-2 grammar-generalization sweep #text[→] `results/sweep/`* (30 runs; independent of stages 1–3, needs only `sweep.py`/`analyze.py`/`train.py`).
```bash
for g in $(seq 0 9); do for s in 0 1 2; do
  uv run python sweep.py --no-wandb --out-root results/sweep --grammar $g --seed $s
done; done
uv run python paper-typst/figures/scripts/grammar_sweep.py
# -> sweep_level_r2.png, sweep_blooming.png, sweep_steering.png, sweep_sanity.csv
```

*Stage 5 — pass-3 architecture sweep #text[→] `results/arch/`* (99 runs; independent of stages 1–4).
```bash
uv run python run_arch.py            # optionally --dry-run first to preview the 99-cell plan
uv run python paper-typst/figures/scripts/arch_sweep.py
# -> arch_marginal_r2.png, arch_depth_accum.png, arch_lossfit.png, arch_sanity.csv
```

*Stage 6 — pass-4 depth-4 sweep #text[→] `results/depth4/`* (60 runs; independent of stages 1–5).
```bash
uv run python run_depth4.py          # optionally --dry-run first
uv run python paper-typst/figures/scripts/depth4_sweep.py
# -> depth4_level_r2.png, depth4_blooming.png, depth4_layer_r2.png,
#    depth4_steering.png, depth4_sanity.csv, depth4_prereg.json
```

*Stage 7 — pass-5 non-trivial-grammar sweep #text[→] `results/noncollapse/`* (27 runs; independent of stages 1–6, but stages 8–9 depend on it).
```bash
uv run python run_noncollapse.py     # optionally --dry-run first
uv run python paper-typst/figures/scripts/noncollapse.py
# -> noncollapse_curve.png, noncollapse_heatmap.png, noncollapse_attractor.png,
#    noncollapse_sanity.csv, noncollapse_prereg.json
```

*Stage 8 — grammar-as-graph figures* (depend on stage 1 for the base grammar, and on stage 7 for the two pinned skew/ambiguity example dirs — must run *after* `run_noncollapse.py`, since its `__main__` regenerates all three PNGs in one process and errors if the noncollapse dirs don't exist yet).
```bash
uv run python paper-typst/figures/scripts/ruletable_graph.py
# -> ruletable_graph.png, ruletable_graph_skew.png, ruletable_graph_amb.png
uv run python paper-typst/figures/scripts/ruletable_tables.py
# -> ruletable_skew.typ, ruletable_amb.typ (included verbatim by main.typ)
```

*Stage 9 — grammar non-isomorphism check* (verifies the pass-2 sweep's 10 grammars are pairwise non-isomorphic; can run any time after stage 4).
```bash
uv run python grammar_iso.py
```

*Stage 10 — compile the paper* (depends on every stage above having produced its PNGs/CSVs/`.typ` includes under `paper-typst/figures/`).
```bash
ninja
```

*Parallelization note.*
Stages 4, 5, 6, and 7 (the four sweeps) don't read each other's outputs and can run concurrently given the wall-clock budget to run multiple MPS training processes at once; stage 8 must still wait for both 1 and 7 to finish, and stage 10 must wait for all figure-producing stages.

== Per-number and per-figure provenance

*Abstract.*

#table(
  columns: (1.6fr, 2fr, 1.6fr),
  inset: 4pt,
  align: left,
  stroke: 0.5pt + luma(200),
  table.header([*Number*], [*File · key*], [*Regenerate*]),
  [#text[≈]399k parameters], [`artifacts/train_summary.json` #text[→] `n_params` (398848)], [Stage 1],
  [1024 equiprobable trees], [derived: $v dot m^(d-1) = 8 dot 2^7$], [arithmetic, not a file],
  [$<10^(-6)$ BP vs brute-force], [`pytest` (10 tests)], [Stage 0],
  [0.88 test CE / 89% gap closed], [`results/refrun/train_summary.json` #text[→] `final_test_loss`, and $(u - f)\/(u - b)$], [Stages 1–3],
  [$R^2 = 0.38$ root / $approx 0$ shuffled], [`results/analysis.json` #text[→] `latent_level_r2.root_L0`, `layer_r2_shuffled[-1]`], [Stage 3],
  [$-0.00 -> 0.15 -> 0.38$ layer accumulation], [`results/analysis.json` #text[→] `layer_r2`], [Stage 3],
  [up to 0.66 deep latents], [`results/analysis.json` #text[→] `latent_level_r2`], [Stage 3],
  [$-0.73 -> -8.5$ steering], [`results/analysis.json` #text[→] `steering.true_lp_steer_wrong`], [Stage 3],
  [three of four predictions held], [`PREREGISTRATION.md` scored by hand against `results/analysis.json`], [Stages 1, 3],
  [30-run sweep, 10 grammars $times$ 3 seeds], [`figures/sweep_sanity.csv` (30 rows)], [Stage 4],
  [99-run architecture sweep], [`figures/arch_sanity.csv` (99 rows)], [Stage 5],
)

*Experimental design — grammar and ground truth posterior.*
The frozen rule table (@ruletable) and "1024 equiprobable trees" are read directly off `artifacts/grammar.npz` (`rules_0/1/2`, `probs_0/1/2` if present).
To regenerate the *graph* version instead of the text table: `uv run python paper-typst/figures/scripts/ruletable_graph.py` (Stage 8; reads `artifacts/grammar.npz` via `analyze.load_grammar`, writes `figures/ruletable_graph.png`).
"$<10^(-6)$ across 10 passing tests" (BP vs brute-force): `uv run pytest -q` (Stage 0).
"entropy 1.89 at $k=1$ falling to 0.00 by $k=3$" (the self-test preview): `uv run python rhm.py` (Stage 0; module self-test prints exactly this trace, see `EXECUTION_OUTPUT.md` § 1).
"0.725 nats" Bayes floor / "2.079" uniform baseline: `artifacts/train_summary.json` #text[→] `bayes_optimal_mean`, `uniform_baseline` (Stage 1; computed inside `train.train()` from the exact grammar, independent of the trained model).

*Model and training* (@loss).
Currently-informative file: `results/refrun/train_summary.json` #text[→] `uniform_baseline`, `bayes_optimal_mean`, `final_test_loss`, `n_params`, `n_train`, `n_test`, `config{n_layer,n_embd,n_head}`, `loss_history` (list of `[step, train_minibatch_CE, held_out_test_CE]`, #text[≈]81 points at `log_every=50`).
This is the file the figure and the "0.880 / 89%" text both cite — *not* `artifacts/train_summary.json`, which has no `loss_history` (pass-1 predates that field).
`artifacts/train_summary.json` holds the original pass-1 numbers (`final_test_loss=0.8911`, `n_params=398848`, `n_train=922`, `n_test=102`), underlying the abstract's "≈399k parameters" and the `EXECUTION_OUTPUT.md` § 3 trace, but Fig. 1's curve and caption numbers come from `results/refrun/` (Stage 2).
Figure: Stage 3, `loss_curve.py`.

*Linear probe* (@simplex).
Data: `artifacts/probe_data.npz` (`beliefs` $N times 8 times 8$, `hidden` $N times 3 times 8 times 128$ — index `[:,2]` is the final-layer residual) for the geometry panel; `results/analysis.json` #text[→] `latent_level_r2.root_L0` (0.38) and `layer_r2_shuffled[-1]` ($-0.085$) for the cited $R^2$ numbers.
Figure: Stage 3, `posterior_simplex.py` (fits its own in-sample probe for the plotted geometry — the annotated "$R^2 approx 0.38$" is the held-out score from `results/analysis.json`, a separate fit; see the script's header comment).

*Layer/position* (@layerpos).
Data: `results/analysis.json` #text[→] `layer_r2` (3 elements: embedding, after block 1, after block 2), `layer_r2_shuffled`, `heatmap_layer_position_r2` ($3 times 8$ matrix).
Figure: Stage 3, `layer_position.py`.

*Latent hierarchy* (@latents).
Data: `results/analysis.json` #text[→] `latent_level_r2` (keys `root_L0, mid_L1a, mid_L1b, low_L2a..d`).
Figure: Stage 3, `latent_levels.py`.
The $L=4$ inverted-U preview (`depth4_level_r2.png`, @latents-invu) comes from the depth-4 sweep, Stage 6.

*Blooming* (@blooming, @rootmatch).
Data: `artifacts/probe_data.npz` (`hidden[:,2]`, `beliefs`, `roots`).
Radius numbers ($0.17 -> 0.39$ readout, $0.17 -> 0.47$ true posterior) and root-match rate ($0.47$) are printed by the script itself at generation time (Stage 3, `blooming.py`, writes both `blooming.png` and `blooming_match.png`; stdout prints "radius-from-prior by pos" and "root match rate").
"raw residual PCA — correlation $approx 0.03$ against $approx 0.81$" (the non-blooming control) is *not* persisted by any committed script — it exists only in prior session transcripts and the paper text (a known gap, below).

*Causal steering* (@steering).
Data: `results/analysis.json` #text[→] `steering{alphas, patch_layer, true_lp_steer_true, true_lp_steer_wrong, target_mass_steer_true, target_mass_steer_wrong}`, computed by `analyze.causal_steering()` inside Stage 1's `analyze.py` (patches layer-1 residual by class-mean difference, $alpha in {0, 0.5, 1, 2, 4}$).
Figure: Stage 3, `steering.py`.

*Pre-registration scorecard* (@scorecard).
`PREREGISTRATION.md` records the four predictions, committed before training (honor-code rule); @scorecardtable scores them by hand against `results/analysis.json` — there is no separate `pass1_prereg.json`.
Unlike passes 4–5, this scoring is manual prose, not a script-emitted JSON.

*Non-collapsing belief geometry* (@nccurve, @ncheatmap, @ncattractor).
Data: `results/noncollapse/*/{config.json,analysis.json}` — 27 run dirs (3 skew $times$ 3 ambiguity $times$ 3 rule-table draws), pinned arch.
Per-run `analysis.json` additionally carries a `noncollapse{ambiguity, skew, root_entropy_full_context, reachable_eff_dim, reachable_hull_area}` block (populated only when `g.ambiguity>0` or `g.skew!='none'`).
Regeneration: Stage 7 (`run_noncollapse.py`, then `noncollapse.py` #text[→] figures, `noncollapse_sanity.csv`, `noncollapse_prereg.json`).
"$2.5 times 10^(-16)$" BP/brute-force agreement under both knobs: Stage 0's pytest suite (weighted sum-product tests added alongside the ambiguity/skew knobs).
The two example grammars in @appendix-example-grammars are the pinned dirs `results/noncollapse/skhigh_am0_g00_L2_d128_h4_t4000_s0` and `results/noncollapse/sknone_am0.6_g00_L2_d128_h4_t4000_s0`, hardcoded in `ruletable_graph.py`'s and `ruletable_tables.py`'s `__main__` blocks (Stage 8).

*Generalization across grammars* (@sweeplevels, @sweepbloom, @sweepsteer, @sanitytable).
Data: `results/sweep/g{00..09}_L2_d128_h4_t4000_s{0,1,2}/{config.json,analysis.json}` — 30 runs; `figures/sweep_sanity.csv` columns `grammar,seed,n_layer,n_embd,n_head,test_ce,loss_gap_closed,root_r2,deepest_r2`.
There is no dedicated `run_sweep.py` wrapper (unlike passes 3–5), so Stage 4's 30-cell reproduction is a shell loop over `sweep.py` directly.
Figures/table: `grammar_sweep.py` (globs `results/sweep/g*`).
"$0.89 plus.minus 0.03$" loss-gap-closed, "$0.35 plus.minus 0.07$" root $R^2$ family mean, "$6.9 plus.minus 0.5$" steering collapse are printed to stdout by `grammar_sweep.py`'s `table_sanity()`/`fig_steering()`; mechanically reproducible via `pandas`: `d=pd.read_csv("figures/sweep_sanity.csv"); d[["loss_gap_closed","root_r2"]].agg(["mean","std"])`.
Grammar non-isomorphism check: Stage 9, `grammar_iso.py`.

*Architecture dependence* (@archmarginal, @archdepth, @archlossfit, @archsanitytable).
Data: `results/arch/g{00,01,02}_L{1..4}_d{16,64,128,256}_h{1,2,4,8}_t{4000,16000}_s{0,1,2}/` — 99 runs (11 one-axis-at-a-time configs $times$ 3 grammars $times$ 3 seeds); `figures/arch_sanity.csv` columns `grammar,seed,n_layer,n_embd,n_head,steps,test_ce,loss_gap_closed,root_r2,deepest_r2`.
Regeneration: Stage 5.
All marginal-effect numbers and the $"corr"("loss_gap_closed", R^2)$ values are printed to stdout by `arch_sweep.py`'s `fig_marginal()`/`fig_lossfit()`.

*Going deeper: $L=4$* (@depth4levels, @depth4bloom, @depth4layer, @depth4steer, @depth4sanitytable).
Data: `results/depth4/g{00..09}_L{2,3}_d128_h4_t4000_s{0,1,2}/` — 60 runs; `figures/depth4_sanity.csv` columns `grammar,seed,n_layer,test_ce,loss_gap_closed,root_r2,deepest_r2` (`deepest_r2` = mean over the eight $L_3$ leaf-parent nodes).
Regeneration: Stage 6.
`PREREGISTRATION_L4.md` is scored against `figures/depth4_prereg.json` (`level_means, root_r2_mean_n2/n3, ladder_monotone_fraction, steering_collapse_mean/sd`, printed by `depth4_sweep.py`'s `if __name__` block).

*Root reconstruction diagnostics* (@rootmatch).
Data: `results/root_reconstruction.json` (`meta{N,positions,root_classes,chance_rate,chance_percent}`, `all_positions`, `per_position[]`, `all_but_last`, `full_sequence`; each a `{matches,total,rate,percent}` pair for `readout` vs `exact_posterior_map`).
Regeneration: Stage 1, `root_reconstruction.py` (reads `artifacts/probe_data.npz`; note `blooming_match.png` itself is written by `blooming.py`, Stage 3, not this script).

*Rule table as a graph and example grammars* (@ruletablegraph, @ruletablegraphskew, @ruletablegraphamb).
Regeneration: Stage 8, `ruletable_graph.py` (regenerates all three graph variants in one run: base from `artifacts/`, skew and ambiguous variants from the two pinned `results/noncollapse/...` dirs) and `ruletable_tables.py` (writes the two `.typ` table includes).

*Per-run sanity tables* (@sanitytable, @archsanitytable, @depth4sanitytable, @ncsanitytable).
All four `*_sanity.csv` files are read directly by this document via `csv(...)` — regenerating the CSV in place does not require touching the Typst source.
Regeneration commands are the four aggregator scripts (Stages 4–7): `grammar_sweep.py` (30 rows), `arch_sweep.py` (99 rows), `depth4_sweep.py` (60 rows), `noncollapse.py` (27 rows).

*Known gaps (not mechanically regenerable as of this writing).*
- Raw-residual-PCA "correlation $approx 0.03$ vs $approx 0.81$" (the blooming-section honesty check) is not computed by any committed script; it exists only in prior session transcripts and the paper text.
- Pass 2 (`results/sweep/`) has no dedicated `run_*.py` wrapper analogous to passes 3–5, so its 30-cell reproduction is a shell loop over `sweep.py` rather than a single `uv run python run_sweep.py`.
- The pass-1 @scorecardtable scoring is hand-authored prose against `results/analysis.json`, not emitted by any script (unlike the $L=4$ and non-trivial-grammar scorecards, which have `*_prereg.json`).

