# Pre-registered prediction

**Committed before any model is trained or any probe/result is computed.** This
is the honor-code artifact: the geometric intuition and light-math predictions
are recorded here, timestamped in git, ahead of looking at results.

Setup: RHM with arity `s=2`, depth `L=3` (context length `d = s**L = 8`),
per-level vocabulary `v=8`, `m=2` rules per symbol, uniform rule choice,
unambiguous rules. A 2-layer `GPT2LMHeadModel` (dropout off) is trained by
next-token prediction. We then fit a single global linear/affine probe from the
residual stream to the **exact** Bayesian posterior over the root class
(computed by sum-product belief propagation on the tree given a leaf prefix).

## What we predict

### 1. The posterior is linearly decodable from the residual stream
Because next-token prediction on a hierarchical grammar requires tracking the
distribution over hidden latents (the optimal predictor IS the Bayesian belief
state), the transformer should represent the belief state in a way a linear
probe can read out. Prediction: **R² (residual → root posterior simplex) well
above a shuffled-label baseline**, rising across context positions and reaching
its maximum at the last residual layer. Concretely we expect R² > 0.5 at the
final position/last layer, with the shuffled baseline near 0.

### 2. Geometry: a simplex that "blooms" with context position
At position 1 (one leaf observed) the posterior is close to the prior — the
belief points sit near the **center** (barycenter) of the probability simplex.
As more leaves are observed (k grows toward 8), the posteriors **sharpen**,
moving from the center out toward the **vertices** (near-certain root class).
Projected into 2–3 PCA components of the residual stream, the cloud of belief
points should trace a low-dimensional image of the simplex that **expands /
blooms** from a tight central blob (early positions) to a wide spread reaching
toward vertices (late positions). With v=8 root classes the limiting object is
the image of a 7-simplex; in 2D PCA we expect a roughly polygonal/star spread.

Light math: the entropy of the exact root posterior should **decrease
monotonically (in expectation) with k**, and the radial distance of activations
from the cloud centroid should **increase with k**, mirroring posterior
sharpening. We pre-register a measurable monotone trend in both.

### 3. Layer-wise additive accumulation
The residual stream is a sum of contributions across layers. We predict the
belief representation is **built up additively**: probe R² should be low at the
embedding layer (index 0), higher after layer 1, highest after layer 2, i.e.
**monotonically increasing with depth**. Each layer adds belief-relevant
signal rather than overwriting it.

### 4. Correlation structure (why hierarchy matters)
RHM leaves carry **power-law (algebraic) token-token correlations vs. distance**,
unlike a Markov chain's exponential decay. This is the structural reason a flat
n-gram baseline is beaten and why deep/hierarchical aggregation helps: distant
tokens remain informative about shared ancestors.

### 5. Causal relevance (the chosen extra analysis)
We pick **causal steering** because it distinguishes a *used* representation
from a merely *decodable* one. Prediction: patching the final-layer residual
along the probe's direction for root-class `c` should shift the model's
next-token distribution toward leaves consistent with `c`. We expect a positive,
ordered effect (larger patch magnitude → larger shift), not just decodability.

## Alternative geometries we might instead see (pre-registered alternatives)
- **Trivial/degenerate:** if L=3 is too easy, posteriors may collapse to vertices
  almost immediately (little blooming visible) — mitigation noted in spec: bump
  L to 4. We will report if blooming is too sharp to see.
- **Non-linear encoding:** the posterior could be present but only non-linearly
  decodable, giving low linear R² despite the information being there. We'd see
  this as linear probe R² near baseline while next-token loss is well below
  uniform.
- **Mean-only / collapsed simplex:** the residual might encode only the MAP class
  (argmax) rather than the full distribution — visible as activations clustering
  at vertices with no graded interior, and good classification accuracy but poor
  full-simplex R².
- **No layer accumulation:** belief signal could appear fully at layer 1 and stay
  flat, rather than accumulating — we'd report a flat R²-vs-depth curve.

## Falsification
The core claim is **falsified** if linear probe R² (residual → exact root
posterior) is not clearly above the shuffled-label baseline at the final layer,
or if the activation geometry shows no relationship to posterior sharpening.
