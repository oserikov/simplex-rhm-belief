# Figures

Publication-quality figures for the RHM belief-geometry result. All figures are
generated from real pipeline outputs (`results/analysis.json`,
`artifacts/probe_data.npz`) by the scripts in `scripts/`. Regenerate any figure
with `uv run python paper-typst/figures/scripts/<name>.py`. Shared style lives in
`scripts/_style.py` (seaborn `whitegrid`, paper context, serif, 300 DPI,
colorblind palette).

Defaults of the run: RHM with arity `s=2`, depth `L=3` (context length 8),
vocabulary `v=8`, `m=2` rules per symbol; a 2-layer GPT-2 decoder, `n_embd=128`.
Probe set `N=400`. "Final layer" = residual-stream hidden state index 2.

---

## blooming.png (headline) — 892 KB
**Data:** `probe_data.npz` (`hidden[:,2]`, `beliefs`, `roots`).
**Caption.** The belief the transformer linearly reads out of its final-layer
residual stream blooms with context position. We fit a least-squares affine map
from the 128-d residual to the 8-class root posterior and project the readout
into a belief-space PCA(2) basis (3200 example×position points). *Left:* points
colored by context position k, with concentric rings marking the mean radius of
each position's cloud about the uniform-prior anchor (×). With little context
(k small) the readout sits in a tight central cluster near the prior; as context
accumulates the cloud expands outward (mean radius grows monotonically from 0.17
at k=0 to ~0.39 at k=7, tracking the true posterior's 0.17→0.47). *Right:* the
same cloud colored by ground-truth root class resolves into separated petals.
Raw residual PCA is dominated by token/position nuisance
variance and does not bloom; the *belief content* of the residual is what blooms.

## blooming_match.png — 486 KB
**Data:** `probe_data.npz` (`hidden[:,2]`, `beliefs`, `roots`).
**Caption.** The same belief-readout PCA cloud as `blooming.png`, colored by
whether the inferred root class from the linear readout (`argmax`) matches the
ground-truth root. Green points are matches; red points are mismatches. The
overall match rate across all example-position points is 49%, consistent with
the root posterior being only partially linearly decoded.

## posterior_simplex.png — 489 KB
**Data:** `probe_data.npz` (`beliefs`, `hidden[:,2]`).
**Caption.** The residual stream is an (affine) image of the exact Bayesian
belief simplex. *Left:* PCA(2) of the exact root posteriors over the probe set;
the posterior lives on a discrete, structured set, with small-k points (dark)
near the prior and large-k points (bright) spread toward simplex vertices.
*Right:* the linear probe's predicted posteriors projected into the *same* PCA
basis, colored identically by context position. The probe recovers the same
region and the same position gradient, though noisily — the held-out root
posterior R² is ≈0.38, so the affine correspondence is real but imperfect for
the global root.

## layer_position.png — 228 KB
**Data:** `analysis.json` (`layer_r2`, `layer_r2_shuffled`,
`heatmap_layer_position_r2`).
**Caption.** Belief decodability accumulates across depth and across context.
*Left:* linear-probe R² for the root posterior rises layer by layer
(−0.00 → 0.15 → 0.38) while a shuffled-label control stays at ≈0, confirming the
signal is genuine. *Right:* R²(layer, position) heatmap. Early context positions
are decodable even at shallow layers, and the final layer pushes decodability
deep into the sequence (positions 0–3 reach R²=1.0). Colors are clipped to
[−1, 1] for legibility, but annotations show raw values; the largest negative
cell is shown in scientific notation.

## latent_levels.png — 130 KB
**Data:** `analysis.json` (`latent_level_r2`).
**Caption.** The residual stream encodes the whole latent hierarchy, and deeper,
more local latents are generally more strongly encoded than the global root.
Linear-probe R² is lowest for the root (L0, 0.38 — the spec's primary target,
highlighted), higher for the level-1 mid latents (0.51, 0.59), and generally
higher for the four deepest level-2 latents (0.61, 0.66, 0.50, 0.66). Local
structure near the leaves is easier to read off the residual than the global
class, with visible node-level variation.

## steering.png — 240 KB
**Data:** `analysis.json` (`steering`).
**Caption.** The belief state is causally used, not merely decodable. Adding
α·(belief direction) to the layer-1 residual toward the *true* latent leaves
predictions untouched (green: the model already holds that belief), whereas
steering toward a *wrong* latent collapses the mean log-probability of the true
next token (left, orange: −0.73 → −8.5) and re-routes probability mass onto the
wrong latent's children (right, orange: 0.22 → 0.53). Intervening on the decoded
belief axis changes behavior in the predicted direction.

## loss_curve.png — 104 KB
**Data:** `probe_data.npz` (`final_test_loss`, `uniform_baseline`, `floor`).
**Caption.** The transformer learned the true RHM predictor. Test cross-entropy
(0.891 nats/token) sits far below the uniform baseline (ln 8 = 2.079) and close
to the Bayes-optimal floor (0.725, the mean per-position posterior entropy over
the seven predicted positions), closing ≈88% of the uniform→Bayes gap.
