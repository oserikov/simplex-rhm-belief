"""Probe + figures: residual stream -> belief simplex.

1. Fit one global linear (affine) probe mapping residual activations to the
   exact root-class posterior. Report R² / accuracy vs. a shuffled-label
   baseline.
2. Visualize the activation cloud (PCA / direct affine projection) showing
   "blooming" with context position and layer-wise accumulation.
3. One extra analysis (default: causal steering -- patch the residual along
   the probe's belief direction and verify the next-token distribution shifts
   toward the implied latent).

TODO: implement probe fitting, figures, and the steering analysis.
"""
