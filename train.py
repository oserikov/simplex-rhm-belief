"""Train a tiny decoder-only transformer on RHM samples (next-token CE).

Default: 2 layers, d_model ~64-128, 2-4 heads, context = d = s ** L.
Device: PyTorch MPS with CPU fallback.

Saves model weights and residual-stream activations (hidden states for every
layer) over a held-out probe set, paired with their exact belief states from
``rhm.belief_propagation`` for the downstream linear probe.

TODO: implement the model, training loop, baselines (uniform / flat n-gram),
and activation dumping.
"""
