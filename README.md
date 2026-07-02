# Simplex RHM Belief

This repository contains the experiments and paper for studying whether a tiny transformer trained on Recursive Hidden Markov Model (RHM) emissions represents the ground-truth posterior over the RHM latent state.

## Paper

The compiled paper is at [`paper-typst/main.pdf`](paper-typst/main.pdf).
The Typst source is at [`paper-typst/main.typ`](paper-typst/main.typ).

The reproducibility guide is included in the paper itself, in the reproducibility appendix.
It documents the end-to-end pipeline, figure generation stages, and the final `ninja paper` compile step.

## Summary

I took RHMs data generation and trained a tiny transformer on emitted sequences, then studied whether the residual stream carries the ground truth posteriors about the RHM latent and where in the network this information lives, what geometry it traces, and whether the model relies on it.

I found that unlike in results of Shai et al. (2026), linear probes manage to achieve only *partial* recovery of the ground truth posteriors, with the core exception being an intentionally naive experiment where knowing the position of the sentence clearly disambiguated the latent.
Each layer of the model populates the residual stream with useful information about the ground truth posteriors.
As context grows, the probe's predicted posteriors bloom outward from the prior toward simplex vertices.
Causal steering confirms the model relies on it rather than merely exposing it.
The phenomenology replicates across the RHM grammars of the same size, and one level deeper.
It is also robust to model capacity.
Finally, breaking the grammar's invertibility with *ambiguity* leaves a genuinely uncertain belief state, and there the probe does not degrade.
It seems to sharpen, but this might be due to the ground truth posteriors being less peaky, which distracts the chosen probe evaluation metrics.

## Rebuilding The Paper

The default build target recompiles the paper PDF:

```bash
ninja paper
```

That runs:

```bash
typst compile --ignore-system-fonts paper-typst/main.typ paper-typst/main.pdf
```

Figure and table regeneration is separate from the Typst compile.
See the reproducibility guide in the paper for the full sequence of commands.

## Files To Know

- `paper-typst/main.pdf` - compiled paper
- `paper-typst/main.typ` - paper source and embedded reproducibility guide
- `paper-typst/figures/` - figures, generated tables, and figure-generation scripts
- `rhm.py` - RHM sampler and exact belief propagation
- `train.py` - tiny transformer training
- `analyze.py` - probes, geometry analysis, and causal steering
- `sweep.py`, `run_arch.py`, `run_depth4.py`, `run_noncollapse.py` - experiment runners
- `build.ninja` - paper build target
- `flake.nix`, `.envrc`, `pyproject.toml` - development environment and Python dependencies
