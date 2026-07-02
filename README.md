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

## Development

<details>
  <summary>System setup (one-time)</summary>

```bash
# Install Nix
sh <(curl -L https://nixos.org/nix/install) --daemon
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf
exit # the commands below need a fresh shell

# Install direnv
nix profile install nixpkgs#direnv nixpkgs#nix-direnv

# Install direnv shell hook
if [[ "$SHELL" == *"/zsh" ]]; then
    echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
elif [[ "$SHELL" == *"/bash" ]]; then
    echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
elif [[ "$SHELL" == *"/fish" ]]; then
    echo 'direnv hook fish | source' >> ~/.config/fish/config.fish
else
    echo "Can't set up direnv hook for your $SHELL, please set it up manually"
fi

# Setup nix-direnv
mkdir -p ~/.config/direnv
echo 'source $HOME/.nix-profile/share/nix-direnv/direnvrc' >> ~/.config/direnv/direnvrc

# Install pre-commit
nix profile install nixpkgs#pre-commit
```

</details>

### Project setup

After cloning, copy `.env.example` to `.env` and fill in any keys.

```bash
direnv allow
pre-commit install && pre-commit run --all-files
```

### Daily workflow

```bash
cd simplex-rhm-belief
uv run my_script.py # run Python scripts
uv add requests     # add dependencies
ninja               # builds the paper (paper-typst/main.typ -> paper-typst/main.pdf)
git commit          # checks format, lints, and type checks via pre-commit
```

### Files to know

- `paper-typst/main.pdf` - compiled paper
- `paper-typst/main.typ` - paper source and embedded reproducibility guide
- `paper-typst/figures/` - figures, generated tables, and figure-generation scripts
- `rhm.py` - RHM sampler and exact belief propagation
- `train.py` - tiny transformer training
- `analyze.py` - probes, geometry analysis, and causal steering
- `sweep.py`, `run_arch.py`, `run_depth4.py`, `run_noncollapse.py` - experiment runners
- `flake.nix` - system dependencies (uv, typst, ninja, nix tools)
- `pyproject.toml` - Python dependencies and ruff config
- `.envrc` - direnv config that activates nix + uv
- `.pre-commit-config.yaml` - commit hooks (ruff, ty, nixfmt, uv-lock)
- `build.ninja` - build targets (`ninja paper` compiles the Typst paper)
- `.github/workflows/paper.yml` - CI: builds paper, uploads as release on main
- `experiment/run_experiment.sh` - containerized experiment runner (devcontainers + Claude Code agents)
- `experiment/.devcontainer/` - Dockerfile, firewall, permission bypass for agent containers
- `experiment/AGENT_PROMPT.md.template` - prompt template with `{{TASK_ID}}` and `{{CONDITION}}` placeholders

### Rebuilding the paper

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

### Running experiments

```bash
# 1. Define tasks (one ID per line)
echo -e "task1\ntask2\ntask3" > experiment/tasks.txt

# 2. Place data per condition
mkdir -p experiment/data/my_condition
cp my_resources.txt experiment/data/my_condition/

# 3. Customize the prompt template
vim experiment/AGENT_PROMPT.md.template

# 4. Run
./experiment/run_experiment.sh --conditions my_condition --budget 50

# 5. Results are in experiment/results/my_condition/*.jsonl
```

Each agent runs in a firewalled Docker container with full tool access (bash, Python, file I/O).
