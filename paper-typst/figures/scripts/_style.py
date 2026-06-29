"""Shared figure style + data paths for all RHM belief-geometry figures."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = "/Users/oleg/simplex-briefing/simplex-rhm-belief"
PROBE = os.path.join(ROOT, "artifacts/probe_data.npz")
ANALYSIS = os.path.join(ROOT, "results/analysis.json")
OUTDIR = os.path.join(ROOT, "paper-typst/figures")


def apply_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.family": "serif",
    })
