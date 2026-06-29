# Figure: blooming.png  (HEADLINE)
# Message: The belief the model linearly reads out of its final-layer residual
#   stream "blooms" with context position k. With little context (small k) the
#   posterior is near the uniform prior, so the readouts sit in a tight central
#   cluster. As context accumulates the posterior sharpens toward one-hot
#   vertices and the cloud expands outward into petals, one per inferred root.
# Honesty note: raw residual PCA is dominated by token/position nuisance variance
#   and does NOT bloom; the *belief content* of the residual is what blooms. We
#   therefore plot the residual's LINEAR belief readout (a least-squares affine
#   map residual(128) -> belief(8)), which is a faithful linear function of the
#   residual. Measured cloud radius grows 0.17 -> 0.39 with k (true belief: 0.17 -> 0.47).
# Data: artifacts/probe_data.npz: hidden[:, 2, :, :] (final layer), beliefs (true
#   root posteriors for fitting the readout), roots (true root class).
# Type: two scatter panels in a shared belief-PCA(2) basis.
#   Left  colored by context position k (0..7, viridis) -> blooming.
#   Right colored by true root class (8 categories, colorblind) -> petals.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from _style import OUTDIR, PROBE, apply_style
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

apply_style()
d = np.load(PROBE)
H = d["hidden"][:, 2, :, :]            # (N, 8, 128) final layer
B = d["beliefs"]                       # (N, 8, 8) true root posteriors
roots = d["roots"]
N, P, V = B.shape
X = H.reshape(N * P, -1)
Y = B.reshape(N * P, V)
assert X.shape[0] > 0, f"Empty data: {PROBE}"

# linear belief readout of the residual, then a belief-space PCA basis
readout = LinearRegression().fit(X, Y).predict(X)
pca = PCA(n_components=2, random_state=0).fit(Y)   # basis from the true simplex
Z = pca.transform(readout)
pos = np.tile(np.arange(P), N)
root_pt = np.repeat(roots, P)
prior = pca.transform(np.full((1, V), 1.0 / V))[0]      # uniform-prior anchor
rad = np.linalg.norm(Z - prior, axis=1).reshape(N, P).mean(0)
print("blooming: readout cloud", Z.shape, "radius-from-prior by pos", np.round(rad, 3))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)

# draw later (larger-k) points first so the tight small-k core sits on top
ordr = np.argsort(-pos)
sc = ax1.scatter(Z[ordr, 0], Z[ordr, 1], c=pos[ordr], cmap="viridis", s=11,
                 alpha=0.55, edgecolors="none")
cmap = plt.get_cmap("viridis")
for k in range(P):                                       # concentric bloom rings
    circ = plt.Circle(prior, rad[k], fill=False, lw=1.8,
                      color=cmap(k / (P - 1)), alpha=0.9)
    ax1.add_patch(circ)
ax1.plot(*prior, "x", color="k", ms=9, mew=2, label="uniform prior")
ax1.set_title("Belief readout blooms with context position")
ax1.set_xlabel("Belief PC1 (arb. units)")
ax1.set_ylabel("Belief PC2 (arb. units)")
ax1.legend(loc="lower right", fontsize=9)
ax1.set_aspect("equal", adjustable="box")
cb = fig.colorbar(sc, ax=ax1, ticks=range(P))
cb.set_label("Context position $k$  (rings: mean radius)")

pal = sns.color_palette("colorblind", 8)
for c in range(8):
    m = root_pt == c
    ax2.scatter(Z[m, 0], Z[m, 1], color=pal[c], s=11, alpha=0.75,
                edgecolors="none", label=f"{c}")
ax2.set_title("Petals = inferred root class")
ax2.set_xlabel("Belief PC1 (arb. units)")
ax2.legend(title="Root class", ncol=2, markerscale=2, framealpha=0.9,
           loc="best", fontsize=8)

plt.tight_layout()
out = os.path.join(OUTDIR, "blooming.png")
plt.savefig(out)
plt.close()
print("wrote", out, os.path.getsize(out), "bytes")
