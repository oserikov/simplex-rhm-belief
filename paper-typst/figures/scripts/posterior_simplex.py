# Figure: posterior_simplex.png
# Message: The residual stream is an affine image of the exact Bayesian belief
#   simplex. PCA of the TRUE root posteriors and PCA-projection of the linear
#   probe's PREDICTED posteriors share the same 2-D shape, colored identically
#   by context position -> the probe recovers the simplex geometry.
# Data: artifacts/probe_data.npz. beliefs[:, :, :] (N x 8 pos x 8 classes) true
#   root posteriors; hidden[:, 2, :, :] final-layer residuals as probe inputs.
#   A linear (least-squares) probe maps residual(128) -> belief(8); predictions
#   are projected with the SAME PCA basis fit on the true posteriors.
# Type: two scatter panels (PC1 vs PC2), colored by context position k (viridis).
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
import numpy as np
from _style import OUTDIR, PROBE, apply_style
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

apply_style()
d = np.load(PROBE)
B = d["beliefs"]                       # (N, 8, 8) true posteriors
H = d["hidden"][:, 2, :, :]            # (N, 8, 128) final layer
N, P, V = B.shape
Ytrue = B.reshape(N * P, V)
X = H.reshape(N * P, -1)
assert Ytrue.shape[0] > 0, f"Empty data: {PROBE}"
print("simplex: true beliefs", Ytrue.shape, "residuals", X.shape)

# linear probe residual -> belief (fit on all probe points; this is geometry, not eval)
probe = LinearRegression().fit(X, Ytrue)
Ypred = probe.predict(X)

# shared PCA basis fit on the TRUE simplex
pca = PCA(n_components=2, random_state=0).fit(Ytrue)
Zt = pca.transform(Ytrue)
Zp = pca.transform(Ypred)
verts = pca.transform(np.eye(V))
prior = pca.transform(np.full((1, V), 1.0 / V))[0]

pos = np.tile(np.arange(P), N)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
for ax, Z, ttl in [(ax1, Zt, "Ground-truth posterior"),
                   (ax2, Zp, "Linear probe of ROOT posterior")]:
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=pos, cmap="viridis", s=18, alpha=0.75,
                    edgecolors="none")
    ax.scatter(verts[:, 0], verts[:, 1], marker="o", s=150, facecolors="none",
               edgecolors="black", linewidths=1.0, zorder=6,
               label="certainty / simplex vertices")
    ax.scatter([prior[0]], [prior[1]], marker="P", s=130, c="crimson",
               edgecolors="white", linewidths=1.0, zorder=6, label="uniform prior")
    ax.set_title(ttl)
    ax.set_xlabel("PC1 (arb. units)")
ax1.set_ylabel("PC2 (arb. units)")
ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax2.text(0.03, 0.03, "held-out root R² ≈ 0.38", transform=ax2.transAxes,
         fontsize=9, va="bottom", ha="left",
         bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
cb = fig.colorbar(sc, ax=[ax1, ax2], ticks=range(P), fraction=0.046, pad=0.04)
cb.set_label("Context position $k$")

out = os.path.join(OUTDIR, "posterior_simplex.png")
plt.savefig(out)
plt.close()
print("wrote", out, os.path.getsize(out), "bytes")
