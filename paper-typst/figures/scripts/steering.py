# Figure: steering.png
# Message: The belief direction is CAUSALLY used, not merely decodable. Adding
#   alpha * (belief axis) at layer 1 toward the TRUE latent leaves predictions
#   untouched (model already believes it); steering toward a WRONG latent
#   collapses log p(true next token) and re-routes probability mass onto the
#   wrong latent's children.
# Data: results/analysis.json: steering.{alphas, true_lp_steer_true,
#   true_lp_steer_wrong, target_mass_steer_true, target_mass_steer_wrong}.
# Type: 2 panels vs steering strength alpha.
#   (left)  mean log p(true next token); (right) P(next token in target children).
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from _style import ANALYSIS, OUTDIR, apply_style

apply_style()
with open(ANALYSIS) as _f:
    A = json.load(_f)
S = A["steering"]
a = np.array(S["alphas"])
assert a.size > 0, "Empty steering data"
pal = sns.color_palette("colorblind")
print("steering alphas", a, "patch_layer", S["patch_layer"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

ax1.plot(a, S["true_lp_steer_true"], "-o", color=pal[2], lw=2.2, ms=8,
         label="steer → true latent")
ax1.plot(a, S["true_lp_steer_wrong"], "-s", color=pal[3], lw=2.2, ms=8,
         label="steer → wrong latent")
ax1.set_xlabel(r"Steering strength $\alpha$")
ax1.set_ylabel("Mean log $p$(true next token)")
ax1.set_title("Wrong-belief steering destroys the prediction")
ax1.legend(loc="lower left")

ax2.plot(a, S["target_mass_steer_true"], "-o", color=pal[2], lw=2.2, ms=8,
         label="steer → true latent")
ax2.plot(a, S["target_mass_steer_wrong"], "-s", color=pal[3], lw=2.2, ms=8,
         label="steer → wrong latent")
ax2.set_xlabel(r"Steering strength $\alpha$")
ax2.set_ylabel("P(next token ∈ target latent's children)")
ax2.set_ylim(0, 1)
ax2.set_title("Mass re-routes toward the injected latent")
ax2.legend(loc="best")

fig.suptitle("The belief state is causally used, not just decodable",
             fontsize=15, y=1.02)
plt.tight_layout()
out = os.path.join(OUTDIR, "steering.png")
plt.savefig(out)
plt.close()
print("wrote", out, os.path.getsize(out), "bytes")
