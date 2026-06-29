# Figure appendix data: root reconstruction rates from the linear belief readout.
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from _style import PROBE, ROOT
from sklearn.linear_model import LinearRegression


def rate_payload(matches, total):
    rate = float(matches / total)
    return {"matches": int(matches), "total": int(total), "rate": rate, "percent": 100.0 * rate}


d = np.load(PROBE)
hidden = d["hidden"][:, 2, :, :]  # final-layer residual stream
beliefs = d["beliefs"]
roots = d["roots"]
N, P, V = beliefs.shape

X = hidden.reshape(N * P, -1)
Y = beliefs.reshape(N * P, V)
readout = LinearRegression().fit(X, Y).predict(X).reshape(N, P, V)
inferred = readout.argmax(axis=-1)
readout_match = inferred == roots[:, None]

exact_map = beliefs.argmax(axis=-1)
exact_match = exact_map == roots[:, None]

per_position = []
for k in range(P):
    per_position.append({
        "k": k,
        "observed_prefix_len": k + 1,
        "readout": rate_payload(readout_match[:, k].sum(), N),
        "exact_posterior_map": rate_payload(exact_match[:, k].sum(), N),
    })

out = {
    "meta": {
        "N": int(N),
        "positions": int(P),
        "root_classes": int(V),
        "chance_rate": 1.0 / V,
        "chance_percent": 100.0 / V,
    },
    "all_positions": rate_payload(readout_match.sum(), N * P),
    "per_position": per_position,
    "all_but_last": per_position[P - 2],
    "full_sequence": per_position[P - 1],
}

path = os.path.join(ROOT, "results/root_reconstruction.json")
with open(path, "w") as f:
    json.dump(out, f, indent=2)
    f.write("\n")

print(json.dumps({
    "all_positions": out["all_positions"],
    "all_but_last": out["all_but_last"],
    "full_sequence": out["full_sequence"],
    "chance_percent": out["meta"]["chance_percent"],
}, indent=2))
print("wrote", path)
