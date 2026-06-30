"""Quick-failure checkpoints for the RHM sampler + belief propagation."""

import numpy as np
import pytest

from rhm import (
    Grammar,
    brute_force_belief_node,
    brute_force_belief_root,
)


def test_sample_length_and_unambiguity():
    g = Grammar.random(s=2, L=3, v=8, m=2, seed=0)
    rng = np.random.default_rng(0)
    x, latents = g.sample(rng)
    assert len(x) == g.d == 2**3 == 8
    g.assert_unambiguous()  # raises if ambiguous
    # latents: one array per level 0..L
    assert len(latents) == g.L + 1
    assert latents[0].shape == (1,)
    assert latents[-1].shape == (8,)


def test_belief_is_distribution():
    g = Grammar.random(s=2, L=3, v=8, m=2, seed=2)
    rng = np.random.default_rng(3)
    x, _ = g.sample(rng)
    for k in range(1, g.d + 1):
        b = g.belief_root(x, k)
        assert np.all(b >= -1e-12)
        assert abs(b.sum() - 1.0) < 1e-9


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_bp_matches_brute_force_root(seed):
    """Belief propagation root posterior == brute-force enumeration (< 1e-6)."""
    g = Grammar.random(s=2, L=3, v=8, m=2, seed=seed)
    rng = np.random.default_rng(100 + seed)
    # test on several real samples, every prefix length
    for _ in range(5):
        x, _ = g.sample(rng)
        for k in range(1, g.d + 1):
            bp = g.belief_root(x, k)
            bf = brute_force_belief_root(g, x, k)
            assert np.max(np.abs(bp - bf)) < 1e-6, (seed, k, bp, bf)


def test_bp_matches_brute_force_midlevel():
    """Full sum-product marginal at a mid-level latent matches brute force."""
    g = Grammar.random(s=2, L=3, v=8, m=2, seed=7)
    rng = np.random.default_rng(7)
    x, _ = g.sample(rng)
    for k in range(1, g.d + 1):
        # level 1 has 2 nodes
        for pos in range(2):
            bp = g.belief_node(x, k, ell=1, pos=pos)
            bf = brute_force_belief_node(g, x, k, ell=1, pos=pos)
            assert np.max(np.abs(bp - bf)) < 1e-6, (k, pos)


def test_belief_node_root_consistency():
    """belief_node at the root equals belief_root."""
    g = Grammar.random(s=2, L=3, v=8, m=2, seed=9)
    rng = np.random.default_rng(9)
    x, _ = g.sample(rng)
    for k in range(1, g.d + 1):
        a = g.belief_root(x, k)
        b = g.belief_node(x, k, ell=0, pos=0)
        assert np.max(np.abs(a - b)) < 1e-9


@pytest.mark.parametrize("ambiguity", [0.3, 0.6])
@pytest.mark.parametrize("skew", ["mid", "high"])
def test_bp_matches_brute_force_generalized(ambiguity, skew):
    """Weighted BP == weighted brute force under BOTH ambiguity and skew (<1e-6)."""
    g = Grammar.random(s=2, L=3, v=8, m=2, seed=1, ambiguity=ambiguity, skew=skew)
    rng = np.random.default_rng(50)
    for _ in range(3):
        x, _ = g.sample(rng)
        for k in range(1, g.d + 1):
            assert np.max(np.abs(g.belief_root(x, k) - brute_force_belief_root(g, x, k))) < 1e-6
            for pos in range(2):
                bp = g.belief_node(x, k, ell=1, pos=pos)
                bf = brute_force_belief_node(g, x, k, ell=1, pos=pos)
                assert np.max(np.abs(bp - bf)) < 1e-6, (k, pos)


def test_noncollapse_invariant_and_monotonicity():
    """rho=0 keeps canonical rules + ~0 full-context entropy; rho>0 raises it."""
    g0 = Grammar.random(seed=0)
    g0n = Grammar.random(seed=0, ambiguity=0.0, skew="none")
    assert all(np.array_equal(a, b) for a, b in zip(g0.rules, g0n.rules, strict=True))

    def mean_full_entropy(amb):
        g = Grammar.random(seed=0, ambiguity=amb)
        rng = np.random.default_rng(3)
        es = [-(b := g.belief_root(g.sample(rng)[0], g.d)) @ np.log(b + 1e-12) for _ in range(40)]
        return float(np.mean(es))

    e0, e3, e6 = mean_full_entropy(0.0), mean_full_entropy(0.3), mean_full_entropy(0.6)
    assert e0 < 1e-6 < e3 < e6  # collapse at rho=0, strictly rising with rho


def test_weights_sum_to_one():
    """enumerate_all weights are a proper distribution under skew."""
    g = Grammar.random(s=2, L=3, v=8, m=2, seed=4, skew="high")
    _, _, w = g.enumerate_all()
    assert abs(w.sum() - 1.0) < 1e-9
    gu = Grammar.random(s=2, L=3, v=8, m=2, seed=4)  # skew=none -> uniform weights
    _, _, wu = gu.enumerate_all()
    assert np.allclose(wu, wu[0])


def test_tiny_tree_full_posterior():
    """On a tiny tree, full prefix pins the root to a near-one-hot posterior."""
    g = Grammar.random(s=2, L=2, v=4, m=2, seed=1)
    leaves, roots, _ = g.enumerate_all()
    x = leaves[0]
    b = g.belief_root(x, g.d)
    bf = brute_force_belief_root(g, x, g.d)
    assert np.max(np.abs(b - bf)) < 1e-6
    # observing all leaves must be consistent with the true root of that tree
    assert b[roots[0]] > 0
