# ClosingTime

**Every new beginning comes from some other beginning's end.**

ClosingTime begins exactly where [OutoTesti](https://github.com/anttiluode/OutoTesti) stopped.

The hidden-tree story died. The general graph story died. One object survived repeated spectrum, gauge, initialization, replication, and behavioral attacks:

> **The geometry of the residual-stream subspaces read by attention heads through Q/K weights predicts similarity of their observed attention behavior.**

Across two TinyStories 1M checkpoints, OutoTesti REAL10 measured a median weight-geometry -> attention-behavior correlation of **+0.385**, positive in **15/16** model-layers. Fresh random initialization was approximately zero.

ClosingTime asks what becomes possible after that result.

## Prior art: the geometry itself is not new

Principal-angle / projection-kernel geometry between attention-head weight subspaces is not ours.

Yamagiwa, Takase & Shimodaira (2026), *Measuring Affinity between Attention-Head Weight Subspaces via the Projection Kernel*, studies basis-invariant attention-head subspace affinity in GPT-2:

- https://arxiv.org/abs/2601.10266

TransformerLens added a maintained Projection Kernel implementation in August 2026:

- https://github.com/TransformerLensOrg/TransformerLens/issues/1720
- https://github.com/TransformerLensOrg/TransformerLens/pull/1721

Their primitive and ours meet at the same invariant. For orthonormal bases U and V,

    PK(U,V) = ||U^T V||_F^2 = sum_i cos(theta_i)^2

For equal rank k, the normalized chordal distance used here is

    d_chord(U,V) = sqrt((k - PK(U,V)) / k)

So this repo does **not** claim to have discovered principal angles between heads.

The question is stronger and more operational:

> **What does this geometry predict, what does it survive, and can geometric neighborhoods be used causally?**

## The object

For one attention head, split Q and K into head blocks Q_h and K_h. The bilinear attention-score operator is

    M_h = Q_h^T K_h / sqrt(d_head)

This gives a useful interpretation of the OutoTesti survivor:

- rowspace(Q_h) is the left/output support available to M_h;
- rowspace(K_h) is the right/input support available to M_h;
- those supports are invariant to ordinary within-head Q/K orthogonal basis changes;
- pairwise Q/K support distances give a data-free description of how differently two heads can route similarity through residual space.

REAL10 says that this static support geometry contains information about the geometry of the heads' actual attention maps on text.

That is the starting fact. It is not yet causality, semantics, importance, redundancy, or a pruning theorem.

## Gate 1 — what part of the weights carries the behavioral signal?

REAL10 compared one weight-space metric with one behavioral metric. Gate 1 decomposes that association.

For every head pair in every layer it compares four predictors of attention-map distance:

1. **SUPPORT** — mean Q/K Grassmann chordal distance. Basis/gauge invariant.
2. **FULL SCORE OPERATOR** — cosine distance between M_h matrices. Gauge invariant and retains orientation plus singular values.
3. **SPECTRUM ONLY** — distance between singular-value spectra of M_h. Singular vectors are removed.
4. **RAW Q/K** — cosine distance between flattened raw Q/K blocks. Deliberately gauge sensitive.

The point is not to make support win. The point is to learn what REAL10 actually measured.

A strong support result says that *where a head can read in residual space* carries behavioral information beyond low-order spectral structure. A full-score win says support is useful but incomplete. A spectrum win demotes the story to a much simpler explanation.

See [Gate 1 protocol](results/GATE1_PROTOCOL.md).

## Gate 2 — can a geometric neighbor substitute for a head's routing?

This is the first causal gate.

For target head A and same-layer donor B, perform:

    before:
      A = (Q_A, K_A, V_A, O_A)
      B = (Q_B, K_B, V_B, O_B)

    intervention:
      A' = (Q_B, K_B, V_A, O_A)

Only the target head's **routing / score operator** is transplanted. Its value/output write path remains its own.

Then run the actual language model and measure final token-distribution damage relative to the untouched model.

The sharp prediction is:

    small support distance(A,B)
                |
                v
    smaller causal damage when B's routing replaces A's

Donors are ranked within each fixed target head, so a globally fragile or important target cannot manufacture the result.

Attackers include:

- full score-operator distance;
- raw Q/K distance;
- actual attention-map distance on the same text, as a data-dependent oracle;
- fresh random initialization;
- nearest-vs-farthest donor contrast;
- a mechanistic check that copied Q/K makes the target attention map match the donor attention map.

If this passes, “nearby heads” gets an operational meaning:

> **Near in weight-only support geometry implies more interchangeable routing.**

If it fails, REAL10 remains a descriptive behavioral association and we do not pretend otherwise.

See [Gate 2 protocol](results/GATE2_PROTOCOL.md).

## What comes after Gate 2

Only follow the branch the measurements earn.

If causal substitutability passes:

1. **Geometry-guided routing sharing** — share Q/K routing among nearest-neighbor heads and compare equal-budget random grouping.
2. **Redundancy without data** — ask whether a weight-only geometry predicts which routing operators can be tied with little damage.
3. **Cross-layer atlas** — move from within-layer constellations to the whole model.
4. **Developmental geometry** — inspect training checkpoints and measure splitting, convergence, neighbor exchange and stabilization of head subspaces.
5. **Cross-seed correspondence** — try to match functionally analogous heads between independently trained models without assuming head-index alignment.

If causal substitutability fails, Gate 1 tells us what richer object deserves promotion: the full score operator, spectrum, activation-conditioned geometry, or nothing.

## Why a new repo?

OutoTesti was a falsification chase around hidden trees and graph-derived transformer weights. Its stopping line is useful precisely because it stays closed.

ClosingTime starts with the survivor as an independent object:

    Q/K weights
        |
        v
    gauge-invariant support subspaces
        |
        v
    head-to-head geometry
        |
        v
    observed attention behavior          [already measured]
        |
        v
    causal routing substitutability      [current test]
        |
        v
    sharing / pruning / development      [only if earned]

No tree is required anywhere in this repo.

## Lightweight tests

    python -m pip install -e .[dev]
    pytest -q

## Real gates

    python -m pip install -e .[models]
    python experiments/gate1_decompose_behavior.py
    python experiments/gate2_qk_transplant.py

The first real-model gates intentionally stay on TinyStories 1M before scaling.

## Scientific limit

Even a successful Gate 2 would not mean two heads have identical semantic functions. Heads are context-dependent and multifunctional; support overlap is a coarse static property. Nor would it establish that support geometry is the best pruning score or that it generalizes to every attention architecture.

The useful claim, if earned, is narrower:

> **Weight-only, gauge-invariant Q/K support geometry predicts a measurable property of routing behavior and the causal cost of substituting one routing operator for another.**

That is already enough to test hard.
