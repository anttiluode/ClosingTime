# Gate 1 protocol — decompose the REAL10 behavioral signal

Locked before the first ClosingTime real-model run.

## Question

OutoTesti REAL10 found that within-layer Q/K head-subspace distance predicts distance between actual head attention maps.

Gate 1 asks:

> Which static property of Q/K weights carries that association?

## Models

Run unchanged on:

- roneneldan/TinyStories-1M
- roneneldan/TinyStories-Instruct-1M

For each, also create a fresh random-initialized copy of the exact architecture.

## Behavior target

Use the same fixed 24 story-like texts as REAL10.

For every layer/head, collect causal attention probabilities, flatten valid token-pair entries across texts, center each head vector, and compute centered-cosine head distance.

Split texts 12/12 and independently verify that the behavioral distance is reliable.

## Weight predictors

For head blocks Q_h and K_h and score operator M_h = Q_h^T K_h / sqrt(d_head):

1. **support** — mean normalized Grassmann chordal distance for Q and K row spaces;
2. **score operator** — cosine distance between flattened M_h matrices;
3. **spectrum** — distance between normalized singular-value spectra of M_h;
4. **raw Q/K** — cosine distance between flattened raw Q/K blocks, retained as a gauge-sensitive attacker.

All primary correlations use the 120 head-pair distances in one 16-head layer. Significance is tested by permuting head labels on the behavioral distance matrix.

## Locked reproduction gate

Before interpreting decomposition, REAL10 must reproduce:

    median behavior split-half reliability      >= .50
    median trained support->behavior r           >= .20
    positive support correlation                 >= 12 / 16 layers
    permutation p < .05                          >= 8 / 16 layers
    trained - random-init median support r       >= .15

If it does not, classification is:

REAL10_SUPPORT_SIGNAL_DID_NOT_REPRODUCE

and no decomposition story is promoted.

## Locked interpretation

If reproduction passes and

    support median r - spectrum median r >= .15
    score median r - support median r    <= .15

classify:

SUPPORT_GEOMETRY_CAPTURES_MOST_BEHAVIORAL_WEIGHT_SIGNAL

If support beats spectrum by >= .15 but the full score operator exceeds support by > .15:

SUPPORT_MATTERS_BUT_FULL_SCORE_OPERATOR_IS_RICHER

Otherwise:

SPECTRUM_OR_OTHER_LOW_ORDER_STRUCTURE_EXPLAINS_SUPPORT_SIGNAL

These labels are descriptive. Gate 1 does not establish causality.
