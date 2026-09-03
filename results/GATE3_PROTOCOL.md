# Gate 3 protocol — dense causal replication

Designed **after** Gate 2 failed its locked nearest/farthest threshold. Gate 2 remains a failure and its threshold is not changed.

## Why another gate is justified

Gate 2 nevertheless exposed three preregistered-compatible facts:

- copying Q/K exactly copied the donor attention pattern;
- trained support distance had a positive within-target association with causal damage (median rho +0.257);
- fresh random initialization had median rho 0.000;
- the full gauge-invariant score operator was stronger (+0.357).

The failed nearest/farthest summary used only six donors per target. Gate 3 asks a new confirmatory question with the complete same-layer donor set.

## Question

> Is there a reproducible weight-only geometry of causal routing substitution, and does the full score operator contain reliably more causal information than support geometry alone?

## Models

Run unchanged on both independently trained checkpoints:

- `roneneldan/TinyStories-1M`
- `roneneldan/TinyStories-Instruct-1M`

For each model also run a fresh random-initialized copy of the exact architecture.

## Intervention

For every layer, every target head A, and **all 15 other same-layer donor heads B**:

    Q_A <- Q_B
    K_A <- K_B

Keep A's V/O write path unchanged.

Measure final-model token KL against the untouched baseline, then restore Q/K.

Thus each trained model contributes:

    8 layers x 16 targets x 15 donors = 1,920 interventions

## Reliability

Use 16 fixed texts.

For every intervention compute KL per text, then independently average:

- texts 1-8;
- texts 9-16.

Within every fixed target head, correlate the 15 donor damages from the two halves.

The primary causal ranking metric is considered reliable only if pooled median split-half Spearman is >= .50.

## Predictors

Within each fixed target head, rank all 15 donors by:

1. **support** — mean Q/K Grassmann chordal distance;
2. **full score operator** — cosine distance between `Q_h^T K_h`;
3. **raw Q/K** — gauge-sensitive coordinate attacker;
4. **attention oracle** — actual baseline attention-map distance on the same 16 texts.

The causal target is donor-induced final-model KL.

## Donor-label permutation test

For each predictor, independently permute donor damage labels **within every fixed target head**, recompute all target-level Spearman correlations, and record the pooled median.

Use 256 seeded controls.

This preserves:

- each target's damage distribution;
- layer identity;
- target importance/fragility;
- predictor geometry.

It destroys only the donor correspondence the hypothesis needs.

## Locked support replication criterion

For **each trained model separately**, require:

    median damage split-half reliability          >= .50
    median support -> damage rho                 >= .15
    support donor-permutation p                  <= .01
    trained - random-init support rho            >= .10
    positive support rho                         >= 60% of layer-targets

Only if both models pass is support called reproducibly causal.

## Locked full-operator criterion

For **each trained model separately**, require:

    median full-score -> damage rho              >= .20
    full-score donor-permutation p               <= .01
    trained - random-init full-score rho          >= .10

If both support and full score pass, call the full operator reliably richer only if:

    full-score median rho - support median rho   >= .05

in **both** trained models.

## Classifications

If support and full score replicate and the full operator is richer:

`REPLICABLE_CAUSAL_ROUTING_GEOMETRY_FULL_OPERATOR_RICHER`

If support replicates but the richer criterion fails:

`REPLICABLE_CAUSAL_SUPPORT_GEOMETRY`

If support fails but the full score operator replicates:

`FULL_SCORE_OPERATOR_ONLY_CAUSAL_SIGNAL`

Otherwise:

`NO_REPLICABLE_WEIGHT_ONLY_CAUSAL_ROUTING_GEOMETRY`

## Interpretation limit

A positive result concerns **routing substitution under Q/K transplantation**.

It does not imply semantic identity, interchangeable V/O writes, global head redundancy, or safe pruning. Those require new interventions.
