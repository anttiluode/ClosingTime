# Gate 2 protocol — causal Q/K routing transplant

Locked before the first real-model intervention run.

## Question

> Does weight-only Q/K support distance predict how safely one head can inherit another head's routing operator?

## Model

Primary model:

- roneneldan/TinyStories-1M

Control:

- a fresh random-initialized copy of the exact architecture.

## Intervention

For target head A and same-layer donor B:

    Q_A <- Q_B
    K_A <- K_B

Leave target V_A and its output path unchanged.

Run the model on the fixed text batch, record mean token-level KL divergence from untouched baseline logits, then restore the original Q/K rows exactly.

This isolates the target head's score/routing operator as much as this simple weight intervention permits.

## Donor sampling

For every target head:

- always include its two nearest support-geometry donors;
- always include its two farthest donors;
- add seeded random remaining donors up to the requested donor count.

Default: six donors per target.

## Mechanistic sanity

After copying donor Q/K into target A, the target's attention map should match the donor's attention map on the same input up to numerical precision.

Required:

    max |attention_target_after_copy - attention_donor| < 1e-5

Failure invalidates the assay.

## Predictors

Rank donor damage using:

1. support distance — mean Q/K Grassmann chordal distance;
2. full score-operator distance — Q^T K matrix distance;
3. raw Q/K distance — gauge-sensitive attacker;
4. attention oracle — actual baseline attention-map distance on the same texts.

## Critical anti-confound

Do not correlate all interventions globally.

For every fixed target head, rank only its measured donors and compute Spearman correlation between predictor distance(target, donor) and KL damage(target <- donor routing). Then aggregate those within-target correlations.

This prevents globally important or fragile targets from manufacturing a spurious geometry effect.

## Locked pass condition

All must hold on the trained model:

    mechanistic attention-copy error            < 1e-5
    median within-target support Spearman       >= .20
    positive support rho                        >= 65% of layer-targets
    median far-donor / near-donor damage        >= 1.25

Pass classification:

HEAD_SUPPORT_GEOMETRY_PREDICTS_CAUSAL_ROUTING_SUBSTITUTABILITY

Otherwise:

HEAD_SUPPORT_GEOMETRY_NOT_CAUSALLY_SUBSTITUTABLE

## Interpretation limit

A pass means geometrically close routing supports are more causally interchangeable under this intervention.

It does not mean the heads have identical semantics, equal importance, or interchangeable V/O write behavior.
