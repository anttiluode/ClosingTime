# Gate 2 receipt — sparse causal routing transplant

Run: GitHub Actions **First real gates** run `33708441615`, job `gate2`.

Gate 2 copied a same-layer donor head's Q/K rows into a target head while leaving the target's V/O write path untouched.

The mechanistic sanity check was exact:

```text
max target-vs-donor attention error after Q/K copy   0.0
```

So the intervention really did replace the target head's routing/attention pattern with the donor's.

## Trained TinyStories-1M

Across targets, donors were ranked **within each fixed target head**. This prevents target importance/fragility from manufacturing the association.

```text
median support-distance -> KL-damage Spearman       +0.2571
median full-score-operator -> damage Spearman       +0.3571
median raw-Q/K -> damage Spearman                    +0.0429
median attention-map oracle -> damage Spearman       +0.9000

positive support correlations                        85 / 128 targets
median far-donor / near-donor damage                 1.1697x
```

Fresh random initialization:

```text
median support Spearman                              +0.0000
median full-score Spearman                           -0.0286
median raw-Q/K Spearman                              -0.0286
```

The locked pass required:

```text
attention-copy error < 1e-5                          PASS
median support rho >= .20                            PASS
positive support rho >= 65%                          PASS (66.4%)
far / near damage >= 1.25x                           FAIL (1.17x)
```

Locked classification:

```text
HEAD_SUPPORT_GEOMETRY_NOT_CAUSALLY_SUBSTITUTABLE
```

The threshold is **not** changed after seeing the result.

## What the failure exposed

The continuous within-target causal signal is training-dependent and nonzero, but the preregistered nearest-vs-farthest contrast was not strong enough.

More importantly, the full `Q^T K` score operator predicted transplant damage better than support alone.

The next confirmatory assay therefore does not rescue Gate 2. It asks a new question using **all 15 donors per target**, both trained checkpoints, split-text reliability, donor-label permutations, and fresh-init controls:

> Is there a reproducible weight-only causal routing geometry at all, and if so is the full score operator reliably richer than its support subspaces?
