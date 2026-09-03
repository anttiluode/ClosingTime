# Gate 1 receipt — support geometry carries most of the REAL10 signal

Run: GitHub Actions **First real gates** run `33708441615`, job `gate1`.

The locked REAL10 reproduction gate passed on the same two independently trained TinyStories 1M checkpoints.

```text
trained model-layers                         16
median behavior split-half reliability      +0.9631
median support -> behavior r                 +0.3847
positive support correlation                 15 / 16
support permutation p < .05                  13 / 16
random-init median support r                 -0.0027
```

The decomposition was:

```text
Q/K support subspaces                        +0.3847
full Q^T K score operator                    +0.4338
singular spectrum only                       +0.1646
raw Q/K coordinates                          +0.0013
```

Thus:

```text
support - spectrum                           +0.2201
full score operator - support                +0.0490
```

Locked classification:

```text
SUPPORT_GEOMETRY_CAPTURES_MOST_BEHAVIORAL_WEIGHT_SIGNAL
```

## Interpretation

The REAL10 association is not primarily explained by the singular-value spectrum and disappears almost completely in raw coordinate distance.

Most of the static weight signal predicting which heads exhibit similar attention maps lives in the basis-invariant Q/K support subspaces: the residual-stream directions available to the head's score operator.

The full gauge-invariant score operator still performs somewhat better, so support is not the whole routing computation.

This earns a sharper object, not causality:

> **where a head can read in residual space carries most of the weight-only signal predicting how it attends.**
