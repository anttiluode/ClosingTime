from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from closingtime.geometry import (
    qk_support_distance_matrix,
    raw_qk_distance_matrix,
    score_operator_distance_matrix,
    score_spectrum_distance_matrix,
)
from closingtime.hf_tools import (
    attention_behavior_distances,
    load_tokenizer,
    model_num_heads,
    qk_numpy,
    token_batch,
)
from closingtime.stats import distance_correlation, label_permutation_test
from common import TEXTS

MODELS = (
    "roneneldan/TinyStories-1M",
    "roneneldan/TinyStories-Instruct-1M",
)


def load_pair(model_name: str, seed: int):
    import torch
    from transformers import AutoModelForCausalLM

    trained = AutoModelForCausalLM.from_pretrained(model_name)
    trained.eval()
    torch.manual_seed(seed)
    fresh = AutoModelForCausalLM.from_config(trained.config)
    fresh.eval()
    return trained, fresh


def one_model(model, tokenizer, *, controls: int, seed: int, max_length: int) -> dict:
    num_heads = model_num_heads(model)
    qk = qk_numpy(model)
    batch = token_batch(tokenizer, TEXTS, model, max_length=max_length)
    behavior = attention_behavior_distances(model, batch)
    half_a = token_batch(tokenizer, TEXTS[:12], model, max_length=max_length)
    half_b = token_batch(tokenizer, TEXTS[12:], model, max_length=max_length)
    beh_a = attention_behavior_distances(model, half_a)
    beh_b = attention_behavior_distances(model, half_b)

    rows = []
    for layer in sorted(qk):
        Wq, Wk = qk[layer]
        predictors = {
            "support": qk_support_distance_matrix(Wq, Wk, num_heads=num_heads),
            "score_operator": score_operator_distance_matrix(Wq, Wk, num_heads=num_heads),
            "spectrum": score_spectrum_distance_matrix(Wq, Wk, num_heads=num_heads),
            "raw_qk": raw_qk_distance_matrix(Wq, Wk, num_heads=num_heads),
        }
        rel = distance_correlation(beh_a[layer], beh_b[layer])
        row = {"layer": int(layer), "behavior_reliability": float(rel)}
        for j, (name, D) in enumerate(predictors.items()):
            test = label_permutation_test(
                D,
                behavior[layer],
                controls=controls,
                seed=seed + layer * 1000 + j,
            )
            row[f"{name}_r"] = float(test["observed"])
            row[f"{name}_p"] = float(test["p_upper"])
        rows.append(row)
        print(
            f"layer {layer}: rel={rel:+.3f} "
            f"support={row['support_r']:+.3f} "
            f"score={row['score_operator_r']:+.3f} "
            f"spectrum={row['spectrum_r']:+.3f} raw={row['raw_qk_r']:+.3f}"
        )

    agg = {"layers": len(rows)}
    for key in ("behavior_reliability", "support_r", "score_operator_r", "spectrum_r", "raw_qk_r"):
        vals = np.asarray([r[key] for r in rows], dtype=float)
        agg[f"median_{key}"] = float(np.median(vals))
        if key.endswith("_r"):
            agg[f"positive_{key}"] = int(np.sum(vals > 0))
    for key in ("support", "score_operator", "spectrum", "raw_qk"):
        agg[f"p_lt_0p05_{key}"] = int(np.sum([r[f"{key}_p"] < 0.05 for r in rows]))
    return {"aggregate": agg, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "gate1" / "audit.json")
    args = ap.parse_args()

    results = {}
    trained_rows = []
    init_rows = []
    for mi, name in enumerate(MODELS):
        print(f"\nMODEL {name}")
        trained, fresh = load_pair(name, args.seed + mi)
        tokenizer = load_tokenizer(name)
        print("TRAINED")
        tr = one_model(trained, tokenizer, controls=args.controls, seed=100000 + mi*10000, max_length=args.max_length)
        print("RANDOM INIT")
        ri = one_model(fresh, tokenizer, controls=args.controls, seed=200000 + mi*10000, max_length=args.max_length)
        results[name] = {"trained": tr, "random_initialization": ri}
        trained_rows.extend(tr["rows"])
        init_rows.extend(ri["rows"])

    def med(rows, key):
        return float(np.median([r[key] for r in rows]))

    aggregate = {
        "trained_layers": len(trained_rows),
        "median_behavior_reliability": med(trained_rows, "behavior_reliability"),
        "trained_median_support_r": med(trained_rows, "support_r"),
        "trained_median_score_operator_r": med(trained_rows, "score_operator_r"),
        "trained_median_spectrum_r": med(trained_rows, "spectrum_r"),
        "trained_median_raw_qk_r": med(trained_rows, "raw_qk_r"),
        "init_median_support_r": med(init_rows, "support_r"),
        "support_minus_spectrum": med(trained_rows, "support_r") - med(trained_rows, "spectrum_r"),
        "score_minus_support": med(trained_rows, "score_operator_r") - med(trained_rows, "support_r"),
        "trained_support_positive": int(np.sum([r["support_r"] > 0 for r in trained_rows])),
        "trained_support_p_lt_0p05": int(np.sum([r["support_p"] < 0.05 for r in trained_rows])),
    }

    reproduction = (
        aggregate["median_behavior_reliability"] >= 0.5
        and aggregate["trained_median_support_r"] >= 0.2
        and aggregate["trained_support_positive"] >= 12
        and aggregate["trained_support_p_lt_0p05"] >= 8
        and aggregate["trained_median_support_r"] - aggregate["init_median_support_r"] >= 0.15
    )
    if not reproduction:
        classification = "REAL10_SUPPORT_SIGNAL_DID_NOT_REPRODUCE"
    elif aggregate["support_minus_spectrum"] >= 0.15 and aggregate["score_minus_support"] <= 0.15:
        classification = "SUPPORT_GEOMETRY_CAPTURES_MOST_BEHAVIORAL_WEIGHT_SIGNAL"
    elif aggregate["support_minus_spectrum"] >= 0.15:
        classification = "SUPPORT_MATTERS_BUT_FULL_SCORE_OPERATOR_IS_RICHER"
    else:
        classification = "SPECTRUM_OR_OTHER_LOW_ORDER_STRUCTURE_EXPLAINS_SUPPORT_SIGNAL"

    summary = {
        "experiment": "GATE1",
        "question": "What part of Q/K weights explains the REAL10 head-behavior geometry?",
        "predictors": {
            "support": "mean Q/K Grassmann chordal distance (basis/gauge invariant)",
            "score_operator": "cosine distance between full gauge-invariant Q^T K score operators",
            "spectrum": "distance between normalized singular spectra of Q^T K",
            "raw_qk": "cosine distance between flattened raw Q/K blocks (gauge-sensitive attacker)",
        },
        "aggregate": aggregate,
        "classification": classification,
        "models": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nGATE1", json.dumps(aggregate, indent=2))
    print("classification:", classification)


if __name__ == "__main__":
    main()
