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
)
from closingtime.hf_tools import (
    attention_behavior_distances,
    baseline_logits,
    load_tokenizer,
    mean_token_kl,
    model_num_heads,
    qk_layers,
    token_batch,
)
from closingtime.stats import masked_within_target_spearman
from common import TEXTS


def load_pair(model_name: str, seed: int):
    import torch
    from transformers import AutoModelForCausalLM

    trained = AutoModelForCausalLM.from_pretrained(model_name)
    trained.eval()
    torch.manual_seed(seed)
    fresh = AutoModelForCausalLM.from_config(trained.config)
    fresh.eval()
    return trained, fresh


def donor_set(D: np.ndarray, target: int, count: int, rng: np.random.Generator) -> list[int]:
    n = D.shape[0]
    candidates = np.asarray([h for h in range(n) if h != target], dtype=int)
    order = candidates[np.argsort(D[target, candidates])]
    chosen: list[int] = []
    for h in list(order[:2]) + list(order[-2:]):
        if int(h) not in chosen:
            chosen.append(int(h))
    remaining = [int(h) for h in candidates if int(h) not in chosen]
    rng.shuffle(remaining)
    for h in remaining:
        if len(chosen) >= min(count, n - 1):
            break
        chosen.append(h)
    return chosen[: min(count, n - 1)]


def run_condition(model, tokenizer, *, texts: list[str], donor_count: int, seed: int, max_length: int) -> dict:
    import torch

    num_heads = model_num_heads(model)
    layers = qk_layers(model)
    batch = token_batch(tokenizer, texts, model, max_length=max_length)
    base_logits = baseline_logits(model, batch)
    behavior = attention_behavior_distances(model, batch)

    rows = []
    verify_error = None
    rng = np.random.default_rng(seed)

    for layer, item in layers.items():
        q0_t = item.q_param.detach().clone()
        k0_t = item.k_param.detach().clone()
        q0 = q0_t.float().cpu().numpy().astype(float)
        k0 = k0_t.float().cpu().numpy().astype(float)
        d_head = q0.shape[0] // num_heads

        predictors = {
            "support": qk_support_distance_matrix(q0, k0, num_heads=num_heads),
            "score_operator": score_operator_distance_matrix(q0, k0, num_heads=num_heads),
            "raw_qk": raw_qk_distance_matrix(q0, k0, num_heads=num_heads),
            "attention_oracle": behavior[layer],
        }
        damage = np.full((num_heads, num_heads), np.nan, dtype=float)
        near_damage = []
        far_damage = []

        for target in range(num_heads):
            donors = donor_set(predictors["support"], target, donor_count, rng)
            ranked = [h for h in range(num_heads) if h != target]
            ranked.sort(key=lambda h: predictors["support"][target, h])
            near = set(ranked[:2])
            far = set(ranked[-2:])
            ts = slice(target*d_head, (target+1)*d_head)
            for donor in donors:
                ds = slice(donor*d_head, (donor+1)*d_head)
                with torch.no_grad():
                    item.q_param[ts].copy_(q0_t[ds])
                    item.k_param[ts].copy_(k0_t[ds])
                    out = model(**batch, use_cache=False, return_dict=True)
                    kl = mean_token_kl(base_logits, out.logits.detach(), batch["attention_mask"])
                    item.q_param[ts].copy_(q0_t[ts])
                    item.k_param[ts].copy_(k0_t[ts])
                damage[target, donor] = kl
                if donor in near:
                    near_damage.append(kl)
                if donor in far:
                    far_damage.append(kl)

                if verify_error is None:
                    with torch.no_grad():
                        item.q_param[ts].copy_(q0_t[ds])
                        item.k_param[ts].copy_(k0_t[ds])
                        check = model(**batch, output_attentions=True, use_cache=False, return_dict=True)
                        item.q_param[ts].copy_(q0_t[ts])
                        item.k_param[ts].copy_(k0_t[ts])
                    A = check.attentions[layer].detach().float().cpu().numpy()
                    verify_error = float(np.max(np.abs(A[:, target] - A[:, donor])))

        row = {
            "layer": int(layer),
            "measured_interventions": int(np.sum(np.isfinite(damage))),
            "median_near_damage": float(np.median(near_damage)),
            "median_far_damage": float(np.median(far_damage)),
            "far_over_near": float(np.median(far_damage) / max(np.median(near_damage), 1e-15)),
        }
        for name, D in predictors.items():
            rho = masked_within_target_spearman(D, damage)
            row[f"{name}_median_target_rho"] = float(np.nanmedian(rho))
            row[f"{name}_positive_targets"] = int(np.sum(rho > 0))
            row[f"{name}_target_rhos"] = [None if not np.isfinite(x) else float(x) for x in rho]
        rows.append(row)
        print(
            f"layer {layer}: support rho={row['support_median_target_rho']:+.3f} "
            f"score={row['score_operator_median_target_rho']:+.3f} "
            f"oracle={row['attention_oracle_median_target_rho']:+.3f} "
            f"far/near={row['far_over_near']:.2f}"
        )

    def median(key):
        return float(np.median([r[key] for r in rows]))
    aggregate = {
        "layers": len(rows),
        "routing_copy_max_attention_error": verify_error,
        "median_support_target_rho": median("support_median_target_rho"),
        "median_score_operator_target_rho": median("score_operator_median_target_rho"),
        "median_raw_qk_target_rho": median("raw_qk_median_target_rho"),
        "median_attention_oracle_target_rho": median("attention_oracle_median_target_rho"),
        "median_far_over_near": median("far_over_near"),
        "support_positive_layer_targets": int(sum(r["support_positive_targets"] for r in rows)),
        "total_layer_targets": int(len(rows) * num_heads),
    }
    return {"aggregate": aggregate, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roneneldan/TinyStories-1M")
    ap.add_argument("--texts", type=int, default=8)
    ap.add_argument("--donors", type=int, default=6)
    ap.add_argument("--max-length", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "gate2" / "audit.json")
    args = ap.parse_args()

    trained, fresh = load_pair(args.model, args.seed)
    tokenizer = load_tokenizer(args.model)
    texts = TEXTS[: args.texts]

    print("TRAINED")
    tr = run_condition(trained, tokenizer, texts=texts, donor_count=args.donors, seed=310000, max_length=args.max_length)
    print("\nRANDOM INIT")
    ri = run_condition(fresh, tokenizer, texts=texts, donor_count=args.donors, seed=410000, max_length=args.max_length)

    ta = tr["aggregate"]
    causal_pass = (
        ta["routing_copy_max_attention_error"] is not None
        and ta["routing_copy_max_attention_error"] < 1e-5
        and ta["median_support_target_rho"] >= 0.20
        and ta["support_positive_layer_targets"] / ta["total_layer_targets"] >= 0.65
        and ta["median_far_over_near"] >= 1.25
    )
    if causal_pass:
        classification = "HEAD_SUPPORT_GEOMETRY_PREDICTS_CAUSAL_ROUTING_SUBSTITUTABILITY"
    else:
        classification = "HEAD_SUPPORT_GEOMETRY_NOT_CAUSALLY_SUBSTITUTABLE"

    summary = {
        "experiment": "GATE2",
        "model": args.model,
        "question": "Does weight-only Q/K support distance predict damage when one head inherits another head's routing operator?",
        "intervention": (
            "For target head A, copy donor B's Q and K row blocks into A while leaving A's V/O path untouched; "
            "measure final-model token KL, then restore weights."
        ),
        "controls": {
            "score_operator": "full gauge-invariant Q^T K operator distance",
            "raw_qk": "gauge-sensitive flattened Q/K distance",
            "attention_oracle": "actual baseline attention-map distance on the same texts",
            "random_initialization": "fresh exact architecture",
            "within_target": "all correlations rank donors only within a fixed target head",
        },
        "trained": tr,
        "random_initialization": ri,
        "classification": classification,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\nGATE2 trained aggregate", json.dumps(ta, indent=2))
    print("GATE2 init aggregate", json.dumps(ri["aggregate"], indent=2))
    print("classification:", classification)


if __name__ == "__main__":
    main()
