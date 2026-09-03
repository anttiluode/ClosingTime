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
    model_num_heads,
    qk_layers,
    token_batch,
    token_kl_by_sequence,
)
from closingtime.stats import masked_within_target_spearman, spearman
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
    torch.manual_seed(int(seed))
    fresh = AutoModelForCausalLM.from_config(trained.config)
    fresh.eval()
    return trained, fresh


def target_reliability(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.shape[0]
    out = np.full(n, np.nan, dtype=float)
    for target in range(n):
        keep = (
            np.isfinite(a[target])
            & np.isfinite(b[target])
            & (np.arange(n) != target)
        )
        if np.sum(keep) >= 3:
            out[target] = spearman(a[target, keep], b[target, keep])
    return out


def donor_permutation_test(
    layer_payload: list[dict],
    predictor_name: str,
    *,
    controls: int,
    seed: int,
) -> dict:
    observed_rhos = []
    for payload in layer_payload:
        observed_rhos.extend(
            masked_within_target_spearman(
                payload["predictors"][predictor_name],
                payload["damage"],
            ).tolist()
        )
    observed = float(np.nanmedian(np.asarray(observed_rhos, dtype=float)))

    rng = np.random.default_rng(seed)
    null = np.empty(controls, dtype=float)
    for c in range(controls):
        rhos = []
        for payload in layer_payload:
            D = payload["predictors"][predictor_name]
            damage = payload["damage"]
            n = D.shape[0]
            for target in range(n):
                donors = np.asarray([h for h in range(n) if h != target], dtype=int)
                shuffled = rng.permutation(donors)
                rhos.append(spearman(D[target, donors], damage[target, shuffled]))
        null[c] = float(np.median(rhos))

    return {
        "observed_median_rho": observed,
        "null_median": float(np.median(null)),
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null, ddof=1)) if controls > 1 else 0.0,
        "p_upper": float((1 + np.sum(null >= observed)) / (controls + 1)),
    }


def run_condition(
    model,
    tokenizer,
    *,
    texts: list[str],
    controls: int,
    seed: int,
    max_length: int,
) -> dict:
    import torch

    num_heads = model_num_heads(model)
    layers = qk_layers(model)
    batch = token_batch(tokenizer, texts, model, max_length=max_length)
    base_logits = baseline_logits(model, batch)
    behavior = attention_behavior_distances(model, batch)
    half = len(texts) // 2

    rows = []
    payloads = []
    verify_error = None

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
        damage_a = np.full_like(damage, np.nan)
        damage_b = np.full_like(damage, np.nan)

        for target in range(num_heads):
            ts = slice(target * d_head, (target + 1) * d_head)
            for donor in range(num_heads):
                if donor == target:
                    continue
                ds = slice(donor * d_head, (donor + 1) * d_head)

                with torch.no_grad():
                    item.q_param[ts].copy_(q0_t[ds])
                    item.k_param[ts].copy_(k0_t[ds])

                    if verify_error is None:
                        out = model(
                            **batch,
                            output_attentions=True,
                            use_cache=False,
                            return_dict=True,
                        )
                        A = out.attentions[layer].detach().float().cpu().numpy()
                        verify_error = float(
                            np.max(np.abs(A[:, target] - A[:, donor]))
                        )
                    else:
                        out = model(**batch, use_cache=False, return_dict=True)

                    item.q_param[ts].copy_(q0_t[ts])
                    item.k_param[ts].copy_(k0_t[ts])

                per_text = token_kl_by_sequence(
                    base_logits,
                    out.logits.detach(),
                    batch["attention_mask"],
                )
                damage[target, donor] = float(np.mean(per_text))
                damage_a[target, donor] = float(np.mean(per_text[:half]))
                damage_b[target, donor] = float(np.mean(per_text[half:]))

        reliability = target_reliability(damage_a, damage_b)
        row = {
            "layer": int(layer),
            "median_damage_split_rho": float(np.nanmedian(reliability)),
            "positive_damage_split_targets": int(np.sum(reliability > 0)),
            "median_intervention_kl": float(np.nanmedian(damage)),
        }
        for name, D in predictors.items():
            rhos = masked_within_target_spearman(D, damage)
            row[f"{name}_median_target_rho"] = float(np.nanmedian(rhos))
            row[f"{name}_positive_targets"] = int(np.sum(rhos > 0))
        rows.append(row)
        payloads.append({"predictors": predictors, "damage": damage})

        print(
            f"layer {layer}: rel={row['median_damage_split_rho']:+.3f} "
            f"support={row['support_median_target_rho']:+.3f} "
            f"score={row['score_operator_median_target_rho']:+.3f} "
            f"raw={row['raw_qk_median_target_rho']:+.3f} "
            f"oracle={row['attention_oracle_median_target_rho']:+.3f}"
        )

    permutation = {}
    for i, name in enumerate(
        ("support", "score_operator", "raw_qk", "attention_oracle")
    ):
        permutation[name] = donor_permutation_test(
            payloads,
            name,
            controls=controls,
            seed=seed + i * 100000,
        )

    total_targets = len(rows) * num_heads

    def all_target_rhos(name: str) -> np.ndarray:
        vals = []
        for payload in payloads:
            vals.extend(
                masked_within_target_spearman(
                    payload["predictors"][name],
                    payload["damage"],
                ).tolist()
            )
        return np.asarray(vals, dtype=float)

    reliability_vals = []
    for payload in payloads:
        # Reliability was already computed before payload storage; reconstructing
        # it is unnecessary. Pull from rows at the layer level for the locked
        # median-of-targets approximation is not precise, so retain below from
        # a second lightweight pass using stored half matrices would cost memory.
        pass

    # Exact target-level reliability was used per layer; pool layer medians only
    # for reporting would hide weak layers. Instead use the median of the eight
    # layer target-medians as a conservative scalar gate.
    aggregate = {
        "layers": len(rows),
        "num_heads": int(num_heads),
        "total_layer_targets": int(total_targets),
        "routing_copy_max_attention_error": verify_error,
        "median_damage_split_reliability": float(
            np.median([r["median_damage_split_rho"] for r in rows])
        ),
        "rows": rows,
        "permutation": permutation,
    }

    for name in ("support", "score_operator", "raw_qk", "attention_oracle"):
        vals = all_target_rhos(name)
        aggregate[f"median_{name}_target_rho"] = float(np.nanmedian(vals))
        aggregate[f"positive_{name}_targets"] = int(np.sum(vals > 0))
        aggregate[f"positive_{name}_fraction"] = float(
            np.mean(vals[np.isfinite(vals)] > 0)
        )
        aggregate[f"{name}_permutation_p"] = float(
            permutation[name]["p_upper"]
        )

    aggregate["score_minus_support"] = float(
        aggregate["median_score_operator_target_rho"]
        - aggregate["median_support_target_rho"]
    )
    return aggregate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", type=int, default=16)
    ap.add_argument("--controls", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "gate3" / "audit.json",
    )
    args = ap.parse_args()

    if args.texts < 8 or args.texts % 2:
        raise ValueError("--texts must be an even integer >= 8")
    texts = TEXTS[: args.texts]

    results = {}
    per_model_flags = {}

    for mi, model_name in enumerate(MODELS):
        print(f"\nMODEL {model_name}")
        trained, fresh = load_pair(model_name, args.seed + mi)
        tokenizer = load_tokenizer(model_name)

        print("TRAINED")
        tr = run_condition(
            trained,
            tokenizer,
            texts=texts,
            controls=args.controls,
            seed=500000 + mi * 10000,
            max_length=args.max_length,
        )
        print("RANDOM INIT")
        ri = run_condition(
            fresh,
            tokenizer,
            texts=texts,
            controls=args.controls,
            seed=700000 + mi * 10000,
            max_length=args.max_length,
        )

        support_pass = (
            tr["routing_copy_max_attention_error"] is not None
            and tr["routing_copy_max_attention_error"] < 1e-5
            and tr["median_damage_split_reliability"] >= 0.50
            and tr["median_support_target_rho"] >= 0.15
            and tr["support_permutation_p"] <= 0.01
            and (
                tr["median_support_target_rho"]
                - ri["median_support_target_rho"]
            ) >= 0.10
            and tr["positive_support_fraction"] >= 0.60
        )
        score_pass = (
            tr["routing_copy_max_attention_error"] is not None
            and tr["routing_copy_max_attention_error"] < 1e-5
            and tr["median_damage_split_reliability"] >= 0.50
            and tr["median_score_operator_target_rho"] >= 0.20
            and tr["score_operator_permutation_p"] <= 0.01
            and (
                tr["median_score_operator_target_rho"]
                - ri["median_score_operator_target_rho"]
            ) >= 0.10
        )

        per_model_flags[model_name] = {
            "support_pass": bool(support_pass),
            "score_pass": bool(score_pass),
            "score_richer_by_0p05": bool(
                tr["score_minus_support"] >= 0.05
            ),
        }
        results[model_name] = {
            "trained": tr,
            "random_initialization": ri,
        }

        print("MODEL SUMMARY")
        print(
            f"  trained support rho={tr['median_support_target_rho']:+.3f} "
            f"p={tr['support_permutation_p']:.4f} "
            f"init={ri['median_support_target_rho']:+.3f}"
        )
        print(
            f"  trained score rho={tr['median_score_operator_target_rho']:+.3f} "
            f"p={tr['score_operator_permutation_p']:.4f} "
            f"init={ri['median_score_operator_target_rho']:+.3f}"
        )
        print(
            f"  score-support={tr['score_minus_support']:+.3f} "
            f"reliability={tr['median_damage_split_reliability']:+.3f}"
        )

    support_all = all(
        per_model_flags[m]["support_pass"] for m in MODELS
    )
    score_all = all(
        per_model_flags[m]["score_pass"] for m in MODELS
    )
    richer_all = all(
        per_model_flags[m]["score_richer_by_0p05"] for m in MODELS
    )

    if support_all and score_all and richer_all:
        classification = (
            "REPLICABLE_CAUSAL_ROUTING_GEOMETRY_FULL_OPERATOR_RICHER"
        )
    elif support_all:
        classification = "REPLICABLE_CAUSAL_SUPPORT_GEOMETRY"
    elif score_all:
        classification = "FULL_SCORE_OPERATOR_ONLY_CAUSAL_SIGNAL"
    else:
        classification = "NO_REPLICABLE_WEIGHT_ONLY_CAUSAL_ROUTING_GEOMETRY"

    summary = {
        "experiment": "GATE3",
        "models": list(MODELS),
        "texts": texts,
        "controls": int(args.controls),
        "design_note": (
            "Designed after Gate 2's locked nearest/farthest failure; "
            "Gate 2 remains failed. Gate 3 uses all 15 donors per target."
        ),
        "per_model_flags": per_model_flags,
        "classification": classification,
        "results": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nGATE3 flags")
    print(json.dumps(per_model_flags, indent=2))
    print("classification:", classification)


if __name__ == "__main__":
    main()
