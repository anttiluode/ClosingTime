from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QKLayer:
    layer: int
    q_name: str
    k_name: str
    q_param: object
    k_param: object


def model_num_heads(model) -> int:
    for name in ("num_heads", "num_attention_heads", "n_head"):
        value = getattr(model.config, name, None)
        if value is not None:
            return int(value)
    raise ValueError("could not infer number of attention heads")


def qk_layers(model) -> dict[int, QKLayer]:
    """Find GPT-Neo/GPT-2-style square q_proj/k_proj matrices by layer.

    ClosingTime deliberately starts narrow rather than pretending every HF
    architecture has the same weight layout.
    """
    found: dict[int, dict[str, tuple[str, object]]] = {}
    for name, parameter in model.named_parameters():
        if getattr(parameter, "ndim", None) != 2:
            continue
        role = None
        if name.endswith(".q_proj.weight"):
            role = "Q"
        elif name.endswith(".k_proj.weight"):
            role = "K"
        if role is None:
            continue
        marker = ".h."
        if marker not in name:
            continue
        try:
            layer = int(name.split(marker, 1)[1].split(".", 1)[0])
        except ValueError:
            continue
        found.setdefault(layer, {})[role] = (name, parameter)

    out: dict[int, QKLayer] = {}
    for layer, roles in found.items():
        if "Q" in roles and "K" in roles:
            q_name, q = roles["Q"]
            k_name, k = roles["K"]
            if tuple(q.shape) != tuple(k.shape):
                raise ValueError(f"layer {layer}: Q/K shapes differ")
            out[layer] = QKLayer(layer, q_name, k_name, q, k)
    if not out:
        raise ValueError("no .q_proj/.k_proj attention matrices found")
    return dict(sorted(out.items()))


def qk_numpy(model) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for layer, item in qk_layers(model).items():
        q = item.q_param.detach().float().cpu().numpy().astype(float)
        k = item.k_param.detach().float().cpu().numpy().astype(float)
        out[layer] = (q, k)
    return out


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def token_batch(tokenizer, texts: list[str], model, *, max_length: int = 64):
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    return {k: v.to(device) for k, v in batch.items()}


def centered_cosine_distance(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=1, keepdims=True)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)
    sim = np.clip(X @ X.T, -1.0, 1.0)
    D = 0.5 * (1.0 - sim)
    np.fill_diagonal(D, 0.0)
    return D


def attention_behavior_distances(model, batch) -> dict[int, np.ndarray]:
    import torch

    with torch.no_grad():
        out = model(
            **batch,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    if out.attentions is None:
        raise RuntimeError("model did not return attention maps")

    mask = batch["attention_mask"].bool().cpu().numpy()
    B, T = mask.shape
    causal = np.tril(np.ones((T, T), dtype=bool))
    valid = mask[:, :, None] & mask[:, None, :] & causal[None]

    result = {}
    for layer, tensor in enumerate(out.attentions):
        A = tensor.detach().float().cpu().numpy()
        if A.ndim != 4:
            raise RuntimeError(f"unexpected attention tensor {A.shape}")
        heads = A.shape[1]
        X = np.empty((heads, int(valid.sum())), dtype=float)
        for h in range(heads):
            X[h] = A[:, h][valid]
        result[layer] = centered_cosine_distance(X)
    return result


def baseline_logits(model, batch):
    import torch

    with torch.no_grad():
        return model(**batch, use_cache=False, return_dict=True).logits.detach()


def mean_token_kl(reference_logits, candidate_logits, attention_mask) -> float:
    import torch
    import torch.nn.functional as F

    ref_logp = F.log_softmax(reference_logits.float(), dim=-1)
    cand_logp = F.log_softmax(candidate_logits.float(), dim=-1)
    ref_p = ref_logp.exp()
    kl = torch.sum(ref_p * (ref_logp - cand_logp), dim=-1)
    valid = attention_mask.bool()
    return float(kl[valid].mean().item())


def token_kl_by_sequence(reference_logits, candidate_logits, attention_mask) -> np.ndarray:
    """Mean token KL for each sequence independently."""
    import torch
    import torch.nn.functional as F

    ref_logp = F.log_softmax(reference_logits.float(), dim=-1)
    cand_logp = F.log_softmax(candidate_logits.float(), dim=-1)
    ref_p = ref_logp.exp()
    kl = torch.sum(ref_p * (ref_logp - cand_logp), dim=-1)
    mask = attention_mask.bool()
    num = (kl * mask).sum(dim=1)
    den = mask.sum(dim=1).clamp_min(1)
    return (num / den).detach().cpu().numpy().astype(float)
