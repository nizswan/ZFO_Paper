"""AdamW, cosine, Prodigy, Muon, and sequential ZFO baselines."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

INITIAL_MODELS = {"t2", "t3", "p2", "p3"}
_GOLDEN_RATIO_CONJUGATE = (math.sqrt(5.0) - 1.0) / 2.0


@dataclass
class SelectionResult:
    selected_a: float
    probes_used: int
    evaluated: List[Tuple[float, float]]
    fallback: bool = False
    safeguard: bool = False
    initial_proposal: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


def _clip(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def _finite(x: float) -> bool:
    return math.isfinite(float(x))


def _best_evaluated(
    evaluated: Sequence[Tuple[float, float]], beta: float
) -> Tuple[float, float]:
    valid = [
        (float(a), float(y))
        for a, y in evaluated
        if 0.0 <= float(a) <= float(beta) and _finite(y)
    ]
    if not valid:
        raise RuntimeError("No finite evaluated point in [0, beta].")
    return max(valid, key=lambda pair: pair[1])


def _already_evaluated(
    a: float, evaluated: Sequence[Tuple[float, float]], tol: float
) -> bool:
    return any((abs(float(a) - float(x)) <= float(tol) for x, _ in evaluated))


def _golden_candidate(
    lo: float, hi: float, evaluated: Sequence[Tuple[float, float]], tol: float
) -> float:
    lo, hi = (float(lo), float(hi))
    if hi <= lo:
        return lo
    cands = [
        lo + (1.0 - _GOLDEN_RATIO_CONJUGATE) * (hi - lo),
        lo + _GOLDEN_RATIO_CONJUGATE * (hi - lo),
        0.5 * (lo + hi),
    ]
    for cand in cands:
        if not _already_evaluated(cand, evaluated, tol):
            return float(cand)
    for i in range(1, 65):
        cand = lo + (hi - lo) * i / 65.0
        if not _already_evaluated(cand, evaluated, tol):
            return float(cand)
    return float(0.5 * (lo + hi))


def _quadratic_argmax(f0: float, d1: float, d2: float, beta: float) -> float:
    candidates = [0.0, float(beta)]
    if abs(float(d2)) > 1e-14:
        root = -float(d1) / float(d2)
        if 0.0 <= root <= float(beta):
            candidates.append(root)
    return max(
        candidates, key=lambda a: float(f0) + a * float(d1) + 0.5 * a * a * float(d2)
    )


def _cubic_argmax(f0: float, d1: float, d2: float, d3: float, beta: float) -> float:
    candidates = [0.0, float(beta)]
    qa, qb, qc = (0.5 * float(d3), float(d2), float(d1))
    if abs(qa) < 1e-14:
        if abs(qb) > 1e-14:
            candidates.append(_clip(-qc / qb, 0.0, beta))
    else:
        disc = qb * qb - 4.0 * qa * qc
        if disc >= 0.0:
            root_disc = math.sqrt(disc)
            for root in (
                (-qb + root_disc) / (2.0 * qa),
                (-qb - root_disc) / (2.0 * qa),
            ):
                if 0.0 <= root <= float(beta):
                    candidates.append(root)

    def value(a: float) -> float:
        return (
            float(f0) + a * float(d1) + 0.5 * a * a * float(d2) + a**3 * float(d3) / 6.0
        )

    return max(candidates, key=value)


def _pade11_argmax(f0: float, d1: float, d2: float, beta: float) -> Tuple[float, bool]:
    if abs(float(d1)) < 1e-14:
        return (_quadratic_argmax(f0, d1, d2, beta), False)
    b1 = -float(d2) / (2.0 * float(d1))
    pole = -1.0 / b1 if abs(b1) > 1e-14 else math.inf
    if 0.0 <= pole <= float(beta):
        return (_quadratic_argmax(f0, d1, d2, beta), False)
    a0 = float(f0)
    a1 = float(d1) + a0 * b1

    def value(a: float) -> float:
        den = 1.0 + b1 * a
        return -math.inf if abs(den) < 1e-12 else (a0 + a1 * a) / den

    candidates = [0.0, float(beta)]
    return (max(candidates, key=value), True)


def _pade12_argmax(
    f0: float, d1: float, d2: float, d3: float, beta: float
) -> Tuple[float, bool]:
    denom = float(d1) ** 2 - 0.5 * float(f0) * float(d2)
    if abs(denom) < 1e-14 or abs(float(f0)) < 1e-14:
        return (_quadratic_argmax(f0, d1, d2, beta), False)
    b1 = (float(f0) * float(d3) / 6.0 - 0.5 * float(d1) * float(d2)) / denom
    b2 = (-0.5 * float(d2) - float(d1) * b1) / float(f0)
    if abs(b2) < 1e-14:
        pole = -1.0 / b1 if abs(b1) > 1e-14 else math.inf
        has_pole = 0.0 <= pole <= float(beta)
    else:
        disc = b1 * b1 - 4.0 * b2
        has_pole = False
        if disc >= 0.0:
            root_disc = math.sqrt(disc)
            roots = ((-b1 + root_disc) / (2.0 * b2), (-b1 - root_disc) / (2.0 * b2))
            has_pole = any((0.0 <= root <= float(beta) for root in roots))
    if has_pole:
        return (_quadratic_argmax(f0, d1, d2, beta), False)
    a0 = float(f0)
    a1 = float(d1) + a0 * b1

    def value(a: float) -> float:
        den = 1.0 + b1 * a + b2 * a * a
        return -math.inf if abs(den) < 1e-12 else (a0 + a1 * a) / den

    grid = [float(beta) * i / 512.0 for i in range(513)]
    return (max(grid, key=value), True)


def _initial_model_proposal(
    model: str, f0: float, d1: float, d2: float, d3: float, beta: float
) -> Tuple[float, bool]:
    if model == "t2":
        return (_quadratic_argmax(f0, d1, d2, beta), False)
    if model == "t3":
        if not _finite(d3) or abs(float(d3)) > 1000000.0 * max(1.0, abs(float(d2))):
            return (_quadratic_argmax(f0, d1, d2, beta), True)
        return (_cubic_argmax(f0, d1, d2, d3, beta), False)
    if model == "p2":
        proposal, ok = _pade11_argmax(f0, d1, d2, beta)
        return (proposal, not ok)
    if model == "p3":
        if not _finite(d3):
            return (_quadratic_argmax(f0, d1, d2, beta), True)
        proposal, ok = _pade12_argmax(f0, d1, d2, d3, beta)
        return (proposal, not ok)
    raise ValueError(f"Unknown initial model: {model}")


def _parabolic_proposal(
    evaluated: Sequence[Tuple[float, float]], beta: float, tol: float
) -> Tuple[float, bool, Tuple[float, float]]:
    points = sorted(
        ((float(a), float(y)) for a, y in evaluated if 0.0 <= float(a) <= float(beta))
    )
    best_a, _ = max(points, key=lambda pair: pair[1])
    left = [point for point in points if point[0] < best_a - tol]
    right = [point for point in points if point[0] > best_a + tol]
    lo = left[-1][0] if left else 0.0
    hi = right[0][0] if right else float(beta)
    bracket = (float(lo), float(hi))
    if not left or not right:
        return (_golden_candidate(lo, hi, points, tol), True, bracket)
    x1, y1 = left[-1]
    x2, y2 = (best_a, dict(points)[best_a])
    x3, y3 = right[0]
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if abs(denom) < 1e-18:
        return (_golden_candidate(lo, hi, points, tol), True, bracket)
    qa = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    qb = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    candidate = -qb / (2.0 * qa) if abs(qa) > 1e-18 else math.nan
    unsafe = (
        qa >= 0.0
        or not _finite(candidate)
        or candidate <= lo
        or (candidate >= hi)
        or _already_evaluated(candidate, points, tol)
    )
    if unsafe:
        return (_golden_candidate(lo, hi, points, tol), True, bracket)
    return (float(candidate), False, bracket)


def select_zfo_seq(
    phi: Callable[[float], float],
    *,
    phi0: float,
    d1: float,
    dalpha: float,
    beta: float,
    k: int = 5,
    initial_model: str = "t2",
    tol: Optional[float] = None,
    search_inside: str = "auto",
) -> SelectionResult:
    if k < 2:
        raise ValueError("zfo_seq requires k >= 2 for its symmetric probes.")
    if initial_model not in INITIAL_MODELS:
        raise ValueError(f"initial_model must be one of {sorted(INITIAL_MODELS)}")
    tol_value = float(tol) if tol is not None else 0.001 * float(beta)
    fp, fm = (float(phi(+float(dalpha))), float(phi(-float(dalpha))))
    evaluated: List[Tuple[float, float]] = [(0.0, float(phi0))]
    if 0.0 <= float(dalpha) <= float(beta):
        evaluated.append((float(dalpha), fp))
    if 0.0 <= -float(dalpha) <= float(beta):
        evaluated.append((-float(dalpha), fm))
    d2 = (fp - 2.0 * float(phi0) + fm) / float(dalpha) ** 2
    d3 = 3.0 * (fp - fm - 2.0 * float(dalpha) * float(d1)) / float(dalpha) ** 3
    proposal, model_fallback = _initial_model_proposal(
        initial_model, phi0, d1, d2, d3, beta
    )
    if search_inside != "auto":
        from zfo import select_model

        method = {"t2": "taylor2", "t3": "taylor3", "p2": "pade2", "p3": "pade3"}[
            initial_model
        ]
        proposal, details = select_model(
            method, phi0, d1, d2, d3, beta, fp, search_inside
        )
        model_fallback = bool(details["fallback"])
    initial_proposal = float(proposal)
    safeguard = bool(model_fallback)
    for j in range(max(0, int(k) - 2)):
        proposal = _clip(proposal, 0.0, beta)
        if _already_evaluated(proposal, evaluated, tol_value):
            proposal = _golden_candidate(0.0, beta, evaluated, tol_value)
            safeguard = True
        evaluated.append((float(proposal), float(phi(float(proposal)))))
        if j < int(k) - 3:
            proposal, used_safeguard, _ = _parabolic_proposal(
                evaluated, beta, tol_value
            )
            safeguard = safeguard or used_safeguard
    selected_a, _ = _best_evaluated(evaluated, beta)
    return SelectionResult(
        selected_a=selected_a,
        probes_used=int(k),
        evaluated=evaluated,
        fallback=model_fallback,
        safeguard=safeguard,
        initial_proposal=initial_proposal,
        details={"d2": d2, "d3": d3},
    )


class OptimizerBundle:
    """One interface for Muon's matrix optimizer and its AdamW remainder."""

    def __init__(self, optimizers):
        self.optimizers = optimizers
        self.param_groups = [g for opt in optimizers for g in opt.param_groups]

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()


def build_optimizer(model, config):
    """Build the requested baseline, or AdamW for a ZFO direction."""
    params = [p for p in model.parameters() if p.requires_grad]
    common = dict(
        betas=(config["beta1"], config["beta2"]),
        eps=config["adam_eps"],
        weight_decay=config["weight_decay"],
    )
    method = config["method"]
    if method == "prodigy":
        from prodigyopt import Prodigy

        return Prodigy(
            params,
            lr=config["prodigy_lr"],
            **common,
            decouple=True,
            d_coef=config["d_coef"],
            d0=config["d0"],
        )
    if method == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("Muon requires PyTorch 2.9 or later")
        # Embeddings and the language-model output head use AdamW, including
        # tied weights. Muon handles only hidden two-dimensional parameters.
        excluded = set()
        for module in (model.get_input_embeddings(), model.get_output_embeddings()):
            if module is not None:
                excluded.update(id(p) for p in module.parameters())
        matrices = [p for p in params if p.ndim == 2 and id(p) not in excluded]
        remaining = [p for p in params if p.ndim != 2 or id(p) in excluded]
        if not matrices:
            raise ValueError("Muon found no hidden two-dimensional parameters")
        opts = [
            torch.optim.Muon(
                matrices,
                lr=config["muon_lr"],
                weight_decay=config["weight_decay"],
                momentum=config["muon_momentum"],
                nesterov=True,
                ns_steps=config["muon_ns_steps"],
                adjust_lr_fn="match_rms_adamw",
            )
        ]
        if remaining:
            opts.append(torch.optim.AdamW(remaining, lr=config["lr"], **common))
        return OptimizerBundle(opts)
    return torch.optim.AdamW(params, lr=config["lr"], **common)


def build_scheduler(optimizer, config, total_steps):
    """Historical warmup/cosine indexing, including its first-step warmup."""
    if config["lr_type"] == "constant":
        return None
    warmup = min(total_steps, max(1, int(round(config["warmup_ratio"] * total_steps))))

    def multiplier(step):
        if step < warmup:
            return float(step + 1) / warmup
        progress = min(1.0, float(step + 1 - warmup) / max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


from torch import nn
from zfo import _trainable_params


def _logprobs_for_tokens(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    return torch.gather(logp, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def _masked_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum(dim=-1)


def _k3_kl_tokens(logp_tok: torch.Tensor, ref_logp_tok: torch.Tensor) -> torch.Tensor:
    log_ratio_ref_over_theta = torch.clamp(ref_logp_tok - logp_tok, min=-20.0, max=20.0)
    return torch.exp(log_ratio_ref_over_theta) - 1.0 - log_ratio_ref_over_theta


@torch.inference_mode()
def _rlvr_zo_objective_on_fixed_batch(
    model: nn.Module,
    ref_logp_tok: torch.Tensor,
    adv_flat: torch.Tensor,
    gen_ids: torch.Tensor,
    gen_attn: torch.Tensor,
    tok_mask: torch.Tensor,
    kl_beta: float,
    policy_kind: str,
    ppo_clip: float,
    seq_logp_old: Optional[torch.Tensor] = None,
) -> Tuple[float, float, float, torch.Tensor]:
    was_training = model.training
    model.eval()
    out = model(gen_ids, attention_mask=gen_attn)
    logits = out.logits[:, :-1, :]
    tokens = gen_ids[:, 1:]
    logp_tok = _logprobs_for_tokens(logits, tokens)
    seq_logp = _masked_sum(logp_tok, tok_mask)
    kl_tok = _k3_kl_tokens(logp_tok, ref_logp_tok)
    seq_kl = _masked_sum(kl_tok, tok_mask) / tok_mask.sum(dim=-1).clamp(min=1.0)
    kl_mean_t = seq_kl.mean()
    if seq_logp_old is None:
        seq_logp_old = seq_logp.detach().clone()
    if policy_kind == "grpo":
        pg_obj_t = (adv_flat * seq_logp).mean()
    else:
        ratio = torch.exp(torch.clamp(seq_logp - seq_logp_old, min=-20.0, max=20.0))
        unclipped = ratio * adv_flat
        clipped = (
            torch.clamp(ratio, 1.0 - float(ppo_clip), 1.0 + float(ppo_clip)) * adv_flat
        )
        pg_obj_t = torch.min(unclipped, clipped).mean()
    obj_t = pg_obj_t - float(kl_beta) * kl_mean_t
    if was_training:
        model.train()
    return (
        float(obj_t.detach().item()),
        float(pg_obj_t.detach().item()),
        float(kl_mean_t.detach().item()),
        seq_logp.detach().clone(),
    )


@torch.no_grad()
def _zo_perturb_trainable(
    model: nn.Module, eps: float, random_seed: int, scaling_factor: float
) -> None:
    torch.manual_seed(int(random_seed))
    for _, p in _trainable_params(model):
        z = torch.normal(
            mean=0.0,
            std=1.0,
            size=p.data.size(),
            device=p.data.device,
            dtype=p.data.dtype,
        )
        p.add_(z, alpha=float(scaling_factor) * float(eps))


def _zo_should_apply_weight_decay(name: str) -> bool:
    lname = str(name).lower()
    return (
        "bias" not in lname
        and "layer_norm" not in lname
        and ("layernorm" not in lname)
        and ("ln_" not in lname)
        and (".ln" not in lname)
    )


@torch.no_grad()
def _zo_apply_update(
    model: nn.Module,
    grad_est: float,
    lr: float,
    random_seed: int,
    *,
    weight_decay: float = 0.0,
    max_update_norm: Optional[float] = None,
) -> Dict[str, float]:
    params = _trainable_params(model)
    if not params:
        raise RuntimeError("No trainable parameters found for ZO update.")
    coef = float(lr) * float(grad_est)
    if not math.isfinite(coef):
        coef = 0.0
    clip_enabled = max_update_norm is not None and float(max_update_norm) > 0.0
    update_scale = 1.0
    update_norm = 0.0
    if clip_enabled:
        torch.manual_seed(int(random_seed))
        sq_pre = 0.0
        for _, p in params:
            z = torch.normal(
                mean=0.0,
                std=1.0,
                size=p.data.size(),
                device=p.data.device,
                dtype=p.data.dtype,
            )
            sq_pre += float(((coef * z.float()) ** 2).sum().item())
        update_norm = math.sqrt(max(0.0, sq_pre))
        if update_norm > float(max_update_norm):
            update_scale = float(max_update_norm) / max(1e-12, update_norm)
    torch.manual_seed(int(random_seed))
    sq = 0.0
    decayed_params = 0
    skipped_decay_params = 0
    for name, p in params:
        z = torch.normal(
            mean=0.0,
            std=1.0,
            size=p.data.size(),
            device=p.data.device,
            dtype=p.data.dtype,
        )
        sq += float(((coef * z.float()) ** 2).sum().item())
        if float(weight_decay) != 0.0 and _zo_should_apply_weight_decay(name):
            p.add_(p, alpha=-float(lr) * float(weight_decay))
            decayed_params += 1
        elif float(weight_decay) != 0.0:
            skipped_decay_params += 1
        p.add_(z, alpha=coef * update_scale)
    if not clip_enabled:
        update_norm = math.sqrt(max(0.0, sq))
    return {
        "zo/update_coef": float(coef),
        "zo/update_scale": float(update_scale),
        "zo/update_norm_unclipped": float(update_norm),
        "zo/weight_decay": float(weight_decay),
        "zo/weight_decay_params": float(decayed_params),
        "zo/weight_decay_skipped_params": float(skipped_decay_params),
    }
