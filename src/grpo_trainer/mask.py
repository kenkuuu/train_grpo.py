"""
Sparse mask utilities for parameter delta analysis and masked training.

Based on: "Reinforcement Learning Finetunes Small Subnetworks in LLMs"
(https://arxiv.org/abs/2505.11711)

Typical workflow:
    # 1. Compute mask from two checkpoints
    mask = compute_delta_mask(base_path, finetuned_path, threshold=1e-5)

    # 2. Inspect sparsity
    stats = get_sparsity_stats(mask)
    print(stats)

    # 3a. Apply to training via gradient hooks (works with any trainer)
    handles = register_grad_masks(model, mask)
    trainer.train()
    for h in handles: h.remove()

    # 3b. Or via TrainerCallback (convenience wrapper)
    from grpo_trainer.mask import MaskedGradCallback
    trainer = GRPOTrainer(..., callbacks=[MaskedGradCallback(model, mask)])
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, TrainerCallback

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data class
# ──────────────────────────────────────────────

@dataclass
class SparsityStats:
    global_sparsity: float    # fraction of params with |delta| <= threshold
    active_fraction: float    # 1 - global_sparsity
    total_params: int
    active_params: int
    layerwise: dict           # layer_idx (str) -> active_fraction
    paramwise: dict           # param_name -> active_fraction

    def __str__(self) -> str:
        lines = [
            f"Global sparsity : {self.global_sparsity:.4f}  "
            f"({self.total_params - self.active_params:,} / {self.total_params:,} params frozen)",
            f"Active fraction : {self.active_fraction:.4f}  "
            f"({self.active_params:,} params updated)",
        ]
        if self.layerwise:
            lines.append("\nLayer-wise active fraction:")
            for layer, frac in sorted(self.layerwise.items(), key=lambda x: int(x[0])):
                bar = "█" * int(frac * 30)
                lines.append(f"  Layer {int(layer):>3d}: {frac:.4f}  {bar}")
        return "\n".join(lines)


# ──────────────────────────────────────────────
# Mask computation
# ──────────────────────────────────────────────

def compute_delta_mask(
    base_path: str | Path,
    finetuned_path: str | Path,
    threshold: float = 1e-5,
) -> dict[str, torch.BoolTensor]:
    """
    Compare two model checkpoints and return a binary mask.

    mask[name] == True  →  |delta| > threshold  (parameter was actively updated)
    mask[name] == False →  |delta| <= threshold (parameter was effectively frozen)

    Both models are loaded on CPU to avoid GPU memory pressure.
    threshold=0 uses exact equality; threshold>0 treats near-zero deltas as frozen.
    """
    base_path = Path(base_path)
    finetuned_path = Path(finetuned_path)

    logger.info(f"Loading base model: {base_path}")
    base_sd = AutoModelForCausalLM.from_pretrained(
        str(base_path), torch_dtype=torch.float32, device_map="cpu"
    ).state_dict()

    logger.info(f"Loading finetuned model: {finetuned_path}")
    ft_sd = AutoModelForCausalLM.from_pretrained(
        str(finetuned_path), torch_dtype=torch.float32, device_map="cpu"
    ).state_dict()

    mask: dict[str, torch.BoolTensor] = {}
    with torch.no_grad():
        for name, base_param in base_sd.items():
            if base_param.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                continue
            if name not in ft_sd:
                continue
            delta = ft_sd[name].float() - base_param.float()
            mask[name] = delta.abs() > threshold

    logger.info(f"Mask computed for {len(mask)} parameter tensors")
    return mask


def compute_delta_mask_from_state_dicts(
    base_sd: dict,
    ft_sd: dict,
    threshold: float = 1e-5,
) -> dict[str, torch.BoolTensor]:
    """Same as compute_delta_mask but accepts pre-loaded state dicts (avoids double loading)."""
    mask: dict[str, torch.BoolTensor] = {}
    with torch.no_grad():
        for name, base_param in base_sd.items():
            if base_param.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                continue
            if name not in ft_sd:
                continue
            delta = ft_sd[name].float() - base_param.float()
            mask[name] = delta.abs() > threshold
    return mask


# ──────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────

def get_sparsity_stats(mask: dict[str, torch.BoolTensor]) -> SparsityStats:
    """Aggregate global and per-layer sparsity statistics from a mask dict."""
    total, active = 0, 0
    paramwise: dict[str, float] = {}
    layerwise_accum: dict[str, list[float]] = {}

    for name, m in mask.items():
        n = m.numel()
        a = int(m.sum().item())
        total += n
        active += a
        frac = a / n if n > 0 else 0.0
        paramwise[name] = frac

        parts = name.split(".")
        if len(parts) > 3 and parts[0] == "model" and parts[1] == "layers":
            layerwise_accum.setdefault(parts[2], []).append(frac)

    layerwise = {
        k: sum(v) / len(v)
        for k, v in layerwise_accum.items()
    }

    return SparsityStats(
        global_sparsity=1.0 - active / total if total > 0 else 0.0,
        active_fraction=active / total if total > 0 else 0.0,
        total_params=total,
        active_params=active,
        layerwise=layerwise,
        paramwise=paramwise,
    )


# ──────────────────────────────────────────────
# Training integration
# ──────────────────────────────────────────────

def register_grad_masks(
    model: torch.nn.Module,
    mask: dict[str, torch.BoolTensor],
    invert: bool = False,
) -> list:
    """
    Register per-parameter gradient hooks so that only active parameters receive updates.

    mask[name] == True  → gradient kept    (parameter updated normally)
    mask[name] == False → gradient zeroed  (parameter effectively frozen)

    invert=True reverses the mask: train only the sparse (inactive) region.

    Returns hook handles. Call handle.remove() on each to deregister.

    Example:
        handles = register_grad_masks(model, mask)
        trainer.train()
        for h in handles:
            h.remove()
    """
    handles = []
    registered = 0

    for name, param in model.named_parameters():
        if name not in mask:
            continue

        m = mask[name]
        if invert:
            m = ~m

        def _make_hook(m_: torch.BoolTensor):
            def _hook(grad: torch.Tensor) -> torch.Tensor:
                return grad.mul(m_.to(grad.device, non_blocking=True))
            return _hook

        handles.append(param.register_hook(_make_hook(m)))
        registered += 1

    logger.info(
        f"Registered gradient masks on {registered} / {len(list(model.named_parameters()))} parameters"
    )
    return handles


class MaskedGradCallback(TrainerCallback):
    """
    TrainerCallback that applies a sparse mask to gradients at training start.

    Registers gradient hooks immediately upon construction, then removes them
    when training ends. Compatible with TRL GRPOTrainer.

    Usage:
        mask = compute_delta_mask(base_path, ckpt_path)
        callback = MaskedGradCallback(model, mask)
        trainer = GRPOTrainer(..., callbacks=[callback])
    """

    def __init__(
        self,
        model: torch.nn.Module,
        mask: dict[str, torch.BoolTensor],
        invert: bool = False,
    ):
        self._handles = register_grad_masks(model, mask, invert=invert)

    def on_train_end(self, args, state, control, **kwargs):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        logger.info("Gradient mask hooks removed")
