"""Rank-one factual editing (ROME), with a null-space constraint.

The procedure is deliberately explicit: locate one matrix, capture the key that
enters it, solve a single value vector, and write a rank-one delta.  Nothing
optimises model weights.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import numpy as np

from ..linalg import rank1_constrained


@dataclass
class RomeResult:
    module: str
    layer: int
    relative_delta: float
    target_probability: float
    target_token_id: int
    steps: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def rome_edit(
    runtime: Any,
    prompt: str,
    target: str,
    layer: int,
    preserve_prompts: Sequence[str] = (),
    steps: int = 250,
    lr: float = 0.02,
    module_suffix: str = "mlp.down_proj",
    max_rel_norm: float | None = None,
) -> RomeResult:
    """Edit ``runtime`` in place so ``prompt`` continues with ``target``."""
    import torch

    if runtime.tokenizer is None:
        raise RuntimeError("rome_edit needs a tokenizer")

    ids = runtime.encode([prompt])[0]
    target_token = runtime.tokenizer(target, add_special_tokens=False)["input_ids"][0]

    weight_name = runtime.edit_target(layer, module_suffix)
    module_name = weight_name[: -len(".weight")]
    module = runtime.resolve(module_name)

    W = runtime.read(weight_name)
    key = runtime.capture_module_input(module_name, ids)[0]

    preserve = None
    if preserve_prompts:
        preserve = np.stack([
            runtime.capture_module_input(module_name, runtime.encode([p])[0])[0]
            for p in preserve_prompts
        ])

    box: Dict[str, Any] = {}

    def capture(_m, _i, output):
        box["y"] = output

    handle = module.register_forward_hook(capture)
    with torch.no_grad():
        runtime.model(ids)
    handle.remove()
    if "y" not in box:
        raise RuntimeError("%s was never called" % module_name)

    value = box["y"][0, -1, :].detach().clone().requires_grad_(True)
    optimiser = torch.optim.Adam([value], lr=lr)
    loss = None
    for _ in range(steps):
        def replace(_m, _i, output, _v=value):
            output = output.clone()
            output[0, -1, :] = _v
            return output

        hook = module.register_forward_hook(replace)
        logits = runtime.model(ids).logits[0, -1]
        loss = -torch.log_softmax(logits, dim=-1)[target_token]
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        hook.remove()

    solved = float(torch.exp(-loss.detach()))
    W_new, delta = rank1_constrained(
        W,
        key.astype(np.float64),
        value.detach().double().cpu().numpy(),
        preserve=preserve,
        max_rel_norm=max_rel_norm,
    )
    runtime.write(weight_name, W_new)

    return RomeResult(
        module=weight_name,
        layer=layer,
        relative_delta=float(np.linalg.norm(delta) / np.linalg.norm(W)),
        target_probability=solved,
        target_token_id=int(target_token),
        steps=steps,
        diagnostics={"preserve_keys": 0 if preserve is None else int(preserve.shape[0])},
    )
