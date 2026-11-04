"""Rank-one factual editing (ROME), with a null-space constraint.

Two value solvers, both of which leave model weights untouched:

``direction`` (default)
    Use the target token's unembedding row as a direction in hidden space, add
    it to the module's current output, and line-search the scale.  No backward
    pass — ten to fifteen forward passes on a 0.5B model take about a second.

``gradient``
    Optimise the value vector directly.  Model parameters are frozen for the
    duration, so backward accumulates a gradient for one vector rather than for
    every weight in the checkpoint.  Useful when the unembedding direction is a
    poor proxy, e.g. for edits deep inside the network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

import numpy as np

from ..linalg import rank1_constrained


@dataclass
class RomeProposal:
    """A solved edit that has not been written yet."""

    module: str
    layer: int
    delta: Any
    relative_delta: float
    target_probability: float
    target_token_id: int
    solver: str
    steps: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RomeResult:
    module: str
    layer: int
    relative_delta: float
    target_probability: float
    target_token_id: int
    solver: str
    steps: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _output_embedding_row(runtime: Any, token: int) -> Any:
    import torch

    model = runtime.model
    embedding = None
    if hasattr(model, "get_output_embeddings"):
        embedding = model.get_output_embeddings()
    if embedding is None or getattr(embedding, "weight", None) is None:
        raise AttributeError("model has no output embedding; use solver='gradient'")
    row = embedding.weight[token].detach().float()
    return row / torch.linalg.norm(row)


def _logits_with(runtime, module, ids, value):
    import torch

    def replace(_m, _i, output, _v=value):
        output = output.clone()
        output[0, -1, :] = _v.to(output.dtype)
        return output

    hook = module.register_forward_hook(replace)
    try:
        with torch.no_grad():
            logits = runtime.model(ids).logits[0, -1].float()
    finally:
        hook.remove()
    return logits


def _solve_direction(runtime, module, ids, target_token, value, alphas):
    """Smallest scale along the target's unembedding row that wins.

    Saturation is cheap and wasteful: the value that pushes the target to
    probability 1.0 can spend ten times the norm of the value that merely makes
    it the argmax.  We take the first alpha that flips the prediction, then
    bisect for the smallest one that still flips.
    """
    import torch

    direction = _output_embedding_row(runtime, target_token).to(value.dtype)

    def evaluate(alpha):
        logits = _logits_with(runtime, module, ids, value + alpha * direction)
        probability = float(torch.softmax(logits, dim=-1)[target_token])
        return probability, int(logits.argmax())

    best_probability = evaluate(0.0)
    best = (best_probability[0], 0.0)
    previous = 0.0
    for alpha in alphas:
        probability, top = evaluate(float(alpha))
        if top == target_token:
            low, high = previous, float(alpha)
            best = (probability, float(alpha))
            for _ in range(6):
                middle = (low + high) / 2.0
                middle_probability, middle_top = evaluate(middle)
                if middle_top == target_token:
                    high = middle
                    best = (middle_probability, middle)
                else:
                    low = middle
            break
        if probability > best[0]:
            best = (probability, float(alpha))
        previous = float(alpha)

    probability, alpha = best
    return value + alpha * direction, probability, len(alphas)


def _solve_gradient(runtime, module, ids, target_token, value, steps, lr):
    import torch

    value = value.detach().clone().requires_grad_(True)
    optimiser = torch.optim.Adam([value], lr=lr)
    frozen = [p for p in runtime.model.parameters() if p.requires_grad]
    for parameter in frozen:
        parameter.requires_grad_(False)
    try:
        loss = None
        for _ in range(steps):
            def replace(_m, _i, output, _v=value):
                output = output.clone()
                output[0, -1, :] = _v.to(output.dtype)
                return output

            hook = module.register_forward_hook(replace)
            logits = runtime.model(ids).logits[0, -1].float()
            loss = -torch.log_softmax(logits, dim=-1)[target_token]
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            hook.remove()
    finally:
        for parameter in frozen:
            parameter.requires_grad_(True)
    return value.detach(), float(torch.exp(-loss.detach())), steps


def solve_edit(
    runtime: Any,
    prompt: str,
    target: str,
    layer: int,
    preserve_prompts: Sequence[str] = (),
    steps: int = 250,
    lr: float = 0.02,
    module_suffix: str = "mlp.down_proj",
    max_rel_norm: float | None = None,
    target_token_id: int | None = None,
    solver: str = "direction",
    alphas: Sequence[float] = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
) -> RomeProposal:
    """Solve an edit and return the delta without touching the weights."""
    import torch

    if runtime.tokenizer is None:
        raise RuntimeError("rome_edit needs a tokenizer")
    if solver not in ("direction", "gradient"):
        raise ValueError("solver must be 'direction' or 'gradient'")

    ids = runtime.encode([prompt])[0]
    if target_token_id is None:
        target_token = runtime.tokenizer(target, add_special_tokens=False)["input_ids"][0]
    else:
        target_token = int(target_token_id)

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

    current = box["y"][0, -1, :].detach()
    if solver == "direction":
        value, probability, used = _solve_direction(
            runtime, module, ids, target_token, current, alphas)
    else:
        value, probability, used = _solve_gradient(
            runtime, module, ids, target_token, current, steps, lr)

    _, delta = rank1_constrained(
        W,
        key.astype(np.float64),
        value.detach().double().cpu().numpy(),
        preserve=preserve,
        max_rel_norm=max_rel_norm,
    )

    return RomeProposal(
        module=weight_name,
        layer=layer,
        delta=delta,
        relative_delta=float(np.linalg.norm(delta) / np.linalg.norm(W)),
        target_probability=float(probability),
        target_token_id=int(target_token),
        solver=solver,
        steps=int(used),
        diagnostics={"preserve_keys": 0 if preserve is None else int(preserve.shape[0])},
    )


def rome_edit(runtime: Any, prompt: str, target: str, layer: int, **kwargs) -> RomeResult:
    """Solve an edit and write it immediately."""
    proposal = solve_edit(runtime, prompt, target, layer, **kwargs)

    W = runtime.read(proposal.module)
    runtime.write(proposal.module, W + proposal.delta)

    return RomeResult(
        module=proposal.module,
        layer=proposal.layer,
        relative_delta=proposal.relative_delta,
        target_probability=proposal.target_probability,
        target_token_id=proposal.target_token_id,
        solver=proposal.solver,
        steps=proposal.steps,
        diagnostics=proposal.diagnostics,
    )
