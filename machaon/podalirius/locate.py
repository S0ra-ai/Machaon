"""Activation patching: which layer carries the behaviour?

The method is causal tracing.  Run a *clean* prompt (model knows the answer)
and a *corrupt* prompt (it does not).  Then, layer by layer, splice the clean
hidden state into the corrupt run and measure how much of the target token's
probability comes back.  The layer with the largest recovery is where the fact
lives -- and therefore where the edit belongs.

Requires torch and a Hugging Face causal LM.  Imported lazily so the core
package stays dependency-free.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

LAYER_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers")


def _resolve(model: Any, path: str) -> Any:
    cur = model
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def find_layers(model: Any):
    """Return ``(layer_module_list, dotted_path)`` for common HF architectures."""
    for path in LAYER_PATHS:
        try:
            return _resolve(model, path), path
        except AttributeError:
            continue
    raise AttributeError(
        "could not find decoder layers; tried %s. "
        "Add your architecture's path to LAYER_PATHS." % (LAYER_PATHS,)
    )


def patch_scores(model: Any, clean_ids: Any, corrupt_ids: Any, target_token_id: int,
                 layers: Optional[List[int]] = None) -> Dict[str, Any]:
    """Score every layer by how much splicing its clean state recovers.

    ``clean_ids`` and ``corrupt_ids`` must have the same sequence length;
    position-wise splicing is what makes the comparison meaningful.
    """
    import torch

    layers_mod, path = find_layers(model)
    n_layers = len(layers_mod)

    if tuple(clean_ids.shape) != tuple(corrupt_ids.shape):
        raise ValueError(
            "clean and corrupt prompts must have the same shape, got %r and %r"
            % (tuple(clean_ids.shape), tuple(corrupt_ids.shape))
        )

    with torch.no_grad():
        clean = model(clean_ids, output_hidden_states=True)
        corrupt = model(corrupt_ids, output_hidden_states=True)

    clean_hs = clean.hidden_states  # tuple of length n_layers + 1
    baseline_p = float(torch.softmax(corrupt.logits[0, -1], dim=-1)[target_token_id])

    selected = range(n_layers) if layers is None else layers
    scores: Dict[int, Dict[str, float]] = {}

    for layer in selected:
        if not (0 <= layer < n_layers):
            raise IndexError("layer %d out of range (0..%d)" % (layer, n_layers - 1))
        replacement = clean_hs[layer + 1]

        def hook(module, inputs, output, _r=replacement):
            if isinstance(output, tuple):
                return (_r,) + output[1:]
            return _r

        handle = layers_mod[layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                patched = model(corrupt_ids)
            p = float(torch.softmax(patched.logits[0, -1], dim=-1)[target_token_id])
        finally:
            handle.remove()

        scores[layer] = {
            "p_target": p,
            "recovery": p - baseline_p,
        }

    ranked = sorted(scores.items(), key=lambda kv: kv[1]["recovery"], reverse=True)
    return {
        "layer_path": path,
        "n_layers": n_layers,
        "baseline_p": baseline_p,
        "scores": scores,
        "ranked": [layer for layer, _ in ranked],
        "best_layer": ranked[0][0] if ranked else None,
    }
