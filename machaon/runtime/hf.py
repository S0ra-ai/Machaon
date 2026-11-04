"""A thin runtime over a Hugging Face causal LM.

Deliberately small: this module knows how to find a layer, read and write one
weight as float64, and produce logits or NLL.  It does not know what an edit is.

Layer discovery covers the common families, so the same code paths work from a
randomly initialised 40k-parameter model up to an 8B checkpoint.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

# Ordered from most to least specific.  First match wins.
LAYER_PATHS = (
    "model.layers",        # Llama, Qwen, Mistral, Gemma, Phi
    "transformer.h",       # GPT-2, GPT-J style
    "gpt_neox.layers",     # GPT-NeoX, Pythia
    "model.decoder.layers",# OPT
    "transformer.blocks",  # MPT
)


def _resolve(model: Any, dotted: str) -> Any:
    cur = model
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur


def layer_container(model: Any) -> tuple:
    """Return ``(container, dotted_path)`` for the decoder blocks."""
    for path in LAYER_PATHS:
        try:
            return _resolve(model, path), path
        except AttributeError:
            continue
    raise AttributeError(
        "could not find decoder layers; tried %s. "
        "Add your architecture to machaon.runtime.hf.LAYER_PATHS." % (LAYER_PATHS,)
    )


class HFModel:
    """A causal LM plus the few operations weight editing needs."""

    def __init__(self, model: Any, tokenizer: Any = None, device: Any = None) -> None:
        import torch

        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        self._torch = torch

    # -- structure ------------------------------------------------------

    @property
    def layers(self) -> List[Any]:
        return list(layer_container(self.model)[0])

    @property
    def layer_path(self) -> str:
        return layer_container(self.model)[1]

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def resolve(self, dotted: str) -> Any:
        return _resolve(self.model, dotted)

    def module_names(self, suffix: Optional[str] = None) -> List[str]:
        names = [name for name, _ in self.model.named_modules() if name]
        if suffix:
            names = [n for n in names if n.endswith(suffix)]
        return names

    def edit_target(self, layer: int, suffix: str = "mlp.down_proj") -> str:
        """Dotted name of ``layers[layer].<suffix>`` for this architecture.

        ``mlp.down_proj`` on Llama-family models; GPT-2 names the same matrix
        ``mlp.c_proj``.  Callers can pass either and get the right one.
        """
        candidates = [suffix]
        if suffix.endswith("down_proj"):
            candidates.append(suffix.replace("down_proj", "c_proj"))
        elif suffix.endswith("c_proj"):
            candidates.append(suffix.replace("c_proj", "down_proj"))
        prefix = "%s.%d" % (self.layer_path, int(layer))
        for candidate in candidates:
            name = "%s.%s.weight" % (prefix, candidate)
            try:
                self.resolve(name)
                return name
            except AttributeError:
                continue
        raise AttributeError(
            "no %r under %s; available: %s"
            % (suffix, prefix, self.module_names()[:0] or "see model.named_modules()")
        )

    # -- weights --------------------------------------------------------

    def read(self, name: str) -> np.ndarray:
        """Read one weight as float64 (exact widening from fp32/bf16/fp16)."""
        tensor = self.resolve(name)
        if not hasattr(tensor, "detach"):
            raise TypeError("%s is not a parameter or buffer" % name)
        return tensor.detach().float().cpu().numpy().astype(np.float64, copy=True)

    def write(self, name: str, values: np.ndarray) -> None:
        """Write one weight, casting back to the module's own dtype."""
        module = self._module_holding(name)
        param = getattr(module, name.rsplit(".", 1)[-1])
        array = np.ascontiguousarray(values, dtype=np.float32)
        if array.shape != tuple(param.shape):
            raise ValueError(
                "shape %r does not match %s %r" % (array.shape, name, tuple(param.shape))
            )
        with self._torch.no_grad():
            param.copy_(self._torch.from_numpy(array).to(device=param.device, dtype=param.dtype))

    def _module_holding(self, name: str) -> Any:
        parent, _, attr = name.rpartition(".")
        return _resolve(self.model, parent) if parent else self.model

    # -- inference ------------------------------------------------------

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.tokenizer is None:
            raise RuntimeError("this runtime has no tokenizer; use *_ids methods")
        encoded = self.tokenizer(list(texts), return_tensors="pt", padding=True)
        return encoded["input_ids"].to(self.device)

    def next_logits_ids(self, input_ids: Any) -> np.ndarray:
        """Next-token logits at the last position, as ``(n, vocab)`` float32."""
        import torch

        ids = torch.as_tensor(np.asarray(input_ids), dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(ids)
        return out.logits[:, -1, :].float().cpu().numpy()

    def _to_ids(self, items: Sequence[Any]):
        """Accept either a list of prompts or a list of 1-D token-id sequences."""
        items = list(items)
        if items and isinstance(items[0], str):
            if self.tokenizer is None:
                raise RuntimeError("text input requires a tokenizer; pass token ids")
            return self.encode(items)
        rows = []
        for item in items:
            if hasattr(item, "detach"):
                item = item.detach().cpu().numpy()
            array = np.asarray(item, dtype=np.int64)
            if array.ndim == 1:
                rows.append(array)
            elif array.ndim == 2:
                rows.extend(list(array))
            else:
                raise ValueError("token ids must be 1-D or 2-D, got %r" % (array.shape,))
        try:
            return np.asarray(rows)
        except ValueError as exc:
            raise ValueError(
                "token id rows have different lengths; pad them yourself or pass text"
            ) from exc

    def next_logits(self, items: Sequence[Any]) -> np.ndarray:
        return self.next_logits_ids(self._to_ids(items))

    def capture_module_input(self, name: str, input_ids: Any, position: int = -1) -> np.ndarray:
        """The tensor entering module ``name`` at one position.

        This is the ROME "key": the input the edited matrix actually sees, not
        the residual stream before the final layer norm.
        """
        import torch

        module = self.resolve(name)
        ids = torch.as_tensor(np.asarray(input_ids), dtype=torch.long, device=self.device)
        box = {}

        def hook(_module, inputs):
            box["x"] = inputs[0].detach()

        handle = module.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                self.model(ids)
        finally:
            handle.remove()
        if "x" not in box:
            raise RuntimeError("module %s was never called" % name)
        return box["x"][:, position, :].float().cpu().numpy().astype(np.float64)

    def nll_ids(self, input_ids: Any) -> np.ndarray:
        """Mean negative log-likelihood per token, as ``(n,)`` float64."""
        import torch

        ids = torch.as_tensor(np.asarray(input_ids), dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(ids).logits.float()
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = ids[:, 1:]
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return (-gathered.mean(dim=-1)).double().cpu().numpy()

    def nll(self, items: Sequence[Any]) -> np.ndarray:
        return self.nll_ids(self._to_ids(items))

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))
