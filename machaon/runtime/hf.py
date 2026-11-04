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

    @staticmethod
    def _stores_weight_transposed(module: Any) -> bool:
        """True when ``module.weight`` is ``(in, out)`` instead of ``(out, in)``.

        ``transformers.Conv1D`` (GPT-2 style) stores ``(nx, nf)`` and computes
        ``x @ W``; ``nn.Linear`` stores ``(out, in)`` and computes ``x @ W.T``.
        The rest of this project speaks one language: ``W`` is ``(d_out, d_in)``.
        """
        return module.__class__.__name__ == "Conv1D"

    def weight_shape(self, name: str) -> tuple:
        """Canonical ``(d_out, d_in)`` of a weight, whatever the module stores."""
        module = self._module_holding(name)
        param = getattr(module, name.rsplit(".", 1)[-1])
        shape = tuple(param.shape)
        return tuple(reversed(shape)) if self._stores_weight_transposed(module) else shape

    def read(self, name: str) -> np.ndarray:
        """Read one weight as float64, always in ``(d_out, d_in)`` orientation."""
        tensor = self.resolve(name)
        if not hasattr(tensor, "detach"):
            raise TypeError("%s is not a parameter or buffer" % name)
        array = tensor.detach().float().cpu().numpy()
        if self._stores_weight_transposed(self._module_holding(name)):
            array = array.T
        return np.ascontiguousarray(array, dtype=np.float64)

    def write(self, name: str, values: np.ndarray) -> None:
        """Write one weight given in ``(d_out, d_in)`` orientation."""
        module = self._module_holding(name)
        param = getattr(module, name.rsplit(".", 1)[-1])
        array = np.ascontiguousarray(values, dtype=np.float32)
        expected = self.weight_shape(name)
        if array.shape != expected:
            raise ValueError(
                "shape %r does not match %s %r" % (array.shape, name, expected)
            )
        if self._stores_weight_transposed(module):
            array = np.ascontiguousarray(array.T)
        with self._torch.no_grad():
            param.copy_(self._torch.from_numpy(array).to(device=param.device, dtype=param.dtype))

    def _module_holding(self, name: str) -> Any:
        parent, _, attr = name.rpartition(".")
        return _resolve(self.model, parent) if parent else self.model

    # -- inference ------------------------------------------------------

    def encode(self, texts: Sequence[str]):
        """Tokenise and right-pad. Returns ``(input_ids, attention_mask)``.

        Padded manually because several tokenizers (GPT-2 among them) ship
        without a pad token; we never mutate the caller's tokenizer.
        """
        import torch

        if self.tokenizer is None:
            raise RuntimeError("this runtime has no tokenizer; use *_ids methods")
        sequences = [self.tokenizer(text)["input_ids"] for text in texts]
        if not sequences:
            raise ValueError("no texts to encode")
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        width = max(len(seq) for seq in sequences)
        ids = np.full((len(sequences), width), pad_id, dtype=np.int64)
        mask = np.zeros((len(sequences), width), dtype=np.int64)
        for row, seq in enumerate(sequences):
            ids[row, : len(seq)] = seq
            mask[row, : len(seq)] = 1
        return (
            torch.as_tensor(ids, device=self.device),
            torch.as_tensor(mask, device=self.device),
        )

    def next_logits_ids(self, input_ids: Any, attention_mask: Any = None) -> np.ndarray:
        """Next-token logits at the last *real* position, as ``(n, vocab)``."""
        import torch

        ids = torch.as_tensor(np.asarray(input_ids), dtype=torch.long, device=self.device)
        mask = None
        if attention_mask is not None:
            mask = torch.as_tensor(
                np.asarray(attention_mask), dtype=torch.long, device=self.device
            )
        with torch.no_grad():
            out = self.model(ids, attention_mask=mask)
        if mask is None:
            return out.logits[:, -1, :].float().cpu().numpy()
        rows = torch.arange(ids.shape[0], device=self.device)
        last = mask.sum(dim=-1) - 1
        return out.logits[rows, last, :].float().cpu().numpy()

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
        items = list(items)
        if items and isinstance(items[0], str):
            ids, mask = self.encode(items)
            return self.next_logits_ids(ids, mask)
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

    def nll_ids(self, input_ids: Any, attention_mask: Any = None) -> np.ndarray:
        """Mean negative log-likelihood per real token, as ``(n,)`` float64."""
        import torch

        ids = torch.as_tensor(np.asarray(input_ids), dtype=torch.long, device=self.device)
        mask = None
        if attention_mask is not None:
            mask = torch.as_tensor(
                np.asarray(attention_mask), dtype=torch.long, device=self.device
            )
        with torch.no_grad():
            logits = self.model(ids, attention_mask=mask).logits.float()
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = ids[:, 1:]
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        if mask is None:
            return (-gathered.mean(dim=-1)).double().cpu().numpy()
        weights = mask[:, 1:].double()
        total = -(gathered.double() * weights).sum(dim=-1)
        return (total / weights.sum(dim=-1).clamp(min=1.0)).cpu().numpy()

    def nll(self, items: Sequence[Any]) -> np.ndarray:
        items = list(items)
        if items and isinstance(items[0], str):
            ids, mask = self.encode(items)
            return self.nll_ids(ids, mask)
        return self.nll_ids(self._to_ids(items))

    def score_continuations(self, pairs: Sequence[tuple]) -> np.ndarray:
        """Mean log-probability of each continuation given its prompt.

        Tokenises prompt and continuation separately so the boundary is exact;
        used for multiple-choice scoring and answer-level edit metrics.
        """
        import torch

        if self.tokenizer is None:
            raise RuntimeError("this runtime has no tokenizer")
        scores = []
        for prompt, continuation in pairs:
            prefix = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            suffix = self.tokenizer(continuation, add_special_tokens=False)["input_ids"]
            if not suffix:
                raise ValueError("empty continuation")
            ids = torch.as_tensor([prefix + suffix], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.model(ids).logits.float()
            log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
            targets = ids[0, 1:]
            start = len(prefix) - 1
            picked = log_probs[start:, :].gather(
                -1, targets[start:].unsqueeze(-1)
            ).squeeze(-1)
            scores.append(float(picked.mean()))
        return np.asarray(scores, dtype=np.float64)

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))
