"""Declared Python edit actions with an explicit execution contract.

This is the L1 action space from the WSP roadmap, not an unrestricted Python
sandbox.  An action is a small Python function that computes a delta for one
explicitly selected tensor.  The source is checked against a strict AST
whitelist, executes in a separate process with a wall-clock deadline and
resource caps, and returns through the same budget and protected-key checks as
the declarative edit executor.

The action may refer to these names:

``W``
    The selected weight matrix as a float64 NumPy array.
``keys``, ``values``, ``preserve``
    Optional caller-supplied arrays.
``max_rel_norm``
    The action's declared relative-norm budget.

It may call the pure primitives exported by :mod:`machaon.linalg` and a small
set of NumPy helpers.  Imports, filesystem access, network access, loops,
classes, lambdas, comprehensions, dunder attributes and arbitrary method calls
are rejected before the worker starts.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np

from . import linalg
from .schema import Target
from .verify import check_protected

__all__ = [
    "ActionError",
    "ActionValidationError",
    "ActionExecutionError",
    "ActionSpec",
    "ActionResult",
    "SAFE_FUNCTIONS",
    "validate_action_source",
    "execute_action",
]

FORMAT = "machaon.action"
FORMAT_VERSION = 1
MAX_SOURCE_BYTES = 64 * 1024
MAX_WORKER_OUTPUT = 16 * 1024


class ActionError(RuntimeError):
    """Base class for action validation and execution failures."""


class ActionValidationError(ActionError, ValueError):
    """The action source is outside the constrained Python subset."""


class ActionExecutionError(ActionError):
    """The isolated worker failed, timed out, or returned an invalid delta."""


SAFE_FUNCTIONS = {
    "rank1": linalg.rank1,
    "rank1_constrained": linalg.rank1_constrained,
    "bake_direction": linalg.bake_direction,
    "ablate_output": linalg.ablate_output,
    "ablate_input": linalg.ablate_input,
    "low_rank_scale": linalg.low_rank_scale,
    "project_complement": linalg.project_complement,
    "nullspace_projector": linalg.nullspace_projector,
    "clamp_rel_norm": linalg.clamp_rel_norm,
    "relative_norm": linalg.relative_norm,
    "zeros_like": np.zeros_like,
    "ones_like": np.ones_like,
    "eye": np.eye,
    "norm": np.linalg.norm,
    "maximum": np.maximum,
    "minimum": np.minimum,
    "abs": np.abs,
}

_GLOBAL_NAMES = frozenset(SAFE_FUNCTIONS) | {
    "W", "keys", "values", "preserve", "max_rel_norm"
}
_BINARY_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.MatMult)
_UNARY_OPS = (ast.UAdd, ast.USub)
_SAFE_ATTRIBUTES = frozenset({"T", "shape", "ndim", "size"})
_SAFE_METHODS = frozenset({
    "copy", "reshape", "ravel", "transpose", "sum", "mean", "max", "min",
})


def _fail(message: str) -> None:
    raise ActionValidationError(message)


def _name(node: ast.Name, names: Set[str]) -> None:
    if node.id.startswith("_"):
        _fail("private names are not available: %r" % node.id)
    if node.id not in names:
        _fail("name %r is not declared in the action environment" % node.id)


def _targets(node: ast.AST, names: Set[str]) -> None:
    if isinstance(node, ast.Name):
        if node.id.startswith("_"):
            _fail("assignment to private name %r is not allowed" % node.id)
        return
    if isinstance(node, (ast.Tuple, ast.List)):
        for item in node.elts:
            _targets(item, names)
        return
    _fail("only simple local assignments are allowed")


def _slice(node: ast.AST, names: Set[str]) -> None:
    if isinstance(node, ast.Slice):
        for part in (node.lower, node.upper, node.step):
            if part is not None:
                _expr(part, names)
        return
    if isinstance(node, ast.Tuple):
        for item in node.elts:
            _slice(item, names)
        return
    # ``ast.Index`` was removed in Python 3.9 but may appear through external
    # AST constructors; handle it defensively.
    if node.__class__.__name__ == "Index":
        _expr(node.value, names)
        return
    _expr(node, names)


def _expr(node: ast.AST, names: Set[str]) -> None:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, bool, str, type(None))):
            _fail("constant %r is not allowed" % (node.value,))
        return
    if isinstance(node, ast.Name):
        _name(node, names)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _BINARY_OPS):
            _fail("operator %s is not allowed" % type(node.op).__name__)
        _expr(node.left, names)
        _expr(node.right, names)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _UNARY_OPS):
            _fail("unary operator %s is not allowed" % type(node.op).__name__)
        _expr(node.operand, names)
        return
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            _fail("private attribute %r is not allowed" % node.attr)
        if node.attr not in _SAFE_ATTRIBUTES and node.attr not in _SAFE_METHODS:
            _fail("attribute %r is not allowed" % node.attr)
        _expr(node.value, names)
        return
    if isinstance(node, ast.Subscript):
        _expr(node.value, names)
        _slice(node.slice, names)
        return
    if isinstance(node, (ast.Tuple, ast.List)):
        for item in node.elts:
            _expr(item, names)
        return
    if isinstance(node, ast.Call):
        if node.keywords and any(k.arg is None for k in node.keywords):
            _fail("**kwargs is not allowed")
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg.startswith("_"):
                _fail("keyword names must be public")
            _expr(keyword.value, names)
        if isinstance(node.func, ast.Name):
            if node.func.id not in SAFE_FUNCTIONS:
                _fail("function %r is not in the action environment" % node.func.id)
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr not in _SAFE_METHODS:
                _fail("method %r is not allowed" % node.func.attr)
            _expr(node.func.value, names)
        else:
            _fail("only direct calls to safe functions or array methods are allowed")
        for arg in node.args:
            _expr(arg, names)
        return
    _fail("syntax node %s is not allowed" % type(node).__name__)


def validate_action_source(source: str) -> ast.Module:
    """Validate the constrained action source and return its parsed tree.

    The returned tree is still parsed in the worker; execution never uses the
    caller's process.  A validation failure is raised as
    :class:`ActionValidationError`.
    """
    if not isinstance(source, str):
        _fail("action source must be a string")
    if not source.strip():
        _fail("action source is empty")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        _fail("action source exceeds %d bytes" % MAX_SOURCE_BYTES)
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ActionValidationError("action syntax error: %s" % exc) from exc

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        _fail("action must define exactly one function named 'edit'")
    function = functions[0]
    if function.name != "edit":
        _fail("action function must be named 'edit'")
    if function.decorator_list:
        _fail("decorators are not allowed")
    if function.returns is not None:
        _fail("return annotations are not allowed")
    if getattr(function, "type_params", ()):
        _fail("generic parameters are not allowed")
    if function.args.posonlyargs or function.args.vararg or function.args.kwarg:
        _fail("edit must accept exactly one positional argument, W")
    if function.args.kwonlyargs or function.args.defaults or function.args.kw_defaults:
        _fail("default or keyword-only arguments are not allowed")
    if len(function.args.args) != 1 or function.args.args[0].arg != "W":
        _fail("edit must be declared as def edit(W)")
    if function.args.args[0].annotation is not None:
        _fail("argument annotations are not allowed")

    for node in tree.body:
        if node is function:
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            continue
        _fail("only a module docstring and the edit function are allowed")

    names: Set[str] = set(_GLOBAL_NAMES)
    names.add("W")
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    if not body:
        _fail("edit has no body")
    if not isinstance(body[-1], ast.Return):
        _fail("edit must end with a return statement")
    for statement in body[:-1]:
        if not isinstance(statement, ast.Assign):
            _fail("only local assignments may precede the final return")
        _expr(statement.value, names)
        for target in statement.targets:
            _targets(target, names)
        for target in statement.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for item in target.elts:
                    if isinstance(item, ast.Name):
                        names.add(item.id)
    if body[-1].value is None:
        _fail("edit must return a delta array")
    _expr(body[-1].value, names)
    return tree


def _as_weight(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ActionExecutionError("W must be a 2-D weight matrix")
    if not np.isfinite(array).all():
        raise ActionExecutionError("W contains non-finite values")
    return array


def _as_optional(name: str, value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim not in (1, 2):
        raise ActionExecutionError("%s must be 1-D or 2-D" % name)
    if not np.isfinite(array).all():
        raise ActionExecutionError("%s contains non-finite values" % name)
    return array


@dataclass(frozen=True)
class ActionSpec:
    """An auditable constrained edit function for one tensor."""

    source: str
    target: Target
    intent: str
    max_rel_norm: float = 0.02
    max_protected_drift: float = 1e-8
    timeout_seconds: float = 10.0
    memory_mb: int = 1024

    def validate(self) -> "ActionSpec":
        if not isinstance(self.intent, str) or not self.intent.strip():
            raise ActionValidationError("intent is required")
        self.target.validate()
        validate_action_source(self.source)
        if not (0.0 < float(self.max_rel_norm) <= 0.5):
            raise ActionValidationError("max_rel_norm must be in (0, 0.5]")
        if not (0.0 < float(self.max_protected_drift) <= 1e-2):
            raise ActionValidationError("max_protected_drift must be in (0, 1e-2]")
        if not (0.1 <= float(self.timeout_seconds) <= 60.0):
            raise ActionValidationError("timeout_seconds must be in [0.1, 60]")
        memory = int(self.memory_mb)
        if memory < 128 or memory > 4096:
            raise ActionValidationError("memory_mb must be in [128, 4096]")
        return self

    @property
    def source_sha256(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "target": {"module": self.target.module, "layer": self.target.layer},
            "intent": self.intent,
            "max_rel_norm": float(self.max_rel_norm),
            "max_protected_drift": float(self.max_protected_drift),
            "timeout_seconds": float(self.timeout_seconds),
            "memory_mb": int(self.memory_mb),
        }


@dataclass(frozen=True)
class ActionResult:
    """The delta returned by a constrained action."""

    delta: np.ndarray
    relative_norm: float
    source_sha256: str
    module: str
    intent: str
    elapsed_ms: float
    protected: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "op": "constrained_python",
            "module": self.module,
            "delta_shape": list(self.delta.shape),
            "relative_norm": self.relative_norm,
            "source_sha256": self.source_sha256,
            "intent": self.intent,
            "elapsed_ms": self.elapsed_ms,
        }
        if self.protected is not None:
            payload["protected"] = self.protected
        return payload


def _worker_environment() -> Dict[str, str]:
    env = os.environ.copy()
    root = str(Path(__file__).resolve().parent.parent)
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root + (os.pathsep + prior if prior else "")
    env["PYTHONHASHSEED"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def _read_error(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    return str(error)[:MAX_WORKER_OUTPUT] if error else ""


def execute_action(
    spec: ActionSpec,
    W: Any,
    *,
    keys: Any = None,
    values: Any = None,
    preserve: Any = None,
) -> ActionResult:
    """Execute one constrained action in an isolated worker process.

    The worker receives only ``W`` and the optional named arrays.  The parent
    validates the returned shape, finiteness, declared budget and protected-key
    drift before exposing the delta.  A timeout, worker crash, invalid shape,
    over-budget result or protected-key violation raises
    :class:`ActionExecutionError`.
    """
    spec = spec.validate()
    weight = _as_weight(W)
    arrays = {
        "W": weight,
        "keys": _as_optional("keys", keys),
        "values": _as_optional("values", values),
        "preserve": _as_optional("preserve", preserve),
    }
    arrays = {name: value for name, value in arrays.items() if value is not None}

    with tempfile.TemporaryDirectory(prefix="machaon-action-") as tmp:
        root = Path(tmp)
        input_path = root / "inputs.npz"
        request_path = root / "request.json"
        output_path = root / "output.npy"
        response_path = root / "response.json"
        np.savez(input_path, **arrays)
        request_path.write_text(json.dumps({
            "source": spec.source,
            "input": input_path.name,
            "max_rel_norm": float(spec.max_rel_norm),
            "memory_mb": int(spec.memory_mb),
            "timeout_seconds": float(spec.timeout_seconds),
        }), encoding="utf-8")
        command = [
            sys.executable, "-m", "machaon.action_worker",
            "--request", str(request_path),
            "--input", str(input_path),
            "--output", str(output_path),
            "--response", str(response_path),
        ]
        started = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=str(root),
                env=_worker_environment(),
                capture_output=True,
                text=True,
                timeout=float(spec.timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ActionExecutionError(
                "action exceeded the %.3f second deadline" % spec.timeout_seconds
            ) from exc
        elapsed_ms = (time.monotonic() - started) * 1000.0

        if completed.returncode != 0:
            detail = _read_error(response_path) or completed.stderr.strip()
            raise ActionExecutionError(
                "action worker exited with code %d%s"
                % (completed.returncode, (": " + detail[:MAX_WORKER_OUTPUT]) if detail else "")
            )
        if not response_path.exists():
            raise ActionExecutionError("action worker produced no response")
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ActionExecutionError("cannot read action response") from exc
        if not isinstance(response, dict) or not response.get("ok"):
            detail = str((response or {}).get("error", "unknown action failure"))
            raise ActionExecutionError(detail[:MAX_WORKER_OUTPUT])
        try:
            delta = np.load(output_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ActionExecutionError("action produced no readable delta") from exc
        delta = np.asarray(delta, dtype=np.float64)
        if delta.shape != weight.shape:
            raise ActionExecutionError(
                "action returned shape %r, expected %r" % (delta.shape, weight.shape)
            )
        if not np.isfinite(delta).all():
            raise ActionExecutionError("action returned non-finite delta values")
        rel = linalg.relative_norm(delta, weight)
        if rel > float(spec.max_rel_norm) + 1e-12:
            raise ActionExecutionError(
                "action returned relative norm %.6g above budget %.6g"
                % (rel, spec.max_rel_norm)
            )
        protected_report = None
        if preserve is not None:
            protected_report = check_protected(
                weight, delta, np.asarray(preserve, dtype=np.float64),
                max_drift=float(spec.max_protected_drift),
            )
            if not protected_report["passed"]:
                raise ActionExecutionError(
                    "action changed a protected key (max drift %.6g > %.6g)"
                    % (protected_report["max_drift"], spec.max_protected_drift)
                )
        return ActionResult(
            delta=delta,
            relative_norm=float(rel),
            source_sha256=spec.source_sha256,
            module=spec.target.module,
            intent=spec.intent,
            elapsed_ms=elapsed_ms,
            protected=protected_report,
        )
