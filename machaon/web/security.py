"""Local-only authentication and checkpoint path policy.

The browser is a local convenience layer over the same Python functions used by
the CLI and MCP server.  Two small guards make that layer safe to leave running:

* a one-time token protects every API route;
* checkpoint reads and writes are restricted to explicitly allowed roots.

Neither guard is a substitute for running Machaon in the same trust boundary as
your shell.  They do keep an arbitrary page on ``localhost`` from driving a
loaded model or reading an unrelated checkpoint.
"""
from __future__ import annotations

import hmac
import os
import secrets
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

__all__ = ["PathPolicy", "TokenAuth", "load_allowed_roots"]


def load_allowed_roots(
    raw: Optional[str] = None,
    default: Optional[Iterable[Path]] = None,
) -> Tuple[Path, ...]:
    """Resolve the allowed roots from an argument or the environment.

    ``MACHAON_WEB_ALLOWED_ROOTS`` accepts the platform path separator and also
    commas, which makes copy-pasted shell configurations less surprising.
    """
    value = raw if raw is not None else os.environ.get("MACHAON_WEB_ALLOWED_ROOTS")
    roots = []
    if value:
        for item in value.replace(",", os.pathsep).split(os.pathsep):
            item = item.strip()
            if item:
                roots.append(Path(item).expanduser().resolve())
    elif default is not None:
        roots.extend(Path(item).expanduser().resolve() for item in default)
    else:
        roots.append(Path.cwd().resolve())
    return tuple(dict.fromkeys(roots))


class PathPolicy:
    """Allow filesystem operations only below one of the configured roots."""

    def __init__(self, roots: Sequence[Path]) -> None:
        if not roots:
            raise ValueError("at least one allowed root is required")
        self.roots = tuple(Path(root).expanduser().resolve() for root in roots)

    def _inside(self, path: Path) -> bool:
        for root in self.roots:
            try:
                path.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def resolve(self, value: str) -> Path:
        """Resolve a user-provided path and enforce the allowlist."""
        if not value or not str(value).strip():
            raise ValueError("path is required")
        path = Path(value).expanduser().resolve()
        if not self._inside(path):
            raise PermissionError(
                "path is outside the configured roots: %s" % path
            )
        return path

    def require(self, value: str) -> str:
        return str(self.resolve(value))

    def require_existing(self, value: str) -> str:
        path = self.resolve(value)
        if not path.exists():
            raise FileNotFoundError(str(path))
        return str(path)


class TokenAuth:
    """Constant-time bearer token check for local API requests."""

    def __init__(self, token: Optional[str] = None) -> None:
        self.token = token or secrets.token_urlsafe(24)

    def matches(self, supplied: Optional[str]) -> bool:
        if not supplied:
            return False
        return hmac.compare_digest(str(supplied), self.token)

    def from_headers(self, headers) -> Optional[str]:
        value = headers.get("x-machaon-token")
        if value:
            return value
        authorization = headers.get("authorization", "")
        prefix = "bearer "
        if authorization.lower().startswith(prefix):
            return authorization[len(prefix):].strip()
        return None
