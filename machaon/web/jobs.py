"""Small, dependency-free job runner and event log for the local web UI.

Model loading and solving are long-running CPU work.  They run in one worker
thread so a browser cannot start two weights-in-flight operations at once.
Every transition is retained as a JSON event; the browser receives the same
records through SSE or a polling endpoint.
"""
from __future__ import annotations

import copy
import math
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

__all__ = ["Job", "JobEvent", "JobRegistry"]


def _jsonable(value: Any) -> Any:
    """Turn values returned by NumPy/Python into JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    # NumPy scalars and arrays are optional here; avoid importing NumPy in the
    # web module's import path, but handle them when a result contains one.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _jsonable(tolist())
        except Exception:
            pass
    return str(value)


@dataclass
class JobEvent:
    seq: int
    type: str
    data: Dict[str, Any]
    created_at: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "type": self.type,
            "data": copy.deepcopy(self.data),
            "created_at": self.created_at,
        }


class Job:
    def __init__(self, kind: str) -> None:
        self.id = uuid.uuid4().hex[:16]
        self.kind = kind
        self.status = "queued"
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.result: Any = None
        self.error: Optional[str] = None
        self.error_type: Optional[str] = None
        self.events: List[JobEvent] = []
        self._seq = 0
        self._lock = threading.RLock()
        self._done = threading.Event()

    def _emit(self, event_type: str, **data: Any) -> Dict[str, Any]:
        with self._lock:
            self._seq += 1
            event = JobEvent(
                seq=self._seq,
                type=event_type,
                data=_jsonable(data),
                created_at=time.time(),
            )
            self.events.append(event)
            return event.as_dict()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "result": copy.deepcopy(self.result),
                "error": self.error,
                "error_type": self.error_type,
                "events": len(self.events),
            }

    def events_after(self, seq: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            return [event.as_dict() for event in self.events if event.seq > seq]

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._done.wait(timeout)


class JobRegistry:
    """One-worker registry with a bounded in-memory audit trail."""

    TERMINAL = frozenset(("done", "error"))

    def __init__(self, max_workers: int = 1, max_jobs: int = 100) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="machaon-web"
        )
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.RLock()
        self._max_jobs = max(1, int(max_jobs))
        self._closed = False

    def start(self, kind: str, function: Callable[[], Any]) -> Job:
        with self._lock:
            if self._closed:
                raise RuntimeError("job registry is closed")
            job = Job(kind)
            self._jobs[job.id] = job
            self._prune_locked()
        job._emit("status", status="queued", kind=kind)
        # The executor's single worker serialises model/session operations.
        self._executor.submit(self._run, job, function)
        return job

    def _run(self, job: Job, function: Callable[[], Any]) -> None:
        with job._lock:
            job.status = "running"
            job.started_at = time.time()
        job._emit("status", status="running", kind=job.kind)
        try:
            result = function()
        except Exception as exc:  # the browser gets a safe message, not a traceback
            with job._lock:
                job.status = "error"
                job.error = str(exc) or exc.__class__.__name__
                job.error_type = exc.__class__.__name__
                job.finished_at = time.time()
            job._emit(
                "error",
                error=job.error,
                error_type=job.error_type,
            )
            job._emit("status", status="error", kind=job.kind)
        else:
            normalized = _jsonable(result)
            with job._lock:
                job.result = normalized
                job.status = "done"
                job.finished_at = time.time()
            job._emit("result", result=normalized)
            job._emit("status", status="done", kind=job.kind)
        finally:
            job._done.set()

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.get(job_id)
        return None if job is None else job.snapshot()

    def events_after(self, job_id: str, seq: int = 0) -> Optional[List[Dict[str, Any]]]:
        job = self.get(job_id)
        return None if job is None else job.events_after(seq)

    def _prune_locked(self) -> None:
        while len(self._jobs) > self._max_jobs:
            oldest_id = next(iter(self._jobs))
            oldest = self._jobs[oldest_id]
            if oldest.status not in self.TERMINAL:
                break
            self._jobs.pop(oldest_id, None)

    def shutdown(self, wait: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def __repr__(self) -> str:
        with self._lock:
            return "JobRegistry(jobs=%d, max=%d)" % (len(self._jobs), self._max_jobs)
