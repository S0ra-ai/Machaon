"""FastAPI app for Machaon's local weight-surgery console.

Run:

    machaon-web --allowed-root /path/to/checkpoints

The server binds to loopback by default.  A random token is generated for each
launch and printed with the URL.  Every API route requires that token; the
static shell and ``/api/health`` are the only public routes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised through the extra
    raise ImportError(
        'the web console needs FastAPI: pip install "machaon[web]"'
    ) from exc

from .. import __version__
from .jobs import JobRegistry
from .security import PathPolicy, TokenAuth, load_allowed_roots
from .service import WebService

__all__ = ["create_app", "main"]

STATIC_DIR = Path(__file__).with_name("static")


class OpenSessionRequest(BaseModel):
    path: str
    dtype: str = "auto"
    device: Optional[str] = None
    quantize: Optional[str] = None
    budget_total: Optional[float] = 0.10


class DescribeRequest(BaseModel):
    path: str


class TensorListRequest(BaseModel):
    path: str
    contains: Optional[str] = None
    suffix: Optional[str] = None
    layer: Optional[int] = None
    limit: int = Field(default=50, ge=1, le=5000)


class FetchModelRequest(BaseModel):
    starter: str
    dest: str
    endpoint: Optional[str] = None


class BaselineRequest(BaseModel):
    neutral_prompts: List[str] = Field(default_factory=list)
    corpus_texts: List[str] = Field(default_factory=list)


class ProposeRequest(BaseModel):
    prompt: str
    target: str
    layer: int = Field(ge=0)
    preserve_prompts: List[str] = Field(default_factory=list)
    solver: str = "gradient"
    steps: int = Field(default=40, ge=1, le=10000)
    lr: float = 0.02
    max_rel_norm: Optional[float] = None
    module_suffix: str = "mlp.down_proj"
    target_token_id: Optional[int] = None
    densify_from_layer: bool = False


class BatchProposeRequest(BaseModel):
    prompts: List[str] = Field(min_length=1)
    values: List[List[float]] = Field(min_length=1)
    layer: int = Field(ge=0)
    preserve_prompts: List[str] = Field(default_factory=list)
    max_rel_norm: Optional[float] = None
    clamp: bool = True
    ridge: float = 0.0
    module_suffix: str = "mlp.down_proj"


class TeachPreviewRequest(BaseModel):
    sentence: str = Field(min_length=1, max_length=4000)


class TeachRequest(BaseModel):
    sentence: Optional[str] = Field(default=None, max_length=4000)
    prompt: Optional[str] = None
    answer: Optional[str] = None
    layers: Optional[List[int]] = None
    solver: str = "direction"
    max_rel_norm: float = Field(default=0.02, gt=0, le=0.5)
    dry_run: bool = False
    out_dir: Optional[str] = None


class CheckRequest(BaseModel):
    path: str
    sentence: Optional[str] = Field(default=None, max_length=4000)
    prompt: Optional[str] = Field(default=None, max_length=4000)
    answer: Optional[str] = Field(default=None, max_length=4000)
    facts: Optional[List[str]] = Field(default=None, max_length=100)
    dtype: str = "auto"
    device: Optional[str] = None
    quantize: Optional[str] = None


class TeachManyRequest(BaseModel):
    facts: List[str] = Field(min_length=1, max_length=100)
    out_dir: str
    budget: float = Field(default=0.10, gt=0.0, le=1.0)
    max_rel_norm: float = Field(default=0.02, gt=0.0, le=0.5)
    keep_steps: bool = False
    continue_on_reject: bool = False


class ProgramRequest(BaseModel):
    program: str = Field(min_length=1, max_length=200_000)
    base_dir: Optional[str] = None


class EvaluateRequest(BaseModel):
    prompts: List[str] = Field(min_length=1)
    targets: List[Union[str, int]] = Field(min_length=1)


class CloseSessionRequest(BaseModel):
    force: bool = False


class SaveCheckpointRequest(BaseModel):
    out_dir: str


class GateRequest(BaseModel):
    metrics: Dict[str, float]


class WspAuditRequest(BaseModel):
    report: Dict[str, Any]


class WspSaveRequest(BaseModel):
    report: Dict[str, Any]
    out_dir: str
    metadata: Optional[Dict[str, Any]] = None


class WspSessionRequest(BaseModel):
    prompts: List[str] = Field(min_length=1)
    targets: List[Union[str, int]] = Field(min_length=1)
    headroom: Optional[Dict[str, float]] = None
    mechanism: Optional[Dict[str, Any]] = None
    seed: int = 0
    target_metric: str = "accuracy"


class WspSessionSaveRequest(WspSessionRequest):
    out_dir: str


class ExportEditRequest(BaseModel):
    out_dir: str
    evidence_sha256: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BindingRequest(BaseModel):
    edit_dir: str
    evidence_dir: str


class AuditRequest(BaseModel):
    base_path: str
    edit_dir: str
    max_rel_norm: Optional[float] = None


class ApplyDeltaRequest(BaseModel):
    base_path: str
    tensor: str
    delta_path: str
    out_dir: str
    refuse_rescale: bool = True


def create_app(
    *,
    token: Optional[str] = None,
    allowed_roots: Optional[Sequence[Union[str, Path]]] = None,
    service: Optional[WebService] = None,
    title: str = "Machaon",
) -> FastAPI:
    """Create the local app.

    ``service`` is injectable for tests and for hosts that want to provide a
    custom session factory.  Production launches use :func:`main`.
    """
    roots = (
        tuple(Path(root) for root in allowed_roots)
        if allowed_roots is not None
        else load_allowed_roots()
    )
    policy = PathPolicy(roots)
    auth = TokenAuth(token)
    jobs = JobRegistry(max_workers=1)
    web_service = service or WebService(policy)
    if getattr(web_service, "policy", None) is None:
        web_service.policy = policy

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            jobs.shutdown(wait=False)

    app = FastAPI(
        title=title,
        version=__version__,
        description="Local control plane for transactional weight edits.",
        lifespan=lifespan,
    )
    app.state.auth = auth
    app.state.jobs = jobs
    app.state.service = web_service
    app.state.policy = policy

    # -- authentication and response hygiene --------------------------

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        public = (
            path in ("/", "/api/health", "/favicon.ico")
            or path.startswith("/static/")
        )
        if not public:
            supplied = auth.from_headers(request.headers)
            if supplied is None:
                supplied = request.query_params.get("token")
            if not auth.matches(supplied):
                return JSONResponse(
                    {"detail": "missing or invalid Machaon web token"},
                    status_code=401,
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self'; "
            "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"
        )
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def start_job(kind: str, function, with_progress: bool = False):
        job = jobs.start(kind, function, with_progress=with_progress)
        return {"job": job.id, "kind": kind}

    # -- public ---------------------------------------------------------

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "service": "machaon-web",
            "version": __version__,
            "repository": "https://github.com/S0ra-ai/Machaon",
            "auth": "required",
            "session": web_service.state(),
            "capabilities": web_service.capabilities(),
            "allowed_roots": [str(root) for root in policy.roots],
        }

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico")
    def favicon():
        return FileResponse(STATIC_DIR / "favicon.svg")

    # -- discovery ------------------------------------------------------

    @app.get("/api/models")
    def models() -> Dict[str, Any]:
        return web_service.models()

    @app.post("/api/models/fetch")
    def fetch_model(body: FetchModelRequest) -> Dict[str, Any]:
        dest = _path_or_http_error(policy, body.dest, must_exist=False)
        return start_job(
            "fetch_model",
            lambda progress: web_service.fetch_model(
                body.starter, dest, endpoint=body.endpoint, progress=progress
            ),
            with_progress=True,
        )

    # -- checkpoint inspection -----------------------------------------

    @app.post("/api/checkpoints/describe")
    def describe(body: DescribeRequest) -> Dict[str, Any]:
        path = _path_or_http_error(policy, body.path, must_exist=True)
        return start_job(
            "describe_checkpoint",
            lambda: web_service.describe_checkpoint(path),
        )

    @app.post("/api/checkpoints/tensors")
    def tensors(body: TensorListRequest) -> Dict[str, Any]:
        path = _path_or_http_error(policy, body.path, must_exist=True)
        return start_job(
            "list_tensors",
            lambda: web_service.list_tensors(
                path,
                contains=body.contains,
                suffix=body.suffix,
                layer=body.layer,
                limit=body.limit,
            ),
        )

    @app.post("/api/checkpoints/audit")
    def audit(body: AuditRequest) -> Dict[str, Any]:
        base = _path_or_http_error(policy, body.base_path, must_exist=True)
        edit = _path_or_http_error(policy, body.edit_dir, must_exist=True)
        return start_job(
            "audit_checkpoint_edit",
            lambda: web_service.audit_checkpoint_edit(
                base, edit, max_rel_norm=body.max_rel_norm
            ),
        )

    @app.post("/api/checkpoints/apply")
    def apply_delta(body: ApplyDeltaRequest) -> Dict[str, Any]:
        base = _path_or_http_error(policy, body.base_path, must_exist=True)
        delta = _path_or_http_error(policy, body.delta_path, must_exist=True)
        out = _path_or_http_error(policy, body.out_dir, must_exist=False)
        return start_job(
            "apply_delta",
            lambda: web_service.apply_delta(
                base,
                body.tensor,
                delta,
                out,
                refuse_rescale=body.refuse_rescale,
            ),
        )

    # -- session --------------------------------------------------------

    @app.post("/api/session/open")
    def open_session(body: OpenSessionRequest) -> Dict[str, Any]:
        path = _path_or_http_error(policy, body.path, must_exist=True)
        return start_job(
            "open_session",
            lambda: web_service.open_session(
                path,
                dtype=body.dtype,
                device=body.device,
                quantize=body.quantize,
                budget_total=body.budget_total,
            ),
        )

    @app.get("/api/session/state")
    def session_state() -> Dict[str, Any]:
        return web_service.state()

    @app.post("/api/session/close")
    def close_session(body: CloseSessionRequest) -> Dict[str, Any]:
        return start_job(
            "close_session",
            lambda: web_service.close_session(force=body.force),
        )

    @app.post("/api/session/baseline")
    def baseline(body: BaselineRequest) -> Dict[str, Any]:
        return start_job(
            "baseline",
            lambda: web_service.baseline(
                body.neutral_prompts, body.corpus_texts
            ),
        )

    @app.post("/api/session/save")
    def save_checkpoint(body: SaveCheckpointRequest) -> Dict[str, Any]:
        out = _path_or_http_error(policy, body.out_dir, must_exist=False)
        return start_job(
            "save_checkpoint",
            lambda: web_service.save_checkpoint(out),
        )

    @app.post("/api/session/export_openwsp")
    def export_openwsp(body: ExportEditRequest) -> Dict[str, Any]:
        out = _path_or_http_error(policy, body.out_dir, must_exist=False)
        return start_job(
            "export_openwsp",
            lambda: web_service.export_session_edit(
                out, evidence_sha256=body.evidence_sha256, metadata=body.metadata
            ),
        )

    @app.post("/api/binding/verify")
    def verify_binding(body: BindingRequest) -> Dict[str, Any]:
        edit = _path_or_http_error(policy, body.edit_dir, must_exist=True)
        evidence = _path_or_http_error(policy, body.evidence_dir, must_exist=True)
        return start_job(
            "verify_binding",
            lambda: web_service.verify_binding(edit, evidence),
        )

    # -- edit -----------------------------------------------------------

    @app.post("/api/teach/preview")
    def teach_preview(body: TeachPreviewRequest) -> Dict[str, Any]:
        try:
            return web_service.preview_teach(body.sentence)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/teach")
    def teach(body: TeachRequest) -> Dict[str, Any]:
        out = None
        if body.out_dir:
            out = _path_or_http_error(policy, body.out_dir, must_exist=False)
        return start_job(
            "teach",
            lambda: web_service.teach(
                sentence=body.sentence,
                prompt=body.prompt,
                answer=body.answer,
                layers=body.layers,
                solver=body.solver,
                max_rel_norm=body.max_rel_norm,
                dry_run=body.dry_run,
                out_dir=out,
            ),
        )

    @app.post("/api/check")
    def check(body: CheckRequest) -> Dict[str, Any]:
        path = _path_or_http_error(policy, body.path, must_exist=True)
        return start_job(
            "check",
            lambda: web_service.check_saved(
                path,
                sentence=body.sentence,
                prompt=body.prompt,
                answer=body.answer,
                facts=body.facts,
                dtype=body.dtype,
                device=body.device,
                quantize=body.quantize,
            ),
        )

    @app.post("/api/program/plan")
    def program_plan(body: ProgramRequest) -> Dict[str, Any]:
        try:
            return web_service.plan_program(body.program, body.base_dir)
        except (OSError, TypeError, ValueError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/program/run")
    def program_run(body: ProgramRequest) -> Dict[str, Any]:
        try:
            # Resolve the base now so a bad path fails before a worker starts.
            base = policy.require_existing(body.base_dir) if body.base_dir else str(
                policy.roots[0]
            )
        except (OSError, TypeError, ValueError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return start_job(
            "program_run",
            lambda: web_service.run_program(body.program, base),
        )

    @app.post("/api/teach_many")
    def teach_many(body: TeachManyRequest) -> Dict[str, Any]:
        out = _path_or_http_error(policy, body.out_dir, must_exist=False)
        return start_job(
            "teach_many",
            lambda: web_service.teach_many(
                body.facts,
                out,
                budget=body.budget,
                max_rel_norm=body.max_rel_norm,
                keep_steps=body.keep_steps,
                continue_on_reject=body.continue_on_reject,
            ),
        )

    @app.post("/api/edit/propose")
    def propose(body: ProposeRequest) -> Dict[str, Any]:
        return start_job(
            "propose_edit",
            lambda: web_service.propose(
                prompt=body.prompt,
                target=body.target,
                layer=body.layer,
                preserve_prompts=body.preserve_prompts,
                solver=body.solver,
                steps=body.steps,
                lr=body.lr,
                max_rel_norm=body.max_rel_norm,
                module_suffix=body.module_suffix,
                target_token_id=body.target_token_id,
                densify_from_layer=body.densify_from_layer,
            ),
        )

    @app.post("/api/edit/propose_batch")
    def propose_batch(body: BatchProposeRequest) -> Dict[str, Any]:
        return start_job(
            "propose_batch",
            lambda: web_service.propose_batch(
                prompts=body.prompts,
                values=body.values,
                layer=body.layer,
                preserve_prompts=body.preserve_prompts,
                max_rel_norm=body.max_rel_norm,
                clamp=body.clamp,
                ridge=body.ridge,
                module_suffix=body.module_suffix,
            ),
        )

    @app.post("/api/edit/apply")
    def apply_edit() -> Dict[str, Any]:
        return start_job("apply_edit", web_service.apply)

    @app.post("/api/edit/evaluate")
    def evaluate(body: EvaluateRequest) -> Dict[str, Any]:
        return start_job(
            "evaluate",
            lambda: web_service.evaluate(body.prompts, body.targets),
        )

    @app.post("/api/edit/rollback")
    def rollback() -> Dict[str, Any]:
        return start_job("rollback", web_service.rollback)

    # -- gates ----------------------------------------------------------

    @app.post("/api/gates/check")
    def gate(body: GateRequest) -> Dict[str, Any]:
        return web_service.gate_metrics(body.metrics)

    @app.post("/api/wsp/audit")
    def wsp_audit(body: WspAuditRequest) -> Dict[str, Any]:
        return start_job("wsp_audit", lambda: web_service.wsp_audit(body.report))

    @app.post("/api/wsp/save")
    def save_wsp(body: WspSaveRequest) -> Dict[str, Any]:
        out_dir = _path_or_http_error(policy, body.out_dir, must_exist=False)
        return start_job(
            "save_wsp_evidence",
            lambda: web_service.save_wsp_evidence(
                out_dir, body.report, body.metadata
            ),
        )

    @app.post("/api/wsp/session")
    def wsp_session(body: WspSessionRequest) -> Dict[str, Any]:
        return start_job(
            "wsp_session",
            lambda: web_service.assess_session_wsp(
                body.prompts,
                body.targets,
                headroom=body.headroom,
                seed=body.seed,
                target_metric=body.target_metric,
                mechanism=body.mechanism,
            ),
        )

    @app.post("/api/wsp/session_evidence")
    def wsp_session_evidence(body: WspSessionSaveRequest) -> Dict[str, Any]:
        out_dir = _path_or_http_error(policy, body.out_dir, must_exist=False)
        return start_job(
            "wsp_session_evidence",
            lambda: web_service.assess_session_evidence(
                out_dir,
                body.prompts,
                body.targets,
                headroom=body.headroom,
                seed=body.seed,
                target_metric=body.target_metric,
                mechanism=body.mechanism,
            ),
        )

    # -- jobs -----------------------------------------------------------

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> Dict[str, Any]:
        snapshot = jobs.snapshot(job_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return snapshot

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request):
        if jobs.get(job_id) is None:
            raise HTTPException(status_code=404, detail="unknown job")

        async def stream():
            seq = 0
            while True:
                job = jobs.get(job_id)
                if job is None:
                    return
                events = job.events_after(seq)
                for event in events:
                    seq = max(seq, int(event["seq"]))
                    yield "data: %s\n\n" % json.dumps(event, ensure_ascii=False)
                snapshot = job.snapshot()
                if snapshot["status"] in JobRegistry.TERMINAL:
                    yield "event: end\ndata: %s\n\n" % json.dumps(
                        {"status": snapshot["status"], "job": job_id}
                    )
                    return
                if await request.is_disconnected():
                    return
                await asyncio.sleep(0.12)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def _path_or_http_error(policy: PathPolicy, value: str, must_exist: bool) -> str:
    """Translate a policy failure into a useful HTTP response."""
    try:
        if must_exist:
            return policy.require_existing(value)
        return policy.require(value)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _localhost(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="machaon-web",
        description="Local browser console for transactional weight editing.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allowed-root",
        action="append",
        default=None,
        help="checkpoint directory the UI may read; repeat for multiple roots",
    )
    parser.add_argument("--token", default=None, help="reuse a fixed local token")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="explicitly allow a non-loopback bind (not recommended)",
    )
    parser.add_argument("--version", action="version", version="machaon-web " + __version__)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if not _localhost(args.host) and not args.allow_remote:
        print(
            "refusing to bind outside loopback; pass --allow-remote only when "
            "you have put the service behind your own authentication proxy",
            file=sys.stderr,
        )
        return 2
    token = args.token or os.environ.get("MACHAON_WEB_TOKEN") or secrets.token_urlsafe(24)
    roots = args.allowed_root or None
    app = create_app(token=token, allowed_roots=roots)
    url = "http://%s:%d/?token=%s" % (args.host, args.port, token)
    print("Machaon web console: %s" % url)
    print("Allowed roots: %s" % ", ".join(str(root) for root in app.state.policy.roots))
    print("Press Ctrl-C to stop.")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        import uvicorn
    except ImportError:
        print('uvicorn is required: pip install "machaon[web]"', file=sys.stderr)
        app.state.jobs.shutdown(wait=False)
        return 2
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        app.state.jobs.shutdown(wait=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
