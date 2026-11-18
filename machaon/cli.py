"""Command line entry point.

Every command is a thin wrapper over the library.  Exit codes are part of the
contract, so the CLI can be used as a CI step without parsing text:

    0  success / gate passed
    1  gate rejected the edit, or validation failed
    2  malformed input, missing file, or unsupported checkpoint
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import metadata
from typing import Any, Dict

from . import __version__


def _error(message: str) -> int:
    print("error: %s" % message, file=sys.stderr)
    return 2


def _cmd_demo(_args) -> int:
    from .demo import main as demo_main
    demo_main()
    return 0


def _cmd_validate(args) -> int:
    from .schema import EditSpec
    try:
        with open(args.path) as f:
            spec = EditSpec.from_dict(json.load(f)).validate()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _error("%s: %s" % (args.path, exc))
    print("OK  %s -> %s (budget %.3f)"
          % (spec.op, spec.target.module, spec.constraints.max_rel_norm))
    return 0


def _cmd_gate(args) -> int:
    from .hygieia.gates import evaluate
    try:
        with open(args.path) as f:
            metrics = json.load(f)
    except (OSError, ValueError) as exc:
        return _error("%s: %s" % (args.path, exc))
    if not isinstance(metrics, dict):
        return _error("%s: metrics must be a JSON object" % args.path)
    verdict = evaluate(metrics)
    print(verdict.summary())
    return 0 if verdict.passed else 1


def _cmd_wsp(args) -> int:
    from .evidence import write_evidence
    from .wsp import assess_report
    try:
        with open(args.path) as f:
            payload = json.load(f)
        report = assess_report(payload)
        metadata = {}
        if args.metadata:
            with open(args.metadata) as f:
                metadata = json.load(f)
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a JSON object")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _error("%s: %s" % (args.path, exc))
    if args.save:
        try:
            document = write_evidence(args.save, report, metadata)
        except (OSError, ValueError, TypeError) as exc:
            return _error("%s: %s" % (args.save, exc))
        report["evidence"] = {
            "path": str(args.save),
            "sha256": document["evidence_sha256"],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["accepted"] else 1


def _cmd_wsp_verify(args) -> int:
    from .evidence import read_evidence
    try:
        document = read_evidence(args.path)
    except (OSError, ValueError, TypeError) as exc:
        return _error("%s: %s" % (args.path, exc))
    print(json.dumps({
        "valid": True,
        "evidence_sha256": document["evidence_sha256"],
        "format": document["format"],
        "format_version": document["format_version"],
        "label": document["report"].get("label"),
        "accepted": document["report"].get("accepted"),
    }, indent=2, sort_keys=True))
    return 0


def _cmd_binding(args) -> int:
    from .binding import verify_binding
    try:
        report = verify_binding(args.edit_dir, args.evidence_dir).to_dict()
    except (OSError, ValueError, TypeError) as exc:
        return _error("%s: %s" % (args.edit_dir, exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def _cmd_wsp_bench(args) -> int:
    from .bench.wsp import load_benchmark, run_benchmark

    try:
        cases = load_benchmark(args.path)
        if args.require_measured and any(
            case.source_kind != "measured" for case in cases
        ):
            return _error("benchmark contains synthetic cases; measured evidence required")
        report = run_benchmark(cases)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _error("%s: %s" % (args.path, exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["summary"]["all_matched"] else 1


def _cmd_plan(args) -> int:
    from .planning import plan_surgery

    try:
        with open(args.path) as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("plan input must be a JSON object")
        plan = plan_surgery(
            payload.get("failure_type", "unknown"),
            layer_scores=payload.get("layer_scores"),
            patch_report=payload.get("patch_report"),
            module_suffix=payload.get("module_suffix", "mlp.down_proj"),
            max_rel_norm=float(payload.get("max_rel_norm", 0.02)),
            min_score=float(payload.get("min_score", 0.0)),
            metadata=payload.get("metadata"),
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _error("%s: %s" % (args.path, exc))
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_diagnose(args) -> int:
    from .planning import DiagnosisProtocol, diagnose_and_plan

    try:
        with open(args.path) as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("diagnosis input must be a JSON object")
        protocol_payload = payload.get("protocol") or {}
        if not isinstance(protocol_payload, dict):
            raise ValueError("protocol must be an object")
        protocol = DiagnosisProtocol(**protocol_payload)
        result = diagnose_and_plan(
            payload.get("base", {}),
            headroom=payload.get("headroom"),
            mechanism=payload.get("mechanism"),
            protocol=protocol,
            plan_kwargs=payload.get("plan") or {},
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return _error("%s: %s" % (args.path, exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_inspect(args) -> int:
    from .adapters import Checkpoint

    try:
        ckpt = Checkpoint(args.path)
    except (OSError, ValueError, ImportError) as exc:
        return _error(str(exc))
    names = ckpt.find(contains=args.contains, suffix=args.suffix, layer=args.layer)
    if args.limit is not None:
        names = names[: args.limit]
    rows = []
    for name in names:
        info = ckpt.info(name)
        rows.append({"name": name, "shape": list(info.shape), "dtype": info.dtype})
    if args.json:
        print(json.dumps({"path": str(ckpt.path), "tensors": rows}, indent=2))
        return 0
    for row in rows:
        print("%-64s %-20s %s" % (row["name"], str(row["shape"]), row["dtype"]))
    print("\n%d tensor(s)" % len(rows))
    return 0


def _parse_layers(value: str):
    """Parse ``8,10,12`` or ``8-12`` into a sorted unique layer list."""
    layers = []
    for part in str(value).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start, end = int(start), int(end)
            if end < start:
                raise ValueError("layer range %r is descending" % part)
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    if not layers:
        raise ValueError("no layers in %r" % value)
    if any(layer < 0 for layer in layers):
        raise ValueError("layers must be non-negative")
    return sorted(set(layers))


def _read_lines(path: str):
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def _default_out(model: str) -> str:
    from pathlib import Path

    source = Path(model).expanduser().resolve()
    name = source.name + "-machaon" if source.is_dir() else source.stem + "-machaon"
    # A Hugging Face cache snapshot is an implementation detail; do not write
    # a new checkpoint inside the cache. Put it in the working directory under
    # the repository's short name instead.
    if (
        source.parent.name == "snapshots"
        and source.parent.parent.name.startswith("models--")
    ):
        repo = source.parent.parent.name.split("models--", 1)[1]
        short = repo.rsplit("--", 1)[-1]
        return str(Path.cwd() / (short + "-machaon"))
    return str(source.with_name(name))


def _save_applied(session, out_dir: str) -> dict:
    from .checkpointing import save_session

    return save_session(session, out_dir)


def _discard_artifacts(artifacts: dict) -> None:
    import shutil
    from pathlib import Path

    for key in ("evidence", "edit"):
        item = artifacts.get(key)
        if item and item.get("path"):
            shutil.rmtree(item["path"], ignore_errors=True)
    report = artifacts.get("report")
    if report:
        Path(report).unlink(missing_ok=True)


def _print_teach(result: dict, intent: dict, artifacts: dict) -> None:
    status_label = {
        "applied": "APPLIED",
        "already_true": "NOTHING TO DO",
        "rejected": "REVERTED",
        "dry_run": "DRY RUN",
    }.get(result["status"], result["status"].upper())
    print("Machaon · teach")
    print("status     : %s" % status_label)
    print("prompt     : %s" % result["prompt"])
    print("answer     : %r" % result["answer_text"])
    print("token      : %r (id %d, %d token%s)" % (
        result["answer_token_text"], result["answer_token"],
        result["answer_token_count"],
        "" if result["answer_token_count"] == 1 else "s",
    ))
    print("base answer: %r" % result.get("base_prediction_text", ""))
    print("intent     : %s confidence, %s form" % (
        intent.get("confidence", "high"), intent.get("source", "arguments"),
    ))
    print("model      : %s (%d layers)" % (result["model"], result["n_layers"]))
    if result["best_layer"] is not None:
        print("layer      : %d (tried %s)" % (
            result["best_layer"],
            ", ".join(str(layer) for layer in result["layers"]),
        ))
    if result["tensor"]:
        print("tensor     : %s" % result["tensor"])
    if result["relative_delta"] is not None:
        print("delta      : rel %.4f%s" % (
            result["relative_delta"],
            "" if result["budget_total"] is None
            else " (budget total %.4f)" % result["budget_total"],
        ))
    for check in result["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        value = check.get("value")
        suffix = ""
        if value is not None:
            if check["name"] == "panel_top1" and result.get("panel_size"):
                suffix = " [%d/%d]" % (
                    round(float(value) * result["panel_size"]),
                    result["panel_size"],
                )
            elif isinstance(value, (int, float)):
                suffix = " [%.4g]" % float(value)
            else:
                suffix = " [%s]" % value
        print("check      : %-14s %-4s %s%s" % (
            check["name"], mark, check["requirement"], suffix,
        ))
    if result["unmeasured"]:
        print("unmeasured : %s" % ", ".join(result["unmeasured"]))
    for warning in result["warnings"]:
        print("warning    : %s" % warning)
    for note in result["notes"]:
        print("note       : %s" % note)
    print("summary    : %s" % result["summary"])
    if artifacts.get("output"):
        print("saved      : %s" % artifacts["output"])
    if artifacts.get("edit"):
        print("openwsp    : %s" % artifacts["edit"])
    if artifacts.get("evidence"):
        print("evidence   : %s (%s)" % (
            artifacts["evidence"]["path"],
            artifacts["evidence"].get("evidence_sha256", "")[:12],
        ))
    if result["next_steps"]:
        print("next       : %s" % result["next_steps"][0])


def _cmd_intent(args) -> int:
    from .intent import IntentError, parse_intent

    try:
        intent = parse_intent(args.sentence)
    except IntentError as exc:
        return _error(str(exc))
    payload = intent.to_dict()
    payload["command"] = intent.to_command()
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    print("kind       : %s" % payload["kind"])
    print("confidence : %s" % payload["confidence"])
    print("source     : %s" % payload["source"])
    print("prompt     : %s" % payload["prompt"])
    print("answer     : %r" % payload["answer"])
    for note in payload["notes"]:
        print("note       : %s" % note)
    print("\n%s" % payload["command"])
    return 0


def _cmd_teach(args) -> int:
    from pathlib import Path

    from .budget import EditBudget
    from .intent import IntentError, parse_intent
    from .recipes import TeachPolicy, teach_fact

    if args.say:
        if args.prompt is not None or args.answer is not None:
            return _error("use --say or --prompt/--answer, not both")
        try:
            intent = parse_intent(args.say)
        except IntentError as exc:
            return _error(str(exc))
        prompt, answer = intent.prompt, intent.answer
        intent_payload = intent.to_dict()
    else:
        if args.prompt is None or args.answer is None or args.answer == "":
            return _error("teach needs --say, or both --prompt and --answer")
        prompt, answer = args.prompt, args.answer
        intent_payload = {
            "kind": "teach",
            "prompt": prompt,
            "answer": answer,
            "source": "arguments",
            "confidence": "high",
            "notes": [],
        }
    if not args.model:
        return _error("teach needs --model")
    if args.dry_run and args.out:
        return _error("--dry-run cannot be combined with --out")
    if args.no_save and args.out:
        return _error("--no-save cannot be combined with --out")
    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        return _error(
            "%s: no such checkpoint; download it first with "
            "python scripts/fetch_model.py --repo <repo> --dest %s"
            % (args.model, args.model)
        )
    if not model_path.is_dir():
        return _error(
            "teach needs a checkpoint directory with config.json, tokenizer "
            "files and weights; %s is a file" % model_path
        )
    for flag, value in (
        ("--evidence-out", args.evidence_out),
        ("--edit-out", args.edit_out),
    ):
        if value and Path(value).expanduser().exists():
            return _error("%s already exists: %s" % (flag, value))
    try:
        layers = _parse_layers(args.layers) if args.layers else None
        preserve = _read_lines(args.preserve_file) if args.preserve_file else None
        corpus = _read_lines(args.corpus_file) if args.corpus_file else ()
    except (OSError, ValueError) as exc:
        return _error(str(exc))

    from .session import Session

    try:
        session = Session.open(
            args.model,
            dtype=args.dtype,
            device=args.device,
            quantize=args.quantize,
            budget=EditBudget(max_total=args.budget),
        )
    except ImportError as exc:
        return _error(
            "real-model support is not installed; pip install \"machaon[hf]\" (%s)"
            % exc
        )
    except (OSError, ValueError) as exc:
        return _error("%s: %s" % (args.model, exc))

    policy = TeachPolicy(
        max_rel_norm=args.max_rel_norm,
        max_kl_p95=args.max_kl,
        max_ppl_ratio=args.max_ppl_ratio,
        min_panel_agreement=args.min_panel_agreement,
    )
    try:
        result = teach_fact(
            session,
            prompt,
            answer,
            layers=layers,
            policy=policy,
            solver=args.solver,
            steps=args.steps,
            lr=args.lr,
            max_layers=args.max_layers,
            preserve_prompts=preserve,
            corpus_texts=corpus,
            max_evaluations=args.max_evaluations,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        return _error(str(exc))

    payload = result.to_dict()
    payload["intent"] = intent_payload
    artifacts: dict = {}
    if result.status == "applied":
        try:
            evidence_hash = None
            if args.evidence_out:
                from .evidence import write_evidence
                from .wsp import WSPProtocol

                report = session.assess_wsp(
                    [prompt],
                    [result.answer_token],
                    headroom={
                        "name": "unedited",
                        "accuracy": float(result.base_accuracy),
                    },
                    protocol=WSPProtocol(
                        target_metric="accuracy", require_gates=False
                    ),
                )
                document = write_evidence(
                    args.evidence_out,
                    report,
                    metadata={
                        "recipe": "teach_fact",
                        "model": result.model,
                        "prompt": result.prompt,
                        "answer": result.answer_text,
                        "layer": result.best_layer,
                        "tensor": result.tensor,
                        "relative_delta": result.relative_delta,
                        "headroom": "unedited model",
                    },
                )
                evidence_hash = document["evidence_sha256"]
                artifacts["evidence"] = {
                    "path": str(Path(args.evidence_out)),
                    "evidence_sha256": evidence_hash,
                }
            if args.edit_out:
                manifest = session.export_openwsp(
                    args.edit_out,
                    base_model=result.model,
                    metadata={
                        "recipe": "teach_fact",
                        "prompt": result.prompt,
                        "answer": result.answer_text,
                        "layer": result.best_layer,
                        "evidence_sha256": evidence_hash,
                    },
                )
                artifacts["edit"] = {
                    "path": str(Path(args.edit_out)),
                    "edit_sha256": manifest.get("edit_sha256"),
                }
            if args.report_out:
                report_path = Path(args.report_out)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
                artifacts["report"] = str(report_path)
            if not args.no_save:
                out_dir = args.out or _default_out(args.model)
                artifacts.update(_save_applied(session, out_dir))
        except Exception as exc:
            _discard_artifacts(artifacts)
            try:
                session.rollback()
            except Exception as rollback_exc:
                return _error(
                    "writing an output failed (%s) and rollback also failed (%s); "
                    "the session is no longer safe" % (exc, rollback_exc)
                )
            return _error(
                "writing an output failed; artifacts were removed, the edit was "
                "rolled back and no checkpoint was kept: %s" % exc
            )
    payload["artifacts"] = artifacts
    if result.status == "applied":
        if args.no_save:
            payload["next_steps"] = [
                "this edit is in memory only; rerun with --out DIR to persist it",
                "run a full gate report before deployment",
            ]
        elif artifacts.get("output"):
            payload["next_steps"] = [
                "saved to %s; run a full gate report before deployment"
                % artifacts["output"],
                "the source checkpoint is unchanged; delete the output to undo",
            ]

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        _print_teach(payload, intent_payload, artifacts)
    return 0 if result.status in ("applied", "already_true", "dry_run") else 1


def _cmd_fetch(args) -> int:
    from .fetch import main as fetch_main

    argv = []
    if args.list:
        argv.append("--list")
    if args.repo:
        argv.extend(["--repo", args.repo])
    if args.starter:
        argv.extend(["--starter", args.starter])
    if args.dest:
        argv.extend(["--dest", args.dest])
    if args.endpoint:
        argv.extend(["--endpoint", args.endpoint])
    if args.json:
        argv.append("--json")
    try:
        return fetch_main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)


def _cmd_teach_many(args) -> int:
    from pathlib import Path

    from .recipes import TeachPolicy
    from .session import Session
    from .teach_many import load_facts, teach_many

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        return _error(
            "%s: no such checkpoint; run machaon fetch --starter qwen05b" % args.model
        )
    if not model_path.is_dir():
        return _error(
            "teach-many needs a checkpoint directory; %s is a file" % model_path
        )
    try:
        layers = _parse_layers(args.layers) if args.layers else None
        facts = load_facts(args.facts)
        preserve = _read_lines(args.preserve_file) if args.preserve_file else None
    except (OSError, ValueError) as exc:
        return _error(str(exc))
    out = Path(args.out) if args.out else Path(_default_out(args.model))
    if not args.out:
        out = out.with_name(out.name + "-many")
    policy = TeachPolicy(
        max_rel_norm=args.max_rel_norm,
        max_kl_p95=args.max_kl,
        max_ppl_ratio=args.max_ppl_ratio,
        min_panel_agreement=args.min_panel_agreement,
    )

    def session_factory(path, budget):
        return Session.open(
            str(path),
            dtype=args.dtype,
            device=args.device,
            quantize=args.quantize,
            budget=budget,
        )

    try:
        result = teach_many(
            session_factory,
            str(model_path),
            facts,
            out,
            budget=args.budget,
            policy=policy,
            preserve_prompts=preserve,
            layers=layers,
            solver=args.solver,
            steps=args.steps,
            lr=args.lr,
            max_layers=args.max_layers,
            keep_steps=args.keep_steps,
            continue_on_reject=args.continue_on_reject,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        return _error(str(exc))
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print("Machaon · teach-many")
        print("model      : %s" % result.base_model)
        print("facts      : %d" % result.facts)
        print("budget     : %.4f cumulative (per-fact cap %.4f)" % (
            result.budget, policy.max_rel_norm,
        ))
        for step in result.steps:
            extra = ""
            if step.global_panel_agreement is not None:
                extra = " · global %d/%d, KL p95 %.3g" % (
                    round(step.global_panel_agreement * result.panel_size),
                    result.panel_size,
                    step.global_kl_p95 or 0.0,
                )
            print("step %-3d %-12s %s%s" % (
                step.index, step.status.upper(), step.prompt[:52], extra,
            ))
        print("final      : %s" % (result.final or "none (source unchanged)"))
        print("cumulative : %.4f" % result.cumulative_relative_delta)
        print("status     : %s" % result.status.upper())
        print("summary    : %s" % result.summary)
        if result.report_path:
            print("report     : %s" % result.report_path)
    return 0 if result.status in ("applied", "no_edits") else 1


def _cmd_start(args) -> int:
    import numpy as np

    print("Machaon · start")
    print("Weight-space programming with a contract, not a training loop.\n")
    print("0. Get a small checkpoint (one time, with mirror fallback):")
    print("   machaon fetch --starter qwen05b --dest ./models/Qwen2.5-0.5B")
    print("   # smaller: machaon fetch --starter smol135m --dest ./models/SmolLM2-135M")
    print()
    print("1. See the whole loop in one second (no model):")
    print("   machaon demo")
    print()
    print("2. Teach a local checkpoint in one sentence:")
    print('   machaon teach --model ./models/Qwen2.5-0.5B \\')
    print('     --say "The Eiffel Tower is in the city of -> Rome"')
    print()
    print("3. Use the local browser console:")
    print("   machaon-web --allowed-root ./models")
    print()
    print("capabilities")
    print("  python      %s" % platform.python_version())
    print("  numpy       %s" % np.__version__)
    for name in ("torch", "transformers", "safetensors", "fastapi"):
        version = _optional_version(name)
        print("  %-11s %s" % (name, version))
    if _optional_version("torch") == "absent":
        print('\ninstall real-model support with: pip install "machaon[hf]"')
    if _optional_version("fastapi") == "absent":
        print('install the browser console with: pip install "machaon[web]"')
    if args.demo:
        from .demo import main as demo_main

        print()
        demo_main()
    return 0


def _optional_version(module: str) -> str:
    try:
        return metadata.version(module)
    except metadata.PackageNotFoundError:
        return "absent"


def _cmd_doctor(args) -> int:
    import numpy as np

    report: Dict[str, Any] = {
        "machaon": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "packages": {
            name: _optional_version(name)
            for name in (
                "torch",
                "transformers",
                "safetensors",
                "pyarrow",
                "mcp",
            )
        },
        "core": {
            "linalg": True,
            "contract_synthesis": True,
            "transactions": True,
            "delta_store": True,
            "gates": True,
            "openwsp_v1": True,
        },
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("machaon %s" % report["machaon"])
    print("python  %s (%s)" % (report["python"], report["platform"]))
    print("numpy   %s" % report["numpy"])
    print("\ncore")
    for name, ready in report["core"].items():
        print("  %-14s %s" % (name, "ready" if ready else "unavailable"))
    print("\noptional")
    for name, version in report["packages"].items():
        print("  %-14s %s" % (name, version))
    if report["packages"]["torch"] == "absent":
        print("\ninstall real-model support with: pip install \"machaon[hf]\"")
    if report["packages"]["mcp"] == "absent":
        print("install the MCP server with:      pip install \"machaon[mcp]\"")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="machaon",
        description="Direct weight surgery for LLMs. Extract the arrow, leave the body intact.",
    )
    parser.add_argument("--version", action="version", version="machaon " + __version__)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="run the synthetic end-to-end demo (no downloads)")
    p_val = sub.add_parser("validate", help="validate an edit spec JSON")
    p_val.add_argument("path")
    p_gate = sub.add_parser("gate", help="check metrics JSON against the acceptance gates")
    p_gate.add_argument("path")
    p_wsp = sub.add_parser(
        "wsp",
        help="classify an edit as repair/acquisition/non-specific from a WSP report",
    )
    p_wsp.add_argument("path")
    p_wsp.add_argument("--save", default=None,
                       help="write a content-addressed evidence bundle")
    p_wsp.add_argument("--metadata", default=None,
                       help="optional JSON object to store with the evidence")
    p_v = sub.add_parser("wsp-verify", help="verify a WSP evidence bundle")
    p_v.add_argument("path")
    p_binding = sub.add_parser(
        "binding", help="verify an OpenWSP edit is bound to its WSP evidence"
    )
    p_binding.add_argument("edit_dir")
    p_binding.add_argument("evidence_dir")
    p_bench = sub.add_parser(
        "wsp-bench",
        help="run labelled obstruction-vs-capability WSP benchmark cases",
    )
    p_bench.add_argument("path")
    p_bench.add_argument(
        "--require-measured",
        action="store_true",
        help="refuse synthetic fixtures and accept only measured cases",
    )
    p_plan = sub.add_parser(
        "plan",
        help="turn failure and layer-site evidence into an edit/diagnose route",
    )
    p_plan.add_argument("path")
    p_diagnose = sub.add_parser(
        "diagnose",
        help="classify obstruction vs capability gap and produce a plan",
    )
    p_diagnose.add_argument("path")
    p_ins = sub.add_parser("inspect", help="list tensors in a safetensors checkpoint")
    p_ins.add_argument("path")
    p_ins.add_argument("--contains", default=None)
    p_ins.add_argument("--suffix", default=None)
    p_ins.add_argument("--layer", type=int, default=None)
    p_ins.add_argument("--limit", type=int, default=None)
    p_ins.add_argument("--json", action="store_true", help="machine-readable output")
    p_doc = sub.add_parser("doctor", help="report installed capabilities and versions")
    p_doc.add_argument("--json", action="store_true", help="machine-readable output")
    p_intent = sub.add_parser(
        "intent",
        help="parse a plain-language teaching request into an exact contract",
    )
    p_intent.add_argument("sentence")
    p_intent.add_argument("--json", action="store_true", help="machine-readable output")
    p_teach = sub.add_parser(
        "teach",
        help="teach one prompt/answer pair with automatic site search and verification",
    )
    p_teach.add_argument("--model", required=True, help="checkpoint directory with config, tokenizer and weights")
    p_teach.add_argument("--say", default=None, help="one-sentence teaching request")
    p_teach.add_argument("--prompt", default=None, help="prompt text (alternative to --say)")
    p_teach.add_argument("--answer", default=None, help="expected continuation")
    p_teach.add_argument("--layers", default=None, help="8,10,12 or 8-12; default: a depth window")
    p_teach.add_argument("--max-layers", type=int, default=7, help="default candidate count")
    p_teach.add_argument(
        "--solver", choices=("direction", "gradient"), default="direction"
    )
    p_teach.add_argument("--steps", type=int, default=250, help="gradient solver steps")
    p_teach.add_argument("--lr", type=float, default=0.02, help="gradient solver learning rate")
    p_teach.add_argument(
        "--max-rel-norm", type=float, default=0.02,
        help="relative budget for the edit (default 0.02)",
    )
    p_teach.add_argument(
        "--budget", type=float, default=0.10,
        help="cumulative session budget (default 0.10)",
    )
    p_teach.add_argument(
        "--max-kl", type=float, default=0.05, help="KL p95 acceptance threshold"
    )
    p_teach.add_argument(
        "--max-ppl-ratio", type=float, default=1.02,
        help="perplexity-ratio threshold when a corpus is supplied",
    )
    p_teach.add_argument(
        "--min-panel-agreement", type=float, default=0.875,
        help="fraction of unrelated prompts that must keep their top-1 token",
    )
    p_teach.add_argument(
        "--preserve-file", default=None,
        help="newline-separated unrelated prompts for the baseline and projection",
    )
    p_teach.add_argument(
        "--corpus-file", default=None,
        help="newline-separated texts for an optional perplexity check",
    )
    p_teach.add_argument("--max-evaluations", type=int, default=None)
    p_teach.add_argument("--dtype", default="float32")
    p_teach.add_argument("--device", default=None)
    p_teach.add_argument("--quantize", choices=("int4", "int8"), default=None)
    p_teach.add_argument(
        "--out", default=None,
        help="write the edited checkpoint here (default: <model>-machaon)",
    )
    p_teach.add_argument("--no-save", action="store_true", help="keep the edit in memory only")
    p_teach.add_argument("--edit-out", default=None, help="export the applied delta as OpenWSP")
    p_teach.add_argument("--evidence-out", default=None, help="write a WSP evidence bundle")
    p_teach.add_argument("--report-out", default=None, help="write the full recipe report JSON")
    p_teach.add_argument("--dry-run", action="store_true", help="search without writing weights")
    p_teach.add_argument("--json", action="store_true", help="machine-readable output")
    p_fetch = sub.add_parser(
        "fetch", help="download a safetensors checkpoint without loading it"
    )
    p_fetch.add_argument("--repo", default=None, help="Hugging Face repository id")
    p_fetch.add_argument(
        "--starter", choices=("smol135m", "qwen05b", "gpt2"), default=None,
        help="use a recommended starter checkpoint",
    )
    p_fetch.add_argument("--dest", default=None, help="destination directory")
    p_fetch.add_argument(
        "--endpoint", choices=("huggingface", "hf-mirror", "modelscope"),
        default=None, help="force one endpoint",
    )
    p_fetch.add_argument("--list", action="store_true", help="list starters and exit")
    p_fetch.add_argument("--json", action="store_true", help="machine-readable output")
    p_many = sub.add_parser(
        "teach-many",
        help="teach several facts into one checkpoint chain",
    )
    p_many.add_argument("--model", required=True, help="base checkpoint directory")
    p_many.add_argument(
        "--facts", required=True,
        help="text (one sentence per line), JSON or JSONL facts file",
    )
    p_many.add_argument("--out", default=None, help="output container directory")
    p_many.add_argument("--budget", type=float, default=0.10, help="cumulative budget")
    p_many.add_argument(
        "--max-rel-norm", type=float, default=0.02, help="per-fact relative cap"
    )
    p_many.add_argument("--max-kl", type=float, default=0.05)
    p_many.add_argument("--max-ppl-ratio", type=float, default=1.02)
    p_many.add_argument("--min-panel-agreement", type=float, default=0.875)
    p_many.add_argument("--preserve-file", default=None)
    p_many.add_argument("--layers", default=None)
    p_many.add_argument("--max-layers", type=int, default=7)
    p_many.add_argument("--solver", choices=("direction", "gradient"), default="direction")
    p_many.add_argument("--steps", type=int, default=250)
    p_many.add_argument("--lr", type=float, default=0.02)
    p_many.add_argument("--dtype", default="float32")
    p_many.add_argument("--device", default=None)
    p_many.add_argument("--quantize", choices=("int4", "int8"), default=None)
    p_many.add_argument(
        "--continue-on-reject", action="store_true",
        help="record a rejected fact and continue with the next one",
    )
    p_many.add_argument(
        "--keep-steps", action="store_true",
        help="keep every step checkpoint, not only the last safe one",
    )
    p_many.add_argument("--json", action="store_true")
    p_start = sub.add_parser("start", help="print the shortest path to a first edit")
    p_start.add_argument("--demo", action="store_true", help="also run the synthetic demo")

    args = parser.parse_args(argv)
    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "gate":
        return _cmd_gate(args)
    if args.command == "wsp":
        return _cmd_wsp(args)
    if args.command == "wsp-verify":
        return _cmd_wsp_verify(args)
    if args.command == "binding":
        return _cmd_binding(args)
    if args.command == "wsp-bench":
        return _cmd_wsp_bench(args)
    if args.command == "plan":
        return _cmd_plan(args)
    if args.command == "diagnose":
        return _cmd_diagnose(args)
    if args.command == "inspect":
        return _cmd_inspect(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "intent":
        return _cmd_intent(args)
    if args.command == "teach":
        return _cmd_teach(args)
    if args.command == "teach-many":
        return _cmd_teach_many(args)
    if args.command == "fetch":
        return _cmd_fetch(args)
    if args.command == "start":
        return _cmd_start(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
