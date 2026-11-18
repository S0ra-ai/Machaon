"""Machaon — direct weight surgery for LLMs.

Extract the arrow. Leave the body intact.

The core is pure NumPy so the maths is testable anywhere; PyTorch and
Hugging Face are optional adapters.
"""
from __future__ import annotations

from .action import (
    ActionError,
    ActionExecutionError,
    ActionResult,
    ActionSpec,
    ActionValidationError,
    execute_action,
    validate_action_source,
)
from .batch import BatchSolution, solve_batch
from .binding import BindingReport, verify_binding
from .contract import ContractSolution, synthesize_contract
from .evidence import (
    EvidenceError,
    evidence_sha256,
    is_wsp_evidence,
    read_evidence,
    write_evidence,
)
from .executor import EditResult, execute_spec
from .fingerprint import (
    TensorFingerprint,
    diff_tensors,
    model_fingerprint,
    tensor_fingerprint,
)
from .headroom import (
    DecodingConfig,
    generation_accuracy,
    make_generation_headroom,
    make_next_token_headroom,
    next_token_accuracy,
    score_generation_texts,
)
from .intent import IntentError, TeachIntent, parse_intent
from .linalg import (
    ablate_input,
    ablate_output,
    bake_direction,
    clamp_rel_norm,
    low_rank_scale,
    nullspace_basis,
    nullspace_projector,
    project_complement,
    rank1,
    rank1_constrained,
    relative_norm,
)
from .mechanism import (
    MechanismProtocol,
    MechanismVerdict,
    assess_mechanism,
    generation_mechanism,
    generation_metrics,
    records_metrics,
)
from .planning import (
    DiagnosisProtocol,
    EditPlan,
    FailureDiagnosis,
    diagnose_and_plan,
    diagnose_failure,
    plan_surgery,
    scores_from_patch_report,
    search_layers,
)
from .recipes import (
    DEFAULT_BASELINE_PROMPTS,
    TeachPolicy,
    TeachResult,
    candidate_layers,
    teach,
    teach_fact,
)
from .schema import Constraints, EditSpec, Target
from .symmetry import (
    MergedMLP,
    MLPAlignment,
    UnitAlignment,
    align_mlp_hidden,
    align_rows,
    functional_alignment_check,
    merge_mlp_aligned,
    mlp_outputs,
    permute_columns,
    permute_rows,
)
from .task_arithmetic import (
    TaskVector,
    TaskVectorCombination,
    combine_task_vectors,
    task_vector,
)
from .transaction import Transaction, Workspace
from .wsp import (
    EffectVerdict,
    WSPProtocol,
    assess_report,
    assess_wsp,
    classify_effect,
    random_control,
    random_controls,
)

__version__ = "0.4.3"

__all__ = [
    "rank1",
    "rank1_constrained",
    "bake_direction",
    "ablate_output",
    "ablate_input",
    "low_rank_scale",
    "nullspace_basis",
    "nullspace_projector",
    "project_complement",
    "clamp_rel_norm",
    "relative_norm",
    "ContractSolution",
    "synthesize_contract",
    "BatchSolution",
    "solve_batch",
    "ActionSpec",
    "ActionResult",
    "ActionError",
    "ActionValidationError",
    "ActionExecutionError",
    "validate_action_source",
    "execute_action",
    "BindingReport",
    "verify_binding",
    "EditResult",
    "execute_spec",
    "TensorFingerprint",
    "tensor_fingerprint",
    "model_fingerprint",
    "diff_tensors",
    "EvidenceError",
    "evidence_sha256",
    "write_evidence",
    "read_evidence",
    "is_wsp_evidence",
    "DecodingConfig",
    "generation_accuracy",
    "score_generation_texts",
    "next_token_accuracy",
    "make_generation_headroom",
    "make_next_token_headroom",
    "MechanismProtocol",
    "MechanismVerdict",
    "generation_metrics",
    "records_metrics",
    "generation_mechanism",
    "assess_mechanism",
    "EditPlan",
    "DiagnosisProtocol",
    "FailureDiagnosis",
    "diagnose_failure",
    "diagnose_and_plan",
    "plan_surgery",
    "scores_from_patch_report",
    "search_layers",
    "DEFAULT_BASELINE_PROMPTS",
    "TeachPolicy",
    "TeachResult",
    "candidate_layers",
    "teach",
    "teach_fact",
    "IntentError",
    "TeachIntent",
    "parse_intent",
    "EditSpec",
    "Target",
    "Constraints",
    "Workspace",
    "Transaction",
    "WSPProtocol",
    "EffectVerdict",
    "random_control",
    "random_controls",
    "classify_effect",
    "assess_wsp",
    "assess_report",
    "UnitAlignment",
    "MLPAlignment",
    "permute_rows",
    "permute_columns",
    "align_rows",
    "align_mlp_hidden",
    "TaskVector",
    "TaskVectorCombination",
    "task_vector",
    "combine_task_vectors",
    "MergedMLP",
    "merge_mlp_aligned",
    "mlp_outputs",
    "functional_alignment_check",
]
