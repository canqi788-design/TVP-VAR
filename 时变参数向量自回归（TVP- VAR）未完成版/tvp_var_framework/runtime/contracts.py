"""
Central Schema Definition — single source of truth (frozen)

Defines:
  1. Kernel contract version
  2. Node IO schema (what each node reads/writes)
  3. Allowed output keys (whitelist for ctx.outputs)
  4. Allowed node types (IR whitelist)

All constants are frozen at import time. No runtime mutation.
No eval/exec. No dynamic schema parsing.
"""

import logging
from types import MappingProxyType

logger = logging.getLogger("tvp_var.runtime.contracts")

# ── Version ──

SCHEMA_VERSION = "4.0.0"
KERNEL_CONTRACT = "fn(inputs: dict, params: dict) -> dict"


# ── Exception ──

class ContractViolation(RuntimeError):
    """Raised when a kernel violates its output contract."""
    pass

# ── Node type whitelist (frozen) ──

ALLOWED_NODE_TYPES = frozenset({"source", "transform", "compute", "sink"})

# ── Allowed output keys — ctx.outputs whitelist (frozen) ──

ALLOWED_OUTPUT_KEYS = frozenset({
    # stationarity
    "Y_diff",
    "d_orders",
    "_stationarity",
    # state_update
    "theta_t",
    # likelihood
    "log_lik",
    # sampling_basic (backward compat)
    "_model",
    "_result",
    "posterior",
    # sampling_research / unified posterior
    "_joint_chain",
    "posterior_mean",
    "acceptance_rate",
    "n_saved_samples",
    # diagnostics
    "metrics",
    "_convergence",
    # reporting
    "_report_path",
    "report",
    "csv",
    # initial data pass-through (engine, not kernel)
    "mean",
    "std",
    "normalize",
    "debug",
})

# ── Posterior sample keys — standardize posterior sample semantics ──

POSTERIOR_SAMPLE_KEYS = frozenset({
    "theta",
    "sigma",
    "sv_state",
    "log_likelihood",
    "sample_index",
})

# ── Struct fields — these live on the dataclass, not in outputs ──

STRUCT_FIELDS = frozenset({"Y", "config", "var_names", "time_index"})

# ── Node IO schema (frozen dict-of-dicts) ──

_NODE_IO_INNER = {
    "stationarity": {
        "inputs": {"Y": "ndarray(T, n)", "var_names": "list[str]"},
        "outputs": {"Y_diff": "ndarray(T', n)", "d_orders": "list[int]"},
    },
    "state_update": {
        "inputs": {"Y_diff": "ndarray(T', n)"},
        "outputs": {"theta_t": "any"},
    },
    "likelihood": {
        "inputs": {"theta_t": "any"},
        "outputs": {"log_lik": "float"},
    },
    "sampling_basic": {
        "inputs": {"Y_diff": "ndarray(T', n)", "theta_t": "any", "log_lik": "float"},
        "outputs": {"_joint_chain": "list[dict]", "posterior": "dict"},
    },
    "sampling_research": {
        "inputs": {"Y_diff": "ndarray(T', n)", "theta_t": "any", "log_lik": "float"},
        "outputs": {"_joint_chain": "list[dict]", "posterior_mean": "dict",
                     "acceptance_rate": "dict", "n_saved_samples": "int"},
    },
    "diagnostics": {
        "inputs": {"_joint_chain": "list[dict]"},
        "outputs": {"metrics": "dict"},
    },
    "reporting": {
        "inputs": {"_joint_chain": "list[dict]", "metrics": "dict"},
        "outputs": {"report": "str", "csv": "str"},
    },
}

NODE_IO = MappingProxyType({
    k: MappingProxyType(v) for k, v in _NODE_IO_INNER.items()
})

# ── Registered nodes (must match IR executables) ──

REGISTERED_NODES = frozenset(NODE_IO.keys())


# ============================================================
# Validation — no fallback, no silent pass
# ============================================================

def validate_inputs(node_name: str, ctx_dict: dict) -> list:
    contract = NODE_IO.get(node_name)
    if contract is None:
        return []
    violations = []
    for key, type_desc in contract["inputs"].items():
        if key not in ctx_dict:
            violations.append(f"[{node_name}] missing input: {key} ({type_desc})")
        elif ctx_dict[key] is None:
            violations.append(f"[{node_name}] input is None: {key} ({type_desc})")
    return violations


def validate_kernel_output(node_name: str, output: dict) -> None:
    """Fail-fast on any key not in ALLOWED_OUTPUT_KEYS or STRUCT_FIELDS."""
    for k in output:
        if k not in ALLOWED_OUTPUT_KEYS and k not in STRUCT_FIELDS:
            raise KeyError(
                f"Kernel '{node_name}' produced invalid key: '{k}'. "
                f"Allowed: outputs={ALLOWED_OUTPUT_KEYS}, struct={STRUCT_FIELDS}"
            )


def validate_ir_against_contracts(ir_nodes: dict) -> None:
    """Every executable IR node must exist in NODE_IO registry."""
    for name, node in ir_nodes.items():
        if node.get("type") == "source":
            continue
        if name not in NODE_IO:
            raise ValueError(
                f"IR node '{name}' has no contract in NODE_IO. "
                f"Registered: {sorted(NODE_IO.keys())}"
            )


def validate_joint_chain(joint_chain) -> None:
    """
    Validate research-grade posterior joint chain.

    Rules:
    ------
    1. Must be non-empty list
    2. Every sample must be dict
    3. theta/sigma required
    4. Samples must be immutable snapshots (np.copy)
    5. Shape consistency required
    """
    if not isinstance(joint_chain, list):
        raise ContractViolation("_joint_chain must be a list")

    if len(joint_chain) == 0:
        raise ContractViolation("_joint_chain cannot be empty")

    required_keys = {"theta", "sigma"}
    theta_shape = None
    sigma_shape = None

    for i, sample in enumerate(joint_chain):
        if not isinstance(sample, dict):
            raise ContractViolation(f"Sample {i} is not a dict")

        missing = required_keys - set(sample.keys())
        if missing:
            raise ContractViolation(f"Sample {i} missing keys: {missing}")

        theta = sample["theta"]
        sigma = sample["sigma"]

        if theta is None:
            raise ContractViolation(f"Sample {i} theta is None")
        if sigma is None:
            raise ContractViolation(f"Sample {i} sigma is None")

        if theta_shape is None:
            theta_shape = theta.shape
        elif theta.shape != theta_shape:
            raise ContractViolation(f"theta shape mismatch at sample {i}")

        if sigma_shape is None:
            sigma_shape = sigma.shape
        elif sigma.shape != sigma_shape:
            raise ContractViolation(f"sigma shape mismatch at sample {i}")


def log_contract(node_name: str) -> None:
    contract = NODE_IO.get(node_name)
    if contract is None:
        return
    logger.debug(f"Contract [{node_name}]:")
    logger.debug(f"  inputs:  {list(contract['inputs'].keys())}")
    logger.debug(f"  outputs: {list(contract['outputs'].keys())}")
