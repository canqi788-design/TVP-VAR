"""
Typed Execution Context — hardened

5 fields. outputs whitelist enforced on write.
No quarter_labels alias. No hidden struct mutation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from tvp_var_framework.runtime.contracts import ALLOWED_OUTPUT_KEYS


@dataclass
class ExecutionContext:
    Y: Optional[np.ndarray] = None
    config: Dict[str, Any] = field(default_factory=dict)
    var_names: List[str] = field(default_factory=list)
    time_index: List[str] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)

    def update(self, key: str, value: Any):
        if key not in ALLOWED_OUTPUT_KEYS:
            raise KeyError(f"Invalid output key: {key}")
        self.outputs[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Y": self.Y,
            "config": self.config,
            "var_names": self.var_names,
            "time_index": self.time_index,
            **self.outputs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionContext":
        known = {"Y", "config", "var_names", "time_index"}
        ctx = cls(
            Y=d.get("Y"),
            config=d.get("config", {}),
            var_names=d.get("var_names", []),
            time_index=d.get("time_index", []),
        )
        for k, v in d.items():
            if k not in known:
                ctx.outputs[k] = v
        return ctx
