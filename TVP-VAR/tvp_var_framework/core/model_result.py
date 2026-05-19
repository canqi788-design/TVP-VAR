"""
统一的模型结果结构
所有 TVP-VAR 模型返回 ModelResult，下游代码不再猜测 dict key
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class ModelResult:
    """标准化的模型输出容器"""

    model_name: str

    # 预测
    pred_mean: Optional[np.ndarray] = None
    pred_lower: Optional[np.ndarray] = None
    pred_upper: Optional[np.ndarray] = None
    future_labels: Optional[list] = None

    # 脉冲响应
    irf_mean: Optional[np.ndarray] = None
    irf_lower: Optional[np.ndarray] = None
    irf_upper: Optional[np.ndarray] = None

    # 模型评估
    log_likelihood: Optional[float] = None
    bic: Optional[float] = None

    # 后验链（chain_A, chain_Q, chain_theta 等）
    chains: Dict[str, np.ndarray] = field(default_factory=dict)

    # 后验摘要
    posterior_summary: Optional[Dict] = None

    # 诊断信息（R-hat, ESS 等）
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # 元数据（迭代数、先验参数等）
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_prediction(self) -> bool:
        return self.pred_mean is not None

    def has_irf(self) -> bool:
        return self.irf_mean is not None

    def has_chains(self) -> bool:
        return len(self.chains) > 0

    def get_chain(self, name: str) -> Optional[np.ndarray]:
        return self.chains.get(name)
