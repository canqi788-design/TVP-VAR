"""
TVP-VAR 模型统一接口
所有模型实现 BaseTVPVARModel，pipeline 通过统一接口调用
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

from .model_result import ModelResult


class BaseTVPVARModel(ABC):
    """TVP-VAR 模型基类"""

    @abstractmethod
    def fit(self, Y: np.ndarray, **kwargs) -> ModelResult:
        """拟合模型，返回 ModelResult"""
        ...

    @abstractmethod
    def predict(self, steps: int, **kwargs) -> ModelResult:
        """预测，返回含 pred_mean/lower/upper 的 ModelResult"""
        ...

    @abstractmethod
    def compute_irf(self, shock_var: int, periods: int, **kwargs) -> ModelResult:
        """计算脉冲响应，返回含 irf_mean/lower/upper 的 ModelResult"""
        ...

    @abstractmethod
    def diagnostics(self) -> Dict[str, Any]:
        """返回诊断信息（R-hat, ESS 等）"""
        ...

    def get_chains(self) -> Dict[str, np.ndarray]:
        """返回后验链，默认从 fit() 结果中获取"""
        return {}

    def summary(self, var_names: Optional[list] = None) -> str:
        """返回文本摘要"""
        return f"{self.__class__.__name__}: no summary available"
