"""
运行时数据上下文
DataContext 封装不可变数据，模型只能读不能写
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DataContext:
    """不可变数据上下文"""

    Y: np.ndarray
    var_names: list
    quarter_labels: list
    mean: Optional[np.ndarray]
    std: Optional[np.ndarray]

    @classmethod
    def create(cls, Y, var_names, quarter_labels, mean=None, std=None):
        """创建 DataContext，强制拷贝所有输入"""
        return cls(
            Y=np.array(Y, copy=True),
            var_names=list(var_names),
            quarter_labels=list(quarter_labels),
            mean=mean.copy() if mean is not None else None,
            std=std.copy() if std is not None else None,
        )

    def replace_Y(self, Y_new) -> "DataContext":
        """返回新的 DataContext（替换 Y，不修改原对象）"""
        return DataContext(
            Y=np.array(Y_new, copy=True),
            var_names=list(self.var_names),
            quarter_labels=list(self.quarter_labels),
            mean=self.mean.copy() if self.mean is not None else None,
            std=self.std.copy() if self.std is not None else None,
        )

    @property
    def n_vars(self) -> int:
        return self.Y.shape[1]

    @property
    def n_obs(self) -> int:
        return self.Y.shape[0]
