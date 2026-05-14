"""
TVP-VAR 通用数据加载器
支持 CSV 文件、年度字典、JSON 配置等多种数据源
"""

import numpy as np
import json
import csv
import os


def load_csv(path, var_columns=None):
    """
    从 CSV 加载数据
    CSV 格式: 第一行为表头, 后续行为数据
    var_columns: 指定使用的列名列表, None 则使用除第一列外所有数值列

    返回: (Y, var_names) — numpy 数组 + 变量名列表
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    if var_columns is None:
        # 自动检测数值列 (跳过第一列, 假设为时间/标签列)
        var_columns = []
        for h in headers[1:]:
            try:
                float(rows[0][h])
                var_columns.append(h)
            except (ValueError, KeyError):
                continue
    else:
        # 校验请求的列名是否存在于 CSV 表头
        missing = [col for col in var_columns if col not in headers]
        if missing:
            raise KeyError(
                f"CSV 列名不匹配: {missing} 不存在于 {path}\n"
                f"  可用列: {headers}\n"
                f"  请检查 config 中的 vars / data_ingestion.vars 是否与 CSV 表头一致"
            )

    # 提取时间标签 (第一列)
    time_col = headers[0]
    time_labels = [row[time_col] for row in rows]

    data = []
    for row in rows:
        row_data = []
        for col in var_columns:
            val = row[col]
            if val and val.strip():
                row_data.append(float(val))
            else:
                row_data.append(np.nan)
        data.append(row_data)

    Y = np.array(data)
    return Y, var_columns, time_labels


def load_json_config(path):
    """
    读取 JSON 配置文件
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def build_quarterly(yearly_dict, keys, noise_scale=0.02):
    """
    从年度数据构建季度序列
    yearly_dict: {year: {key: value, ...}, ...}
    keys: 要提取的键名列表
    noise_scale: 季度噪声比例

    返回: (Y_raw, quarter_labels)
    """
    flow_keys = {"revenue", "depreciation", "net_profit", "op_profit", "rd",
                 "cost", "expense", "capex", "fcf", "ebitda"}

    quarters = []
    quarter_labels = []
    for year in sorted(yearly_dict.keys()):
        d = yearly_dict[year]
        for q in range(4):
            row = []
            for k in keys:
                base = d[k] / 4 if k in flow_keys else d[k]
                seasonal = 1.0 + 0.03 * np.sin(q * np.pi / 2)
                noise = np.random.normal(0, abs(base) * noise_scale)
                row.append(base * seasonal + noise)
            quarters.append(row)
            quarter_labels.append(f"{year}Q{q + 1}")

    return np.array(quarters), quarter_labels


def normalize(Y):
    """Z-score 标准化"""
    mean = Y.mean(axis=0)
    std = Y.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)  # 防止除零
    return (Y - mean) / std, mean, std


def load_company_data(data_source, vars_to_use, noise_scale=0.02, normalize_data=True):
    """
    统一数据加载入口
    data_source: CSV 文件路径 或 年度字典
    vars_to_use: 变量列表

    返回: (Y, quarter_labels, var_names, mean, std)
    """
    np.random.seed(42)

    if isinstance(data_source, str):
        # CSV 文件
        Y_raw, var_names, quarter_labels = load_csv(data_source, vars_to_use)
    elif isinstance(data_source, dict):
        # 年度字典
        Y_raw, quarter_labels = build_quarterly(data_source, vars_to_use, noise_scale)
        var_names = vars_to_use
    else:
        raise ValueError(f"不支持的数据源类型: {type(data_source)}")

    # 处理 NaN: 前向填充 + 后向填充
    if np.any(np.isnan(Y_raw)):
        n_nan = np.sum(np.isnan(Y_raw))
        for j in range(Y_raw.shape[1]):
            col = Y_raw[:, j]
            mask = np.isnan(col)
            if np.all(mask):
                continue
            # 前向填充
            idx = np.where(~mask, np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            col[mask] = col[idx[mask]]
            # 后向填充剩余NaN
            mask = np.isnan(col)
            if np.any(mask):
                idx = np.where(~mask, np.arange(len(col)), len(col) - 1)
                idx = np.minimum.accumulate(idx[::-1])[::-1]
                col[mask] = col[idx[mask]]
            Y_raw[:, j] = col
        import logging
        logging.getLogger("tvp_var").info(f"  NaN 插值完成: {n_nan} 个缺失值已填充")

    if normalize_data:
        Y, mean, std = normalize(Y_raw)
    else:
        Y = Y_raw
        mean = np.zeros(Y.shape[1])
        std = np.ones(Y.shape[1])

    return Y, quarter_labels, var_names, mean, std
