"""
TVP-VAR 通用数据加载器
支持 CSV 文件、年度字典、JSON 配置等多种数据源
"""

import numpy as np
import json
import csv
import os
import re


def _parse_numeric_value(value):
    if value is None:
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def _forward_backward_fill(Y_raw):
    """Fill missing values column-wise with forward then backward fill."""
    Y_filled = np.array(Y_raw, copy=True)
    n_nan = int(np.sum(np.isnan(Y_filled)))
    for j in range(Y_filled.shape[1]):
        col = Y_filled[:, j]
        mask = np.isnan(col)
        if np.all(mask):
            continue
        idx = np.where(~mask, np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        col[mask] = col[idx[mask]]
        mask = np.isnan(col)
        if np.any(mask):
            idx = np.where(~mask, np.arange(len(col)), len(col) - 1)
            idx = np.minimum.accumulate(idx[::-1])[::-1]
            col[mask] = col[idx[mask]]
        Y_filled[:, j] = col
    return Y_filled, n_nan


def _read_csv_numeric_table(path, var_columns=None):
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    if not headers:
        raise ValueError(f"CSV 文件为空或缺少表头: {path}")

    time_col = headers[0]
    time_labels = [row[time_col] for row in rows]

    if var_columns is None:
        var_columns = []
        for h in headers[1:]:
            parsed = []
            for row in rows:
                try:
                    parsed.append(_parse_numeric_value(row.get(h)))
                except (ValueError, KeyError):
                    parsed.append(np.nan)
            if np.any(np.isfinite(parsed)):
                var_columns.append(h)
    else:
        missing = [col for col in var_columns if col not in headers]
        if missing:
            raise KeyError(
                f"CSV 列名不匹配: {missing} 不存在于 {path}\n"
                f"  可用列: {headers}\n"
                f"  请检查 config 中的 vars / data_ingestion.vars 是否与 CSV 表头一致"
            )

    data = []
    for row in rows:
        row_data = []
        for col in var_columns:
            try:
                row_data.append(_parse_numeric_value(row.get(col)))
            except ValueError:
                row_data.append(np.nan)
        data.append(row_data)

    return np.asarray(data, dtype=float), list(var_columns), time_labels


def load_csv(path, var_columns=None):
    """
    从 CSV 加载数据
    CSV 格式: 第一行为表头, 后续行为数据
    var_columns: 指定使用的列名列表, None 则使用除第一列外所有数值列

    返回: (Y, var_names) — numpy 数组 + 变量名列表
    """
    return _read_csv_numeric_table(path, var_columns)


def load_json_config(path):
    """
    读取 JSON 配置文件
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def build_quarterly(yearly_dict, keys, noise_scale=0.02, rng=None):
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
                generator = rng if rng is not None else np.random.default_rng()
                noise = generator.normal(0, abs(base) * noise_scale)
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


def _parse_quarter_label(label):
    text = str(label).strip()
    match = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", text)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        quarter = (month - 1) // 3 + 1
        return year, quarter

    match = re.match(r"^(\d{4})Q([1-4])$", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))

    return None, None


def _looks_like_ytd(labels, Y_raw):
    parsed = [_parse_quarter_label(label) for label in labels]
    if not parsed or any(year is None for year, _ in parsed):
        return False

    seen_q1 = any(q == 1 for _, q in parsed)
    has_reset = False
    monotone_runs = 0
    total_runs = 0
    for idx in range(1, len(parsed)):
        prev_year, prev_q = parsed[idx - 1]
        year, quarter = parsed[idx]
        prev = Y_raw[idx - 1]
        curr = Y_raw[idx]
        if year == prev_year and quarter > prev_q:
            total_runs += 1
            if np.nanmean(curr >= prev) >= 0.6:
                monotone_runs += 1
        elif year > prev_year and quarter == 1:
            if np.nanmean(curr < prev) >= 0.6:
                has_reset = True

    return seen_q1 and has_reset and total_runs > 0 and monotone_runs / total_runs >= 0.6


def ytd_to_single_period(Y_raw, labels):
    """
    Convert year-to-date financial statement flows to single-period flows.

    Q1 stays unchanged. Q2-Q4 become current cumulative minus previous
    quarter cumulative in the same fiscal year.
    """
    Y_raw = np.asarray(Y_raw, dtype=float)
    converted = Y_raw.copy()
    parsed = [_parse_quarter_label(label) for label in labels]
    for idx in range(1, len(labels)):
        prev_year, prev_q = parsed[idx - 1]
        year, quarter = parsed[idx]
        if year is not None and year == prev_year and quarter is not None and quarter > prev_q:
            converted[idx] = Y_raw[idx] - Y_raw[idx - 1]
    return converted


_FLOW_RATIO_SPECS = {
    "毛利率": ("营业总收入_亿元", "营业成本_亿元", "gross_margin"),
    "销售费用率": ("销售费用_亿元", "营业总收入_亿元", "expense_ratio"),
    "管理费用率": ("管理费用_亿元", "营业总收入_亿元", "expense_ratio"),
    "净利率": ("净利润_亿元", "营业总收入_亿元", "ratio"),
}


def _safe_ratio(numerator, denominator):
    denom = np.asarray(denominator, dtype=float)
    num = np.asarray(numerator, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    valid = np.isfinite(num) & np.isfinite(denom) & (np.abs(denom) > 1e-12)
    out[valid] = num[valid] / denom[valid]
    return out


def ytd_financial_table_to_single_period(Y_raw, labels, var_names):
    """
    Convert an A-share style cumulative financial table into single-quarter data.

    Flow amount columns are converted by within-year first difference. Derived
    ratio columns are then rebuilt from the converted flow amounts instead of
    being differenced directly.
    """
    Y_raw = np.asarray(Y_raw, dtype=float)
    var_names = list(var_names)
    converted = ytd_to_single_period(Y_raw, labels)
    name_to_idx = {name: idx for idx, name in enumerate(var_names)}

    for ratio_name, (left_name, right_name, formula) in _FLOW_RATIO_SPECS.items():
        if ratio_name not in name_to_idx or left_name not in name_to_idx or right_name not in name_to_idx:
            continue

        ratio_idx = name_to_idx[ratio_name]
        left = converted[:, name_to_idx[left_name]]
        right = converted[:, name_to_idx[right_name]]
        if formula == "gross_margin":
            converted[:, ratio_idx] = _safe_ratio(left - right, left)
        else:
            converted[:, ratio_idx] = _safe_ratio(left, right)

    return converted


def apply_seasonal_transform(Y_raw, labels, transform="none"):
    """Apply quarterly seasonal transforms after flow normalization."""
    transform = (transform or "none").lower()
    Y_raw = np.asarray(Y_raw, dtype=float)
    labels = list(labels)

    if transform in {"none", "raw"}:
        return Y_raw, labels

    if transform == "ttm":
        if Y_raw.shape[0] < 4:
            raise ValueError("TTM 需要至少 4 个季度观测")
        rows = []
        new_labels = []
        for idx in range(3, Y_raw.shape[0]):
            rows.append(np.sum(Y_raw[idx - 3:idx + 1], axis=0))
            new_labels.append(labels[idx])
        return np.asarray(rows, dtype=float), new_labels

    if transform == "yoy":
        if Y_raw.shape[0] < 5:
            raise ValueError("YoY 需要至少 5 个季度观测")
        prev = Y_raw[:-4]
        curr = Y_raw[4:]
        denom = np.where(np.abs(prev) < 1e-12, np.nan, prev)
        yoy = (curr - prev) / denom
        if np.any(~np.isfinite(yoy)):
            raise ValueError("YoY 变换遇到零或非有限基期值")
        return yoy, labels[4:]

    raise ValueError(f"不支持的 seasonal_transform: {transform}")


def load_company_data(
    data_source,
    vars_to_use,
    noise_scale=0.02,
    normalize_data=True,
    handle_missing="none",
    random_state=None,
    flow_transform="auto",
    seasonal_transform="none",
):
    """
    统一数据加载入口
    data_source: CSV 文件路径 或 年度字典
    vars_to_use: 变量列表

    返回: (Y, quarter_labels, var_names, mean, std)
    """
    rng = np.random.default_rng(random_state)
    flow_transform = (flow_transform or "none").lower()
    if flow_transform not in {"none", "auto", "ytd_to_quarter"}:
        raise ValueError(f"不支持的 flow_transform: {flow_transform}")

    if isinstance(data_source, str):
        # CSV 文件
        if flow_transform in {"auto", "ytd_to_quarter"}:
            Y_all, all_names, quarter_labels = _read_csv_numeric_table(data_source, None)
            should_convert = flow_transform == "ytd_to_quarter" or _looks_like_ytd(quarter_labels, Y_all)
            if should_convert:
                Y_all = ytd_financial_table_to_single_period(Y_all, quarter_labels, all_names)
                import logging

                logging.getLogger("tvp_var").info("  A股累计口径已转换为单季口径，并重算派生财务比率")

            missing = [col for col in vars_to_use if col not in all_names]
            if missing:
                raise KeyError(
                    f"CSV 列名不匹配: {missing} 不存在于 {data_source}\n"
                    f"  可用列: {all_names}\n"
                    f"  请检查 config 中的 vars / data_ingestion.vars 是否与 CSV 表头一致"
                )
            idx = [all_names.index(col) for col in vars_to_use]
            Y_raw = Y_all[:, idx]
            var_names = list(vars_to_use)
        else:
            Y_raw, var_names, quarter_labels = load_csv(data_source, vars_to_use)
    elif isinstance(data_source, dict):
        # 年度字典
        Y_raw, quarter_labels = build_quarterly(data_source, vars_to_use, noise_scale, rng=rng)
        var_names = vars_to_use
    else:
        raise ValueError(f"不支持的数据源类型: {type(data_source)}")

    if np.any(np.isnan(Y_raw)):
        import logging

        if handle_missing == "drop":
            row_mask = np.any(np.isnan(Y_raw), axis=1)
            dropped = int(np.sum(row_mask))
            Y_raw = Y_raw[~row_mask]
            quarter_labels = [q for q, keep in zip(quarter_labels, ~row_mask) if keep]
            logging.getLogger("tvp_var").info(f"  NaN 行删除完成: {dropped} 行已移除")
        elif handle_missing == "interpolate":
            Y_raw, n_nan = _forward_backward_fill(Y_raw)
            logging.getLogger("tvp_var").info(f"  NaN 插值完成: {n_nan} 个缺失值已填充")

    if not isinstance(data_source, str) and flow_transform == "ytd_to_quarter":
        Y_raw = ytd_to_single_period(Y_raw, quarter_labels)

    Y_raw, quarter_labels = apply_seasonal_transform(Y_raw, quarter_labels, seasonal_transform)

    if normalize_data:
        Y, mean, std = normalize(Y_raw)
    else:
        Y = Y_raw
        mean = np.zeros(Y.shape[1])
        std = np.ones(Y.shape[1])

    return Y, quarter_labels, var_names, mean, std


def split_exogenous_columns(Y, var_names, exog_vars):
    """
    将已加载的变量矩阵拆分为内生 Y 与外生 X。
    返回: (Y_endog, X_exog, endog_names, exog_names)
    """
    if not exog_vars:
        return Y, None, list(var_names), []

    var_names = list(var_names)
    exog_set = set(exog_vars)
    exog_idx = [i for i, name in enumerate(var_names) if name in exog_set]
    endog_idx = [i for i, name in enumerate(var_names) if name not in exog_set]

    if not exog_idx:
        return Y, None, list(var_names), []
    if not endog_idx:
        raise ValueError("外生变量不能覆盖全部变量")

    return Y[:, endog_idx], Y[:, exog_idx], [var_names[i] for i in endog_idx], [var_names[i] for i in exog_idx]
