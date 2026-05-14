"""
配置加载器 — 从 config/ 目录合并 3 个配置文件 (深合并)
"""
import json
import os


def _deep_merge(base, override):
    """递归合并字典，override 中的标量覆盖 base，嵌套字典递归合并。"""
    merged = base.copy()
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(config_dir):
    """
    从目录加载 data.json, model.json, output.json 并深合并为一个配置字典。
    后加载的文件中嵌套字典会递归合并，而非整体覆盖。
    """
    cfg = {}
    for name in ("data.json", "model.json", "output.json"):
        path = os.path.join(config_dir, name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = _deep_merge(cfg, json.load(f))
    return cfg
