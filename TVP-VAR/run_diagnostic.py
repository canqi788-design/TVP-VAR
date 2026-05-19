import numpy as np
import logging
from tvp_var_framework.runtime.context import ExecutionContext
from tvp_var_framework.utils.stationarity import adf_test

# 配置日志，看到内部报错
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tvp_var.debug")

def run_diagnostic():
    # 1. 模拟数据 (5变量, 40个季度)
    T, N = 40, 5
    Y_mock = np.random.randn(T, N)
    var_names = [f"var_{i}" for i in range(N)]

    logger.info(f"--- 阶段 1: 数据输入 ---")
    logger.info(f"原始维度: {Y_mock.shape} (T={T}, N={N})")

    ctx = ExecutionContext(
        Y=Y_mock,
        var_names=var_names,
        config={
            "inference_control": {"n_iter": 10, "burnin": 2}, # 极小采样仅供测试
            "stationarity": {"test": "adf"}
        }
    )

    # 2. 模拟差分模块 (stationarity.py)
    # 风险点：差分后 T 会减 1，下游是否知晓？
    Y_diff = np.diff(Y_mock, axis=0)
    ctx.update("Y_diff", Y_diff)
    logger.info(f"差分后维度: {Y_diff.shape} (预期 T={T-1})")

    # 3. 模拟状态空间转换 (analyst.py 中的逻辑)
    # 风险点：k = n + n*n。当 n=5 时，k 应为 30
    n = N
    expected_k = n + n * n
    logger.info(f"--- 阶段 2: 状态维度校验 ---")
    logger.info(f"预期状态维度 k: {expected_k}")

    try:
        # 调用 estimators.py 中的 state_transition (如果已注册)
        # 这里模拟内部调用 logic
        from tvp_var_framework.models.analyst import TVP_VAR_Analyst
        analyst = TVP_VAR_Analyst(n_vars=n)

        # 测试单步构建 Z 矩阵
        y_prev = Y_diff[0]
        Z = analyst._build_Z(y_prev)
        logger.info(f"Z 矩阵维度: {Z.shape} (预期应为 {n}x{expected_k})")

        if Z.shape != (n, expected_k):
            logger.error(f"❌ 维度不匹配！Z 矩阵应该是 ({n}, {expected_k})，但实际是 {Z.shape}")
    except Exception as e:
        logger.error(f"❌ 状态转换模块崩溃: {e}")

    # 4. 模拟后验契约校验 (contracts.py)
    logger.info(f"--- 阶段 3: 契约一致性校验 ---")
    try:
        # 模拟采样器输出
        mock_joint_chain = [
            {
                "theta": np.random.randn(T-1, expected_k),
                "sigma": np.random.randn(n, n),
                "sample_index": 0
            }
        ]
        # 测试 update 到 context
        ctx.update("_joint_chain", mock_joint_chain)
        logger.info("✅ 契约校验通过: _joint_chain 已被正确识别")
    except KeyError as e:
        logger.error(f"❌ 契约违背: 键名未在 ALLOWED_OUTPUT_KEYS 中注册: {e}")

if __name__ == "__main__":
    run_diagnostic()
