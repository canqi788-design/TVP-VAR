"""
TVP-VAR 回测引擎

滚动窗口 / 扩展窗口交叉验证
计算 RMSE, MAE, MAPE 等样本外预测指标
"""

import numpy as np
import logging

logger = logging.getLogger("tvp_var")


def compute_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted
    abs_errors = np.abs(errors)

    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_errors)
    me = np.mean(errors)

    nonzero = np.abs(actual) > 1e-10
    if np.any(nonzero):
        mape = np.mean(abs_errors[nonzero] / np.abs(actual[nonzero])) * 100
    else:
        mape = np.nan

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "mse": float(mse),
        "me": float(me),
    }


class BacktestResult:
    def __init__(self, predictions, actuals, metrics_per_step, overall_metrics,
                 fold_indices, horizon):
        self.predictions = predictions
        self.actuals = actuals
        self.metrics_per_step = metrics_per_step
        self.overall_metrics = overall_metrics
        self.fold_indices = fold_indices
        self.horizon = horizon
        self.n_folds = len(fold_indices)

    def summary(self, var_names=None):
        n = self.predictions[0].shape[1] if self.predictions else 0
        if var_names is None:
            var_names = [f"x{i}" for i in range(n)]

        lines = []
        lines.append("回测结果摘要")
        lines.append("=" * 60)
        lines.append(f"窗口数: {self.n_folds}")
        lines.append(f"预测步数: {self.horizon}")
        lines.append("")

        lines.append(f"{'步':<6} {'RMSE':>10} {'MAE':>10} {'MAPE(%)':>10} {'ME':>10}")
        lines.append("-" * 46)
        for s, m in enumerate(self.metrics_per_step):
            lines.append(f"t+{s+1:<3} {m['rmse']:>10.4f} {m['mae']:>10.4f} {m['mape']:>10.2f} {m['me']:>10.4f}")

        lines.append("-" * 46)
        m = self.overall_metrics
        lines.append(f"{'总体':<6} {m['rmse']:>10.4f} {m['mae']:>10.4f} {m['mape']:>10.2f} {m['me']:>10.4f}")

        lines.append("")
        lines.append(f"{'变量':<12} {'RMSE':>10} {'MAE':>10}")
        lines.append("-" * 34)
        for j, name in enumerate(var_names):
            all_actual = np.concatenate([a[:, j] for a in self.actuals])
            all_pred = np.concatenate([p[:, j] for p in self.predictions])
            m_j = compute_metrics(all_actual, all_pred)
            lines.append(f"{name:<12} {m_j['rmse']:>10.4f} {m_j['mae']:>10.4f}")

        return "\n".join(lines)

    def to_dict(self):
        return {
            "n_folds": self.n_folds,
            "horizon": self.horizon,
            "metrics_per_step": self.metrics_per_step,
            "overall_metrics": self.overall_metrics,
        }


class BacktestingEngine:
    def __init__(self, initial_window=None, horizon=1, step=1, mode='expanding'):
        self.initial_window = initial_window
        self.horizon = horizon
        self.step = step
        self.mode = mode

    def run(self, model_class, Y, model_kwargs=None, verbose=True):
        Y = np.asarray(Y, dtype=float)
        T, n = Y.shape
        if model_kwargs is None:
            model_kwargs = {}

        initial = self.initial_window or T // 2
        initial = max(initial, n + 2)

        predictions = []
        actuals = []
        fold_indices = []

        folds = []
        train_end = initial
        while train_end + self.horizon <= T:
            test_start = train_end
            test_end = train_end + self.horizon
            folds.append((train_end, test_start, test_end))
            train_end += self.step

        if not folds:
            logger.warning(f"回测: 数据太短 (T={T}), 无法生成任何折叠")
            return BacktestResult([], [], [], {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "me": np.nan}, [], self.horizon)

        for i, (train_end, test_start, test_end) in enumerate(folds):
            if self.mode == 'expanding':
                Y_train = Y[:train_end]
            else:
                Y_train = Y[max(0, train_end - initial):train_end]

            Y_test = Y[test_start:test_end]

            if verbose:
                logger.info(f"回测 [{i+1}/{len(folds)}]: 训练={len(Y_train)}, 测试={len(Y_test)}")

            try:
                model = model_class(**model_kwargs)
                model.fit(Y_train)
                means, lowers, uppers = model.predict_interval(
                    Y_train, steps=self.horizon, n_samples=200
                )
                predictions.append(means[:len(Y_test)])
                actuals.append(Y_test[:len(means)])
                fold_indices.append((train_end, test_start))
            except Exception as e:
                logger.warning(f"回测折叠 {i+1} 失败: {e}")
                continue

        if not predictions:
            logger.warning("回测: 所有折叠均失败")
            return BacktestResult([], [], [], {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "me": np.nan}, [], self.horizon)

        metrics_per_step = []
        for s in range(self.horizon):
            step_actual = []
            step_pred = []
            for fold_idx in range(len(predictions)):
                if s < predictions[fold_idx].shape[0]:
                    step_actual.append(actuals[fold_idx][s])
                    step_pred.append(predictions[fold_idx][s])
            if step_actual:
                metrics_per_step.append(compute_metrics(
                    np.array(step_actual), np.array(step_pred)
                ))
            else:
                metrics_per_step.append({"rmse": np.nan, "mae": np.nan, "mape": np.nan, "mse": np.nan, "me": np.nan})

        all_actual = np.concatenate(actuals)
        all_pred = np.concatenate(predictions)
        overall_metrics = compute_metrics(all_actual, all_pred)

        result = BacktestResult(
            predictions=predictions,
            actuals=actuals,
            metrics_per_step=metrics_per_step,
            overall_metrics=overall_metrics,
            fold_indices=fold_indices,
            horizon=self.horizon,
        )

        if verbose:
            logger.info(f"回测完成: {result.n_folds} 折叠, RMSE={overall_metrics['rmse']:.4f}")

        return result
