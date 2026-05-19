# Posterior Dependency Graph (PDG)

## Generative Model (True DAG)

```
h_{t-1} ──→ h_t ──→ h_{t+1}       AR(1) log-volatility
  │           │           │
  ▼           ▼           ▼
 Σ_{t-1}    Σ_t     Σ_{t+1}       Σ_t = diag(exp(h_t)) ⊙ R_corr
  │           │           │
θ_{t-1} ──→ θ_t ──→ θ_{t+1}       Random walk: θ_t = θ_{t-1} + w_t, w_t ~ N(0, Q)
  │           │           │
  ▼           ▼           ▼
 Y_{t-1}    Y_t     Y_{t+1}       Y_t = Z_t θ_t + e_t, e_t ~ N(0, Σ_t)
```

## Sampler DAG (v4 — unified posterior)

```
kernel_sampler_research:
  ┌─────────────────────────────────────────────────────┐
  │  (θ | Σ, SV, Y)     FFBS forward-backward           │
  │       ↓                                             │
  │  (Σ | θ, SV, Y)     Inverse-Wishart                 │
  │       ↓                                             │
  │  (SV | θ, Σ, Y)     10-component mixture FFBS       │
  │       ↓                                             │
  │  Save coupled snapshot:                              │
  │    _joint_chain[k] = {theta, sigma, sv_state, ll}   │
  └─────────────────────────────────────────────────────┘
  Output: _joint_chain (list of coupled posterior states)

kernel_sampler_basic:
  ┌─────────────────────────────────────────────────────┐
  │  Dispatch to estimation backends:                    │
  │    fully_bayesian, bayesian, v2, research            │
  │       ↓                                             │
  │  Restructure chains into _joint_chain format         │
  │  (post-hoc coupling for backward compatibility)      │
  └─────────────────────────────────────────────────────┘
  Output: _joint_chain (list of restructured posterior states)
```

## Execution DAG (v4)

```
data → stationarity → state_update → likelihood → sampling_basic    → diagnostics → reporting
                                       |                                        ^
                                       +→ sampling_research ────────────────────-+
```

Both samplers share the same upstream pipeline. Both output `_joint_chain`.
DAG nodes determine kernels directly — no mode-driven routing.

## Output DAG (v4 — coupled posterior)

```
_joint_chain[k] = {
    "theta": θ_k,          # coupled
    "sigma": Σ_k,          # coupled (same Gibbs iteration)
    "sv_state": sv_k,      # coupled
    "log_likelihood": ll_k,
    "sample_index": k,
}
         │
         ├──→ IRF(s):  A^{(s)} = θ^{(s)}[n:].reshape(n,n), Σ^{(s)} = sample["sigma"]
         │     Ψ^{(s)}(h) = A^{(s)} · Ψ^{(s)}(h-1)
         │     IRF^{(s)} = Ψ^{(s)} · chol(Σ^{(s)})
         │
         ├──→ FEVD(s): same coupled (A, Σ) per sample
         │
         ├──→ Forecast(s): θ^{(s)}, Σ^{(s)} from same iteration
         │
         └──→ Report: marginalize over s → mean, 2.5%, 97.5%
```

## Consistency Constraints — RESOLVED

### C1: θ_t ↔ Σ_t 时间一致性 ✅ RESOLVED

```
约束: θ_t 和 Σ_t 必须来自同一个后验样本 t
方案: _joint_chain[k] 存储耦合的 (theta, sigma) 快照
状态: ✅ 已解决 — kernel_sampler_research 保证采样时耦合
```

### C2: A ↔ Σ 后验相关性 ✅ RESOLVED

```
约束: IRF 中 A_i 和 Σ_i 必须来自同一样本 i
方案: StructuralAnalysis.from_joint_chain() 提取耦合 (A, Σ)
状态: ✅ 已解决 — 逐样本耦合
```

### C3: IRF 递推中的参数一致性 ⚠️ PARTIALLY RESOLVED

```
约束: Ψ(h) = A_{t+h} · Ψ(h-1), A 应随 h 变化
方案: _joint_chain 存储完整 θ_T (含时变结构)
状态: ⚠️ 部分解决 — 当前 IRF 仍用标量 A 递推
      未来: 从 _joint_chain 中提取时变 A 轨迹
```

### C4: FEVD 中方差分解一致性 ✅ RESOLVED

```
约束: FEVD 应使用与 IRF 相同的 Σ_t
方案: from_joint_chain() 确保 FEVD 和 IRF 使用相同的耦合 (A, Σ)
状态: ✅ 已解决 — 统一后验源
```

### C5: BQ likelihood 一致性 ✅ RESOLVED

```
约束: BQ 旋转是逐样本确定性变换
方案: 从 _joint_chain 逐样本应用 BQ 旋转
状态: ✅ 已解决 — 输入 (A, Σ) 来自同一样本
```

## Summary Table (v4)

| 约束 | kernel_sampler_research | kernel_sampler_basic | 状态 |
|------|------------------------|---------------------|------|
| C1: θ↔Σ 同一后验 | ✅ 采样时耦合 | ✅ 后重构耦合 | ✅ |
| C2: A↔Σ 同一样本 | ✅ | ✅ | ✅ |
| C3: IRF 时变参数 | ⚠️ 标量 A | ⚠️ 标量 A | ⚠️ |
| C4: FEVD Σ 一致 | ✅ | ✅ | ✅ |
| C5: BQ likelihood | ✅ | ✅ | ✅ |

## Design Rules

```
1. One Gibbs iteration = one coherent posterior world-state
2. _joint_chain is the ONLY posterior output format
3. theta_chain / sigma_chain are DEPRECATED
4. Decoupled posterior means are REJECTED by downstream analysis
5. PosteriorSample = {theta, sigma, sv_state, log_likelihood, sample_index}
6. All samples use np.copy() — mutable references forbidden
```

## Legacy Deprecation

```
DEPRECATED:
  - PathwiseMCPredictor.predict(theta_chain, Q_chain, R_chain, ...)
  - PathwiseMCPredictor.predict_with_regime(theta_chain, Q_chain, R_chain, ...)
  - theta_chain / sigma_chain as separate posterior stores

USE INSTEAD:
  - PathwiseMCPredictor.predict_from_joint_chain(joint_chain, Y_last, ...)
  - StructuralAnalysis.orthogonalized_irf_from_joint_chain(joint_chain, ...)
  - FullyBayesianTVPVAR.impulse_response_from_joint_chain(joint_chain, ...)
  - FullyBayesianTVPVAR.predict_from_joint_chain(joint_chain, ...)
```
