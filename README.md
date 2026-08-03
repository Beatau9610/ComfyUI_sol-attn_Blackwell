# ComfyUI Sol-Attn (SM120 / RTX 5090) — MiniMax H3

Sol-Attn（Sparsified On-the-fly Attention）稀疏注意力加速，专门为 **NVIDIA RTX 5090 (SM120)** 上的 **MiniMax H3 视频生成**工作流实现。

本插件通过 PyTorch 的 `flex_attention`（`torch.compile` 后生成 FlashAttention-3 风格内核）执行 Sol-Attn 的稀疏路由，在长序列上相比标准 SDPA 有显著加速。

## 特性

- **SM120 原生优化**：flex_attention 编译内核，无串行循环，无 OOM
- **真·加速**：长序列（>8k tokens）实测超越 SDPA

| 序列长度 | SDPA | 本插件 | 加速 | 稀疏度 |
|---------|------|--------|------|--------|
| 4096  | 2.35ms | 1.61ms | 1.46x | 0.15 |
| 8192  | 9.24ms | 2.35ms | 3.93x | 0.11 |
| 16384 | 36.47ms | 5.28ms | 6.91x | 0.09 |
| 32768 | 144.5ms | 16.06ms | 9.00x | 0.08 |

（MiniMax H3 实际配置：H=56 头、bfloat16、BTHD 布局，预热后数据）

- **自动回退**：`flex_attention` → Triton 参考内核 → 标准 SDPA，失败不影响生成
- **零编译依赖**：为 SM120 设计的纯 Python + torch 方案，不需要 CUTLASS / CuTe DSL

## 环境要求

- NVIDIA RTX 5090 (SM120) 或其它 Blackwell consumer GPU
- **PyTorch ≥ 2.6**（flex_attention 完整性能需要；实测基于 2.11.0+cu130）
- CUDA 12.x / 13.x
- ComfyUI

## 安装

1. 将本目录放到 ComfyUI 的插件目录：

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/<你的用户名>/ComfyUI-SolAttn.git
```

或手动下载整个目录，放到 `ComfyUI/custom_nodes/` 下（目录名需为 `ComfyUI-SolAttn`）。

2. 重启 ComfyUI（无需额外 pip 安装，依赖已随 torch 内置）。

启动日志应出现：

```
[Sol-Attn] ✓ GPU: NVIDIA GeForce RTX 5090 (SM120) — Sol-Attn via flex_attention (Python) path
[SolAttnFlex] flex_attention kernel compiled (warmup done)
```

## 使用

在 MiniMax H3 工作流中加入节点 **Sol-Attn MiniMax H3 Patcher 🚀**：

```
[Load Diffusion Model] → [Sol-Attn MiniMaxH3 Patcher] → [KSampler / 采样节点]
```

节点参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `enabled` | true | 是否启用 |
| `tau` | 1.0 | 稀疏化温度，越大跳过越多 KV 块（更快但精度下降），推荐 0.8~1.5 |
| `thresh_type` | diag | 阈值估算方式，`diag` 更快，`exact` 更精确 |

## 工作原理

Sol-Attn 的核心思想：用每个 KV 块的均值向量 `kc` 做廉价路由，筛出"重要"的 KV 块，只对它们做精确 attention。

本插件把路由决策在 128-token 粒度上用纯 torch 计算，然后用 `flex_attention` 的编译内核执行被选中的块。相比原始 Sol-Attn 的 Triton 参考实现（串行处理每个 KV 块、慢 3~17 倍甚至 OOM），本方案在 SM120 上得到了规格化的 FlashAttention-3 性能。

## 文件结构

```
ComfyUI-SolAttn/
├── __init__.py          # 插件入口：inductor fix + flex 预热 + 节点注册
├── inductor_fix.py      # 修复 torch 2.11 的 torch.compile 崩溃（duplicate template）
├── sol_attn_flex.py     # SM120 主后端（flex_attention + Sol-Attn 路由）
├── minimax_h3_patch.py  # MiniMax H3 的 optimized_attention_override 钩子
├── sol_attn_node.py     # ComfyUI 节点
├── sol_attn_loader.py   # 加载器（含回退）
└── sol_attn/
    ├── __init__.py
    ├── interface.py
    ├── preprocess.py
    └── triton_ref/
        └── fwd.py       # 回退后端（Triton 参考实现）
```

## 已知限制

- 本版本针对 **SM120 (RTX 5090)** 和 **MiniMax H3**。H100 (SM90) / B200 (SM100) 的 CuTe DSL 后端不在本仓库中。
- 仅拦截 MiniMax H3 的 self-attention（`skip_reshape=True` + head_dim=128 + bfloat16）。跨注意力、其它 dtype 自动回退。
- 首次生成有一次 `torch.compile` 编译延迟（插件启动时已预热，约 3-4s）。

## 参考

- [Sol-Attn 论文](https://arxiv.org/abs/2501.17209)
- [NVLabs/Sana 源码](https://github.com/NVlabs/Sana)
- [PyTorch flex_attention 文档](https://pytorch.org/docs/stable/nn.attention.flex_attention.html)