"""
Sol-Attn x MiniMax H3 DiT 集成补丁
通过 ComfyUI 的 transformer_options["optimized_attention_override"] 钩子
将 Minimax H3 的 self-attention 替换为 Sol-Attn (SM120 Triton 路径)。

MiniMax H3 attention 数据流：
  Attention.forward() 内部:
    q/k/v: [S, H, D]  →  transpose + unsqueeze  →  [1, H, S, D]
    调用 optimized_attention(q, k, v, heads, ..., skip_reshape=True, ...)
    返回 [1, S, H*D]  →  squeeze(0)  →  [S, H*D]  →  out_proj

Sol-Attn 要求:
    - BTHD 格式: [B, T, H, D] = [1, S, H, D]
    - dtype: bfloat16
    - head_dim == 128
    - 连续张量

Kijai 的两条建议 (2026-08):
  1. 最后一步（或最后几步）不做稀疏化：采样末尾的噪声最弱，最后一步的
     稀疏误差最容易被看见。用 dense_steps 让前 N 个 forward 走全量密
     attention（回到原始 xformers），并可选 step_off 按调度位置强制最后
     一步也 dense。
  2. MiniMax-H3 only 的 exact_kv / exact_kv_and_rows：
     - exact_kv: 打包序列里 text/cond/参考/audio 这些"前缀"行的 KV 块对
       每个 query 都精确保留（+3% 成本）。
     - exact_kv_and_rows: 额外把这些前缀 query 行也跑成 dense，使"生成的
       audio 流"（位于前缀末尾）完全精确（+20% 成本）。
     对其它模型无影响。
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)

# Minimax H3 固定 head_dim
_H3_HEAD_DIM = 128

# 模块级 sol_attn 函数缓存
_sol_attn_fn = None
_sol_attn_fallback_fn = None
_sol_attn_load_attempted = False


def _load_sol_attn():
    """延迟加载 sol_attn，避免 import 阶段就触发 kernel 编译。"""
    global _sol_attn_fn, _sol_attn_load_attempted
    if _sol_attn_load_attempted:
        return _sol_attn_fn
    _sol_attn_load_attempted = True
    try:
        import sys, os
        # 确保 sol_attn 包路径在 sys.path 中
        _pkg_dir = os.path.dirname(__file__)
        if _pkg_dir not in sys.path:
            sys.path.insert(0, _pkg_dir)
        # SM120: 优先使用 flex_attention 后端（compiled FlashAttn3），
        # 失败时回退到 triton_ref，再回退到 SDPA。
        from sol_attn_flex import sol_attn_flex as _flex
        from sol_attn.interface import sol_attn as _triton
        _sol_attn_fn = _flex
        _sol_attn_fallback_fn = _triton
        logger.info("[MiniMaxH3-SolAttn] sol_attn_flex 加载成功")
    except Exception as e:
        try:
            from sol_attn.interface import sol_attn
            _sol_attn_fn = sol_attn
            _sol_attn_fallback_fn = None
            logger.info("[MiniMaxH3-SolAttn] sol_attn (triton_ref) 加载成功")
        except Exception as e2:
            logger.error(f"[MiniMaxH3-SolAttn] 无法加载 sol_attn: {e2}")
    return _sol_attn_fn


def _trailing_dense_steps(total: int, dense_steps: int, step_off: float) -> int:
    """计算采样末尾需要走 dense 的步数。

    - dense_steps: 直接指定末尾 N 步（Kijai 建议 1 的「最后一步或几步」）。
    - step_off: 用调度序号比例指定，0..1，表示末尾 fraction 比例的步走 dense
      （0.5 = 末尾一半）。
    两者取较大者。
    """
    n = int(dense_steps)
    if step_off > 0.0 and total > 1:
        n = max(n, int(math.ceil(total * min(step_off, 1.0))))
    return max(0, n)


def _make_forward_wrapper(
    dense_steps: int,
    step_off: float,
    exact_mode: str,
    sink_tokens: int,
):
    """构建 DIFFUSION_MODEL wrapper，把每步状态写进 transformer_options。

    MiniMaxH3Model.forward (comfy/ldm/minimax/model.py:498) 会跑这个 wrapper
    执行器。我们用它：
      - 维护「本采样 run 内已经 forward 了几次」的步计数器；
      - 从 minimax_payload["layout"] 里取出 video 段起点 = sink_tokens，
        并写进 transformer_options，供 override 读取；
      - 决定本步是否走 dense（dense_steps 前 N 步 或 step_off 末尾步）。

    wrapper 签名与 MyDiffusionModel 的 _forward 一致：
      (executor, x, timestep, context, transformer_options, minimax_payload, **kwargs)
    """
    step_counter = {"n": 0}

    def wrapper(
        executor,
        x,
        timestep,
        context,
        transformer_options={},
        minimax_payload=None,
        **kwargs,
    ):
        to = transformer_options

        # 需要知道总步数；transformer_options["sample_sigmas"] 是完整调度
        # （长度 = 步数 + 1，末尾补 0）。采样 loop 跑 total-1 步。
        total = len(to.get("sample_sigmas", []))
        n_steps = max(1, total - 1)

        # 关键：DIFFUSION_MODEL wrapper 是在节点首次执行时创建一次，之后被
        # ComfyUI 的节点输出缓存复用（输入不变时不会重新执行节点），所以闭包
        # 里的 step_counter 会跨多个采样 run 一直累加、从不归零。若不重置，
        # 第二次及以后的 run 里 step 会一直落在「末尾 dense」窗口（step >
        # n_steps - need）内，导致每一次 forward 都走全量 dense——速度退回
        # 没开 Sol-Attn 的水平，画质也与原始一致。这里按调度长度把计数器归零，
        # 让每个 run 独立从 1 计数。
        if step_counter["n"] >= n_steps:
            step_counter["n"] = 0
        step_counter["n"] += 1
        step = step_counter["n"]

        # 从 layout 里取 sink_tokens（video 段起点 = 前缀 = text+cond+ref+audio）。
        # t2va 布局: [text | audio | video]；ref2va: [text | refs | audio | video]。
        s_tokens = int(sink_tokens)
        layout = None
        if minimax_payload is not None:
            layout = minimax_payload.get("layout", None)
        if layout is not None and hasattr(layout, "seq_len"):
            if s_tokens <= 0:
                # 自动推导：video 段起点
                for a, b, kind in layout.segments:
                    if kind == "video":
                        s_tokens = int(a)
                        break
        to["_solattn_sink_tokens"] = s_tokens

        # 本步是否走 dense（末尾段）
        force_dense = False
        need = _trailing_dense_steps(n_steps, dense_steps, step_off)
        if need > 0:
            # 末尾 need 步 dense：step 从 n_steps-need+1 到 n_steps
            if step > n_steps - need:
                force_dense = True
        to["_solattn_dense"] = bool(force_dense)

        return executor.original(
            x,
            timestep,
            context,
            transformer_options=transformer_options,
            minimax_payload=minimax_payload,
            **kwargs,
        )

    return wrapper


def make_minimax_h3_override(
    tau: float = 1.0,
    thresh_type: str = "diag",
    dense_steps: int = 0,
    step_off: float = 0.0,
    exact_mode: str = "off",
    sink_tokens: int = 0,
):
    """
    创建 optimized_attention_override 函数，注入 Sol-Attn。

    Args:
        tau: 稀疏化温度，越大越稀疏，越小越接近全量 attention (default: 1.0)
        thresh_type: 阈值计算方式 "diag"(快) 或 "exact"(精确) (default: "diag")
        dense_steps: 前 N 个 forward 步强制走全量 dense（不做稀疏）(default: 0)
        step_off: 按 sigma 阈值把采样末尾的步骤强制 dense（"最后一步"改进）
            (default: 0.0)
        exact_mode: "off" / "exact_kv" / "exact_kv_and_rows" (default: "off")
            - exact_kv: 前缀（text/cond/ref/audio）KV 行对每个 query 精确
            - exact_kv_and_rows: 前缀 KV 精确 + 前缀 query 行也 dense
        sink_tokens: 精确前缀 token 数；<=0 且从 layout 可推导时自动取
            video 段起点 (default: 0)

    Returns:
        override 函数，传给 transformer_options["optimized_attention_override"]
    """

    def override(func, q, k, v, heads,
                 mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False,
                 **kwargs):
        """
        拦截 attention_basic 调用，将满足条件的调用替换为 Sol-Attn。

        满足以下全部条件才走 Sol-Attn：
          1. skip_reshape=True  (Minimax H3 才会设置)
          2. q.ndim == 4，即已是 [B, H, S, D] 格式
          3. q.shape[-1] == 128  (Sol-Attn 固定 head_dim=128)
          4. dtype == bfloat16
          5. sol_attn 函数已成功加载
        否则回退到原始 attention。
        """
        sol_fn = _load_sol_attn()

        # 前提条件检查
        if (
            sol_fn is None
            or not skip_reshape
            or q.ndim != 4
            or q.shape[-1] != _H3_HEAD_DIM
            or q.dtype != torch.bfloat16
        ):
            return func(q, k, v, heads,
                        mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape,
                        **kwargs)

        to = kwargs.get("transformer_options", None) or {}

        # 本步强制 dense（Kijai 建议 1）：直接走原始 attention，跳过 Sol-Attn。
        if to.get("_solattn_dense", False):
            return func(q, k, v, heads,
                        mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape,
                        **kwargs)

        try:
            B, H, S, D = q.shape  # [1, 56, SeqLen, 128]

            # 从 wrapper 里读 sink_tokens（前缀）。只对打包序列生效：
            # 打包序列 T ≈ 3.1e7，而 token_refiner 的 T=text_len（≤~512）。
            sink_tok = int(to.get("_solattn_sink_tokens", 0) or 0)
            if sink_tok > 0 and S <= sink_tok:
                # 这不是打包序列（比如 token_refiner 的 T=text_len），
                # 前缀精确只对打包序列有意义，这里保持纯稀疏。
                sink_tok = 0

            dense_sink = bool(exact_mode == "exact_kv_and_rows")
            if exact_mode == "off":
                sink_tok = 0
                dense_sink = False

            # [B, H, S, D] → [B, S, H, D]  (BTHD，Sol-Attn 的标准格式)
            q_sol = q.permute(0, 2, 1, 3).contiguous()
            k_sol = k.permute(0, 2, 1, 3).contiguous()
            v_sol = v.permute(0, 2, 1, 3).contiguous()

            # Sol-Attn 执行；scale 由 sol_attn 内部计算 (1/sqrt(128))
            # 回退链：sol_attn_flex → triton_ref → SDPA
            try:
                out_sol = sol_fn(
                    q_sol, k_sol, v_sol, tau=tau, thresh_type=thresh_type,
                    sink_tokens=sink_tok, dense_sink_rows=dense_sink,
                )
            except Exception as _fe:
                logger.warning(
                    f"[MiniMaxH3-SolAttn] flex 后端失败 ({_fe.__class__.__name__}: {_fe})，"
                    f"回退到 triton_ref"
                )
                if _sol_attn_fallback_fn is not None:
                    out_sol = _sol_attn_fallback_fn(
                        q_sol, k_sol, v_sol, tau=tau, thresh_type=thresh_type,
                    )
                else:
                    raise
            # out_sol: [B, S, H, D]

            # 合并 head 维度：[B, S, H, D] → [B, S, H*D]
            # 与 attention_basic(skip_reshape=True) 的返回值格式一致
            return out_sol.reshape(B, S, H * D)

        except Exception as exc:
            logger.warning(
                f"[MiniMaxH3-SolAttn] kernel 执行失败 ({exc.__class__.__name__}: {exc})，"
                f"回退到标准 attention"
            )
            return func(q, k, v, heads,
                        mask=mask, attn_precision=attn_precision,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape,
                        **kwargs)

    return override


def patch_minimax_h3(
    model,
    tau: float = 1.0,
    thresh_type: str = "diag",
    dense_steps: int = 0,
    step_off: float = 0.0,
    exact_mode: str = "off",
    sink_tokens: int = 0,
):
    """
    给 ComfyUI MODEL 对象注入 Sol-Attn，专为 MiniMax H3 设计。

    不修改模型权重，只在 transformer_options 中注入 attention override，
    并注册一个 DIFFUSION_MODEL wrapper 来维护每步状态（dense/稀疏、sink）。
    对原始 model 无副作用（返回 clone）。

    Args:
        model: ComfyUI ModelPatcher 对象
        tau: 稀疏化温度 (default: 1.0)
        thresh_type: "diag" 或 "exact" (default: "diag")
        dense_steps: 前 N 步强制 dense (default: 0)
        step_off: 采样末尾按 sigma 阈值强制 dense (default: 0.0)
        exact_mode: "off"/"exact_kv"/"exact_kv_and_rows" (default: "off")
        sink_tokens: 精确前缀 token 数；<=0 自动推导 (default: 0)

    Returns:
        patched: 注入了 Sol-Attn 的 ModelPatcher 副本
    """
    patched = model.clone()
    # 复制 transformer_options，不污染原始对象
    to = patched.model_options.get("transformer_options", {}).copy()
    to["optimized_attention_override"] = make_minimax_h3_override(
        tau=tau, thresh_type=thresh_type,
        dense_steps=dense_steps, step_off=step_off,
        exact_mode=exact_mode, sink_tokens=sink_tokens,
    )
    patched.model_options["transformer_options"] = to

    # 注册 DIFFUSION_MODEL wrapper，维护每步状态。
    # 它会被 prepare_model_patcher 并入 transformer_options["wrappers"]，
    # 再由 MiniMaxH3Model.forward 的 wrapper 执行器调用。
    if dense_steps > 0 or step_off > 0.0 or exact_mode != "off":
        wrapper = _make_forward_wrapper(
            int(dense_steps), float(step_off), exact_mode, int(sink_tokens),
        )
        try:
            patched.add_wrapper_with_key(
                "diffusion_model", "sol_attn_h3_state", wrapper,
            )
        except Exception as e:
            logger.warning(
                f"[MiniMaxH3-SolAttn] 无法注册 DIFFUSION_MODEL wrapper: {e}；"
                f"dense/exact 边界将不可用"
            )

    logger.info(
        f"[MiniMaxH3-SolAttn] 注入完成 | tau={tau} | thresh_type={thresh_type} | "
        f"dense_steps={dense_steps} | step_off={step_off} | "
        f"exact_mode={exact_mode} | sink_tokens={sink_tokens}"
    )
    return patched