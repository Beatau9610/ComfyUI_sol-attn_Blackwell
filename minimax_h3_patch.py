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
"""

from __future__ import annotations

import logging
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


def make_minimax_h3_override(tau: float = 1.0, thresh_type: str = "diag"):
    """
    创建 optimized_attention_override 函数，注入 Sol-Attn。

    Args:
        tau: 稀疏化温度，越大越稀疏，越小越接近全量 attention (default: 1.0)
        thresh_type: 阈值计算方式 "diag"(快) 或 "exact"(精确) (default: "diag")

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

        try:
            B, H, S, D = q.shape  # [1, 56, SeqLen, 128]

            # [B, H, S, D] → [B, S, H, D]  (BTHD，Sol-Attn 的标准格式)
            q_sol = q.permute(0, 2, 1, 3).contiguous()
            k_sol = k.permute(0, 2, 1, 3).contiguous()
            v_sol = v.permute(0, 2, 1, 3).contiguous()

            # Sol-Attn 执行；scale 由 sol_attn 内部计算 (1/sqrt(128))
            # 回退链：sol_attn_flex → triton_ref → SDPA
            try:
                out_sol = sol_fn(q_sol, k_sol, v_sol, tau=tau, thresh_type=thresh_type)
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


def patch_minimax_h3(model, tau: float = 1.0, thresh_type: str = "diag"):
    """
    给 ComfyUI MODEL 对象注入 Sol-Attn，专为 MiniMax H3 设计。

    不修改模型权重，只在 transformer_options 中注入 attention override。
    对原始 model 无副作用（返回 clone）。

    Args:
        model: ComfyUI ModelPatcher 对象
        tau: 稀疏化温度 (default: 1.0)
        thresh_type: "diag" 或 "exact" (default: "diag")

    Returns:
        patched: 注入了 Sol-Attn 的 ModelPatcher 副本
    """
    patched = model.clone()
    # 复制 transformer_options，不污染原始对象
    to = patched.model_options.get("transformer_options", {}).copy()
    to["optimized_attention_override"] = make_minimax_h3_override(
        tau=tau, thresh_type=thresh_type
    )
    patched.model_options["transformer_options"] = to
    logger.info(
        f"[MiniMaxH3-SolAttn] 注入完成 | tau={tau} | thresh_type={thresh_type}"
    )
    return patched
