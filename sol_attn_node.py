"""
ComfyUI节点实现 - Sol-Attn加速器
支持 H100 / B200 / RTX 5090
"""

import torch
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class SolAttnModelPatcher:
    """
    Sol-Attn模型补丁节点
    用于给LTX 2.3等DiT模型注入Sol-Attn加速
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "tau": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "thresh_type": (["diag", "full"], {
                    "default": "diag",
                    "tooltip": "阈值计算方式，diag更快"
                }),
                "kv_splits": (["1", "2", "4"], {
                    "default": "1",
                    "tooltip": "KV分割数，长序列时可提升性能"
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "model_patches/optimization"
    DESCRIPTION = "为DiT模型注入Sol-Attn加速 (支持RTX 5090/H100/B200)"

    def patch_model(self, model, enabled: bool, tau: float,
                    thresh_type: str, kv_splits: str):
        """
        给模型注入Sol-Attn

        Args:
            model: ComfyUI MODEL对象
            enabled: 是否启用Sol-Attn
            tau: 稀疏化温度参数
            thresh_type: 阈值类型
            kv_splits: KV分割数量

        Returns:
            (patched_model,): 修改后的模型
        """
        if not enabled:
            logger.info("Sol-Attn disabled, returning original model")
            return (model,)

        try:
            from ltx_patch import patch_ltx_attention

            # ComfyUI的MODEL是个wrapper，获取实际的PyTorch模型
            if hasattr(model, 'model'):
                pytorch_model = model.model
            elif hasattr(model, 'diffusion_model'):
                pytorch_model = model.diffusion_model
            else:
                pytorch_model = model

            # 应用patch
            patched_model, stats = patch_ltx_attention(
                pytorch_model,
                tau=tau,
                thresh_type=thresh_type,
                kv_splits=int(kv_splits),
                verbose=True
            )

            logger.info(
                f"✓ Sol-Attn patch applied: "
                f"{stats['patched_layers']}/{stats['total_layers']} layers patched"
            )

            # 包装回ComfyUI格式
            if hasattr(model, 'model'):
                model.model = patched_model
            elif hasattr(model, 'diffusion_model'):
                model.diffusion_model = patched_model

            return (model,)

        except Exception as e:
            logger.error(f"Failed to patch model with Sol-Attn: {e}")
            logger.warning("Returning original model without Sol-Attn")
            return (model,)


class SolAttnInfo:
    """
    Sol-Attn信息节点
    显示当前GPU的Sol-Attn支持状态
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "get_info"
    CATEGORY = "utils"
    OUTPUT_NODE = True
    DESCRIPTION = "显示Sol-Attn加速器状态信息"

    def get_info(self):
        """获取Sol-Attn状态信息"""
        try:
            from sol_attn_loader import get_sol_attn_loader

            loader = get_sol_attn_loader()
            gpu_info = loader.gpu_info

            if not gpu_info["available"]:
                info = "❌ CUDA不可用\n"
                info += f"原因: {gpu_info['reason']}"
                return (info,)

            info = "=== Sol-Attn加速器状态 ===\n\n"
            info += f"GPU: {gpu_info['name']}\n"
            info += f"架构: {gpu_info['arch_name']} (SM{gpu_info['sm']})\n"
            info += f"计算能力: {gpu_info['capability'][0]}.{gpu_info['capability'][1]}\n\n"

            if loader.is_available:
                info += "✅ Sol-Attn可用\n"
                info += "预期加速: 3-5x vs 标准attention\n"
                info += "支持模型: LTX 2.3, Wan 2.1, DiT类模型\n"
            else:
                info += "❌ Sol-Attn不可用\n"
                if loader.error_msg:
                    info += f"原因: {loader.error_msg}\n"
                info += "\n💡 将自动回退到PyTorch标准attention\n"

            return (info,)

        except Exception as e:
            return (f"❌ 获取信息失败: {e}",)


class SolAttnConfig:
    """
    Sol-Attn配置节点
    生成配置对象供其他节点使用
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tau": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "稀疏化温度，越大越稀疏"
                }),
                "thresh_type": (["diag", "full"], {
                    "default": "diag",
                    "tooltip": "阈值计算方式"
                }),
                "kv_splits": (["1", "2", "4"], {
                    "default": "1",
                    "tooltip": "KV分割数量（长序列优化）"
                }),
            },
        }

    RETURN_TYPES = ("SOL_ATTN_CONFIG",)
    FUNCTION = "create_config"
    CATEGORY = "model_patches/optimization"

    def create_config(self, tau: float, thresh_type: str, kv_splits: str):
        """创建Sol-Attn配置"""
        config = {
            "tau": tau,
            "thresh_type": thresh_type,
            "kv_splits": int(kv_splits),
        }
        return (config,)


class SolAttnMiniMaxH3Patcher:
    """
    Sol-Attn MiniMax H3 专用补丁节点
    为 MiniMax H3 视频生成 DiT 注入 Sol-Attn 加速（SM120 Triton 原生路径）

    原理：通过 transformer_options["optimized_attention_override"] 钩子
    将每次 optimized_attention 调用重定向到 Sol-Attn kernel。
    不修改模型权重，支持随时启用/禁用。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "tau": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": (
                        "稀疏化温度。越大 → 跳过更多 KV 块 → 更快但可能损失精度；"
                        "越小 → 更接近全量 attention。推荐 0.8~1.5。"
                    ),
                }),
                "thresh_type": (["diag", "exact"], {
                    "default": "diag",
                    "tooltip": (
                        "阈值估算方式。"
                        "diag: 用 KV 均值对角近似，速度更快，适合生产环境；"
                        "exact: 全协方差估算，阈值更精确但预处理略慢。"
                    ),
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/optimization"
    DESCRIPTION = (
        "为 MiniMax H3 DiT 注入 Sol-Attn 加速 (RTX 5090 SM120 原生 Triton 路径)。"
        "相比 Flash Attention，Sol-Attn 在长序列（>8k tokens）上可额外节省 30-50% 计算量。"
    )

    def patch(self, model, enabled: bool, tau: float, thresh_type: str):
        if not enabled:
            logger.info("[SolAttnMiniMaxH3Patcher] 已禁用，返回原始模型")
            return (model,)
        try:
            from minimax_h3_patch import patch_minimax_h3
            patched = patch_minimax_h3(model, tau=tau, thresh_type=thresh_type)
            logger.info(
                f"[SolAttnMiniMaxH3Patcher] ✓ Sol-Attn 注入成功 "
                f"| tau={tau} | thresh_type={thresh_type}"
            )
            return (patched,)
        except Exception as e:
            logger.error(f"[SolAttnMiniMaxH3Patcher] 注入失败: {e}，返回原始模型")
            return (model,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "SolAttnModelPatcher": SolAttnModelPatcher,
    "SolAttnMiniMaxH3Patcher": SolAttnMiniMaxH3Patcher,
    "SolAttnInfo": SolAttnInfo,
    "SolAttnConfig": SolAttnConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SolAttnModelPatcher": "Sol-Attn Model Patcher (LTX)",
    "SolAttnMiniMaxH3Patcher": "Sol-Attn MiniMax H3 Patcher 🚀",
    "SolAttnInfo": "Sol-Attn Info",
    "SolAttnConfig": "Sol-Attn Config",
}
