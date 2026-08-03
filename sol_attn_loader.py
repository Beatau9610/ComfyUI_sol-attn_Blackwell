"""
Sol-Attn 加载器 - RTX 5090 / H100 / B200 支持
提供智能fallback和架构自适应编译
包含SM120→SM100 workaround
"""

import torch
import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# 导入SM120 workaround
try:
    from sm120_workaround import apply_sm120_workaround, remove_sm120_workaround
    _has_workaround = True
except ImportError:
    _has_workaround = False
    logger.warning("SM120 workaround not available")


class SolAttnLoader:
    """Sol-Attn动态加载器，支持多架构和fallback"""

    def __init__(self):
        self.sol_attn_func = None
        self.is_available = False
        self.gpu_info = self._detect_gpu()
        self.error_msg = None

    def _detect_gpu(self) -> dict:
        """检测GPU架构信息"""
        if not torch.cuda.is_available():
            return {"available": False, "reason": "CUDA not available"}

        device = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device)
        sm = capability[0] * 10 + capability[1]
        gpu_name = torch.cuda.get_device_name(device)

        return {
            "available": True,
            "name": gpu_name,
            "sm": sm,
            "capability": capability,
            "arch_name": self._get_arch_name(sm),
            "supported": sm in [90, 100, 120]  # H100, B200, RTX 5090
        }

    def _get_arch_name(self, sm: int) -> str:
        """获取架构名称"""
        arch_map = {
            90: "Hopper (H100)",
            100: "Blackwell Datacenter (B200)",
            120: "Blackwell Consumer (RTX 5090)",
        }
        return arch_map.get(sm, f"SM{sm}")

    def _setup_environment(self):
        """设置编译环境变量"""
        sm = self.gpu_info["sm"]

        if sm == 90:
            os.environ['CUTE_DSL_ARCH'] = "sm_90a"
            os.environ['TORCH_CUDA_ARCH_LIST'] = "9.0"
        elif sm == 100:
            os.environ['CUTE_DSL_ARCH'] = "sm_100a"
            os.environ['TORCH_CUDA_ARCH_LIST'] = "10.0"
        elif sm == 120:
            # RTX 5090: 使用 flex_attention 后端（不需要 CuTe DSL / CUTLASS 环境变量）
            # 保留 TORCH_CUDA_ARCH_LIST 以便 Triton 正确编译 SM120 目标。
            os.environ['TORCH_CUDA_ARCH_LIST'] = "12.0;12.0+PTX"
        else:
            logger.warning(f"Unsupported SM{sm}, attempting PTX fallback")
            os.environ['CUTE_DSL_ARCH'] = f"sm_{sm}a"
            cap = self.gpu_info["capability"]
            os.environ['TORCH_CUDA_ARCH_LIST'] = f"{cap[0]}.{cap[1]}+PTX"

    def load(self) -> Tuple[bool, str]:
        """
        加载Sol-Attn kernel

        Returns:
            (success: bool, message: str)
        """
        # 检查GPU支持
        if not self.gpu_info["available"]:
            return False, "CUDA not available"

        if not self.gpu_info["supported"]:
            return False, f"GPU {self.gpu_info['name']} (SM{self.gpu_info['sm']}) not supported by Sol-Attn"

        # SM120 (RTX 5090) uses the native Triton path in sol_attn/triton_ref/fwd.py.
        # Do NOT apply the SM120→SM100 workaround here: that workaround patches
        # Triton's driver to report SM100, which causes Triton to compile SM100
        # ISA binaries that cannot execute on SM120 hardware.
        # Triton compiles JIT for the actual device, so no masquerade is needed.

        # 设置环境
        self._setup_environment()

        # 尝试导入Sol-Attn
        try:
            # 添加sol_attn到Python路径
            sol_attn_path = Path(__file__).parent / "sol_attn"
            if sol_attn_path.exists() and str(sol_attn_path.parent) not in sys.path:
                sys.path.insert(0, str(sol_attn_path.parent))

            from sol_attn.interface import sol_attn
            self.sol_attn_func = sol_attn
            self.is_available = True

            arch_name = self.gpu_info["arch_name"]
            sm = self.gpu_info["sm"]
            if sm == 120:
                return True, f"✓ Sol-Attn loaded for {arch_name} (flex_attention backend)"
            else:
                return True, f"✓ Sol-Attn loaded successfully for {arch_name}"

        except ImportError as e:
            self.error_msg = f"Failed to import Sol-Attn: {e}"
            logger.error(self.error_msg)
            return False, self.error_msg

        except Exception as e:
            self.error_msg = f"Failed to load Sol-Attn: {e}"
            logger.error(self.error_msg)
            return False, self.error_msg

    def __call__(self, q, k, v, tau=1.0, thresh_type='diag', kv_splits=1,
                 sink_tokens=0, scale=None, **kwargs):
        """
        执行attention，带自动fallback

        Args:
            q, k, v: Query, Key, Value tensors
                     Shape: (batch, seqlen, nheads, headdim)
            tau: Temperature parameter for sparsification (default: 1.0)
            thresh_type: Threshold type - 'diag' or other (default: 'diag')
            kv_splits: KV分割数量，用于长序列优化 (default: 1)
            sink_tokens: 保留的sink token数量 (default: 0)
            scale: 缩放因子，None时自动计算 (default: None)
            **kwargs: 其他参数传递给sol_attn

        Returns:
            output: Attention output, same shape as q
        """
        if not self.is_available or self.sol_attn_func is None:
            # Fallback to PyTorch SDPA
            logger.warning(
                f"Sol-Attn not available ({self.error_msg}), "
                f"falling back to torch.nn.functional.scaled_dot_product_attention"
            )
            return self._fallback_attention(q, k, v)

        try:
            # 调用Sol-Attn kernel
            return self.sol_attn_func(
                q, k, v,
                scale=scale,
                tau=tau,
                thresh_type=thresh_type,
                kv_splits=kv_splits,
                sink_tokens=sink_tokens,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Sol-Attn execution failed: {e}, falling back to standard attention")
            return self._fallback_attention(q, k, v)

    def _fallback_attention(self, q, k, v):
        """
        Fallback到PyTorch标准attention

        Sol-Attn expects: (batch, seqlen, nheads, headdim)
        PyTorch SDPA expects: (batch, nheads, seqlen, headdim)
        """
        # 转换维度
        q_sdpa = q.transpose(1, 2)  # (B, H, S, D)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)

        # 使用PyTorch SDPA
        out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa
        )

        # 转回Sol-Attn格式
        return out.transpose(1, 2)  # (B, S, H, D)


# 全局实例
_global_loader = None

def get_sol_attn_loader() -> SolAttnLoader:
    """获取全局Sol-Attn加载器实例"""
    global _global_loader
    if _global_loader is None:
        _global_loader = SolAttnLoader()
        success, msg = _global_loader.load()
        logger.info(msg)
    return _global_loader


def sol_attn_forward(q, k, v, tau=1.0, thresh_type='diag', kv_splits=1,
                     sink_tokens=0, scale=None, **kwargs):
    """
    便捷函数：执行Sol-Attn前向传播

    自动处理加载、fallback和错误
    """
    loader = get_sol_attn_loader()
    return loader(q, k, v, tau=tau, thresh_type=thresh_type,
                  kv_splits=kv_splits, sink_tokens=sink_tokens, scale=scale, **kwargs)
