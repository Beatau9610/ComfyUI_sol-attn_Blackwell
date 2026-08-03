"""
ComfyUI Sol-Attn Node
NVIDIA Sol-Attn (Sparsified On-the-fly Attention) for video DiT acceleration.

Supported GPUs:
  SM120 (RTX 5090)  — native Triton JIT path (triton_ref/fwd.py)
  SM90  (H100)      — CuTe DSL warp-level MMA kernel
  SM100 (B200)      — CuTe DSL tcgen05 kernel

Nodes:
  Sol-Attn MiniMax H3 Patcher  — for MiniMax H3 video generation workflows
  Sol-Attn Model Patcher (LTX) — for LTX 2.3 / Wan 2.1 / generic DiTs
  Sol-Attn Info                — GPU status display
  Sol-Attn Config              — shared config object
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[Sol-Attn] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inductor fix: this torch build ships obsolete duplicate top-level kernel
# files (flex_attention.py / flex_decoding.py / mm_scaled_grouped.py) next to
# the new flex/ subpackage, which makes torch.compile(flex_attention) crash
# with "duplicate template name".  Apply the fix before anything compiles.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
try:
    from inductor_fix import apply_inductor_fix_if_needed
    apply_inductor_fix_if_needed()
except Exception as _e:
    logger.warning(f"Inductor fix skipped: {_e}")

# 当前目录加入 sys.path，供子模块互相引用
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from sol_attn_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    logger.info(f"✓ Sol-Attn nodes loaded: {list(NODE_CLASS_MAPPINGS.keys())}")

    # 打印 GPU 信息（不触发 kernel 编译，只做能力探测）
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            sm = cap[0] * 10 + cap[1]
            if sm == 120:
                logger.info(f"✓ GPU: {name} (SM{sm}) — Sol-Attn via flex_attention (Python) path")
                # 预热 flex_attention 编译内核，避免首次生成时 30-60s 编译延迟
                try:
                    from sol_attn_flex import warmup
                    warmup()
                except Exception as _we:
                    logger.warning(f"flex_attention warmup skipped: {_we}")
            elif sm in (90, 100):
                logger.info(f"✓ GPU: {name} (SM{sm}) — Sol-Attn via CuTe DSL path")
            else:
                logger.warning(f"⚠ GPU: {name} (SM{sm}) — Sol-Attn 不支持此架构，将回退到 SDPA")
    except Exception as e:
        logger.warning(f"⚠ GPU 探测失败: {e}")

except Exception as e:
    logger.error(f"❌ Sol-Attn 节点加载失败: {e}", exc_info=True)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
