# Kernel and Orchestration Configuration

from pathlib import Path

from simpler.task_interface import ArgDirection as _D

_ROOT_DIR = Path(__file__).parent

# Runtime configuration for tensormap_and_ringbuffer.
# AICPU thread count 0 selects the runtime's architecture default (a2a3: 4; a5: 5).
RUNTIME_CONFIG = {
	"runtime": "tensormap_and_ringbuffer",
	"aicpu_thread_num": 0,
}

ORCHESTRATION = {
	"source": str(_ROOT_DIR / "orchestration" / "paged_attention_pypto_dynamic.cpp"),
	"function_name": "aicpu_orchestration_entry",
	"signature": [_D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT],
}

KERNELS = [
	{"func_id": 0, "name": "attn_swpipe_aic", "source": str(_ROOT_DIR / "kernels" / "aic" / "attn_swpipe_aic.cpp"), "core_type": "aic", "signature": [_D.IN, _D.INOUT, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.INOUT, _D.OUT]},
	{"func_id": 1, "name": "attn_swpipe_aiv", "source": str(_ROOT_DIR / "kernels" / "aiv" / "attn_swpipe_aiv.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.INOUT, _D.INOUT, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT, _D.INOUT, _D.OUT]},
]
