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
	"source": str(_ROOT_DIR / "orchestration" / "_sparse_attn.cpp"),
	"function_name": "aicpu_orchestration_entry",
	"signature": [_D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT],
}

KERNELS = [
	{"func_id": 0, "name": "prefill_sparse_sources_ready", "source": str(_ROOT_DIR / "kernels" / "aiv" / "prefill_sparse_sources_ready.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 1, "name": "physical_sparse_packed_init", "source": str(_ROOT_DIR / "kernels" / "aiv" / "physical_sparse_packed_init.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 2, "name": "staged_swa_rope_cs", "source": str(_ROOT_DIR / "kernels" / "aiv" / "staged_swa_rope_cs.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.OUT, _D.OUT]},
	{"func_id": 3, "name": "physical_sparse_gather_ori", "source": str(_ROOT_DIR / "kernels" / "aiv" / "physical_sparse_gather_ori.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.OUT]},
	{"func_id": 4, "name": "physical_sparse_gather_cmp", "source": str(_ROOT_DIR / "kernels" / "aiv" / "physical_sparse_gather_cmp.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 5, "name": "physical_sparse_build_bias", "source": str(_ROOT_DIR / "kernels" / "aiv" / "physical_sparse_build_bias.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN]},
	{"func_id": 6, "name": "staged_swa_qk_pv_aic", "source": str(_ROOT_DIR / "kernels" / "aic" / "staged_swa_qk_pv_aic.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT]},
	{"func_id": 7, "name": "staged_swa_qk_pv_aiv", "source": str(_ROOT_DIR / "kernels" / "aiv" / "staged_swa_qk_pv_aiv.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT]},
	{"func_id": 8, "name": "staged_swa_merge_rope_pack", "source": str(_ROOT_DIR / "kernels" / "aiv" / "staged_swa_merge_rope_pack.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT]},
	{"func_id": 9, "name": "staged_swa_proj_a_mm", "source": str(_ROOT_DIR / "kernels" / "aic" / "staged_swa_proj_a_mm.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.OUT]},
	{"func_id": 10, "name": "staged_swa_quant", "source": str(_ROOT_DIR / "kernels" / "aiv" / "staged_swa_quant.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.OUT]},
	{"func_id": 11, "name": "staged_swa_proj_b_mm", "source": str(_ROOT_DIR / "kernels" / "aic" / "staged_swa_proj_b_mm.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 12, "name": "staged_swa_proj_b_act", "source": str(_ROOT_DIR / "kernels" / "aiv" / "staged_swa_proj_b_act.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.OUT]},
	{"func_id": 13, "name": "physical_sparse_publish", "source": str(_ROOT_DIR / "kernels" / "aiv" / "physical_sparse_publish.cpp"), "core_type": "aiv", "signature": [_D.INOUT]},
]
