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
	"source": str(_ROOT_DIR / "orchestration" / "l2_moe.cpp"),
	"function_name": "aicpu_orchestration_entry",
	"signature": [_D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT, _D.INOUT, _D.OUT, _D.OUT, _D.INOUT, _D.INOUT, _D.OUT, _D.INOUT],
}

KERNELS = [
	{"func_id": 0, "name": "hc_pre_rms", "source": str(_ROOT_DIR / "kernels" / "aiv" / "hc_pre_rms.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN]},
	{"func_id": 1, "name": "hc_pre_linear", "source": str(_ROOT_DIR / "kernels" / "aic" / "hc_pre_linear.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 2, "name": "hc_pre_linear_reduce", "source": str(_ROOT_DIR / "kernels" / "aiv" / "hc_pre_linear_reduce.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT]},
	{"func_id": 3, "name": "split_pre_post", "source": str(_ROOT_DIR / "kernels" / "aiv" / "split_pre_post.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 4, "name": "comb_sinkhorn", "source": str(_ROOT_DIR / "kernels" / "aiv" / "comb_sinkhorn.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.OUT]},
	{"func_id": 5, "name": "mix_x", "source": str(_ROOT_DIR / "kernels" / "aiv" / "mix_x.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN]},
	{"func_id": 6, "name": "ffn_norm", "source": str(_ROOT_DIR / "kernels" / "aiv" / "ffn_norm.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.OUT, _D.OUT, _D.OUT, _D.OUT]},
	{"func_id": 7, "name": "x_norm_quant", "source": str(_ROOT_DIR / "kernels" / "aiv" / "x_norm_quant.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN]},
	{"func_id": 8, "name": "gate_pre_route", "source": str(_ROOT_DIR / "kernels" / "aiv" / "gate_pre_route.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.OUT, _D.OUT, _D.OUT]},
	{"func_id": 9, "name": "gate_aic", "source": str(_ROOT_DIR / "kernels" / "aic" / "gate_aic.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.IN, _D.IN, _D.OUT, _D.INOUT, _D.OUT]},
	{"func_id": 10, "name": "gate_aiv", "source": str(_ROOT_DIR / "kernels" / "aiv" / "gate_aiv.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.IN, _D.OUT, _D.INOUT, _D.OUT]},
	{"func_id": 11, "name": "route_hash", "source": str(_ROOT_DIR / "kernels" / "aiv" / "route_hash.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.INOUT, _D.INOUT]},
	{"func_id": 12, "name": "route_sort", "source": str(_ROOT_DIR / "kernels" / "aiv" / "route_sort.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.INOUT, _D.INOUT]},
	{"func_id": 13, "name": "sh_gate_mm", "source": str(_ROOT_DIR / "kernels" / "aic" / "sh_gate_mm.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.OUT]},
	{"func_id": 14, "name": "sh_up_mm", "source": str(_ROOT_DIR / "kernels" / "aic" / "sh_up_mm.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.OUT]},
	{"func_id": 15, "name": "sh_h_tile_i8_init", "source": str(_ROOT_DIR / "kernels" / "aiv" / "sh_h_tile_i8_init.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 16, "name": "sh_gate_up_act_q", "source": str(_ROOT_DIR / "kernels" / "aiv" / "sh_gate_up_act_q.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT, _D.INOUT]},
	{"func_id": 17, "name": "sh_w2_mm", "source": str(_ROOT_DIR / "kernels" / "aic" / "sh_w2_mm.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.OUT]},
	{"func_id": 18, "name": "sh_w2_act", "source": str(_ROOT_DIR / "kernels" / "aiv" / "sh_w2_act.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN, _D.IN]},
	{"func_id": 19, "name": "dispatch_meta", "source": str(_ROOT_DIR / "kernels" / "aiv" / "dispatch_meta.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.INOUT, _D.INOUT, _D.OUT, _D.OUT]},
	{"func_id": 20, "name": "dispatch_push", "source": str(_ROOT_DIR / "kernels" / "aiv" / "dispatch_push.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.OUT, _D.IN, _D.IN, _D.OUT, _D.INOUT]},
	{"func_id": 21, "name": "dispatch_wait", "source": str(_ROOT_DIR / "kernels" / "aiv" / "dispatch_wait.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN]},
	{"func_id": 22, "name": "dispatch_gather", "source": str(_ROOT_DIR / "kernels" / "aiv" / "dispatch_gather.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.OUT, _D.OUT, _D.OUT]},
	{"func_id": 23, "name": "exp_gate_mm", "source": str(_ROOT_DIR / "kernels" / "aic" / "exp_gate_mm.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 24, "name": "exp_up_mm", "source": str(_ROOT_DIR / "kernels" / "aic" / "exp_up_mm.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 25, "name": "exp_gate_up_act", "source": str(_ROOT_DIR / "kernels" / "aiv" / "exp_gate_up_act.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 26, "name": "exp_h_q", "source": str(_ROOT_DIR / "kernels" / "aiv" / "exp_h_q.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.OUT]},
	{"func_id": 27, "name": "exp_w2_mm", "source": str(_ROOT_DIR / "kernels" / "aic" / "exp_w2_mm.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 28, "name": "exp_w2_act", "source": str(_ROOT_DIR / "kernels" / "aiv" / "exp_w2_act.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.OUT, _D.IN, _D.IN]},
	{"func_id": 29, "name": "expert_scatter", "source": str(_ROOT_DIR / "kernels" / "aiv" / "expert_scatter.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.INOUT, _D.IN]},
	{"func_id": 30, "name": "combine_wait", "source": str(_ROOT_DIR / "kernels" / "aiv" / "combine_wait.cpp"), "core_type": "aiv", "signature": [_D.INOUT]},
	{"func_id": 31, "name": "shared_routed", "source": str(_ROOT_DIR / "kernels" / "aiv" / "shared_routed.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 32, "name": "hc_post", "source": str(_ROOT_DIR / "kernels" / "aiv" / "hc_post.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 33, "name": "moe_signal_clear", "source": str(_ROOT_DIR / "kernels" / "aiv" / "moe_signal_clear.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.INOUT, _D.INOUT, _D.INOUT]},
]
