import torch
from simpler.task_interface import CallConfig, CommBufferSpec, DataType, TaskArgs, TensorArgType
from pypto.runtime.tensor_arg import make_tensor_arg
from pypto.runtime.distributed_runner import add_chip_tensor_arg, _submit_chip

def _alloc_intermediates(tensors, world_size=1):
    tensors["__comm_d0_ord"] = [torch.zeros((1,), dtype=torch.int32).share_memory_() for _ in range(max(world_size, 1))]

def l3_moe(orch, _args, config, *, tensors, callables, sub_ids, _keep, world_size, _domain_provider=None):
    layer_id__ssa_v0 = tensors["layer_id__ssa_v0"]
    num_tokens__ssa_v0 = tensors["num_tokens__ssa_v0"]
    with (_domain_provider or orch.allocate_domain)(
        name="comm_d0",
        workers=[*range(world_size)],
        window_size=((((256 + 31) // 32) * 32)) + ((((2097152 + 31) // 32) * 32)) + ((((16384 + 31) // 32) * 32)) + ((((8 + 31) // 32) * 32)) + ((((8 + 31) // 32) * 32)) + ((((393216 + 31) // 32) * 32)) + ((((8 + 31) // 32) * 32)),
        buffers=[
            CommBufferSpec(name="recv_meta_buf__ssa_v0", dtype="opaque", count=256, nbytes=(((256 + 31) // 32) * 32)),
            CommBufferSpec(name="recv_x_buf__ssa_v0", dtype="opaque", count=2097152, nbytes=(((2097152 + 31) // 32) * 32)),
            CommBufferSpec(name="recv_aux_buf__ssa_v0", dtype="opaque", count=16384, nbytes=(((16384 + 31) // 32) * 32)),
            CommBufferSpec(name="arrived_buf__ssa_v0", dtype="opaque", count=8, nbytes=(((8 + 31) // 32) * 32)),
            CommBufferSpec(name="data_arrived_buf__ssa_v0", dtype="opaque", count=8, nbytes=(((8 + 31) // 32) * 32)),
            CommBufferSpec(name="routed_y_buf_buf__ssa_v0", dtype="opaque", count=393216, nbytes=(((393216 + 31) // 32) * 32)),
            CommBufferSpec(name="combine_arrived_buf__ssa_v0", dtype="opaque", count=8, nbytes=(((8 + 31) // 32) * 32)),
        ],
    ) as __comm_d0:
        t__tmp_v0 = world_size
        for r__idx_v0 in range(0, t__tmp_v0, 1):
            tensors["t__tmp_v1"] = tensors["x_hc__ssa_v0"][r__idx_v0, 0:8, 0:4, 0:4096]
            tensors["t__tmp_v2"] = tensors["hc_ffn_fn__ssa_v0"][r__idx_v0, 0:24, 0:16384]
            tensors["t__tmp_v3"] = tensors["hc_ffn_scale__ssa_v0"][r__idx_v0, 0:3]
            tensors["t__tmp_v4"] = tensors["hc_ffn_base__ssa_v0"][r__idx_v0, 0:24]
            tensors["t__tmp_v5"] = tensors["norm_w__ssa_v0"][r__idx_v0, 0:4096]
            tensors["t__tmp_v6"] = tensors["gate_w__ssa_v0"][r__idx_v0, 0:64, 0:4096]
            tensors["t__tmp_v7"] = tensors["gate_bias__ssa_v0"][r__idx_v0, 0:64]
            tensors["t__tmp_v8"] = tensors["tid2eid__ssa_v0"][r__idx_v0, 0:129280, 0:6]
            tensors["t__tmp_v9"] = tensors["input_ids__ssa_v0"][r__idx_v0, 0:8]
            tensors["t__tmp_v10"] = tensors["routed_w1__ssa_v0"][r__idx_v0, 0:32, 0:2048, 0:4096]
            tensors["t__tmp_v11"] = tensors["routed_w1_scale__ssa_v0"][r__idx_v0, 0:32, 0:2048]
            tensors["t__tmp_v12"] = tensors["routed_w3__ssa_v0"][r__idx_v0, 0:32, 0:2048, 0:4096]
            tensors["t__tmp_v13"] = tensors["routed_w3_scale__ssa_v0"][r__idx_v0, 0:32, 0:2048]
            tensors["t__tmp_v14"] = tensors["routed_w2__ssa_v0"][r__idx_v0, 0:32, 0:4096, 0:2048]
            tensors["t__tmp_v15"] = tensors["routed_w2_scale__ssa_v0"][r__idx_v0, 0:32, 0:4096]
            tensors["t__tmp_v16"] = tensors["shared_w1__ssa_v0"][r__idx_v0, 0:2048, 0:4096]
            tensors["t__tmp_v17"] = tensors["shared_w1_scale__ssa_v0"][r__idx_v0, 0:2048]
            tensors["t__tmp_v18"] = tensors["shared_w3__ssa_v0"][r__idx_v0, 0:2048, 0:4096]
            tensors["t__tmp_v19"] = tensors["shared_w3_scale__ssa_v0"][r__idx_v0, 0:2048]
            tensors["t__tmp_v20"] = tensors["shared_w2__ssa_v0"][r__idx_v0, 0:4096, 0:2048]
            tensors["t__tmp_v21"] = tensors["shared_w2_scale__ssa_v0"][r__idx_v0, 0:4096]
            tensors["t__tmp_v22"] = tensors["x_next__ssa_v0"][r__idx_v0, 0:8, 0:4, 0:4096]
            _ta_0 = TaskArgs()
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v1"], TensorArgType.INPUT, callables, "l2_moe", 0)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v2"], TensorArgType.INPUT, callables, "l2_moe", 1)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v3"], TensorArgType.INPUT, callables, "l2_moe", 2)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v4"], TensorArgType.INPUT, callables, "l2_moe", 3)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v5"], TensorArgType.INPUT, callables, "l2_moe", 4)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v6"], TensorArgType.INPUT, callables, "l2_moe", 5)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v7"], TensorArgType.INPUT, callables, "l2_moe", 6)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v8"], TensorArgType.INPUT, callables, "l2_moe", 7)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v9"], TensorArgType.INPUT, callables, "l2_moe", 8)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v10"], TensorArgType.INPUT, callables, "l2_moe", 9)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v11"], TensorArgType.INPUT, callables, "l2_moe", 10)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v12"], TensorArgType.INPUT, callables, "l2_moe", 11)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v13"], TensorArgType.INPUT, callables, "l2_moe", 12)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v14"], TensorArgType.INPUT, callables, "l2_moe", 13)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v15"], TensorArgType.INPUT, callables, "l2_moe", 14)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v16"], TensorArgType.INPUT, callables, "l2_moe", 15)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v17"], TensorArgType.INPUT, callables, "l2_moe", 16)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v18"], TensorArgType.INPUT, callables, "l2_moe", 17)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v19"], TensorArgType.INPUT, callables, "l2_moe", 18)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v20"], TensorArgType.INPUT, callables, "l2_moe", 19)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v21"], TensorArgType.INPUT, callables, "l2_moe", 20)
            add_chip_tensor_arg(_ta_0, orch._worker, tensors["t__tmp_v22"], TensorArgType.OUTPUT_EXISTING, callables, "l2_moe", 21)
            _ta_0.add_tensor(__comm_d0[r__idx_v0].buffers["recv_meta_buf__ssa_v0"].tensor(shapes=(2, 32), dtype=DataType.INT32), TensorArgType.INOUT)
            _ta_0.add_tensor(__comm_d0[r__idx_v0].buffers["recv_x_buf__ssa_v0"].tensor(shapes=(512, 4096), dtype=DataType.INT8), TensorArgType.OUTPUT_EXISTING)
            _ta_0.add_tensor(__comm_d0[r__idx_v0].buffers["recv_aux_buf__ssa_v0"].tensor(shapes=(512, 8), dtype=DataType.FLOAT32), TensorArgType.OUTPUT_EXISTING)
            _ta_0.add_tensor(__comm_d0[r__idx_v0].buffers["arrived_buf__ssa_v0"].tensor(shapes=(2, 1), dtype=DataType.INT32), TensorArgType.INOUT)
            _ta_0.add_tensor(__comm_d0[r__idx_v0].buffers["data_arrived_buf__ssa_v0"].tensor(shapes=(2, 1), dtype=DataType.INT32), TensorArgType.INOUT)
            _ta_0.add_tensor(__comm_d0[r__idx_v0].buffers["routed_y_buf_buf__ssa_v0"].tensor(shapes=(48, 4096), dtype=DataType.BFLOAT16), TensorArgType.OUTPUT_EXISTING)
            _ta_0.add_tensor(__comm_d0[r__idx_v0].buffers["combine_arrived_buf__ssa_v0"].tensor(shapes=(2, 1), dtype=DataType.INT32), TensorArgType.INOUT)
            _ta_0.add_tensor(make_tensor_arg(orch._worker, tensors["__comm_d0_ord"][r__idx_v0]), TensorArgType.INOUT)
            _ta_0.add_scalar(layer_id__ssa_v0)
            _ta_0.add_scalar(num_tokens__ssa_v0)
            _ta_0.add_scalar(r__idx_v0)
            _ta_0.add_scalar(1)
            _ta_0.add_scalar(__comm_d0[r__idx_v0].device_ctx)
            _ta_0.add_scalar(__comm_d0[r__idx_v0].device_ctx)
            _ta_0.add_scalar(__comm_d0[r__idx_v0].device_ctx)
            _ta_0.add_scalar(__comm_d0[r__idx_v0].device_ctx)
            _ta_0.add_scalar(__comm_d0[r__idx_v0].device_ctx)
            _ta_0.add_scalar(__comm_d0[r__idx_v0].device_ctx)
            _ta_0.add_scalar(__comm_d0[r__idx_v0].device_ctx)
            _keep.append(_ta_0)
            _submit_chip(orch, callables["l2_moe"], _ta_0, config, r__idx_v0)

l3_moe._pypto_distributed_entry = True

