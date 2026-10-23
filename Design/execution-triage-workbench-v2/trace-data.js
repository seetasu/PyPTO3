window.TRACE_SUMMARY = {
  "schemaVersion": 1,
  "source": "PyTorch profiler Chrome trace from archive",
  "eventCount": 6458204,
  "traceSpanMs": 74216.235,
  "timestampRangeUs": [
    1788868836492767,
    1788868910709002
  ],
  "categories": [
    {
      "key": "(none)",
      "count": 3149107,
      "durMs": 221929.689
    },
    {
      "key": "cpu_op",
      "count": 1152320,
      "durMs": 82700.916
    },
    {
      "key": "dequeue",
      "count": 252364,
      "durMs": 7072.131
    },
    {
      "key": "enqueue",
      "count": 252364,
      "durMs": 792.513
    },
    {
      "key": "async_task_queue",
      "count": 504728,
      "durMs": 0
    },
    {
      "key": "HostToDevice",
      "count": 660037,
      "durMs": 0
    },
    {
      "key": "async_npu",
      "count": 487284,
      "durMs": 0
    }
  ],
  "phases": [
    {
      "key": "X",
      "count": 4805951,
      "durMs": 312495.249
    },
    {
      "key": "f",
      "count": 909764,
      "durMs": 0
    },
    {
      "key": "s",
      "count": 742285,
      "durMs": 0
    },
    {
      "key": "M",
      "count": 202,
      "durMs": 0
    },
    {
      "key": "C",
      "count": 2,
      "durMs": 0
    }
  ],
  "topEvents": [
    {
      "key": "(none) · NOTIFY_WAIT_SQE",
      "count": 170801,
      "durMs": 59855.646
    },
    {
      "key": "(none) · EVENT_WAIT",
      "count": 26782,
      "durMs": 34874.489
    },
    {
      "key": "cpu_op · npu_fx_compiler inference",
      "count": 29,
      "durMs": 19485.645
    },
    {
      "key": "cpu_op · vllm::moe_forward_shared",
      "count": 1535,
      "durMs": 14320.156
    },
    {
      "key": "(none) · Communication",
      "count": 20025,
      "durMs": 12413.539
    },
    {
      "key": "(none) · Communication(Not Overlapped)",
      "count": 22528,
      "durMs": 12305.954
    },
    {
      "key": "(none) · Computing",
      "count": 266323,
      "durMs": 9395.845
    },
    {
      "key": "(none) · NOTIFY_WAIT",
      "count": 14938,
      "durMs": 8765.333
    },
    {
      "key": "(none) · AivKernel",
      "count": 15023,
      "durMs": 8636.57
    },
    {
      "key": "(none) · Notify_Wait",
      "count": 18703,
      "durMs": 7974.517
    },
    {
      "key": "(none) · Free",
      "count": 283214,
      "durMs": 7246.537
    },
    {
      "key": "cpu_op · aten::item",
      "count": 9522,
      "durMs": 7161.42
    },
    {
      "key": "cpu_op · aten::_local_scalar_dense",
      "count": 9522,
      "durMs": 7146.553
    },
    {
      "key": "cpu_op · aten::repeat_interleave",
      "count": 2999,
      "durMs": 6834.126
    },
    {
      "key": "cpu_op · vllm::dsa_forward",
      "count": 1535,
      "durMs": 6302.195
    },
    {
      "key": "(none) · AscendCL@aclrtSynchronizeStreamWithTimeout",
      "count": 1461,
      "durMs": 5124.689
    },
    {
      "key": "(none) · alltoallvAivKernel",
      "count": 3612,
      "durMs": 5077.899
    },
    {
      "key": "(none) · Node@launch",
      "count": 158568,
      "durMs": 2906.667
    },
    {
      "key": "(none) · allgatherAivKernel",
      "count": 4615,
      "durMs": 2713.248
    },
    {
      "key": "(none) · AscendCL@aclrtSynchronizeEvent",
      "count": 290,
      "durMs": 2127.252
    },
    {
      "key": "cpu_op · npu::npu_quant_matmul",
      "count": 8143,
      "durMs": 1813.278
    },
    {
      "key": "(none) · AscendCL@aclnnQuantMatmulWeightNzGetWorkspaceSize",
      "count": 7901,
      "durMs": 1376.445
    },
    {
      "key": "cpu_op · gloo:all_reduce",
      "count": 192,
      "durMs": 1259.6
    },
    {
      "key": "(none) · SDMA_SQE",
      "count": 71076,
      "durMs": 1148.455
    },
    {
      "key": "(none) · AscendCL@QuantBatchMatmulV3_Tiling",
      "count": 15802,
      "durMs": 1030.495
    },
    {
      "key": "(none) · Notify_Record",
      "count": 13855,
      "durMs": 1004.927
    },
    {
      "key": "(none) · MoeDistributeDispatchV2",
      "count": 331,
      "durMs": 976.715
    },
    {
      "key": "(none) · AscendCL@aclrtLaunchKernelWithHostArgs",
      "count": 150973,
      "durMs": 945.549
    },
    {
      "key": "cpu_op · aten::linear",
      "count": 5692,
      "durMs": 873.28
    },
    {
      "key": "cpu_op · Event::synchronize",
      "count": 290,
      "durMs": 850.87
    },
    {
      "key": "cpu_op · aten::copy_",
      "count": 33437,
      "durMs": 842.605
    },
    {
      "key": "cpu_op · aten::matmul",
      "count": 5721,
      "durMs": 788.578
    },
    {
      "key": "cpu_op · empty_tensor",
      "count": 202790,
      "durMs": 724.673
    },
    {
      "key": "cpu_op · vllm::matmul_and_reduce",
      "count": 1399,
      "durMs": 713.934
    },
    {
      "key": "cpu_op · npu::npu_grouped_matmul",
      "count": 1535,
      "durMs": 659.773
    },
    {
      "key": "dequeue · Dequeue@record_event",
      "count": 46490,
      "durMs": 658.536
    },
    {
      "key": "cpu_op · vllm::all_gather",
      "count": 2837,
      "durMs": 605.578
    },
    {
      "key": "cpu_op · npu::npu_dynamic_quant",
      "count": 7001,
      "durMs": 600.539
    },
    {
      "key": "dequeue · Dequeue@aclnnInplaceCopy",
      "count": 19937,
      "durMs": 579.567
    },
    {
      "key": "(none) · aclnnGroupedMatmulSwigluQuantWeightNZ_GroupedMatmulSwigluQuant_GroupedMatmulSwigluQuant",
      "count": 1535,
      "durMs": 576.018
    }
  ]
}
;\n