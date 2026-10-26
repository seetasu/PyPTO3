// Generated from archived Paged Attention data. Do not edit by hand.
window.PAGED_ATTENTION_DATA = {
  "schemaVersion": 1,
  "run": "Paged Attention · 20261015_232459",
  "sourceDirectory": "Data/paged attention 编译产物 马金 20261015_232459_qw2ym_86",
  "soc": {
    "name": "72 核",
    "aicCores": 24,
    "aivCores": 48,
    "source": "chip_swimlane_records.metadata"
  },
  "ranks": [
    {
      "rank": "capture",
      "program": "paged_attention_pypto_dynamic",
      "source": "Data/paged attention 编译产物 马金 20261015_232459_qw2ym_86/dfx_outputs",
      "windowUs": 36.96,
      "kpi": {
        "windowUs": 36.96,
        "criticalPathUs": 36.96,
        "criticalPathTasks": 1,
        "totalTasks": 1,
        "singleBlockOnPathUs": 0,
        "singleBlockOnPathShare": 0,
        "computeUs": 1751.78,
        "setupUs": 429.98,
        "slotUs": 2181.76,
        "tailWasteUs": 623.26,
        "capacityUs": 2661.12,
        "occupancy": 0.6582867364117364,
        "setupShare": 0.24545319617760222,
        "aicCores": 24,
        "aivCores": 48,
        "totalCores": 72,
        "byType": {
          "AIC": {
            "cores": 24,
            "computeUs": 586.14,
            "setupUs": 136.32,
            "slotUs": 722.46,
            "waitUs": 0,
            "computeShare": 0.6607819264069265,
            "slotShare": 0.8144615800865802
          },
          "AIV": {
            "cores": 48,
            "computeUs": 1165.64,
            "setupUs": 293.66,
            "slotUs": 1459.3,
            "waitUs": 0,
            "computeShare": 0.6570391414141415,
            "slotShare": 0.8225671897546898
          }
        },
        "bytesTotal": 0,
        "tailWasteRatio": 0.35578668554270515
      },
      "launches": [
        {
          "id": "4294967305|attn_swpipe_aic",
          "taskId": "4294967305",
          "tag": "r1t9",
          "leaf": "attn_swpipe_aic",
          "ct": "AIC",
          "coresUsed": 24,
          "blockNum": 24,
          "logicalBlocks": 24,
          "startUs": 0,
          "endUs": 36.36,
          "spanUs": 36.36,
          "computeUs": 586.14,
          "setupUs": 136.32,
          "slotUs": 722.46,
          "waitUs": 0,
          "tailWasteUs": 198.18,
          "bytes": 0,
          "gbPerS": null,
          "blocks": {
            "n": 24,
            "min": 12.96,
            "p50": 27.4,
            "p90": 31.38,
            "max": 32.68,
            "mean": 24.4225,
            "spread": 1.33811,
            "zero": 0,
            "durs": [
              12.96,
              13.42,
              14.5,
              14.56,
              16.08,
              16.28,
              16.38,
              17.36,
              24.58,
              25.58,
              26.42,
              27.4,
              27.6,
              28.12,
              28.24,
              28.84,
              29.4,
              29.56,
              29.58,
              31.34,
              31.38,
              31.5,
              32.38,
              32.68
            ],
            "items": [
              {
                "core": 0,
                "startUs": 0,
                "endUs": 20.2,
                "slotUs": 20.2,
                "computeUs": 17.36,
                "setupUs": 2.84
              },
              {
                "core": 1,
                "startUs": 0.02,
                "endUs": 34.56,
                "slotUs": 34.54,
                "computeUs": 31.5,
                "setupUs": 3.04
              },
              {
                "core": 2,
                "startUs": 0.02,
                "endUs": 35.94,
                "slotUs": 35.92,
                "computeUs": 32.38,
                "setupUs": 3.54
              },
              {
                "core": 3,
                "startUs": 0.16,
                "endUs": 20.24,
                "slotUs": 20.08,
                "computeUs": 16.38,
                "setupUs": 3.7
              },
              {
                "core": 4,
                "startUs": 0.18,
                "endUs": 34.9,
                "slotUs": 34.72,
                "computeUs": 29.56,
                "setupUs": 5.16
              },
              {
                "core": 5,
                "startUs": 0.24,
                "endUs": 36,
                "slotUs": 35.76,
                "computeUs": 27.6,
                "setupUs": 8.16
              },
              {
                "core": 6,
                "startUs": 0.18,
                "endUs": 20.14,
                "slotUs": 19.96,
                "computeUs": 12.96,
                "setupUs": 7
              },
              {
                "core": 7,
                "startUs": 0.2,
                "endUs": 35.56,
                "slotUs": 35.36,
                "computeUs": 28.12,
                "setupUs": 7.24
              },
              {
                "core": 8,
                "startUs": 0.24,
                "endUs": 36.3,
                "slotUs": 36.06,
                "computeUs": 28.24,
                "setupUs": 7.82
              },
              {
                "core": 9,
                "startUs": 0.18,
                "endUs": 20.18,
                "slotUs": 20,
                "computeUs": 16.08,
                "setupUs": 3.92
              },
              {
                "core": 10,
                "startUs": 0.22,
                "endUs": 35.66,
                "slotUs": 35.44,
                "computeUs": 29.4,
                "setupUs": 6.04
              },
              {
                "core": 11,
                "startUs": 0.26,
                "endUs": 36.36,
                "slotUs": 36.1,
                "computeUs": 32.68,
                "setupUs": 3.42
              },
              {
                "core": 12,
                "startUs": 0.22,
                "endUs": 20.5,
                "slotUs": 20.28,
                "computeUs": 16.28,
                "setupUs": 4
              },
              {
                "core": 13,
                "startUs": 0.52,
                "endUs": 34.18,
                "slotUs": 33.66,
                "computeUs": 26.42,
                "setupUs": 7.24
              },
              {
                "core": 14,
                "startUs": 0.32,
                "endUs": 36.3,
                "slotUs": 35.98,
                "computeUs": 31.38,
                "setupUs": 4.6
              },
              {
                "core": 15,
                "startUs": 0.24,
                "endUs": 20.46,
                "slotUs": 20.22,
                "computeUs": 14.56,
                "setupUs": 5.66
              },
              {
                "core": 16,
                "startUs": 0.5,
                "endUs": 34.4,
                "slotUs": 33.9,
                "computeUs": 27.4,
                "setupUs": 6.5
              },
              {
                "core": 17,
                "startUs": 0.34,
                "endUs": 36.34,
                "slotUs": 36,
                "computeUs": 31.34,
                "setupUs": 4.66
              },
              {
                "core": 18,
                "startUs": 0.26,
                "endUs": 20.32,
                "slotUs": 20.06,
                "computeUs": 13.42,
                "setupUs": 6.64
              },
              {
                "core": 19,
                "startUs": 0.6,
                "endUs": 33.54,
                "slotUs": 32.94,
                "computeUs": 24.58,
                "setupUs": 8.36
              },
              {
                "core": 20,
                "startUs": 0.4,
                "endUs": 36.12,
                "slotUs": 35.72,
                "computeUs": 29.58,
                "setupUs": 6.14
              },
              {
                "core": 21,
                "startUs": 0.26,
                "endUs": 20.56,
                "slotUs": 20.3,
                "computeUs": 14.5,
                "setupUs": 5.8
              },
              {
                "core": 22,
                "startUs": 0.58,
                "endUs": 33.88,
                "slotUs": 33.3,
                "computeUs": 25.58,
                "setupUs": 7.72
              },
              {
                "core": 23,
                "startUs": 0.4,
                "endUs": 36.36,
                "slotUs": 35.96,
                "computeUs": 28.84,
                "setupUs": 7.12
              }
            ]
          },
          "compile": {
            "type": "AIC",
            "level": "AIC",
            "launchKind": "spmd_submit",
            "coreNum": "24",
            "coreNumConst": 24,
            "via": "attn_swpipe_spmd",
            "bornPass": null,
            "bornPassName": null
          }
        },
        {
          "id": "4294967305|attn_swpipe_aiv",
          "taskId": "4294967305",
          "tag": "r1t9",
          "leaf": "attn_swpipe_aiv",
          "ct": "AIV",
          "coresUsed": 48,
          "blockNum": 24,
          "logicalBlocks": 24,
          "startUs": 0.14,
          "endUs": 36.96,
          "spanUs": 36.82,
          "computeUs": 1165.64,
          "setupUs": 293.66,
          "slotUs": 1459.3,
          "waitUs": 0,
          "tailWasteUs": 425.08,
          "bytes": 0,
          "gbPerS": null,
          "blocks": {
            "n": 48,
            "min": 11.78,
            "p50": 27.14,
            "p90": 31.22,
            "max": 33.14,
            "mean": 24.284167,
            "spread": 1.364675,
            "zero": 0,
            "durs": [
              11.78,
              12.1,
              12.3,
              13.4,
              13.42,
              13.86,
              14.48,
              14.6,
              14.86,
              14.98,
              15.1,
              16.02,
              16.12,
              16.66,
              17.28,
              17.48,
              25.34,
              25.56,
              26.46,
              26.76,
              27.06,
              27.08,
              27.12,
              27.14,
              27.26,
              27.76,
              27.82,
              28.44,
              28.44,
              28.72,
              28.92,
              28.94,
              28.96,
              29.5,
              29.6,
              29.68,
              29.72,
              30.12,
              30.14,
              30.36,
              30.84,
              31.06,
              31.22,
              31.36,
              31.66,
              32.06,
              32.96,
              33.14
            ],
            "items": [
              {
                "core": 24,
                "startUs": 0.14,
                "endUs": 20.32,
                "slotUs": 20.18,
                "computeUs": 16.66,
                "setupUs": 3.52
              },
              {
                "core": 25,
                "startUs": 0.18,
                "endUs": 20.26,
                "slotUs": 20.08,
                "computeUs": 16.02,
                "setupUs": 4.06
              },
              {
                "core": 26,
                "startUs": 0.22,
                "endUs": 35,
                "slotUs": 34.78,
                "computeUs": 27.82,
                "setupUs": 6.96
              },
              {
                "core": 27,
                "startUs": 0.22,
                "endUs": 34.98,
                "slotUs": 34.76,
                "computeUs": 27.14,
                "setupUs": 7.62
              },
              {
                "core": 28,
                "startUs": 0.28,
                "endUs": 36.32,
                "slotUs": 36.04,
                "computeUs": 28.44,
                "setupUs": 7.6
              },
              {
                "core": 29,
                "startUs": 0.26,
                "endUs": 36.36,
                "slotUs": 36.1,
                "computeUs": 31.36,
                "setupUs": 4.74
              },
              {
                "core": 30,
                "startUs": 0.2,
                "endUs": 20.32,
                "slotUs": 20.12,
                "computeUs": 15.1,
                "setupUs": 5.02
              },
              {
                "core": 31,
                "startUs": 0.2,
                "endUs": 20.32,
                "slotUs": 20.12,
                "computeUs": 14.86,
                "setupUs": 5.26
              },
              {
                "core": 32,
                "startUs": 0.22,
                "endUs": 35.28,
                "slotUs": 35.06,
                "computeUs": 29.72,
                "setupUs": 5.34
              },
              {
                "core": 33,
                "startUs": 0.22,
                "endUs": 35.42,
                "slotUs": 35.2,
                "computeUs": 28.44,
                "setupUs": 6.76
              },
              {
                "core": 34,
                "startUs": 0.26,
                "endUs": 36.5,
                "slotUs": 36.24,
                "computeUs": 29.5,
                "setupUs": 6.74
              },
              {
                "core": 35,
                "startUs": 0.26,
                "endUs": 36.68,
                "slotUs": 36.42,
                "computeUs": 31.22,
                "setupUs": 5.2
              },
              {
                "core": 36,
                "startUs": 0.2,
                "endUs": 20.22,
                "slotUs": 20.02,
                "computeUs": 14.98,
                "setupUs": 5.04
              },
              {
                "core": 37,
                "startUs": 0.22,
                "endUs": 20.26,
                "slotUs": 20.04,
                "computeUs": 13.42,
                "setupUs": 6.62
              },
              {
                "core": 38,
                "startUs": 0.24,
                "endUs": 36,
                "slotUs": 35.76,
                "computeUs": 28.94,
                "setupUs": 6.82
              },
              {
                "core": 39,
                "startUs": 0.26,
                "endUs": 35.96,
                "slotUs": 35.7,
                "computeUs": 27.12,
                "setupUs": 8.58
              },
              {
                "core": 40,
                "startUs": 0.3,
                "endUs": 36.7,
                "slotUs": 36.4,
                "computeUs": 27.76,
                "setupUs": 8.64
              },
              {
                "core": 41,
                "startUs": 0.3,
                "endUs": 36.74,
                "slotUs": 36.44,
                "computeUs": 30.14,
                "setupUs": 6.3
              },
              {
                "core": 42,
                "startUs": 0.22,
                "endUs": 20.22,
                "slotUs": 20,
                "computeUs": 13.86,
                "setupUs": 6.14
              },
              {
                "core": 43,
                "startUs": 0.22,
                "endUs": 20.24,
                "slotUs": 20.02,
                "computeUs": 14.48,
                "setupUs": 5.54
              },
              {
                "core": 44,
                "startUs": 0.24,
                "endUs": 36.08,
                "slotUs": 35.84,
                "computeUs": 30.12,
                "setupUs": 5.72
              },
              {
                "core": 45,
                "startUs": 0.24,
                "endUs": 36.18,
                "slotUs": 35.94,
                "computeUs": 26.76,
                "setupUs": 9.18
              },
              {
                "core": 46,
                "startUs": 0.3,
                "endUs": 36.86,
                "slotUs": 36.56,
                "computeUs": 27.26,
                "setupUs": 9.3
              },
              {
                "core": 47,
                "startUs": 0.28,
                "endUs": 36.9,
                "slotUs": 36.62,
                "computeUs": 29.6,
                "setupUs": 7.02
              },
              {
                "core": 48,
                "startUs": 0.22,
                "endUs": 20.58,
                "slotUs": 20.36,
                "computeUs": 13.4,
                "setupUs": 6.96
              },
              {
                "core": 49,
                "startUs": 0.24,
                "endUs": 20.6,
                "slotUs": 20.36,
                "computeUs": 12.3,
                "setupUs": 8.06
              },
              {
                "core": 50,
                "startUs": 0.5,
                "endUs": 34.64,
                "slotUs": 34.14,
                "computeUs": 26.46,
                "setupUs": 7.68
              },
              {
                "core": 51,
                "startUs": 0.5,
                "endUs": 34.62,
                "slotUs": 34.12,
                "computeUs": 27.06,
                "setupUs": 7.06
              },
              {
                "core": 52,
                "startUs": 0.3,
                "endUs": 36.72,
                "slotUs": 36.42,
                "computeUs": 28.96,
                "setupUs": 7.46
              },
              {
                "core": 53,
                "startUs": 0.3,
                "endUs": 36.76,
                "slotUs": 36.46,
                "computeUs": 28.92,
                "setupUs": 7.54
              },
              {
                "core": 54,
                "startUs": 0.24,
                "endUs": 20.52,
                "slotUs": 20.28,
                "computeUs": 17.48,
                "setupUs": 2.8
              },
              {
                "core": 55,
                "startUs": 0.24,
                "endUs": 20.54,
                "slotUs": 20.3,
                "computeUs": 17.28,
                "setupUs": 3.02
              },
              {
                "core": 56,
                "startUs": 0.5,
                "endUs": 34.8,
                "slotUs": 34.3,
                "computeUs": 27.08,
                "setupUs": 7.22
              },
              {
                "core": 57,
                "startUs": 0.5,
                "endUs": 34.94,
                "slotUs": 34.44,
                "computeUs": 30.36,
                "setupUs": 4.08
              },
              {
                "core": 58,
                "startUs": 0.32,
                "endUs": 36.9,
                "slotUs": 36.58,
                "computeUs": 33.14,
                "setupUs": 3.44
              },
              {
                "core": 59,
                "startUs": 0.38,
                "endUs": 36.9,
                "slotUs": 36.52,
                "computeUs": 31.06,
                "setupUs": 5.46
              },
              {
                "core": 60,
                "startUs": 0.24,
                "endUs": 20.4,
                "slotUs": 20.16,
                "computeUs": 14.6,
                "setupUs": 5.56
              },
              {
                "core": 61,
                "startUs": 0.24,
                "endUs": 20.42,
                "slotUs": 20.18,
                "computeUs": 11.78,
                "setupUs": 8.4
              },
              {
                "core": 62,
                "startUs": 0.56,
                "endUs": 34,
                "slotUs": 33.44,
                "computeUs": 25.34,
                "setupUs": 8.1
              },
              {
                "core": 63,
                "startUs": 0.56,
                "endUs": 33.96,
                "slotUs": 33.4,
                "computeUs": 28.72,
                "setupUs": 4.68
              },
              {
                "core": 64,
                "startUs": 0.38,
                "endUs": 36.56,
                "slotUs": 36.18,
                "computeUs": 31.66,
                "setupUs": 4.52
              },
              {
                "core": 65,
                "startUs": 0.38,
                "endUs": 36.74,
                "slotUs": 36.36,
                "computeUs": 32.06,
                "setupUs": 4.3
              },
              {
                "core": 66,
                "startUs": 0.46,
                "endUs": 20.64,
                "slotUs": 20.18,
                "computeUs": 16.12,
                "setupUs": 4.06
              },
              {
                "core": 67,
                "startUs": 0.48,
                "endUs": 20.66,
                "slotUs": 20.18,
                "computeUs": 12.1,
                "setupUs": 8.08
              },
              {
                "core": 68,
                "startUs": 0.56,
                "endUs": 34.3,
                "slotUs": 33.74,
                "computeUs": 25.56,
                "setupUs": 8.18
              },
              {
                "core": 69,
                "startUs": 0.56,
                "endUs": 34.44,
                "slotUs": 33.88,
                "computeUs": 29.68,
                "setupUs": 4.2
              },
              {
                "core": 70,
                "startUs": 0.36,
                "endUs": 36.88,
                "slotUs": 36.52,
                "computeUs": 32.96,
                "setupUs": 3.56
              },
              {
                "core": 71,
                "startUs": 0.6,
                "endUs": 36.96,
                "slotUs": 36.36,
                "computeUs": 30.84,
                "setupUs": 5.52
              }
            ]
          },
          "compile": {
            "type": "AIV",
            "level": "AIV",
            "launchKind": "spmd_submit",
            "coreNum": "24",
            "coreNumConst": 24,
            "via": "attn_swpipe_spmd",
            "bornPass": null,
            "bornPassName": null
          }
        }
      ],
      "cores": [
        {
          "id": 0,
          "ct": "AIC",
          "slotUs": 20.2,
          "computeUs": 17.36,
          "setupUs": 2.84,
          "waitUs": 0
        },
        {
          "id": 1,
          "ct": "AIC",
          "slotUs": 34.54,
          "computeUs": 31.5,
          "setupUs": 3.04,
          "waitUs": 0
        },
        {
          "id": 2,
          "ct": "AIC",
          "slotUs": 35.92,
          "computeUs": 32.38,
          "setupUs": 3.54,
          "waitUs": 0
        },
        {
          "id": 3,
          "ct": "AIC",
          "slotUs": 20.08,
          "computeUs": 16.38,
          "setupUs": 3.7,
          "waitUs": 0
        },
        {
          "id": 4,
          "ct": "AIC",
          "slotUs": 34.72,
          "computeUs": 29.56,
          "setupUs": 5.16,
          "waitUs": 0
        },
        {
          "id": 5,
          "ct": "AIC",
          "slotUs": 35.76,
          "computeUs": 27.6,
          "setupUs": 8.16,
          "waitUs": 0
        },
        {
          "id": 6,
          "ct": "AIC",
          "slotUs": 19.96,
          "computeUs": 12.96,
          "setupUs": 7,
          "waitUs": 0
        },
        {
          "id": 7,
          "ct": "AIC",
          "slotUs": 35.36,
          "computeUs": 28.12,
          "setupUs": 7.24,
          "waitUs": 0
        },
        {
          "id": 8,
          "ct": "AIC",
          "slotUs": 36.06,
          "computeUs": 28.24,
          "setupUs": 7.82,
          "waitUs": 0
        },
        {
          "id": 9,
          "ct": "AIC",
          "slotUs": 20,
          "computeUs": 16.08,
          "setupUs": 3.92,
          "waitUs": 0
        },
        {
          "id": 10,
          "ct": "AIC",
          "slotUs": 35.44,
          "computeUs": 29.4,
          "setupUs": 6.04,
          "waitUs": 0
        },
        {
          "id": 11,
          "ct": "AIC",
          "slotUs": 36.1,
          "computeUs": 32.68,
          "setupUs": 3.42,
          "waitUs": 0
        },
        {
          "id": 12,
          "ct": "AIC",
          "slotUs": 20.28,
          "computeUs": 16.28,
          "setupUs": 4,
          "waitUs": 0
        },
        {
          "id": 13,
          "ct": "AIC",
          "slotUs": 33.66,
          "computeUs": 26.42,
          "setupUs": 7.24,
          "waitUs": 0
        },
        {
          "id": 14,
          "ct": "AIC",
          "slotUs": 35.98,
          "computeUs": 31.38,
          "setupUs": 4.6,
          "waitUs": 0
        },
        {
          "id": 15,
          "ct": "AIC",
          "slotUs": 20.22,
          "computeUs": 14.56,
          "setupUs": 5.66,
          "waitUs": 0
        },
        {
          "id": 16,
          "ct": "AIC",
          "slotUs": 33.9,
          "computeUs": 27.4,
          "setupUs": 6.5,
          "waitUs": 0
        },
        {
          "id": 17,
          "ct": "AIC",
          "slotUs": 36,
          "computeUs": 31.34,
          "setupUs": 4.66,
          "waitUs": 0
        },
        {
          "id": 18,
          "ct": "AIC",
          "slotUs": 20.06,
          "computeUs": 13.42,
          "setupUs": 6.64,
          "waitUs": 0
        },
        {
          "id": 19,
          "ct": "AIC",
          "slotUs": 32.94,
          "computeUs": 24.58,
          "setupUs": 8.36,
          "waitUs": 0
        },
        {
          "id": 20,
          "ct": "AIC",
          "slotUs": 35.72,
          "computeUs": 29.58,
          "setupUs": 6.14,
          "waitUs": 0
        },
        {
          "id": 21,
          "ct": "AIC",
          "slotUs": 20.3,
          "computeUs": 14.5,
          "setupUs": 5.8,
          "waitUs": 0
        },
        {
          "id": 22,
          "ct": "AIC",
          "slotUs": 33.3,
          "computeUs": 25.58,
          "setupUs": 7.72,
          "waitUs": 0
        },
        {
          "id": 23,
          "ct": "AIC",
          "slotUs": 35.96,
          "computeUs": 28.84,
          "setupUs": 7.12,
          "waitUs": 0
        },
        {
          "id": 24,
          "ct": "AIV",
          "slotUs": 20.18,
          "computeUs": 16.66,
          "setupUs": 3.52,
          "waitUs": 0
        },
        {
          "id": 25,
          "ct": "AIV",
          "slotUs": 20.08,
          "computeUs": 16.02,
          "setupUs": 4.06,
          "waitUs": 0
        },
        {
          "id": 26,
          "ct": "AIV",
          "slotUs": 34.78,
          "computeUs": 27.82,
          "setupUs": 6.96,
          "waitUs": 0
        },
        {
          "id": 27,
          "ct": "AIV",
          "slotUs": 34.76,
          "computeUs": 27.14,
          "setupUs": 7.62,
          "waitUs": 0
        },
        {
          "id": 28,
          "ct": "AIV",
          "slotUs": 36.04,
          "computeUs": 28.44,
          "setupUs": 7.6,
          "waitUs": 0
        },
        {
          "id": 29,
          "ct": "AIV",
          "slotUs": 36.1,
          "computeUs": 31.36,
          "setupUs": 4.74,
          "waitUs": 0
        },
        {
          "id": 30,
          "ct": "AIV",
          "slotUs": 20.12,
          "computeUs": 15.1,
          "setupUs": 5.02,
          "waitUs": 0
        },
        {
          "id": 31,
          "ct": "AIV",
          "slotUs": 20.12,
          "computeUs": 14.86,
          "setupUs": 5.26,
          "waitUs": 0
        },
        {
          "id": 32,
          "ct": "AIV",
          "slotUs": 35.06,
          "computeUs": 29.72,
          "setupUs": 5.34,
          "waitUs": 0
        },
        {
          "id": 33,
          "ct": "AIV",
          "slotUs": 35.2,
          "computeUs": 28.44,
          "setupUs": 6.76,
          "waitUs": 0
        },
        {
          "id": 34,
          "ct": "AIV",
          "slotUs": 36.24,
          "computeUs": 29.5,
          "setupUs": 6.74,
          "waitUs": 0
        },
        {
          "id": 35,
          "ct": "AIV",
          "slotUs": 36.42,
          "computeUs": 31.22,
          "setupUs": 5.2,
          "waitUs": 0
        },
        {
          "id": 36,
          "ct": "AIV",
          "slotUs": 20.02,
          "computeUs": 14.98,
          "setupUs": 5.04,
          "waitUs": 0
        },
        {
          "id": 37,
          "ct": "AIV",
          "slotUs": 20.04,
          "computeUs": 13.42,
          "setupUs": 6.62,
          "waitUs": 0
        },
        {
          "id": 38,
          "ct": "AIV",
          "slotUs": 35.76,
          "computeUs": 28.94,
          "setupUs": 6.82,
          "waitUs": 0
        },
        {
          "id": 39,
          "ct": "AIV",
          "slotUs": 35.7,
          "computeUs": 27.12,
          "setupUs": 8.58,
          "waitUs": 0
        },
        {
          "id": 40,
          "ct": "AIV",
          "slotUs": 36.4,
          "computeUs": 27.76,
          "setupUs": 8.64,
          "waitUs": 0
        },
        {
          "id": 41,
          "ct": "AIV",
          "slotUs": 36.44,
          "computeUs": 30.14,
          "setupUs": 6.3,
          "waitUs": 0
        },
        {
          "id": 42,
          "ct": "AIV",
          "slotUs": 20,
          "computeUs": 13.86,
          "setupUs": 6.14,
          "waitUs": 0
        },
        {
          "id": 43,
          "ct": "AIV",
          "slotUs": 20.02,
          "computeUs": 14.48,
          "setupUs": 5.54,
          "waitUs": 0
        },
        {
          "id": 44,
          "ct": "AIV",
          "slotUs": 35.84,
          "computeUs": 30.12,
          "setupUs": 5.72,
          "waitUs": 0
        },
        {
          "id": 45,
          "ct": "AIV",
          "slotUs": 35.94,
          "computeUs": 26.76,
          "setupUs": 9.18,
          "waitUs": 0
        },
        {
          "id": 46,
          "ct": "AIV",
          "slotUs": 36.56,
          "computeUs": 27.26,
          "setupUs": 9.3,
          "waitUs": 0
        },
        {
          "id": 47,
          "ct": "AIV",
          "slotUs": 36.62,
          "computeUs": 29.6,
          "setupUs": 7.02,
          "waitUs": 0
        },
        {
          "id": 48,
          "ct": "AIV",
          "slotUs": 20.36,
          "computeUs": 13.4,
          "setupUs": 6.96,
          "waitUs": 0
        },
        {
          "id": 49,
          "ct": "AIV",
          "slotUs": 20.36,
          "computeUs": 12.3,
          "setupUs": 8.06,
          "waitUs": 0
        },
        {
          "id": 50,
          "ct": "AIV",
          "slotUs": 34.14,
          "computeUs": 26.46,
          "setupUs": 7.68,
          "waitUs": 0
        },
        {
          "id": 51,
          "ct": "AIV",
          "slotUs": 34.12,
          "computeUs": 27.06,
          "setupUs": 7.06,
          "waitUs": 0
        },
        {
          "id": 52,
          "ct": "AIV",
          "slotUs": 36.42,
          "computeUs": 28.96,
          "setupUs": 7.46,
          "waitUs": 0
        },
        {
          "id": 53,
          "ct": "AIV",
          "slotUs": 36.46,
          "computeUs": 28.92,
          "setupUs": 7.54,
          "waitUs": 0
        },
        {
          "id": 54,
          "ct": "AIV",
          "slotUs": 20.28,
          "computeUs": 17.48,
          "setupUs": 2.8,
          "waitUs": 0
        },
        {
          "id": 55,
          "ct": "AIV",
          "slotUs": 20.3,
          "computeUs": 17.28,
          "setupUs": 3.02,
          "waitUs": 0
        },
        {
          "id": 56,
          "ct": "AIV",
          "slotUs": 34.3,
          "computeUs": 27.08,
          "setupUs": 7.22,
          "waitUs": 0
        },
        {
          "id": 57,
          "ct": "AIV",
          "slotUs": 34.44,
          "computeUs": 30.36,
          "setupUs": 4.08,
          "waitUs": 0
        },
        {
          "id": 58,
          "ct": "AIV",
          "slotUs": 36.58,
          "computeUs": 33.14,
          "setupUs": 3.44,
          "waitUs": 0
        },
        {
          "id": 59,
          "ct": "AIV",
          "slotUs": 36.52,
          "computeUs": 31.06,
          "setupUs": 5.46,
          "waitUs": 0
        },
        {
          "id": 60,
          "ct": "AIV",
          "slotUs": 20.16,
          "computeUs": 14.6,
          "setupUs": 5.56,
          "waitUs": 0
        },
        {
          "id": 61,
          "ct": "AIV",
          "slotUs": 20.18,
          "computeUs": 11.78,
          "setupUs": 8.4,
          "waitUs": 0
        },
        {
          "id": 62,
          "ct": "AIV",
          "slotUs": 33.44,
          "computeUs": 25.34,
          "setupUs": 8.1,
          "waitUs": 0
        },
        {
          "id": 63,
          "ct": "AIV",
          "slotUs": 33.4,
          "computeUs": 28.72,
          "setupUs": 4.68,
          "waitUs": 0
        },
        {
          "id": 64,
          "ct": "AIV",
          "slotUs": 36.18,
          "computeUs": 31.66,
          "setupUs": 4.52,
          "waitUs": 0
        },
        {
          "id": 65,
          "ct": "AIV",
          "slotUs": 36.36,
          "computeUs": 32.06,
          "setupUs": 4.3,
          "waitUs": 0
        },
        {
          "id": 66,
          "ct": "AIV",
          "slotUs": 20.18,
          "computeUs": 16.12,
          "setupUs": 4.06,
          "waitUs": 0
        },
        {
          "id": 67,
          "ct": "AIV",
          "slotUs": 20.18,
          "computeUs": 12.1,
          "setupUs": 8.08,
          "waitUs": 0
        },
        {
          "id": 68,
          "ct": "AIV",
          "slotUs": 33.74,
          "computeUs": 25.56,
          "setupUs": 8.18,
          "waitUs": 0
        },
        {
          "id": 69,
          "ct": "AIV",
          "slotUs": 33.88,
          "computeUs": 29.68,
          "setupUs": 4.2,
          "waitUs": 0
        },
        {
          "id": 70,
          "ct": "AIV",
          "slotUs": 36.52,
          "computeUs": 32.96,
          "setupUs": 3.56,
          "waitUs": 0
        },
        {
          "id": 71,
          "ct": "AIV",
          "slotUs": 36.36,
          "computeUs": 30.84,
          "setupUs": 5.52,
          "waitUs": 0
        }
      ],
      "criticalPath": [
        {
          "taskId": "4294967305",
          "name": "attn_swpipe",
          "spanUs": 36.96,
          "startUs": 0,
          "endUs": 36.96,
          "pathContributionUs": 36.96,
          "computeUs": 1751.78,
          "blockNum": 24,
          "onCore": true,
          "ids": [
            "4294967305|attn_swpipe_aic",
            "4294967305|attn_swpipe_aiv"
          ]
        }
      ],
      "edges": [],
      "taskNames": {
        "4294967305": "attn_swpipe"
      },
      "paged": {
        "taskId": "4294967305",
        "logicalBlocks": 24,
        "physicalRecords": 72,
        "mixedKernels": [
          0,
          1,
          1
        ],
        "originUs": 120.08,
        "logicalBlockMappingAvailable": false,
        "nonComputeTasks": 7
      }
    }
  ],
  "caveats": [
    "仅包含一个计算 Task 的 72 条核上记录；编排依赖中的非计算条目不作为算子绘制。",
    "24 个逻辑 block 的结构来自 block_num 与 MixedKernels；原始核记录不含 block_idx，不按核号推断实际 block 派发映射。",
    "时间来自同目录 merged_swimlane，与原始 cycles 按 50 MHz 逐核交叉核对；坐标原点为最早核记录开始。"
  ]
};
