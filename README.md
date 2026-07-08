# GPU Waste in Reasoning Model Inference

**DeepSeek-R1 at production scale: 44% structural GPU waste, $477,000/month.**

This repository measures structural GPU waste in LLM inference clusters —
particularly for reasoning models (DeepSeek-R1, o1-class) where the problem
is dramatically worse than general LLMs.

---

## The core finding

Reasoning models output **25x more tokens** per request than general LLMs.

```
DeepSeek-R1:  avg 2,073 output tokens  (thinking chains)
General 14B:  avg 82 output tokens
```

More tokens per request = GPU held longer per request = larger gap between
peak demand and average demand = more structural waste.

This waste is a **lower bound**, not an estimate:

```
min_waste = 1 - (avg_demand / peak_demand)
          = 1 - (1 / burstiness)
```

No batching strategy, quantization, or framework tuning eliminates it.
The only fix is pooling across services.

---

## Results across five model types

From ServeGen (NSDI 2026) — parameterized from **3.54 billion real Alibaba
Cloud production requests** across DeepSeek-R1, Qwen2.5-VL, and general LLMs:

| Model | Burstiness | Avg utilization | Structural waste | Monthly cost |
|---|---|---|---|---|
| General 14B | 1.9x | 52% | 25% | $3,600 |
| General 72B | 2.0x | 49% | 36% | $25,200 |
| General 310B | 1.9x | 53% | 31% | $37,800 |
| Qwen2.5-VL Image | 1.9x | 53% | 33% | $3,600 |
| **DeepSeek-R1 671B** | **2.3x** | **43%** | **44%** | **$477,000** |
| **Combined (pooled)** | — | — | **42%** | **$549,000** |

DeepSeek-R1 dominates because it is both bursty **and** compute-heavy:
at 200 tokens/sec per GPU, each request holds the GPU for ~10 seconds.
The capacity required to handle peak demand is large.
When demand drops, that capacity sits idle at enormous cost.

---

## Why reasoning makes it worse

General LLM request: 82 output tokens → 0.04s GPU time at 2,000 tok/s
DeepSeek-R1 request: 2,073 output tokens → 10.4s GPU time at 200 tok/s

The GPU occupancy per request is **260x higher** for DeepSeek-R1.

This means:
- You need more GPUs to handle the same request rate at peak
- Between bursts, those GPUs sit idle
- The idle cost scales with model size and token length

The structural waste formula captures this directly:
peak demand grows with output length, average demand does not change
proportionally, so the ratio widens.

---

## Cross-dataset validation

The same structural waste pattern appears across every public dataset:

| Dataset | Paper | Year | Waste |
|---|---|---|---|
| Azure LLM Inference | Splitwise ISCA 2024 | Nov 2023 | 17–53% |
| Azure LMM Multimodal | ModServe SoCC 2025 | Oct 2024 | 64% |
| BurstGPT | KDD 2025 | 2022–23 | ~50% |
| **ServeGen** | **NSDI 2026** | **2026** | **25–44%** |

Alibaba proved this at hyperscale with Aegaeon (SOSP 2025):
1,192 GPUs → 213 GPUs (−82%) with pooling across 47 models.

---

## Run the analysis

```bash
# Clone ServeGen (data files are in the repo)
git clone https://github.com/alibaba/ServeGen.git
pip install -e ServeGen scipy numpy matplotlib

# Run the waste analysis
python servegen_waste_2026.py
```

No login required. Data is Apache-2.0 licensed.

---

## What this means for teams running DeepSeek-R1

If you run a dedicated DeepSeek-R1 instance sized for peak demand:

- Your average GPU utilization is ~43%
- 44% of your GPU budget is structurally idle
- At H100 spot pricing ($2.50/hr), that's ~$477k/month at this scale

The fix is not a better serving framework.
The fix is pooling across models so idle capacity from one
serves demand from another.

---

## Data and citations

```bibtex
@inproceedings{xiang2026servegen,
  title={ServeGen: Workload Characterization and Generation of
         Large Language Model Serving in Production},
  author={Xiang, Yuxing and Li, Xue and Qian, Kun and Yu, Wenyuan
          and Zhai, Ennan and Jin, Xin},
  booktitle={NSDI 2026},
  year={2026}
}

@inproceedings{xiang2025aegaeon,
  title={Aegaeon: Effective GPU Pooling for Concurrent LLM Serving
         on the Market},
  booktitle={SOSP 2025},
  year={2025}
}
```

*Data: [alibaba/ServeGen](https://github.com/alibaba/ServeGen) (Apache-2.0)*
*Code: MIT License*
