# Marathi LLM Study — Benchmark Evaluation Summary

Generated automatically from `results/runs` across 2 run(s).

## 1. Overview Comparison Table

| Run ID | Model | Adapter | Prompt Format | MILU Val Acc (%) | Flores en→mr (chrF++) | Flores mr→en (chrF++) | Held-out BPC | Script Purity (%) | EOS Rate (%) | IFEval Ground (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `base_qwen25_1p5b` | Qwen2.5-1.5B-Instruct-bnb-4bit | none | `chat` | 27.16 [24.20, 30.37] | 9.34 [9.05, 9.60] | 25.87 [25.17, 26.57] | 2.9586 [2.9964, 3.0486] | 90.77% | 37.41% | 8.57% |
| `v0_adapter_qwen25_1p5b` | Qwen2.5-1.5B-Instruct-bnb-4bit | marathi-qwen2.5-lora | `alpaca_v0` | 34.94 [31.85, 38.15] | 14.40 [14.00, 14.80] | 29.19 [28.18, 30.16] | 2.2872 [2.3176, 2.3706] | 100.00% | 67.40% | — |

## 2. Statistical Significance & Paired Tests

### Paired Comparison: `v0_adapter_qwen25_1p5b` vs `base_qwen25_1p5b`

| Benchmark / Metric | Baseline (base_qwen25_1p5b) | Target (v0_adapter_qwen25_1p5b) | Delta (Δ) | 95% CI of Δ | Significant? (p < 0.05) |
| --- | --- | --- | --- | --- | --- |
| **MILU Validation Acc** | 27.16% | 34.94% | **+7.78 pp** | [+4.07, +11.36] | ✅ Yes (McNemar p=0.0000) |
| **Flores en→mr (chrF++)** | 9.34 | 14.40 | **+5.06** | [+4.56, +5.65] | ✅ Yes (p=0.0000) |
| **Flores mr→en (chrF++)** | 25.87 | 29.19 | **+3.32** | [+2.47, +4.31] | ✅ Yes (p=0.0000) |
| **Held-out Flores BPC** | 3.0221 | 2.3430 | **-0.6790** | [-0.6948, -0.6641] | ✅ Yes (p=0.0000) |

## 3. Pre-Registered Model Ranking

### Pre-Registered Stage A Model Ranking (EVAL.md Protocol)

| Rank | Run ID | Mean Rank | MILU Rank | en→mr Rank | mr→en Rank | IFEval Ground (%) | Held-out BPC |
|---|---|---|---|---|---|---|---|
| **#1** | `v0_adapter_qwen25_1p5b` | **1.00** | #1 | #1 | #1 | 0.00% | 2.2872 |
| **#2** | `base_qwen25_1p5b` | **2.00** | #2 | #2 | #2 | 8.57% | 2.9586 |
