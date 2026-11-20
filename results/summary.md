# Marathi LLM Study — Benchmark Evaluation Summary

Generated automatically from `results/runs` across 11 run(s).

## 1. Overview Comparison Table

| Run ID | Model | Adapter | Prompt Format | MILU Val Acc (%) | Flores en→mr (chrF++) | Flores mr→en (chrF++) | Flores n | Held-out BPC | Script Purity (%) | EOS Rate (%) | IFEval Ground (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `A_qwen25_1p5b` | Qwen2.5-1.5B-Instruct-bnb-4bit | adapter | `chat` | 28.64 [25.68, 31.98] | 10.70 [10.40, 11.03] | 28.26 [27.48, 29.09] | 997 | 2.2342 | 100.00% | 37.11% | — |
| `A_qwen25_1p5b_768` | Qwen2.5-1.5B-Instruct-bnb-4bit | adapter | `chat` | 29.14 [25.93, 32.22] | 11.20 [10.86, 11.56] | 28.86 [27.91, 29.78] | 997 | 2.1724 | 100.00% | 42.43% | — |
| `B2_s1_indicalign_12k` | Qwen2.5-1.5B-Instruct-bnb-4bit | adapter | `chat` | 28.77 [25.43, 31.98] | 11.12 [10.52, 11.74] | 23.32 [21.29, 25.54] | 200 | 2.1860 | 100.00% | 16.50% | — |
| `B3_s2_sandlogic_12k` | Qwen2.5-1.5B-Instruct-bnb-4bit | adapter | `chat` | 30.12 [27.16, 33.33] | 10.69 [9.97, 11.39] | 29.67 [27.51, 31.79] | 200 | 2.1721 | 84.39% | 33.00% | — |
| `B4_s3_aya_1p2k` | Qwen2.5-1.5B-Instruct-bnb-4bit | adapter | `chat` | 28.77 [25.80, 31.85] | 7.62 [7.10, 8.15] | 22.98 [21.37, 24.70] | 200 | 2.6156 | 99.38% | 13.50% | — |
| `E3_s1_6k_cap512` | Qwen2.5-1.5B-Instruct-bnb-4bit | adapter | `chat` | 27.41 [24.44, 30.49] | 9.85 [9.29, 10.40] | 21.67 [19.69, 23.95] | 200 | 2.3245 | 99.69% | 23.00% | — |
| `E3_s1_6k_cap768` | Qwen2.5-1.5B-Instruct-bnb-4bit | adapter | `chat` | 27.41 [24.32, 30.37] | 9.92 [9.38, 10.43] | 24.76 [22.78, 27.10] | 200 | 2.2879 | 99.77% | 27.00% | — |
| `base_alpaca` | Qwen2.5-1.5B-Instruct-bnb-4bit | none | `alpaca_v0` | 26.54 [23.58, 29.63] | 10.59 [10.42, 10.76] | 12.48 [12.11, 12.85] | 997 | 2.8584 | 93.83% | 0.00% | — |
| `base_chat` | Qwen2.5-1.5B-Instruct-bnb-4bit | none | `chat` | 27.16 [24.20, 30.37] | 9.45 [9.16, 9.73] | 25.47 [24.71, 26.15] | 997 | 2.8584 | 91.86% | 36.61% | — |
| `v0_alpaca` | Qwen2.5-1.5B-Instruct-bnb-4bit | marathi-qwen2.5-lora | `alpaca_v0` | 35.06 [31.85, 38.40] | 14.44 [14.05, 14.84] | 29.33 [28.34, 30.29] | 997 | 2.1942 | 100.00% | 67.70% | — |
| `v0_chat` | Qwen2.5-1.5B-Instruct-bnb-4bit | marathi-qwen2.5-lora | `chat` | 33.33 [30.12, 36.54] | 14.39 [13.98, 14.78] | 29.69 [28.71, 30.70] | 997 | 2.1942 | 100.00% | 69.21% | — |
