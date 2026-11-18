# E2 Worst-Case VRAM & Throughput Probe (`Qwen2.5-1.5B-Instruct`)

| Cap | Config | Status | Peak Alloc (GB) | Peak Reserved (GB) | Steps/s | Tokens/s | Temp (°C) | ETA S0 (min) | ETA S1 (min) | ETA S2 (min) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | b2×a4 | OK | 1.386 | 1.422 | 0.198 | 768.8 | 58.5 | 66.4 | 67.0 | 77.5 |
| 768 | b2×a4 | OK | 1.429 | 1.463 | 0.141 | 824.2 | 58.2 | 79.0 | 82.7 | 103.5 |
| 1024 | b2×a4 | OK | 1.469 | 1.504 | 0.109 | 842.5 | 59.2 | 82.9 | 92.7 | 124.6 |
| 1536 | b2×a4 | OK | 1.553 | 1.598 | 0.073 | 846.1 | 58.2 | 86.0 | 122.8 | 153.2 |
| 2048 | b2×a4 | OK | 1.640 | 1.695 | 0.054 | 844.9 | 58.2 | 87.2 | 150.2 | 174.7 |

## E4 Gate Evaluation (Candidates: 512, 768, 1024)

| Candidate | Hardware (≤3.4 GB res) | Time (S0/S1/S2 ETA ≤150m) | Benefit (min S1/S2 Δ ≥ +15 pp) | Verdict |
|---:|---|---|---|---|
| 512 | PASS (1.42 GB) | PASS (66m / 67m / 78m) | BASELINE (28.6%, +0.0 pp) | **PASS** |
| 768 | PASS (1.46 GB) | PASS (79m / 83m / 104m) | PASS (45.8%, +17.2 pp) | **PASS** |
| 1024 | PASS (1.50 GB) | PASS (83m / 93m / 125m) | PASS (58.3%, +29.7 pp) | **PASS** |
