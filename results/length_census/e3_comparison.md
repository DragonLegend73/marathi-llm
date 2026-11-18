# E3 Paired Comparison: `E3_s1_6k_cap768` vs `E3_s1_6k_cap512` (S1 IndicAlign, 6k, 750 steps, seed 0)

**Pre-Registered E3 Verdict**: **ADOPT 768**

| Metric | cap=512 | cap=768 | Paired Δ (768 − 512) | Paired 95% CI | Crosses 0? |
|---|---:|---:|---:|---|---|
| **MILU val acc (%)** *(n=810)* | 27.41 | 27.41 | +0.00 pp | [-1.48, +1.48] | Yes *(McNemar p=1.0)* |
| **Flores en→mr chrF++** *(n=200)* | 9.85 | 9.92 | +0.07 | [-0.63, +0.77] | Yes |
| **Flores mr→en chrF++** *(n=200)* | 21.67 | 24.76 | +3.09 | [+0.76, +5.23] | **No** |
| **Held-out Marathi BPC** *(n=997, lower better)* | 2.3245 | 2.2879 | -0.0366 | [-0.0408, -0.0323] | **No** |

### Arm Composition & Token Budget (chat-template counter)
- **cap=512**: `n=6,000`, mean=`244.4` tok (~`1.47M` total tok), rows >512: `0` (`0.0%`), sub-configs: `{'S1_Dolly_T': 1766, 'S1_HHRLHF_T': 971, 'S1_OpenAssistant_T': 696, 'S1_WikiHow': 7, 'S1_Wiki_Chat': 267, 'S1_Wiki_Conv': 2293}`
- **cap=768**: `n=6,000`, mean=`320.2` tok (~`1.92M` total tok), rows >512: `1222` (`20.4%`), sub-configs: `{'S1_Dolly_T': 1602, 'S1_HHRLHF_T': 1307, 'S1_OpenAssistant_T': 807, 'S1_WikiHow': 41, 'S1_Wiki_Chat': 374, 'S1_Wiki_Conv': 1869}`

*Note: The 768 arm sees more tokens and a shifted S1 sub-corpus mix under equal examples (6,000 rows), so any observed difference reflects longer rows allowed plus more tokens and sub-corpus shift, not sequence length alone.*

### Translation Auto-Metrics (do not read EOS-rate shifts as quality changes)
- **cap=512**: script_purity=99.69%, lang_id_mr=96.50%, repeat_4gram=71.50%, eos_rate=23.00%, mean_len=218.8 tok
- **cap=768**: script_purity=99.77%, lang_id_mr=96.50%, repeat_4gram=70.50%, eos_rate=27.00%, mean_len=208.4 tok
