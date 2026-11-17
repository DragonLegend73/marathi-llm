# Marathi-LLM — What's Needed to Finish (resume-project standard)

Goal: a repo and README that are honest, reproducible enough to rerun, and free of claims a reviewer could
poke holes in. Not publication-grade. Everything not listed here is optional.

---

## 1. Verdict on the two files you just sent

### `generate.py` change — no effect on any v0 number you have
The only edit is the **with-input** Alpaca preamble:

| | Text |
|---|---|
| Old | "...वर्णन करते, आणि त्यासोबत जोडलेले इनपुट अतिरिक्त संदर्भ पुरवते. ..." |
| New | "...वर्णन करते, पुढील संदर्भ प्रदान करणार्या इनपुटसह जोडली जाते. ..." |

The old text did not match the training data, so your agent corrected it. That matters only for prompts that
pass an `input`. MILU, Flores and XQuAD items have no `input` key, so they all use the **no-input** template,
which was not changed. BPC uses raw text with no template. So **all existing v0 results (MILU, Flores, BPC,
auto-metrics) are unaffected.** No need to explain this in the README beyond "template matched to training data".

One cheap safeguard, since the new string was copied from data that may contain invisible characters
(e.g. ZWJ in `करणार्या`). Add this test and move on:

```python
# tests/test_prompt_template.py
from datasets import load_dataset
from marathi_llm.generate import ALPACAV0_PROMPT_NO_INPUT, ALPACAV0_PROMPT_WITH_INPUT

def test_templates_match_training_data():
    ds = load_dataset("rachittshah/alpaca-marahti", split="train")
    no_in = with_in = None
    for row in ds:
        t = row["translated_text"]
        if "### इनपुट:" in t and with_in is None:
            with_in = t
        elif "### इनपुट:" not in t and no_in is None:
            no_in = t
        if no_in and with_in:
            break
    assert no_in.startswith(ALPACAV0_PROMPT_NO_INPUT.split("{")[0])
    assert with_in.startswith(ALPACAV0_PROMPT_WITH_INPUT.split("{")[0])
```

### Contamination audit — one hit, ignore it
Row 35,950 matched an IndicIFEval-trans riddle prompt at n = 8, and nothing matched at n = 12 or 20. The match
is inside the instruction/response (the shared preamble is in every row, so it cannot be the cause). It is one
row in 52,002 and does not touch Flores or MILU, which are the benchmarks you report for v0. **Do nothing
further.** In the README, say "no overlap detected with Flores or MILU" (not "verified out-of-distribution").

---

## 2. To-do (about one day)

### 2.1 BPC: one-line fix
In `src/marathi_llm/eval/bpc.py`, Qwen has no BOS token and the code falls back to `<|im_end|>` (151645).
Use `<|endoftext|>` (151643) instead:

```python
bos_id = tokenizer.bos_token_id
if bos_id is None:
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    bos_id = eot if eot is not None and eot != tokenizer.unk_token_id else tokenizer.eos_token_id
```

### 2.2 Four runs from one commit (this is the main fix)
This removes the biggest weakness in the current comparison: base ran under `chat`, v0 under `alpaca_v0`, on
different commits. Commit everything first, then do not touch code until all four finish. Use fresh run IDs
so the old runs stay on disk (re-running an existing run ID with `--force` wipes its metrics). Skip IFEval.

```bash
# runner.py lives under src/marathi_llm/eval/ (adjust if different)
python -m marathi_llm.eval.runner --run-id base_chat    --prompt-format chat      --skip-ifeval
python -m marathi_llm.eval.runner --run-id base_alpaca  --prompt-format alpaca_v0 --skip-ifeval
python -m marathi_llm.eval.runner --run-id v0_alpaca    --prompt-format alpaca_v0 --skip-ifeval \
    --adapter DragonLegend/marathi-qwen2.5-lora
python -m marathi_llm.eval.runner --run-id v0_chat      --prompt-format chat      --skip-ifeval \
    --adapter DragonLegend/marathi-qwen2.5-lora
```
`HF_HUB_OFFLINE=1` is the default in the runner, so the adapter must already be in the HF cache (or run once
with `HF_HUB_OFFLINE=0`). Check `adapter_loaded: true` in the two v0 `metrics.json` files.

How to read the result:
- **Headline comparison = same format:** `base_alpaca` vs `v0_alpaca`. That isolates the adapter.
- Secondary: `base_chat` vs `v0_chat` (what a user would see with the stock chat template).
- If `base_alpaca` is much better than `base_chat`, say the template, not the adapter, explains part of the
  old gap, and quote the smaller matched-format gap.

### 2.3 README (the part recruiters actually read)
Use wording like this and fill the table from the four `metrics.json` files.

**Results table (fill in):**

| Metric | Base (chat) | Base (alpaca) | v0 (alpaca) | v0 (chat) |
|---|---|---|---|---|
| MILU-Marathi val acc (%) | | | | |
| MILU predicted-letter spread (A/B/C/D %) | | | | |
| Flores en→mr chrF++ | | | | |
| Flores mr→en chrF++ | | | | |
| Held-out Marathi BPC (lower is better) | | | | |

**Caveats paragraph (copy and edit):**
> v0 is a pilot QLoRA adapter (Qwen2.5-1.5B-Instruct, trained on Alpaca-Marathi, ~52k machine-translated
> examples). Evaluation is zero-shot with greedy decoding: MILU (810 validation questions, next-token scoring
> over A–D), Flores-IN translation (chrF++, 997 sentences per direction) and held-out Marathi BPC. Intervals
> are bootstrap 95% CIs. MILU scores are close to chance and strongly affected by option-letter bias, so I
> report the predicted-letter distribution alongside accuracy. Generation-quality statistics (script purity,
> repetition, EOS rate) were measured on translation outputs only. Contamination checks found no overlap
> between the training data and Flores or MILU.

**Data pipeline paragraph (shows you audited your own work — keep it):**
> While auditing the data pipeline I found that word tokenization used `\w+`, which splits Devanagari words at
> vowel signs, so my "8-word" overlap check actually matched ~3-word spans and rejected ~11k rows
> (~22%) as false benchmark contamination. I fixed tokenization (shared `text.py` helper), widened the
> decontamination index to MILU val/test, Flores, XQuAD-IN, IndicIFEval and native prompts, and applied the
> 512-token length filter, which had silently not been running. Re-auditing the v0 training data with the
> fixed matcher found 1 overlapping row out of 52,002 (an IndicIFEval riddle prompt, none with Flores/MILU).
> Regression tests cover the tokenizer bug. The rebuilt 12k S0 set (30,509 clean rows after filtering) is used
> for later experiments; v0 itself was trained on the raw dataset.

(Check the numbers against your new `data/processed/s0_alpaca.stats.json` before pasting.)

---

## 3. Wording rules (so nothing can be called inflated)

| Say | Don't say |
|---|---|
| "no overlap detected with Flores/MILU" | "verified out-of-distribution", "zero leakage guaranteed" |
| "letter bias reduced (98.5% A → 49% A / 38% B)" | "MCQ bias resolved" |
| "gains on Flores and BPC, with 95% bootstrap CIs" | "statistically significant" with p-values from 50 resamples |
| "generation stats on translation outputs" | "natural stopping learned" (a third of v0 translations still hit the 256-token cap) |
| "pilot adapter" | "ranked #1 in a pre-registered study" (it is a two-model comparison) |
| "perplexity (token-level)" | "perplexity on BPC" |

Also drop the ranking table and the IFEval column from anything you show; IFEval exists only for base.

---

## 4. Explicitly skipped (fine for a resume project)

Vectorized chrF++ bootstrap, `git_dirty` tracking, p-value formatting, per-tag index counts, equal-token
budgets across sources, n = 12/20 audits, IFEval for v0, the pre-registered ranking, the new trainer
with response-only loss (only needed if you want to claim a data-source comparison; v0 + controls is enough).

---

## 5. Definition of done

- [ ] `tests/` pass, including the tokenizer regression test and the template test above
- [ ] BPC BOS fix committed
- [ ] Four runs completed from one commit, `adapter_loaded: true` for both v0 runs
- [ ] README table filled, caveats and pipeline-fix paragraphs added
- [ ] No claim in the README that the wording rules in section 3 forbid
