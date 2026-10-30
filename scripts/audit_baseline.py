"""
scripts/audit_baseline.py — Phase 0: Baseline Audit and Clean Split Generation
-----------------------------------------------------------------------------
Performs:
1. Leakage audit: checks the 10 infer.py prompts against the filtered training split
2. Reference audit: checks evaluate.py references vs dataset ground truth
3. Validation loss audit: confirms how v0 eval loss was calculated
4. S0 deduplication and clean hash-split generation (test 1000, dev 500, train rest)
"""

import os
import json
import re
import hashlib
from datasets import load_dataset
from transformers import AutoTokenizer
from rapidfuzz import fuzz

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "rachittshah/alpaca-marahti"
MAX_SEQ_LEN = 512

PROMPTS = [
    "निरोगी राहण्यासाठी तीन टिप्स द्या.",
    "तीन मुख्य रंग कोणते आहेत?",
    "पाण्याचे महत्त्व सांगा.",
    "भारताची राजधानी कोणती आहे?",
    "सूर्यमाला म्हणजे काय?",
    "मित्राला पत्र कसे लिहावे?",
    "झाडे लावणे का महत्त्वाचे आहे?",
    "संगणक म्हणजे काय आणि त्याचा उपयोग काय?",
    "वेळेचे व्यवस्थापन कसे करावे?",
    "चांगल्या विद्यार्थ्याचे गुण कोणते असतात?",
]

REFERENCES = [
    "1. संतुलित आहार घ्या आणि भरपूर फळे आणि भाज्या खा. 2. नियमित व्यायाम करा. 3. पुरेशी झोप घ्या.",
    "लाल, निळा आणि पिवळा असे तीन प्राथमिक रंग आहेत.",
    "पाणी जीवनासाठी अत्यंत आवश्यक आहे. ते शरीरातील पेशींना कार्यक्षम ठेवते आणि पचनक्रिया सुधारते.",
    "भारताची राजधानी नवी दिल्ली आहे.",
    "सूर्यमाला म्हणजे सूर्य आणि त्याच्याभोवती फिरणारे ग्रह, उपग्रह, लघुग्रह आणि धूमकेतू यांचा समूह.",
    "प्रिय मित्र, मी तुला हे पत्र लिहित आहे कारण मला तुझ्याशी काही महत्त्वाच्या गोष्टी सांगायच्या आहेत.",
    "झाडे आपल्याला ऑक्सिजन देतात, पर्यावरण स्वच्छ ठेवतात आणि पाऊस आणण्यास मदत करतात.",
    "संगणक एक इलेक्ट्रॉनिक उपकरण आहे जे माहिती साठवते, प्रक्रिया करते आणि विविध कामांसाठी वापरले जाते.",
    "वेळेचे व्यवस्थापन करण्यासाठी दैनंदिन वेळापत्रक बनवा, प्राधान्यक्रम ठरवा आणि विलंब टाळा.",
    "चांगला विद्यार्थी नियमित अभ्यास करतो, शिक्षकांचा आदर करतो आणि जिज्ञासू वृत्ती ठेवतो.",
]

def normalize_text(text):
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def extract_instruction_and_response(translated_text):
    """
    In alpaca-marahti, translated_text format usually is:
    'खाली एक सूचना आहे... ### सूचना:\n...\n\n### प्रतिसाद:\n...'
    """
    inst = ""
    resp = ""
    if "### सूचना:" in translated_text and "### प्रतिसाद:" in translated_text:
        parts = translated_text.split("### प्रतिसाद:")
        resp = parts[1].strip() if len(parts) > 1 else ""
        inst_part = parts[0]
        if "### सूचना:" in inst_part:
            inst = inst_part.split("### सूचना:")[1].strip()
            # If there's an input section:
            if "### इनपुट:" in inst:
                inst = inst.split("### इनपुट:")[0].strip()
    return inst, resp

def main():
    print("1. Loading dataset:", DATASET_NAME)
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"Total rows in raw dataset: {len(ds):,}")
    print(f"Columns: {ds.column_names}")

    # Add row indices to track IDs
    rows = []
    for idx, item in enumerate(ds):
        row = dict(item)
        row["row_id"] = idx
        rows.append(row)

    print("\n2. Initializing tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    eos_token = tokenizer.eos_token or "<|im_end|>"

    # Reproduce v0 formatting and filtering
    print("\n3. Reproducing v0 filtering (length <= 512)...")
    v0_filtered_rows = []
    for row in rows:
        translated = (row.get("translated_text") or "").strip()
        text = f"{translated}{eos_token}"
        input_ids = tokenizer(text)["input_ids"]
        row["token_len"] = len(input_ids)
        row["v0_text"] = text
        inst, resp = extract_instruction_and_response(translated)
        row["extracted_inst"] = inst
        row["extracted_resp"] = resp
        if len(input_ids) <= MAX_SEQ_LEN:
            v0_filtered_rows.append(row)

    print(f"v0 Filtered count: {len(v0_filtered_rows):,} / {len(rows):,} "
          f"({(1 - len(v0_filtered_rows)/len(rows))*100:.2f}% dropped)")

    # Reproduce v0 train_test_split (95% train, 5% test, seed=42)
    # Datasets train_test_split uses random permutation with seed 42
    import numpy as np
    rng = np.random.default_rng(42)
    indices = np.arange(len(v0_filtered_rows))
    rng.shuffle(indices)
    test_size = int(np.ceil(0.05 * len(v0_filtered_rows)))
    test_idx = set(indices[:test_size])
    
    v0_train_rows = [r for i, r in enumerate(v0_filtered_rows) if i not in test_idx]
    v0_eval_rows = [r for i, r in enumerate(v0_filtered_rows) if i in test_idx]
    v0_train_ids = set(r["row_id"] for r in v0_train_rows)
    v0_eval_ids = set(r["row_id"] for r in v0_eval_rows)

    print(f"v0 Train split: {len(v0_train_rows):,}")
    print(f"v0 Eval split : {len(v0_eval_rows):,}")

    # --- Task 0.2 & 0.3: Leakage & Reference Audit ---
    print("\n" + "="*80)
    print("TASK 0.2 & 0.3: LEAKAGE & REFERENCE AUDIT")
    print("="*80)

    audit_results = []
    for p_idx, prompt in enumerate(PROMPTS):
        ref = REFERENCES[p_idx]
        norm_prompt = normalize_text(prompt)
        
        matches = []
        for r in rows:
            # Check against raw instruction, extracted_inst, and translated_text
            raw_inst = r.get("instruction") or ""
            ext_inst = r.get("extracted_inst") or ""
            trans_text = r.get("translated_text") or ""
            
            # Direct exact or normalized check
            ext_norm = normalize_text(ext_inst)
            ratio_ext = fuzz.ratio(norm_prompt, ext_norm)
            
            # Partial ratio or direct containment
            contains = prompt in trans_text or norm_prompt in normalize_text(trans_text)
            
            if ratio_ext >= 80 or contains:
                split_status = "dropped_by_length"
                if r["row_id"] in v0_train_ids:
                    split_status = "IN_V0_TRAIN"
                elif r["row_id"] in v0_eval_ids:
                    split_status = "IN_V0_EVAL"
                    
                matches.append({
                    "row_id": r["row_id"],
                    "split_status": split_status,
                    "token_len": r["token_len"],
                    "fuzz_ratio": ratio_ext,
                    "extracted_inst": ext_inst,
                    "extracted_resp": r.get("extracted_resp"),
                    "raw_output": r.get("output"),
                })
        
        # Sort matches by fuzz_ratio descending
        matches.sort(key=lambda x: x["fuzz_ratio"], reverse=True)
        audit_results.append({
            "idx": p_idx + 1,
            "prompt": prompt,
            "eval_reference": ref,
            "matches": matches
        })

    # Print summary of matches
    leak_count = 0
    for res in audit_results:
        print(f"\n[{res['idx']}] Prompt: {res['prompt']}")
        print(f"    Reference in evaluate.py: {res['eval_reference']}")
        if res["matches"]:
            top = res["matches"][0]
            print(f"    Matched Dataset Row ID: {top['row_id']} | Status: {top['split_status']} | Ratio: {top['fuzz_ratio']:.1f}%")
            print(f"    Dataset Extracted Inst: {top['extracted_inst'][:100]}")
            print(f"    Dataset Ground Truth  : {top['extracted_resp'][:120] if top['extracted_resp'] else 'N/A'}")
            if "TRAIN" in top['split_status']:
                leak_count += 1
        else:
            print("    NO match found in dataset (ratio >= 80% or substring).")

    print("\n" + "="*80)
    print(f"Total prompts leaked into training split: {leak_count} / {len(PROMPTS)}")
    print("="*80)

    # --- Task 0.4: Clean Deduplication and Splits for S0 ---
    print("\n" + "="*80)
    print("TASK 0.4: S0 DEDUPLICATION & DETERMINISTIC HASH SPLITS")
    print("="*80)

    # Deduplicate rows by normalized instruction
    seen_instructions = {}
    deduped_rows = []
    duplicate_count = 0

    for r in rows:
        inst = r.get("extracted_inst") or r.get("instruction") or ""
        norm_inst = normalize_text(inst)
        if not norm_inst:
            # Fallback to normalized translated text prefix if instruction was empty
            norm_inst = normalize_text(r.get("translated_text", "")[:100])
        
        if norm_inst in seen_instructions:
            duplicate_count += 1
        else:
            seen_instructions[norm_inst] = r["row_id"]
            deduped_rows.append(r)

    print(f"Total rows before deduplication: {len(rows):,}")
    print(f"Duplicates found and removed   : {duplicate_count:,}")
    print(f"Unique rows remaining           : {len(deduped_rows):,}")

    # Deterministic hash split
    # Sort by row_id to ensure exact determinism across platforms
    deduped_rows.sort(key=lambda x: x["row_id"])

    # Compute hash for each row based on row_id and normalized instruction
    hashed_items = []
    for r in deduped_rows:
        inst = r.get("extracted_inst") or r.get("instruction") or ""
        h = hashlib.sha256(f"s0_salt_{r['row_id']}_{normalize_text(inst)}".encode('utf-8')).hexdigest()
        hashed_items.append((h, r))

    # Sort deterministically by hash
    hashed_items.sort(key=lambda x: x[0])

    # Assign splits: test 1000, dev 500, train rest
    test_set = [item[1] for item in hashed_items[:1000]]
    dev_set  = [item[1] for item in hashed_items[1000:1500]]
    train_set = [item[1] for item in hashed_items[1500:]]

    test_ids = [r["row_id"] for r in test_set]
    dev_ids  = [r["row_id"] for r in dev_set]
    train_ids = [r["row_id"] for r in train_set]
    test_200_ids = test_ids[:200]  # Subset of 200 for Task 0.5 fast rescore

    # Verification: check for overlap
    s_test = set(test_ids)
    s_dev = set(dev_ids)
    s_train = set(train_ids)

    assert len(s_test.intersection(s_dev)) == 0, "Leakage: test and dev overlap!"
    assert len(s_test.intersection(s_train)) == 0, "Leakage: test and train overlap!"
    assert len(s_dev.intersection(s_train)) == 0, "Leakage: dev and train overlap!"
    print("\n✓ Split integrity verified: ZERO overlap between train, dev, and test sets.")
    print(f"  - Test set  : {len(test_ids):,} examples")
    print(f"  - Dev set   : {len(dev_ids):,} examples")
    print(f"  - Train set : {len(train_ids):,} examples")
    print(f"  - Test 200  : {len(test_200_ids)} examples (for Task 0.5)")

    # Save split manifests
    os.makedirs("data/splits", exist_ok=True)
    manifest_paths = {
        "data/splits/s0_test_ids.json": test_ids,
        "data/splits/s0_dev_ids.json": dev_ids,
        "data/splits/s0_train_ids.json": train_ids,
        "data/splits/s0_test_200.json": test_200_ids,
    }
    for p, ids in manifest_paths.items():
        with open(p, "w", encoding="utf-8") as f:
            json.dump({
                "source": "rachittshah/alpaca-marahti",
                "count": len(ids),
                "row_ids": ids
            }, f, indent=2)
        print(f"  Saved manifest -> {p}")

    # Also save the audit details to a json file to compile markdown report
    audit_data = {
        "dataset_name": DATASET_NAME,
        "total_raw": len(rows),
        "v0_filtered_count": len(v0_filtered_rows),
        "v0_train_count": len(v0_train_rows),
        "v0_eval_count": len(v0_eval_rows),
        "leak_count": leak_count,
        "prompts_audit": audit_results,
        "dedup_stats": {
            "total_before": len(rows),
            "duplicates_removed": duplicate_count,
            "unique_after": len(deduped_rows),
            "splits": {
                "test": len(test_ids),
                "dev": len(dev_ids),
                "train": len(train_ids),
            }
        }
    }
    os.makedirs("docs/legacy", exist_ok=True)
    with open("docs/legacy/audit_raw_results.json", "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    print("\nRaw audit results saved to docs/legacy/audit_raw_results.json")

if __name__ == "__main__":
    main()
