"""
scripts/inspect_phase1_datasets.py — Inspect field names and splits of Phase 1 datasets
---------------------------------------------------------------------------------------
Checks:
1. google/IndicGenBench_flores_in
2. google/IndicGenBench_xquad_in
3. ai4bharat/MILU (Marathi)
"""

import os
import sys
from datasets import load_dataset, get_dataset_config_names

def inspect_dataset(name, config=None, split=None, **kwargs):
    print("\n" + "="*70)
    print(f"Inspecting: {name} (config={config}, split={split})")
    print("="*70)
    try:
        if config is None:
            try:
                configs = get_dataset_config_names(name)
                print(f"Available configs: {configs[:10]} (total: {len(configs)})")
                if "mr" in configs:
                    config = "mr"
                elif "marathi" in configs:
                    config = "marathi"
                elif configs:
                    config = configs[0]
            except Exception as ce:
                print(f"Could not list configs automatically: {ce}")
        
        ds = load_dataset(name, config, split=split, **kwargs)
        if hasattr(ds, "keys"):
            print(f"Available splits: {list(ds.keys())}")
            first_split = list(ds.keys())[0]
            sample = ds[first_split][0]
            print(f"Sample row from '{first_split}' (length: {len(ds[first_split])}):")
        else:
            sample = ds[0]
            print(f"Sample row (length: {len(ds)}):")
        
        print("Fields/Columns:")
        for k, v in sample.items():
            val_str = str(v)
            if len(val_str) > 80:
                val_str = val_str[:80] + "..."
            print(f"  - {k} ({type(v).__name__}): {val_str}")
        return True
    except Exception as e:
        print(f"Failed to inspect {name}: {e}")
        return False

def main():
    print("Starting inspection of Phase 1 benchmark datasets...")
    
    # 1. Flores-IN
    inspect_dataset("google/IndicGenBench_flores_in")
    
    # 2. XQuAD-IN
    inspect_dataset("google/IndicGenBench_xquad_in")
    
    # 3. MILU
    inspect_dataset("ai4bharat/MILU")

if __name__ == "__main__":
    main()
