#!/usr/bin/env bash
# Clone and setup IndicIFEval at the exact pinned commit with verification mode patch applied.
set -euo pipefail

PINNED_COMMIT="1bb5f1bc4936cb20b8544e185bfd7cbed8f31464"
TARGET_DIR="IndicIFEval"
PATCH_FILE="scripts/patches/indicifeval_verification_mode.patch"

if [ -d "$TARGET_DIR" ]; then
    echo "[$TARGET_DIR] Directory already exists. Verifying commit..."
    git -C "$TARGET_DIR" rev-parse HEAD
else
    echo "Cloning ai4bharat/IndicIFEval..."
    git clone https://github.com/ai4bharat/IndicIFEval "$TARGET_DIR"
    git -C "$TARGET_DIR" checkout "$PINNED_COMMIT"
    echo "Applying verification mode patch..."
    git -C "$TARGET_DIR" apply "$(pwd)/$PATCH_FILE"
    echo "IndicIFEval successfully configured at commit $PINNED_COMMIT"
fi
