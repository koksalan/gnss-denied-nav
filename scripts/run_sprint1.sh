#!/usr/bin/env bash
# Sprint 1 experiment chain: pretrained baseline -> fine-tune -> fine-tuned eval -> LightGlue refinement.
# Split is by region (flights over the same area stay on the same side).
set -euo pipefail
PY=${PY:-python}
TRAIN=01,02,05,08,09,10
VAL=06
TEST=03,04,11
$PY scripts/evaluate.py --flights $TEST --tag test_pretrained
$PY scripts/train_finetune.py --train $TRAIN --val $VAL --epochs 15 --out outputs/finetune/s1
$PY scripts/evaluate.py --flights $TEST --weights outputs/finetune/s1/best.pt --tag test_finetuned
$PY scripts/evaluate.py --flights $TEST --weights outputs/finetune/s1/best.pt --rerank-k 10 --tag test_finetuned_lg
$PY scripts/evaluate.py --flights $TEST --rerank-k 10 --tag test_pretrained_lg
