#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-python}
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

run_pids_candidate() {
  local TAG=$1 DATASET=$2 SEED=$3 PATCH=$4
  shift 4
  local EXTRA=("$@")
  local STAGE1_ROOT="runs_final_ablations/${TAG}/pids_stage1"
  local STAGE2_ROOT="runs_final_ablations/${TAG}/pids_stage2"
  local SELECT_PATH="runs_final_ablations/${TAG}/selection/${DATASET}/seed_${SEED}/selection.json"
  local PREFIX="ablation_${TAG}_${DATASET}_seed${SEED}"

  if [[ ! -f "${STAGE1_ROOT}/${DATASET}/pids/seed_${SEED}/metrics.json" ]]; then
    "$PY" src/train.py \
      --data-root data/raw --output-root "$STAGE1_ROOT" \
      --dataset "$DATASET" --variant pids --seed "$SEED" --epochs 100 \
      --batch-size 128 --eval-batch-size 512 --workers 4 --patch-size "$PATCH" \
      --backbone ccf --learning-rate 3e-4 --weight-decay 1e-4 \
      --validation-interval 5 --patience 35 --evaluation-draws 10 \
      --selection-objective robust "${EXTRA[@]}" \
      > "${PREFIX}_stage1.log" 2>&1
  fi

  if [[ ! -f "${STAGE2_ROOT}/${DATASET}/pids/seed_${SEED}/metrics.json" ]]; then
    "$PY" src/train.py \
      --data-root data/raw --output-root "$STAGE2_ROOT" \
      --dataset "$DATASET" --variant pids --seed "$SEED" --epochs 80 \
      --batch-size 128 --eval-batch-size 512 --workers 4 --patch-size "$PATCH" \
      --backbone ccf --learning-rate 1e-4 --weight-decay 1e-4 \
      --initial-run "${STAGE1_ROOT}/${DATASET}/pids/seed_${SEED}" \
      --validation-interval 5 --patience 35 --evaluation-draws 10 \
      --selection-objective robust "${EXTRA[@]}" \
      > "${PREFIX}_stage2.log" 2>&1
  fi

  if [[ ! -f "$SELECT_PATH" ]]; then
    "$PY" src/select_stage.py \
      --stage1 "${STAGE1_ROOT}/${DATASET}/pids/seed_${SEED}" \
      --stage2 "${STAGE2_ROOT}/${DATASET}/pids/seed_${SEED}" \
      --output "$SELECT_PATH" \
      > "${PREFIX}_selection.log" 2>&1
  fi
}

for DATASET in IndianPines PaviaU Salinas; do
  for SEED in 23 37 51 71 101; do
    run_pids_candidate uniform_risk "$DATASET" "$SEED" 11 \
      --training-compound --group-mixture 0.0
    run_pids_candidate random_subsets "$DATASET" "$SEED" 11
    run_pids_candidate spectral_only "$DATASET" "$SEED" 11 \
      --training-compound --spectral-only
    run_pids_candidate patch7 "$DATASET" "$SEED" 7 \
      --training-compound
    run_pids_candidate patch15 "$DATASET" "$SEED" 15 \
      --training-compound

    CLEAN_SPECTRAL="runs_final_ablations/spectral_only/clean_stage"
    if [[ ! -f "${CLEAN_SPECTRAL}/${DATASET}/clean/seed_${SEED}/metrics.json" ]]; then
      "$PY" src/train.py \
        --data-root data/raw --output-root "$CLEAN_SPECTRAL" \
        --dataset "$DATASET" --variant clean --seed "$SEED" --epochs 100 \
        --batch-size 128 --eval-batch-size 512 --workers 4 --patch-size 11 \
        --backbone ccf --spectral-only --learning-rate 3e-4 --weight-decay 1e-4 \
        --validation-interval 5 --patience 35 --evaluation-draws 10 \
        --selection-objective clean \
        > "ablation_spectral_only_${DATASET}_seed${SEED}_clean.log" 2>&1
    fi
  done
done

