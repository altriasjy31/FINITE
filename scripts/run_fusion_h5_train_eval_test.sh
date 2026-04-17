#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/data0/shaojiangyi/pprogo-flg-2/results/union_space_preds_only1"
OUT_ROOT="/data0/shaojiangyi/pprogo-flg-2/results/fusion_runs_1"

SCRIPT_TRAIN="train_fusion_project_then_fuse.py"
SCRIPT_INFER="infer_fusion.py"

EPOCHS=10
BATCH_SIZE=32
NUM_WORKERS=4

LR="1e-4"
OPT="adamw"
WEIGHT_DECAY="1e-2"

HIDDEN=256
DROPOUT=0.1
LATENT_TOKENS=3
HEAD=8
DEPTH=6
BP_QUERIES=512
MF_QUERIES=384
CC_QUERIES=256
GO_DATA_ROOT="/data0/shaojiangyi/pprogo-flg-2/data"
PROT_FEATURES="/data0/shaojiangyi/pprogo-flg-2/models/node_embedding"

# Explicit, shared critical flags (MUST be consistent between train and infer)
USE_LOGIT_FLAG="--use_logit_input"          # or "--no_logit_input"
# use distillation
# USE_KD_FLAG="--use_kd"
# KD_LAMBDA=0.3
USE_EMA_KD_FLAG="--use_ema_kd"
KD_LAMBDA=0.3

MIXUP_P=0.
MIXUP_ALPHA=0.2

LOSS="asl_opt"
GAMMA_NEG=4
GAMMA_POS=1
CLIP=0.05

PCT_START=0.1
DIV_FACTOR=25
FINAL_DIV_FACTOR=1000

DEVICE="cuda:4"
AMP="--amp"   # remove if not using AMP


# BASE_WEIGHTS_CC=(0.250 0.500 0.250)
# BASE_WEIGHTS_MF=(0.400 0.450 0.150)
# BASE_WEIGHTS_BP=(0.000 0.750 0.250)
BASE_WEIGHTS_CC=(0.3333 0.3333 0.3333)
BASE_WEIGHTS_MF=(0.3333 0.3333 0.3333)
BASE_WEIGHTS_BP=(0.3333 0.3333 0.3333)

# term gate
TERM_GATE_FLAG="--use_term_gate"     # or "--no_term_gate"
GATE_REG_LAMBDA_BP="0.10"
GATE_REG_LAMBDA_MF="0.05"
GATE_REG_LAMBDA_CC="0.05"
GATE_SCALE_INIT="1.0"


timestamp() { date +"%Y%m%d_%H%M%S"; }

run_one() {
  local go="$1"
  local train_h5="${DATA_ROOT}/${go}/train/${go}_train.h5"
  local valid_h5="${DATA_ROOT}/${go}/valid/${go}_valid.h5"
  local test_h5="${DATA_ROOT}/${go}/test/${go}_test.h5"
  local go_features="${GO_DATA_ROOT}/${go}/hgat_esm2_inductive_features/union_go_features.npy"
  # local go_features="${GO_DATA_ROOT}/${go}/hgat_esm2_inductive_features/union_go_wangsim.npy"
  local train_prot_features="${PROT_FEATURES}/${go}/${go}_train_protein_emb.npy"
  local valid_prot_features="${PROT_FEATURES}/${go}/${go}_valid_protein_emb.npy"
  local test_prot_features="${PROT_FEATURES}/${go}/${go}_test_protein_emb.npy"

  local train_go_features="${PROT_FEATURES}/${go}/${go}_train_go_emb.npy"
  local valid_go_features="${PROT_FEATURES}/${go}/${go}_valid_go_emb.npy"
  local test_go_features="${PROT_FEATURES}/${go}/${go}_test_go_emb.npy"

  [[ -f "${train_h5}" ]] || { echo "[ERROR] Missing ${train_h5}" >&2; exit 1; }
  [[ -f "${valid_h5}" ]] || { echo "[ERROR] Missing ${valid_h5}" >&2; exit 1; }
  [[ -f "${test_h5}"  ]] || { echo "[ERROR] Missing ${test_h5}"  >&2; exit 1; }

  local run_id="${go}_$(timestamp)"
  local run_dir="${OUT_ROOT}/${run_id}"
  mkdir -p "${run_dir}/ckpt" "${run_dir}/pred" "${run_dir}/logs"

  echo "============================================================"
  echo "[${go}] RUN: ${run_dir}"
  echo "  train: ${train_h5}"
  echo "  valid: ${valid_h5}"
  echo "  test : ${test_h5}"
  echo "============================================================"
  

  # ------------------------
  # Train (force blocking)
  # ------------------------
  local train_log="${run_dir}/logs/train_${go}.log"
  echo "[${go}] Start training ..."

  # Run python in foreground, but tee output; explicitly wait for python pid
  ( python "${SCRIPT_TRAIN}" \
      --backend h5 \
      --data_root "${DATA_ROOT}" \
      --go_name "${go}" \
      --train_h5 "${train_h5}" \
      --valid_h5 "${valid_h5}" \
      --save_dir "${run_dir}/ckpt" \
      --device "${DEVICE}" \
      ${AMP} \
      ${USE_LOGIT_FLAG} \
      ${USE_EMA_KD_FLAG} \
      --kd_lambda ${KD_LAMBDA} \
      --teacher_weights "${BASE_WEIGHTS[0]}" "${BASE_WEIGHTS[1]}" "${BASE_WEIGHTS[2]}" \
      --num_decode_queries ${NUM_QUERIES} \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --hidden_dim "${HIDDEN}" \
      --latent_tokens "${LATENT_TOKENS}" \
      --num_heads "${HEAD}" \
      --depth "${DEPTH}" \
      --dropout "${DROPOUT}" \
      --train_go_features "${train_go_features}" \
      --valid_go_features "${valid_go_features}" \
      --train_prot_features "${train_prot_features}" \
      --valid_prot_features "${valid_prot_features}" \
      --mixup_p "${MIXUP_P}" \
      --mixup_alpha "${MIXUP_ALPHA}" \
      --optimizer "${OPT}" \
      --lr "${LR}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --pct_start "${PCT_START}" \
      --div_factor "${DIV_FACTOR}" \
      --final_div_factor "${FINAL_DIV_FACTOR}" \
      --loss "${LOSS}" \
      --gate_scale_init "${GATE_SCALE_INIT}" \
      --gate_reg_lambda "${GATE_REG_LAMBDA}" \
      --gamma_neg "${GAMMA_NEG}" \
      --gamma_pos "${GAMMA_POS}" \
      --clip "${CLIP}" \
      --val_every 1 \
      2>&1 | tee "${train_log}"
  )
  # If python exits non-zero, set -e will stop here.

  echo "[${go}] Training finished."

  local best_ckpt="${run_dir}/ckpt/best_${go}.pt"
  if [[ ! -f "${best_ckpt}" ]]; then
    echo "[WARN] Best ckpt not found, fallback to last."
    best_ckpt="${run_dir}/ckpt/last_${go}.pt"
  fi
  [[ -f "${best_ckpt}" ]] || { echo "[ERROR] No checkpoint found for ${go}." >&2; exit 1; }

  # ------------------------
  # Test inference (after train completes)
  # ------------------------
  local infer_log="${run_dir}/logs/test_infer_${go}.log"
  echo "[${go}] Start test inference ..."

  python "${SCRIPT_INFER}" \
    --backend h5 \
    --hidden_dim "${HIDDEN}" \
    --latent_tokens "${LATENT_TOKENS}" \
    --num_heads "${HEAD}" \
    --depth "${DEPTH}" \
    --dropout "${DROPOUT}" \
    --test_go_features "${test_go_features}" \
    --test_prot_features "${test_prot_features}" \
    --go_name "${go}" \
    --train_h5 "${train_h5}" \
    --h5 "${test_h5}" \
    --ckpt "${best_ckpt}" \
    --out_dir "${run_dir}/pred" \
    --device "${DEVICE}" \
    ${AMP} \
    ${USE_LOGIT_FLAG} \
    --teacher_weights "${BASE_WEIGHTS[0]}" "${BASE_WEIGHTS[1]}" "${BASE_WEIGHTS[2]}" \
    --num_decode_queries ${NUM_QUERIES} \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    2>&1 | tee "${infer_log}"

  echo "[${go}] Completed."
  echo "  ckpt: ${run_dir}/ckpt"
  echo "  pred: ${run_dir}/pred"
  echo "  logs: ${run_dir}/logs"
}

mkdir -p "${OUT_ROOT}"

for go in bp cc mf; do
case "$go" in
  bp)
    GATE_REG_LAMBDA="${GATE_REG_LAMBDA_BP}"
    BASE_WEIGHTS=("${BASE_WEIGHTS_BP[@]}")
    NUM_QUERIES="${BP_QUERIES}"
    ;;
  mf)
    GATE_REG_LAMBDA="${GATE_REG_LAMBDA_MF}"
    BASE_WEIGHTS=("${BASE_WEIGHTS_MF[@]}")
    NUM_QUERIES="${MF_QUERIES}"
    ;;
  cc)
    GATE_REG_LAMBDA="${GATE_REG_LAMBDA_CC}"
    BASE_WEIGHTS=("${BASE_WEIGHTS_CC[@]}")
    NUM_QUERIES="${CC_QUERIES}"
    ;;
  *)
    echo "Error: invalid value for go='$go'. Expected one of: bp, mf, cc." >&2
    exit 1
    ;;
esac

  run_one "${go}"
done

echo "All done."