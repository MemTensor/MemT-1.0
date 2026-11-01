#!/bin/sh
set -eu
if [ "$#" -lt 1 ]; then
    printf '%s\n' 'Usage: sh start_memt_nano.sh MODEL_PATH [extra launcher arguments]' >&2
    exit 2
fi
model_path=$1
shift
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
exec "${MEMT_PYTHON:-python}" -m memt_sglang.launch \
    --model-path "$model_path" \
    --port "${MEMT_PORT:-30000}" \
    --tp-size "${MEMT_TP_SIZE:-2}" --ep "${MEMT_EP_SIZE:-1}" --dp-size "${MEMT_DP_SIZE:-1}" --max-prefill-tokens 16384 \
    --state-directory "${MEMT_STATE_DIRECTORY:-./memt-state}" \
    --max-sessions-per-rank 6 --max-inflight-commits-per-rank 2 \
    --min-kv-tokens 2048 --max-total-tokens 65536 \
    --max-running-requests 8 --mem-fraction-static 0.60 \
    --dtype bfloat16 --attention-backend triton --sampling-backend pytorch \
    --linear-attn-backend triton --mamba-ssm-dtype float32 \
    --moe-runner-backend triton "$@"
