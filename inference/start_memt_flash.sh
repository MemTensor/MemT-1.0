#!/bin/sh
set -eu
if [ "$#" -lt 1 ]; then
    printf '%s\n' 'Usage: sh start_memt_flash.sh MODEL_PATH [extra launcher arguments]' >&2
    exit 2
fi
model_path=$1
shift
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
exec "${MEMT_PYTHON:-python}" -m memt_sglang.launch \
    --model-path "$model_path" --port "${MEMT_PORT:-30000}" \
    --tp-size "${MEMT_TP_SIZE:-8}" --ep "${MEMT_EP_SIZE:-8}" --dp-size "${MEMT_DP_SIZE:-8}" --max-prefill-tokens 16384 \
    --state-directory "${MEMT_STATE_DIRECTORY:-./memt-state}" \
    --max-sessions-per-rank 6 --max-inflight-commits-per-rank 1 \
    --min-kv-tokens 2048 --max-total-tokens 65536 \
    --max-running-requests 48 --context-length 16384 \
    --mem-fraction-static 0.70 --dtype bfloat16 \
    --kv-cache-dtype fp8_e4m3 --sampling-backend pytorch \
    --moe-runner-backend triton \
    --enable-streaming-session --stream-interval 1 "$@"
