#!/bin/sh
set -eu
if [ "$#" -lt 1 ]; then
    printf '%s\n' 'Usage: sh start_memt_flash.sh MODEL_PATH [extra launcher arguments]' >&2
    exit 2
fi
model_path=$1
shift
# Flash needs attention TP1, so tp == dp == ep and all three move together.
# The topology is read from the environment rather than hardcoded to the
# reference 8-GPU layout: --dp-size given twice is rejected, and a layout that
# does not match the visible devices fails deep inside NCCL init.
tp=${MEMT_TP_SIZE:-8}
ep=${MEMT_EP_SIZE:-$tp}
dp=${MEMT_DP_SIZE:-$tp}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
exec "${MEMT_PYTHON:-python}" -m memt_sglang.launch \
    --model-path "$model_path" --port "${MEMT_PORT:-30000}" \
    --tp-size "$tp" --ep "$ep" --dp-size "$dp" --max-prefill-tokens 16384 \
    --state-directory "${MEMT_STATE_DIRECTORY:-./memt-state}" \
    --max-sessions-per-rank 6 --max-inflight-commits-per-rank 1 \
    --min-kv-tokens 2048 --max-total-tokens 65536 \
    --max-running-requests 48 --context-length 16384 \
    --mem-fraction-static 0.70 --dtype bfloat16 \
    --kv-cache-dtype fp8_e4m3 --sampling-backend pytorch \
    --moe-runner-backend triton \
    --enable-streaming-session --stream-interval 1 \
    ${MEMT_CUSTOM_ALL_REDUCE_DISABLED:+--disable-custom-all-reduce} "$@"
