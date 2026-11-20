#!/bin/sh
set -eu
if [ "$#" -lt 1 ]; then
    printf '%s\n' 'Usage: sh start_memt_pro.sh MODEL_PATH [extra launcher arguments]' >&2
    exit 2
fi
model_path=$1
shift
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
exec "${MEMT_PYTHON:-python}" -m memt_sglang.launch \
    --model-path "$model_path" --port "${MEMT_PORT:-30000}" \
    --tp-size 8 --ep 8 --dp-size 1 --max-prefill-tokens 16384 \
    --state-directory "${MEMT_STATE_DIRECTORY:-./memt-state}" \
    --max-sessions-per-rank 6 --max-inflight-commits-per-rank 2 \
    --min-kv-tokens 2048 --max-total-tokens 65536 \
    --max-running-requests 8 --context-length 16384 \
    --mem-fraction-static 0.70 --dtype bfloat16 \
    --attention-backend flashinfer --page-size 1 \
    --mamba-ssm-dtype float32 --moe-runner-backend triton \
    --sampling-backend pytorch --disable-custom-all-reduce \
    --enable-streaming-session --stream-interval 1 "$@"
