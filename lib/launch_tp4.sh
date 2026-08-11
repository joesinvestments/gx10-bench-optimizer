#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash-0731 at TP=4 across all four Sparks.
#   ./launch_tp4.sh          -> 256K  <-- DEFAULT = THE STANDING PRODUCTION CONFIG
#   ./launch_tp4.sh 128      -> --max-model-len 131072  (only if you have a specific reason)
#
# 256K IS REQUIRED, NOT A PREFERENCE: measured Hermes traffic has median session-mean
# prompt 87.6K and 26.35% of sessions AVERAGE more than 128K (only 0.35% exceed 256K).
# 128K imposes real compression/segmentation pressure. See hermes-real-workload-profile.
# (An older header here claimed ">=8 resident 131K docs / >=4 resident 262K docs" and
#  "pick by single-request length" — those came from a document-rotation benchmark that
#  answers a question this workload does not ask. Ignore them.)
#
# Standing config baked in below: max_num_seqs 12, capture 96, max_num_batched_tokens 8264
# (= 8192 + 6*seqs, offsetting spec-decode drafting), gmu 0.85, and DYNAMIC batch-aware
# speculative decoding [[1,4,7],[5,8,5],[9,12,3]] which lives in launch_rank_tp4.sh.
set -uo pipefail
# ★★ 384 IS THE STANDING BAND AS OF 2026-08-06, AND THE 384 ARM DID NOT EXIST UNTIL TODAY.
# Production has run --max-model-len 393216 since 2026-08-05T19:37Z. This script accepted
# only 128|256, and the watchdog's auto-recovery calls it — so any unattended recovery would
# have silently DEMOTED the cluster to 262144 and then reported "SUCCESS". Two other flags
# were dropped the same way (see EXTRA_SERVE_ARGS below).
#   393216 is required, not a preference: reasoning_effort "max" needs >= 393216, and 34.4%
#   of live requests carry prompts above 200K tokens against this ceiling.
BAND="${1:-384}"   # 384K = 393216 is the standing production band
case "$BAND" in
  128) MML=131072;;
  256) MML=262144;;
  320) MML=327680;;
  384) MML=393216;;
  *) echo "usage: $0 [128|256|320|384]   (384 = standing production)" >&2; exit 2;;
esac
RANK_SCRIPT="$(cd "$(dirname "$0")" && pwd)/launch_rank_tp4.sh"

# ★ THESE TWO FLAGS ARE PART OF THE PRODUCTION CONFIG AND USED TO BE INVISIBLE HERE.
# They are passed to `vllm serve` via launch_rank_tp4.sh's EXTRA_SERVE_ARGS passthrough.
# Because this script never set it, every recovery launched WITHOUT them.
#   --override-generation-config {"temperature":0.0}
#       ⚠ OPEN QUESTION, not a settled good. It defeats the checkpoint's own
#       generation_config.json (temperature 1.0, top_p 1.0) which is the distribution the
#       DSpark drafter was fitted at, and at temperature 0 vLLM routes to
#       rejection_greedy_sample_kernel where a probabilistic draft is strictly wasted work.
#       It is pinned here to REPRODUCE production, not to endorse it. Scheduled A/B.
#   --compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}
#       Verified in force; capture 96 accepted with zero truncation (0 "Truncating" lines
#       in 49,034 log lines). FULL_AND_PIECEWISE is NOT ours — an older echo said so.
# ⚠ `export` does NOT cross ssh. The value must ride inside the env string handed to the
# remote shell (see $E below), single-quoted there, exactly as restart_acceptance_fixes.sh
# does it. Getting this wrong is silent: the flags simply vanish and the server still boots.
# ⚠ DO NOT write this as ${EXTRA_SERVE_ARGS:-<default>}. Bash parameter expansion ends at the
# first unescaped `}`, and the default is JSON full of them — so the value silently truncates
# to `{"temperature":0.0` and the rest becomes literal junk. That is exactly what happened
# when this was first written today; verify_recovery_fidelity.sh caught it on its first run.
if [ -z "${EXTRA_SERVE_ARGS:-}" ]; then
  EXTRA_SERVE_ARGS='--override-generation-config {"temperature":0.0} --compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"} --enable-prompt-tokens-details'
fi
# ★ DUAL RAIL 2026-08-06. Ports A and B are SEPARATE PCIe functions with SEPARATE GID
# tables (rocep1s0f0 / roceP2p1s0f0); naming only one pinned ALL traffic to rail A.
# Both rails verified ACTIVE with IPs on different /24s on all 4 nodes before this
# became the default. Expected gain is SMALL and honestly computed: decode collectives
# are 64 KiB and latency-bound (bandwidth term 0.41ms of a 48.83ms step -> ~0.4%);
# prefill collectives are large but 94.88% prefix-cache hit means only ~6,793 new
# tokens are actually prefilled. Under 1% end-to-end. Correct on principle, not a win.
E="MODE=dspark7 FABRIC_INTERFACE=enp1s0f0np0 RDMA_HCA=rocep1s0f0,roceP2p1s0f0 \
MODEL_PATH=/srv/models/DeepSeek-V4-Flash-0731 SERVED_MODEL=deepseek-v4-flash-0731 \
IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1 MAX_MODEL_LEN=$MML MAX_NUM_SEQS=12 \
MAX_NUM_BATCHED_TOKENS=8264 GRAPH_CAPTURE=96 GPU_MEMORY_UTILIZATION=0.85 API_PORT=8000 \
MASTER_PORT=25310 MASTER_ADDR=192.168.100.11 NNODES=4 \
EXTRA_SERVE_ARGS='$EXTRA_SERVE_ARGS'"
for b in gx10-1 gx10-2 gx10-3 gx10-4; do
  ssh -o BatchMode=yes "$b" 'docker rm -f deepseek-v4-0731-vllm-head deepseek-v4-0731-vllm-worker >/dev/null 2>&1' || true
done
# WORKERS FIRST (ranks 1-3), head last — the author's ordering, and it matters
for spec in gx10-4:3:192.168.100.14 gx10-3:2:192.168.100.13 gx10-2:1:192.168.100.12; do
  h=${spec%%:*}; rest=${spec#*:}; r=${rest%%:*}; ip=${rest#*:}
  scp -q "$RANK_SCRIPT" "$h:/tmp/tp4.sh"
  ssh -o BatchMode=yes "$h" "$E RANK=$r HOST_IP=$ip bash /tmp/tp4.sh" >/dev/null 2>&1
  echo "  rank $r -> $h"
done
sleep 5
scp -q "$RANK_SCRIPT" gx10-1:/tmp/tp4.sh
ssh -o BatchMode=yes gx10-1 "$E RANK=0 HOST_IP=192.168.100.11 bash /tmp/tp4.sh" >/dev/null 2>&1
echo "  rank 0 -> gx10-1 (head)"
for i in $(seq 1 12); do
  c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://192.168.1.16:8000/v1/models)
  echo "  [${i}m] $c"; [ "$c" = "200" ] && { echo "  SERVING: http://192.168.1.16:8000/v1 (${BAND}K, TP=4)"; exit 0; }
  sleep 60
done
echo "  TIMEOUT — check: docker logs deepseek-v4-0731-vllm-head on gx10-1" >&2; exit 1
