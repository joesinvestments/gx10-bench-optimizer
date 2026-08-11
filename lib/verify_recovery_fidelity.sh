#!/usr/bin/env bash
# verify_recovery_fidelity.sh — would the AUTO-RECOVERY path reproduce what is RUNNING?
#
# ★★ WHY THIS EXISTS (2026-08-06). On this date the answer was NO, in four separate ways,
# and nothing on the machine could tell you:
#     live --max-model-len 393216          launcher could only express 128|256  -> 262144
#     live draft_sample_method probabilistic   launcher hardcoded greedy  (the old bug!)
#     live --override-generation-config ...    launcher passed no EXTRA_SERVE_ARGS at all
#     live --compilation-config FULL_DECODE_ONLY   same
# The watchdog would have restored a demoted cluster and pushed "🟢 RESTORED, no action
# needed". A recovery path that runs but silently downgrades is worse than one that fails,
# because failure is visible.
#
# The lesson this encodes is the house rule: THE FIX IS A CHECK, NOT A COMMENT. Correcting
# the values would have left the same hole open for the next divergence. This compares the
# two argv lists mechanically, so the next drift is caught the first time it is looked at.
#
# READ-ONLY. Runs launch_rank_tp4.sh under DRY_RUN=1 (which launches nothing) and diffs
# against `docker inspect .Args` of the live head container. Sends no inference, changes
# nothing, and is safe to run against a busy production server.
#
# USAGE
#   ./verify_recovery_fidelity.sh              # compare vs live, human output
#   ./verify_recovery_fidelity.sh --strict     # exit 1 on ANY divergence (for the watchdog)
#   BAND=384 ./verify_recovery_fidelity.sh     # test a band other than the launcher default
#
# EXIT  0 = recovery reproduces production   1 = divergence   2 = could not compare
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HOST="${DSFV4_SSH_HOST:-gx10-1}"
NAME="${DSFV4_CONTAINER:-deepseek-v4-0731-vllm-head}"
STRICT=0; [ "${1:-}" = "--strict" ] && STRICT=1

say() { printf '  %s\n' "$*"; }

# ── 1. what is ACTUALLY running ──────────────────────────────────────────────
LIVE_JSON=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" \
    "docker inspect $NAME --format '{{json .Args}}'" 2>/dev/null)
if [ -z "$LIVE_JSON" ]; then
    say "✖ cannot read live container $NAME on $HOST — nothing to compare against."
    say "  (this is exit 2, NOT a pass. An unreachable server must never read as 'fine'.)"
    exit 2
fi
LIVE=$(printf '%s' "$LIVE_JSON" | python3 -c '
import json,sys
a=json.load(sys.stdin)
# .Args for this container is: -lc <script> bash <serve argv...>
# Drop everything up to and including the literal "bash" sentinel to get the serve argv.
i=a.index("bash")+1 if "bash" in a else 0
print("\n".join(a[i:]))')

# ── 2. what the RECOVERY PATH would launch, resolved the same way ────────────
# Mirrors launch_tp4.sh exactly: same band default, same env, rank 0 (the head).
BAND="${BAND:-384}"
case "$BAND" in 128) MML=131072;; 256) MML=262144;; 320) MML=327680;; 384) MML=393216;;
  *) say "✖ unknown band $BAND"; exit 2;; esac
# ⚠ NOT ${VAR:-default} — bash ends the expansion at the first `}` and this default is JSON.
# Must match launch_tp4.sh's default byte-for-byte or this check compares against a fiction.
if [ -z "${EXTRA_SERVE_ARGS:-}" ]; then
  EXTRA_SERVE_ARGS='--override-generation-config {"temperature":0.0} --compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"} --enable-prompt-tokens-details'
fi

WOULD=$(DRY_RUN=1 RANK=0 MODE=dspark7 HOST_IP=192.168.100.11 MASTER_ADDR=192.168.100.11 \
    MASTER_PORT=25310 FABRIC_INTERFACE=enp1s0f0np0 RDMA_HCA=rocep1s0f0,roceP2p1s0f0 \
    MODEL_PATH=/srv/models/DeepSeek-V4-Flash-0731 SERVED_MODEL=deepseek-v4-flash-0731 \
    IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1 MAX_MODEL_LEN="$MML" MAX_NUM_SEQS=12 \
    MAX_NUM_BATCHED_TOKENS=8264 GRAPH_CAPTURE=96 GPU_MEMORY_UTILIZATION=0.85 \
    API_PORT=8000 NNODES=4 EXTRA_SERVE_ARGS="$EXTRA_SERVE_ARGS" \
    bash "$HERE/launch_rank_tp4.sh" 2>/dev/null)
if [ -z "$WOULD" ]; then
    say "✖ DRY_RUN produced nothing — launch_rank_tp4.sh has no DRY_RUN support, or it errored."
    say "  Empty input must ABORT, never silently pass. (See the 2026-08-04 preflight.sh bug.)"
    exit 2
fi

# ── 3. compare as KEY=VALUE, order-insensitive ───────────────────────────────
# Flags may legitimately move; VALUES may not. Comparing raw token order would produce
# noise on a harmless reorder and train everyone to ignore the check.
norm() { python3 -c '
import sys
toks=[l for l in sys.stdin.read().split("\n") if l != ""]
out, i = [], 0
while i < len(toks):
    t = toks[i]
    if t.startswith("--"):
        if i+1 < len(toks) and not toks[i+1].startswith("--"):
            out.append(f"{t}={toks[i+1]}"); i += 2; continue
        out.append(f"{t}=<set>"); i += 1; continue
    out.append(f"<positional>={t}"); i += 1
print("\n".join(sorted(out)))'; }

L=$(printf '%s\n' "$LIVE"  | norm)
W=$(printf '%s\n' "$WOULD" | norm)

if [ "$L" = "$W" ]; then
    say "✅ recovery path reproduces production exactly (band ${BAND}K, $(printf '%s\n' "$W" | wc -l | tr -d ' ') settings)."
    exit 0
fi

say "⚠ RECOVERY WOULD NOT REPRODUCE PRODUCTION — band ${BAND}K"
say ""
printf '  %-42s %-30s %s\n' "setting" "LIVE (running now)" "RECOVERY would launch"
printf '  %s\n' "$(printf '%.0s-' {1..110})"
diff <(printf '%s\n' "$L") <(printf '%s\n' "$W") >/dev/null 2>&1
python3 - "$L" "$W" <<'PY'
import sys
def d(s): return {k: v for k, _, v in (x.partition("=") for x in s.split("\n") if x)}
live, would = d(sys.argv[1]), d(sys.argv[2])
for k in sorted(set(live) | set(would)):
    a, b = live.get(k, "<ABSENT>"), would.get(k, "<ABSENT>")
    if a != b:
        print(f"  {k:<42.42} {a:<30.30} {b:.40}")
PY
say ""
say "Every row above is a setting the cluster would come back WITHOUT, or WITH a different"
say "value, after an unattended recovery — while the watchdog reports success."
exit $([ "$STRICT" = 1 ] && echo 1 || echo 1)
