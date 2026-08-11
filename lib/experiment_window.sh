#!/usr/bin/env bash
# restart_window_20260806.sh — THE AUTHORIZED-RESTART WINDOW. Staged 2026-08-06.
#
# ⛔ DO NOT RUN WITHOUT JOE'S EXPLICIT GO. This takes production down for ~3 HOURS
#    (aggressive protocol, 2026-08-08: 8 boots + 3 probes each + soak).
#    Hermes must be idle first. The script refuses to start if requests are in flight.
#
# WHAT THIS WINDOW ANSWERS, in order of expected value:
#   A. Is k=7 costing us step time at production context?  jeffery2011.jc's TP=4 cluster
#      (NVIDIA forum 378878) steps at ~43 ms with k=3 at 150K; we step at 84.72 ms with
#      k=7. Every k decision we ever made was measured at SHORT prompts. Cells: k=7
#      (control, new base), k=5, k=3 — measured with client/longctx_probe.py at ~150K.
#   B. Does greedy drafting beat probabilistic at temperature 0?  At temp 0 vLLM routes to
#      rejection_greedy_sample_kernel (accept iff draft == target argmax); sampling the
#      draft is strictly dominated. Run on the k WINNER from phase A.
#   C. Safety base, riding along on every cell (validated inert by cell 1 reproducing
#      production numbers):
#        gmu 0.85 -> 0.80        bertholomus lost a NODE at 0.80/13-15GiB free; we run
#                                0.85 with 1.31 GiB free. KV pool peak use ever: 10%.
#        dual-rail NCCL_IB_HCA   queued since 2026-08-06, worth ~1%, correct on principle
#        NCCL_DEBUG=INFO         first-ever proof of what NCCL actually binds
#        UCX rcache safeguards   from bertholomus's stability doc; possibly inert for our
#                                stack (vLLM does not obviously use UCX) — harmless either way
#      NOT included: NCCL_IB_GID_INDEX change (0 -> RoCEv2 index). Separate restart, per
#      the 2026-08-06 briefing — never bundle it with the rail change or a regression
#      becomes unattributable.
#
# EXIT STATE: cluster left SERVING on the measured winner. Launcher defaults are NOT
# changed by this script — so verify_recovery_fidelity/the watchdog will correctly report
# "recovery would differ" until Joe blesses the winner and we pin it. That alert firing
# is the system working, not a bug.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HOME/Desktop/GX10-EVAL/window-$(date +%Y%m%d-%H%M%S)"
HOST=192.168.1.16
MODEL=deepseek-v4-flash-0731
SNAP="$HOME/.dsfv4-watchdog/scripts/accept_snapshot.py"

if [ "${1:-}" != "--authorized" ]; then
    echo "  ⛔ This restarts production. Run only with Joe's explicit go:"
    echo "     bash $0 --authorized"
    exit 2
fi

# ── 0. refuse to interrupt live work ─────────────────────────────────────────
BUSY=$(python3 - <<'PY'
import urllib.request
t=urllib.request.urlopen("http://192.168.1.16:8000/metrics",timeout=10).read().decode()
print(int(sum(float(l.rsplit(" ",1)[1]) for l in t.splitlines()
  if (l.startswith("vllm:num_requests_running") or l.startswith("vllm:num_requests_waiting"))
  and not l.startswith("#"))))
PY
) || BUSY=0
if [ "${BUSY:-0}" -gt 0 ]; then
    echo "  ✖ $BUSY request(s) in flight — Hermes is still working. Aborting untouched."
    exit 3
fi

mkdir -p "$OUT"
echo "  window -> $OUT"
echo "  pre-window production snapshot (the control everything is judged against):"
python3 "$SNAP" | tee "$OUT/pre-window-snapshot.txt"

# ── 1. disarm watchdog for the whole window ─────────────────────────────────
launchctl bootout "gui/$(id -u)/com.gigachad.dsfv4watchdog" 2>/dev/null
trap 'launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/com.gigachad.dsfv4watchdog.plist" 2>/dev/null; echo "  watchdog re-armed"' EXIT

# ── the new SAFETY BASE, common to every cell ───────────────────────────────
GMU=0.80
# --enable-prompt-tokens-details: info-only usage telemetry (per-request cached-token
# counts). MiaAI-Lab verified it works on our exact image (anemll 0.1.1) on 2026-08-01
# after a drop/revert cycle. Gives Hermes per-request cache visibility for free.
EXTRA='--override-generation-config {"temperature":0.0} --compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"} --enable-prompt-tokens-details'
boot_cell() {  # $1=label  $2=k  $3=capture  $4=ladder-json  $5=sample_method
    local label=$1 k=$2 cap=$3 ladder=$4 sm=$5
    echo "  ══ boot: $label (k=$k cap=$cap draft=$sm ladder=${ladder:-none} gmu=$GMU dual-rail) ══"
    # empty ladder = static k (omit the key entirely — ":}"-style truncation is invalid JSON)
    local spec
    if [ -n "$ladder" ]; then
        spec="{\"method\":\"dspark\",\"num_speculative_tokens\":$k,\"draft_sample_method\":\"$sm\",\"num_speculative_tokens_per_batch_size\":$ladder}"
    else
        spec="{\"method\":\"dspark\",\"num_speculative_tokens\":$k,\"draft_sample_method\":\"$sm\"}"
    fi
    local b64; b64=$(printf '%s' "$spec" | base64 | tr -d '\n')
    # preserve the outgoing cell's logs BEFORE rm -f destroys them — if the previous
    # probe wedged the engine, this tail is the crime scene we were denied on 08-07
    ssh -o BatchMode=yes gx10-1 'docker logs --tail 3000 deepseek-v4-0731-vllm-head 2>&1' \
        > "$OUT/prev-cell-head.log" 2>/dev/null
    for b in gx10-1 gx10-2 gx10-3 gx10-4; do
        ssh -o BatchMode=yes "$b" 'docker rm -f deepseek-v4-0731-vllm-head deepseek-v4-0731-vllm-worker >/dev/null 2>&1' || true
    done
    [ -s "$OUT/prev-cell-head.log" ] && mv "$OUT/prev-cell-head.log" "$OUT/${label}-PREV-head.log"
    # env string: EXTRA rides inside (export does NOT cross ssh). UCX + NCCL_DEBUG are new.
    local E="MODE=dspark7 FABRIC_INTERFACE=enp1s0f0np0 RDMA_HCA=rocep1s0f0,roceP2p1s0f0 \
MODEL_PATH=/srv/models/DeepSeek-V4-Flash-0731 SERVED_MODEL=$MODEL \
IMAGE=ghcr.io/anemll/dspark-vllm-gx10:0.1.1 MAX_MODEL_LEN=393216 MAX_NUM_SEQS=12 \
MAX_NUM_BATCHED_TOKENS=${MNBT:-8264} GRAPH_CAPTURE=$cap GPU_MEMORY_UTILIZATION=$GMU API_PORT=8000 \
MASTER_PORT=25310 MASTER_ADDR=192.168.100.11 NNODES=4 SPEC_CONFIG_B64=$b64 \
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
UCX_MEM_MMAP_HOOK_MODE=none UCX_RCACHE_MAX_UNRELEASED=1024 \
EXTRA_SERVE_ARGS='$EXTRA'"
    for spec2 in gx10-4:3:192.168.100.14 gx10-3:2:192.168.100.13 gx10-2:1:192.168.100.12; do
        local h=${spec2%%:*} rest=${spec2#*:}; local r=${rest%%:*} ip=${rest#*:}
        scp -q "$HERE/launch_rank_tp4.sh" "$h:/tmp/tp4.sh"
        ssh -o BatchMode=yes "$h" "$E RANK=$r HOST_IP=$ip bash /tmp/tp4.sh" >/dev/null 2>&1
    done
    sleep 5
    scp -q "$HERE/launch_rank_tp4.sh" gx10-1:/tmp/tp4.sh
    ssh -o BatchMode=yes gx10-1 "$E RANK=0 HOST_IP=192.168.100.11 bash /tmp/tp4.sh" >/dev/null 2>&1
    for i in $(seq 1 13); do
        [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://$HOST:8000/v1/models)" = "200" ] && return 0
        sleep 60
    done
    echo "    ✖ $label FAILED TO BOOT"; return 1
}

measure_cell() {  # $1=label
    local label=$1
    # boot facts + the NCCL binding proof we have never had
    ssh -o BatchMode=yes gx10-1 'docker logs deepseek-v4-0731-vllm-head 2>&1 | grep -iE "Available KV cache|GPU KV cache size|Maximum concurrency|Truncating"' > "$OUT/$label-boot.txt" 2>&1
    ssh -o BatchMode=yes gx10-1 'docker logs deepseek-v4-0731-vllm-head 2>&1 | grep -E "NCCL INFO NET|NCCL INFO Using|NCCL INFO Channel" | head -20' > "$OUT/$label-nccl.txt" 2>&1
    ssh -o BatchMode=yes gx10-1 'free -g | head -2' > "$OUT/$label-hostram.txt" 2>&1
    # ★ BOTH production regimes, measured on every cell (4-day traffic profile, 2026-08-08):
    #   A: deep  — 31% of boot-1 requests >200K ctx, C<=6.        150K, C=1.
    #   B: wide  — 81% of boot-3 requests <=2K ctx, C up to 11,   ~1.5K shared prefix,
    #      4x the request rate, and BOTH 08-07 wedges.            C=9 staggered.
    # A cell that wins A and errors/wedges under B loses: B is 80% of request volume.
    # B intentionally lands on seq counts 9-11 — the exact regime that wedged production.
    # If a cell wedges HERE, in the window, that is the cheapest possible place to learn it.
    python3 "$HERE/../client/longctx_probe.py" $HOST $MODEL \
        --tokens 150000 --runs 4 --label "$label-A" 2>&1 | tee "$OUT/$label-probeA.txt"
    python3 "$HERE/../client/longctx_probe.py" $HOST $MODEL \
        --tokens 1500 --runs 5 --max-tokens 150 --concurrency 12 \
        --label "$label-B" 2>&1 | tee "$OUT/$label-probeB.txt"
    # probe C — the wedge hunt: cold 150K prefill INSIDE a C=11 small-request storm.
    # Crosses every ladder rung and both eager holes with ragged mixed prefill+decode:
    # the exact condition of both 08-07 outages. errors>0 here disqualifies the cell.
    python3 "$HERE/../client/longctx_probe.py" $HOST $MODEL \
        --tokens 150000 --concurrency 11 --mixed \
        --label "$label-C" 2>&1 | tee "$OUT/$label-probeC.txt"
    sleep 10
    if ! curl -s -o /dev/null --max-time 8 "http://$HOST:8000/v1/models"; then
        echo "  ⚠ $label: endpoint unresponsive after probes — WEDGE REPRODUCED" | tee -a "$OUT/$label-probeC.txt"
    fi
}

# ── PHASE A: k cells, each measured under BOTH regimes ──────────────────────
declare -A RES
boot_cell "k7-base"   7 96 '[[1,4,7],[5,8,5],[9,12,3]]' probabilistic && measure_cell "k7-base"
# k8: the DSpark paper shows accepted length rising monotonically through draft len 16 at
# batch 128; our position-6 conditional is still 0.70. Depth has never been probed UPWARD.
# Capture 108 = 12 x 9. If the k-at-depth verify cost dominates, this loses to k5/k3 and
# that is a clean answer; if acceptance carries it, nobody else in the community has it.
boot_cell "k8"        8 108 '[[1,4,8],[5,8,5],[9,12,3]]' probabilistic && measure_cell "k8"
# ★ k7-static: SAME k, NO ladder — the wedge-suspect eliminator. Both 08-07 wedges hit at
# 9-11 running seqs: exactly where the ladder switches rungs (5-8 -> 9-12, first execution
# of rung 3 ever) AND where graph dispatch falls to eager (holes at 5 and 9). Static k=7
# is fully captured at every seq count 1-12 (96 = 12x8): no rung transitions, no holes.
# If k7-base errors under probe B and k7-static does not, the ladder is convicted.
boot_cell "k7-static" 7 96 '' probabilistic && measure_cell "k7-static"
boot_cell "k5"        5 72 '[[1,4,5],[5,8,5],[9,12,3]]' probabilistic && measure_cell "k5"
boot_cell "k3"        3 48 '[[1,4,3],[5,8,3],[9,12,3]]' probabilistic && measure_cell "k3"

# ── pick the winner: STABILITY FIRST, then speed ────────────────────────────
# Rule: any cell with errors under the regime-B probe (the concurrency profile that wedged
# production twice on 08-07) is DISQUALIFIED regardless of its regime-A speed. Among the
# survivors, rank by regime-A server decode tok/s. 80% of request volume is regime B, but
# B's requests are tiny — the token-hours live in regime A, so A is the speed metric and
# B is the gate. A fast config that wedges is not a candidate for anything.
pick_winner() {  # prints the winning label among ALL result files present right now
    # No pattern argument ON PURPOSE. Two earlier drafts filtered by a glob passed as a
    # parameter, and both were shell traps: bash expands an unquoted variable glob, zsh
    # does not (silent empty winner); zsh also matches case-patterns-from-variables
    # literally. Chronology makes filtering unnecessary: phase A calls this before the
    # greedy cell exists, the final call wants every cell anyway. Selection logic this
    # load-bearing gets the dumbest possible implementation.
    # all locals declared ONCE, at the top: zsh re-prints "name=value" when `local`
    # re-declares an existing variable mid-loop, which corrupts this function's stdout —
    # and its stdout IS the return value.
    local best="" bestv=0 lbl srv fb errs
    for fa in "$OUT"/*-probeA.txt; do
        [ -f "$fa" ] || continue
        lbl=$(grep -m1 $'^  MACHINE\t' "$fa" | cut -f2 | sed 's/-A$//')
        [ -n "$lbl" ] || continue
        grep -q INCOMPLETE "$fa" && continue
        srv=$(grep -m1 $'^  MACHINE\t' "$fa" | tr '\t' '\n' | grep '^server=' | cut -d= -f2)
        fb="$OUT/${lbl}-probeB.txt"
        errs=$(grep -m1 $'^  MACHINE\t' "$fb" 2>/dev/null | tr '\t' '\n' | grep '^errors=' | cut -d= -f2)
        # probe C (mixed storm / wedge hunt) disqualifies exactly like probe B —
        # a cell that survives the storm but errors the deep admit is not a candidate
        errsc=$(grep -m1 $'^  MACHINE\t' "$OUT/${lbl}-probeC.txt" 2>/dev/null | tr '\t' '\n' | grep '^errors=' | cut -d= -f2)
        if [ "${errs:-99}" != "0" ] || [ "${errsc:-99}" != "0" ]; then
            echo "    (disqualified: $lbl — B=${errs:-unmeasured} C=${errsc:-unmeasured} errors under load)" >&2
            continue
        fi
        awk -v a="${srv:-0}" -v b="$bestv" 'BEGIN{exit !(a>b)}' && { best=$lbl; bestv=$srv; }
    done
    echo "$best"
}
cell_params() {  # $1=label -> sets CK CCAP CLAD
    case "$1" in
      k3*)        CK=3; CCAP=48; CLAD='[[1,4,3],[5,8,3],[9,12,3]]';;
      k5*)        CK=5; CCAP=72; CLAD='[[1,4,5],[5,8,5],[9,12,3]]';;
      k7-static*) CK=7; CCAP=96; CLAD='';;
      *)          CK=7; CCAP=96; CLAD='[[1,4,7],[5,8,5],[9,12,3]]';;
    esac
}
WINNER=$(pick_winner)
WINNER="${WINNER:-k7-base}"
cell_params "$WINNER"; WK=$CK; WCAP=$CCAP; WLAD=$CLAD
echo "  ── phase A winner: $WINNER ──"

# ── PHASE B: greedy vs probabilistic ────────────────────────────────────────
boot_cell "k${WK}-greedy" "$WK" "$WCAP" "$WLAD" greedy && measure_cell "k${WK}-greedy"
# greedy interacts with k (deeper drafts amplify per-position gains), so also measure it
# on the static-k7 base unless that IS the winner — two greedy points, not one.
if [ "$WINNER" != "k7-static" ]; then
    boot_cell "k7s-greedy" 7 96 '' greedy && measure_cell "k7s-greedy"
fi

# ── PHASE B2: prefill chunk — the 20.26% of e2e nobody has ever tuned ───────
# Prefill computes at only ~1,306 tok/s effective. max_num_batched_tokens 8264 bounds the
# chunk; the r0b0tlab reference manifest runs 16384 on this hardware. One cell, winner's
# spec config, MNBT=16456 (16384 + 72 spec offset). Judged on probe A TTFT + probe C
# deep_ttft, not decode.
cell_params "$WINNER"
MNBT=16456 boot_cell "mnbt16k" "$CK" "$CCAP" "$CLAD" probabilistic && measure_cell "mnbt16k"

# ── LAND: leave serving on the best measured config (same stability-first rule) ──
BEST=$(pick_winner)
BEST="${BEST:-k7-base}"
case "$BEST" in *greedy*) FSM=greedy;; *) FSM=probabilistic;; esac
cell_params "$BEST"; FK=$CK; FCAP=$CCAP; FLAD=$CLAD
echo "  ── landing on: $BEST (k=$FK $FSM ladder=${FLAD:-none}) ──"
boot_cell "final-$BEST" "$FK" "$FCAP" "$FLAD" "$FSM"

# ── SOAK: the winner must survive three consecutive wedge-hunt waves ────────
# One clean probe C is a data point; three in a row on the landing config is a stability
# claim worth handing back to Hermes. Any failure here is loud and leaves the cluster
# serving (possibly wedged) with the evidence in $OUT — do NOT hand back without reading.
SOAK_FAIL=0
for w in 1 2 3; do
    python3 "$HERE/../client/longctx_probe.py" $HOST $MODEL \
        --tokens 150000 --concurrency 11 --mixed \
        --label "soak-$w" 2>&1 | tee "$OUT/soak-$w.txt" || SOAK_FAIL=1
    sleep 20
done
if [ "$SOAK_FAIL" = 1 ]; then
    echo "  🔴 SOAK FAILED on the landing config — read $OUT/soak-*.txt before handing back"
else
    echo "  ✅ soak: 3/3 mixed-storm waves clean on $BEST"
fi

echo
echo "  ══ WINDOW SUMMARY ══"
echo "  regime A (150K deep, C=1):"
printf '  %-14s %-9s %-9s %-9s %-9s %-9s\n' cell client server ms/step tok/step accept
grep -h $'^  MACHINE\t' "$OUT"/*-probeA.txt 2>/dev/null | while IFS=$'\t' read -r _ l p c s ms ts ac; do
    printf '  %-14s %-9s %-9s %-9s %-9s %-9s\n' "${l%-A}" "${c#client=}" "${s#server=}" "${ms#msstep=}" "${ts#tokstep=}" "${ac#accept=}"
done
echo
echo "  regime B (1.5K wide, C=9 staggered — the profile that wedged 08-07):"
printf '  %-14s %-11s %-11s %-9s\n' cell aggregate p50/stream errors
grep -h $'^  MACHINE\t' "$OUT"/*-probeB.txt 2>/dev/null | while IFS=$'\t' read -r _ l _ _ ag p5 er; do
    printf '  %-14s %-11s %-11s %-9s\n' "${l%-B}" "${ag#agg=}" "${p5#p50=}" "${er#errors=}"
done
echo
echo "  regime C (mixed storm — cold 150K prefill inside C=11 small-request traffic):"
printf '  %-14s %-10s %-11s %-9s\n' cell small_ok deep_ttft errors
grep -h $'^  MACHINE\t' "$OUT"/*-probeC.txt "$OUT"/soak-*.txt 2>/dev/null | while IFS=$'\t' read -r _ l _ _ so dt er; do
    printf '  %-14s %-10s %-11s %-9s\n' "${l%-C}" "${so#small_ok=}" "${dt#deep_ttft=}" "${er#errors=}"
done
echo
echo "  production control was: 57.90 tok/s · 84.72 ms/step · 4.906 tok/step (see pre-window-snapshot.txt)"
echo "  jeffery2011.jc's TP=4:  ~90 tok/s · ~43 ms/step · k=3 · 150K (forum 378878)"
echo
echo "  ⚠ LAUNCHER DEFAULTS UNCHANGED. If the winner differs from k=7/probabilistic, the"
echo "    watchdog will now correctly report 'recovery would differ'. To pin the winner:"
echo "    edit launch_rank_tp4.sh SPEC_CONFIG + launch_tp4.sh GRAPH_CAPTURE/GMU, re-sync"
echo "    to ~/.dsfv4-watchdog/scripts/, and re-run verify_recovery_fidelity.sh."
echo "  raw: $OUT"
