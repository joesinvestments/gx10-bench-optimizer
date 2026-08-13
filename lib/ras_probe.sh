#!/usr/bin/env bash
# ras_probe.sh — single-shot NCCL RAS query across a TP group, model-agnostic.
#
# NCCL runs a RAS agent by default since 2.24, one localhost:28028 socket per rank
# (reachable directly with --network host, the common case for multi-node TP on this
# fleet; use SSH_PREFIX below if a rank isn't reachable that way). One query per node,
# sequential, never looped: repeated RAS polling has historical race reports against a
# live job. Proven against a live, confirmed-stuck GLM-5.2 wedge (see the GLM-5.2 repo's
# v027/MEMORY-AND-KERNEL-FINDINGS.md) before being generalized into this kit.
#
# Reading the output: a frozen AND mismatched per-rank operation count on a query run
# twice in a row is real cross-rank divergence. Frozen and equal, or a clean RUNNING/OK
# on every rank, means RAS isn't the layer that's stuck (check the workers directly,
# e.g. with py-spy, before concluding the job is healthy).
#
# Usage: NODES="gx10-1 gx10-2 gx10-3 gx10-4" ./ras_probe.sh
set -uo pipefail
NODES="${NODES:-gx10-1}"
RAS_PORT="${RAS_PORT:-28028}"
SSH_PREFIX="${SSH_PREFIX:-ssh -o BatchMode=yes -o ConnectTimeout=5}"

for n in $NODES; do
  echo "===== $n ====="
  $SSH_PREFIX "$n" "timeout 5 sh -c 'echo \"verbose status\" | nc -w 3 localhost $RAS_PORT' 2>&1" \
    || echo "[probe failed on $n]"
done
