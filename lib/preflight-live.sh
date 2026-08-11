#!/usr/bin/env bash
# preflight-live.sh — run the fidelity gate against a RUNNING container.
#
#   ./preflight-live.sh [host] [profile]
#       host     default gx10-1
#       profile  default production_profile
#
# Read-only: docker inspect on the container and on its image, then preflight.sh.
# The image inspect matters — .Config.Env on a CONTAINER returns the image's baked-in
# env merged with the launcher's -e flags. Without subtracting the image baseline this
# reports 34 deviations instead of the real 8, and the author's own image env (PATH,
# NVARCH, UV_*, VLLM_BUILD_*) gets blamed on us.
#
# Manifest-declared keys are still judged on the EFFECTIVE value (image default merged
# with -e override) — that is how `use_breakable_cudagraph` was caught: his image bakes
# VLLM_USE_BREAKABLE_CUDAGRAPH=1 while his manifest declares false.
set -uo pipefail
HOST="${1:-gx10-1}"
PROFILE="${2:-production_profile}"
B="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="${MANIFEST:-$B/recipe/r0b0tlab/runtime-manifest.vllm-sm121.3.json}"
APPROVED="${APPROVED:-$B/recipe/approved-deviations.txt}"

[[ -f "$MANIFEST" ]] || { echo "preflight-live: manifest not found: $MANIFEST" >&2
                          echo "  (expected the vendored copy at recipe/r0b0tlab/)" >&2; exit 2; }

# ★ THIS FILTER WAS `name=dsfv4` UNTIL 2026-08-06 AND MATCHED ZERO CONTAINERS ON ALL FOUR
# NODES. The container is named deepseek-v4-0731-vllm-head; "dsfv4" is our shorthand for the
# model, never a container name. So this gate has NEVER produced a pass for this deployment —
# it exited 2 every time, which reads as "could not check", not "checked and fine". It failed
# closed, which is the only reason it was harmless. Overridable via DSFV4_CONTAINER.
CID=$(ssh -o BatchMode=yes "$HOST" "docker ps -q --filter name=${DSFV4_CONTAINER:-deepseek-v4-0731-vllm-head} | head -1")
[[ -z "$CID" ]] && { echo "preflight-live: no running dsfv4 container on $HOST" >&2; exit 2; }

TMP=$(mktemp -d -t preflight-live); trap 'rm -rf "$TMP"' EXIT
ssh -o BatchMode=yes "$HOST" "docker inspect $CID --format '{{json .Config.Env}}'" > "$TMP/env.json"
ssh -o BatchMode=yes "$HOST" "docker inspect $CID --format '{{json .Args}}'"       > "$TMP/args.json"
IMG=$(ssh -o BatchMode=yes "$HOST" "docker inspect $CID --format '{{.Config.Image}}'")
ssh -o BatchMode=yes "$HOST" "docker inspect '$IMG' --format '{{range .Config.Env}}{{println .}}{{end}}'" > "$TMP/imgenv.txt"

python3 - "$TMP" "$IMG" <<'PY' > "$TMP/argv.txt"
import json, sys
t, img = sys.argv[1], sys.argv[2]
out = []
for e in json.load(open(f"{t}/env.json")): out += ["-e", e]
out.append(img); out += json.load(open(f"{t}/args.json"))
print("\n".join(out))
PY

echo "  host $HOST   container ${CID:0:12}   image $IMG"
A=(); while IFS= read -r l; do A+=("$l"); done < "$TMP/argv.txt"
"$B/scripts/preflight.sh" --manifest "$MANIFEST" --profile "$PROFILE" \
    --image-baseline "$TMP/imgenv.txt" --approved "$APPROVED" -- "${A[@]}"
