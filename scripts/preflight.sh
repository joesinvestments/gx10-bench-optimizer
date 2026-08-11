#!/usr/bin/env bash
# preflight.sh — PRE-LAUNCH fidelity gate. Refuses to let a launcher run until the
# resolved command line has been diffed, KEY=VALUE, against a named manifest profile.
#
#   WHY THIS EXISTS (2026-08-03). Two failures on the same day, both from prose:
#     * "I'm executing his production_profile verbatim" — written 38s after building a
#       launcher that carried VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 in from an old file
#       of mine. The manifest does not declare that key at all.
#     * "one variable moved (262,144 context)" — his max_model_len is 327,680, and the
#       KV cap was 60 GiB against his 14.9 GiB. Three moved.
#   A names-only diff was run and came back CLEAN while the wrong VALUE went through.
#   So: values, or it does not count.
#
#   NOT a duplicate of r0b0tlab's scripts/runtime_gate.py. That gate runs POST-launch
#   against a live container (image id, source revision, vllm commit, file sha256,
#   pid1 env for rank identity) and its manifest declares profiles as FLAG SETS only —
#   it has no env-var expectations, so it cannot see the contamination above. Run both:
#   preflight.sh before docker run, runtime_gate.py after the server is up.
#
# USAGE (from inside a launcher, immediately before `docker run`):
#     DOCKER_ARGS=( --name "$CONTAINER" --gpus all ... "$IMAGE" ... )
#     "$(dirname "$0")/preflight.sh" \
#         --manifest "$MANIFEST" --profile production_profile -- "${DOCKER_ARGS[@]}"
#     docker run -d "${DOCKER_ARGS[@]}"
#
#   --approved <file>   deviations you have explicitly OK'd, one `key=value` per line
#                       with a `#` comment giving date + who approved. Absent file = no
#                       deviations allowed.
#   --print-only        show the diff, always exit 0. For inspecting a config you are
#                       NOT about to launch. Never use this in a launcher.
#
# EXIT 1 on any MINE key that is not in --approved. A gate that stops you beats a doc
# that warns you.
set -uo pipefail

MANIFEST=""; PROFILE=""; APPROVED=""; BASELINE=""; PRINT_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --profile)  PROFILE="$2";  shift 2 ;;
    --approved) APPROVED="$2"; shift 2 ;;
    --image-baseline) BASELINE="$2"; shift 2 ;;
    --print-only) PRINT_ONLY=1; shift ;;
    --) shift; break ;;
    *) echo "preflight: unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -z "$MANIFEST" || -z "$PROFILE" ]] && { echo "preflight: --manifest and --profile are required" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "preflight: manifest not found: $MANIFEST" >&2; exit 2; }
[[ $# -gt 0 ]] || { echo "preflight: no command line given after --" >&2; exit 2; }

ARGV_FILE="$(mktemp -t preflight-argv)"; trap 'rm -f "$ARGV_FILE"' EXIT
printf '%s\n' "$@" > "$ARGV_FILE"

python3 - "$MANIFEST" "$PROFILE" "$ARGV_FILE" "${APPROVED:-}" "$PRINT_ONLY" "${BASELINE:-}" <<'PYEOF'
import hashlib, json, re, sys

manifest_path, profile, argv_path, approved_path, print_only, baseline_path = sys.argv[1:7]
print_only = print_only == "1"
argv = [l.rstrip("\n") for l in open(argv_path) if l.strip() != ""]
raw = open(manifest_path, "rb").read()
manifest = json.loads(raw)
if profile not in manifest:
    print(f"preflight: profile '{profile}' not in manifest. Have: "
          f"{[k for k in manifest if k.endswith('_profile')]}", file=sys.stderr)
    sys.exit(2)
prof = manifest[profile]

# ── manifest key -> how it appears on a real launch line ──────────────────────
# flag: value follows the flag.  bool_flag: presence == True.  env: -e KEY=VALUE.
# spec: lives inside the --speculative-config JSON blob.
SPEC = {
    "max_model_len":                 ("flag", "--max-model-len"),
    "max_num_seqs":                  ("flag", "--max-num-seqs"),
    "max_num_batched_tokens":        ("flag", "--max-num-batched-tokens"),
    "gpu_memory_utilization":        ("flag", "--gpu-memory-utilization"),
    "kv_cache_memory_bytes":         ("flag", "--kv-cache-memory-bytes"),
    "enable_flashinfer_autotune":    ("bool_flag", "--enable-flashinfer-autotune"),
    "enforce_eager":                 ("bool_flag", "--enforce-eager"),
    "mtp_tokens":                    ("spec", "num_speculative_tokens"),
    "vllm_disable_compile_cache":    ("env", "VLLM_DISABLE_COMPILE_CACHE"),
    "vllm_dsv4_enable_multi_stream": ("env", "VLLM_DSV4_ENABLE_MULTI_STREAM"),
    "vllm_use_aot_compile":          ("env", "VLLM_USE_AOT_COMPILE"),
    # env name confirmed from the author's own image baseline, 2026-08-03
    "use_breakable_cudagraph":       ("env", "VLLM_USE_BREAKABLE_CUDAGRAPH"),
    # our own profile's keys (recipe/gx10-verified-profiles.json)
    "max_cudagraph_capture_size":    ("flag", "--max-cudagraph-capture-size"),
    "vllm_sparse_indexer_max_logits_mb":        ("env", "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"),
    "vllm_allow_long_max_model_len":            ("env", "VLLM_ALLOW_LONG_MAX_MODEL_LEN"),
    "vllm_use_flashinfer_sampler":              ("env", "VLLM_USE_FLASHINFER_SAMPLER"),
    "vllm_use_b12x_moe":                        ("env", "VLLM_USE_B12X_MOE"),
    "vllm_memory_profiler_estimate_cudagraphs": ("env", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"),
}
# Declared in the manifest but not expressible as a single launch token. Reported as
# UNCHECKED rather than silently dropped — an unchecked key is not a passing key.
METADATA = {"runtime_lane", "status", "qualification_mode", "upstream_evidence",
            "performance_mode", "optimization_level", "cudagraph_implementation",
            "cudagraph_strict"}

# Env that is legitimately environment-specific: node identity, paths, fabric. These
# are tagged ENV and never require approval. Anything else undeclared is MINE.
ENV_OK = re.compile(
    r"^(VLLM_HOST_IP|VLLM_CACHE_ROOT|NCCL_.*|GLOO_SOCKET_IFNAME|TP_SOCKET_IFNAME"
    r"|HF_.*|TRANSFORMERS_.*|TORCH_CUDA_ARCH_LIST|FLASHINFER_.*|CUTE_DSL_ARCH"
    r"|TILELANG_.*|PYTORCH_CUDA_ALLOC_CONF)$")
ENV_PROOF = {
    "VLLM_HOST_IP":       "ip -4 addr show $FABRIC_INTERFACE | awk '/inet /{print $2}'",
    "VLLM_CACHE_ROOT":    "ls -d $HOME/.cache/huggingface/vllm-cache",
    "NCCL_IB_HCA":        "ssh $HOST ibv_devices",
    "GLOO_SOCKET_IFNAME": "ssh $HOST ip -br link show $FABRIC_INTERFACE",
    "TP_SOCKET_IFNAME":   "ssh $HOST ip -br link show $FABRIC_INTERFACE",
}

def norm(v):
    if isinstance(v, bool): return "true" if v else "false"
    s = str(v).strip()
    if s.lower() in ("true", "1", "yes"):  return "true"
    if s.lower() in ("false", "0", "no"):  return "false"
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s

# ── parse the real command line ───────────────────────────────────────────────
env, flags, bools, spec_cfg, image = {}, {}, set(), {}, None
i = 0
while i < len(argv):
    a = argv[i]
    if a == "-e" and i + 1 < len(argv):
        kv = argv[i + 1]
        k, _, v = kv.partition("=")
        env[k] = v if "=" in kv else "<inherited>"
        i += 2; continue
    if a.startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
        flags.setdefault(a, argv[i + 1])
    if a.startswith("--"):
        bools.add(a)
    if a == "--speculative-config" and i + 1 < len(argv):
        try: spec_cfg = json.loads(argv[i + 1])
        except Exception: spec_cfg = {}
    if image is None and ("ghcr.io" in a or re.match(r"^[\w.\-]+/[\w.\-/]+:[\w.\-]+$", a)):
        image = a
    i += 1

# ── SANITY: refuse to grade a command line we did not actually receive ────────
# The first run of this gate was handed an EMPTY argv (zsh has no `mapfile`) and
# reported "10 unapproved deviations" — a confident red verdict on nothing. A gate
# that cannot distinguish "no input" from "everything differs" is not a gate.
if not env and not any(a.startswith("--") for a in argv):
    print("preflight: ABORT — the command line after `--` contains no `-e` env and no "
          "`--flags`.\n  Got %d token(s): %r\n  This is almost certainly an empty or "
          "unexpanded array (zsh has no `mapfile`; use a bash while-read loop or run "
          "the launcher under bash). Refusing to report a diff against nothing."
          % (len(argv), argv[:5]), file=sys.stderr)
    sys.exit(2)

# ── subtract the IMAGE's own baked-in env ─────────────────────────────────────
# `docker inspect <container> .Config.Env` returns image defaults MERGED with the
# launcher's -e flags. Without this subtraction a live-container audit reports the
# author's own image (PATH, NVARCH, UV_*, VLLM_BUILD_*) as MY deviations. Only keys
# whose value DIFFERS from the image default, or that the image never set, are mine.
image_env = {}
env_effective = dict(env)
if baseline_path:
    try:
        for line in open(baseline_path):
            line = line.rstrip("\n")
            if not line: continue
            k, _, v = line.partition("=")
            image_env[k] = v
    except FileNotFoundError:
        pass
# Two views, deliberately. env_effective = what the process ACTUALLY sees (image
# default overridden by -e) and is what manifest-DECLARED keys must be judged on.
# env = launcher-attributable only, used to enumerate UNDECLARED keys, so the
# author's own image env is not reported as my deviation.
env_effective = {**image_env, **env}
if image_env:
    env = {k: v for k, v in env.items() if image_env.get(k) != v}

approved = {}
if approved_path:
    try:
        for line in open(approved_path):
            line = line.split("#")[0].strip()
            if not line: continue
            k, _, v = line.partition("=")
            approved[k.strip()] = norm(v)
    except FileNotFoundError:
        pass

# ── diff ──────────────────────────────────────────────────────────────────────
rows, mine, unchecked = [], [], []
for key, want in sorted(prof.items()):
    if key.startswith("_"):
        continue
    if key in METADATA:
        unchecked.append(f"{key}={norm(want)}"); continue
    if key not in SPEC:
        unchecked.append(f"{key}={norm(want)}  (no launch-line mapping)"); continue
    kind, token = SPEC[key]
    if kind == "flag":       got = flags.get(token)
    elif kind == "bool_flag": got = "true" if token in bools else "false"
    elif kind == "env":       got = env_effective.get(token)
    else:                     got = (spec_cfg or {}).get(token)
    got = norm(got) if got is not None else None
    want_n = norm(want)
    if got is None:
        rows.append(("MISSING", key, want_n, "<absent>")); mine.append((key, "<absent>"))
    elif got != want_n:
        rows.append(("MINE", key, want_n, got)); mine.append((key, got))

for k, v in sorted(env.items()):
    if any(s[0] == "env" and s[1] == k for s in SPEC.values()):
        continue
    if ENV_OK.match(k):
        rows.append(("ENV", k, "<environment-specific>", v))
    else:
        rows.append(("UNDECLARED", k, "<not in manifest>", v)); mine.append((k, v))

launch_sha = hashlib.sha256("\n".join(argv).encode()).hexdigest()
man_sha    = hashlib.sha256(raw).hexdigest()

W = max([len(r[1]) for r in rows] + [12])
print()
print(f"  ── PREFLIGHT — {profile} ──")
print(f"  manifest {manifest_path}")
print(f"  manifest_sha256 {man_sha[:16]}   launch_sha256 {launch_sha[:16]}   image {image or '<not found>'}")
print()
for tag, key, want, got in sorted(rows, key=lambda r: ({"MINE":0,"MISSING":0,"UNDECLARED":1,"ENV":2}[r[0]], r[1])):
    ok = " " if tag == "ENV" else ("~" if (key in approved and approved[key] == got) else "!")
    print(f"  {ok} {tag:<11} {key:<{W}}  manifest={want:<22} launch={got}")
    if tag == "ENV" and key in ENV_PROOF:
        print(f"                {'':<{W}}  proof: {ENV_PROOF[key]}")
print()
if unchecked:
    print(f"  UNCHECKED ({len(unchecked)}) — declared in the manifest, no single launch token to compare:")
    for u in unchecked: print(f"      {u}")
    print()

unapproved = [(k, v) for k, v in mine if approved.get(k) != v]
if not mine:
    print("  ✅ resolved launch line matches the profile on every mappable key.")
elif not unapproved:
    print(f"  ✅ {len(mine)} deviation(s), all present in {approved_path}:")
    for k, v in mine: print(f"      {k}={v}")
else:
    print(f"  ❌ {len(unapproved)} UNAPPROVED deviation(s) from {profile}:")
    for k, v in unapproved: print(f"      {k}={v}")
    print()
    print("  This launch is NOT the manifest profile. Either fix the launcher, or get")
    print("  each line above approved in chat and add it to the --approved file as:")
    for k, v in unapproved:
        print(f"      {k}={v}   # approved by Joe YYYY-MM-DD: <reason>")
    print()
    print("  ★ Until then, no fidelity word — \"verbatim\", \"his config\", \"one variable\",")
    print("    \"same fixture\", \"isolates cleanly\" — may be used about this run, and no")
    print("    result from it may be compared against a published number.")

print()
print(f"  FINGERPRINT profile={profile} manifest_sha={man_sha[:16]} launch_sha={launch_sha[:16]} "
      f"image={image or 'unknown'} deviations={len(mine)} unapproved={len(unapproved)}")
print()
sys.exit(0 if (print_only or not unapproved) else 1)
PYEOF
