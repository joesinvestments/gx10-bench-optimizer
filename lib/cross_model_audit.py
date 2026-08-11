#!/usr/bin/env python3
"""
cross_model_audit.py — flag knobs where OUR OWN models disagree with each other,
or with a vendored reference recipe, for no recorded reason.

★★ WHY THIS EXISTS (2026-08-04)
   Our GLM recipe has used `draft_sample_method: probabilistic` for weeks. Our DeepSeek
   launcher used `greedy`. That single divergence is the leading explanation for DeepSeek
   running at 27-48% spec-decode acceptance instead of ~80%, and it sat there unnoticed
   while a full day was spent tuning TP size, vLLM versions and KV caps.

   It was not that the fact was unknown. It was recorded TWICE — as CONFIGURATION:
     memory/dynamic-speculative-decoding.md:61  ... "draft_sample_method":"greedy" ...
     memory/glm52-0xdfi-1m-comparison.md:22     ... probabilistic draft ...
   A config snapshot has no cross-model reach. Nothing linked the parameter to what it
   DOES, so nothing fired when the other model's launcher was written.

   Prose cannot fix that; a doc is only read when you already suspect something. This
   is a CHECK. It reads every recipe/launcher we own plus the vendored references, and
   prints the knobs where we disagree with ourselves. Run it before any launch change.

USAGE
    python3 cross_model_audit.py            # table + divergences
    python3 cross_model_audit.py --strict   # exit 1 if any UNEXPLAINED divergence
"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path

HOME = Path.home()
DS = HOME / "Desktop/DEEPSEEK-V4-RESTORE-BUNDLE"
GLM = HOME / "Desktop/GLM52-RESTORE-BUNDLE"

# Knobs that mean the SAME THING across models, with what they actually do. A knob only
# belongs here if a wrong value is silently costly — that is what makes divergence a bug
# rather than a legitimate difference.
KNOBS = {
    "draft_sample_method": (
        "How the DRAFT model samples. `greedy` = point mass, so spec-decode acceptance "
        "collapses to p_target(argmax) — flat across draft positions, and BAD whenever "
        "the target samples at temperature > 0. `probabilistic` matches the target's "
        "distribution. THE 2026-08-04 BUG."),
    "num_speculative_tokens": ("draft depth k. tok/step ceiling is k+1."),
    "VLLM_MARLIN_USE_ATOMIC_ADD": (
        "Marlin mixed-precision GEMM reduce path. Default OFF = fp32 global reduction, "
        "worst at tiny-M shapes — and spec-decode DRAFT forwards (tiny-M, run k times "
        "sequentially per step) are the pathological case. Measured on GLM-5.2/GB10 "
        "2026-08-10: quantized draft 6.5 tok/s without, 44.6 with (7x); large-GEMM main "
        "path unaffected. Safe on cc>=90 (bf16 atomics native). EVERY model serving "
        "compressed-tensors int4/int8 through Marlin with MTP/EAGLE drafting must set "
        "or deliberately decline this."),
    "kv_cache_dtype": ("KV bytes/token. nvfp4_ds_mla << fp8_ds_mla."),
    "generation_config": ("`vllm` IGNORES the checkpoint's generation_config.json, so a "
                          "client omitting temperature gets vLLM's default 1.0."),
    "gpu_memory_utilization": ("fraction of unified memory vLLM may claim."),
    "max_cudagraph_capture_size": ("graph memory; competes directly with the KV pool."),
    "moe_backend": ("MoE kernel. flashinfer_b12x is BANNED on sm_121 for GLM."),
    # ── FABRIC (env vars, not serve flags — invisible to this audit until 2026-08-05) ──
    "NCCL_IB_HCA": (
        "Which RDMA devices NCCL may use. On dual-port GB10, ports A and B are SEPARATE "
        "PCIe functions with separate GID tables (rocep1s0f0 / roceP2p1s0f0). Listing only "
        "ONE pins all traffic to one rail: measured ~110 Gb/s vs ~222 Gb/s using both — the "
        "single PCIe Gen5 x4 limit. Both rails must ALSO have IPs on DIFFERENT subnets, or "
        "Linux routing pins the outbound route to one interface and you lose the second rail "
        "silently. THE 2026-08-05 FINDING: DeepSeek pinned one HCA while GLM used both."),
    "NCCL_NET": ("IB vs Socket. A silent TCP fallback costs ~12 vs 30+ tok/s and is INVISIBLE "
                 "from nvidia-smi. Must be IB."),
    "NCCL_IB_DISABLE": ("must be 0. Some upstream launch scripts hard-set 1, clobbering overrides."),
    "NCCL_IB_GID_INDEX": ("RoCEv2 GID index. Not stable across boots on every stack — GLM "
                          "resolves it dynamically, DeepSeek pins 0 as proven-working."),
}

# Divergences that ARE deliberate, with the reason. Anything NOT here is unexplained —
# which is the exact category the greedy-vs-probabilistic miss lived in for weeks.
DELIBERATE = {
    "moe_backend": "GLM must use flashinfer_cutlass — flashinfer_b12x is BANNED on sm_121 "
                   "for GLM (standing rule e). DeepSeek correctly uses flashinfer_b12x.",
    "num_speculative_tokens": "Model-and-stack-specific, measured. DeepSeek runs a "
                              "batch-aware ladder 7/5/3. GLM DCP-stack recipes measured "
                              "k=2 best; the LEGACY stack serving since 2026-08-10 runs "
                              "k=4 with the quantized MTP draft (viable only with "
                              "VLLM_MARLIN_USE_ATOMIC_ADD=1, 6.5->44.6 tok/s draft).",
    "gpu_memory_utilization": "Sized per model+stack to the unified-memory KV/host-headroom "
                              "trade, not a shared constant: DeepSeek port-026 default 0.85; "
                              "GLM DCP recipes 0.90; GLM legacy stack 0.91 (pairs with its "
                              "explicit --kv-cache-memory-bytes 10.95GB cap); jvr0x vendor "
                              "reference 0.80 is a conservative shipping default we "
                              "deliberately exceed. Shrink only in steps with the 80% "
                              "cache-hit-rate canary watching.",
    "max_cudagraph_capture_size": "Graph memory competes with the KV pool and unused capture "
                                  "range is walked every step (the GLM 5.7->14.7 decode "
                                  "lesson). GLM DCP recipes capture 36 to match max-num-seqs; "
                                  "DeepSeek captures 96 for its batch-aware k ladder's larger "
                                  "batch shapes. The GLM legacy stack sets no size — it uses "
                                  "compilation-config cudagraph_mode FULL with vLLM defaults.",
    "kv_cache_dtype": "DeepSeek serves nvfp4_ds_mla (nvfp4 checkpoint, flashinfer path). "
                      "The GLM legacy stack serves fp8_ds_mla: the QuantTrio Int4-Int8Mix "
                      "checkpoint on the legacy sparse-MLA kernel overlay has no nvfp4 KV "
                      "path. Not interchangeable; re-evaluate only on a stack change.",
    "NCCL_IB_GID_INDEX": "Not stable across boots on every stack. DeepSeek pins 0 as "
                         "proven-working on its image; the GLM legacy stack pins 4 (its "
                         "image's `show_gids`-verified RoCEv2 index, per the launcher's "
                         "EDIT note). Pinned per image, both verified live.",
    "VLLM_MARLIN_USE_ATOMIC_ADD": "GLM=1 (compressed-tensors weights serve through Marlin; "
                                  "measured 6.5->44.6 tok/s on the quantized draft, "
                                  "2026-08-10). DeepSeek unset ON PURPOSE: nvfp4 weights + "
                                  "flashinfer paths never touch Marlin, flag is moot there. "
                                  "Re-evaluate for any NEW model with int4/int8 "
                                  "compressed-tensors weights.",
}

# ★ EDIT THIS FOR YOUR OWN FLEET. These paths are the author's; nothing else in this file
# is site-specific. A "source" is any file that declares serve flags — a launcher script, a
# recipe YAML, a vendored reference config. The audit reads each, extracts the KNOBS below,
# and flags where they DISAGREE with no recorded reason. It needs at least two sources to be
# useful, and it is most useful when they are DIFFERENT MODELS on the same hardware — that is
# the case where a knob silently diverges and nobody notices.
# The live-container row needs no editing: it comes from DSFV4_SSH_HOST / DSFV4_CONTAINER.
SOURCES = [
    ("deepseek/live-tp4", DS / "scripts/launch_rank_tp4.sh"),
    ("deepseek/port-026", DS / "scripts/launch_tp4_port026.sh"),
    ("glm/longctx-v2", GLM / "recipe/glm-dcp2-v2-longctx.yaml"),
    ("glm/speed128k-v2", GLM / "recipe/glm-dcp2-v2-speed128k.yaml"),
    ("ref:jvr0x/ds-dual-tp2", DS / "recipe/jvr0x/deepseek-v4-flash-0731-dual.yaml"),
]

# Sources that live on another machine — read over ssh so the audit compares what is
# ACTUALLY serving, not a local copy that can rot. Each successful read is cached next
# to this script; if ssh fails the cache is used and marked stale in the table header.
# ★ 2026-08-10: GLM production moved off the DCP recipes above to the legacy stack
# launched from gx10-1 — auditing only the DCP yamls was auditing a retired config.
REMOTE_SOURCES = [
    ("glm/legacy-gx10-1", "gx10-1", "~/glm-legacy-stack/launch_gx10.sh"),
]
REMOTE_CACHE = Path(__file__).resolve().parent / ".remote_cache"


def remote_text(name: str, host: str, rpath: str):
    """Returns (text, stale). text=None if unreachable and no cache exists."""
    cache = REMOTE_CACHE / (name.replace("/", "_") + ".cached")
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                            host, f"cat {rpath}"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            REMOTE_CACHE.mkdir(exist_ok=True)
            cache.write_text(r.stdout)
            return r.stdout, False
    except Exception:
        pass
    if cache.exists():
        return cache.read_text(), True
    return None, False


def resolve_var(val, text, depth=0):
    """Launchers pass knobs as "$VAR". Comparing that literal against other sources'
    real values produced permanent false UNEXPLAINED rows (gmu / capture / HCA,
    2026-08-10). Resolve $VAR to the launcher's own in-file default; a var with NO
    in-file assignment is caller-supplied — return None so the row shows '—' and the
    LIVE(running) container row remains the only truth for it."""
    m = re.fullmatch(r'\$\{?([A-Za-z_]\w*)[^}]*\}?', val)
    if not m:
        return val
    name = m.group(1)
    a = re.search(rf'^\s*(?:export\s+)?{name}=(.+)$', text, re.M)
    if not a:
        return None
    rhs = a.group(1).strip().strip('\'"')
    d = re.fullmatch(r'\$\{(\w+):-([^}]*)\}', rhs)
    if d:
        rhs = d.group(2).strip('\'"')
    if rhs.startswith("$") and depth < 3:
        return resolve_var(rhs, text, depth + 1)
    return rhs or None


def extract_text(t: str) -> dict:
    out = {}
    for k in KNOBS:
        # JSON form inside --speculative-config, then CLI flag form, then YAML key
        # form, then docker env form (quote-tolerant: -e K=V and -e "K=V" both occur)
        m = (re.search(rf'"{k}"\s*:\s*"([^"]+)"', t)
             or re.search(rf'"{k}"\s*:\s*([0-9.]+)', t)
             or re.search(rf'--{k.replace("_", "-")}[= ]+[\'"]?([^\s\'"\\]+)', t)
             or re.search(rf'^\s*{k}:\s*[\'"]?([^\'"\n]+)', t, re.M)
             or re.search(rf'-e\s+[\'"]?{k}=[\'"]?([^\s\'"\\]+)', t))
        if m:
            v = m.group(1).strip()
            if v.startswith("$"):
                v = resolve_var(v, t)
            if v is not None:
                out[k] = v
    return out


def extract(path: Path) -> dict:
    if not path.exists():
        return {}
    return extract_text(path.read_text(errors="replace"))


def live_container(host=None, name=None):
    """Reads the RUNNING config over ssh. Host/container from env so this file carries
    no site-specific names:  export DSFV4_SSH_HOST=<host>  DSFV4_CONTAINER=<name>"""
    host = host or os.environ.get("DSFV4_SSH_HOST", "localhost")
    name = name or os.environ.get("DSFV4_CONTAINER", "vllm-head")
    """Read the RUNNING config. Launchers hold $VARS; only the live container has values."""
    import subprocess
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
                            f'docker inspect {name} --format "{{{{json .Args}}}}"'],
                           capture_output=True, text=True, timeout=30)
        args = json.loads(r.stdout.strip())
        # ★ Config.Env too. Fabric settings (NCCL_*) live in the ENVIRONMENT, never in the
        # serve args — reading only .Args made the single-rail bug structurally unfindable.
        e = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
                            f'docker inspect {name} --format "{{{{json .Config.Env}}}}"'],
                           capture_output=True, text=True, timeout=30)
        env = json.loads(e.stdout.strip() or "[]")
    except Exception:
        return {}
    blob = " ".join(args) + "\n" + "\n".join(f"-e {v}" for v in env)
    out = {}
    for k in KNOBS:
        m = (re.search(rf'"{k}"\s*:\s*"([^"]+)"', blob)
             or re.search(rf'--{k.replace("_", "-")}\s+([^\s]+)', blob)
             # env form — MUST be here too, not only in extract(). Omitting it is why the
             # live single-rail config still showed blank after the knobs were added.
             or re.search(rf'^-e {k}=(.*)$', blob, re.M))
        if m:
            out[k] = m.group(1).strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()

    data = {name: extract(p) for name, p in SOURCES}
    missing = [n for n, p in SOURCES if not p.exists()]
    names = [n for n, _ in SOURCES if n not in missing]
    stale = []
    for name, host, rpath in REMOTE_SOURCES:
        t, is_stale = remote_text(name, host, rpath)
        if t is None:
            missing.append(f"{name} (ssh {host} unreachable, no cache)")
            continue
        if is_stale:
            stale.append(name)
        data[name] = extract_text(t)
        names.append(name)
    live = live_container()
    if live:
        data["LIVE(running)"] = live
        names.insert(0, "LIVE(running)")

    w = max(len(k) for k in KNOBS) + 2
    print(f"\n  {'knob':<{w}}" + "".join(f"{n[:21]:<23}" for n in names))
    print("  " + "-" * (w + 23 * len(names)))
    divergences = []
    for k in KNOBS:
        vals = {n: data[n].get(k, "—") for n in names}
        real = {v for v in vals.values() if v != "—"}
        flag = "  " if len(real) <= 1 else "! "
        if len(real) > 1:
            divergences.append((k, vals))
        print(f"  {flag}{k:<{w-2}}" + "".join(f"{vals[n][:21]:<23}" for n in names))

    if missing:
        print(f"\n  (not found, skipped: {', '.join(missing)})")
    if stale:
        print(f"\n  ⚠ STALE remote source(s) — ssh failed, using last cached copy: {', '.join(stale)}")

    if not divergences:
        print("\n  ✅ no cross-model divergence on the audited knobs.\n")
        return 0

    unexplained = [(k, v) for k, v in divergences if k not in DELIBERATE]
    known = [(k, v) for k, v in divergences if k in DELIBERATE]
    if known:
        print(f"\n  {len(known)} DELIBERATE divergence(s), reason on file:")
        for k, _ in known:
            print(f"    {k}: {DELIBERATE[k]}")
    if not unexplained:
        print("\n  ✅ no UNEXPLAINED divergence.\n")
        return 0
    print(f"\n  ⚠ {len(unexplained)} UNEXPLAINED knob(s) where our own configs disagree:\n")
    for k, vals in unexplained:
        print(f"  ── {k}")
        print(f"     {KNOBS[k]}")
        for n, v in vals.items():
            if v != "—":
                print(f"       {n:<24} {v}")
        print()
    print("  Each divergence is either a DELIBERATE, model-specific choice — in which case")
    print("  record WHY next to it — or it is a bug that has been sitting there unread.")
    print("  A value that differs across our own models with no stated reason is the exact")
    print("  shape of the greedy-vs-probabilistic miss.\n")
    return 1 if a.strict else 0


if __name__ == "__main__":
    sys.exit(main())
