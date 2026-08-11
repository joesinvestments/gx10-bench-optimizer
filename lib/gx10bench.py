#!/usr/bin/env python3
"""
gx10bench — ONE benchmark tool for the GX10 fleet. Ours.

Combines the three engines behind a single entrypoint, one flag surface, one output
schema, one provenance row:

    tool-eval-bench   quality scenarios, spec-decode acceptance, context-pressure
    llama-benchy      throughput depth x concurrency  (driven via tool-eval-bench,
                      which resolves it from PATH or falls back to `uvx llama-benchy`)
    r0b0bench         canary, NIAH long-context retrieval, latency, throughput
    sessbench3 (ours) session-affine reuse — the 94.88% cache-read shape that NONE
                      of the three model

★ WHY THIS EXISTS, AND WHAT IT DELIBERATELY DOES NOT DO
  The predecessor (scripts/gx10-eval.sh) passed FIVE flags that do not exist —
  --tokenizer, --fail-on-safety, --weight-by-difficulty, --export, --export-output —
  to tool-eval-bench. They were taken from a README instead of from `--help`, the
  script was never run, and nothing caught it. So:

  1. This tool NEVER re-implements a measurement. Each lane shells out to the engine
     whose authors validated it. We orchestrate; they measure.
  2. VALIDATE-BEFORE-RUN is a hard gate. Every flag this tool will pass is checked
     against that engine's real `--help` at startup. A bad flag exits 2 BEFORE any
     load reaches the server. `--dry-run` runs only the gate.

USAGE
    gx10bench lanes                      # what exists, and which engine backs it
    gx10bench run --lanes canary,spec --dry-run
    gx10bench run --lanes quality,niah --out ~/Desktop/GX10-EVAL/run1
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

DEF_BASE = "http://192.168.1.16:8000/v1"
DEF_MODEL = "deepseek-v4-flash-0731"
BUNDLE = Path(__file__).resolve().parent.parent
DEF_TOK = str(BUNDLE / "client" / "dsv4-tok")
# spark-bench is a script, not a PATH binary. Vendored clone; override with SPARK_BENCH.
SPARK_BENCH = os.environ.get("SPARK_BENCH",
    os.path.expanduser("~/Desktop/bench-engines/spark-bench/spark_bench.py"))

# ── lane registry ─────────────────────────────────────────────────────────────
# argv is built by a function so validation and execution use the SAME list — a
# lane cannot be validated one way and run another.
def _teb(*extra):
    return ["tool-eval-bench", "bench", *extra]

def _sb(*extra):
    return [sys.executable, SPARK_BENCH, "eval", *extra]

def _r0b0(only, *extra):
    return ["r0b0bench", "run", "--profile", "systems", "--only", only, *extra]

LANES = {
    # name:        (engine, builder(cfg) -> argv, one-line what-it-measures)
    "canary": ("r0b0bench", lambda c: _r0b0(
        "canary", "--base-url", c.base_url, "--model", c.model,
        "--tokenizer", c.tokenizer, "--output", str(c.out / "r0b0"),
        "--run-id", f"{c.label}-canary"),
        "deterministic API gate — run this first, it is cheap"),

    "niah": ("r0b0bench", lambda c: _r0b0(
        "niah", "--base-url", c.base_url, "--model", c.model,
        "--tokenizer", c.tokenizer, "--output", str(c.out / "r0b0"),
        "--run-id", f"{c.label}-niah", "--timeout", "7200"),
        "long-context RETRIEVAL. capacity != usability"),

    "latency": ("r0b0bench", lambda c: _r0b0(
        "latency", "--base-url", c.base_url, "--model", c.model,
        "--tokenizer", c.tokenizer, "--output", str(c.out / "r0b0"),
        "--run-id", f"{c.label}-latency"),
        "TTFT / ITL / TPOT, server-side definitions"),

    "throughput": ("r0b0bench", lambda c: _r0b0(
        "throughput", "--base-url", c.base_url, "--model", c.model,
        "--tokenizer", c.tokenizer, "--output", str(c.out / "r0b0"),
        "--run-id", f"{c.label}-throughput"),
        "r0b0tlab's own throughput lane"),

    "spec": ("tool-eval-bench", lambda c: _teb(
        "--spec-bench", "--spec-method", "auto",
        "--spec-prompts", "code,structured,filler",
        "--depth", "0,32768",
        "--base-url", c.base_url, "--model", c.model, "--metrics-url", c.metrics,
        "--json-file", str(c.out / "spec.json")),
        "spec-decode acceptance. replaces our hand-rolled accept.py"),

    "perf": ("tool-eval-bench", lambda c: _teb(
        "--perf-only", "--pp", "2048", "--tg", "128",
        "--depth", "0,32768,131072", "--concurrency", c.concurrency,
        "--base-url", c.base_url, "--model", c.model, "--metrics-url", c.metrics,
        # llama-benchy needs a tokenizer and cannot fetch one offline; tool-eval-bench has NO
        # --tokenizer flag, so it must ride in as a benchy PASS-THROUGH. Without this the
        # perf lane dies in 2.5s with "offline mode with no tokenizer in the HuggingFace cache".
        "--benchy-args", f"--runs 4 --exact-tg --latency-mode generation --tokenizer {c.tokenizer}",
        "--json-file", str(c.out / "perf.json")),
        "throughput depth x concurrency, via llama-benchy"),

    "quality": ("tool-eval-bench", lambda c: _teb(
        "--seed", "42", "--trials", "3", "--hardmode",
        "--temperature", c.temperature, "--top-p", c.top_p,
        "--error-rate", "0.05", "--max-turns", "8",
        "--base-url", c.base_url, "--model", c.model, "--metrics-url", c.metrics,
        "--json-file", str(c.out / "quality.json")),
        "scenarios + hardmode, 3 trials, tool errors injected"),

    "pressure": ("tool-eval-bench", lambda c: _teb(
        "--context-pressure-sweep", "0.3-0.9", "--sweep-steps", "4",
        "--temperature", c.temperature, "--top-p", c.top_p,
        "--base-url", c.base_url, "--model", c.model, "--metrics-url", c.metrics,
        "--json-file", str(c.out / "pressure.json")),
        "quality as context FILLS. our median session sits at 0.33"),

    "agentic": ("spark-bench", lambda c: _sb(
        "--endpoint", c.base_url, "--model", c.model,
        "--label", f"{c.label}-agentic", "--out-dir", str(c.out / "spark-bench"),
        "--tier", "challenge", "--domains", "tool_use,coding",
        "--repeats", "3", "--temperature", c.temperature, "--thinking", "auto",
        "--topology", "4x DGX Spark", "--parallelism", "TP4",
        "--spec-decode", "dspark", "--run-kind", "benchmark",
        "--notes", "gx10bench agentic lane"),
        "multi-turn agentic + tool-FAILURE injection, partial-credit graded"),

    "truescore": ("spark-bench", lambda c: _sb(
        "--endpoint", c.base_url, "--model", c.model,
        "--label", f"{c.label}-truescore", "--out-dir", str(c.out / "spark-bench"),
        "--tier", "all", "--repeats", "3",
        "--temperature", c.temperature, "--thinking", "auto",
        "--topology", "4x DGX Spark", "--parallelism", "TP4",
        "--spec-decode", "dspark", "--run-kind", "benchmark",
        "--notes", "gx10bench truescore lane"),
        "full 76-scenario TrueScore (quality/calibration/reliability), executable code"),

    "accept-live": ("gx10", lambda c: [
        sys.executable, str(BUNDLE / "client" / "accept_live.py"), "60", c.host],
        "spec-decode acceptance off LIVE traffic. SENDS NOTHING. safe mid-production"),

    "session": ("gx10", lambda c: [
        sys.executable, str(BUNDLE / "client" / "sessbench3.py"),
        c.label, c.host, "4", "12"],
        "session-affine reuse at 94.88% cache-read. OURS — no engine models this"),
}

PRESETS = {
    "quick":   ["canary", "spec", "perf"],
    "health":  ["accept-live"],
    "quality": ["quality", "pressure", "agentic"],
    "context": ["niah", "pressure"],
    "full":    ["canary", "spec", "perf", "quality", "pressure", "agentic", "niah", "session"],
    "hermes":  ["agentic", "session", "pressure", "accept-live"],
}


class Cfg:
    def __init__(self, a):
        self.base_url = a.base_url
        self.model = a.model
        self.tokenizer = a.tokenizer
        self.label = a.label
        self.out = Path(a.out).expanduser()
        self.metrics = a.base_url.rsplit("/v1", 1)[0] + "/metrics"
        self.host = a.base_url.split("//", 1)[-1].split(":", 1)[0]
        self.concurrency = a.concurrency
        self.temperature = a.temperature
        self.top_p = a.top_p


# ── the gate ──────────────────────────────────────────────────────────────────
# Flags whose VALUE is a pass-through argv for a different engine. Their contents
# must be validated against THAT engine, not the parent. Checking them against the
# parent is a false FAIL; skipping them entirely would be a false PASS.
PASSTHROUGH = {"--benchy-args": "llama-benchy"}


def engine_help(engine: str) -> str | None:
    """Real --help text, or None if the engine is not installed."""
    if engine == "gx10":
        return ""                                   # our own script, nothing to check
    if engine == "llama-benchy":
        # tool-eval-bench resolves it from PATH, else `uvx llama-benchy`. Mirror that.
        cmd = ["llama-benchy"] if shutil.which("llama-benchy") else (
              ["uvx", "llama-benchy"] if shutil.which("uvx") else None)
        if cmd is None:
            return None
        try:
            r = subprocess.run([*cmd, "--help"], capture_output=True, text=True, timeout=180)
            return (r.stdout or "") + (r.stderr or "")
        except Exception:
            return None
    if engine == "spark-bench":
        if not os.path.exists(SPARK_BENCH):
            return None
        try:
            r = subprocess.run([sys.executable, SPARK_BENCH, "eval", "--help"],
                               capture_output=True, text=True, timeout=120)
            return (r.stdout or "") + (r.stderr or "")
        except Exception:
            return None
    if not shutil.which(engine.split()[0]):
        return None
    sub = {"tool-eval-bench": ["bench"], "r0b0bench": ["run"]}.get(engine, [])
    try:
        r = subprocess.run([engine, *sub, "--help"], capture_output=True, text=True, timeout=60)
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return None


def validate(lanes, cfg, verbose=True):
    """Every flag we will pass must exist in that engine's --help. Hard gate."""
    helps, problems = {}, []
    for name in lanes:
        engine, build, _ = LANES[name]
        if engine not in helps:
            helps[engine] = engine_help(engine)
        h = helps[engine]
        if h is None:
            problems.append((name, engine, "ENGINE NOT INSTALLED"))
            continue
        argv = build(cfg)
        skip_next = None
        for tok in argv:
            if skip_next:
                pt_engine, pt_flag = skip_next
                skip_next = None
                if pt_engine not in helps:
                    helps[pt_engine] = engine_help(pt_engine)
                ph = helps[pt_engine]
                if ph is None:
                    problems.append((name, pt_engine, f"pass-through engine for {pt_flag} NOT AVAILABLE"))
                    continue
                for inner in str(tok).split():
                    if inner.startswith("--") and inner not in ph:
                        problems.append((name, pt_engine, f"pass-through flag {inner} not in {pt_engine} --help"))
                continue
            if isinstance(tok, str) and tok.startswith("--"):
                if tok in PASSTHROUGH:
                    if tok not in h:
                        problems.append((name, engine, f"flag {tok} not in --help"))
                    skip_next = (PASSTHROUGH[tok], tok)
                    continue
                if tok not in h:
                    problems.append((name, engine, f"flag {tok} not in --help"))
        if verbose:
            status = "ok" if not any(p[0] == name for p in problems) else "FAIL"
            print(f"  {status:<4} {name:<11} {engine:<16} {LANES[name][2]}")
    return problems


def fingerprint(cfg, lanes):
    """Provenance row. A table without one is not publishable."""
    def sh(p):
        try: return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
        except Exception: return "missing"
    vers = {}
    for e in ("tool-eval-bench", "r0b0bench"):
        if shutil.which(e):
            try:
                r = subprocess.run([e, "--version"], capture_output=True, text=True, timeout=30)
                vers[e] = (r.stdout or r.stderr).strip().splitlines()[-1][:40]
            except Exception:
                vers[e] = "?"
        else:
            vers[e] = "ABSENT"
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": cfg.label, "base_url": cfg.base_url, "model": cfg.model,
        "lanes": lanes, "engines": vers,
        "gx10bench_sha": sh(__file__), "sessbench_sha": sh(BUNDLE / "client" / "sessbench3.py"),
    }


def main():
    p = argparse.ArgumentParser(prog="gx10bench", description="one benchmark tool for the GX10 fleet")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lanes", help="list lanes and which engine backs each")
    r = sub.add_parser("run", help="run lanes")
    r.add_argument("--lanes", default="quick",
                   help="comma list, or a preset: " + ", ".join(PRESETS))
    r.add_argument("--label", default="run")
    r.add_argument("--base-url", default=DEF_BASE)
    r.add_argument("--model", default=DEF_MODEL)
    r.add_argument("--tokenizer", default=DEF_TOK)
    r.add_argument("--concurrency", default="1,4,12")
    r.add_argument("--temperature", default="1.0")
    r.add_argument("--top-p", default="0.95")
    r.add_argument("--out", default="~/Desktop/GX10-EVAL/gx10bench")
    r.add_argument("--dry-run", action="store_true", help="validate only; send NO load")
    a = p.parse_args()

    if a.cmd == "lanes":
        print()
        for n, (e, _, d) in LANES.items():
            mark = "  " if (e == "gx10" or shutil.which(e) or (e == "spark-bench" and os.path.exists(SPARK_BENCH))) else "!!"
            print(f" {mark} {n:<11} {e:<16} {d}")
        print("\n  presets: " + "  ".join(f"{k}={'+'.join(v)}" for k, v in PRESETS.items()))
        print()
        return 0

    lanes = PRESETS.get(a.lanes, None) or [x.strip() for x in a.lanes.split(",") if x.strip()]
    bad = [l for l in lanes if l not in LANES]
    if bad:
        print(f"gx10bench: unknown lane(s) {bad}. Known: {list(LANES)}", file=sys.stderr)
        return 2

    a.out = os.path.expanduser(a.out)
    cfg = Cfg(a)
    cfg.out.mkdir(parents=True, exist_ok=True)

    print(f"\n  === gx10bench {a.label} — {len(lanes)} lane(s) ===")
    print(f"  {cfg.base_url}  {cfg.model}\n")
    print("  VALIDATE (every flag checked against the engine's real --help):")
    problems = validate(lanes, cfg)
    if problems:
        print("\n  ❌ GATE FAILED — no load was sent:")
        for n, e, msg in problems:
            print(f"      {n} [{e}]: {msg}")
        print("\n  Fix the lane definition in gx10bench.py. Do NOT work around this by\n"
              "  removing the check — five invented flags is exactly how the last harness died.")
        return 2
    print("  ✅ all flags exist\n")

    fp = fingerprint(cfg, lanes)
    (cfg.out / "provenance.json").write_text(json.dumps(fp, indent=2))
    print("  PROVENANCE " + json.dumps({k: fp[k] for k in ("utc", "engines", "gx10bench_sha")}))

    if a.dry_run:
        print("\n  --dry-run: validated only, nothing executed.\n")
        return 0

    results = {}
    for name in lanes:
        engine, build, _ = LANES[name]
        argv = build(cfg)
        print(f"\n  ── {name} [{engine}] ──")
        print("  $ " + " ".join(argv[:8]) + (" …" if len(argv) > 8 else ""))
        t0 = time.time()
        rc = subprocess.call(argv)
        results[name] = {"rc": rc, "seconds": round(time.time() - t0, 1), "argv": argv}
        print(f"  {name}: rc={rc} in {results[name]['seconds']}s")

    (cfg.out / "results.json").write_text(json.dumps({"provenance": fp, "lanes": results}, indent=2))
    failed = [n for n, v in results.items() if v["rc"] != 0]
    print(f"\n  === done: {cfg.out} ===")
    print(f"  {len(results) - len(failed)}/{len(results)} lanes ok" + (f", FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
