#!/usr/bin/env python3
"""
accept_snapshot.py — freeze the live spec-decode + serving state of the DeepSeek TP=4
cluster into one ledger row. READ-ONLY: one HTTP GET to /metrics, nothing else.

★ WHY (2026-08-06, Tier 0). Every performance claim we have ever made about this cluster
came from a benchmark harness firing synthetic prompts. Those numbers describe the harness
as much as the server: the fixed short-prompt protocol reads 123.13 tok/s where production
aggregate is 57.4, because the step is 75% slower at real context. This reads the SERVER'S
OWN counters under REAL traffic, so the control we A/B against is the thing we actually run.

★ THE PARSE RULE THAT MATTERS. Prometheus emits large counters in scientific notation
(1.270805e+06). Shell truncation `${x%.*}` on that yields "1" — that exact bug produced 663
false wedge alarms from the watchdog on this same endpoint. Everything here is parsed by
float() in Python. Never shell-parse this endpoint. See memory/instruments-must-be-tested.

★ COUNTERS ARE CUMULATIVE OVER CONTAINER LIFETIME. A single row is a lifetime average, not
a snapshot of "now" — it includes idle stretches. Differencing two rows gives you the
interval. That is the whole point of running this daily: row N minus row N-1 is a real
window, and the spread across rows is the error bar the one-shot reading never had.

USAGE
    accept_snapshot.py                 # append a row, print the summary
    accept_snapshot.py --print         # print the ledger with per-interval deltas
"""
import argparse, json, os, sys, time, urllib.request
from pathlib import Path

ENDPOINT = os.environ.get("DSFV4_METRICS", "http://192.168.1.16:8000/metrics")
LEDGER = Path.home() / ".dsfv4-watchdog" / "accept-ledger.tsv"
COLS = ["utc", "uptime_s", "requests", "gen_tokens", "prompt_tokens", "cache_hit_pct",
        "mean_prompt", "tok_per_step", "ms_per_step", "decode_toks", "avg_batch",
        "accept_p0", "accept_p1", "accept_p2", "accept_p3", "accept_p4", "accept_p5",
        "accept_p6", "draft_toks", "blocks", "draft_toks_per_block", "accepted_total",
        "preemptions", "errors"]


def scrape(url=ENDPOINT, timeout=15):
    """Return {metric_name: [(labels_dict, value), ...]}. float() everywhere — never shell."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        text = r.read().decode("utf-8", "replace")
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        try:
            head, val = line.rsplit(" ", 1)
            v = float(val)                      # <- handles 1.270805e+06 correctly
        except ValueError:
            continue
        name, _, lbl = head.partition("{")
        labels = {}
        if lbl:
            for part in lbl.rstrip("}").split(","):
                k, _, raw = part.partition("=")
                labels[k.strip()] = raw.strip().strip('"')
        out.setdefault(name.strip(), []).append((labels, v))
    return out


def one(m, name, **want):
    """Sum every series of `name` whose labels match `want`. Missing metric -> None."""
    series = m.get(name)
    if not series:
        return None
    tot, hit = 0.0, False
    for labels, v in series:
        if all(labels.get(k) == str(x) for k, x in want.items()):
            tot += v
            hit = True
    return tot if hit else None


def collect():
    m = scrape()
    d = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    req = one(m, "vllm:request_success_total")
    gen = one(m, "vllm:generation_tokens_total")
    pt = one(m, "vllm:prompt_tokens_total")
    # ITL sum/count is the honest per-stream decode measure: it excludes queueing and TTFT.
    itl_s = one(m, "vllm:inter_token_latency_seconds_sum") or one(m, "vllm:time_per_output_token_seconds_sum")
    itl_n = one(m, "vllm:inter_token_latency_seconds_count") or one(m, "vllm:time_per_output_token_seconds_count")
    # NOTE: no fallback here on purpose. An earlier draft fell back to num_preemptions_total,
    # which is a different quantity entirely — it would have silently produced a garbage
    # avg_batch instead of an honest "?". Missing metric must read as missing.
    iters = one(m, "vllm:iteration_tokens_total_count")

    hits = one(m, "vllm:prefix_cache_hits_total") or one(m, "vllm:gpu_prefix_cache_hits_total")
    qrys = one(m, "vllm:prefix_cache_queries_total") or one(m, "vllm:gpu_prefix_cache_queries_total")

    # ★ TWO DIFFERENT DENOMINATORS, and using the wrong one is the easy mistake:
    #   num_draft_tokens_total = every drafted TOKEN   (1,960,998 — sum over all k positions)
    #   num_drafts_total       = every draft BLOCK     (282,348  — one per decode step)
    # Position-0 acceptance is accepted_p0 / BLOCKS. Dividing by tokens gives 0.13 instead
    # of 0.91 — off by a factor of k, and it looks like a catastrophic drafter rather than a
    # healthy one. Positions 1..6 are conditional on the prior position, so they chain.
    draft_toks = one(m, "vllm:spec_decode_num_draft_tokens_total")
    blocks = one(m, "vllm:spec_decode_num_drafts_total")
    acc = one(m, "vllm:spec_decode_num_accepted_tokens_total")
    # Per-position acceptance: vLLM exposes one series per draft position.
    pos = {}
    for labels, v in m.get("vllm:spec_decode_num_accepted_tokens_per_pos_total", []):
        p = labels.get("position")
        if p is not None:
            pos[int(p)] = pos.get(int(p), 0.0) + v

    d["requests"] = req
    d["gen_tokens"] = gen
    d["prompt_tokens"] = pt
    d["cache_hit_pct"] = round(100 * hits / qrys, 3) if hits and qrys else None
    d["mean_prompt"] = round(pt / req) if pt and req else None
    d["ms_per_step"] = round(1000 * itl_s / itl_n, 2) if itl_s and itl_n else None
    d["decode_toks"] = round(gen / itl_s, 2) if gen and itl_s else None
    d["tok_per_step"] = round(gen / itl_n, 3) if gen and itl_n else None
    d["avg_batch"] = round(itl_n / iters, 3) if itl_n and iters else None
    d["draft_toks"] = draft_toks
    d["blocks"] = blocks
    d["draft_toks_per_block"] = round(draft_toks/blocks,3) if draft_toks and blocks else None
    d["accepted_total"] = acc
    for i in range(7):
        d[f"accept_p{i}"] = pos.get(i)
    d["preemptions"] = one(m, "vllm:num_preemptions_total")
    d["errors"] = one(m, "vllm:request_failure_total") or 0.0
    d["uptime_s"] = None
    return d, pos, blocks


def fmt(v):
    if v is None:
        return "?"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", dest="show")
    a = ap.parse_args()

    if a.show:
        if LEDGER.exists():
            print(LEDGER.read_text())
        else:
            print("  no ledger yet")
        return 0

    try:
        d, pos, blocks = collect()
    except Exception as e:
        print(f"  ✖ scrape failed: {e}", file=sys.stderr)
        return 1

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        LEDGER.write_text("\t".join(COLS) + "\n")
    with LEDGER.open("a") as f:
        f.write("\t".join(fmt(d.get(c)) for c in COLS) + "\n")

    print(f"  snapshot {d['utc']}  ->  {LEDGER}")
    print(f"    requests {fmt(d['requests'])}   mean prompt {fmt(d['mean_prompt'])} tok   "
          f"prefix cache {fmt(d['cache_hit_pct'])}%")
    print(f"    decode {fmt(d['decode_toks'])} tok/s   {fmt(d['ms_per_step'])} ms/step   "
          f"{fmt(d['tok_per_step'])} tok/step   avg batch {fmt(d['avg_batch'])}")
    print(f"    preemptions {fmt(d['preemptions'])}   errors {fmt(d['errors'])}")

    if pos and blocks:
        print(f"\n    draft blocks {fmt(d['blocks'])}   draft tokens/block {fmt(d['draft_toks_per_block'])} of k=7")
        print("    per-position acceptance (pos 0 / blocks; pos n conditional on pos n-1)")
        print(f"      {'pos':<5}{'accepted':>12}{'of blocks':>12}{'conditional':>13}{'delta':>9}")
        prev = blocks
        prev_cond = None
        for i in sorted(pos):
            cond = pos[i] / prev if prev else 0.0
            delta = "" if prev_cond is None else f"{100*(cond-prev_cond):+.1f}pt"
            print(f"      {i:<5}{pos[i]:>12,.0f}{100*pos[i]/blocks:>11.1f}%{cond:>13.4f}{delta:>9}")
            prev, prev_cond = pos[i], cond
    return 0


if __name__ == "__main__":
    sys.exit(main())
