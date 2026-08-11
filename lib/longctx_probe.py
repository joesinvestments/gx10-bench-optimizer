#!/usr/bin/env python3
"""
longctx_probe.py — decode speed at PRODUCTION-CLASS context, C=1.

★ WHY (2026-08-06). Every k-ladder decision we ever made was measured at SHORT prompts
(48.83 ms/step), but production runs 84.72 ms/step at ~155K mean context. jeffery2011.jc's
4-node TP=4 cluster (NVIDIA forum 378878) steps at ~43 ms with k=3 at 150K — half our step
time on identical hardware. The suspect is k=7's verify cost against long KV: the target
verifies (k+1) attention rows per step, and the DSpark paper's "deep drafts are nearly free"
was measured at ctx 512-4096 and explicitly waives long context. This probe re-runs the k
decision at the context class we actually serve.

PROTOCOL (matches the forum post, so results are comparable):
  1. build a deterministic ~TARGET_TOKENS prompt (seeded, reproducible byte-for-byte)
  2. cold prime: send once, report TTFT + prefill tok/s
  3. N cache-hit repeats: same prompt, max_tokens=512, temperature=0, ignore_eos
  4. decode measured TWO ways and both reported:
       client: generated_tokens / (t_last_token - t_first_token)   per run
       server: delta(generation_tokens_total) / delta(request_decode_time_seconds_sum)
     If they disagree badly, say so — that is a finding, not an embarrassment.
  5. per-position acceptance deltas from /metrics around the measured block.

FOREIGN-TRAFFIC GUARD: refuses to run (exit 4) if the server shows running/waiting
requests, unless --force. Concurrent traffic corrupts both measurements.

USAGE
  longctx_probe.py HOST MODEL [--tokens 150000] [--runs 3] [--label k7-base] [--force]
"""
import argparse, json, random, sys, time, urllib.request


def scrape(host):
    t = urllib.request.urlopen(f"http://{host}:8000/metrics", timeout=15).read().decode()
    out = {}
    for line in t.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        head, val = line.rsplit(" ", 1)
        try:
            v = float(val)          # scientific notation safe — never shell-parse this
        except ValueError:
            continue
        out[head] = out.get(head, 0.0) + v
    return out


def msum(m, prefix):
    return sum(v for k, v in m.items() if k.startswith(prefix))


def build_prompt(n_tokens, seed=20260806):
    """Deterministic filler at ~3.5 chars/token for this tokenizer class. The server's
    reported prompt_tokens is the authority; this only needs to land in the right band."""
    rng = random.Random(seed)
    words = ("the quick brown fox jumps over lazy dog while seventeen engineers "
             "measure tensor parallel decode throughput across four spark nodes "
             "verifying speculative draft acceptance at every position under load").split()
    chunks = []
    # ~22 tokens per chunk for word-soup like this ("[i]" marker + 12 mostly-1-token words).
    # First calibration used n_tokens//8 and built a 400-560K-token prompt against a 393,216
    # ceiling — the server would have 400'd every run. Aim UNDER target; the server's
    # reported prompt_tokens is the authority and is printed with every result.
    for i in range(n_tokens // 24):
        rng.shuffle(words)
        chunks.append(f"[{i}] " + " ".join(words[:12]))
    return ("You are indexing a document. Read it fully, then answer.\n\n"
            + "\n".join(chunks)
            + "\n\nSummarize the document's structure in detail.")


def one_request(host, model, prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"http://{host}:8000/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    t_first = t_last = None
    n_chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
                now = time.monotonic()
                if t_first is None:
                    t_first = now
                t_last = now
                n_chunks += 1
    return {
        "ttft_s": (t_first - t0) if t_first else None,
        "decode_window_s": (t_last - t_first) if t_first and t_last and t_last > t_first else None,
        "chunks": n_chunks,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def concurrent_probe(a):
    """Regime-B probe: many small agentic-shaped requests at real concurrency.
    ★ WHY (2026-08-08): 4 days of production show TWO regimes, and the wide one —
    81% of requests ≤2K tokens, concurrency up to 11 — is where both 08-07 wedges
    happened. A C=1 long-context probe cannot see this regime at all. Shape mimics
    production: shared ~1.5K prefix (the cache-hit part, 96.7% live) + unique tail,
    ~150-token outputs, staggered arrivals (not a synchronized burst)."""
    import concurrent.futures, threading
    shared = build_prompt(a.tokens, seed=777)          # shared prefix -> prefix-cache hits
    pre = scrape(a.host)
    t_start = time.monotonic()
    lock = threading.Lock(); done = []
    def worker(i):
        time.sleep(0.4 * i)                            # staggered arrivals, like real traffic
        for j in range(a.runs):
            p = shared + f"\n\n[worker {i} turn {j}] Answer briefly: what is item {i*37+j} about?"
            try:
                r = one_request(a.host, a.model, p, a.max_tokens)
                with lock: done.append(r)
            except Exception as e:
                with lock: done.append({"error": str(e)})
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(worker, range(a.concurrency)))
    wall = time.monotonic() - t_start
    post = scrape(a.host)
    ok = [d for d in done if d.get("completion_tokens")]
    errs = len(done) - len(ok)
    gen = sum(d["completion_tokens"] for d in ok)
    dg = msum(post,"vllm:generation_tokens_total") - msum(pre,"vllm:generation_tokens_total")
    dt = (msum(post,"vllm:request_decode_time_seconds_sum")
          - msum(pre,"vllm:request_decode_time_seconds_sum"))
    ds = msum(post,"vllm:spec_decode_num_drafts_total") - msum(pre,"vllm:spec_decode_num_drafts_total")
    rates = [(d["completion_tokens"]-1)/d["decode_window_s"] for d in ok
             if d.get("decode_window_s") and d["completion_tokens"] > 1]
    rates.sort()
    p50 = rates[len(rates)//2] if rates else 0
    print(f"\n  ══ [{a.label}] CONCURRENT C={a.concurrency}×{a.runs} ══")
    print(f"  {len(ok)}/{len(done)} ok, {errs} errors · wall {wall:.1f}s · aggregate {gen/wall:.1f} tok/s")
    if dt > 0:
        print(f"  server decode {dg/dt:.2f} tok/s summed-stream · ms/step {1000*dt/ds:.2f}" if ds else "")
    print(f"  per-request decode p50 {p50:.2f} tok/s")
    # errors here are a FINDING — this is the regime that wedged production twice
    print(f"  MACHINE\t{a.label}\tconcurrent\tC={a.concurrency}\tagg={gen/wall:.1f}\t"
          f"p50={p50:.2f}\terrors={errs}")
    return 1 if errs else 0


def mixed_storm(a):
    """Probe C — the WEDGE HUNT. One worker fires a COLD ~150K prefill into the middle of a
    sustained small-request storm at high concurrency. This forces the exact condition the
    08-07 wedges lived in: chunked deep prefill RAGGED-MIXED with many decode streams at
    seq counts 9-12, crossing every ladder rung and both cudagraph eager holes. Community
    patches exist for precisely this path in the other DSpark lineage (Keys patch 2:
    'mixed prefill+decode -> non-uniform rows'). If a config survives three waves of this,
    it has earned production. If it wedges, we just reproduced in a 6-minute-restart window
    what cost two outages — and the boot log of the wedged container gets captured.
    Success = zero errors AND the endpoint still generating afterward."""
    import concurrent.futures, threading
    shared = build_prompt(1500, seed=777)
    deep = build_prompt(a.tokens, seed=int(time.time()) % 100000)   # unique -> genuinely cold
    lock = threading.Lock(); done = []; stop = threading.Event()
    def small_worker(i):
        j = 0
        while not stop.is_set():
            p = shared + f"\n\n[storm {i}.{j}] Answer briefly: describe item {i*41+j}."
            try:
                r = one_request(a.host, a.model, p, 120)
                with lock: done.append(("small", r))
            except Exception as e:
                with lock: done.append(("small", {"error": str(e)}))
            j += 1
    def deep_worker():
        time.sleep(6)                               # let the storm establish first
        try:
            r = one_request(a.host, a.model, deep, 256)
            with lock: done.append(("deep", r))
        except Exception as e:
            with lock: done.append(("deep", {"error": str(e)}))
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency + 1) as ex:
        futs = [ex.submit(small_worker, i) for i in range(a.concurrency)]
        dw = ex.submit(deep_worker)
        dw.result(timeout=900)                       # wave ends when the deep request lands
        stop.set()
        concurrent.futures.wait(futs, timeout=180)
    wall = time.monotonic() - t0
    small_ok = [r for k, r in done if k == "small" and r.get("completion_tokens")]
    small_err = sum(1 for k, r in done if k == "small" and r.get("error"))
    deep_res = next((r for k, r in done if k == "deep"), {})
    deep_ok = bool(deep_res.get("completion_tokens"))
    # post-wave liveness: is the engine still generating, or did we wedge it?
    alive = False
    try:
        g0 = msum(scrape(a.host), "vllm:generation_tokens_total")
        probe = one_request(a.host, a.model, "Reply with exactly: OK", 8)
        alive = bool(probe.get("completion_tokens"))
    except Exception:
        pass
    errs = small_err + (0 if deep_ok else 1) + (0 if alive else 10)
    print(f"\n  ══ [{a.label}] MIXED STORM C={a.concurrency}+deep ══")
    print(f"  wall {wall:.1f}s · small ok {len(small_ok)} err {small_err} · "
          f"deep {'ok TTFT %.1fs' % deep_res.get('ttft_s', -1) if deep_ok else 'FAILED'} · "
          f"post-wave engine {'ALIVE' if alive else '✖ NOT GENERATING — WEDGE'}")
    print(f"  MACHINE\t{a.label}\tmixed\tC={a.concurrency}\tsmall_ok={len(small_ok)}\t"
          f"deep_ttft={deep_res.get('ttft_s', -1):.1f}\terrors={errs}")
    return 1 if errs else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host"); ap.add_argument("model")
    ap.add_argument("--tokens", type=int, default=150000)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--label", default="unlabeled")
    ap.add_argument("--concurrency", type=int, default=0,
                    help="if >0: regime-B mode — N staggered workers × runs small requests")
    ap.add_argument("--mixed", action="store_true",
                    help="probe C: cold deep prefill inside a small-request storm (wedge hunt)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.mixed:
        m0 = scrape(a.host)
        busy = msum(m0, "vllm:num_requests_running") + msum(m0, "vllm:num_requests_waiting")
        if busy > 0 and not a.force:
            print(f"  ✖ {busy:.0f} foreign request(s) active — aborting.", file=sys.stderr)
            return 4
        return mixed_storm(a)

    if a.concurrency > 0:
        m0 = scrape(a.host)
        busy = msum(m0,"vllm:num_requests_running") + msum(m0,"vllm:num_requests_waiting")
        if busy > 0 and not a.force:
            print(f"  ✖ {busy:.0f} foreign request(s) active — aborting.", file=sys.stderr)
            return 4
        return concurrent_probe(a)

    m0 = scrape(a.host)
    busy = msum(m0, "vllm:num_requests_running") + msum(m0, "vllm:num_requests_waiting")
    if busy > 0 and not a.force:
        print(f"  ✖ server has {busy:.0f} active/queued requests — foreign traffic would "
              f"corrupt both measurements. Re-run when idle or pass --force.", file=sys.stderr)
        return 4

    prompt = build_prompt(a.tokens)
    print(f"  [{a.label}] target ~{a.tokens:,} tokens · {a.runs} measured cache-hit runs · temp 0")

    cold = one_request(a.host, a.model, prompt, a.max_tokens)
    pt = cold["prompt_tokens"] or 0
    print(f"  cold prime: prompt={pt:,} tok  TTFT={cold['ttft_s']:.2f}s  "
          f"prefill={pt/cold['ttft_s']:,.0f} tok/s" if cold["ttft_s"] else "  cold prime: FAILED")

    pre = scrape(a.host)
    client_rates, ttfts = [], []
    for i in range(a.runs):
        r = one_request(a.host, a.model, prompt, a.max_tokens)
        if r["decode_window_s"] and r["completion_tokens"]:
            # first token excluded from the window, so rate = (n-1)/window
            rate = (r["completion_tokens"] - 1) / r["decode_window_s"]
            client_rates.append(rate); ttfts.append(r["ttft_s"])
            print(f"    run {i+1}: TTFT={r['ttft_s']:.3f}s  decode={rate:.2f} tok/s "
                  f"({r['completion_tokens']} tok)")
        else:
            print(f"    run {i+1}: FAILED / no stream window")
    post = scrape(a.host)

    dg = msum(post, "vllm:generation_tokens_total") - msum(pre, "vllm:generation_tokens_total")
    dt = (msum(post, "vllm:request_decode_time_seconds_sum")
          - msum(pre, "vllm:request_decode_time_seconds_sum"))
    ds = (msum(post, "vllm:spec_decode_num_drafts_total")
          - msum(pre, "vllm:spec_decode_num_drafts_total"))
    dacc = (msum(post, "vllm:spec_decode_num_accepted_tokens_total")
            - msum(pre, "vllm:spec_decode_num_accepted_tokens_total"))
    dtok = (msum(post, "vllm:spec_decode_num_draft_tokens_total")
            - msum(pre, "vllm:spec_decode_num_draft_tokens_total"))

    print(f"\n  ══ [{a.label}] RESULTS at {pt:,} prompt tokens ══")
    if client_rates:
        mean = sum(client_rates) / len(client_rates)
        print(f"  client decode : {mean:.2f} tok/s  (runs: {', '.join(f'{x:.1f}' for x in client_rates)})")
    if dg > 0 and dt > 0:
        print(f"  server decode : {dg/dt:.2f} tok/s   ms/step: {1000*dt/ds:.2f}   "
              f"tok/step: {dg/ds:.3f}" if ds > 0 else f"  server decode : {dg/dt:.2f} tok/s")
    if ds > 0 and dtok > 0:
        print(f"  spec decode   : accept {100*dacc/dtok:.1f}%  drafted/step {dtok/ds:.2f}")
    # machine-readable line for the window script's summary table
    print(f"  MACHINE\t{a.label}\tprompt={pt}\tclient={mean:.2f}\t"
          f"server={dg/dt if dt else 0:.2f}\tmsstep={1000*dt/ds if ds else 0:.2f}\t"
          f"tokstep={dg/ds if ds else 0:.3f}\taccept={100*dacc/dtok if dtok else 0:.1f}"
          if client_rates and dt > 0 else f"  MACHINE\t{a.label}\tINCOMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
