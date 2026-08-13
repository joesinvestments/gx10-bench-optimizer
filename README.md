# GX10 Bench Optimizer

Model-agnostic benchmarking + tuning for multi-node LLM serving, in one command: `fleet`.

One coherent system, two divisions, one command. **A model IS a profile file**
`models/<name>.env`, ~15 keys. Adding a model to the entire system is writing that file.

```
fleet ls                                   profiles
fleet <model> status                       endpoint, containers, live health
fleet <model> corpus                       map of every ledger, result, reference, lesson

── DIVISION 1: BENCH, measurement (read-mostly, foreign-traffic-guarded) ──
fleet <model> bench lanes                  12 lanes across 4 engines (r0b0bench,
                                           tool-eval-bench, spark-bench, our own)
fleet <model> bench run <preset|lanes..>   quick · health · quality · context · full · hermes
fleet <model> bench probe-a|b|c            deep C=1 · storm C=max · mixed wedge-hunt
fleet <model> bench parity [C] [label]     community apples-to-apples (published protocol)
fleet <model> bench accept [--sweep ..]    spec-decode acceptance + temperature sweep
fleet <model> bench snapshot               production EKG → append-only ledger

── DIVISION 2: TUNE, config lifecycle (gated; nothing mutates without --authorized) ──
fleet <model> tune preflight               manifest-vs-live config diff (read-only)
fleet <model> tune fidelity                would auto-recovery reproduce production?
fleet <model> tune audit [--strict]        cross-model knob divergence (reasons required)
fleet <model> tune window [--authorized]   staged experiment window: N single-variable
                                           boots × 3 probes each, stability-first winner,
                                           evidence capture before every teardown, 3-wave soak
fleet <model> tune floors                  quality-floor status (acceptance / RAM / cache-hit)
```

## Architecture

```
bin/fleet          the dispatcher, division routing, profile resolution, fail-closed
models/*.env       one profile per model (the ONLY thing a new model needs)
lib/               every instrument, symlinked from its maintained home, single source
                   of truth; a community release vendors these as copies
recipe/ scripts/   bundle-shaped links so instruments that resolve paths relative to
                   their bundle root work unmodified through the kit
```

Design rules, each paid for by a real failure:
- **Instruments are wrapped, never rewritten**, their self-tests and histories hold.
- **Fail closed everywhere**: unknown model/command/division → exit 2; unreachable server →
  "could not compare", never a pass; a launcher without DRY_RUN support → fidelity exit 2.
  (The first lib/ integration broke fidelity's launcher resolution, and it ABORTED loudly
  instead of passing. That is the contract working.)
- **Measurements enforce their own validity** (`lib/probe_battery_v2.py`): the battery
  REFUSES to start against a non-idle server, DISCARDS any segment where foreign requests
  complete mid-measurement, cache-busts every prompt by construction, and labels the
  content class on every number (spec-decode throughput depends on output predictability;
  a peak and a sustained number are different measurements). Paid for three times before
  it became code: prefix-cache pollution, content-class conflation, and live-traffic
  pollution each produced a published number that had to be walked back. A precondition
  that lives in the operator's head instead of the instrument will be violated.
- **Measure at the production shape**: probes read their prompt band and concurrency from
  the profile, not from constants.
- **Stability first**: the window disqualifies any config that errors under storm or
  mixed-storm probes regardless of its speed.

## Profile contract (`models/<name>.env`)
Required: `FLEET_MODEL_ID HOST PORT HEAD_CONTAINER WORKER_CONTAINER SSH_HEAD NODES BUNDLE`.
Shape: `FLEET_DEEP_TOKENS STORM_CONCURRENCY MIXED_CONCURRENCY MAX_NUM_SEQS`.
Launch interface (tune division): `FLEET_LAUNCHER` takes a band argument; its rank script
supports `DRY_RUN=1` (print resolved argv, launch nothing). Non-conforming → exit 2.

## Coverage matrix, the entire corpus, accounted for
| corpus artifact | command | notes |
|---|---|---|
| gx10bench.py (4 engines, 12 lanes) | `bench lanes` / `bench run` | the 3-repos-combined tool |
| longctx_probe.py A/B/C | `bench probe-a/b/c` | regime probes at the profile's band |
| bst_parity.py | `bench parity` | community apples-to-apples protocol |
| accept.py / accept_live.py | `bench accept` / `bench run health` | sweep harness / live lane |
| accept_snapshot.py | `bench snapshot` | EKG → ledger |
| preflight-live.sh + preflight.sh | `tune preflight` | manifest-vs-live gate |
| verify_recovery_fidelity.sh | `tune fidelity` | DRY_RUN argv vs live; fail-closed |
| cross_model_audit.py | `tune audit` | divergence needs a recorded reason |
| restart_window (experiment_window.sh) | `tune window` | gated; cells per-campaign by design |
| watchdog quality floors | `tune floors` | status view; floors live in the watchdog |
| upstream ecosystems (HF + GitHub) | `upstream` | movement diff vs last-check stamp |
| ledgers, results, recipes, vendor refs, lessons | `corpus` | the knowledge map |
| k/ctx sweeps | superseded | absorbed into the window |
| sessbench3 / prefill_probe | niche | listed in `corpus` |
| ras_probe.sh | not yet wired to a verb, run directly | NCCL RAS query across a TP group, single-shot, no loop; classifies a live stuck job as cross-rank collective divergence vs something else. Proven against a live GLM-5.2 wedge before landing here, see honest limits below. |

## Honest limits (the refinement backlog before community release)
1. **Window cells are per-campaign**, the current cells are the DeepSeek k/greedy/mnbt
   experiments. Generalizing = cells move into the profile. Do it against GLM as the second
   real consumer, not speculatively.
2. **Quality floors watch one endpoint**, multi-model = loop profiles in check_quality.
3. **Metrics names are vLLM-v1**, llama.cpp/other engines need a metrics adapter.
4. **Pre-share scrub list**: profiles carry private IPs/hostnames (fine, profiles are
   local by design and example profiles ship sanitized), but several lib/ instruments
   still embed default IPs, absolute home paths, and internal memory references. `make
   dist` must vendor + scrub: default endpoints → required args, home paths → kit-relative,
   internal doc references → removed. Publish under the owner's handle per house rules.
5. **`ras_probe.sh` isn't wired to a verb yet.** It's a real, proven instrument (caught a
   genuine frozen cross-rank mismatch on a live GLM-5.2 wedge, confirmed by re-querying and
   getting the identical frozen numbers twice), but it landed here straight from that
   campaign and hasn't been generalized into `tune` the way the window cells were. The
   natural home is a `tune diag` verb that runs it alongside a memory snapshot on WEDGED,
   the same dual-snapshot pattern proven in that campaign: single-shot, never gates the
   verdict, logged raw. Do that against a second real wedge on a different model, not
   speculatively, same rule as the window cells above.


## Where this came from

I built this running DeepSeek-V4 and GLM-5.2 in production on a 4x NVIDIA DGX Spark
cluster, serving my coding agent's real workload. Every instrument in here earned its
place in a live campaign: the probe battery caught crashes that only fire on deep cold
prefill, the acceptance EKG diagnoses broken drafters in one read, the fidelity check
exists because a recovery path once nearly restored a demoted cluster and reported
success, and the audit refuses to let two of your own models disagree on a knob without
a recorded reason. Paths and profiles in this repo are my cluster's; a model is one
~15-key env file, so pointing it at yours is an afternoon, not a port.

Results it produced:
- https://github.com/joesinvestments/DeepSeek-V4-Flash-0731-TP4-4x-DGX-Spark
- https://github.com/joesinvestments/GLM-5.2-QuantTrio-TP4-DCP2-4x-DGX-Spark
