# Production-shape probes

Added 2026-08-16 after a full-day sweep on GLM-5.2 / 4x DGX Spark where the
existing short-prompt probes gave confident answers to the wrong questions.

## Why these exist

The standard prose/peak probes send ~60-token prompts at C1/C4. That measures
decode in isolation and is **blind** to most things worth tuning:

- `max-num-seqs 8/16` -- C4 never fills more than 4 slots, so the lever is invisible
- `max-num-batched-tokens` -- prefill on a 60-token prompt is a rounding error
- KV pool size -- buys capacity, not decode speed
- cascade / shared-prefix optimizations -- nonce-busted prompts share nothing by construction

Every one of those produced a "no effect" result that was an artifact of the
instrument, not a property of the system.

## The probes

| probe | measures | use for |
|---|---|---|
| `midctx_probe.py` | decode AND prefill, split by streaming TTFT, at any context depth and concurrency | anything context- or concurrency-sensitive |
| `cachehit_probe.py` | TTFT on turn 1 vs turns 2+ of one growing conversation | whether prefix caching is working, and what it's worth |
| `cache_capacity_probe.py` | how many sessions stay resident before the oldest is evicted | the real operating envelope |
| `sharedprefix_probe.py` | N concurrent requests sharing one prefix | cascade attention, shared-prefix paths |
| `correctness_probe.py` | generated TEXT: truncation, repetition loops, known answers, plus concurrent batches | gate every config before believing its numbers |

## Three rules these encode

**1. Separate prefill from decode.** At 32K+ context the wall clock is dominated
by prefill, so `output_tokens / wall` is a prefill number wearing a decode label.
The first version of `midctx_probe` made exactly that mistake and reported
0.5 tok/s. Use streaming TTFT:

```
prefill_tok_s = prompt_tokens / TTFT
decode_tok_s  = completion_tokens / (wall - TTFT)
```

**2. Demand a long answer.** A prompt ending "summarize in two sentences" emits
~40 tokens against a 50K prefill, leaving no decode phase to measure.

**3. Discard the first batch after a boot.** Measured ~14% cold-start penalty,
nearly 3x the run-to-run noise floor, and it biases against whichever config is
measured first.

## Correctness gating

Throughput probes count tokens; a repetition loop still emits full `max_tokens`
at a healthy tok/s. `correctness_probe.py` reads the words.

Its concurrent phases exist because a config can pass every sequential single
request and still kill the engine on the first 4-concurrent batch -- observed
directly with `max-num-seqs 16` and a gapped cudagraph capture ladder.

The detector is unit-tested against synthetic corruption (1-word and 3-word
repetition loops, single-token domination, truncation) and stays silent on
healthy prose, so the predicate is known to be able to fire.

## Usage

```bash
export O14_BASE_URL=http://HEAD_IP:PORT O14_MODEL=your-model

python3 midctx_probe.py 8 16000 512 label      # C8, 16K ctx, 512 max tokens
python3 cachehit_probe.py 100000 3 label       # 100K prefix, 3 turns
python3 cache_capacity_probe.py 100000 6 label # 6 sessions at 100K
python3 sharedprefix_probe.py 16 16000 256 label
python3 correctness_probe.py HEAD_IP PORT label
```
