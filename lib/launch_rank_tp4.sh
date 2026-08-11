#!/usr/bin/env bash
set -euo pipefail
required=(RANK MODE HOST_IP MASTER_ADDR MASTER_PORT FABRIC_INTERFACE RDMA_HCA MODEL_PATH
          SERVED_MODEL IMAGE MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
          GPU_MEMORY_UTILIZATION API_PORT NNODES)
for n in "${required[@]}"; do [[ -z "${!n:-}" ]] && { echo "missing $n" >&2; exit 1; }; done
if [[ "$RANK" == "0" ]]; then
  container=deepseek-v4-0731-vllm-head; port=$API_PORT; headless=()
else
  container=deepseek-v4-0731-vllm-worker; port=$((API_PORT + 1 + RANK)); headless=(--headless)
fi
if [[ -n "${SPEC_CONFIG_B64:-}" ]]; then
  SPEC_CONFIG="$(echo "$SPEC_CONFIG_B64" | base64 -d)"
else
  # STANDING CONFIG 2026-08-01: dynamic (batch-aware) speculative decoding.
  # k=7 at 1-4 lanes (95% of traffic, identical to the old static config),
  # k=5 at 5-8, k=3 at 9-12 -> worst-case decode batch 48 rows < the old 64.
  #
  # ★★ draft_sample_method WAS `greedy` HERE UNTIL 2026-08-06, AND THAT WAS A LANDMINE.
  # `greedy` makes the drafter a point mass, so acceptance collapses to p_target(argmax).
  # It cost us weeks at 27-48% acceptance. The LIVE server has run `probabilistic` since
  # 2026-08-05 (measured: 56.2% acceptance, per-position 0.9061/0.8726/0.8468/0.8162/
  # 0.8026/0.7441/0.6992), but THIS FILE still said greedy — so the watchdog's automatic
  # recovery path would have silently reinstated the bug and then reported success.
  # The value now matches production. If you change it, change it in a measured A/B and
  # write the result next to it. Verified by scripts/verify_recovery_fidelity.sh.
  #
  # NOTE k=7 vs the checkpoint's declared dspark_block_size=5: deliberate and measured.
  # Positions 5-6 accept at 74.4%/69.9% conditional and contribute 14.16% of all accepted
  # tokens; clamping to k=5 would cost 11.3% of tok/step. vLLM does NOT clamp and does not
  # warn. Do not "fix" this to 5 without re-measuring.
  # ★★ CHAMPION 2026-08-08 (window-20260808-182629, all k∈{3,5,7,8} measured, 24 probes +
  # 6 soaks, zero errors): STATIC k=7, NO ladder. The batch-aware ladder was measured as
  # pure overhead at real concurrency (-6.8% at C=1); k=7 beats k5 by 39% and k3 by 69%
  # on this image (kernels are dspark7-specialized — community k=3/5 advice does NOT
  # transfer). Do not re-add the ladder or change k without re-running the window.
  SPEC_CONFIG='{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
fi
speculative=(--speculative-config "$SPEC_CONFIG")

# EXTRA_SERVE_ARGS — optional passthrough appended to `vllm serve`. Added 2026-08-04 so a
# staged experiment can inject a flag WITHOUT editing this file. Unset = empty array =
# byte-identical launch, so the watchdog's auto-recovery path is unaffected.
# Space-split, so values must not contain spaces (JSON with no spaces is fine, e.g.
#   EXTRA_SERVE_ARGS='--override-generation-config {"temperature":0.0}'
extra_serve=()
[[ -n "${EXTRA_SERVE_ARGS:-}" ]] && read -r -a extra_serve <<< "$EXTRA_SERVE_ARGS"
graph_capture="${GRAPH_CAPTURE:-96}"

# ── the serve argv, declared ONCE ────────────────────────────────────────────
# ★ WHY AN ARRAY (2026-08-06): DRY_RUN below prints exactly what would be launched, and it
# must print THE SAME TOKENS the launch uses — not a hand-maintained copy of them. A second
# copy is precisely the drift that let the recovery path fall out of sync with production
# for two days without anyone being able to see it. One array, two consumers.
serve_args=(
    /model --served-model-name "$SERVED_MODEL" --host 0.0.0.0 --port "$port"
    --trust-remote-code --tensor-parallel-size 4 --pipeline-parallel-size 1
    --kv-cache-dtype nvfp4_ds_mla --block-size 256
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-cudagraph-capture-size "$graph_capture"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --enable-prefix-caching --async-scheduling --enable-chunked-prefill
    --tokenizer-mode deepseek_v4 --distributed-executor-backend mp
    --moe-backend flashinfer_b12x --tool-call-parser deepseek_v4 --enable-auto-tool-choice
    --reasoning-parser deepseek_v4
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
    --default-chat-template-kwargs '{"thinking":false}' --enable-flashinfer-autotune
    --generation-config vllm --nnodes "$NNODES" --node-rank "$RANK"
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
    "${speculative[@]}" ${extra_serve[@]+"${extra_serve[@]}"} ${headless[@]+"${headless[@]}"}
)

# DRY_RUN=1 prints the resolved argv, one token per line, and launches NOTHING. This is what
# verify_recovery_fidelity.sh diffs against the running container's docker-inspect .Args.
if [[ -n "${DRY_RUN:-}" ]]; then
    printf '%s\n' "${serve_args[@]}"
    exit 0
fi

# ★ 2026-08-08 env notes (a comment cannot live inside the docker-run continuation):
#   NCCL_IB_GID_INDEX=0  — ★★ REVERTED 2026-08-09 AFTER A 10-HOUR OUTAGE. I changed the
#     proven-working 0 to 3 ("correct" RoCE v2 index, verified symmetric at the time).
#     Then the recovery path's rail-IP flush+re-add REBUILT the GID tables and index 3
#     moved to 4 on three nodes — every recovery boot failed NCCL init all night. GID
#     indexes are NOT stable across IP re-application. The dsfv4_launch.sh note said
#     exactly this and said DO NOT change the pin. 0 works with ROCE_VERSION_NUM=2 on
#     this stack and has weeks of uptime behind it. NEVER hardcode a GID index other
#     than 0 here; if v2-index correctness ever matters, resolve it DYNAMICALLY at
#     launch like the GLM launcher does.
#   NCCL_IB_QPS_PER_CONNECTION=1 + NCCL_CROSS_NIC=0 — ★ WEDGE MITIGATION step 1
#     (staged 2026-08-09). Root cause of the recurring engine freezes (4+ events,
#     08-07→08-09): NCCL net_ib/p2p.cc receiver request-matching desync — completions
#     arrive for already-retired 'unused' request slots; the resiliency path tolerates
#     each one but the collective never completes -> all ranks block, engine frozen,
#     HTTP alive. Terminal log signature (93-231 repeats, all remote workers):
#       NET/IB: ncclIbCompletionEventProcess: Receiver got a completion for a CTS
#       but retrieved an 'unused' request
#     NCCL 2.30.7 (current); zero public issues match — upstream report filed from
#     crash-logs/20260809-150445 evidence. These two vars shrink the QP/completion
#     matching surface. If wedges persist: step 2 = single-rail (RDMA_HCA one device).
#   NCCL_DEBUG=INFO,SUBSYS=INIT,NET — baked so EVERY boot logs its fabric binding proof.
#     The experiment window tried passing these through the launch env and they silently
#     vanished: this docker run forwards ONLY its explicit -e list; env vars in the ssh
#     command string do NOT propagate into the container. UCX rcache flags: same story.
docker rm -f "$container" >/dev/null 2>&1 || true
docker run -d --name "$container" --gpus all --ipc host --network host \
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576:1048576 \
  --security-opt label=disable \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e VLLM_HOST_IP="$HOST_IP" \
  -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e ENABLE_VLLM_GB10_PATCH=0 -e GB10_HYBRID_NVFP4_M_THRESHOLD=128 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_B12X_MOE=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e CUTE_DSL_ARCH=sm_121a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e FLASHINFER_WORKSPACE_BASE=/cache/huggingface/flashinfer \
  -e TILELANG_CLEANUP_TEMP_FILES=1 -e DG_JIT_USE_NVRTC=0 -e DG_JIT_NVCC_COMPILER=/usr/local/cuda/bin/nvcc \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e GLOO_SOCKET_IFNAME="$FABRIC_INTERFACE" -e TP_SOCKET_IFNAME="$FABRIC_INTERFACE" \
  -e NCCL_NET=IB -e NCCL_SOCKET_IFNAME="$FABRIC_INTERFACE" -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$RDMA_HCA" -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_GID_INDEX=0 -e NCCL_IB_TIMEOUT=22 -e NCCL_IB_RETRY_CNT=7 -e NCCL_CROSS_NIC=0 \
  -e NCCL_IB_QPS_PER_CONNECTION=1 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET \
  -e UCX_MEM_MMAP_HOOK_MODE=none -e UCX_RCACHE_MAX_UNRELEASED=1024 \
  -v "$MODEL_PATH:/model:ro" -v "$HOME/.cache/huggingface:/cache/huggingface" \
  --entrypoint /bin/bash "$IMAGE" -lc '
    export PATH="/usr/local/cuda/bin:/usr/local/bin:${PATH:-}"
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"; export CUDA_PATH="$CUDA_HOME"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    exec /usr/local/bin/vllm serve "$@"' bash "${serve_args[@]}"
echo "started $container (rank $RANK of $NNODES, TP=4)"
