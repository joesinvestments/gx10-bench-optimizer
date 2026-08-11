#!/usr/bin/env python3
"""Fail-closed dual-rank runtime, identity, semantic, and native CUDA-graph gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3
REGULAR_RUNTIME_MARKERS = (
    "CUDA_GRAPH_RUNTIME",
    "requested=regular",
    "resolved=regular",
    "breakable=false",
    "enforce_eager=false",
    "CUDA_GRAPH_CAPTURE implementation=regular status=complete",
    "CUDA_GRAPH_REPLAY implementation=regular status=observed",
    "Capturing model for DSpark speculator...",
)
BREAKABLE_RUNTIME_MARKERS = (
    "CUDA_GRAPH_RUNTIME",
    "requested=breakable",
    "resolved=breakable",
    "breakable=true",
    "enforce_eager=false",
    "Breakable CUDA graph enabled",
    "Capturing model for DSpark speculator...",
    "CUDA_GRAPH_CAPTURE implementation=breakable status=complete",
    "CUDA_GRAPH_REPLAY implementation=breakable status=observed",
)
NATIVE_BACKEND_MARKERS = ("DSpark", "nvfp4_ds_mla", "B12X", "NET/IB")
REGULAR_FORBIDDEN_MARKERS = (
    "Breakable CUDA graph enabled",
    "implementation=breakable",
    "resolved=breakable",
    "enforce_eager=true",
    "Cudagraph is disabled under eager mode",
    "decode_mode=NONE",
    "fallback to eager",
    "fallback to breakable",
    "Marlin",
    "emulation",
    "CUDA_GRAPH_AUXILIARY implementation=regular component=dflash mode=eager",
    "does not support full CUDA graphs; running the draft eagerly",
)
BREAKABLE_FORBIDDEN_MARKERS = (
    "resolved=regular",
    "enforce_eager=true",
    "Cudagraph is disabled under eager mode",
    "decode_mode=NONE",
    "fallback to eager",
    "Marlin",
    "emulation",
    # Note: DFlash may run as breakable auxiliary eager under PIECEWISE; that is
    # recorded in evidence logs but is not an automatic fallback of the selector.
)


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {value}")
        ),
    )


def command_has_json_option(
    command: str, option: str, expected_value: Any
) -> bool:
    """Match one shell option by parsed JSON value.

    The launcher may pass a *superset* of the expected compilation config
    (for example mode + cudagraph_mode + implementation + strict). Accept when
    every expected key is present with an equal value.
    """
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return False

    values = [
        tokens[index + 1]
        for index, token in enumerate(tokens[:-1])
        if token == option
    ]
    if len(values) != 1:
        return False

    try:
        actual_value = strict_loads(values[0])
    except (json.JSONDecodeError, ValueError, TypeError):
        return False
    if actual_value == expected_value:
        return True
    if isinstance(expected_value, dict) and isinstance(actual_value, dict):
        return all(actual_value.get(key) == value for key, value in expected_value.items())
    return False


def validate_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            validate_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_finite(child, f"{path}[{index}]")


def run(command: list[str], *, host: str | None = None, timeout: float = 60) -> str:
    # OpenSSH concatenates arguments after the destination and executes the
    # result through the remote shell.  Pass one shell-quoted command so Python
    # snippets containing parentheses/semicolons survive that boundary.
    effective = ["ssh", host, "--", shlex.join(command)] if host else command
    completed = subprocess.run(
        effective,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"command failed on {host or 'head'} ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def docker_logs(container: str, host: str | None) -> str:
    command = ["docker", "logs", "--tail", "20000", container]
    effective = ["ssh", host, "--", shlex.join(command)] if host else command
    completed = subprocess.run(
        effective,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"docker logs failed on {host or 'head'}: {completed.stderr.strip()}"
        )
    return completed.stdout + completed.stderr


def inspect_object(kind: str, identifier: str, host: str | None) -> dict[str, Any]:
    output = run(["docker", kind, "inspect", identifier], host=host)
    value = strict_loads(output)
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise RuntimeError(f"unexpected docker {kind} inspect response")
    return value[0]


def pid1_environment(container: str, host: str | None) -> set[str]:
    script = (
        "from pathlib import Path; "
        "data=Path('/proc/1/environ').read_bytes().split(b'\\0'); "
        "print('\\n'.join(sorted(x.decode(errors='strict') for x in data if x)))"
    )
    output = run(
        ["docker", "exec", container, "/usr/bin/python3", "-c", script],
        host=host,
    )
    return set(output.splitlines())


def container_file_sha256(container: str, path: str, host: str | None) -> str:
    script = (
        "import hashlib; from pathlib import Path; "
        f"print(hashlib.sha256(Path({path!r}).read_bytes()).hexdigest())"
    )
    return run(
        ["docker", "exec", container, "/usr/bin/python3", "-c", script],
        host=host,
    ).strip()


def request_json(
    url: str, payload: dict[str, Any] | None = None, timeout: float = 300
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = strict_loads(response.read().decode())
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"non-object response from {url}")
    validate_finite(value)
    return value


def chat_payload(model: str, prompt: str, max_tokens: int = 64) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"thinking": False},
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def marker_hits(logs: str, markers: tuple[str, ...]) -> dict[str, bool]:
    return {marker: marker.lower() in logs.lower() for marker in markers}


CUDA_GRAPH_RUNTIME_RE = re.compile(
    r"CUDA_GRAPH_RUNTIME "
    r"requested=(?P<requested>\S+) "
    r"resolved=(?P<resolved>\S+) "
    r"mode=(?P<mode>\S+) "
    r"decode_mode=(?P<decode_mode>\S+) "
    r"breakable=(?P<breakable>true|false) "
    r"enforce_eager=(?P<enforce_eager>true|false) "
    r"strict=(?P<strict>true|false)"
)


def authoritative_cuda_graph_runtime(logs: str) -> dict[str, Any]:
    """Parse the final worker-execution CUDA-graph resolution marker.

    Parent/API processes log the requested config before backend capability
    normalization. Worker_TP logs after model/backend initialization are the
    authoritative execution mode.
    """
    matches = [
        match
        for line in logs.splitlines()
        if "Worker_TP" in line
        for match in [CUDA_GRAPH_RUNTIME_RE.search(line)]
        if match is not None
    ]
    if not matches:
        raise ValueError("authoritative Worker_TP CUDA_GRAPH_RUNTIME marker missing")
    values = matches[-1].groupdict()
    return {
        "requested": values["requested"],
        "resolved": values["resolved"],
        "mode": values["mode"],
        "decode_mode": values["decode_mode"],
        "breakable": values["breakable"] == "true",
        "enforce_eager": values["enforce_eager"] == "true",
        "strict": values["strict"] == "true",
    }


def rank_identity(
    container: str,
    host: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    container_inspect = inspect_object("container", container, host)
    image_id = str(container_inspect.get("Image") or "")
    image_inspect = inspect_object("image", image_id, host)
    labels = image_inspect.get("Config", {}).get("Labels") or {}
    if not isinstance(labels, dict):
        raise RuntimeError("image labels are not an object")
    command = " ".join(
        str(value)
        for value in (
            container_inspect.get("Path"),
            *(container_inspect.get("Args") or []),
        )
    )
    pid_env = pid1_environment(container, host)
    legacy_entries = sorted(
        entry for entry in pid_env if entry.startswith("VLLM_USE_BREAKABLE_CUDAGRAPH=")
    )
    dsv4_multi_stream_entries = sorted(
        entry for entry in pid_env if entry.startswith("VLLM_DSV4_ENABLE_MULTI_STREAM=")
    )
    audit_destination = "/opt/r0b0tlab/runtime-manifest.json"
    audit_mounts = [
        mount
        for mount in container_inspect.get("Mounts", [])
        if isinstance(mount, dict) and mount.get("Destination") == audit_destination
    ]
    audit_mount_read_only = (
        len(audit_mounts) == 1 and audit_mounts[0].get("RW") is False
    )
    audit_manifest_sha256 = container_file_sha256(
        container, audit_destination, host
    )

    expected_labels = {
        "org.opencontainers.image.revision": args.expected_source_revision,
        "io.r0b0tlab.model.revision": args.expected_model_revision,
        "io.r0b0tlab.vllm.integrated.commit": args.expected_integrated_vllm_commit,
        "io.r0b0tlab.vllm.package.version": args.expected_vllm_version,
        "io.r0b0tlab.cudagraph.selector": "native-v1",
        "io.r0b0tlab.kv-cache.dtype": "nvfp4_ds_mla",
        "io.r0b0tlab.speculative.method": "dspark",
        "io.r0b0tlab.speculative.tokens": "6",
    }
    mismatched_labels = {
        key: {"expected": expected, "actual": labels.get(key)}
        for key, expected in expected_labels.items()
        if labels.get(key) != expected
    }
    expected_config_value = strict_loads(args.expected_compilation_config)
    expected_config = json.dumps(
        expected_config_value,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    config_present = command_has_json_option(
        command, "--compilation-config", expected_config_value
    )
    optimization_present = (
        f"--optimization-level {args.expected_optimization_level}" in command
    )
    performance_present = f"--performance-mode {args.expected_performance_mode}" in command

    return {
        "container_id": container_inspect.get("Id"),
        "image_id": image_id,
        "image_architecture": image_inspect.get("Architecture"),
        "source_revision": labels.get("org.opencontainers.image.revision"),
        "model_revision": labels.get("io.r0b0tlab.model.revision"),
        "vllm_version": labels.get("io.r0b0tlab.vllm.package.version"),
        "integrated_vllm_commit": labels.get("io.r0b0tlab.vllm.integrated.commit"),
        "cudagraph_implementation": args.expected_cudagraph_implementation,
        "cudagraph_mode_expected": args.expected_cudagraph_mode,
        "enforce_eager": False,
        "serialized_compilation_config": expected_config,
        "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "compilation_config_present": config_present,
        "optimization_level": args.expected_optimization_level,
        "optimization_level_present": optimization_present,
        "performance_mode": args.expected_performance_mode,
        "performance_mode_present": performance_present,
        "legacy_selector_pid1_entries": legacy_entries,
        "dsv4_multi_stream_pid1_entries": dsv4_multi_stream_entries,
        "runtime_audit_manifest_sha256": audit_manifest_sha256,
        "runtime_audit_manifest_read_only": audit_mount_read_only,
        "label_mismatches": mismatched_labels,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="deepseek-v4-flash-dspark")
    parser.add_argument("--container", default="dspark_vllm")
    parser.add_argument("--worker-host", required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument(
        "--expected-model-revision",
        default="9e165c30e2704aec5d9d593cce3eebd58bbef1cb",
    )
    parser.add_argument("--expected-integrated-vllm-commit", required=True)
    parser.add_argument(
        "--expected-vllm-version",
        default="0.26.0+dspark.sm121.3",
    )
    parser.add_argument("--expected-profile", required=True)
    parser.add_argument(
        "--expected-cudagraph-implementation",
        choices=("regular", "breakable"),
        default="regular",
    )
    parser.add_argument(
        "--expected-cudagraph-mode",
        choices=("FULL", "FULL_AND_PIECEWISE", "FULL_DECODE_ONLY", "PIECEWISE"),
        default="FULL_AND_PIECEWISE",
    )
    parser.add_argument("--expected-compilation-config", required=True)
    parser.add_argument("--expected-runtime-audit-manifest-sha256", required=True)
    parser.add_argument("--expected-optimization-level", choices=("2", "3"), required=True)
    parser.add_argument(
        "--expected-performance-mode",
        choices=("balanced", "throughput", "interactivity"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    evidence: dict[str, Any] = {
        "profile": args.expected_profile,
        "expected_cudagraph_implementation": args.expected_cudagraph_implementation,
    }

    try:
        identities = {
            "head": rank_identity(args.container, None, args),
            "worker": rank_identity(args.container, args.worker_host, args),
        }
        evidence["runtime_identity"] = identities
        for rank, identity in identities.items():
            if identity["image_id"] != args.expected_image_id:
                errors.append(f"{rank} image ID mismatch")
            if identity["image_architecture"] != "arm64":
                errors.append(f"{rank} image architecture is not arm64")
            if identity["label_mismatches"]:
                errors.append(f"{rank} image label mismatch")
            if not identity["compilation_config_present"]:
                errors.append(f"{rank} command lacks serialized compilation config")
            if not identity["optimization_level_present"]:
                errors.append(f"{rank} command lacks expected optimization level")
            if not identity["performance_mode_present"]:
                errors.append(f"{rank} command lacks expected performance mode")
            if identity["legacy_selector_pid1_entries"]:
                errors.append(f"{rank} PID 1 retains legacy CUDA-graph selector")
            if identity["dsv4_multi_stream_pid1_entries"] != [
                "VLLM_DSV4_ENABLE_MULTI_STREAM=0"
            ]:
                errors.append(f"{rank} PID 1 does not pin compile-safe DSV4 streams")
            if (
                identity["runtime_audit_manifest_sha256"]
                != args.expected_runtime_audit_manifest_sha256
            ):
                errors.append(f"{rank} runtime audit manifest checksum mismatch")
            if not identity["runtime_audit_manifest_read_only"]:
                errors.append(f"{rank} runtime audit manifest is not one read-only mount")
        if identities["head"]["image_id"] != identities["worker"]["image_id"]:
            errors.append("rank image IDs disagree")
        if (
            identities["head"]["serialized_compilation_config"]
            != identities["worker"]["serialized_compilation_config"]
        ):
            errors.append("rank compilation configurations disagree")
    except Exception as exc:
        evidence["runtime_identity"] = None
        errors.append(f"runtime identity gate failed: {exc}")

    try:
        models = request_json(f"{args.base_url.rstrip('/')}/v1/models")
        entries = models.get("data") if isinstance(models.get("data"), list) else []
        ids = [entry.get("id") for entry in entries if isinstance(entry, dict)]
        evidence["models"] = {"ids": ids, "entries": entries}
        if args.model not in ids:
            errors.append(f"served model {args.model!r} missing from /v1/models")
    except Exception as exc:
        evidence["models"] = None
        errors.append(str(exc))

    base = args.base_url.rstrip("/")
    try:
        semantic = request_json(
            f"{base}/v1/chat/completions",
            chat_payload(
                args.model,
                "Compute 7 multiplied by 19. Reply with only the number.",
            ),
        )
        choice = semantic["choices"][0]
        content = str(choice["message"].get("content") or "").strip()
        finish_reason = choice.get("finish_reason")
        passed = content == "133" and finish_reason == "stop"
        evidence["semantic"] = {
            "content": content,
            "finish_reason": finish_reason,
            "pass": passed,
        }
        if not passed:
            errors.append(
                f"semantic canary expected content='133'/finish_reason='stop', got {content!r}/{finish_reason!r}"
            )
    except Exception as exc:
        evidence["semantic"] = None
        errors.append(f"semantic canary failed: {exc}")

    try:
        tool_payload = chat_payload(args.model, "Use get_weather for Paris.")
        tool_payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Return weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        tool_payload["tool_choice"] = {
            "type": "function",
            "function": {"name": "get_weather"},
        }
        tool = request_json(f"{base}/v1/chat/completions", tool_payload)
        calls = tool["choices"][0]["message"].get("tool_calls") or []
        names = [
            call.get("function", {}).get("name")
            for call in calls
            if isinstance(call, dict)
        ]
        evidence["tool_call"] = {"names": names, "pass": "get_weather" in names}
        if "get_weather" not in names:
            errors.append(f"tool canary did not emit get_weather: {names}")
    except Exception as exc:
        evidence["tool_call"] = None
        errors.append(f"tool canary failed: {exc}")

    needle = "R0B0TLAB-DSVF-NVFP4-7319"
    filler = "A deterministic context sentence for retrieval validation. " * 900
    prompt = f"{filler}\nSecret code: {needle}\n{filler}\nReply with the secret code only."
    try:
        retrieval = request_json(
            f"{base}/v1/chat/completions",
            chat_payload(args.model, prompt, max_tokens=48),
            timeout=600,
        )
        choice = retrieval["choices"][0]
        content = str(choice["message"].get("content") or "").strip()
        finish_reason = choice.get("finish_reason")
        passed = needle in content and finish_reason == "stop"
        evidence["retrieval"] = {
            "content": content,
            "finish_reason": finish_reason,
            "usage": retrieval.get("usage"),
            "pass": passed,
        }
        if not passed:
            errors.append("retrieval canary did not return the exact code naturally")
    except Exception as exc:
        evidence["retrieval"] = None
        errors.append(f"retrieval canary failed: {exc}")

    try:
        logs_by_rank = {
            "head": docker_logs(args.container, None),
            "worker": docker_logs(args.container, args.worker_host),
        }
        required_markers = (
            REGULAR_RUNTIME_MARKERS
            if args.expected_cudagraph_implementation == "regular"
            else BREAKABLE_RUNTIME_MARKERS
        ) + NATIVE_BACKEND_MARKERS
        forbidden_markers = (
            REGULAR_FORBIDDEN_MARKERS
            if args.expected_cudagraph_implementation == "regular"
            else BREAKABLE_FORBIDDEN_MARKERS
        )
        runtime_evidence: dict[str, Any] = {}
        for rank, logs in logs_by_rank.items():
            required = marker_hits(logs, required_markers)
            forbidden = marker_hits(logs, forbidden_markers)
            resolved_runtime = authoritative_cuda_graph_runtime(logs)
            runtime_evidence[rank] = {
                "required": required,
                "forbidden": forbidden,
                "authoritative_cuda_graph_runtime": resolved_runtime,
                "log_bytes": len(logs.encode()),
                "log_sha256": hashlib.sha256(logs.encode()).hexdigest(),
            }
            if resolved_runtime["mode"] != args.expected_cudagraph_mode:
                errors.append(
                    f"{rank} resolved CUDA-graph mode mismatch: expected "
                    f"{args.expected_cudagraph_mode}, got {resolved_runtime['mode']}"
                )
            if resolved_runtime["resolved"] != args.expected_cudagraph_implementation:
                errors.append(
                    f"{rank} resolved CUDA-graph implementation mismatch: expected "
                    f"{args.expected_cudagraph_implementation}, got "
                    f"{resolved_runtime['resolved']}"
                )
            runtime_identity = evidence.get("runtime_identity")
            if isinstance(runtime_identity, dict):
                rank_identity_evidence = runtime_identity.get(rank)
                if isinstance(rank_identity_evidence, dict):
                    rank_identity_evidence["cudagraph_mode"] = resolved_runtime["mode"]
            for marker, present in required.items():
                if not present:
                    errors.append(f"{rank} required runtime marker missing: {marker}")
            for marker, present in forbidden.items():
                if present:
                    errors.append(f"{rank} forbidden runtime marker present: {marker}")
        evidence["runtime_logs"] = runtime_evidence
    except Exception as exc:
        evidence["runtime_logs"] = None
        errors.append(f"runtime log gate failed: {exc}")

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "evidence": evidence,
        "errors": errors,
    }
    validate_finite(result)
    atomic_write_json(args.output, result)
    print(json.dumps({"status": result["status"], "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
