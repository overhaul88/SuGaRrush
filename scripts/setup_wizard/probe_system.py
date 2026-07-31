#!/usr/bin/env python3
"""Probe a host for the SuGaRrush setup wizard.

This helper intentionally uses only Python's standard library so it can run
before any conda environment exists. It emits a detailed JSON profile and a
small shell-safe projection consumed by setup_wizard.sh.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


CUDA_TARGET = "11.8"
MIN_DRIVER_MAJOR = 520
MIN_DISK_GB = 12.0
RECOMMENDED_DISK_GB = 25.0


def run(command: list[str], timeout: int = 10) -> dict[str, Any]:
    """Run a non-interactive probe without raising."""
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in read_text(Path("/etc/os-release")).splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def memory_profile() -> dict[str, float]:
    values: dict[str, int] = {}
    for raw in read_text(Path("/proc/meminfo")).splitlines():
        match = re.match(r"^(\w+):\s+(\d+)\s+kB$", raw)
        if match:
            values[match.group(1)] = int(match.group(2))

    def gib(key: str) -> float:
        return round(values.get(key, 0) / (1024 * 1024), 2)

    return {
        "total_gib": gib("MemTotal"),
        "available_gib": gib("MemAvailable"),
        "swap_total_gib": gib("SwapTotal"),
        "swap_free_gib": gib("SwapFree"),
    }


def version_from_output(text: str, program: str) -> str | None:
    patterns = {
        "nvcc": r"\brelease\s+(\d+\.\d+)",
        "gcc": r"\b(\d+)(?:\.\d+){1,2}\b",
        "python": r"\b(\d+\.\d+(?:\.\d+)?)\b",
    }
    match = re.search(patterns[program], text)
    return match.group(1) if match else None


def executable_candidates(names: Iterable[str]) -> list[str]:
    result: list[str] = []
    for name in names:
        found = shutil.which(name)
        if found and found not in result:
            result.append(str(Path(found).resolve()))
    return result


def compiler_profile() -> dict[str, Any]:
    candidates = executable_candidates(
        ("gcc-11", "gcc-10", "gcc", "cc", "clang")
    )
    inspected: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for path in candidates:
        result = run([path, "--version"])
        version = version_from_output(
            f"{result['stdout']}\n{result['stderr']}", "gcc"
        )
        entry = {
            "path": path,
            "version": version,
            "returncode": result["returncode"],
        }
        inspected.append(entry)
        if selected is None and version:
            major = int(version.split(".", 1)[0])
            if major <= 11:
                selected = entry

    cxx_candidates = executable_candidates(
        ("g++-11", "g++-10", "g++", "c++", "clang++")
    )
    selected_cxx = None
    if selected:
        cc_name = Path(selected["path"]).name
        preferred = {
            "gcc-11": "g++-11",
            "gcc-10": "g++-10",
            "gcc": "g++",
            "cc": "c++",
            "clang": "clang++",
        }.get(cc_name)
        selected_cxx = shutil.which(preferred) if preferred else None
    if not selected_cxx and cxx_candidates:
        selected_cxx = cxx_candidates[0]

    return {
        "candidates": inspected,
        "selected_cc": selected["path"] if selected else None,
        "selected_cxx": (
            str(Path(selected_cxx).resolve()) if selected_cxx else None
        ),
        "private_fallback_required": selected is None,
    }


def cuda_profile() -> dict[str, Any]:
    candidates: list[str] = []
    env_home = os.environ.get("CUDA_HOME")
    if env_home:
        candidates.append(str(Path(env_home).expanduser()))
    nvcc_on_path = shutil.which("nvcc")
    if nvcc_on_path:
        candidates.append(str(Path(nvcc_on_path).resolve().parent.parent))
    candidates.extend(sorted(glob.glob("/usr/local/cuda-*"), reverse=True))
    if Path("/usr/local/cuda").exists():
        candidates.append("/usr/local/cuda")

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    selected = None
    for candidate in candidates:
        home = str(Path(candidate).resolve())
        if home in seen:
            continue
        seen.add(home)
        nvcc = Path(home) / "bin" / "nvcc"
        if not nvcc.is_file():
            continue
        result = run([str(nvcc), "--version"])
        version = version_from_output(
            f"{result['stdout']}\n{result['stderr']}", "nvcc"
        )
        entry = {
            "home": home,
            "nvcc": str(nvcc),
            "version": version,
            "returncode": result["returncode"],
        }
        unique.append(entry)
        if selected is None and version == CUDA_TARGET:
            selected = entry
    return {
        "target_version": CUDA_TARGET,
        "candidates": unique,
        "compatible_system_toolkit": selected,
        "private_fallback_required": selected is None,
    }


def gpu_profile() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    profile: dict[str, Any] = {
        "nvidia_smi": str(Path(nvidia_smi).resolve()) if nvidia_smi else None,
        "available": False,
        "gpus": [],
        "driver_minimum_met": False,
    }
    if not nvidia_smi:
        return profile

    fields = "index,name,driver_version,memory.total,compute_cap"
    result = run(
        [
            nvidia_smi,
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
    )
    has_compute_cap = result["returncode"] == 0
    if not has_compute_cap:
        fields = "index,name,driver_version,memory.total"
        result = run(
            [
                nvidia_smi,
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ]
        )
    if result["returncode"] != 0:
        profile["error"] = result["stderr"] or result["stdout"]
        return profile

    gpus: list[dict[str, Any]] = []
    for line in result["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        memory_match = re.search(r"[\d.]+", parts[3])
        compute_cap = parts[4] if has_compute_cap and len(parts) > 4 else None
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "driver_version": parts[2],
                "memory_mib": (
                    int(float(memory_match.group(0))) if memory_match else None
                ),
                "compute_capability": compute_cap,
            }
        )

    profile["gpus"] = gpus
    profile["available"] = bool(gpus)
    if gpus:
        majors = []
        for gpu in gpus:
            match = re.match(r"(\d+)", gpu["driver_version"])
            if match:
                majors.append(int(match.group(1)))
        profile["driver_minimum_met"] = bool(majors) and min(majors) >= MIN_DRIVER_MAJOR
    return profile


def conda_profile() -> dict[str, Any]:
    candidates: list[str] = []
    for variable in ("CONDA_EXE",):
        value = os.environ.get(variable)
        if value:
            candidates.append(value)
    candidates.extend(executable_candidates(("conda",)))

    home = Path.home()
    for path in (
        home / "miniforge3/bin/conda",
        home / "mambaforge/bin/conda",
        home / "miniconda3/bin/conda",
        home / "anaconda3/bin/conda",
    ):
        if path.is_file():
            candidates.append(str(path))

    seen: set[str] = set()
    inspected: list[dict[str, Any]] = []
    selected = None
    for raw in candidates:
        path = str(Path(raw).expanduser().resolve())
        if path in seen:
            continue
        seen.add(path)
        result = run([path, "info", "--json"], timeout=20)
        entry: dict[str, Any] = {
            "path": path,
            "returncode": result["returncode"],
        }
        if result["returncode"] == 0:
            try:
                data = json.loads(result["stdout"])
                entry["base_prefix"] = data.get("root_prefix")
                entry["version"] = data.get("conda_version")
                entry["envs"] = data.get("envs", [])
                if selected is None:
                    selected = entry
            except json.JSONDecodeError:
                entry["error"] = "conda returned invalid JSON"
        else:
            entry["error"] = result["stderr"]
        inspected.append(entry)
    return {
        "candidates": inspected,
        "selected": selected,
        "bootstrap_required": selected is None,
    }


def recommended_jobs(memory_gib: float, cpu_count: int) -> int:
    # CUDA extension builds peak around 2–3 GiB per translation unit. Leave
    # headroom for the OS and conda solver instead of blindly using nproc.
    if memory_gib < 6:
        by_memory = 1
    elif memory_gib < 12:
        by_memory = 2
    elif memory_gib < 20:
        by_memory = 3
    elif memory_gib < 32:
        by_memory = 4
    else:
        by_memory = 8
    return max(1, min(cpu_count or 1, by_memory, 8))


def gpu_arch_list(gpus: list[dict[str, Any]]) -> str:
    capabilities = sorted(
        {
            gpu["compute_capability"]
            for gpu in gpus
            if gpu.get("compute_capability")
            and re.fullmatch(r"\d+\.\d+", gpu["compute_capability"])
        },
        key=lambda value: tuple(int(part) for part in value.split(".")),
    )
    if not capabilities:
        return ""
    capabilities[-1] = capabilities[-1] + "+PTX"
    return ";".join(capabilities)


def build_profile(repo: Path) -> dict[str, Any]:
    release = os_release()
    memory = memory_profile()
    disk = shutil.disk_usage(repo)
    gpu = gpu_profile()
    cuda = cuda_profile()
    compiler = compiler_profile()
    conda = conda_profile()
    cpu_count = os.cpu_count() or 1
    kernel_text = (
        f"{platform.release()} {read_text(Path('/proc/version'))}".lower()
    )
    wsl = "microsoft" in kernel_text or "wsl" in kernel_text
    system = platform.system().lower()
    machine = platform.machine().lower()
    disk_free_gib = round(disk.free / (1024**3), 2)

    blockers: list[str] = []
    warnings: list[str] = []
    if system != "linux":
        blockers.append("The automated pipeline installer currently supports Linux and WSL2 only.")
    if machine not in {"x86_64", "amd64"}:
        blockers.append(
            f"Architecture {machine!r} is unsupported by the pinned PyTorch3D/CUDA stack."
        )
    if not gpu["available"]:
        blockers.append(
            "No NVIDIA GPU is visible through nvidia-smi; install/repair the host driver first."
        )
    elif not gpu["driver_minimum_met"]:
        blockers.append(
            f"The NVIDIA driver must be new enough for CUDA 11.8 (major >= {MIN_DRIVER_MAJOR})."
        )
    if disk_free_gib < MIN_DISK_GB:
        blockers.append(
            f"Only {disk_free_gib:.1f} GiB is free; at least {MIN_DISK_GB:.0f} GiB is required."
        )
    elif disk_free_gib < RECOMMENDED_DISK_GB:
        warnings.append(
            f"Only {disk_free_gib:.1f} GiB is free; {RECOMMENDED_DISK_GB:.0f}+ GiB is recommended."
        )
    if memory["total_gib"] and memory["total_gib"] < 8.0:
        warnings.append(
            "Less than 8 GiB RAM is available; native builds and reconstruction may be unstable."
        )
    for device in gpu["gpus"]:
        memory_mib = device.get("memory_mib")
        if memory_mib and memory_mib < 4096:
            warnings.append(
                f"{device['name']} has {memory_mib} MiB VRAM; 4 GiB is the tested minimum."
            )
    if cuda["private_fallback_required"]:
        warnings.append(
            "No system CUDA 11.8 toolkit was selected; the wizard will use its private conda toolchain."
        )
    if compiler["private_fallback_required"]:
        warnings.append(
            "No CUDA-11.8-compatible host GCC was selected; the wizard will use conda GCC 11."
        )

    return {
        "schema_version": 1,
        "repo": str(repo.resolve()),
        "platform": {
            "system": system,
            "machine": machine,
            "kernel": platform.release(),
            "wsl": wsl,
            "distribution": release,
        },
        "cpu": {
            "logical_count": cpu_count,
            "machine": platform.processor() or platform.machine(),
        },
        "memory": memory,
        "disk": {
            "path": str(repo.resolve()),
            "free_gib": disk_free_gib,
            "total_gib": round(disk.total / (1024**3), 2),
        },
        "gpu": gpu,
        "cuda": cuda,
        "compiler": compiler,
        "conda": conda,
        "recommendations": {
            "max_jobs": recommended_jobs(memory["total_gib"], cpu_count),
            "torch_cuda_arch_list": gpu_arch_list(gpu["gpus"]),
            "cuda_version": CUDA_TARGET,
            "use_private_cuda_toolchain": cuda["private_fallback_required"],
            "use_private_compiler": compiler["private_fallback_required"],
        },
        "blockers": blockers,
        "warnings": warnings,
    }


def shell_projection(profile: dict[str, Any]) -> str:
    selected_conda = profile["conda"]["selected"] or {}
    selected_cuda = profile["cuda"]["compatible_system_toolkit"] or {}
    gpu_names = ", ".join(gpu["name"] for gpu in profile["gpu"]["gpus"])
    driver_versions = ", ".join(
        sorted({gpu["driver_version"] for gpu in profile["gpu"]["gpus"]})
    )
    values = {
        "PROBE_OS": profile["platform"]["system"],
        "PROBE_ARCH": profile["platform"]["machine"],
        "PROBE_WSL": "1" if profile["platform"]["wsl"] else "0",
        "PROBE_DISTRO": profile["platform"]["distribution"].get("PRETTY_NAME", ""),
        "PROBE_CPU_COUNT": str(profile["cpu"]["logical_count"]),
        "PROBE_RAM_GIB": str(profile["memory"]["total_gib"]),
        "PROBE_SWAP_GIB": str(profile["memory"]["swap_total_gib"]),
        "PROBE_DISK_FREE_GIB": str(profile["disk"]["free_gib"]),
        "PROBE_GPU_COUNT": str(len(profile["gpu"]["gpus"])),
        "PROBE_GPU_NAMES": gpu_names,
        "PROBE_DRIVER_VERSIONS": driver_versions,
        "PROBE_ARCH_LIST": profile["recommendations"]["torch_cuda_arch_list"],
        "PROBE_MAX_JOBS": str(profile["recommendations"]["max_jobs"]),
        "PROBE_CONDA_EXE": selected_conda.get("path", ""),
        "PROBE_CONDA_BASE": selected_conda.get("base_prefix", ""),
        "PROBE_SYSTEM_CUDA_HOME": selected_cuda.get("home", ""),
        "PROBE_SYSTEM_CUDA_VERSION": selected_cuda.get("version", ""),
        "PROBE_SYSTEM_CC": profile["compiler"].get("selected_cc") or "",
        "PROBE_SYSTEM_CXX": profile["compiler"].get("selected_cxx") or "",
        "PROBE_BLOCKER_COUNT": str(len(profile["blockers"])),
        "PROBE_WARNING_COUNT": str(len(profile["warnings"])),
        "PROBE_BLOCKERS": "\n".join(profile["blockers"]),
        "PROBE_WARNINGS": "\n".join(profile["warnings"]),
    }
    return "".join(
        f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()
    )


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="repository root used for disk and path checks",
    )
    parser.add_argument("--json-out", type=Path, help="write the full JSON profile")
    parser.add_argument("--shell-out", type=Path, help="write shell-safe assignments")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="print compact JSON instead of indented JSON",
    )
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    if not repo.is_dir():
        parser.error(f"repository path does not exist: {repo}")

    profile = build_profile(repo)
    encoded = json.dumps(
        profile,
        indent=None if args.compact else 2,
        sort_keys=True,
    ) + "\n"
    if args.json_out:
        atomic_write(args.json_out, encoded)
    if args.shell_out:
        atomic_write(args.shell_out, shell_projection(profile))
    if not args.json_out:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
