#!/usr/bin/env python3
"""Run post-install SuGaRrush checks and emit a machine-readable report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Check:
    name: str
    group: str
    command: list[str]
    description: str
    required: bool = True
    timeout: int = 120


def conda_python(conda: str, environment: str, source: str) -> list[str]:
    return [conda, "run", "-n", environment, "python", "-c", source]


def build_checks(
    repo: Path,
    conda: str,
    require_model: bool,
    include_gpu: bool,
) -> list[Check]:
    checks: list[Check] = [
        Check(
            "repository-layout",
            "source",
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    f"""
                    from pathlib import Path
                    root = Path({str(repo)!r})
                    required = [
                        "environment.yml",
                        "scripts/reconstruct_object.sh",
                        "scripts/select_sharp.py",
                        "gaussian_splatting/train.py",
                        "gaussian_splatting/submodules/diff-gaussian-rasterization/setup.py",
                        "gaussian_splatting/submodules/simple-knn/setup.py",
                    ]
                    missing = [item for item in required if not (root / item).is_file()]
                    assert not missing, "missing source files: " + ", ".join(missing)
                    print("repository layout ok")
                    """
                ),
            ],
            "Required source trees and pipeline entry points exist.",
        ),
        Check(
            "launcher-syntax",
            "source",
            ["bash", "-n", str(repo / "scripts/reconstruct_object.sh")],
            "The end-to-end launcher parses as valid Bash.",
        ),
        Check(
            "nvidia-driver",
            "host",
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            "The host NVIDIA driver exposes at least one GPU.",
        ),
        Check(
            "cuda-compiler",
            "toolchain",
            [str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc"), "--version"],
            "The selected private CUDA 11.8 compiler is executable.",
        ),
        Check(
            "host-compiler",
            "toolchain",
            [os.environ["CXX"], "--version"],
            "The selected GCC 11 C++ compiler is executable.",
        ),
        Check(
            "colmap-cli",
            "colmap",
            [conda, "run", "-n", "colmap", "colmap", "-h"],
            "COLMAP starts inside its isolated environment.",
        ),
        Check(
            "ffmpeg-functional",
            "colmap",
            [
                conda,
                "run",
                "-n",
                "colmap",
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=32x32:d=0.1",
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            "ffmpeg decodes and processes a synthetic frame.",
        ),
        Check(
            "segmentation-imports",
            "segmentation",
            conda_python(
                conda,
                "seg",
                (
                    "import cv2, onnxruntime, rembg; "
                    "from importlib.metadata import version; "
                    "print('rembg', version('rembg'), "
                    "'onnxruntime', onnxruntime.__version__, "
                    "'opencv', cv2.__version__)"
                ),
            ),
            "rembg, ONNX Runtime and OpenCV import together.",
        ),
        Check(
            "sugar-python-stack",
            "sugar",
            conda_python(
                conda,
                "sugar",
                textwrap.dedent(
                    """
                    import coacd
                    import nvdiffrast.torch
                    import numpy
                    import open3d
                    import plyfile
                    import pymeshfix
                    import pytorch3d
                    import scipy
                    import torch
                    import trimesh
                    import xatlas
                    from diff_gaussian_rasterization import GaussianRasterizer
                    from simple_knn import _C
                    print(
                        "torch", torch.__version__,
                        "torch-cuda", torch.version.cuda,
                        "pytorch3d", pytorch3d.__version__,
                        "open3d", open3d.__version__,
                    )
                    """
                ),
            ),
            "The pinned reconstruction, geometry and native modules import.",
        ),
        Check(
            "geometry-libraries",
            "geometry",
            conda_python(
                conda,
                "sugar",
                textwrap.dedent(
                    """
                    import numpy as np
                    import coacd
                    import open3d as o3d
                    import pymeshfix
                    import trimesh
                    import xatlas

                    box = trimesh.creation.box()
                    vertices = np.asarray(box.vertices, dtype=np.float64)
                    faces = np.asarray(box.faces, dtype=np.int32)

                    mesh = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(vertices),
                        o3d.utility.Vector3iVector(faces),
                    )
                    assert mesh.is_edge_manifold()

                    vmapping, indices, uvs = xatlas.parametrize(
                        vertices.astype(np.float32), faces.astype(np.uint32)
                    )
                    assert len(indices) == len(faces) and len(uvs) > 0

                    repaired = pymeshfix.MeshFix(vertices, faces)
                    repaired.repair(verbose=False)
                    assert len(repaired.v) > 0 and len(repaired.f) > 0

                    coacd.set_log_level("error")
                    parts = coacd.run_coacd(
                        coacd.Mesh(vertices, faces),
                        threshold=0.1,
                        resolution=1000,
                        preprocess_mode="off",
                    )
                    assert len(parts) > 0
                    print("geometry operations ok; coacd parts", len(parts))
                    """
                ),
            ),
            "Open3D, xatlas, MeshFix, trimesh and CoACD process a cube.",
            timeout=180,
        ),
    ]

    if require_model:
        checks.append(
            Check(
                "u2net-inference",
                "segmentation",
                conda_python(
                    conda,
                    "seg",
                    textwrap.dedent(
                        """
                        from PIL import Image
                        from rembg import new_session, remove
                        session = new_session("u2net")
                        image = Image.new("RGB", (64, 64), (127, 127, 127))
                        mask = remove(image, session=session, only_mask=True)
                        assert mask.size == image.size
                        print("U2Net inference ok", mask.size)
                        """
                    ),
                ),
                "U²-Net loads its cached weights and completes one inference.",
                timeout=300,
            )
        )

    if include_gpu:
        checks.extend(
            [
                Check(
                    "pytorch-cuda",
                    "cuda",
                    conda_python(
                        conda,
                        "sugar",
                        textwrap.dedent(
                            """
                            import torch
                            assert torch.version.cuda == "11.8", torch.version.cuda
                            assert torch.cuda.is_available(), "CUDA is unavailable to PyTorch"
                            a = torch.arange(256, device="cuda", dtype=torch.float32)
                            value = float((a @ a).cpu())
                            capability = torch.cuda.get_device_capability(0)
                            print(torch.cuda.get_device_name(0), capability, value)
                            """
                        ),
                    ),
                    "PyTorch sees the GPU and executes a CUDA reduction.",
                ),
                Check(
                    "cuda-extension-kernels",
                    "cuda",
                    conda_python(
                        conda,
                        "sugar",
                        textwrap.dedent(
                            """
                            import torch
                            from diff_gaussian_rasterization import (
                                GaussianRasterizationSettings,
                                GaussianRasterizer,
                            )
                            from simple_knn._C import distCUDA2

                            points = torch.rand((128, 3), device="cuda")
                            distances = distCUDA2(points)
                            assert distances.shape == (128,)

                            identity = torch.eye(4, device="cuda")
                            settings = GaussianRasterizationSettings(
                                image_height=8,
                                image_width=8,
                                tanfovx=1.0,
                                tanfovy=1.0,
                                bg=torch.zeros(3, device="cuda"),
                                scale_modifier=1.0,
                                viewmatrix=identity,
                                projmatrix=identity,
                                sh_degree=0,
                                campos=torch.zeros(3, device="cuda"),
                                prefiltered=False,
                                debug=False,
                            )
                            visible = GaussianRasterizer(settings).markVisible(
                                torch.tensor([[0.0, 0.0, 0.5]], device="cuda")
                            )
                            assert visible.numel() == 1
                            print("simple-knn and Gaussian rasterizer kernels ok")
                            """
                        ),
                    ),
                    "Both project-specific CUDA extensions launch kernels.",
                ),
                Check(
                    "pytorch3d-kernel",
                    "cuda",
                    conda_python(
                        conda,
                        "sugar",
                        textwrap.dedent(
                            """
                            import torch
                            from pytorch3d.ops import knn_points
                            points = torch.rand((1, 32, 3), device="cuda")
                            result = knn_points(points, points, K=2)
                            assert result.dists.shape == (1, 32, 2)
                            print("PyTorch3D CUDA kernel ok")
                            """
                        ),
                    ),
                    "PyTorch3D launches a CUDA nearest-neighbour kernel.",
                ),
                Check(
                    "nvdiffrast-kernel",
                    "cuda",
                    conda_python(
                        conda,
                        "sugar",
                        textwrap.dedent(
                            """
                            import torch
                            import nvdiffrast.torch as dr
                            context = dr.RasterizeCudaContext()
                            positions = torch.tensor(
                                [[[-0.8, -0.8, 0.0, 1.0],
                                  [ 0.8, -0.8, 0.0, 1.0],
                                  [ 0.0,  0.8, 0.0, 1.0]]],
                                device="cuda",
                                dtype=torch.float32,
                            )
                            triangles = torch.tensor(
                                [[0, 1, 2]], device="cuda", dtype=torch.int32
                            )
                            raster, _ = dr.rasterize(
                                context, positions, triangles, resolution=[16, 16]
                            )
                            assert raster.shape == (1, 16, 16, 4)
                            print("nvdiffrast CUDA kernel ok")
                            """
                        ),
                    ),
                    "nvdiffrast builds/loads its plugin and rasterizes a triangle.",
                    timeout=300,
                ),
            ]
        )
    return checks


def execute(check: Check, environment: dict[str, str]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            check.command,
            cwd=environment["SUGARRUSH_REPO"],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=check.timeout,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        status = "passed" if returncode == 0 else "failed"
    except (OSError, subprocess.TimeoutExpired) as exc:
        returncode = None
        stdout = getattr(exc, "stdout", "") or ""
        stderr = str(exc)
        status = "failed"
    return {
        "name": check.name,
        "group": check.group,
        "description": check.description,
        "required": check.required,
        "status": status,
        "returncode": returncode,
        "duration_seconds": round(time.monotonic() - started, 3),
        "command": shlex.join(check.command),
        "stdout": stdout[-8000:].strip(),
        "stderr": stderr[-8000:].strip(),
    }


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--conda-exe", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="skip the U²-Net model-load and inference check",
    )
    parser.add_argument(
        "--skip-gpu",
        action="store_true",
        help="skip CUDA kernel checks (for diagnostics only, not a valid full install)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list checks without executing them",
    )
    args = parser.parse_args()

    repo = args.repo.resolve()
    environment = os.environ.copy()
    environment["SUGARRUSH_REPO"] = str(repo)
    environment.setdefault(
        "NUMBA_CACHE_DIR", str(repo / ".setup/cache/numba")
    )

    missing_variables = [
        name for name in ("CUDA_HOME", "CC", "CXX") if not environment.get(name)
    ]
    if missing_variables:
        parser.error(
            "runtime environment is missing: " + ", ".join(missing_variables)
        )

    checks = build_checks(
        repo,
        args.conda_exe,
        require_model=not args.skip_model,
        include_gpu=not args.skip_gpu,
    )
    if args.list:
        for check in checks:
            print(f"{check.group:12} {check.name:28} {check.description}")
        return 0

    results: list[dict[str, Any]] = []
    for index, check in enumerate(checks, start=1):
        print(f"[{index:02d}/{len(checks):02d}] {check.name} ... ", end="", flush=True)
        result = execute(check, environment)
        results.append(result)
        print(
            f"{result['status'].upper()} ({result['duration_seconds']:.1f}s)",
            flush=True,
        )
        if result["status"] == "failed" and result["stderr"]:
            final_line = result["stderr"].splitlines()[-1]
            print(f"         {final_line[:240]}", flush=True)

    required_failures = [
        result
        for result in results
        if result["required"] and result["status"] != "passed"
    ]
    report = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo": str(repo),
        "conda_executable": str(Path(args.conda_exe).resolve()),
        "cuda_home": environment["CUDA_HOME"],
        "cc": environment["CC"],
        "cxx": environment["CXX"],
        "torch_cuda_arch_list": environment.get("TORCH_CUDA_ARCH_LIST", ""),
        "summary": {
            "total": len(results),
            "passed": sum(result["status"] == "passed" for result in results),
            "failed": sum(result["status"] == "failed" for result in results),
            "required_failures": len(required_failures),
            "status": "passed" if not required_failures else "failed",
        },
        "checks": results,
    }
    atomic_json(args.report, report)
    print(f"Report: {args.report}")
    if required_failures:
        print(
            "Required checks failed: "
            + ", ".join(result["name"] for result in required_failures),
            file=sys.stderr,
        )
        return 1
    print("All required setup checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
