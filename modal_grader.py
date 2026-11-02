"""Run the Assignment 2 checker on Modal's L4 GPU."""

from pathlib import Path
import subprocess
import sys

import modal


ROOT = Path(__file__).resolve().parent
app = modal.App("gstar-assignment-2-kda")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .workdir("/workspace")
)
# Mount Python files at container startup, outside the cached dependency image.
# Code edits do not rebuild the image; copy=False is also Modal's default.
for source in sorted(ROOT.glob("*.py")):
    image = image.add_local_file(
        str(source), f"/workspace/{source.name}", copy=False
    )


@app.function(image=image, gpu="T4", timeout=30 * 60)
def run_grader(problem: int = 1, all: bool = False, dtype: str = "float32",
               benchmark: bool = False, verbose: bool = False,
               p9_edge: bool = False):
    if not p9_edge and not all and problem not in range(1, 10):
        raise ValueError("problem must be between 1 and 9")
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("dtype must be float32, float16, or bfloat16")
    if p9_edge and benchmark:
        raise ValueError("p9_edge and benchmark are separate checks")

    if p9_edge:
        cmd = [sys.executable, "p9_edge_check.py", "--dtype", dtype]
    else:
        cmd = [sys.executable, "autograder.py", "--all" if all else f"--p{problem}",
               "--device", "cuda", "--dtype", dtype]
        if benchmark:
            cmd.append("--benchmark")
        if verbose:
            cmd.append("--verbose")

    print("$", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd="/workspace")
    if result.returncode != 0:
        raise RuntimeError(f"Autograder exited with code {result.returncode}")


@app.local_entrypoint()
def main(problem: int = 1, all: bool = False, dtype: str = "float32",
         benchmark: bool = False, verbose: bool = False,
         p9_edge: bool = False):
    run_grader.remote(problem, all, dtype, benchmark, verbose, p9_edge)
