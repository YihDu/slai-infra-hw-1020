"""One simulator call from JSON; no search, agent, compilation or optimization harness."""

import argparse
import hashlib
import json
from pathlib import Path

from agentic.gb10.sim.kernel import simulate_kernel
from agentic.gb10.sim.sm import kernel_sass
from agentic.gb10.sim.whole_tma import kernel_us


def simulate(request, directory):
    sass = (directory / request["sass"]).resolve()
    symbol = request.get("kernel", "matmul_kernel")
    if not kernel_sass(sass, kernel=symbol):
        raise ValueError(
            "No encoded instructions for the requested kernel; use nvdisasm -hex"
        )
    h, shape = request["holes"], request["shape"]
    if any(type(shape[k]) is not int or shape[k] <= 0 for k in ("M", "N", "K")):
        raise ValueError("M/N/K must be positive integers")
    if any(
        type(h[k]) is not int or h[k] <= 0
        for k in ("block_M", "block_N", "block_K", "threads", "num_stages")
    ):
        raise ValueError(
            "Tile dimensions, threads and stages must be positive integers"
        )
    if request["path"] == "cpasync":
        result = simulate_kernel(
            sass,
            h,
            shape,
            request["regs"],
            smem=request.get("smem_bytes"),
            warm=request.get("warm", True),
            split=request.get("split", 1),
            kernel=symbol,
        )
    elif request["path"] == "tma":
        if request.get("split", 1) != 1:
            raise ValueError("TMA entry supports split=1")
        result = {
            "us": kernel_us(
                "",
                h,
                shape,
                warm=request.get("warm", True),
                sass_path=sass,
                kernel=symbol,
            )
        }
    else:
        raise ValueError("path must be cpasync or tma")
    return {
        "source": "gb10_instruction_simulator",
        "unit": "us",
        "sass_sha256": hashlib.sha256(sass.read_bytes()).hexdigest(),
        "simulation": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = simulate(json.loads(args.input.read_text()), args.input.resolve().parent)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
