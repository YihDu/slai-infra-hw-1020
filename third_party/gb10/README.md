# GB10 simulator and GEMM inputs

Extracted from AI-Infra-Co-Design at `a628c9d963cf9c029fa596b872d6e8920ca883e8`.
`SOURCE.json` records source paths, hashes and the TMA module extraction. No agent,
optimizer harness, solution, benchmark comparison or saved performance results are included.

- `workloads/*.json`: upstream GEMM input shapes, tensor names/dtypes and C=A@B semantics.
  `baseline_config` is empty; choose your implementation and search space.
- `agentic/gb10/sim/`: SASS parser, SM scheduler and cp.async/TMA timing simulator.
- `agentic/gb10/model.py`: dependency providing memory-traffic/architecture calculations.
- `calibration/spark-6951/gb10.json`: GB10 hardware/timing parameters and their sources.
- `examples/simulation-input.json`: input format for one compiled candidate.

CPU simulation needs Python 3.10+ and the standard library. Compile your candidate for
GB10 (`sm_121a`), export its encoded SASS, then supply its actual shape, tile choices,
kernel symbol and resource metadata in the JSON:

```sh
nvdisasm -hex kernel.cubin > examples/kernel.sass
python3 simulate.py --input examples/simulation-input.json --output prediction.json
```

`sass` is relative to the input JSON. `path` is `cpasync` or `tma`; `holes.threads`
is the consumer count on TMA (the upstream simulator adds 128 producer threads).
`regs` comes from compilation. Optional `smem_bytes` is total static + dynamic shared
memory per block; when omitted, the upstream cp.async path derives its stage-buffer
estimate. Do not pass static shared memory alone as the total. TMA uses the
upstream resident-one model; its whole-kernel function does not consume those two
fields. `warm` selects back-to-back warm-cache timing; `split` defaults to 1.
The example contains input metadata only: replace it to match your own compiled kernel.
This is a timing simulator for the upstream GEMM paths; it does not compute or check
output values. Workload tolerances are development defaults, not simulator parameters.
