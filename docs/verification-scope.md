# Numerical verification and performance scope

The executable kernel is a parameterized TileLang family. `build(workload, config)`
compiles its PrimFunc to Metal for MPS tensors (CPU tensors are transferred to MPS),
and to sm_121a for compatible CUDA tensors. CUDA execution was not available in this
project's development environment and is not claimed as tested.

For each eligible candidate, the numerical subprocess calls the actual generated
module's build function, runs the compiled Metal shader on the full workload shape,
and compares every output element to independently computed CPU float64 GEMM rounded
to float16. Tests include two random seeds and structured zero/signed patterns.
Output is initialized with NaNs to expose unwritten regions. An integration test
changes the kernel arithmetic and confirms that it fails, rather than merely comparing
two references. Tolerances come from the workload; no mismatch fraction is allowed.

The kernel/config hashes, input TIR hash and Metal shader hash are saved. The GB10
compiler separately imports the same saved source/config; its input TIR must match
byte-for-byte. Actual cubin/SASS hashes and resource usage are then saved. Only
numerically passed, compilation-valid, same-IR candidates can become retained results.
Only completed, resource-valid simulator estimates can rank their performance.

`checked_target: metal` and `cuda_binary_executed: false` are present in result.json.
This verifies the exact source-level TileLang computation on a different backend,
not the CUDA binary or all code-generation/synchronization behavior on GB10. The
course assignment does not explicitly clarify acceptance of cross-backend numerical
checks; this scope remains visible in the report and is not claimed away.

For serpentine traversal, the actual kernel computes the same coordinate mapping as
the course simulator's launch_order. Unit tests compare the formulas, Z3 checks fixed
swizzled-grid bounds, and a finite check verifies that the tile mapping is bijective.
These are scoped integer properties; no whole-family floating-point or pipeline proof
is claimed. Original simulator files/parameters are hash-verified and unchanged.

The GB10 model's memory/cache calibration and instruction/control approximations
remain limitations. Model estimates are not hardware measurements; small differences
need hardware corroboration. The model's reported busy counters are not presented as
normalized hardware utilization. No simulator value substitutes for a numerical check.

Latest public-case records are the final ZIP's results/<workload-id>. Development
experiments and their failures are under results/history. Older runs are never
retroactively labeled checked. Source snapshots identify the exact code used per run.
