# Agentic GB10 optimizer — submission report

The final delivery retains the best complete observed run per workload (two v3 runs,
three v2 runs), with original snapshots and traces. `results/selection.json` identifies
provenance. Latest API/SMT feedback changes are separately tested; historical performance
is not attributed to a new five-case rerun. Earlier experiments, including regressions,
remain under `results/history/` in the ZIP.

## Result and validation boundary

The five published GEMM workloads completed through real DeepSeek formulation,
HiGHS selection, generated TileLang execution/checking and the course GB10 simulator.
Numerical tests execute **the exact generated TileLang source and configuration on
Apple Metal**. The CUDA compiler receives byte-identical input TileLang IR.
`checked` has this explicit Metal scope; no GB10/CUDA binary was executed. Cross-backend
checks do not prove CUDA code-generation or hardware correctness. The course document
does not explicitly settle whether this validation scope is sufficient for its grading;
this limitation is not hidden behind the status field.

Hardware: Apple M4 / 16 GiB / macOS 26.5.1, with a 4-vCPU/6-GiB ARM64 Linux VM for
cross-compilation. There is no NVIDIA GPU. Native TileLang 0.1.15 / PyTorch 2.11.0
execute through Metal; CUDA nvcc 13.0.88 targets sm_121a; nvdisasm 13.4.92 exports real
encoded SASS. Full versions, sources and compiled image IDs are retained.

All times below are **GB10 simulator estimates in μs**, not Mac/GB10 hardware measurements.
Speedup compares the first numerically checked candidate and the best eligible candidate
under the same source family, simulator and test protocol; it is not a cuBLAS comparison.

| Workload | Initial μs | Best μs | Simulated ratio | Usable candidates | Search seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| gemm_128x4096x4096 | 157.292 | 155.921 | 1.009× | 11 | 252.23 |
| gemm_2048x2048x2048 | 234.820 | 203.694 | 1.153× | 7 | 81.84 |
| gemm_32x4096x4096 | 195.806 | 147.524 | 1.327× | 9 | 193.09 |
| gemm_4096x4096x1024 | 470.776 | 399.545 | 1.178× | 8 | 140.02 |
| gemm_4096x4096x4096 | 2558.094 | 1597.230 | 1.602× | 8 | 391.61 |

Numerical checks: 129 full-shape cases across the candidates.
Each candidate is tested with independent normal inputs at seeds 42 and 7 and a structured
zero-row/signed-column pattern at seed 42. FP16 inputs and output, FP32 TileLang accumulation;
reference is a separate CPU PyTorch float64 full matmul rounded to FP16. Every output element
is checked against the workload tolerance, with NaN initialization to detect unwritten values.
The test suite also deliberately corrupts a generated shader and verifies numerical rejection.

Final acceptance: 27 automated tests pass, including actual Metal execution, deliberate
numerical corruption, three protocol adapters, retry accounting and concrete SMT witnesses.
Four local protocol fixtures (Responses, Chat, Anthropic and legacy Chat) drive the full
entry point with only a TA-style key variable, compiler and simulator. Fixture runs are
contract evidence, not performance evidence or access to a real TA endpoint. A separate
real DeepSeek API call passed using a renamed key variable. See setup/acceptance.json.

Historical v2 acceptance also includes an unseen 96×192×96 GEMM and fresh Python
CPU/Metal environments after archive extraction. That two-round run took 17.92 seconds
on the same Mac with the prepared compiler image. This historical environment installation
check is not relabeled as a fresh-machine test of the final archive.

## Agent/MIP formulation and implementation

This is a custom Python/httpx harness, used instead of an interactive CLI to make model access,
deadlines and artifacts explicit. Model: `deepseek-flash`, configured Chat Completions API,
thinking disabled; credentials are provided at runtime. Disk TileLang/autotuning caches are disabled during final runs. No personal agent settings/history
are loaded. Three declared protocol adapters are tested; public runs use DeepSeek.

The Agent selects tile/stage/thread domains, a shared-memory cap and positive analytic
objective weights. The deterministic builder creates one binary variable per allowed tuple,
requires exactly one selection, and imposes the resource cap. HiGHS solves this MIP and saves
LP/MPS, incumbent, bound and gap. Tile choices actually control the TileLang PrimFunc, launch
geometry, buffer sizes, GEMM and pipeline. Previously evaluated points are excluded, so the
loop evaluates distinct selected candidates rather than exhaustively measuring and encoding
a known winner. Optimality is only within the explicit finite proxy formulation.

Current catalog: block_M={16,32,64,128}, block_N={32,64,128,256}, block_K={32,64,128}, stages={1,2,3,4},
threads={64,128,256}, serpentine row-panel swizzle={0,4,8,16,32,64}.
Historical v2 runs use the narrower catalog captured in their source snapshots. The first model plan uses
row-major swizzle=0 as a conventional comparison baseline; later plans explore all
allowed swizzles. Only tiles dividing the case dimensions are included. A 32768-byte portability
cap permits the same source to execute on Metal. The memory estimate is
max(2*block_K*(block_M+block_N)*stages, 4*block_M*block_N), reflecting separated buffer lifetimes
and observed compiler aliasing; actual compiled resources and actual Metal execution are
checked separately. This excludes 128×128 FP32 epilogues and does not claim to cover every
fast GB10 kernel. Thread counts are searched; split-K and TMA are not.

The objective combines normalized traffic (with an analytical panel-locality reuse term),
launch-count, inverse pipeline-depth and memory
footprint features. It is dimensionless, not a latency prediction. The Agent receives previous
selections, numeric/SMT results, actual compiler resources and simulator estimates, then revises
its formulation. Improving a proxy objective does not guarantee lower simulated latency.

## One complete decision trace

Case: `gemm_4096x4096x4096`. Full trace is `results/gemm_4096x4096x4096/trace.jsonl`.

1. Workload tensors/precision are read; source/prompt/config snapshots are saved.
2. Agent plan `attempt-00` weights: `{"traffic": 1, "launches": 1, "pipeline": 1, "footprint": 1}`.
3. HiGHS selects `{"block_M": 128, "block_N": 64, "block_K": 64, "num_stages": 1, "threads": 128, "swizzle": 0}`; LP/MPS and solver results are saved.
4. The generated source executes on Metal and passes all numerical cases; identical input IR
   compiles to sm_121a and its actual SASS simulates at 2558.094 μs.
5. Observations are included in the next model request. Revisions/exclusions lead to plan
   `attempt-02` weights `{"traffic": 1.5, "launches": 1, "pipeline": 1.5, "footprint": 1}` and choice `{"block_M": 64, "block_N": 128, "block_K": 32, "num_stages": 2, "threads": 128, "swizzle": 16}`.
6. That implementation passes the same checks and simulates at 1597.230 μs.
   The source/config/TIR/SASS identities are in result.json and evaluation.json; only eligible
   candidates can replace the checkpointed best. Later failures or timeouts preserve it.

## Proxy disagreement and failed attempts

To compare like with like, the **first plan's fixed weights** are reapplied to all measured
feature vectors; costs from differently weighted rounds are not directly compared as latency.
The resulting preferred points versus observed winners are:

- gemm_128x4096x4096: first-plan proxy prefers attempt-01 among measured candidates (186.637 μs); observed best is attempt-11 (155.921 μs).
- gemm_2048x2048x2048: first-plan proxy prefers attempt-01 among measured candidates (235.125 μs); observed best is attempt-04 (203.694 μs).
- gemm_32x4096x4096: first-plan proxy prefers attempt-01 among measured candidates (176.447 μs); observed best is attempt-11 (147.524 μs).
- gemm_4096x4096x1024: first-plan proxy prefers attempt-01 among measured candidates (402.232 μs); observed best is attempt-02 (399.545 μs).
- gemm_4096x4096x4096: first-plan proxy prefers attempt-01 among measured candidates (1616.240 μs); observed best is attempt-02 (1597.230 μs).

This exposes cost-model limitations without fabricating a microsecond prediction or a
hardware-calibrated error bound. Limited samples and small simulated improvements do not establish
real-hardware speedups. The model's bandwidth/cache assumptions and mapping approximations remain
limitations, particularly for fine differences between candidates.

Incomplete/rejected public-run observations: `[{"workload": "gemm_128x4096x4096", "attempt": "attempt-04", "stage": "MIP", "status": "Infeasible", "numerical_status": null, "simulation_status": null}, {"workload": "gemm_2048x2048x2048", "attempt": "attempt-07", "stage": "MIP", "status": "Infeasible", "numerical_status": null, "simulation_status": null}, {"workload": "gemm_32x4096x4096", "attempt": "attempt-05", "stage": null, "status": null, "numerical_status": "not_run", "simulation_status": "not_run"}, {"workload": "gemm_32x4096x4096", "attempt": "attempt-08", "stage": "MIP", "status": "Infeasible", "numerical_status": null, "simulation_status": null}, {"workload": "gemm_32x4096x4096", "attempt": "attempt-09", "stage": "MIP", "status": "Infeasible", "numerical_status": null, "simulation_status": null}]`.
Raw failed/rejected plans and traces are retained. Development history additionally preserves
empty-model-output failures, missing Target-context compilation, Metal FP16 epilogue type errors,
numerical NaNs before explicit epilogue barriers, and an unsuccessful sliced-fragment
epilogue experiment (memory limit/type-layout errors) which was rejected rather than shipped. These development runs are not counted as
part of a single public-case budget or used as checked performance baselines.

## TVM, SMT and integrity

TileLang's bundled TVM lowers the program; input/device TIR and CUDA/Metal code are saved.
This assignment's published cases are single GEMMs, so no multi-operator graph-fusion benefit
is claimed. A general network importer is not provided.

Z3 checks fixed, divisible tile indices for out-of-bounds access, recording SMT2, assumptions,
shape/config and source hash. Swizzled tile bounds are also checked with Z3; complete
finite coordinate enumeration establishes bijective tile coverage. Unit tests compare
the mapping against the unchanged course launch_order, including partial panels.
Nondivisible counterexamples are tested. The final runtime saves SAT variable assignments
and violated bounds (or the reason for UNKNOWN), and includes these in subsequent Agent feedback. This is neither a
floating-point proof nor a pipeline/whole-family proof; numerical tests remain independent.
The official simulator and parameters retain their original SOURCE.json hashes. A wrapper
checks that every simulated warp completed, without changing timing parameters or the algorithm.
Unsupported/resource-invalid/incomplete/timeout results cannot enter performance rankings.

## Usage, prebuilt components and reproducibility

Aggregate public-run usage: `{"model_calls": 48, "input_tokens": 205092, "output_tokens": 25279, "cached_tokens": 166912, "reasoning_tokens": null, "cache_creation_tokens": null, "evaluations": 43, "compilations": 43, "simulations": 43, "numerical_checks": 43, "solver_calls": 48}`. Counts include failed model attempts; unavailable
provider metrics are null. These are API/tool records, not a billing estimate. Per-case elapsed
wall time includes model calls, solving, checking, compilation and simulation, within 1200 seconds.

Prewritten: kernel family, portable epilogue and serpentine block traversal, MIP feature/constraint builder, test generator,
independent reference, Z3 checker, model adapter, compiler/simulator wrappers and checkpoint logic.
Runtime changes: Agent domains/cap/weights/rationale, HiGHS assignment, generated program
configuration and evaluation-guided selection. The kernel source is intentionally a declared
parameterized family, not claimed to be written from scratch by the Agent on each invocation.

The source family is based on TileLang's public GEMM examples. The course simulator source is
commit a628c9d963cf9c029fa596b872d6e8920ca883e8, unmodified. TileLang is release wheel 0.1.15;
we do not claim it is identical to the upstream Spark runner's separate pinned source commit.
The Docker base digest/Python packages are pinned; apt resolution is not a historical snapshot.

See README.md for clean setup, model configuration, standard entry point and verification.
The declared numerical runtime requires Mac Metal; an arbitrary Linux CPU-only grader would
need a different explicitly supported numerical backend. No NVIDIA server rental is assumed.
There is no claim of full-family proof, global optimum, CUDA binary execution, or guaranteed
acceptance of cross-backend numerical validation by the course. These are the main remaining
scope limitations, not missing artifacts concealed as successful tests.
