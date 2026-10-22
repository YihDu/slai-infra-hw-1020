# Agent + HiGHS + TileLang optimizer for GB10

Development continues after the frozen v2 submission candidate. The existing ZIP,
`report.md`, and `setup/acceptance.json` describe v2; no course upload has been made.
The expanded 12-round exploration is saved separately in `results/exploration-v3/`.
Its `comparison.json` compares against the **previous best**, not just a new initial
candidate. `best-known.json` indexes the retained best complete runs and final acceptance.
Per-run source snapshots preserve the version actually evaluated.

This workspace runs a real DeepSeek/API formulation agent, HiGHS MIP selection,
TileLang numerical checks, GB10 cross-compilation and the supplied timing simulator.
The supported development environment is **Apple Silicon macOS with Metal**, plus
an ARM64 Linux Docker VM for CUDA compilation. No NVIDIA GPU is required locally.

**Validation scope:** `checked` records the exact TileLang source/configuration
executed on Metal against an independent CPU float64 reference. The CUDA compiler
receives byte-identical input TileLang IR; its sm_121a SASS is timing-simulated.
The CUDA binary is **not executed**. Every result explicitly records
`checked_target: metal`, `cuda_binary_executed: false` and its source/IR/artifact hashes.
Cross-backend testing does not prove CUDA lowering correctness. The course document
does not explicitly specify acceptance of this validation scope; the report discloses it.

## Prepare the environment before timing

Tested: Apple M4, 16 GiB, macOS 26.5.1; Python 3.13.13 (host), Python 3.12.12
(compiler); uv 0.12.18, Colima 0.10.3, Docker CLI 29.9.1.
Exact versions are recorded in `setup/toolchain.json` and dependency lock files.
Install uv, Docker CLI and Colima beforehand, then:

```sh
sh setup/cpu.sh
sh setup/mac-numeric.sh
colima start slai-compiler --vm-type vz --arch aarch64 --cpu 4 --memory 6 \
  --disk 40 --mount-type virtiofs --activate=false
sh setup/compiler-build.sh
```

`setup/cpu.sh` creates `.venv`; `setup/mac-numeric.sh` creates `.venv-metal`.
The image `slai-gb10-compiler:0.1.15` contains TileLang 0.1.15, CPU PyTorch,
CUDA nvcc 13.0.88 and nvdisasm 13.4.92. CUDA containers run with `--network=none`
and no GPU. Base image digest and Python dependencies are pinned; the built image
ID is saved in each result. apt package resolution is not a historical snapshot.
The macOS wheel supplies Metal execution through `torch.mps.compile_shader`; a full
Xcode installation was not needed on the tested machine.

The build script's default proxy is the local development proxy forwarded to
`192.168.5.2:25662`. Override `SLAI_BUILD_PROXY` for your network (an empty value means
no build proxy). It is an installation-time setting, not a model credential.
Use `SLAI_DOCKER_CONTEXT` / `SLAI_COMPILER_IMAGE` to select a prepared compiler;
`SLAI_NUMERIC_PYTHON` selects the prepared native numerical Python interpreter.
A generic Linux CPU-only machine cannot execute the Metal numerical backend.

## Required execution interface

Provide the credential through the environment variable named in the config, then:

```sh
./run.sh --workload /cases/example/workload.json \
  --model-config /evaluation/model.json --output /results/example
```

Default configuration is `configs/deepseek.json` (DeepSeek Flash, Chat Completions).
There is no implicit dependency on the developer's credential file, personal agent
settings, conversation history, MCP servers or global skills. Credentials are never
packaged. This is a custom Python/httpx harness, chosen to control reproducible state,
strict model configuration, tool calls and the workload deadline directly.

Supported protocols: `openai_responses`, `openai_chat_completions`,
`anthropic_messages`. Honor `base_url`, `model`, `api_key_env`; the model must return
structured JSON with domains, resource constraints, weights and a rationale.
All three adapters have fixture tests; real public-case runs use DeepSeek.
Optional `max_tokens_field`, `trust_env` and `thinking` extensions are documented by
`configs/deepseek.json`. Model responses and provider usage are saved without headers.

Output must be new or empty. Twelve rounds are attempted by default, bounded by 1200
seconds. `--rounds N` and `--budget-seconds S` are available for experiments.
Disk TileLang/autotuning caches are explicitly disabled, and cleanup time is reserved
within the workload deadline. The timed entry point never installs dependencies or retrieves external knowledge;
only requests to the configured model service are needed. It returns 0 for `checked`
or `no_valid_result`, and nonzero for an execution error without a retained result.

## Workload and search scope

The current family implements `C = A @ B`: rank-2 FP16 inputs/output, FP32 accumulation.
It reads tensor sizes and tolerance from the workload, not the case ID. Tile domains
are M={16,32,64,128}, N={32,64,128,256}, K={32,64,128}, stages={1,2,3,4}, threads={64,128,256}, serpentine row-panel swizzle={0,4,8,16,32,64}.
The first Agent plan uses conventional row-major swizzle=0 as the comparison baseline.
The prompt establishes a core-domain row-major baseline, then explores core-domain
locality before opening the expanded tile/stage/thread/swizzle choices. Launch-equivalent swizzles
are deduplicated, and the bounds proof uses constant-divisor complete/tail panels. Only divisible tuples are searched; workloads with no supported tuple are recorded
as no valid result, without pretending to support arbitrary operators.
Case-side reference/baseline/README files are resolved relative to workload.json and
retained; this family replaces any supplied starting implementation and uses its
own independent full-matrix CPU reference. A general network importer is not included.

DeepSeek chooses domains, a memory cap and objective weights. HiGHS creates binary
choices, enforces one choice and resource limits, and exports LP/MPS and solver status,
bound and gap. Previously evaluated configurations are excluded. The proxy objective
is dimensionless, not predicted microseconds. Numerical, SMT, compiler and simulator
feedback drives further formulation changes. No measured exhaustive winner is encoded
as a MIP. A panel-locality term models possible B-tile reuse; it is tested against real simulated
traffic and latency. SMT checks tile bounds, and a finite mapping check ensures every
output tile is covered once; the mapping matches the original simulator.
The best eligible candidate is checkpointed before continuing the search.

A portable FP32 shared-memory epilogue and explicit synchronization are prewritten.
The compiler reuses epilogue/pipeline storage; the MIP estimates the maximum of the two
lifetimes, then actual compilation checks resources. Z3 verifies fixed, divisible
integer tile bounds only. Single GEMM has no graph-fusion opportunity; the bundled
TVM is used for TileLang lowering, not a claimed graph-level speedup.

## Evidence, verification and packaging

The final ZIP places public-case runs under `results/<workload-id>/`. In the development
repository the latest batch is under `results/submission-v2/`; older CPU and compiled
runs remain separate, and are not retroactively relabeled. Each run saves:

- `result.json`, `checks.json`, `evaluation.json`: selected identity, validation scope,
  numerical errors, usable simulator values and model/tool usage.
- `implementation/`, `formulations/`: executable source/configuration, LP/MPS, solver decisions.
- `compilation/`: actual CUDA source, cubin, SASS, actual resources and raw simulator evidence.
- `trace.jsonl`, `state/`: model/tool actions, failures, feedback, source snapshots and hashes.

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/validate_submission.py --results results/submission-v2
.venv/bin/python tools/build_submission.py --results results/submission-v2
```

In the extracted ZIP, use `--results results` (or omit it) for these tools.

Tests include real Metal execution and a deliberately incorrect shader that must fail,
MIP infeasibility, Z3 counterexamples, best-result retention on timeout, model adapters,
and container timeout cleanup. `report.md` describes results, the decision trace,
proxy/model limitations, resource use and remaining validation boundaries.

Course source: `homework-agentic-optimizer.md`; original simulator and hashes:
`third_party/gb10/SOURCE.json`. The simulator is unchanged; an outer audit rejects
unfinished warp execution. Failed/unsupported/timed-out simulations do not enter rankings.

The ZIP excludes credentials, virtual environments, caches and Git internals. It has
no enclosing directory or symbolic links. Save the portal receipt after uploading.
Use `colima stop slai-compiler` when finished to release the VM's memory.
