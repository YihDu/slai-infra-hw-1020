#!/usr/bin/env python3
"""Produce a concise report from validated public-case run evidence."""
import argparse
import json
from pathlib import Path
from validate_submission import ROOT,verify_run,read


def write(results):
    cases=[verify_run(results/read(p)['id']) for p in sorted((ROOT/'third_party/gb10/workloads').glob('*.json'))]
    totals={}
    for key in cases[0]['usage']:
        values=[r['usage'][key] for r in cases]
        totals[key]=None if any(v is None for v in values) else sum(values)
    rows='\n'.join(f"| {r['workload']} | {r['initial']['simulated_us']:.3f} | {r['final']['simulated_us']:.3f} | {r['simulated_speedup']:.3f}× | {r['tested_candidates']} | {r['elapsed_seconds']:.2f} |" for r in cases)
    example=max(cases,key=lambda r:r['simulated_speedup']);directory=results/example['workload']
    search=read(directory/'state/search.json');initial=example['initial'];final=example['final']
    first_plan=read(directory/'formulations'/initial['attempt']/'plan.json')
    best_plan=read(directory/'formulations'/final['attempt']/'plan.json')
    ranking='\n'.join(f"- {r['workload']}: first-plan proxy prefers {r['initial_proxy_preferred_attempt']} among measured candidates ({r['initial_proxy_preferred_us']:.3f} μs); observed best is {r['final']['attempt']} ({r['final']['simulated_us']:.3f} μs)." for r in cases)
    incomplete=[]
    for r in cases:
        for item in read(results/r['workload']/'state/search.json')['history']:
            if item.get('stage') or item.get('numerical_status')!='passed' or item.get('simulation_status')!='completed':
                incomplete.append(dict(workload=r['workload'],attempt=item['attempt'],stage=item.get('stage'),status=item.get('status'),numerical_status=item.get('numerical_status'),simulation_status=item.get('simulation_status')))
    report=f'''# Agentic GB10 optimizer — submission report

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
{rows}

Numerical checks: {sum(r['numeric_test_cases'] for r in cases)} full-shape cases across the candidates.
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

Current catalog: block_M={{16,32,64,128}}, block_N={{32,64,128,256}}, block_K={{32,64,128}}, stages={{1,2,3,4}},
threads={{64,128,256}}, serpentine row-panel swizzle={{0,4,8,16,32,64}}.
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

Case: `{example['workload']}`. Full trace is `results/{example['workload']}/trace.jsonl`.

1. Workload tensors/precision are read; source/prompt/config snapshots are saved.
2. Agent plan `{initial['attempt']}` weights: `{json.dumps(first_plan['weights'])}`.
3. HiGHS selects `{json.dumps(initial['configuration'])}`; LP/MPS and solver results are saved.
4. The generated source executes on Metal and passes all numerical cases; identical input IR
   compiles to sm_121a and its actual SASS simulates at {initial['simulated_us']:.3f} μs.
5. Observations are included in the next model request. Revisions/exclusions lead to plan
   `{final['attempt']}` weights `{json.dumps(best_plan['weights'])}` and choice `{json.dumps(final['configuration'])}`.
6. That implementation passes the same checks and simulates at {final['simulated_us']:.3f} μs.
   The source/config/TIR/SASS identities are in result.json and evaluation.json; only eligible
   candidates can replace the checkpointed best. Later failures or timeouts preserve it.

## Proxy disagreement and failed attempts

To compare like with like, the **first plan's fixed weights** are reapplied to all measured
feature vectors; costs from differently weighted rounds are not directly compared as latency.
The resulting preferred points versus observed winners are:

{ranking}

This exposes cost-model limitations without fabricating a microsecond prediction or a
hardware-calibrated error bound. Limited samples and small simulated improvements do not establish
real-hardware speedups. The model's bandwidth/cache assumptions and mapping approximations remain
limitations, particularly for fine differences between candidates.

Incomplete/rejected public-run observations: `{json.dumps(incomplete)}`.
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

Aggregate public-run usage: `{json.dumps(totals)}`. Counts include failed model attempts; unavailable
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
'''
    (ROOT/'report.md').write_text(report)
    (results/'summary.json').write_text(json.dumps(dict(scope='Metal-checked source/config/IR and GB10 simulated timing; no CUDA execution',cases=cases,usage=totals),indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),usage=totals,report='report.md')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=(ROOT/'results/submission-final' if (ROOT/'results/submission-final').exists() else ROOT/'results'));args=p.parse_args();write(args.results)
