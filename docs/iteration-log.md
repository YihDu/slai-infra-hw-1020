# Baseline 迭代记录

2026-10-22：复核课程网页，要求与本地 markdown 一致。

1. 初始实现限定为 CPU 开发 baseline：真实 DeepSeek 建模、HiGHS 求解与输出，
   TileLang 源码生成、CPU shadow 诊断、Z3 检查，诚实记录缺失的编译/模拟步骤。
2. 首次 32x4096x4096 运行，模型两轮正文均为空。原始 usage 证明每轮 4096 tokens
   全部用于 reasoning。失败保留于 results/development-failures/empty-model-output。
3. 开发时将 DeepSeek thinking 显式关闭；五个公开案例各执行两轮建模与诊断。
   10 次有效模型调用，20 次全形状 CPU 数值检查通过。记录中无 GPU/GB10 性能值。
4. 通过测试确认：HiGHS 不可行时不选配置，非整除 shape 的无 guard 索引可被 Z3
   检出，CPU shadow 可以处理尾块，驱动不会将 CPU 通过提升为 checked，超时保留结果。
5. 补充运行时源文件快照。对已完成运行按其启动时的 SHA-256 重建准确版本，所有
   快照均与原 manifest 哈希一致；没有将后续代码冒充当次运行代码。
6. 编译与模拟仍不可用：课程示例缺少 kernel.sass；原始模拟器文件哈希全通过。

Gemini read-only 审查尝试因 worker 服务认证 401 失败，未获得审查结论；
开发和复核由主 agent 完成。该调用不属于任何 workload 的计时运行和结果用量。

7. 复核上游 compile_only/lower 与 GB10 Spark runner 后，在 Mac 安装独立 ARM64 Linux
   Colima VM，配置 CPU PyTorch + TileLang 0.1.15 + CUDA 13.0 工具链，无 GPU 编译 sm_121a。
8. 初次 lower 因缺少 Target context 失败，保留 compilation.json；添加显式 Target
   context 后真实生成 cubin/SASS。为匹配 cp.async 禁用 warp specialization/TMA copy。
9. 接入未修改的课程模拟器，额外审计 warp 完成状态；模拟时延单独保存，不当作数值检查。
   编译/模拟反馈进入下一轮真实模型请求。新增超时清理/缺失 Docker 测试。
10. 新的五案例两轮运行证据见 results/compiled-baseline；工具链差异、剩余正确性
    缺口和模拟估计汇总见 docs/compiler-integration.md。历史结果不覆盖。

11. Mac 原生 TileLang/Metal 实测五个完整 GEMM：同源码/config 的 input TIR 与 CUDA
    端逐字节相同；数值参考独立 CPU float64。修复 FP32→FP16 epilogue 与显式同步。
12. 接入正式 runner、最佳结果检查点、硬截止与清理余量，加入真实错误 shader 负例。
    第一版六轮搜索完成 30 次模型调用、29 个有效候选，完整记录位于 submission-v1。
13. 为提升性能尝试分段 epilogue，但发现共享内存越界与数值错误，拒绝纳入正式模板。
14. 加入与课程 launch_order 一致的 serpentine swizzle；Z3 检查映射边界，有限坐标
    枚举检查双射，测试覆盖不满 panel 的情况。首次计划用 swizzle=0 建立基线，
    后续由 DeepSeek 修改域/目标并由 HiGHS 选择。未把测量赢家硬编码为 MIP。
15. 最终八轮五案例结果位于 submission-v2；原生磁盘 kernel cache 禁用，模拟器原文
    不变。17 项测试和额外未知 shape 通过。checked 明确为 Metal 源级数值验证，
    未执行 CUDA 二进制；两者范围在 result.json/报告中明确，不把历史失败改为成功。

## Final submission acceptance

- Final eight-round runs: five public workloads, 40 model calls, 39 evaluated candidates and 117 full-shape numerical tests. Best simulated speedup over the first checked candidate is 1.6016× on 4096³; all case searches finish within 20 minutes.
- 17 automated tests passed, including real Metal corruption rejection, timeout checkpointing, protocol adapters and swizzle coverage. An unseen 96×192×96 workload passed without code edits.
- Extracted the release-candidate ZIP into a separate workspace, installed fresh CPU/Metal virtual environments, and verified archive content hashes. The validation CLI regenerates results/summary.json; archive bytes were restored before the full hash comparison.
- The extracted entry point completed two rounds in 17.924 seconds with EVAL_MODEL_API_KEY and without DEEPSEEK_API_KEY in its environment. All five archived case evidence sets passed validation after extraction. The same Mac and already prepared Docker compiler image were reused; dependency installation was outside the timed run.
- Final validation remains explicitly Metal numerical execution plus byte-identical input TIR CUDA compilation and original GB10 SASS simulation. CUDA binary execution is not claimed.

## Post-v2 exploration (not submitted)

- Keep the v2 ZIP and evidence unchanged. Added a separate five-case, 12-round search under `results/exploration-v3`, comparing each result against the previous best rather than only its new first candidate.
- Expanded M/N/K tiles, stages, thread counts and launch panels. Six bounded development branches contain 16 probes: smaller/wider/deeper tiles, 64/256 threads, FP16 shared epilogues, scalar/direct epilogues, larger panels, and alternate warp partitioning. These probes are diagnostic experiments, not a measured winner table in the MIP.
- Direct/FP16 epilogues hit Metal code-generation limitations. Alternative warp partitions failed the independent numerical reference. None was promoted to the production kernel. Larger panels increased simulated time on 4096³; they are not automatically assumed faster.
- During v3, some symbolic-stride SMT queries returned unknown. A separate constant-divisor full/tail-panel proof passed 160 checks (maximum about 0.040 seconds). Its integration and launch-equivalence deduplication passed three additional tests, bringing the suite to 20. One later concurrent test run timed out in the Metal integration test; that failure is retained and requires a serial rerun.
- V3 case source snapshots precede the proof/deduplication refinement. The refined four-round real-model smoke run passed 12 full-size numerical tests, with four distinct launch configurations and all SMT checks passing, in 87.608 seconds. No v3 history was rewritten to claim use of the new implementation.
- The expanded domain can hurt search efficiency. The subsequent prompt uses a generic two-step core-domain baseline/locality phase before expanding; it contains no workload-specific winner or measured latency table. Default rounds increase from 8 to 12; the wall-clock cap remains 1200 seconds.

### Exploration outcome and final acceptance

- V3 completed all five 12-round searches. Its longest search took 1105.579 seconds, below 1200. Schema failures, infeasible/proof-rejected attempts and a simulation timeout remain in the original traces; no failed candidate was promoted.
- Compared with v2 bests, M=128 improved from 156.322388 to 155.921338 μs (1.002572×), and M=32 from 147.667532 to 147.523888 μs (1.000974×). The 2048³ and 4096×4096×1024 cases tied v2; expanded-domain 4096³ was worse (2500.849862 vs 1597.229841 μs), so v2 remains selected for all three large cases.
- The revised generic staged-search prompt was then tested on 4096³ for two rounds. It reached 1616.240131 μs in 106.243 seconds, within about 1.2% of the previous best, without a workload-specific winner table. The baseline/locality sequence therefore repairs a search-efficiency regression; it is not claimed as a new performance record.
- Added specific schema-failure feedback to the next model request. A regression test verifies that the model receives the invalid resource-budget reason rather than only ValueError.
- The final serial suite passed all 21 tests in 10.336 seconds. The earlier concurrent Metal test timeout is retained separately; it did not recur serially. The current runtime files match the staged run's saved source manifest.
- Across the five v3 searches and two refined validation runs: 66 real model calls, 58 numerical checks/compilations/simulations, 301506 input tokens and 34092 output tokens (249728 cached input tokens reported). Failed/simulation-timeout attempts remain included; these are provider records, not billing estimates. The 16 manual development probes are separate and not counted as Agent-generated candidates.
- `results/exploration-v3/comparison.json` retains the direct v3-versus-v2 comparison, including the regression. `best-known.json` selects complete verified run records without overwriting originals. The existing submission ZIP retains its original SHA-256; no new course upload or ZIP replacement was performed.
