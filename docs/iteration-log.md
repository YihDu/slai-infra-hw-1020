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
