# 无 GPU 编译与 GB10 模拟接入

> 阶段记录：后续正式实现与最终实测结果见根目录 report.md；本页保留当时的计划或事实。

2026-10-22。已在 Apple M4 / 16 GB 上完成真实 TileLang → CUDA → cubin → 编码 SASS →
课程模拟器链路；运行容器没有 NVIDIA GPU、没有网络。数值验证仍只执行 CPU shadow，
因此没有 `checked` GPU 实现，也不把模拟改进当作已经验证的正确实现加速比。

## 工具链

- 独立 Colima `slai-compiler`，原生 ARM64 Linux，4 核 / 6 GiB / 40 GiB 虚拟磁盘。
- Python 3.12.12；TileLang 0.1.15 发布 wheel（包含其自带 TVM）；torch 2.11.0+cpu。
- nvcc、runtime、crt、nvvm 13.0.88；CCCL 13.0.85；nvdisasm 13.4.92。
- Python 完整版本：`setup/compiler-python.lock`；基础镜像 digest 固定于 Dockerfile。
- 每次 compilation.json 保存实际镜像 ID 和关键版本；系统包通过该镜像 ID 标识，
  Dockerfile 中 apt 包未用历史仓库快照锁定，重新构建不保证逐字节相同。
- 上游 GB10 记录使用 TileLang 源码 commit `fe6d319c78c0738186fc6e83eb8436ddef65204d`；
  本次使用发布 wheel，并非声称与该 commit 完全一致。编译目标明确为 `sm_121a`。

上游依据：
[compile_only 实现](https://github.com/tile-ai/tilelang/blob/fe6d319c78c0738186fc6e83eb8436ddef65204d/tilelang/tools/compile_only.py)
说明可通过显式 target 与 lower 生成设备源码，不需要执行 GPU 程序。
`tools/compiler_worker.py` 在 Target context 中执行 lower，提取实际动态共享内存，
再调用 nvcc 与 nvdisasm，保存原始日志和 SHA-256。

为匹配课程 cp.async 路径，模板禁用 warp specialization，两个 global→shared copy
均显式 `disable_tma=True`。线程数从 CUDA launch bounds 读取，寄存器来自 ptxas；
校验实际线程数、共享内存与寄存器上限，拒绝意外 TMA mapping。
这些改动由开发阶段完成，运行时 Agent 只修改域、权重、资源预算和选中配置。

## 模拟与正确性边界

`tools/simulate_audited.py` 先核对课程包全部源码哈希，调用未修改的模拟器。
外层仅检查 SM.run 返回时全部 warp 已完成：原模拟器达到内部循环上限时也可能
返回 cycles，因此未完成结果必须排除。未完成、异常、外层超时和编译资源无效
均不能参加性能比较。该包装不修改延迟参数或时序算法。

新结果位于 `results/compiled-baseline/`，每例两轮真实 DeepSeek 调用；第二轮请求
含第一轮编译资源和模拟结果。运行不下载资料、不构建镜像，模型调用是唯一外部网络。
完整证据校验与汇总：

```sh
.venv/bin/python tools/summarize_compiled.py
```

`state/source` 保留当次驱动/编译工具/模板的精确快照。模型 usage 与工具调用次数
在各 result.json 中，失败也保留。`results/compiler-integration/` 是接入阶段的独立
试验：首次 lower 缺少 Target context 报错，修复后首例编译完成并模拟得 147.819 μs；
它不计入新的五案例 workload 运行统计。

课程作业第 60 行要求精确程序与配置的独立 reference 数值测试；第 133 行明确
模拟器不检查输出。当前 CPU shadow 比较 FP16 输入/输出、FP32 分块累加与 float64
reference，只覆盖算法诊断；Z3 只覆盖固定配置整数索引边界。
将模板改为 CPU TileLang 后执行，也不能直接证明原 CUDA 并行映射与异步流水正确。
因此剩余工作是精确 GPU 实现的数值执行与验证，再选择 checked winner、形成可信性能
对比和最终提交包。目前不宣称作业全部完成，也不要求购置 GPU 服务器。

## 五案例实测

| Workload | 整次运行秒数 | 第一轮模拟 μs | 第二轮模拟 μs |
| --- | ---: | ---: | ---: |
| gemm_128x4096x4096 | 13.693 | 236.957 | 236.957 |
| gemm_2048x2048x2048 | 15.918 | 271.066 | 271.066 |
| gemm_32x4096x4096 | 12.233 | 147.846 | 147.846 |
| gemm_4096x4096x1024 | 28.670 | 575.695 | 432.647 |
| gemm_4096x4096x4096 | 101.819 | 1989.087 | 1974.749 |

五例均完成两轮编译和模拟，共 10 次模型调用、10 次编译、10 次模拟；
输入 14762 tokens、输出 3371 tokens。20 个全形状 CPU shadow 测试通过。
所有模拟值来自真实 SASS，全部 warp 完成。上述值是未经数值验证内核的模型估计，
不能作为正确实现的性能提升结论。前三例第二轮配置未变；4096×4096×1024 的模拟
估计下降约 24.8%，4096³ 将 block_K 从 64 改为 32、流水级数从 1 改为 2，估计下降约
0.72%。这些仅是模拟模型内的差异；后者尤其不能据此宣称真实硬件收益。
新旧运行的统计相互独立。最终源码另增加输出目录防覆盖检查，经单元测试；当次运行
快照仍保存运行时版本，没有用后续版本替换历史代码。
