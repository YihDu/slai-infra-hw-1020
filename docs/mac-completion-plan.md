# Mac 本地完成提交版的方案

> 阶段记录：后续正式实现与最终实测结果见根目录 report.md；本页保留当时的计划或事实。

核查日期：2026-10-22。课程线上 homework.md 与仓库版本逐字一致。
本次结论来自实际运行，并非仅依据支持列表。

## 已验证的本地路径

同一份 TileLang kernel.py + 配置 → 同一 PrimFunc/输入 TIR：

1. Mac 原生 TileLang 0.1.15 + PyTorch 2.11.0 Metal adapter：实际执行数值计算。
2. Mac 上 ARM64 Linux 容器：交叉编译 sm_121a，导出 cubin、SASS 和实际资源。
3. 本机 CPU：原课程模拟器估计 GB10 时延，检查所有 warp 完成。
4. 独立 CPU float64 reference：检查 FP16 输入输出、FP32 累加结果的误差。

结果：五个公开尺寸各两项全形状测试通过；五例输入 TIR 在 Metal/CUDA 两端逐字节一致；
五例均成功编译 sm_121a；32×4096×4096 的新模板完成模拟，估计 147.697 μs。
最大绝对误差不超过 0.001953125，各元素均满足 workload 的 rtol/atol。
来源哈希和完整原始证据在 `results/mac-feasibility/`。本次配置是人工固定的可行性探针，
不是新的 Agent 搜索成绩；此前 no_valid_result 运行不改写。

实测解决的兼容问题：FP32 fragment 直接复制到 FP16 输出会触发 Metal 类型错误；
通过统一的 FP32 shared epilogue 加显式 FP16 cast 解决。仅加 epilogue 在大矩阵上发现
NaN；加入 epilogue 前后显式同步后全部数值测试通过。CUDA 路径使用完全相同的源码，
没有为 Metal 单独替换 GEMM 算法。Metal adapter 要求调用方显式提供输出张量，不能
假定其像 CUDA adapter 一样自动分配并返回 T.empty 输出。

原生 Metal 使用 torch.mps.compile_shader，本机无需安装完整 Xcode 或命令行 metal 编译器。
其依赖位于独立 `.cache/mac-check`，没有修改已有 CPU baseline 虚拟环境。

## 证据边界与课程口径

数值证据是同源、同配置、同输入 TIR 在 Metal 后端执行的结果；GB10 SASS 仅参与
时序模拟。不同后端的代码生成、同步 lowering 和硬件浮点行为仍然不同，不能声称
Metal 测试验证了 CUDA 二进制。这不是完整编译器正确性证明。

作业要求 exact implementation 的数值测试，但未明确讨论跨后端测试是否可判 checked。
因此技术可行性已成立，课程对此证据范围的评分认定仍不能由本次测试保证。
最终系统应保留 checked_target、source/config/TIR 哈希、CUDA artifact 哈希及
cuda_binary_executed=false；若没有明确接受跨后端的规则，不应将它静默标为 CUDA 已验证。

若要核对课程口径，具体问题是：“同一 TileLang 源码、配置和输入 TIR 在 Mac Metal 上
通过全尺寸 reference 测试，另编译 sm_121a 并运行官方模拟器，明确标注未执行 CUDA
二进制，是否可作为 checked 实现？”这是验证范围问题，不是要求购买 GPU 的理由。

另一个未公布清楚的信息是最终运行平台。当前方案已实证可在这台 Mac 完成本地流程，
但 Metal 不能在普通 Linux CPU 环境中执行；若最终评测平台不同，需按课程公布的
环境适配，不能把 Mac 实测直接当作任意 Linux 评测环境也能运行的保证。
本方案覆盖当前已发布的五个 GEMM，未公布的其他算子不作虚假承诺。

## 开发与验收顺序

| 阶段 | 实施内容 | 完成标准 |
| --- | --- | --- |
| 1. 统一可执行模板 | 将探针的可移植 epilogue/同步整合到正式模板；build 根据显式后端编译并正确分配输出；两端共享 PrimFunc | 同一源码/config/TIR，无独立 shadow 冒充实际实现 |
| 2. 真正数值检查 | 自动生成随机与结构化输入，调用上述 build，CPU float64 reference 比较；保存种子/误差/硬件/后端/哈希 | 五例均通过；故意破坏计算的负例能被检出 |
| 3. 合法搜索空间 | MIP 增加 Metal 与 CUDA 共同可执行的线程、tile、shared memory 限制；新 epilogue 资源记入预算 | 求解控制实际程序；不能只把手工探针配置写成固定赢家 |
| 4. 反馈与候选保存 | 数值失败、Z3 反例、编译资源、有效模拟时延进入下一轮；保存最佳通过验证的候选，失败不覆盖 | 结果身份一致；验证范围明确；超时仍有正确检查点 |
| 5. 有效搜索 | 有限预算内选多个有区别的候选；保留初始正确 baseline；观察成本代理与模拟排序差异 | 真实 Agent 修改变量域/约束/目标，HiGHS 实际求解，无固定加速门槛 |
| 6. 接口与稳健性 | 标准三参数；case 相对路径；新 shape；模型协议适配；离线编译；异常/超时/缺凭据/模拟未完成 | 无人工干预；只允许模型服务网络；每例 20 分钟内 |
| 7. 重新正式运行 | 五公开案例各跑完整 Agent→MIP→执行检查→编译→模拟→反馈；收集全部用量 | 成功与失败原样保存，结果身份与证据哈希校验通过 |
| 8. 复现与报告 | Mac 原生数值环境、Linux 编译镜像分别锁定；从新工作目录复现；更新根报告 | 说明预写组件、搜索空间、决策轨迹、SMT范围、TVM单GEMM适用性和跨后端限制 |
| 9. 打包与提交 | 最终公开案例置于 results/<workload-id>；根目录 run.sh/README.md/report.md；秘密/符号链接/大小审计 | ZIP无外层目录，≤256MiB压缩/1GiB展开/10,000条目；上传后保存回执 |

不要求：租 NVIDIA 服务器、实现任意模型导入器、强行加入双算子演示、导出开发聊天全文，
或达到固定 cuBLAS 性能门槛。需要的是优化器运行轨迹与诚实、可追溯的证据。
上传阶段需要姓名、学号和提交回执；这些不阻塞开发。

## 已核对的上游资料

- 课程要求：http://10.204.87.113:8001/agentic-optimizer/homework.md
- 后端目标：https://github.com/tile-ai/tilelang/blob/v0.1.15/docs/get_started/targets.md
- Metal 实现说明：https://www.tilelang.com/compiler_internals/metal_tilelang_development.html
- 官方 GEMM：https://github.com/tile-ai/tilelang/blob/v0.1.15/benchmark/matmul_metal/benchmark_matmul_metal.py
- Metal adapter：https://github.com/tile-ai/tilelang/blob/v0.1.15/tilelang/jit/adapter/torch/metal.py

按仓库指令尝试 Gemini 只读复核，但其服务返回 401，未获得独立结论；以上由主 agent
直接读源码和实测确认，该失败不影响项目使用的 DeepSeek API。
