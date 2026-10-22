# Agentic AI Infrastructure Optimizer — preparation workspace

当前为环境准备阶段，尚未实现优化闭环，也没有通过检查的 kernel。

## 已准备

- `third_party/gb10/`：课程原始模拟器、参数、5 个 GEMM workload；保留 SOURCE.json。
- `requirements-cpu.lock`、`setup/cpu.sh`：固定全部直接和间接 CPU 依赖与安装入口。
- `tools/doctor.py`：工具、求解器、模拟器检查；结果存入 `setup/environment.json`。
- `optimizer/trace.py`：追加式 JSONL 事件记录与输出目录内的证据路径检查。
- `configs/model.example.json`：模型接口配置示例，不含凭据。

安装和检查：

```sh
sh setup/cpu.sh
.venv/bin/python tools/doctor.py --output setup/environment.json
```

当前本机 CPU 环境已安装并通过 HiGHS、Z3、模拟器 CLI 检查，版本见 `setup/environment.json`。
重新获取本机安装包可运行 `python3 tools/fetch_wheels.py --direct`；下载后核对 SHA-256。
安装阶段需要网络。离线安装可提前将对应平台的所有依赖 wheel 放入 `wheelhouse/`。
`.venv/`、缓存与 wheelhouse 不纳入提交。

## CPU 与 GPU 分工

Agent API、HiGHS、Z3 和 GB10 时序模拟可在 CPU 上运行。模拟器不执行 GEMM，
不能证明输出正确。课程包未提供数值模拟器、reference.py 或 baseline.py。
按课程方“不需要 GPU”的说明，当前采用 CPU 配置，不要求购买或租用服务器。
课程提供的无 GPU 编译和数值验证路径仍待确认；时序模拟不能替代正确性检查。

GB10 SASS 可在没有 GPU 的 Linux 环境中用适当工具链交叉编译；需要支持
`sm_121a` 的 CUDA 工具链（CUDA 13.0 文档包含该目标）、nvdisasm 和兼容 TileLang。
不能将较旧 GPU 当作可以执行 sm_121a 程序的设备。

暂不在 Mac 上安装 CUDA/TileLang。课程编译工具链公布后再选择兼容平台和版本。
TileLang 使用定制 TVM，避免盲目安装另一套 TVM。

## 模型配置

当前默认开发配置为 `configs/deepseek.json`，使用 DeepSeek 官方端点和 `deepseek-flash`。
API 已通过一次真实连通性测试，证据见 `setup/deepseek-connectivity.json`。
已验证 Python HTTP 客户端端到端调用并返回 OK（2026-10-22）；通用配置模板仍保留。
`optimizer/model.py` 支持三种协议的纯文本请求，模型需能输出代码和结构化决策；
当前尚未实现多轮 Agent 工具循环。不读取个人 Agent 配置，不将密钥写入项目。

```sh
.venv/bin/python tools/check_model_config.py configs/model.example.json
# 填好配置并在运行环境设置指定密钥变量后，显式发起一次测试：
.venv/bin/python tools/check_model_config.py configs/model.example.json --ping
```

默认检查不会发起模型调用。端点应包含服务所需的版本前缀（如 `/v1`）。
HTTP 客户端默认不继承系统代理；DeepSeek 配置显式设置 `trust_env: true`。
DeepSeek 使用 `max_tokens` 字段，通用 Chat Completions 配置默认使用 `max_completion_tokens`。
密钥通过 `DEEPSEEK_API_KEY` 环境变量提供，不保存在仓库。开发时可从指定文件安全加载：

```sh
.venv/bin/python tools/with_deepseek_key.py --credential-file /path/to/API -- \
  .venv/bin/python tools/check_model_config.py configs/deepseek.json --ping
```

该开发工具只读取 `# DeepSeek-yihang` 节的 `key=`，不执行凭据文件，
不会作为提交系统的隐式依赖；正式运行应直接设置配置指定的环境变量。
DeepSeek 接口文档：https://api-docs.deepseek.com/api/create-chat-completion/
接口字段参考官方文档：
https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
https://developers.openai.com/api/reference/resources/responses/methods/create

下载来源：
http://10.204.87.113:8001/agentic-optimizer/downloads/gb10-simulator-inputs-v2.zip

课程编译/验证接口、run.sh 和优化闭环仍待完成；本目录不能作为最终作业提交。
