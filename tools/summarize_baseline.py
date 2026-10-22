#!/usr/bin/env python3
"""Validate saved evidence and generate a development baseline report."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
rows=[];totals=dict(model_calls=0,input_tokens=0,output_tokens=0)
for directory in sorted((root/'results').glob('gemm_*')):
    result=json.loads((directory/'result.json').read_text())
    checks=json.loads((directory/'checks.json').read_text())['attempts']
    evaluations=json.loads((directory/'evaluation.json').read_text())['attempts']
    manifest=json.loads((directory/'state/manifest.json').read_text())['resources']
    for name,sha in manifest.items():
        assert hashlib.sha256((directory/'state/source'/name).read_bytes()).hexdigest()==sha
    for line in (directory/'trace.jsonl').read_text().splitlines():
        event=json.loads(line)
        for reference in event['artifacts']:
            p=(directory/reference).resolve()
            assert p.is_relative_to(directory.resolve()) and p.is_file(),reference
    for item in evaluations:
        if 'kernel_sha256' not in item:continue
        impl=directory/'implementation'/item['attempt']
        assert hashlib.sha256((impl/'kernel.py').read_bytes()).hexdigest()==item['kernel_sha256']
        assert hashlib.sha256((impl/'config.json').read_bytes()).hexdigest()==item['config_sha256']
    assert result['status']=='no_valid_result' and result['implementation'] is None
    assert len(checks)==2 and all(x['passed'] for x in checks)
    solvers=[json.loads(p.read_text()) for p in sorted((directory/'formulations').glob('*/solver.json'))]
    assert len(solvers)==2 and all(x['status']=='Optimal' for x in solvers)
    row=dict(workload=result['workload_id'],seconds=round(result['elapsed_ms']/1000,3),
             cpu_checks=sum(len(x['cases']) for x in checks),
             max_abs_error=max(c['max_abs_error'] for x in checks for c in x['cases']),
             initial_config=checks[0]['config'],final_config=checks[-1]['config'],usage=result['usage'])
    rows.append(row)
    for metric in totals:totals[metric]+=result['usage'][metric]
failures=[]
for p in (root/'results/development-failures').glob('*/result.json'):
    r=json.loads(p.read_text());failures.append(dict(path=str(p.relative_to(root)),usage=r['usage']))
summary=dict(scope='CPU development baseline only; no checked TileLang result',cases=rows,
             successful_run_usage=totals,retained_failed_runs=failures)
(root/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
table='\n'.join(f"| {r['workload']} | {r['seconds']} | {r['cpu_checks']} | {r['max_abs_error']:.8g} |" for r in rows)
report=f'''# 初始 CPU baseline 历史报告

本文仅记录接入编译器之前的运行，保留其原始结论。最新进展见 [编译接入记录](docs/compiler-integration.md)。

## 当时结论

五个公开 GEMM 的两轮模型建模、HiGHS 求解与 CPU shadow 数值诊断已运行。
所有结果为 `no_valid_result`：尚无经过编译、数值执行和 GB10 模拟的 TileLang 结果。
本报告不是最终课程性能报告；缺失项不计为通过。

| Workload | 总耗时（秒） | 全形状 CPU 测试数 | 最大绝对误差 |
| --- | ---: | ---: | ---: |
{table}

每轮包含随机输入 seed=42 与零行/正负列结构输入。FP16 输入输出、FP32 分块累加，
独立 reference 为完整 float64 矩阵乘法后转 FP16。阈值来自各 workload；全部逐元素通过。
这些测试执行的是 `cpu_shadow.py`，**不是** `kernel.py`，不能推导 CUDA 正确性。
开发硬件为 Apple M4 / 16 GB；任何 CPU 时间都不是 GB10 模拟或实测延迟。

## 方法与决策轨迹

DeepSeek Flash 通过显式 API 配置提出 tile/stage/thread 域、共享内存 cap 与成本权重。
构建器为笛卡尔积内整除 shape 的 tuple 建立二元变量，约束恰好选择一个，且估计共享内存
不超过 Agent 的 cap。HiGHS 实际求解并导出 LP/MPS，两个选中点各执行一次诊断。
全部记录的解在该有限解析模型内为 Optimal、gap=0；这不代表 GPU 性能最优。
解析特征未经 GB10 校准，不是时延预测。没有穷举测量候选性能。

具体轨迹见 `results/gemm_32x4096x4096/trace.jsonl`：
workload → attempt-00 plan → HiGHS choice → kernel/config → CPU passed + compiler unavailable
→ Agent 收到反馈 → attempt-01 将 launches 权重从 0.5 调整到 0.25 → 再次求解。
该案例最终配置未变，也没有证据表明这次权重修改改善性能；Agent 的解释是待验证假说。
每个案例初始与最终配置见 `results/summary.json`，实现哈希在各自 evaluation.json 中。

## 验证范围与失败

Z3 对固定 shape/config 的 tile 索引边界做 UNSAT 检查；范围仅整数索引，
不包括浮点、异步同步、内存布局或编译器 lowering。非整除 shape 的测试能产生 SAT 反例。
单 GEMM 不存在多算子融合选择，未实施 TVM 图变换。

第一次 DeepSeek 默认思考模式运行消耗输出上限但正文为空，两轮 JSON 解码失败。
原始响应与用量保留在 `results/development-failures/empty-model-output/`。
随后显式设置 thinking=disabled，两轮结构化输出成功。这个改动由开发过程完成，
不是运行时 Agent 自我修改。失败记录与成功运行不可混作单次 20 分钟运行。

成功运行合计：{totals['model_calls']} 次模型调用，输入 {totals['input_tokens']} tokens，
输出 {totals['output_tokens']} tokens；失败用量另见 summary.json。cached/reasoning 原始统计
保存在响应中，缺失汇总为 null。实际 TileLang 编译次数、数值执行次数、GB10 模拟次数均为 0。

原版模拟器哈希校验通过；直接运行课程示例由于未提供 kernel.sass 而失败，
证据见 `setup/simulator-baseline-blocker.json`。禁止用伪造 SASS 或 shape-only 估算替代该输入。

## 预写组件与后续工作

MIP 构建器、CPU shadow、TileLang 家族模板、Z3 检查与驱动均为开发时预写；
每次运行改变域、约束预算、权重与配置。生成的 TileLang 模板参考上游
https://github.com/tile-ai/tilelang/blob/main/examples/gemm/example_gemm.py ，尚未编译验证。
模拟器来源及固定 commit 在 `third_party/gb10/SOURCE.json`。Python/httpx 是自建 harness，
用于精确控制时间、证据与配置文件接口，全部依赖锁定于 requirements-cpu.lock。

下一阶段首先确认课程无 GPU 编译和数值执行方式，再产出真实 sm_121a SASS 和资源数据，
接通模拟反馈；之后才形成可提交的 checked baseline。详见 docs/development-plan.md。
'''
(root/'report.md').write_text(report)
print(json.dumps(dict(cases=len(rows),usage=totals,evidence_validation='passed')))
