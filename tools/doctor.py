#!/usr/bin/env python3
"""Inspect local tools without installing packages or contacting services."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = {'platform': platform.platform(), 'machine': platform.machine(),
              'python': sys.version, 'packages': {}, 'tools': {}, 'checks': {}}
    for name in ['highspy', 'z3-solver', 'numpy', 'httpx', 'torch', 'tilelang']:
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][name] = None
    for name in ['nvcc', 'nvdisasm', 'nvidia-smi', 'codex', 'uv']:
        report['tools'][name] = shutil.which(name)
    if report['packages']['highspy']:
        try:
            import highspy
            h = highspy.Highs()
            h.setOptionValue('output_flag', False)
            h.addVar(0, 1)
            h.changeColIntegrality(0, highspy.HighsVarType.kInteger)
            h.changeColCost(0, -1)
            h.run()
            report['checks']['highs'] = h.getSolution().col_value == [1.0]
        except Exception as exc:
            report['checks']['highs'] = {'error': str(exc)}
    if report['packages']['z3-solver']:
        try:
            import z3
            x = z3.Int('x')
            s = z3.Solver()
            s.add(x > 0, x < 0)
            report['checks']['z3'] = s.check() == z3.unsat
        except Exception as exc:
            report['checks']['z3'] = {'error': str(exc)}
    sim = ROOT / 'third_party/gb10/simulate.py'
    if sim.exists():
        result = subprocess.run([sys.executable, str(sim), '--help'],
                                capture_output=True, text=True, timeout=20)
        report['checks']['simulator_cli'] = result.returncode == 0
        report['simulator_source_sha256'] = hashlib.sha256(
            (sim.parent / 'SOURCE.json').read_bytes()).hexdigest()
    if report['packages']['torch']:
        try:
            import torch
            report['checks']['cuda_available'] = torch.cuda.is_available()
            report['gpu'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except Exception as exc:
            report['checks']['cuda_available'] = {'error': str(exc)}
    report['cpu_ready'] = all(report['packages'][n] for n in ['highspy', 'z3-solver', 'numpy', 'httpx']) and all(
        report['checks'].get(k) is True for k in ['highs', 'z3', 'simulator_cli'])
    report['note'] = 'Environment inspection only; no kernel correctness or timing evaluation performed.'
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
