"""Run full-shape CPU shadow against an independent float64 reference."""
import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path
import numpy as np


def check(workload, config, source):
    spec = importlib.util.spec_from_file_location('candidate_shadow', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = module.build(workload, config)
    m, k = workload['inputs']['A']['shape']; n = workload['inputs']['B']['shape'][1]
    rng = np.random.default_rng(42)
    cases = []
    for pattern in ['normal_seed_42', 'zero_rows_signed_columns']:
        if pattern.startswith('normal'):
            a = (rng.standard_normal((m,k), dtype=np.float32)*0.1).astype(np.float16)
            b = (rng.standard_normal((k,n), dtype=np.float32)*0.1).astype(np.float16)
        else:
            a = np.ones((m,k),dtype=np.float16); a[::2] = 0
            b = np.full((k,n),0.125,dtype=np.float16); b[:,::2] *= -1
        reference = (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float16)
        start = time.monotonic(); actual = run({'A':a,'B':b})['C']; elapsed=(time.monotonic()-start)*1000
        error = np.abs(actual.astype(np.float64)-reference.astype(np.float64))
        limit = workload['tolerance']['atol']+workload['tolerance']['rtol']*np.abs(reference.astype(np.float64))
        passed = actual.shape == reference.shape and actual.dtype == np.float16 and bool(np.all(np.isfinite(actual))) and bool(np.all(error<=limit))
        cases.append(dict(pattern=pattern, passed=passed, max_abs_error=float(error.max()),
                          mismatches=int(np.count_nonzero(error>limit)), elapsed_ms=elapsed,
                          timing_source='CPU shadow wall clock; not GPU or GB10 simulator'))
    return dict(scope='CPU shadow only; not TileLang execution', passed=all(c['passed'] for c in cases),
                source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),config=config,
                reference='NumPy float64 full matmul, rounded to float16',
                input_dtype='float16', accumulation_dtype='float32', output_dtype='float16',
                tolerance=workload['tolerance'],hardware=platform.platform(),numpy_version=np.__version__, cases=cases)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--workload',required=True);p.add_argument('--config',required=True)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=check(json.loads(Path(a.workload).read_text()),json.loads(Path(a.config).read_text()),a.source)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
