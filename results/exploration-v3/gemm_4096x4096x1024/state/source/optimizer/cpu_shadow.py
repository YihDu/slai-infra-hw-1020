"""Executable CPU diagnostic only; not an execution of generated TileLang."""
import numpy as np


def build(workload, config):
    bm, bn, bk = (config[k] for k in ('block_M', 'block_N', 'block_K'))
    def run(inputs):
        a, b = inputs['A'].astype(np.float32), inputs['B'].astype(np.float32)
        m, k = a.shape
        n = b.shape[1]
        c = np.empty((m, n), dtype=np.float16)
        for i in range(0, m, bm):
            for j in range(0, n, bn):
                tile = np.zeros((min(bm, m-i), min(bn, n-j)), dtype=np.float32)
                for p in range(0, k, bk):
                    tile += a[i:i+bm, p:p+bk] @ b[p:p+bk, j:j+bn]
                c[i:i+bm, j:j+bn] = tile.astype(np.float16)
        return {'C': c}
    return run
