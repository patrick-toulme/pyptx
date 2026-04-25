# Blackwell / Tcgen05 Epilogue Partition Probe

[:material-github: View on GitHub](https://github.com/patrick-toulme/pyptx/blob/dev/examples/blackwell/tcgen05_epilogue_partition_probe.py){ .md-button } 
[:material-file-code: `examples/blackwell/tcgen05_epilogue_partition_probe.py`](https://github.com/patrick-toulme/pyptx/blob/dev/examples/blackwell/tcgen05_epilogue_partition_probe.py){ .md-button }

## Overview

Blackwell tcgen05 epilogue partition diagnostic.

This uses the current no-TMA GEMM to measure the row/column support produced by
the handwritten TMEM epilogue. The goal is to formalize the observed residue
class lattice before we rewrite the epilogue partition.

## Source

??? example "Full source"

    ```python
    """Blackwell tcgen05 epilogue partition diagnostic.

    This uses the current no-TMA GEMM to measure the row/column support produced by
    the handwritten TMEM epilogue. The goal is to formalize the observed residue
    class lattice before we rewrite the epilogue partition.
    """
    from __future__ import annotations

    import os

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import numpy as np
    import torch

    try:
        from pyptx.examples.blackwell.gemm_experimental_blackwell import build_gemm_no_tma_debug
    except ImportError:
        from examples.blackwell.gemm_experimental_blackwell import build_gemm_no_tma_debug


    def run():
        kfun = build_gemm_no_tma_debug(128, 256, 64)
        kk = 0

        a = np.zeros((128, 64), dtype=np.float32)
        a[:, kk] = np.arange(1, 129, dtype=np.float32)
        b = np.zeros((64, 256), dtype=np.float32)
        b[kk, :] = 1.0
        out_rows = kfun(
            torch.tensor(a, device="cuda", dtype=torch.bfloat16),
            torch.tensor(b, device="cuda", dtype=torch.bfloat16).t().contiguous(),
        )
        torch.cuda.synchronize()

        a.fill(0)
        a[:, kk] = 1.0
        b.fill(0)
        b[kk, :] = np.arange(1, 257, dtype=np.float32)
        out_cols = kfun(
            torch.tensor(a, device="cuda", dtype=torch.bfloat16),
            torch.tensor(b, device="cuda", dtype=torch.bfloat16).t().contiguous(),
        )
        torch.cuda.synchronize()

        row0 = out_cols[0].float().cpu().numpy()
        row1 = out_cols[1].float().cpu().numpy()
        col0 = out_rows[:, 0].float().cpu().numpy()
        col1 = out_rows[:, 1].float().cpu().numpy()

        for name, vec in (("row0", row0), ("row1", row1), ("col0", col0), ("col1", col1)):
            nz = np.nonzero(vec)[0]
            print(name, "nz_count", len(nz), "nz", nz[:64].tolist())
            print(name, "vals", vec[nz[:32]].tolist())


    if __name__ == "__main__":
        run()
    ```
