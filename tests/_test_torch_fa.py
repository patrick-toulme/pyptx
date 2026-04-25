"""Quick torch path test on the FA kloop kernel — tests the hardest
wgmma + TMA + K-loop + online softmax case end-to-end through torch."""
import sys; sys.path.insert(0, ".")
import torch
import math

device = torch.device("cuda")
torch.manual_seed(0)

from examples.hopper.experimental.flash_attention_wgmma_kloop import build_flash_attention_kloop, BM, HEAD_DIM

for N_seq in [32, 64, 128]:
    k_fn = build_flash_attention_kloop(N_seq)
    q = (torch.randn(BM, HEAD_DIM, device=device) * 0.3).to(torch.bfloat16)
    k = (torch.randn(N_seq, HEAD_DIM, device=device) * 0.3).to(torch.bfloat16)
    v = (torch.randn(N_seq, HEAD_DIM, device=device) * 0.3).to(torch.bfloat16)
    k_t = k.T.contiguous()                # (HEAD_DIM, N_seq)
    out = k_fn(q, k_t, v)
    # torch reference
    qf = q.float(); kf = k.float(); vf = v.float()
    ref = torch.softmax(qf @ kf.T / math.sqrt(HEAD_DIM), dim=-1) @ vf
    diff = float((out - ref).abs().max())
    ok = torch.allclose(out, ref, atol=2e-2, rtol=1e-2)
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] fa_kloop N_seq={N_seq:4d}  max_abs={diff:.3e}")
