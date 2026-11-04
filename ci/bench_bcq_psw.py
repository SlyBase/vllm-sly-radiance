"""PSW GEMM timing on checkpoint shapes, for A/B of two radiance_mxfp4_fp8.so builds (run once per .so).
launch (decode/folded) and launch_at (A-tiled prefill) with set_at_psw(2); BCQ builds get a table upload.
env: MS (8,64,2048,4096), SHAPES, ITERS (40). Prints one TIME line per (shape, path, M) in microseconds.
"""
import json, os, sys
os.environ.setdefault("RADIANCE_MXFP4_TN4_MIN_M", "513")
import torch
sys.path.insert(0, "/w")
import radiance_mxfp4 as R
from safetensors import safe_open
from tile_layout import tile

dev = torch.device("cuda"); stream = torch.cuda.current_stream().cuda_stream
E = R._ext
CK = "/ck"; PFX = "model.language_model.layers.12."
IDX = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
MS = [int(x) for x in os.environ.get("MS", "8,64,2048,4096").split(",")]
IT = int(os.environ.get("ITERS", "40"))
E.set_at_skip_pad(1)
SH = {"gate_up": ["mlp.gate_proj", "mlp.up_proj"], "down": ["mlp.down_proj"],
      "qkvz": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]}
R._decode_scratch[0] = torch.empty(4 * 128 * 36864, dtype=torch.float32, device=dev)
R._decode_scratch[1] = torch.zeros(36864 // 128 + 8, dtype=torch.int32, device=dev)
E.set_decode_scratch(R._decode_scratch[0].data_ptr(), R._decode_scratch[0].numel() * 4, R._decode_scratch[1].data_ptr())
bcq = hasattr(E, "set_mag_table")
if bcq:
    tb = torch.load("/w/bcq_tables.pt").float().to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    E.set_mag_table(tb.data_ptr())
print("BUILD", "bcq" if bcq else "ocp", flush=True)


def tens(k):
    with safe_open(f"{CK}/{IDX[k]}", "pt", device="cpu") as f:
        return f.get_tensor(k)


E.set_at_psw(2)
for name in os.environ.get("SHAPES", ",".join(SH)).split(","):
    w = torch.cat([tens(PFX + p + ".weight_packed") for p in SH[name]]).to(dev)
    s = torch.cat([tens(PFX + p + ".weight_scale") for p in SH[name]]).to(dev)
    N, K = w.shape[0], w.shape[1] * 2
    st = s.T.contiguous(); ref = R.make_row_ref(st)
    W2 = R.permute_w(w, N, K).contiguous(); R.psw_pack_w_(W2)
    S2 = R.psw_fold_ws(st, ref)
    if not bcq:
        S2 = (S2.int() * 8).to(torch.uint8)   # pre-BCQ builds take the byte offset
    else:
        S2 = torch.randint(0, 256, S2.shape, dtype=torch.uint8, device=dev)
    for M in MS:
        x = torch.randn(M, K, device=dev).to(torch.float8_e4m3fn)
        xs = torch.full((M,), 1e-3, device=dev)
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        for pname, a in [("launch", x)] + ([("launch_at", tile(x.view(torch.uint8)).contiguous())] if M >= 513 else []):
            fn = getattr(E, pname)
            go = lambda: fn(a.data_ptr(), W2.data_ptr(), S2.data_ptr(), ref.data_ptr(), xs.data_ptr(), out.data_ptr(), M, N, K, stream)
            for _ in range(5):
                go()
            t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
            ts = []
            for _ in range(5):
                t0.record()
                for _ in range(IT):
                    go()
                t1.record(); torch.cuda.synchronize(); ts.append(t0.elapsed_time(t1) * 1000 / IT)
            print(f"TIME {name} {pname} M={M} us={sorted(ts)[2]:.1f}", flush=True)
E.set_at_psw(0)
