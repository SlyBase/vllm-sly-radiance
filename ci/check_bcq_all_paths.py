"""BCQ (block-clustered (BCQ)) block tables through EVERY PSW kernel band, checkpoint nibbles.

set_mag_table() replaces kMag with 256 learned 8-entry e4m3 tables; the PSW-folded Ws byte is then
a free table id per 32-block instead of clamp(Wref - Ws, 0, 15). Per shape and M this runs the
PSW kernels (launch: decode / wide / folded, launch_at: A-tiled) with random table ids on the
checkpoint's nibbles and compares against an fp32 torch reference of the SAME dequantized weights:
  W[n, k] = sign * T[id[k/32, n]][mag] * 2^(Wref[n] - 127)
rel <= 2e-3 (only fp32 accumulation order differs; the e4m3 table values are exact in the WMMA).
Then it restores the OCP table and checks the plain fold digest is unchanged (OCP rows 0..15 are
still the default kMag). env: MS, SHAPES, LAYER (12), ATTN_LAYER (3), TABLES (bcq_tables.pt path).
"""
import hashlib, json, os, sys
os.environ.setdefault("RADIANCE_MXFP4_TN4_MIN_M", "513")
import torch
sys.path.insert(0, "/w")
import radiance_mxfp4 as R
from safetensors import safe_open
from tile_layout import tile

dev = torch.device("cuda"); stream = torch.cuda.current_stream().cuda_stream
E = R._ext
assert R.WPERM and R.DECODE_MAX_M >= 128, "run with RADIANCE_MXFP4_WPERM=1 DECODE_MAX_M=128"
assert hasattr(E, "set_mag_table"), "this .so has no BCQ reader"
CK = "/ck"; PFX = "model.language_model."
IDX = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
L = int(os.environ.get("LAYER", "12")); LA = int(os.environ.get("ATTN_LAYER", "3"))
MS = [int(x) for x in os.environ.get("MS", "1,8,33,64,72,128,136,192,256,512,880,1514").split(",")]
E.set_at_skip_pad(int(os.environ.get("SKIP_PAD", "1")))
SH = {"gate_up": (L, "mlp", ["gate_proj", "up_proj"]), "down": (L, "mlp", ["down_proj"]),
      "qkvz": (L, "linear_attn", ["in_proj_qkv", "in_proj_z"]), "ba": (L, "linear_attn", ["in_proj_a", "in_proj_b"]),
      "out_proj": (L, "linear_attn", ["out_proj"]), "qkv": (LA, "self_attn", ["q_proj", "k_proj", "v_proj"]),
      "o_proj": (LA, "self_attn", ["o_proj"])}
SHAPES = os.environ.get("SHAPES", ",".join(SH)).split(",")

R._decode_scratch[0] = torch.empty(4 * 128 * 36864, dtype=torch.float32, device=dev)
R._decode_scratch[1] = torch.zeros(36864 // 128 + 8, dtype=torch.int32, device=dev)
E.set_decode_scratch(R._decode_scratch[0].data_ptr(), R._decode_scratch[0].numel() * 4, R._decode_scratch[1].data_ptr())

T = torch.load(os.environ.get("TABLES", "/w/bcq_tables.pt")).float()          # [256, 8], e4m3-exact
assert T.shape == (256, 8)
TB = T.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()                   # 2048 B, row-major
OCP = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
TOCP = torch.zeros(256, 8)
TOCP[:16] = torch.stack([OCP * 2.0 ** -d for d in range(16)])
TOCPB = TOCP.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()


def tens(key):
    with safe_open(f"{CK}/{IDX[key]}", "pt", device="cpu") as f:
        return f.get_tensor(key)


def h(t):
    return hashlib.sha256(t.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]


def dequant(w, ids, ref, tab):
    """checkpoint-order nibbles [N, K/2] + table ids [K/32, N] -> fp32 [N, K]."""
    nib = torch.stack([w & 15, w >> 4], -1).reshape(w.shape[0], -1).long()
    t = ids.T.long().repeat_interleave(32, 1)                                  # [N, K]
    mag = tab.to(dev)[t, nib & 7]
    return torch.where(nib >= 8, -mag, mag) * torch.exp2(ref.float() - 127.0)[:, None]


def run(fn, a, Wt, St, ref, xs, M, N, K):
    out = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
    fn(a.data_ptr(), Wt.data_ptr(), St.data_ptr(), ref.data_ptr(), xs.data_ptr(), out.data_ptr(), M, N, K, stream)
    torch.cuda.synchronize()
    return out


from vllm import _custom_ops as ops
emb = tens(f"{PFX}embed_tokens.weight")
g = torch.Generator().manual_seed(0)
x_emb = emb[torch.randint(0, emb.shape[0], (max(MS),), generator=g)].to(dev, torch.bfloat16)
ok_n = bad = 0
for name in SHAPES:
    Ly, blk, parts = SH[name]
    w = torch.cat([tens(f"{PFX}layers.{Ly}.{blk}.{n}.weight_packed") for n in parts], 0).to(dev)
    s = torch.cat([tens(f"{PFX}layers.{Ly}.{blk}.{n}.weight_scale") for n in parts], 0).to(dev)
    N, K = w.shape[0], w.shape[1] * 2
    st = s.T.contiguous(); ref = R.make_row_ref(st)
    W2 = R.permute_w(w, N, K).contiguous(); R.psw_pack_w_(W2)
    ids = torch.randint(0, 256, (K // 32, N), generator=g, dtype=torch.uint8).to(dev)
    Wd = dequant(w, ids, ref, T)
    S_ocp = R.psw_fold_ws(st, ref)
    xb = x_emb if K == x_emb.shape[1] else torch.cat([x_emb] * -(-K // x_emb.shape[1]), 1)[:, :K].contiguous()
    print(f"SHAPE {name} N={N} K={K} layer={Ly}", flush=True)
    for M in MS:
        xq, xs = ops.scaled_fp8_quant(xb[:M].contiguous(), scale=None, use_per_token_if_dynamic=True)
        xs = xs.view(-1).float().contiguous()
        rows = min(M, 64)
        r_ = (xq[:rows].float() * xs[:rows, None]) @ Wd.T
        paths = [("launch", xq)]
        if M >= 513 and K % 128 == 0:
            paths.append(("launch_at", tile(xq.view(torch.uint8)).contiguous()))
        for pname, a in paths:
            fn = getattr(E, pname)
            E.set_at_psw(2)
            E.set_mag_table(TOCPB.data_ptr()); d0 = h(run(fn, a, W2, S_ocp, ref, xs, M, N, K))
            E.set_mag_table(TB.data_ptr()); out = run(fn, a, W2, ids, ref, xs, M, N, K)
            E.set_mag_table(TOCPB.data_ptr()); d1 = h(run(fn, a, W2, S_ocp, ref, xs, M, N, K))
            E.set_at_psw(0)
            rel = ((out[:rows].float() - r_).norm() / r_.norm().clamp_min(1e-9)).item()
            good = rel <= 2e-3 and d0 == d1
            ok_n += good; bad += not good
            print(f"BCQ {name} {pname} M={M} rel={rel:.6f} ocp_digest {'SAME' if d0 == d1 else 'CHANGED'} "
                  f"{'OK' if good else 'WRONG'}", flush=True)
    del w, s, W2, Wd, ids, S_ocp; torch.cuda.empty_cache()
print(f"TOTAL OK={ok_n} BAD={bad}", flush=True)
