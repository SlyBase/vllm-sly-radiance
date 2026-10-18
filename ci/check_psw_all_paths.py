"""PSW weight layout (RADIANCE_MXFP4_AT_PSW=2) across EVERY MXFP4 kernel band, checkpoint weights.

For each production shape (gate_up, down, qkvz, ba, out_proj, qkv, o_proj) the same layer runs
twice per M: plain fragment-order weights with set_at_psw(0), and the PSW-repacked copy (the
loader's in-place psw_pack_w_ + psw_fold_ws) with set_at_psw(2). The sha256 of the bf16 output
must match (DIGEST ... SAME). Bands covered through launch(): decode M 1..64 and 72..128 (all
DTM), wide decode 136..192, folded 256..512, folded prefill 880/1514/1760; through launch_at
(fragment-tiled A, real-producer layout via tile_layout.tile): 880/1514/1760.
v0 is also checked against the fp32 reference on the first 64 rows (rel <= 0.02).
Refusal: with set_at_psw(2) a launch without wref (per-block path) must throw.
env: MS (override M list), SHAPES, LAYER (12), ATTN_LAYER (3).
"""
import hashlib, json, os, sys
# production GT knobs: TN4 from 513 (read by the .so at its first launch) and SKIPPAD (set below)
os.environ.setdefault("RADIANCE_MXFP4_TN4_MIN_M", "513")
import torch
sys.path.insert(0, "/w")
import radiance_mxfp4 as R
from safetensors import safe_open
from tile_layout import tile

dev = torch.device("cuda"); stream = torch.cuda.current_stream().cuda_stream
E = R._ext
assert R.WPERM and R.DECODE_MAX_M >= 128, "run with RADIANCE_MXFP4_WPERM=1 DECODE_MAX_M=128"
assert E.psw_reader_mode() >= 2, "this .so has no PSW decode reader"
CK = "/ck"; PFX = "model.language_model."
IDX = json.load(open(f"{CK}/model.safetensors.index.json"))["weight_map"]
L = int(os.environ.get("LAYER", "12")); LA = int(os.environ.get("ATTN_LAYER", "3"))
MS = [int(x) for x in os.environ.get("MS", ",".join(map(str, list(range(1, 129)) + [129, 136, 144, 152, 160, 168, 176, 184, 192, 193, 200, 256, 400, 512, 513, 700, 880, 1100, 1514, 1760, 2048]))).split(",")]
E.set_at_skip_pad(int(os.environ.get("SKIP_PAD", "1")))
SH = {"gate_up": (L, "mlp", ["gate_proj", "up_proj"]), "down": (L, "mlp", ["down_proj"]),
      "qkvz": (L, "linear_attn", ["in_proj_qkv", "in_proj_z"]), "ba": (L, "linear_attn", ["in_proj_a", "in_proj_b"]),
      "out_proj": (L, "linear_attn", ["out_proj"]), "qkv": (LA, "self_attn", ["q_proj", "k_proj", "v_proj"]),
      "o_proj": (LA, "self_attn", ["o_proj"])}
SHAPES = os.environ.get("SHAPES", ",".join(SH)).split(",")

R._decode_scratch[0] = torch.empty(4 * 128 * 36864, dtype=torch.float32, device=dev)
R._decode_scratch[1] = torch.zeros(36864 // 128 + 8, dtype=torch.int32, device=dev)
E.set_decode_scratch(R._decode_scratch[0].data_ptr(), R._decode_scratch[0].numel() * 4, R._decode_scratch[1].data_ptr())


def tens(key):
    with safe_open(f"{CK}/{IDX[key]}", "pt", device="cpu") as f:
        return f.get_tensor(key)


def h(t):
    return hashlib.sha256(t.view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]


from vllm import _custom_ops as ops
emb = tens(f"{PFX}embed_tokens.weight")
g = torch.Generator().manual_seed(0)
ids = torch.randint(0, emb.shape[0], (max(MS),), generator=g)
x_emb = emb[ids].to(dev, torch.bfloat16)
nsame = nwrong = 0
for name in SHAPES:
    Ly, blk, parts = SH[name]
    w = torch.cat([tens(f"{PFX}layers.{Ly}.{blk}.{n}.weight_packed") for n in parts], 0).to(dev)
    s = torch.cat([tens(f"{PFX}layers.{Ly}.{blk}.{n}.weight_scale") for n in parts], 0).to(dev)
    N, K = w.shape[0], w.shape[1] * 2
    st = s.T.contiguous(); ref = R.make_row_ref(st)
    W0 = R.permute_w(w, N, K).contiguous()
    W2 = W0.clone(); R.psw_pack_w_(W2)                      # the loader's in-place, banded pack
    assert torch.equal(W2, R.psw_pack_w(W0, 2)), "banded in-place pack != psw_pack_w"
    S2 = R.psw_fold_ws(st, ref)
    # activation: real token embeddings (K = 5120), repeated / random-signed widening for other K
    if K == x_emb.shape[1]:
        xb = x_emb
    else:
        rep = -(-K // x_emb.shape[1])
        xb = torch.cat([x_emb] * rep, 1)[:, :K].contiguous()
    print(f"SHAPE {name} N={N} K={K} layer={Ly}", flush=True)
    for M in MS:
        xq, xs = ops.scaled_fp8_quant(xb[:M].contiguous(), scale=None, use_per_token_if_dynamic=True)
        xs = xs.view(-1).float().contiguous()
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        paths = [("launch", xq)]
        if M >= 513 and K % 128 == 0:
            paths.append(("launch_at", tile(xq.view(torch.uint8)).contiguous()))
        for pname, a in paths:
            fn = getattr(E, pname)
            res = {}
            for v, Wt, St in ((0, W0, st), (2, W2, S2), (3, W2, S2)):   # 3 = PSW again (determinism)
                E.set_at_psw(2 if v == 3 else v)
                out.fill_(0)
                fn(a.data_ptr(), Wt.data_ptr(), St.data_ptr(), ref.data_ptr(), xs.data_ptr(), out.data_ptr(), M, N, K, stream)
                torch.cuda.synchronize(); res[v] = h(out)
                if v == 0 and M in (1, 8, 64, 136, 400, 1514):
                    rows = min(M, 64)
                    r_ = R._exact_ref(xq[:rows], xs[:rows], W0, st, N, K)
                    rel = ((out[:rows].float() - r_).norm() / r_.norm().clamp_min(1e-9)).item()
                    print(f"REF {name} {pname} M={M} rel={rel:.5f}" + (" WRONG" if rel > 0.02 else ""), flush=True)
            E.set_at_psw(0)
            ok = res[0] == res[2] == res[3]
            nsame += ok; nwrong += not ok
            print(f"DIGEST {name} {pname} M={M} v0={res[0]} v2={res[2]} v2b={res[3]} {'SAME' if ok else 'WRONG'}", flush=True)
    del w, s, W0, W2, S2; torch.cuda.empty_cache()

# refusal: the per-block path (wref = 0) has no PSW reader and must throw while PSW is set
E.set_at_psw(2)
try:
    E.launch(0, 0, 0, 0, 0, 0, 1, 16, 128, stream)   # throws before any launch
    print("REFUSE perblock NOT RAISED BAD", flush=True); nwrong += 1
except Exception as e:
    print(f"REFUSE perblock raised ok: {str(e)[:90]}", flush=True)
E.set_at_psw(0)
print(f"TOTAL SAME={nsame} BAD={nwrong}", flush=True)
