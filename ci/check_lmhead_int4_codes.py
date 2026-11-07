"""RADIANCE_LMHEAD_INT4_CODES: offline codes must pack bit-identically to the load-time RTN when they ARE the RTN
codes (same q, same scales), and load_codes must refuse shape/group/range mismatches. CPU only. Prints TOTAL OK/BAD."""
import os, sys, tempfile
import torch
sys.path.insert(0, "/w")
f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
os.environ["RADIANCE_LMHEAD_INT4_CODES"] = f
import radiance_lmhead_int4 as L

ok = bad = 0


def check(name, cond):
    global ok, bad
    print(f"{name}: {'OK' if cond else 'WRONG'}", flush=True)
    ok += bool(cond); bad += not cond


torch.manual_seed(0)
n, k, g = 384, 512, 128
w = torch.randn(n, k) * 0.02
packed, scales = L.quantize_int4_rows(w, g)
blk = w.view(n, k // g, g)
q = torch.round(blk / scales.float()[..., None]).clamp(-8, 7).to(torch.int8).view(n, k)   # same rule, chosen scale
p2, s2 = L.pack_codes(q, scales, "cpu")
check("pack_codes == load-time RTN packing", torch.equal(p2, packed) and torch.equal(s2, scales))
torch.save({"q": q, "s": scales, "group": g}, f)
q_, s_ = L.load_codes(n, k, g)
check("load_codes round trip", torch.equal(q_, q) and torch.equal(s_, scales))
check("fewer rows than vocab (padding) accepted", L.load_codes(n + 64, k, g)[0].shape[0] == n)
for what, d in (("group", {"q": q, "s": scales, "group": 64}), ("K", {"q": q[:, :256], "s": scales[:, :2], "group": g}),
                ("range", {"q": q.clone().fill_(-9), "s": scales, "group": g}), ("dtype", {"q": q.int(), "s": scales, "group": g})):
    torch.save(d, f)
    try:
        L.load_codes(n, k, g); check(f"refuses bad {what}", False)
    except RuntimeError:
        check(f"refuses bad {what}", True)
os.unlink(f)
print(f"TOTAL OK={ok} BAD={bad}")
