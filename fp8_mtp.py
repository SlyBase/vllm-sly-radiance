#!/usr/bin/env python3
"""MXFP4 body + FP8 drafter, in one checkpoint.

Run this once against amd/Qwen3.8-27B-Quark-AWQ-MXFP4 to produce the checkpoint serve-mxfp4.sh
serves (./setup-mxfp4.sh drives it for you). It is not optional: AMD ships the MTP head bf16 but names it in neither `exclude` nor
`layer_quant_config`, so vLLM's quark config falls through to `global_quant_config` (mxfp4) for
`mtp.*`, builds a packed uint8 weight of half the input width, and dies loading the full-width
bf16 tensor into it --

    AssertionError: Attempted to load weight (torch.Size([5120, 10240]))
                    into parameter (torch.Size([5120, 5120]))

-- before any of the quality argument below comes into play. Writing an explicit
`layer_quant_config` for the eight MTP projections is what makes the head loadable at all.

MXFP4 on the drafter failed twice: plain RTN cost acceptance 2.5 -> 2.21, and AWQ calibration did
not rescue it (0-5% error improvement; the alpha search chose a=0.1-0.2, and a=0.0 for mtp.fc,
because MXFP4's per-32 E8M0 block exponent already does most of what per-channel scaling would).
The error is intrinsic to 4 bits, and for a drafter accuracy IS throughput.

FP8 trades a smaller bandwidth win for a much smaller error: e4m3 per-channel is ~2-3% relative
versus MXFP4's ~11.6%. The drafter is 34% of decode weight traffic at n=8, so fp8 removes ~17% of
total decode traffic instead of 25% -- but should actually hold acceptance.

vLLM's quark config supports this natively: `layer_quant_config` is matched with fnmatch, and
QuarkW8A8Fp8 wants weight fp8_e4m3 static per_channel + input_tensors fp8_e4m3 dynamic (so no
input_scale is stored). Explicit layer names are used rather than a `*q_proj` glob, which would
also match the body's 64 layers.

Checkpoint layout. The source may be a single file (AMD ships the release as one ~19 GiB
`model.safetensors`) or a sharded one (`model.safetensors.index.json` + `model-0000X-of-0000Y.safetensors`,
the standard HF layout a re-sharded copy uses). The output is sharded by default -- HF's layout,
5 GiB per shard, `RADIANCE_SHARD_GIB` to change, `RADIANCE_SINGLE_FILE=1` to write one file.
Sharding is what matters on low-RAM hosts: vLLM loads a sharded checkpoint shard-by-shard
(maps one shard at a time) instead of a single monolithic file, so the peak host RAM for the
weight load is the largest shard, not the whole model -- the difference between a 16 GiB box
loading in minutes and one swapping for hours. The serve path (serve-mxfp4.sh) hands vLLM the
checkpoint directory and is layout-agnostic, so a sharded build serves with no further change.
"""
import json, os, shutil, struct, sys, pathlib

# Checked before importing torch, so a bare invocation on a host without it still explains itself.
if len(sys.argv) != 3:
    sys.exit(f"usage: {pathlib.Path(sys.argv[0]).name} <src-checkpoint> <dst-checkpoint>\n"
             "  src  a Quark AWQ MXFP4 snapshot, single-file or sharded (model.safetensors.index.json),\n"
             "       e.g. the directory under ~/.cache/huggingface/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/\n"
             "  dst  where to write it, e.g. $MODELS/Qwen3.8-27B-MXFP4-mtpfp8\n"
             "\n"
             "Output is sharded by default (5 GiB shards; RADIANCE_SHARD_GIB to resize,\n"
             "RADIANCE_SINGLE_FILE=1 to write one file). Needs torch. If the host has none,\n"
             "run it inside the image -- see the README.")

import torch

MTP = ["mtp.fc", "mtp.layers.0.mlp.down_proj", "mtp.layers.0.mlp.gate_proj",
       "mtp.layers.0.mlp.up_proj", "mtp.layers.0.self_attn.k_proj",
       "mtp.layers.0.self_attn.o_proj", "mtp.layers.0.self_attn.q_proj",
       "mtp.layers.0.self_attn.v_proj"]
FP8_MAX = 448.0
TDT = {"BF16": torch.bfloat16, "U8": torch.uint8, "F32": torch.float32}
SDT = {torch.uint8: "U8", torch.float32: "F32", torch.bfloat16: "BF16",
       torch.float8_e4m3fn: "F8_E4M3"}
# Default shard size for the output (bytes). Sharding is the low-RAM win: vLLM maps one shard at a time.
SHARD_MAX = int(float(os.environ.get("RADIANCE_SHARD_GIB", "5")) * 1024**3)
SINGLE_FILE = os.environ.get("RADIANCE_SINGLE_FILE") == "1"

def spec(dtype, dynamic, qscheme, ch_axis):
    return {"block_size": None, "ch_axis": ch_axis, "dtype": dtype, "enable_buffer_reuse": False,
            "group_size": None, "is_dynamic": dynamic, "is_scale_quant": False,
            "max_input_numel": 4194304, "mx_element_dtype": None,
            "observer_cls": "PerChannelMinMaxObserver" if qscheme == "per_channel"
                            else "PerTensorMinMaxObserver",
            "qscheme": qscheme, "round_method": "half_even", "scale_calculation_mode": None,
            "scale_format": None, "scale_type": "float", "symmetric": True}

FP8_CFG = {"bias": None, "output_tensors": None, "target_device": None,
           "weight": spec("fp8_e4m3", False, "per_channel", 0),
           "input_tensors": spec("fp8_e4m3", True, "per_tensor", -1)}

def read_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

def load(p, hdr, base, name):
    m = hdr[name]; s, e = m["data_offsets"]
    with open(p, "rb") as f:
        f.seek(base + s); buf = bytearray(f.read(e - s))
    return torch.frombuffer(buf, dtype=TDT[m["dtype"]]).reshape(m["shape"])

def raw(t):
    return t.contiguous().view(torch.uint8).numpy().tobytes()

def open_source(src):
    """Map every source tensor to (file path, safetensors entry, file data base offset).

    loc: tensor name -> (path, entry, base). A single-file source maps everything to
    model.safetensors; a sharded one (model.safetensors.index.json) maps each tensor to its shard
    via the weight_map. base is the file's 8-byte header padding, so tensor data sits at
    base + entry["data_offsets"][0].
    """
    idx = src / "model.safetensors.index.json"
    if idx.exists():
        weight_map = json.loads(idx.read_text())["weight_map"]
        hdrs = {}
        loc = {}
        for name, shard in weight_map.items():
            p = src / shard
            if p not in hdrs:
                hdrs[p] = read_header(p)
            loc[name] = (p, hdrs[p][0][name], hdrs[p][1])
        return loc
    p = src / "model.safetensors"
    hdr, base = read_header(p)
    return {k: (p, hdr[k], base) for k in hdr if k != "__metadata__"}

def pack_shards(sizes, max_bytes):
    """Greedy first-fit packing of an ordered [name, nbytes] list into shards of <= max_bytes.

    Returns a list of shards, each a list of (name, nbytes). A tensor larger than max_bytes gets
    its own shard (safetensors cannot split a tensor across files).
    """
    shards = []; cur = []; cur_bytes = 0
    for name, nb in sizes:
        if cur and cur_bytes + nb > max_bytes:
            shards.append(cur); cur = []; cur_bytes = 0
        cur.append((name, nb)); cur_bytes += nb
    if cur:
        shards.append(cur)
    return shards

def write_safetensors(outf, order, bytes_of, entries):
    """Write one safetensors file: 8-byte length + JSON header (8-byte padded) + tensor data.

    order: ordered tensor names. bytes_of(name) -> raw bytes for that tensor. entries: name ->
    {dtype, shape} (data_offsets are filled here).
    """
    hdr, off = {}, 0
    for name in order:
        nb = len(bytes_of(name)) if isinstance(bytes_of(name), bytes) else entries[name]["_nb"]
        hdr[name] = {k: entries[name][k] for k in ("dtype", "shape")}
        hdr[name]["data_offsets"] = [off, off + nb]
        off += nb
    hdr["__metadata__"] = {"format": "pt"}
    blob = json.dumps(hdr).encode()
    blob += b" " * ((8 - (len(blob) % 8)) % 8)
    with open(outf, "wb") as fout:
        fout.write(struct.pack("<Q", len(blob))); fout.write(blob)
        for name in order:
            b = bytes_of(name)
            if isinstance(b, bytes):
                fout.write(b)
            else:
                # (path, data_start, nbytes): stream from source in 32 MiB chunks
                sp, s0, left = b
                with open(sp, "rb") as fin:
                    fin.seek(s0)
                    while left:
                        c = fin.read(min(left, 32 << 20)); fout.write(c); left -= len(c)

def main(src_dir, dst_dir):
    src, dst = pathlib.Path(src_dir), pathlib.Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    loc = open_source(src)

    # Requantize the MTP head to fp8: per-channel weight + weight_scale, replacing each bf16 weight.
    new = {}
    print(f"{'tensor':34s} {'shape':>18} {'rel err':>9}   (MXFP4 was ~0.116)")
    for name in MTP:
        p, entry, base = loc[name + ".weight"]
        w = load(p, read_header(p)[0], base, name + ".weight").float()
        amax = w.abs().amax(dim=1).clamp(min=1e-12)          # per output channel
        s = (amax / FP8_MAX).float()
        q = (w / s.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        rel = ((q.float() * s.unsqueeze(1) - w).norm() / w.norm()).item()
        print(f"{name:34s} {str(tuple(w.shape)):>18} {rel:9.4f}")
        new[name + ".weight"] = q
        new[name + ".weight_scale"] = s

    # Ordered output tensor list: source order, with each MTP .weight replaced by its fp8 weight +
    # scale. Each entry carries its bytes provider (raw bytes for the quantized head, a stream
    # spec for copied tensors) and its safetensors metadata.
    order = []; entries = {}
    def add(name, kind, nbytes):
        order.append(name)
        if kind == "new":
            entries[name] = {"dtype": SDT[new[name].dtype], "shape": list(new[name].shape), "_nb": nbytes}
        else:
            _, entry, _ = loc[name]
            entries[name] = {"dtype": entry["dtype"], "shape": entry["shape"], "_nb": nbytes}

    for name in loc:
        if name in new:
            # A source tensor that gets transformed (an MTP .weight): emit its fp8 weight plus
            # the weight_scale that only exists in `new` (the source had none).
            add(name, "new", new[name].numel() * new[name].element_size())
            scale_name = name + "_scale"
            if scale_name in new:
                add(scale_name, "new", new[scale_name].numel() * new[scale_name].element_size())
        else:
            _, entry, base = loc[name]
            add(name, "copy", entry["data_offsets"][1] - entry["data_offsets"][0])

    def bytes_of(name):
        if name in new:
            return raw(new[name])
        p, entry, base = loc[name]
        s0 = base + entry["data_offsets"][0]
        return (p, s0, entry["data_offsets"][1] - entry["data_offsets"][0])

    if SINGLE_FILE:
        write_safetensors(dst / "model.safetensors", order, bytes_of, entries)
        outf = dst / "model.safetensors"
        print(f"\nwrote {outf} ({outf.stat().st_size / 2**30:.2f} GiB)")
    else:
        sizes = [(name, entries[name]["_nb"]) for name in order]
        shards = pack_shards(sizes, SHARD_MAX)
        nsh = len(shards); width = max(len(str(nsh)), 3)
        weight_map = {}
        for i, shard in enumerate(shards):
            fname = f"model-{str(i + 1).zfill(width)}-of-{str(nsh).zfill(width)}.safetensors"
            shard_order = [name for name, _ in shard]
            write_safetensors(dst / fname, shard_order, bytes_of, entries)
            for name in shard_order:
                weight_map[name] = fname
            print(f"  {fname} ({(dst / fname).stat().st_size / 2**30:.2f} GiB)")
        total_size = sum(entries[n]["_nb"] for n in order)
        (dst / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2))
        print(f"\nwrote {nsh} shard(s), {total_size / 2**30:.2f} GiB total, "
              f"index at {dst / 'model.safetensors.index.json'}")

    cfg = json.loads((src / "config.json").read_text())
    qc = cfg["quantization_config"]
    qc["exclude"] = [e for e in qc["exclude"] if e not in MTP]
    qc["layer_quant_config"] = {name: FP8_CFG for name in MTP}
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"exclude {len(qc['exclude'])} entries; layer_quant_config {len(qc['layer_quant_config'])} fp8 layers")
    for n in ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
              "merges.txt", "preprocessor_config.json", "processor_config.json",
              "video_preprocessor_config.json", "chat_template.jinja"):
        if (src / n).exists():
            shutil.copy(src / n, dst / n)

main(sys.argv[1], sys.argv[2])
