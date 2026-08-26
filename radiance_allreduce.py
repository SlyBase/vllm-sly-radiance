#!/usr/bin/env python3
"""radiance custom all-reduce: P2P-BAR AR for R9700s (gfx1201, TP 2/4/8, PCIe).

Mirrors vLLM's CustomAllreduce (ca_comm) interface, so it slots into CudaCommunicator.all_reduce's
existing size-gated dispatch. should_custom_ar returns True only for messages in a served band;
larger or other messages fall through to RCCL. The routing lives here, in Python: the library
exports plain per-kernel entry points and this file owns the bands.

TP2 (unchanged): Kernels: r4d.ar_oneshot_2rank_exact (exact bf16) and r4d.ar_oneshot_2rank_wht6 (Walsh-Hadamard rotated
6-bit payload, used for large messages when RADIANCE_USE_R4D_AR_QUANT=1). Both come from the R4D gfx1201 kernel
library; both are one-shot push all-reduces for exactly two P2P-connected ranks, which is what their
names say, and they share the same scratch, flags and seq counters.

TP4/TP8 (the wide family, one flag/seq/scratch set shared by every kernel, they publish in
one 2*seq flag value space, so a mixed stream is protocol-safe, the timing tests for these were done
on a system with ~28 GB/s write bandwidth, PCIe 5.0 x8):
  1..7 tokens          ar_oneshot_{ws}rank_exact   (both modes: latency band, exact)
  8..4096 tokens       ar_twoshot_{ws}rank_ti8     when RADIANCE_USE_R4D_AR_QUANT=1
                       ar_twoshot_{ws}rank_exact   otherwise (and wherever ti8 cannot serve:
                                                   no ws-8 ti8 kernel, fp32, non-tiling numel)
  above 4096 tokens    decline -> RCCL

PUSH model throughout. Each rank writes into its peers' IPC scratch, then reduces into out.
Only scratch and flags are IPC-shared; input/output stay local, so no cudagraph buffer
registration is needed. cudagraph-safe: the sequence number is a device-resident counter the
kernel increments (a host-passed seq would freeze at capture), and scratch is double-buffered
by seq parity. Multi-block throughout: each block owns a contiguous chunk and drives its own
flag handshake (one flag word per block at TP2, per (block, source rank) at width), spreading
the push across CUs (PCIe) and the reduce across CUs; the one-shots scale block count with the
message, the two-shots with the shard.

Env:
  RADIANCE_USE_R4D_AR (1)      1 = install the kernel on the TP group, 0 = RCCL
  RADIANCE_USE_R4D_AR_QUANT (1) 1 = the quantized wires (tiered-int8 two-shot at width,
                                rotated 6-bit at TP2) in their bands; 0 = exact everywhere
"""
import os
import sys
from contextlib import contextmanager

import torch
import torch.distributed as dist

_DTYPE_CODE = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}

# Largest message the TP2 kernel takes; anything above it falls back to RCCL. Sized to hold one
# prefill chunk's all-reduce (4096 tokens x 5120 channels x bf16 = 40 MiB), so a serve running the
# shipped --max-num-batched-tokens keeps the kernel for prefill as well as decode. Costs 2x this in
# IPC scratch per rank, which is trivial on 32 GB. The kernel is bit-identical to RCCL at every size.
_MAX_BYTES = 49152 * 1024
# Below this the exact bf16 kernel wins at TP2: compression only pays once the transfer is
# bandwidth-bound.
_QUANT_MIN_BYTES = 128 * 1024
# The wide one-shot/two-shot crossover: 6 tokens of the 5120-channel residual stream. At and
# below it the one-shot's single hop wins on latency; above it the two-shot's ring-class bytes
# win, on the quantized wire when the flag is up. (7 tokens measured as a spike as one-shot)
_WIDE_ONESHOT_MAX_ELEMS = 6 * 5120


def _log(msg):
    sys.stderr.write(f"[radiance] {msg}\n")
    sys.stderr.flush()


class RadianceAllreduce:
    """Drop-in for vLLM's ca_comm (CustomAllreduce). Same public surface:
    `disabled`, `should_custom_ar(inp)`, `custom_all_reduce(inp)`, `capture()`."""

    def __init__(self, group, device):
        self.disabled = True
        self._ar_exact = None
        self._ar_wht6 = None
        self.group = group
        try:
            self.world_size = dist.get_world_size(group)
            self.rank = dist.get_rank(group)
        except Exception as e:
            _log(f"custom AR: no process group ({e!r})")
            return
        try:
            import r4d as ext
        except Exception as e:
            _log(f"custom AR disabled: r4d import failed ({e!r})")
            return
        self._ext = ext

        # "exactly 2 ranks" is a property of the kernel, not of this file, so ask the library
        # whether it has an all-reduce for this group instead of keeping a copy of the number.
        # The dtype in the question is the production one; the kernel takes fp16 and fp32 through
        # the same entry point, and should_custom_ar() is what gates a message's dtype per call.
        name = ext.select("allreduce", world_size=self.world_size, exact=1, dtype="bf16")
        if name is None:
            _log(f"custom AR disabled: no exact all-reduce kernel for "
                 f"world_size={self.world_size}")
            return
        self._ar_exact = getattr(ext, name)
        self.wide = self.world_size > 2

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        self.device = device
        self.drain = 3        # s_wait_storecnt drain (correct for fine-grained/uncached scratch)
        self.acq = 0          # no explicit acquire fence (uncached reads are already fresh)
        self.nt = 1024        # threads/block (tuned)
        self.min_nb = 4       # block count scales with message size, clamped to [min_nb, max_nb]
        self.words_per_block = 1400   # 16B words/block target for the block-count heuristics
        fine = True           # fine-grained (uncached) IPC scratch: posts straight to the fabric

        if not self.wide:
            self.max_bytes = _MAX_BYTES
            self.max_nb = min(24, int(ext.AR_MAX_BLOCKS))
            maxb = int(ext.AR_MAX_BLOCKS)
        else:
            ws = self.world_size
            # The two-shots are bound by NAME: the registry cannot separate them from the
            # one-shot (no capability cap does, the 7-token edge is a routing decision this
            # file owns), and picking a kernel by its full name is deliberate. ti8 exists at
            # 4 ranks only; without it the quant flag simply routes the two-shot band to the 
            # exact wire.
            self._ar_ts = getattr(ext, f"ar_twoshot_{ws}rank_exact")
            self._ar_qts = getattr(ext, f"ar_twoshot_{ws}rank_ti8", None)
            self.oneshot_max_elems = _WIDE_ONESHOT_MAX_ELEMS
            self.ts_max_elems = int(ext.AR_TWOSHOT_WIDE_MAX_ELEMS)
            self._q_tile = int(ext.AR_TI8_GROUP) * ws  # ti8 groups must tile every shard
            self.pub = 1                # one release fence per publish, not one per peer
            self.max_nb = min(int(ext.AR_WIDE_NB_DESIGN), int(ext.AR_WIDE_MAX_BLOCKS))
            self.ts_max_nb = 24   # two-shot cap for the SHARD-based heuristic (measured:
                                  # full-message nb over-launches the mid band)
            maxb = int(ext.AR_WIDE_MAX_BLOCKS)
            # Slot stride: the 16-bit two-shot at its ceiling dominates (regions A+B =
            # 2x shard); the ti8 wire and the 7-token one-shot both need less. An fp32
            # message the 16-bit sizing cannot hold declines in _ts_ok rather than
            # inflating every slot.
            self.max_bytes = max(self.oneshot_max_elems * 4,
                                 self.ts_max_elems * 2 * 2 // ws)
        self.max_bytes = (self.max_bytes // 16) * 16     # 16B (uint4) alignment
        self.slot16 = self.max_bytes // 16               # per-slot capacity in 16B words

        try:
            torch.cuda.set_device(self.device)
            # Scratch is one slot per (parity, source rank) at width, 2 slots at ws=2 where
            # parity alone names the single remote writer. Flags: one word per (block,
            # source rank) at width, one per block at ws=2. One set serves every kernel in
            # the family (shared 2*seq flag space).
            nslots = 2 * self.world_size if self.wide else 2
            flag_words = maxb * self.world_size if self.wide else maxb
            self._scratch, sc_h, fine_used = self._alloc(nslots * self.max_bytes, fine)
            self._flags, fl_h, _ = self._alloc(flag_words * 4, fine)
            sc_handles = [None] * self.world_size
            fl_handles = [None] * self.world_size
            dist.all_gather_object(sc_handles, sc_h, group=group)
            dist.all_gather_object(fl_handles, fl_h, group=group)
            # Peers in ascending global rank with this rank removed -- the order the wide
            # entry points take their pointer arguments in. At ws=2 this is the other rank.
            self._peers = [r for r in range(self.world_size) if r != self.rank]
            self._peer_scratch = [ext.ar_ipc_open(sc_handles[p]) for p in self._peers]
            self._peer_flags = [ext.ar_ipc_open(fl_handles[p]) for p in self._peers]
            # per-BLOCK device-resident seq counters (kernel increments -> replay-safe;
            # all ranks run identical graph sequences so seq_ctr[b] stays in lockstep)
            self._seq = torch.zeros(maxb, dtype=torch.int32, device=self.device)
        except Exception as e:
            _log(f"custom AR disabled: IPC setup failed ({e!r})")
            return

        self.disabled = False
        if not self.wide:
            _log(f"custom all-reduce INSTALLED (rank={self.rank} max={self.max_bytes // 1024}KB "
                 f"finegrained={fine_used} drain={self.drain} acq={self.acq} nt={self.nt} "
                 f"nb={self.min_nb}..{self.max_nb})")
        else:
            _log(f"custom all-reduce INSTALLED (rank={self.rank}/{self.world_size} "
                 f"slot={self.max_bytes // 1024}KB finegrained={fine_used} "
                 f"1shot<= {self.oneshot_max_elems} elems, 2shot<= {self.ts_max_elems})")

        # Quantized wires (RADIANCE_USE_R4D_AR_QUANT). Wide: the ti8 two-shot bound above
        # owns the whole 8..4096-token band; nothing more to allocate (the encoder writes
        # its own slot). ws=2: the rotated 6-bit one-shot for large messages, which needs
        # this rank's packed copy.
        self.ar_quant = os.environ.get("RADIANCE_USE_R4D_AR_QUANT", "1") == "1"
        self.quant_min_bytes = _QUANT_MIN_BYTES
        self.qnt = 1024        # threads/block for the compressed push (wire-bound; not sensitive)
        self.qmax_nb = 48      # block cap for the ws=2 compressed path
        self._qext = None
        self._qgroup = 0
        self._locpk = None
        if self.ar_quant and not self.wide:
            try:
                import r4d as qext
                self._qgroup = int(qext.AR_WHT6_GROUP)
                # numel is the one per-message term in the question: a group is the smallest
                # message this path will send, and _quant_ok() enforces the multiple per call.
                qname = qext.select("allreduce", world_size=self.world_size, exact=0,
                                    dtype="bf16", numel=self._qgroup)
                if qname is None:
                    raise RuntimeError("no lossy all-reduce kernel in this build")
                self._ar_wht6 = getattr(qext, qname)
                self._qext = qext
                # This rank's half, kept so the reduce folds exactly the bytes it sent. Same
                # scale_off split as the peer scratch. Held on the instance for a stable address
                # under cudagraph replay.
                loc_bytes = self.max_bytes // 2 + self.max_bytes // 32 + 4096
                self._locpk = torch.empty(loc_bytes, dtype=torch.uint8, device=self.device)
                _log(f"AR_QUANT ON (rotated {int(qext.AR_WHT6_BITS)}-bit packed payload; "
                     f"min={self.quant_min_bytes // 1024}KB group={self._qgroup} "
                     f"nt={self.qnt} max_nb={self.qmax_nb} local={loc_bytes >> 20}MB)")
            except Exception as e:
                _log(f"AR_QUANT disabled: {e!r}")
                self.ar_quant = False
        elif self.ar_quant:
            if self._ar_qts is not None:
                _log(f"AR_QUANT ON (tiered int8 two-shot wire, both hops; "
                     f"{self.oneshot_max_elems // 5120}..{self.ts_max_elems // 5120} tok)")
            else:
                _log("AR_QUANT: no ti8 kernel at this width; two-shot band stays exact")

    def _alloc(self, nbytes, fine):
        """Allocate a shared buffer, falling back fine->coarse if fine-grained IPC fails.
        Deterministic across ranks (same env/HW), so the fallback stays symmetric."""
        try:
            ptr, h = self._ext.ar_ipc_alloc(nbytes, fine)
            return ptr, h, fine
        except Exception as e:
            if fine:
                _log(f"fine-grained alloc failed ({e!r}); falling back to coarse")
                ptr, h = self._ext.ar_ipc_alloc(nbytes, False)
                return ptr, h, False
            raise

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if inp.dtype not in _DTYPE_CODE:
            return False
        nbytes = inp.numel() * inp.element_size()
        if nbytes == 0 or nbytes % 16 != 0:  # uint4 push alignment
            return False
        if not inp.is_contiguous():
            return False
        if not self.wide:
            return nbytes <= self.max_bytes
        # The wide bands: one-shot to the crossover, a two-shot to the ceiling, RCCL above.
        if inp.numel() <= self.oneshot_max_elems:
            return True
        return self._quant_ok(inp) or self._ts_ok(inp)

    def _nblocks(self, n16: int) -> int:
        # PCIe saturates with few blocks; extra blocks only help the reduce. Scale with
        # message size, clamped. Baked per-batch-size at cudagraph capture (n16 fixed).
        nb = n16 // self.words_per_block
        if nb < self.min_nb:
            nb = self.min_nb
        if nb > self.max_nb:
            nb = self.max_nb
        if nb > n16:
            nb = max(1, n16)
        return nb

    def _nblocks_ts(self, shard16: int) -> int:
        # The two-shots move shards, so their block count scales with the shard, not the
        # message: the full-message form over-launches the whole mid band.
        nb = shard16 // self.words_per_block
        if nb < self.min_nb:
            nb = self.min_nb
        if nb > self.ts_max_nb:
            nb = self.ts_max_nb
        return nb

    def _ts_ok(self, inp: torch.Tensor) -> bool:
        # exact two-shot band: above the crossover, inside the ceiling, sharding evenly,
        # and 2x the shard fits the slot (declines an fp32 message the 16-bit sizing
        # cannot hold)
        n = inp.numel()
        nbytes = n * inp.element_size()
        return (n > self.oneshot_max_elems and n <= self.ts_max_elems
                and (nbytes // 16) % self.world_size == 0
                and nbytes * 2 // self.world_size <= self.max_bytes)

    def _quant_ok(self, inp: torch.Tensor) -> bool:
        if not self.ar_quant:
            return False
        if inp.dtype not in (torch.bfloat16, torch.float16):
            return False
        n = inp.numel()
        if not self.wide:
            if self._qext is None:
                return False
            if n * inp.element_size() < self.quant_min_bytes:
                return False
            return n % self._qgroup == 0
        return (self._ar_qts is not None and n > self.oneshot_max_elems
                and n <= self.ts_max_elems and n % self._q_tile == 0)

    def _nblocks_q(self, nbytes: int) -> int:
        nb = nbytes // (self.words_per_block * 16)
        if nb < self.min_nb:
            nb = self.min_nb
        if nb > self.qmax_nb:
            nb = self.qmax_nb
        return nb

    def custom_all_reduce(self, inp: torch.Tensor):
        if not self.should_custom_ar(inp):
            return None
        out = torch.empty_like(inp)
        nbytes = inp.numel() * inp.element_size()
        stream = torch.cuda.current_stream().cuda_stream
        if not self.wide:
            if self._quant_ok(inp):
                self._ar_wht6(
                    self._peer_scratch[0], self._scratch, self._peer_flags[0], self._flags,
                    self._seq.data_ptr(), self._locpk.data_ptr(),
                    self.max_bytes, self.max_bytes // 2,
                    inp.data_ptr(), out.data_ptr(), inp.numel(),
                    _DTYPE_CODE[inp.dtype], stream, self._nblocks_q(nbytes), self.qnt,
                    self.drain, self.acq,
                )
            else:
                self._ar_exact(
                    self._peer_scratch[0], self._scratch, self._peer_flags[0], self._flags,
                    self._seq.data_ptr(), self.slot16,
                    inp.data_ptr(), out.data_ptr(), inp.numel(),
                    _DTYPE_CODE[inp.dtype], stream, self._nblocks(nbytes // 16), self.nt,
                    self.drain, self.acq,
                )
            return out
        # The wide ABI: every peer scratch pointer, every peer flag pointer (ascending rank,
        # matching self._peers), then rank in place of the 2-rank kernel's implicit peer, and
        # the pub knob.
        if inp.numel() <= self.oneshot_max_elems:
            self._ar_exact(
                *self._peer_scratch, *self._peer_flags, self._scratch, self._flags,
                self._seq.data_ptr(), self.slot16,
                inp.data_ptr(), out.data_ptr(), inp.numel(),
                _DTYPE_CODE[inp.dtype], self.rank, stream,
                self._nblocks(nbytes // 16), self.nt,
                self.drain, self.acq, self.pub,
            )
        elif self._quant_ok(inp):
            # ti8 stride is in BYTES (the wire is not 16B-aligned)
            self._ar_qts(
                *self._peer_scratch, *self._peer_flags, self._scratch, self._flags,
                self._seq.data_ptr(), self.max_bytes,
                inp.data_ptr(), out.data_ptr(), inp.numel(),
                _DTYPE_CODE[inp.dtype], self.rank, stream,
                self._nblocks_ts(nbytes // 16 // self.world_size), self.nt,
                self.drain, self.acq, self.pub,
            )
        else:
            self._ar_ts(
                *self._peer_scratch, *self._peer_flags, self._scratch, self._flags,
                self._seq.data_ptr(), self.slot16,
                inp.data_ptr(), out.data_ptr(), inp.numel(),
                _DTYPE_CODE[inp.dtype], self.rank, stream,
                self._nblocks_ts(nbytes // 16 // self.world_size), self.nt,
                self.drain, self.acq, self.pub,
            )
        return out

    @contextmanager
    def capture(self):
        # No graph-buffer registration needed (push model: only fixed IPC scratch/flags
        # are peer-shared; input/output stay local & are captured with stable addresses).
        yield


def install_custom_ar():
    """Wrap CudaCommunicator.all_reduce so small TP messages take the one-shot kernel and everything
    else falls through to RCCL. Wrapping rather than replacing ca_comm is required: vLLM's RocmAiter fusion
    pass asserts isinstance(ca_comm, CustomAllreduce), and ca_comm is already inert on ROCm. Env-gated
    by RADIANCE_USE_R4D_AR, idempotent."""
    # RADIANCE_USE_R4D is the master switch for the whole libr4d integration (patch_r4d.py):
    # with it off the TP group keeps RCCL, exactly as in an image built without the library.
    if os.environ.get("RADIANCE_USE_R4D", "1") != "1":
        _log("custom all-reduce not installed: RADIANCE_USE_R4D=0")
        return
    if os.environ.get("RADIANCE_USE_R4D_AR", "1") != "1":
        return
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

    if getattr(CudaCommunicator, "_radiance_ar_patched", False):
        return
    _orig_init = CudaCommunicator.__init__
    _orig_all_reduce = CudaCommunicator.all_reduce

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.radiance_comm = None
        try:
            # Only the TP group does custom AR (matches vLLM's own gating). No world-size test
            # here: which widths have a kernel is the library's answer.
            if "tp" in getattr(self, "unique_name", ""):
                comm = RadianceAllreduce(self.cpu_group, self.device)
                if not comm.disabled:
                    self.radiance_comm = comm
        except Exception as e:
            _log(f"custom AR attach failed: {e!r}")

    def _patched_all_reduce(self, input_):
        rc = getattr(self, "radiance_comm", None)
        if rc is not None and not rc.disabled and rc.should_custom_ar(input_):
            out = rc.custom_all_reduce(input_)
            if out is not None:
                return out
        return _orig_all_reduce(self, input_)

    CudaCommunicator.__init__ = _patched_init
    CudaCommunicator.all_reduce = _patched_all_reduce
    CudaCommunicator._radiance_ar_patched = True
    _log("fast-reduce hook armed (RADIANCE_USE_R4D_AR=1, all_reduce wrap)")
