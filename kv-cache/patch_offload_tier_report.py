#!/usr/bin/env python3
"""Per-tier sizing verdict for the KV offload tiers (portable subset).

This is the vLLM-0.32 re-anchor of the KV offload tier-report, reduced to the
part of it that is NOT superseded by the native ``TieringMetricsTracker``
(vllm/v1/kv_offload/tiering/metrics.py), which both 0.30.0 and 0.32.0 ship.

WHAT THIS PATCH KEEPS (the genuinely-novel sizing verdict)
=========================================================
  P3 -- size (filesystem tier)
    tier_capacity_bytes{tier}            gauge: filesystem capacity
    tier_used_bytes{tier}                gauge: filesystem used bytes
    tier_occupancy_ratio{tier}           HISTOGRAM: occupancy over time

  P4/P6 -- the sizing verdict (CPU primary tier)
    tier_reads_before_evict{tier}        HISTOGRAM: read count before eviction
    tier_evictions_total{tier}           counter
    tier_evicted_bytes_total{tier}       counter
    tier_eviction_to_reuse_seconds{tier} HISTOGRAM: eviction -> next want
    tier_lookup_miss_evicted_total{tier} counter: would-have-hit misses

These are the "point of the exercise": the two histograms that distinguish
oversized (mass at 0 reads) from undersized (low reuse-delay mass) from
wrong-shape (both), which no occupancy gauge can do.

WHAT THIS PATCH DROPS (superseded by the native tracker in both 0.30.0 and 0.32.0)
=================================================================================
  P1 -- attribution (per-tier hit blocks/tokens)  -> native CHUNK_HITS/QUERIES
  P2 -- per-batch load/store bytes/secs/ops/latency -> native READ/WRITE_BYTES/TIME
  P5 -- per-tier stall seconds -> native LOOKUP_ASYNC_DELAY
The 0.30-era code these hung off (the ``secondary_lookup_start_time`` deferred-
lookup mechanism) does not exist in the released 0.30.0 tree either, so the
original patch was written against a transitional commit. The native tracker
covers all three.

RE-ANCHOR: 0.30.0 and 0.32.0 are near-identical here (both use "chunk" naming;
the "block" -> "chunk" rename predates the 0.30.0 release). The only real
deltas are the ``record_access`` guard in prepare_load (3e), the
``_cache_generation`` line in reset_cache (3g), and the import blocks (3a/3b).
Those use ``apply_any``; the rest are a single ``apply`` shared by both.

RUNS AFTER patch_kv_offload_instrumentation.py (it adds the fs tier's
``get_stats`` that the P3 size hunk extends).
"""

import sysconfig
from pathlib import Path

from _patchlib import apply, apply_any

SP = Path(sysconfig.get_paths()["purelib"])
KVO = SP / "vllm/v1/kv_offload"
KVO_BASE = KVO / "base.py"
CPU_MANAGER = KVO / "cpu/manager.py"
FS_MANAGER = KVO / "tiering/fs/manager.py"
CONN = SP / "vllm/distributed/kv_transfer/kv_connector/v1/offloading"
CONN_METRICS = CONN / "metrics.py"

print("radiance: KV offload tier-report (portable sizing verdict) for 0.30/0.32")

# ---------------------------------------------------------------------------
# 1. base.py: the name table. The anchor (end of OffloadingMetricMetadata) is
#    identical in 0.30.0 and 0.32.0, so a single apply covers both.
# ---------------------------------------------------------------------------
apply(
    KVO_BASE,
    """@dataclass(frozen=True)
class OffloadingMetricMetadata:
    documentation: str
    labelnames: tuple[str, ...] = ()""",
    """@dataclass(frozen=True)
class OffloadingMetricMetadata:
    documentation: str
    labelnames: tuple[str, ...] = ()


class TierReportMetrics:
    \"\"\"radiance tier-report: the sizing verdict, per tier.

    Every name carries a single ``tier`` label: "cpu" for the primary tier,
    the configured tier_type ("fs", ...) for each secondary tier. P1/P2/P5
    (attribution, per-batch transfer, stall) are delegated to vLLM's native
    TieringMetricsTracker; only the size (P3) and the sizing verdict (P4/P6)
    live here.
    \"\"\"

    TIER = ("tier",)  # a tuple: labelnames is tuple[str, ...]; the bare string
    # "tier" has len 4 and the connector's label-count assert rejects the 1-label
    # emission ("expects 4 labels, got 1").

    # P3: size. Occupancy is a distribution over time, not a level.
    CAPACITY_BYTES = "vllm:kv_offload_tier_capacity_bytes"
    USED_BYTES = "vllm:kv_offload_tier_used_bytes"
    OCCUPANCY_RATIO = "vllm:kv_offload_tier_occupancy_ratio"

    # P4/P6: the sizing verdict.
    READS_BEFORE_EVICT = "vllm:kv_offload_tier_reads_before_evict"
    EVICTIONS = "vllm:kv_offload_tier_evictions"
    EVICTED_BYTES = "vllm:kv_offload_tier_evicted_bytes"
    EVICTION_TO_REUSE = "vllm:kv_offload_tier_eviction_to_reuse_seconds"
    MISS_EVICTED = "vllm:kv_offload_tier_lookup_miss_evicted"

    # One second to one day: how long after eviction the block was wanted again.
    REUSE_BUCKETS = (1, 5, 15, 60, 300, 900, 3600, 14400, 86400)
    # Integer read counts; edges at .5 so a block read exactly N times lands in
    # the bucket labelled N and "never read back" is exactly the le="0.5" bin.
    READS_BUCKETS = (0.5, 1.5, 2.5, 4.5, 8.5, 16.5, 32.5, 64.5)
    OCCUPANCY_BUCKETS = (0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.999, 1.0)""",
    "class TierReportMetrics:",
    "1  kv_offload/base.py: TierReportMetrics name table",
)

# ---------------------------------------------------------------------------
# 2. Register the definitions at the CONNECTOR level, so they exist for every
#    offloading spec. OffloadPromMetrics.observe() asserts that every emitted
#    name is registered; registering here means that assertion cannot be tripped.
# ---------------------------------------------------------------------------
# The eight sizing-verdict definitions, shared by both variants below (the 0.30.0 and
# 0.32.0 definitions-dict tails differ only in the entry the block is appended after).
_TIER_REPORT_DEFS = """        # radiance tier-report: the sizing verdict. Everything carries a `tier`
        # label so one scrape describes every tier separately.
        TierReportMetrics.CAPACITY_BYTES: OffloadingGaugeMetadata(
            documentation=(
                "Configured capacity of this tier, in bytes. For a filesystem "
                "tier this is the size of the filesystem holding the block "
                "store, since that is what actually bounds it."
            ),
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.USED_BYTES: OffloadingGaugeMetadata(
            documentation=(
                "Bytes currently held by this tier. For a filesystem tier this "
                "is the filesystem's used bytes, which may include data that is "
                "not KV blocks -- it is the number that decides whether the "
                "tier runs out of room."
            ),
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.OCCUPANCY_RATIO: OffloadingHistogramMetadata(
            documentation=(
                "Histogram of this tier's occupancy, sampled once per metrics "
                "interval. The DISTRIBUTION is the sizing signal."
            ),
            buckets=TierReportMetrics.OCCUPANCY_BUCKETS,
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.READS_BEFORE_EVICT: OffloadingHistogramMetadata(
            documentation=(
                "Histogram of how many times a block was read back before this "
                "tier evicted it. Mass in the le=0.5 bucket means blocks are "
                "being stored that nobody ever reads: the tier is oversized."
            ),
            buckets=TierReportMetrics.READS_BUCKETS,
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.EVICTIONS: OffloadingCounterMetadata(
            documentation="Blocks evicted from this tier.",
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.EVICTED_BYTES: OffloadingCounterMetadata(
            documentation="Bytes evicted from this tier.",
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.EVICTION_TO_REUSE: OffloadingHistogramMetadata(
            documentation=(
                "Histogram of the time between this tier evicting a block and a "
                "later lookup asking for that same block, in seconds. Mass at "
                "the low end is cache thrashing measured rather than inferred: "
                "the tier is undersized."
            ),
            buckets=TierReportMetrics.REUSE_BUCKETS,
            labelnames=TierReportMetrics.TIER,
        ),
        TierReportMetrics.MISS_EVICTED: OffloadingCounterMetadata(
            documentation=(
                "Lookups that missed on a block this tier previously held and "
                "evicted. Distinguishes 'never cached' from 'cached and thrown "
                "away too early'; only the second justifies buying capacity."
            ),
            labelnames=TierReportMetrics.TIER,
        ),"""

_TAIL_030 = """        _ConnectorMetricName.LOOKUP_SERVED_TOKENS: OffloadingCounterMetadata(
            documentation="Tokens returned by successful offload lookups.",
        ),"""

_TAIL_032 = """        _ConnectorMetricName.ALLOCATION_FAILURE: OffloadingCounterMetadata(
            documentation=(
                "Number of KV offload store allocation attempts that failed."
            ),
        ),"""

apply_any(
    CONN_METRICS,
    [
        # 0.30.0: the definitions dict tail is LOOKUP_SERVED_TOKENS.
        (
            _TAIL_030 + "\n    }",
            _TAIL_030 + "\n" + _TIER_REPORT_DEFS + "\n    }",
        ),
        # 0.32.0: the definitions dict tail is ALLOCATION_FAILURE.
        (
            _TAIL_032 + "\n    }",
            _TAIL_032 + "\n" + _TIER_REPORT_DEFS + "\n    }",
        ),
    ],
    "TierReportMetrics.CAPACITY_BYTES: OffloadingGaugeMetadata(",
    "2b offloading/metrics.py: tier-report metric definitions",
)

# The import of TierReportMetrics: the 0.30.0 import block ends with
# RequestOffloadingContext, the 0.32.0 one with get_offload_group_idx.
apply_any(
    CONN_METRICS,
    [
        (
            """from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
)""",
            """from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingMetricMetadata,
    TierReportMetrics,
)""",
        ),
    ],
    "    TierReportMetrics,\n)",
    "2a offloading/metrics.py: import TierReportMetrics",
)

# ---------------------------------------------------------------------------
# 3. The CPU primary tier: read counts, evictions, the ghost list, occupancy.
#    This is where P4 and P6 live -- the sizing verdict.
# ---------------------------------------------------------------------------
apply_any(
    CPU_MANAGER,
    [
        # 0.30.0: no `import time` at the top.
        (
            """from collections import OrderedDict
from collections.abc import Collection, Iterable""",
            """import time
from collections import OrderedDict
from collections.abc import Collection, Iterable""",
        ),
        # 0.32.0: `import time` already present (no-op).
        (
            """import time
from collections.abc import Collection, Iterable""",
            """import time
from collections.abc import Collection, Iterable""",
        ),
    ],
    "import time\nfrom collections",
    "3a cpu/manager.py: import time",
)

apply_any(
    CPU_MANAGER,
    [
        # 0.30.0: import block ends with RequestOffloadingContext.
        (
            """    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.common import (""",
            """    RequestOffloadingContext,
    TierReportMetrics,
)
from vllm.v1.kv_offload.cpu.common import (""",
        ),
        # 0.32.0: import block ends with get_offload_group_idx.
        (
            """    RequestOffloadingContext,
    get_offload_group_idx,
)""",
            """    RequestOffloadingContext,
    TierReportMetrics,
    get_offload_group_idx,
)""",
        ),
    ],
    "    TierReportMetrics,",
    "3b cpu/manager.py: import TierReportMetrics",
)

# Identical in 0.30.0 and 0.32.0 (both use "chunk" naming).
apply(
    CPU_MANAGER,
    """        # Number of chunk references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )""",
    """        # Number of chunk references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

        # radiance tier-report: state for the sizing verdict. ChunkStatus is a
        # ctypes.Structure with fixed fields, so the per-chunk read count cannot
        # live on the chunk and needs a side map. Both maps are FIFO-bounded.
        self._tr_tier_name: str = "cpu"
        self._tr_block_bytes: int = 0
        self._tr_cap: int = max(1024, num_chunks * 4)
        self._tr_reads: OrderedDict[OffloadKey, int] = OrderedDict()
        self._tr_ghost: OrderedDict[OffloadKey, float] = OrderedDict()
        self._tr_reads_before_evict: list[int] = []
        self._tr_reuse_delays: list[float] = []
        self._tr_evictions: int = 0
        self._tr_miss_evicted: int = 0""",
    "self._tr_reads_before_evict",
    "3c cpu/manager.py: tier-report state",
)

# Identical in both (the `chunk` lookup path).
apply(
    CPU_MANAGER,
    """        chunk = self._policy.get(key)
        if chunk is None:
            return LookupResult.MISS
        if not chunk.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT""",
    """        chunk = self._policy.get(key)
        if chunk is None:
            # radiance tier-report (P4/P6): was this chunk ever here? A miss on
            # a chunk we evicted is the only kind of miss that argues for buying
            # capacity; a miss on a chunk we never had does not.
            evicted_at = self._tr_ghost.pop(key, None)
            if evicted_at is not None:
                self._tr_miss_evicted += 1
                self._tr_reuse_delays.append(time.monotonic() - evicted_at)
            return LookupResult.MISS
        if not chunk.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT""",
    "radiance tier-report (P4/P6): was this",
    "3d cpu/manager.py: would-have-hit on lookup miss",
)

# 0.30.0 has no `record_access` guard; 0.32.0 does.
apply_any(
    CPU_MANAGER,
    [
        # 0.30.0: append then straight to the return.
        (
            """            chunk.ref_cnt += 1
            chunks.append(chunk)
        return self._get_load_store_spec(keys, chunks)""",
            """            chunk.ref_cnt += 1
            chunks.append(chunk)
            # radiance tier-report (P4): a prepare_load IS the read-back.
            self._tr_reads[key] = self._tr_reads.get(key, 0) + 1
            self._tr_reads.move_to_end(key)
            if len(self._tr_reads) > self._tr_cap:
                self._tr_reads.popitem(last=False)
        return self._get_load_store_spec(keys, chunks)""",
        ),
        # 0.32.0: a `record_access` guard sits between append and the return.
        (
            """            chunk.ref_cnt += 1
            chunks.append(chunk)
        if record_access:
            self._record_request_cache_access(keys, req_context, reused_keys=keys)
        return self._get_load_store_spec(keys, chunks)""",
            """            chunk.ref_cnt += 1
            chunks.append(chunk)
            # radiance tier-report (P4): a prepare_load IS the read-back.
            self._tr_reads[key] = self._tr_reads.get(key, 0) + 1
            self._tr_reads.move_to_end(key)
            if len(self._tr_reads) > self._tr_cap:
                self._tr_reads.popitem(last=False)
        if record_access:
            self._record_request_cache_access(keys, req_context, reused_keys=keys)
        return self._get_load_store_spec(keys, chunks)""",
        ),
    ],
    "radiance tier-report (P4): a prepare_load IS the read-back.",
    "3e cpu/manager.py: count read-backs",
)

# Identical in both (the eviction loop).
apply(
    CPU_MANAGER,
    """            for key, chunk in evicted:
                self._free_chunk(chunk)
                to_evict.append(key)""",
    """            _tr_now = time.monotonic()  # radiance tier-report
            for key, chunk in evicted:
                self._free_chunk(chunk)
                to_evict.append(key)
                # radiance tier-report (P4): how many times was it read back
                # before it died, and when did it die?
                self._tr_reads_before_evict.append(self._tr_reads.pop(key, 0))
                self._tr_evictions += 1
                self._tr_ghost[key] = _tr_now
                if len(self._tr_ghost) > self._tr_cap:
                    self._tr_ghost.popitem(last=False)""",
    "radiance tier-report (P4): how many times was it read back",
    "3f cpu/manager.py: eviction accounting",
)

# 0.30.0 reset_cache has no `_cache_generation += 1`; 0.32.0 does.
apply_any(
    CPU_MANAGER,
    [
        # 0.30.0: no cache_generation line.
        (
            """        self._policy.clear()
        self._num_evictable_cache_chunks = 0
        self._num_write_pending_chunks = 0""",
            """        self._policy.clear()
        self._num_evictable_cache_chunks = 0
        self._num_write_pending_chunks = 0

        # radiance tier-report: a reset is not an eviction. Dropping the ghost
        # list here stops a post-reset miss being reported as a would-have-hit.
        self._tr_reads.clear()
        self._tr_ghost.clear()""",
        ),
        # 0.32.0: a `_cache_generation += 1` sits between the clear and the counters.
        (
            """        self._policy.clear()
        self._cache_generation += 1
        self._num_evictable_cache_chunks = 0
        self._num_write_pending_chunks = 0""",
            """        self._policy.clear()
        self._cache_generation += 1
        self._num_evictable_cache_chunks = 0
        self._num_write_pending_chunks = 0

        # radiance tier-report: a reset is not an eviction. Dropping the ghost
        # list here stops a post-reset miss being reported as a would-have-hit.
        self._tr_reads.clear()
        self._tr_ghost.clear()""",
        ),
    ],
    "radiance tier-report: a reset is not an eviction.",
    "3g cpu/manager.py: clear tier-report state on reset",
)

# Identical in both (the get_stats tail; `_num_chunks`/`_num_allocated_chunks`
# naming is the same in 0.30.0 and 0.32.0).
apply(
    CPU_MANAGER,
    """        if self.store_threshold >= 2:
            stats.increase_counter(
                CPUOffloadingMetrics.STORES_SKIPPED,
                self.stores_skipped_in_current_batch,
            )
            self.stores_skipped_in_current_batch = 0

        return stats""",
    """        if self.store_threshold >= 2:
            stats.increase_counter(
                CPUOffloadingMetrics.STORES_SKIPPED,
                self.stores_skipped_in_current_batch,
            )
            self.stores_skipped_in_current_batch = 0

        self._tr_emit_tier_report(stats)

        return stats

    # ---- radiance tier-report -------------------------------------------
    def _tr_emit_tier_report(self, stats: OffloadingConnectorStats) -> None:
        \"\"\"Flush this tier's cumulative sizing counters onto the stats payload.\"\"\"
        if getattr(self, "_tr_reads", None) is None:
            return
        label = (self._tr_tier_name,)
        block_bytes = self._tr_block_bytes

        if self._num_chunks > 0:
            resident = self._num_allocated_chunks - len(self._free_list)
            stats.observe_histogram(
                TierReportMetrics.OCCUPANCY_RATIO,
                resident / self._num_chunks,
                label,
            )
            if block_bytes:
                stats.set_gauge(
                    TierReportMetrics.CAPACITY_BYTES,
                    self._num_chunks * block_bytes,
                    label,
                )
                stats.set_gauge(
                    TierReportMetrics.USED_BYTES, resident * block_bytes, label
                )

        for num_reads in self._tr_reads_before_evict:
            stats.observe_histogram(
                TierReportMetrics.READS_BEFORE_EVICT, num_reads, label
            )
        self._tr_reads_before_evict.clear()

        for delay in self._tr_reuse_delays:
            stats.observe_histogram(
                TierReportMetrics.EVICTION_TO_REUSE, delay, label
            )
        self._tr_reuse_delays.clear()

        if self._tr_evictions:
            stats.increase_counter(
                TierReportMetrics.EVICTIONS, self._tr_evictions, label
            )
            if block_bytes:
                stats.increase_counter(
                    TierReportMetrics.EVICTED_BYTES,
                    self._tr_evictions * block_bytes,
                    label,
                )
            self._tr_evictions = 0

        if self._tr_miss_evicted:
            stats.increase_counter(
                TierReportMetrics.MISS_EVICTED, self._tr_miss_evicted, label
            )
            self._tr_miss_evicted = 0""",
    "def _tr_emit_tier_report",
    "3h cpu/manager.py: emit tier-report counters",
)

# ---------------------------------------------------------------------------
# 8. The filesystem tier: report its size (P3). The fs tier holds no capacity
#    number of its own -- it writes files until the filesystem says no -- so
#    the filesystem IS the capacity, and its used bytes are what decide whether
#    the tier runs out of room.
# ---------------------------------------------------------------------------
apply_any(
    FS_MANAGER,
    [
        # 0.30.0: no `import os` alias issues; the anchor is the same.
        (
            """        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)""",
            """        self._lookup_manager = FsAsyncLookupManager(tier=self, tier_type=self.tier_type)

        # radiance tier-report (P3): size of the filesystem holding the store.
        # statvfs is a syscall against the block device's filesystem; sampling
        # it on every metrics interval would put it on the hot path for a
        # number that moves at the speed of a disk filling up.
        try:
            self._tr_root_dir = os.path.dirname(
                self.file_mapper.get_config_file_path()
            )
        except Exception:
            self._tr_root_dir = None
        self._tr_statvfs_every = 256
        self._tr_statvfs_countdown = 0
        self._tr_capacity_bytes = 0.0
        self._tr_used_bytes = 0.0""",
        ),
    ],
    "self._tr_statvfs_every",
    "8a fs/manager.py: tier-report size state",
)

# The fs tier's get_stats is added by patch_kv_offload_instrumentation.py (the
# cascade-backlog gauge) and is identical in 0.30.0 and 0.32.0, so a single
# apply extends it to also report the P3 size via a new _tr_emit_tier_report().
# The inflight gate is relaxed to "always build a stats payload" so the size is
# reported even when the pool exposes no inflight count.
apply(
    FS_MANAGER,
    """    def get_stats(self):
        \"\"\"radiance: emit the cascade backlog. Guarded -- a rename must not crash metrics.\"\"\"
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
            OffloadingConnectorStats,
        )

        pool = getattr(self, "_pool", None)
        inflight = getattr(pool, "_inflight_jobs", None) if pool is not None else None
        if not isinstance(inflight, int):
            return None
        stats = OffloadingConnectorStats()
        stats.set_gauge(self.FS_INFLIGHT_JOBS, float(inflight))
        return stats""",
    """    def get_stats(self):
        \"\"\"radiance: emit the cascade backlog + tier size (P3).\"\"\"
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
            OffloadingConnectorStats,
        )

        pool = getattr(self, "_pool", None)
        inflight = getattr(pool, "_inflight_jobs", None) if pool is not None else None
        stats = OffloadingConnectorStats()
        if isinstance(inflight, int):
            stats.set_gauge(self.FS_INFLIGHT_JOBS, float(inflight))
        self._tr_emit_tier_report(stats)
        return stats

    def _tr_emit_tier_report(self, stats) -> None:
        \"\"\"radiance tier-report (P3): drain the filesystem size counters.\"\"\"
        from vllm.v1.kv_offload.base import TierReportMetrics

        root_dir = getattr(self, "_tr_root_dir", None)
        if root_dir is None:
            return
        self._tr_statvfs_countdown -= 1
        if self._tr_statvfs_countdown <= 0:
            self._tr_statvfs_countdown = self._tr_statvfs_every
            try:
                st = os.statvfs(root_dir)
                self._tr_capacity_bytes = float(st.f_blocks) * float(st.f_frsize)
                self._tr_used_bytes = (
                    float(st.f_blocks - st.f_bfree) * float(st.f_frsize)
                )
            except OSError:
                pass
        if self._tr_capacity_bytes > 0.0:
            label = (self.tier_type,)
            stats.set_gauge(
                TierReportMetrics.CAPACITY_BYTES, self._tr_capacity_bytes, label
            )
            stats.set_gauge(TierReportMetrics.USED_BYTES, self._tr_used_bytes, label)
            stats.observe_histogram(
                TierReportMetrics.OCCUPANCY_RATIO,
                self._tr_used_bytes / self._tr_capacity_bytes,
                label,
            )""",
    "def _tr_emit_tier_report",
    "8b fs/manager.py: emit the tier size",
)

print("[radiance] tier-report (portable sizing verdict) applied -- P3 + P4/P6")
