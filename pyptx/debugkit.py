"""Kernel debugging utilities: beacon waits and phase cycle counters.

Warp-specialized kernels fail in two characteristic ways that are painful
to diagnose in environments without profiler access (containers commonly
block ``ncu`` and ``cuda-gdb``):

1. **Deadlocks** — some barrier wait never completes, the kernel hangs,
   and the host can never read anything back because the launch never
   finishes.
2. **Latency mysteries** — the pipeline runs but slower than it should,
   and you need per-phase cycle counts to see which wait is eating time.

``DebugKit`` packages the two patterns that proved reliable while
debugging the Blackwell flash-attention kernels:

- **Beacon waits**: ``kit.wait(mbar, phase, site=...)`` compiles (when
  enabled) to a bounded ``mbarrier.try_wait`` loop. On timeout the thread
  records a first-blame ``(site, extra)`` code into a dedicated debug
  buffer and *falls through* pretending the wait succeeded, so the whole
  pipeline drains, the kernel exits, and the host can decode exactly which
  wait deadlocked first. When disabled it emits a plain blocking wait with
  zero overhead.
- **Cycle counters**: ``t = kit.stamp(); ...; kit.accumulate("name", t)``
  accumulates ``%clock`` deltas per thread; ``kit.flush()`` writes them to
  the debug buffer for host-side readout.

The kit needs a dedicated debug output tensor — never point it at a
buffer the kernel also writes as real output; epilogue stores will
clobber the diagnostics (ask us how we know). Declare an extra
``Tile(n_slots, SLOT_WORDS, b32)`` output when building in debug mode and
attach it once at the top of the kernel body::

    kit = DebugKit(enabled=debug)
    ...
    def body(Q, K, V, O, DBG=None):
        ...
        slot = ...  # e.g. ctaid.x * block_threads + tid
        if kit.enabled:
            kit.attach(ptx.global_ptrs(DBG)[0], slot)

Host side, after the (now guaranteed to finish) run::

    for rec in debugkit.decode_beacons(dbg_tensor):
        print(rec)  # {"slot": ..., "site": "p_full", "extra": 1}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

BEACON_MAGIC = 0xBEAC
SLOT_WORDS = 8  # words reserved per slot: [beacon, counters 0..6]
DEFAULT_TIMEOUT_TRIES = 3_000_000  # ~50-200 ms of spinning before blame


@dataclass
class DebugKit:
    """Trace-time helper that instruments waits and phases in a kernel."""

    enabled: bool = False
    timeout_tries: int = DEFAULT_TIMEOUT_TRIES

    _sites: list[str] = field(default_factory=list)
    _counters: list[str] = field(default_factory=list)
    _counter_regs: dict = field(default_factory=dict)
    _ptr: Any = None            # u64 global pointer to the debug buffer
    _slot_off: Any = None       # u64 byte offset of this thread's slot
    _blamed: Any = None         # pred: this thread already recorded a beacon
    _label_n: int = 0

    # ---------------------------------------------------------------
    # wiring
    # ---------------------------------------------------------------

    def attach(self, dbg_ptr: Any, slot_reg: Any) -> None:
        """Bind the debug buffer pointer and this thread's slot index.

        ``slot_reg`` is a u32 register unique per participating thread
        (e.g. ``ctaid.x * threads_per_cta + tid``). Must be called before
        any ``wait``/``flush`` when the kit is enabled.
        """
        if not self.enabled:
            return
        from pyptx import ptx, reg
        from pyptx.types import pred as pred_t, u32, u64

        self._ptr = dbg_ptr
        off = reg.scalar(u64)
        ptx.inst.mul.wide.u32(off, slot_reg, SLOT_WORDS * 4)
        self._slot_off = off
        self._blamed = reg.scalar(pred_t)
        with_zero = reg.scalar(u32, init=0)
        ptx.inst.setp.ne.u32(self._blamed, with_zero, 0)

    # ---------------------------------------------------------------
    # beacon waits
    # ---------------------------------------------------------------

    def wait(self, mbar: Any, phase: Any, *, site: str, extra: int = 0) -> None:
        """mbarrier parity wait; in debug mode, bounded with first-blame
        recording and fall-through on timeout."""
        from pyptx import ptx, reg
        from pyptx.types import b32, pred as pred_t, u32

        if not self.enabled:
            ptx.mbarrier.wait(mbar, phase)
            return
        if self._ptr is None:
            raise RuntimeError("DebugKit.wait before attach()")
        if site not in self._sites:
            self._sites.append(site)
        site_id = self._sites.index(site) + 1

        self._label_n += 1
        lbl = f"dk_{site_id}_{extra}_{self._label_n}"
        tries = reg.scalar(u32, init=0)
        ptx.label(lbl)
        done = ptx.mbarrier.try_wait(mbar, phase)
        ptx.bra(lbl + "_ok", pred=done)
        ptx.inst.add.u32(tries, tries, 1)
        keep = reg.scalar(pred_t)
        ptx.inst.setp.lt.u32(keep, tries, self.timeout_tries)
        ptx.bra(lbl, pred=keep)
        # timed out: record first blame, then pretend success so the
        # pipeline drains and the kernel can exit
        ptx.bra(lbl + "_skip", pred=self._blamed)
        val = reg.scalar(b32, init=(BEACON_MAGIC << 16) | (site_id << 8) | (extra & 0xFF))
        ptx.inst.st.global_.b32(ptx.addr(self._ptr + self._slot_off), val)
        with ptx.scope():
            one = reg.scalar(u32, init=1)
            ptx.inst.setp.ne.u32(self._blamed, one, 0)
        ptx.label(lbl + "_skip")
        ptx.label(lbl + "_ok")

    # ---------------------------------------------------------------
    # cycle counters
    # ---------------------------------------------------------------

    def stamp(self) -> Any:
        """Return a u32 register holding %clock (None when disabled)."""
        if not self.enabled:
            return None
        from pyptx import ptx, reg
        from pyptx.types import u32

        t = reg.scalar(u32)
        ptx.inst.mov.u32(t, ptx.special.clock())
        return t

    def accumulate(self, name: str, t0: Any) -> None:
        """counter[name] += clock() - t0."""
        if not self.enabled:
            return
        from pyptx import ptx, reg
        from pyptx.types import u32

        if name not in self._counters:
            if len(self._counters) >= SLOT_WORDS - 1:
                raise ValueError(f"DebugKit supports {SLOT_WORDS - 1} counters")
            self._counters.append(name)
            acc = reg.scalar(u32, init=0)
            self._counter_regs[name] = acc
        acc = self._counter_regs[name]
        t1 = reg.scalar(u32)
        ptx.inst.mov.u32(t1, ptx.special.clock())
        ptx.inst.sub.u32(t1, t1, t0)
        ptx.inst.add.u32(acc, acc, t1)

    def flush(self) -> None:
        """Write all counters to this thread's slot (words 1..)."""
        if not self.enabled or not self._counters:
            return
        from pyptx import ptx

        for i, name in enumerate(self._counters):
            ptx.inst.st.global_.b32(
                ptx.addr(self._ptr + self._slot_off, 4 * (1 + i)),
                self._counter_regs[name],
            )

    # ---------------------------------------------------------------
    # host-side decode
    # ---------------------------------------------------------------

    def decode(self, dbg_tensor) -> dict:
        """Decode a debug tensor produced by this kit.

        Returns ``{"beacons": [...], "counters": {slot: {name: cycles}}}``.
        Beacons are aggregated by (site, extra) with slot lists.
        """
        import numpy as np

        w = np.ascontiguousarray(
            dbg_tensor.detach().cpu().view(-1).numpy()
        ).view(np.uint32).reshape(-1, SLOT_WORDS)
        beacons: dict = {}
        counters: dict = {}
        for slot in range(w.shape[0]):
            v = int(w[slot, 0])
            if (v >> 16) == BEACON_MAGIC:
                site_id = (v >> 8) & 0xFF
                extra = v & 0xFF
                name = (
                    self._sites[site_id - 1]
                    if 0 < site_id <= len(self._sites)
                    else f"site{site_id}"
                )
                beacons.setdefault((name, extra), []).append(slot)
            row = {}
            for i, cname in enumerate(self._counters):
                c = int(w[slot, 1 + i])
                if c:
                    row[cname] = c
            if row:
                counters[slot] = row
        return {
            "beacons": [
                {"site": k[0], "extra": k[1], "threads": len(v), "slots": v[:4]}
                for k, v in sorted(beacons.items())
            ],
            "counters": counters,
        }


def n_slot_words(n_slots: int) -> int:
    """Total b32 words to allocate for a debug buffer of n_slots."""
    return n_slots * SLOT_WORDS


# =====================================================================
# WaitMap: trace-time auto-instrumentation of barrier waits
# =====================================================================

WAITMAP_MAX_SITES = 16


@dataclass
class WaitMap:
    """Wraps every ``ptx.mbarrier.wait`` and ``ptx.bar.sync`` emitted while
    active, accumulating per-site %clock cycles into per-thread counters.
    Sites are auto-named from the caller's source line, so kernels need no
    source changes beyond attach/flush.

    Usage in a builder::

        wm = WaitMap()
        ...
        def body(...):
            if wm_enabled:
                wm.attach_flush_late(...)  # see attach()/flush()
            with wm.active():
                ...   # trace the kernel body; waits get instrumented

    Buffer layout: row per thread (tid), WAITMAP_MAX_SITES words per row.
    """

    _sites: list = field(default_factory=list)
    _accs: dict = field(default_factory=dict)
    _ptr: Any = None
    _row_off: Any = None

    def site_names(self) -> list:
        return list(self._sites)

    def attach(self, dbg_ptr: Any, tid_reg: Any) -> None:
        from pyptx import ptx, reg
        from pyptx.types import u32, u64

        self._ptr = dbg_ptr
        off = reg.scalar(u64)
        ptx.inst.mul.wide.u32(off, tid_reg, WAITMAP_MAX_SITES * 4)
        self._row_off = off
        # pre-create every accumulator here, in the uniform entry region:
        # a lazily created reg would get its zero-init emitted inside
        # whichever role branch first hit the site, leaving other roles'
        # threads accumulating into garbage
        for sid in range(WAITMAP_MAX_SITES):
            self._accs[sid] = reg.scalar(u32, init=0)

    def _site_id(self) -> int:
        import inspect

        for fr in inspect.stack()[2:8]:
            fn = fr.filename
            if "debugkit" in fn or "/ptx.py" in fn:
                continue
            name = f"{fn.rsplit('/', 1)[-1].removesuffix('.py')}:{fr.lineno}"
            break
        else:
            name = f"site{len(self._sites)}"
        if name not in self._sites:
            if len(self._sites) >= WAITMAP_MAX_SITES:
                return -1
            self._sites.append(name)
        return self._sites.index(name)

    def _wrap(self, emit_fn, args, kwargs) -> None:
        from pyptx import ptx, reg
        from pyptx.types import u32

        sid = self._site_id()
        if sid < 0 or sid not in self._accs:
            emit_fn(*args, **kwargs)
            return
        t0 = reg.scalar(u32)
        ptx.inst.mov.u32(t0, ptx.special.clock())
        emit_fn(*args, **kwargs)
        t1 = reg.scalar(u32)
        ptx.inst.mov.u32(t1, ptx.special.clock())
        ptx.inst.sub.u32(t1, t1, t0)
        ptx.inst.add.u32(self._accs[sid], self._accs[sid], t1)

    def active(self):
        """Context manager: instrument waits emitted inside."""
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            from pyptx import ptx

            orig_wait = ptx.mbarrier.wait
            orig_sync = ptx.bar.sync
            wm = self

            def wait_hook(*a, **k):
                wm._wrap(orig_wait, a, k)

            def sync_hook(*a, **k):
                wm._wrap(orig_sync, a, k)

            ptx.mbarrier.wait = wait_hook
            ptx.bar.sync = sync_hook
            try:
                yield self
            finally:
                ptx.mbarrier.wait = orig_wait
                ptx.bar.sync = orig_sync

        return _cm()

    def flush(self, pred: Any = None) -> None:
        """Write all site counters for this thread (guard with a pred that
        selects one CTA to keep the buffer small)."""
        from pyptx import ptx

        if self._ptr is None:
            raise RuntimeError("WaitMap.flush before attach()")
        for sid, acc in sorted(self._accs.items()):
            ptx.inst.st.global_.b32(
                ptx.addr(self._ptr + self._row_off, 4 * sid), acc, pred=pred
            )

    def decode(self, dbg_tensor, n_threads: int = 512, roles: dict | None = None) -> dict:
        """Aggregate per-site cycles by role. roles: {name: (lo_tid, hi_tid)}."""
        import numpy as np

        w = np.ascontiguousarray(
            dbg_tensor.detach().cpu().view(-1).numpy()
        ).view(np.uint32).reshape(-1, WAITMAP_MAX_SITES)[:n_threads]
        roles = roles or {"all": (0, n_threads)}
        out = {}
        for sid, name in enumerate(self._sites):
            row = {}
            for rname, (lo, hi) in roles.items():
                seg = w[lo:hi, sid]
                nz = seg[seg > 0]
                if len(nz):
                    row[rname] = int(nz.mean())
            if row:
                out[name] = row
        return out
