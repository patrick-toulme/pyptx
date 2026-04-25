# Mbarriers and Async Synchronization

Every non-trivial pyptx kernel uses mbarriers. The Hopper GEMM has one
per K-slice. The Blackwell GEMM has a 4-stage ring of `load` +
`consumed` barriers plus one `mma` barrier. The grouped GEMM does a
two-phase parity dance inside a runtime loop.

If you don't have a mental model for mbarriers, these kernels read as
noise. This page gives you that model.

## The One-Sentence Mental Model

> An **mbarrier is a completion object living in shared memory** that
> tracks pending async work. You `arrive` to tell it "I'm done with my
> share," and you `wait` to block until everyone who was supposed to
> arrive has arrived.

That's it. Everything else — phases, `expect_tx`, `try_wait.parity`,
cluster scope — is a refinement of that one idea to handle the
specifics of TMA, WGMMA, and tcgen05.

## Step 1: The Pattern

Every mbarrier use has four calls:

```python
# Setup (once per barrier, usually thread 0)
bar = smem.mbarrier(count=1)
with ptx.if_(tid == 0):
    ptx.mbarrier.init(bar[0], 1)
    ptx.fence.proxy_async_shared_cta()

# Hot path (per iteration)
with ptx.if_(tid == 0):
    ptx.mbarrier.arrive_expect_tx(bar[0], EXPECTED_BYTES)
    ptx.cp.async_.bulk.tensor_2d(..., mbar=bar[0])
ptx.bar.sync(0)
ptx.mbarrier.wait(bar[0], phase)
phase ^= 1
```

- **`init(bar, count)`**: allocate the barrier, set the expected
  arrival count. For a single-producer TMA pattern, count=1 (just the
  TMA engine).
- **`arrive_expect_tx(bar, bytes)`**: announce to the barrier that we
  expect `bytes` of async traffic. Replaces the simple `arrive` when
  the work is an async copy — the barrier doesn't become ready after
  an immediate arrive, but after the async bytes actually land.
- **`cp.async.bulk.tensor_2d(..., mbar=bar)`**: the TMA issue itself.
  Writes data into SMEM; when done, signals the barrier.
- **`mbarrier.wait(bar, phase)`**: blocking wait. Returns once the
  barrier has been "flipped" this phase.

**Why `phase ^= 1`?** The barrier alternates between phase 0 and
phase 1 on each completion. `wait` checks the current expected phase
bit and blocks until it matches the barrier's. Toggling `phase` on
every iteration is how you reuse one barrier across many iterations
without re-initializing it. If you don't toggle, the second `wait`
returns immediately because the barrier is still on phase 1 from
last time.

## Step 2: Why `arrive_expect_tx` Exists

A normal `arrive` says "this participant is done." It's immediate.

But TMA doesn't finish immediately — it kicks off a DRAM → SMEM copy
that may take hundreds of cycles. If you `arrive` after issuing the
TMA, the barrier would release while the bytes were still in flight,
and the consumer would read stale SMEM.

`arrive_expect_tx` solves this. It tells the barrier: "expect N bytes
of async traffic; mark me arrived only after those bytes actually
land in SMEM." The TMA engine reports its completion against the same
barrier, and the wait is correct.

Calculating `EXPECTED_BYTES`:

```python
# Single tile load:
BM * BK * sizeof(dtype)

# Two tiles sharing one barrier (A + B):
BM * BK * sizeof(dtype) + BN * BK * sizeof(dtype)
```

Pay attention when multiple TMA loads share one barrier — the
`expect_tx` count is the **sum** of both loads. Under-counting
deadlocks the consumer forever; over-counting never fires.

## Step 3: Phase Toggling In Detail

A two-phase mbarrier maintains an internal phase bit. Each completion
flips it:

```
Iteration 0:  bar.phase = 0 → TMA completes → bar.phase = 1
              wait(phase=0) returns once bar.phase differs from 0

Iteration 1:  bar.phase = 1 → TMA completes → bar.phase = 0
              wait(phase=1) returns once bar.phase differs from 1

...
```

Your code maintains its own `phase` register and toggles it each
iteration:

```python
phase = reg.scalar(b32, init=0)
with ptx.loop("k_loop", pred=keep_going):
    # ... issue TMA, arrive_expect_tx ...
    ptx.bar.sync(0)                      # CTA lockstep
    ptx.mbarrier.wait(bar[0], phase)     # wait this iter's completion
    phase ^= 1                           # toggle for next iter
    # ... consume the data ...
```

`ptx.bar.sync(0)` is the **CTA-wide sync** — all threads reach the
same point before anyone waits. Without it, some threads could rush
ahead and wait before the TMA even issued.

## Step 4: The `try_wait.parity` Spin-Loop (Blackwell)

The Blackwell GEMM uses a different wait shape — not `mbarrier.wait`,
but a manual spin loop around `try_wait.parity`:

```python
with ptx.scope():
    ready = reg.scalar(pred)
    ptx.label(f"lwait_{ki}")
    ptx.inst.mbarrier.try_wait.parity.shared__cta.b64(
        ready, ptx.addr(mbar_l), load_phase
    )
    ptx.bra(f"ldone_{ki}", pred=ready)
    ptx.bra(f"lwait_{ki}")
    ptx.label(f"ldone_{ki}")
```

Three differences from `mbarrier.wait`:

- **`try_wait.parity`** is non-blocking. It checks the barrier's
  phase bit once and returns a predicate — "ready" or "not yet."
- **The spin loop is explicit** — the kernel branches back to the
  label until the barrier flips.
- **Phase is passed explicitly** (`load_phase = (ki // stages) & 1`),
  not toggled register-side.

Why use this form? Two reasons:

1. **Multi-stage ring buffers**: with N stages, the producer runs
   ahead of the consumer by up to N iterations. Each iteration's
   expected phase is `(ki // stages) & 1` — a function of how many
   full cycles through the ring you've done. Simpler to compute per
   iteration than to maintain N separate `phase` registers.
2. **Bit-exact match with Pallas reference**: Blackwell programs that
   use `mbarrier.wait` sometimes get different PTX from programs that
   use `try_wait.parity + spin`. Being able to emit either is useful
   when matching another codebase's output.

The `with ptx.scope():` block keeps the `ready` register and labels
local, so you can expand this pattern inline multiple times per
iteration without label collisions.

## Step 5: Producer/Consumer Warp Specialization

Blackwell GEMM (and some Hopper kernels) splits the CTA into
**specialized warps**:

- **TMA warp**: the only warp that issues TMA loads. Uses one barrier
  per slot (`bar_load[slot]`) to signal completion.
- **MMA warp**: the only warp that issues `tcgen05.mma`. Waits on
  `bar_load` before consuming a slot, signals `bar_consumed` after.
- **All warps**: wait on `bar_mma` at the end of the kernel for the
  MMA commit.

The ring buffer gives you producer/consumer decoupling:

```
Time →
TMA warp:  load[0] load[1] load[2] load[3]  (stall on consumed[0])  load[4] ...
MMA warp:  (wait)  (wait)  (wait)  mma[0]   mma[1]  mma[2]  mma[3]  mma[4] ...
                                   ↑signal  ↑signal  ↑signal
                                   consumed[0..3]
```

With 4 stages, the TMA warp can stay 4 K-tiles ahead of MMA, hiding
HBM latency completely. Once the ring fills, the producer waits for
the consumer to free a slot (`bar_consumed`) before issuing the next
TMA.

The code pattern is:

```python
# Allocate STAGES barriers for loads, STAGES for consumed, 1 for MMA.
bar_load = base + SMEM_BAR_LOAD        # STAGES barriers, one per slot
bar_consumed = base + SMEM_BAR_CONSUMED
bar_mma = base + SMEM_BAR_MMA

with ptx.if_(is_tma_warp):
    for ki in range(k_iters):
        slot = ki % STAGES
        if ki >= STAGES:
            # Wait for MMA to free this slot before refilling.
            wait(bar_consumed[slot], phase=((ki // STAGES) - 1) & 1)
        arrive_expect_tx(bar_load[slot], A_BYTES + B_BYTES)
        tma_load(..., mbar=bar_load[slot])

with ptx.if_(is_mma_warp):
    for ki in range(k_iters):
        slot = ki % STAGES
        wait(bar_load[slot], phase=(ki // STAGES) & 1)
        issue_mma(...)
        arrive(bar_consumed[slot])

    # After all K-tiles, commit the MMA work against bar_mma.
    ptx.tcgen05.commit(bar_mma, space="cluster")

# Everybody waits for the MMA commit before epilogue.
all_threads_wait(bar_mma, phase=0)
```

## Step 6: `ptx.bar.sync(0)` vs mbarriers

Two different sync primitives, often confused:

- **`ptx.bar.sync(0)`** — a **CTA-wide hardware barrier**. All 128
  (or however many) threads in the CTA wait here before anyone
  proceeds. Cheap, uses dedicated hardware, no state in SMEM. Use
  for lockstep sync where every thread should reach the same point.
- **`ptx.mbarrier.wait(bar, phase)`** — an **async completion wait**.
  Blocks until the barrier's phase flips. State lives in SMEM. Use
  for "wait until this async thing finishes" — TMA loads, MMA
  commits, producer→consumer handoff.

Typical pattern in one iteration:

```python
with ptx.if_(tid == 0):
    arrive_expect_tx + issue_tma(...)
ptx.bar.sync(0)                          # hardware barrier: lockstep
ptx.mbarrier.wait(bar[0], phase)         # async wait: TMA done
# ...consume data from SMEM...
```

The hardware barrier keeps all threads together; the mbarrier waits
for the actual bytes. Both are needed — remove `bar.sync(0)` and
some threads might wait on a barrier that hasn't even been asked to
arrive yet.

## Step 7: `ptx.fence.proxy_async_shared_cta`

After `mbarrier.init`, you'll see:

```python
with ptx.if_(tid == 0):
    ptx.mbarrier.init(bar[0], 1)
    ptx.fence.proxy_async_shared_cta()
```

The fence is required so that the async proxy (TMA engine) sees the
initialized barrier. Without it, the TMA engine might read the barrier
in its uninitialized state and never complete.

You only need this fence **once after init**. Don't sprinkle it into
the hot path.

## Step 8: Cluster-Scoped Barriers

On Blackwell with `cta_group::2` MMA, one MMA issue covers two CTAs.
The `bar_mma` barrier is **cluster-scoped** — both CTAs can wait on
the same barrier:

```python
ptx.tcgen05.commit(bar_mma, space="cluster")
```

The `space="cluster"` attribute tells the hardware that this barrier
crosses CTA boundaries. The barrier itself still lives in SMEM (one
of the two CTAs' SMEM), but cluster-scoped primitives let both CTAs
signal and wait against it.

Cluster barriers require the cluster-launch decorator config and
only work on Blackwell+. For most Hopper kernels you don't need them.

## Checklist For New Code

Before shipping a kernel that uses mbarriers:

1. **Each barrier initialized exactly once, by one thread.** Inside
   `with ptx.if_(tid == 0):` block. Followed by
   `ptx.fence.proxy_async_shared_cta()`.
2. **Every `arrive_expect_tx` has the right byte count.** Sum of all
   TMA loads against that barrier, in bytes.
3. **Every `wait` is followed by `phase ^= 1`** if the barrier will
   be reused next iteration. Or pass an explicit
   `phase = (ki // STAGES) & 1` for ring buffers.
4. **`ptx.bar.sync(0)` before every `mbarrier.wait`** to keep the CTA
   in lockstep. Most bugs without this are "some threads wait on a
   barrier that's already flipped twice ago."
5. **Producer/consumer pairs have both a `load` and `consumed` barrier
   per slot.** Not sharing, not one barrier used for both roles.
6. **Match the `cta_group::2` scope.** If you're using cluster-scoped
   MMA, the commit barrier needs `space="cluster"`.

## Why This Works

Mbarriers are a hardware implementation of a CSP-style channel —
each stage is a send/receive handshake between the TMA engine (or an
MMA issuer, or a cluster peer) and the consumer thread. The phase bit
is just the channel's "last signaled" flag, and the `expect_tx` count
is the payload size the channel agrees to transfer.

Once you see it as channels, the ring-buffer pattern falls out
naturally: each slot is a channel, the producer sends one message per
K-tile, the consumer receives in order, and the ring lets the
producer run ahead. The parity wait is just the receiver checking
"is there a new message here yet?"

## What To Read Next

- [Tiles, Layouts, and TMA](tiles-layouts-tma.md) — the companion
  page that explains what `mbar=bar[0]` is passed to and how the
  TMA descriptor is built.
- [Blackwell GEMM](blackwell-gemm.md) — the most mbarrier-intensive
  kernel in the repo; 4-stage ring buffer, producer/consumer warp
  specialization, cluster-scoped commit.
- [Hopper GEMM](handwritten-gemm.md) — the simplest mbarrier use: one
  barrier per K-slice, a toggled phase register, nothing more.
