"""Kernel tracing, specialization, and runtime dispatch.

The :func:`kernel` decorator is the main entry point for authoring
``pyptx`` kernels. A decorated Python function is traced into PTX and
can then be:

- inspected as PTX text with ``.ptx(...)``
- launched through the JAX runtime path
- launched through the PyTorch eager path
- launched through the ``torch.compile`` custom-op path

Example:

```python
from pyptx import kernel, Tile, Layout
from pyptx.types import bf16, f32

@kernel(
    in_specs=(Tile("M", "K", bf16, Layout.ROW),
              Tile("K", "N", bf16, Layout.COL)),
    out_specs=(Tile("M", "N", f32, Layout.ROW),),
    grid=lambda M, N, K: (M // 128, N // 256),
    block=(128, 1, 1),
    cluster=(2, 1, 1),
    arch="sm_90a",
)
def gemm(A, B, C, *, BM=128, BN=256, BK=64): ...
```

Key concepts:

- Positional parameters correspond to tensor inputs and outputs.
- Keyword-only parameters act as template parameters and are baked into
  the trace.
- ``Tile`` and ``Layout`` describe the tensor boundary contract.
- The kernel body itself emits PTX by calling into ``reg``, ``smem``,
  and ``ptx``.

Practical workflow:

```python
print(gemm.ptx(M=4096, N=4096, K=4096))
```

and then later:

```python
@jax.jit
def fwd(x, w):
    return gemm(x, w)
```

or:

```python
@torch.compile
def fwd(x, w):
    return gemm(x, w)
```
"""

from __future__ import annotations

import functools
import inspect
import struct
from typing import Any, Callable, Sequence

from pyptx._trace import trace_scope
from pyptx.emitter import emit
from pyptx.ir.nodes import (
    AddressSize,
    Function,
    FunctionDirective,
    Module,
    Param,
    Target,
    VarDecl,
    Version,
)
from pyptx.ir.types import LinkingDirective, ScalarType, StateSpace
from pyptx.specs import Tile, Layout, unify_envs


def _build_entry_params(
    positional_names: Sequence[str],
    raw_params: Sequence[tuple[str, str]] = (),
    tma_tensor_names: Sequence[str] = (),
) -> tuple[Param, ...]:
    """Build a Param list for an .entry function.

    All buffer arguments to a pyptx kernel are passed as 64-bit pointers
    in .param space, named exactly as in the Python signature. The kernel
    body is responsible for ``ld.param.u64`` / ``cvta.to.global.u64`` to
    convert them to addressable global pointers.

    Raw scalar params are appended after the regular buffer params. These are
    used by callable kernels that need a few non-tensor launch params (for
    example symbolic sizes packed as plain ``.param .u32`` scalars).

    TMA descriptor pointers are appended after both the regular buffer params
    and the raw scalar params as additional ``.param .u64
    <name>_tma_desc`` entries after the regular buffer params, in the
    same order that the kernel body first called ``<tensor>.tma_desc()``.
    The shim populates these slots at launch time via its TMA spec
    machinery (see ``pyptx.jax_support.synthesize_tma_descriptor``).
    """
    regular = tuple(
        Param(
            state_space=StateSpace.PARAM,
            type=ScalarType.U64,
            name=name,
        )
        for name in positional_names
    )
    raw = tuple(_parse_raw_param_decl(raw_type, name) for raw_type, name in raw_params)
    tma = tuple(
        Param(
            state_space=StateSpace.PARAM,
            type=ScalarType.U64,
            name=f"{name}_tma_desc",
        )
        for name in tma_tensor_names
    )
    return regular + raw + tma


def _parse_raw_param_decl(raw_type: str, pname: str) -> Param:
    """Parse a raw param declaration into an IR Param."""
    parts = raw_type.split(".")
    base_type = parts[0]
    has_ptr = "ptr" in parts
    ptr_ss = None
    ptr_align = None
    alignment = None
    array_size = None
    for part in parts[1:]:
        if part == "ptr":
            continue
        if part in ("global", "shared", "local", "const", "param"):
            if has_ptr:
                ptr_ss = StateSpace.from_ptx(f".{part}")
        elif part.startswith("align"):
            alignment = int(part[5:])
        elif part.startswith("palign"):
            ptr_align = int(part[6:])
        elif part.startswith("array"):
            array_size = int(part[5:])
    if has_ptr and ptr_align is None:
        ptr_align = 1
    return Param(
        state_space=StateSpace.PARAM,
        type=ScalarType.from_ptx(f".{base_type}"),
        name=pname,
        ptr_state_space=ptr_ss,
        ptr_alignment=ptr_align,
        alignment=alignment,
        array_size=array_size,
    )


def _raw_param_is_scalar(raw_type: str) -> bool:
    """Return True if ``raw_type`` is a scalar .param value, not a pointer/array."""
    parts = raw_type.split(".")
    return "ptr" not in parts and not any(part.startswith("array") for part in parts[1:])


def _pack_scalar_raw_param(raw_type: str, value: Any) -> tuple[int, int]:
    """Pack a scalar raw param value into (little-endian bits, size_bytes)."""
    base_type = raw_type.split(".", 1)[0]
    if base_type in {"pred", "u8", "s8", "b8"}:
        size = 1
    elif base_type in {"u16", "s16", "b16", "f16"}:
        size = 2
    elif base_type in {"u32", "s32", "b32", "f32"}:
        size = 4
    elif base_type in {"u64", "s64", "b64", "f64"}:
        size = 8
    else:
        raise TypeError(
            f"pyptx: callable raw param type {raw_type!r} is not a supported scalar"
        )

    if base_type == "pred":
        return (1 if bool(value) else 0, size)
    if base_type.startswith("f"):
        try:
            if size == 2:
                packed = struct.pack("<e", float(value))
            elif size == 4:
                packed = struct.pack("<f", float(value))
            else:
                packed = struct.pack("<d", float(value))
        except Exception as e:
            raise TypeError(
                f"pyptx: raw param {raw_type!r} expects a float-compatible value, "
                f"got {value!r}"
            ) from e
        return (int.from_bytes(packed, "little"), size)

    try:
        ivalue = int(value)
    except Exception as e:
        raise TypeError(
            f"pyptx: raw param {raw_type!r} expects an int-compatible value, got {value!r}"
        ) from e

    mask = (1 << (size * 8)) - 1
    return (ivalue & mask, size)


class TensorSpec:
    """Placeholder for a tensor argument at trace time.

    Carries the parameter name plus (if known) shape and dtype information
    derived from the input/output specs. At execution time inside jax.jit
    these are bound to real JAX arrays.

    Methods like ``tma_desc()`` return symbolic handles that get resolved
    to real pointers by the FFI launcher at kernel launch time.
    """

    __slots__ = ("name", "shape", "dtype", "layout")

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...] | None = None,
        dtype: Any = None,
        layout: Layout | None = None,
    ) -> None:
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.layout = layout

    def __repr__(self) -> str:
        if self.shape is not None:
            return f"TensorSpec({self.name!r}, shape={self.shape}, dtype={self.dtype})"
        return f"TensorSpec({self.name!r})"

    def tma_desc(self):
        """Return a TMA descriptor reference for this tensor.

        Used inside a kernel to pass the tensor to a TMA load/store:

            ptx.cp.async_.bulk.tensor_2d(
                dst=sA[0],
                src=A.tma_desc(),
                coord=(x, y),
                mbar=bar[0],
            )

        Inside an active kernel trace this function:
          1. Records ``self.name`` on the trace context so the driver
             knows to append a ``.param .u64 <name>_tma_desc`` slot to
             the emitted entry signature and to synthesize a real TMA
             descriptor at compile time.
          2. Emits an ``ld.param.u64`` prologue (once per tensor) that
             loads the descriptor pointer into a fresh register.
          3. Returns that register so it can be used directly as the
             ``src`` of a ``cp.async.bulk.tensor.*`` instruction.

        Outside a trace (e.g. in unit tests that probe the TensorSpec
        API without entering a kernel), this returns a ``TmaDescriptorHandle``
        for backwards compatibility.
        """
        # Lazy imports to avoid circular deps.
        from pyptx._trace import _local as _trace_local

        ctx = getattr(_trace_local, "ctx", None)
        if ctx is None:
            # No active trace — return the inert handle for test fixtures.
            return TmaDescriptorHandle(self)

        # Active trace. Memoize the loaded register per tensor name so
        # repeated `A.tma_desc()` calls reuse the same Reg.
        if not hasattr(ctx, "tma_tensor_names"):
            ctx.tma_tensor_names = []
            ctx.tma_desc_regs = {}

        if self.name in ctx.tma_desc_regs:
            return ctx.tma_desc_regs[self.name]

        # First use of this tensor's descriptor. Record the name so
        # _build_entry_params adds an extra `.param .u64 <name>_tma_desc`.
        ctx.tma_tensor_names.append(self.name)

        # Emit `ld.param.u64 %rd, [<name>_tma_desc];` and return the reg.
        from pyptx import ptx as _ptx
        from pyptx.reg import scalar as reg_scalar
        from pyptx.types import u64

        rd = reg_scalar(u64)
        _ptx.inst.ld.param.u64(rd, _ptx.addr(f"{self.name}_tma_desc"))
        ctx.tma_desc_regs[self.name] = rd
        return rd


class TmaDescriptorHandle:
    """Symbolic handle for a TMA descriptor.

    Carries a reference back to the TensorSpec so the FFI launcher can
    build the real cuTensorMap at runtime from the JAX array metadata.
    In the emitted PTX it's rendered as the symbolic name (e.g. ``A_desc``).
    """

    __slots__ = ("tensor", "name")

    def __init__(self, tensor: TensorSpec) -> None:
        self.tensor = tensor
        self.name = f"{tensor.name}_tma_desc"

    def __repr__(self) -> str:
        return f"TmaDescriptorHandle({self.name!r})"


def _default_version_for_arch(arch: str) -> tuple[int, int]:
    """Return the default PTX ISA version for a target architecture.

    Hopper examples in this repo were originally emitted as PTX 8.5.
    Blackwell tcgen05 kernels need a newer PTX target than Hopper, but
    CUDA 12.9 toolchains still top out at PTX 8.8. Workstation Blackwell
    (sm_120, RTX Pro 6000) was added in CUDA 12.8 / PTX 8.7. Default to
    the newest version that assembles on the current bring-up stack for
    each arch instead of emitting PTX 9.2 unconditionally.
    """
    if arch.startswith("sm_100") or arch.startswith("sm_101"):
        return (8, 8)
    if arch.startswith("sm_12"):
        return (8, 7)
    return (8, 5)


class Kernel:
    """A traced PTX kernel. Wraps a Python function that uses ptx.* calls."""

    def __init__(
        self,
        fn: Callable,
        arch: str = "sm_90a",
        version: tuple[int, int] | None = None,
        in_specs: Sequence[Tile] | None = None,
        out_specs: Sequence[Tile] | None = None,
        grid: Callable[..., tuple[int, int, int]] | tuple[int, int, int] | None = None,
        block: tuple[int, int, int] = (1, 1, 1),
        cluster: tuple[int, int, int] = (1, 1, 1),
        smem: int = 0,
        raw_params: Sequence[tuple[str, str]] | None = None,
        extern_smem: bool | str = False,
        reqntid: tuple[int, ...] | None = None,
        raw_directives: Sequence[tuple[str, tuple]] | None = None,
    ) -> None:
        self._fn = fn
        self._arch = arch
        self._version = version if version is not None else _default_version_for_arch(arch)
        self._signature = inspect.signature(fn)

        # JAX integration specs
        self._in_specs: tuple[Tile, ...] = tuple(in_specs) if in_specs else ()
        self._out_specs: tuple[Tile, ...] = tuple(out_specs) if out_specs else ()
        self._grid = grid
        self._block: tuple[int, int, int] = tuple(block)  # type: ignore
        self._cluster: tuple[int, int, int] = tuple(cluster)  # type: ignore
        self._smem: int = smem
        self._raw_params = raw_params
        self._extern_smem = extern_smem
        self._reqntid = reqntid
        self._raw_directives = raw_directives

        # Split the signature into positional (tensor placeholders) and
        # keyword-only (template params).
        self._positional_names: tuple[str, ...] = tuple(
            name
            for name, p in self._signature.parameters.items()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        self._template_defaults: dict[str, Any] = {
            name: (p.default if p.default is not inspect.Parameter.empty else None)
            for name, p in self._signature.parameters.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        }
        self._template_names: frozenset[str] = frozenset(self._template_defaults)
        self._raw_param_names: frozenset[str] = frozenset(
            pname for _, pname in (raw_params or ())
        )

        # Specialization cache: key -> Module
        self._cache: dict[tuple, Module] = {}

        # Cubin cache (keyed by concrete input shapes + template kwargs + arch)
        self._cubin_handles: dict[tuple, int] = {}

        functools.update_wrapper(self, fn)

    # -- JAX spec accessors -------------------------------------------------

    @property
    def in_specs(self) -> tuple[Tile, ...]:
        """Input tensor specs declared on the kernel."""
        return self._in_specs

    @property
    def out_specs(self) -> tuple[Tile, ...]:
        """Output tensor specs declared on the kernel."""
        return self._out_specs

    @property
    def grid(self):
        """Configured grid tuple or grid resolver callable."""
        return self._grid

    @property
    def block(self) -> tuple[int, int, int]:
        """Static CUDA block dimensions for the kernel."""
        return self._block

    @property
    def cluster(self) -> tuple[int, int, int]:
        """CTA cluster dimensions used at launch time."""
        return self._cluster

    @property
    def smem(self) -> int:
        """Requested dynamic/shared memory size in bytes."""
        return self._smem

    # -- Inspection ---------------------------------------------------------

    @property
    def template_params(self) -> dict[str, Any]:
        """Return the declared template parameters and their default values.

        Only keyword-only parameters in the function signature count as
        template parameters. Positional args are tensor placeholders, not
        template parameters.
        """
        return dict(self._template_defaults)

    @property
    def arch(self) -> str:
        """Target PTX architecture string, e.g. ``sm_90a``."""
        return self._arch

    # -- Internal helpers ---------------------------------------------------

    @property
    def _shape_var_names(self) -> frozenset[str]:
        """Symbolic shape variables from in_specs + out_specs.

        e.g. if in_specs = (Tile("M","K",bf16), Tile("K","N",bf16)),
        this returns frozenset({"M","K","N"}).
        """
        names: set[str] = set()
        for spec in self._in_specs:
            names.update(spec.symbolic_dims)
        for spec in self._out_specs:
            names.update(spec.symbolic_dims)
        return frozenset(names)

    def _split_kwargs(
        self, kwargs: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
        """Partition kwargs into (template_kwargs, shape_env, raw_param_kwargs).

        Shape variables are names declared as symbolic dims in in_specs/
        out_specs. Template parameters are keyword-only params in the
        function signature. Unknown kwargs raise TypeError.
        """
        shape_names = self._shape_var_names
        template_kwargs: dict[str, Any] = {}
        shape_env: dict[str, int] = {}
        raw_param_kwargs: dict[str, Any] = {}
        unknown: list[str] = []
        for k, v in kwargs.items():
            if k in shape_names:
                if not isinstance(v, int):
                    raise TypeError(
                        f"Shape variable {k!r} must be an int, got {type(v).__name__}"
                    )
                shape_env[k] = v
            elif k in self._template_names:
                template_kwargs[k] = v
            elif k in self._raw_param_names:
                raw_param_kwargs[k] = v
            else:
                unknown.append(k)
        if unknown:
            avail_t = ", ".join(sorted(self._template_names)) or "(none)"
            avail_s = ", ".join(sorted(shape_names)) or "(none)"
            avail_r = ", ".join(sorted(self._raw_param_names)) or "(none)"
            raise TypeError(
                f"Kernel {self._fn.__name__} has no template parameter or "
                f"shape variable {sorted(unknown)!r}. "
                f"Template params: {avail_t}. Shape vars: {avail_s}. "
                f"Raw params: {avail_r}"
            )
        return template_kwargs, shape_env, raw_param_kwargs

    def _resolve_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Validate kwargs and fill in defaults from the signature.

        Accepts both template parameters and shape variables; shape vars
        are passed through separately via _shape_env. Raises TypeError
        for unknown kwargs.
        """
        template_kwargs, _, _ = self._split_kwargs(kwargs)
        resolved = dict(self._template_defaults)
        resolved.update(template_kwargs)
        return resolved

    def _resolve_callable_raw_params(
        self,
        *,
        shape_env: dict[str, int],
        template_kwargs: dict[str, Any],
        raw_param_kwargs: dict[str, Any],
    ) -> tuple[tuple[str, str, int, int], ...]:
        """Resolve callable raw params to concrete packed scalar values."""
        resolved: list[tuple[str, str, int, int]] = []
        for raw_type, pname in (self._raw_params or ()):
            if not _raw_param_is_scalar(raw_type):
                raise NotImplementedError(
                    f"pyptx: callable kernels currently only support scalar raw params; "
                    f"{pname!r} uses unsupported type {raw_type!r}. Use normal tensor "
                    f"specs + tensor.tma_desc() for descriptors/pointers."
                )
            if pname in raw_param_kwargs:
                value = raw_param_kwargs[pname]
            elif pname in shape_env:
                value = shape_env[pname]
            elif pname in template_kwargs and template_kwargs[pname] is not None:
                value = template_kwargs[pname]
            else:
                raise TypeError(
                    f"pyptx: callable kernel {self._fn.__name__} requires raw param "
                    f"{pname!r}; pass it explicitly or make it available via the "
                    f"shape/template environment"
                )
            bits, size = _pack_scalar_raw_param(raw_type, value)
            resolved.append((raw_type, pname, bits, size))
        return tuple(resolved)

    def _cache_key(self, resolved_kwargs: dict[str, Any]) -> tuple:
        """Build a hashable cache key from arch/version + resolved kwargs."""
        try:
            items = tuple(sorted(resolved_kwargs.items()))
            # Force hashability check
            hash(items)
        except TypeError:
            # Fall back to repr-based key for unhashable values
            items = tuple(
                (k, repr(v)) for k, v in sorted(resolved_kwargs.items())
            )
        return (self._arch, self._version, items)

    def _bind_positional(
        self,
        resolved_kwargs: dict[str, Any] | None = None,
        shape_env: dict[str, int] | None = None,
    ) -> tuple[TensorSpec, ...]:
        """Build TensorSpec placeholders for each positional parameter.

        If in_specs / out_specs are declared, resolve their symbolic dims
        against shape_env (if provided) or the template kwargs (as a
        fallback — the user may pass M=4096, N=4096, K=4096 to .ptx()).
        This lets kernel bodies reference A.shape[1], A.dtype, etc.
        """
        # Build the shape env to resolve symbolic dims with.
        env: dict[str, int] = {}
        if shape_env:
            env.update(shape_env)
        if resolved_kwargs:
            # Template kwargs that look like shape vars (M, N, K, ...)
            # get treated as a shape env when no real one is provided.
            for k, v in resolved_kwargs.items():
                if isinstance(v, int) and k not in env:
                    env[k] = v

        # All declared specs: in_specs first, then out_specs
        all_specs: list[Tile] = []
        all_specs.extend(self._in_specs)
        all_specs.extend(self._out_specs)

        specs_list: list[TensorSpec] = []
        for i, name in enumerate(self._positional_names):
            if i < len(all_specs):
                spec = all_specs[i]
                try:
                    concrete_shape = spec.resolve_shape(env)
                except KeyError:
                    concrete_shape = None
                specs_list.append(TensorSpec(
                    name=name,
                    shape=concrete_shape,
                    dtype=spec.dtype,
                    layout=spec.layout,
                ))
            else:
                specs_list.append(TensorSpec(name=name))
        return tuple(specs_list)

    # -- Tracing ------------------------------------------------------------

    def _trace(
        self,
        _shape_env: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> Module:
        """Trace the kernel function and return an IR Module.

        kwargs may include BOTH template parameters (keyword-only params
        in the function signature like BM, BN, BK) AND shape variables
        (symbolic dims declared in in_specs/out_specs like M, N, K).
        They are partitioned automatically by name.

        If _shape_env is provided (from __call__ with concrete JAX arrays),
        it seeds the shape variables and any caller kwargs override it.
        """
        # Split kwargs into template params vs shape vars
        template_kwargs, caller_shape_env, _ = self._split_kwargs(kwargs)
        resolved = dict(self._template_defaults)
        resolved.update(template_kwargs)

        # Merge shape envs: _shape_env (from real JAX arrays) + caller kwargs
        merged_shape_env: dict[str, int] = {}
        if _shape_env:
            merged_shape_env.update(_shape_env)
        merged_shape_env.update(caller_shape_env)

        # Cache key: template kwargs + shape env
        key_parts: list[Any] = [self._arch, self._version]
        try:
            key_parts.append(tuple(sorted(resolved.items())))
        except TypeError:
            key_parts.append(tuple(
                (k, repr(v)) for k, v in sorted(resolved.items())
            ))
        if merged_shape_env:
            key_parts.append(tuple(sorted(merged_shape_env.items())))
        key = tuple(key_parts)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        positional = self._bind_positional(
            resolved_kwargs=resolved,
            shape_env=merged_shape_env,
        )

        force_dynamic_trace = self._smem > 48 * 1024
        while True:
            with trace_scope(ptx_version=self._version) as ctx:
                # If the user declared smem > 48KB, or a prior trace
                # discovered that this kernel needs >48KB, activate
                # dynamic mode BEFORE tracing so all smem allocs use
                # offset-based addressing.
                if force_dynamic_trace:
                    ctx.force_dynamic_smem = True
                ctx.raw_param_types = {
                    pname: raw_type for raw_type, pname in (self._raw_params or ())
                }
                ctx.extern_smem_name = (
                    self._extern_smem
                    if isinstance(self._extern_smem, str)
                    else "global_smem" if self._extern_smem else None
                )

                # Call the user's function — all ptx.*, reg.*, smem.* calls
                # record into ctx via the thread-local trace context.
                self._fn(*positional, **resolved)

                # Pick up any TMA tensors the body asked for via .tma_desc();
                # those get additional trailing .param .u64 entries.
                tma_names: tuple[str, ...] = tuple(
                    getattr(ctx, "tma_tensor_names", []) or ()
                )

                # Entry param list: one .param .u64 per positional arg so the
                # buffer pointers JAX passes in are visible inside the kernel,
                # plus one extra per TMA-described tensor.
                if self._raw_params is not None:
                    # Raw params: user declares params explicitly (for transpiled kernels).
                    # Type string examples:
                    #   "u64.ptr.global.palign1" → .param .u64 .ptr .global .align 1 name
                    #   "u32" → .param .u32 name
                    #   "b8.align64.array128" → .param .align 64 .b8 name[128]
                    import re
                    raw_params_list = tuple(
                        _parse_raw_param_decl(raw_type, pname)
                        for raw_type, pname in self._raw_params
                    )
                    if self._in_specs or self._out_specs:
                        params = _build_entry_params(
                            self._positional_names,
                            raw_params=self._raw_params,
                            tma_tensor_names=tma_names,
                        )
                    else:
                        params = raw_params_list
                elif self._in_specs or self._out_specs:
                    params = _build_entry_params(self._positional_names, tma_tensor_names=tma_names)
                else:
                    params = ()

                total_smem = ctx.dyn_smem_offset
                use_dynamic_smem = total_smem > 48 * 1024

                # First trace can discover that the kernel actually needs
                # dynamic shared memory. Retrace in dynamic mode so every
                # shared allocation and mbarrier is lowered consistently to
                # dyn_smem-based addressing instead of leaving orphaned
                # smem_* / mbar_* symbols behind.
                if use_dynamic_smem and not ctx.force_dynamic_smem:
                    force_dynamic_trace = True
                    continue

                if use_dynamic_smem:
                    # Remove static .shared VarDecls from function body
                    ctx.var_decls = [
                        v for v in ctx.var_decls
                        if not (hasattr(v, 'state_space') and v.state_space == StateSpace.SHARED)
                    ]

                self._last_trace_smem = total_smem if use_dynamic_smem else 0

                func_directives: list[FunctionDirective] = []
                if self._raw_directives is not None:
                    for dname, dvals in self._raw_directives:
                        func_directives.append(FunctionDirective(
                            name=dname,
                            values=tuple(dvals),
                        ))
                elif self._reqntid is not None:
                    func_directives.append(FunctionDirective(
                        name="reqntid",
                        values=tuple(self._reqntid),
                    ))

                func = Function(
                    is_entry=True,
                    name=self._fn.__name__,
                    params=params,
                    body=ctx.body(),
                    linking=LinkingDirective.VISIBLE,
                    directives=tuple(func_directives),
                )
                break

        # Build module directives: optionally prepend extern .shared
        module_directives: list = []
        if self._extern_smem:
            from pyptx.ir.types import LinkingDirective as LD
            smem_name = self._extern_smem if isinstance(self._extern_smem, str) else "global_smem"
            module_directives.append(VarDecl(
                state_space=StateSpace.SHARED,
                type=ScalarType.from_ptx(".b8"),
                name=smem_name,
                array_size=None,
                alignment=128,
                linking=LD.EXTERN,
            ))
        elif use_dynamic_smem:
            from pyptx.ir.types import LinkingDirective as LD
            module_directives.append(VarDecl(
                state_space=StateSpace.SHARED,
                type=ScalarType.from_ptx(".b8"),
                name="dyn_smem",
                array_size=None,
                alignment=128,
                linking=LD.EXTERN,
            ))
        module_directives.append(func)

        module = Module(
            version=Version(self._version[0], self._version[1]),
            target=Target((self._arch,)),
            address_size=AddressSize(64),
            directives=tuple(module_directives),
        )
        # Stash the TMA tensor list on a side table keyed by the
        # Module's id so Kernel.__call__ can wire up shim TMA specs
        # without re-tracing. Module is a frozen dataclass so we can't
        # attach attributes to it directly.
        if not hasattr(self, "_tma_names_by_module_id"):
            self._tma_names_by_module_id: dict[int, tuple[str, ...]] = {}
        self._tma_names_by_module_id[id(module)] = tma_names

        self._cache[key] = module
        return module

    # -- Public inspection API ---------------------------------------------

    def ptx(self, **kwargs: Any) -> str:
        """Trace and emit PTX text. The inspection API.

        Pass template kwargs (BM, BN, BK, etc.) and/or shape variables
        (M, N, K, ...) to specialize. Defaults from the function signature
        fill in any kwargs you don't supply.

        Usage:
            print(my_kernel.ptx(M=4096, N=4096, K=4096, BM=128))
        """
        module = self._trace(**kwargs)
        return emit(module)

    def module(self, **kwargs: Any) -> Module:
        """Trace and return the IR Module (for programmatic inspection)."""
        return self._trace(**kwargs)

    def sass(self, **kwargs: Any) -> str:
        """Compile PTX to cubin and disassemble to SASS via cuobjdump.

        This is the "what actually ran on the GPU" view — useful for
        performance tuning and understanding how ptxas lowered your PTX.

        Requires the CUDA toolkit to be installed (for ptxas + cuobjdump).
        Raises RuntimeError with a helpful message if the toolkit is not
        available.

        Usage:
            print(my_kernel.sass(M=4096, N=4096, K=4096))
        """
        import os
        import shutil
        import subprocess
        import tempfile

        ptxas = shutil.which("ptxas")
        cuobjdump = shutil.which("cuobjdump")
        if ptxas is None or cuobjdump is None:
            missing = [n for n, p in (("ptxas", ptxas), ("cuobjdump", cuobjdump)) if p is None]
            raise RuntimeError(
                f"kernel.sass() requires the CUDA toolkit ({', '.join(missing)} "
                f"not found in PATH). Install CUDA or add its bin directory to "
                f"PATH. On Linux: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
            )

        ptx_source = self.ptx(**kwargs)
        with tempfile.TemporaryDirectory() as tmpdir:
            ptx_path = os.path.join(tmpdir, f"{self._fn.__name__}.ptx")
            cubin_path = os.path.join(tmpdir, f"{self._fn.__name__}.cubin")
            with open(ptx_path, "w") as f:
                f.write(ptx_source)

            # Compile PTX → cubin
            result = subprocess.run(
                [ptxas, f"-arch={self._arch}", "-o", cubin_path, ptx_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ptxas failed for kernel {self._fn.__name__}:\n"
                    f"{result.stderr}"
                )

            # Disassemble cubin → SASS
            result = subprocess.run(
                [cuobjdump, "--dump-sass", cubin_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"cuobjdump failed for kernel {self._fn.__name__}:\n"
                    f"{result.stderr}"
                )
            return result.stdout

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the kernel with concrete JAX arrays.

        This builds a jax.ffi.ffi_call that XLA will dispatch to our
        registered FFI target at execution time. Works inside @jax.jit.

        Args:
            *args: JAX arrays matching in_specs + out_specs (in that order).
                   If out_specs has N entries, the last N args are treated
                   as output buffers (or omitted for return-by-value).
            **kwargs: Template kwargs that override defaults.

        Returns:
            A JAX array (or tuple of arrays) matching out_specs. If there's
            exactly one output, returns a single array; otherwise a tuple.
        """
        if not self._in_specs:
            raise NotImplementedError(
                f"Kernel {self._fn.__name__} has no in_specs — calling it "
                f"directly requires declaring in_specs / out_specs on the "
                f"@kernel decorator. Use .ptx() to inspect the traced PTX."
            )

        # All the runtime glue that doesn't depend on JAX / PyTorch.
        from pyptx.jax_support import (
            add_tma_spec_to_shim,
            compile_ptx_to_cubin,
            get_cubin_registry,
            register_launch_config,
            shim_is_available,
            synthesize_tma_descriptor,
            synthesize_tma_descriptor_3d,
        )
        from pyptx.torch_support import any_torch_tensors

        # Input arrays are the positional args matching in_specs
        input_arrays = args[: len(self._in_specs)]
        if len(input_arrays) != len(self._in_specs):
            raise TypeError(
                f"Kernel {self._fn.__name__} expected {len(self._in_specs)} "
                f"input arrays, got {len(input_arrays)}"
            )

        # --- Turbo fast path: skip all Python overhead for repeated
        # Torch eager calls with the same input shapes. ---
        if not kwargs and len(input_arrays) == len(self._in_specs) and any_torch_tensors(input_arrays):
            turbo = getattr(self, '_turbo_torch', None)
            if turbo is not None:
                shapes_key = tuple(a.shape for a in input_arrays)
                if turbo[0] == shapes_key:
                    _ext, _handle, _out_shapes, _out_dtypes = turbo[1], turbo[2], turbo[3], turbo[4]
                    import torch
                    _s = torch.cuda.current_stream(input_arrays[0].device)
                    stream_ptr = int(_s.cuda_stream if hasattr(_s, 'cuda_stream') else _s)
                    results = _ext.launch_kernel(
                        _handle, stream_ptr, list(input_arrays),
                        _out_shapes, _out_dtypes,
                    )
                    if len(results) == 1:
                        return results[0]
                    return tuple(results)

        # Dispatch: if any input is a torch.Tensor, go down the PyTorch
        # runtime path. Otherwise use the JAX FFI path. The compile /
        # register / TMA machinery below is shared.
        use_torch = any_torch_tensors(input_arrays)

        # Extract and unify the shape env from the input avals / tensors.
        # Both JAX arrays and torch.Tensors expose ``.shape`` that is
        # iterable of ints; we convert to a tuple for extract_env.
        envs: list[dict[str, int]] = []
        for arr, spec in zip(input_arrays, self._in_specs):
            envs.append(spec.extract_env(tuple(int(d) for d in arr.shape)))
        shape_env = unify_envs(envs)

        # Resolve template kwargs + callable raw params from the explicit kwargs.
        template_overrides, caller_shape_kwargs, raw_param_kwargs = self._split_kwargs(kwargs)
        template_kwargs = dict(self._template_defaults)
        template_kwargs.update(template_overrides)

        # Build the cubin cache key.
        # Fast path: for repeated calls with the same shapes, reuse the
        # last cache key to avoid tuple/sort overhead.
        input_shapes = tuple(
            (a.shape, a.dtype) for a in input_arrays
        )
        _last = getattr(self, '_last_cache_hit', None)
        if _last is not None and _last[0] == input_shapes and not kwargs:
            cache_key = _last[1]
        else:
            cache_key = (
                self._arch,
                self._version,
                tuple(sorted(template_kwargs.items())),
                tuple(sorted(shape_env.items())),
                input_shapes,
            )
        if self._raw_params:
            raw_param_values = self._resolve_callable_raw_params(
                shape_env={**shape_env, **caller_shape_kwargs},
                template_kwargs=template_kwargs,
                raw_param_kwargs=raw_param_kwargs,
            )
            cache_key = cache_key + (tuple((name, bits, size) for _, name, bits, size in raw_param_values),)
        else:
            raw_param_values = ()

        if cache_key not in self._cubin_handles:
            # Trace → PTX → cubin → register
            # Pass the shape env so TensorSpec.shape is resolved inside
            # the kernel body (A.shape[1] works at trace time).
            module = self._trace(_shape_env=shape_env, **template_kwargs)
            ptx_source = emit(module)
            grid_tuple = self._resolve_grid(shape_env)

            # TMA tensors the body called `.tma_desc()` on, in order.
            tma_names: tuple[str, ...] = getattr(
                self, "_tma_names_by_module_id", {}
            ).get(id(module), ())

            # JIT the PTX into a CUfunction via cuda-python. On machines
            # without cuda-python / CUDA, returns None and we still
            # register the PTX so lowering tests can inspect the HLO.
            # On a real GPU, failures propagate — a broken kernel should
            # fail loudly at compile time, not at launch.
            dyn_smem = getattr(self, '_last_trace_smem', 0)
            requested_smem = max(self._smem, dyn_smem)
            compiled = compile_ptx_to_cubin(
                ptx_source,
                self._arch,
                kernel_name=self._fn.__name__,
                dynamic_smem_bytes=requested_smem,
            )
            if compiled is not None:
                cu_function, cu_module = compiled
            else:
                cu_function, cu_module = None, None

            handle = get_cubin_registry().register(
                ptx_source=ptx_source,
                kernel_name=self._fn.__name__,
                smem_bytes=self._smem,
                grid=grid_tuple,
                block=self._block,
                cu_function=cu_function,
                module=cu_module,
            )

            # If the shim is available but JIT-compile returned no CUfunction,
            # the user is in a real-GPU runtime path missing cuda-python.
            # Without this, the launch later fails with the cryptic
            # "no launch config registered for handle N" from the C++ shim.
            if cu_function is None and shim_is_available():
                try:
                    import cuda.bindings.driver  # noqa: F401
                except ImportError:
                    raise RuntimeError(
                        "pyptx: PTX-to-cubin JIT requires cuda-python on a real "
                        "GPU. Install it with: pip install cuda-python"
                    )

            # If the shim and a real CUfunction are both available, push
            # the launch config into the shim's registry so XLA's dispatch
            # can find it. On laptop this is a no-op.
            if cu_function is not None and shim_is_available():
                register_launch_config(
                    handle=handle,
                    cu_function=cu_function,
                    grid=grid_tuple,
                    block=self._block,
                    cluster=self._cluster,
                    smem_bytes=requested_smem,
                )

                if raw_param_values:
                    from pyptx.jax_support import add_scalar_param_to_shim

                    for _raw_type, _name, bits, size in raw_param_values:
                        add_scalar_param_to_shim(
                            handle=handle,
                            value_bits=bits,
                            size_bytes=size,
                        )

                # For each TMA tensor, synthesize a host-side CUtensorMap
                # + device-side 128-byte slot and register both with the
                # shim. The shim patches the host blob with the real
                # XLA buffer pointer and uploads it per launch.
                if tma_names:
                    # Keep these alive for the life of the kernel so the
                    # CUtensorMap Python objects and the device buffers
                    # aren't freed underneath the shim.
                    if not hasattr(self, "_tma_slots_by_handle"):
                        self._tma_slots_by_handle: dict[int, list] = {}
                    slots: list = []
                    n_inputs = len(self._in_specs)
                    n_outputs = len(self._out_specs)
                    for name in tma_names:
                        # Find the positional index of this tensor.
                        try:
                            pos_idx = self._positional_names.index(name)
                        except ValueError:
                            raise RuntimeError(
                                f"pyptx: kernel {self._fn.__name__!r} called "
                                f"tma_desc() on an unknown tensor {name!r}"
                            )
                        # Resolve against in_specs OR out_specs. The shim
                        # uses a unified inputs-then-outputs index space,
                        # so outputs get xla_arg_index = n_inputs + out_idx.
                        if pos_idx < n_inputs:
                            spec = self._in_specs[pos_idx]
                            xla_arg_index = pos_idx
                        else:
                            out_idx = pos_idx - n_inputs
                            if out_idx >= n_outputs:
                                raise RuntimeError(
                                    f"pyptx: tensor {name!r} at position "
                                    f"{pos_idx} has no matching spec"
                                )
                            spec = self._out_specs[out_idx]
                            xla_arg_index = n_inputs + out_idx

                        concrete_shape = spec.resolve_shape(shape_env)
                        # Resolve the per-TMA box shape (from Tile.wgmma_a's
                        # tile_k / Tile.wgmma_b's tile_k,tile_n) against the
                        # same shape env, so K-loop kernels load exactly one
                        # slice per cp.async.bulk.tensor instead of the
                        # whole tensor.
                        concrete_tma_box: tuple[int, ...] | None = None
                        if getattr(spec, "tma_box", None) is not None:
                            box_resolved: list[int] = []
                            for d in spec.tma_box:
                                if isinstance(d, int):
                                    box_resolved.append(d)
                                elif isinstance(d, str):
                                    if d not in shape_env:
                                        raise KeyError(
                                            f"pyptx: tma_box dim {d!r} not "
                                            f"bound in shape env {sorted(shape_env)}"
                                        )
                                    box_resolved.append(shape_env[d])
                                else:
                                    raise TypeError(
                                        f"pyptx: tma_box dim must be int or str, got {type(d).__name__}"
                                    )
                            concrete_tma_box = tuple(box_resolved)
                        if getattr(spec, "tma_rank", 2) == 3:
                            if concrete_tma_box is None:
                                raise ValueError(
                                    f"pyptx: tensor {name!r} requested a rank-3 TMA descriptor "
                                    f"but spec.tma_box is not set"
                                )
                            host_tmap, host_blob_ptr, dev_blob_ptr = synthesize_tma_descriptor_3d(
                                height=int(concrete_shape[0]),
                                width=int(concrete_shape[1]),
                                dtype=spec.dtype,
                                box_major=int(concrete_tma_box[0]),
                                box_minor=int(concrete_tma_box[1]),
                                swizzle_128b=(spec.layout == Layout.TMA_128B),
                                padding=bool(getattr(spec, "tma_padding", False)),
                            )
                        else:
                            host_tmap, host_blob_ptr, dev_blob_ptr = synthesize_tma_descriptor(
                                shape=concrete_shape,
                                dtype=spec.dtype,
                                layout=spec.layout,
                                box_shape=concrete_tma_box,
                            )
                        add_tma_spec_to_shim(
                            handle=handle,
                            xla_arg_index=xla_arg_index,
                            host_blob_ptr=host_blob_ptr,
                            device_blob_ptr=dev_blob_ptr,
                        )
                        slots.append((host_tmap, host_blob_ptr, dev_blob_ptr))
                    self._tma_slots_by_handle[handle] = slots

            self._cubin_handles[cache_key] = handle

        cubin_handle = self._cubin_handles[cache_key]
        self._last_cache_hit = (input_shapes, cache_key)

        # Resolve grid from the shape env (again, for the call site —
        # already registered above in the cache-miss branch).
        grid_tuple = self._resolve_grid(shape_env)

        # Dispatch to the right runtime path.
        if use_torch:
            from pyptx.torch_support import _try_load_cpp_ext
            ext = _try_load_cpp_ext()
            if ext is not None and not kwargs:
                shapes_key = tuple(a.shape for a in input_arrays)
                _dtype_map = {5: 5, 15: 15, 6: 6, 7: 7, 1: 1, 2: 2, 3: 3, 4: 4, 0: 0, 11: 11}
                import torch as _th
                _DTYPE_TO_INT = {_th.float16: 5, _th.bfloat16: 15, _th.float32: 6,
                                 _th.float64: 7, _th.int8: 1, _th.int16: 2,
                                 _th.int32: 3, _th.int64: 4, _th.uint8: 0, _th.bool: 11}
                from pyptx.torch_support import _ptx_type_to_torch_dtype
                out_shapes = [list(s.resolve_shape(shape_env)) for s in self._out_specs]
                out_dtypes = [_DTYPE_TO_INT.get(_ptx_type_to_torch_dtype(s.dtype), 6)
                              for s in self._out_specs]
                self._turbo_torch = (shapes_key, ext, cubin_handle, out_shapes, out_dtypes)

            from pyptx.torch_support import call_kernel_via_torch_compile
            return call_kernel_via_torch_compile(
                *input_arrays,
                cubin_handle=cubin_handle,
                out_specs=self._out_specs,
                out_shape_env=shape_env,
                grid=grid_tuple,
                block=self._block,
                cluster=self._cluster,
                smem_bytes=self._smem,
            )
        else:
            from pyptx.jax_support import call_kernel_via_ffi
            return call_kernel_via_ffi(
                *input_arrays,
                cubin_handle=cubin_handle,
                out_specs=self._out_specs,
                out_shape_env=shape_env,
                grid=grid_tuple,
                block=self._block,
                cluster=self._cluster,
                smem_bytes=self._smem,
            )

    def _resolve_grid(self, shape_env: dict[str, int]) -> tuple[int, int, int]:
        """Resolve the grid spec (which may be a lambda over shape vars) to a concrete tuple."""
        if self._grid is None:
            return (1, 1, 1)
        if callable(self._grid):
            try:
                result = self._grid(**shape_env)
            except TypeError:
                # Try positional call with shape values in a consistent order
                result = self._grid(*shape_env.values())
            if isinstance(result, int):
                return (result, 1, 1)
            result = tuple(result)
            while len(result) < 3:
                result = result + (1,)
            return result[:3]  # type: ignore
        return tuple(self._grid)  # type: ignore

    def __repr__(self) -> str:
        return f"Kernel({self._fn.__name__}, arch={self._arch!r})"


def kernel(
    fn: Callable | None = None,
    *,
    arch: str = "sm_90a",
    version: tuple[int, int] | None = None,
    # Auto-detection: pass arch="auto" to query the first CUDA device's
    # compute capability and target that arch (e.g. "sm_80" on A100,
    # "sm_90a" on H100, "sm_100a" on B200). See pyptx.detect_arch().
    in_specs: Sequence[Tile] | None = None,
    out_specs: Sequence[Tile] | None = None,
    grid: Any = None,
    block: tuple[int, int, int] = (1, 1, 1),
    cluster: tuple[int, int, int] = (1, 1, 1),
    smem: int = 0,
    raw_params: Sequence[tuple[str, str]] | None = None,
    extern_smem: bool = False,
    reqntid: tuple[int, ...] | None = None,
    raw_directives: Sequence[tuple[str, tuple]] | None = None,
) -> Kernel | Callable[[Callable], Kernel]:
    """Decorator to define a PTX kernel.

    Can be used with or without arguments:

        @kernel
        def simple(): ...

        @kernel(arch="sm_100a")
        def blackwell(): ...

        @kernel(
            in_specs=(Tile("M", "K", bf16), Tile("K", "N", bf16)),
            out_specs=(Tile("M", "N", f32),),
            grid=lambda M, N, K: (M // 128, N // 256),
            block=(128, 1, 1),
            cluster=(2, 1, 1),
            arch="sm_90a",
        )
        def gemm(A, B, C, *, BM=128): ...
    """
    # Resolve arch="auto" once, here, so all build paths see a concrete
    # arch string. Keeps Kernel.__init__ free of detection logic.
    if arch == "auto":
        from pyptx._arch import detect_arch
        resolved_arch = detect_arch()
    else:
        resolved_arch = arch

    def _build(fn: Callable) -> Kernel:
        return Kernel(
            fn,
            arch=resolved_arch,
            version=version,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            block=block,
            cluster=cluster,
            smem=smem,
            raw_params=raw_params,
            extern_smem=extern_smem,
            reqntid=reqntid,
            raw_directives=raw_directives,
        )

    if fn is not None:
        # Used without parentheses: @kernel
        return _build(fn)

    # Used with parentheses: @kernel(arch="sm_100a", ...)
    return _build
