# Install

`pyptx` keeps runtime dependencies optional. Pick the install that
matches what you want to do:

| Command | What you get |
| --- | --- |
| `pip install pyptx` | DSL, parser, emitter, transpiler (no GPU runtime) |
| `pip install 'pyptx[torch]'` | + PyTorch eager and `torch.compile` launch path |
| `pip install 'pyptx[jax]'` | + `jax.jit` launch path via typed FFI |
| `pip install 'pyptx[all]'` | + both PyTorch and JAX |
| `pip install 'pyptx[docs]'` | + mkdocs-material for building the docs site |

## Base package

```bash
pip install pyptx
```

Zero required dependencies. This is enough for:

- the DSL and `@kernel` tracing
- parsing and emitting PTX
- the PTX → pyptx transpiler
- anything that doesn't need to actually execute a kernel

If you only want to read/write PTX or prototype kernels without a GPU,
stop here.

## PyTorch runtime

```bash
pip install 'pyptx[torch]'
```

Pulls in `torch>=2.1`. Gives you PyTorch eager dispatch through
`torch.library.custom_op` and `torch.compile` integration.

!!! tip "Install `ninja` for the fast path"
    `pyptx` ships a C++ torch extension (`pyptx/_shim/torch_ext.cpp`)
    that JIT-compiles on first launch and drops dispatch overhead
    from ~34 µs to ~14 µs. The JIT build needs `ninja`:

    ```bash
    pip install ninja
    ```

    Without `ninja`, pyptx silently falls back to the ctypes path.

## JAX runtime

```bash
pip install 'pyptx[jax]'
```

Pulls in `jax[cuda12]>=0.4.20` and `cuda-python>=12.3`. Kernels
register as typed FFI handlers that run inside `jax.jit`.

!!! tip "Blackwell / CUDA 13 setups"
    On a B200 box running CUDA 13 / driver ≥ 575, the CUDA-12 jaxlib
    falls back to CPU and pyptx will fail to launch with
    `CUDA_ERROR_INVALID_CONTEXT`. Install the CUDA-13 jax build:

    ```bash
    pip install 'jax[cuda13]'
    ```

    And set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.2` (or
    `XLA_PYTHON_CLIENT_PREALLOCATE=false`) when the host also runs
    other CUDA processes (vLLM, etc.) that already claim HBM.

## Everything

```bash
pip install 'pyptx[all]'
```

Equivalent to `pyptx[jax,torch]`. Useful if you're building kernels
that need to be callable from both frameworks.

## Docs tooling

```bash
pip install 'pyptx[docs]'
```

Pulls in `mkdocs-material[imaging]` and `mkdocstrings[python]`. Build
and serve locally with:

```bash
python3 docs/scripts/generate_docs.py
PYTHONPATH=. mkdocs serve
```

## Requirements

- **NVIDIA GPU** for actually running kernels.
    - `examples/hopper/*` needs `sm_90a` — an H100 or H200.
    - `examples/blackwell/*` needs `sm_100a` — a B200.
- **CUDA driver** (not the full toolkit). pyptx JITs PTX through
  `cuModuleLoadData`; there's no ptxas dependency at install time.
- **Python ≥ 3.10.**

## Building the launch shim

The C++ launch shim handles the `cuLaunchKernel` call for both JAX and
PyTorch. It ships with the package and builds automatically via the
wheel. If you're installing editable (`pip install -e .`), build it
once manually:

```bash
cd pyptx/_shim
bash build.sh
```

This needs `jaxlib` installed (for the FFI header). If you only want
the PyTorch path, the C++ extension gets JIT-compiled on first use
and doesn't need this step — just `pip install ninja`.

## System-Python installs

Debian-derived distros ship a PEP 668 "externally-managed" Python. If
you hit `error: externally-managed-environment`, either use a venv:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install 'pyptx[all]'
```

or explicitly opt out:

```bash
pip install --break-system-packages 'pyptx[all]'
```

A venv is the safer default; `--break-system-packages` is fine for
throwaway dev boxes.
