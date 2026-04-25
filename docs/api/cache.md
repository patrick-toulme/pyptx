# `pyptx.cache`

> This page is generated from source docstrings and public symbols.

Disk cache for compiled PTX → cubin.

Each cubin is stored at:
    ~/.cache/pyptx/<cache_key_hash>.cubin

Plus a metadata sidecar:
    ~/.cache/pyptx/<cache_key_hash>.json

The cache key is derived from (fn_id, template_kwargs, input_avals, arch).
PTX source is also cached so users can inspect what was compiled without
needing the original Python function.

## Public API

- [`DEFAULT_CACHE_DIR`](#default-cache-dir)
- [`CacheKey`](#cachekey)
- [`CacheEntry`](#cacheentry)
- [`CubinCache`](#cubincache)
- [`get_global_cache`](#get-global-cache)
- [`set_global_cache`](#set-global-cache)

<a id="default-cache-dir"></a>

## `DEFAULT_CACHE_DIR`

- Kind: `namespace`

- Type: `PosixPath`

Path subclass for non-Windows systems.

On a POSIX system, instantiating a Path should return this object.

<a id="cachekey"></a>

## `CacheKey`

- Kind: `class`

```python
class CacheKey(fn_id: 'str', template_kwargs: 'tuple[tuple[str, Any], ...]', input_shapes: 'tuple[tuple[tuple[int, ...], str], ...]', arch: 'str') -> None
```

Identifies a unique kernel specialization.

fn_id: a stable identifier for the function (typically fn.__qualname__)
template_kwargs: sorted tuple of template parameter values
input_shapes: tuple of (shape_tuple, dtype_name) per input
arch: target architecture string

### Members

#### `hash() -> 'str'`

- Kind: `method`

Return a stable SHA-256 hex digest of this cache key.

<a id="cacheentry"></a>

## `CacheEntry`

- Kind: `class`

```python
class CacheEntry(key: 'CacheKey', ptx_source: 'str', cubin_bytes: 'bytes | None', metadata: 'dict[str, Any]' = <factory>) -> None
```

A single cached compilation result.

<a id="cubincache"></a>

## `CubinCache`

- Kind: `class`

```python
class CubinCache(cache_dir: 'Path | None' = None, enable_disk: 'bool' = True) -> 'None'
```

Process-local in-memory cache with optional disk persistence.

Thread-safe. Disk-backed by default under ~/.cache/pyptx/.

### Members

#### `get(key: 'CacheKey') -> 'CacheEntry | None'`

- Kind: `method`

Look up a cache entry. Checks memory first, then disk.

#### `put(entry: 'CacheEntry') -> 'None'`

- Kind: `method`

Store a cache entry in memory and (optionally) on disk.

#### `clear(*, disk: 'bool' = False) -> 'None'`

- Kind: `method`

Drop the in-memory cache. Optionally wipe disk too.

<a id="get-global-cache"></a>

## `get_global_cache`

- Kind: `function`

```python
get_global_cache() -> 'CubinCache'
```

Return the process-wide cubin cache.

<a id="set-global-cache"></a>

## `set_global_cache`

- Kind: `function`

```python
set_global_cache(cache: 'CubinCache') -> 'None'
```

Override the global cache (primarily for tests).
