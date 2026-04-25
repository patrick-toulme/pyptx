"""Disk cache for compiled PTX → cubin.

Each cubin is stored at:
    ~/.cache/pyptx/<cache_key_hash>.cubin

Plus a metadata sidecar:
    ~/.cache/pyptx/<cache_key_hash>.json

The cache key is derived from (fn_id, template_kwargs, input_avals, arch).
PTX source is also cached so users can inspect what was compiled without
needing the original Python function.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CACHE_DIR = Path(os.environ.get("PYPTX_CACHE_DIR", "")) \
    if os.environ.get("PYPTX_CACHE_DIR") \
    else Path.home() / ".cache" / "pyptx"


@dataclass(frozen=True)
class CacheKey:
    """Identifies a unique kernel specialization.

    fn_id: a stable identifier for the function (typically fn.__qualname__)
    template_kwargs: sorted tuple of template parameter values
    input_shapes: tuple of (shape_tuple, dtype_name) per input
    arch: target architecture string
    """

    fn_id: str
    template_kwargs: tuple[tuple[str, Any], ...]
    input_shapes: tuple[tuple[tuple[int, ...], str], ...]
    arch: str

    def hash(self) -> str:
        """Return a stable SHA-256 hex digest of this cache key."""
        blob = json.dumps(
            {
                "fn_id": self.fn_id,
                "template_kwargs": [list(kv) for kv in self.template_kwargs],
                "input_shapes": [
                    [list(shape), dtype] for shape, dtype in self.input_shapes
                ],
                "arch": self.arch,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:32]


@dataclass
class CacheEntry:
    """A single cached compilation result."""

    key: CacheKey
    ptx_source: str
    cubin_bytes: bytes | None  # None means "PTX only, cubin compiled on demand"
    metadata: dict[str, Any] = field(default_factory=dict)


class CubinCache:
    """Process-local in-memory cache with optional disk persistence.

    Thread-safe. Disk-backed by default under ~/.cache/pyptx/.
    """

    def __init__(self, cache_dir: Path | None = None, enable_disk: bool = True) -> None:
        self._mem: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._enable_disk = enable_disk
        if enable_disk:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: CacheKey) -> CacheEntry | None:
        """Look up a cache entry. Checks memory first, then disk."""
        digest = key.hash()
        with self._lock:
            entry = self._mem.get(digest)
            if entry is not None:
                return entry
            if self._enable_disk:
                entry = self._load_from_disk(digest)
                if entry is not None:
                    self._mem[digest] = entry
                    return entry
        return None

    def put(self, entry: CacheEntry) -> None:
        """Store a cache entry in memory and (optionally) on disk."""
        digest = entry.key.hash()
        with self._lock:
            self._mem[digest] = entry
            if self._enable_disk:
                self._save_to_disk(digest, entry)

    def clear(self, *, disk: bool = False) -> None:
        """Drop the in-memory cache. Optionally wipe disk too."""
        with self._lock:
            self._mem.clear()
            if disk and self._enable_disk and self._cache_dir.exists():
                for f in self._cache_dir.iterdir():
                    if f.suffix in (".cubin", ".ptx", ".json"):
                        f.unlink(missing_ok=True)

    def __len__(self) -> int:
        return len(self._mem)

    def __contains__(self, key: CacheKey) -> bool:
        return self.get(key) is not None

    # -- disk I/O ----------------------------------------------------------

    def _load_from_disk(self, digest: str) -> CacheEntry | None:
        meta_path = self._cache_dir / f"{digest}.json"
        ptx_path = self._cache_dir / f"{digest}.ptx"
        cubin_path = self._cache_dir / f"{digest}.cubin"
        if not meta_path.exists() or not ptx_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
            ptx_source = ptx_path.read_text()
            cubin_bytes: bytes | None = None
            if cubin_path.exists():
                cubin_bytes = cubin_path.read_bytes()
            key = CacheKey(
                fn_id=meta["key"]["fn_id"],
                template_kwargs=tuple(
                    tuple(kv) for kv in meta["key"]["template_kwargs"]
                ),
                input_shapes=tuple(
                    (tuple(shape), dtype)
                    for shape, dtype in meta["key"]["input_shapes"]
                ),
                arch=meta["key"]["arch"],
            )
            return CacheEntry(
                key=key,
                ptx_source=ptx_source,
                cubin_bytes=cubin_bytes,
                metadata=meta.get("metadata", {}),
            )
        except (OSError, KeyError, json.JSONDecodeError):
            return None

    def _save_to_disk(self, digest: str, entry: CacheEntry) -> None:
        meta_path = self._cache_dir / f"{digest}.json"
        ptx_path = self._cache_dir / f"{digest}.ptx"
        cubin_path = self._cache_dir / f"{digest}.cubin"
        try:
            ptx_path.write_text(entry.ptx_source)
            if entry.cubin_bytes is not None:
                cubin_path.write_bytes(entry.cubin_bytes)
            meta_path.write_text(json.dumps({
                "key": {
                    "fn_id": entry.key.fn_id,
                    "template_kwargs": [list(kv) for kv in entry.key.template_kwargs],
                    "input_shapes": [
                        [list(s), d] for s, d in entry.key.input_shapes
                    ],
                    "arch": entry.key.arch,
                },
                "metadata": entry.metadata,
            }, default=str))
        except OSError:
            pass  # disk full / permissions — continue with memory cache


# Process-global cache singleton
_GLOBAL_CACHE: CubinCache | None = None


def get_global_cache() -> CubinCache:
    """Return the process-wide cubin cache."""
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = CubinCache()
    return _GLOBAL_CACHE


def set_global_cache(cache: CubinCache) -> None:
    """Override the global cache (primarily for tests)."""
    global _GLOBAL_CACHE
    _GLOBAL_CACHE = cache
