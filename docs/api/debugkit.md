# `pyptx.debugkit`

> This page is generated from source docstrings and public symbols.

Kernel debugging utilities: beacon waits and phase cycle counters.

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

## Public API

- [`BEACON_MAGIC`](#beacon-magic)
- [`SLOT_WORDS`](#slot-words)
- [`DEFAULT_TIMEOUT_TRIES`](#default-timeout-tries)
- [`DebugKit`](#debugkit)
- [`n_slot_words`](#n-slot-words)
- [`WAITMAP_MAX_SITES`](#waitmap-max-sites)
- [`WaitMap`](#waitmap)

<a id="beacon-magic"></a>

## `BEACON_MAGIC`

- Kind: `namespace`

- Type: `int`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

### Members

#### `conjugate`

- Kind: `attribute`

- Value: `<built-in method conjugate of int object at 0x7f4d78df3770>`

Returns self, the complex conjugate of any int.

#### `bit_length`

- Kind: `attribute`

- Value: `<built-in method bit_length of int object at 0x7f4d78df3770>`

Number of bits necessary to represent self in binary.

>>> bin(37)
'0b100101'
>>> (37).bit_length()
6

#### `bit_count`

- Kind: `attribute`

- Value: `<built-in method bit_count of int object at 0x7f4d78df3770>`

Number of ones in the binary representation of the absolute value of self.

Also known as the population count.

>>> bin(13)
'0b1101'
>>> (13).bit_count()
3

#### `to_bytes`

- Kind: `attribute`

- Value: `<built-in method to_bytes of int object at 0x7f4d78df3770>`

Return an array of bytes representing an integer.

length
  Length of bytes object to use.  An OverflowError is raised if the
  integer is not representable with the given number of bytes.  Default
  is length 1.
byteorder
  The byte order used to represent the integer.  If byteorder is 'big',
  the most significant byte is at the beginning of the byte array.  If
  byteorder is 'little', the most significant byte is at the end of the
  byte array.  To request the native byte order of the host system, use
  `sys.byteorder' as the byte order value.  Default is to use 'big'.
signed
  Determines whether two's complement is used to represent the integer.
  If signed is False and a negative integer is given, an OverflowError
  is raised.

#### `from_bytes`

- Kind: `attribute`

- Value: `<built-in method from_bytes of type object at 0x7f4d79b30840>`

Return the integer represented by the given array of bytes.

bytes
  Holds the array of bytes to convert.  The argument must either
  support the buffer protocol or be an iterable object producing bytes.
  Bytes and bytearray are examples of built-in objects that support the
  buffer protocol.
byteorder
  The byte order used to represent the integer.  If byteorder is 'big',
  the most significant byte is at the beginning of the byte array.  If
  byteorder is 'little', the most significant byte is at the end of the
  byte array.  To request the native byte order of the host system, use
  `sys.byteorder' as the byte order value.  Default is to use 'big'.
signed
  Indicates whether two's complement is used to represent the integer.

#### `as_integer_ratio`

- Kind: `attribute`

- Value: `<built-in method as_integer_ratio of int object at 0x7f4d78df3770>`

Return a pair of integers, whose ratio is equal to the original int.

The ratio is in lowest terms and has a positive denominator.

>>> (10).as_integer_ratio()
(10, 1)
>>> (-10).as_integer_ratio()
(-10, 1)
>>> (0).as_integer_ratio()
(0, 1)

#### `is_integer`

- Kind: `attribute`

- Value: `<built-in method is_integer of int object at 0x7f4d78df3770>`

Returns True. Exists for duck type compatibility with float.is_integer.

#### `real`

- Kind: `attribute`

- Value: `48812`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

#### `imag`

- Kind: `attribute`

- Value: `0`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

#### `numerator`

- Kind: `attribute`

- Value: `48812`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

#### `denominator`

- Kind: `attribute`

- Value: `1`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

<a id="slot-words"></a>

## `SLOT_WORDS`

- Kind: `namespace`

- Type: `int`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

### Members

#### `conjugate`

- Kind: `attribute`

- Value: `<built-in method conjugate of int object at 0x7f4d79c1d3c8>`

Returns self, the complex conjugate of any int.

#### `bit_length`

- Kind: `attribute`

- Value: `<built-in method bit_length of int object at 0x7f4d79c1d3c8>`

Number of bits necessary to represent self in binary.

>>> bin(37)
'0b100101'
>>> (37).bit_length()
6

#### `bit_count`

- Kind: `attribute`

- Value: `<built-in method bit_count of int object at 0x7f4d79c1d3c8>`

Number of ones in the binary representation of the absolute value of self.

Also known as the population count.

>>> bin(13)
'0b1101'
>>> (13).bit_count()
3

#### `to_bytes`

- Kind: `attribute`

- Value: `<built-in method to_bytes of int object at 0x7f4d79c1d3c8>`

Return an array of bytes representing an integer.

length
  Length of bytes object to use.  An OverflowError is raised if the
  integer is not representable with the given number of bytes.  Default
  is length 1.
byteorder
  The byte order used to represent the integer.  If byteorder is 'big',
  the most significant byte is at the beginning of the byte array.  If
  byteorder is 'little', the most significant byte is at the end of the
  byte array.  To request the native byte order of the host system, use
  `sys.byteorder' as the byte order value.  Default is to use 'big'.
signed
  Determines whether two's complement is used to represent the integer.
  If signed is False and a negative integer is given, an OverflowError
  is raised.

#### `from_bytes`

- Kind: `attribute`

- Value: `<built-in method from_bytes of type object at 0x7f4d79b30840>`

Return the integer represented by the given array of bytes.

bytes
  Holds the array of bytes to convert.  The argument must either
  support the buffer protocol or be an iterable object producing bytes.
  Bytes and bytearray are examples of built-in objects that support the
  buffer protocol.
byteorder
  The byte order used to represent the integer.  If byteorder is 'big',
  the most significant byte is at the beginning of the byte array.  If
  byteorder is 'little', the most significant byte is at the end of the
  byte array.  To request the native byte order of the host system, use
  `sys.byteorder' as the byte order value.  Default is to use 'big'.
signed
  Indicates whether two's complement is used to represent the integer.

#### `as_integer_ratio`

- Kind: `attribute`

- Value: `<built-in method as_integer_ratio of int object at 0x7f4d79c1d3c8>`

Return a pair of integers, whose ratio is equal to the original int.

The ratio is in lowest terms and has a positive denominator.

>>> (10).as_integer_ratio()
(10, 1)
>>> (-10).as_integer_ratio()
(-10, 1)
>>> (0).as_integer_ratio()
(0, 1)

#### `is_integer`

- Kind: `attribute`

- Value: `<built-in method is_integer of int object at 0x7f4d79c1d3c8>`

Returns True. Exists for duck type compatibility with float.is_integer.

#### `real`

- Kind: `attribute`

- Value: `8`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

#### `imag`

- Kind: `attribute`

- Value: `0`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

#### `numerator`

- Kind: `attribute`

- Value: `8`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

#### `denominator`

- Kind: `attribute`

- Value: `1`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

<a id="default-timeout-tries"></a>

## `DEFAULT_TIMEOUT_TRIES`

- Kind: `namespace`

- Type: `int`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

### Members

#### `conjugate`

- Kind: `attribute`

- Value: `<built-in method conjugate of int object at 0x7f4d78df2210>`

Returns self, the complex conjugate of any int.

#### `bit_length`

- Kind: `attribute`

- Value: `<built-in method bit_length of int object at 0x7f4d78df2210>`

Number of bits necessary to represent self in binary.

>>> bin(37)
'0b100101'
>>> (37).bit_length()
6

#### `bit_count`

- Kind: `attribute`

- Value: `<built-in method bit_count of int object at 0x7f4d78df2210>`

Number of ones in the binary representation of the absolute value of self.

Also known as the population count.

>>> bin(13)
'0b1101'
>>> (13).bit_count()
3

#### `to_bytes`

- Kind: `attribute`

- Value: `<built-in method to_bytes of int object at 0x7f4d78df2210>`

Return an array of bytes representing an integer.

length
  Length of bytes object to use.  An OverflowError is raised if the
  integer is not representable with the given number of bytes.  Default
  is length 1.
byteorder
  The byte order used to represent the integer.  If byteorder is 'big',
  the most significant byte is at the beginning of the byte array.  If
  byteorder is 'little', the most significant byte is at the end of the
  byte array.  To request the native byte order of the host system, use
  `sys.byteorder' as the byte order value.  Default is to use 'big'.
signed
  Determines whether two's complement is used to represent the integer.
  If signed is False and a negative integer is given, an OverflowError
  is raised.

#### `from_bytes`

- Kind: `attribute`

- Value: `<built-in method from_bytes of type object at 0x7f4d79b30840>`

Return the integer represented by the given array of bytes.

bytes
  Holds the array of bytes to convert.  The argument must either
  support the buffer protocol or be an iterable object producing bytes.
  Bytes and bytearray are examples of built-in objects that support the
  buffer protocol.
byteorder
  The byte order used to represent the integer.  If byteorder is 'big',
  the most significant byte is at the beginning of the byte array.  If
  byteorder is 'little', the most significant byte is at the end of the
  byte array.  To request the native byte order of the host system, use
  `sys.byteorder' as the byte order value.  Default is to use 'big'.
signed
  Indicates whether two's complement is used to represent the integer.

#### `as_integer_ratio`

- Kind: `attribute`

- Value: `<built-in method as_integer_ratio of int object at 0x7f4d78df2210>`

Return a pair of integers, whose ratio is equal to the original int.

The ratio is in lowest terms and has a positive denominator.

>>> (10).as_integer_ratio()
(10, 1)
>>> (-10).as_integer_ratio()
(-10, 1)
>>> (0).as_integer_ratio()
(0, 1)

#### `is_integer`

- Kind: `attribute`

- Value: `<built-in method is_integer of int object at 0x7f4d78df2210>`

Returns True. Exists for duck type compatibility with float.is_integer.

#### `real`

- Kind: `attribute`

- Value: `3000000`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

#### `imag`

- Kind: `attribute`

- Value: `0`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

#### `numerator`

- Kind: `attribute`

- Value: `3000000`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

#### `denominator`

- Kind: `attribute`

- Value: `1`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

<a id="debugkit"></a>

## `DebugKit`

- Kind: `class`

```python
class DebugKit(enabled: 'bool' = False, timeout_tries: 'int' = 3000000, _sites: 'list[str]' = <factory>, _counters: 'list[str]' = <factory>, _counter_regs: 'dict' = <factory>, _ptr: 'Any' = None, _slot_off: 'Any' = None, _blamed: 'Any' = None, _label_n: 'int' = 0) -> None
```

Trace-time helper that instruments waits and phases in a kernel.

### Members

#### `enabled`

- Kind: `attribute`

- Value: `False`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

#### `timeout_tries`

- Kind: `attribute`

- Value: `3000000`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

#### `attach(dbg_ptr: 'Any', slot_reg: 'Any') -> 'None'`

- Kind: `method`

Bind the debug buffer pointer and this thread's slot index.

``slot_reg`` is a u32 register unique per participating thread
(e.g. ``ctaid.x * threads_per_cta + tid``). Must be called before
any ``wait``/``flush`` when the kit is enabled.

#### `wait(mbar: 'Any', phase: 'Any', *, site: 'str', extra: 'int' = 0) -> 'None'`

- Kind: `method`

mbarrier parity wait; in debug mode, bounded with first-blame
recording and fall-through on timeout.

#### `stamp() -> 'Any'`

- Kind: `method`

Return a u32 register holding %clock (None when disabled).

#### `accumulate(name: 'str', t0: 'Any') -> 'None'`

- Kind: `method`

counter[name] += clock() - t0.

#### `flush() -> 'None'`

- Kind: `method`

Write all counters to this thread's slot (words 1..).

#### `decode(dbg_tensor) -> 'dict'`

- Kind: `method`

Decode a debug tensor produced by this kit.

Returns ``{"beacons": [...], "counters": {slot: {name: cycles}}}``.
Beacons are aggregated by (site, extra) with slot lists.

<a id="n-slot-words"></a>

## `n_slot_words`

- Kind: `function`

```python
n_slot_words(n_slots: 'int') -> 'int'
```

Total b32 words to allocate for a debug buffer of n_slots.

<a id="waitmap-max-sites"></a>

## `WAITMAP_MAX_SITES`

- Kind: `namespace`

- Type: `int`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

### Members

#### `conjugate`

- Kind: `attribute`

- Value: `<built-in method conjugate of int object at 0x7f4d79c1d4c8>`

Returns self, the complex conjugate of any int.

#### `bit_length`

- Kind: `attribute`

- Value: `<built-in method bit_length of int object at 0x7f4d79c1d4c8>`

Number of bits necessary to represent self in binary.

>>> bin(37)
'0b100101'
>>> (37).bit_length()
6

#### `bit_count`

- Kind: `attribute`

- Value: `<built-in method bit_count of int object at 0x7f4d79c1d4c8>`

Number of ones in the binary representation of the absolute value of self.

Also known as the population count.

>>> bin(13)
'0b1101'
>>> (13).bit_count()
3

#### `to_bytes`

- Kind: `attribute`

- Value: `<built-in method to_bytes of int object at 0x7f4d79c1d4c8>`

Return an array of bytes representing an integer.

length
  Length of bytes object to use.  An OverflowError is raised if the
  integer is not representable with the given number of bytes.  Default
  is length 1.
byteorder
  The byte order used to represent the integer.  If byteorder is 'big',
  the most significant byte is at the beginning of the byte array.  If
  byteorder is 'little', the most significant byte is at the end of the
  byte array.  To request the native byte order of the host system, use
  `sys.byteorder' as the byte order value.  Default is to use 'big'.
signed
  Determines whether two's complement is used to represent the integer.
  If signed is False and a negative integer is given, an OverflowError
  is raised.

#### `from_bytes`

- Kind: `attribute`

- Value: `<built-in method from_bytes of type object at 0x7f4d79b30840>`

Return the integer represented by the given array of bytes.

bytes
  Holds the array of bytes to convert.  The argument must either
  support the buffer protocol or be an iterable object producing bytes.
  Bytes and bytearray are examples of built-in objects that support the
  buffer protocol.
byteorder
  The byte order used to represent the integer.  If byteorder is 'big',
  the most significant byte is at the beginning of the byte array.  If
  byteorder is 'little', the most significant byte is at the end of the
  byte array.  To request the native byte order of the host system, use
  `sys.byteorder' as the byte order value.  Default is to use 'big'.
signed
  Indicates whether two's complement is used to represent the integer.

#### `as_integer_ratio`

- Kind: `attribute`

- Value: `<built-in method as_integer_ratio of int object at 0x7f4d79c1d4c8>`

Return a pair of integers, whose ratio is equal to the original int.

The ratio is in lowest terms and has a positive denominator.

>>> (10).as_integer_ratio()
(10, 1)
>>> (-10).as_integer_ratio()
(-10, 1)
>>> (0).as_integer_ratio()
(0, 1)

#### `is_integer`

- Kind: `attribute`

- Value: `<built-in method is_integer of int object at 0x7f4d79c1d4c8>`

Returns True. Exists for duck type compatibility with float.is_integer.

#### `real`

- Kind: `attribute`

- Value: `16`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

#### `imag`

- Kind: `attribute`

- Value: `0`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

#### `numerator`

- Kind: `attribute`

- Value: `16`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

#### `denominator`

- Kind: `attribute`

- Value: `1`

int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating-point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4

<a id="waitmap"></a>

## `WaitMap`

- Kind: `class`

```python
class WaitMap(_sites: 'list' = <factory>, _accs: 'dict' = <factory>, _ptr: 'Any' = None, _row_off: 'Any' = None) -> None
```

Wraps every ``ptx.mbarrier.wait`` and ``ptx.bar.sync`` emitted while
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

### Members

#### `site_names() -> 'list'`

- Kind: `method`

No docstring yet.

#### `attach(dbg_ptr: 'Any', tid_reg: 'Any') -> 'None'`

- Kind: `method`

No docstring yet.

#### `active()`

- Kind: `method`

Context manager: instrument waits emitted inside.

#### `flush(pred: 'Any' = None) -> 'None'`

- Kind: `method`

Write all site counters for this thread (guard with a pred that
selects one CTA to keep the buffer small).

#### `decode(dbg_tensor, n_threads: 'int' = 512, roles: 'dict | None' = None) -> 'dict'`

- Kind: `method`

Aggregate per-site cycles by role. roles: {name: (lo_tid, hi_tid)}.
