# `pyptx.types`

> This page is generated from source docstrings and public symbols.

PTX scalar type descriptors.

The public API of this module is the set of singleton :class:`PtxType`
instances such as ``u32``, ``bf16``, ``f32``, and ``pred``.

These type objects are used throughout the DSL:

```python
from pyptx.types import bf16, f32, u32, pred

acc = reg.array(f32, 64)
sA = smem.alloc(bf16, (STAGES, BM, BK))
tid = reg.from_(ptx.special.tid.x(), u32)
p = reg.scalar(pred)
```

The type singletons are intentionally lightweight. They mostly serve as
an explicit bridge between Python code and PTX type spelling.

## Public API

- [`PtxType`](#ptxtype)
- [`b8`](#b8)
- [`b16`](#b16)
- [`b32`](#b32)
- [`b64`](#b64)
- [`b128`](#b128)
- [`u8`](#u8)
- [`u16`](#u16)
- [`u32`](#u32)
- [`u64`](#u64)
- [`s8`](#s8)
- [`s16`](#s16)
- [`s32`](#s32)
- [`s64`](#s64)
- [`f16`](#f16)
- [`f16x2`](#f16x2)
- [`bf16`](#bf16)
- [`bf16x2`](#bf16x2)
- [`tf32`](#tf32)
- [`f32`](#f32)
- [`f64`](#f64)
- [`e4m3`](#e4m3)
- [`e5m2`](#e5m2)
- [`pred`](#pred)
- [`from_name`](#from-name)

<a id="ptxtype"></a>

## `PtxType`

- Kind: `class`

```python
class PtxType(name: 'str', bits: 'int') -> 'None'
```

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

- Kind: `attribute`

- Value: `<member 'bits' of 'PtxType' objects>`

No docstring yet.

#### `name`

- Kind: `attribute`

- Value: `<member 'name' of 'PtxType' objects>`

No docstring yet.

<a id="b8"></a>

## `b8`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

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

#### `name`

- Kind: `attribute`

- Value: `'b8'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="b16"></a>

## `b16`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

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

#### `name`

- Kind: `attribute`

- Value: `'b16'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="b32"></a>

## `b32`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

- Kind: `attribute`

- Value: `32`

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

#### `name`

- Kind: `attribute`

- Value: `'b32'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="b64"></a>

## `b64`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

- Kind: `attribute`

- Value: `64`

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

#### `name`

- Kind: `attribute`

- Value: `'b64'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="b128"></a>

## `b128`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

- Kind: `attribute`

- Value: `128`

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

#### `name`

- Kind: `attribute`

- Value: `'b128'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="u8"></a>

## `u8`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

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

#### `name`

- Kind: `attribute`

- Value: `'u8'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="u16"></a>

## `u16`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

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

#### `name`

- Kind: `attribute`

- Value: `'u16'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="u32"></a>

## `u32`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

- Kind: `attribute`

- Value: `32`

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

#### `name`

- Kind: `attribute`

- Value: `'u32'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="u64"></a>

## `u64`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

- Kind: `attribute`

- Value: `64`

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

#### `name`

- Kind: `attribute`

- Value: `'u64'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="s8"></a>

## `s8`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

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

#### `name`

- Kind: `attribute`

- Value: `'s8'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="s16"></a>

## `s16`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

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

#### `name`

- Kind: `attribute`

- Value: `'s16'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="s32"></a>

## `s32`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

- Kind: `attribute`

- Value: `32`

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

#### `name`

- Kind: `attribute`

- Value: `'s32'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="s64"></a>

## `s64`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

- Kind: `attribute`

- Value: `64`

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

#### `name`

- Kind: `attribute`

- Value: `'s64'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="f16"></a>

## `f16`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

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

#### `name`

- Kind: `attribute`

- Value: `'f16'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="f16x2"></a>

## `f16x2`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

- Kind: `attribute`

- Value: `32`

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

#### `name`

- Kind: `attribute`

- Value: `'f16x2'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="bf16"></a>

## `bf16`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

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

#### `name`

- Kind: `attribute`

- Value: `'bf16'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="bf16x2"></a>

## `bf16x2`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

- Kind: `attribute`

- Value: `32`

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

#### `name`

- Kind: `attribute`

- Value: `'bf16x2'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="tf32"></a>

## `tf32`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

- Kind: `attribute`

- Value: `32`

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

#### `name`

- Kind: `attribute`

- Value: `'tf32'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="f32"></a>

## `f32`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

- Kind: `attribute`

- Value: `32`

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

#### `name`

- Kind: `attribute`

- Value: `'f32'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="f64"></a>

## `f64`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

- Kind: `attribute`

- Value: `64`

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

#### `name`

- Kind: `attribute`

- Value: `'f64'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="e4m3"></a>

## `e4m3`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

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

#### `name`

- Kind: `attribute`

- Value: `'e4m3'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="e5m2"></a>

## `e5m2`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

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

#### `name`

- Kind: `attribute`

- Value: `'e5m2'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="pred"></a>

## `pred`

- Kind: `namespace`

- Type: `PtxType`

A PTX scalar type.

Singleton instances (bf16, f32, etc.) are the public API.

### Members

#### `ptx`

- Kind: `property`

PTX text form with leading dot: '.f32'.

#### `bits`

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

#### `name`

- Kind: `attribute`

- Value: `'pred'`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

<a id="from-name"></a>

## `from_name`

- Kind: `function`

```python
from_name(name: 'str') -> 'PtxType'
```

Look up a PtxType by name (with or without leading dot).
