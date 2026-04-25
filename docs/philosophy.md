# Philosophy

`pyptx` is intentionally not a “hide the GPU” project.

The core idea is simple:

- Python is the authoring language
- PTX is the execution model
- the DSL should make common patterns easier without erasing what the hardware is doing

## The Design Bias

The library is biased toward:

- explicit instructions over opaque compiler magic
- readable low-level kernels over giant metaprogramming systems
- direct interoperability with existing Python runtimes
- making Hopper-era features like WGMMA and TMA practical to use from Python
- making Blackwell-era features like `tcgen05.mma`, TMEM, and
  `cta_group::2` cooperative MMA writable directly, not buried under an
  abstraction that loses the one instruction you need

That means `pyptx` is comfortable exposing things that many DSLs try to hide:

- registers
- predicates
- shared-memory address math
- barriers
- cluster launch structure
- PTX instruction spelling

## The Goal

The goal is not “replace CUDA.”

The goal is to make a class of work dramatically easier:

- prototyping low-level kernels
- translating or studying PTX
- building custom kernels that stay close to the machine model
- integrating those kernels into JAX and PyTorch workflows

## Where Sugar Helps

The right kind of sugar in `pyptx` removes repetition in places that are mechanical and bug-prone:

- parameter loading
- stage cursor logic
- barrier arrays
- descriptor assembly
- common epilogues

The wrong kind of sugar hides the instruction shape or makes PTX harder to predict.

That tradeoff is central to the project. The DSL should compress boilerplate, not blur the hardware model.
