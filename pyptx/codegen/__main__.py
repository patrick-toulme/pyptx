"""CLI entry point: python -m pyptx.codegen kernel.ptx [--sugar] [--name NAME]"""

import sys

from pyptx.codegen.codegen import ptx_to_python


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = [a for a in sys.argv[1:] if a.startswith("-")]

    if not args:
        print("Usage: python -m pyptx.codegen <file.ptx> [--sugar] [--name NAME]",
              file=sys.stderr)
        sys.exit(1)

    path = args[0]
    sugar = "--sugar" in flags
    kernel_name = "matmul_kernel"
    for i, f in enumerate(flags):
        if f == "--name" and i + 1 < len(flags):
            kernel_name = flags[i + 1]

    with open(path) as f:
        source = f.read()

    print(ptx_to_python(source, sugar=sugar, kernel_name=kernel_name))


if __name__ == "__main__":
    main()
