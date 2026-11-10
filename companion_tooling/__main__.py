import sys

from companion_tooling import generate, l0, normalize_headers, pointers, slice


def main() -> int:
    commands = {
        "check_slice": slice.main,
        "generate": generate.main,
        "normalize_headers": normalize_headers.main,
        "check_pointers": pointers.main,
        "l0": l0.main,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("usage: python -m companion_tooling {" + ",".join(commands) + "} ...")
        return 2
    return commands[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
