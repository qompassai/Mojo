#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise a supplied compiled executable without running a nonempty GPU workload."""
from pathlib import Path
import subprocess
import sys


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tests/test_cli.py /path/to/tiger-kernel")
    executable = Path(sys.argv[1]).resolve(strict=True)
    cases = [
        ([], True, "Usage:"),
        (["--help"], True, "Usage:"),
        (["plan", "--rows", "3", "--cols", "257"], True, "elements: 771"),
        (["plan", "--rows", "0"], True, "elements: 0"),
        (["unknown"], False, "unknown command"),
        (["--help", "extra"], False, "help accepts"),
        (["plan", "--rows"], False, "option value"),
        (["plan", "--rows", "-1"], False, "decimal digits"),
        (["plan", "--cols", "0"], False, "columns must"),
        (["plan", "--rows", "1", "--rows", "2"], False, "duplicate"),
        (["plan", "--unknown", "1"], False, "unknown option"),
        (["plan", "--samples", "52"], False, "samples must"),
        (["plan", "--memory-mib", "1"], False, "array budget"),
        (["verify", "--rows", "2"], False, "accepts only --device"),
        (["plan", "--rows", "9" * 33], False, "32 bytes"),
        (["plan", "--rows", "9999999999"], False, "1..9 decimal digits"),
    ]
    for arguments, success, expected in cases:
        result = subprocess.run(
            [str(executable), *arguments], capture_output=True, text=True,
            timeout=10, check=False,
        )
        output = result.stdout + result.stderr
        if (result.returncode == 0) != success or expected not in output:
            raise SystemExit(f"FAIL {arguments!r}: exit={result.returncode}\n{output}")
    print(f"PASS: {len(cases)} CLI cases; no nonempty GPU workload launched")


if __name__ == "__main__":
    main()
