#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Run from the project root. Read-only; no root and no automatic installation.
set -eu
printf '%s\n' '# Tiger Kernel environment'
date -u '+UTC: %Y-%m-%dT%H:%M:%SZ'
uname -srmo
uv run --offline --locked mojo --version
printf '%s\n' '# Source and dependency identity'
sha256sum pyproject.toml uv.lock *.mojo tests/*.mojo
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
fi
if [ -r /proc/driver/nvidia/version ]; then
    cat /proc/driver/nvidia/version
fi
printf '%s\n' '# Build contract: -O3 -D ASSERT=all --fp-mode=contract=off'
printf '%s\n' '# Record any extra build target flags and GPU visibility settings alongside results.'
