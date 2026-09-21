# Validation record

Date: 2026-09-21 UTC.

## Environment actually used

- Mojo `1.1.0 (8189361e)`.
- MAX `26.6.0`.
- CPython `3.12.14`; `uv.lock` resolved and `uv sync --locked` succeeded.
- Linux x86-64 build container; kernel `6.18.44`. This is not the user's Arch laptop.
- No functioning NVIDIA runtime in the container: `libnvidia-ml.so.1` is unavailable.
- GPU compilation target: `sm_89` (RTX 4070 Laptop architecture).

## Results

| Check | Result |
| --- | --- |
| Pinned `mojo format` on all Mojo sources | Passed |
| CPU tests, `-O0 -D ASSERT=all` | 10 passed, 0 failed, 0 skipped |
| CPU tests, `-O3 -D ASSERT=all --fp-mode=contract=off` | 10 passed, 0 failed, 0 skipped |
| CLI tests against compiled GPU-target executable | 16 passed |
| Full tool `sm_89` cross-compilation | Passed |
| GPU test module `sm_89` cross-compilation | Passed, including locked environment |
| Reusable workspace example `sm_89` cross-compilation | Passed |
| Environment helper shell syntax and execution | Passed |
| No-GPU `verify` invocation | Nonzero exit with missing NVIDIA library error; no false PASS |
| GPU correctness execution | UNAVAILABLE |
| GPU benchmarks and speedups | UNAVAILABLE; no performance numbers invented |
| Real RTX 4070 / Arch integration | NOT RUN |

The CPU suite covers zero/exact/overflow shapes; byte and work budgets; numeric and
CLI parsing; duplicate/unknown/missing options; finite-value policy; ReLU values,
extremes, NaN/infinities; corruption detection; and bounded statistics. CLI checks
exercise argument admission and `plan` without executing nonempty GPU workloads.

GPU tests are supplied, compiled, and pending execution. They cover 17 boundary
shapes across all five candidates, user input loading, rejected-mutation preservation,
invalid candidate rejection, guard corruption, and an empty workspace. Compilation
is not evidence that memory access, synchronization, driver behavior, or numeric
results are correct at runtime.

## Reproduction

From the extracted project root after `uv sync --locked --python 3.12`:

```sh
mkdir -p build
uv run --locked mojo build -O0 -D ASSERT=all -I . \
    tests/test_cpu.mojo -o build/test-cpu-debug
./build/test-cpu-debug
uv run --locked mojo run -O3 -D ASSERT=all --fp-mode=contract=off \
    -I . tests/test_cpu.mojo
uv run --locked mojo build -O3 -D ASSERT=all --fp-mode=contract=off \
    --target-accelerator sm_89 -I . tiger_kernel.mojo -o build/tiger-kernel
python3 tests/test_cli.py ./build/tiger-kernel
uv run --locked mojo build -O3 -D ASSERT=all --fp-mode=contract=off \
    --target-accelerator sm_89 -I . tests/test_gpu.mojo -o build/test-gpu
uv run --locked mojo build -O3 -D ASSERT=all --fp-mode=contract=off \
    --target-accelerator sm_89 -I . examples/reuse.mojo -o build/reuse
```

On the actual supported GPU host, complete the pending checks:

```sh
./build/test-gpu
./build/reuse
./build/tiger-kernel verify --device 0
./build/tiger-kernel tune --rows 1024 --cols 4096 --device 0
```

Retain complete stdout/stderr and exit codes. Trust tuning output only when the
process exits successfully and contains the final PASS marker. Rerun after changes
to compiler, runtime, driver, kernel, GPU, shape, operation, or numeric policy.
