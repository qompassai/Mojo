# Tiger Kernel

A bounded Mojo tool for verifying and tuning **GPU compute kernels for AI operators**.
The first operator is a fused FP32 bias + residual + ReLU epilogue:

```text
y[row, col] = max((x[row, col] + bias[col]) + residual[row, col], 0)
```

It compares one fused GPU pass with a two-pass GPU baseline, checks every output
against a scalar CPU reference, and reports the fastest measured launch configuration.
Use it for workloads that already contain this exact operation, such as suitable MLP
inference epilogues. ReLU is not interchangeable with GELU or SiLU.

This is a compute-kernel tool. It does not build the Linux kernel, tune sysctls,
change NVIDIA clocks, alter security mitigations, or automatically modify a model.
It is not a GEMM, attention, quantization, or full-model inference optimizer.

**Validation:** Mojo 1.1.0 CPU tests and CLI tests passed; GPU application, GPU tests,
and integration example cross-compiled for `sm_89`. GPU execution and speedups have
not been verified in the build environment. See [VALIDATION.md](VALIDATION.md).

## Requirements and pinned environment

- Linux x86-64; your Arch laptop is the intended deployment host.
- Mojo **1.1.0** and MAX **26.6.0**, pinned together in `pyproject.toml` and `uv.lock`.
- Python **3.12** for the project environment; the computation itself uses no Python interop.
- A supported GPU and functioning userspace driver. Your RTX 4070 Laptop is an Ada
  device (`sm_89`), but successful cross-compilation does not verify its runtime.
- `uv` installed through a source you have reviewed. Do not use root for this project.

Modular lists RTX 40-series GPUs as known compatible for development and documents
its Linux/driver requirements in [MAX Packages](https://max.modular.com/packages/).
Arch is not its continuously tested Ubuntu environment. Keep the working driver/kernel
configuration while testing this user-space tool.

From the extracted `tiger-kernel` directory, these commands work in Bash and Fish:

```sh
uv sync --locked --python 3.12
uv run --locked mojo --version
uv run --locked mojo run -I . -D ASSERT=all tests/test_cpu.mojo
mkdir -p build
uv run --locked mojo build -O3 -D ASSERT=all --fp-mode=contract=off \
    -I . tiger_kernel.mojo -o build/tiger-kernel
./build/tiger-kernel --help
python3 tests/test_cli.py ./build/tiger-kernel
```

A normal build discovers the local accelerator target. If building on a machine
without the RTX 4070 present, add `--target-accelerator sm_89` to **build** commands.
This selects the compile target; it does not supply a GPU driver or emulate execution.
Rebuild for a different architecture. The host CPU target also defaults to the build
machine; do not assume a compiled executable is portable to older CPUs.

Dependency installation is explicit. `uv.lock` pins resolved dependencies and artifact
hashes; use `--locked`. When changing Mojo/MAX, review the pair, update the lockfile,
format, compile, run both test suites, and rerun measurements as one change.

## Verify before tuning

```sh
./build/tiger-kernel plan --rows 1024 --cols 4096
./build/tiger-kernel verify --device 0
uv run --locked mojo run -I . -D ASSERT=all --fp-mode=contract=off tests/test_gpu.mojo
```

The CLI `verify` checks 17 shapes against all five candidates, covering tails around
32, 64, 128, 256, and 512 threads plus a multi-row shape with an irregular bias width.
The separate GPU test module adds user-supplied inputs, cancellation-sensitive numeric
values, nonfinite rejection, rejected-mutation preservation, invalid candidate rejection,
intentional guard corruption, and empty-workspace rejection. Its device defaults to 0.

No device is opened by `--help` or `plan`. `verify` accepts only `--device`; its shapes
are fixed. A missing GPU or failing runtime is an error, not a silently passing test.
The ten CPU test groups require no accelerator execution.

## Tune representative shapes

```sh
./build/tiger-kernel tune --rows 1024 --cols 4096 --device 0
./build/tiger-kernel tune --rows 1 --cols 4096 --samples 21 --iterations 50
./build/tiger-kernel tune --rows 256 --cols 4096 --samples 15 --iterations 20
```

The selected device ID is relative to the accelerator runtime's visible devices.
The tool prints the context's device name; verify that it is the intended NVIDIA GPU,
especially with GPU visibility variables or multiple accelerators. Desktop rendering
on the Intel GPU does not determine which compute device this tool uses.

| Candidate | Implementation | Threads/block |
| --- | --- | ---: |
| 0 | Bias pass, then residual + ReLU pass | 256 |
| 1 | Fused bias + residual + ReLU | 64 |
| 2 | Fused bias + residual + ReLU | 128 |
| 3 | Fused bias + residual + ReLU | 256 |
| 4 | Fused bias + residual + ReLU | 512 |

A nonempty tune run:

1. Validates dimensions, aggregate array bytes, sample limits, and work budget.
2. Allocates reusable buffers and populates deterministic mixed-sign inputs.
3. Warms and fully verifies each candidate, excluding first-use work from samples.
4. Rotates candidate order across samples, collecting completed GPU batches.
5. Reports median, nearest-rank p95, minimum, maximum, and baseline/median ratio.
6. Rechecks every candidate and its output guard after measurement.
7. Separately measures upload + compute + download for the fastest candidate.
8. Publishes `# status: PASS` and the winner only after those checks succeed.

The two-pass baseline is eligible to win. A ratio above 1 means the candidate's median
was lower than the baseline's median in this run; a small difference can be noise.
No confidence interval or universal optimum is claimed. Repeat representative shapes
under comparable temperatures, power settings, and competing GPU load.

**Timing meaning:** a monotonic host clock brackets launches and stream synchronization.
The table measures **batch-mean wall time per operation on resident buffers**, including
submission overhead, with transfers and allocations excluded. It is not CUDA-event-only
kernel duration. The separate roundtrip includes three input copies, the operation,
output copy (including the guard), and synchronization, using existing allocations.
It excludes initial allocation, input construction, validation, and model/framework work.
The reported p95 is over batch means, not individual request latency. With nine samples,
nearest-rank p95 is the maximum sample. Small kernels may be dominated by submission costs.

Results describe this operator, these shapes, and these inputs. They do not establish
end-to-end model speedup or superiority to fused PyTorch/MAX/cuBLAS implementations.

## Resource and numerical contracts

| Limit | Policy |
| --- | --- |
| Rows | 0..16,384; empty tune runs perform no allocations/launches |
| Columns | 1..16,384 |
| Elements | At most 16,777,216; checked before multiplying dimensions |
| Samples | 3..51 |
| Iterations/sample | 1..100 |
| Device ID | 0..31, then actual runtime validation |
| Array memory budget | 1..1,024 MiB; default 512 MiB |
| Kernel element visits | At most 34,359,738,368, conservatively counted across the run |
| CLI | At most six option/value pairs; no duplicates or unknown options |
| Input values | Finite FP32, absolute value <= 1,000,000 |
| Comparison | `abs(actual-expected) <= 1e-5 + 1e-5*abs(expected)` |
| Guard | 512 trailing FP32 elements; verified unchanged |

For `N = rows*columns` and `C = columns`, the owned array budget is:

```text
device: (4*N + C + 512) * 4 bytes
host:   (3*N + C + 512) * 4 bytes
total:  (7*N + 2*C + 1024) * 4 bytes
```

This is an **array allocation budget**, not a process-RSS or VRAM hard limit. Driver
contexts, runtime caches, executable code, list metadata, and allocator overhead are
additional. Caller-owned input lists are also outside this budget. Allocation failures
propagate. Array storage and work limits are checked independently of removable assertions.

The addition order is explicit and contraction is disabled in documented builds.
NaN/infinity inputs are rejected by the reusable loader. Finite bounds keep intermediate
sums finite. ReLU returns positive zero for nonpositive results; signed-zero preservation
is not promised. Cross-device subnormal behavior is not required to be bit-identical;
the stated tolerance applies. Synthetic inputs are exact binary fractions; the GPU tests
also include non-exact decimal fractions and cancellation-sensitive values.

Each thread writes a disjoint element after a tail check. Kernels have no reductions,
barriers, atomics, shared memory, or allocator calls. Raw pointer operations are isolated
in `kernels.mojo`, with a documented extent/aliasing/lifetime contract. Host code uses
owning MAX buffers and never dereferences device pointers.

All buffers share one context/stream. Normal completion is explicit; the owner remains
live until completion. MAX's buffer destructor schedules frees on its associated stream.
If submission, transfer, or synchronization fails, let the error propagate and discard
the workspace. Partial device state is not reported as a successful transaction.

There is no hard GPU execution deadline or in-kernel cancellation. Bounds limit admitted
work but cannot recover a hung driver. Ctrl-C terminates the foreground tool; do not treat
that as proof of immediate GPU cancellation. For a supervised run, an outer process timeout
can bound host waiting, but it cannot guarantee recovery of a stuck device.

## Reuse from your Mojo workflow

`FusionWorkspace` owns a fixed-shape allocation and exposes:

- `load(x, bias, residual)`: validate lengths and all values before changing host data;
  then upload. Rejected inputs preserve prior input data. Transfer failures may be partial.
- `execute(candidate)`: validate candidate, reset the output guard, enqueue, and download;
  results are ready on return.
- `verify()`: check `host_output` against the retained host inputs and trailing guard.

Read only the first `count` elements of `host_output`; the remainder is the guard.
Treat the shape and buffer fields as implementation-owned; do not resize, rebind, or
alias them. The workspace is single-owner and must not be used concurrently.

See the complete [examples/reuse.mojo](examples/reuse.mojo):

```sh
uv run --locked mojo run -I . -D ASSERT=all --fp-mode=contract=off examples/reuse.mojo
```

The example intentionally uses native Mojo lists and copies. It does not claim zero-copy
PyTorch interoperability. For production integration, profile the existing model first,
match the exact operation/dtype/layout, and measure the whole graph after integrating a
custom operator. A CPU↔GPU roundtrip can erase a kernel-level improvement.

## Keep reports and environment evidence

The tool writes reports to stdout and does not auto-load project configuration or persist
winner selections. Check its exit status: output from a failed run can contain incomplete
candidate rows and must not be treated as a valid recommendation.

Bash:

```bash
state_dir="${XDG_STATE_HOME:-$HOME/.local/state}/tiger-kernel"
install -d -m 700 "$state_dir"
(umask 077; ./scripts/record_environment.sh > "$state_dir/environment.txt")
(umask 077; ./build/tiger-kernel tune > "$state_dir/tune.tsv")
```

Fish:

```fish
set -l state_base "$HOME/.local/state"
if set -q XDG_STATE_HOME; and test -n "$XDG_STATE_HOME"
    set state_base "$XDG_STATE_HOME"
end
set -l state_dir "$state_base/tiger-kernel"
install -d -m 700 "$state_dir"
# Run a child shell so restrictive umask does not change your interactive shell.
sh -c 'umask 077; ./scripts/record_environment.sh > "$1"' sh "$state_dir/environment.txt"
sh -c 'umask 077; ./build/tiger-kernel tune > "$1"' sh "$state_dir/tune.tsv"
```

These commands overwrite the named reports; choose new filenames to retain older runs.
Use your own trusted state directory. The read-only environment helper records compiler,
source/lock hashes, kernel/OS, device listing, and NVIDIA kernel module version when
available. Keep actual build flags, power mode, visibility settings, and workload context
alongside results. It deliberately does not dump your environment or model data.

## Source layout and maintenance

| File | Responsibility |
| --- | --- |
| `tiger_kernel.mojo` | CLI admission and dispatch |
| `contracts.mojo` | Shape, byte, work, and input-argument limits |
| `reference.mojo` | Scalar contract, fixtures, result comparison |
| `kernels.mojo` | Three device entry points and unsafe boundary |
| `workspace.mojo` | Owning buffers, checked input loading, dispatch, verification |
| `statistics.mojo` | Bounded sample statistics |
| `runner.mojo` | Warmup, correctness gating, measurement, reporting |
| `tests/` | CPU, CLI, and device tests |
| `examples/reuse.mojo` | Reusable-buffer integration example |
| `scripts/record_environment.sh` | Read-only evidence capture |

Formatting follows the pinned formatter:

```sh
uv run --locked mojo format *.mojo tests/*.mojo examples/*.mojo
```

For CI, format a disposable checkout and use `git diff --exit-code`. Record GPU tests
as unavailable when a GPU is absent; do not replace their results with CPU-test success.
No dependency installation, benchmark, or GPU access should run merely from opening
these files in Neovim. Use an explicit build/run action in a trusted workspace.

## Reference APIs

- [Mojo ownership](https://mojolang.org/docs/manual/values/ownership/)
- [Mojo explicit lifetime extension](https://mojolang.org/docs/manual/lifecycle/death/)
- [Mojo pointer contracts](https://mojolang.org/docs/std/memory/pointer/Pointer/)
- [MAX DeviceContext](https://max.modular.com/api/mojo/max/gpu/host/device_context/DeviceContext/)
- [MAX DeviceBuffer](https://max.modular.com/api/mojo/max/gpu/host/device_context/DeviceBuffer/)
- [Mojo timing functions](https://mojolang.org/docs/std/time/time/)

License: Apache-2.0; see [LICENSE](LICENSE).
