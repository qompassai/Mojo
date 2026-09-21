# SPDX-License-Identifier: Apache-2.0
"""Correctness-gated, bounded tuning. Timings include submission and synchronization."""
from std.collections import List
from std.time import perf_counter_ns
from contracts import (
    CANDIDATE_COUNT,
    WARMUP_COUNT,
    RunConfig,
    allocation_bytes,
    checked_elements,
    validate_config,
)
from statistics import Summary, summarize
from workspace import FusionWorkspace, block_size


def measure_batch(
    mut work: FusionWorkspace, candidate: Int, iterations: Int
) raises -> Float64:
    work.ctx.synchronize()
    var start = perf_counter_ns()
    for _ in range(iterations):
        work.enqueue(candidate)
    work.ctx.synchronize()
    var elapsed = perf_counter_ns() - start
    if elapsed <= 0:
        raise Error("nonpositive timing interval")
    return Float64(elapsed) / Float64(iterations) / 1000


def measure_roundtrip(
    mut work: FusionWorkspace, candidate: Int, iterations: Int
) raises -> Float64:
    work.ctx.synchronize()
    var start = perf_counter_ns()
    for _ in range(iterations):
        work.upload()
        work.enqueue(candidate)
        work.download()
    var elapsed = perf_counter_ns() - start
    if elapsed <= 0:
        raise Error("nonpositive roundtrip interval")
    return Float64(elapsed) / Float64(iterations) / 1000


def warm_and_verify(mut work: FusionWorkspace) raises:
    for candidate in range(CANDIDATE_COUNT):
        work.reset_output()
        for _ in range(WARMUP_COUNT):
            work.enqueue(candidate)
        work.download()
        _ = work.verify()


def collect_samples(
    mut work: FusionWorkspace, config: RunConfig
) raises -> List[List[Float64]]:
    var samples = List[List[Float64]](capacity=CANDIDATE_COUNT)
    for _ in range(CANDIDATE_COUNT):
        samples.append(List[Float64](capacity=config.samples))
    # Rotate which candidate runs first; no one candidate always gets cold clocks.
    for sample in range(config.samples):
        for offset in range(CANDIDATE_COUNT):
            var candidate = (sample + offset) % CANDIDATE_COUNT
            samples[candidate].append(
                measure_batch(work, candidate, config.iterations)
            )
    return samples^


def print_result(candidate: Int, result: Summary, baseline_us: Float64):
    var kind = String("fused")
    if candidate == 0:
        kind = "two_pass"
    # Candidate has already passed block_size validation in the timed run.
    var block = 256
    if candidate > 0:
        block = 32 << candidate
    print(
        candidate,
        kind,
        block,
        result.median_us,
        result.p95_us,
        result.minimum_us,
        result.maximum_us,
        baseline_us / result.median_us,
        sep="\t",
    )


def tune(config: RunConfig) raises:
    validate_config(config)
    var count = checked_elements(config.rows, config.columns)
    if count == 0:
        print("# empty shape: no device allocation, no launch, no timing")
        return
    var work = FusionWorkspace(config)
    print("# operation: y=relu((x+bias[column])+residual); dtype=float32")
    print("# device_id:", config.device, "device:", work.ctx.name())
    print(
        "# rows:", config.rows, "columns:", config.columns, "elements:", count
    )
    print(
        "# aggregate_array_bytes:",
        allocation_bytes(config.rows, config.columns),
    )
    print(
        "# samples:",
        config.samples,
        "iterations_per_sample:",
        config.iterations,
    )
    print("# warmup_launch_groups_per_candidate:", WARMUP_COUNT)
    print(
        "# timing: synchronized host wall clock; resident buffers; batch means"
        " in us"
    )
    warm_and_verify(work)
    var samples = collect_samples(work, config)
    var baseline = summarize(samples[0])
    var best = 0
    var best_us = baseline.median_us
    print(
        "candidate\tkind\tblock\tmedian_us\tp95_us\tmin_us\tmax_us\tbaseline_ratio"
    )
    for candidate in range(CANDIDATE_COUNT):
        var result = summarize(samples[candidate])
        print_result(candidate, result, baseline.median_us)
        if result.median_us < best_us:
            best = candidate
            best_us = result.median_us
    # Recheck all variants after sustained operation; do not publish a winner first.
    for candidate in range(CANDIDATE_COUNT):
        work.execute(candidate)
        _ = work.verify()
    var roundtrip = List[Float64](capacity=config.samples)
    for _ in range(config.samples):
        roundtrip.append(measure_roundtrip(work, best, config.iterations))
    _ = work.verify()
    var end_to_end = summarize(roundtrip)
    print(
        "# status: PASS; every candidate passed full CPU comparison and output"
        " guards"
    )
    print("# fastest_measured_candidate:", best, "block:", block_size(best))
    print("# fastest_measured_baseline_ratio:", baseline.median_us / best_us)
    print("# winner_roundtrip_median_us:", end_to_end.median_us)
    print("# winner_roundtrip_p95_us:", end_to_end.p95_us)
    print(
        "# roundtrip: upload + compute + download + synchronization; reusable"
        " allocations"
    )
    print("# p95 is over batch means, not individual invocation tail latency")
    # Explicit last use keeps every owning buffer alive through all completed work.
    _ = work


def verify_gpu(config: RunConfig) raises:
    validate_config(config)
    var widths: List[Int] = [
        1,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
        513,
    ]
    for width in widths:
        var test_config = RunConfig(rows=1, columns=width, device=config.device)
        var work = FusionWorkspace(test_config)
        for candidate in range(CANDIDATE_COUNT):
            work.execute(candidate)
            _ = work.verify()
        _ = work
    var shape = RunConfig(rows=3, columns=257, device=config.device)
    var work = FusionWorkspace(shape)
    for candidate in range(CANDIDATE_COUNT):
        work.execute(candidate)
        _ = work.verify()
    _ = work
    print(
        "PASS: 17 shapes x 5 candidates; scalar comparison and 512-element"
        " output guard"
    )
