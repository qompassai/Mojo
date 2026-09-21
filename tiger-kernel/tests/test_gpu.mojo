# SPDX-License-Identifier: Apache-2.0
"""Requires a compatible GPU. Missing hardware is an error, never a passing test."""
from std.collections import List
from std.testing import TestSuite, assert_equal, assert_raises
from std.sys import has_accelerator
from contracts import RunConfig
from runner import verify_gpu
from workspace import FusionWorkspace


def test_gpu_boundaries() raises:
    verify_gpu(RunConfig())


def test_user_inputs_and_rejected_mutation() raises:
    var work = FusionWorkspace(RunConfig(rows=1, columns=7))
    var x: List[Float32] = [1000000, -1000000, 0.1, -0.1, 0, -0.0, 1]
    var b: List[Float32] = [-999999, 999999, 0.2, -0.2, 0, 0, -1]
    var residual: List[Float32] = [0.125, 2, 0.3, 0.4, -1, 0, 0.000001]
    work.load(x, b, residual)
    for candidate in range(5):
        work.execute(candidate)
        _ = work.verify()
    x[0] = Float32(Float64(String("nan")))
    with assert_raises():
        work.load(x, b, residual)
    # Validation failed before host/device mutation; previous data remain usable.
    assert_equal(work.host_x[0], Float32(1000000))
    work.execute(3)
    _ = work.verify()
    var short = List[Float32]()
    with assert_raises():
        work.load(short, b, residual)
    with assert_raises():
        work.execute(5)
    _ = work


def test_guard_corruption_detected() raises:
    var work = FusionWorkspace(RunConfig(rows=1, columns=1))
    work.execute(1)
    work.host_output[work.count] = Float32(0)
    with assert_raises():
        _ = work.verify()
    _ = work


def test_empty_workspace_rejected() raises:
    with assert_raises():
        var work = FusionWorkspace(RunConfig(rows=0, columns=1))
        _ = work


def main() raises:
    comptime if not has_accelerator():
        raise Error("GPU tests unavailable: compile on a supported GPU host")
    else:
        var suite = TestSuite.discover_tests[__functions_in_module()]()
        suite^.run()
