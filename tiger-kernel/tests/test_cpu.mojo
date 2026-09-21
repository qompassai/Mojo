# SPDX-License-Identifier: Apache-2.0
from std.collections import List
from std.testing import TestSuite, assert_equal, assert_raises
from contracts import (
    ELEMENTS_MAX,
    RunConfig,
    allocation_bytes,
    checked_elements,
    parse_config,
    parse_number,
    validate_config,
)
from reference import checked_error, scalar_reference, validate_value
from statistics import summarize


def test_shape_limits() raises:
    assert_equal(checked_elements(0, 1), 0)
    assert_equal(checked_elements(1024, 16384), ELEMENTS_MAX)
    with assert_raises():
        _ = checked_elements(-1, 1)
    with assert_raises():
        _ = checked_elements(1, 0)
    with assert_raises():
        _ = checked_elements(1, 16385)
    with assert_raises():
        _ = checked_elements(1025, 16384)
    with assert_raises():
        _ = checked_elements(Int.MAX, Int.MAX)


def test_budget_boundary() raises:
    var config = RunConfig(rows=256, columns=256, memory_mib=1)
    with assert_raises():
        validate_config(config)
    config.memory_mib = 2
    validate_config(config)
    assert_equal(allocation_bytes(1, 1), (7 + 2 + 1024) * 4)


def test_cli_validation() raises:
    var args: List[String] = ["tool", "tune", "--rows", "3", "--cols", "257"]
    var config = parse_config(args)
    assert_equal(config.rows, 3)
    assert_equal(config.columns, 257)
    var duplicate: List[String] = ["tool", "tune", "--rows", "3", "--rows", "4"]
    with assert_raises():
        _ = parse_config(duplicate)
    var missing: List[String] = ["tool", "tune", "--rows"]
    with assert_raises():
        _ = parse_config(missing)
    var unknown: List[String] = ["tool", "tune", "--typo", "4"]
    with assert_raises():
        _ = parse_config(unknown)


def test_numeric_parser() raises:
    assert_equal(parse_number("00017"), 17)
    var invalid: List[String] = [
        "",
        "-1",
        "+1",
        "1.5",
        " 2",
        "9999999999",
        "abc",
    ]
    for value in invalid:
        with assert_raises():
            _ = parse_number(value)


def test_runtime_bounds() raises:
    var config = RunConfig()
    config.samples = 0
    with assert_raises():
        validate_config(config)
    config.samples = 52
    with assert_raises():
        validate_config(config)
    config.samples = 3
    config.iterations = 101
    with assert_raises():
        validate_config(config)
    config.iterations = 1
    config.device = 32
    with assert_raises():
        validate_config(config)


def test_aggregate_work_budget() raises:
    var config = RunConfig(rows=1024, columns=16384, samples=51, iterations=100)
    with assert_raises():
        validate_config(config)


def test_empty_argument_list_rejected() raises:
    var args = List[String]()
    with assert_raises():
        _ = parse_config(args)


def test_reference_and_extremes() raises:
    assert_equal(scalar_reference(1, 2, 3), Float32(6))
    assert_equal(scalar_reference(-1, -2, -3), Float32(0))
    assert_equal(scalar_reference(-0.0, 0, 0), Float32(0))
    assert_equal(scalar_reference(1000000, 1000000, 1000000), Float32(3000000))
    with assert_raises():
        validate_value(Float32(1000001))
    with assert_raises():
        validate_value(Float32(Float64(String("nan"))))
    with assert_raises():
        validate_value(Float32(Float64(String("inf"))))
    with assert_raises():
        validate_value(Float32(Float64(String("-inf"))))


def test_verifier_rejects_corruption() raises:
    assert_equal(checked_error(1, 1), Float32(0))
    with assert_raises():
        _ = checked_error(2, 1)
    with assert_raises():
        _ = checked_error(Float32(Float64(String("nan"))), 1)


def test_statistics() raises:
    var odd: List[Float64] = [5, 1, 3]
    var result = summarize(odd)
    assert_equal(result.median_us, Float64(3))
    assert_equal(result.p95_us, Float64(5))
    assert_equal(odd[0], Float64(5))
    var even: List[Float64] = [4, 1, 2, 3]
    assert_equal(summarize(even).median_us, Float64(2.5))
    var bad: List[Float64] = [0]
    with assert_raises():
        _ = summarize(bad)
    var empty = List[Float64]()
    with assert_raises():
        _ = summarize(empty)


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    suite^.run()
