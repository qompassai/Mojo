# SPDX-License-Identifier: Apache-2.0
"""Validated resource limits and command-line configuration for Tiger Kernel."""
from std.collections import List

comptime DIMENSION_MAX = 16384
comptime ELEMENTS_MAX = 16777216
comptime MEMORY_MIB_MAX = 1024
comptime SAMPLES_MAX = 51
comptime ITERATIONS_MAX = 100
comptime GUARD_ELEMENTS = 512
comptime CANDIDATE_COUNT = 5
comptime WARMUP_COUNT = 2
comptime WORK_ELEMENTS_MAX = 34359738368


struct RunConfig(Copyable, Movable):
    var rows: Int
    var columns: Int
    var samples: Int
    var iterations: Int
    var device: Int
    var memory_mib: Int

    def __init__(
        out self,
        rows: Int = 1024,
        columns: Int = 4096,
        samples: Int = 9,
        iterations: Int = 20,
        device: Int = 0,
        memory_mib: Int = 512,
    ):
        self.rows = rows
        self.columns = columns
        self.samples = samples
        self.iterations = iterations
        self.device = device
        self.memory_mib = memory_mib


def checked_elements(rows: Int, columns: Int) raises -> Int:
    if rows < 0 or rows > DIMENSION_MAX:
        raise Error("rows must be in 0..16384")
    if columns < 1 or columns > DIMENSION_MAX:
        raise Error("columns must be in 1..16384")
    if rows > ELEMENTS_MAX // columns:
        raise Error("shape exceeds 16777216 elements")
    return rows * columns


def allocation_bytes(rows: Int, columns: Int) raises -> Int:
    var count = checked_elements(rows, columns)
    # Four full-sized device arrays, three host arrays, and two bias arrays.
    # Output has a trailing guard on both host and device. Float32 = four bytes.
    # The preceding shape limit proves this calculation fits a 64-bit Int.
    return (7 * count + 2 * columns + 2 * GUARD_ELEMENTS) * 4


def validate_config(config: RunConfig) raises:
    var required = allocation_bytes(config.rows, config.columns)
    if config.samples < 3 or config.samples > SAMPLES_MAX:
        raise Error("samples must be in 3..51")
    if config.iterations < 1 or config.iterations > ITERATIONS_MAX:
        raise Error("iterations must be in 1..100")
    # Six passes across five candidates per sample; at most two winner passes
    # per roundtrip. Include warmup and final verification. Products are bounded.
    var passes = config.samples * config.iterations * 8 + WARMUP_COUNT * 6 + 6
    if (
        checked_elements(config.rows, config.columns)
        > WORK_ELEMENTS_MAX // passes
    ):
        raise Error("aggregate kernel work exceeds 34359738368 element visits")
    if config.device < 0 or config.device > 31:
        raise Error("device must be in 0..31")
    if config.memory_mib < 1 or config.memory_mib > MEMORY_MIB_MAX:
        raise Error("memory-mib must be in 1..1024")
    if required > config.memory_mib * 1024 * 1024:
        raise Error("shape exceeds aggregate host/device array budget")


def parse_number(text: String) raises -> Int:
    if text.byte_length() == 0 or text.byte_length() > 9:
        raise Error("numeric arguments require 1..9 decimal digits")
    for byte in text.as_bytes():
        if byte < 48 or byte > 57:
            raise Error("numeric arguments must contain only decimal digits")
    return Int(text)


def parse_config(args: List[String]) raises -> RunConfig:
    if len(args) < 2 or len(args) > 14 or (len(args) - 2) % 2 != 0:
        raise Error("expected at most six --option value pairs")
    var config = RunConfig()
    var seen = List[String]()
    for index in range(2, len(args), 2):
        var key = args[index]
        for previous in seen:
            if previous == key:
                raise Error("duplicate option")
        seen.append(key)
        var value = parse_number(args[index + 1])
        if key == "--rows":
            config.rows = value
        elif key == "--cols":
            config.columns = value
        elif key == "--samples":
            config.samples = value
        elif key == "--iterations":
            config.iterations = value
        elif key == "--device":
            config.device = value
        elif key == "--memory-mib":
            config.memory_mib = value
        else:
            raise Error("unknown option; use --help")
    validate_config(config)
    return config^
