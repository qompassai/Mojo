# SPDX-License-Identifier: Apache-2.0
"""Numerical contract and deterministic CPU reference; no accelerator imports."""
from std.math import abs, isfinite

comptime INPUT_ABS_MAX = Float32(1000000)
comptime ABS_TOLERANCE = Float32(0.00001)
comptime REL_TOLERANCE = Float32(0.00001)
comptime OUTPUT_SENTINEL = Float32(-12345)


def validate_value(value: Float32) raises:
    if not isfinite(value) or abs(value) > INPUT_ABS_MAX:
        raise Error("inputs must be finite and abs(value) <= 1000000")


def scalar_reference(
    x: Float32, bias: Float32, residual: Float32
) raises -> Float32:
    validate_value(x)
    validate_value(bias)
    validate_value(residual)
    # Keep the same explicit addition order as the device implementation.
    var biased = x + bias
    var result = biased + residual
    if result > 0:
        return result
    return Float32(0)


def sample_x(index: Int) -> Float32:
    return Float32(index % 257 - 128) / Float32(32)


def sample_residual(index: Int) -> Float32:
    return Float32(index % 31 - 15) / Float32(16)


def sample_bias(column: Int) -> Float32:
    return Float32(column % 17 - 8) / Float32(8)


def checked_error(actual: Float32, expected: Float32) raises -> Float32:
    if not isfinite(actual) or not isfinite(expected):
        raise Error("nonfinite verification result")
    var error = abs(actual - expected)
    if error > ABS_TOLERANCE + REL_TOLERANCE * abs(expected):
        raise Error("GPU output disagrees with the scalar reference")
    return error
