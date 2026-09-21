# SPDX-License-Identifier: Apache-2.0
"""Private device entry points. Use FusionWorkspace for checked host dispatch.

Unsafe contract: dedicated, nonaliasing Float32 buffers on one context; initialized
inputs; count <= ELEMENTS_MAX; columns > 0; bias length == columns; remaining
buffers have at least count elements. Only index < Int(count) can access memory.
Each thread owns one output element, so no barriers or atomic writes are needed.
Device pointers are never dereferenced by host code.
"""
from max.gpu import block_dim, block_idx, thread_idx
from std.memory import Pointer
from std.origin import MutUnsafeAnyOrigin

comptime FloatPointer = Pointer[Float32, MutUnsafeAnyOrigin]


def fused_bias_residual_relu(
    x: FloatPointer,
    bias: FloatPointer,
    residual: FloatPointer,
    output: FloatPointer,
    count: Int32,
    columns: Int32,
):
    var index = Int(block_idx.x * block_dim.x + thread_idx.x)
    if index < Int(count):
        var biased = (
            x[unsafe_offset=index] + bias[unsafe_offset=index % Int(columns)]
        )
        var value = biased + residual[unsafe_offset=index]
        output[unsafe_offset=index] = max(value, Float32(0))


def add_bias(
    x: FloatPointer,
    bias: FloatPointer,
    temporary: FloatPointer,
    count: Int32,
    columns: Int32,
):
    var index = Int(block_idx.x * block_dim.x + thread_idx.x)
    if index < Int(count):
        temporary[unsafe_offset=index] = (
            x[unsafe_offset=index] + bias[unsafe_offset=index % Int(columns)]
        )


def residual_relu(
    temporary: FloatPointer,
    residual: FloatPointer,
    output: FloatPointer,
    count: Int32,
):
    var index = Int(block_idx.x * block_dim.x + thread_idx.x)
    if index < Int(count):
        var value = (
            temporary[unsafe_offset=index] + residual[unsafe_offset=index]
        )
        output[unsafe_offset=index] = max(value, Float32(0))
