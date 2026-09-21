# SPDX-License-Identifier: Apache-2.0
"""Own all buffers for one shape and one explicitly selected accelerator stream."""
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.collections import List
from contracts import (
    GUARD_ELEMENTS,
    RunConfig,
    checked_elements,
    validate_config,
)
from kernels import add_bias, fused_bias_residual_relu, residual_relu
from reference import (
    OUTPUT_SENTINEL,
    checked_error,
    sample_bias,
    sample_residual,
    sample_x,
    scalar_reference,
    validate_value,
)

comptime F32 = DType.float32


def block_size(candidate: Int) raises -> Int:
    if candidate == 0:
        return 256
    if candidate == 1:
        return 64
    if candidate == 2:
        return 128
    if candidate == 3:
        return 256
    if candidate == 4:
        return 512
    raise Error("candidate must be in 0..4")


struct FusionWorkspace(Movable):
    var ctx: DeviceContext
    var count: Int
    var columns: Int
    var host_x: HostBuffer[F32]
    var host_bias: HostBuffer[F32]
    var host_residual: HostBuffer[F32]
    var host_output: HostBuffer[F32]
    var device_x: DeviceBuffer[F32]
    var device_bias: DeviceBuffer[F32]
    var device_residual: DeviceBuffer[F32]
    var device_temporary: DeviceBuffer[F32]
    var device_output: DeviceBuffer[F32]

    def __init__(out self, config: RunConfig) raises:
        validate_config(config)
        var count = checked_elements(config.rows, config.columns)
        if count == 0:
            raise Error("empty shapes do not need a GPU workspace")
        self.ctx = DeviceContext(config.device)
        self.count = count
        self.columns = config.columns
        self.host_x = self.ctx.enqueue_create_host_buffer[F32](count)
        self.host_bias = self.ctx.enqueue_create_host_buffer[F32](
            config.columns
        )
        self.host_residual = self.ctx.enqueue_create_host_buffer[F32](count)
        self.host_output = self.ctx.enqueue_create_host_buffer[F32](
            count + GUARD_ELEMENTS
        )
        self.device_x = self.ctx.enqueue_create_buffer[F32](count)
        self.device_bias = self.ctx.enqueue_create_buffer[F32](config.columns)
        self.device_residual = self.ctx.enqueue_create_buffer[F32](count)
        self.device_temporary = self.ctx.enqueue_create_buffer[F32](count)
        self.device_output = self.ctx.enqueue_create_buffer[F32](
            count + GUARD_ELEMENTS
        )
        self.ctx.synchronize()
        self.load_fixture()

    def load_fixture(mut self) raises:
        """Deterministic mixed-sign exact binary fractions; overwrites prior inputs.
        """
        self.ctx.synchronize()
        for index in range(self.count):
            self.host_x[index] = sample_x(index)
            self.host_residual[index] = sample_residual(index)
        for column in range(self.columns):
            self.host_bias[column] = sample_bias(column)
        self.upload()

    def load(
        mut self,
        x: List[Float32],
        bias: List[Float32],
        residual: List[Float32],
    ) raises:
        """Validate all input before mutation. Caller-owned lists are extra memory.

        A device/transfer failure after validation can leave partial device state;
        discard the workspace on such an error. Rejected values preserve inputs.
        """
        if len(x) != self.count or len(residual) != self.count:
            raise Error("x/residual length does not match workspace shape")
        if len(bias) != self.columns:
            raise Error("bias length does not match columns")
        for index in range(self.count):
            validate_value(x[index])
            validate_value(residual[index])
        for value in bias:
            validate_value(value)
        self.ctx.synchronize()
        for index in range(self.count):
            self.host_x[index] = x[index]
            self.host_residual[index] = residual[index]
        for column in range(self.columns):
            self.host_bias[column] = bias[column]
        self.upload()

    def upload(mut self) raises:
        self.ctx.enqueue_copy(dst_buf=self.device_x, src_buf=self.host_x)
        self.ctx.enqueue_copy(dst_buf=self.device_bias, src_buf=self.host_bias)
        self.ctx.enqueue_copy(
            dst_buf=self.device_residual, src_buf=self.host_residual
        )
        self.ctx.synchronize()

    def reset_output(mut self) raises:
        self.ctx.synchronize()
        for index in range(self.count + GUARD_ELEMENTS):
            self.host_output[index] = OUTPUT_SENTINEL
        self.ctx.enqueue_copy(
            dst_buf=self.device_output, src_buf=self.host_output
        )
        self.ctx.synchronize()

    def enqueue(mut self, candidate: Int) raises:
        """Internal timed path; constructor/load established shapes and finiteness.

        Keep this workspace alive until synchronize() completes. Enqueues are on
        the same stream as buffer allocation/free; no detached work is created.
        """
        var block = block_size(candidate)
        var grid = (self.count + block - 1) // block
        if candidate == 0:
            self.ctx.enqueue_function[add_bias](
                self.device_x,
                self.device_bias,
                self.device_temporary,
                Int32(self.count),
                Int32(self.columns),
                grid_dim=grid,
                block_dim=block,
            )
            self.ctx.enqueue_function[residual_relu](
                self.device_temporary,
                self.device_residual,
                self.device_output,
                Int32(self.count),
                grid_dim=grid,
                block_dim=block,
            )
        else:
            self.ctx.enqueue_function[fused_bias_residual_relu](
                self.device_x,
                self.device_bias,
                self.device_residual,
                self.device_output,
                Int32(self.count),
                Int32(self.columns),
                grid_dim=grid,
                block_dim=block,
            )

    def download(mut self) raises:
        self.ctx.enqueue_copy(
            dst_buf=self.host_output, src_buf=self.device_output
        )
        self.ctx.synchronize()

    def execute(mut self, candidate: Int) raises:
        """Synchronous public operation; output is available in host_output on return.
        """
        _ = block_size(candidate)
        self.reset_output()
        self.enqueue(candidate)
        self.download()

    def verify(self) raises -> Float32:
        """Full CPU comparison plus guard check, after a completed download."""
        var maximum_error = Float32(0)
        for index in range(self.count):
            var expected = scalar_reference(
                self.host_x[index],
                self.host_bias[index % self.columns],
                self.host_residual[index],
            )
            maximum_error = max(
                maximum_error, checked_error(self.host_output[index], expected)
            )
        for index in range(self.count, self.count + GUARD_ELEMENTS):
            if self.host_output[index] != OUTPUT_SENTINEL:
                raise Error("output guard changed: out-of-range GPU write")
        return maximum_error
