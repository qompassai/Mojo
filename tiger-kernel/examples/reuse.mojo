# SPDX-License-Identifier: Apache-2.0
"""Synchronous integration example: reuse a workspace for two real input batches."""
from std.collections import List
from contracts import RunConfig
from workspace import FusionWorkspace


def main() raises:
    var work = FusionWorkspace(RunConfig(rows=1, columns=3, device=0))
    var x: List[Float32] = [1, -2, 3]
    var bias: List[Float32] = [0.25, 0.5, -0.25]
    var residual: List[Float32] = [-0.5, 1, 0.25]
    # Candidate 3 means fused/256; substitute the winner for this workload/device.
    var candidate = 3
    for batch in range(2):
        x[0] = Float32(batch + 1)
        work.load(x, bias, residual)
        work.execute(candidate)
        _ = work.verify()
        print(work.host_output[0], work.host_output[1], work.host_output[2])
    _ = work
