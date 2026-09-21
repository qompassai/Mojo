# SPDX-License-Identifier: Apache-2.0
"""Small bounded sample statistics. No unbounded sorting or hidden allocations."""
from std.collections import List
from std.math import isfinite
from contracts import SAMPLES_MAX


@fieldwise_init
struct Summary(Copyable, Movable):
    var median_us: Float64
    var p95_us: Float64
    var minimum_us: Float64
    var maximum_us: Float64


def summarize(samples: List[Float64]) raises -> Summary:
    var count = len(samples)
    if count < 1 or count > SAMPLES_MAX:
        raise Error("sample count outside 1..51")
    var ordered = List[Float64](capacity=count)
    for value in samples:
        if not isfinite(value) or value <= 0:
            raise Error("timing samples must be finite and positive")
        ordered.append(value)
    # Insertion sort: at most 51*50/2 comparisons; deterministic tie behavior.
    for index in range(1, count):
        var value = ordered[index]
        var position = index
        while position > 0:
            if ordered[position - 1] <= value:
                break
            ordered[position] = ordered[position - 1]
            position -= 1
        ordered[position] = value
    var median = ordered[count // 2]
    if count % 2 == 0:
        median = (ordered[count // 2 - 1] + median) / 2
    var p95_index = (95 * count + 99) // 100 - 1
    return Summary(median, ordered[p95_index], ordered[0], ordered[count - 1])
