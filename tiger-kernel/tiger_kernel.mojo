# SPDX-License-Identifier: Apache-2.0
"""Tiger Kernel: explicit, unprivileged AI operator fusion verification and tuning."""
from std.sys import argv, has_accelerator
from contracts import allocation_bytes, checked_elements, parse_config
from std.collections import List
from runner import tune, verify_gpu


def print_help():
    print("Tiger Kernel 0.1.0 -- Mojo 1.1.0 / MAX 26.6.0")
    print("Usage: tiger-kernel COMMAND [--option value ...]")
    print("Commands: tune, verify, plan, --help")
    print("Options: --rows 1024 --cols 4096 --samples 9 --iterations 20")
    print("         --device 0 --memory-mib 512")
    print("plan: validate shape/budgets without device access")
    print("verify: check a fixed GPU boundary-shape suite on --device")
    print("tune: check and benchmark two-pass vs fused bias+residual+ReLU")
    print("CPU tests: mojo run -I . -D ASSERT=all tests/test_cpu.mojo")
    print(
        "No command runs by default. No root, network, or system configuration"
        " writes."
    )


def main() raises:
    var raw_args = argv()
    if len(raw_args) > 14:
        raise Error("too many command-line arguments")
    var args = List[String](capacity=len(raw_args))
    for index in range(len(raw_args)):
        # argv[0] can contain a long executable path; never copy that path.
        if index == 0:
            args.append("tiger-kernel")
        else:
            if raw_args[index].byte_length() > 32:
                raise Error("command-line argument exceeds 32 bytes")
            args.append(String(raw_args[index]))
    if len(args) == 1:
        print_help()
        return
    if args[1] == "--help" or args[1] == "-h":
        if len(args) != 2:
            raise Error("help accepts no additional arguments")
        print_help()
        return
    var command = args[1]
    if command != "tune" and command != "verify" and command != "plan":
        raise Error("unknown command; use --help")
    if command == "verify":
        for index in range(2, len(args), 2):
            if args[index] != "--device":
                raise Error(
                    "verify uses fixed shapes and accepts only --device"
                )
    var config = parse_config(args)
    if command == "plan":
        print("rows:", config.rows, "columns:", config.columns)
        print("elements:", checked_elements(config.rows, config.columns))
        print(
            "aggregate_array_bytes:",
            allocation_bytes(config.rows, config.columns),
        )
        print("samples:", config.samples, "iterations:", config.iterations)
        print("device_id:", config.device, "budget_mib:", config.memory_mib)
        return
    comptime if has_accelerator():
        if command == "verify":
            verify_gpu(config)
        else:
            tune(config)
    else:
        raise Error(
            "GPU target unavailable; rebuild on a supported GPU host or set"
            " --target-accelerator"
        )
