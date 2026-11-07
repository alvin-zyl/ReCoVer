#!/usr/bin/env python3
"""Standalone failure-schedule generator.

Usage examples::

    # 3D parallelism, write to file
    python scripts/generate_failure_schedule.py \\
        --kind 3d --tp 2 --pp 2 --dp 4 --ep 1 \\
        --num-failures 3 --step-range 10 200 \\
        --seed 42 \\
        --locations grad-sync:first:0.6 forward:0.4 \\
        --output schedules/3d_demo.yaml

    # HSDP, print to stdout
    python scripts/generate_failure_schedule.py \\
        --kind hsdp --world-size 8 --shard-size 2 \\
        --num-failures 2 --step-range 50 500 --seed 7 \\
        --locations grad-sync:first:1.0 --print

    # Human-readable summary only, no file written
    python scripts/generate_failure_schedule.py \\
        --kind hsdp --world-size 8 --shard-size 2 \\
        --num-failures 2 --step-range 50 500 --seed 7 \\
        --locations grad-sync:first:1.0 --dry-run

This script does *not* import torch, MPI, CUDA, or any distributed primitive.
Output is byte-identical to what the in-process generator produces at
training init given the same inputs.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple


# Import the ``failure`` subpackage as a top-level package so we skip
# ``ulfm_collectives/__init__.py`` (which pulls in the compiled _C extension
# and thus torch). The standalone generator must stay torch-free.
_THIS = os.path.dirname(os.path.abspath(__file__))
_EXT_ROOT = os.path.dirname(_THIS)
_ULFM_PKG = os.path.join(_EXT_ROOT, "ulfm_collectives")
if _ULFM_PKG not in sys.path:
    sys.path.insert(0, _ULFM_PKG)

from failure import (  # noqa: E402  (imported as top-level package; see above)
    FailureSchedule,
    ParallelismSpec,
    generate,
    replicas_for,
)


def _parse_location_weights(pairs: List[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for pair in pairs:
        if ":" not in pair:
            raise argparse.ArgumentTypeError(
                f"--locations entry must be name:weight, got {pair!r}"
            )
        name, _, weight = pair.rpartition(":")
        name = name.strip()
        if not name:
            raise argparse.ArgumentTypeError(f"empty location name in {pair!r}")
        try:
            w = float(weight)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid weight in {pair!r}")
        if w <= 0:
            raise argparse.ArgumentTypeError(f"weight must be > 0 in {pair!r}")
        if name in result:
            raise argparse.ArgumentTypeError(f"duplicate location {name!r}")
        result[name] = w
    return result


def _spec_from_args(args: argparse.Namespace) -> ParallelismSpec:
    if args.kind == "3d":
        for required in ("tp", "pp", "dp", "ep"):
            if getattr(args, required) is None:
                raise SystemExit(
                    f"--{required} is required with --kind 3d"
                )
        return ParallelismSpec(
            kind="3d", tp=args.tp, pp=args.pp, dp=args.dp, ep=args.ep
        )
    if args.kind == "hsdp":
        if args.world_size is None or args.shard_size is None:
            raise SystemExit(
                "--world-size and --shard-size are required with --kind hsdp"
            )
        return ParallelismSpec(
            kind="hsdp", world_size=args.world_size, shard_size=args.shard_size
        )
    raise SystemExit(f"unknown --kind: {args.kind!r}")


def _print_summary(sched: FailureSchedule) -> None:
    replicas = replicas_for(sched.parallelism)
    print(f"parallelism: {sched.parallelism.to_dict()}")
    print(
        f"{len(replicas)} replicas, "
        f"{len(replicas[0]) if replicas else 0} ranks per replica"
    )
    if sched.generator_config is not None:
        gc = sched.generator_config
        print(
            f"generator: seed={gc.seed} num_failures={gc.num_failures} "
            f"step_range={tuple(gc.step_range)} sampling={gc.sampling} "
            f"locations={dict(sorted(gc.location_weights.items()))}"
        )
    if not sched.entries:
        print("(no failures scheduled)")
        return
    print()
    print(f"{'step':>6}  {'replica':>7}  {'local':>5}  {'global':>6}  location")
    print(f"{'-'*6}  {'-'*7}  {'-'*5}  {'-'*6}  {'-'*20}")
    for e in sched.entries:
        gr = replicas[e.replica_id][e.local_rank]
        print(f"{e.step:>6}  {e.replica_id:>7}  {e.local_rank:>5}  {gr:>6}  {e.location}")


def main(argv: Tuple[str, ...] = None) -> int:
    p = argparse.ArgumentParser(
        description="Deterministic failure-schedule generator (standalone).",
    )

    p.add_argument("--kind", choices=("3d", "hsdp"), required=True)

    # 3d
    p.add_argument("--tp", type=int, default=None)
    p.add_argument("--pp", type=int, default=None)
    p.add_argument("--dp", type=int, default=None)
    p.add_argument("--ep", type=int, default=None)

    # hsdp
    p.add_argument("--world-size", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=None)

    # generator inputs
    p.add_argument("--num-failures", type=int, required=True)
    p.add_argument(
        "--step-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        required=True,
        help="half-open [START, END)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sampling",
        choices=("iid", "stratified"),
        default="stratified",
    )
    p.add_argument(
        "--locations",
        nargs="+",
        default=None,
        metavar="NAME:WEIGHT",
        help="one or more name:weight entries, e.g. grad-sync:first:0.6 forward:0.4; "
        "locations: forward, backward, grad-sync:first, grad-sync:mid, grad-sync:last, optimizer",
    )
    p.add_argument(
        "--exclude-replicas",
        type=str,
        default=None,
        metavar="IDS",
        help="comma-separated replica ids to exclude from selection "
             "(e.g. '0' to spare the wandb logger replica).",
    )

    # output
    g = p.add_mutually_exclusive_group()
    g.add_argument("--output", type=str, default=None, help="write YAML to this path")
    g.add_argument("--print", dest="print_yaml", action="store_true", help="write YAML to stdout")
    g.add_argument("--dry-run", action="store_true", help="print summary only, no YAML")

    args = p.parse_args(argv)

    spec = _spec_from_args(args)

    if args.num_failures > 0:
        if not args.locations:
            p.error("--locations is required when --num-failures > 0")
        location_weights = _parse_location_weights(args.locations)
    else:
        location_weights = {}

    exclude_ids = None
    if args.exclude_replicas:
        try:
            exclude_ids = [int(x) for x in args.exclude_replicas.split(",") if x.strip()]
        except ValueError:
            p.error(f"--exclude-replicas expects comma-separated integers, got {args.exclude_replicas!r}")

    sched = generate(
        parallelism=spec,
        seed=args.seed,
        num_failures=args.num_failures,
        step_range=(args.step_range[0], args.step_range[1]),
        location_weights=location_weights,
        sampling=args.sampling,
        exclude_replica_ids=exclude_ids,
    )

    if args.dry_run:
        _print_summary(sched)
        return 0

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        sched.save(args.output)
        print(f"wrote {len(sched.entries)} entries to {args.output}")
        return 0

    # Default / explicit --print: YAML to stdout.
    sys.stdout.write(sched.to_yaml_str())
    return 0


if __name__ == "__main__":
    sys.exit(main())
