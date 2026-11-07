"""recover: launch ReCoVer training on Open MPI's ULFM.

``recover hsdp`` (train/main_hsdp.py) and ``recover 3d`` (train/main_3d.py, on
Nanotron) build the mpirun command and run it. The command turns on fault
tolerance (``--with-ft ulfm``), sets the NCCL timeouts that recovery needs, and
exports the environment to every rank. Training options come from a YAML config
and from the arguments after ``--``, which go to the training script unchanged.
"""

import argparse
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from recover import __version__

HOME = Path(os.environ.get("RECOVER_HOME") or Path(__file__).resolve().parent.parent)
HSDP_SCRIPT = HOME / "train" / "main_hsdp.py"
NANOTRON_SCRIPT = HOME / "train" / "main_3d.py"
SCHEDULE_SCRIPT = HOME / "scripts" / "generate_failure_schedule.py"
DEFAULT_DATASET = "/data/c4/tokenized/seq_len_4096"

# Variables exported to every rank when they are set in the caller's environment.
PASSTHROUGH_PREFIXES = ("WANDB_", "HF_", "NCCL_", "TORCH_NCCL_")
PASSTHROUGH_NAMES = ("RECOVER_DATASET_PATH", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES",
                     "CUDA_DEVICE_MAX_CONNECTIONS")


def _load_locations():
    """Failure locations, read from ulfm_collectives/failure/locations.py without importing the
    package (importing it loads torch and the C++ extension)."""
    try:
        spec = importlib.util.spec_from_file_location(
            "_recover_locations", HOME / "ulfm_collectives" / "failure" / "locations.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return tuple(mod.LOCATIONS), dict(mod.LOCATION_ALIASES)
    except (OSError, AttributeError):
        return ("forward", "backward", "grad-sync:first", "grad-sync:mid", "grad-sync:last", "optimizer"), {}


LOCATIONS, LOCATION_ALIASES = _load_locations()


class _Formatter(argparse.RawDescriptionHelpFormatter):
    """Keeps the descriptions' line breaks and shows defaults only where there is one."""

    def __init__(self, prog):
        super().__init__(prog, max_help_position=34, width=min(shutil.get_terminal_size().columns, 110))

    def _get_help_string(self, action):
        text = action.help or ""
        if (action.option_strings and "%(default)" not in text
                and action.default not in (None, False, [], argparse.SUPPRESS)):
            text += " (default: %(default)s)"
        return text


class _Error(Exception):
    pass


# --------------------------------------------------------------------------- parsers

TOP_DESCRIPTION = "Launch ReCoVer: fault-tolerant LLM pre-training on Open MPI's ULFM."

TOP_EPILOG = """\
examples:
  recover hsdp -n 4 --config examples/8xH100/hsdp_llama_60m.yaml \\
      --failure-count 1 --failure-step-range 100 200 --failure-locations forward
  recover 3d -n 8 --config examples/8xH100/3d_llama_60m.yaml \\
      --failure-count 1 --failure-step-range 5 10 --failure-locations grad-sync:first
  recover hsdp -n 4 --config examples/8xH100/hsdp_llama_60m.yaml --dry-run

Ready-to-run scripts for each platform are in examples/.

Run 'recover <command> -h' for the options of a command.
"""

HSDP_DESCRIPTION = """\
HSDP training. Each replica shards the model over --shard-size GPUs with FSDP
(HYBRID_SHARD). The gradient all-reduce across replicas runs on ReCoVer's ULFM
backend.

Training options are those of train/main_hsdp.py. Put them in --config (a YAML
file of option: value pairs) or after '--'. Options after '--' override the
config. 'recover hsdp --script-help' lists them all.
"""

HSDP_EPILOG = """\
examples:
  # 2 replicas x 2 GPUs, one replica killed before a forward pass in steps 100-199
  recover hsdp -n 4 --config examples/8xH100/hsdp_llama_60m.yaml \\
      --failure-count 1 --failure-step-range 100 200 --failure-locations forward

  # the same run, with a training option overridden after '--'
  recover hsdp -n 4 --config examples/8xH100/hsdp_llama_60m.yaml -- --num_training_steps 500

  # the failure-free reference on NCCL (torchrun, no fault tolerance)
  recover hsdp -n 4 --config examples/8xH100/hsdp_llama_60m.yaml --nccl-baseline

  # multi-node, 4 GPUs per node (e.g. inside a PBS job)
  recover hsdp -n 32 --hostfile $PBS_NODEFILE --map-by ppr:4:node:PE=8 --bind-to core \\
      --master-addr $(head -1 $PBS_NODEFILE) --shard-size 8 --config my_run.yaml
"""

THREED_DESCRIPTION = """\
3D-parallel training on Nanotron. The Nanotron YAML config (--config) sets the
data, tensor and pipeline parallelism. Each data-parallel replica is one TP x PP
group. The gradient all-reduce across replicas runs on ReCoVer's ULFM backend.

Other options of train/main_3d.py go after '--'. 'recover 3d --script-help'
lists them.
"""

THREED_EPILOG = """\
examples:
  # DP=2, TP=2, PP=2 on 8 GPUs, one replica killed during gradient synchronization
  recover 3d -n 8 --config examples/8xH100/3d_llama_60m.yaml \\
      --failure-count 1 --failure-step-range 5 10 --failure-locations grad-sync:first

  # replay a fixed schedule
  recover schedule --kind 3d --tp 2 --pp 2 --dp 2 --num-failures 2 \\
      --step-range 5 20 --locations grad-sync:first:1 --output sched.yaml
  recover 3d -n 8 --config examples/8xH100/3d_llama_60m.yaml --failure-schedule sched.yaml
"""

TIMEOUT_DESCRIPTION = """\
After a failure, the other ranks of the failed replica exit only when their
NCCL collectives time out (--pg-timeout, and --pp-timeout in 3D). Meanwhile,
some ranks of the healthy replicas wait at a barrier inside their replica. The
barrier times out after --mp-timeout. Keep --mp-timeout longer than
--pg-timeout (or --pp-timeout) plus 4 x --dump-wait-ms plus the time the
processes take to exit. Otherwise a healthy replica times out too. See
docker/README.md.
"""

FAILURE_DESCRIPTION = """\
The failure simulator kills a scheduled rank with SIGKILL. Either sample a
schedule (--failure-count, --failure-step-range, --failure-locations) or
replay one (--failure-schedule). Without these options nothing is injected.
"""


def _add_common(p, kind):
    g = p.add_argument_group("processes")
    g.add_argument("-n", "--nproc", type=int, metavar="N",
                   help="number of processes (one per GPU)")
    g.add_argument("--hostfile", metavar="FILE",
                   help="Open MPI hostfile for multi-node runs, e.g. $PBS_NODEFILE")
    g.add_argument("--map-by", metavar="SPEC",
                   help="mpirun --map-by, e.g. ppr:4:node:PE=8")
    g.add_argument("--bind-to", metavar="UNIT", help="mpirun --bind-to, e.g. core")
    g.add_argument("--btl", default="tcp,self,sm", metavar="LIST",
                   help="Open MPI transports (--mca btl). '' uses Open MPI's default")
    g.add_argument("--master-addr", default="127.0.0.1", metavar="ADDR",
                   help="address of global rank 0 for the NCCL groups. For multi-node runs, the first host")
    g.add_argument("--master-port", type=int, default=29500, metavar="PORT",
                   help="port for the NCCL groups' rendezvous")
    g.add_argument("--mpirun", default="mpirun", metavar="CMD",
                   help="launcher of Open MPI built with ULFM (mpirun or mpiexec)")
    g.add_argument("--mpi-arg", action="append", default=[], metavar="ARGS",
                   help="extra mpirun arguments, split like a shell command. Repeatable")
    g.add_argument("--env", action="append", default=[], metavar="NAME[=VALUE]",
                   help="export an environment variable to all ranks, and set it if VALUE is given. Repeatable")
    g.add_argument("--python", default=sys.executable, metavar="PATH",
                   help="Python interpreter that runs the training script")

    t = p.add_argument_group("timeouts (seconds)", TIMEOUT_DESCRIPTION)
    t.add_argument("--pg-timeout", type=int, default=20, metavar="SEC",
                   help="NCCL_PG_TIMEOUT_SECONDS: timeout of the intra-replica NCCL groups")
    if kind == "3d":
        t.add_argument("--pp-timeout", type=int, default=20, metavar="SEC",
                       help="NCCL_PP_TIMEOUT_SECONDS: timeout of the pipeline-parallel groups")
    t.add_argument("--mp-timeout", type=int, default=160, metavar="SEC",
                   help="NCCL_MP_TIMEOUT_SECONDS: timeout of the barrier inside each replica")
    t.add_argument("--dump-wait-ms", type=int, default=1000, metavar="MS",
                   help="TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: the NCCL watchdog waits 4x this before it aborts a timed-out rank")
    t.add_argument("--no-timeout-check", action="store_true",
                   help="do not check the order of the timeouts")

    f = p.add_argument_group("failure injection", FAILURE_DESCRIPTION)
    f.add_argument("--failure-count", type=int, metavar="N",
                   help="number of replicas to kill")
    f.add_argument("--failure-step-range", type=int, nargs=2, metavar=("START", "END"),
                   help="kill at training steps in [START, END)")
    f.add_argument("--failure-locations", nargs="+", metavar="NAME[:WEIGHT]",
                   help="where in the step to kill, with relative weights (default 1): " + ", ".join(LOCATIONS))
    f.add_argument("--failure-seed", type=int, metavar="S", help="seed of the sampled schedule, the same on every rank (the scripts default to 0 for hsdp and 42 for 3d)")
    f.add_argument("--failure-sampling", choices=("stratified", "iid"),
                   help="stratified (default) puts one failure in each equal slice of the step range. iid draws the steps independently")
    f.add_argument("--failure-exclude-replicas", metavar="IDS",
                   help="comma-separated ids of replicas that are never killed. '' allows all. "
                        "By default the scripts spare replica 0, which hosts the logger")
    f.add_argument("--failure-schedule", metavar="FILE",
                   help="replay a YAML schedule from 'recover schedule' instead of sampling one")

    o = p.add_argument_group("other")
    o.add_argument("--ulfm-verbose", action="store_true",
                   help="verbose logging from the ULFM backend and the control plane")
    o.add_argument("--dry-run", action="store_true",
                   help="print the environment and the mpirun command instead of running them")
    o.add_argument("--script-help", action="store_true",
                   help="show the training script's own options and exit")


def build_parser():
    top = argparse.ArgumentParser(prog="recover", description=TOP_DESCRIPTION, epilog=TOP_EPILOG,
                                  formatter_class=_Formatter, usage="recover [-h] [--version] <command> ...")
    top.add_argument("--version", action="version", version=f"recover {__version__}")
    sub = top.add_subparsers(dest="command", metavar="<command>", title="commands")

    h = sub.add_parser("hsdp", help="HSDP training (PyTorch FSDP HYBRID_SHARD), runs train/main_hsdp.py", description=HSDP_DESCRIPTION,
                       epilog=HSDP_EPILOG, formatter_class=_Formatter,
                       usage="recover hsdp [options] [-- TRAINING_OPTIONS ...]")
    hg = h.add_argument_group("training")
    hg.add_argument("--config", metavar="YAML",
                    help="training options as a YAML mapping (option: value), e.g. examples/8xH100/hsdp_llama_60m.yaml")
    hg.add_argument("--shard-size", type=int, metavar="N",
                    help="GPUs per replica (--hsdp_shard_size); -n must be a multiple of it")
    hg.add_argument("--dataset", metavar="DIR",
                    help="pre-tokenized dataset saved with datasets.save_to_disk (sets --offline_mode --dataset_path)")
    hg.add_argument("--nccl-baseline", action="store_true",
                    help="run the same training on NCCL with torchrun and without fault tolerance: the "
                         "failure-free reference. It runs on one node, needs a shard size of 2 or more and "
                         "takes no failure options")
    _add_common(h, "hsdp")

    t = sub.add_parser("3d", help="3D-parallel training on Nanotron, runs train/main_3d.py",
                       description=THREED_DESCRIPTION, epilog=THREED_EPILOG, formatter_class=_Formatter,
                       usage="recover 3d [options] [-- TRAINING_OPTIONS ...]")
    tg = t.add_argument_group("training")
    tg.add_argument("--config", metavar="YAML", help="Nanotron config file (--config-file), e.g. examples/8xH100/3d_llama_60m.yaml")
    tg.add_argument("--dp", type=int, metavar="N", help="override the config's data-parallel size")
    _add_common(t, "3d")

    sub.add_parser("schedule", help="write a failure schedule (YAML) for --failure-schedule",
                   add_help=False)
    sub.add_parser("env", help="check the installation: MPI with ULFM, PyTorch, GPUs, data", description="Check the installation: Open MPI with "
                   "ULFM, PyTorch with the ulfm backend, GPUs, the container's ptrace capability and the dataset.",
                   formatter_class=_Formatter)
    return top


# --------------------------------------------------------------------------- helpers

def _split_dashdash(argv):
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def _resolve(path, *roots):
    """A path as given if it exists, else under the first root where it exists, else as given."""
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return str(p)
    for root in roots:
        if (root / p).exists():
            return str(root / p)
    return str(p)


def _load_yaml(path):
    try:
        import yaml
    except ImportError as e:
        raise _Error("--config needs PyYAML (pip install pyyaml)") from e
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise _Error(f"{path}: expected a YAML mapping of option: value")
    return data


def _yaml_to_flags(data):
    """{'batch_size': 64, 'offline_mode': True, 'failure_step_range': [5, 10]} -> main_hsdp.py flags."""
    flags = []
    for key, value in data.items():
        flag = "--" + str(key).lstrip("-")
        if value is None or value is False:
            continue
        if value is True:
            flags.append(flag)
        elif isinstance(value, (list, tuple)):
            flags += [flag] + [str(v) for v in value]
        else:
            flags += [flag, str(value)]
    return flags


def _location_pairs(items):
    """'forward' -> 'forward:1'; 'grad-sync:first:0.5' stays. Checks the names."""
    pairs = []
    for item in items:
        name, weight = item, "1"
        head, sep, tail = item.rpartition(":")
        if sep:
            try:
                float(tail)
                name, weight = head, tail
            except ValueError:
                pass
        if LOCATION_ALIASES.get(name, name) not in LOCATIONS:
            aliases = f" (aliases: {', '.join(LOCATION_ALIASES)})" if LOCATION_ALIASES else ""
            raise _Error(f"unknown failure location {name!r}; expected one of {', '.join(LOCATIONS)}{aliases}")
        pairs.append(f"{name}:{weight}")
    return pairs


def _failure_flags(a, dash):
    """The training script's failure-injection flags (main_hsdp.py uses '_', Nanotron '-')."""
    def flag(name):
        return "--failure" + ("-" if dash else "_") + name.replace("-", "-" if dash else "_")
    count = a.failure_count or 0
    if a.failure_schedule and count > 0:
        raise _Error("--failure-schedule and --failure-count are mutually exclusive")
    if count > 0 and (a.failure_step_range is None or not a.failure_locations):
        raise _Error("--failure-count needs --failure-step-range START END and --failure-locations")
    if a.failure_step_range and a.failure_step_range[0] >= a.failure_step_range[1]:
        raise _Error("--failure-step-range START END needs START < END")
    out = []
    if a.failure_schedule:
        out += [flag("schedule"), _resolve(a.failure_schedule)]
    if count > 0:
        out += [flag("count"), str(count), flag("step-range")] + [str(s) for s in a.failure_step_range]
        out += [flag("locations")] + _location_pairs(a.failure_locations)
    if a.failure_seed is not None:
        out += [flag("seed"), str(a.failure_seed)]
    if a.failure_sampling:
        out += [flag("sampling"), a.failure_sampling]
    if a.failure_exclude_replicas is not None:
        out += [flag("exclude-replicas"), a.failure_exclude_replicas]
    return out


def _environment(a, kind):
    """(variables to set, names to export with -x)."""
    pp = getattr(a, "pp_timeout", None)
    inner = max(a.pg_timeout, pp or 0)
    if not a.no_timeout_check and a.mp_timeout <= inner + 4 * a.dump_wait_ms / 1000 + 5:
        raise _Error(
            f"--mp-timeout {a.mp_timeout} is too short. It must be longer than the NCCL timeout inside a "
            f"replica ({inner} s) + 4 x --dump-wait-ms ({4 * a.dump_wait_ms / 1000:g} s) + 5 s for the "
            f"processes to exit. Otherwise a healthy replica can time out while the failed replica's ranks "
            f"are still exiting. Raise --mp-timeout, or pass --no-timeout-check.")
    env = {
        "MASTER_ADDR": a.master_addr,
        "MASTER_PORT": str(a.master_port),
        "TORCH_NCCL_DUMP_ON_TIMEOUT": "0",
        "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC": str(a.dump_wait_ms),
        "NCCL_PG_TIMEOUT_SECONDS": str(a.pg_timeout),
        "NCCL_MP_TIMEOUT_SECONDS": str(a.mp_timeout),
    }
    if kind == "3d":
        env["NCCL_PP_TIMEOUT_SECONDS"] = str(pp)
        env["TORCH_NCCL_PROPAGATE_ERROR"] = "0"
    exports = list(env)
    for item in a.env:
        name, sep, value = item.partition("=")
        if not name.isidentifier():
            raise _Error(f"--env {item!r}: not a variable name")
        if sep:
            env[name] = value
        elif name not in os.environ:
            raise _Error(f"--env {name}: not set in the environment; use --env {name}=VALUE")
        if name not in exports:
            exports.append(name)
    for name in sorted(os.environ):
        if (name.startswith(PASSTHROUGH_PREFIXES) or name in PASSTHROUGH_NAMES) and name not in exports:
            exports.append(name)
    return env, exports


def _mpirun(a, exports):
    if not a.nproc or a.nproc < 1:
        raise _Error("-n/--nproc N is required (one process per GPU)")
    cmd = [a.mpirun, "-np", str(a.nproc), "--with-ft", "ulfm"]
    if a.hostfile:
        cmd += ["--hostfile", a.hostfile]
    if a.map_by:
        cmd += ["--map-by", a.map_by]
    if a.bind_to:
        cmd += ["--bind-to", a.bind_to]
    cmd += ["--mca", "pml", "ob1"]
    if a.btl:
        cmd += ["--mca", "btl", a.btl]
    for extra in a.mpi_arg:
        cmd += shlex.split(extra)
    for name in exports:
        cmd += ["-x", name]
    return cmd


def _run(a, env_set, cmd, cwd=None):
    shown = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_set.items())
    if a.dry_run:
        print(f"export {shown}")
        if cwd:
            print(f"cd {shlex.quote(str(cwd))}")
        print(shlex.join(cmd))
        return 0
    if shutil.which(cmd[0]) is None:
        raise _Error(f"{cmd[0]} not found; put Open MPI built with ULFM on PATH (the container has it), or pass --mpirun")
    print(f"recover: {shown}", file=sys.stderr)
    print(f"recover: {shlex.join(cmd)}", file=sys.stderr, flush=True)
    env = dict(os.environ, **env_set)
    if cwd:
        os.chdir(cwd)
    os.execvpe(cmd[0], cmd, env)


def _script_help(a, script, cwd=None):
    return subprocess.call([a.python, str(script), "--help"], cwd=cwd)


# --------------------------------------------------------------------------- commands

def _last_value(tokens, flag):
    """The value of the last `flag VALUE` or `flag=VALUE` in tokens (argparse keeps the last one), or None."""
    value = None
    for i, tok in enumerate(tokens):
        if tok == flag and i + 1 < len(tokens):
            value = tokens[i + 1]
        elif tok.startswith(flag + "="):
            value = tok[len(flag) + 1:]
    return value


def cmd_hsdp(a, script_args):
    if a.script_help:
        return _script_help(a, HSDP_SCRIPT, cwd=HOME)
    flags = []
    if a.config:
        cfg = _resolve(a.config, HOME)
        data = _load_yaml(cfg)
        if "model_config" in data and data["model_config"]:
            data["model_config"] = _resolve(str(data["model_config"]), Path(cfg).parent, HOME)
        flags += _yaml_to_flags(data)
    if a.shard_size:
        if a.nproc and a.nproc % a.shard_size:
            raise _Error(f"-n {a.nproc} is not a multiple of --shard-size {a.shard_size}")
        flags += ["--hsdp_shard_size", str(a.shard_size)]
    if a.dataset:
        flags += ["--offline_mode", "--dataset_path", a.dataset]
    env_set, exports = _environment(a, "hsdp")
    if a.nccl_baseline:
        if a.failure_count or a.failure_schedule:
            raise _Error("--nccl-baseline runs without fault tolerance; drop the failure-injection options")
        if a.hostfile:
            raise _Error("--nccl-baseline runs on one node; drop --hostfile")
        if not a.nproc or a.nproc < 1:
            raise _Error("-n/--nproc N is required (one process per GPU)")
        if a.nproc > 1 and (_last_value(flags + script_args, "--hsdp_shard_size") or "").strip() == "1":
            raise _Error("--nccl-baseline needs a shard size of 2 or more. With one GPU per replica, FSDP1 "
                         "replaces HYBRID_SHARD with NO_SHARD and no longer reduces gradients across "
                         "replicas (pytorch/pytorch#154888)")
        cmd = [a.python, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={a.nproc}",
               f"--master-port={a.master_port}", str(HSDP_SCRIPT), "--backend", "nccl"] + flags + script_args
        return _run(a, env_set, cmd)
    flags += _failure_flags(a, dash=False)
    if a.ulfm_verbose:
        flags.append("--ulfm_verbose")
    cmd = _mpirun(a, exports) + [a.python, str(HSDP_SCRIPT), "--backend", "ulfm"] + flags + script_args
    return _run(a, env_set, cmd)


def _nanotron_world(config_path, dp_override):
    try:
        par = _load_yaml(config_path).get("parallelism") or {}
        return (dp_override or int(par["dp"])) * int(par["tp"]) * int(par["pp"]), par
    except (_Error, KeyError, TypeError, ValueError):
        return None, None


def cmd_3d(a, script_args):
    if a.script_help:
        return _script_help(a, NANOTRON_SCRIPT, cwd=HOME)
    if not a.config:
        raise _Error("--config YAML (a Nanotron config) is required")
    cfg = _resolve(a.config, HOME)
    if not Path(cfg).exists():
        raise _Error(f"config not found: {a.config}")
    world, par = _nanotron_world(cfg, a.dp)
    if world and a.nproc and world != a.nproc:
        raise _Error(f"{a.config} needs dp x tp x pp = {a.dp or par['dp']} x {par['tp']} x {par['pp']} = {world} "
                     f"processes, but -n is {a.nproc}")
    flags = ["--config-file", str(Path(cfg).resolve())]
    if a.dp:
        flags += ["--dp", str(a.dp)]
    flags += _failure_flags(a, dash=True)
    if a.ulfm_verbose:
        flags.append("--ulfm-verbose")
    env_set, exports = _environment(a, "3d")
    cmd = _mpirun(a, exports) + [a.python, str(NANOTRON_SCRIPT)] + flags + script_args
    return _run(a, env_set, cmd)


def cmd_schedule(argv):
    return subprocess.call([sys.executable, str(SCHEDULE_SCRIPT), *argv])


def cmd_env(_a):
    rows = []

    def row(status, what, detail):
        rows.append((status, what, detail))

    mpirun = shutil.which("mpirun")
    if mpirun:
        ver = subprocess.run([mpirun, "--version"], capture_output=True, text=True).stdout.splitlines()
        row("ok", "mpirun", f"{mpirun} ({ver[0] if ver else '?'})")
        info = subprocess.run(["ompi_info"], capture_output=True, text=True).stdout if shutil.which("ompi_info") else ""
        ext = next((l.split(":", 1)[1].strip() for l in info.splitlines() if "MPI extensions" in l), "")
        row("ok" if "ftmpi" in ext else "FAIL", "ULFM", f"MPI extensions: {ext or 'unknown'}")
    else:
        row("FAIL", "mpirun", "not on PATH")
    try:
        import torch
        import torch.distributed as dist
        row("ok", "PyTorch", f"{torch.__version__}, CUDA {torch.version.cuda}")
        ulfm = "ulfm" in getattr(dist.Backend, "backend_list", [])
        row("ok" if ulfm else "FAIL", "ulfm backend", "registered in torch.distributed" if ulfm else
            "not registered: build PyTorch from pytorch/ of this repository")
        n = torch.cuda.device_count()
        row("ok" if n else "FAIL", "GPUs", f"{n} visible" + (f", NCCL {'.'.join(map(str, torch.cuda.nccl.version()))}" if n else ""))
    except Exception as e:  # noqa: BLE001 - report, do not crash
        row("FAIL", "PyTorch", repr(e))
    for mod, what in (("ulfm_collectives._C", "ReCoVer extension"), ("nanotron", "Nanotron")):
        try:
            __import__(mod)
            row("ok", what, f"import {mod}")
        except Exception as e:  # noqa: BLE001
            row("FAIL", what, f"import {mod}: {e}")
    try:
        cap = int(next(l for l in open("/proc/self/status") if l.startswith("CapEff")).split()[1], 16)
        has = bool(cap & (1 << 19))
        in_container = Path("/.dockerenv").exists()
        row("ok" if has or not in_container else "warn", "CAP_SYS_PTRACE",
            "present" if has else ("missing: start the container with --cap-add SYS_PTRACE" if in_container
                                   else "not granted (not in a container)"))
    except (OSError, StopIteration, ValueError):
        pass
    data = os.environ.get("RECOVER_DATASET_PATH") or DEFAULT_DATASET
    row("ok" if Path(data).exists() else "warn", "dataset", f"{data} ({'found' if Path(data).exists() else 'not found'})")
    width = max(len(r[1]) for r in rows)
    for status, what, detail in rows:
        print(f"[{status:>4}] {what:<{width}}  {detail}")
    return 1 if any(r[0] == "FAIL" for r in rows) else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if argv and argv[0] == "schedule":
        sys.exit(cmd_schedule(argv[1:]))
    own, script_args = _split_dashdash(argv)
    a = parser.parse_args(own)
    if a.command is None:
        parser.print_help()
        sys.exit(2)
    try:
        if a.command == "hsdp":
            sys.exit(cmd_hsdp(a, script_args))
        if a.command == "3d":
            sys.exit(cmd_3d(a, script_args))
        if script_args:
            parser.error(f"'recover {a.command}' takes no arguments after '--'")
        sys.exit(cmd_env(a))
    except _Error as e:
        parser.exit(2, f"recover {a.command}: error: {e}\n")
