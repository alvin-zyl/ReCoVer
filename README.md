<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img src="assets/logo-light.svg" alt="ReCoVer" width="440">
  </picture>
</p>

<p align="center">
  <b><ins>Re</ins>silient LLM Pre-Training System via Fault-Tolerant <ins>Co</ins>llective and <ins>Ver</ins>satile Workload</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.11215"><img src="https://img.shields.io/badge/arXiv-2605.11215-b31b1b.svg" alt="arXiv"></a>
  <a href="https://alvin-zyl.github.io/recover/"><img src="https://img.shields.io/badge/Project-Page-1b4d90.svg" alt="Project page"></a>
  <img src="https://img.shields.io/badge/NeurIPS-2026-c08a20.svg" alt="NeurIPS 2026">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-647285.svg" alt="License"></a>
</p>

---

ReCoVer keeps LLM pre-training running through GPU and node failures. It needs no checkpoint-restart and does not change the training trajectory. It keeps one invariant: each iteration commits gradients from the same number of microbatches as the failure-free run, no matter when or where failures occur. Each update therefore stays statistically the same. Three decoupled layers keep this invariant:

- **Fault-tolerant collectives.** A guarded all-reduce and a consensus barrier, built on ULFM (User-Level Failure Mitigation), detect a failed rank and shrink the communicator to the survivors. All survivors agree on what failed, so the failure is never fatal.
- **Recovery inside the failing iteration.** Each gradient bucket is saved before it is reduced. After a failure, only the buckets that were reduced together with the failed replica are restored and reduced again among the survivors. The rest of the work done in the iteration is kept.
- **Versatile workload.** Survivors take major, minor or spare roles. The policy redistributes the microbatches so that the global batch stays the same.

The protocol works with 3D parallelism (on [Nanotron](https://github.com/huggingface/nanotron)) and with HSDP (PyTorch FSDP). Communication inside a replica stays on NCCL. Only the gradient all-reduce across replicas goes through the guarded collectives.

In a 7B run that lost 256 of its 512 GPUs, the loss curve stays on the failure-free one. Compared with checkpoint-restart, ReCoVer reaches 2.23× the effective throughput after eight failures and processes 74.9% more tokens within 234 GPU-hours. See the [paper](https://arxiv.org/abs/2605.11215) and the [project page](https://alvin-zyl.github.io/recover/).

## Repository layout

```
recover/            the `recover` command: launches training, checks the installation
ulfm_collectives/   control plane: per-iteration orchestrator, policies, comm hooks, failure simulator
csrc/               ProcessGroupULFM, the ULFM-aware torch.distributed backend (C++)
train/              training scripts: main_hsdp.py (HSDP), main_3d.py (3D parallelism on Nanotron)
configs/            model configs
examples/           ready-to-run scripts and configs per platform: 8xH100/, polaris/
scripts/            failure-schedule generator
tests/              unit tests
nanotron/           Nanotron with ReCoVer's 3D-parallel trainer
pytorch/            PyTorch 2.9 (development build) with the "ulfm" backend and the deferred FSDP hook path
docker/             container that rebuilds the software stack of our 8x H100 server
setup.py            builds the C++ backend and installs ulfm_collectives and the `recover` command
```

## Tested hardware

**Our server.** We developed and tested ReCoVer on one server with:

- 8 NVIDIA H100 NVL GPUs with 94 GB of memory each. NVLink bridges connect the GPUs in pairs, and PCIe Gen5 connects the pairs.
- 2 AMD EPYC 9654 CPUs with 96 cores each, and 1.5 TB of memory.
- Ubuntu 22.04 and NVIDIA driver 570.207. [Our container](docker/) has the rest of the software.

**Polaris.** The paper's experiments ran on up to 128 nodes (512 GPUs) of Polaris at the Argonne Leadership Computing Facility (ALCF). Each node has:

- 4 NVIDIA A100 GPUs with 40 GB of memory each, connected by NVLink.
- 1 AMD EPYC 7543P CPU with 32 cores, and 512 GB of memory.
- 2 HPE Slingshot 11 network adapters.

## Installation

ReCoVer needs:

- Open MPI 5.0.8 built with ULFM (`--with-ft=ulfm`),
- PyTorch from `pytorch/`, built with `USE_MPI=1` against that Open MPI,
- this package, compiled against both.

Our container has all three and is the easiest way to start. Its PyTorch is compiled only for compute capability 9.0, as on our server, so the container needs Hopper GPUs such as the H100. Pull the prebuilt image (a 12 GB download, 20 GB on disk):

```bash
docker pull ghcr.io/alvin-zyl/recover:h100
docker tag ghcr.io/alvin-zyl/recover:h100 recover:h100
```

or build it from the recipe in [`docker/`](docker/):

```bash
git clone https://github.com/alvin-zyl/ReCoVer.git && cd ReCoVer
git submodule update --init --recursive          # PyTorch's third-party submodules
docker build -f docker/Dockerfile -t recover:h100 .
```

The build compiles Open MPI, PyTorch and flash-attn and takes a few hours. [`docker/README.md`](docker/README.md) lists every version and flag. To build without Docker, follow the steps of [`docker/Dockerfile`](docker/Dockerfile):

1. Build Open MPI with `--with-ft=ulfm`.
2. In `pytorch/`, run `pip install -e .` with `USE_MPI=1`.
3. At the repository root, run `MPI_HOME=<Open MPI prefix> pip install -e .`, then `pip install -e nanotron`.

**Data.** The experiments use C4, tokenized with the Llama-2 tokenizer into sequences of 4097 tokens and saved with `datasets.save_to_disk`. The container looks for it at `/data/c4/tokenized/seq_len_4096`. For HSDP, you can instead set `RECOVER_DATASET_PATH` or pass `--dataset` to `recover hsdp`. For 3D, set `hf_dataset_or_datasets` in the Nanotron config.

```bash
docker run --gpus all --ipc=host --shm-size=64g --ulimit core=0 --cap-add SYS_PTRACE -it \
  -v /path/to/seq_len_4096:/data/c4/tokenized/seq_len_4096:ro recover:h100
recover env        # inside the container: checks MPI with ULFM, PyTorch, GPUs and the data
```

Runs with failures need `--ulimit core=0` and `--cap-add SYS_PTRACE`. [`docker/README.md`](docker/README.md#run) explains why.

## Running

[`examples/`](examples/) has ready-to-run scripts and configs per platform:

| Platform | |
|---|---|
| [`examples/8xH100/`](examples/8xH100/) | one server with 8 H100 GPUs, in our container: HSDP and 3D examples with an injected failure, and the failure-free NCCL reference |
| [`examples/polaris/`](examples/polaris/) | the paper's runs on Polaris at ALCF (to be added) |

For example, inside the container:

```bash
cd /opt/recover
examples/8xH100/hsdp_llama_60m.sh     # HSDP on 4 GPUs, one replica killed during gradient synchronization
examples/8xH100/3d_llama_60m.sh       # DP=2, TP=2, PP=2 on 8 GPUs, the same failure
```

The scripts call `recover`, which builds the mpirun command and runs it. The command turns on fault tolerance (`--with-ft ulfm`), sets the NCCL timeouts that recovery needs (see [Timeouts](#timeouts)), and exports the environment to every rank.

| Command | What it runs |
|---|---|
| `recover hsdp` | HSDP training, [`train/main_hsdp.py`](train/main_hsdp.py). Training options come from `--config` (a YAML file of `option: value`) and from the arguments after `--`. `--nccl-baseline` runs the same training on NCCL without fault tolerance. |
| `recover 3d` | 3D-parallel training on Nanotron, [`train/main_3d.py`](train/main_3d.py), with a Nanotron config. |
| `recover schedule` | writes a failure schedule (YAML) for `--failure-schedule`. |
| `recover env` | checks the installation. |

Every command has a help page, e.g. `recover hsdp -h`. `--script-help` shows the training script's own options, and `--dry-run` prints the mpirun command instead of running it. Arguments after `--` go to the training script unchanged and override the config:

```bash
recover hsdp -n 4 --config examples/8xH100/hsdp_llama_60m.yaml -- --num_training_steps 500
```

**Multi-node.** Pass the hostfile, the process mapping and the address of the first host. For example, in a PBS job with four GPUs per node:

```bash
recover 3d -n 32 --hostfile $PBS_NODEFILE --map-by ppr:4:node:PE=8 --bind-to core \
    --master-addr $(head -1 $PBS_NODEFILE) --config my_config.yaml
```

### Failure injection

The failure simulator kills a scheduled rank with `SIGKILL` at a chosen point of a chosen step. To sample a schedule, pass `--failure-count`, `--failure-step-range` and `--failure-locations`. To replay a schedule written by `recover schedule`, pass `--failure-schedule`. The locations are `forward`, `backward`, `grad-sync:first`, `grad-sync:mid`, `grad-sync:last` and `optimizer`, described in [`docker/README.md`](docker/README.md#failure-injection). Each can carry a relative weight, e.g. `forward:0.4 grad-sync:first:0.6`. The paper's runs inject every failure during gradient synchronization (`grad-sync:first`).

### Timeouts

When a rank dies, the other ranks of its replica are still alive. They hang in an NCCL collective with the dead rank until the collective times out, and only then exit. Meanwhile, some ranks of the healthy replicas wait at a barrier inside their own replica. The barrier has its own timeout, `--mp-timeout`. It must be long enough for the failed replica's ranks to exit first. Otherwise a healthy replica times out too. `recover` refuses settings that break this rule.

By default, the NCCL collectives inside a replica time out after 20 s (`--pg-timeout`, and `--pp-timeout` in 3D) and the barrier after 160 s, as in the paper's 3D runs. A failure during gradient synchronization pauses the run for about `--mp-timeout`. On one server, 60 s passed all our tests, so the examples in [`examples/8xH100/`](examples/8xH100/) use `--mp-timeout 60`. [`docker/README.md`](docker/README.md#timeouts) explains the rule and the pauses in detail.

## Reproducing the paper

The paper trained a 7B model with 3D parallelism (TP=4, PP=2) and a 1B model with HSDP (shard size 8) on Polaris ([Tested hardware](#tested-hardware)). Their job scripts and configurations will be added to [`examples/polaris/`](examples/polaris/).

## Citation

```bibtex
@inproceedings{liu2026recover,
  title     = {ReCoVer: Resilient LLM Pre-Training System via Fault-Tolerant Collective and Versatile Workload},
  author    = {Liu, Ziyue and Wang, Zhengyang and Zhang, Ruijie and Maurya, Avinash and Zhou, Hui and
               Hovland, Paul and Di, Sheng and Cappello, Franck and Nicolae, Bogdan and Zhang, Zheng},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

ReCoVer is released under the [Apache License 2.0](LICENSE). It builds on [PyTorch](https://github.com/pytorch/pytorch), [Nanotron](https://github.com/huggingface/nanotron) and [Open MPI](https://www.open-mpi.org/)'s ULFM. `pytorch/` and `nanotron/` keep their own licenses ([`pytorch/LICENSE`](pytorch/LICENSE), [`nanotron/LICENSE`](nanotron/LICENSE)).
