# ReCoVer container

The image has everything ReCoVer needs: Open MPI 5.0.8 with ULFM, PyTorch from
`pytorch/`, flash-attn, Nanotron and this package. It has the same software
versions as our test server, which
[Tested hardware](../README.md#tested-hardware) describes.
[Software in the image](#software-in-the-image) lists them.

PyTorch in the image is compiled only for compute capability 9.0, as on our
server. The image therefore needs Hopper GPUs, such as the H100.

## Build

A prebuilt image of this recipe is `ghcr.io/alvin-zyl/recover:h100`
(`docker pull ghcr.io/alvin-zyl/recover:h100`). To build it yourself:

```bash
git clone https://github.com/alvin-zyl/ReCoVer.git && cd ReCoVer
git submodule update --init --recursive      # PyTorch's third-party submodules
docker build -f docker/Dockerfile -t recover:h100 .
```

The build compiles Open MPI, PyTorch and flash-attn. It needs network
access and takes a few hours. `--build-arg MAX_JOBS=N` sets the number of
parallel compile jobs (default 32). flash-attn needs a lot of memory per job.

## Run

The image does not include the pre-tokenized C4 dataset (sequences of 4097
Llama-2 tokens, saved with `datasets.save_to_disk`). Mount it at
`/data/c4/tokenized/seq_len_4096`, or point `RECOVER_DATASET_PATH` or
`recover hsdp --dataset` to it.

```bash
docker run --gpus all --ipc=host --shm-size=64g --ulimit core=0 --cap-add SYS_PTRACE -it \
  -v /path/to/seq_len_4096:/data/c4/tokenized/seq_len_4096:ro \
  recover:h100
# inside, in /opt/recover:
recover env                                   # checks MPI with ULFM, PyTorch, GPUs, data
recover hsdp -n 4 --config examples/8xH100/hsdp_llama_60m.yaml --mp-timeout 60 \
    --failure-count 1 --failure-step-range 100 200 --failure-locations forward
```

`recover` builds the mpirun command and runs it. Add `--dry-run` to print the
command instead. `recover hsdp -h` and `recover 3d -h` list the options.

**Disable core dumps (`--ulimit core=0`) for runs with failures.** After a
failure, the NCCL watchdog stops the failed replica's remaining ranks with
`abort()`. Docker does not limit the size of core files by default, so each of
these ranks writes a core file of several GB through the host's `core_pattern`
(apport on Ubuntu). A rank cannot be killed while it writes. MPI reports a rank
as failed only after it has exited, so the surviving replicas wait for many
minutes, or forever.

**Add `--cap-add SYS_PTRACE`.** Inside a container, Open MPI's shared-memory
transport needs this capability for single-copy transfers (CMA,
`process_vm_readv`). Without it, the transport falls back to copying through
shared-memory buffers. Open MPI's ULFM documentation lists shared memory as
fault-tolerant when it uses CMA or XPMEM.

## Failure injection

A failure location is the phase of the training step in which the scheduled
rank is killed. Pass locations to `--failure-locations`, each optionally with a
weight (`name:weight`). To replay a fixed schedule instead, write one with
`recover schedule` and pass it to `--failure-schedule`.

| Location | The rank is killed | What recovery undoes |
|---|---|---|
| `forward` | before a microbatch's forward pass | Nothing. No gradient of this step has been reduced across replicas yet. |
| `backward` | in the backward pass, right after a gradient bucket was saved and queued | Usually nothing, because a rank starts its cross-replica reduction only after its own backward pass. With pipeline parallelism, later stages of the failed replica finish backward first and may already have reduced some buckets. Those buckets are rolled back. |
| `grad-sync:first`, `grad-sync:mid`, `grad-sync:last` | in the cross-replica gradient all-reduce, after the first, middle or last bucket completed | The buckets reduced so far contain the failed replica's gradients. They are rolled back to their saved copies. |
| `optimizer` | after the step's last cross-replica collective (the loss all-reduce), before the local `optimizer.step()` | Nothing. The step is complete, and the failure is detected at the next step. |

The research code's names still work: `pre-forward` (`forward`),
`post-deferred-hook-firing` (`backward`) and `post-allreduce`
(`grad-sync:first`). The paper's runs used `post-allreduce`.

### Timeouts

When a rank dies, the other ranks of its replica are still alive. Each of them
hangs in the next NCCL collective that includes the dead rank until the
collective times out, and only then exits. MPI reports a rank as failed only
after it has exited, so the survivors cannot finish recovery before that.

Meanwhile, some ranks of the healthy replicas wait at a barrier inside their
own replica. In nanotron the barrier is the model-parallel group. In HSDP it is
a group over the replica's shard ranks. If the barrier times out before the
failed replica's ranks have exited, a healthy replica goes down too. The
timeouts must therefore keep this order:

    NCCL_MP_TIMEOUT_SECONDS > NCCL_PG_TIMEOUT_SECONDS (and NCCL_PP_TIMEOUT_SECONDS)
                              + 4 × TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC
                              + process exit time

- `NCCL_MP_TIMEOUT_SECONDS` (`recover --mp-timeout`) is the barrier's timeout.
- `NCCL_PG_TIMEOUT_SECONDS` and `NCCL_PP_TIMEOUT_SECONDS` (`--pg-timeout`,
  `--pp-timeout`) are the timeouts of the TP, FSDP and PP collectives.
- Before PyTorch's NCCL watchdog aborts a timed-out rank, it waits
  `4 × TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC` to leave time for a debug dump. The
  wait is 60 s by default, even with `TORCH_NCCL_DUMP_ON_TIMEOUT=0`. `recover`
  sets the variable to `1000`, which cuts the wait to 4 s.
- Process exit took under a second in most of our tests, once about 2 minutes,
  and several minutes when the ranks wrote core files.

`recover` uses 20 s for the TP, FSDP and PP collectives and 160 s for the
barrier, as in the paper's 3D runs. It refuses settings that break the order
unless you pass `--no-timeout-check`.

How long a failure pauses the run depends on where the failed replica's other
ranks are waiting:

- If they all wait in a TP, PP or FSDP collective, they exit after about 25 s,
  and the failure adds about 30 s to the run. This is the case for `forward`
  and `optimizer`, and for `backward` in HSDP.
- If some of them have already reached the barrier, they exit only when the
  barrier times out, and the failure adds about `NCCL_MP_TIMEOUT_SECONDS`. This
  is the case for every `grad-sync:*` failure, and for `backward` in nanotron,
  where later pipeline stages finish backward first.

We tested every location on one server with a 60 s barrier. The tests covered
2 and 4 replicas, for HSDP (shard size 2) and for nanotron (dp=2 with tp=pp=2,
and dp=4 with tp=1, pp=2). All runs passed, including runs with three failures
at different locations. The examples in [`examples/8xH100/`](../examples/8xH100/)
therefore set `--mp-timeout 60`. This shortens the pause after a failure during
gradient synchronization. After a failure at `forward`, `backward` or
`optimizer`, the shorter barrier also leaves the failed replica's ranks less
time to exit: about 35 s instead of about 135 s. The 2-minute exit mentioned
above would not have finished in time, and a healthy replica would have gone
down too.

## Software in the image

| Component | Version and build options |
|---|---|
| OS and compiler | Ubuntu 22.04, GCC 11.4.0 |
| CUDA | 12.9.1 at run time. PyTorch is compiled with CUDA 12.4.1. |
| Open MPI | 5.0.8 with ULFM (`--with-ft=ulfm`) and CUDA support, in `/opt/openmpi-5.0.8-install`. `--enable-debug` is on, as in the Open MPI builds on our server and on Polaris. |
| NCCL | 2.27.5, built from source and linked statically into PyTorch |
| PyTorch | `pytorch/` of this repository: PyTorch 2.9 (development build) with the `ulfm` backend and the deferred FSDP hook path. Release build with `USE_MPI=1`, for compute capability 9.0 only, without cuDNN. |
| Python | 3.10.12, with the packages in [`requirements/`](requirements/) |
| flash-attn | 2.5.1.post1, built from source for compute capability 8.0 and 9.0 |
| Nanotron | `nanotron/` of this repository |
| ReCoVer | this repository, in `/opt/recover` |
