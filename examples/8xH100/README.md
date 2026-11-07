# One server with 8 NVIDIA H100 GPUs

Ready-to-run examples for one server with 8 H100 GPUs, such as [ours](../../README.md#tested-hardware), using the container from [`docker/`](../../docker/).

## Start the container

Build or pull the image (see the [main README](../../README.md#installation)). Then start it with the pre-tokenized C4 dataset mounted (sequences of 4097 Llama-2 tokens, saved with `datasets.save_to_disk`):

```bash
docker run --gpus all --ipc=host --shm-size=64g --ulimit core=0 --cap-add SYS_PTRACE -it \
  -v /path/to/seq_len_4096:/data/c4/tokenized/seq_len_4096:ro recover:h100
cd /opt/recover
recover env          # checks Open MPI with ULFM, PyTorch, GPUs and the dataset
```

Runs with failures need `--ulimit core=0` and `--cap-add SYS_PTRACE`. [`docker/README.md`](../../docker/README.md#run) explains why.

## Examples

| Script | GPUs | What it runs |
|---|---|---|
| [`hsdp_llama_60m.sh`](hsdp_llama_60m.sh) | 4 | HSDP, Llama 60M, 2 replicas of 2 GPUs. One replica is killed during the gradient synchronization of a step between 100 and 199. |
| [`hsdp_llama_60m_nccl.sh`](hsdp_llama_60m_nccl.sh) | 4 | The same training on NCCL, without failures: the failure-free reference. |
| [`3d_llama_60m.sh`](3d_llama_60m.sh) | 8 | 3D parallelism on Nanotron, Llama 60M, DP=2, TP=2, PP=2. One replica is killed during the gradient synchronization of a step between 5 and 9. |

Each script takes its training settings from the YAML file next to it ([`hsdp_llama_60m.yaml`](hsdp_llama_60m.yaml), [`3d_llama_60m.yaml`](3d_llama_60m.yaml)). Arguments to a script go to `recover`. For example, `--dry-run` prints the mpirun command instead of running it, failure options replace the script's own, and options after `--` go to the training script:

```bash
examples/8xH100/hsdp_llama_60m.sh --dry-run
examples/8xH100/hsdp_llama_60m.sh -- --num_training_steps 300
```

## What happens at the failure

After the failure, the run pauses for about a minute. The other ranks of the killed replica wait at the barrier inside their replica and exit only when the barrier times out, after `--mp-timeout`. The scripts set `--mp-timeout 60` instead of the default 160 s, because 60 s is enough on one server. [Timeouts](../../docker/README.md#timeouts) explains why. When the run resumes, the log shows which ranks the survivors agree have failed, which gradient buckets they restore, and how many extra microbatches they run to keep the global batch. Training then continues, and the loss stays on the failure-free curve.
