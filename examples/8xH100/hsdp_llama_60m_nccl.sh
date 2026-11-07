#!/usr/bin/env bash
# The failure-free reference for hsdp_llama_60m.sh: the same training on NCCL (torchrun),
# without fault tolerance and without failures. Extra arguments go to `recover hsdp`.
set -euo pipefail
cd "$(dirname "$0")/../.."
recover hsdp -n 4 --config examples/8xH100/hsdp_llama_60m.yaml --nccl-baseline "$@"
