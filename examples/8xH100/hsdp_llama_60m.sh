#!/usr/bin/env bash
# HSDP training of Llama 60M on 4 GPUs (2 replicas x 2 GPUs) with ReCoVer. One replica is killed
# during the gradient synchronization of a step between 100 and 199. The other replica takes over
# its microbatches, and training continues. Extra arguments go to `recover hsdp` (e.g. --dry-run).
# The barrier inside each replica times out after 60 s instead of the default 160 s. That is
# enough on one server, and it shortens the pause after the failure (docker/README.md#timeouts).
set -euo pipefail
cd "$(dirname "$0")/../.."
recover hsdp -n 4 --config examples/8xH100/hsdp_llama_60m.yaml --mp-timeout 60 \
    --failure-count 1 --failure-step-range 100 200 --failure-locations grad-sync:first "$@"
