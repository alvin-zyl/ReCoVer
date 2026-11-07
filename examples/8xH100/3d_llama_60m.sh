#!/usr/bin/env bash
# 3D-parallel training of Llama 60M on 8 GPUs (DP=2, TP=2, PP=2) with ReCoVer on Nanotron.
# One replica (a TP x PP group of 4 GPUs) is killed during the gradient synchronization of a
# step between 5 and 9. The other replica takes over its microbatches, and training continues.
# Extra arguments go to `recover 3d` (e.g. --dry-run).
# The barrier inside each replica times out after 60 s instead of the default 160 s. That is
# enough on one server, and it shortens the pause after the failure (docker/README.md#timeouts).
set -euo pipefail
cd "$(dirname "$0")/../.."
recover 3d -n 8 --config examples/8xH100/3d_llama_60m.yaml --mp-timeout 60 \
    --failure-count 1 --failure-step-range 5 10 --failure-locations grad-sync:first "$@"
