# syntax=docker/dockerfile:1
#
# ReCoVer container for single H100 servers.
#
# Rebuilds the software of the 8x H100 server on which we developed and tested
# ReCoVer, with the same versions and build options (see docker/README.md). The
# three stages follow the order in which the software was built on that server:
#   ompi  - Open MPI 5.0.8, built twice (2025-07-17 and 2025-07-23)
#   torch - PyTorch from pytorch/ of this repository, CUDA 12.4.1 (2025-08-04)
#   final - CUDA 12.9 runtime, Python packages, flash-attn, ReCoVer (2026-04)
#
# From the repository root, with submodules checked out:
#   git submodule update --init --recursive
#   docker build -f docker/Dockerfile -t recover:h100 .

ARG CUDA_12_4_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:da6791294b0b04d7e65d87b7451d6f2390b4d36225ab0701ee7dfec5769829f5
ARG CUDA_12_9_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu22.04@sha256:bd4e2680a261c212f1e2fea241606f71497dc67a417f73175d794ec8212b5ba8
# Ubuntu archive snapshots: system packages as of the torch build and of the runs.
ARG APT_SNAPSHOT_BUILD=20250804T000000Z
ARG APT_SNAPSHOT_RUN=20260401T000000Z
ARG MAX_JOBS=32

###############################################################################
FROM ${CUDA_12_9_IMAGE} AS ompi
###############################################################################
ARG APT_SNAPSHOT_BUILD
ARG MAX_JOBS
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN sed -i -E "s#https?://(archive|security)\.ubuntu\.com/ubuntu/?#http://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT_BUILD}#" /etc/apt/sources.list \
 && apt-get -o Acquire::Check-Valid-Until=false update \
 && apt-get install -y --no-install-recommends build-essential perl python3 wget ca-certificates bzip2 file \
 && rm -rf /var/lib/apt/lists/*

ARG OMPI_VERSION=5.0.8
ARG OMPI_SHA256=53131e1a57e7270f645707f8b0b65ba56048f5b5ac3f68faabed3eb0d710e449
WORKDIR /src
RUN wget -q https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-${OMPI_VERSION}.tar.bz2 \
 && echo "${OMPI_SHA256}  openmpi-${OMPI_VERSION}.tar.bz2" | sha256sum -c - \
 && tar -xjf openmpi-${OMPI_VERSION}.tar.bz2 && rm openmpi-${OMPI_VERSION}.tar.bz2

# Build 1 (2025-07-17): a plain build into /usr/local. It brings hwloc 2.7.1,
# libevent 2.1.12, PMIx 5.0.8a1 and PRRTE 3.0.11a1 with FT enabled. Build 2
# links these libraries and launches its processes through this build's prterun.
RUN cp -a openmpi-${OMPI_VERSION} ompi-l1 && cd ompi-l1 \
 && ./configure --with-ft=ulfm \
 && make -j"${MAX_JOBS}" \
 && make install DESTDIR=/stage/l1 \
 && cp -a /stage/l1/usr/local/. /usr/local/ && ldconfig \
 && cd .. && rm -rf ompi-l1 \
 && /usr/local/bin/ompi_info | grep -E 'Configure command|MCA accelerator'

# Build 2 (2025-07-23): the build with ULFM and CUDA that every ReCoVer process
# uses. On our server, /usr/local/cuda-12 pointed to CUDA 12.9 by then.
ENV MPI_HOME=/opt/openmpi-5.0.8-install
RUN cd openmpi-${OMPI_VERSION} \
 && ./configure --with-cuda=/usr/local/cuda-12/targets/x86_64-linux \
      LDFLAGS=-L/usr/local/cuda-12/targets/x86_64-linux/lib \
      CPPFLAGS=-I/usr/local/cuda-12/targets/x86_64-linux/include \
      --with-ft=ulfm --enable-dlopen --prefix="${MPI_HOME}" --enable-debug \
 && make -j"${MAX_JOBS}" && make install \
 && cd .. && rm -rf openmpi-${OMPI_VERSION} \
 && "${MPI_HOME}"/bin/ompi_info | grep -E 'Configure command|Build LIBS|FT MPI support|MCA accelerator' \
 && test ! -e "${MPI_HOME}"/bin/prterun

###############################################################################
FROM ${CUDA_12_4_IMAGE} AS torch
###############################################################################
ARG APT_SNAPSHOT_BUILD
ARG MAX_JOBS
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN sed -i -E "s#https?://(archive|security)\.ubuntu\.com/ubuntu/?#http://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT_BUILD}#" /etc/apt/sources.list \
 && apt-get -o Acquire::Check-Valid-Until=false update \
 && apt-get install -y --no-install-recommends build-essential git ca-certificates \
      python3.10 python3.10-dev python3.10-venv libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ompi /stage/l1/usr/local/ /usr/local/
COPY --from=ompi /opt/openmpi-5.0.8-install /opt/openmpi-5.0.8-install
RUN ldconfig

ENV MPI_HOME=/opt/openmpi-5.0.8-install \
    VIRTUAL_ENV=/opt/envs/mpi-pytorch
ENV PATH=${MPI_HOME}/bin:${VIRTUAL_ENV}/bin:${PATH}
RUN python3.10 -m venv "${VIRTUAL_ENV}"
COPY docker/requirements/01-torch-build.txt /tmp/
RUN pip install --no-deps -r /tmp/01-torch-build.txt

COPY . /opt/recover
WORKDIR /opt/recover/pytorch
# On our server, PyTorch was compiled with Lambda Stack's CUDA 12.4.1 under /usr.
# That CUDA has no CUPTI where PyTorch looks for it, so Kineto was built without
# CUPTI (-DLIBKINETO_NOCUPTI). NVIDIA's image has CUPTI in extras/ and also in the
# main lib and include directories, so remove all three.
RUN rm -rf /usr/local/cuda-12.4/extras/CUPTI \
 && rm -f /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcupti* \
          /usr/local/cuda-12.4/targets/x86_64-linux/include/cupti*.h \
 && ! ls /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcupti* 2>/dev/null
# The build environment of PyTorch on our server. Its GPUs were detected as
# compute capability 9.0.
ENV USE_CUDA=1 USE_DISTRIBUTED=1 USE_MPI=1 USE_CUFILE=0 \
    CMAKE_PREFIX_PATH=/opt/openmpi-5.0.8-install: \
    CC=mpicc CXX=mpicxx \
    CUDA_HOME=/usr/local/cuda-12.4 \
    TORCH_CUDA_ARCH_LIST=9.0
# PyTorch takes the +git suffix of its version from pytorch/.git, which does not
# exist here. Take it from this repository instead (2.9.0a0+git<commit>).
RUN PYTORCH_BUILD_VERSION="$(cat version.txt)+git$(git -C /opt/recover rev-parse --short=7 HEAD)" \
    PYTORCH_BUILD_NUMBER=0 MAX_JOBS="${MAX_JOBS}" LD_LIBRARY_PATH="${MPI_HOME}/lib" \
      python -m pip install --no-build-isolation -v -e . 2>&1 | tee /opt/torch-build.log \
 && grep -q 'Could not find CUPTI library, using CPU-only Kineto build' /opt/torch-build.log \
 && rm -rf /opt/recover/pytorch/build \
 && python -c "import torch; print(torch.__config__.show())"

###############################################################################
FROM ${CUDA_12_9_IMAGE} AS final
###############################################################################
ARG APT_SNAPSHOT_RUN
ARG MAX_JOBS
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN sed -i -E "s#https?://(archive|security)\.ubuntu\.com/ubuntu/?#http://snapshot.ubuntu.com/ubuntu/${APT_SNAPSHOT_RUN}#" /etc/apt/sources.list \
 && apt-get -o Acquire::Check-Valid-Until=false update \
 && apt-get install -y --no-install-recommends build-essential git ca-certificates openssh-client \
      python3.10 python3.10-dev python3.10-venv libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ompi /stage/l1/usr/local/ /usr/local/
COPY --from=ompi /opt/openmpi-5.0.8-install /opt/openmpi-5.0.8-install
COPY --from=torch /opt/envs/mpi-pytorch /opt/envs/mpi-pytorch
COPY --from=torch /opt/recover /opt/recover
COPY --from=torch /opt/torch-build.log /opt/torch-build.log
RUN ldconfig

ENV MPI_HOME=/opt/openmpi-5.0.8-install \
    VIRTUAL_ENV=/opt/envs/mpi-pytorch
ENV PATH=${MPI_HOME}/bin:${VIRTUAL_ENV}/bin:${PATH}
COPY docker/requirements/02-runtime.txt /tmp/
RUN pip install --no-deps -r /tmp/02-runtime.txt

# flash-attn was compiled from source against this torch on 2026-04-01.
RUN FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS}" \
      pip install --no-build-isolation --no-deps flash-attn==2.5.1.post1

WORKDIR /opt/recover
RUN FORCE_CUDA=1 pip install --no-build-isolation --no-deps -e . \
 && pip install --no-build-isolation --no-deps -e nanotron \
 && python -c "import torch, ulfm_collectives, nanotron, flash_attn; print(torch.__version__, torch.version.cuda)"

LABEL org.opencontainers.image.source="https://github.com/alvin-zyl/ReCoVer" \
      org.opencontainers.image.description="ReCoVer: fault-tolerant LLM pre-training on Open MPI ULFM (H100 stack)" \
      org.opencontainers.image.licenses="Apache-2.0"
ENV OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    RECOVER_DATASET_PATH=/data/c4/tokenized/seq_len_4096
