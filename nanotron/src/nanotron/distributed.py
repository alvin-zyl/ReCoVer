import datetime
import os
from functools import cache, lru_cache
from typing import List, Optional, Tuple

import torch
from packaging import version
from torch import distributed as dist
from torch.distributed import *  # noqa
from torch.distributed.distributed_c10d import ProcessGroup

from nanotron.utils import find_free_port

torch_version_above_1_13 = version.parse(torch.__version__) >= version.parse("1.13.0")
Work = dist.Work if torch_version_above_1_13 else dist._Work
default_pg_timeout = datetime.timedelta(minutes=10)


def new_group(  # pylint: disable=function-redefined
    ranks=None, timeout=default_pg_timeout, backend=None, pg_options=None
) -> ProcessGroup:
    if len(ranks) == 0:
        raise ValueError("Cannot create a group with not ranks inside it")

    return dist.new_group(ranks=ranks, timeout=timeout, backend=backend, pg_options=pg_options)


def reduce_scatter_tensor(  # pylint: disable=function-redefined
    output: torch.Tensor,
    input: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[Work]:
    if group is None:
        group = dist.torch_dist.distributed_c10d._get_default_group()

    assert (
        group.size() > 1
    ), "You should probably not call `reduce_scatter_tensor` with a single rank, as it copies data over"

    if torch_version_above_1_13:
        return dist.reduce_scatter_tensor(output=output, input=input, group=group, op=op, async_op=async_op)
    else:
        # Support pytorch 1.12
        return dist._reduce_scatter_base(output=output, input=input, group=group, op=op, async_op=async_op)


def all_gather_into_tensor(  # pylint: disable=function-redefined
    output_tensor, input_tensor, group: Optional[ProcessGroup] = None, async_op: bool = False
) -> Optional[Work]:
    if group is None:
        group = dist.torch_dist.distributed_c10d._get_default_group()

    assert (
        group.size() > 1
    ), "You should probably not call `all_gather_into_tensor` with a single rank, as it copies data over"

    if torch_version_above_1_13:
        return dist.all_gather_into_tensor(
            output_tensor=output_tensor, input_tensor=input_tensor, group=group, async_op=async_op
        )
    else:
        # Support Pytorch 1.12
        return dist.distributed_c10d._all_gather_base(
            output_tensor=output_tensor, input_tensor=input_tensor, group=group, async_op=async_op
        )


def reduce_scatter_coalesced(
    output_tensor_list: List[torch.Tensor],
    input_tensor_lists: List[List[torch.Tensor]],
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[torch._C.Future]:
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    Args:
        output_tensor_list (list[Tensor]): Output tensor.
        input_tensor_lists (list[list[Tensor]]): List of tensors to reduce and scatter.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    assert len(output_tensor_list) > 0
    assert len(input_tensor_lists) == len(output_tensor_list)
    device = output_tensor_list[0].device
    dtype = output_tensor_list[0].dtype
    group_size = len(input_tensor_lists[0])

    assert (
        group_size > 1
    ), "You should probably not call `reduce_scatter_coalesced` with a single rank, as it copies data over"

    for output_tensor in output_tensor_list:
        assert device == output_tensor.device
        assert dtype == output_tensor.dtype

    for input_tensor_list in input_tensor_lists:
        assert len(input_tensor_list) == group_size, f"Expected {len(input_tensor_list)} == {group_size}"
        for input_tensor in input_tensor_list:
            assert device == input_tensor.device
            assert dtype == input_tensor.dtype

    output_tensor_buffer = torch._utils._flatten_dense_tensors(output_tensor_list)
    input_tensor_buffer_list = [
        torch._utils._flatten_dense_tensors(
            [input_tensor_list[group_rank] for input_tensor_list in input_tensor_lists]
        )
        for group_rank in range(group_size)
    ]

    work = dist.reduce_scatter(output_tensor_buffer, input_tensor_buffer_list, op=op, group=group, async_op=async_op)

    def update_output():
        for original_buffer, reduced_buffer in zip(
            output_tensor_list, torch._utils._unflatten_dense_tensors(output_tensor_buffer, output_tensor_list)
        ):
            original_buffer.copy_(reduced_buffer)

    if async_op is True:
        return work.get_future().then(lambda fut: update_output())
    else:
        # No need to run `work.wait()` since `dist.reduce_scatter` already waits
        update_output()


def all_reduce_coalesced(  # pylint: disable=function-redefined
    tensors: List[torch.Tensor],
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[torch._C.Future]:
    if group is None:
        group = dist.torch_dist.distributed_c10d._get_default_group()

    if group.size() == 1:
        return

    return dist.all_reduce_coalesced(tensors, op=op, group=group, async_op=async_op)


def all_gather_coalesced(  # pylint: disable=function-redefined
    output_tensor_lists: List[List[torch.Tensor]],
    input_tensor_list: List[torch.Tensor],
    group: Optional[ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[torch._C.Future]:
    """
    `torch` has a deprecated version of this method that doesn't work over NCCL.
    All gathers a list of tensors to all processes in a group.

    Args:
        output_tensor_lists (list[list[Tensor]]): Output tensor.
        input_tensor_list (list[Tensor]): List of tensors to all_gather from.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    assert len(output_tensor_lists) > 0
    assert len(input_tensor_list) == len(output_tensor_lists)
    device = input_tensor_list[0].device
    dtype = input_tensor_list[0].dtype
    group_size = len(output_tensor_lists[0])

    assert (
        group_size > 1
    ), "You should probably not call `all_gather_coalesced` with a single rank, as it copies data over"

    for input_tensor in input_tensor_list:
        assert device == input_tensor.device
        assert dtype == input_tensor.dtype

    for output_tensor_list in output_tensor_lists:
        assert len(output_tensor_list) == group_size
        for output_tensor in output_tensor_list:
            assert device == output_tensor.device
            assert dtype == output_tensor.dtype

    # Invert from `[param_idx][group_rank]` to `[group_rank][param_idx]`
    output_tensor_lists = [
        [output_tensor_list[group_rank] for output_tensor_list in output_tensor_lists]
        for group_rank in range(group_size)
    ]

    input_tensor_buffer = torch._utils._flatten_dense_tensors(input_tensor_list)
    output_tensor_buffer_list = [
        torch._utils._flatten_dense_tensors(output_tensor_list) for output_tensor_list in output_tensor_lists
    ]

    work = dist.all_gather(output_tensor_buffer_list, input_tensor_buffer, group=group, async_op=async_op)

    def update_output():
        for original_buffer_list, gathered_buffer_tensor in zip(output_tensor_lists, output_tensor_buffer_list):
            for original_buffer, gathered_buffer in zip(
                original_buffer_list,
                torch._utils._unflatten_dense_tensors(gathered_buffer_tensor, original_buffer_list),
            ):
                original_buffer.copy_(gathered_buffer)

    if async_op is True:
        return work.get_future().then(lambda fut: update_output())
    else:
        # No need to run `work.wait()` since `dist.reduce_scatter` already waits
        update_output()


# This cache has a speedup of 4 tflops on a 7b model
@cache
def get_global_rank(group: ProcessGroup, group_rank: int) -> int:  # pylint: disable=function-redefined
    if torch_version_above_1_13:
        return dist.get_global_rank(group, group_rank=group_rank)
    else:
        # Support pytorch 1.12
        return dist.distributed_c10d._get_global_rank(group=group, rank=group_rank)


def get_global_ranks(group: ProcessGroup) -> Tuple[int]:
    return tuple(sorted((get_global_rank(group, i) for i in range(group.size()))))


# We cache for dp, pp, tp process groups, world group, and tied process group for tied params
@lru_cache
def get_rank(group: Optional[ProcessGroup] = None) -> int:  # pylint: disable=function-redefined
    """Similar to `get_rank` except we raise an exception instead of return -1 when current rank is not part of the group"""
    result = dist.get_rank(group)
    if result == -1:
        raise RuntimeError("Can not call `get_rank` on a group in which current process is not a part of")
    return result


def initialize_torch_distributed():
    """Initializes torch distributed with the environment variables"""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        # Set the device id.
        # `torch.cuda.device_count` should return the number of device on a single node.
        # We assume the nodes to be homogeneous (same number of gpus per node)
        device_id = local_rank
        torch.cuda.set_device(torch.cuda.device(device_id))
        backend = "nccl"
    else:
        # TODO @thomasw21: Maybe figure out a way to do distributed `cpu` training at some point
        raise NotImplementedError(f"CUDA was not found: torch.cuda.is_available(): {torch.cuda.is_available()}")
        backend = "gloo"

    # Call the init process.

    port = os.getenv("MASTER_PORT")
    if port is None:
        port = find_free_port()
    else:
        port = int(port)

    init_method = f"env://localhost:{port}"
    dist.init_process_group(
        init_method=init_method, backend=backend, world_size=world_size, rank=rank, timeout=dist.default_pg_timeout
    )
    return True


def initialize_torch_distributed_ulfm():
    """Initialize torch.distributed with the ULFM (MPI) backend.

    Requires MASTER_ADDR and MASTER_PORT to be set (e.g. via the launch command),
    because NCCL subgroup creation needs a TCPStore that implements set()/get().

    PyTorch hardcodes a Store() placeholder for the ULFM backend at
    distributed_c10d.py:1735, silently ignoring any store= argument.
    We work around this by monkey-patching _world.pg_map after init to replace the
    placeholder with the real TCPStore, so NCCL new_group() calls can exchange
    ncclUniqueIds via store.set()/get().

    Launch example:
        recover 3d -n N --config CONFIG   (runs train/main_3d.py under mpirun)
    """
    dist.init_process_group(backend="ulfm")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # rank       = int(os.getenv("OMPI_COMM_WORLD_RANK",       os.getenv("RANK",       "0")))
    # world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE",       os.getenv("WORLD_SIZE", "1")))
    local_rank = int(os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", os.getenv("LOCAL_RANK", "0")))
    torch.cuda.set_device(local_rank)

    master_addr = os.getenv("MASTER_ADDR", "localhost")
    master_port = int(os.getenv("MASTER_PORT", "29500"))

    # Avoid thundering herd: after init_process_group (an MPI collective), all
    # ranks are tightly synchronized and would blast TCP connections at rank 0
    # simultaneously, overwhelming the TCP listen backlog.  Instead:
    #   1. Rank 0 starts the server with wait_for_workers=False (returns immediately)
    #   2. MPI barrier ensures the server is up before workers connect
    #   3. Workers connect (still bursty, but the server is already accepting)
    #   4. Final MPI barrier ensures all workers are connected before proceeding
    if rank == 0:
        print(f"[Rank {rank}] Starting TCP store server at {master_addr}:{master_port}")
        store = dist.TCPStore(
            host_name=master_addr,
            port=master_port,
            world_size=world_size,
            is_master=True,
            timeout=dist.default_pg_timeout,
            wait_for_workers=False,
        )
    print(f"[Rank {rank}] Entering pre-TCP barrier")
    dist.barrier()  # ensure server is listening before workers connect
    print(f"[Rank {rank}] Leaving pre-TCP barrier")
    if rank != 0:
        print(f"[Rank {rank}] Connecting to TCP store server at {master_addr}:{master_port}")
        store = dist.TCPStore(
            host_name=master_addr,
            port=master_port,
            world_size=world_size,
            is_master=False,
            timeout=dist.default_pg_timeout,
        )
    print(f"[Rank {rank}] Entering post-TCP barrier")
    dist.barrier()  # ensure all workers connected before proceeding
    print(f"[Rank {rank}] Leaving post-TCP barrier")

    # Monkey-patch: replace the Store() placeholder PyTorch hardcodes for ULFM
    # with the real TCPStore, so subsequent new_group(backend="nccl") calls
    # can call store.set()/get() for ncclUniqueId exchange.
    import torch.distributed.distributed_c10d as _c10d
    _default_pg = _c10d._get_default_group()
    _backend_str, _ = _c10d._world.pg_map[_default_pg]
    _c10d._world.pg_map[_default_pg] = (_backend_str, store)

    return True
