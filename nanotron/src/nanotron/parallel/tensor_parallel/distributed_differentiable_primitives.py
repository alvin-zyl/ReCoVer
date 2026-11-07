# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from torch import distributed as torch_dist

from nanotron import distributed as dist
from nanotron.distributed import ProcessGroup


class DifferentiableIdentity(torch.autograd.Function):
    """All-reduce gradients in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        return DifferentiableAllReduceSum.apply(grad_output, group), None


class DifferentiableAllReduceSum(torch.autograd.Function):
    """All-reduce in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        if group.size() == 1:
            return tensor

        if not tensor.is_contiguous():
            # torch.cuda.nvtx.range_push("contiguous")
            tensor = tensor.permute(2, 0, 1, 3)
            # print("tensor 0.stride", tensor.stride())
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
            tensor = tensor.permute(1, 2, 0, 3)
            # print("tensor 1.stride", tensor.stride())
            # torch.cuda.nvtx.range_pop()
            return tensor
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class DifferentiableCoalescedAllReduceSum(torch.autograd.Function):
    """Coalesced All-reduce in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        if group.size() == 1:
            return tensor

        dist.all_reduce_coalesced(tensor, op=dist.ReduceOp.SUM, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class DifferentiableAllGather(torch.autograd.Function):
    """All gather in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        ctx.group = group

        if group.size() == 1:
            return tensor

        # TODO @thomasw21: gather along another dimension
        sharded_batch_size, *rest_size = tensor.shape
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()
        unsharded_batch_size = sharded_batch_size * group.size()

        unsharded_tensor = torch.empty(
            unsharded_batch_size,
            *rest_size,
            device=tensor.device,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
        )

        # `tensor` can sometimes not be contiguous
        # https://cs.github.com/pytorch/pytorch/blob/2b267fa7f28e18ca6ea1de4201d2541a40411457/torch/distributed/nn/functional.py#L317
        tensor = tensor.contiguous()

        dist.all_gather_into_tensor(unsharded_tensor, tensor, group=group)
        return unsharded_tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        out = DifferentiableReduceScatterSum.apply(grad_output, group)
        return out, None


class DifferentiableReduceScatterSum(torch.autograd.Function):
    """Reduce scatter in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        ctx.group = group

        if group.size() == 1:
            return tensor

        # TODO @thomasw21: shard along another dimension
        unsharded_batch_size, *rest_size = tensor.shape
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()
        assert unsharded_batch_size % group.size() == 0

        # TODO @thomasw21: Collectives seem to require tensors to be contiguous
        # https://cs.github.com/pytorch/pytorch/blob/2b267fa7f28e18ca6ea1de4201d2541a40411457/torch/distributed/nn/functional.py#L305
        tensor = tensor.contiguous()

        sharded_tensor = torch.empty(
            unsharded_batch_size // group.size(),
            *rest_size,
            device=tensor.device,
            dtype=tensor.dtype,
            requires_grad=False,
        )
        dist.reduce_scatter_tensor(sharded_tensor, tensor, group=group, op=dist.ReduceOp.SUM)
        return sharded_tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        return DifferentiableAllGather.apply(grad_output, group), None


class DifferentiableAllGatherLastDim(torch.autograd.Function):
    """All-gather shards along the LAST dimension (dim=-1), differentiably.

    Forward: concat shards along dim=-1.
    Backward: split grad along dim=-1 (no reduce needed).
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: Optional[ProcessGroup]):
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()
        ctx.group = group

        tp = group.size()
        if tp == 1:
            return tensor

        # Make contiguous before view/transpose
        tensor = tensor.contiguous()

        # We want to gather along last dim, but all_gather_into_tensor gathers along dim0.
        # So reshape to 2D and swap dims: [N, d_local] -> [d_local, N], gather on dim0 -> [tp*d_local, N]
        prefix = tensor.shape[:-1]
        d_local = tensor.shape[-1]
        N = int(torch.prod(torch.tensor(prefix))) if len(prefix) > 0 else 1

        x2 = tensor.view(N, d_local).transpose(0, 1).contiguous()  # [d_local, N]

        gathered = torch.empty(
            tp * d_local,
            N,
            device=tensor.device,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
        )

        dist.all_gather_into_tensor(gathered, x2, group=group)  # [tp*d_local, N]

        # Undo transpose/reshape back to [..., tp*d_local]
        y2 = gathered.transpose(0, 1).contiguous()  # [N, tp*d_local]
        out = y2.view(*prefix, tp * d_local)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        group = ctx.group
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()

        tp = group.size()
        if tp == 1:
            return grad_output, None

        # Split grad along last dim (because forward was concat along last dim)
        rank = dist.get_rank(group)
        d_full = grad_output.shape[-1]
        assert d_full % tp == 0, f"last dim {d_full} must be divisible by tp {tp}"
        d_local = d_full // tp

        grad_local = grad_output[..., rank * d_local : (rank + 1) * d_local].contiguous()
        return grad_local, None


# -----------------
# Helper functions.
# -----------------


def differentiable_identity(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableIdentity.apply(tensor, group)


def differentiable_all_reduce_sum(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableAllReduceSum.apply(tensor, group)


def differentiable_all_gather(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableAllGather.apply(tensor, group)


def differentiable_reduce_scatter_sum(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableReduceScatterSum.apply(tensor, group)


def differentiable_coalesced_all_reduce_sum(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableCoalescedAllReduceSum.apply(tensor, group)

def differentiable_all_gather_last_dim(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableAllGatherLastDim.apply(tensor, group)