"""All-reduce outside the training step: evaluation, validation, saver election.

These run on a ULFM group of their own, with the same ranks as the training
group (dp_pg / replicate_pg). A failure that such an all-reduce runs into
repairs only that group, and the all-reduce is repeated over the survivors.
The training group is left as it was: it still holds the failure, and the
next training step finds it and lets the policy handle it, as for a failure
after the step's gradient all-reduce. Done on the training group instead,
the repair would hide the failure from the policy (the failed replica's
microbatches would silently go missing from every later step).
"""

import torch.distributed as dist

import ulfm_collectives as ULFM


def ulfm_all_reduce_isolated(tensor, op, group, max_attempts: int = 10) -> None:
    """In-place all-reduce of ``tensor`` over the live ranks of ``group``.

    An all-reduce that ran into a failure is repeated from the local value:
    one that found the failure beforehand returns without reducing (NOOP),
    and one interrupted by it leaves the buffer undefined, which the
    consensus that follows reveals.
    """
    local = tensor.clone()
    opts = ULFM.ULFMOptions(auto_repair=True)
    for _ in range(max_attempts):
        work = dist.ulfm_all_reduce(tensor, op=op, group=group, async_op=True)
        work.wait()
        check = group.consensus(opts)
        check.wait()
        if not (work.has_failures() or work.was_noop() or check.has_failures()):
            return
        tensor.copy_(local)
    raise RuntimeError(f"all-reduce still failing after {max_attempts} attempts")
