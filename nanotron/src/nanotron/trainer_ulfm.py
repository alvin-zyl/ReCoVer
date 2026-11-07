# src/nanotron/trainer_ulfm.py
"""
ULFMDistributedTrainer: Nanotron DistributedTrainer with ULFM fault-tolerant DP.

Changes from parent:
  - _create_parallel_context(): returns ULFMParallelContext (ULFM dp_pg, NCCL tp/pp/ep)
  - init_model(): monkey-patches register_comm_hook to prevent default hook
    registration; _setup_ulfm_manager() restores it and registers ULFM hook
  - training_step(): while-loop that runs pipeline, fires deferred allreduces,
    checks for failure, handles blocking/non-blocking restore, and loops for
    extra microbatches at policy boundaries
  - train(): failure simulator tick, dynamic microbatch count, dynamic GBS

All ULFM recovery logic lives in NanotronULFMTrainingManager / StepTxnOrchestrator /
policy; this class only manages the nanotron-side lifecycle.

Note: The `train()` override does not include the parent's profiler (`get_profiler`),
NVTX ranges, or CUDA profiler start/stop. These are correctness-neutral omissions;
nanotron profiling tools will not capture ULFM runs. TODO: add profiler support.
"""

import gc
import time
import shutil
import os
import wandb
from typing import Dict, Iterable, Iterator, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from nanotron import distributed as dist
from nanotron import logging as nanotron_logging
from nanotron.models import NanotronModel
from nanotron.parallel.context import ParallelContext
from nanotron.parallel.context_ulfm import ULFMParallelContext
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.sanity_checks import (
    after_optim_step_sanity_checks,
    after_tbi_sanity_checks,
    before_optim_step_sanity_checks,
    before_tbi_sanity_checks,
)
from nanotron.dataloader import sanity_check_dataloader
from nanotron.trainer import DistributedTrainer
from nanotron.training_manager_nanotron import NanotronULFMTrainingManager
from ulfm_collectives.side_collectives import ulfm_all_reduce_isolated
from nanotron.parallel.pipeline_parallel.engine import (
    OneForwardOneBackwardPipelineEngine,
)
from nanotron.parallel.pipeline_parallel.engine_ulfm import (
    ULFMOneForwardOneBackwardPipelineEngine,
)
from nanotron.logging import (
    LogItem,
    log_memory,
    log_rank,
)
from nanotron.helpers import (
    log_throughput,
)
from nanotron.optim.clip_grads import clip_grad_norm
from nanotron.parallel.tied_parameters import sync_tied_weights_gradients
from nanotron.serialize.optimizer import state_dict_to_device

import ulfm_collectives as ULFM
from ulfm_collectives.policy import GradRestoreMode
from ulfm_collectives.failure_simulator import FailureSimulator, set_failure_simulator

logger = nanotron_logging.get_logger(__name__)


class ULFMDistributedTrainer(DistributedTrainer):
    """
    Nanotron trainer with ULFM fault-tolerant data parallelism.

    Constructor args (in addition to DistributedTrainer args):
      failure_simulator: Optional[FailureSimulator]
          Pre-configured FailureSimulator. If None, no failure injection.
      ulfm_policy_type: str (default "static")
          Policy type forwarded to NanotronULFMTrainingManager.
    """

    def __init__(
        self,
        config_or_config_file,
        failure_simulator: Optional[FailureSimulator] = None,
        ulfm_policy_type: str = "static",
        **kwargs,
    ):
        self._failure_simulator = failure_simulator
        self._ulfm_policy_type = ulfm_policy_type
        # ulfm_manager is set after super().__init__(); declare early for type checkers
        self.ulfm_manager: Optional[NanotronULFMTrainingManager] = None

        super().__init__(config_or_config_file, **kwargs)

        # Now model + optimizer + accumulator exist. Create ULFM manager
        # and register the appropriate hook (bf16 deferred or fp32 deferred).
        self._setup_ulfm_manager()

        # Register failure simulator globally (ulfm_hook.py reads it via get_failure_simulator()).
        # Replica exclusion (e.g. sparing dp_rank=0 that hosts wandb) is handled
        # at schedule-generation time via --failure-exclude-replicas, not here.
        if self._failure_simulator is not None:
            set_failure_simulator(self._failure_simulator)
            self._failure_simulator.initialize(
                rank=dist.get_rank(self.parallel_context.world_pg),
                world_size=dist.get_world_size(self.parallel_context.world_pg),
            )

    def _setup_ulfm_manager(self):
        """Create the ULFM training manager after model + optimizer + accumulator are ready.

        By this point, init_model() has monkey-patched register_comm_hook to
        swallow the default hook registration (if fp32 accum is enabled).
        The optimizer and accumulator are properly created via
        OptimizerFromGradientAccumulator.  Now we restore register_comm_hook
        and register our own ULFM hook.
        """
        if not isinstance(self.model, DistributedDataParallel):
            logger.warning("dp_size=1, ULFM manager not created (single-GPU mode).")
            return

        # Restore register_comm_hook if it was monkey-patched in init_model()
        orig = getattr(self.model, "_original_register_comm_hook", None)
        if orig is not None:
            self.model.register_comm_hook = orig
            del self.model._original_register_comm_hook

        dp_size = self.parallel_context.data_parallel_size
        grad_accum = self.config.tokens.batch_accumulation_per_replica

        # Pass the fp32 accumulator if the parent created one via
        # OptimizerFromGradientAccumulator (accumulate_grad_in_fp32=True).
        from nanotron.optim.gradient_accumulator import FP32GradientAccumulator
        fp32_accum = (
            self.grad_accumulator
            if isinstance(self.grad_accumulator, FP32GradientAccumulator)
            else None
        )

        self.ulfm_manager = NanotronULFMTrainingManager(
            ddp_model=self.model,
            dp_pg=self.parallel_context.dp_pg,
            grad_accum_steps=grad_accum,
            policy_type=self._ulfm_policy_type,
            grad_accumulator=fp32_accum,
            initial_world_size=dp_size,
        )
        # The replica of a world rank is its data-parallel index.
        _replica_of = {
            int(world_rank): int(index[2])  # world_rank_matrix is [ep, pp, dp, tp]
            for index, world_rank in np.ndenumerate(self.parallel_context.world_rank_matrix)
        }
        self.ulfm_manager.txn.set_replica_map(_replica_of.__getitem__, dist.get_rank())

    # ------------------------------------------------------------------
    # Override 1: use ULFMParallelContext
    # ------------------------------------------------------------------

    def _create_parallel_context(self) -> ParallelContext:
        return ULFMParallelContext(
            tensor_parallel_size=self.config.parallelism.tp,
            pipeline_parallel_size=self.config.parallelism.pp,
            data_parallel_size=self.config.parallelism.dp,
            expert_parallel_size=self.config.parallelism.expert_parallel_size,
        )

    # ------------------------------------------------------------------
    # Override 2: register ULFM hook on the DDP model nanotron creates
    # ------------------------------------------------------------------

    def init_model(self) -> Union[NanotronModel, DistributedDataParallel]:
        model = super().init_model()

        # If fp32 accumulation is enabled, the parent's __init__ will next call
        # init_optimizer_and_grad_accumulator(), which creates the optimizer +
        # accumulator (OptimizerFromGradientAccumulator) AND registers the default
        # DDP comm hook.  DDP only allows one hook, so we monkey-patch
        # register_comm_hook to a no-op here.  _setup_ulfm_manager() restores it
        # and registers our ULFM hook instead.
        if (
            isinstance(model, DistributedDataParallel)
            and getattr(self.config.optimizer, "accumulate_grad_in_fp32", False)
        ):
            model._original_register_comm_hook = model.register_comm_hook
            model.register_comm_hook = lambda state, hook: None

        return model

    # ------------------------------------------------------------------
    # Override 3: training_step — while-loop with ULFM recovery
    # ------------------------------------------------------------------

    def _microbatches(self, dataloader, n_micro: int):
        """Yield ``n_micro`` batches. The pipeline engine pulls each one right
        before that microbatch's forward pass, which is where a ``forward``
        failure is injected."""
        for _ in range(n_micro):
            micro_batch = next(dataloader)
            if self._failure_simulator is not None:
                with self._failure_simulator.may_fail_here("forward"):
                    pass
            yield micro_batch

    def training_step(
        self, dataloader: Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]
    ) -> Tuple[Iterable[Dict], Optional[torch.Tensor]]:
        """
        ULFM-aware training step with loop-based failure recovery.

        Cases:
          - No failure: run pipeline, allreduce, step
          - Non-boundary failure: blocking restore + re-reduce, step
          - Policy boundary: non-blocking restore + extra microbatches (loop back)
          - Failure during re-reduce or extended pass: same logic re-applies
        """
        before_tbi_sanity_checks(
            self.config,
            self.parallel_context,
            self.unwrapped_model,
            self.grad_accumulator,
            self.lr_scheduler,
        )

        if self.iteration_step < self.initial_iter_step + 5:
            log_memory(logger=logger)

        is_first_pass = True
        all_outputs = []

        while True:
            # --- Determine microbatch count ---
            if self.ulfm_manager is None:
                n_micro = self.n_micro_batches_per_batch
            elif is_first_pass:
                n_micro = self.ulfm_manager.get_effective_n_microbatches()
            else:
                # Extended pass at policy boundary
                n_micro = self.ulfm_manager.get_n_extra_microbatches()

            if self.ulfm_manager is not None:
                self.ulfm_manager.prepare_iteration(
                    iteration_step=self.iteration_step,
                    n_microbatches=n_micro,
                    is_first_pass=is_first_pass
                )
                # Global sync: ULFM consensus on world_pg
                ulfm_opts = ULFM.ULFMOptions(auto_repair=True)
                logger.debug("Entering the global ULFM consensus barrier")
                work = self.parallel_context.world_pg.consensus(ulfm_opts)
                work.wait()
                # Extended pass: take in failures the world consensus found
                # that this DP group's decision does not include.
                if not is_first_pass and self.ulfm_manager.reconcile_extended_pass(work):
                    n_micro = self.ulfm_manager.get_n_extra_microbatches()

            # --- Wire pre_first_backward_fn (only for extended passes) ---
            # Waits for async gradient restore before the first microbatch's
            # backward — early micros' `_accumulate_grad` writes the fp32 buffer
            # on the main stream, which races the non-blocking side-stream
            # `view.copy_(snap)` until the wait_event is inserted.
            if self.ulfm_manager is not None and not is_first_pass:
                self.pipeline_engine.pre_first_backward_fn = (
                    lambda: self.ulfm_manager.wait_restore_before_backward()
                )
            else:
                self.pipeline_engine.pre_first_backward_fn = None

            # --- Run pipeline: all microbatches fwd+bwd ---
            self.pipeline_engine.nb_microbatches = n_micro
            outputs = self.pipeline_engine.train_batch_iter(
                model=self.model,
                pg=self.parallel_context.pp_pg,
                batch=self._microbatches(dataloader, n_micro),
                nb_microbatches=n_micro,
                grad_accumulator=self.grad_accumulator,
            )
            all_outputs.extend(outputs if outputs else [])
            is_first_pass = False

            if self.iteration_step < self.initial_iter_step + 5:
                log_memory(logger=logger)

            after_tbi_sanity_checks(
                self.config, self.parallel_context, self.unwrapped_model, self.grad_accumulator
            )

            # --- Fire deferred ULFM gradient allreduces ---
            if self.ulfm_manager is not None and self.parallel_context.data_parallel_size > 1:
                self.ulfm_manager.fire_deferred_allreduces()

            # --- Replica-consistency gate ---
            if self.ulfm_manager is not None and self.parallel_context.data_parallel_size > 1:
                mp_pg = self.parallel_context.mp_pg
                if dist.get_world_size(mp_pg) > 1:
                    dist.barrier(group=mp_pg, device_ids=[torch.cuda.current_device()])
                self.ulfm_manager.on_consensus_step()

            # --- Check restore mode (set by handle_work_completion during allreduce) ---
            if self.ulfm_manager is None:
                break  # Single-GPU: no recovery needed

            if self.ulfm_manager.get_restore_mode() == GradRestoreMode.SKIP:
                # No failure — proceed to optimizer step
                break
            
            if self.ulfm_manager.get_restore_mode() == GradRestoreMode.NON_BLOCKING:
                # Policy boundary: start async restore, loop for extra microbatches
                n_extra = self.ulfm_manager.get_n_extra_microbatches()
                logger.warning(
                    f"[Rank {dist.get_rank(self.parallel_context.world_pg)}] "
                    f"Policy boundary — need {n_extra} extra microbatches. "
                    "Starting non-blocking restore."
                )
                self.ulfm_manager.start_nonblocking_restore()
                continue  # Loop back for extra microbatches

            blocking_restore_attempt = 0
            while self.ulfm_manager.get_restore_mode() == GradRestoreMode.BLOCKING:
                # Non-boundary failure: blocking restore with re-reduce.
                # Disable internal retry — we cross-synchronize DP groups
                # below via the replica barrier + ULFM DP barrier, so any
                # re-reduction failure must return here so every rank makes
                # the same outer-loop decision (otherwise early-discoverer
                # ranks retry to success and set SKIP while late-discoverer
                # ranks re-enter BLOCKING alone, deadlocking the next
                # collective — MPI has no timeout).
                logger.warning(
                    f"[Rank {dist.get_rank(self.parallel_context.world_pg)}] "
                    "BLOCKING gradient restore before optimizer step"
                )
                self.ulfm_manager.start_blocking_restore(allow_internal_retry=False)
                blocking_restore_attempt += 1

                # If failure happened during the re-reduce, some ranks in the failed replica may survived
                # and other ranks in healthy replica may have incosistent views from the one directly involved with the failure
                # The additional replica barrier kills the survivors in the failed replica,
                # and ensures all ranks in a healthy replica have consistent views before proceeding.
                if self.ulfm_manager is not None and self.parallel_context.data_parallel_size > 1:
                    ulfm_opts = ULFM.ULFMOptions(auto_repair=True)
                    logger.debug("Entering the global ULFM consensus barrier")
                    work = self.parallel_context.world_pg.consensus(ulfm_opts)
                    work.wait()

                    mp_pg = self.parallel_context.mp_pg
                    if dist.get_world_size(mp_pg) > 1:
                        dist.barrier(group=mp_pg, device_ids=[torch.cuda.current_device()])
                    self.ulfm_manager.on_consensus_step()

                # Check if blocking restore retries crossed a policy boundary
                if self.ulfm_manager.is_at_policy_boundary():
                    logger.warning(
                        f"[Rank {dist.get_rank(self.parallel_context.world_pg)}] "
                        "Crossed policy boundary during blocking restore — "
                        "starting non-blocking restore for extra microbatches"
                    )
                    self.ulfm_manager.start_nonblocking_restore()
                    break  # Loop back for extra microbatches
                elif self.ulfm_manager.get_restore_mode() == GradRestoreMode.BLOCKING:
                    # Still on the same boundary after blocking restore + re-reduce: proceed to optimizer step
                    logger.warning(
                        f"[Rank {dist.get_rank(self.parallel_context.world_pg)}] "
                        "Failure happend during re-reduction, but not crossing policy boundary, retrying."
                    )
                    if blocking_restore_attempt <= 3:
                        continue
                    else:
                        raise RuntimeError(
                            f"[Rank {dist.get_rank(self.parallel_context.world_pg)}] "
                            "Re-attempt of blocking restoration has reached limit, abort."
                        )
            else:
                logger.warning(
                    f"[Rank {dist.get_rank(self.parallel_context.world_pg)}] "
                    "Restored successfully without crossing policy boundary — proceeding to optimizer step"
                )
                break

        # --- Post-loop: sync tied weights, clip grads, optimizer step ---

        # Sync gradients of tied parameters across PP stages (NCCL groups, not DP)
        sync_tied_weights_gradients(
            module=self.unwrapped_model,
            parallel_context=self.parallel_context,
            grad_accumulator=self.grad_accumulator,
        )

        if self.ulfm_manager is not None:
            self.ulfm_manager.normalize_gradients()

        # Clip gradients
        if self.config.optimizer.clip_grad is not None:
            named_parameters = [
                (name, param)
                for name, param in self.unwrapped_model.get_named_params_with_correct_tied()
                if param.requires_grad
            ]
            self.grad_norm_unclipped = clip_grad_norm(
                mp_pg=self.parallel_context.mp_pg,
                named_parameters=named_parameters,
                grad_accumulator=self.grad_accumulator,
                max_norm=self.config.optimizer.clip_grad,
            )

        before_optim_step_sanity_checks(
            self.config,
            self.parallel_context,
            self.unwrapped_model,
            self.grad_accumulator,
            self.optimizer,
        )

        # Move optimizer states to GPU on the first step of a checkpoint-resume run.
        if (
            self.init_checkpoint_path is not None
            and self.config.checkpoints.load_optimizer
            and self.iteration_step == self.initial_iter_step
        ):
            state_dict_to_device(self.optimizer.state_dict(), "cuda")

        # Loss average across DP
        if all_outputs and isinstance(all_outputs[0]["loss"], torch.Tensor):
            loss_avg = torch.stack([out["loss"] for out in all_outputs]).sum()
            handle = dist.ulfm_all_reduce(
                loss_avg,
                group=self.parallel_context.dp_pg,
                async_op=True,
                op=dist.ReduceOp.SUM,
            )
            handle.wait()
            loss_avg.div_(self.ulfm_manager._get_grad_div_factor())
        else:
            loss_avg = None

        if self.ulfm_manager is not None:
            self.ulfm_manager.optimizer_step(self.optimizer)
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()

        after_optim_step_sanity_checks(
            self.config, self.parallel_context, self.unwrapped_model, self.grad_accumulator
        )

        # LR scheduler advances only on successful optimizer steps
        self.lr_scheduler.step()

        self.post_train_step()

        return all_outputs, loss_avg

    # ------------------------------------------------------------------
    # Override 4: ULFM-safe logging
    # ------------------------------------------------------------------

    def _is_log_rank(self) -> bool:
        """Check dynamically if this rank should log, using dp_pg.current_rank()
        instead of the static dp_rank from the world_rank_matrix."""
        my_global_rank = dist.get_rank(self.parallel_context.world_pg)
        ep_rank, pp_rank, _, tp_rank = self.parallel_context.get_local_ranks(my_global_rank)
        return (
            ep_rank == 0
            and pp_rank == self.unwrapped_model.output_pp_rank
            and tp_rank == 0
            and self.parallel_context.dp_pg.current_rank() == 0
        )

    def train_step_logs(self, outputs, loss_avg):
        """Override parent to use dp_pg.current_rank() for the log-rank check."""
        import numpy as np

        if self._is_log_rank():
            my_global_rank = dist.get_rank(self.parallel_context.world_pg)
            self.logger_ranks = np.array([my_global_rank])
            if self.loggerwriter is None:
                self.loggerwriter = self.setup_log_writers()
        else:
            self.logger_ranks = np.array([])

        torch.cuda.synchronize()
        elapsed_time_per_iteration_ms = (time.time() - self.iteration_start_time) * 1000
        tokens_per_sec = (
            self.global_batch_size * self.sequence_length / (elapsed_time_per_iteration_ms / 1000)
        )  # tokens_per_sec is calculated using sequence_length
        model_tflops, hardware_tflops = self.unwrapped_model.get_flops_per_sec(
            iteration_time_in_sec=elapsed_time_per_iteration_ms / 1000,
            sequence_length=self.sequence_length,
            global_batch_size=self.global_batch_size,
        )

        if dist.get_rank(self.parallel_context.world_pg) in self.logger_ranks:
            assert self.loggerwriter is not None, "loggerwriter should be defined on logger ranks"

            lr = self.lr_scheduler.get_last_lr()[0]

            log_entries = [
                # LogItem("consumed_samples", self.consumed_train_samples, "human_format"),  # , "12d"),
                LogItem(
                    "consumed_tokens",
                    self.metadata.consumed_train_samples * self.config.tokens.sequence_length,
                    "human_format",
                ),  # , "12d"),
                LogItem("elapsed_time_per_iteration_ms", elapsed_time_per_iteration_ms, "human_format"),  # , ".1f"),
                LogItem("tokens_per_sec", tokens_per_sec, "human_format"),  # , "1.6E"),
                LogItem(
                    "tokens_per_sec_per_gpu", tokens_per_sec / self.parallel_context.world_pg.current_size(), "human_format"
                ),  # , "1.6E"),
                LogItem("total_gpus", self.parallel_context.world_pg.current_size(), "human_format"),  # , "5d"),
                LogItem("global_batch_size", self.global_batch_size, "human_format"),  # , "5d"),
                LogItem("lm_loss", loss_avg.item(), "human_format"),  # , "1.6E"),
                LogItem("lr", lr, "human_format"),  # , ".3E"),
                LogItem("model_tflops_per_gpu", model_tflops, "human_format"),  # , ".2f"),
                LogItem("hardware_tflops_per_gpu", hardware_tflops, "human_format"),  # , ".2f"),
            ]

            if self.config.optimizer.clip_grad is not None:
                log_entries.append(LogItem("grad_norm", self.grad_norm_unclipped.item(), "human_format"))  # , ".3f"))

            if self.ulfm_manager is not None:
                txn = self.ulfm_manager.txn
                curr_grad_accum = txn.curr_grad_accum_steps
                curr_dp_size = txn.curr_world_size
                gbs = txn.effective_batch_size
                total_workload = curr_dp_size * curr_grad_accum
                redundancy = 1.0 - (gbs / total_workload) if total_workload > 0 else 0.0
                log_entries.extend([
                    LogItem("ulfm_num_majors", txn.num_major_procs, "human_format"),
                    LogItem("ulfm_num_minors", txn.num_minor_procs, "human_format"),
                    LogItem("ulfm_num_major_spares", txn.num_major_spare_procs, "human_format"),
                    LogItem("ulfm_num_minor_spares", txn.num_minor_spare_procs, "human_format"),
                    LogItem("ulfm_curr_grad_accum", curr_grad_accum, "human_format"),
                    LogItem("ulfm_curr_minor_grad_accum", txn.minor_proc_grad_accum_steps, "human_format"),
                    LogItem("ulfm_redundancy", redundancy, "human_format"),
                ])

            # Log not too often the memory
            if self.iteration_step < 5 or (self.iteration_step - 1) % self.config.checkpoints.checkpoint_interval == 0:
                total, used, free = shutil.disk_usage("/")
                log_entries.extend(
                    [
                        LogItem(
                            "cuda_memory_allocated", torch.cuda.memory_allocated(), "human_format"
                        ),  #  / 1024**2, ".2f"),
                        LogItem(
                            "cuda_max_memory_reserved", torch.cuda.max_memory_reserved(), "human_format"
                        ),  #  / 1024**2, ".2f"),
                        LogItem("hd_total_memory_tb", total, "human_format"),  #  / (2**40), ".2f"),
                        LogItem("hd_used_memory_tb", used, "human_format"),  #  / (2**40), ".2f"),
                        LogItem("hd_free_memory_tb", free, "human_format"),  #  / (2**40), ".2f"),
                    ]
                )

            # NOTE: only one rank writes to wandb. A rank that took over
            # logging after a failure has no wandb run (wandb.init ran on the
            # first logger rank only); it logs to stdout only.
            if (
                dist.get_rank(self.parallel_context.world_pg) == self.logger_ranks[0]
                and wandb is not None
                and wandb.run is not None
            ):
                wandb.log(
                    {
                        **{log_item.tag: log_item.scalar_value for log_item in log_entries},
                        "iteration_step": self.iteration_step,
                    },
                    step=self.iteration_step,
                )

            self.loggerwriter.add_scalars_from_list(log_entries, self.iteration_step)

        # Nanotron Benchmark mode: we log the throughput and exit
        if os.environ.get("NANOTRON_BENCHMARK", "0") == "1" and self.iteration_step == 3:
            log_throughput(
                self.config,
                self.parallel_context,
                model_tflops,
                hardware_tflops,
                tokens_per_sec,
            )
            log_rank("Throughput logging complete", logger=logger, level=nanotron_logging.INFO)
            if "SLURM_JOB_ID" in os.environ:
                os.system("scancel " + os.environ["SLURM_JOB_ID"])
            else:
                exit(0)

    # ------------------------------------------------------------------
    # Override 5: ULFM-safe validation
    # ------------------------------------------------------------------

    def run_validation(self) -> None:
        """ULFM-safe validation.

        Mirrors the parent's run_validation, but syncs the loss across DP the
        same way training_step syncs loss_avg: `dist.ulfm_all_reduce` (SUM) on
        the ULFM dp_pg + manual divide, instead of the parent's plain
        `dist.all_reduce(op=AVG)` — a non-ULFM collective on a communicator
        that may have been shrunk/revoked by an earlier training failure.

        Assumes no simulated failure fires during validation (the simulator
        only ticks in train() via begin_minibatch), so no recovery loop is
        needed: the dp_pg entering validation was already repaired by the
        consensus/restore logic of the preceding training_step.
        """
        log_rank(
            f"[Validation] Running validation at step {self.iteration_step}",
            logger=logger,
            level=nanotron_logging.INFO,
            rank=0,
        )

        # Reinitialize validation dataloader from factory to always start from beginning
        val_dl = (
            self._val_dataloader_factory()
            if callable(self._val_dataloader_factory)
            else self._val_dataloader_factory
        )
        val_dataloader = sanity_check_dataloader(
            dataloader=val_dl, parallel_context=self.parallel_context, config=self.config
        )

        self.model.eval()

        val_outputs = self.validation_step(dataloader=val_dataloader)

        # Each output["loss"] is already divided by the number of validation
        # microbatches (validate_batch_iter uses the base PipelineEngine.forward),
        # so the local sum is this replica's mean loss. SUM-allreduce across the
        # live DP ranks, then divide by the live DP size to get the global mean.
        if val_outputs and isinstance(val_outputs[0]["loss"], torch.Tensor):
            val_loss = torch.stack([output["loss"] for output in val_outputs]).sum()

            if self.ulfm_manager is not None and self.parallel_context.data_parallel_size > 1:
                # On the DP eval group, not dp_pg: a failure found here must
                # stay in dp_pg for the next training step (side_collectives).
                eval_pg = self.parallel_context.dp_eval_pg
                ulfm_all_reduce_isolated(val_loss, dist.ReduceOp.SUM, eval_pg)
                val_loss.div_(eval_pg.current_size())

            val_perplexity = torch.exp(val_loss)
            self.validation_step_logs(val_loss=val_loss, val_perplexity=val_perplexity)
        else:
            log_rank(
                "[Validation] No loss computed (not on output PP rank)",
                logger=logger,
                level=nanotron_logging.INFO,
                rank=0,
            )

        self.model.train()

    def validation_step_logs(self, val_loss, val_perplexity) -> None:
        """Override parent to use the dynamic dp_pg.current_rank() log-rank
        check (static self.logger_ranks may point at a dead rank after a
        failure, or be unset if train_step_logs hasn't run this step)."""
        import numpy as np

        if self._is_log_rank():
            my_global_rank = dist.get_rank(self.parallel_context.world_pg)
            self.logger_ranks = np.array([my_global_rank])
            if self.loggerwriter is None:
                self.loggerwriter = self.setup_log_writers()

            log_entries = [
                LogItem("val_loss", val_loss.item(), "human_format"),
                LogItem("val_perplexity", val_perplexity.item(), "human_format"),
            ]

            log_rank(
                f"[Validation] Step {self.iteration_step}: val_loss={val_loss.item():.4f}, "
                f"val_perplexity={val_perplexity.item():.4f}",
                logger=logger,
                level=nanotron_logging.INFO,
            )

            # NOTE: _is_log_rank() is true on exactly one rank, so it is the
            # wandb writer, if it has a wandb run (see train_step_logs)
            if wandb is not None and wandb.run is not None:
                wandb.log(
                    {
                        "val_loss": val_loss.item(),
                        "val_perplexity": val_perplexity.item(),
                        "iteration_step": self.iteration_step,
                    },
                    step=self.iteration_step,
                )

            self.loggerwriter.add_scalars_from_list(log_entries, self.iteration_step)

    # ------------------------------------------------------------------
    # Override 6: train() — dynamic workload + failure handling
    # ------------------------------------------------------------------

    def train(
        self,
        dataloader_or_dls,
        **kwargs,
    ) -> None:
        """
        ULFM training loop with dynamic workload adjustment.

        Each iteration:
          - Queries policy for effective microbatch count (may change after failure)
          - Ticks failure simulator
          - Runs training_step (which handles extended windows internally via loop)
          - Dynamically computes global_batch_size for consumed_train_samples
        """
        self.pre_training(**kwargs)

        if self.config.checkpoints.save_initial_state and self.init_checkpoint_path is None:
            self.save_checkpoint()

        # Use ULFM pipeline engine
        base_engine = self.config.parallelism.pp_engine
        if isinstance(base_engine, OneForwardOneBackwardPipelineEngine):
            self.pipeline_engine = ULFMOneForwardOneBackwardPipelineEngine()
        else:
            self.pipeline_engine = base_engine
        self.pipeline_engine.nb_microbatches = self.n_micro_batches_per_batch

        # Wire should_zero_loss_fn once — dp_pg.should_contribute() /
        # increment_contributed() must be called every microbatch to track
        # contribution counts (same as _may_zero_grad in reference).
        if self.ulfm_manager is not None:
            self.pipeline_engine.should_zero_loss_fn = (
                lambda: self.ulfm_manager.should_zero_grad()
            )

        self.unwrapped_model = (
            self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        )
        self.unwrapped_model.module_id_to_prefix = {
            id(module): f"{module_name}."
            for module_name, module in self.unwrapped_model.named_modules()
        }
        self.unwrapped_model.module_id_to_prefix[id(self.unwrapped_model)] = ""

        self.initial_iter_step = self.metadata.last_train_step + 1
        self.last_iter_step = self.config.tokens.train_steps

        gc.collect()
        torch.cuda.empty_cache()

        for self.iteration_step in range(self.initial_iter_step, self.last_iter_step + 1):
            self.iteration_start_time = time.time()
            self._update_dataloader_based_on_training_stages(dataloader_or_dls)

            # Dynamic microbatch count from policy
            if self.ulfm_manager is not None:
                effective_n_micro = self.ulfm_manager.get_effective_n_microbatches()
                self.pipeline_engine.nb_microbatches = effective_n_micro

            # Tick failure simulator
            if self._failure_simulator is not None:
                self._failure_simulator.begin_minibatch(self.iteration_step)

            outputs, loss_avg = self.training_step(dataloader=self.current_dataloader)

            # Failure iteration: outputs=[], loss_avg=None → skip bookkeeping
            if not outputs and loss_avg is None:
                logger.warning(
                    f"[Step {self.iteration_step}] Failure detected — skipping metadata/log/checkpoint."
                )
                continue

            # Dynamic global_batch_size for consumed_train_samples
            if self.ulfm_manager is not None:
                effective_gbs = self.micro_batch_size * self.ulfm_manager._get_grad_div_factor()
            else:
                effective_gbs = self.global_batch_size

            self.metadata.consumed_train_samples += effective_gbs
            self.metadata.last_train_step = self.iteration_step
            self.metadata.data_stages[
                self.metadata.last_stage_idx
            ].consumed_train_samples += effective_gbs

            if (self.iteration_step - 1) % self.config.logging.iteration_step_info_interval == 0:
                self.train_step_logs(outputs=outputs, loss_avg=loss_avg)

            if self.iteration_step % self.config.checkpoints.checkpoint_interval == 0:
                self.save_checkpoint()

            if (
                self.config.tokens.val_check_interval > 0
                and self._val_dataloader_factory is not None
                and self.iteration_step % self.config.tokens.val_check_interval == 0
            ):
                self.run_validation()

        self.parallel_context.world_pg.barrier()

        if self.config.checkpoints.save_final_state:
            self.save_checkpoint()

        self.post_training()
