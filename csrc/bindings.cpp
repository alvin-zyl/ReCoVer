#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ProcessGroupULFM.hpp"
#include "TypesULFM.hpp"
#include "ULFMLogging.hpp"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupULFM", &c10d::ProcessGroupULFM::createProcessGroupULFM);

  py::class_<c10d::ULFMOptions>(m, "ULFMOptions")
      .def(py::init<>())
      .def(py::init([](bool auto_repair, bool track_rank_types, bool auto_elect, bool consensus_on_rank_types) {
          c10d::ULFMOptions opts;
          opts.auto_repair = auto_repair;
          opts.track_rank_types = track_rank_types;
          opts.auto_elect = auto_elect;
          opts.consensus_on_rank_types = consensus_on_rank_types;
          return opts;
      }),
      py::arg("auto_repair") = false,
      py::arg("track_rank_types") = false,
      py::arg("auto_elect") = true,
      py::arg("consensus_on_rank_types") = false)
      .def_readwrite("auto_repair", &c10d::ULFMOptions::auto_repair)
      .def_readwrite("track_rank_types", &c10d::ULFMOptions::track_rank_types)
      .def_readwrite("auto_elect", &c10d::ULFMOptions::auto_elect)
      .def_readwrite("consensus_on_rank_types", &c10d::ULFMOptions::consensus_on_rank_types)
      .def_readwrite("failure_strategy", &c10d::ULFMOptions::failure_strategy)
      .def("copy", [](const c10d::ULFMOptions& self) {
          return c10d::ULFMOptions(self);
      }, "Create a copy of this ULFMOptions object");

  py::enum_<c10d::ULFMFailureHandlingStrategy>(m, "ULFMFailureHandlingStrategy")
      .value("CONTINUE_WITH_SURVIVORS", c10d::ULFMFailureHandlingStrategy::CONTINUE_WITH_SURVIVORS)
      .value("RESTART_FAILED_PROCESSES", c10d::ULFMFailureHandlingStrategy::RESTART_FAILED_PROCESSES)
      .value("ABORT_ON_FAILURE", c10d::ULFMFailureHandlingStrategy::ABORT_ON_FAILURE);

  py::class_<c10d::ProcessGroupULFM, c10d::ProcessGroup, c10::intrusive_ptr<c10d::ProcessGroupULFM>>(m, "ProcessGroupULFM")
      .def("ulfm_allreduce", &c10d::ProcessGroupULFM::ulfm_allreduce,
      py::arg("tensors"),
      py::arg("opts") = c10d::AllreduceOptions(),
      py::arg("ulfm_opts") = c10d::ULFMOptions())
      .def("repair_communicator", &c10d::ProcessGroupULFM::repair_communicator)
      .def("notify_all_ranks_of_failure", &c10d::ProcessGroupULFM::notify_all_ranks_of_failure)
      .def("check_for_failures", &c10d::ProcessGroupULFM::check_for_failures)
      .def("detect_and_recover_failures", [](c10d::ProcessGroupULFM& self, bool auto_repair) {
          std::vector<int> failed_ranks;
          bool success = self.detect_and_recover_failures(auto_repair, &failed_ranks);
          return py::make_tuple(success, failed_ranks);
      }, py::arg("auto_repair") = true,
         "Comprehensive failure detection and recovery workflow. Returns (success, failed_ranks)")
      .def("consensus", &c10d::ProcessGroupULFM::consensus,
         py::arg("ulfm_opts") = c10d::ULFMOptions(),
         "Perform consensus operation for failure detection and recovery")
      .def("set_quiesce", &c10d::ProcessGroupULFM::set_quiesce, py::arg("v"),
         "Set the quiesce state of the process group")
      .def("is_quiesced", &c10d::ProcessGroupULFM::is_quiesced,
         "Check if the process group is currently quiesced")
      .def("worldEpoch", &c10d::ProcessGroupULFM::worldEpoch,
         "Get the current world epoch (increments after communicator repairs)")
      .def("original_rank_of", &c10d::ProcessGroupULFM::original_rank_of,
           "Rank in the group as created of the process with the given rank in the current communicator (-1 if none)")
      .def("current_rank", &c10d::ProcessGroupULFM::current_rank,
         "Get the current rank (may change after communicator repairs)")
      .def("current_size", &c10d::ProcessGroupULFM::current_size,
         "Get the current world size (may change after communicator repairs)")
      .def("set_minor", &c10d::ProcessGroupULFM::set_minor,
         "Set this rank as a minor rank")
      .def("reset_minor", &c10d::ProcessGroupULFM::reset_minor,
         "Reset this rank to not be a minor rank")
      .def("is_minor", &c10d::ProcessGroupULFM::is_minor,
         "Check if this rank is a minor rank")
      .def("set_major_minor_split", &c10d::ProcessGroupULFM::set_major_minor_split, py::arg("boundary"),
         "Set the major/minor split: ranks < boundary are major, ranks >= boundary are minor")
      .def("set_major_minor_split_with_spares", &c10d::ProcessGroupULFM::set_major_minor_split_with_spares,
         py::arg("num_majors"), py::arg("num_minors"), py::arg("num_major_spares"), py::arg("num_minor_spares"),
         "Set rank type based on explicit counts. Layout: [major workers | major spares | minor workers | minor spares]")
      .def("set_spare", &c10d::ProcessGroupULFM::set_spare,
         "Set this rank as a spare rank")
      .def("reset_spare", &c10d::ProcessGroupULFM::reset_spare,
         "Reset this rank to not be a spare rank")
      .def("is_spare", &c10d::ProcessGroupULFM::is_spare,
         "Check if this rank is a spare rank")
      .def("set_boundary_minor", &c10d::ProcessGroupULFM::set_boundary_minor,
         "Set this rank as a boundary minor rank")
      .def("reset_boundary_minor", &c10d::ProcessGroupULFM::reset_boundary_minor,
         "Reset this rank to not be a boundary minor rank")
      .def("is_boundary_minor", &c10d::ProcessGroupULFM::is_boundary_minor,
         "Check if this rank is a boundary minor rank")
      .def("set_boundary_minor_split", &c10d::ProcessGroupULFM::set_boundary_minor_split,
         py::arg("num_boundary_majors"),
         py::arg("boundary_major_workload"),
         py::arg("boundary_minor_workload"),
         "Set the boundary minor split and directly populate the boundary-phase "
         "target contribution: ranks < num_boundary_majors take boundary_major_workload, "
         "ranks >= num_boundary_majors take boundary_minor_workload. "
         "boundary_contributed_ is cleared by reset_contributed() at iteration end, "
         "so this is safe to call multiple times within the same boundary.")
      .def("is_at_policy_boundary", &c10d::ProcessGroupULFM::is_at_policy_boundary,
         "Check if PG has reached policy boundary (sticky flag)")
      .def("reset_policy_boundary", &c10d::ProcessGroupULFM::reset_policy_boundary,
         "Reset PG-level policy boundary flag")
      .def("update_rank_type_counts", [](c10d::ProcessGroupULFM& self,
                                         int majors, int minors,
                                         int major_spares, int minor_spares,
                                         int boundary_minors) {
          c10d::RankTypeCounts counts(majors, minors, major_spares, minor_spares, boundary_minors);
          self.update_rank_type_counts(counts);
      }, py::arg("majors"), py::arg("minors"),
         py::arg("major_spares"), py::arg("minor_spares"),
         py::arg("boundary_minors") = 0,
         "Update rank type counts directly (without MPI communication)")
      .def("get_num_major_procs", &c10d::ProcessGroupULFM::get_num_major_procs,
         "Get count of major workers")
      .def("get_num_minor_procs", &c10d::ProcessGroupULFM::get_num_minor_procs,
         "Get count of minor workers")
      .def("get_num_major_spare_procs", &c10d::ProcessGroupULFM::get_num_major_spare_procs,
         "Get count of major spares")
      .def("get_num_minor_spare_procs", &c10d::ProcessGroupULFM::get_num_minor_spare_procs,
         "Get count of minor spares")
      .def("get_num_boundary_minor_procs", &c10d::ProcessGroupULFM::get_num_boundary_minor_procs,
         "Get count of boundary minor ranks")
      .def("get_contributed", &c10d::ProcessGroupULFM::get_contributed,
         "Get local count of gradient contributions made by this rank")
      .def("increment_contributed", &c10d::ProcessGroupULFM::increment_contributed,
         "Increment the local gradient contribution counter. Call from the control plane "
         "when this rank's gradient was not zeroed before the allreduce.")
      .def("reset_contributed", &c10d::ProcessGroupULFM::reset_contributed,
         "Reset the local gradient contribution counter to zero.")
      .def("get_target_contribution", &c10d::ProcessGroupULFM::get_target_contribution,
         "Get the target contribution value for this rank.")
      .def("set_target_contribution", &c10d::ProcessGroupULFM::set_target_contribution,
         py::arg("major_value"), py::arg("minor_value") = -1,
         "Set the target contribution. major_value applies to major ranks; "
         "minor_value applies to minor ranks (defaults to major_value if omitted).")
      .def("increment_target_contribution", &c10d::ProcessGroupULFM::increment_target_contribution,
         py::arg("delta") = 1,
         "Increment the target contribution by a positive delta (default 1).")
      .def("get_boundary_contributed", &c10d::ProcessGroupULFM::get_boundary_contributed,
         "Get local count of gradient contributions made during the boundary phase.")
      .def("reset_boundary_contributed", &c10d::ProcessGroupULFM::reset_boundary_contributed,
         "Reset the boundary-phase gradient contribution counter to zero.")
      .def("merge_boundary_contributed", &c10d::ProcessGroupULFM::merge_boundary_contributed,
         "Fold boundary_contributed into contributed and zero both "
         "boundary_contributed and boundary_target_contribution. "
         "Call before issuing a new set_boundary_minor_split at a nested boundary.")
      .def("get_boundary_target_contribution", &c10d::ProcessGroupULFM::get_boundary_target_contribution,
         "Get the boundary-phase target contribution value for this rank.")
      .def("set_boundary_target_contribution", &c10d::ProcessGroupULFM::set_boundary_target_contribution,
         py::arg("value"),
         "Set the boundary-phase target contribution (non-negative).")
      .def("should_contribute", &c10d::ProcessGroupULFM::should_contribute,
         "Return True if this rank still owes a contribution. During the extended "
         "pass at a policy boundary compares boundary_contributed < boundary_target_contribution; "
         "otherwise compares contributed < target_contribution.")
      .def("elect_promotion", &c10d::ProcessGroupULFM::elect_promotion,
         py::arg("failed_majors"), py::arg("failed_minors"),
         "Elect spare promotion via collective. Returns True if THIS rank was promoted")
      .def("record_failure", [](c10d::ProcessGroupULFM& self,
                                const std::vector<int>& failed_ranks,
                                const c10d::ULFMOptions& ulfm_opts,
                                c10::intrusive_ptr<c10d::ProcessGroupULFM::WorkULFM> ulfm_work) {
          self.record_and_handling_failure(failed_ranks, ulfm_opts, ulfm_work.get());
      }, py::arg("failed_ranks"), py::arg("ulfm_opts"), py::arg("ulfm_work"),
         "Combined helper: track rank types, compute failures, auto-elect, and record");

  py::class_<c10d::ProcessGroupULFM::WorkULFM, c10d::Work, c10::intrusive_ptr<c10d::ProcessGroupULFM::WorkULFM>>(m, "WorkULFM")
      .def("has_failures", &c10d::ProcessGroupULFM::WorkULFM::has_failures)
      .def("get_failed_ranks", &c10d::ProcessGroupULFM::WorkULFM::get_failed_ranks)
      .def("get_failed_world_ranks", &c10d::ProcessGroupULFM::WorkULFM::get_failed_world_ranks)
      .def("was_noop", &c10d::ProcessGroupULFM::WorkULFM::was_noop,
         "Check if this work was marked as a no-op due to failures")
      .def("markNoop", &c10d::ProcessGroupULFM::WorkULFM::markNoop,
         "Mark this work as a no-op (used internally for failure handling)")
      .def("get_failure_stats", [](const c10d::ProcessGroupULFM::WorkULFM& self) {
          const auto& stats = self.get_failure_stats();
          return py::make_tuple(stats.failed_majors, stats.failed_minors,
                                stats.failed_major_spares, stats.failed_minor_spares,
                                stats.failed_boundary_minors,
                                stats.at_policy_boundary);
      }, "Get failure stats as tuple: (failed_majors, failed_minors, failed_major_spares, failed_minor_spares, failed_boundary_minors, at_policy_boundary)")
      .def("get_current_counts", [](const c10d::ProcessGroupULFM::WorkULFM& self) {
          const auto& counts = self.get_current_counts();
          return py::make_tuple(counts.majors, counts.minors,
                                counts.major_spares, counts.minor_spares,
                                counts.boundary_minors,
                                counts.contributed,
                                counts.boundary_contributed);
      }, "Get current rank type counts as tuple: (majors, minors, major_spares, minor_spares, boundary_minors, contributed, boundary_contributed)");

  // ULFM logging control
  m.def("set_ulfm_verbose_logging", &c10d::set_ulfm_verbose_logging, 
        py::arg("verbose"), 
        "Enable or disable verbose ULFM logging");
  
  m.def("is_ulfm_verbose_logging", &c10d::is_ulfm_verbose_logging, 
        "Check if verbose ULFM logging is enabled");
}
