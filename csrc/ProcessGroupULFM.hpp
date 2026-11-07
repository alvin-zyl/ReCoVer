#pragma once

// #ifdef USE_C10D_MPI

#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/core/ivalue_inl.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <mpi.h>
#include <torch/python.h>
#include <pybind11/chrono.h>
#include "TypesULFM.hpp"

namespace c10d {

constexpr const char* ULFM_BACKEND_NAME = "mpi";

// Recovery status enum to differentiate between different outcomes
enum class RecoveryStatus {
  NO_FAILURE,        // No failure detected, system is healthy
  RECOVERED,         // Failure(s) detected and successfully recovered
  FAILED_TO_RECOVER, // Failure(s) detected but recovery failed
  NOT_RECOVERED      // Failure(s) detected but recovery not attempted
};

// Result type for recovery operations that carries status, error info, and failure details
struct RecoveryResult {
  RecoveryStatus status;
  std::string error_message;
  int error_code;               // MPI error code if applicable (0 if not applicable)
  int num_failed_ranks;         // Number of failed ranks detected (0 if no failure)

  // Constructors
  RecoveryResult()
      : status(RecoveryStatus::NO_FAILURE), error_code(0), num_failed_ranks(0) {}

  RecoveryResult(RecoveryStatus s, const std::string& msg = "", int code = 0, int failed = 0)
      : status(s), error_message(msg), error_code(code), num_failed_ranks(failed) {}

  // Factory methods for convenience
  static RecoveryResult NoFailure() {
    return RecoveryResult(RecoveryStatus::NO_FAILURE);
  }

  static RecoveryResult Recovered(int num_failed = 0) {
    return RecoveryResult(RecoveryStatus::RECOVERED, "", 0, num_failed);
  }

  static RecoveryResult Error(const std::string& msg, int code = 0) {
    return RecoveryResult(RecoveryStatus::FAILED_TO_RECOVER, msg, code, 0);
  }

  static RecoveryResult NotRecovered(int num_failed = 0) {
    return RecoveryResult(RecoveryStatus::NOT_RECOVERED, "", 0, num_failed);
  }

  // Query methods
  bool is_ok() const {
    return status == RecoveryStatus::NO_FAILURE || status == RecoveryStatus::RECOVERED;
  }

  bool has_failure() const {
    return status == RecoveryStatus::RECOVERED || status == RecoveryStatus::FAILED_TO_RECOVER ||
           status == RecoveryStatus::NOT_RECOVERED;
  }

  bool recovered() const {
    return status == RecoveryStatus::RECOVERED;
  }

  bool failed_to_recover() const {
    return status == RecoveryStatus::FAILED_TO_RECOVER;
  }

  // Implicit conversion to bool for backward compatibility (true if ok)
  operator bool() const { return is_ok(); }
};

// WorkEntry is the state associated with a single MPI run instance.
// It include the source Tensor list and destination Tensor list, as well as
// The actual run function that will operate either on src or dst or both.
struct WorkEntry {
  explicit WorkEntry(
      std::vector<at::Tensor>* srcPtr,
      std::vector<at::Tensor>* dstPtr,
      std::function<void(std::unique_ptr<WorkEntry>&)> run)
      : dst(dstPtr ? *dstPtr : std::vector<at::Tensor>()), run(std::move(run)) {
    if (srcPtr) {
      src = *srcPtr;
    }
  }

  // Not copyable
  WorkEntry(const WorkEntry&) = delete;
  // Not copy assignable
  WorkEntry& operator=(const WorkEntry&) = delete;

  // For input and output tensors (in-place), we will always use src
  std::vector<at::Tensor> src;

  // Copy of user provided outputs.
  const std::vector<at::Tensor> dst;

  // src rank returned, for recv only
  int* srcRank = nullptr;
  std::function<void(std::unique_ptr<WorkEntry>&)> run;
  
  // For ULFM work entries - pointer to WorkULFM for failure recording  
  void* ulfmWork = nullptr;
};

// ProcessGroupMPI implements MPI bindings for c10d.
//
// All functions on this class are expected to be called in the same
// order across processes in the group. This is the only way that we
// can guarantee to match up the same calls across processes.
//
// All MPI functions provided by this class is asynchronously scheduled on a
// Worker thread. Therefore, ProcessGroupMPI requires the MPI implementation
// that is used to have a minimum thread support value of MPI_THREAD_SERIALIZED.
// That is, The process may be multi-threaded, and multiple threads may make
// MPI calls, but only one at a time: MPI calls are not made concurrently from
// two distinct threads (all MPI calls are serialized). However, with
// MPI_THREAD_SERIALIZED, ProcessGroupMPI will only support a single process
// group. In other words, no more than 1 process group can be created globally.
//
// If you would like to use multiple ProcessGroupMPI, it requires your MPI
// implementation to have a thread support value of MPI_THREAD_MULTIPLE, that
// is, multiple threads may call MPI, with no restriction.
//
// Also note that ProcessGroupMPI only supports a single Tensor operation. In
// other words, the size of the input Tensor vector should always be 1.
//
// CUDA tensor can be supported if the MPI used is CUDA-aware MPI, and
// ProcessGroupMPI will automatically detect this support.
class TORCH_API ProcessGroupULFM : public ProcessGroup {
 public:
  class WorkMPI : public Work {
   public:
    explicit WorkMPI(
        std::vector<at::Tensor> outputTensors,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors =
            std::nullopt)
        : Work(-1, OpType::UNKNOWN, profilingTitle, inputTensors),
          outputTensors_(std::move(outputTensors)),
          future_(c10::make_intrusive<at::ivalue::Future>(
              c10::ListType::create(c10::TensorType::get()))) {}

    std::vector<at::Tensor> result() override;

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

   protected:
    friend class ProcessGroupULFM;

   private:
    void finishWorkMPI();
    void finishWorkMPIError(const std::exception_ptr& eptr);

    std::vector<at::Tensor> outputTensors_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
  };

  class WorkULFM : public WorkMPI {
   public:
    explicit WorkULFM(
        std::vector<at::Tensor> outputTensors,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors =
            std::nullopt)
        : WorkMPI(std::move(outputTensors), profilingTitle, inputTensors),
          hasFailures_(false) {}

    // Failure detection methods (no recovery)
    bool has_failures() const;
    std::vector<int> get_failed_ranks() const;
    // The same failed processes as ranks of MPI_COMM_WORLD (empty if the
    // work saw no failure).
    std::vector<int> get_failed_world_ranks() const;
    bool was_noop() const {
      return was_noop_.load(std::memory_order_acquire);
    }
    void markNoop() {
      was_noop_.store(true, std::memory_order_release);
    }

    // Failure statistics and current counts accessors
    const FailureStats& get_failure_stats() const { return failureStats_; }
    const RankTypeCounts& get_current_counts() const { return currentCounts_; }

   protected:
    friend class ProcessGroupULFM;
    void recordFailure(const std::vector<int>& failedRanks,
                       const FailureStats& failureStats,
                       const RankTypeCounts& currentCounts);
    void recordFailedWorldRanks(const std::vector<int>& failedWorldRanks);

   private:
    mutable std::mutex failureMutex_;
    bool hasFailures_;
    std::vector<int> failedRanks_;
    std::vector<int> failedWorldRanks_;
    std::atomic<bool> was_noop_{false};

    FailureStats failureStats_;
    RankTypeCounts currentCounts_;
  };

  class AsyncWork : public Work {
   public:
    AsyncWork(
        MPI_Request request,
        std::vector<at::Tensor> outputTensors,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors =
            std::nullopt);

    ~AsyncWork() override;

    bool isCompleted() override;

    bool isSuccess() const override;

    int sourceRank() const override;

    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;

    void abort() override;

    std::vector<at::Tensor> result() override;

   protected:
    void populateException();

   private:
    const std::vector<at::Tensor> outputTensors_;
    MPI_Request request_;
    MPI_Status status_{};
  };

  // Constructor will spawn up the worker thread loop
  explicit ProcessGroupULFM(int rank, int size, MPI_Comm pgComm);

  ~ProcessGroupULFM() override;

  // Abort the MPI program, needs to be called when exception is detected
  void abort() override;

  const std::string getBackendName() const override {
      return std::string(ULFM_BACKEND_NAME);
    }

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> ulfm_allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions(),
      const ULFMOptions& ulfm_opts = ULFMOptions()
    );

  // Recovery methods (ProcessGroup-level operations)
  RecoveryResult repair_communicator();
  void notify_all_ranks_of_failure();
  bool check_for_failures() const;

  void set_quiesce(bool v) {
    quiesce_.store(v, std::memory_order_release);
  }
  bool is_quiesced() const {
    return quiesce_.load(std::memory_order_acquire);
  }

  // Epoch (bumps after repair)
  int worldEpoch() const noexcept {
    return world_epoch_.load(std::memory_order_acquire);
  }

  // Query current rank and size (may change after repairs)
  int current_rank() const noexcept {
    return currentRank_;
  }

  int current_size() const noexcept {
    return currentSize_;
  }

  // Minor rank flag: minor procs may zero their loss at the last few steps
  // to match exact global batch size configuration
  void set_minor();
  void reset_minor();
  bool is_minor() const {
    return is_minor_.load(std::memory_order_acquire);
  }

  // Set the major/minor split boundary: ranks < boundary are major, ranks >= boundary are minor
  void set_major_minor_split(int boundary);

  // Boundary minor rank flag: boundary minor procs are at the boundary between major and minor
  void set_boundary_minor();
  void reset_boundary_minor();
  bool is_boundary_minor() const {
    return is_boundary_minor_.load(std::memory_order_acquire);
  }

  // Set the boundary minor split and directly populate the boundary-phase
  // contribution target for this rank:
  //   ranks <  num_boundary_majors -> boundary_target_contribution_ = boundary_major_workload
  //   ranks >= num_boundary_majors -> boundary_target_contribution_ = boundary_minor_workload
  // boundary_contributed_ is reset to 0 on every call. The caller is
  // responsible for sizing the two workloads so they sum to the total
  // global contribution target required by the policy.
  void set_boundary_minor_split(int num_boundary_majors,
                                int64_t boundary_major_workload,
                                int64_t boundary_minor_workload);

  // Set rank type based on explicit counts
  // Layout: [major workers | major spares | minor workers | minor spares]
  // - Ranks 0 to (num_majors - 1): Major workers
  // - Ranks num_majors to (num_majors + num_major_spares - 1): Major spares
  // - Ranks (num_majors + num_major_spares) to (num_majors + num_major_spares + num_minors - 1): Minor workers
  // - Remaining ranks: Minor spares
  void set_major_minor_split_with_spares(int num_majors, int num_minors, int num_major_spares, int num_minor_spares);

  // Spare rank flag: spare procs are standby workers that can replace failed ranks
  void set_spare();
  void reset_spare();
  bool is_spare() const {
    return is_spare_.load(std::memory_order_acquire);
  }

  // Count all rank types via single MPI_Allreduce (reusable helper)
  // Can be called from consensus, ulfm_allreduce, or independently
  void count_rank_types(RankTypeCounts& counts);

  // Update rank type counts directly (without MPI communication)
  void update_rank_type_counts(const RankTypeCounts& counts);

  // Count rank types via MPI_Allreduce and update internal counters
  RankTypeCounts count_and_update_rank_types();

  // Compute failed counts from before/after survivor counts
  // Input: old counts (before failure), new counts (after failure)
  // Output: populates failure_stats
  void compute_failed_counts(const RankTypeCounts& old_counts,
                             const RankTypeCounts& new_counts,
                             FailureStats& failure_stats);

  // Check if at policy boundary based on failure stats and current counts
  // Returns true if: major failed with no major spares OR minor failed with no minor spares
  static bool check_at_policy_boundary(const FailureStats& failure_stats,
                                       const RankTypeCounts& current_counts);

  // Update PG-level policy boundary flag
  // Once set to true, stays true (sticky) until explicitly reset
  void update_policy_boundary(bool event_boundary);

  // Get PG-level policy boundary flag
  bool is_at_policy_boundary() const {
    return atPolicyBoundary_.load(std::memory_order_acquire);
  }

  // Reset PG-level policy boundary flag (e.g., after policy change)
  void reset_policy_boundary() {
    atPolicyBoundary_.store(false, std::memory_order_release);
  }

  // Elect spare promotion via collective
  // Returns true if THIS rank was promoted
  // Automatically calls reset_spare() on promoted ranks
  bool elect_promotion(int failed_majors, int failed_minors);

  // Combined helper: track rank types, compute failures, auto-elect, and record
  // Encapsulates the entire failure handling workflow for reuse
  void record_and_handling_failure(const std::vector<int>& failed_ranks,
                      const ULFMOptions& ulfm_opts,
                      WorkULFM* ulfm_work);

  // Getters for rank type counts
  int get_num_major_procs() const {
    return num_major_procs_.load(std::memory_order_acquire);
  }
  int get_num_minor_procs() const {
    return num_minor_procs_.load(std::memory_order_acquire);
  }
  int get_num_major_spare_procs() const {
    return num_major_spare_procs_.load(std::memory_order_acquire);
  }
  int get_num_minor_spare_procs() const {
    return num_minor_spare_procs_.load(std::memory_order_acquire);
  }
  int get_num_boundary_minor_procs() const {
    return num_boundary_minor_procs_.load(std::memory_order_acquire);
  }

  // Local count of how many times this rank contributed gradients to allreduce.
  // Call increment_contributed() from the Python control plane when this rank's
  // gradient was not zeroed before the allreduce. During the extended pass at
  // a policy boundary (is_at_policy_boundary() == true) all ranks route their
  // increments into boundary_contributed_ so the boundary phase is tracked
  // separately from the regular accumulation window.
  int64_t get_contributed() const {
    return contributed_.load(std::memory_order_acquire);
  }
  void increment_contributed() {
    if (is_at_policy_boundary()) {
      boundary_contributed_.fetch_add(1, std::memory_order_relaxed);
    } else {
      contributed_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  void reset_contributed() {
    contributed_.store(0, std::memory_order_release);
    boundary_contributed_.store(0, std::memory_order_release);
  }

  // Boundary-phase counterpart of contributed_: incremented by every rank
  // during the extended microbatches at a policy boundary.
  int64_t get_boundary_contributed() const {
    return boundary_contributed_.load(std::memory_order_acquire);
  }
  void reset_boundary_contributed() {
    boundary_contributed_.store(0, std::memory_order_release);
  }
  // Fold the boundary-phase contribution counter into the regular counter
  // and zero both boundary_contributed_ and boundary_target_contribution_
  // (the stale target from the previous extension is repopulated by the
  // next set_boundary_minor_split call). Called when a failure is observed
  // during an active boundary extended pass so the next extension can
  // stack on top of the already-accounted-for contributions.
  void merge_boundary_contributed() {
    int64_t boundary =
        boundary_contributed_.exchange(0, std::memory_order_acq_rel);
    if (boundary != 0) {
      contributed_.fetch_add(boundary, std::memory_order_relaxed);
    }
    boundary_target_contribution_.store(0, std::memory_order_release);
  }

  // Target contribution: a settable/incrementable goal value controlled from
  // the Python control plane (e.g. expected number of gradient contributions).
  // set replaces the value; increment adds a positive delta.
  int64_t get_target_contribution() const {
    return target_contribution_.load(std::memory_order_acquire);
  }
  void set_target_contribution(int64_t major_value, int64_t minor_value = -1) {
    TORCH_CHECK(major_value > 0, "target_contribution major_value must be positive");
    int64_t effective_minor = (minor_value <= 0) ? major_value : minor_value;
    TORCH_CHECK(effective_minor > 0, "target_contribution minor_value must be positive");
    int64_t value = is_minor() ? effective_minor : major_value;
    target_contribution_.store(value, std::memory_order_release);
  }
  void increment_target_contribution(int64_t delta = 1) {
    TORCH_CHECK(delta >= 0, "target_contribution delta must non negative");
    target_contribution_.fetch_add(delta, std::memory_order_relaxed);
  }

  // Boundary-phase target: how many contributions a rank should make during
  // the extended pass at a policy boundary (workload for non-boundary-minor
  // ranks, workload - 1 for boundary-minor ranks).
  int64_t get_boundary_target_contribution() const {
    return boundary_target_contribution_.load(std::memory_order_acquire);
  }
  void set_boundary_target_contribution(int64_t value) {
    TORCH_CHECK(value >= 0, "boundary_target_contribution must be non-negative");
    boundary_target_contribution_.store(value, std::memory_order_release);
  }

  // Returns true if this rank has not yet reached its target contribution.
  // During the extended pass at a policy boundary all ranks compare against
  // the boundary-phase counters so the boundary workload is tracked
  // independently from the regular accumulation window.
  bool should_contribute() const {
    if (is_at_policy_boundary()) {
      return boundary_contributed_.load(std::memory_order_acquire) <
             boundary_target_contribution_.load(std::memory_order_acquire);
    }
    return contributed_.load(std::memory_order_acquire) <
           target_contribution_.load(std::memory_order_acquire);
  }

  // Comprehensive failure detection and recovery workflow
  RecoveryResult detect_and_recover_failures(bool auto_repair = true, std::vector<int>* failed_ranks = nullptr, int max_retries = 5, std::vector<int>* failed_world_ranks = nullptr);

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensor,
      int tag) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<Work> consensus(
    const ULFMOptions& ulfm_opts = ULFMOptions());

  // Creating a new ProcessGroupMPI, will initialize MPI if not initialized
  static c10::intrusive_ptr<ProcessGroup> createProcessGroupULFM(
      std::vector<int> ranks = {});

  static void ProcessGroupULFMConstructor() __attribute__((constructor)) {
    py::object module = py::module::import("torch.distributed");
    py::object register_backend =
        module.attr("Backend").attr("register_backend");
    register_backend("ulfm", py::cpp_function(createProcessGroupULFM));
  }

 protected:
  using WorkType =
      std::tuple<std::unique_ptr<WorkEntry>, c10::intrusive_ptr<WorkMPI>>;
  // Worker thread loop
  void runLoop();
  // Helper function that is called by the destructor
  void destroy();

  // Modular failure recovery helper methods (corrected workflow order)
  bool notice_failure();
  RecoveryResult get_failed_ranks_internal(std::vector<int>& failed_ranks_comm, std::vector<int>* failed_ranks_world = nullptr, int max_retries = 5);
  bool should_repair_communicator(const std::vector<int>& failed_ranks);
  RecoveryResult repair_communicator_internal();
  void bumpEpoch() {
    world_epoch_.fetch_add(1, std::memory_order_acq_rel);
  }

  c10::intrusive_ptr<Work> enqueue(
      std::unique_ptr<WorkEntry> entry,
      const char* profilingTitle = nullptr,
      const std::optional<std::vector<at::Tensor>>& inputTensors =
          std::nullopt);

  c10::intrusive_ptr<WorkULFM> enqueueULFM(
      std::unique_ptr<WorkEntry> entry,
      const char* profilingTitle = nullptr,
      const std::optional<std::vector<at::Tensor>>& inputTensors =
          std::nullopt);

  bool stop_;

  std::mutex pgMutex_;
  std::thread workerThread_;

  std::deque<WorkType> queue_;
  std::condition_variable queueProduceCV_;
  std::condition_variable queueConsumeCV_;

  // Global states
  static void initMPIOnce();
  static void mpiExit();

  static std::mutex pgGlobalMutex_;
  static int mpiThreadSupport_;

  MPI_Comm pgComm_;
  // The communicator's group as created. Ranks that torch passes as roots
  // are ranks of this group; after a shrink they must be mapped to the
  // current communicator (currentRankOf).
  MPI_Group originalGroup_ = MPI_GROUP_NULL;
  int currentRankOf(int originalRank) const;

 public:
  // Rank, in the group as created, of the process that has rank
  // `current_rank` in the current (possibly shrunk) communicator; -1 if
  // there is no such rank. Identity until the first shrink.
  int original_rank_of(int current_rank);

 private:
  int currentRank_;  // Current rank after repairs (may change)
  int currentSize_;  // Current size after repairs (may change)

 private:
  std::atomic<bool> quiesce_{false};
  std::atomic<int>  world_epoch_{0};
  std::atomic<bool> is_minor_{false};  // Minor rank flag for major/minor split
  std::atomic<bool> is_spare_{false};  // Spare rank flag for standby workers
  std::atomic<bool> is_boundary_minor_{false};  // Boundary minor rank flag
  std::atomic<bool> atPolicyBoundary_{false};  // PG-level policy boundary flag (sticky once set)

  // Results from rank type counting
  std::atomic<int> num_major_procs_{0};       // !is_minor && !is_spare
  std::atomic<int> num_minor_procs_{0};       // is_minor && !is_spare
  std::atomic<int> num_major_spare_procs_{0}; // !is_minor && is_spare
  std::atomic<int> num_minor_spare_procs_{0}; // is_minor && is_spare
  std::atomic<int> num_boundary_minor_procs_{0}; // is_boundary_minor

  // Local count of gradient contributions (incremented each time this rank's gradient not being zeroed
  std::atomic<int64_t> contributed_{0};

  // Target contribution: settable/incrementable goal value controlled from Python
  std::atomic<int64_t> target_contribution_{0};

  // Boundary-phase counterparts used only while is_boundary_minor_ is true.
  std::atomic<int64_t> boundary_contributed_{0};
  std::atomic<int64_t> boundary_target_contribution_{0};
};

} // namespace c10d

// #endif // USE_C10D_MPI
