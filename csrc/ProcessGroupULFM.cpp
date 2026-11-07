#include "ProcessGroupULFM.hpp"
#include "TypesULFM.hpp"
#include "ULFMLogging.hpp"

// #ifdef USE_C10D_MPI

#include <iostream>
#include <map>

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/irange.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h> // Needed for CUDA-aware check
#endif

namespace c10d {

#define MPI_CHECK(cmd)                                                   \
  do {                                                                   \
    int mpiStatus = cmd;                                                 \
    if (mpiStatus != MPI_SUCCESS) {                                      \
      std::string err = "MPI error in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) +                                     \
          ", with error code: " + std::to_string(mpiStatus);             \
      TORCH_CHECK(false, err);                                           \
    }                                                                    \
  } while (0)

#define ULFM_MPI_CHECK(cmd, rank)                                       \
  do {                                                                   \
    int mpiStatus = cmd;                                                 \
    if (mpiStatus != MPI_SUCCESS) {                                      \
      int errClass;                                                      \
      MPI_Error_class(mpiStatus, &errClass);                             \
      if (errClass == MPIX_ERR_PROC_FAILED) {                            \
        /* Process failure detected, do not terminate */                 \
        ULFM_LOG_WARN(rank, "Process failure detected at " << __FILE__ << ":" << __LINE__); \
        break;                                                           \
      }                                                                  \
      std::string err = "MPI error in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) +                                     \
          ", with error code: " + std::to_string(mpiStatus);             \
      TORCH_CHECK(false, err);                                           \
    }                                                                    \
  } while (0)

namespace {

// DEPRECATED: This static helper is no longer used. Use ProcessGroupULFM::get_failed_ranks_internal() instead.
// Kept for backward compatibility but should be removed in future versions.
// Note: This function still uses ULFM_LOG_ERROR which throws immediately, making return false unreachable.
static bool get_failed_ranks(MPI_Comm comm,
                             const int& rank,
                             const int& size,
                             std::vector<int>& failed_ranks_comm,
                             std::vector<int>* failed_ranks_world = nullptr) {
  failed_ranks_comm.clear();
  if (failed_ranks_world) failed_ranks_world->clear();

  // MPI_Barrier(comm);
  int flag = 1;
  int rc_flag = MPIX_Comm_agree(comm, &flag);

  int num_acked;
  int rc_ack = MPIX_Comm_ack_failed(comm, size, &num_acked);
  if (rc_ack != MPI_SUCCESS) {
    ULFM_LOG_ERROR(rank, "Failed to ack failures");
    return false;
  }

  MPI_Group failed_grp = MPI_GROUP_NULL;
  int rc_get = MPIX_Comm_get_failed(comm, &failed_grp);
  if (rc_get != MPI_SUCCESS || failed_grp == MPI_GROUP_NULL) {
    ULFM_LOG_ERROR(rank, "Failed to get failed group");
    return false;
  }

  int fsize = 0;
  MPI_Group_size(failed_grp, &fsize);
  if (fsize <= 0) {
    MPI_Group_free(&failed_grp);
    std::string err = "[ULFM Rank " + std::to_string(rank) + "] Got failed group size " + std::to_string(fsize);
    TORCH_CHECK(false, err);
    return false; // nothing recorded yet
  }

  // Build index array 0..fsize-1 in the failed group's own indexing
  std::vector<int> idx(fsize);
  for (int i = 0; i < fsize; ++i) idx[i] = i;

  // Translate to ranks in 'comm'
  MPI_Group comm_grp = MPI_GROUP_NULL;
  MPI_Comm_group(comm, &comm_grp);
  failed_ranks_comm.resize(fsize);
  MPI_Group_translate_ranks(failed_grp, fsize, idx.data(),
                            comm_grp, failed_ranks_comm.data());
  MPI_Group_free(&comm_grp);

  ULFM_LOG_DEBUG(rank, "Number of failures acked: " << num_acked);
  if (is_ulfm_verbose_logging() && fsize > 0) {
    std::ostringstream oss;
    oss << "Failed ranks in comm: ";
    for (int i = 0; i < fsize; ++i) {
      if (i > 0) oss << ", ";
      oss << failed_ranks_comm[i];
    }
    std::string debug_msg = oss.str();
    if (!debug_msg.empty() && debug_msg != "Failed ranks in comm: ") {
      ULFM_LOG_DEBUG(rank, debug_msg);
    }
  }

  // Optionally translate to MPI_COMM_WORLD ranks
  if (failed_ranks_world) {
    MPI_Group world_grp = MPI_GROUP_NULL;
    MPI_Comm_group(MPI_COMM_WORLD, &world_grp);
    failed_ranks_world->resize(fsize);
    MPI_Group_translate_ranks(failed_grp, fsize, idx.data(),
                              world_grp, failed_ranks_world->data());
    MPI_Group_free(&world_grp);
  }

  MPI_Group_free(&failed_grp);
  return true;
}

// Op mapping
std::map<ReduceOp::RedOpType, MPI_Op> mpiOp = {
    {ReduceOp::MIN, MPI_MIN},
    {ReduceOp::MAX, MPI_MAX},
    {ReduceOp::SUM, MPI_SUM},
    {ReduceOp::PRODUCT, MPI_PROD},
};
// Type mapping
std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
    // 2-byte types treated as opaque 16-bit units.
    // Correct for bit-copy ops (broadcast, allgather, scatter, send, recv).
    // For allreduce on these types, upcast to float before MPI reduce (see allreduce impl).
    {at::kBFloat16, MPI_UNSIGNED_SHORT},
    {at::kHalf,     MPI_UNSIGNED_SHORT},
    {at::kBool,     MPI_UNSIGNED_CHAR},
};

// Checking CUDA-aware MPI support, currently we only support CUDA aware
// MPI ops through Open MPI
bool cudaAwareMpiCheck() {
// Run time check
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (MPIX_Query_cuda_support() == 1) {
    return true;
  } else {
    return false;
  }
#else // !defined(MPIX_CUDA_AWARE_SUPPORT)
  return false;
#endif // MPIX_CUDA_AWARE_SUPPORT
}

// Checking the input tensor's validity
void checkSingleTensorHelper(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    TORCH_CHECK(false, "input tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    TORCH_CHECK(false, "input tensor has to be dense");
  }
  if (tensor.is_cuda() && !cudaAwareMpiCheck()) {
    TORCH_CHECK(
        false,
        "CUDA tensor detected and the MPI used doesn't "
        "have CUDA-aware MPI support");
  }
}

void checkSingleTensor(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    TORCH_CHECK(
        false, "MPI process group does not support multi-GPU collectives");
  }
  checkSingleTensorHelper(tensors[0]);
}

void checkSameSizeAndType(
    const at::Tensor& t_in,
    const std::vector<at::Tensor>& tensors) {
  for (const auto& tensor : tensors) {
    if ((tensor.numel() != t_in.numel()) ||
        (tensor.scalar_type() != t_in.scalar_type())) {
      TORCH_CHECK(false, "Tensors are not equal in size or data type");
    }
    checkSingleTensorHelper(tensor);
  }
}

} // namespace

std::vector<at::Tensor> ProcessGroupULFM::WorkMPI::result() {
  return outputTensors_;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupULFM::WorkMPI::getFuture() {
  return future_;
}

void ProcessGroupULFM::WorkMPI::finishWorkMPIError(
    const std::exception_ptr& eptr) {
  future_->setError(eptr);
  finish(eptr);
}

void ProcessGroupULFM::WorkMPI::finishWorkMPI() {
  future_->markCompleted(at::IValue(outputTensors_));
  finish();
}

ProcessGroupULFM::AsyncWork::AsyncWork(
    MPI_Request request,
    std::vector<at::Tensor> outputTensors,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors)
    : Work(-1, OpType::UNKNOWN, profilingTitle, inputTensors),
      outputTensors_(std::move(outputTensors)),
      request_(request) {
  memset(&status_, 0, sizeof(status_));
}

ProcessGroupULFM::AsyncWork::~AsyncWork() {
  if (request_ != MPI_REQUEST_NULL) {
    std::cerr
        << "Attempted destruction of AsyncWork before work has completed, "
        << "terminating the program." << '\n';
    std::terminate();
  }
}

bool ProcessGroupULFM::AsyncWork::isCompleted() {
  if (request_ == MPI_REQUEST_NULL) {
    return true;
  }

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  int flag = 0;
  MPI_CHECK(MPI_Test(&request_, &flag, &status_));
  if (request_ != MPI_REQUEST_NULL) {
    return false;
  }

  // request_ == MPI_REQUEST_NULL; the work has completed
  // Populate exception if request was not successful
  if (status_.MPI_ERROR != MPI_SUCCESS) {
    populateException();
  }

  return true;
}

bool ProcessGroupULFM::AsyncWork::isSuccess() const {
  if (request_ != MPI_REQUEST_NULL) {
    TORCH_CHECK(
        false,
        "Invalid call to AsyncWork::isSuccess before work has completed");
  }

  return status_.MPI_ERROR == MPI_SUCCESS;
}

int ProcessGroupULFM::AsyncWork::sourceRank() const {
  return status_.MPI_SOURCE;
}

bool ProcessGroupULFM::AsyncWork::wait(std::chrono::milliseconds /* unused */) {
  if (request_ == MPI_REQUEST_NULL) {
    // AsyncWork needs to manually call profiling end callbacks if they are set,
    // since it does not call ProcessGroup::finish().
    if (Work::recordFunctionEndCallback_) {
      Work::recordFunctionEndCallback_();
      Work::recordFunctionEndCallback_ = nullptr;
    }
    return true;
  }

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  MPI_CHECK(MPI_Wait(&request_, &status_));
  auto ok = (status_.MPI_ERROR == MPI_SUCCESS);

  // AsyncWork needs to manually call profiling end callbacks if they are set,
  // since it does not call ProcessGroup::finish().
  if (Work::recordFunctionEndCallback_) {
    Work::recordFunctionEndCallback_();
    Work::recordFunctionEndCallback_ = nullptr;
  }

  if (!ok) {
    populateException();
    std::rethrow_exception(exception_);
  }
  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(
        c10::intrusive_ptr<
            ProcessGroupULFM::AsyncWork>::unsafe_reclaim_from_nonowning(this));
  }
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupULFM::AsyncWork::abort(){
    TORCH_CHECK(false, "ProcessGroupULFM::AsyncWork::abort not implemented.")}

std::vector<at::Tensor> ProcessGroupULFM::AsyncWork::result() {
  return outputTensors_;
}

void ProcessGroupULFM::AsyncWork::populateException() {
  std::array<char, MPI_MAX_ERROR_STRING> buf{};
  int len = buf.size();
  MPI_CHECK(MPI_Error_string(status_.MPI_ERROR, buf.data(), &len));
  exception_ =
      std::make_exception_ptr(std::runtime_error(std::string(buf.data(), len)));
}

// Static global states
int ProcessGroupULFM::mpiThreadSupport_ = 0;
std::mutex ProcessGroupULFM::pgGlobalMutex_;

void ProcessGroupULFM::mpiExit() {
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  MPI_CHECK(MPI_Finalize());
}

void ProcessGroupULFM::initMPIOnce() {
  // Initialize MPI environment. We only want to initialize once.
  static bool init_mpi_flag [[maybe_unused]] = []() {
    int mpi_was_initialized = 0;
    MPI_CHECK(MPI_Initialized(&mpi_was_initialized));
    if (mpi_was_initialized == 0) {
      MPI_CHECK(MPI_Init_thread(
          nullptr, nullptr, MPI_THREAD_SERIALIZED, &mpiThreadSupport_));
      if (mpiThreadSupport_ < MPI_THREAD_SERIALIZED) {
        TORCH_CHECK(
            false,
            "Used MPI implementation doesn't have the "
            "minimum level of threading support: "
            "MPI_THREAD_SERIALIZED. This is required by "
            "c10d package");
      }
      if (std::atexit(ProcessGroupULFM::mpiExit)) {
        TORCH_CHECK(false, "Fail to register the MPI exit handler");
      }
    } else {
      TORCH_WARN_ONCE("MPI was previously initialized.");
    }
    return true;
  }();
}

c10::intrusive_ptr<ProcessGroup> ProcessGroupULFM::createProcessGroupULFM(
    std::vector<int> ranks) {
  // Once initialization
  initMPIOnce();

  MPI_Comm groupComm = MPI_COMM_WORLD;
  int rank = -1;
  int size = -1;

  {
    std::lock_guard<std::mutex> globalLock(pgGlobalMutex_);

    // If no ranks are specified, assume we're creating the root group
    if (!ranks.empty()) {
      MPI_Group worldGroup{};
      MPI_Group ranksGroup{};
      MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &worldGroup));
      MPI_CHECK(
          MPI_Group_incl(worldGroup, ranks.size(), ranks.data(), &ranksGroup));
      // `MPI_Comm_create` can be flaky in certain cases.
      // See: https://github.com/pytorch/pytorch/issues/53899
      constexpr int kMaxNumRetries = 3;
      bool groupComm_updated = false;
      MPI_Barrier(MPI_COMM_WORLD);
      for (const auto i : c10::irange(kMaxNumRetries)) {
        (void)i;
        if (MPI_Comm_create(MPI_COMM_WORLD, ranksGroup, &groupComm)) {
          groupComm_updated = true;
          break;
        }
      }
      MPI_CHECK(groupComm_updated);
      MPI_CHECK(MPI_Group_free(&worldGroup));
      MPI_CHECK(MPI_Group_free(&ranksGroup));
    }

    // Fetch rank and world size for this group (MPI_COMM_WORLD or new)
    if (groupComm != MPI_COMM_NULL) {
      MPI_CHECK(MPI_Comm_rank(groupComm, &rank));
      MPI_CHECK(MPI_Comm_size(groupComm, &size));

      if (rank < 0 || size < 0) {
        TORCH_CHECK(false, "Failed to get the world_size / rank");
      }
    }
  }

  // If this process is not part of the group, we don't construct a
  // process group instance. This is in line with the semantics of the
  // other process group types.
  if (groupComm == MPI_COMM_NULL) {
    return c10::intrusive_ptr<ProcessGroupULFM>();
  }

  MPI_Comm_set_errhandler(groupComm, MPI_ERRORS_RETURN);
  return c10::make_intrusive<ProcessGroupULFM>(rank, size, groupComm);
}

ProcessGroupULFM::ProcessGroupULFM(int rank, int size, MPI_Comm pgComm)
    : ProcessGroup(rank, size), stop_(false), pgComm_(pgComm),
      currentRank_(rank), currentSize_(size) {
  if (pgComm_ == MPI_COMM_NULL) {
    TORCH_CHECK(false, "pgComm_ must not be MPI_COMM_NULL");
  }

  {
    std::lock_guard<std::mutex> globalLock(pgGlobalMutex_);
    MPI_Comm_group(pgComm_, &originalGroup_);
  }

  // Initialize num_major_procs_ to world size (all ranks start as major workers)
  num_major_procs_.store(size, std::memory_order_release);
  
  char hostname[256];
  gethostname(hostname, sizeof(hostname));

  ULFM_LOG_WARN(rank, "ULFM MPI Constructor initialized with rank " << rank << ", size " << size << ", on host " << hostname);

  // Start the worker thread accepting MPI calls
  workerThread_ = std::thread(&ProcessGroupULFM::runLoop, this);

  init();
}

ProcessGroupULFM::~ProcessGroupULFM() {
  destroy();
}

// A rank of the group as created, translated to the current (possibly
// shrunk) communicator; MPI_UNDEFINED if that process has been removed.
// Identity until the first shrink.
int ProcessGroupULFM::currentRankOf(int originalRank) const {
  if (currentSize_ == size_) {
    return originalRank;
  }
  MPI_Group currentGroup = MPI_GROUP_NULL;
  MPI_Comm_group(pgComm_, &currentGroup);
  int current = MPI_UNDEFINED;
  MPI_Group_translate_ranks(originalGroup_, 1, &originalRank, currentGroup, &current);
  MPI_Group_free(&currentGroup);
  return current;
}

int ProcessGroupULFM::original_rank_of(int current_rank) {
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  if (current_rank < 0 || current_rank >= currentSize_) {
    return -1;
  }
  if (currentSize_ == size_) {
    return current_rank;
  }
  MPI_Group currentGroup = MPI_GROUP_NULL;
  MPI_Comm_group(pgComm_, &currentGroup);
  int original = MPI_UNDEFINED;
  MPI_Group_translate_ranks(currentGroup, 1, &current_rank, originalGroup_, &original);
  MPI_Group_free(&currentGroup);
  return original == MPI_UNDEFINED ? -1 : original;
}

void ProcessGroupULFM::destroy() {
  std::unique_lock<std::mutex> lock(pgMutex_);
  queueConsumeCV_.wait(lock, [&] { return queue_.empty(); });

  // Queue is empty, signal stop
  stop_ = true;

  // Release lock to allow threads to terminate
  lock.unlock();
  queueProduceCV_.notify_all();

  // Join the single worker thread
  workerThread_.join();
}

void ProcessGroupULFM::abort() {
  destroy();
  MPI_Abort(pgComm_, EXIT_FAILURE);
}

void ProcessGroupULFM::runLoop() {
  // Set CUDA device on the worker thread so OpenMPI's CUDA-aware layer
  // can query the device context (cuCtxGetDevice) without error 201.
  // {
  //   int device_count = 0;
  //   auto err = cudaGetDeviceCount(&device_count);
  //   if (err == cudaSuccess && device_count > 0) {
  //     cudaSetDevice(rank_ % device_count);
  //   }
  // }

  std::unique_lock<std::mutex> lock(pgMutex_);

  while (!stop_) {
    if (queue_.empty()) {
      queueProduceCV_.wait(lock);
      continue;
    }

    auto workTuple = std::move(queue_.front());

    queue_.pop_front();

    auto& workEntry = std::get<0>(workTuple);
    auto& work = std::get<1>(workTuple);

    lock.unlock();
    queueConsumeCV_.notify_one();

    try {
      workEntry->run(workEntry);
      work->finishWorkMPI();
    } catch (...) {
      work->finishWorkMPIError(std::current_exception());
    }

    lock.lock();
  }
}

c10::intrusive_ptr<Work> ProcessGroupULFM::enqueue(
    std::unique_ptr<WorkEntry> entry,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors) {
  auto work =
      c10::make_intrusive<WorkMPI>(entry->dst, profilingTitle, inputTensors);
  std::unique_lock<std::mutex> lock(pgMutex_);
  queue_.emplace_back(std::move(entry), work);
  lock.unlock();
  queueProduceCV_.notify_one();
  return work;
}

c10::intrusive_ptr<ProcessGroupULFM::WorkULFM> ProcessGroupULFM::enqueueULFM(
    std::unique_ptr<WorkEntry> entry,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors) {
  auto work =
      c10::make_intrusive<WorkULFM>(entry->dst, profilingTitle, inputTensors);
  
  // Store work pointer in entry for access in runFunc
  entry->ulfmWork = static_cast<void*>(work.get());
  
  std::unique_lock<std::mutex> lock(pgMutex_);
  queue_.emplace_back(std::move(entry), work);
  lock.unlock();
  queueProduceCV_.notify_one();
  return work;
}

// WorkULFM method implementations
bool ProcessGroupULFM::WorkULFM::has_failures() const {
  std::lock_guard<std::mutex> lock(failureMutex_);
  return hasFailures_;
}

std::vector<int> ProcessGroupULFM::WorkULFM::get_failed_ranks() const {
  std::lock_guard<std::mutex> lock(failureMutex_);
  return failedRanks_;
}

std::vector<int> ProcessGroupULFM::WorkULFM::get_failed_world_ranks() const {
  std::lock_guard<std::mutex> lock(failureMutex_);
  return failedWorldRanks_;
}

void ProcessGroupULFM::WorkULFM::recordFailedWorldRanks(
    const std::vector<int>& failedWorldRanks) {
  std::lock_guard<std::mutex> lock(failureMutex_);
  failedWorldRanks_ = failedWorldRanks;
}

void ProcessGroupULFM::WorkULFM::recordFailure(
    const std::vector<int>& failedRanks,
    const FailureStats& failureStats,
    const RankTypeCounts& currentCounts) {
  std::lock_guard<std::mutex> lock(failureMutex_);
  hasFailures_ = true;
  failedRanks_ = failedRanks;
  failureStats_ = failureStats;
  currentCounts_ = currentCounts;
}

c10::intrusive_ptr<Work> ProcessGroupULFM::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  checkSingleTensor(tensors);
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        c10::DeviceGuard guard(data.device());
        // Ensure any pending async GPU ops (e.g. non_blocking CPU→GPU copies
        // from DDP's sync_bucket_indices) have completed before MPI reads
        // the buffer.  MPI_Bcast is a host-side call and does not
        // synchronize with the CUDA stream.
        if (data.is_cuda()) {
          cudaDeviceSynchronize();
        }
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        const int root = currentRankOf(opts.rootRank);
        TORCH_CHECK(root != MPI_UNDEFINED, "broadcast: root rank ",
                    opts.rootRank, " is no longer in the communicator");
        MPI_CHECK(MPI_Bcast(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            root,
            pgComm_));
      };
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  return enqueue(
      std::move(entry),
      "mpi:broadcast",
      std::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<Work> ProcessGroupULFM::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {

  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

        // BF16/FP16 have no native MPI reduction op (MPI_UNSIGNED_SHORT would
        // give wrong sums).  Upcast to FP32, reduce, then cast back.
        const bool needs_upcast = (data.scalar_type() == at::kBFloat16 ||
                                   data.scalar_type() == at::kHalf);
        at::Tensor reduce_buf = needs_upcast ? data.data().to(at::kFloat) : data;

        // ReduceOp::AVG has no native MPI equivalent; emulate with SUM + divide.
        MPI_Op mpi_op = (opts.reduceOp == ReduceOp::AVG)
            ? MPI_SUM
            : mpiOp.at(opts.reduceOp);
        MPI_CHECK(MPI_Allreduce(
            MPI_IN_PLACE,
            reduce_buf.data_ptr(),
            reduce_buf.numel(),
            mpiDatatype.at(reduce_buf.scalar_type()),
            mpi_op,
            pgComm_));
        if (opts.reduceOp == ReduceOp::AVG) {
          reduce_buf.div_(static_cast<double>(size_));
        }
        if (needs_upcast) {
          // Write result back into original tensor without triggering autograd.
          data.data().copy_(reduce_buf);
        }
      };
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  return enqueue(
      std::move(entry),
      "mpi:all_reduce",
      std::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<Work> ProcessGroupULFM::ulfm_allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts,
    const ULFMOptions& ulfm_opts) {

  checkSingleTensor(tensors);
  const int epoch_at_enqueue = worldEpoch();

  // Record a CUDA event on the caller's current stream so the worker thread
  // can block on GPU-side producers (reduce_scatter, snapshot clone, etc.)
  // without the caller having to torch.cuda.synchronize() on the main thread.
  std::shared_ptr<at::cuda::CUDAEvent> input_ready;
  if (tensors[0].is_cuda()) {
    input_ready = std::make_shared<at::cuda::CUDAEvent>();
    input_ready->record(
        c10::cuda::getCurrentCUDAStream(tensors[0].device().index()));
  }

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, ulfm_opts, this, epoch_at_enqueue, input_ready](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        c10::DeviceGuard guard(data.device());
        if (input_ready) {
          input_ready->synchronize();
        }
        ULFM_LOG_DEBUG(currentRank_, "ulfm_allreduce: waiting for pgGlobalMutex_");
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        ULFM_LOG_DEBUG(currentRank_, "ulfm_allreduce: pgGlobalMutex_ acquired");

        // Get the WorkULFM instance to record failures
        WorkULFM* ulfm_work = static_cast<WorkULFM*>(entry->ulfmWork);

        // Early NOOP if quiesced
        if (is_quiesced()) {
          if (ulfm_work) ulfm_work->markNoop();
          ULFM_LOG_INFO(currentRank_, "Quiesced before entering ULFM logic, marked as NOOP");
          return;
        }

        // Use the new modular failure recovery system
        ULFM_LOG_DEBUG(currentRank_, "ulfm_allreduce: entering detect_and_recover_failures");
        std::vector<int> failed_ranks, failed_world_ranks;
        RecoveryResult recovery_success = detect_and_recover_failures(ulfm_opts.auto_repair, &failed_ranks, ulfm_opts.max_retries, &failed_world_ranks);
        ULFM_LOG_DEBUG(currentRank_, "ulfm_allreduce: detect_and_recover_failures returned");

        // Always record failure for inspection if any were detected
        if (recovery_success.has_failure()) {
          record_and_handling_failure(failed_ranks, ulfm_opts, ulfm_work);
          if (ulfm_work) ulfm_work->recordFailedWorldRanks(failed_world_ranks);
        }
        
        // If quiesced (latch) or epoch changed during detect/repair, NOOP
        if (is_quiesced() || worldEpoch() != epoch_at_enqueue) {
          if (ulfm_work) ulfm_work->markNoop();
          ULFM_LOG_INFO(currentRank_, "Quiesced: " << (is_quiesced() ? "true" : "false") 
                                     << " or epoch changed: " << (worldEpoch() != epoch_at_enqueue ? "true" : "false") 
                                     << " during ULFM logic, marked as NOOP");
          return;
        }
        
        if (!failed_ranks.empty()) {
          if (ulfm_opts.auto_repair) {
            if (!recovery_success) {
              // Auto-repair failed, throw exception
              std::string error_msg = "[ULFM Rank " + std::to_string(currentRank_) +
                                      "] Auto-repair failed after detecting failures: ";
              for (int failed_rank : failed_ranks) {
                error_msg += std::to_string(failed_rank) + " ";
              }
              throw std::runtime_error(error_msg);
            }
            // Auto-repair succeeded, continue to allreduce below
            ULFM_LOG_INFO(currentRank_, "Auto-repair succeeded, continuing with allreduce");
          } else {
            if (ulfm_work) ulfm_work->markNoop();
            // Don't auto-repair, record failure and bypass allreduce
            ULFM_LOG_WARN(currentRank_, "Failures detected, bypassing allreduce (manual recovery needed)");
            // Skip allreduce - tensor data remains unchanged, which is correct
            // The work will complete normally but with failure recorded
            return;
          }
        }
        if (is_spare() && !is_at_policy_boundary()) {
          data.zero_();
          ULFM_LOG_DEBUG(currentRank_, "Spare process zeroing out grad data");
        }

        // BF16/FP16 have no native MPI reduction op — upcast to FP32 for reduce.
        const bool needs_upcast = (data.scalar_type() == at::kBFloat16 ||
                                   data.scalar_type() == at::kHalf);
        at::Tensor reduce_buf = needs_upcast ? data.data().to(at::kFloat) : data;
        
        ULFM_LOG_DEBUG(currentRank_, "Entering MPI_Allreduce");
        int rc = MPI_Allreduce(
            MPI_IN_PLACE,
            reduce_buf.data_ptr(),
            reduce_buf.numel(),
            mpiDatatype.at(reduce_buf.scalar_type()),
            mpiOp.at(opts.reduceOp),
            pgComm_);
        ULFM_LOG_DEBUG(currentRank_, "Leaving MPI_Allreduce");
        if (rc != MPI_SUCCESS) {
          int cls;
          MPI_Error_class(rc, &cls);
          if (cls != MPIX_ERR_PROC_FAILED && cls != MPIX_ERR_PROC_FAILED_PENDING &&
              cls != MPIX_ERR_REVOKED) {
            MPI_CHECK(rc);
          }
          // A process died while the reduction was in flight (the agreement
          // above only covers failures before it). The buffer is undefined
          // now, and some peers may have completed the reduction. Revoke, so
          // that no peer stays blocked in it, and complete the work as if the
          // bucket had been reduced: every rank then notices the failure at
          // its next agreement on this communicator (the next bucket's
          // notice_failure() or the step's consensus) and rolls the bucket
          // back to its snapshot, as for a failure right after the bucket.
          ULFM_LOG_WARN(currentRank_, "Process failure during MPI_Allreduce (error class "
                                      << cls << "); communicator revoked, the failure is "
                                      << "handled at the next agreement");
          MPIX_Comm_revoke(pgComm_);
          return;
        }

        if (needs_upcast) {
          data.data().copy_(reduce_buf);
        }
      };
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  return enqueueULFM(
      std::move(entry),
      "mpi:ulfm_allreduce",
      std::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<Work> ProcessGroupULFM::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  TORCH_CHECK(false, "allreduce_coalesced is currently not supported with MPI");
}

c10::intrusive_ptr<Work> ProcessGroupULFM::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        auto dataPtr = (entry->src)[0].data_ptr();

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        const int root = currentRankOf(opts.rootRank);
        TORCH_CHECK(root != MPI_UNDEFINED, "reduce: root rank ",
                    opts.rootRank, " is no longer in the communicator");
        void* sendbuf = (currentRank_ == root) ? MPI_IN_PLACE : dataPtr;
        void* recvbuf = (currentRank_ == root) ? dataPtr : nullptr;
        MPI_CHECK(MPI_Reduce(
            sendbuf,
            recvbuf,
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            mpiOp.at(opts.reduceOp),
            root,
            pgComm_));
      };
  auto entry =
      std::make_unique<WorkEntry>(&tensors, &tensors, std::move(runFunc));
  return enqueue(
      std::move(entry),
      "mpi:reduce",
      std::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<Work> ProcessGroupULFM::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  checkSingleTensor(inputTensors);
  if (outputTensors.size() != 1) {
    TORCH_CHECK(
        false,
        "MPI process group only supports a single "
        "tensor op");
  }
  if (static_cast<size_t>(size_) != outputTensors[0].size()) {
    TORCH_CHECK(
        false,
        "All gather: number of output tensors should equal "
        "to the world size");
  }

  checkSameSizeAndType(inputTensors[0], outputTensors[0]);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        std::vector<at::Tensor> outputDataVec = entry->dst;
        auto flatOutputTensor = newLikeFlat(outputDataVec);

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Allgather(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            flatOutputTensor.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            pgComm_));

        for (const auto i : c10::irange(outputDataVec.size())) {
          outputDataVec[i].copy_(flatOutputTensor[static_cast<int64_t>(i)]);
        }
      };
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors[0], std::move(runFunc));
  return enqueue(
      std::move(entry),
      "mpi:all_gather",
      std::optional<std::vector<at::Tensor>>(inputTensors));
}

c10::intrusive_ptr<Work> ProcessGroupULFM::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */) {
  TORCH_CHECK(false, "ProcessGroupULFM does not support allgather_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupULFM::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  checkSingleTensor(inputTensors);

  if (rank_ != opts.rootRank) {
    if (!outputTensors.empty()) {
      TORCH_CHECK(
          false,
          "Gather: number of output tensors should be 0 "
          "for non-root");
    }
  } else {
    if (outputTensors.size() != 1) {
      TORCH_CHECK(false, "Gather: multi-GPU collective is not supported");
    }
    if (static_cast<size_t>(size_) != outputTensors[0].size()) {
      TORCH_CHECK(
          false,
          "Gather: number of output tensors should equal "
          "to the world size");
    }
    checkSameSizeAndType(inputTensors[0], outputTensors[0]);
  }

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        void* recvbuf = nullptr;
        at::Tensor flatOutputTensor;

        std::vector<at::Tensor> dstdata = entry->dst;
        if (rank_ == opts.rootRank) {
          flatOutputTensor = newLikeFlat(dstdata);
          recvbuf = flatOutputTensor.data_ptr();
        }

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Gather(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            recvbuf,
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            opts.rootRank,
            pgComm_));

        if (rank_ == opts.rootRank) {
          const std::vector<at::Tensor>& outputDataVec = entry->dst;
          // copy the flattened output tensors to the outputs
          for (const auto i : c10::irange(outputDataVec.size())) {
            outputDataVec.at(i).copy_(
                flatOutputTensor[static_cast<int64_t>(i)]);
          }
        }
      };

  if (rank_ == opts.rootRank) {
    auto entry = std::make_unique<WorkEntry>(
        &inputTensors, &outputTensors[0], std::move(runFunc));
    return enqueue(
        std::move(entry),
        "mpi:gather",
        std::optional<std::vector<at::Tensor>>(inputTensors));
  } else {
    auto entry =
        std::make_unique<WorkEntry>(&inputTensors, nullptr, std::move(runFunc));
    return enqueue(
        std::move(entry),
        "mpi:gather",
        std::optional<std::vector<at::Tensor>>(inputTensors));
  }
}

c10::intrusive_ptr<Work> ProcessGroupULFM::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  checkSingleTensor(outputTensors);

  if (rank_ != opts.rootRank) {
    if (!inputTensors.empty()) {
      TORCH_CHECK(
          false,
          "Scatter: number of input tensors should be 0 "
          "for non-root");
    }
  } else {
    if (inputTensors.size() != 1) {
      TORCH_CHECK(false, "Scatter: multi-GPU collective is not supported");
    }
    if (static_cast<size_t>(size_) != inputTensors[0].size()) {
      TORCH_CHECK(
          false,
          "Scatter: number of input tensors should equal "
          "to the world size");
    }
    checkSameSizeAndType(outputTensors[0], inputTensors[0]);
  }

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->dst)[0];
        void* sendbuf = nullptr;
        at::Tensor flatInputTensor;

        if (rank_ == opts.rootRank) {
          std::vector<at::Tensor>& inputDataVec = entry->src;
          flatInputTensor = newLikeFlat(inputDataVec);
          sendbuf = flatInputTensor.data_ptr();

          // copy the input tensors to the flatten large send buffer
          for (const auto i : c10::irange(inputDataVec.size())) {
            flatInputTensor[static_cast<int64_t>(i)].copy_(inputDataVec.at(i));
          }
        }

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Scatter(
            sendbuf,
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            opts.rootRank,
            pgComm_));
      };

  if (rank_ == opts.rootRank) {
    auto entry = std::make_unique<WorkEntry>(
        &inputTensors[0], &outputTensors, std::move(runFunc));
    return enqueue(
        std::move(entry),
        "mpi:scatter",
        !inputTensors.empty()
            ? std::optional<std::vector<at::Tensor>>(inputTensors[0])
            : std::nullopt);
  } else {
    auto entry = std::make_unique<WorkEntry>(
        nullptr, &outputTensors, std::move(runFunc));
    return enqueue(
        std::move(entry),
        "mpi:scatter",
        !inputTensors.empty()
            ? std::optional<std::vector<at::Tensor>>(inputTensors[0])
            : std::nullopt);
  }
}

c10::intrusive_ptr<Work> ProcessGroupULFM::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  checkSingleTensor(outputTensors);
  if (inputTensors.size() != 1) {
    TORCH_CHECK(
        false,
        "MPI process group only supports a single "
        "tensor op");
  }
  if (static_cast<size_t>(size_) != inputTensors[0].size()) {
    TORCH_CHECK(
        false,
        "Reduce scatter: number of input tensors should equal "
        "to the world size");
  }
  checkSameSizeAndType(outputTensors[0], inputTensors[0]);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->dst)[0];
        auto flatInputTensor = newLikeFlat(entry->src);
        for (const auto i : c10::irange(entry->src.size())) {
          flatInputTensor[static_cast<int64_t>(i)].copy_(entry->src[i]);
        }
        int recvcount = flatInputTensor.numel() / size_;

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Reduce_scatter_block(
            flatInputTensor.data_ptr(),
            data.data_ptr(),
            recvcount,
            mpiDatatype.at(data.scalar_type()),
            mpiOp.at(opts.reduceOp),
            pgComm_));
      };

  auto entry = std::make_unique<WorkEntry>(
      &inputTensors[0], &outputTensors, std::move(runFunc));
  return enqueue(
      std::move(entry),
      "mpi:reduce_scatter",
      std::optional<std::vector<at::Tensor>>(inputTensors[0]));
}

c10::intrusive_ptr<Work> ProcessGroupULFM::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts) {
  checkSingleTensorHelper(inputTensor);
  checkSingleTensorHelper(outputTensor);

  if (outputSplitSizes.empty() && inputSplitSizes.empty()) {
    // We can use alltoall
    TORCH_CHECK(
        outputTensor.numel() == inputTensor.numel() &&
            outputTensor.type() == inputTensor.type(),
        "Tensors are not equal in size or data type");
    TORCH_CHECK(
        outputTensor.size(0) % size_ == 0,
        "Tensor's dim 0 does not divide equally across group size");

    std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
        [this](std::unique_ptr<WorkEntry>& entry) {
          auto srcdata = (entry->src)[0];
          auto dstdata = (entry->dst)[0];
          c10::DeviceGuard guard(srcdata.device());
          std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
          MPI_CHECK(MPI_Alltoall(
              srcdata.data_ptr(),
              srcdata.numel() / size_,
              mpiDatatype.at(srcdata.scalar_type()),
              dstdata.data_ptr(),
              dstdata.numel() / size_,
              mpiDatatype.at(dstdata.scalar_type()),
              pgComm_));
        };
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    auto entry = std::make_unique<WorkEntry>(
        &inputTensors, &outputTensors, std::move(runFunc));
    return enqueue(
        std::move(entry),
        "mpi:all_to_all",
        std::optional<std::vector<at::Tensor>>(inputTensors));
  } else {
    // Need alltoallv
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);
    std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
        [this, inputSplitSizes, outputSplitSizes](
            std::unique_ptr<WorkEntry>& entry) {
          auto srcdata = (entry->src)[0];
          auto dstdata = (entry->dst)[0];
          std::vector<int> send_lengths(size_);
          std::vector<int> recv_lengths(size_);
          std::vector<int> send_offsets(size_);
          std::vector<int> recv_offsets(size_);
          c10d::computeLengthsAndOffsets(
              inputSplitSizes, srcdata, &send_lengths, &send_offsets);
          c10d::computeLengthsAndOffsets(
              outputSplitSizes, dstdata, &recv_lengths, &recv_offsets);
          c10::DeviceGuard guard(srcdata.device());
          std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
          MPI_CHECK(MPI_Alltoallv(
              srcdata.data_ptr(),
              send_lengths.data(),
              send_offsets.data(),
              mpiDatatype.at(srcdata.scalar_type()),
              dstdata.data_ptr(),
              recv_lengths.data(),
              recv_offsets.data(),
              mpiDatatype.at(dstdata.scalar_type()),
              pgComm_));
        };
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    auto entry = std::make_unique<WorkEntry>(
        &inputTensors, &outputTensors, std::move(runFunc));
    return enqueue(
        std::move(entry),
        "mpi:all_to_all",
        std::optional<std::vector<at::Tensor>>(inputTensors));
  }
}

c10::intrusive_ptr<Work> ProcessGroupULFM::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts) {
  TORCH_CHECK(
      inputTensors.size() == static_cast<size_t>(size_),
      "Number of input tensors are not equal to group size");
  TORCH_CHECK(
      outputTensors.size() == static_cast<size_t>(size_),
      "Number of output tensors are not equal to group size");
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [this](std::unique_ptr<WorkEntry>& entry) {
        std::vector<int> send_lengths(size_);
        std::vector<int> recv_lengths(size_);
        std::vector<int> send_offsets(size_);
        std::vector<int> recv_offsets(size_);
        auto srcdata = entry->src;
        auto dstdata = entry->dst;
        auto src_len = c10d::computeLengthsAndOffsets(
            srcdata, &send_lengths, &send_offsets);
        auto dst_len = c10d::computeLengthsAndOffsets(
            dstdata, &recv_lengths, &recv_offsets);
        std::vector<int64_t> send_lengthsL(
            send_lengths.begin(), send_lengths.end());
        std::vector<int64_t> recv_lengthsL(
            recv_lengths.begin(), recv_lengths.end());
        at::Tensor srcFlatData =
            at::empty({static_cast<int64_t>(src_len)}, srcdata[0].options());
        at::Tensor dstFlatData =
            at::empty({static_cast<int64_t>(dst_len)}, dstdata[0].options());
        auto srcFlatDataSplits =
            srcFlatData.split_with_sizes(c10::IntArrayRef(send_lengthsL), 0);
        for (const auto i : c10::irange(size_)) {
          srcFlatDataSplits[i].copy_(srcdata[i].view({-1}));
        }
        c10::DeviceGuard guard1(srcdata[0].device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Alltoallv(
            srcFlatData.data_ptr(),
            send_lengths.data(),
            send_offsets.data(),
            mpiDatatype.at(srcdata[0].scalar_type()),
            dstFlatData.data_ptr(),
            recv_lengths.data(),
            recv_offsets.data(),
            mpiDatatype.at(dstdata[0].scalar_type()),
            pgComm_));

        auto dstFlatDataSplits =
            dstFlatData.split_with_sizes(c10::IntArrayRef(recv_lengthsL), 0);
        for (const auto i : c10::irange(size_)) {
          dstdata[i].view({-1}).copy_(dstFlatDataSplits[i]);
        }
      };
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors, std::move(runFunc));
  return enqueue(
      std::move(entry),
      "mpi:all_to_all",
      std::optional<std::vector<at::Tensor>>(inputTensors));
}

c10::intrusive_ptr<Work> ProcessGroupULFM::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  checkSingleTensor(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    c10::DeviceGuard guard(tensor.device());
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Isend(
        tensor.data_ptr(),
        tensor.numel(),
        mpiDatatype.at(tensor.scalar_type()),
        dstRank,
        tag,
        pgComm_,
        &request));
  }

  return c10::make_intrusive<AsyncWork>(
      request,
      std::vector<at::Tensor>(),
      "mpi:send",
      std::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<Work> ProcessGroupULFM::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  checkSingleTensor(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    c10::DeviceGuard guard(tensor.device());
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Irecv(
        tensor.data_ptr(),
        tensor.numel(),
        mpiDatatype.at(tensor.scalar_type()),
        srcRank,
        tag,
        pgComm_,
        &request));
  }

  return c10::make_intrusive<AsyncWork>(
      request,
      tensors,
      "mpi:recv",
      std::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<Work> ProcessGroupULFM::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  checkSingleTensor(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    c10::DeviceGuard guard(tensor.device());
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Irecv(
        tensor.data_ptr(),
        tensor.numel(),
        mpiDatatype.at(tensor.scalar_type()),
        MPI_ANY_SOURCE,
        tag,
        pgComm_,
        &request));
  }

  return c10::make_intrusive<AsyncWork>(
      request,
      tensors,
      "mpi:recvAnySource",
      std::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<Work> ProcessGroupULFM::barrier(const BarrierOptions& opts) {
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [this](std::unique_ptr<WorkEntry>& entry) {
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Barrier(pgComm_));
      };
  auto entry =
      std::make_unique<WorkEntry>(nullptr, nullptr, std::move(runFunc));
  return enqueue(std::move(entry), "mpi:barrier", std::nullopt);
}

c10::intrusive_ptr<Work> ProcessGroupULFM::consensus(const ULFMOptions& ulfm_opts) {
  const int epoch_at_enqueue = worldEpoch();

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [ulfm_opts, this, epoch_at_enqueue](std::unique_ptr<WorkEntry>& entry) {
        ULFM_LOG_DEBUG(currentRank_, "consensus: waiting for pgGlobalMutex_");
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        ULFM_LOG_DEBUG(currentRank_, "consensus: pgGlobalMutex_ acquired");

        // Get the WorkULFM instance to record failures
        WorkULFM* ulfm_work = static_cast<WorkULFM*>(entry->ulfmWork);
        // Use the new modular failure recovery system
        std::vector<int> failed_ranks, failed_world_ranks;
        RecoveryResult recovery_success = detect_and_recover_failures(
          ulfm_opts.auto_repair, &failed_ranks, ulfm_opts.max_retries, &failed_world_ranks
        );
        // Always record failure for inspection if any were detected
        if (recovery_success.has_failure()) {
          record_and_handling_failure(failed_ranks, ulfm_opts, ulfm_work);
          if (ulfm_work) ulfm_work->recordFailedWorldRanks(failed_world_ranks);

          if (ulfm_opts.auto_repair) {
            ULFM_LOG_INFO(currentRank_, "Consensus: Auto-repair succeeded during consensus, epoch change from "
                                      << epoch_at_enqueue << " to " << worldEpoch());
          } else {
            ULFM_LOG_WARN(currentRank_, "Consensus: Failures detected during consensus at epoch "
                                      << epoch_at_enqueue << ", manual recovery needed");
          }
        }
      };
  auto entry =
      std::make_unique<WorkEntry>(nullptr, nullptr, std::move(runFunc));
  return enqueueULFM(std::move(entry), "mpi:consensus", std::nullopt);
}

c10::intrusive_ptr<Work> ProcessGroupULFM::_allgather_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const AllgatherOptions& opts) {
  TORCH_CHECK(
      outputTensor.numel() == inputTensor.numel() * size_,
      "All gather: output tensor size must be equal to input tensor size times the world size");

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [this](std::unique_ptr<WorkEntry>& entry) {
        auto dstdata = (entry->dst)[0];
        auto srcdata = (entry->src)[0];
        c10::DeviceGuard guard(srcdata.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Allgather(
            srcdata.data_ptr(),
            srcdata.numel(),
            mpiDatatype.at(srcdata.scalar_type()),
            dstdata.data_ptr(),
            srcdata.numel(),
            mpiDatatype.at(dstdata.scalar_type()),
            pgComm_));
      };

  auto inputTensors = std::vector<at::Tensor>({inputTensor});
  auto outputTensors = std::vector<at::Tensor>({outputTensor});
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors, std::move(runFunc));
  return enqueue(
      std::move(entry),
      "mpi:_allgather_base",
      std::optional<std::vector<at::Tensor>>(inputTensors));
}

c10::intrusive_ptr<Work> ProcessGroupULFM::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK(
      outputTensor.numel() * size_ == inputTensor.numel(),
      "Reduce scatter: input tensor size must be equal to output tensor size times the world size");

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto dstdata = (entry->dst)[0];
        auto srcdata = (entry->src)[0];
        c10::DeviceGuard guard(srcdata.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Reduce_scatter_block(
            srcdata.data_ptr(),
            dstdata.data_ptr(),
            dstdata.numel(),
            mpiDatatype.at(srcdata.scalar_type()),
            mpiOp.at(opts.reduceOp),
            pgComm_));
      };

  auto inputTensors = std::vector<at::Tensor>({inputTensor});
  auto outputTensors = std::vector<at::Tensor>({outputTensor});
  auto entry = std::make_unique<WorkEntry>(
      &inputTensors, &outputTensors, std::move(runFunc));
  return enqueue(
      std::move(entry),
      "mpi:_reduce_scatter_base",
      std::optional<std::vector<at::Tensor>>(inputTensors));
}

// Comprehensive failure detection and recovery workflow (corrected ULFM protocol order)
// Uses ULFM spec-compliant consistent failure detection pattern.
RecoveryResult ProcessGroupULFM::detect_and_recover_failures(bool auto_repair, std::vector<int>* failed_ranks, int max_retries, std::vector<int>* failed_world_ranks) {

  // Step 1: Notice failure (comm_agree first - detects failures, revokes communicator)
  if (!notice_failure()) {
    // No failures detected
    return RecoveryResult::NoFailure();
  }

  // Step 2: Get failed ranks using spec-compliant ack-agree loop
  // This internally performs:
  //   do {
  //       MPIX_Comm_ack_failed(c, size, &num_acked);
  //       rc = MPIX_Comm_agree(c, &T);
  //   } while (rc != MPI_SUCCESS);
  // ensuring consistent failure detection across all survivors
  std::vector<int> failed_ranks_comm, failed_ranks_world;
  RecoveryResult get_result = get_failed_ranks_internal(failed_ranks_comm, &failed_ranks_world, max_retries);
  if (!get_result) {
    ULFM_LOG_ERROR(currentRank_, get_result.error_message);
    return get_result;
  }

  // Return failed ranks to caller if requested
  if (failed_ranks) {
    *failed_ranks = failed_ranks_comm;
  }
  if (failed_world_ranks) {
    *failed_world_ranks = failed_ranks_world;
  }

  int num_failures = static_cast<int>(failed_ranks_comm.size());

  RecoveryResult repair_result = repair_communicator_internal();
  if (!repair_result) {
    ULFM_LOG_ERROR(currentRank_, repair_result.error_message);
    return repair_result;
  }
  ULFM_LOG_INFO(currentRank_, "Failure detection and recovery completed successfully");
  return RecoveryResult::Recovered(num_failures);


  // Step 3: Repair communicator if needed and requested
  // if (auto_repair) {
  //   RecoveryResult repair_result = repair_communicator_internal();
  //   if (!repair_result) {
  //     ULFM_LOG_ERROR(currentRank_, repair_result.error_message);
  //     return repair_result;
  //   }
  //   ULFM_LOG_INFO(currentRank_, "Failure detection and recovery completed successfully");
  //   return RecoveryResult::Recovered(num_failures);
  // } else {
  //   ULFM_LOG_WARN(currentRank_, "Repair needed but auto_repair disabled");
  //   // Failure detected but not repaired (caller's choice)
  //   return RecoveryResult::NotRecovered(num_failures);
  // }
}

// Step 1: Notice failure (comm_agree first - best practice)
bool ProcessGroupULFM::notice_failure() {
  int flag = 1;
  int rc = MPIX_Comm_agree(pgComm_, &flag);
  int cls;
  MPI_Error_class(rc, &cls);
  
  // Check if failure was noticed
  if (cls == MPIX_ERR_PROC_FAILED || cls == MPIX_ERR_REVOKED) {
    ULFM_LOG_WARN(currentRank_, "Failure noticed via MPIX_Comm_agree");
    MPIX_Comm_revoke(pgComm_); // Ensure communicator is revoked
    // set_quiesce(true); // latch for the rest of the (failed) step
    return true; // Failure detected
  }
  
  return false; // No failure
}

// Step 2: Get failed ranks with ULFM spec-compliant consistent failure detection
// Uses the ack-agree loop pattern from the ULFM spec:
//   do {
//       MPIX_Comm_ack_failed(c, size, &num_acked);
//       rc = MPIX_Comm_agree(c, &T);
//   } while (rc != MPI_SUCCESS);
// This ensures all survivors agree on the same set of failed processes.
// The exit test itself is agreed on as well (see the loop), so that every
// survivor runs the same number of agreements.
RecoveryResult ProcessGroupULFM::get_failed_ranks_internal(std::vector<int>& failed_ranks_comm, std::vector<int>* failed_ranks_world, int max_retries) {
  failed_ranks_comm.clear();
  if (failed_ranks_world) failed_ranks_world->clear();

  // === ULFM spec-compliant consistent failure detection ===
  int num_acked = 0;
  int rc = MPI_SUCCESS;
  int loop_count = 0;
  int done = 0;

  do {
    loop_count++;
    // Acknowledge all known failures
    int rc_ack = MPIX_Comm_ack_failed(pgComm_, currentSize_, &num_acked);
    if (rc_ack != MPI_SUCCESS) {
      return RecoveryResult::Error("MPIX_Comm_ack_failed failed", rc_ack);
    }

    // Try to reach agreement; flag (bitwise AND) tells whether every
    // survivor has acknowledged a failure.
    int flag = num_acked > 0 ? 1 : 0;
    rc = MPIX_Comm_agree(pgComm_, &flag);
    // Loop continues if new failure discovered during agree

    // Leave the loop on an agreed value only. num_acked is local, and the
    // agree can succeed while a rank has acknowledged nothing yet: Open
    // MPI's ERA agreement compares the acknowledged sets, but misses an
    // empty one that is combined first. Deciding on num_acked let that rank
    // run one more MPIX_Comm_agree than the others, which met their
    // MPIX_Comm_shrink (assertion era_combine_agreement_values in debug
    // builds). A second agreement makes the decision the same on every
    // survivor.
    done = (rc == MPI_SUCCESS && flag) ? 1 : 0;
    MPIX_Comm_agree(pgComm_, &done);
    {
      int cls = MPI_SUCCESS;
      if (rc != MPI_SUCCESS) MPI_Error_class(rc, &cls);
      ULFM_LOG_DEBUG(currentRank_, "ack-agree iteration " << loop_count
                     << ": num_acked=" << num_acked << " rc_class=" << cls
                     << " flag=" << flag << " done=" << done);
    }

    // Check max retries to prevent infinite loop
    if (loop_count >= max_retries && !done) {
      std::ostringstream oss;
      oss << "Ack-agree loop exceeded max_retries=" << max_retries << " (last rc=" << rc << ", num_acked=" << num_acked << ")";
      return RecoveryResult::Error(oss.str(), rc);
    }
  } while (!done);

  ULFM_LOG_DEBUG(currentRank_, "Consistent failure agreement reached after "
                 << loop_count << " iterations, num_acked=" << num_acked);
  
  // Handle no failures case
  if (num_acked == 0) {
    ULFM_LOG_WARN(currentRank_, "No failures acknowledged despite earlier detection via earlier MPIX_Comm_agree.");
  }

  // Get failed group (now consistent across all survivors)
  MPI_Group failed_grp = MPI_GROUP_NULL;
  int rc_get = MPIX_Comm_get_failed(pgComm_, &failed_grp);
  if (rc_get != MPI_SUCCESS || failed_grp == MPI_GROUP_NULL) {
    std::ostringstream oss;
    oss << "Failed to get failed group, rc=" << rc_get;
    return RecoveryResult::Error(oss.str(), rc_get);
  }

  int fsize = 0;
  MPI_Group_size(failed_grp, &fsize);
  if (fsize <= 0) {
    MPI_Group_free(&failed_grp);
    std::ostringstream oss;
    oss << "Empty failed group (size=" << fsize << ")";
    return RecoveryResult::Error(oss.str());
  }

  // Use num_acked to get the agreed-upon failures
  MPI_Group consistent_failed_grp = MPI_GROUP_NULL;
  if (fsize > num_acked) {
    // More failures detected than acknowledged - use only acknowledged ones
    int ranges[1][3] = {{0, num_acked - 1, 1}};
    MPI_Group_range_incl(failed_grp, 1, ranges, &consistent_failed_grp);
    ULFM_LOG_DEBUG(currentRank_, "Using " << num_acked << " acknowledged failures out of " << fsize << " detected");
    MPI_Group_free(&failed_grp);
    failed_grp = consistent_failed_grp;
    fsize = num_acked;
  }

  // Build index array 0..fsize-1 in the failed group's own indexing
  std::vector<int> idx(fsize);
  for (int i = 0; i < fsize; ++i) idx[i] = i;

  // Translate to ranks in 'comm'
  MPI_Group comm_grp = MPI_GROUP_NULL;
  MPI_Comm_group(pgComm_, &comm_grp);
  failed_ranks_comm.resize(fsize);
  MPI_Group_translate_ranks(failed_grp, fsize, idx.data(),
                            comm_grp, failed_ranks_comm.data());
  MPI_Group_free(&comm_grp);

  if (is_ulfm_verbose_logging() && fsize > 0) {
    std::ostringstream oss;
    oss << "Failed ranks in comm: ";
    for (int i = 0; i < fsize; ++i) {
      if (i > 0) oss << ", ";
      oss << failed_ranks_comm[i];
    }
    std::string debug_msg = oss.str();
    if (!debug_msg.empty() && debug_msg != "Failed ranks in comm: ") {
      ULFM_LOG_DEBUG(currentRank_, debug_msg);
    }
  }

  // Optionally translate to MPI_COMM_WORLD ranks
  if (failed_ranks_world) {
    MPI_Group world_grp = MPI_GROUP_NULL;
    MPI_Comm_group(MPI_COMM_WORLD, &world_grp);
    failed_ranks_world->resize(fsize);
    MPI_Group_translate_ranks(failed_grp, fsize, idx.data(),
                              world_grp, failed_ranks_world->data());
    MPI_Group_free(&world_grp);
  }

  MPI_Group_free(&failed_grp);
  return RecoveryResult::NoFailure();
}

// Step 3: Check if repair is needed and advisable
bool ProcessGroupULFM::should_repair_communicator(const std::vector<int>& failed_ranks) {
  if (failed_ranks.empty()) {
    return false; // No failures, no repair needed
  }
  
  // Check if we have enough surviving ranks to continue
  int surviving_ranks = currentSize_ - static_cast<int>(failed_ranks.size());
  if (surviving_ranks <= 0) {
    ULFM_LOG_WARN(currentRank_, "No surviving ranks, cannot repair\n");
    return false;
  }
  
  // Additional policy checks could go here
  // For now, repair if we have failures and survivors
  ULFM_LOG_INFO(currentRank_, "Repair recommended: " << failed_ranks.size() << " failures, " << surviving_ranks << " survivors");
  return true;
}

// Step 4: Repair communicator (shrink operation)
RecoveryResult ProcessGroupULFM::repair_communicator_internal() {
  // Shrink the communicator to remove failed processes
  MPI_Comm new_comm;
  int rc_shrink = MPIX_Comm_shrink(pgComm_, &new_comm);
  if (rc_shrink != MPI_SUCCESS) {
    return RecoveryResult::Error("Failed to shrink communicator", rc_shrink);
  }

  // Replace old communicator
  MPI_Comm_free(&pgComm_);
  pgComm_ = new_comm;

  // Update current rank and size
  MPI_Comm_rank(pgComm_, &currentRank_);
  MPI_Comm_size(pgComm_, &currentSize_);

  int old_epoch_ = worldEpoch();             // Save old epoch for logging
  bumpEpoch();                               // <<— bump after successful repair
  int curr_epoch_ = worldEpoch();            // Current epoch after bump
  ULFM_LOG_INFO(currentRank_, "Epoch bumped: old_epoch=" << old_epoch_ << ", new_epoch=" << curr_epoch_);

  ULFM_LOG_INFO(currentRank_, "Communicator repaired: original_rank=" << rank_ << ", new_rank=" << currentRank_ << ", new_size=" << currentSize_);

  return RecoveryResult::Recovered();
}

// ProcessGroup-level recovery methods
RecoveryResult ProcessGroupULFM::repair_communicator() {
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  try {
    constexpr int MAX_REPAIR_ATTEMPTS = 3;
    MPI_Comm new_comm = MPI_COMM_NULL;
    int rc;
    int flag;

    for (int attempt = 0; attempt < MAX_REPAIR_ATTEMPTS; ++attempt) {
      // Step 1: Shrink the communicator to exclude failed processes
      rc = MPIX_Comm_shrink(pgComm_, &new_comm);
      if (rc != MPI_SUCCESS) {
        ULFM_LOG_WARN(currentRank_, "Shrink failed on attempt " << (attempt + 1) << "/" << MAX_REPAIR_ATTEMPTS << ", rc=" << rc);
        if (new_comm != MPI_COMM_NULL) {
          MPI_Comm_free(&new_comm);
          new_comm = MPI_COMM_NULL;
        }
        continue;
      }

      // Step 2: Agree on the new communicator to ensure consistency
      flag = 1;
      rc = MPIX_Comm_agree(new_comm, &flag);
      if (rc != MPI_SUCCESS || flag == 0) {
        ULFM_LOG_WARN(currentRank_, "Agree failed on attempt " << (attempt + 1) << "/" << MAX_REPAIR_ATTEMPTS << ", rc=" << rc << ", flag=" << flag);
        MPI_Comm_free(&new_comm);
        new_comm = MPI_COMM_NULL;
        continue;
      }

      // Both shrink and agree succeeded
      MPI_Comm_free(&pgComm_);
      pgComm_ = new_comm;

      // Update current rank and size after repair (keep original rank_ unchanged)
      MPI_Comm_rank(pgComm_, &currentRank_);
      MPI_Comm_size(pgComm_, &currentSize_);

      ULFM_LOG_INFO(currentRank_, "Communicator repaired, new rank=" << currentRank_ << ", size=" << currentSize_);
      return RecoveryResult::Recovered();
    }

    // All attempts failed
    return RecoveryResult::Error("Failed to repair communicator after " + std::to_string(MAX_REPAIR_ATTEMPTS) + " attempts", rc);
  } catch (const std::exception& e) {
    std::string error_msg = std::string("Exception during communicator repair: ") + e.what();
    return RecoveryResult::Error(error_msg);
  } catch (...) {
    return RecoveryResult::Error("Unknown exception during communicator repair");
  }
}

void ProcessGroupULFM::notify_all_ranks_of_failure() {
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  
  // Use MPI_Barrier to synchronize all surviving ranks
  // This will fail if there are failures, alerting all ranks
  int flag = 0;  // Signal failure to all ranks
  MPIX_Comm_agree(pgComm_, &flag);
}

bool ProcessGroupULFM::check_for_failures() const {
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  int flag = 1;
  int rc = MPIX_Comm_agree(pgComm_, &flag);
  int cls;
  MPI_Error_class(rc, &cls);

  return (cls == MPIX_ERR_PROC_FAILED || cls == MPIX_ERR_REVOKED);
}

void ProcessGroupULFM::count_rank_types(RankTypeCounts& counts) {
  // Use array for MPI_Allreduce, then copy to struct
  // Index: 0=major_workers, 1=minor_workers, 2=major_spares, 3=minor_spares, 4=boundary_minors
  int data[5] = {0, 0, 0, 0, 0};
  data[(is_minor() ? 1 : 0) + (is_spare() ? 2 : 0)] = 1;
  data[4] = is_boundary_minor() ? 1 : 0;

  MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, data, 5, MPI_INT, MPI_SUM, pgComm_));

  counts.majors = data[0];
  counts.minors = data[1];
  counts.major_spares = data[2];
  counts.minor_spares = data[3];
  counts.boundary_minors = data[4];

  // Reduce local contribution counts globally (SUM) without modifying the
  // per-rank atomics. Coalesced into a single Allreduce for both phases.
  int64_t contrib_buf[2] = {
      contributed_.load(std::memory_order_acquire),
      boundary_contributed_.load(std::memory_order_acquire),
  };
  MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, contrib_buf, 2, MPI_INT64_T, MPI_SUM, pgComm_));
  counts.contributed = contrib_buf[0];
  counts.boundary_contributed = contrib_buf[1];

  ULFM_LOG_DEBUG(currentRank_, "count_rank_types: majors=" << counts.majors
                 << " minors=" << counts.minors
                 << " major_spares=" << counts.major_spares
                 << " minor_spares=" << counts.minor_spares
                 << " boundary_minors=" << counts.boundary_minors
                 << " contributed(global)=" << counts.contributed
                 << " boundary_contributed(global)=" << counts.boundary_contributed);
}

void ProcessGroupULFM::compute_failed_counts(
    const RankTypeCounts& old_counts,
    const RankTypeCounts& new_counts,
    FailureStats& failure_stats) {
  failure_stats.failed_majors = old_counts.majors - new_counts.majors;
  failure_stats.failed_minors = old_counts.minors - new_counts.minors;
  failure_stats.failed_major_spares = old_counts.major_spares - new_counts.major_spares;
  failure_stats.failed_minor_spares = old_counts.minor_spares - new_counts.minor_spares;
  failure_stats.failed_boundary_minors = old_counts.boundary_minors - new_counts.boundary_minors;

  ULFM_LOG_DEBUG(currentRank_, "compute_failed_counts: failed_majors=" << failure_stats.failed_majors
                 << " failed_minors=" << failure_stats.failed_minors
                 << " failed_major_spares=" << failure_stats.failed_major_spares
                 << " failed_minor_spares=" << failure_stats.failed_minor_spares
                 << " failed_boundary_minors=" << failure_stats.failed_boundary_minors);
}

bool ProcessGroupULFM::check_at_policy_boundary(
    const FailureStats& failure_stats,
    const RankTypeCounts& current_counts) {
  // At policy boundary if surviving spares cannot cover every failed slot.
  // current_counts is taken among survivors pre-promotion, so major_spares /
  // minor_spares is the number of spares available to fill gaps. Boundary
  // when failed_majors exceeds available major spares (or the same for minors).
  return (failure_stats.failed_majors > current_counts.major_spares) ||
         (failure_stats.failed_minors > current_counts.minor_spares);
}

void ProcessGroupULFM::update_policy_boundary(bool event_boundary) {
  // PG-level boundary flag is sticky: once true, stays true until explicitly reset
  // If PG-level flag is already on, per-event flag should not override it
  // If PG-level flag is off, it should be set by per-event flag
  if (event_boundary && !atPolicyBoundary_.load(std::memory_order_acquire)) {
    atPolicyBoundary_.store(true, std::memory_order_release);
    ULFM_LOG_DEBUG(currentRank_, "Policy boundary reached - no spares available for failed workers");
  }
}

bool ProcessGroupULFM::elect_promotion(int failed_majors, int failed_minors) {
  bool promoted = false;

  // Major spare election: ALL ranks must participate in collective
  // Only major spares (spares that are NOT minor) contribute 1
  {
    int my_contrib = (is_spare() && !is_minor()) ? 1 : 0;
    int my_index = 0;
    ULFM_LOG_DEBUG(currentRank_, "Participating in major spare election (contrib=" << my_contrib << ")");
    MPI_CHECK(MPI_Exscan(&my_contrib, &my_index, 1, MPI_INT, MPI_SUM, pgComm_));
    ULFM_LOG_DEBUG(currentRank_, "Major spare election: my_index=" << my_index);
    // Rank 0 in the comm gets 0 from Exscan (undefined behavior, MPI sets it to 0)
    if (currentRank_ == 0) {
      my_index = 0;
    }
    // Only major spares check if they should be promoted
    if (is_spare() && !is_minor() && failed_majors > 0 && my_index < failed_majors) {
      reset_spare();
      promoted = true;
      ULFM_LOG_INFO(currentRank_, "Promoted from major spare to major worker (index=" << my_index << ")");
    }
  }

  // Minor spare election: ALL ranks must participate in collective
  // Only minor spares (spares that ARE minor) contribute 1
  {
    int my_contrib = (is_spare() && is_minor()) ? 1 : 0;
    int my_index = 0;
    ULFM_LOG_DEBUG(currentRank_, "Participating in minor spare election (contrib=" << my_contrib << ")");
    MPI_CHECK(MPI_Exscan(&my_contrib, &my_index, 1, MPI_INT, MPI_SUM, pgComm_));
    ULFM_LOG_DEBUG(currentRank_, "Minor spare election: my_index=" << my_index);
    // Rank 0 in the comm gets 0 from Exscan (undefined behavior, MPI sets it to 0)
    if (currentRank_ == 0) {
      my_index = 0;
    }
    // Only minor spares check if they should be promoted
    if (is_spare() && is_minor() && failed_minors > 0 && my_index < failed_minors) {
      reset_spare();
      promoted = true;
      ULFM_LOG_INFO(currentRank_, "Promoted from minor spare to minor worker (index=" << my_index << ")");
    }
  }

  return promoted;
}

void ProcessGroupULFM::record_and_handling_failure(
    const std::vector<int>& failed_ranks,
    const ULFMOptions& ulfm_opts,
    WorkULFM* ulfm_work) {
  if (!ulfm_work) return;

  FailureStats failure_stats;
  RankTypeCounts current_counts;

  if (ulfm_opts.track_rank_types) {
    // Snapshot old counts
    RankTypeCounts old_counts(
      num_major_procs_.load(), num_minor_procs_.load(),
      num_major_spare_procs_.load(), num_minor_spare_procs_.load(),
      num_boundary_minor_procs_.load()
    );

    // Get new counts among survivors
    count_rank_types(current_counts);

    // Compute failed counts
    compute_failed_counts(old_counts, current_counts, failure_stats);

    // Check if this event is at policy boundary and update PG-level flag
    bool event_boundary = check_at_policy_boundary(failure_stats, current_counts);
    update_policy_boundary(event_boundary);

    // Record PG-level boundary flag (not per-event) in failure_stats
    failure_stats.at_policy_boundary = is_at_policy_boundary();

    // Auto-elect if enabled and PG is NOT at policy boundary
    if (ulfm_opts.auto_elect && !failure_stats.at_policy_boundary) {
      ULFM_LOG_INFO(currentRank_, "Auto-electing promotions for failed ranks");
      elect_promotion(failure_stats.failed_majors, failure_stats.failed_minors);
      // Recount after promotion
      count_rank_types(current_counts);
    }
    // Update stored counts
    update_rank_type_counts(current_counts);
  }

  ulfm_work->recordFailure(failed_ranks, failure_stats, current_counts);
}

void ProcessGroupULFM::update_rank_type_counts(const RankTypeCounts& counts) {
  num_major_procs_.store(counts.majors, std::memory_order_release);
  num_minor_procs_.store(counts.minors, std::memory_order_release);
  num_major_spare_procs_.store(counts.major_spares, std::memory_order_release);
  num_minor_spare_procs_.store(counts.minor_spares, std::memory_order_release);
  num_boundary_minor_procs_.store(counts.boundary_minors, std::memory_order_release);

  ULFM_LOG_DEBUG(currentRank_, "update_rank_type_counts: majors=" << counts.majors
                 << " minors=" << counts.minors
                 << " major_spares=" << counts.major_spares
                 << " minor_spares=" << counts.minor_spares
                 << " boundary_minors=" << counts.boundary_minors);
}

RankTypeCounts ProcessGroupULFM::count_and_update_rank_types() {
  RankTypeCounts counts;
  count_rank_types(counts);
  update_rank_type_counts(counts);
  return counts;
}

void ProcessGroupULFM::set_minor() {
  is_minor_.store(true, std::memory_order_release);
  ULFM_LOG_WARN(currentRank_, "Set as a minor rank");
}

void ProcessGroupULFM::reset_minor() {
  bool was_minor = is_minor_.exchange(false, std::memory_order_acq_rel);
  if (was_minor) {
    ULFM_LOG_WARN(currentRank_, "Reset from minor to a major rank");
  }
}

void ProcessGroupULFM::set_spare() {
  is_spare_.store(true, std::memory_order_release);
  ULFM_LOG_WARN(currentRank_, "Set as a spare rank");
}

void ProcessGroupULFM::reset_spare() {
  bool was_spare = is_spare_.exchange(false, std::memory_order_acq_rel);
  if (was_spare) {
    ULFM_LOG_WARN(currentRank_, "Reset from spare to a worker rank");
  }
}

void ProcessGroupULFM::set_major_minor_split(int boundary) {
  // Ranks < boundary are major (is_minor = false)
  // Ranks >= boundary are minor (is_minor = true)
  if (currentRank_ >= boundary) {
    set_minor();
  } else {
    reset_minor();
  }
}

void ProcessGroupULFM::set_boundary_minor() {
  is_boundary_minor_.store(true, std::memory_order_release);
  ULFM_LOG_WARN(currentRank_, "Set as a boundary minor rank");
}

void ProcessGroupULFM::reset_boundary_minor() {
  bool was_boundary_minor = is_boundary_minor_.exchange(false, std::memory_order_acq_rel);
  if (was_boundary_minor) {
    ULFM_LOG_WARN(currentRank_, "Reset from boundary minor rank");
  }
}

void ProcessGroupULFM::set_boundary_minor_split(int num_boundary_majors,
                                                int64_t boundary_major_workload,
                                                int64_t boundary_minor_workload) {
  TORCH_CHECK(boundary_major_workload >= 0,
              "boundary_major_workload must be non-negative");
  TORCH_CHECK(boundary_minor_workload >= 0,
              "boundary_minor_workload must be non-negative");
  // Ranks <  num_boundary_majors are not boundary minor -> boundary_major_workload
  // Ranks >= num_boundary_majors are boundary minor     -> boundary_minor_workload
  // During the extended pass (is_at_policy_boundary() == true) all ranks
  // route their contribution increments into boundary_contributed_ and
  // check against boundary_target_contribution_. boundary_contributed_ is
  // reset by reset_contributed() at iteration end, so this call is safe to
  // invoke multiple times within the same boundary.
  if (currentRank_ >= num_boundary_majors) {
    set_boundary_minor();
    boundary_target_contribution_.store(boundary_minor_workload,
                                        std::memory_order_release);
  } else {
    reset_boundary_minor();
    boundary_target_contribution_.store(boundary_major_workload,
                                        std::memory_order_release);
  }
}

void ProcessGroupULFM::set_major_minor_split_with_spares(int num_majors, int num_minors, int num_major_spares, int num_minor_spares) {
  // Layout: [major workers | major spares | minor workers | minor spares]
  // - Ranks 0 to (num_majors - 1): Major workers
  // - Ranks num_majors to (num_majors + num_major_spares - 1): Major spares
  // - Ranks (num_majors + num_major_spares) to (num_majors + num_major_spares + num_minors - 1): Minor workers
  // - Remaining ranks: Minor spares

  int rank = currentRank_;

  int major_end = num_majors;
  int major_spare_end = major_end + num_major_spares;
  int minor_end = major_spare_end + num_minors;

  bool should_be_minor = (rank >= major_spare_end);
  bool should_be_spare = (rank >= major_end && rank < major_spare_end) ||  // major spare
                         (rank >= minor_end);                               // minor spare

  // Set minor/major status
  if (should_be_minor) {
    set_minor();
  } else {
    reset_minor();
  }

  // Set spare/worker status
  if (should_be_spare) {
    set_spare();
  } else {
    reset_spare();
  }
}

} // namespace c10d

// #endif // USE_C10D_MPI
