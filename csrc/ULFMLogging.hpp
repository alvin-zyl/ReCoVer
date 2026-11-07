#pragma once

#include <torch/csrc/distributed/c10d/logger.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <c10/util/Exception.h>
#include <sstream>

namespace c10d {

// Global verbose logging flag - can be controlled from Python
extern bool g_ulfm_verbose_logging;

// Set verbose logging mode
void set_ulfm_verbose_logging(bool verbose);

// Get current verbose logging state
bool is_ulfm_verbose_logging();

// ULFM-specific logging macros using PyTorch's logging system
#define ULFM_LOG_ERROR(rank, msg) \
  do { \
    std::ostringstream oss; \
    oss << "[ULFM Rank " << rank << "] [ERROR] " << msg; \
    TORCH_CHECK(false, oss.str()); \
  } while(0)

#define ULFM_LOG_WARN(rank, msg) \
  do { \
    std::ostringstream oss; \
    oss << "[ULFM Rank " << rank << "] [WARN] " << msg; \
    TORCH_WARN(oss.str()); \
  } while(0)

#define ULFM_LOG_INFO(rank, msg) \
  do { \
    if (c10d::is_ulfm_verbose_logging()) { \
      std::ostringstream oss; \
      oss << "[ULFM Rank " << rank << "] [INFO] " << msg; \
      TORCH_WARN(oss.str()); \
    } \
  } while(0)

#define ULFM_LOG_DEBUG(rank, msg) \
  do { \
    if (c10d::is_ulfm_verbose_logging()) { \
      std::ostringstream oss; \
      oss << "[ULFM Rank " << rank << "] [DEBUG] " << msg; \
      TORCH_WARN(oss.str()); \
    } \
  } while(0)

#define ULFM_LOG_TRACE(rank, msg) \
  do { \
    if (c10d::is_ulfm_verbose_logging()) { \
      std::ostringstream oss; \
      oss << "[ULFM Rank " << rank << "] [TRACE] " << msg; \
      TORCH_WARN(oss.str()); \
    } \
  } while(0)

// Helper macros for common patterns
#define ULFM_LOG_FAILURE_DETECTED(rank, failed_ranks) \
  do { \
    std::string ranks_str = "["; \
    for (size_t i = 0; i < failed_ranks.size(); ++i) { \
      if (i > 0) ranks_str += ", "; \
      ranks_str += std::to_string(failed_ranks[i]); \
    } \
    ranks_str += "]"; \
    ULFM_LOG_WARN(rank, "Failure detected in ranks: " + ranks_str); \
  } while(0)

#define ULFM_LOG_RECOVERY_SUCCESS(rank, operation) \
  do { \
    ULFM_LOG_INFO(rank, operation + " completed successfully"); \
  } while(0)

#define ULFM_LOG_RECOVERY_FAILED(rank, operation, error) \
  do { \
    ULFM_LOG_ERROR(rank, operation + " failed: " + std::string(error)); \
  } while(0)

#define ULFM_LOG_COMMUNICATOR_REPAIRED(rank, orig_rank, new_rank, new_size) \
  do { \
    ULFM_LOG_INFO(rank, "Communicator repaired: original_rank=" + std::to_string(orig_rank) + \
                        ", new_rank=" + std::to_string(new_rank) + \
                        ", new_size=" + std::to_string(new_size)); \
  } while(0)

} // namespace c10d
