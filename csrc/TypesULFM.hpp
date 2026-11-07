#pragma once

namespace c10d {

// Counts for each rank type (majors, minors, major_spares, minor_spares, boundary_minors)
struct RankTypeCounts {
  int majors = 0;
  int minors = 0;
  int major_spares = 0;
  int minor_spares = 0;
  int boundary_minors = 0;
  int64_t contributed = 0;           // Global sum of regular-phase gradient contributions
  int64_t boundary_contributed = 0;  // Global sum of boundary-phase gradient contributions

  RankTypeCounts() = default;
  RankTypeCounts(int m, int n, int ms, int ns, int bm = 0)
      : majors(m), minors(n), major_spares(ms), minor_spares(ns), boundary_minors(bm) {}
};

// Statistics about failed ranks by type
struct FailureStats {
  int failed_majors = 0;
  int failed_minors = 0;
  int failed_major_spares = 0;
  int failed_minor_spares = 0;
  int failed_boundary_minors = 0;
  bool at_policy_boundary = false;  // True when worker failed with no matching spares

  FailureStats() = default;
  FailureStats(int fm, int fn, int fms, int fns, int fbm = 0, bool boundary = false)
      : failed_majors(fm), failed_minors(fn),
        failed_major_spares(fms), failed_minor_spares(fns),
        failed_boundary_minors(fbm),
        at_policy_boundary(boundary) {}
};

// Failure handling strategies for ULFM
enum class ULFMFailureHandlingStrategy {
  CONTINUE_WITH_SURVIVORS,  // Continue training with remaining processes
  RESTART_FAILED_PROCESSES, // Restart failed processes (if supported by scheduler)
  ABORT_ON_FAILURE         // Abort training on any process failure
};

struct ULFMOptions {
  bool auto_repair = false;
  ULFMFailureHandlingStrategy failure_strategy = ULFMFailureHandlingStrategy::CONTINUE_WITH_SURVIVORS;
  int max_retries = 5;
  int retry_delay_ms = 100;
  bool track_rank_types = false;  // Enable rank type tracking & failure counting
  bool auto_elect = true;         // Auto-elect spare promotion (default ON)
  bool consensus_on_rank_types = false; // Whether to do consensus on rank types
  // Add more in future, like timeout, etc.
};

}
