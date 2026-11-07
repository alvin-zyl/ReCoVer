#include "ULFMLogging.hpp"

namespace c10d {

// Global verbose logging flag
bool g_ulfm_verbose_logging = false;

void set_ulfm_verbose_logging(bool verbose) {
  g_ulfm_verbose_logging = verbose;
}

bool is_ulfm_verbose_logging() {
  return g_ulfm_verbose_logging;
}

} // namespace c10d