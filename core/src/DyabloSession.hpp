#pragma once

#include <memory>
#include <functional>

#include "utils/mpi/GlobalMpiSession.h"

namespace dyablo {

class GlobalMpiSession;
class MpiBufferPool;

/**
 * DyabloSession serves as a scope guard for MPI and Kokkos initialization/finalization
 * As for MPI and Kokkos, only one session can be opened per process.
 */
class DyabloSession {
private:
  static DyabloSession*& unique_session();

  std::unique_ptr<dyablo::GlobalMpiSession> mpiSession;
  std::unique_ptr<dyablo::MpiBufferPool> mpi_pool;
  std::vector< std::function<void()> > finalize_callbacks;
public:
  DyabloSession(int& argc, char *argv[]);
  ~DyabloSession();

  /**
   * Get the current DyabloSession.
   * Asserts when there is no active session (not opened or already finalized)
   **/
  static DyabloSession& get_DyabloSession();

  static dyablo::MpiBufferPool& get_MpiBufferPool();
  
  // Add callbacks to be called before finalization
  void call_before_finalize( std::function<void()> finalize_callback );
};


} // namespace dyablo
