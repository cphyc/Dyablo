#pragma once

#include <memory>

#include "GhostCommunicator_full_blocks.h"
#include "GhostCommunicator_partial_blocks.h"

namespace dyablo{

/***
 * Class to generate Subset instances for per-level mpi communications
 * 
 * This precomputes Subsets for every level to quickly get 
 * per-level Subsets for multiple levels.
 ***/
class Subset_levels
{
public:
    Subset_levels(const LightOctree& lmesh, int level_max);
    ~Subset_levels();

    /// Instanciate an OctSubset to perform per-level MPI communication with GhostCommunicator_full_blocks
    GhostCommunicator_full_blocks::OctSubset getGhostCommunicatorSubset_level( int level, const GhostCommunicator_full_blocks& comm );
    /// Instanciate an OctSubset to perform per-level MPI communication with GhostCommunicator_partial_blocks
    GhostCommunicator_partial_blocks::OctSubset getGhostCommunicatorSubset_level( int level, const GhostCommunicator_partial_blocks& comm );

public:
    struct Pdata;
    std::unique_ptr<Pdata> pdata;
};

} // namespace dyablo