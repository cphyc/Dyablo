#pragma once

#include <cassert>
#include <utility>

#include "utils/misc/Dyablo_assert.h"
#include "amr/LightOctree_base.h"
#include "enums.h"

namespace dyablo { 

template< typename MemorySpace_ = Kokkos::View<int*>::memory_space >
class LightOctree_storage
{
public:
  using MemorySpace = MemorySpace_;
  using logical_coord_t = uint32_t;
  using level_t = logical_coord_t;
  using oct_data_t = Kokkos::View< logical_coord_t**, Kokkos::LayoutLeft, MemorySpace >;
  using pos_t = Kokkos::Array<real_t,3>;
  using coarse_grid_size_t = Kokkos::Array<logical_coord_t,3>;
protected:
  using OctantIndex = LightOctree_base::OctantIndex;
  //! Index to access different fields in `oct_data`
  enum oct_data_field_t{
      ICORNERX, 
      ICORNERY, 
      ICORNERZ, 
      ILEVEL,
      OCT_DATA_COUNT
  };

public:
  LightOctree_storage() = default;
  LightOctree_storage(const LightOctree_storage& lmesh) = default;
  LightOctree_storage& operator=(const LightOctree_storage& lmesh) = default;
  LightOctree_storage(LightOctree_storage&& lmesh) = default;
  LightOctree_storage& operator=(LightOctree_storage&& lmesh) = default;

  template< typename MemorySpace_t >
  LightOctree_storage<MemorySpace_t> deep_copy() const
  {
    const LightOctree_storage& storage = *this;
    LightOctree_storage<MemorySpace_t> res( storage.getNdim(), storage.getNumOctants(), storage.getNumGhosts(), storage.getNumIntermediates(), storage.getNumIntermediateGhosts(), storage.level_min, storage.coarse_grid_size );
    Kokkos::deep_copy( res.oct_data, storage.oct_data );
    return res;
  }

  // Create an empty LightOctree_storage
  LightOctree_storage( int ndim, uint32_t numLocalLeaves, uint32_t numGhostLeaves,  uint32_t numLocalIntermediates, uint32_t numGhostIntermediates, level_t level_min, const coarse_grid_size_t& coarse_grid_size )
  : LightOctree_storage(ndim, numLocalLeaves, numGhostLeaves, numLocalIntermediates, numGhostIntermediates, level_min)
  {
    this->coarse_grid_size = coarse_grid_size;
  }

  // Create an empty LightOctree_storage (full coarse grid version)
  LightOctree_storage( int ndim, uint32_t numLocalLeaves, uint32_t numGhostLeaves,  uint32_t numLocalIntermediates, uint32_t numGhostIntermediates, level_t level_min )
  : ndim(ndim), numLocalLeaves(numLocalLeaves), numGhostLeaves(numGhostLeaves), numLocalIntermediates(numLocalIntermediates), numGhostIntermediates(numGhostIntermediates), level_min(level_min),
    coarse_grid_size( { (1U << level_min), (1U << level_min), (ndim==3)?(1U << level_min):1 }  ),
    oct_data("LightOctree_storage", numLocalLeaves+numGhostLeaves+numLocalIntermediates+numGhostIntermediates, oct_data_field_t::OCT_DATA_COUNT)
  {}

public:
  //! @copydoc LightOctree_base::getNumOctants()
  KOKKOS_INLINE_FUNCTION 
  uint32_t getNumOctants() const
  {
    return numLocalLeaves;
  }

  //! @copydoc LightOctree_base::getNumGhosts()
  KOKKOS_INLINE_FUNCTION 
  uint32_t getNumGhosts() const
  {
    return numGhostLeaves;
  }

  //! @copydoc LightOctree_base::getNumIntermediates()
  KOKKOS_INLINE_FUNCTION 
  uint32_t getNumIntermediates() const
  {
      return numLocalIntermediates;
  }  

  //! @copydoc LightOctree_base::getNumIntermediateGhosts()
  KOKKOS_INLINE_FUNCTION 
  uint32_t getNumIntermediateGhosts() const
  {
      return numGhostIntermediates;
  }  

  //! @copydoc LightOctree_base::getNdim()
  KOKKOS_INLINE_FUNCTION 
  uint32_t getNdim() const
  {
      return ndim;
  }
  //! @copydoc LightOctree_base::getCenter()
  KOKKOS_INLINE_FUNCTION 
  pos_t getCenter(const OctantIndex& iOct)  const
  {
      pos_t pos = getCorner(iOct);
      auto oct_size = getSize(iOct);
      return {
          pos[IX] + oct_size[IX]/2,
          pos[IY] + oct_size[IY]/2,
          pos[IZ] + (ndim-2)*(oct_size[IZ]/2)
      };
  }
  //! @copydoc LightOctree_base::getCorner()
  KOKKOS_INLINE_FUNCTION 
  pos_t getCorner(const OctantIndex& iOct)  const
  {
      auto lp = get_logical_coords(iOct);
      auto size = getSize(iOct);
      return {
        lp[IX] * size[IX],
        lp[IY] * size[IY],
        lp[IZ] * size[IZ]
      };
  }
   //! @copydoc LightOctree_base::getBound()
  KOKKOS_INLINE_FUNCTION 
  bool getBound(const OctantIndex& iOct)  const
  {
    auto lp = get_logical_coords(iOct);
    level_t level = getLevel(iOct);
    logical_coord_t last_oct_x = cell_count(IX, level)-1;
    logical_coord_t last_oct_y = cell_count(IY, level)-1;
    logical_coord_t last_oct_z = cell_count(IZ, level)-1;

    return lp[IX] == 0 || lp[IY] == 0 || lp[IZ] == 0 
        || lp[IX] == last_oct_x || lp[IY] == last_oct_y || lp[IZ] == last_oct_z;
  }

  KOKKOS_INLINE_FUNCTION 
  pos_t getSize(level_t level)  const
  {
      return { 1.0/cell_count( IX, level ),
               1.0/cell_count( IY, level ),
               1.0/cell_count( IZ, level ) };
  }

  //! @copydoc LightOctree_base::getSize()
  KOKKOS_INLINE_FUNCTION 
  pos_t getSize(const OctantIndex& iOct)  const
  {
    return getSize(getLevel(iOct));
  }

  //! @copydoc LightOctree_base::getLevel()
  KOKKOS_INLINE_FUNCTION level_t getLevel(const OctantIndex& iOct)  const
  {
      return oct_data(OctantIndex_to_iOctLocal(iOct), ILEVEL);
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Array<logical_coord_t, 3> get_logical_coords( const OctantIndex& iOct )  const
  {
    uint32_t iOct_local = OctantIndex_to_iOctLocal(iOct);
    return {
        oct_data(iOct_local, ICORNERX),
        oct_data(iOct_local, ICORNERY),
        oct_data(iOct_local, ICORNERZ),
    };
  }

  KOKKOS_INLINE_FUNCTION 
  void set( const OctantIndex& iOct, 
            logical_coord_t ix, logical_coord_t iy, logical_coord_t iz, 
            level_t level ) const
  {
    uint32_t iOct_local = OctantIndex_to_iOctLocal(iOct);
    oct_data(iOct_local, ICORNERX) = ix;
    oct_data(iOct_local, ICORNERY) = iy;
    oct_data(iOct_local, ICORNERZ) = iz;
    oct_data(iOct_local, ILEVEL) = level;
  }

  oct_data_t getLocalSubview() const
  {
    return Kokkos::subview( 
      oct_data, 
      std::make_pair( (uint32_t)0, 
                      getNumOctants() ),
      Kokkos::ALL() );
  }

  oct_data_t getLocalIntermediatesSubview() const
  {
    return Kokkos::subview( 
      oct_data, 
      std::make_pair( getNumOctants(), 
                      getNumOctants()+getNumIntermediates() ),
      Kokkos::ALL() );
  }

  oct_data_t getAllLocalsSubview() const
  {
    return Kokkos::subview( 
      oct_data, 
      std::make_pair( (uint32_t)0, 
                      getNumOctants()+getNumIntermediates() ),
      Kokkos::ALL() );
  }

  oct_data_t getGhostsSubview() const
  {
    return Kokkos::subview( 
      oct_data, 
      std::make_pair( getNumOctants()+getNumIntermediates(), 
                      getNumOctants()+getNumIntermediates()+getNumGhosts() ),
      Kokkos::ALL() );
  }

  oct_data_t getGhostIntermediatesSubview() const
  {
    return Kokkos::subview( 
      oct_data, 
      std::make_pair( getNumOctants()+getNumIntermediates()+getNumGhosts(), 
                      getNumOctants()+getNumIntermediates()+getNumGhosts()+getNumIntermediateGhosts() ),
      Kokkos::ALL() );
  }

  int ndim;
  uint32_t numLocalLeaves, numGhostLeaves; //! Number of local leaf octants (no ghosts), Number of leaf ghosts.
  uint32_t numLocalIntermediates, numGhostIntermediates; //! Number of local intermediate octants (no ghosts), Number of intermediate ghosts.
  level_t level_min;
  coarse_grid_size_t coarse_grid_size;

  KOKKOS_INLINE_FUNCTION
  logical_coord_t cell_count( ComponentIndex3D idim, level_t n ) const
  {
      DYABLO_ASSERT_KOKKOS_DEBUG( n>=level_min, "Cannot ask cell_count with level < level_min" );
      DYABLO_ASSERT_KOKKOS_DEBUG( n < sizeof(logical_coord_t)*8, "Overflow : cell_count too big for logical_coord_t" );
      return coarse_grid_size[idim] << (n-level_min);
  }

  //! Kokkos::view containing octants position and level 
  //! ex: (oct_data(iOct, ILEVEL) is octant level)
  oct_data_t oct_data;
  
  KOKKOS_INLINE_FUNCTION 
  uint32_t OctantIndex_to_iOctLocal(const OctantIndex& oct) const
  {
    // Octants are stored in this order : local leaves, local intermediates, ghost leaves, ghost intermediates
    if( !oct.isGhost )
    {
      return oct.iOct + oct.isIntermediate * this->numLocalLeaves;
    }
    else // if(oct.isGhost)
    {
      return oct.iOct + (this->numLocalLeaves + this->numLocalIntermediates) + oct.isIntermediate * this->numGhostLeaves;
    }
  }

  KOKKOS_INLINE_FUNCTION 
  OctantIndex iOctLocal_to_OctantIndex(uint32_t ioct_local) const 
  { 
    uint32_t total_local = (this->getNumOctants() + this->getNumIntermediates());

    bool ghost = ioct_local >= total_local;
    uint32_t iOct = ioct_local - ghost * total_local;

    bool intermediate = false;
    if( ghost && iOct >= this->getNumGhosts() )
    {
      intermediate = true;
      iOct -= this->getNumGhosts();
    }
    else if( !ghost && iOct >= this->getNumOctants() )
    {
      intermediate = true;
      iOct -= this->getNumOctants();
    }
    return OctantIndex{
      .iOct = iOct, 
      .isGhost=ghost,
      .isIntermediate=intermediate,
    };
  }
};

}