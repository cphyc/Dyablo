#pragma once

#include "LightOctree_hashmap.h"

namespace dyablo { 

namespace{
KOKKOS_INLINE_FUNCTION 
static uint32_t offset_to_index(int32_t offset_x, int32_t offset_y, int32_t offset_z, uint32_t ndims) 
{
    return offset_x+1 + 3*(offset_y+1) + (ndims==3)*3*3*(offset_z+1);
}
KOKKOS_INLINE_FUNCTION 
static uint32_t offset_to_index(const LightOctree_base::offset_t& offset, uint32_t ndims) 
{
    return offset_to_index(offset[IX], offset[IY], offset[IZ], ndims);
}

KOKKOS_INLINE_FUNCTION 
static LightOctree_base::offset_t index_to_offset(uint32_t index, uint32_t ndims)
{
    int32_t iz = index / (3*3);
    int32_t iy = (index - 3*3*iz)/3;
    int32_t ix = index % 3;

    return LightOctree_base::offset_t{ ix-1, iy-1, (ndims==2)?0:iz-1};
}

constexpr uint32_t NEIGHBOR_PRECOMPUTE_NOT_FOUND = (uint32_t)(-1);

} // namespace

inline void LightOctree_hashmap_precompute_init( 
    const LightOctree_hashmap& lmesh_hashmap, 
    const LightOctree_hashmap::Storage_t& storage,
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft >& neighbor_iOct_leaves,
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft >& neighbor_iOct_intermediates,
    Kokkos::View< uint32_t*, Kokkos::LayoutLeft >& neighbor_iOct_parents,
    Kokkos::View< uint32_t*, Kokkos::LayoutLeft >& neighbor_iOct_children,
    uint32_t ndims )
{
    using OctantIndex = LightOctree_base::OctantIndex;
    using offset_t = LightOctree_base::offset_t;

    uint32_t nneighbors = (ndims==2) ? 3*3 : 3*3*3;
    uint32_t nbOcts = storage.getNumOctants();
    uint32_t nbGhosts = storage.getNumGhosts();
    uint32_t nbIntermediates = storage.getNumIntermediates();
    uint32_t nbIntermediateGhosts = storage.getNumIntermediateGhosts();
    uint32_t level_min = storage.level_min;

    constexpr bool ghosts_enabled = true;
    constexpr bool no_assert_on_failure = false;
    {
        std::string name = "neighbor_iOct_leaves";
        uint32_t nbOcts_total = nbOcts+nbGhosts;
        Kokkos::View< uint32_t**, Kokkos::LayoutLeft > neighbor_iOcts(name, nbOcts_total, nneighbors );
        Kokkos::parallel_for( name+"::compute", nbOcts_total*nneighbors,
            KOKKOS_LAMBDA(uint32_t index)
        {
            uint32_t iOct_local = index%nbOcts_total;
            uint32_t neighbor_id = index/nbOcts_total;

            OctantIndex iOct = iOct_local < nbOcts  
                                ? OctantIndex{iOct_local, false}
                                : OctantIndex{iOct_local-nbOcts, true};
            offset_t offset = index_to_offset(neighbor_id, ndims);

            OctantIndex iOct_neighbor = lmesh_hashmap.findNeighbor<ghosts_enabled, no_assert_on_failure>( iOct, offset );
            uint32_t iOct_local_neighbor = LightOctree_base::neighbor_not_found( iOct_neighbor )
                                            ? NEIGHBOR_PRECOMPUTE_NOT_FOUND
                                            : storage.OctantIndex_to_iOctLocal( iOct_neighbor );
            neighbor_iOcts(iOct_local, neighbor_id) = iOct_local_neighbor;            
        });
        neighbor_iOct_leaves = neighbor_iOcts;
    }

    {
        std::string name = "neighbor_iOct_intermediates";
        uint32_t nbOcts_total = nbOcts+nbGhosts+nbIntermediates+nbIntermediateGhosts;
        Kokkos::View< uint32_t**, Kokkos::LayoutLeft > neighbor_iOcts(name, nbOcts_total, nneighbors );
        Kokkos::parallel_for( name+"::compute", nbOcts_total*nneighbors,
            KOKKOS_LAMBDA(uint32_t index)
        {
            uint32_t iOct_local = index%nbOcts_total;
            uint32_t neighbor_id = index/nbOcts_total;

            OctantIndex iOct = storage.iOctLocal_to_OctantIndex( iOct_local );
            offset_t offset = index_to_offset(neighbor_id, ndims);
            
            OctantIndex iOct_neighbor = lmesh_hashmap.findNeighbor_intermediate<ghosts_enabled, no_assert_on_failure>( iOct, offset );
            uint32_t iOct_local_neighbor = LightOctree_base::neighbor_not_found( iOct_neighbor )
                                            ? NEIGHBOR_PRECOMPUTE_NOT_FOUND
                                            : storage.OctantIndex_to_iOctLocal( iOct_neighbor );
            neighbor_iOcts(iOct_local, neighbor_id) = iOct_local_neighbor;
        });
        neighbor_iOct_intermediates = neighbor_iOcts;
    }

    {
        std::string name = "neighbor_iOct_parents";
        uint32_t nbOcts_total = nbOcts+nbIntermediates;
        Kokkos::View< uint32_t*, Kokkos::LayoutLeft > neighbor_iOcts(name, nbOcts_total);
        Kokkos::parallel_for( name+"::compute", nbOcts_total,
            KOKKOS_LAMBDA(uint32_t iOct_local)
        {
            OctantIndex iOct = ( iOct_local<nbOcts ) ? OctantIndex{iOct_local, false, false} : OctantIndex{iOct_local-nbOcts, false, true};

            if(lmesh_hashmap.getLevel( iOct ) > level_min)
            {
                OctantIndex iOct_neighbor = lmesh_hashmap.findParent( iOct );
                uint32_t iOct_local_neighbor = storage.OctantIndex_to_iOctLocal( iOct_neighbor );
                neighbor_iOcts(iOct_local) = iOct_local_neighbor;
            }
        });
        neighbor_iOct_parents = neighbor_iOcts;
    }

    {
        std::string name = "neighbor_iOct_children";
        uint32_t nbOcts_total = nbIntermediates;
        Kokkos::View< uint32_t*, Kokkos::LayoutLeft > neighbor_iOcts(name, nbOcts_total);
        Kokkos::parallel_for( name+"::compute", nbOcts_total,
            KOKKOS_LAMBDA(uint32_t iOct_local)
        {
            OctantIndex iOct = OctantIndex{iOct_local, false, true};

            OctantIndex iOct_neighbor = lmesh_hashmap.findChild( iOct, {0,0,0} );
            uint32_t iOct_local_neighbor = storage.OctantIndex_to_iOctLocal( iOct_neighbor );
            neighbor_iOcts(iOct_local) = iOct_local_neighbor;
        });
        neighbor_iOct_children = neighbor_iOcts;
    }
}


class LightOctree_hashmap_precompute : public LightOctree_hashmap{
public:
    LightOctree_hashmap_precompute() = default;
    LightOctree_hashmap_precompute(const LightOctree_hashmap_precompute& lmesh) = default;

    template < typename AMRmesh_t >
    LightOctree_hashmap_precompute( const AMRmesh_t* pmesh, level_t level_min, level_t level_max )
    : LightOctree_hashmap(pmesh, level_min, level_max)
    {
        LightOctree_hashmap_precompute_init(
            *this, this->storage, 
            this->neighbor_iOct_leaves, 
            this->neighbor_iOct_intermediates, 
            this->neighbor_iOct_parents, 
            this->neighbor_iOct_children, 
            getNdim());
    }


    template< bool has_intermediates, bool accepts_ghosts, bool assert_on_failure >
    KOKKOS_INLINE_FUNCTION
    OctantIndex findNeighbor_aux(const OctantIndex& iOct, int32_t offset_x, int32_t offset_y, int32_t offset_z, const Kokkos::View< uint32_t**, Kokkos::LayoutLeft >& neighbor_iOcts) const
    {
        static_assert( assert_on_failure, "Only assert_on_failure = true is supported for LighOctree_hashmap_precompute::findNeighbor*" );

        if( offset_x == 0 && offset_y == 0 && offset_z == 0 )
            return iOct;

        DYABLO_ASSERT_KOKKOS_DEBUG( !this->isBoundary(iOct, offset_x, offset_y, offset_z), "findNeighbor not compatible with boundaries, please check isBoundary() before" );
        if( !has_intermediates )
        {
            DYABLO_ASSERT_KOKKOS_DEBUG( !iOct.isIntermediate, "findNeighbor can't get neighbor of intermediate octant, use findNeighbor_intermediate" );
        }
        if( !accepts_ghosts )
        {
            DYABLO_ASSERT_KOKKOS_DEBUG(!iOct.isGhost, "LightOctree_hashmap_precompute::findNeighbor_aux : iOct is a ghost but ghosts are disabled");
        }

        uint32_t iOct_local;
        if constexpr ( has_intermediates )
        {
            iOct_local = storage.OctantIndex_to_iOctLocal( iOct );
        }
        else
        {
            iOct_local = iOct.iOct + storage.getNumOctants() * iOct.isGhost;
        }
        uint32_t neighbor_id = offset_to_index(offset_x, offset_y, offset_z, getNdim());
        uint32_t iOct_local_neighbor = neighbor_iOcts( iOct_local, neighbor_id );

        OctantIndex iOct_n = storage.iOctLocal_to_OctantIndex(iOct_local_neighbor);

        [[maybe_unused]] auto test_same_as_LightOctree_hashmap = [&]()
        {
            OctantIndex iOct_expected = LightOctree_hashmap::findNeighbor_aux<has_intermediates, accepts_ghosts, assert_on_failure>( iOct, offset_x, offset_y, offset_z );
            return iOct_expected.iOct == iOct_n.iOct && iOct_expected.isGhost == iOct_n.isGhost && iOct_expected.isIntermediate == iOct_n.isIntermediate;
        };
        DYABLO_ASSERT_KOKKOS_DEBUG( test_same_as_LightOctree_hashmap() 
            , "LightOctree_hashmap_precompute::findNeighbor does not match LightOctree_hashmap" );

        DYABLO_ASSERT_KOKKOS_DEBUG( iOct_local_neighbor != NEIGHBOR_PRECOMPUTE_NOT_FOUND, "Neighbor not found!" );
        

        return iOct_n;
    }

    //! @copydoc LightOctree_base::findNeighbor()
    template< bool accepts_ghosts = false, bool assert_on_failure = true >
    KOKKOS_INLINE_FUNCTION
    OctantIndex findNeighbor(const OctantIndex& iOct, const offset_t& offset) const
    {
        return findNeighbor_aux<false, accepts_ghosts, assert_on_failure>( iOct, offset[IX], offset[IY], offset[IZ], neighbor_iOct_leaves );
    }

    //! @copydoc LightOctree_base::findNeighbor_intermediates()
    template< bool accepts_ghosts = false, bool assert_on_failure = true >
    KOKKOS_INLINE_FUNCTION
    OctantIndex findNeighbor_intermediate(const OctantIndex& iOct, const offset_t& offset) const
    {
        return findNeighbor_aux<true, accepts_ghosts, assert_on_failure>( iOct, offset[IX], offset[IY], offset[IZ], neighbor_iOct_intermediates );
    }

     //! @copydoc LightOctree_base::findNeighbor()
    template< bool accepts_ghosts = false, bool assert_on_failure = true >
    KOKKOS_INLINE_FUNCTION
    OctantIndex findNeighbor(const OctantIndex& iOct, int32_t offset_x, int32_t offset_y, int32_t offset_z) const
    {
        return findNeighbor_aux<false, accepts_ghosts, assert_on_failure>(iOct, offset_x, offset_y, offset_z, neighbor_iOct_leaves);
    }

    //! @copydoc LightOctree_base::findNeighbor_intermediate()
    template< bool accepts_ghosts = false, bool assert_on_failure = true >
    KOKKOS_INLINE_FUNCTION 
    OctantIndex findNeighbor_intermediate( const OctantIndex& iOct, int32_t offset_x, int32_t offset_y, int32_t offset_z)  const
    {
        return findNeighbor_aux<true, accepts_ghosts, assert_on_failure>(iOct, offset_x, offset_y, offset_z, neighbor_iOct_intermediates);
    }

    KOKKOS_INLINE_FUNCTION
    OctantIndex findParent( const OctantIndex& iOct )  const
    {
        DYABLO_ASSERT_KOKKOS_DEBUG( this->getLevel( iOct ) > this->min_level, "Can't get parent at coarse level" );
        
        if( iOct.isGhost )
            return LightOctree_hashmap::findParent( iOct );
        
        uint32_t iOct_local = iOct.iOct + iOct.isIntermediate * getNumOctants();
        uint32_t iOct_local_p = neighbor_iOct_parents(iOct_local);

        OctantIndex iOct_p = storage.iOctLocal_to_OctantIndex( iOct_local_p );

        DYABLO_ASSERT_KOKKOS_DEBUG( [&]()
            {
                OctantIndex iOct_expected = LightOctree_hashmap::findParent( iOct );
                return iOct_expected.iOct == iOct_p.iOct && iOct_expected.isGhost == iOct_p.isGhost && iOct_expected.isIntermediate == iOct_p.isIntermediate;
            }() 
            , "LightOctree_hashmap_precompute::findParent does not match LightOctree_hashmap" );

        return iOct_p;
    }

    KOKKOS_INLINE_FUNCTION
    OctantIndex findChild( const OctantIndex& iOct, const offset_t& offset )  const
    {
        DYABLO_ASSERT_KOKKOS_DEBUG( iOct.isIntermediate, "Leaves have no children" );
    
        if( iOct.isGhost )
            return LightOctree_hashmap::findChild( iOct, offset );
        
        uint32_t iOct_local = iOct.iOct;
        uint32_t iOct_local_c0 = neighbor_iOct_children(iOct_local);
        OctantIndex iOct_c0 = storage.iOctLocal_to_OctantIndex( iOct_local_c0 );

        OctantIndex iOct_c = findNeighbor_intermediate(iOct_c0, offset);
        DYABLO_ASSERT_KOKKOS_DEBUG( [&]()
            {
                OctantIndex iOct_expected = LightOctree_hashmap::findChild( iOct, offset );
                return iOct_expected.iOct == iOct_c.iOct && iOct_expected.isGhost == iOct_c.isGhost && iOct_expected.isIntermediate == iOct_c.isIntermediate;
            }() 
            , "LightOctree_hashmap_precompute::findChild does not match LightOctree_hashmap" );
        return iOct_c;
    }


private:
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft > neighbor_iOct_leaves; 
    Kokkos::View< uint32_t**, Kokkos::LayoutLeft > neighbor_iOct_intermediates; 
    Kokkos::View< uint32_t*, Kokkos::LayoutLeft > neighbor_iOct_parents; 
    Kokkos::View< uint32_t*, Kokkos::LayoutLeft > neighbor_iOct_children; 
};

} //namespace dyablo
