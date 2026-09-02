#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include "morton_utils.h"
#include "amr/AMRmesh.h"
#include "amr/LightOctree_storage.h"
#include "utils/misc/Dyablo_assert.h"
#include "amr/LightOctree_base.h"

namespace dyablo { 

class LightOctree_hashmap : public LightOctree_base{
public:
    using Storage_t = LightOctree_storage<>;
    using LightOctree_base::OctantIndex;
    using LightOctree_base::pos_t;
    using morton_t = uint64_t;

    LightOctree_hashmap() = default;
    LightOctree_hashmap(const LightOctree_hashmap& lmesh) = default;

    LightOctree_hashmap( Storage_t&& storage, 
                         level_t level_min, level_t level_max,
                         Kokkos::Array<bool,3> periodic );

    LightOctree_hashmap( const AMRmesh* pmesh, level_t level_min, level_t level_max );

    KOKKOS_INLINE_FUNCTION
    int getNdim() const
    {return storage.getNdim();}
    
    KOKKOS_INLINE_FUNCTION
    uint32_t getNumOctants() const
    {return storage.getNumOctants();}
    
    KOKKOS_INLINE_FUNCTION
    uint32_t getNumGhosts() const
    {return storage.getNumGhosts();}

    KOKKOS_INLINE_FUNCTION
    uint32_t getNumIntermediates() const
    {return storage.getNumIntermediates();}

    KOKKOS_INLINE_FUNCTION
    uint32_t getNumIntermediateGhosts() const
    {return storage.getNumIntermediateGhosts();}

    KOKKOS_INLINE_FUNCTION
    pos_t getCenter(const OctantIndex& iOct)  const
    {return storage.getCenter(iOct);}
   
    KOKKOS_INLINE_FUNCTION
    pos_t getCorner(const OctantIndex& iOct)  const
    {return storage.getCorner(iOct);}

    KOKKOS_INLINE_FUNCTION
    pos_t getSize(const OctantIndex& iOct)  const
    {return storage.getSize(iOct);}

    KOKKOS_INLINE_FUNCTION
    pos_t getSize(Storage_t::level_t level)  const
    {return storage.getSize(level);}
    
    KOKKOS_INLINE_FUNCTION
    level_t getLevel(const OctantIndex& iOct)  const
    {return storage.getLevel(iOct);}
    
    KOKKOS_INLINE_FUNCTION
    bool getBound(const OctantIndex& iOct)  const
    {return storage.getBound(iOct);}

    const Storage_t getStorage() const 
    {
        return storage;
    }

    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<uint32_t, 3> get_logical_coords(const OctantIndex& iOct)  const
    {
        return storage.get_logical_coords(iOct);
    }

    //! @copydoc LightOctree_base::findNeighbor()
    template< bool accepts_ghosts = false, bool assert_on_failure = true >
    KOKKOS_INLINE_FUNCTION
    OctantIndex findNeighbor(const OctantIndex& iOct, const offset_t& offset) const
    {
        return findNeighbor_aux<false, accepts_ghosts, assert_on_failure>(iOct, offset[IX], offset[IY], offset[IZ]);
    }

    //! @copydoc LightOctree_base::findNeighbor_intermediate()
    template< bool accepts_ghosts = false, bool assert_on_failure = true >
    KOKKOS_INLINE_FUNCTION 
    OctantIndex findNeighbor_intermediate( const OctantIndex& iOct, const offset_t& offset )  const
    {
        return findNeighbor_aux<true, accepts_ghosts, assert_on_failure>(iOct, offset[IX], offset[IY], offset[IZ]);
    }

    //! @copydoc LightOctree_base::findNeighbor()
    template< bool accepts_ghosts = false, bool assert_on_failure = true >
    KOKKOS_INLINE_FUNCTION
    OctantIndex findNeighbor(const OctantIndex& iOct, int32_t offset_x, int32_t offset_y, int32_t offset_z) const
    {
        return findNeighbor_aux<false, accepts_ghosts, assert_on_failure>(iOct, offset_x, offset_y, offset_z);
    }

    //! @copydoc LightOctree_base::findNeighbor_intermediate()
    template< bool accepts_ghosts = false, bool assert_on_failure = true >
    KOKKOS_INLINE_FUNCTION 
    OctantIndex findNeighbor_intermediate( const OctantIndex& iOct, int32_t offset_x, int32_t offset_y, int32_t offset_z)  const
    {
        return findNeighbor_aux<true, accepts_ghosts, assert_on_failure>(iOct, offset_x, offset_y, offset_z);
    }

    template< bool search_intermediate, bool accepts_ghosts, bool assert_on_failure >
    KOKKOS_INLINE_FUNCTION
    OctantIndex findNeighbor_aux(const OctantIndex& iOct, int32_t offset_x, int32_t offset_y, int32_t offset_z) const
    {

#define findNeighbor_aux_ASSERT_OR_FAIL( cond, message )    \
{                                                           \
    if constexpr ( assert_on_failure )                      \
    {                                                       \
        DYABLO_ASSERT_KOKKOS_DEBUG( cond, message );        \
    }                                                       \
    else                                                    \
    {                                                       \
        if( !(cond) )                                         \
        {                                                   \
            return LightOctree_base::NEIGHBOR_NOT_FOUND;    \
        }                                                   \
    }                                                       \
}                                                           \

        if( offset_x == 0 && offset_y == 0 && offset_z == 0 )
            return iOct;

        if constexpr ( !accepts_ghosts )
        {
           findNeighbor_aux_ASSERT_OR_FAIL(!iOct.isGhost, "LightOctree_hashmap::findNeighbor_aux : iOct is a ghost but ghosts are disabled");
        }

        findNeighbor_aux_ASSERT_OR_FAIL( !this->isBoundary(iOct, offset_x, offset_y, offset_z), "findNeighbor doesn't support boundaries." );

        // Get logical coordinates of neighbor        
        level_t level = getLevel(iOct);
        auto lc = storage.get_logical_coords(iOct);
        logical_coord_t octant_count_x = storage.cell_count(IX, level );
        logical_coord_t octant_count_y = storage.cell_count(IY, level );
        logical_coord_t octant_count_z = storage.cell_count(IZ, level );
        key_t logical_coords{
            .level = getLevel(iOct),
            .i = (lc[IX] + octant_count_x + offset_x) % octant_count_x, // Periodic coord only works if offset > -octant_count
            .j = (lc[IY] + octant_count_y + offset_y) % octant_count_y,
            .k = (lc[IZ] + octant_count_z + offset_z) % octant_count_z,
        };

        if constexpr ( search_intermediate )
        {
            // Looking for intermediates : same level should exist
            auto it = oct_map.find(logical_coords);
            if( oct_map.valid_at(it))
            {
                return oct_map.value_at(it);
            }
            else // If not, is for sure a larger leaf
            {
                key_t logical_coords_bigger {
                    .level = logical_coords.level-1,
                    .i = logical_coords.i >> 1,
                    .j = logical_coords.j >> 1,
                    .k = logical_coords.k >> 1
                };
                // Search octant at coarser level
                auto it = oct_map.find(logical_coords_bigger);
                findNeighbor_aux_ASSERT_OR_FAIL(oct_map.valid_at(it), "Could not find neighbor : not found");
                findNeighbor_aux_ASSERT_OR_FAIL(!oct_map.value_at(it).isIntermediate, "Bigger neighbour must be a leaf ");
                return oct_map.value_at(it);
            }
        }
        else
        {
            // Looking for leaves : look at all levels
            OctantIndex res{};
            // Search octant at same level
            auto it = oct_map.find(logical_coords);
            if( oct_map.valid_at(it) && !oct_map.value_at(it).isIntermediate )
            {
                // Found at same level
                res = oct_map.value_at(it);
            }
            else
            {
                key_t logical_coords_bigger{
                    .level = logical_coords.level-1,
                    .i = logical_coords.i >> 1,
                    .j = logical_coords.j >> 1,
                    .k = logical_coords.k >> 1,
                };

                // Search octant at coarser level
                auto it = oct_map.find(logical_coords_bigger);
                if( oct_map.valid_at(it) && !oct_map.value_at(it).isIntermediate ) 
                {
                    // Found at coarser level
                    res = oct_map.value_at(it);
                }
                else
                {
                    // Neighbor(s) is(are) at finer level
                    findNeighbor_aux_ASSERT_OR_FAIL(level+1 <= max_level, "Could not find neighbor : already at level_max");
                    
                    // Compute logical coord of first neighbor
                    key_t logical_coords_smaller_origin{
                        .level = logical_coords.level+1,
                        .i = (logical_coords.i << 1) + (offset_x==-1),
                        .j = (logical_coords.j << 1) + (offset_y==-1),
                        .k = (logical_coords.k << 1) + (offset_z==-1),
                    };
                    
                    auto it = oct_map.find(logical_coords_smaller_origin);
                    findNeighbor_aux_ASSERT_OR_FAIL(oct_map.valid_at(it), "Could not find neighbor : not found");
                    findNeighbor_aux_ASSERT_OR_FAIL(!oct_map.value_at(it).isIntermediate, "Could not find neighbor : is intermediate");
                    res = oct_map.value_at(it);
                }            
            }
            return res;
        }
#undef findNeighbor_aux_ASSERT_OR_FAIL
    
    }

    //! @copydoc LightOctree_base::findNeighbors()
    [[deprecated]]
    KOKKOS_INLINE_FUNCTION 
    NeighborList findNeighbors( const OctantIndex& iOct, const offset_t& offset )  const
    {
        if( this->isBoundary(iOct, offset) )
            return NeighborList{0,{}};

        OctantIndex iOct_n = findNeighbor(iOct, offset);

        if( this->getLevel( iOct_n ) <= this->getLevel( iOct ) )
        {
            // Neighbor is same size of bigger
            return NeighborList{1, {iOct_n}};
        }
        else //( this->getLevel( iOct_n ) > this->getLevel( iOct ) )
        {
            // Compute logical coord of first neighbor    
            auto lc = storage.get_logical_coords( iOct_n );
            key_t logical_coords_smaller_origin;
            logical_coords_smaller_origin.level = getLevel(iOct_n);
            logical_coords_smaller_origin.i = lc[IX];
            logical_coords_smaller_origin.j = lc[IY];
            logical_coords_smaller_origin.k = lc[IZ];
            uint32_t ndim = getNdim();  
            uint32_t sz_max = (ndim==2) ? 0 : (offset[IZ]==0); // No offset in z in 2D
            uint32_t sy_max = (offset[IY]==0);
            uint32_t sx_max = (offset[IX]==0); // Constrained to plane adjacent to neighbor if offset in this direction
            
            NeighborList res{0};
            for( uint32_t sz=0; sz<=sz_max; sz++ )
            for( uint32_t sy=0; sy<=sy_max; sy++ )
            for( uint32_t sx=0; sx<=sx_max; sx++ )
            {
                res.m_size++;
                key_t logical_coords_smaller = logical_coords_smaller_origin;
                logical_coords_smaller.i += sx;
                logical_coords_smaller.j += sy;
                logical_coords_smaller.k += sz;
                auto it = oct_map.find(logical_coords_smaller);
                DYABLO_ASSERT_KOKKOS_DEBUG(oct_map.valid_at(it), "Could not find neighbor");
                res.m_neighbors[res.m_size-1] = oct_map.value_at(it);
            }
            DYABLO_ASSERT_KOKKOS_DEBUG(res.m_size<=2*(ndim-1), "Too many neighbors");
            return res;
        }
        
    }    

    /// @copydoc LightOctree_base::isBoundary()
    KOKKOS_INLINE_FUNCTION
    bool isBoundary(const OctantIndex& iOct, int32_t offset_x, int32_t offset_y, int32_t offset_z) const {       
        if(     (offset_x == 0 || this->is_periodic[IX]) 
            &&  (offset_y == 0 || this->is_periodic[IY]) 
            &&  (offset_z == 0 || this->is_periodic[IZ]) )
        {
            return false;
        }
        auto logical_coord = this->get_logical_coords( iOct );
        int level = this->getLevel( iOct );
        uint32_t cell_count_x = this->storage.cell_count( IX, level );
        uint32_t cell_count_y = this->storage.cell_count( IY, level );
        uint32_t cell_count_z = this->storage.cell_count( IZ, level );
        return ( offset_x != 0 && !this->is_periodic[IX] && ( /*(logical_coord[IX] + offset_x) < 0 ||*/ ( logical_coord[IX] + offset_x ) >= cell_count_x ) )
            || ( offset_y != 0 && !this->is_periodic[IY] && ( /*(logical_coord[IY] + offset_y) < 0 ||*/ ( logical_coord[IY] + offset_y ) >= cell_count_y ) )
            || ( offset_z != 0 && !this->is_periodic[IZ] && ( /*(logical_coord[IZ] + offset_z) < 0 ||*/ ( logical_coord[IZ] + offset_z ) >= cell_count_z ) );
    }


    KOKKOS_INLINE_FUNCTION
    bool isBoundary(const OctantIndex& iOct, const offset_t& offset) const {
      return isBoundary(iOct, offset[IX], offset[IY], offset[IZ]);      
    }

    /// @copydoc LightOctree_base::findChild()
    KOKKOS_INLINE_FUNCTION
    OctantIndex findChild( const OctantIndex& iOct, const offset_t& offset )  const
    {
        DYABLO_ASSERT_KOKKOS_DEBUG( iOct.isIntermediate, "Leaves have no children" );
        DYABLO_ASSERT_KOKKOS_DEBUG(     offset[IX] >=0 && offset[IX] < 2 
                                    &&  offset[IY] >=0 && offset[IY] < 2 
                                    &&  offset[IZ] >=0 && offset[IZ] < 2,
                                    "findChild offset must be in [0,1]" );
        
        auto lc = storage.get_logical_coords(iOct);
        
        key_t logical_coords{
            .level = (logical_coord_t)(this->getLevel(iOct)+1),
            .i = 2u*lc[IX] + offset[IX],
            .j = 2u*lc[IY] + offset[IY],
            .k = 2u*lc[IZ] + offset[IZ],
        };

        return this->getiOctFromCoordinates(logical_coords.i, logical_coords.j, logical_coords.k, logical_coords.level);
    }

    KOKKOS_INLINE_FUNCTION
    OctantIndex findParent( const OctantIndex& iOct )  const
    {
        DYABLO_ASSERT_KOKKOS_DEBUG( this->getLevel( iOct ) > this->min_level, "Can't get parent at coarse level" );

        const auto lc = storage.get_logical_coords(iOct);
        key_t logical_coords{
            .level = this->getLevel(iOct) - 1u,
            .i = (lc[IX] >> 1u),
            .j = (lc[IY] >> 1u),
            .k = (lc[IZ] >> 1u),
        };
        
        return this->getiOctFromCoordinates(logical_coords.i, logical_coords.j, logical_coords.k, logical_coords.level);
    }


    // ------------------------
    // Only in LightOctree_hashmap
    // ------------------------
    /**
     * Get octant from logical position
     **/
    KOKKOS_INLINE_FUNCTION
    OctantIndex getiOctFromCoordinates(uint16_t ix, uint16_t iy, uint16_t iz, uint16_t level) const
    {
        int ndim = getNdim();

        DYABLO_ASSERT_KOKKOS_DEBUG( ix < storage.cell_count(IX, level ), "ix out of bound" );
        DYABLO_ASSERT_KOKKOS_DEBUG( iy < storage.cell_count(IY, level ), "iy out of bound"  );
        if(ndim == 3)
        {
            DYABLO_ASSERT_KOKKOS_DEBUG( iz < storage.cell_count(IZ, level ), "iz out of bound" );
        }
        else 
            DYABLO_ASSERT_KOKKOS_DEBUG( iz == 0, "iz must be 0 in 2D"  );

        auto it = oct_map.find({level, ix, iy, iz});

        DYABLO_ASSERT_KOKKOS_DEBUG( oct_map.valid_at(it), "Could not find iOct" );

        return oct_map.value_at(it);
    }
    /**
     * Get octant containing position pos
     **/
    KOKKOS_INLINE_FUNCTION
    OctantIndex getiOctFromPos(const pos_t& pos) const
    {
        int ndim = getNdim();

        DYABLO_ASSERT_KOKKOS_DEBUG( 0 < pos[IX] && pos[IX] < 1, "pos_x out of bounds" );
        DYABLO_ASSERT_KOKKOS_DEBUG( 0 < pos[IY] && pos[IY] < 1, "pos_y out of bounds" );
        if(ndim == 3)
        {
            DYABLO_ASSERT_KOKKOS_DEBUG( 0 < pos[IZ] && pos[IZ] < 1, "pos_z out of bounds" );
        }
        else
            DYABLO_ASSERT_KOKKOS_DEBUG( pos[IZ] == 0, "pos_z should be 0 in 2D");

        key_t logical_coords;
        {
            real_t octant_size_x = 1.0/( storage.cell_count(IX, max_level) );
            real_t octant_size_y = 1.0/( storage.cell_count(IY, max_level) );
            real_t octant_size_z = 1.0/( storage.cell_count(IZ, max_level) );
            logical_coords.level = max_level;
            logical_coords.i = std::floor(pos[IX]/octant_size_x);
            logical_coords.j = std::floor(pos[IY]/octant_size_y);
            logical_coords.k = (ndim-2)*std::floor(pos[IZ]/octant_size_z);
        }

        for(level_t level=max_level; level>=min_level; level--)
        {
            auto it = oct_map.find(logical_coords);

            if( oct_map.valid_at(it) ) 
            {
                return oct_map.value_at(it);
            }
            logical_coords.level = logical_coords.level-1;
            logical_coords.i = logical_coords.i >> 1;
            logical_coords.j = logical_coords.j >> 1;
            logical_coords.k = logical_coords.k >> 1;
        }

        DYABLO_ASSERT_KOKKOS_DEBUG(false, "Could not find octant at this position");
        return {};
    }

    KOKKOS_INLINE_FUNCTION
    int getDomainFromPos(const pos_t& pos) const
    {
        int ndim = getNdim();

        assert( 0 < pos[IX] && pos[IX] < 1 );
        assert( 0 < pos[IY] && pos[IY] < 1 );
        if(ndim == 3)
            assert( 0 < pos[IZ] && pos[IZ] < 1 );
        else
            assert( pos[IZ] == 0 );
        morton_t morton;
        {
            index_t<3> logical_coords;
            real_t octant_size_x = 1.0/( storage.cell_count(IX, max_level) );
            real_t octant_size_y = 1.0/( storage.cell_count(IY, max_level) );
            real_t octant_size_z = 1.0/( storage.cell_count(IZ, max_level) );
            logical_coords[IX] = std::floor(pos[IX]/octant_size_x);
            logical_coords[IY] = std::floor(pos[IY]/octant_size_y);
            logical_coords[IZ] = (ndim-2)*std::floor(pos[IZ]/octant_size_z);

            morton = compute_morton_key( logical_coords );
        }

        // first i with morton_intervals[i] > morton, minus 1
        int res = -1;
        for(size_t i=0; i<morton_intervals.size(); i++)
            if( morton_intervals(i) > morton )
            {
                res = i-1; 
                break;
            }

        assert( 0 <= res && res < (int)morton_intervals.size()-1 );

        return res;
    }


    using logical_coord_t = uint32_t;
    using level_t = logical_coord_t;
    struct key_t //! key type for hashmap (morton+level)
    {
        logical_coord_t level, i, j, k;
    };

    using oct_ref_t = OctantIndex; //! value type for the hashmap
    using oct_map_t = Kokkos::UnorderedMap<key_t, oct_ref_t>; //! hashmap returning an octant form a key

protected:
    Storage_t storage;
    
    oct_map_t oct_map; //! hashmap returning an octant form a key

    level_t min_level; //! Coarser level of the octree
    level_t max_level; //! Finer level of the octree
    Kokkos::Array<bool, 3> is_periodic;
    Kokkos::View<morton_t*> morton_intervals;
};

} //namespace dyablo
