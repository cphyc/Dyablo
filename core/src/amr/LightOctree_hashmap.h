#pragma once

#include <utility>

#include "morton_utils.h"
#include "kokkos_shared.h"
#include "amr/AMRmesh.h"
#include "amr/LightOctree_storage.h"
#include "Kokkos_UnorderedMap.hpp"
#include "utils/misc/Dyablo_assert.h"

#include "amr/LightOctree_base.h"

namespace dyablo { 



class LightOctree_hashmap : public LightOctree_base{
public:
    using Storage_t = LightOctree_storage<>;
    using LightOctree_base::OctantIndex;
    using LightOctree_base::pos_t;
    using morton_t = uint64_t;
    using logical_coord_t = uint32_t;
    using level_t = logical_coord_t;
    struct key_t //! key type for hashmap (morton+level)
    {
        logical_coord_t level, i, j, k;
    };

    LightOctree_hashmap() = default;
    LightOctree_hashmap(const LightOctree_hashmap& lmesh) = default;

    LightOctree_hashmap( Storage_t&& storage, 
                         uint8_t level_min, uint8_t level_max,
                         Kokkos::Array<bool,3> periodic )
    : storage( std::move(storage) ),
      oct_map(storage.getNumOctants()+storage.getNumGhosts()),
      oct_map_intermediate(storage.getNumOctants()+storage.getNumGhosts()),
      min_level(level_min), max_level(level_max),
      is_periodic(periodic)
    {
        private_init();
    }

    template < typename AMRmesh_t >
    LightOctree_hashmap( const AMRmesh_t* pmesh, uint8_t level_min, uint8_t level_max )
    : storage( *pmesh ),
      oct_map(pmesh->getNumOctants()+pmesh->getNumGhosts()),
      oct_map_intermediate(storage.getNumOctants()+storage.getNumGhosts()),
      min_level(level_min), max_level(level_max),
      is_periodic( {pmesh->getPeriodic(2*IX), pmesh->getPeriodic(2*IY), pmesh->getPeriodic(2*IZ)} ),
      morton_intervals( "morton_intervals", pmesh->getMpiComm().MPI_Comm_size()+1 )
    {    
        private_init();

        // TODO use logical coords directly or even morton_intervals from AMRmesh
        morton_t first_morton;
        {
            int ndim = getNdim();
            auto pos = pmesh->getCoordinates((uint32_t)0);
            index_t<3> logical_coords;
            uint32_t octant_count = std::pow(2, max_level );
            real_t octant_size = 1.0/octant_count;
            logical_coords[IX] = std::floor(pos[IX]/octant_size);
            logical_coords[IY] = std::floor(pos[IY]/octant_size);
            logical_coords[IZ] = (ndim-2)*std::floor(pos[IZ]/octant_size);

            first_morton = compute_morton_key( logical_coords );
        }
        auto morton_intervals_host = Kokkos::create_mirror_view( morton_intervals );
        pmesh->getMpiComm().MPI_Allgather( &first_morton, morton_intervals_host.data(), 1 );
        morton_intervals_host(morton_intervals_host.size()-1) = uint64_t(-1);
        Kokkos::deep_copy(morton_intervals, morton_intervals_host);

    }

    /**
     * DO NOT CALL THIS YOURSELF
     * this is here because KOKKOS_LAMBDAS cannot be declared in constructors or private methods
     **/
    void private_init()
    {
        const Storage_t& storage = this->storage;
        Storage_t& storage_intermediate = this->storage_intermediate;
        const oct_map_t& oct_map = this->oct_map;
        const oct_map_t& oct_map_intermediate = this->oct_map_intermediate;
        uint32_t nbOcts = storage.getNumOctants();
        uint32_t numOctants_tot = nbOcts + storage.getNumGhosts();
        const level_t min_multigrid_level = 0;
        // Put octants into hashmap on device
        Kokkos::parallel_for( "LightOctree_hashmap::hash",
                            Kokkos::RangePolicy<>(0, numOctants_tot),
                            KOKKOS_LAMBDA(uint32_t ioct_local)
        {   
            OctantIndex iOct = OctantIndex::iOctLocal_to_OctantIndex( ioct_local, nbOcts );

            auto logical_pos = storage.get_logical_coords( iOct );
            level_t level = storage.getLevel( iOct );
            key_t logical_coords;
            logical_coords.level = level;
            logical_coords.i = logical_pos[IX];
            logical_coords.j = logical_pos[IY];
            logical_coords.k = logical_pos[IZ];           

            [[maybe_unused]] oct_map_t::insert_result inserted = oct_map.insert( logical_coords, iOct );
            DYABLO_ASSERT_KOKKOS_DEBUG(inserted.success(), "oct_map::insert() failed");

            //for( level_t i_level = level-1; i_level >= min_level; i_level-- )
            for( level_t i_level = level; i_level-- > min_multigrid_level; )
            {
                logical_coords.level = i_level;
                logical_coords.i /= 2;
                logical_coords.j /= 2;
                logical_coords.k /= 2;
                oct_map_t::insert_result inserted = this->oct_map_intermediate.insert( logical_coords, OctantIndex{0, false, true} );
                if( inserted.existing() )
                    break;
                DYABLO_ASSERT_KOKKOS_DEBUG(inserted.success(), "oct_map::insert() failed");
            }
        });

        uint32_t nbIntermediate = 0;
        Kokkos::parallel_scan("LightOctree_hashmap::intermediate_numbering",oct_map_intermediate.capacity(), 
            KOKKOS_LAMBDA(uint32_t i, uint32_t& iOct, bool final)
        {
            if( oct_map_intermediate.valid_at(i) )
            {
                if( final )
                {
                    oct_map_intermediate.value_at(i).iOct = iOct;
                }
                iOct++;
            }
        }, nbIntermediate);
        //storage_intermediate = LightOctree_storage( storage.ndim, nbIntermediate, 0, min_level, storage.coarse_grid_size);
        storage_intermediate = LightOctree_storage( storage.ndim, nbIntermediate, 0, min_multigrid_level, storage.coarse_grid_size);
        auto& oct_data_intermediate = storage_intermediate.oct_data;
        using oct_data_field_t = Storage_t::oct_data_field_t;

        Kokkos::parallel_for( "LightOctree_hashmap::intermediate_storage", oct_map_intermediate.capacity(), 
            KOKKOS_LAMBDA(uint32_t i)
        {
            if( oct_map_intermediate.valid_at(i) )
            {
                uint32_t iOct = oct_map_intermediate.value_at(i).iOct;
                key_t logical_coords = oct_map_intermediate.key_at(i);

                oct_data_intermediate( iOct, oct_data_field_t::ICORNERX ) = logical_coords.i;
                oct_data_intermediate( iOct, oct_data_field_t::ICORNERY ) = logical_coords.j;
                oct_data_intermediate( iOct, oct_data_field_t::ICORNERZ ) = logical_coords.k;
                oct_data_intermediate( iOct, oct_data_field_t::ILEVEL) = logical_coords.level;
            }
        });
    }

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
    uint32_t getNumIntermediate() const
    {return storage_intermediate.getNumOctants();}
    
    KOKKOS_INLINE_FUNCTION
    pos_t getCenter(const OctantIndex& iOct)  const
    {return iOct.isIntermediate? storage_intermediate.getCenter(iOct) : storage.getCenter(iOct);}
   
    KOKKOS_INLINE_FUNCTION
    pos_t getCorner(const OctantIndex& iOct)  const
    {return iOct.isIntermediate? storage_intermediate.getCorner(iOct) : storage.getCorner(iOct);}

    KOKKOS_INLINE_FUNCTION
    pos_t getSize(const OctantIndex& iOct)  const
    {return iOct.isIntermediate? storage_intermediate.getSize(iOct) : storage.getSize(iOct);}
    
    KOKKOS_INLINE_FUNCTION
    uint8_t getLevel(const OctantIndex& iOct)  const
    {return iOct.isIntermediate? storage_intermediate.getLevel(iOct) : storage.getLevel(iOct);}
    
    KOKKOS_INLINE_FUNCTION
    bool getBound(const OctantIndex& iOct)  const
    {return iOct.isIntermediate? storage_intermediate.getBound(iOct) : storage.getBound(iOct);}

    KOKKOS_INLINE_FUNCTION
    uint32_t get_level_min()  const
    {return this->min_level;}

    KOKKOS_INLINE_FUNCTION
    uint32_t get_level_max()  const
    {return this->max_level;}

    const Storage_t getStorage() const 
    {
        return storage;
    }
    const Storage_t getStorageIntermediate() const 
    {
        return storage_intermediate;
    }


    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<logical_coord_t, 3> get_logical_coords(const OctantIndex& iOct)  const
    {
        return iOct.isIntermediate ? storage_intermediate.get_logical_coords(iOct) : storage.get_logical_coords(iOct);
    }

    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<logical_coord_t, 3> get_parent_logical_coords(const  Kokkos::Array<logical_coord_t, 3>& logical_coords)  const
    {
        return {logical_coords[IX] >> 1, logical_coords[IY] >> 1, logical_coords[IZ] >> 1 };
    }

    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<logical_coord_t, 3> get_child_logical_coords(const  Kokkos::Array<logical_coord_t, 3>& logical_coords, const offset_t& offset )  const
    {
        return {(logical_coords[IX] << 1) + offset[IX], (logical_coords[IY] << 1) + offset[IY], (logical_coords[IZ] << 1) + offset[IZ] };
    }
    
    
    //! @copydoc LightOctree_base::findNeighbors()
    KOKKOS_INLINE_FUNCTION NeighborList findNeighbors( const OctantIndex& iOct, const offset_t& offset )  const
    {
        int ndim = getNdim();

        if( offset[IX] == 0 && offset[IY] == 0 && offset[IZ] == 0 )
            return NeighborList{1,{iOct}};

        if( this->isBoundary(iOct, offset) )
            return NeighborList{0,{}};

        // Get logical coordinates of neighbor
        
        level_t level = this->getLevel(iOct);
        auto lc = this->get_logical_coords(iOct);
        logical_coord_t octant_count_x = storage.cell_count(IX, level );
        logical_coord_t octant_count_y = storage.cell_count(IY, level );
        logical_coord_t octant_count_z = storage.cell_count(IZ, level );
        key_t logical_coords;
        logical_coords.level = getLevel(iOct);
        logical_coords.i = (lc[IX] + octant_count_x + offset[IX]) % octant_count_x; // Periodic coord only works if offset > -octant_count
        logical_coords.j = (lc[IY] + octant_count_y + offset[IY]) % octant_count_y;
        logical_coords.k = (lc[IZ] + octant_count_z + offset[IZ]) % octant_count_z; 
  
        NeighborList res = {0};
        // Search octant at same level
        auto it = oct_map.find(logical_coords);
        if( oct_map.valid_at(it) )
        {
            // Found at same level
            res =  NeighborList{1, {oct_map.value_at(it)}};
        }
        else
        {
            key_t logical_coords_bigger;
            logical_coords_bigger.level = logical_coords.level-1;
            logical_coords_bigger.i = logical_coords.i >> 1;
            logical_coords_bigger.j = logical_coords.j >> 1;
            logical_coords_bigger.k = logical_coords.k >> 1;

            // Search octant at coarser level
            auto it = oct_map.find(logical_coords_bigger);
            if( oct_map.valid_at(it) ) 
            {
                // Found at coarser level
                res = NeighborList{1, {oct_map.value_at(it)}};
            }
            else
            {
                // Neighbor(s) is(are) at finer level
                DYABLO_ASSERT_KOKKOS_DEBUG(level+1 <= max_level, "Could not find neighbor : already at level_max");
                
                // Compute logical coord of first neighbor
                key_t logical_coords_smaller_origin;
                logical_coords_smaller_origin.level = logical_coords.level+1;
                logical_coords_smaller_origin.i = (logical_coords.i << 1) + (offset[IX]==-1);
                logical_coords_smaller_origin.j = (logical_coords.j << 1) + (offset[IY]==-1);
                logical_coords_smaller_origin.k = (logical_coords.k << 1) + (offset[IZ]==-1);
                int sz_max = (ndim==2) ? 0 : (offset[IZ]==0); // No offset in z in 2D
                int sy_max = (offset[IY]==0);
                int sx_max = (offset[IX]==0); // Constrained to plane adjacent to neighbor if offset in this direction
                
                for( int sz=0; sz<=sz_max; sz++ )
                for( int sy=0; sy<=sy_max; sy++ )
                for( int sx=0; sx<=sx_max; sx++ )
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
            }            
        }
        return res;
    }

    //! @copydoc LightOctree_base::findNeighbors()
    KOKKOS_INLINE_FUNCTION NeighborList findNeighbors_intermediate( const OctantIndex& iOct, const offset_t& offset )  const
    {
        int ndim = getNdim();

        if( offset[IX] == 0 && offset[IY] == 0 && offset[IZ] == 0 )
            return NeighborList{1,{iOct}};

        if( this->isBoundary(iOct, offset) )
            return NeighborList{0,{}};

        // Get logical coordinates of neighbor
        
        level_t level = this->getLevel(iOct);
        auto lc = this->get_logical_coords(iOct);
        logical_coord_t octant_count_x = storage_intermediate.cell_count(IX, level );
        logical_coord_t octant_count_y = storage_intermediate.cell_count(IY, level );
        logical_coord_t octant_count_z = storage_intermediate.cell_count(IZ, level );
        key_t logical_coords;
        logical_coords.level = getLevel(iOct);
        logical_coords.i = (lc[IX] + octant_count_x + offset[IX]) % octant_count_x; // Periodic coord only works if offset > -octant_count
        logical_coords.j = (lc[IY] + octant_count_y + offset[IY]) % octant_count_y;
        logical_coords.k = (lc[IZ] + octant_count_z + offset[IZ]) % octant_count_z; 

        NeighborList res = {0};
        // Search octant at same level
        auto it = oct_map_intermediate.find(logical_coords);
        if( oct_map_intermediate.valid_at(it) )
        {
            // Found at same level
            res =  NeighborList{1, {oct_map_intermediate.value_at(it)}};
        }
        else
        {
            key_t logical_coords_bigger;
            logical_coords_bigger.level = logical_coords.level-1;
            logical_coords_bigger.i = logical_coords.i >> 1;
            logical_coords_bigger.j = logical_coords.j >> 1;
            logical_coords_bigger.k = logical_coords.k >> 1;

            // Search octant at coarser level
            auto it = oct_map_intermediate.find(logical_coords_bigger);
            if( oct_map_intermediate.valid_at(it) ) 
            {
                // Found at coarser level
                res = NeighborList{1, {oct_map_intermediate.value_at(it)}};
            }
            else
            {
                // Neighbor(s) is(are) at finer level
                DYABLO_ASSERT_KOKKOS_DEBUG(level+1 <= max_level, "Could not find neighbor : already at level_max");
                
                // Compute logical coord of first neighbor
                key_t logical_coords_smaller_origin;
                logical_coords_smaller_origin.level = logical_coords.level+1;
                logical_coords_smaller_origin.i = (logical_coords.i << 1) + (offset[IX]==-1);
                logical_coords_smaller_origin.j = (logical_coords.j << 1) + (offset[IY]==-1);
                logical_coords_smaller_origin.k = (logical_coords.k << 1) + (offset[IZ]==-1);
                int sz_max = (ndim==2) ? 0 : (offset[IZ]==0); // No offset in z in 2D
                int sy_max = (offset[IY]==0);
                int sx_max = (offset[IX]==0); // Constrained to plane adjacent to neighbor if offset in this direction
                
                for( int sz=0; sz<=sz_max; sz++ )
                for( int sy=0; sy<=sy_max; sy++ )
                for( int sx=0; sx<=sx_max; sx++ )
                {
                    res.m_size++;
                    key_t logical_coords_smaller = logical_coords_smaller_origin;
                    logical_coords_smaller.i += sx;
                    logical_coords_smaller.j += sy;
                    logical_coords_smaller.k += sz;
                    auto it = oct_map_intermediate.find(logical_coords_smaller);
                    DYABLO_ASSERT_KOKKOS_DEBUG(oct_map_intermediate.valid_at(it), "Could not find neighbor");
                    res.m_neighbors[res.m_size-1] = oct_map_intermediate.value_at(it);
                }
                DYABLO_ASSERT_KOKKOS_DEBUG(res.m_size<=2*(ndim-1), "Too many neighbors");
            }            
        }
        return res;
    }
    
    KOKKOS_INLINE_FUNCTION
    OctantIndex findChild( const OctantIndex& iOct, const offset_t& offset )  const
    {
        DYABLO_ASSERT_KOKKOS_DEBUG( iOct.isIntermediate, "Leaves have no children" );

        auto lc = storage_intermediate.get_logical_coords(iOct);
        key_t logical_coords;
        logical_coords.level = storage_intermediate.getLevel(iOct)+1;
        logical_coords.i = 2*lc[IX] + offset[IX];
        logical_coords.j = 2*lc[IY] + offset[IY];
        logical_coords.k = 2*lc[IZ] + offset[IZ];

        auto it_l = oct_map.find(logical_coords);
        if( oct_map.valid_at(it_l) )
        {
            return oct_map.value_at(it_l);
        }
        else
        {
            auto it_i = oct_map_intermediate.find(logical_coords);
            DYABLO_ASSERT_KOKKOS_DEBUG( oct_map_intermediate.valid_at(it_i), "Could not find child octant." );
            return oct_map_intermediate.value_at( it_i );
        }
    }

    KOKKOS_INLINE_FUNCTION
    OctantIndex findParent( const OctantIndex& iOct )  const
    {
        auto lc = this->get_logical_coords(iOct);
        key_t logical_coords;
        logical_coords.level = this->getLevel(iOct)-1;
        logical_coords.i = (lc[IX] >> 1);
        logical_coords.j = (lc[IY] >> 1);
        logical_coords.k = (lc[IZ] >> 1);

        if (!iOct.isIntermediate){
            auto it_i = oct_map_intermediate.find(logical_coords);
            DYABLO_ASSERT_KOKKOS_DEBUG( oct_map_intermediate.valid_at(it_i), "Could not find intermediate parent octant from leaf cell." );
            return oct_map_intermediate.value_at(it_i);
        }

        auto it_l = oct_map.find(logical_coords);
        if( oct_map.valid_at(it_l) )
        {
            return oct_map.value_at(it_l);
        }
        else
        {
            auto it_i = oct_map_intermediate.find(logical_coords);
            DYABLO_ASSERT_KOKKOS_DEBUG( oct_map_intermediate.valid_at(it_i), "Could not find Parent octant." );
            return oct_map_intermediate.value_at( it_i );
        }
    }
    
    /// @copydoc LightOctree_base::isBoundary()
    KOKKOS_INLINE_FUNCTION
    bool isBoundary(const OctantIndex& iOct, const offset_t& offset) const {
      auto dh = this->getSize(iOct);
      pos_t center = this->getCenter(iOct);    
      pos_t pos {
          center[IX] + offset[IX]*dh[IX],
          center[IY] + offset[IY]*dh[IY],
          center[IZ] + offset[IZ]*dh[IZ]
      };
  
      //       Not periodic   and     not inside domain
      // in at least one dimension
      return (!this->is_periodic[IX] && !( 0<pos[IX] && pos[IX]<1 ))
          || (!this->is_periodic[IY] && !( 0<pos[IY] && pos[IY]<1 ))
          || (!this->is_periodic[IZ] && !( 0<=pos[IZ] && pos[IZ]<1 )) ;            
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



private:
    Storage_t storage;
    Storage_t storage_intermediate;

    using oct_ref_t = OctantIndex; //! value type for the hashmap
    using oct_map_t = Kokkos::UnorderedMap<key_t, oct_ref_t>; //! hashmap returning an octant form a key
    oct_map_t oct_map; //! hashmap returning an octant form a key
    oct_map_t oct_map_intermediate; //! hashmap returning an octant form a key

    level_t min_level; //! Coarser level of the octree
    level_t max_level; //! Finer level of the octree
    Kokkos::Array<bool, 3> is_periodic;
    Kokkos::View<morton_t*> morton_intervals;
};

} //namespace dyablo
