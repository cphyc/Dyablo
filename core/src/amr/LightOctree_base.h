#pragma once

#include <Kokkos_Core.hpp>

#include "real_type.h"

namespace dyablo { 

/**
 * Interface for a simplified read-only octree that can be accessed to get octant information
 * (position, amr level, neighbors, ...)
 * 
 * Class containing types and methods to implement a LightOctree
 * LightOctree implementations implement this interface, but don't necessarily derive from it 
 * (this base class is not polymorphic!).
 **/
class LightOctree_base{
public:
    
    /// Index to an octant
    struct OctantIndex
    {
        uint32_t iOct; //! Octant index
        bool isGhost; //! Is this a MPI ghost octant?
        bool isIntermediate; //! Is this an intermediate octant ( not a leaf )
    };

    static constexpr OctantIndex NEIGHBOR_NOT_FOUND = {(uint32_t)(-1)};
    KOKKOS_INLINE_FUNCTION
    static bool neighbor_not_found( OctantIndex& i ) 
    {return i.iOct == NEIGHBOR_NOT_FOUND.iOct;}

    /// Physical cell position
    using pos_t = Kokkos::Array<real_t,3>;
    /// Relative position of a neighbor octant relative to local octant
    using offset_t = Kokkos::Array<int32_t,3>;
    using level_t = uint32_t;
    /// Container for 0-4 neighbor(s)
    struct NeighborList
    {
        uint32_t m_size;
        Kokkos::Array<OctantIndex,4> m_neighbors;
        /// Number of neighbors in container (0-4)
        KOKKOS_INLINE_FUNCTION uint32_t size() const
        {
            return m_size;
        }
        /// Get i-th neighbor index in container
        KOKKOS_INLINE_FUNCTION const OctantIndex& operator[](uint32_t i) const
        {
            return m_neighbors[i];
        }
    };
    /// Get local (MPI) octant count
    uint32_t getNumOctants() const;
    /// Get local (MPI) ghost octant count
    uint32_t getNumGhosts() const;
    //bool getBound(const OctantIndex& iOct)  const;
    /// Get physical position of Octant center
    pos_t getCenter(const OctantIndex& iOct)  const;
    /// Get physical position of octant corner (smallest position inside octant)
    pos_t getCorner(const OctantIndex& iOct)  const;
    /// Get physical size of octant in all dimensions (Octant is a cube)
    pos_t getSize(uint32_t level)  const;
    pos_t getSize(const OctantIndex& iOct)  const;
    /// Get amr level of octant
    level_t getLevel(const OctantIndex& iOct)  const;
    /**
     * Get neighbors of `iOct` at relative position `offset`
     * 
     * @param iOct local Octant index (cannot be a ghost octant)
     * @param offset Relative position of neighbor(s) to fetch.
     *               offset in each dimension is either -1, 0 or 1; {0,0,0} is invalid
     *               in 2D, third dimension is always 0
     *               ex : {-1,0,0} is left neighbor; {-1,-1,0} is lower-left edge(3D)/corner(2D) 
     *
     * @returns between 1 and 4 neighbors packed in a NeighborList
     * 
     * ex in 2D:
     * ```
     *   ___________ __________
     *  |           |     |    |
     *  |           |  14 | 15 |
     *  |    11     |-----+----|
     *  |           |  12 | 13 |
     *  |___________|_____|____|
     *  |     |     | 8|9 |    |
     *  |  2  |  3  | 6|7 | 10 |
     *  |-----+-----|-----+----|
     *  |  0  |  1  |  4  | 5  |
     *  |_____|_____|_____|____|
     * 
     * findNeighbors({8,true},{-1, 1, 0}) -> 11
     * findNeighbors({8,true},{-1, 0, 0}) -> 3
     * findNeighbors({8,true},{-1,-1, 0}) -> 3 (Note that same neighbor can be returned twice)
     * findNeighbors({3,true},{ 1, 0, 0}) -> {8,6}
     * findNeighbors({3,true},{ 1, 1, 0}) -> 12
     * 
     * findNeighbors({1,true},{ 1, 1, 0}) -> 6 (Note that there is only 1 smaller neighbor in corners)
     * ```
     *
     * @note findNeighbors(), always returns all neighbors in corner 
     * @note Requesting a neighbor outside the domain when octree is not periodic returns 
     *       an empty neighbor list (but you should use isBoundary() if you only want to test that)
     **/
    NeighborList findNeighbors( const OctantIndex& iOct, const offset_t& offset ) const;

    
    static constexpr bool ACCEPTS_GHOSTS_DEFAULT = true;
    /**
     * Same as findNeighbors but only returns first octant in list (with smallest morton)  
     * Note : works only if findNeighbors returns at least one octant (check for boundaries before)
     * @tparam accepts_ghosts : by default iOct must not point to a ghost octant. 
     *          If you override this, make sure that the neighbor that you are trying to fetch 
     *          is present in local rank ( the neighbor of a ghost might be missing )
     * @tparam assert_on_failure : when false return NEIGHBOR_NOT_FOUND on error instead of assert. 
     *          (But don't use this to test if neighbor is present, there might be false positives!)
     *          when true, trying to fetch a non-existing neighbor is undefined behavior
     */
    template< bool accepts_ghosts = ACCEPTS_GHOSTS_DEFAULT, bool assert_on_failure = true >
    NeighborList findNeighbor( const OctantIndex& iOct, const offset_t& offset ) const;

    /**
     * Find same-size neighbor including intermediate octants
     * 
     * @param iOct Octant index
     * @param offset Relative position of neighbor(s) to fetch.
     *               offset in each dimension is either -1, 0 or 1; {0,0,0} is invalid
     *               in 2D, third dimension is always 0
     *               ex : {-1,0,0} is left neighbor; {-1,-1,0} is lower-left edge(3D)/corner(2D) 
     * 
     * @tparam accepts_ghosts : by default iOct must not point to a ghost octant. 
     *          If you override this, make sure that the neighbor that you are trying to fetch 
     *          is present in local rank ( the neighbor of a ghost might be missing )
     * @tparam assert_on_failure : when false return NEIGHBOR_NOT_FOUND on error instead of assert. 
     *          (But don't use this to test if neighbor is present, there might be false positives!)
     *          when true, trying to fetch a non-existing neighbor is undefined behavior
     * 
     * @returns an Octant index pointing to the resuested same-size neighbor
     * Since intermediate octants are included only one same-size octant matches
     * If same-size octant doesn't exist (neighbor leaf is bigger), the bigger leaf is returned
     **/
    template< bool accepts_ghosts = ACCEPTS_GHOSTS_DEFAULT, bool assert_on_failure = true >
    OctantIndex findNeighbor_intermediate( const OctantIndex& iOct, const offset_t& offset )  const;

    /**
     * Find children of an intermediate octant
     * 
     * @param iOct octant of parent cell (must be intermediate)
     * @param offset offset in [0,1]^3 of suboctant according to first suboctant
     **/
    OctantIndex findChild( const OctantIndex& iOct, const offset_t& offset ) const;

    /// Is the given face of the given oct an external boundary ?
    bool isBoundary(const OctantIndex& iOct, const offset_t& offset) const;

};

} //namespace dyablo
