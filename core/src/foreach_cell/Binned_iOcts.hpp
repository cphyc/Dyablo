#pragma once

#include <memory>

#include "utils/misc/Dyablo_assert.h"
#include "amr/LightOctree_forward.h"

namespace dyablo {

struct Binned_iOcts{
    Kokkos::View<uint32_t*> iOcts; /// All iOcts binned by level
    std::vector<uint32_t> begin_v; /// first index for bin `level` in iOcts;
};

/***
 * Sort iOcts to gather octants from the same bin together
 * @param bin_max number of bins (gen_bin must return value between 0 and bin_max)
 * @param nbOcts the number of octants to sort (gen_bin will be called from 0 to nbOcts)
 * @param gen_bin is a uint32_t -> uint32_t function that returns the bin number for a given iOct
 */
template<typename F>
Binned_iOcts bin_iOcts( int bin_max, int nbOcts, const F& gen_bin)
{
    int bin_count = bin_max + 1;
    Kokkos::View<uint32_t*> bin_count_device("level_count",bin_count);
    Kokkos::parallel_for( "bin_iOcts_count", nbOcts, 
        KOKKOS_LAMBDA( uint32_t iOct )
    {
        int bin = gen_bin(iOct);
        DYABLO_ASSERT_KOKKOS_DEBUG( 0 <= bin && bin <= bin_max, "bin_iOcts : gen_bin out of bounds" );

        Kokkos::atomic_inc( &bin_count_device(bin) );
    });

    auto bin_count_host = Kokkos::create_mirror_view( bin_count_device );
    Kokkos::deep_copy( bin_count_host, bin_count_device );

    Kokkos::View<uint32_t*> iOcts("binned_iOcts", nbOcts);
    std::vector<uint32_t> begin_v(bin_count+1);
    uint32_t bin_begin = 0;
    for( int bin = 0; bin <= bin_max; bin++ )
    {
        begin_v[bin] = bin_begin;
        Kokkos::parallel_scan( "bin_iOcts_sort", nbOcts,
            KOKKOS_LAMBDA( uint32_t iOct, int& i_local, bool final )
        {
            int oct_bin = gen_bin(iOct);
            if( oct_bin == bin )
            {
                if( final )
                   iOcts(bin_begin + i_local) = iOct; 
                i_local++;
            }
        });
        bin_begin += bin_count_host(bin);
    }
    begin_v[bin_max+1] = bin_begin;

    return Binned_iOcts{
        .iOcts = iOcts,
        .begin_v = begin_v,
    };
}

/// Get a subview from all iOcts in a given bin 
Kokkos::View<uint32_t*> get_bin( const Binned_iOcts& iOcts, uint32_t bin );

/**
 * Class to generate per-level octant lists
 * 
 * This precomputes octant lists for every level to quickly get 
 * per-level octant lists for multiple levels.
 **/
class Binned_iOcts_levels
{
public:
    Binned_iOcts_levels(const LightOctree& lmesh, int level_max);
    Binned_iOcts_levels( Binned_iOcts_levels&& );
    ~Binned_iOcts_levels();

    /// get list of non-ghost leave iOcts
    Kokkos::View<uint32_t*> get_iOcts_leaves(int level);
    /// get list of ghost leave iOcts
    Kokkos::View<uint32_t*> get_iOcts_ghost_leaves(int level);
    /// get list of non-ghost intermediate iOcts
    Kokkos::View<uint32_t*> get_iOcts_intermediates(int level);
    /// get list of ghost intermediate iOcts
    Kokkos::View<uint32_t*> get_iOcts_ghost_intermediates(int level);

//private:
public:
    struct Pdata;
    std::unique_ptr<Pdata> pdata;
};

} // namespace dyablo