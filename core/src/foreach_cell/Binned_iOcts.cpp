#include "Binned_iOcts.hpp"

#include "amr/LightOctree.h"

namespace dyablo {

Kokkos::View<uint32_t*> get_bin( const Binned_iOcts& iOcts, uint32_t bin )
{
    DYABLO_ASSERT_HOST_DEBUG( bin+1 < iOcts.begin_v.size(), "Subset_levels::get_bin : bin out of bounds" );

    return Kokkos::subview( iOcts.iOcts, std::make_pair( iOcts.begin_v.at(bin), iOcts.begin_v.at(bin+1) ) );
}

struct Binned_iOcts_levels::Pdata
{
    const LightOctree& lmesh;
    int level_max;
    std::unique_ptr<Binned_iOcts> local_leaves;
    std::unique_ptr<Binned_iOcts> ghost_leaves;
    std::unique_ptr<Binned_iOcts> local_intermediates;
    std::unique_ptr<Binned_iOcts> ghost_intermediates;
};

namespace {

/// Helper function for get_iOcts_xxx
template< bool ghost, bool intermediate >
Kokkos::View<uint32_t*> get_iOcts( const Binned_iOcts_levels::Pdata& pdata, std::unique_ptr<Binned_iOcts>& binned_iOcts, uint32_t nbOcts, int level )
{
    if( !binned_iOcts )
    {
        auto& lmesh = pdata.lmesh;
        auto gen_bin = KOKKOS_LAMBDA( uint32_t iOct )
        {
            return lmesh.getLevel({iOct, ghost, intermediate});
        };
        binned_iOcts = std::make_unique<Binned_iOcts>( 
            bin_iOcts( pdata.level_max, nbOcts, gen_bin)
        );
    }
    return get_bin( *binned_iOcts, level );
}

} // namespace

Binned_iOcts_levels::Binned_iOcts_levels(const LightOctree& lmesh, int level_max)
    : pdata(std::make_unique<Binned_iOcts_levels::Pdata>(
                    Binned_iOcts_levels::Pdata{
                        .lmesh=lmesh, 
                        .level_max=level_max,
                    }))
{/*empty*/}

Binned_iOcts_levels::Binned_iOcts_levels(Binned_iOcts_levels&&) = default;

Binned_iOcts_levels::~Binned_iOcts_levels()
{/*empty*/}

Kokkos::View<uint32_t*> Binned_iOcts_levels::get_iOcts_leaves(int level)
{ 
    return get_iOcts<false, false>( *pdata, pdata->local_leaves, pdata->lmesh.getNumOctants(), level );
}

Kokkos::View<uint32_t*> Binned_iOcts_levels::get_iOcts_ghost_leaves(int level)
{   
    return get_iOcts<true, false>( *pdata, pdata->ghost_leaves, pdata->lmesh.getNumGhosts(), level );
}

Kokkos::View<uint32_t*> Binned_iOcts_levels::get_iOcts_intermediates(int level)
{
    return get_iOcts<false, true>( *pdata, pdata->local_intermediates, pdata->lmesh.getNumIntermediates(), level );
}

Kokkos::View<uint32_t*> Binned_iOcts_levels::get_iOcts_ghost_intermediates(int level)
{
    return get_iOcts<true, true>( *pdata, pdata->ghost_intermediates, pdata->lmesh.getNumIntermediateGhosts(), level );
}


} // namespace dyablo