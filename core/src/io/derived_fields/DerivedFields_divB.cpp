#include "DerivedFields_base.h"

#include "foreach_cell/ForeachCell_utils.h"
#include "user_data/FieldAccessor.h"

namespace dyablo {

/**
 * @brief Derived field plugin computing the divergence of the magnetic field
 */
class DerivedFields_divB : public DerivedFields
{
public:

  DerivedFields_divB(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
  : 
    foreach_cell(foreach_cell),
    ndim(configMap.getValue<int>("mesh", "ndim", 2))
  {}
  
  std::vector<std::string> get_fields_names() const {
    return {"divB"};
  }

  void compute_derived_fields( OutputArray &out, const UserData &U ) const {
    
    using FieldAccessor = UserData::FieldAccessor;
    using CellIndex = ForeachCell::CellIndex;
    using offset_t = CellIndex::offset_t;

    enum VarIndexIn {IBX, IBY, IBZ};
    enum VarIndexOut {IDIVB};

    FieldAccessor Uin = U.getAccessor({{"Bx", IBX},
                                       {"By", IBY},
                                       {"Bz", IBZ}});

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    int ndim = this->ndim;

    foreach_cell.foreach_cell("DerivedFields::divB",
                              Uin.getShape(),
                              CELL_LAMBDA(const CellIndex &iCell) 
    {
      const auto& lmesh = cellmetadata.getLightOctree();
      ForeachCell::SearchMode_neighbor search_neighbor( lmesh, ForeachCell::SearchMode_neighbor::CLOSEST );
      ForeachCell::SearchMode_neighbor search_neighbor_origin( lmesh, ForeachCell::SearchMode_neighbor::ORIGIN );

      // Centered field is only used when one side is a boundary
      const real_t Bx = Uin.at(iCell, IBX);
      const real_t By = Uin.at(iCell, IBY);
      real_t Bz = 0;
      if (ndim == 3) 
        Bz = Uin.at(iCell, IBZ);
      const real_t BC[3] = {Bx, By, Bz};

      // Size ratio when dealing with non-conformal interfaces
      constexpr real_t size_diff[3] = {0.75, 1.0, 1.5}; 

      auto get_dB_dh = [&](const ComponentIndex3D dir) {
        const VarIndex IB = (dir == IX ? IBX : (dir == IY ? IBY : IBZ));
        offset_t off_p = {};
        offset_t off_m = {};
        off_p[dir] = 1;
        off_m[dir] = -1;

        real_t Bm, Bp, size_fac;
        size_fac = 0.0;
        // Minus offset
        {
          auto iCell_m = iCell.getNeighbor(off_m, search_neighbor);
          int ldiff = iCell_m.level_diff();
          
          // Boundary -> forward derivative
          if (iCell_m.is_boundary())
            Bm = BC[dir];
          // Same size of bigger neighbor
          else if (ldiff >= 0) {
            Bm = Uin.at(iCell_m, IB);
            size_fac += size_diff[ldiff+1];
          }
          // Smaller neighbor
          else {
            Bm = 0.0;
            foreach_smaller_neighbor_scattered(ndim, iCell, off_m, lmesh, [&](const CellIndex iCell_smaller) 
            {
              Bm += Uin.at(iCell_smaller, IB);
            });
            size_fac += size_diff[ldiff+1];
            Bm *= (ndim == 2 ? 0.5 : 0.25);
          }
        }
        // Plus offset
        {
          auto iCell_p = iCell.getNeighbor(off_p, search_neighbor);
          int ldiff = iCell_p.level_diff();
          
          // Boundary -> backward derivative
          if (iCell_p.is_boundary())
            Bp = BC[dir];
          // Same size or bigger neighbor
          else if (ldiff >= 0) {
            Bp = Uin.at(iCell_p, IB);
            size_fac += size_diff[ldiff+1];
          }
          // Samller neighbor
          else {
            Bp = 0.0;
            foreach_smaller_neighbor_scattered(ndim, iCell, off_p, lmesh, [&](const CellIndex iCell_smaller) 
            {
              Bp += Uin.at(iCell_smaller, IB);
            });
            size_fac += size_diff[ldiff+1];
            Bp *= (ndim == 2 ? 0.5 : 0.25);
          }
        }

        auto dh = cellmetadata.getCellSize(iCell)[dir];
        const real_t dB_dh = (Bp - Bm) / (dh * size_fac); 
        return dB_dh;
      };
      const real_t divB = get_dB_dh(IX) + get_dB_dh(IY) + (ndim == 3 ? get_dB_dh(IZ) : 0.0);

      out.at(iCell, IDIVB) = divB;
    });
  }

private:
  ForeachCell& foreach_cell;
  int ndim;
 
};

} //namespace dyablo

FACTORY_REGISTER(dyablo::DerivedFieldsFactory, 
                 dyablo::DerivedFields_divB, 
                 "DerivedFields_divB");