#pragma once

#include "IOManager_base.h"

namespace dyablo {


class IOManager_hdf5;
class IOManager_checkpoint;
class IOManager_diagnostics;

} //namespace dyablo 


template<>
inline bool dyablo::IOManagerFactory::init()
{
  #ifdef DYABLO_USE_HDF5
  DECLARE_REGISTERED(dyablo::IOManager_hdf5);
  DECLARE_REGISTERED(dyablo::IOManager_checkpoint);
  DECLARE_REGISTERED(dyablo::IOManager_diagnostics);
  #endif
  return true;
}