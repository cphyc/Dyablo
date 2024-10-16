set -ex

if [ $# -ge 1 ]
then
    Kokkos_ARCH=$1
else
    Kokkos_ARCH='PASCAL61'
fi

mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Release -DDYABLO_ENABLE_UNIT_TESTING=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH=${Kokkos_ARCH} ..
make -j `nproc`
