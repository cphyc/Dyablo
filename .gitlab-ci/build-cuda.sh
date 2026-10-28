set -ex

# Kokkos GPU architecture (e.g. AMPERE86) : first argument, or else $KOKKOS_ARCH.
# When empty, Kokkos detects the GPU of the build machine.
Kokkos_ARCH=${1:-${KOKKOS_ARCH}}

mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Release -DDYABLO_ENABLE_UNIT_TESTING=ON -DKokkos_ENABLE_CUDA=ON ${Kokkos_ARCH:+-DKokkos_ARCH=${Kokkos_ARCH}} ..
cmake --build . -j `nproc`
