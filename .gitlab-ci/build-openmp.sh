set -ex

mkdir build
cd build

cmake_args=()
if [[ -n "${DYABLO_CMAKE_ARGS:-}" ]]
then
    cmake_args+=("-DDYABLO_CMAKE_ARGS=${DYABLO_CMAKE_ARGS}")
fi

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DDYABLO_ENABLE_UNIT_TESTING=ON \
    "${cmake_args[@]}" \
    ..
cmake --build . -j `nproc`
