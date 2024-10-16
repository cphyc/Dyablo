set -ex

mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Release -DDYABLO_ENABLE_UNIT_TESTING=ON ..
make -j `nproc`
