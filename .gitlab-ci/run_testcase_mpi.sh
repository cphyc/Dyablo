set -xe

root=`pwd`

if [ $# -lt 2 ]
then
    echo "run_testcase.sh <.ini filename local to build/dyablo/bin> <MPI proc count>"
    exit -1
fi
TESTCASE=$1
nmpi=$2

# Share half of the available cores between MPI processes (at least 1 thread each)
nthreads=$(( `nproc` / 2 / nmpi ))
if [ $nthreads -lt 1 ]
then
    nthreads=1
fi

cd build/dyablo/bin
OMP_NUM_THREADS=$nthreads mpirun --oversubscribe -np $nmpi ./dyablo ${TESTCASE}
