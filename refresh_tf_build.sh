#!/bin/bash
set -e
N_JOBS=96
BINARY_DIRECTORY="$(pwd)/my_bin_out"
BAZEL_BIN="./bazel"
TF_VERSION="2.7.0"

#--experimental_local_memory_estimate \

function try_build() {
  "${BAZEL_BIN}" build \
    --config=opt \
    -c opt \
    -j $N_JOBS \
    --verbose_failures \
    //tensorflow/tools/pip_package:build_pip_package
}

function try_build_debug() {
  "${BAZEL_BIN}" build \
    --config=opt \
    -c opt \
    --per_file_copt=//tensorflow/core/kernels/data/.*\.cc@-g,-O0 \
    --strip=never \
    -j $N_JOBS \
    --verbose_failures \
    //tensorflow/tools/pip_package:build_pip_package
}

function try_build_while() {
  i=0
  until [ "$i" -ge 10 ];
  do
    try_build && break
    i=$((i+1))
    echo "Build crashed"
    N_JOBS=$((N_JOBS-1))
  done
}

function update_golden() {
  ./bazel run tensorflow/tools/api/tests:api_compatibility_test -- --update_goldens=True
}

try_build
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
mkdir -p "${BINARY_DIRECTORY}"
python -m pip uninstall -y tensorflow
python -m pip install "/tmp/tensorflow_pkg/tensorflow-${TF_VERSION}-cp37-cp37m-linux_x86_64.whl"
cp "/tmp/tensorflow_pkg/tensorflow-${TF_VERSION}-cp37-cp37m-linux_x86_64.whl" "${BINARY_DIRECTORY}"
