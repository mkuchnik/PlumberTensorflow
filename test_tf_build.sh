#!/bin/bash
N_JOBS=96
BAZEL_BIN="./bazel"

function try_test() {
  "${BAZEL_BIN}" test \
    --config=opt \
    -c opt \
    -j $N_JOBS \
    //tensorflow/core/platform/...
    #//tensorflow:all
}

function try_test_tf_data_internal() {
  "${BAZEL_BIN}" test \
    --config=opt \
    -c opt \
    -j $N_JOBS \
    //tensorflow/core/kernels/data/...
}

function try_test_python() {
  "${BAZEL_BIN}" test \
    --config=opt \
    -c opt \
    -j $N_JOBS \
    --test_timeout=1200 \
    //tensorflow/python/...
}

function try_test_tf_data() {
  "${BAZEL_BIN}" test \
    --config=opt \
    -c opt \
    -j $N_JOBS \
    --test_timeout=1200 \
    //tensorflow/python/data/...
}

try_test
try_test_tf_data_internal
#try_test_python
try_test_tf_data
