# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "analysis_lib",
    srcs = ["analysis_lib.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "analysis_lib_testdata",
    srcs = [
        "testdata/stats.pb",
    ],
    visibility = ["//visibility:public"],
)

tf_py_test(
    name = "analysis_lib_test",
    srcs = ["analysis_lib_test.py"],
    data = [":analysis_lib_testdata"],
    deps = [
        ":analysis_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "analysis",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":analysis_lib",
    ],
)
