package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "@org_tensorflow//tensorflow/lite:build_def.bzl",
    "gen_selected_ops",
    "tflite_copts",
    "tflite_jni_binary",
)
load("@org_tensorflow//tensorflow/lite/java:aar_with_jni.bzl", "aar_with_jni")

gen_selected_ops(
    name = "smartreply_ops",
    # adds the necessary operators to the OpResolver
    model = [
        # "//libs/cc/testdata:smartreply.tflite", 
        "//libs/cc/testdata:fm_search_scores.tflite"
        ]
)

cc_library(
    name = "custom_ops",
    srcs = [
        "ops/lower_bound.cc",
        ":smartreply_ops",
    ],
    copts = tflite_copts(),
    deps = [
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "predictor_lib",
    srcs = ["predictor.cc"],
    hdrs = ["predictor.h"],
    copts = tflite_copts(),
    deps = [
        ":custom_ops",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

# TODO(b/118895218): Make this test compatible with oss.
# tf_cc_test(
#     name = "predictor_test",
#     srcs = ["predictor_test.cc"],
#     data = [
#         "//libs/cc/testdata:smartreply.tflite",
#         "//libs/cc/testdata:smartreply_samples.tsv",
#     ],
#     tags = ["no_oss"],
#     deps = [
#         ":predictor_lib",
#         "@org_tensorflow//tensorflow/core:test",
#         "@org_tensorflow//tensorflow/lite:string_util",
#         "@org_tensorflow//tensorflow/lite/testing:util",
#         "@com_google_absl//absl/strings",
#         "@com_google_googletest//:gtest",
#     ],
# )

# cc_test(
#     name = "extract_feature_op_test",
#     size = "small",
#     srcs = ["ops/extract_feature_test.cc"],
#     tags = ["no_oss"],
#     deps = [
#         ":custom_ops",
#         "@org_tensorflow//tensorflow/lite:framework",
#         "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
#         "@org_tensorflow//tensorflow/lite/kernels:test_util",
#         "@com_google_googletest//:gtest",
#         "@farmhash_archive//:farmhash",
#     ],
# )

# cc_test(
#     name = "normalize_op_test",
#     size = "small",
#     srcs = ["ops/normalize_test.cc"],
#     tags = ["no_oss"],
#     deps = [
#         ":custom_ops",
#         "@org_tensorflow//tensorflow/lite:framework",
#         "@org_tensorflow//tensorflow/lite:string_util",
#         "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
#         "@org_tensorflow//tensorflow/lite/kernels:test_util",
#         "@com_google_googletest//:gtest",
#     ],
# )

# cc_test(
#     name = "predict_op_test",
#     size = "small",
#     srcs = ["ops/predict_test.cc"],
#     tags = ["no_oss"],
#     deps = [
#         ":custom_ops",
#         "@org_tensorflow//tensorflow/lite:framework",
#         "@org_tensorflow//tensorflow/lite:string_util",
#         "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
#         "@org_tensorflow//tensorflow/lite/kernels:test_util",
#         "@com_google_googletest//:gtest",
#     ],
# )

cc_library(
    name = "smartreply_jni_lib",
    srcs = [
        "smartreply_jni.cc",
    ],
    copts = tflite_copts(),
    linkopts = [
        "-lm",
        "-ldl",
    ],
    deps = [
        ":predictor_lib",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/java/jni",
    ],
    alwayslink = 1,
)

cc_library(
    name = "smartreply_runtime",
    srcs = ["libsmartreply_jni.so"],
    alwayslink = 1,
)

tflite_jni_binary(
    name = "libsmartreply_jni.so",
    deps = [
        ":smartreply_jni_lib",
    ],
)

android_library(
    name = "smartreply_jni",
    custom_package = "org.tensorflow.lite.examples.smartreply",
    manifest = "DummyManifest.xml",
    resource_files = [],
    deps = [
        ":smartreply_runtime",  # build_cleaner: skip
    ],
)

aar_with_jni(
    name = "smartreply_runtime_aar",
    android_library = ":smartreply_jni",
)
