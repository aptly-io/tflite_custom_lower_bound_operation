/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <jni.h>
#include <utility>
#include <vector>

#include "libs/cc/predictor.h"
#include "tensorflow/lite/model.h"

const char kIllegalStateException[] = "java/lang/IllegalStateException";
const char kSmartReply[] = "org/tensorflow/lite/examples/smartreply/SmartReply";

using tflite::custom::smartreply::GetScores;

struct JNIStorage {
  std::unique_ptr<::tflite::FlatBufferModel> model;
};

template <typename T>
T CheckNotNull(JNIEnv* env, T&& t) {
  if (t == nullptr) {
    env->ThrowNew(env->FindClass(kIllegalStateException), "");
    return nullptr;
  }
  return std::forward<T>(t);
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_examples_smartreply_SmartReplyClient_loadJNI(
    JNIEnv* env, jobject thiz, jobject model_buffer) {
  const char* buf =
      static_cast<char*>(env->GetDirectBufferAddress(model_buffer));
  jlong capacity = env->GetDirectBufferCapacity(model_buffer);

  JNIStorage* storage = new JNIStorage;
  storage->model = tflite::FlatBufferModel::BuildFromBuffer(
      buf, static_cast<size_t>(capacity));

  if (!storage->model) {
    delete storage;
    env->ThrowNew(env->FindClass(kIllegalStateException), "");
    return 0;
  }
  return reinterpret_cast<jlong>(storage);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_org_tensorflow_lite_examples_smartreply_SmartReplyClient_predictJNI(
    JNIEnv* env, jobject /*thiz*/, jlong storage_ptr, jfloatArray averages) {
  if (storage_ptr == 0) {
    return nullptr;
  }
  JNIStorage* storage = reinterpret_cast<JNIStorage*>(storage_ptr);
  if (storage == nullptr) {
    return nullptr;
  }

  const jfloat* p = env->GetFloatArrayElements(averages, 0);
  const jsize count = env->GetArrayLength(averages);
  std::vector<float> average_scores;
  average_scores.reserve(count);
  for (int i = 0; i < count; ++i) {
    average_scores.emplace_back(*p++);
  }

  std::vector<float> scores;
  GetScores(average_scores, *storage->model, &scores);

  jfloatArray array = CheckNotNull(env, env->NewFloatArray(scores.size()));
  for (int i = 0; i < scores.size(); ++i) {
    env->SetFloatArrayRegion(array, i, 1, &scores[i]);
  }

  return array;
}

extern "C" JNIEXPORT void JNICALL
Java_org_tensorflow_lite_examples_smartreply_SmartReplyClient_unloadJNI(
    JNIEnv* env, jobject thiz, jlong storage_ptr) {
  if (storage_ptr != 0) {
    JNIStorage* storage = reinterpret_cast<JNIStorage*>(storage_ptr);
    delete storage;
  }
}
