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

#include "libs/cc/predictor.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/minimal_logging.h"

void RegisterSelectedOps(::tflite::MutableOpResolver* resolver);

namespace tflite {
namespace custom {
namespace smartreply {

void GetScores(
    const std::vector<float>& input,
    const ::tflite::FlatBufferModel& model, 
    std::vector<float>* scores) 
{
  std::unique_ptr<::tflite::Interpreter> interpreter;
  ::tflite::MutableOpResolver resolver;
  RegisterSelectedOps(&resolver);
  ::tflite::InterpreterBuilder(model, resolver)(&interpreter);

  if (!model.initialized()) {
    fprintf(stderr, "Failed to mmap model\n");
    return;
  }

  const std::vector<int> dims = { 3 };
  interpreter->ResizeInputTensor(0, dims); // note if not resizing, input_tensor->data.f will be a nullptr
  
  interpreter->AllocateTensors(); // note if not calling, Invoke() fails with a not ready error

  TfLiteTensor* input_tensor = interpreter->tensor(interpreter->inputs()[0]);
  for (int i = 0; i < input_tensor->dims->data[0]; ++i) {
    input_tensor->data.f[i] = input[i];
  }

  interpreter->Invoke();

  TfLiteTensor* output_tensor = interpreter->tensor(interpreter->outputs()[0]);

  for (int i = 0; i < output_tensor->dims->data[0]; i++) {
    scores->emplace_back(output_tensor->data.i32[i]);
  }
}

}  // namespace smartreply
}  // namespace custom
}  // namespace tflite
