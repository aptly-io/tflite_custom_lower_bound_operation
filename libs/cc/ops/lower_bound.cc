#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace ops {
namespace custom {

namespace lower_bound {

constexpr int kInputTensorHaystack = 0;
constexpr int kInputTensorNeedles = 1;
constexpr int kOutputTensor = 0;

// todo dimension issue? why are haystack, needles and indices having dims->size == 2 or dims->data (1, x) as shape?

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) 
{
    int num_inputs = NumInputs(node);
    TF_LITE_ENSURE(context, num_inputs == 2);
    TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

    const TfLiteTensor* haystack = &context->tensors[node->inputs->data[kInputTensorHaystack]];
    TF_LITE_ENSURE_EQ(context, haystack->type, kTfLiteFloat32); // limit to float-type input
    const TfLiteTensor* needles = &context->tensors[node->inputs->data[kInputTensorNeedles]];
    TF_LITE_ENSURE_EQ(context, needles->type, kTfLiteFloat32);

    // see todo on the dimension issue
    // TF_LITE_ENSURE_EQ(context, NumDimensions(haystack), 1); // limit to vector's
    TF_LITE_ENSURE_EQ(context, haystack->type, needles->type);

    const TfLiteIntArray* needles_dims = needles->dims;
    TfLiteIntArray* output_dims = TfLiteIntArrayCopy(needles_dims);
    TfLiteTensor* indices = &context->tensors[node->outputs->data[kOutputTensor]];
    return context->ResizeTensor(context, indices, output_dims); // return kTfLiteOk
}


template<typename T>
int binary_search_leftmost(int haystack_size, const T* haystack, T needle)
{
    int l = 0, r = haystack_size, m;
    while (l < r) {
        m = (l + r) / 2;
        if (haystack[m] < needle) {
            l = m + 1;
        } else {
            r = m;
        }
    }
    return l;
}


template<typename T>
void EvalImpl(const TfLiteTensor* haystack, const TfLiteTensor* needles, TfLiteTensor* indices)
{
    const T* haystack_data = haystack->data.f;
    const int haystack_size = haystack->dims->data[1]; // see todo on the dimension issue

    const T* needles_data = needles->data.f;
    const int needles_size = needles->dims->data[1]; // see todo on the dimension issue

    int* indices_data = indices->data.i32;

    for (int i = 0; i < needles_size; ++i) {
        indices_data[i] = binary_search_leftmost<T>(haystack_size, haystack_data, needles_data[i]);
    }
}


TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) 
{
    const TfLiteTensor* haystack = &context->tensors[node->inputs->data[kInputTensorHaystack]];
    const TfLiteTensor* needles = &context->tensors[node->inputs->data[kInputTensorNeedles]];
    TfLiteTensor* indices = &context->tensors[node->outputs->data[kOutputTensor]];

    switch (haystack->type) {
        case kTfLiteFloat32:
            EvalImpl<float>(haystack, needles, indices);
            break;
        default:
            context->ReportError(context, "Only float support ");
            break;
    }

    return kTfLiteOk;
}

}  // namespace lower_bound

TfLiteRegistration* Register_LOWER_BOUND() {
    static TfLiteRegistration r = {
        nullptr, nullptr, lower_bound::Prepare, lower_bound::Eval
    };
    return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
