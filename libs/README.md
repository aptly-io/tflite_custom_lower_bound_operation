# Tensorflow lite custom operator #

Experiment with a custom operation (LowerBound),
to make searchsorted(side='left') work in a tensorflow lite model on android.

It's based on Google's example for custom operations.
See here https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply.

The difference of library size is substantial:
- the default tensorflow-lite-0.0.0-nightly.aar contains a jni/armeabi-v7a/libtensorflowlite_jni.so of 1.6Mb
- the custom build variant that only uses the operators effectively used in the model: 382K

To try, checkout this repo and tensorflow examples repo.
Then run the Bazel build and copy the result into the tensorflow example.

```bash
git clone https://github.com/aptly-io/tflite_custom_lower_bound_operation.git
cd tflite_custom_lower_bound_operation
# Build A JNI layer to load a tflite model,
# a c++ layer to call tensorflow lite,
# the custom operator implementation in a AAR
bazel build libs/cc:smartreply_runtime_aar
# Move the custom build tensorflow lite AAR into the Android app
cp bazel-bin//libs/cc/smartreply_runtime_aar.aar ../tensorflow_examples/lite/examples/smart_reply/android/app/libs/smartreply_runtime_aar.aar
# Move the model using searchsorted into the Android app assets
cp libs/cc/testdata/fm_search_scores.tflite ../tensorflow_examples/lite/examples/smart_reply/android/app/src/main/assets/
```

Just the relevant subset of libs/cc is used.
The android java-code is brutally changed like this:

```java
  @WorkerThread
  public synchronized SmartReply[] predict(String[] input) {
    if (storage != 0) {
      float[] input_floats = {2.0f, 4.0f, 9.0f};
      float[] values = predictJNI(storage, input_floats);
      // should return [1, 2, 2] for the model checked in model fm_search_scores.tflite
      Log.d(TAG, "values are back:" + values); 
      return new SmartReply[] {};
    } else {
      return new SmartReply[] {};
    }
  }
```

The TensorFlow lite model is generated as follows:

```python
# This example only works with # pip3 install tensorflow==1.14.0
import tensorflow as tf
print(tf.__version__)

sorted_sequence = [ 0.0, 3.0, 9.0, 9.0, 10.0 ]
imposter_table = tf.constant(sorted_sequence, shape=(5,))

output_scores = tf.placeholder(tf.int32, shape = (3, ), name = 'output_scores')
input_avg_scores = tf.placeholder(tf.float32, shape = (3, ), name = 'input_avg_scores')
scores = tf.searchsorted(imposter_table, input_avg_scores)
output_scores = scores

with tf.Session() as session:
    values = [ 2.0, 4.0, 9.0 ]
    print(session.run(scores, feed_dict = {input_avg_scores: values}))    # should return [[1 2 2]
    print(output_scores.name, input_avg_scores.name)

    converter = tf.lite.TFLiteConverter.from_session(session, [input_avg_scores], [output_scores])

    converter.allow_custom_ops = True

    tflite_model = converter.convert()
    open("fm_search_scores.tflite", "wb").write(tflite_model)
```