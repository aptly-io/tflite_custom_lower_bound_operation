Based on Google's example for custom operations.
See here https://github.com/tensorflow/examples/tree/master/lite/examples/smart_reply

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