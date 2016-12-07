/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.demo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Trace;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

/** A classifier specialized to label images using TensorFlow. */
public class TensorFlowImageClassifierV2 implements Classifier {
  static {
    System.loadLibrary("tensorflow_demo");
  }

  private static final String TAG = "TensorFlowImageClassifier";

  // Only return this many results with at least this confidence.
  private static final int MAX_RESULTS = 5;
  private static final float THRESHOLD = 0.1f;

  // Config values.
  private String inputName = "tf_train_dataset";
  final private int batchSize = 32;
  final private int inputSize = 32;

  // Pre-allocated buffers.
  private Vector<String> labels = new Vector<String>();
  private int[] intValues;
  private float[] floatValues;
  private float[] outputs;
  private String[] outputNames = new String[]{"p_digit_length",
          "p_digit_0", "p_digit_1", "p_digit_2", "p_digit_3", "p_digit_4"};

  float[] outputs_digit_length = new float[10];
  float[] outputs_digit_0 = new float[11];
  float[] outputs_digit_1 = new float[11];
  float[] outputs_digit_2 = new float[11];
  float[] outputs_digit_3 = new float[11];
  float[] outputs_digit_4 = new float[11];

  private TensorFlowInferenceInterface inferenceInterface;

  /**
   * Initializes a native TensorFlow session for classifying images.
   *
   * @param assetManager The asset manager to be used to load assets.
   * @return The native return value, 0 indicating success.
   * @throws IOException
   */
  public int initializeTensorFlow(
      AssetManager assetManager
  ) throws IOException {
    // Pre-allocate buffers.
    intValues = new int[inputSize * inputSize];
    floatValues = new float[inputSize * inputSize * 3];
    outputs = new float[11];

    outputs_digit_length = new float[11];
    outputs_digit_0 = new float[11];
    outputs_digit_1 = new float[11];
    outputs_digit_2 = new float[11];
    outputs_digit_3 = new float[11];
    outputs_digit_4 = new float[11];

    inferenceInterface = new TensorFlowInferenceInterface();
//    String fileName = "model_2.pb";
//    String fileName = "model_3_use_blank.pb";
//    String fileName = "model_4_use_blank_add_margin1.pb";
    String fileName = "model_5_preserve_ratio.pb";
//    String fileName = "model_6_large_image.pb";
    return inferenceInterface.initializeTensorFlow(assetManager, "file:///android_asset/" + fileName);
  }

  @Override
  public List<Recognition> recognizeImage(final Bitmap bitmap) {
    // Log this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("preprocessBitmap");
    // Preprocess the image data from 0-255 int to normalized float based
    // on the provided parameters.
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    for (int i = 0; i < intValues.length; ++i) {
      final int val = intValues[i];
      floatValues[i * 3 + 0] = ((val >> 16) & 0xFF);
      floatValues[i * 3 + 1] = ((val >> 8) & 0xFF);
      floatValues[i * 3 + 2] = (val & 0xFF);
    }
    Trace.endSection();

    // Copy the input data into TensorFlow.
    Trace.beginSection("fillNodeFloat");
    float[] newInput = new float[floatValues.length * batchSize];
    System.arraycopy(floatValues, 0, newInput, 0, floatValues.length);
    inferenceInterface.fillNodeFloat(
        inputName, new int[] {batchSize, inputSize, inputSize, 3}, newInput);
    Trace.endSection();

    // Run th
    // e inference call.
    Trace.beginSection("runInference");
    inferenceInterface.runInference(outputNames);
    Trace.endSection();

    // Copy the output Tensor back into the output array.
    Trace.beginSection("readNodeFloat");
    inferenceInterface.readNodeFloat("p_digit_0", outputs);

    inferenceInterface.readNodeFloat("p_digit_length", outputs_digit_length);
    inferenceInterface.readNodeFloat("p_digit_0", outputs_digit_0);
    inferenceInterface.readNodeFloat("p_digit_1", outputs_digit_1);
    inferenceInterface.readNodeFloat("p_digit_2", outputs_digit_2);
    inferenceInterface.readNodeFloat("p_digit_3", outputs_digit_3);
    inferenceInterface.readNodeFloat("p_digit_4", outputs_digit_4);
    Trace.endSection();

    // Find the best classifications.
    PriorityQueue<Recognition> pq = new PriorityQueue<Recognition>(3,
        new Comparator<Recognition>() {
          @Override
          public int compare(Recognition lhs, Recognition rhs) {
            // Intentionally reversed to put high confidence at the head of the queue.
            return Float.compare(rhs.getConfidence(), lhs.getConfidence());
          }
        });
    for (int i = 0; i < outputs.length; ++i) {
      if (outputs[i] > THRESHOLD) {
        pq.add(new Recognition(
            "" + i, "" + i, outputs[i], null, -1));
      }
    }

    PriorityQueue<Recognition> pqDigitLength = generatePQ("len", outputs_digit_length);
    PriorityQueue<Recognition> pqDigit0 = generatePQ("1th", outputs_digit_0);
    PriorityQueue<Recognition> pqDigit1 = generatePQ("2th", outputs_digit_1);
    PriorityQueue<Recognition> pqDigit2 = generatePQ("3th", outputs_digit_2);
    PriorityQueue<Recognition> pqDigit3 = generatePQ("4th", outputs_digit_3);
    PriorityQueue<Recognition> pqDigit4 = generatePQ("5th", outputs_digit_4);

    final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
//    int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
//    for (int i = 0; i < recognitionsSize; ++i) {
//      recognitions.add(pq.poll());
//    }
    Recognition digitLength = pqDigitLength.poll();
    Recognition digit0 = pqDigit0.poll();
    Recognition digit1 = pqDigit1.poll();
    Recognition digit2 = pqDigit2.poll();
    Recognition digit3 = pqDigit3.poll();
    Recognition digit4 = pqDigit4.poll();

    StringBuilder sb = new StringBuilder();
    sb.append("full [");
    if (digitLength.getIndex() > 0) {
      sb.append(digit0.getIndex() + "");
    }
    if (digitLength.getIndex() > 1) {
      sb.append(digit1.getIndex() + "");
    }
    if (digitLength.getIndex() > 2) {
      sb.append(digit2.getIndex() + "");
    }
    if (digitLength.getIndex() > 3) {
      sb.append(digit3.getIndex() + "");
    }
    if (digitLength.getIndex() > 4) {
      sb.append(digit4.getIndex() + "");
    }
    sb.append("]");
    Recognition digit = new Recognition("id_full", sb.toString(), 0.0f, null, 0);

    recognitions.add(digitLength);
    recognitions.add(digit);
    recognitions.add(digit0);
    recognitions.add(digit1);
    recognitions.add(digit2);
    recognitions.add(digit3);
    recognitions.add(digit4);

    Trace.endSection(); // "recognizeImage"
    return recognitions;
  }

  private PriorityQueue<Recognition>  generatePQ(String name, float[] outputs) {
    PriorityQueue<Recognition> pq = new PriorityQueue<Recognition>(3,
            new Comparator<Recognition>() {
              @Override
              public int compare(Recognition lhs, Recognition rhs) {
                // Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
              }
            });
    for (int i = 0; i < outputs.length; ++i) {
      if (outputs[i] > THRESHOLD) {
        pq.add(new Recognition(
                name + i, name + " [" + i +  "]", outputs[i], null, i));
      }
    }
    return pq;
  }

  @Override
  public void close() {
    inferenceInterface.close();
  }
}
