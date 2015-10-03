/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package spark_ml.discretization

/**
 * Find bins that minimize the label variance. Essentially equivalent to
 * building a decision tree on a single feature with variance criteria.
 * @param maxSampleSize The maximum sample.
 * @param seed The seed to use with random sampling.
 */
class VarianceMinimizingBinFinderFromSample(maxSampleSize: Int, seed: Int)
  extends NumericBinFinderFromSample("LabelVariance", maxSampleSize, seed) {

  /**
   * Calculate the label variance of the segment.
   * @param labelValues A sample label values.
   * @param featureValues A sample feature values.
   * @param labelSummary This is useful if there's a need to find the label
   *                     class cardinality.
   * @param s The segment starting index (inclusive).
   * @param e The segment ending index (exclusive).
   * @return The loss of the segment.
   */
  override def findSegmentLoss(
    labelValues: Array[Double],
    featureValues: Array[Double],
    labelSummary: LabelSummary,
    s: Int,
    e: Int): Double = {
    val segSize = (e - s).toDouble
    var sum = 0.0
    var sqrSum = 0.0
    var i = s
    while (i < e) {
      val labelValue = labelValues(i).toInt
      sum += labelValue
      sqrSum += labelValue * labelValue
      i += 1
    }

    val avg = sum / segSize
    sqrSum / segSize - avg * avg
  }
}
