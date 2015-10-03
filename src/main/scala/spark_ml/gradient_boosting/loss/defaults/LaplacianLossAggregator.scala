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

package spark_ml.gradient_boosting.loss.defaults

import scala.util.Random

import spark_ml.gradient_boosting.loss.LossAggregator
import spark_ml.util.{Selection, ReservoirSample}

/**
 * Aggregator for Laplacian Losses.
 * It uses approximate medians for initial value estimates but that is not
 * likely to affect the performance drastically.
 */
class LaplacianLossAggregator extends LossAggregator {
  private val maxSample = 10000

  private var labelSample = new ReservoirSample(maxSample)
  private var labelPredDiffSample = new ReservoirSample(maxSample)

  private var weightSum: Double = 0.0
  private var weightedAbsLabelPredDiffSum: Double = 0.0

  // For reservoir sampling.
  @transient private var rng: Random = null

  /**
   * Add a sample to the aggregator.
   * @param label Label of the sample.
   * @param weight Weight of the sample.
   * @param curPred Current prediction (e.g., using available trees).
   */
  def addSamplePoint(
    label: Double,
    weight: Double,
    curPred: Double): Unit = {
    if (weight > 0.0) {
      if (this.rng == null) {
        this.rng = new Random()
      }

      this.labelSample.doReservoirSampling(label, rng)

      val labelPredDiff = label - curPred
      this.labelPredDiffSample.doReservoirSampling(labelPredDiff, rng)

      this.weightSum += weight
      this.weightedAbsLabelPredDiffSum += weight * math.abs(labelPredDiff)
    }
  }

  /**
   * Compute the gradient.
   * @param label Label of the sample.
   * @param curPred Current prediction (e.g., using available trees).
   * @return The gradient of the sample.
   */
  def computeGradient(
    label: Double,
    curPred: Double): Double = {
    math.signum(label - curPred)
  }

  /**
   * Merge the aggregated values with another aggregator.
   * @param b The other aggregator to merge with.
   * @return This.
   */
  def mergeInPlace(b: LossAggregator): this.type = {
    if (this.rng == null) {
      this.rng = new Random()
    }

    this.weightSum += b.asInstanceOf[LaplacianLossAggregator].weightSum
    this.weightedAbsLabelPredDiffSum += b.asInstanceOf[LaplacianLossAggregator].weightedAbsLabelPredDiffSum
    // Now merge samples (distributed reservoir sampling).
    this.labelSample = ReservoirSample.mergeReservoirSamples(
      this.labelSample,
      b.asInstanceOf[LaplacianLossAggregator].labelSample,
      maxSample,
      this.rng
    )
    this.labelPredDiffSample = ReservoirSample.mergeReservoirSamples(
      this.labelPredDiffSample,
      b.asInstanceOf[LaplacianLossAggregator].labelPredDiffSample,
      maxSample,
      this.rng
    )
    this
  }

  /**
   * Using the aggregated values, compute deviance.
   * @return Deviance.
   */
  def computeDeviance(): Double = {
    weightedAbsLabelPredDiffSum / weightSum
  }

  /**
   * Using the aggregated values, compute the initial value.
   * @return Inital value.
   */
  def computeInitialValue(): Double = {
    if (this.rng == null) {
      this.rng = new Random()
    }

    // Get the initial value by computing the label median.
    Selection.quickSelect(
      this.labelSample.sample,
      0,
      this.labelSample.numSamplePoints,
      this.labelSample.numSamplePoints / 2,
      rng = this.rng
    )
  }

  /**
   * Using the aggregated values for a particular node, compute the estimated
   * node value.
   * @return Node estimate.
   */
  def computeNodeEstimate(): Double = {
    if (this.rng == null) {
      this.rng = new Random()
    }

    // Get the node estimate by computing the label pred diff median.
    Selection.quickSelect(
      this.labelPredDiffSample.sample,
      0,
      this.labelPredDiffSample.numSamplePoints,
      this.labelPredDiffSample.numSamplePoints / 2,
      rng = this.rng
    )
  }
}
