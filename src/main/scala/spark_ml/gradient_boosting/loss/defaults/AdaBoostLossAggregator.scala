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

import spark_ml.gradient_boosting.loss.LossAggregator
import spark_ml.util.RobustMath

/**
 * The AdaBoost loss aggregator. This is used for binary classifications.
 * Gradient boosting's AdaBoost is an approximation of the original AdaBoosting
 * in that the exponential loss is reduced through gradient steps (original
 * AdaBoost has a different optimization routine).
 */
class AdaBoostLossAggregator extends LossAggregator {
  private var weightSum: Double = 0.0
  private var weightedLabelSum: Double = 0.0
  private var weightedLossSum: Double = 0.0
  private var weightedNumeratorSum: Double = 0.0

  // If we want to show log loss.
  // private var weightedLogLossSum: Double = 0.0

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
    val a = 2 * label - 1.0
    val sampleLoss = math.exp(-a * curPred)

    weightSum += weight
    weightedLabelSum += weight * label
    weightedLossSum += weight * sampleLoss
    weightedNumeratorSum += weight * a * sampleLoss

    // val prob = 1.0 / (1.0 + math.exp(-2.0 * curPred))
    // val logLoss = -(label * math.log(prob) + (1.0 - label) * math.log(1.0 - prob))
    // weightedLogLossSum += weight * logLoss
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
    val a = 2 * label - 1.0
    a * math.exp(-a * curPred)
  }

  /**
   * Merge the aggregated values with another aggregator.
   * @param b The other aggregator to merge with.
   * @return This.
   */
  def mergeInPlace(b: LossAggregator): this.type = {
    this.weightSum += b.asInstanceOf[AdaBoostLossAggregator].weightSum
    this.weightedLabelSum += b.asInstanceOf[AdaBoostLossAggregator].weightedLabelSum
    this.weightedLossSum += b.asInstanceOf[AdaBoostLossAggregator].weightedLossSum
    this.weightedNumeratorSum += b.asInstanceOf[AdaBoostLossAggregator].weightedNumeratorSum
    // this.weightedLogLossSum += b.asInstanceOf[AdaBoostLossAggregator].weightedLogLossSum
    this
  }

  /**
   * Using the aggregated values, compute deviance.
   * @return Deviance.
   */
  def computeDeviance(): Double = {
    weightedLossSum / weightSum
    // weightedLogLossSum / weightSum
  }

  /**
   * Using the aggregated values, compute the initial value.
   * @return Inital value.
   */
  def computeInitialValue(): Double = {
    RobustMath.log(weightedLabelSum / (weightSum - weightedLabelSum)) / 2.0
  }

  /**
   * Using the aggregated values for a particular node, compute the estimated
   * node value.
   * @return Node estimate.
   */
  def computeNodeEstimate(): Double = {
    // For the adaboost loss (exponential loss), the node estimate is
    // approximated as one Newton-Raphson method step's result, as the optimal
    // value will be either negative or positive infinities.
    weightedNumeratorSum / weightedLossSum
  }
}
