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
 * The log loss aggregator. This is a binary logistic regression aggregator.
 */
class LogLossAggregator extends LossAggregator {
  private var weightSum: Double = 0.0
  private var weightedLabelSum: Double = 0.0
  private var weightedLossSum: Double = 0.0
  private var weightedProbSum: Double = 0.0
  private var weightedProbSquareSum: Double = 0.0

  private val eps = 1e-15

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
    val prob = math.min(math.max(1.0 / (1.0 + math.exp(-curPred)), eps), 1.0 - eps)
    val logLoss = -(label * math.log(prob) + (1.0 - label) * math.log(1.0 - prob))
    weightSum += weight
    weightedLabelSum += weight * label
    weightedLossSum += weight * logLoss
    weightedProbSum += weight * prob
    weightedProbSquareSum += weight * prob * prob
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
    val prob = math.min(math.max(1.0 / (1.0 + math.exp(-curPred)), eps), 1.0 - eps)
    label - prob
  }

  /**
   * Merge the aggregated values with another aggregator.
   * @param b The other aggregator to merge with.
   * @return This.
   */
  def mergeInPlace(b: LossAggregator): this.type = {
    this.weightSum += b.asInstanceOf[LogLossAggregator].weightSum
    this.weightedLabelSum += b.asInstanceOf[LogLossAggregator].weightedLabelSum
    this.weightedLossSum += b.asInstanceOf[LogLossAggregator].weightedLossSum
    this.weightedProbSum += b.asInstanceOf[LogLossAggregator].weightedProbSum
    this.weightedProbSquareSum += b.asInstanceOf[LogLossAggregator].weightedProbSquareSum
    this
  }

  /**
   * Using the aggregated values, compute deviance.
   * @return Deviance.
   */
  def computeDeviance(): Double = {
    weightedLossSum / weightSum
  }

  /**
   * Using the aggregated values, compute the initial value.
   * @return Inital value.
   */
  def computeInitialValue(): Double = {
    RobustMath.log(weightedLabelSum / (weightSum - weightedLabelSum))
  }

  /**
   * Using the aggregated values for a particular node, compute the estimated
   * node value.
   * @return Node estimate.
   */
  def computeNodeEstimate(): Double = {
    // For the log loss, the node estimate is approximated as one Newton-Raphson
    // method step's result, as the optimal value will be either negative or
    // positive infinities.
    (weightedLabelSum - weightedProbSum) / (weightedProbSum - weightedProbSquareSum)
  }
}
