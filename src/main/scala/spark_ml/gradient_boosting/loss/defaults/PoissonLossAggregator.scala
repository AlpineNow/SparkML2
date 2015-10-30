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
 * Aggregator for Poisson Losses.
 */
class PoissonLossAggregator extends LossAggregator {
  private var weightSum: Double = 0.0
  private var weightedLabelPredMulSum = 0.0
  private var weightedExpPredSum: Double = 0.0
  private var weightedLabelSum: Double = 0.0

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
    val labelPredMul = label * curPred
    val expPred = RobustMath.exp(curPred)
    weightSum += weight
    weightedLabelPredMulSum += weight * labelPredMul
    weightedExpPredSum += weight * expPred
    weightedLabelSum += weight * label
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
    val expPred = RobustMath.exp(curPred)
    label - expPred
  }

  /**
   * Merge the aggregated values with another aggregator.
   * @param b The other aggregator to merge with.
   * @return This.
   */
  def mergeInPlace(b: LossAggregator): this.type = {
    this.weightSum += b.asInstanceOf[PoissonLossAggregator].weightSum
    this.weightedLabelSum += b.asInstanceOf[PoissonLossAggregator].weightedLabelSum
    this.weightedLabelPredMulSum += b.asInstanceOf[PoissonLossAggregator].weightedLabelPredMulSum
    this.weightedExpPredSum += b.asInstanceOf[PoissonLossAggregator].weightedExpPredSum
    this
  }

  /**
   * Using the aggregated values, compute deviance.
   * @return Deviance.
   */
  def computeDeviance(): Double = {
    -2.0 * (weightedLabelPredMulSum - weightedExpPredSum) / weightSum
  }

  /**
   * Using the aggregated values, compute the initial value.
   * @return Inital value.
   */
  def computeInitialValue(): Double = {
    RobustMath.log(weightedLabelSum / weightSum)
  }

  /**
   * Using the aggregated values for a particular node, compute the estimated
   * node value.
   * @return Node estimate.
   */
  def computeNodeEstimate(): Double = {
    RobustMath.log(weightedLabelSum / weightedExpPredSum)
  }
}
