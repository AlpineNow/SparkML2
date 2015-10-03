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

class TruncHingeLossAggregator(maxProb: Double, eps: Double) extends LossAggregator {
  private var weightSum: Double = 0.0
  private var weightedLabelSum: Double = 0.0
  private var weightedLossSum: Double = 0.0

  private val minProb = 1.0 - maxProb
  private val maxLinearValue = RobustMath.log(maxProb / minProb)

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
    val positiveLoss = math.min(math.max(-(curPred - eps), 0.0), maxLinearValue)
    val negativeLoss = math.min(math.max(curPred + eps, 0.0), maxLinearValue)
    val loss = label * positiveLoss + (1.0 - label) * negativeLoss
    weightSum += weight
    weightedLabelSum += weight * label
    weightedLossSum += weight * loss
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
    val positiveLoss = math.max(-(curPred - eps), 0.0)
    val positiveLossGradient =
      if (positiveLoss == 0.0 || positiveLoss > maxLinearValue) {
        0.0
      } else {
        1.0
      }
    val negativeLoss = math.max(curPred + eps, 0.0)
    val negativeLossGradient =
      if (negativeLoss == 0.0 || negativeLoss > maxLinearValue) {
        0.0
      } else {
        -1.0
      }
    label * positiveLossGradient + (1.0 - label) * negativeLossGradient
  }

  /**
   * Merge the aggregated values with another aggregator.
   * @param b The other aggregator to merge with.
   * @return This.
   */
  def mergeInPlace(b: LossAggregator): this.type = {
    this.weightSum += b.asInstanceOf[TruncHingeLossAggregator].weightSum
    this.weightedLabelSum += b.asInstanceOf[TruncHingeLossAggregator].weightedLabelSum
    this.weightedLossSum += b.asInstanceOf[TruncHingeLossAggregator].weightedLossSum
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
  def computeNodeEstimate(): Double = ???
}
