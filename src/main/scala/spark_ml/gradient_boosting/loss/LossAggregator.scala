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

package spark_ml.gradient_boosting.loss

/**
 * Trait for aggregators used for loss calculations in gradient boosting
 * algorithms. The gradient boosting algorithm supports pluggable aggregators.
 * The implementations for different loss functions determine the type of
 * statistics to aggregate.
 */
trait LossAggregator extends Serializable {
  /**
   * Add a sample to the aggregator.
   * @param label Label of the sample.
   * @param weight Weight of the sample.
   * @param curPred Current prediction
   *                (should be computed using available trees).
   */
  def addSamplePoint(
    label: Double,
    weight: Double,
    curPred: Double): Unit

  /**
   * Compute the gradient of the sample at the current prediction.
   * @param label Label of the sample.
   * @param curPred Current prediction
   *                (should be computed using available trees).
   */
  def computeGradient(
    label: Double,
    curPred: Double): Double

  /**
   * Merge the aggregated values with another aggregator.
   * @param b The other aggregator to merge with.
   * @return This.
   */
  def mergeInPlace(b: LossAggregator): this.type

  /**
   * Using the aggregated values, compute deviance.
   * @return Deviance.
   */
  def computeDeviance(): Double

  /**
   * Using the aggregated values, compute the initial value.
   * @return Inital value.
   */
  def computeInitialValue(): Double

  /**
   * Using the aggregated values for a particular node,
   * compute the estimated node value.
   * @return Node estimate.
   */
  def computeNodeEstimate(): Double
}
