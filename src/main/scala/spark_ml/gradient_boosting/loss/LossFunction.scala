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
 * Each loss function should implement this trait and pass it onto the gradient
 * boosting algorithm. The job is currently mainly to provide proper aggregators
 * for computing losses and gradients.
 */
trait LossFunction extends Serializable {
  /**
   * String name of the loss function.
   * @return The string name of the loss function.
   */
  def lossFunctionName: String

  /**
   * Create the loss aggregator for this loss function.
   * @return
   */
  def createAggregator: LossAggregator

  /**
   * If this loss function is used for categorical labels, this function returns
   * the expected label cardinality. E.g., loss functions like AdaBoost,
   * logistic losses are used for binary classification, so this should return
   * Some(2). For regressions like the Gaussian loss, this should return None.
   * @return either Some(cardinality) or None.
   */
  def getLabelCardinality: Option[Int]

  /**
   * Whether tree node estimate refinement is possible.
   * @return true if node estimates can be refined. false otherwise.
   */
  def canRefineNodeEstimate: Boolean

  /**
   * Convert the raw tree ensemble prediction into a usable form by applying
   * the mean function. E.g., this gives the actual regression prediction and/or
   * a probability of a class.
   * @param rawPred The raw tree ensemble prediction. E.g., this could be
   *                unbounded negative or positive numbers for
   *                AdaaBoost/LogLoss/PoissonLoss. We want to return bounded
   *                numbers or actual count estimate for those losses, for
   *                instance.
   * @return A mean function applied value.
   */
  def applyMeanFunction(rawPred: Double): Double
}
