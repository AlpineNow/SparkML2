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

import spark_ml.gradient_boosting.loss.LossFunction

/**
 * The AdaBoost loss function. This is used for binary classifications.
 * Gradient boosting's AdaBoost is an approximation of the original AdaBoosting
 * in that the exponential loss is reduced through gradient steps (original
 * AdaBoost has a different optimization routine).
 */
class AdaBoostLossFunction extends LossFunction {
  private val eps = 1e-15

  def lossFunctionName = "AdaBoost(Exponential)"
  def createAggregator = new AdaBoostLossAggregator
  def getLabelCardinality: Option[Int] = Some(2)
  def canRefineNodeEstimate: Boolean = true

  def applyMeanFunction(rawPred: Double): Double = {
    math.min(math.max(1.0 / (1.0 + math.exp(-2.0 * rawPred)), eps), 1.0 - eps)
  }
}
