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

class TruncHingeLossFunction extends LossFunction {
  private var maxProb = 0.0
  private var eps = 0.0
  private var minProb = 0.0
  private var maxLinearValue = 0.0

  // Default values.
  setMaxProb(0.9)
  setEps(0.12)

  def setMaxProb(maxp: Double): Unit = {
    maxProb = maxp
    minProb = 1.0 - maxProb
    maxLinearValue = math.log(maxProb / minProb)
  }

  def setEps(e: Double): Unit = {
    eps = e
  }

  def lossFunctionName = "TruncHingeLoss"
  def createAggregator = new TruncHingeLossAggregator(maxProb, eps)
  def getLabelCardinality: Option[Int] = Some(2)
  def canRefineNodeEstimate: Boolean = false

  def applyMeanFunction(rawPred: Double): Double = {
    math.min(math.max(1.0 / (1.0 + math.exp(-rawPred)), minProb), maxProb)
  }
}
