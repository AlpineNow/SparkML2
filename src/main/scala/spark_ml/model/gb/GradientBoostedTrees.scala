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

package spark_ml.model.gb

import spark_ml.gradient_boosting.loss.LossFunction
import spark_ml.model.DecisionTree
import spark_ml.transformation.ColumnTransformer

case class GradientBoostedTreesDefault(
  lossFunctionClassName: String,
  labelTransformer: ColumnTransformer,
  featureTransformers: Array[ColumnTransformer],
  labelName: String,
  labelIsCat: Boolean,
  featureNames: Array[String],
  featureIsCat: Array[Boolean],
  sortedVarImportance: Seq[(String, java.lang.Double)],
  shrinkage: Double,
  initValue: Double,
  decisionTrees: Array[DecisionTree],
  optimalTreeCnt: Option[java.lang.Integer],
  trainingDevianceHistory: Seq[java.lang.Double],
  validationDevianceHistory: Option[Seq[java.lang.Double]]) extends GradientBoostedTrees {

  def lossFunction: LossFunction = {
    Class.forName(lossFunctionClassName).newInstance().asInstanceOf[LossFunction]
  }

  def numTrees = decisionTrees.length

  /**
   * The prediction is done on raw features. Internally, the model should
   * transform them to proper forms before predicting.
   * @param rawFeatures Raw features.
   * @param useOptimalTreeCnt A flag to indicate that we want to use optimal
   *                          number of trees.
   * @return The predicted value.
   */
  def predict(rawFeatures: Seq[Any], useOptimalTreeCnt: Boolean): Double = {
    // Transform the raw features first.
    val transformedFeatures =
      featureTransformers.zip(rawFeatures).map {
        case (featTransformer, rawFeatVal) =>
          if (rawFeatVal == null) {
            featTransformer.transform(null)
          } else {
            featTransformer.transform(rawFeatVal.toString)
          }
      }
    val predictedValue =
      (
        if (useOptimalTreeCnt && optimalTreeCnt.isDefined) {
          decisionTrees.slice(0, optimalTreeCnt.get)
        } else {
          decisionTrees
        }
      ).foldLeft(initValue) {
        case (curPred, tree) =>
          curPred + shrinkage * tree.predict(transformedFeatures)
      }

    lossFunction.applyMeanFunction(predictedValue)
  }
}

/**
 * A public model for the Gradient boosted trees that have been trained should
 * extend this trait.
 */
trait GradientBoostedTrees extends Serializable {
  def lossFunction: LossFunction
  def initValue: Double
  def numTrees: Int
  def optimalTreeCnt: Option[java.lang.Integer]

  /**
   * Sorted variable importance.
   * @return Sorted variable importance.
   */
  def sortedVarImportance: Seq[(String, java.lang.Double)]

  /**
   * The prediction is done on raw features. Internally, the model should
   * transform them to proper forms before predicting.
   * @param rawFeatures Raw features.
   * @param useOptimalTreeCnt A flag to indicate that we want to use optimal
   *                          number of trees.
   * @return The predicted value.
   */
  def predict(rawFeatures: Seq[Any], useOptimalTreeCnt: Boolean): Double
}
