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

import scala.collection.mutable

import spark_ml.discretization.Bins
import spark_ml.model.{DecisionTree, DecisionTreeNode, DecisionTreeUtil}
import spark_ml.transformation.ColumnTransformer

/**
 * The trainer will store the trained model using the internal types defined
 * in this file. If a developer wants to return a customized GBT type as a
 * result, then he/she can implement this factory trait and pass it onto the
 * trainer.
 */
trait GradientBoostedTreesFactory {
  /**
   * The GB tree ensemble factory needs to know how to transform label and
   * features from potentially categorical/string values to enumerated numeric
   * values. E.g. the final model might use enumerated categorical values to
   * perform predictions.
   * @param labelTransformer Label transformer.
   * @param featureTransformers Feature transformers.
   */
  def setColumnTransformers(
    labelTransformer: ColumnTransformer,
    featureTransformers: Array[ColumnTransformer]
  ): Unit

  /**
   * The GB tree ensemble factory also needs to know label/feature names and
   * types.
   * @param labelName Label name.
   * @param labelIsCat Whether the label is categorical.
   * @param featureNames Feature names.
   * @param featureIsCat Whether the individual features are categorical.
   */
  def setColumnNamesAndTypes(
    labelName: String,
    labelIsCat: Boolean,
    featureNames: Array[String],
    featureIsCat: Array[Boolean]
  ): Unit

  /**
   * Set the optimal tree count, as determined through validations.
   * @param optimalTreeCnt The optimal tree count.
   */
  def setOptimalTreeCnt(optimalTreeCnt: Int): Unit

  /**
   * Set the training deviance history.
   * @param trainingDevianceHistory Training deviance history.
   */
  def setTrainingDevianceHistory(
    trainingDevianceHistory: mutable.ListBuffer[Double]
  ): Unit

  /**
   * Set the validation deviance history.
   * @param validationDevianceHistory Validation deviance history.
   */
  def setValidationDevianceHistory(
    validationDevianceHistory: mutable.ListBuffer[Double]
  ): Unit

  /**
   * Set the feature bins.
   * @param featureBins Feature bins.
   */
  def setFeatureBins(featureBins: Array[Bins]): Unit

  /**
   * Create a final model that incorporates all the transformations and
   * discretizations and can predict on the raw feature values.
   * @param store The store that contains internally trained models.
   * @return A final GBT model.
   */
  def createGradientBoostedTrees(
    store: GradientBoostedTreesStore
  ): GradientBoostedTrees
}

/**
 * A default implementation of the GBT factory.
 */
class GradientBoostedTreesFactoryDefault extends GradientBoostedTreesFactory {
  var labelTransformer: Option[ColumnTransformer] = None
  var featureTransformers: Option[Array[ColumnTransformer]] = None
  var labelName: Option[String] = None
  var labelIsCat: Option[Boolean] = None
  var featureNames: Option[Array[String]] = None
  var featureIsCat: Option[Array[Boolean]] = None
  var optimalTreeCnt: Option[Int] = None
  var trainingDevianceHistory: Option[mutable.ListBuffer[Double]] = None
  var validationDevianceHistory: Option[mutable.ListBuffer[Double]] = None
  var featureBins: Option[Array[Bins]] = None

  def setColumnTransformers(
    labelTransformer: ColumnTransformer,
    featureTransformers: Array[ColumnTransformer]
  ): Unit = {
    this.labelTransformer = Some(labelTransformer)
    this.featureTransformers = Some(featureTransformers)
  }

  def setColumnNamesAndTypes(
    labelName: String,
    labelIsCat: Boolean,
    featureNames: Array[String],
    featureIsCat: Array[Boolean]
  ): Unit = {
    this.labelName = Some(labelName)
    this.labelIsCat = Some(labelIsCat)
    this.featureNames = Some(featureNames)
    this.featureIsCat = Some(featureIsCat)
  }

  def setOptimalTreeCnt(optimalTreeCnt: Int): Unit = {
    this.optimalTreeCnt = Some(optimalTreeCnt)
  }

  def setTrainingDevianceHistory(
    trainingDevianceHistory: mutable.ListBuffer[Double]
  ): Unit = {
    this.trainingDevianceHistory = Some(trainingDevianceHistory)
  }

  def setValidationDevianceHistory(
    validationDevianceHistory: mutable.ListBuffer[Double]
  ): Unit = {
    this.validationDevianceHistory = Some(validationDevianceHistory)
  }

  def setFeatureBins(featureBins: Array[Bins]): Unit = {
    this.featureBins = Some(featureBins)
  }

  def createGradientBoostedTrees(store: GradientBoostedTreesStore): GradientBoostedTrees = {
    val numFeatures = featureNames.get.length
    val featureImportance = Array.fill[Double](numFeatures)(0.0)
    val decisionTrees = store.trees.map {
      case (internalTree) =>
        val treeNodes = mutable.Map[java.lang.Integer, DecisionTreeNode]()
        val _ =
          DecisionTreeUtil.createDecisionTreeNode(
            internalTree.nodes(1),
            internalTree.nodes,
            featureImportance,
            this.featureBins.get,
            treeNodes
          )
        DecisionTree(treeNodes.toMap, internalTree.nodes.size)
    }.toArray

    GradientBoostedTreesDefault(
      lossFunctionClassName = store.lossFunction.getClass.getCanonicalName,
      labelTransformer = this.labelTransformer.get,
      featureTransformers = this.featureTransformers.get,
      labelName = this.labelName.get,
      labelIsCat = this.labelIsCat.get,
      featureNames = this.featureNames.get,
      featureIsCat = this.featureIsCat.get,
      sortedVarImportance =
        scala.util.Sorting.stableSort(
          featureNames.get.zip(featureImportance.map(new java.lang.Double(_))).toSeq,
          // We want to sort in a descending importance order.
          (e1: (String, java.lang.Double), e2: (String, java.lang.Double)) => e1._2 > e2._2
        ),
      shrinkage = store.shrinkage,
      initValue = store.initVal,
      decisionTrees = decisionTrees,
      optimalTreeCnt = this.optimalTreeCnt.map(new java.lang.Integer(_)),
      trainingDevianceHistory =
        this.trainingDevianceHistory.get.map(new java.lang.Double(_)).toSeq,
      validationDevianceHistory =
        this.validationDevianceHistory.map(vdh => vdh.map(new java.lang.Double(_)).toSeq)
    )
  }
}
