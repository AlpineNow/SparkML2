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

package spark_ml.model.rf

import scala.collection.mutable

import spark_ml.discretization.Bins
import spark_ml.model._
import spark_ml.tree_ensembles._
import spark_ml.util.MapWithSequentialIntKeys

/**
 * A forest is simply a collection of equal-weight trees.
 * @param trees Trees in the forest.
 * @param splitCriteriaStr Tree split criteria (e.g. infogain or variance).
 * @param sortedVarImportance Sorted variable importance.
 * @param sampleCounts The number of training samples per tree.
 */
case class RandomForest(
  trees: Array[DecisionTree],
  splitCriteriaStr: String,
  sortedVarImportance: Seq[(String, java.lang.Double)],
  sampleCounts: Array[Long]
) {
  /**
   * Predict from the features.
   * @param features A double array of features.
   * @return Prediction(s) and corresponding weight(s) (e.g. probabilities or
   *         variances of predictions, etc.)
   */
  def predict(features: Array[Double]): Array[(Double, Double)] = {
    SplitCriteria.withName(splitCriteriaStr) match {
      case SplitCriteria.Classification_InfoGain => predictClass(features)
      case SplitCriteria.Regression_Variance => predictRegression(features)
    }
  }

  /**
   * Predict the class ouput from the given features - features should be in the
   * same order as the ones that the tree trained on.
   * @param features Feature values.
   * @return Predictions and their probabilities an array of (Double, Double)
   */
  private def predictClass(features: Array[Double]): Array[(Double, Double)] = {
    val predictions = mutable.Map[Double, Double]() // Predicted label and its count.
    var treeId = 0
    while (treeId < trees.length) {
      val tree = trees(treeId)
      val prediction = tree.predict(features)
      predictions.getOrElseUpdate(prediction, 0)
      predictions(prediction) += 1.0

      treeId += 1
    }

    // Sort the predictions by the number of occurrences.
    // The first element has the highest number of occurrences.
    val sortedPredictions = predictions.toArray.sorted(
      Ordering.by[(Double, Double), Double](-_._2)
    )
    sortedPredictions.map(p => (p._1, p._2 / trees.length.toDouble))
  }

  /**
   * Predict a continuous output from the given features - features should be in
   * the same order as the ones that the tree trained on.
   * @param features Feature values.
   * @return Prediction and its variance (a single element array of (Double, Double))
   */
  private def predictRegression(features: Array[Double]): Array[(Double, Double)] = {
    var predictionSum = 0.0
    var predictionSqrSum = 0.0
    var treeId = 0
    while (treeId < trees.length) {
      val tree = trees(treeId)
      val prediction = tree.predict(features)
      predictionSum += prediction
      predictionSqrSum += prediction * prediction

      treeId += 1
    }

    val predAvg = predictionSum / trees.length.toDouble
    val predVar = predictionSqrSum / trees.length.toDouble - predAvg * predAvg
    Array[(Double, Double)]((predAvg, predVar))
  }
}

class RFInternalTree extends Serializable {
  val nodes = mutable.Map[Int, NodeInfo]()

  /**
   * Add a new node to the tree.
   * @param nodeInfo node to add.
   */
  def addNode(nodeInfo: NodeInfo): Unit = {
    // Sanity check !
    assert(
      !nodes.contains(nodeInfo.nodeId),
      "A tree node with the Id " + nodeInfo.nodeId + " already exists."
    )

    nodes.put(nodeInfo.nodeId, nodeInfo)
  }
}

class RandomForestWriter(store: RandomForestStore)
  extends TreeEnsembleWriter {
  def writeNodeInfo(nodeInfo: NodeInfo): Unit = {
    if (!store.trees.contains(nodeInfo.treeId)) {
      store.trees.put(nodeInfo.treeId, new RFInternalTree)
    }
    store.trees.get(nodeInfo.treeId).addNode(nodeInfo)
  }
}

/**
 * A default random forest store.
 * @param splitCriteria The split criteria for trees.
 */
class RandomForestStore(
  splitCriteria: SplitCriteria.SplitCriteria,
  featureNames: Array[String],
  featureBins: Array[Bins]
) extends TreeEnsembleStore {
  val trees = new MapWithSequentialIntKeys[RFInternalTree](
    initCapacity = 100
  )

  def getWriter: TreeEnsembleWriter = {
    new RandomForestWriter(this)
  }

  def createRandomForest: RandomForest = {
    val featureImportance = Array.fill[Double](featureBins.length)(0.0)
    val (startTreeId, endTreeId) = this.trees.getKeyRange
    val decisionTrees = (startTreeId to endTreeId).map {
      case treeId =>
        val internalTree = this.trees.get(treeId)
        val treeNodes = mutable.Map[java.lang.Integer, DecisionTreeNode]()
        val _ =
          DecisionTreeUtil.createDecisionTreeNode(
            internalTree.nodes(1),
            internalTree.nodes,
            featureImportance,
            this.featureBins,
            treeNodes
          )
        DecisionTree(treeNodes.toMap, internalTree.nodes.size)
    }.toArray

    RandomForest(
      trees = decisionTrees,
      splitCriteriaStr = splitCriteria.toString,
      sortedVarImportance =
        scala.util.Sorting.stableSort(
          featureNames.zip(featureImportance.map(new java.lang.Double(_))).toSeq,
          // We want to sort in a descending importance order.
          (e1: (String, java.lang.Double), e2: (String, java.lang.Double)) => e1._2 > e2._2
        ),
      sampleCounts = decisionTrees.map{ _.nodes(1).nodeWeight.toLong }
    )
  }
}
