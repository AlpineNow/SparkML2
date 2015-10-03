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

import spark_ml.gradient_boosting.loss.{LossAggregator, LossFunction}
import spark_ml.tree_ensembles._
import spark_ml.util.{DiscretizedFeatureHandler, MapWithSequentialIntKeys}
import spire.implicits._

/**
 * Used internally to store tree data while GB is training. This is not the
 * final trained model.
 */
class GBInternalTree extends Serializable {
  val nodes = mutable.Map[Int, NodeInfo]()
  val nodeAggregators = new MapWithSequentialIntKeys[LossAggregator](
    initCapacity = 256
  )

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

  /**
   * Initialize aggregators for existing nodes. The aggregators are used to
   * aggregate samples per node to update node predictions, instead of using
   * predictions from CART training.
   */
  def initNodeFinetuners(lossFunction: LossFunction): Unit = {
    val (startNodeId, endNodeId) = (nodes.keys.min, nodes.keys.max)
    // Sanity checks.
    assert(
      startNodeId == 1,
      "The starting root node must always have the Id 1. But we have " + startNodeId
    )
    assert(
      nodes.size == (endNodeId - startNodeId + 1),
      "The number of nodes should equal " + (endNodeId - startNodeId + 1).toString +
      " but instead, we have " + nodes.size + " nodes."
    )

    cfor(startNodeId)(_ <= endNodeId, _ + 1)(
      nodeId => nodeAggregators.put(nodeId, lossFunction.createAggregator)
    )
  }

  /**
   * Add the given sample point to all the nodes that match the point.
   * @param samplePoint Sample point to add. This includes the label, the
   *                    current prediction and the features.
   * @param weight Weight of the sample point.
   * @param featureHandler Feature handler.
   * @tparam T Type of the discretized feature.
   */
  def addSamplePointToMatchingNodes[@specialized(Byte, Short) T](
    samplePoint: ((Double, Double), Array[T]),
    weight: Double,
    featureHandler: DiscretizedFeatureHandler[T]): Unit = {
    val ((label, curPred), features) = samplePoint
    // First add the point to the root node.
    nodeAggregators.get(1).addSamplePoint(
      label = label,
      weight = weight,
      curPred = curPred
    )
    var curNode = nodes(1)

    // Then add the point to all the matching descendants.
    while (curNode.splitInfo.nonEmpty) {
      val splitInfo = curNode.splitInfo.get
      val featId = splitInfo.featureId
      val binId = featureHandler.convertToInt(features(featId))
      val childId = splitInfo.chooseChildNode(binId).nodeId

      // We can't directly use the child node contained in splitInfo.
      // That child node is not the same as the one contained in nodes and
      // is incomplete.
      curNode = nodes(childId)
      val aggregator = nodeAggregators.get(childId)
      aggregator.addSamplePoint(
        label = label,
        weight = weight,
        curPred = curPred
      )
    }
  }

  /**
   * Update a node's prediction with the given one.
   * @param nodeId Id of the node whose prediction we want to update.
   * @param newPrediction The new prediction for the node.
   */
  def updateNodePrediction(nodeId: Int, newPrediction: Double): Unit = {
    nodes(nodeId).prediction = newPrediction
  }

  /**
   * Predict on the given features.
   * @param features Features.
   * @tparam T Type of features.
   * @return Prediction result.
   */
  def predict[@specialized(Byte, Short) T](
    features: Array[T],
    featureHandler: DiscretizedFeatureHandler[T]
  ): Double = {
    var curNode = nodes(1)
    while (curNode.splitInfo.nonEmpty) {
      val splitInfo = curNode.splitInfo.get
      val featId = splitInfo.featureId
      val binId = featureHandler.convertToInt(features(featId))

      // We can't directly use the child node contained in splitInfo.
      // That child node is not the same as the one contained in nodes and
      // is incomplete.
      val childId = splitInfo.chooseChildNode(binId).nodeId
      curNode = nodes(childId)
    }
    curNode.prediction
  }

  /**
   * Print a visual representation of the internal tree with ASCII.
   * @return A string representation of the internal tree.
   */
  override def toString: String = {
    val treeStringBuilder = new mutable.StringBuilder()
    val queue = new mutable.Queue[Int]()
    queue.enqueue(1)
    while (queue.nonEmpty) {
      val nodeInfo = nodes(queue.dequeue())
      treeStringBuilder.
        append("nodeId:").
        append(nodeInfo.nodeId).
        append(",prediction:").
        append(nodeInfo.prediction)
      if (nodeInfo.splitInfo.isDefined) {
        treeStringBuilder.
          append(",splitFeatureId:").
          append(nodeInfo.splitInfo.get.featureId)
        nodeInfo.splitInfo.get match {
          case numericNodeSplitInfo: NumericNodeSplitInfo =>
            treeStringBuilder.
              append(",splitBinId:").
              append(numericNodeSplitInfo.splitBinId)
            if (numericNodeSplitInfo.nanChildNode.isDefined) {
              treeStringBuilder.
                append(",hasNanChild")
            }
          case catNodeSplitInfo: CatNodeSplitInfo =>
            treeStringBuilder.
              append(",splitMapping:")
            catNodeSplitInfo.binIdToChildNode.foreach {
              case (binId, childNodeId) =>
                treeStringBuilder.
                  append(";").
                  append(binId.toString + "->" + childNodeId.toString)
            }
        }

        queue.enqueue(
          nodeInfo.splitInfo.get.getOrderedChildNodes.map(_.nodeId).toSeq : _*
        )
      }

      if (queue.nonEmpty && (nodes(queue.front).depth > nodeInfo.depth)) {
        treeStringBuilder.append("\n")
      } else {
        treeStringBuilder.append("  ")
      }
    }

    treeStringBuilder.toString()
  }
}

/**
 * Gradient boosted trees writer.
 * @param store The gradient boosted trees store object.
 */
class GradientBoostedTreesWriter(store: GradientBoostedTreesStore)
  extends TreeEnsembleWriter {
  var curTree: GBInternalTree = store.curTree

  /**
   * Write the node info to the currently active tree.
   * @param nodeInfo Node info to write.
   */
  def writeNodeInfo(nodeInfo: NodeInfo): Unit = {
    curTree.addNode(nodeInfo)
  }
}

/**
 * Gradient boosted trees store.
 * @param lossFunction Loss function with which the gradient boosted trees are
 *                     trained.
 * @param initVal Initial prediction value for the model.
 * @param shrinkage Shrinkage value.
 */
class GradientBoostedTreesStore(
  val lossFunction: LossFunction,
  val initVal: Double,
  val shrinkage: Double
) extends TreeEnsembleStore {
  val trees: mutable.ArrayBuffer[GBInternalTree] = mutable.ArrayBuffer[GBInternalTree]()
  var curTree: GBInternalTree = null

  /**
   * Add a new tree. This tree becomes the new active tree.
   */
  def initNewTree(): Unit = {
    curTree = new GBInternalTree
    trees += curTree
  }

  /**
   * Get a tree ensemble writer.
   * @return A tree ensemble writer.
   */
  def getWriter: TreeEnsembleWriter = {
    new GradientBoostedTreesWriter(this)
  }

  /**
   * Get an internal tree.
   * @param idx Index of the tree to get.
   * @return An internal tree.
   */
  def getInternalTree(idx: Int): GBInternalTree = {
    trees(idx)
  }
}
