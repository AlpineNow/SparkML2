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

package spark_ml.sequoia_forest

import scala.collection.mutable

/**
 * A trait for node splits to be used in executors to filter samples to matching child nodes.
 * The statistics for child nodes are then individually aggregated in the executors.
 */
trait NodeSplitOnBinId extends Serializable {
  val parentNodeId: Int
  val featureId: Int
  def selectChildNode(featureBinId: Int): Int
  def getOrderedChildNodeIds: Array[Int]
  def getChildNodeWeight(childNodeId: Int): Double

  def getSubTreeHash(childNodeId: Int): Int // -1 means no sub tree.
  def setSubTreeHash(childNodeId: Int, subTreeHash: Int): Unit
}

/**
 * To be used on the very first iteration to get statistics for the root node.
 */
class RootGetter extends NodeSplitOnBinId {
  val parentNodeId = 0
  val featureId = 0
  def selectChildNode(featureBinId: Int): Int = 1 // Always select root, assuming that sample's the node Id is 0.
  def getOrderedChildNodeIds: Array[Int] = {
    Array[Int](1)
  }

  def getChildNodeWeight(childNodeId: Int): Double = 0.0

  def getSubTreeHash(childNodeId: Int): Int = -1
  def setSubTreeHash(childNodeId: Int, subTreeHash: Int): Unit = {}
}

/**
 * To be used on nodes that are split on numeric features.
 * @param parentNodeId The parent node Id.
 * @param featureId The index of the feature to split on.
 * @param splitBinId The split point (bin Id of the split).
 * @param leftId The left child node Id.
 * @param rightId The right child node Id.
 * @param leftWeight The weight (number of samples) on the left child.
 * @param rightWeight The weight (number of samples) on the right child.
 */
case class NumericSplitOnBinId(
    parentNodeId: Int,
    featureId: Int,
    splitBinId: Int,
    leftId: Int,
    rightId: Int,
    leftWeight: Double,
    rightWeight: Double,
    var leftSubTreeHash: Int = -1,
    var rightSubTreeHash: Int = -1) extends NodeSplitOnBinId {
  def selectChildNode(featureBinId: Int): Int = {
    if (featureBinId >= splitBinId) {
      rightId
    } else {
      leftId
    }
  }

  def getOrderedChildNodeIds: Array[Int] = {
    Array[Int](leftId, rightId)
  }

  def getChildNodeWeight(childNodeId: Int): Double = {
    if (leftId == childNodeId) leftWeight else if (rightId == childNodeId) rightWeight else 0.0
  }

  def getSubTreeHash(childNodeId: Int): Int = {
    if (childNodeId == leftId) leftSubTreeHash else if (childNodeId == rightId) rightSubTreeHash else -1
  }

  def setSubTreeHash(childNodeId: Int, subTreeHash: Int): Unit = {
    if (childNodeId == leftId) leftSubTreeHash = subTreeHash else if (childNodeId == rightId) rightSubTreeHash = subTreeHash
  }
}

/**
 * To be used on nodes that are split on categorical features.
 * @param parentNodeId The parent node Id.
 * @param featureId The index of the feature to split on.
 * @param binIdToNodeIdMap Mapping from bin Id to node Id.
 * @param nodeWeights Child node weights (sample counts).
 */
case class CategoricalSplitOnBinId(
    parentNodeId: Int,
    featureId: Int,
    binIdToNodeIdMap: mutable.Map[Int, Int],
    nodeWeights: mutable.Map[Int, Double],
    nodeSubTreeHash: mutable.Map[Int, Int] = mutable.Map[Int, Int]()) extends NodeSplitOnBinId {
  def selectChildNode(featureBinId: Int): Int = {
    binIdToNodeIdMap(featureBinId)
  }

  def getOrderedChildNodeIds: Array[Int] = {
    binIdToNodeIdMap.values.toArray.sorted
  }

  def getChildNodeWeight(childNodeId: Int): Double = {
    nodeWeights(childNodeId)
  }

  def getSubTreeHash(childNodeId: Int): Int = {
    nodeSubTreeHash.getOrElse(childNodeId, -1)
  }

  def setSubTreeHash(childNodeId: Int, subTreeHash: Int): Unit = {
    nodeSubTreeHash.put(childNodeId, subTreeHash)
  }
}

/**
 * A helper object (e.g. to incrementally rotate through an array collection.)
 * @param value The number.
 * @param maxVal The maximum value, incrementing past this rotates the number back to 0.
 */
case class RotatingNaturalNumber(var value: Int, maxVal: Int) {
  def increment(): Unit = {
    value += 1
    if (value > maxVal) {
      value = 0
    }
  }
}

/**
 * A fast look up object to quickly select the node split for a given parent node.
 * This object is constructed in the driver and then passed on to the executors.
 * This is used by the executors to quickly select a child node to aggregate statistics for.
 */
case class ScheduledNodeSplitLookup(
    parentNodeLookup: Array[Array[Int]],
    nodeSplitTable: Array[mutable.ArrayBuffer[NodeSplitOnBinId]],
    numTrees: Int,
    nodeSplitCount: Int,
    subTreeCount: Int = 0) {

  /**
   * Get the node split for the given tree and parent node Id.
   * It may return null if the parent node is not among those scheduled for splits.
   * @param treeId The tree that the node belongs to.
   * @param parentNodeId The node that we want to get the split of.
   * @return The matching splitter if it's found among the scheduled ones. Otherwise, null.
   */
  def getNodeSplit(treeId: Int, parentNodeId: Int): NodeSplitOnBinId = {
    val parentNodeIdStart = parentNodeLookup(treeId)(0)
    val parentNodeIdEnd = parentNodeLookup(treeId)(0) + parentNodeLookup(treeId)(1)
    if (parentNodeIdStart <= parentNodeId && parentNodeId < parentNodeIdEnd) {
      val nodeSplitIdx = parentNodeId - parentNodeIdStart
      nodeSplitTable(treeId)(nodeSplitIdx)
    } else {
      null
    }
  }
}

object ScheduledNodeSplitLookup {
  /**
   * Create a fast look up object to quickly select the node split for a given parent node.
   * This object is constructed in the driver and then passed on to the executors.
   * This is used by the executors to quickly select a child node to aggregate statistics for.
   * @param nodeSplitsPerTree The queue of currently scheduled node splits. The parent IDs are always in increasing order.
   * @param maxSplits The maximum number of nodes that we want to split in one iteration.
   * @return look up object.
   */
  def createLookupForNodeSplits(nodeSplitsPerTree: Array[mutable.Queue[NodeSplitOnBinId]], maxSplits: Int): ScheduledNodeSplitLookup = {
    var numNodeSplitsInQueues = nodeSplitsPerTree.foldLeft(0)((curSum, queue) => curSum + queue.size)

    var nodeSplitCount: Int = 0
    val numTrees = nodeSplitsPerTree.length

    // To enable fast lookups for matching node splits for a sample, we use arrays.
    // parentNodeLookup is used to find out the index into nodeSplitTable.
    val parentNodeLookup = Array.fill[Array[Int]](numTrees)(Array[Int](-1, -1))
    val nodeSplitTable = Array.fill[mutable.ArrayBuffer[NodeSplitOnBinId]](numTrees)(new mutable.ArrayBuffer[NodeSplitOnBinId]())

    // Construct the lookup tables using the constructor arguments.
    var treeId = RotatingNaturalNumber(0, numTrees - 1)
    while (nodeSplitCount <= maxSplits && numNodeSplitsInQueues > 0) {
      // Skip to the non-empty tree queue.
      while (nodeSplitsPerTree(treeId.value).length == 0) {
        treeId.increment()
      }

      val nodeSplit = nodeSplitsPerTree(treeId.value).dequeue()
      nodeSplitCount += 1

      val parentNodeId = nodeSplit.parentNodeId
      if (parentNodeLookup(treeId.value)(0) == -1) {
        parentNodeLookup(treeId.value)(0) = parentNodeId
        parentNodeLookup(treeId.value)(1) = 1
        nodeSplitTable(treeId.value) += nodeSplit
      } else {
        val expectedParentId = parentNodeLookup(treeId.value)(0) + parentNodeLookup(treeId.value)(1)

        // The current node split's parent Id can't be smaller than the expected one.
        if (parentNodeId < expectedParentId) {
          throw new AssertionError("The node Ids of splits should be in an increasing order.")
        }

        // We'll just add null entries for skipped parent node Ids.
        val numNodesToSkip = parentNodeId - expectedParentId
        (1 to numNodesToSkip).foreach(_ => {
          parentNodeLookup(treeId.value)(1) += 1
          nodeSplitTable(treeId.value) += null
        })

        parentNodeLookup(treeId.value)(1) += 1
        nodeSplitTable(treeId.value) += nodeSplit
      }

      numNodeSplitsInQueues = nodeSplitsPerTree.foldLeft(0)((curSum, queue) => curSum + queue.size)
      treeId.increment()
    }

    ScheduledNodeSplitLookup(parentNodeLookup, nodeSplitTable, numTrees, nodeSplitCount)
  }

  /**
   * Create a fast look up object to quickly select the node split for a given parent node.
   * This object is constructed in the driver and then passed on to the executors.
   * This is used by the executors to quickly select a child node that we want to train as a sub-tree by shuffling matching data rows to an executor.
   * @param nodeSplitsPerTree The queue of currently scheduled node splits. The parent IDs are always in increasing order.
   * @param maxSubTrees The maximum number of sub trees we want to train with this schedule object.
   * @return look up object.
   */
  def createLookupForSubTreeTraining(nodeSplitsPerTree: Array[mutable.Queue[NodeSplitOnBinId]], maxSubTrees: Int): ScheduledNodeSplitLookup = {
    var numNodeSplitsInQueues = nodeSplitsPerTree.foldLeft(0)((curSum, queue) => curSum + queue.size)

    var subTreeCount: Int = 0
    var nodeSplitCount: Int = 0
    val numTrees = nodeSplitsPerTree.length

    // To enable fast lookups for matching node splits for a sample, we use arrays.
    // parentNodeLookup is used to find out the index into nodeSplitTable.
    val parentNodeLookup = Array.fill[Array[Int]](numTrees)(Array[Int](-1, -1))
    val nodeSplitTable = Array.fill[mutable.ArrayBuffer[NodeSplitOnBinId]](numTrees)(new mutable.ArrayBuffer[NodeSplitOnBinId]())

    // Construct the lookup tables using the constructor arguments.
    var treeId = RotatingNaturalNumber(0, numTrees - 1)
    while (subTreeCount < maxSubTrees && numNodeSplitsInQueues > 0) {
      // Skip to the non-empty tree queue.
      while (nodeSplitsPerTree(treeId.value).length == 0) {
        treeId.increment()
      }

      val nodeSplit = nodeSplitsPerTree(treeId.value).dequeue()
      nodeSplitCount += 1

      val childNodeIds = nodeSplit.getOrderedChildNodeIds
      var childNodeIdx = 0
      while (childNodeIdx < childNodeIds.length) {
        val childNodeId = childNodeIds(childNodeIdx)
        if (nodeSplit.getSubTreeHash(childNodeId) >= 0) {
          nodeSplit.setSubTreeHash(childNodeId, subTreeCount)
          subTreeCount += 1
        }

        childNodeIdx += 1
      }

      val parentNodeId = nodeSplit.parentNodeId
      if (parentNodeLookup(treeId.value)(0) == -1) {
        parentNodeLookup(treeId.value)(0) = parentNodeId
        parentNodeLookup(treeId.value)(1) = 1
        nodeSplitTable(treeId.value) += nodeSplit
      } else {
        val expectedParentId = parentNodeLookup(treeId.value)(0) + parentNodeLookup(treeId.value)(1)

        // The current node split's parent Id can't be smaller than the expected one.
        if (parentNodeId < expectedParentId) {
          throw new AssertionError("The node Ids of splits should be in an increasing order.")
        }

        // We'll just add null entries for skipped parent node Ids.
        val numNodesToSkip = parentNodeId - expectedParentId
        (1 to numNodesToSkip).foreach(_ => {
          parentNodeLookup(treeId.value)(1) += 1
          nodeSplitTable(treeId.value) += null
        })

        parentNodeLookup(treeId.value)(1) += 1
        nodeSplitTable(treeId.value) += nodeSplit
      }

      numNodeSplitsInQueues = nodeSplitsPerTree.foldLeft(0)((curSum, queue) => curSum + queue.size)
      treeId.increment()
    }

    ScheduledNodeSplitLookup(parentNodeLookup, nodeSplitTable, numTrees, nodeSplitCount, subTreeCount)
  }
}