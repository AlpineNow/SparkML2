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

import spark_ml.discretization.{ CategoricalBins, Discretizer, Bins }
import scala.collection.mutable
import spark_ml.util.RandomSet
import scala.util.Random

/**
 * To be returned after computing node predictions and splits.
 * @param treeId The Id of the tree that the node belongs to.
 * @param nodeId The node Id.
 * @param prediction The node prediction.
 * @param weight The node weight.
 * @param impurity The node impurity.
 * @param splitImpurity The impurity after the node split.
 * @param nodeSplit The node split containing info about how it's split.
 */
case class TrainedNodeInfo(
  treeId: Int,
  nodeId: Int,
  prediction: Double,
  weight: Double,
  impurity: Double,
  splitImpurity: Option[Double],
  nodeSplit: Option[NodeSplitOnBinId])

/**
 * This should be used to create bin statistics.
 * Different bin stats array/builder should be used for different types of statistics.
 * E.g., classification and regression would use different statistics objects.
 */
trait BinStatisticsArrayBuilder {
  /**
   * Add numBins to the builder, and this will return the size of bins added afterward.
   * @param numBins Number of bins to add.
   * @return The size of the bins that were added.
   */
  def addBins(numBins: Int): Int

  /**
   * Call this, after adding all the bins, to create an actual BinStatistics array object.
   * @return BinStatisticsArray object
   */
  def createBinStatisticsArray: BinStatisticsArray
}

/**
 * Different bin stats array/builder should be used for different types of statistics.
 * E.g., classification and regression would use different statistics objects.
 */
trait BinStatisticsArray extends Serializable {
  /**
   * Add a sample's label and count to the bin statistics of a feature of a tree's node.
   * @param offset The offset into the feature within this object.
   * @param featBinId The bin ID that we want to add to.
   * @param label The label of the sample.
   * @param sampleCount The count of the sample (weight of the sample).
   */
  def add(offset: Int, featBinId: Int, label: Double, sampleCount: Int): Unit

  /**
   * Merge the stat values in this one with another one into this one.
   * @param another The other stat array object.
   */
  def mergeInPlace(another: BinStatisticsArray): BinStatisticsArray

  /**
   * Summarize a series of bins. I.e., get prediction, impurity and BinStatisticsArray that summarizes the series into single summarizer bin.
   * This should be used to compute prediction, impurity of a node.
   * @param offset Offset from which to do summarizing.
   * @param numBins Number of bins to summarize over.
   * @return Prediction, impurity, single summarizer bin, and weight sum.
   */
  def getSummaryOverBins(offset: Int, numBins: Int): (Double, Double, BinStatisticsArray, Long)

  /**
   * Compute split on a feature.
   * @param offset Offset to the feature.
   * @param numBins Number of bins in the feature.
   * @param summaryStatsArray One summary bin over all the bins.
   * @param weightSum Weight sum over all the bins.
   * @param isFeatureCategorical Whether this is a categorical feature.
   * @return split impurity, split bin ID and split weights.
   */
  def computeSplit(offset: Int, numBins: Int, summaryStatsArray: BinStatisticsArray, weightSum: Long, isFeatureCategorical: Boolean): (Double, Int, Array[Long])
}

/**
 * This abstract class represents the aggregated statistics.
 * This is where per bin partitions get aggregated.
 * Additionally, this provides the function to compute splits from the aggregated statistics.
 */
abstract class AggregatedStatistics(
    @transient private val nodeSplitLookup: ScheduledNodeSplitLookup,
    numBinsPerFeature: Array[Int],
    treeSeeds: Array[Int],
    mtry: Int,
    @transient private var binStatsArrayBuilder: BinStatisticsArrayBuilder) extends Serializable {

  // Features are randomly selected per node here.
  val numTrees = nodeSplitLookup.numTrees
  val numFeatures = numBinsPerFeature.length
  val numSelectedFeaturesPerNode = math.min(mtry, numFeatures)

  // Auxiliary structures to convert node Ids to array indices.
  // We calculate the node array indices by subtracting the starting node Ids from a given node Id.
  // The nodes should exist in this when a caller wants to add samples for a node.
  private[spark_ml] val startNodeIds = Array.fill[Int](numTrees)(0)

  // This is the object that will be created within the constructor.
  // This is the object that keeps track of actual statistics.
  var binStatsArray: BinStatisticsArray = null

  // This is the look up table to quickly find offsets within binStatsArray.
  private[spark_ml] val offsetLookup = new Array[Array[Array[Int]]](numTrees)

  // This is used to quickly find randomly selected features per node.
  private[spark_ml] val selectedFeaturesLookup = new Array[Array[Array[Int]]](numTrees)

  var numNodes = 0 // Number of nodes whose statistics this object will contain.

  // Constructor routine.
  {
    // To quickly find the offset into this one dimensional array, we use a 3 dimensional array.
    // There are three keys - tree Id, node Id and feature Id - each is used as array indices.
    // Node Id and feature Id both have to be converted to array indices through an auxiliary structures.
    val offsetLookupBuilder = Array.fill[mutable.ArrayBuilder[Array[Int]]](numTrees)(mutable.ArrayBuilder.make[Array[Int]]())

    // Keep track of selected features per node.
    // The values of the last dimension are the indices of selected features. The last dimension's array index can be used to access the last dimension of offsetLookup's.
    val selectedFeaturesLookupBuilder = Array.fill[mutable.ArrayBuilder[Array[Int]]](numTrees)(mutable.ArrayBuilder.make[Array[Int]]())

    var curOffset = 0 // Offset into binStatsArray object.
    var treeId = 0
    while (treeId < numTrees) {
      val nodeSplits = nodeSplitLookup.nodeSplitTable(treeId)
      var nodeSplitIdx = 0
      var prevNodeId = 0
      while (nodeSplitIdx < nodeSplits.length) {
        val nodeSplit = nodeSplits(nodeSplitIdx)
        if (nodeSplit != null) {
          val childNodeIds = nodeSplit.getOrderedChildNodeIds
          var childNodeIdx = 0
          while (childNodeIdx < childNodeIds.length) {
            val childNodeId = childNodeIds(childNodeIdx)
            if (nodeSplit.getSubTreeHash(childNodeId) == -1) {
              // We add this node to statistics only if this node is not trained as a sub-tree.
              // It's indicated by the node's sub-tree-hash value.
              // -1 hash value means that it's not trained as a sub-tree.
              if (startNodeIds(treeId) == 0) {
                startNodeIds(treeId) = childNodeId
              }

              var numSkips = childNodeId - prevNodeId - 1
              while (prevNodeId != 0 && numSkips > 0) {
                selectedFeaturesLookupBuilder(treeId) += Array[Int]()
                offsetLookupBuilder(treeId) += Array[Int]()
                numSkips -= 1
              }

              prevNodeId = childNodeId

              // Randomly choose features for this child node.
              val selectedFeatureIndices = RandomSet.nChooseK(numSelectedFeaturesPerNode, numFeatures, new Random(treeSeeds(treeId) + childNodeId))
              selectedFeaturesLookupBuilder(treeId) += selectedFeatureIndices
              numNodes += 1

              val featureOffsets = Array.fill[Int](numSelectedFeaturesPerNode)(0)
              offsetLookupBuilder(treeId) += featureOffsets

              var featIdx = 0
              while (featIdx < selectedFeatureIndices.length) {
                val featId = selectedFeatureIndices(featIdx)
                val numBins = numBinsPerFeature(featId)

                featureOffsets(featIdx) = curOffset
                curOffset += binStatsArrayBuilder.addBins(numBins)
                featIdx += 1
              }
            }

            childNodeIdx += 1
          }
        }

        nodeSplitIdx += 1
      }

      treeId += 1
    }

    // Now, get concrete objects from various builders.
    binStatsArray = binStatsArrayBuilder.createBinStatisticsArray

    treeId = 0
    while (treeId < numTrees) {
      offsetLookup(treeId) = offsetLookupBuilder(treeId).result()
      treeId += 1
    }

    treeId = 0
    while (treeId < numTrees) {
      selectedFeaturesLookup(treeId) = selectedFeaturesLookupBuilder(treeId).result()
      treeId += 1
    }

    // To trigger GC on this builder object.
    binStatsArrayBuilder = null
  }

  /**
   * Add a sample for a tree's node.
   * @param treeId The tree that we are adding this sample to.
   * @param nodeId The node of the tree.
   * @param sample The sample with unsigned bytes as features.
   */
  def addUnsignedByteSample(
    treeId: Int,
    nodeId: Int,
    sample: (Double, Array[Byte], Array[Byte])): Unit = {
    // Find the selected features.
    val nodeIdx = nodeId - startNodeIds(treeId)
    val selectedFeatures = selectedFeaturesLookup(treeId)(nodeIdx)
    val label = sample._1
    val sampleCount = sample._3(treeId).toInt

    // Add the bin stats for all the selected features.
    var featIdx = 0
    while (featIdx < selectedFeatures.length) {
      val featId = selectedFeatures(featIdx)
      val featBinId = Discretizer.readUnsignedByte(sample._2(featId))
      val offset = offsetLookup(treeId)(nodeIdx)(featIdx)
      binStatsArray.add(offset, featBinId, label, sampleCount)
      featIdx += 1
    }
  }

  /**
   * Add a sample for a tree's node.
   * @param treeId The tree that we are adding this sample to.
   * @param nodeId The node of the tree.
   * @param sample The sample with unsigned shorts as features.
   */
  def addUnsignedShortSample(
    treeId: Int,
    nodeId: Int,
    sample: (Double, Array[Short], Array[Byte])): Unit = {
    // Find the selected features.
    val nodeIdx = nodeId - startNodeIds(treeId)
    val selectedFeatures = selectedFeaturesLookup(treeId)(nodeIdx)
    val label = sample._1
    val sampleCount = sample._3(treeId).toInt

    // Add the bin stats for all the selected features.
    var featIdx = 0
    while (featIdx < selectedFeatures.length) {
      val featId = selectedFeatures(featIdx)
      val featBinId = Discretizer.readUnsignedShort(sample._2(featId))
      val offset = offsetLookup(treeId)(nodeIdx)(featIdx)
      binStatsArray.add(offset, featBinId, label, sampleCount)
      featIdx += 1
    }
  }

  /**
   * Use this to merge statistics from a different partition.
   * @param another Statistics from a different partition.
   * @return This
   */
  def mergeInPlace(another: AggregatedStatistics): this.type = {
    binStatsArray.mergeInPlace(another.binStatsArray)
    this
  }

  /**
   * Given the bin statistics contained in this object, compute node predictions and splits.
   * @param featureBins Need to pass in the bin information for each feature - e.g. to find out if the feature is categorical or not.
   * @param nextNodeIdsPerTree The node Ids to assign to new child nodes from splits.
   * @param nodeDepths The depths of the currently being-trained nodes.
   * @param options Options to refer to.
   * @return An iterator of trained node info objects. The trainer will use this to build trees.
   */
  def computeNodePredictionsAndSplits(
    featureBins: Array[Bins],
    nextNodeIdsPerTree: Array[Int],
    nodeDepths: Array[mutable.Map[Int, Int]],
    options: SequoiaForestOptions): Iterator[TrainedNodeInfo] = {
    val output = new mutable.ArrayBuffer[TrainedNodeInfo]()
    val depthLimit = options.maxDepth match {
      case x if x == -1 => Int.MaxValue
      case x => x
    }

    var treeIdx = 0
    while (treeIdx < numTrees) {
      var nodeIdx = 0
      while (nodeIdx < offsetLookup(treeIdx).length) {
        if (offsetLookup(treeIdx)(nodeIdx).length > 0) {
          // Compute the prediction, impurity and summary from the first feature.
          // This should be the same for every other feature.
          val (prediction, impurity, summaryStatsArray, weightSum) = binStatsArray.getSummaryOverBins(
            offsetLookup(treeIdx)(nodeIdx)(0),
            numBinsPerFeature(selectedFeaturesLookup(treeIdx)(nodeIdx)(0)))

          // Quantities that we have to compute for this node to determine best split for this node.
          var splitImpurity: Option[Double] = None
          var splitFeatId: Option[Int] = None
          var splitBinId: Option[Int] = None
          var splitWeights: Option[Array[Long]] = None

          val nodeId = startNodeIds(treeIdx) + nodeIdx
          val nodeDepth = nodeDepths(treeIdx)(nodeId)

          nodeDepths(treeIdx).remove(nodeId) // We don't need this any more.

          // If the impurity is greater than 0 and the node weight is greater than split limit,
          // then look for possible feature splits.
          if (impurity > 0.0 && weightSum >= options.minSplitSize && nodeDepth < depthLimit) {
            var featIdx = 0
            while (featIdx < offsetLookup(treeIdx)(nodeIdx).length) {
              val featId = selectedFeaturesLookup(treeIdx)(nodeIdx)(featIdx)
              val numBins = numBinsPerFeature(featId)
              val binOffset = offsetLookup(treeIdx)(nodeIdx)(featIdx)

              val (featSplitImpurity, featSplitBinId, featSplitWeights) = binStatsArray.computeSplit(
                binOffset,
                numBins,
                summaryStatsArray,
                weightSum,
                isFeatureCategorical = featureBins(featId).isInstanceOf[CategoricalBins])

              if (featSplitBinId != -1 && (splitImpurity == None || featSplitImpurity < splitImpurity.get)) {
                splitImpurity = Some(featSplitImpurity)
                splitFeatId = Some(featId)
                splitBinId = Some(featSplitBinId)
                splitWeights = Some(featSplitWeights)
              }

              featIdx += 1
            }
          }

          // Now, let's see if we have a good enough split.
          var nodeSplit: Option[NodeSplitOnBinId] = None
          if (splitFeatId != None && splitImpurity.get < impurity && weightSum >= options.minSplitSize && nodeDepth < depthLimit) {
            if (featureBins(splitFeatId.get).isInstanceOf[CategoricalBins]) {
              val binIdToNodeIdMap = mutable.Map[Int, Int]()
              val nodeWeights = mutable.Map[Int, Double]()
              var binId = 0
              while (binId < splitBinId.get) { // split bin ID for categorical split is the number of bins.
                if (splitWeights.get(binId) > 0) {
                  val childNodeId = nextNodeIdsPerTree(treeIdx)

                  // The depth of the child node is the parent's depth + 1.
                  nodeDepths(treeIdx).put(childNodeId, nodeDepth + 1)

                  binIdToNodeIdMap.put(binId, childNodeId)
                  nodeWeights.put(childNodeId, splitWeights.get(binId).toDouble)

                  nextNodeIdsPerTree(treeIdx) += 1
                }

                binId += 1
              }

              nodeSplit = Some(CategoricalSplitOnBinId(
                nodeId,
                splitFeatId.get,
                binIdToNodeIdMap,
                nodeWeights))
            } else {
              val leftChildNodeId = nextNodeIdsPerTree(treeIdx)
              nextNodeIdsPerTree(treeIdx) += 1
              val rightChildNodeId = nextNodeIdsPerTree(treeIdx)
              nextNodeIdsPerTree(treeIdx) += 1

              // The depths of the child nodes should be the parent's depth + 1.
              nodeDepths(treeIdx).put(leftChildNodeId, nodeDepth + 1)
              nodeDepths(treeIdx).put(rightChildNodeId, nodeDepth + 1)

              nodeSplit = Some(NumericSplitOnBinId(
                nodeId,
                splitFeatId.get,
                splitBinId.get,
                leftChildNodeId,
                rightChildNodeId,
                splitWeights.get(0),
                splitWeights.get(1)))
            }
          }

          output += TrainedNodeInfo(treeIdx, nodeId, prediction, weightSum.toDouble, impurity, splitImpurity, nodeSplit)
        }

        nodeIdx += 1
      }

      treeIdx += 1
    }

    output.toIterator
  }
}
