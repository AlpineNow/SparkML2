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

import spire.implicits._

/**
 * To be returned after computing node predictions and splits.
 * @param treeId The Id of the tree that the node belongs to.
 * @param nodeId The node Id.
 * @param prediction The node prediction.
 * @param depth The depth of the node.
 * @param weight The node weight.
 * @param impurity The node impurity.
 * @param splitImpurity The impurity after the node split.
 * @param nodeSplit The node split containing info about how it's split.
 */
case class TrainedNodeInfo(
  treeId: Int,
  nodeId: Int,
  prediction: Double,
  depth: Int,
  weight: Double,
  impurity: Double,
  splitImpurity: Option[Double],
  nodeSplit: Option[NodeSplitOnBinId])

/**
 * Different split criteria (e.g., information gain, variance) would use different bin statistics.
 * The implementation of adding/computing statistics should be different for each split criteria.
 * This factory class is used to specific types of build statistics array.
 * @param numElemsPerBin The number of statistics per bin (e.g. 3 for Variance, numClasses for InfoGain, etc.).
 */
abstract class BinStatisticsArrayBuilder(val numElemsPerBin: Int) {
  protected val binStatsBuilder = new mutable.ArrayBuilder.ofDouble

  /**
   * Add numBins to the builder, and this will return the number of array elements added afterward.
   * @param numBins Number of bins to add.
   * @return The number of array elements that were added.
   */
  def addBins(numBins: Int): Int = {
    val numElemsToAdd = numBins * numElemsPerBin
    binStatsBuilder ++= Array.fill[Double](numElemsToAdd)(0.0)
    numElemsToAdd
  }

  /**
   * Call this, after adding all the bins, to create an actual BinStatistics array object.
   * @return BinStatisticsArray object
   */
  def createBinStatisticsArray: BinStatisticsArray
}

/**
 * Summary of data accumulated for a node.
 * @param prediction Prediction for the node.
 * @param impurity Impurity of the node.
 * @param summaryStatistics Summary statistics for the node (different for each split criteria).
 * @param weightSum Weight of the node (number of the training samples that fall on the node).
 */
case class NodeSummary(
  prediction: Double,
  impurity: Double,
  summaryStatistics: BinStatisticsArray,
  weightSum: Double)

/**
 * Summary of a split.
 * @param splitImpurity Impurity of the split.
 * @param splitGroups For a numeric split, a single number representing the id of the split bin point. For a categorical split, bins Ids for different groups. Omits the missing value bin ID.
 * @param splitWeights The weights after split groups. The last one contains the weight of the missing value group, if one exists.
 */
case class SplitSummary(
  splitImpurity: Double,
  splitGroups: Array[List[Int]],
  splitWeights: Array[Double])

/**
 * Different split criteria (e.g., information gain, variance) would use different bin statistics.
 * The implementation of adding/computing statistics should be different for each split criteria.
 * @param binStats An array that will store all the statistics numbers.
 * @param numElemsPerBin The number of statistics per bin (e.g. 3 for Variance, numClasses for InfoGain, etc.).
 */
abstract class BinStatisticsArray(val binStats: Array[Double], val numElemsPerBin: Int) extends Serializable {
  /**
   * Add a sample's label and count to the bin statistics of a feature of a tree's node.
   * The logic should be implemented by actual bin statistics object (different for different split criteria).
   * @param offset The offset into the feature.
   * @param featBinId The bin ID that we want to add to.
   * @param label The label of the sample.
   * @param sampleCount The count of the sample (weight of the sample).
   */
  def add(offset: Int, featBinId: Int, label: Double, sampleCount: Int): Unit

  /**
   * Add len values in arraySrc (from srcPos) to arrayDst (from dstPos).
   * @param arraySrc Array containing source values.
   * @param srcPos Start position for source.
   * @param arrayDst Array containing destination values.
   * @param dstPos Start position for destination.
   * @param len Length of values to add.
   */
  protected def addArrays(arraySrc: Array[Double], srcPos: Int, arrayDst: Array[Double], dstPos: Int, len: Int): Unit = {
    var curSrcPos = srcPos
    var curDstPos = dstPos
    cfor(0)(_ < len, _ + 1)(
      _ => {
        arrayDst(curDstPos) += arraySrc(curSrcPos)
        curSrcPos += 1
        curDstPos += 1
      }
    )
  }

  /**
   * Subtract len values in arraySrc (from srcPos) from arrayDst (from dstPos).
   * @param arraySrc Array containing source values.
   * @param srcPos Start position for source.
   * @param arrayDst Array containing destination values.
   * @param dstPos Start position for destination.
   * @param len Length of values to subtract.
   */
  protected def subtractArrays(arraySrc: Array[Double], srcPos: Int, arrayDst: Array[Double], dstPos: Int, len: Int): Unit = {
    var curSrcPos = srcPos
    var curDstPos = dstPos
    cfor(0)(_ < len, _ + 1)(
      _ => {
        arrayDst(curDstPos) -= arraySrc(curSrcPos)
        curSrcPos += 1
        curDstPos += 1
      }
    )
  }

  /**
   * Merge the stat values in this one with another one into this one.
   * @param b The other stat array object.
   */
  def mergeInPlace(b: BinStatisticsArray): this.type = {
    addArrays(b.binStats, 0, binStats, 0, binStats.length)
    this
  }

  /**
   * Summarize a sequence of bins.
   * This is equivalent to computing various node statistics based on one feature statistics.
   * I.e., get the prediction, the impurity, summary BinStatisticsArray and the weight.
   * Different for different split criteria.
   * @param offset Offset from which to do summarizing.
   * @param numBins Number of bins to summarize over.
   * @return A summary object that contains prediction, impurity, summary array, and weight sum.
   */
  def getNodeSummary(offset: Int, numBins: Int): NodeSummary

  /**
   * Use this to get the weight (number of samples) in a bin.
   * @param offset Offset to the feature of the node.
   * @param binId The bin id.
   * @return The weight of the bin.
   */
  def getBinWeight(offset: Int, binId: Int): Double

  /**
   * Use this to compute split impurity of the given group stats array (into each group).
   * @param splitGroupStats Group stats with N groups. This function should try to split the groups in N groups.
   * @param weightSum The total weight of all the groups.
   * @return Split impurity and the weights of the split groups.
   */
  def computeSplitImpurity(splitGroupStats: Array[Double], weightSum: Double): (Double, Array[Double])

  /**
   * Call this function to get a sorted array (according to some criteria determined by the child class) of a categorical feature bin Ids.
   * @param numBins number of bins in this categorical feature.
   * @param offset offset to the feature of the node.
   * @param randGen Random number generator in case it's needed. Useful for testing.
   */
  def sortCategoricalFeatureBins(numBins: Int, offset: Int, randGen: scala.util.Random): mutable.ArrayBuffer[Int]

  /**
   * The computeSplit uses a class derived from this interface to find best splits.
   * Different child classes for categorical/numerical splits.
   */
  trait BestSplitFinderForSingleFeature {
    def findBestSplit(splitGroupStats: Array[Double], weightSum: Double): SplitSummary
  }

  /**
   * Use this to find the best split of numerical features.
   */
  class BestSplitFinderForNumericalFeature(
      val numBinsWithoutNaN: Int,
      val offset: Int) extends BestSplitFinderForSingleFeature {
    def findBestSplit(splitGroupStats: Array[Double], weightSum: Double): SplitSummary = {
      var bestSplitImpurity = Double.MaxValue // Smaller == better
      var bestSplitBinId = 0
      var bestSplitGroupWeights: Array[Double] = null

      // First compute split with an empty left side.
      // I.e., if there's only one or fewer categories (e.g. everything is NaN), compute the impurity anyways.
      val (emptyLeftImpurity, emptyLeftWeights) = computeSplitImpurity(splitGroupStats, weightSum)
      bestSplitImpurity = emptyLeftImpurity
      bestSplitGroupWeights = emptyLeftWeights
      if (numBinsWithoutNaN > 1) {
        cfor(1)(_ < numBinsWithoutNaN, _ + 1)(
          binIdToSplit => {
            val prevBinId = binIdToSplit - 1
            val binOffset = prevBinId * numElemsPerBin
            addArrays(binStats, offset + binOffset, splitGroupStats, 0, numElemsPerBin)
            subtractArrays(binStats, offset + binOffset, splitGroupStats, numElemsPerBin, numElemsPerBin)
            val (featSplitImpurity, featSplitWeights) = computeSplitImpurity(splitGroupStats, weightSum)
            if (featSplitImpurity < bestSplitImpurity) {
              bestSplitImpurity = featSplitImpurity
              bestSplitBinId = binIdToSplit
              bestSplitGroupWeights = featSplitWeights
            }
          }
        )
      }

      SplitSummary(
        splitImpurity = bestSplitImpurity,
        splitGroups = Array.fill[List[Int]](1)(List[Int](bestSplitBinId)),
        splitWeights = bestSplitGroupWeights)
    }
  }

  /**
   * Use this to find the best split of categorical features.
   */
  class BestSplitFinderForCategoricalFeature(
      val numBinsWithoutNaN: Int,
      val offset: Int,
      val randGen: scala.util.Random) extends BestSplitFinderForSingleFeature {
    def findBestSplit(splitGroupStats: Array[Double], weightSum: Double): SplitSummary = {
      val sortedBinIds = sortCategoricalFeatureBins(numBinsWithoutNaN, offset, randGen)
      var bestSplitImpurity = Double.MaxValue // Smaller == better
      var bestSplitIdx = 0 // This is the index into the sorted array, and not the actual bin Id.
      val bestSplitGroups = Array.fill[List[Int]](2)(List[Int]())
      var bestSplitGroupWeights: Array[Double] = null
      bestSplitGroups(1) ++= sortedBinIds

      // First compute split with an empty left side.
      // I.e., if there's only one or fewer categories (e.g. everything is NaN), compute the impurity anyways.
      val (emptyLeftImpurity, emptyLeftWeights) = computeSplitImpurity(splitGroupStats, weightSum)
      bestSplitImpurity = emptyLeftImpurity
      bestSplitGroupWeights = emptyLeftWeights
      if (sortedBinIds.length > 1) {
        cfor(1)(_ < sortedBinIds.length, _ + 1)(
          i => {
            val prevBinId = sortedBinIds(i - 1)
            val binOffset = prevBinId * numElemsPerBin
            addArrays(binStats, offset + binOffset, splitGroupStats, 0, numElemsPerBin)
            subtractArrays(binStats, offset + binOffset, splitGroupStats, numElemsPerBin, numElemsPerBin)
            val (featSplitImpurity, featSplitWeights) = computeSplitImpurity(splitGroupStats, weightSum)
            if (featSplitImpurity < bestSplitImpurity) {
              bestSplitImpurity = featSplitImpurity
              bestSplitGroupWeights = featSplitWeights
              cfor(bestSplitIdx)(_ < i, _ + 1)(
                _ => {
                  val head :: tail = bestSplitGroups(1)
                  bestSplitGroups(0) ++= List[Int](head)
                  bestSplitGroups(1) = tail
                }
              )

              bestSplitIdx = i
            }
          }
        )
      }

      SplitSummary(
        splitImpurity = bestSplitImpurity,
        splitGroups = bestSplitGroups,
        splitWeights = bestSplitGroupWeights)
    }
  }

  /**
   * Compute the best split on a feature of a node.
   * @param offset Offset to the feature of the node.
   * @param numBins Number of bins in the feature.
   * @param summaryStatsArray Summary statistics of the node.
   * @param weightSum Weight of the node.
   * @param isFeatureCategorical Whether this is a categorical feature.
   * @param missingValueBinId If there's no missing value BinID, -1, otherwise, the BinID for missing values.
   * @param imputationType Imputation type.
   * @param randGen Random number generator in case it's needed. Useful for testing purposes.
   * @return Split summary object.
   */
  def computeSplit(
    offset: Int,
    numBins: Int,
    summaryStatsArray: BinStatisticsArray,
    weightSum: Double,
    isFeatureCategorical: Boolean,
    missingValueBinId: Int,
    imputationType: ImputationType.ImputationType,
    randGen: scala.util.Random): SplitSummary = {

    var missingValueGroupWeight = 0.0
    val numBinsWithoutNaN = if (missingValueBinId != -1) {
      missingValueGroupWeight = getBinWeight(offset, missingValueBinId)
      numBins - 1
    } else {
      numBins
    }

    val numSplitGroups = if (missingValueGroupWeight > 0.0 && imputationType == ImputationType.SplitOnMissing) 3 else 2
    val splitGroupStats = Array.fill[Double](numSplitGroups * numElemsPerBin)(0.0)

    // The right side of the binary split starts with the entire summary stat.
    // The missing value group would be the last split group, if it exists.
    summaryStatsArray.binStats.copyToArray(splitGroupStats, numElemsPerBin)
    val bestSplitFinder: BestSplitFinderForSingleFeature = if (isFeatureCategorical) {
      new BestSplitFinderForCategoricalFeature(numBinsWithoutNaN = numBinsWithoutNaN, offset = offset, randGen = randGen)
    } else {
      new BestSplitFinderForNumericalFeature(numBinsWithoutNaN = numBinsWithoutNaN, offset = offset)
    }

    if (missingValueGroupWeight == 0.0) { // There's no missing value for this feature.
      // Go through each split point and find the best split.
      // This is a 2-way split.
      bestSplitFinder.findBestSplit(splitGroupStats, weightSum)
    } else { // If there're missing values for this feature.
      if (imputationType == ImputationType.SplitOnMissing) {
        // This is a 3-way split.
        // Subtract the missing value group values from the right side and put them into the last group.
        subtractArrays(binStats, offset + missingValueBinId * numElemsPerBin, splitGroupStats, numElemsPerBin, numElemsPerBin)
        addArrays(binStats, offset + missingValueBinId * numElemsPerBin, splitGroupStats, 2 * numElemsPerBin, numElemsPerBin)
        bestSplitFinder.findBestSplit(splitGroupStats, weightSum)
      } else {
        // This is still a 2-way split.
        // However, we subtract all the missing value group values from the right side stats.
        subtractArrays(binStats, offset + missingValueBinId * numElemsPerBin, splitGroupStats, numElemsPerBin, numElemsPerBin)
        bestSplitFinder.findBestSplit(splitGroupStats, weightSum - missingValueGroupWeight)
      }
    }
  }
}

/**
 * This abstract class represents the aggregated statistics.
 * This object gets constructed in each RDD partition and then bin statistics get aggregated through this, per RDD partition.
 * During the construction, this will also perform the random feature selection.
 * During the reduce phase, AggregatedStatistics objects from different RDD partitions get merged (or their bin statistics array objects get merged through simple additions).
 * The actual logic of aggregation and statistics are handled by the BinStatisticsArray object as they may be different for different split criteria.
 * This also provides the function to compute splits from the aggregated statistics.
 * @param nodeSplitLookup A lookup table of node splits to be performed through this object. Used to determine bin statistics array size and offsets of different bins/features/nodes/trees.
 * @param binStatsArrayBuilder Bin statistics array builder (similar to array builder).
 * @param numBinsPerFeature Number of bins in each feature to help with determining offsets of bin statistics.
 * @param treeSeeds Seeds for the random number generator used to select random set of features.
 * @param mtry The number of random features to select per node.
 */
abstract class AggregatedStatistics(
    @transient private val nodeSplitLookup: ScheduledNodeSplitLookup,
    @transient private var binStatsArrayBuilder: BinStatisticsArrayBuilder,
    numBinsPerFeature: Array[Int],
    treeSeeds: Array[Int],
    mtry: Int) extends Serializable {

  // During the construction, we'll also build lookup arrays to find offsets to bins/features/nodes/trees quickly.
  // This is a multi-tiered array lookup table.
  // Features are randomly selected per node here.
  val numTrees = nodeSplitLookup.numTrees
  val numFeatures = numBinsPerFeature.length
  val numSelectedFeaturesPerNode = math.min(mtry, numFeatures)

  // Auxiliary structures to convert node Ids to array indices.
  // We calculate the node array indices by subtracting the starting node Ids from a given node Id.
  // The node is expected to be in this object when a caller wants to add samples for a node.
  private[spark_ml] val startNodeIds = Array.fill[Int](numTrees)(0)

  // This is the object that will be created within the constructor.
  // This is the object that keeps track of actual statistics.
  var binStatsArray: BinStatisticsArray = null

  // This is the multi-dimensional look up table to quickly find offsets within binStatsArray.
  // The first dimension index chooses the tree.
  // The second dimension index chooses the node.
  // The last dimension index refers to different features.
  // The final value is the offset to the first bin of the feature/node/tree within the bin stats array.
  private[spark_ml] val offsetLookup = new Array[Array[Array[Int]]](numTrees)

  // This is the multi-dimensional look up table to quickly find the feature Id of corresponding bin statistics array offsets.
  // This is used to quickly find randomly selected features per node.
  private[spark_ml] val selectedFeaturesLookup = new Array[Array[Array[Int]]](numTrees)

  var numNodes = 0 // Number of nodes we are processing over all the trees.

  // Constructor routine.
  {
    // To quickly find the offset into this one dimensional array, we use a 3 dimensional array.
    // There are three keys - tree Id, node Id and feature Id - each is used as array indices.
    // Node Id and feature Id both have to be converted to array indices through an auxiliary structures.
    val offsetLookupBuilder = Array.fill[mutable.ArrayBuilder[Array[Int]]](numTrees)(mutable.ArrayBuilder.make[Array[Int]]())

    // Keep track of selected features per node.
    // The values of the last dimension are the indices of selected features. The last dimension's array index can be used to access the last dimension of offsetLookup's.
    val selectedFeaturesLookupBuilder = Array.fill[mutable.ArrayBuilder[Array[Int]]](numTrees)(mutable.ArrayBuilder.make[Array[Int]]())

    var curOffset = 0 // The current offset into binStatsArray object.
    cfor(0)(_ < numTrees, _ + 1)(
      treeId => {
        val nodeSplits = nodeSplitLookup.nodeSplitTable(treeId)
        var prevNodeId = 0
        cfor(0)(_ < nodeSplits.length, _ + 1)(
          nodeSplitIdx => {
            val nodeSplit = nodeSplits(nodeSplitIdx)
            if (nodeSplit != null) { // nodeSplit might be null. TODO: Fix this ugliness.
              val childNodeIds = nodeSplit.getOrderedChildNodeIds // Get child node Ids ordered by the Id, to make sure that we process things in a breadth-first manner.
              cfor(0)(_ < childNodeIds.length, _ + 1)(
                childNodeIdx => {
                  val childNodeId = childNodeIds(childNodeIdx)
                  if (nodeSplit.getSubTreeHash(childNodeId) == -1) {
                    // We add this node to statistics only if this node is not trained as a sub-tree.
                    // It's indicated by the node's sub-tree-hash value.
                    // -1 hash value means that it's not trained as a sub-tree.
                    // TODO: Fix this ugliness.
                    if (startNodeIds(treeId) == 0) {
                      startNodeIds(treeId) = childNodeId
                    }

                    // Since we can skip certain nodes that are to be trained locally as sub-trees,
                    // we might be skipping node Ids.
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

                    // Now add the bins per selected feature and mark the offsets.
                    cfor(0)(_ < selectedFeatureIndices.length, _ + 1)(
                      featIdx => {
                        val featId = selectedFeatureIndices(featIdx)
                        val numBins = numBinsPerFeature(featId)
                        featureOffsets(featIdx) = curOffset
                        curOffset += binStatsArrayBuilder.addBins(numBins)
                      }
                    )
                  }
                }
              )
            }
          }
        )
      }
    )

    // Now, get concrete objects from various builders.
    binStatsArray = binStatsArrayBuilder.createBinStatisticsArray
    cfor(0)(_ < numTrees, _ + 1)(treeId => offsetLookup(treeId) = offsetLookupBuilder(treeId).result())
    cfor(0)(_ < numTrees, _ + 1)(treeId => selectedFeaturesLookup(treeId) = selectedFeaturesLookupBuilder(treeId).result())

    // To trigger GC on this builder object that was passed through the constructor.
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
    cfor(0)(_ < selectedFeatures.length, _ + 1)(
      featIdx => {
        val featId = selectedFeatures(featIdx)
        val featBinId = Discretizer.readUnsignedByte(sample._2(featId))
        val offset = offsetLookup(treeId)(nodeIdx)(featIdx)
        binStatsArray.add(offset, featBinId, label, sampleCount)
      }
    )
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
    cfor(0)(_ < selectedFeatures.length, _ + 1)(
      featIdx => {
        val featId = selectedFeatures(featIdx)
        val featBinId = Discretizer.readUnsignedShort(sample._2(featId))
        val offset = offsetLookup(treeId)(nodeIdx)(featIdx)
        binStatsArray.add(offset, featBinId, label, sampleCount)
      }
    )
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
   * @param randGen Random number generator to use in case rand numbers are needed. Useful for testing.
   * @return An iterator of trained node info objects. The trainer will use this to build trees.
   */
  def computeNodePredictionsAndSplits(
    featureBins: Array[Bins],
    nextNodeIdsPerTree: Array[Int],
    nodeDepths: Array[mutable.Map[Int, Int]],
    options: SequoiaForestOptions,
    randGen: scala.util.Random): Iterator[TrainedNodeInfo] = {
    val output = new mutable.ArrayBuffer[TrainedNodeInfo]()
    val depthLimit = options.maxDepth match {
      case x if x == -1 => Int.MaxValue
      case x => x
    }

    cfor(0)(_ < numTrees, _ + 1)(
      treeId => {
        cfor(0)(_ < offsetLookup(treeId).length, _ + 1)(
          nodeIdx => {
            if (offsetLookup(treeId)(nodeIdx).length > 0) {
              // Compute the node summary from the first feature.
              // This should only be done once per node.
              val nodeSummary = binStatsArray.getNodeSummary(
                offsetLookup(treeId)(nodeIdx)(0),
                numBinsPerFeature(selectedFeaturesLookup(treeId)(nodeIdx)(0)))

              // Now go through all the features and select the best split among them.
              var bestSplitImpurity: Option[Double] = None
              var bestSplitFeatId: Option[Int] = None
              var bestSplitGroups: Option[Array[List[Int]]] = None
              var bestSplitGroupWeights: Option[Array[Double]] = None

              val nodeId = startNodeIds(treeId) + nodeIdx
              val nodeDepth = nodeDepths(treeId)(nodeId)
              nodeDepths(treeId).remove(nodeId) // We don't need to keep track of this node's depth any more. TODO: Fix this ugliness.

              // See if the conditions meet the split criteria.
              if (nodeSummary.impurity > 0.0 &&
                nodeSummary.weightSum >= options.minSplitSize &&
                nodeDepth < depthLimit) {
                cfor(0)(_ < offsetLookup(treeId)(nodeIdx).length, _ + 1) {
                  featIdx =>
                    {
                      val featId = selectedFeaturesLookup(treeId)(nodeIdx)(featIdx)
                      val numBins = numBinsPerFeature(featId)
                      val binOffset = offsetLookup(treeId)(nodeIdx)(featIdx)

                      val splitSummary = binStatsArray.computeSplit(
                        offset = binOffset,
                        numBins = numBins,
                        summaryStatsArray = nodeSummary.summaryStatistics,
                        weightSum = nodeSummary.weightSum,
                        isFeatureCategorical = featureBins(featId).isInstanceOf[CategoricalBins],
                        missingValueBinId = featureBins(featId).getMissingValueBinIdx,
                        imputationType = options.imputationType,
                        randGen = randGen)

                      if (splitSummary.splitWeights != null &&
                        (bestSplitImpurity == None || splitSummary.splitImpurity < bestSplitImpurity.get)) {
                        bestSplitImpurity = Some(splitSummary.splitImpurity)
                        bestSplitFeatId = Some(featId)
                        bestSplitGroups = Some(splitSummary.splitGroups)
                        bestSplitGroupWeights = Some(splitSummary.splitWeights)
                      }
                    }
                }
              }

              // Now, let's see if we have a good enough split.
              var nodeSplit: Option[NodeSplitOnBinId] = None
              if (bestSplitFeatId != None && bestSplitImpurity.get < nodeSummary.impurity) {
                val missingValueBinId = featureBins(bestSplitFeatId.get).getMissingValueBinIdx

                // We have a good enough split.
                // Two different ways to do this based on feature type.
                if (featureBins(bestSplitFeatId.get).isInstanceOf[CategoricalBins]) { // If this is a categorical feature.
                  val binIdToNodeIdMap = mutable.Map[Int, Int]()
                  val nodeWeights = mutable.Map[Int, Double]()

                  if (bestSplitGroupWeights.get(0) > 0.0) {
                    // Left child processing if it contains some elements.
                    val leftChildNodeId = nextNodeIdsPerTree(treeId)
                    nodeDepths(treeId).put(leftChildNodeId, nodeDepth + 1) // Record the depth of the child node for later reference.
                    bestSplitGroups.get(0).foreach(binId => binIdToNodeIdMap.put(binId, leftChildNodeId))
                    nodeWeights.put(leftChildNodeId, bestSplitGroupWeights.get(0))
                    nextNodeIdsPerTree(treeId) += 1
                  }

                  if (bestSplitGroupWeights.get(1) > 0.0) {
                    // Right child processing.
                    val rightChildNodeId = nextNodeIdsPerTree(treeId)
                    nodeDepths(treeId).put(rightChildNodeId, nodeDepth + 1) // Record the depth of the child node for later reference.
                    bestSplitGroups.get(1).foreach(binId => binIdToNodeIdMap.put(binId, rightChildNodeId))
                    nodeWeights.put(rightChildNodeId, bestSplitGroupWeights.get(1))
                    nextNodeIdsPerTree(treeId) += 1
                  }

                  // See if there's a missing value group.
                  if (missingValueBinId != -1) {
                    if (bestSplitGroupWeights.get.length == 3 && bestSplitGroupWeights.get(2) > 0.0) {
                      val nanChildNodeId = nextNodeIdsPerTree(treeId)
                      nodeDepths(treeId).put(nanChildNodeId, nodeDepth + 1) // Record the depth of the child node for later reference.
                      binIdToNodeIdMap.put(missingValueBinId, nanChildNodeId)
                      nodeWeights.put(nanChildNodeId, bestSplitGroupWeights.get(2))
                      nextNodeIdsPerTree(treeId) += 1
                    }
                  }

                  nodeSplit = Some(CategoricalSplitOnBinId(
                    parentNodeId = nodeId,
                    featureId = bestSplitFeatId.get,
                    binIdToNodeIdMap = binIdToNodeIdMap,
                    nodeWeights = nodeWeights))
                } else { // If this is a numerical feature.
                  // If either side has 0 weight, we'll unite it into a single right side.
                  var splitBinId = bestSplitGroups.get(0).head
                  var leftChildNodeId = -1
                  var leftChildNodeWeight = bestSplitGroupWeights.get(0)
                  var rightChildNodeId = -1
                  var rightChildNodeWeight = bestSplitGroupWeights.get(1)
                  if (leftChildNodeWeight == 0.0 ||
                    rightChildNodeWeight == 0.0) {
                    splitBinId = 0
                    rightChildNodeWeight = math.max(leftChildNodeWeight, rightChildNodeWeight)
                    leftChildNodeWeight = 0.0

                    rightChildNodeId = nextNodeIdsPerTree(treeId)
                    nodeDepths(treeId).put(rightChildNodeId, nodeDepth + 1)
                    nextNodeIdsPerTree(treeId) += 1
                  } else {
                    leftChildNodeId = nextNodeIdsPerTree(treeId)
                    nodeDepths(treeId).put(leftChildNodeId, nodeDepth + 1) // Record the depth of the child node for later reference.
                    nextNodeIdsPerTree(treeId) += 1

                    rightChildNodeId = nextNodeIdsPerTree(treeId)
                    nodeDepths(treeId).put(rightChildNodeId, nodeDepth + 1)
                    nextNodeIdsPerTree(treeId) += 1
                  }

                  var nanNodeId = -1
                  var nanWeight = 0.0
                  // See if there's a missing value group.
                  if (missingValueBinId != -1) {
                    if (bestSplitGroupWeights.get.length == 3 && bestSplitGroupWeights.get(2) > 0.0) {
                      val nanChildNodeId = nextNodeIdsPerTree(treeId)
                      nodeDepths(treeId).put(nanChildNodeId, nodeDepth + 1)
                      val nanChildNodeWeight = bestSplitGroupWeights.get(2)
                      nextNodeIdsPerTree(treeId) += 1

                      nanNodeId = nanChildNodeId
                      nanWeight = nanChildNodeWeight
                    }
                  }

                  nodeSplit = Some(NumericSplitOnBinId(
                    parentNodeId = nodeId,
                    featureId = bestSplitFeatId.get,
                    splitBinId = splitBinId,
                    leftId = leftChildNodeId,
                    rightId = rightChildNodeId,
                    leftWeight = leftChildNodeWeight,
                    rightWeight = rightChildNodeWeight,
                    leftSubTreeHash = -1,
                    rightSubTreeHash = -1,
                    nanBinId = missingValueBinId,
                    nanNodeId = nanNodeId,
                    nanWeight = nanWeight,
                    nanSubTreeHash = -1))
                }
              }

              output += TrainedNodeInfo(
                treeId = treeId,
                nodeId = nodeId,
                prediction = nodeSummary.prediction,
                depth = nodeDepth,
                weight = nodeSummary.weightSum,
                impurity = nodeSummary.impurity,
                splitImpurity = bestSplitImpurity,
                nodeSplit = nodeSplit
              )
            }
          }
        )
      }
    )

    output.toIterator
  }
}
