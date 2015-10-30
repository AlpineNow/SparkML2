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

package spark_ml.tree_ensembles

import scala.collection.mutable
import scala.util.Random

import spark_ml.discretization._
import spark_ml.util.Sorting._
import spark_ml.util._
import spire.implicits._

/**
 * Internal node split representation.
 */
trait NodeSplitInfo extends Serializable {
  /**
   * Return the index of the feature that was split.
   */
  val featureId: Int

  /**
   * Return the split impurity.
   */
  val splitImpurity: Double

  /**
   * Select the child node based on the feature's bin Id.
   * @param binId Bin Id of the feature.
   * @return The child node to go to after splitting on the bin Id.
   */
  def chooseChildNode(binId: Int): NodeInfo

  /**
   * Get an ordered array of child nodes. The child nodes returned are ordered
   * by their node Ids.
   * @return An ordered array of child nodes.
   */
  def getOrderedChildNodes: Array[NodeInfo]

  /**
   * (Deep) copy myself. This is needed because this is mutable and this is used
   * within RDD map, which has lazy evaluation semantics.
   * @return A (deep) copy of this.
   */
  def copy: NodeSplitInfo
}

/**
 * Internal numeric node split representation.
 * @param featureId Index of the feature to split on.
 * @param splitBinId Bin ID of the split point.
 * @param splitImpurity Impurity of the split.
 * @param leftChildNode Left child node.
 * @param rightChildNode Right child node.
 * @param nanBinId NaN Bin ID of the feature (if it exists.).
 * @param nanChildNode NaN child node (if it exists.).
 */
class NumericNodeSplitInfo(
  val featureId: Int,
  val splitBinId: Int,
  val splitImpurity: Double,
  val leftChildNode: NodeInfo,
  val rightChildNode: NodeInfo,
  val nanBinId: Option[Int],
  val nanChildNode: Option[NodeInfo]) extends NodeSplitInfo {
  private val orderedChildArray =
    if (nanChildNode.nonEmpty) {
      if (leftChildNode != null) {
        Array(leftChildNode, rightChildNode, nanChildNode.get)
      } else {
        Array(rightChildNode, nanChildNode.get)
      }
    } else {
      Array(leftChildNode, rightChildNode)
    }

  /**
   * (Deep) copy myself. This is needed because this is mutable and this is used
   * within RDD map, which has lazy evaluation semantics.
   * @return A (deep) copy of this.
   */
  def copy: NodeSplitInfo = {
    new NumericNodeSplitInfo(
      featureId = this.featureId,
      splitBinId = this.splitBinId,
      splitImpurity = this.splitImpurity,
      leftChildNode =
        if (this.leftChildNode != null) {
          this.leftChildNode.copy
        } else {
          null.asInstanceOf[NodeInfo]
        },
      rightChildNode = this.rightChildNode.copy,
      nanBinId = this.nanBinId,
      nanChildNode =
        this.nanChildNode match {
          case Some(ncn) => Some(ncn.copy)
          case None => None
        }
    )
  }

  /**
   * Select the child node based on the feature bin Id.
   * @param binId Bin Id of the feature.
   * @return The child node to go to after splitting on the bin Id.
   */
  def chooseChildNode(binId: Int): NodeInfo = {
    if (nanBinId.nonEmpty && nanBinId.get == binId) {
      if (nanChildNode.nonEmpty) {
        nanChildNode.get
      } else {
        // It's possible that the binId is a nanBinId, but
        // we don't have a nan child.
        // This happens in GBT scenarios or validation scenarios.
        // In that case, we return the heavier child.
        if (rightChildNode.weight > leftChildNode.weight) {
          rightChildNode
        } else {
          leftChildNode
        }
      }
    } else if (binId >= splitBinId) {
      rightChildNode
    } else {
      leftChildNode
    }
  }

  /**
   * Get an ordered array of child nodes.
   * The child nodes returned are ordered by their node Ids.
   * @return An ordered array of child nodes.
   */
  def getOrderedChildNodes: Array[NodeInfo] = {
    orderedChildArray
  }
}

/**
 * Internal categorical node split representation.
 * @param featureId Index of the feature to split on.
 * @param splitImpurity Impurity of the split.
 * @param orderedChildNodes Ordered array of child nodes (ordered by node Ids).
 * @param binIdToChildNode Mapping from feature Bins to child Nodes.
 * @param largestChildNode The child node with the largest number of samples.
 */
class CatNodeSplitInfo(
  val featureId: Int,
  val splitImpurity: Double,
  val orderedChildNodes: Array[NodeInfo],
  val binIdToChildNode: mutable.Map[Int, Int],
  val largestChildNode: Int) extends NodeSplitInfo {

  /**
   * (Deep) copy myself. This is needed because this is mutable and this is used
   * within RDD map, which has lazy evaluation semantics.
   * @return A (deep) copy of this.
   */
  def copy: NodeSplitInfo = {
    new CatNodeSplitInfo(
      featureId = this.featureId,
      splitImpurity = this.splitImpurity,
      orderedChildNodes = this.orderedChildNodes.map(_.copy),
      binIdToChildNode = mutable.Map[Int, Int](this.binIdToChildNode.toSeq : _*),
      largestChildNode = this.largestChildNode
    )
  }

  /**
   * Select the child node based on the feature bin Id.
   * @param binId Bin Id of the feature.
   * @return The child node that the bin Id maps to.
   */
  def chooseChildNode(binId: Int): NodeInfo = {
    val i = binIdToChildNode.getOrElse(binId, -1)
    if (i != -1) {
      orderedChildNodes(i)
    } else {
      orderedChildNodes(largestChildNode)
    }
  }

  /**
   * Get an ordered array of child nodes.
   * The child nodes returned are ordered by their node Ids.
   * @return An ordered array of child nodes.
   */
  def getOrderedChildNodes: Array[NodeInfo] = {
    orderedChildNodes
  }
}

/**
 * Internal trained node information.
 * @param treeId Id of the tree that the node belongs to.
 * @param nodeId The node Id. This is used to store both split Id (in-progress)
 *               temporary Ids to keep track of nodes to split and real node Ids
 *               (final trained model node Ids).
 * @param prediction Prediction of the node.
 * @param addendum Addendum to the prediction (e.g. probability).
 * @param depth Depth of the node.
 * @param weight Weight of the node (number of samples).
 * @param impurity Impurity of the node.
 * @param splitInfo Split information (if it can be split).
 */
class NodeInfo(
  var treeId: Int,
  var nodeId: Int,
  var prediction: Double,
  val addendum: Double, // This can be used to store the prediction probability.
  var depth: Int,
  val weight: Double,
  val impurity: Double,
  val splitInfo: Option[NodeSplitInfo]) extends Serializable {

  /**
   * (Deep) copy myself. This is needed because this is mutable and this is used
   * within RDD map, which has lazy evaluation semantics.
   * @return A (deep) copy of this.
   */
  def copy: NodeInfo = {
    new NodeInfo(
      treeId = this.treeId,
      nodeId = this.nodeId,
      prediction = this.prediction,
      addendum = this.addendum,
      depth = this.depth,
      weight = this.weight,
      impurity = this.impurity,
      splitInfo =
        this.splitInfo match {
          case Some(nsi) => Some(nsi.copy)
          case None => None
        }
    )
  }

  /**
   * Determine whether this node is terminal or not based on the given
   * constraints.
   * @param maxDepth Maximum depth that's allowed.
   * @param minSplitSize Minimum node size to be eligible for splitting.
   * @return true if it's terminal. false otherwise.
   */
  def isTerminal(
    maxDepth: Int,
    minSplitSize: Int): Boolean = {
    !NodeStats.isSplitEligible(
      nodeImpurity = impurity,
      nodeDepth = depth,
      nodeWeight = weight,
      maxDepth = maxDepth,
      minSplitSize = minSplitSize
    )
  }
}

/**
 * Description of a selected feature.
 * @param featId Id of the feature.
 * @param numBins Number of bins in the feature (including NaN bins).
 * @param offset Stats offset to the feature in the node statistics array.
 * @param isCat Whether this feature is categorical.
 * @param nanBinId Id of the NaN Bin, if one exists.
 */
case class SelectedFeatureInfo(
  featId: Int,
  numBins: Int,
  offset: Int,
  isCat: Boolean,
  nanBinId: Option[Int])

/**
 * Node statistics to keep track of during the aggregation phase.
 * The statistics per feature bin are kept track of.
 * Different split criteria have different implementations since each criteria
 * requires different statistics.
 * @param treeId Id of the tree that the node belongs to.
 * @param nodeId Id of the node.
 * @param nodeDepth Depth of the node.
 * @param statsArray The actual statistics array.
 * @param mtryFeatures Selected feature descriptions.
 * @param numElemsPerBin Number of statistical elements per feature bin.
 */
abstract class NodeStats(
  val treeId: Int,
  val nodeId: Int,
  val nodeDepth: Int,
  val statsArray: Array[Double],
  val mtryFeatures: Array[SelectedFeatureInfo],
  val numElemsPerBin: Int) extends Serializable {

  /**
   * Add statistics related to a sample (label and features).
   * @param label Label of the sample.
   * @param features Features of the sample.
   * @param sampleCnt Sample count of the sample.
   * @param featureHandler Feature type handler.
   * @tparam T Feature type (Byte or Short).
   */
  def addSample[@specialized(Byte, Short) T](
    label: Double,
    features: Array[T],
    sampleCnt: Int,
    featureHandler: DiscretizedFeatureHandler[T]): Unit = {
    throw new UnsupportedOperationException(
      "addSample has not been implemented."
    )
  }

  /**
   * Merge this with another node stats.
   * The merged statistics are stored in this object.
   * @param ns Another node stats to merge with.
   */
  def mergeInPlace(ns: NodeStats): Unit = {
    cfor(0)(_ < statsArray.length, _ + 1)(
      i => statsArray(i) += ns.statsArray(i)
    )
  }

  /**
   * Internal object used to store temporary values of a trained node. To avoid
   * repeated allocations, this will be pre-allocated once and then reused to
   * store the values of the currently being-examined node values.
   * @param prediction Prediction of the node.
   * @param addendum Addendum to the prediction (e.g. probability).
   * @param weight Weight of the node (number of samples).
   * @param impurity Impurity of the node.
   * @param sumStats Sum of the Node statistics (e.g. distribution of labels).
   */
  protected class PreallocatedNodeValues(
    var prediction: Double,
    var addendum: Double,
    var weight: Double,
    var impurity: Double,
    val sumStats: Array[Double]) { // The aggregated node stats.

    /**
     * Copy the values from another node to this one.
     * @param another The object to copy from.
     */
    def copyFrom(another: PreallocatedNodeValues): Unit = {
      this.prediction = another.prediction
      this.addendum = another.addendum
      this.weight = another.weight
      this.impurity = another.impurity
      another.sumStats.copyToArray(this.sumStats)
    }
  }

  /**
   * Create a PreallocatedNodeValues object.
   * This is used to preallocate the internal node values object
   * that is reused multiple times.
   * @return A PreallocatedNodeValues object.
   */
  protected def preallocateNodeValues: PreallocatedNodeValues = {
    new PreallocatedNodeValues(
      prediction = 0.0,
      addendum = 0.0,
      weight = 0.0,
      impurity = 0.0,
      sumStats = Array.fill[Double](numElemsPerBin)(0.0))
  }

  /**
   * Internal object used to store temporary node split results.
   * This is pre-allocated and then reused multiple times.
   * @param splitImpurity Impurity of the split.
   * @param childNodeValues Child node values.
   * @param childNodeCount How many child nodes are in the above array.
   */
  protected class PreallocatedSplitResult(
    var splitImpurity: Double,
    val childNodeValues: Array[PreallocatedNodeValues],
    var childNodeCount: Int) {

    /**
     * Copy the values from another split result to this one.
     * @param another The object to copy from.
     */
    def copyFrom(another: PreallocatedSplitResult): Unit = {
      this.splitImpurity = another.splitImpurity
      this.childNodeCount = another.childNodeCount
      cfor(0)(_ < this.childNodeCount, _ + 1)(
        i => {
          this.childNodeValues(i).copyFrom(another.childNodeValues(i))
        }
      )
    }
  }

  /**
   * Create a PreallocatedSplitResult object.
   * This is used to preallocate the internal split result object that is reused
   * multiple times.
   * @return A PreallocatedSplitResult object.
   */
  protected def preallocateSplitResult: PreallocatedSplitResult = {
    // We allocate preallocate 65536 PreallocatedNodeValues objects
    // since that's the maximum number of split children possible in
    // the current implementation.
    new PreallocatedSplitResult(
      splitImpurity = Double.MaxValue,
      childNodeValues = Array.fill[PreallocatedNodeValues](65536)(
        preallocateNodeValues
      ),
      childNodeCount = 0)
  }

  /**
   * Learn the node values and then split the node if possible.
   * @param nodeSeed Node's random number seed.
   * @param minSplitSize Minimum node size for split eligibility.
   * @param maxDepth Maximum depth allowed.
   * @param catSplitType Categorical split type.
   * @return Learned node values and splits.
   */
  def splitNode(
    nodeSeed: Int,
    minSplitSize: Int,
    maxDepth: Int,
    catSplitType: CatSplitType.CatSplitType): NodeInfo = {
    val rng = new Random(nodeSeed)
    val nodeValues = calculateNodeValues
    val nodeWeight = nodeValues.weight
    var splitInfo: Option[NodeSplitInfo] = None

    // Let's see if this node is eligible for splitting.
    if (NodeStats.isSplitEligible(
      nodeImpurity = nodeValues.impurity,
      nodeDepth = nodeDepth,
      nodeWeight = nodeValues.weight,
      maxDepth = maxDepth,
      minSplitSize = minSplitSize)) {

      // If it's eligible for splits, then we have to go through selected
      // features and compute the best split impurity for each feature.
      // Additionally, we need to keep track of the overall best split.
      val bestSplitResult: PreallocatedSplitResult = preallocateSplitResult
      var bestSplitFeatId: Int = -1
      var bestSplitNanBinId: Option[Int] = None

      // If the best split comes from a numeric feature, this will be non
      // negative.
      var bestSplitNumericBinId: Int = -1

      // If the best split comes from a categorical feature, the following two
      // variables will contain relevant bin Id to child information.
      val bestSplitCatBinToGroupIds: Array[Int] = Array.fill[Int](65536)(0)
      // How many bins are in the categorical feature.
      var bestSplitNumCatBins: Int = -1

      // The following function is used to potentially update the best split
      // values after comparing the given split impurity with the current best
      // split impurity.
      def updateBestSplitInfo(
        splitResult: PreallocatedSplitResult,
        splitFeatId: Int,
        splitNanBinId: Option[Int] = None,
        splitNumericBinId: Int = -1,
        splitCatBinToGroupIds: Array[Int] = null,
        splitNumCatBins: Int = -1): Unit = {
        // We have to make sure that the split impurity is at least as good as
        // the node's unsplit impurity. Additionally, the new split impurity
        // should be better than the previous best split impurity.
        if (nodeValues.impurity > splitResult.splitImpurity &&
          bestSplitResult.splitImpurity > splitResult.splitImpurity) {
          // To avoid allocating a new split. We copy values.
          bestSplitResult.copyFrom(splitResult)
          bestSplitFeatId = splitFeatId
          bestSplitNanBinId = splitNanBinId
          bestSplitNumericBinId = splitNumericBinId
          bestSplitNumCatBins = splitNumCatBins
          if (splitNumCatBins > 0) {
            cfor(0)(_ < splitNumCatBins, _ + 1)(
              binId =>
                bestSplitCatBinToGroupIds(binId) = splitCatBinToGroupIds(binId)
            )
          }
        }
      }

      // Keep track of the current split values. We use preallocated objects to
      // avoid any new memory allocation.
      val curSplitResult: PreallocatedSplitResult = preallocateSplitResult
      var curSplitNumericBinId: Int = -1
      val curSplitCatBinToGroupIds: Array[Int] = Array.fill[Int](65536)(0)
      var curSplitNumCatBins: Int = -1

      // For performing a multi-way split on categorical features.
      def doMultiwayCatFeatureSplit(
        featId: Int,
        featOffset: Int,
        numBins: Int,
        binWeights: Array[Double]): Unit = {
        // Split across bin boundaries.
        // Every category value becomes one split partition.
        calculateSplit(
          partitionStatsArray = statsArray,
          offset = featOffset,
          numPartitions = numBins,
          partitionWeights = binWeights,
          weightSum = nodeWeight,
          output = curSplitResult)
        var curChildNodeId = 0
        cfor(0)(_ < numBins, _ + 1)(
          binId => {
            if (binWeights(binId) > 0.0) {
              // We'll have a child node for a particular bin
              // iff the bin is not empty.
              curSplitCatBinToGroupIds(binId) = curChildNodeId
              curChildNodeId += 1
            } else {
              // If the cat bin is empty, then there's no child for it.
              // Negative one indicates no child.
              curSplitCatBinToGroupIds(binId) = -1
            }
          }
        )

        curSplitNumericBinId = -1
        curSplitNumCatBins = numBins

        updateBestSplitInfo(
          splitResult = curSplitResult,
          splitFeatId = featId,
          splitCatBinToGroupIds = curSplitCatBinToGroupIds,
          splitNumCatBins = curSplitNumCatBins)
      }

      // Now, prepare variables to be used during binary splits. These are used
      // for both categorical and numerical binary splits.
      val curNonEmptyBinIds = Array.fill[Int](65536)(0)
      var curNonEmptyBinCount = 0
      val curBinLabelAverages: Array[Double] = Array.fill[Double](65536)(0.0)
      val curPartitionStats = Array.fill[Double](3 * numElemsPerBin)(0.0)
      val curPartitionWeights = Array.fill[Double](3)(0.0)
      var curNumPartitions = 0

      // For performing a binary split on categorical features.
      def doBinaryCatFeatureSplit(
        featId: Int,
        featOffset: Int,
        numBins: Int,
        binWeights: Array[Double]): Unit = {
        if (catSplitType == CatSplitType.OrderedBinarySplit &&
          (!forClassification || (forClassification && numElemsPerBin == 2))) {
          // We can perform 'ordered' binary splits for categorical features iff
          // the problem is regression or the number of target classes is 2.

          // Get the bin Ids of non-empty bins.
          // The following function will also sort the bin Ids by the labels.
          curNonEmptyBinCount = getNonEmptyBins(
            statsArray = statsArray,
            offset = featOffset,
            numBins = numBins,
            sortByLabelAverages = true, // Sort by label average of bins.
            rng = rng,
            binWeights = binWeights,
            tmpBinLabelAverages = curBinLabelAverages,
            output = curNonEmptyBinIds)
        } else {
          // Otherwise, we'll be doing a randomly sorted binary split.
          curNonEmptyBinCount = getNonEmptyBins(
            statsArray = statsArray,
            offset = featOffset,
            numBins = numBins,
            sortByLabelAverages = false, // Sort randomly.
            rng = rng,
            binWeights = binWeights,
            tmpBinLabelAverages = curBinLabelAverages,
            output = curNonEmptyBinIds)
        }

        val numNonEmptyBins = curNonEmptyBinCount
        curSplitNumericBinId = -1
        curSplitNumCatBins = numBins
        cfor(0)(_ < numBins, _ + 1)(
          binId => curSplitCatBinToGroupIds(binId) = -1
        )

        // Initially, all non-empty bins should belong
        // to the right side.
        cfor(0)(_ < numNonEmptyBins, _ + 1)(
          i => {
            val binId = curNonEmptyBinIds(i)
            curSplitCatBinToGroupIds(binId) = 1
          }
        )

        // Keep track of binary split stats.
        curNumPartitions = 2
        cfor(0)(_ < numElemsPerBin, _ + 1)(
          i => {
            // Left side.
            curPartitionStats(i) = 0.0

            // Right side.
            curPartitionStats(numElemsPerBin + i) = nodeValues.sumStats(i)
          }
        )
        curPartitionWeights(0) = 0.0
        curPartitionWeights(1) = nodeWeight

        // Now, perform binary splits over bins.
        cfor(0)(_ < (numNonEmptyBins - 1), _ + 1)(
          i => {
            // Move the bin stats from right to left.
            val binId = curNonEmptyBinIds(i)
            cfor(0)(_ < numElemsPerBin, _ + 1)(
              j => {
                val si = featOffset + binId * numElemsPerBin + j
                val elemVal = statsArray(si)
                // Add the current bin statistics to the left side.
                curPartitionStats(j) += elemVal
                // Subtract the current bin statistics from the right side.
                curPartitionStats(numElemsPerBin + j) -= elemVal
              }
            )

            val binWeight = binWeights(binId)
            curPartitionWeights(0) += binWeight
            curPartitionWeights(1) -= binWeight

            // 0 means the left side (1 means the right side).
            curSplitCatBinToGroupIds(binId) = 0

            // Calculate the results for this binary split.
            calculateSplit(
              partitionStatsArray = curPartitionStats,
              offset = 0,
              numPartitions = 2,
              partitionWeights = curPartitionWeights,
              weightSum = nodeWeight,
              output = curSplitResult)

            // Update the best split numbers if needed.
            updateBestSplitInfo(
              splitResult = curSplitResult,
              splitFeatId = featId,
              splitCatBinToGroupIds = curSplitCatBinToGroupIds,
              splitNumCatBins = curSplitNumCatBins)
          }
        )
      }

      // For performing binary numerical feature splits.
      // It could potentially be tri-splits if there are missing values.
      def doNumericalFeatureSplit(
        featId: Int,
        featOffset: Int,
        numBins: Int,
        nanBinId: Option[Int],
        binWeights: Array[Double]): Unit = {
        // Find out if we have to split on NaN feature values.
        val numNonNaNBins = numBins - (if (nanBinId.nonEmpty) 1 else 0)
        val numPartitions =
          if (nanBinId.nonEmpty && binWeights(nanBinId.get) > 0.0) {
            // If NaN exists and there are samples with NaN values, then
            // we'll split 3-way.
            3
          } else {
            2
          }

        // Keep track of split stats. Initially, everything is on the right side.
        curNumPartitions = numPartitions
        cfor(0)(_ < numElemsPerBin, _ + 1)(
          i => {
            // Left side.
            curPartitionStats(i) = 0.0

            // Right side.
            curPartitionStats(numElemsPerBin + i) = nodeValues.sumStats(i)

            // NaN side.
            curPartitionStats(2 * numElemsPerBin + i) = 0.0
          }
        )
        curPartitionWeights(0) = 0.0
        curPartitionWeights(1) = nodeWeight
        curPartitionWeights(2) = 0.0
        if (numPartitions == 3) {
          // If NaN bin is not empty, adjust the initial split stats accordingly
          // by adding the stats to the nan partition.
          val nanBinOffset = featOffset + nanBinId.get * numElemsPerBin
          cfor(0)(_ < numElemsPerBin, _ + 1)(
            i => {
              val elemVal = statsArray(nanBinOffset + i)
              curPartitionStats(numElemsPerBin + i) -= elemVal
              curPartitionStats(2 * numElemsPerBin + i) += elemVal
            }
          )

          val nanBinWeight = binWeights(nanBinId.get)
          curPartitionWeights(1) -= nanBinWeight
          curPartitionWeights(2) += nanBinWeight
        }

        // Now, perform binary splits over bins. The NaN bin won't change.
        cfor(0)(_ < (numNonNaNBins - 1), _ + 1)(
          i => {
            // i is equal to the bin Id in this case.
            val binWeight = binWeights(i)
            if (binWeight > 0.0) {
              // Move the bin stats from the right side to the left side.
              cfor(0)(_ < numElemsPerBin, _ + 1)(
                j => {
                  val si = featOffset + i * numElemsPerBin + j
                  val elemVal = statsArray(si)
                  // Add the current bin statistics to the left side.
                  curPartitionStats(j) += elemVal
                  // Subtract the current bin statistics from the right side.
                  curPartitionStats(numElemsPerBin + j) -= elemVal
                }
              )

              curPartitionWeights(0) += binWeight
              curPartitionWeights(1) -= binWeight

              // Calculate the results for this binary split.
              calculateSplit(
                partitionStatsArray = curPartitionStats,
                offset = 0,
                numPartitions = numPartitions,
                partitionWeights = curPartitionWeights,
                weightSum = nodeWeight,
                output = curSplitResult)

              // Handle cases where one of the non-Nan partitions is empty.
              // This can happen if, for example, all the non-NaN feature values
              // are the same, but NaN values exist.
              // This still results in a (non-standard) binary split case.
              val splitNumericBinId =
                if (numPartitions == 3 && curSplitResult.childNodeCount == 2) {
                  0
                } else {
                  i + 1
                }

              // Update the best split numbers if needed.
              updateBestSplitInfo(
                splitResult = curSplitResult,
                splitFeatId = featId,
                splitNanBinId = nanBinId,
                splitNumericBinId = splitNumericBinId)
            }
          }
        )
      }

      // Preallocate bin weights array.
      val curBinWeights = Array.fill[Double](65536)(0.0)

      // Go through all the selected features and
      // see if we can find an eligible best split.
      cfor(0)(_ < mtryFeatures.length, _ + 1)(
        idx => {
          val mtryFeature = mtryFeatures(idx)
          val featId = mtryFeature.featId
          val featOffset = mtryFeature.offset
          val nanBinId = mtryFeature.nanBinId
          val numBins = mtryFeature.numBins
          val isCat = mtryFeature.isCat

          // Get the bin weights.
          getBinWeights(
            statsArray = statsArray,
            offset = featOffset,
            numBins = numBins,
            output = curBinWeights)

          // Separate handling for categorical and numerical features.
          if (isCat) {
            // For categorical features.
            if (catSplitType == CatSplitType.MultiwaySplit) {
              // For multi-way categorical split, just split K-ways. This
              // function will automatically update the best split values when
              // applicable.
              doMultiwayCatFeatureSplit(
                featId = featId,
                featOffset = featOffset,
                numBins = numBins,
                binWeights = curBinWeights)
            } else {
              // For binary splits, we have to order the bins in some fashion.
              // And then we will split binarily, just like the numeric features.
              // This function will automatically update the best split values
              // when applicable.
              doBinaryCatFeatureSplit(
                featId = featId,
                featOffset = featOffset,
                numBins = numBins,
                binWeights = curBinWeights)
            }
          } else {
            // For numerical features.
            // This function will automatically update the best
            // split values when applicable.
            doNumericalFeatureSplit(
              featId = featId,
              featOffset = featOffset,
              numBins = numBins,
              nanBinId = nanBinId,
              binWeights = curBinWeights)
          }
        }
      )

      // Now create a split info if there's an eligible split that reduces the
      // node impurity.
      if (bestSplitFeatId != -1) {
        splitInfo = if (bestSplitNumericBinId != -1) {
          // See if we have a valid numeric split.
          // If this is false, it means that this is a binary split
          // between a NaN-bin and the rest of the bins.
          val numberSplitExists: Boolean = bestSplitNumericBinId > 0
          var curChildNodeIdx = 0
          val childNodeValues = bestSplitResult.childNodeValues
          val leftChildNode: NodeInfo = if (numberSplitExists) {
            // If bestSplitNumericBinId is not zero, then this is a numerical
            // feature split. Get the left child node.
            val tmp = new NodeInfo(
              treeId = treeId,
              nodeId = -1, // NodeId is not known yet.
                           // The driver will determine this.
              prediction = childNodeValues(curChildNodeIdx).prediction,
              addendum = childNodeValues(curChildNodeIdx).addendum,
              depth = nodeDepth + 1,
              weight = childNodeValues(curChildNodeIdx).weight,
              impurity = childNodeValues(curChildNodeIdx).impurity,
              splitInfo = None) // Can't know until the next iteration.
            curChildNodeIdx += 1
            tmp
          } else {
            // If the split's between a NaN-bin and the rest, then the left side
            // child node doesn't exist.
            null
          }

          // Get the right child node.
          val rightChildNode: NodeInfo = new NodeInfo(
            treeId = treeId,
            nodeId = -1, // NodeId is not known yet.
                         // The driver will determine this.
            prediction = childNodeValues(curChildNodeIdx).prediction,
            addendum = childNodeValues(curChildNodeIdx).addendum,
            depth = nodeDepth + 1,
            weight = childNodeValues(curChildNodeIdx).weight,
            impurity = childNodeValues(curChildNodeIdx).impurity,
            splitInfo = None) // Can't know until the next iteration.
          curChildNodeIdx += 1

          // Potentially get a NaN node.
          val nanChildNode: Option[NodeInfo] =
            if (!numberSplitExists ||
              bestSplitResult.childNodeCount == 3) {
              // We also have a nan Node.
              Some(new NodeInfo(
                treeId = treeId,
                nodeId = -1, // NodeId is not known yet.
                             // The driver witll determine this.
                prediction = childNodeValues(curChildNodeIdx).prediction,
                addendum = childNodeValues(curChildNodeIdx).addendum,
                depth = nodeDepth + 1,
                weight = childNodeValues(curChildNodeIdx).weight,
                impurity = childNodeValues(curChildNodeIdx).impurity,
                splitInfo = None)) // Can't know until the next iteration.
            } else {
              None
            }

          Some(
            new NumericNodeSplitInfo(
              featureId = bestSplitFeatId,
              splitBinId = bestSplitNumericBinId,
              splitImpurity = bestSplitResult.splitImpurity,
              leftChildNode = leftChildNode,
              rightChildNode = rightChildNode,
              nanBinId = bestSplitNanBinId,
              nanChildNode = nanChildNode
            )
          )
        } else {
          // Otherwise, this is a categorical feature split.
          val childNodeValues = bestSplitResult.childNodeValues
          var maxChildWeight = childNodeValues(0).weight
          val numChildren = bestSplitResult.childNodeCount
          val childNodes = new Array[NodeInfo](numChildren)
          val binIdToChildNode = mutable.Map[Int, Int]()
          var maxChildWeightIdx = 0
          val numCatBins = bestSplitNumCatBins
          val catBinToGroupIds = bestSplitCatBinToGroupIds
          cfor(0)(_ < numChildren, _ + 1)(
            i => {
              val childWeight = childNodeValues(i).weight
              if (childWeight > maxChildWeight) {
                maxChildWeight = childWeight
                maxChildWeightIdx = i
              }

              // Create individual child nodes.
              childNodes(i) = new NodeInfo(
                treeId = treeId,
                nodeId = -1, // NodeId is not known yet.
                             // The driver will determine this.
                prediction = childNodeValues(i).prediction,
                addendum = childNodeValues(i).addendum,
                depth = nodeDepth + 1,
                weight = childWeight,
                impurity = childNodeValues(i).impurity,
                splitInfo = None) // Can't know until the next iteration.
            }
          )
          cfor(0)(_ < numCatBins, _ + 1)(
            i =>
              if (catBinToGroupIds(i) >= 0) {
                binIdToChildNode.put(i, catBinToGroupIds(i))
              }
          )

          Some(
            new CatNodeSplitInfo(
              featureId = bestSplitFeatId,
              splitImpurity = bestSplitResult.splitImpurity,
              orderedChildNodes = childNodes,
              binIdToChildNode = binIdToChildNode,
              largestChildNode = maxChildWeightIdx
            )
          )
        }
      }
    }

    new NodeInfo(
      treeId = treeId,
      nodeId = nodeId,
      prediction = nodeValues.prediction,
      addendum = nodeValues.addendum,
      depth = nodeDepth,
      weight = nodeValues.weight,
      impurity = nodeValues.impurity,
      splitInfo = splitInfo)
  }

  /**
   * Get Ids of the bins that are not empty.
   * @param statsArray Stats array.
   * @param offset Start offset.
   * @param numBins Number of bins.
   * @param sortByLabelAverages Whether or not the return array should be
   *                            ordered by the label average. If false,
   *                            it'll be ordered randomly.
   * @param rng Random number generator to be used for random sorting.
   * @param binWeights Bin weights.
   * @param tmpBinLabelAverages A temporary storage for label averages.
   *                            This is here to avoid unnecessary allocations.
   * @param output The non empty bin Ids will be stored in this.
   * @return The number of non empty bins.
   */
  def getNonEmptyBins(
    statsArray: Array[Double],
    offset: Int,
    numBins: Int,
    sortByLabelAverages: Boolean,
    rng: Random,
    binWeights: Array[Double],
    tmpBinLabelAverages: Array[Double],
    output: Array[Int]): Int = {
    var nonEmptyBinCount = 0
    cfor(0)(_ < numBins, _ + 1)(
      binId => {
        if (binWeights(binId) > 0.0) {
          output(nonEmptyBinCount) = binId
          nonEmptyBinCount += 1
          if (sortByLabelAverages) {
            val binOffset = offset + binId * numElemsPerBin
            tmpBinLabelAverages(binId) = getBinLabelAverage(
              statsArray = statsArray,
              binOffset = binOffset
            )
          }
        }
      }
    )

    if (sortByLabelAverages) {
      // Sort by the label averages if required.
      quickSort[Int](output, nonEmptyBinCount)(
        Ordering.by[Int, Double](
          tmpBinLabelAverages(_)
        )
      )
    } else {
      // Otherwise, sort randomly.
      quickSort[Int](output, nonEmptyBinCount)(
        Ordering.by[Int, Double](
          _ => rng.nextDouble()
        )
      )
    }

    nonEmptyBinCount
  }

  /**
   * Calculate the node values of the current node (prediction, weight, etc.)
   * @return Node values.
   */
  def calculateNodeValues: PreallocatedNodeValues = {
    // Get the number of bins of the first feature.
    val numBins = mtryFeatures(0).numBins
    val sumStats = Array.fill[Double](numElemsPerBin)(0.0)
    val nodeValues = preallocateNodeValues

    // To get the node summary stats,
    // aggregate stats across all the bins of the first feature.
    cfor(0)(_ < numBins, _ + 1)(
      binId => {
        val binOffset = binId * numElemsPerBin
        cfor(0)(_ < numElemsPerBin, _ + 1)(
          i => sumStats(i) += statsArray(binOffset + i)
        )
      }
    )

    calculateNodeValues(
      sumStats = sumStats,
      offset = 0,
      output = nodeValues)

    nodeValues
  }

  /**
   * Calculate the split result for the given partitions.
   * @param partitionStatsArray Partitioned stats array.
   * @param offset Starting offset in the array.
   * @param numPartitions Number of partitions.
   * @param partitionWeights Partition weights.
   * @param weightSum Weight sum of the node.
   * @param output SplitResult object to store the output.
   */
  def calculateSplit(
    partitionStatsArray: Array[Double],
    offset: Int,
    numPartitions: Int,
    partitionWeights: Array[Double],
    weightSum: Double,
    output: PreallocatedSplitResult): Unit = {
    var splitImpurity = 0.0
    var numNonEmptyPartitions = 0
    cfor(0)(_ < numPartitions, _ + 1)(
      partId => {
        // We will compute child node values and splitImpurity using only
        // non-empty partitions.
        if (partitionWeights(partId) > 0.0) {
          val partOffset = offset + partId * numElemsPerBin
          calculateNodeValues(
            sumStats = partitionStatsArray,
            offset = partOffset,
            output = output.childNodeValues(numNonEmptyPartitions))
          val childNodeValues = output.childNodeValues(numNonEmptyPartitions)
          val portion = childNodeValues.weight / weightSum
          splitImpurity += portion * childNodeValues.impurity
          numNonEmptyPartitions += 1
        }
      }
    )

    // We only want non-empty child nodes.
    output.childNodeCount = numNonEmptyPartitions
    if (numNonEmptyPartitions > 1) {
      output.splitImpurity = splitImpurity
    } else {
      // If we don't have 2 or more child nodes, then the split is meaningless,
      // so we set the split impurity to be a max double value.
      output.splitImpurity = Double.MaxValue
    }
  }

  /**
   * Calculate the node values from the given summary stats.
   * @param sumStats Summary stats.
   * @param offset Starting offset.
   * @param output Where the output will be stored.
   * @return Same as output.
   */
  def calculateNodeValues(
    sumStats: Array[Double],
    offset: Int,
    output: PreallocatedNodeValues): PreallocatedNodeValues = {
    throw new UnsupportedOperationException(
      "calculateNodeValues has not been implemented."
    )
  }

  /**
   * Get bin weights.
   * @param statsArray Stats array.
   * @param offset Start offset.
   * @param numBins Number of bins.
   * @param output This is where the weights will be stored.
   * @return Returns the same output that was passed in.
   */
  def getBinWeights(
    statsArray: Array[Double],
    offset: Int,
    numBins: Int,
    output: Array[Double]): Array[Double] = {
    throw new UnsupportedOperationException(
      "getBinWeights has not been implemented."
    )
  }

  /**
   * Calculate the label average for the given bin.
   * @param statsArray Stats array.
   * @param binOffset Offset to the bin.
   * @return The label average for the bin.
   */
  def getBinLabelAverage(
    statsArray: Array[Double],
    binOffset: Int): Double = {
    throw new UnsupportedOperationException(
      "getBinLabelAverage has not been implemented."
    )
  }

  /**
   * Whether or not these statistics are for classification.
   * @return true for classification stats. false for regression stats.
   */
  def forClassification: Boolean = {
    throw new UnsupportedOperationException(
      "forClassification has not been implemented."
    )
  }
}

object NodeStats {
  /**
   * Find out whether the split-eligibility condition is met for the given node
   * values.
   * @param nodeImpurity Impurity of the node.
   * @param nodeDepth Depth of the node.
   * @param nodeWeight Weight of the node.
   * @param maxDepth Maximum depth allowed.
   * @param minSplitSize Minimum node size to be split-eligible.
   * @return true if split eligible. false otherwise.
   */
  def isSplitEligible(
    nodeImpurity: Double,
    nodeDepth: Int,
    nodeWeight: Double,
    maxDepth: Int,
    minSplitSize: Int): Boolean = {
    (nodeImpurity > 0.0) &&
      (nodeWeight >= minSplitSize.toDouble) &&
      (nodeDepth < maxDepth)
  }

  /**
   * Create a new NodeStats object, either for regression or classification.
   * @param treeId Tree Id.
   * @param nodeId Node Id (could be an internal split Id).
   * @param nodeDepth Node Depth.
   * @param treeType TreeType (split criteria).
   * @param featureBinsInfo Feature bins information.
   * @param mtryFeatureIds Selected (mtry) feature Ids.
   * @param numClasses Optional number of target classes
   *                   (only for classification).
   * @return a NodeStats object.
   */
  def createNodeStats(
    treeId: Int,
    nodeId: Int,
    nodeDepth: Int,
    treeType: SplitCriteria.SplitCriteria,
    featureBinsInfo: Array[Bins],
    mtryFeatureIds: Array[Int],
    numClasses: Option[Int]): NodeStats = {
    // How many stats array elements we have per feature bin.
    val numElemsPerBin =
      if (treeType == SplitCriteria.Regression_Variance) {
        3
      } else {
        numClasses.get
      }

    // This is where we store the selected feature information, such as stats
    // array offset, etc.
    val mtryFeatures = new Array[SelectedFeatureInfo](mtryFeatureIds.length)
    var curOffset = 0
    cfor(0)(_ < mtryFeatureIds.length, _ + 1)(
      i => {
        // Selected feature Id.
        val mtryFeatId = mtryFeatureIds(i)

        // Whether the feature is categorical.
        val isCat = featureBinsInfo(mtryFeatId).isInstanceOf[CategoricalBins]

        // To find out whether the feature has missing value bin.
        val nanBinId: Option[Int] =
          if (isCat) {
            // We don't have NaN bins for categorical features.
            None
          } else {
            featureBinsInfo(mtryFeatId).asInstanceOf[NumericBins].missingValueBinIdx
          }

        val mtryFeatNumBins = featureBinsInfo(mtryFeatId).getCardinality

        mtryFeatures(i) =
          SelectedFeatureInfo(
            featId = mtryFeatId,
            numBins = mtryFeatNumBins,
            offset = curOffset,
            isCat = isCat,
            nanBinId = nanBinId
          )
        curOffset += mtryFeatNumBins * numElemsPerBin
      }
    )

    val statsArray = Array.fill[Double](curOffset)(0.0)
    if (treeType == SplitCriteria.Regression_Variance) {
      new VarianceNodeStats(
        treeId = treeId,
        nodeId = nodeId,
        nodeDepth = nodeDepth,
        statsArray = statsArray,
        mtryFeatures = mtryFeatures,
        numElemsPerBin = numElemsPerBin)
    } else {
      new InfoGainNodeStats(
        treeId = treeId,
        nodeId = nodeId,
        nodeDepth = nodeDepth,
        statsArray = statsArray,
        mtryFeatures = mtryFeatures,
        numElemsPerBin = numElemsPerBin)
    }
  }
}
