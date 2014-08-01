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
 * This is used to build a ClassificationStatisticsArray object.
 * @param numClasses The number of classes in the classification problem.
 */
case class ClassificationStatisticsArrayBuilder(numClasses: Int) extends BinStatisticsArrayBuilder {
  // We could use either Int or Long to keep track of bin counts.
  // For big data with more than a few billion rows, Int may not work and Long is preferable.

  // The entire bin statistics for all trees and nodes are stored in a one dimensional array.
  // Long to keep track of bin counts.
  private val binStatsBuilder: Object = new mutable.ArrayBuilder.ofLong

  /**
   * Add numBins to the builder, and this will return the size of bins added afterward.
   * @param numBins Number of bins to add.
   * @return The size of the bins that were added.
   */
  def addBins(numBins: Int): Int = {
    val size = numBins * numClasses
    binStatsBuilder.asInstanceOf[mutable.ArrayBuilder[Long]] ++= Array.fill[Long](size)(0)
    size
  }

  /**
   * Call this, after adding all the bins, to create an actual BinStatistics array object.
   * @return BinStatisticsArray object
   */
  def createBinStatisticsArray: BinStatisticsArray = {
    ClassificationStatisticsArray(binStatsBuilder.asInstanceOf[mutable.ArrayBuilder[Long]].result(), numClasses)
  }
}

/**
 * This is the wrapper around the stats array.
 * @param binStats The stats array.
 * @param numClasses The number of classes in the classification problem.
 */
case class ClassificationStatisticsArray(binStats: Array[Long], numClasses: Int) extends BinStatisticsArray {
  /**
   * Add a sample's label and count to the bin statistics of a feature of a tree's node.
   * @param offset The offset into the feature within this object.
   * @param featBinId The bin ID that we want to add to.
   * @param label The label of the sample.
   * @param sampleCount The count of the sample (weight of the sample).
   */
  def add(offset: Int, featBinId: Int, label: Double, sampleCount: Int): Unit = {
    binStats(offset + featBinId * numClasses + label.toInt) += sampleCount
  }

  /**
   * Merge the stat values in this one with another one into this one.
   * @param another The other stat array object.
   */
  def mergeInPlace(another: BinStatisticsArray): BinStatisticsArray = {
    var i = 0
    while (i < binStats.length) {
      binStats(i) += another.asInstanceOf[ClassificationStatisticsArray].binStats(i)
      i += 1
    }

    this
  }

  /**
   * Compute the entropy for the class distribution.
   * @param classWeights Counts for each class.
   * @param weightSum Total count.
   * @return Entropy
   */
  private def computeEntropy(classWeights: Array[Long], weightSum: Long, startIdx: Int = 0): Double = {
    var entropy = 0.0
    var labelId = 0
    while (labelId < numClasses) {
      val w = classWeights(startIdx + labelId)
      if (w > 0) {
        val prob = w.toDouble / weightSum.toDouble
        entropy -= prob * math.log(prob) / math.log(2.0)
      }

      labelId += 1
    }

    entropy
  }

  /**
   * Split the bins as given and compute the split impurity and weights.
   * @param featBinStats An array of bins and their label counts in one-dimensional array.
   * @param weightSum The total weight.
   * @return A tuple of split impurity and bin weights
   */
  private def computeSplitEntropy(featBinStats: Array[Long], weightSum: Long): (Double, Array[Long]) = {
    val numBins = featBinStats.length / numClasses
    var splitEntropy = 0.0
    val binWeights = Array.fill[Long](numBins)(0)
    var binId = 0
    while (binId < numBins) {
      var binWeight: Long = 0
      var labelId = 0
      while (labelId < numClasses) {
        binWeight += featBinStats(binId * numClasses + labelId)
        labelId += 1
      }

      binWeights(binId) = binWeight
      splitEntropy += binWeight.toDouble / weightSum.toDouble * computeEntropy(featBinStats, binWeight, binId * numClasses)

      binId += 1
    }

    (splitEntropy, binWeights)
  }

  /**
   * Summarize a series of bins. I.e., get prediction, impurity and BinStatisticsArray that summarizes the series into single summarizer bin.
   * This should be used to compute prediction, impurity of a node.
   * @param offset Offset from which to do summarizing.
   * @param numBins Number of bins to summarize over.
   * @return Prediction, impurity, single summarizer bin, and weight sum.
   */
  def getSummaryOverBins(offset: Int, numBins: Int): (Double, Double, BinStatisticsArray, Long) = {
    var maxClassWeight: Long = 0
    val classWeights = Array.fill[Long](numClasses)(0)
    var weightSum: Long = 0
    var prediction: Double = 0.0
    var binId = 0
    while (binId < numBins) {
      var labelId = 0
      while (labelId < numClasses) {
        val count = binStats(offset + binId * numClasses + labelId)
        classWeights(labelId) += count
        weightSum += count

        if (maxClassWeight < classWeights(labelId)) {
          maxClassWeight = classWeights(labelId)
          prediction = labelId.toDouble
        }

        labelId += 1
      }

      binId += 1
    }

    (prediction, computeEntropy(classWeights, weightSum), ClassificationStatisticsArray(classWeights, numClasses), weightSum)
  }

  /**
   * Add len values in binArraySrc (from srcPos) to binArrayDst (from dstPos).
   * @param binArraySrc Array containing source values.
   * @param srcPos Start position for source.
   * @param binArrayDst Array containing destination values.
   * @param dstPos Start position for destination.
   * @param len Length of values to add.
   */
  private def addBins(binArraySrc: Array[Long], srcPos: Int, binArrayDst: Array[Long], dstPos: Int, len: Int): Unit = {
    var curSrcPos = srcPos
    var curDstPos = dstPos
    var numCopied = 0
    while (numCopied < len) {
      binArrayDst(curDstPos) += binArraySrc(curSrcPos)
      numCopied += 1
      curSrcPos += 1
      curDstPos += 1
    }
  }

  /**
   * Subtract len values in binArraySrc (from srcPos) from binArrayDst (from dstPos).
   * @param binArraySrc Array containing source values.
   * @param srcPos Start position for source.
   * @param binArrayDst Array containing destination values.
   * @param dstPos Start position for destination.
   * @param len Length of values to subtract.
   */
  private def subtractBins(binArraySrc: Array[Long], srcPos: Int, binArrayDst: Array[Long], dstPos: Int, len: Int): Unit = {
    var curSrcPos = srcPos
    var curDstPos = dstPos
    var numCopied = 0
    while (numCopied < len) {
      binArrayDst(curDstPos) -= binArraySrc(curSrcPos)
      numCopied += 1
      curSrcPos += 1
      curDstPos += 1
    }
  }

  /**
   * Compute split on a feature.
   * @param offset Offset to the feature.
   * @param numBins Number of bins in the feature.
   * @param summaryStatsArray One summary bin over all the bins.
   * @param weightSum Weight sum over all the bins.
   * @param isFeatureCategorical Whether this is a categorical feature.
   * @return split impurity, split bin ID and split weights.
   */
  def computeSplit(offset: Int, numBins: Int, summaryStatsArray: BinStatisticsArray, weightSum: Long, isFeatureCategorical: Boolean): (Double, Int, Array[Long]) = {
    var bestFeatSplitImpurity = Double.MaxValue
    var bestFeatSplitBinId = -1
    var bestFeatSplitWeights = Array[Long]()

    // Compute the best split for this feature.
    // We'll do a simple K-way split for a categorical feature.
    // TODO: This may not be ideal. Might want to fix it at some point. Or at least investigate.
    if (isFeatureCategorical) {
      // Get the bin stats.
      val featBinStats = binStats.slice(offset, offset + numBins * numClasses)
      val (featSplitImpurity, binWeights) = computeSplitEntropy(featBinStats, weightSum)

      bestFeatSplitImpurity = featSplitImpurity
      bestFeatSplitBinId = numBins // For categorical features, we just put the number of bins, instead of split bin Id.
      bestFeatSplitWeights = binWeights
    } else {
      // If it's a numeric feature, we'll scan through
      // each bin as a split point.
      val splitBinStats = Array.fill[Long](2 * numClasses)(0)
      summaryStatsArray.asInstanceOf[ClassificationStatisticsArray].binStats.copyToArray(splitBinStats, numClasses)

      var binIdToSplit = 1
      while (binIdToSplit < numBins) {
        val prevBinId = binIdToSplit - 1
        val binOffset = prevBinId * numClasses
        addBins(binStats, offset + binOffset, splitBinStats, 0, numClasses)
        subtractBins(binStats, offset + binOffset, splitBinStats, numClasses, numClasses)
        val (featSplitImpurity, featSplitWeights) = computeSplitEntropy(splitBinStats, weightSum)

        if (featSplitImpurity < bestFeatSplitImpurity) {
          bestFeatSplitImpurity = featSplitImpurity
          bestFeatSplitBinId = binIdToSplit
          bestFeatSplitWeights = featSplitWeights
        }

        binIdToSplit += 1
      }
    }

    (bestFeatSplitImpurity, bestFeatSplitBinId, bestFeatSplitWeights)
  }
}

/**
 * Keeps track of node statistics for computing information gain. This will also select random features for each node.
 * @param nodeSplitLookup We use the scheduled node splits to find out the nodes whose bin statistics we want to store in this object.
 * @param numBinsPerFeature The count of bins for each feature.
 * @param treeSeeds Random seeds to use for each tree when selecting a random set of features.
 * @param mtry The number of random features per node.
 * @param numClasses The number of target classes. (greater than 2 for multi-class classification).
 */
case class InfoGainStatistics(
    @transient private val nodeSplitLookup: ScheduledNodeSplitLookup,
    numBinsPerFeature: Array[Int],
    treeSeeds: Array[Int],
    mtry: Int,
    numClasses: Int) extends AggregatedStatistics(nodeSplitLookup, numBinsPerFeature, treeSeeds, mtry, ClassificationStatisticsArrayBuilder(numClasses)) {

  /**
   * Get the bin count. This is slow because of linear lookup of feature Id (shouldn't be used other than for some testing purposes).
   * @param treeId The Id of the tree
   * @param nodeId The Id of the node
   * @param featureId The Id of the feature
   * @param binId The Id of the bin
   * @param label The label whose count we want
   * @return The count
   */
  private[spark_ml] def getBinCount(
    treeId: Int,
    nodeId: Int,
    featureId: Int,
    binId: Int,
    label: Int): Long = {
    val nodeIdx = nodeId - startNodeIds(treeId)
    val nodeFeatures = selectedFeaturesLookup(treeId)(nodeIdx)
    var found = false
    var selectedFeatIdx = 0
    while (!found) {
      if (nodeFeatures(selectedFeatIdx) == featureId) {
        found = true
      } else {
        selectedFeatIdx += 1
      }
    }

    val offset = offsetLookup(treeId)(nodeIdx)(selectedFeatIdx)
    binStatsArray.asInstanceOf[ClassificationStatisticsArray].binStats(offset + binId * numClasses + label)
  }
}
