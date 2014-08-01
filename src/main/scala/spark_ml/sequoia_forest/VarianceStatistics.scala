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
 * This is used to build a RegressionStatisticsArray object.
 */
class RegressionStatisticsArrayBuilder extends BinStatisticsArrayBuilder {
  private val binStatsBuilder = new mutable.ArrayBuilder.ofDouble

  /**
   * Add numBins to the builder, and this will return the size of bins added afterward.
   * @param numBins Number of bins to add.
   * @return The size of the bins that were added.
   */
  def addBins(numBins: Int): Int = {
    val size = numBins * 3
    binStatsBuilder ++= Array.fill[Double](size)(0.0)
    size
  }

  /**
   * Call this, after adding all the bins, to create an actual BinStatistics array object.
   * @return BinStatisticsArray object
   */
  def createBinStatisticsArray: BinStatisticsArray = {
    RegressionStatisticsArray(binStatsBuilder.result())
  }
}

/**
 * This is the wrapper around the stats array (contains sum, square sum and count).
 * @param binStats The stats array.
 */
case class RegressionStatisticsArray(binStats: Array[Double]) extends BinStatisticsArray {
  /**
   * Add a sample's label and count to the bin statisitics of a feature of a tree's node.
   * @param offset The offset into the feature within this object.
   * @param featBinId The bin ID that we want to add to.
   * @param label The label of the sample.
   * @param sampleCount The count of the sample (weight of the sample).
   */
  def add(offset: Int, featBinId: Int, label: Double, sampleCount: Int): Unit = {
    val binPos = featBinId * 3

    val sumPos = offset + binPos
    val sqrSumPos = sumPos + 1
    val countPos = sqrSumPos + 1

    val sampleCountInDouble = sampleCount.toDouble
    val labelSum = label * sampleCountInDouble
    val sqrLabelSum = label * labelSum

    binStats(sumPos) += labelSum
    binStats(sqrSumPos) += sqrLabelSum
    binStats(countPos) += sampleCountInDouble
  }

  /**
   * Add bin values in binArraySrc (from srcPos) to binArrayDst (from dstPos).
   * @param binArraySrc Array containing source bin stat values.
   * @param srcPos Start position for source.
   * @param binArrayDst Array containing destination bin stat values.
   * @param dstPos Start position for destination.
   * @param numBins Number of bin values to add.
   */
  private def addBins(binArraySrc: Array[Double], srcPos: Int, binArrayDst: Array[Double], dstPos: Int, numBins: Int): Unit = {
    var bi = 0
    while (bi < numBins) {
      val bOffset = bi * 3

      val srcSumPos = srcPos + bOffset
      val srcSqrSumPos = srcSumPos + 1
      val srcCountPos = srcSqrSumPos + 1

      val dstSumPos = dstPos + bOffset
      val dstSqrSumPos = dstSumPos + 1
      val dstCountPos = dstSqrSumPos + 1

      binArrayDst(dstSumPos) += binArraySrc(srcSumPos)
      binArrayDst(dstSqrSumPos) += binArraySrc(srcSqrSumPos)
      binArrayDst(dstCountPos) += binArraySrc(srcCountPos)

      bi += 1
    }
  }

  /**
   * Subtract numBin values in binArraySrc (from srcPos) from binArrayDst (from dstPos).
   * @param binArraySrc Array containing source values.
   * @param srcPos Start position for source.
   * @param binArrayDst Array containing destination values.
   * @param dstPos Start position for destination.
   * @param numBins Number of bin values to add.
   */
  private def subtractBins(binArraySrc: Array[Double], srcPos: Int, binArrayDst: Array[Double], dstPos: Int, numBins: Int): Unit = {
    var bi = 0
    while (bi < numBins) {
      val bOffset = bi * 3

      val srcSumPos = srcPos + bOffset
      val srcSqrSumPos = srcSumPos + 1
      val srcCountPos = srcSqrSumPos + 1

      val dstSumPos = dstPos + bOffset
      val dstSqrSumPos = dstSumPos + 1
      val dstCountPos = dstSqrSumPos + 1

      binArrayDst(dstSumPos) -= binArraySrc(srcSumPos)
      binArrayDst(dstSqrSumPos) -= binArraySrc(srcSqrSumPos)
      binArrayDst(dstCountPos) -= binArraySrc(srcCountPos)

      bi += 1
    }
  }

  /**
   * Merge the stat values in this one with another one into this one.
   * @param another The other stat array object.
   */
  def mergeInPlace(another: BinStatisticsArray): BinStatisticsArray = {
    var i = 0
    while (i < binStats.length) {
      binStats(i) += another.asInstanceOf[RegressionStatisticsArray].binStats(i)
      i += 1
    }

    this
  }

  /**
   * Compute the weighted variance from three values (sum, square sum and count).
   * @param binStats An array of three values (sum, square sum and count).
   * @return Variance.
   */
  private def computeVariance(binStats: Array[Double], offset: Int = 0): Double = {
    val count = binStats(offset + 2)

    if (count > 0.0) {
      val avg = binStats(offset) / count
      binStats(offset + 1) / count - avg * avg
    } else {
      0.0
    }
  }

  /**
   * Summarize a series of bins. I.e., get prediction, impurity and BinStatisticsArray that summarizes the series into single summarizer bin.
   * This should be used to compute prediction, impurity of a node.
   * @param offset Offset from which to do summarizing.
   * @param numBins Number of bins to summarize over.
   * @return Prediction, impurity, single summarizer bin, and weight sum.
   */
  def getSummaryOverBins(offset: Int, numBins: Int): (Double, Double, BinStatisticsArray, Long) = {
    var binId = 0
    var summaryStats = Array.fill[Double](3)(0.0)
    while (binId < numBins) {
      val binOffset = offset + binId * 3
      addBins(binStats, binOffset, summaryStats, 0, 1)
      binId += 1
    }

    (summaryStats(0) / summaryStats(2), computeVariance(summaryStats), RegressionStatisticsArray(summaryStats), summaryStats(2).toLong)
  }

  /**
   * Compute split weighted variances.
   * @param featBinStats An array of bins and their stats in one-dimensional array.
   * @param weightSum The total weight.
   * @return A tuple of split impurity and bin weights
   */
  private def computeSplitVariance(featBinStats: Array[Double], weightSum: Long): (Double, Array[Long]) = {
    val numBins = featBinStats.length / 3
    var splitVariance = 0.0
    val binWeights = Array.fill[Long](numBins)(0)
    var binId = 0
    while (binId < numBins) {
      val binOffset = binId * 3
      var binWeight: Long = featBinStats(binOffset + 2).toLong
      binWeights(binId) = binWeight
      splitVariance += binWeight.toDouble / weightSum.toDouble * computeVariance(featBinStats, binOffset)
      binId += 1
    }

    (splitVariance, binWeights)
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
      val featBinStats = binStats.slice(offset, offset + numBins * 3)
      val (featSplitImpurity, binWeights) = computeSplitVariance(featBinStats, weightSum)

      bestFeatSplitImpurity = featSplitImpurity
      bestFeatSplitBinId = numBins // For categorical features, we just put the number of bins, instead of split bin Id.
      bestFeatSplitWeights = binWeights
    } else {
      // If it's a numeric feature, we'll scan through
      // each bin as a split point.
      val splitBinStats = Array.fill[Double](6)(0)
      summaryStatsArray.asInstanceOf[RegressionStatisticsArray].binStats.copyToArray(splitBinStats, 3)

      var binIdToSplit = 1
      while (binIdToSplit < numBins) {
        val prevBinId = binIdToSplit - 1
        val binOffset = prevBinId * 3
        addBins(binStats, offset + binOffset, splitBinStats, 0, 1)
        subtractBins(binStats, offset + binOffset, splitBinStats, 3, 1)
        val (featSplitImpurity, featSplitWeights) = computeSplitVariance(splitBinStats, weightSum)

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
 * Keeps track of node statistics for computing variance. This will also select random features for each node.
 * @param nodeSplitLookup We use the scheduled node splits to find out the nodes whose bin statistics we want to store in this object.
 * @param numBinsPerFeature The count of bins for each feature.
 * @param treeSeeds Random seeds to use for each tree when selecting a random set of features.
 * @param mtry The number of random features per node.
 */
case class VarianceStatistics(
    @transient private val nodeSplitLookup: ScheduledNodeSplitLookup,
    numBinsPerFeature: Array[Int],
    treeSeeds: Array[Int],
    mtry: Int) extends AggregatedStatistics(nodeSplitLookup, numBinsPerFeature, treeSeeds, mtry, new RegressionStatisticsArrayBuilder()) {

  /**
   * Get the bin statisitcs (sum, square sum and count). This is slow because of linear lookup of feature Id (shouldn't be used other than for some testing purposes).
   * @param treeId The Id of the tree
   * @param nodeId The Id of the node
   * @param featureId The Id of the feature
   * @param binId The Id of the bin
   * @return A tuple of (sum, square sum, count).
   */
  private[spark_ml] def getBinStats(
    treeId: Int,
    nodeId: Int,
    featureId: Int,
    binId: Int): (Double, Double, Double) = {
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
    val binStats = binStatsArray.asInstanceOf[RegressionStatisticsArray].binStats
    (binStats(offset + binId * 3), binStats(offset + binId * 3 + 1), binStats(offset + binId * 3 + 2))
  }
}
