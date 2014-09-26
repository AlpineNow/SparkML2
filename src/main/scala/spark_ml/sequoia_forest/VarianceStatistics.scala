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

import spire.implicits._

import spark_ml.util.Sorting._

/**
 * This is used to build a RegressionStatisticsArray object.
 */
class RegressionStatisticsArrayBuilder extends BinStatisticsArrayBuilder(3) {
  /**
   * Call this, after adding all the bins, to create an actual BinStatistics array object.
   * @return BinStatisticsArray object
   */
  def createBinStatisticsArray: BinStatisticsArray = {
    new RegressionStatisticsArray(binStatsBuilder.result())
  }
}

/**
 * This is the wrapper around the stats array (contains sum, square sum and count).
 * @param binStats The stats array.
 */
class RegressionStatisticsArray(binStats: Array[Double]) extends BinStatisticsArray(binStats, 3) {
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
   * Summarize a sequence of bins.
   * This is equivalent to computing various node statistics based on one feature statistics.
   * I.e., get the prediction, the impurity, summary BinStatisticsArray and the weight.
   * Different for different split criteria.
   * @param offset Offset from which to do summarizing.
   * @param numBins Number of bins to summarize over.
   * @return A summary object that contains prediction, impurity, summary array, and weight sum.
   */
  override def getNodeSummary(offset: Int, numBins: Int): NodeSummary = {
    val summaryStats = Array.fill[Double](3)(0.0)
    cfor(0)(_ < numBins, _ + 1)(binId => addArrays(binStats, offset + binId * 3, summaryStats, 0, 3))

    NodeSummary(
      prediction = summaryStats(0) / summaryStats(2),
      impurity = computeVariance(summaryStats),
      summaryStatistics = new RegressionStatisticsArray(summaryStats),
      weightSum = summaryStats(2))
  }

  /**
   * Use this to get the weight (number of samples) in a bin.
   * @param offset Offset to the feature of the node.
   * @param binId The bin id.
   * @return The weight of the bin.
   */
  override def getBinWeight(offset: Int, binId: Int): Double = {
    binStats(offset + binId * 3 + 2)
  }

  /**
   * Use this to compute split impurity of the given group stats array (into each group).
   * @param splitGroupStats Group stats with N groups. This function should try to split the groups in N groups.
   * @param weightSum The total weight of all the groups.
   * @return Split impurity and the weights of the split groups.
   */
  override def computeSplitImpurity(
    splitGroupStats: Array[Double],
    weightSum: Double): (Double, Array[Double]) = {
    val numGroups = splitGroupStats.length / 3
    var splitVariance = 0.0
    val groupWeights = Array.fill[Double](numGroups)(0.0)
    cfor(0)(_ < numGroups, _ + 1)(
      groupId => {
        val groupOffset = groupId * 3
        val groupWeight = splitGroupStats(groupOffset + 2)
        groupWeights(groupId) = groupWeight
        splitVariance += groupWeight / weightSum * computeVariance(splitGroupStats, groupOffset)
      }
    )

    (splitVariance, groupWeights)
  }

  /**
   * Call this function to get a sorted array (according to some criteria determined by the child class) of a categorical feature bin Ids.
   * @param numBins number of bins in this categorical feature.
   * @param offset offset to the feature of the node.
   * @param randGen not used for this.
   */
  override def sortCategoricalFeatureBins(numBins: Int, offset: Int, randGen: scala.util.Random): Array[Int] = {
    // In order to perform binary splits on categorical features, we need to sort the bin Ids by the average.
    // We also want to exclude all the zero weight category values.
    // Go through the bins and compute the averages.
    val binIds_notEmpty = Array.fill[Int](numBins)(0)
    var curBinIdCursor = 0
    val binCriteria = Array.fill[Double](numBins)(0.0)
    cfor(0)(_ < numBins, _ + 1)(
      binId => {
        val binOffset = offset + binId * 3
        val binWeight = binStats(binOffset + 2)
        if (binWeight > 0.0) {
          binIds_notEmpty(curBinIdCursor) = binId
          curBinIdCursor += 1
          binCriteria(binId) = binStats(binOffset) / binWeight
        }
      }
    )

    // Now sort the non empty bins according to the proportion criteria.
    val toReturn = binIds_notEmpty.slice(0, curBinIdCursor)
    quickSort[Int](toReturn)(Ordering.by[Int, Double](binCriteria(_)))
    toReturn
  }

  /**
   * Compute the variance from three values (sum, square sum and count).
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
}

/**
 * Keeps track of node statistics for computing variance. This will also select random features for each node.
 * @param nodeSplitLookup We use the scheduled node splits to find out the nodes whose bin statistics we want to store in this object.
 * @param numBinsPerFeature The count of bins for each feature.
 * @param treeSeeds Random seeds to use for each tree when selecting a random set of features.
 * @param mtry The number of random features per node.
 */
class VarianceStatistics(
    @transient private val nodeSplitLookup: ScheduledNodeSplitLookup,
    numBinsPerFeature: Array[Int],
    treeSeeds: Array[Int],
    mtry: Int) extends AggregatedStatistics(nodeSplitLookup, new RegressionStatisticsArrayBuilder(), numBinsPerFeature, treeSeeds, mtry) {

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

    // Look for the requested feature.
    var selectedFeatIdx = 0
    while (!found) {
      if (nodeFeatures(selectedFeatIdx) == featureId) {
        found = true
      } else {
        selectedFeatIdx += 1
      }
    }

    val offset = offsetLookup(treeId)(nodeIdx)(selectedFeatIdx)
    val binOffset = offset + binId * 3
    val binStats = binStatsArray.binStats
    (binStats(binOffset), binStats(binOffset + 1), binStats(binOffset + 2))
  }
}
