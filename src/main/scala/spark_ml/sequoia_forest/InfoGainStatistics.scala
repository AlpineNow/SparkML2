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

import spire.implicits._
import breeze.numerics.log2

/**
 * This is used to build a ClassificationStatisticsArray object.
 * @param numClasses The number of classes in the classification problem.
 */
class ClassificationStatisticsArrayBuilder(numClasses: Int) extends BinStatisticsArrayBuilder(numClasses) {
  /**
   * Call this, after adding all the bins, to create an actual BinStatistics array object.
   * @return BinStatisticsArray object
   */
  def createBinStatisticsArray: BinStatisticsArray = {
    new ClassificationStatisticsArray(binStatsBuilder.result(), numClasses)
  }
}

/**
 * This is the wrapper around the stats array.
 * @param binStats The stats array.
 * @param numClasses The number of classes in the classification problem.
 */
class ClassificationStatisticsArray(binStats: Array[Double], numClasses: Int) extends BinStatisticsArray(binStats, numClasses) {
  /**
   * Add a sample's label and count to the bin statistics of a feature of a tree's node.
   * @param offset The offset into the feature within this object.
   * @param featBinId The bin ID that we want to add to.
   * @param label The label of the sample.
   * @param sampleCount The count of the sample (weight of the sample).
   */
  override def add(offset: Int, featBinId: Int, label: Double, sampleCount: Int): Unit = {
    binStats(offset + featBinId * numClasses + label.toInt) += sampleCount
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
    var maxClassWeight: Double = 0.0
    val classWeights = Array.fill[Double](numClasses)(0.0)
    var weightSum: Double = 0
    var prediction: Double = 0.0
    cfor(0)(_ < numBins, _ + 1)(
      binId => {
        cfor(0)(_ < numClasses, _ + 1)(
          labelId => {
            val weight = binStats(offset + binId * numClasses + labelId)
            classWeights(labelId) += weight
            weightSum += weight

            if (maxClassWeight < classWeights(labelId)) {
              maxClassWeight = classWeights(labelId)
              prediction = labelId.toDouble
            }
          }
        )
      }
    )

    NodeSummary(
      prediction = prediction,
      impurity = computeEntropy(classWeights, weightSum),
      summaryStatistics = new ClassificationStatisticsArray(classWeights, numClasses),
      weightSum = weightSum)
  }

  /**
   * Use this to get the weight (number of samples) in a bin.
   * @param offset Offset to the feature of the node.
   * @param binId The bin id.
   * @return The weight of the bin.
   */
  override def getBinWeight(offset: Int, binId: Int): Double = {
    var binWeight = 0.0
    cfor(0)(_ < numClasses, _ + 1)(labelId => binWeight += binStats(offset + binId * numClasses + labelId))
    binWeight
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
    val numGroups = splitGroupStats.length / numClasses
    var splitEntropy = 0.0
    val groupWeights = Array.fill[Double](numGroups)(0.0)
    cfor(0)(_ < numGroups, _ + 1)(
      groupId => {
        val groupOffset = groupId * numClasses
        var groupWeight: Double = 0.0
        cfor(0)(_ < numClasses, _ + 1)(labelId => groupWeight += splitGroupStats(groupOffset + labelId))
        groupWeights(groupId) = groupWeight
        splitEntropy += groupWeight / weightSum * computeEntropy(splitGroupStats, groupWeight, groupOffset)
      }
    )

    (splitEntropy, groupWeights)
  }

  /**
   * Call this function to get a sorted array (according to some criteria determined by the child class) of a categorical feature bin Ids.
   * @param numBins number of bins in this categorical feature.
   * @param offset offset to the feature of the node.
   * @param randGen Random number generator in case it's needed.
   */
  override def sortCategoricalFeatureBins(numBins: Int, offset: Int, randGen: scala.util.Random): mutable.ArrayBuffer[Int] = {
    // In order to perform binary splits on categorical features, we need to sort the bin Ids based on some criteria.
    // We also want to exclude all the zero weight category values.
    // Go through the bins and compute the criteria.
    if (numClasses == 2) { // For binary classifications, we can sort by the proportions of 1's.
      val binIds_notEmpty = new mutable.ArrayBuffer[Int]()
      val binCriteria = Array.fill[Double](numBins)(0.0)
      cfor(0)(_ < numBins, _ + 1)(
        binId => {
          val binOffset = offset + binId * 2
          val numZeroes = binStats(binOffset)
          val numOnes = binStats(binOffset + 1)
          val binWeight = numZeroes + numOnes
          if (binWeight > 0.0) {
            binIds_notEmpty += binId
            binCriteria(binId) = numOnes / binWeight
          }
        }
      )

      // Now sort the non empty bins according to the proportion criteria.
      binIds_notEmpty.sorted(Ordering.by[Int, Double](binCriteria(_)))
    } else {
      val binIds_notEmpty = new mutable.ArrayBuffer[Int]()
      cfor(0)(_ < numBins, _ + 1)(
        binId => {
          val binOffset = offset + binId * numClasses
          var binWeight = 0.0
          cfor(0)(_ < numClasses, _ + 1)(labelId => binWeight += binStats(binOffset + labelId))
          if (binWeight > 0.0) {
            binIds_notEmpty += binId
          }
        }
      )

      // For multi-class classification, the sorting is done through a random order.
      binIds_notEmpty.sorted(Ordering.by[Int, Double](_ => randGen.nextDouble()))
    }
  }

  /**
   * Compute the entropy for the class distribution.
   * @param classWeights Weights for each class.
   * @param weightSum Total weight.
   * @param startIdx The starting offset within the classWeights array.
   * @return Entropy
   */
  private def computeEntropy(
    classWeights: Array[Double],
    weightSum: Double,
    startIdx: Int = 0): Double = {
    var entropy = 0.0
    cfor(0)(_ < numClasses, _ + 1)(
      labelId => {
        val w = classWeights(startIdx + labelId)
        if (w > 0.0) {
          val prob = w.toDouble / weightSum.toDouble
          entropy -= prob * log2(prob)
        }
      }
    )

    entropy
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
class InfoGainStatistics(
    @transient private val nodeSplitLookup: ScheduledNodeSplitLookup,
    numBinsPerFeature: Array[Int],
    treeSeeds: Array[Int],
    mtry: Int,
    numClasses: Int) extends AggregatedStatistics(nodeSplitLookup, new ClassificationStatisticsArrayBuilder(numClasses), numBinsPerFeature, treeSeeds, mtry) {

  /**
   * Used for testing purposes.
   * Get the bin weight. This is slow because of linear lookup of feature Id.
   * @param treeId The Id of the tree
   * @param nodeId The Id of the node
   * @param featureId The Id of the feature
   * @param binId The Id of the bin
   * @param label The label whose weight we want
   * @return The weight of the label
   */
  private[spark_ml] def getBinLabelWeight(
    treeId: Int,
    nodeId: Int,
    featureId: Int,
    binId: Int,
    label: Int): Double = {
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
    binStatsArray.binStats(offset + binId * numClasses + label)
  }
}
