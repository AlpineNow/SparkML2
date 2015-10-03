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

import spark_ml.util.DiscretizedFeatureHandler
import spire.implicits._

/**
 * Variance node statistics.
 * @param treeId Id of the tree that the node belongs to.
 * @param nodeId Id of the node.
 * @param nodeDepth Depth of the node.
 * @param statsArray The actual statistics array.
 * @param mtryFeatures Selected feature descriptions.
 * @param numElemsPerBin Number of statistical elements per feature bin.
 */
class VarianceNodeStats(
  treeId: Int,
  nodeId: Int,
  nodeDepth: Int,
  statsArray: Array[Double],
  mtryFeatures: Array[SelectedFeatureInfo],
  numElemsPerBin: Int) extends NodeStats(
  treeId = treeId,
  nodeId = nodeId,
  nodeDepth = nodeDepth,
  statsArray = statsArray,
  mtryFeatures = mtryFeatures,
  numElemsPerBin = numElemsPerBin) {

  /**
   * Add statistics related to a sample (label and features).
   * @param label Label of the sample.
   * @param features Features of the sample.
   * @param sampleCnt Sample count of the sample.
   * @param featureHandler Feature type handler.
   * @tparam T Feature type (Byte or Short).
   */
  override def addSample[@specialized(Byte, Short) T](
    label: Double,
    features: Array[T],
    sampleCnt: Int,
    featureHandler: DiscretizedFeatureHandler[T]): Unit = {
    val mtry = mtryFeatures.length
    cfor(0)(_ < mtry, _ + 1)(
      i => {
        // Add to the bins of all the selected features.
        val featId = mtryFeatures(i).featId
        val featOffset = mtryFeatures(i).offset
        val binId = featureHandler.convertToInt(features(featId))

        val binOffset = featOffset + binId * numElemsPerBin

        val sampleCntInDouble = sampleCnt.toDouble
        val labelSum = label * sampleCntInDouble
        val labelSqrSum = label * labelSum

        statsArray(binOffset) += labelSum
        statsArray(binOffset + 1) += labelSqrSum
        statsArray(binOffset + 2) += sampleCntInDouble
      }
    )
  }

  /**
   * Calculate the node values from the summary stats. Also computes variance.
   * @param sumStats Summary stats (label sum, label sqr sum, count).
   * @param offset Starting offset.
   * @param output Where the output will be stored.
   * @return Node values.
   */
  override def calculateNodeValues(
    sumStats: Array[Double],
    offset: Int,
    output: PreallocatedNodeValues): PreallocatedNodeValues = {
    val prediction = sumStats(offset) / sumStats(offset + 2)
    val variance =
      sumStats(offset + 1) / sumStats(offset + 2) - prediction * prediction

    output.prediction = prediction
    output.addendum = variance
    output.weight = sumStats(offset + 2)
    output.impurity = variance
    output.sumStats(0) = sumStats(offset)
    output.sumStats(1) = sumStats(offset + 1)
    output.sumStats(2) = sumStats(offset + 2)

    output
  }

  /**
   * Get bin weights.
   * @param statsArray Stats array.
   * @param offset Start offset.
   * @param numBins Number of bins.
   * @param output This is where the weights will be stored.
   * @return Returns the same output that was passed in.
   */
  override def getBinWeights(
    statsArray: Array[Double],
    offset: Int,
    numBins: Int,
    output: Array[Double]): Array[Double] = {
    cfor(0)(_ < numBins, _ + 1)(
      binId => output(binId) = statsArray(offset + binId * numElemsPerBin + 2)
    )

    output
  }

  /**
   * Calculate the label average for the given bin.
   * @param statsArray Stats array.
   * @param binOffset Offset to the bin.
   * @return The label average for the bin.
   */
  override def getBinLabelAverage(
    statsArray: Array[Double],
    binOffset: Int): Double = {
    statsArray(binOffset) / statsArray(binOffset + 2)
  }

  /**
   * This is for regression, so return false.
   * @return false
   */
  override def forClassification: Boolean = false
}
