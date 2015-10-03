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

import breeze.numerics.log2
import spark_ml.util.DiscretizedFeatureHandler
import spire.implicits._

/**
 * Information gain node statistics.
 * @param treeId Id of the tree that the node belongs to.
 * @param nodeId Id of the node.
 * @param nodeDepth Depth of the node.
 * @param statsArray The actual statistics array.
 * @param mtryFeatures Selected feature descriptions.
 * @param numElemsPerBin Number of statistical elements per feature bin.
 */
class InfoGainNodeStats(
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
        val featId = mtryFeatures(i).featId
        val featOffset = mtryFeatures(i).offset
        val binId = featureHandler.convertToInt(features(featId))
        statsArray(featOffset + binId * numElemsPerBin + label.toInt) +=
          sampleCnt
      }
    )
  }

  /**
   * Calculate the node values from the given class distribution. This also
   * includes calculating the entropy.
   * @param classWeights The class weight distribution.
   * @param offset Starting offset.
   * @param output Where the output will be stored.
   * @return Node values.
   */
  override def calculateNodeValues(
    classWeights: Array[Double],
    offset: Int,
    output: PreallocatedNodeValues): PreallocatedNodeValues = {
    var weightSum: Double = 0.0
    var prediction: Double = 0.0
    var maxClassWeight: Double = 0.0
    var entropy: Double = 0.0

    // Determine weightSum and prediction.
    cfor(0)(_ < numElemsPerBin, _ + 1)(
      labelId => {
        val weight = classWeights(offset + labelId)
        output.sumStats(labelId) = weight
        if (maxClassWeight < weight) {
          maxClassWeight = weight
          prediction = labelId
        }

        weightSum += weight
      }
    )

    // Compute entropy.
    cfor(0)(_ < numElemsPerBin, _ + 1)(
      labelId => {
        val weight = classWeights(offset + labelId)
        if (weight > 0.0) {
          val prob = weight.toDouble / weightSum
          entropy -= prob * log2(prob)
        }
      }
    )

    output.prediction = prediction
    output.addendum = maxClassWeight / weightSum // Probability of the class.
    output.weight = weightSum
    output.impurity = entropy

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
      binId => {
        val binOffset = offset + binId * numElemsPerBin
        // Sum all the label weights per bin.
        cfor(0)(_ < numElemsPerBin, _ + 1)(
          labelId => output(binId) += statsArray(binOffset + labelId)
        )
      }
    )

    output
  }

  /**
   * Calculate the label average for the given bin. This is only meaningful for
   * binary classifications.
   * @param statsArray Stats array.
   * @param binOffset Offset to the bin.
   * @return The label average for the bin.
   */
  override def getBinLabelAverage(
    statsArray: Array[Double],
    binOffset: Int): Double = {
    var labelSum = 0.0
    var weightSum = 0.0
    cfor(0)(_ < numElemsPerBin, _ + 1)(
      labelId => {
        val labelWeight = statsArray(binOffset + labelId)
        labelSum += labelId.toDouble * labelWeight
        weightSum += labelWeight
      }
    )

    labelSum / weightSum
  }

  /**
   * This is for classification, so return true.
   * @return true
   */
  override def forClassification: Boolean = true
}
