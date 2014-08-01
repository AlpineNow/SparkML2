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

package spark_ml.discretization

import org.apache.spark.rdd.RDD
import org.apache.spark.util.MutablePair
import scala.collection.mutable

/**
 * Generate equal width bins for numeric features.
 * For categorical features, it's one to one mapping.
 * This will throw an exception if a categorical feature has a higher cardinality than the MaxNumBins option.
 */
object EqualWidthDiscretizer extends Discretizer {
  override def discretizeFeatures(
    input: RDD[(Double, Array[Double])],
    categoricalFeatureIndices: Set[Int],
    labelIsCategorical: Boolean,
    config: Map[String, String]): (Double, Array[Bins]) = {
    // Numeric feature options.
    val numBins = config(StringConstants.NumBins_Numeric).toInt

    // Categorical feature options.
    val maxCardinality = config(StringConstants.MaxCardinality_Categoric).toInt

    val numFeatures = input.first()._2.length

    // Find the maximum and the minimum for each feature.
    val minMaxValues = input.aggregate((Double.NegativeInfinity, Array.fill[(MutablePair[Double, Double])](numFeatures)(MutablePair[Double, Double](Double.PositiveInfinity, Double.NegativeInfinity))))(
      (partitionMinMaxValues, row) => {
        val label = row._1
        val features = row._2
        var featIdx = 0
        while (featIdx < numFeatures) {
          val value = features(featIdx)

          // If it's a categorical feature value, make sure that it's a non-negative integer value.
          // Also, the maximum number should be smaller than the maximum cardinality.
          if (categoricalFeatureIndices.contains(featIdx)) {
            if (value < 0.0 || value.toInt.toDouble != value) {
              throw InvalidCategoricalValueException(value + " is not a valid categorical value.")
            }

            if (value >= maxCardinality) {
              throw CardinalityOverLimitException("The categorical feature " + featIdx + " has a cardinality that exceeds the limit " + maxCardinality)
            }
          }

          val curMin = partitionMinMaxValues._2(featIdx)._1
          val curMax = partitionMinMaxValues._2(featIdx)._2
          partitionMinMaxValues._2(featIdx).update(math.min(curMin, value), math.max(curMax, value))

          featIdx += 1
        }

        if (labelIsCategorical && (label < 0.0 || label.toInt.toDouble != label)) {
          throw InvalidCategoricalValueException(label + " is not a valid categorical label.")
        }

        (math.max(label, partitionMinMaxValues._1), partitionMinMaxValues._2)
      },

      (minMaxValuesA, minMaxValuesB) => {
        var featIdx = 0
        while (featIdx < numFeatures) {
          val aMin = minMaxValuesA._2(featIdx)._1
          val aMax = minMaxValuesA._2(featIdx)._2
          val bMin = minMaxValuesB._2(featIdx)._1
          val bMax = minMaxValuesB._2(featIdx)._2
          minMaxValuesA._2(featIdx).update(math.min(aMin, bMin), math.max(aMax, bMax))
          featIdx += 1
        }

        (math.max(minMaxValuesA._1, minMaxValuesB._1), minMaxValuesA._2)
      }
    )

    // Now, let's create equi-width numeric bins.
    var featIdx = 0
    val featureBins = new mutable.ArrayBuffer[Bins]()
    while (featIdx < numFeatures) {
      val featMax = minMaxValues._2(featIdx)._2
      val featMin = minMaxValues._2(featIdx)._1
      if (!categoricalFeatureIndices.contains(featIdx)) {
        if (featMax == featMin) {
          // In this case, only one bin necessary.
          featureBins += NumericBins(Array[NumericBin](NumericBin(Double.NegativeInfinity, Double.PositiveInfinity)))
        } else {
          // Otherwise, we get the exact number of bins.
          val binWidth = (featMax - featMin) / numBins.toDouble
          var binId = 0
          val bins = new Array[NumericBin](numBins)
          var curBinLower = Double.NegativeInfinity
          var curBinUpper = Double.NegativeInfinity
          while (binId < numBins) {
            curBinUpper = featMin + (binId + 1).toDouble * binWidth
            if (binId == (numBins - 1)) {
              curBinUpper = Double.PositiveInfinity
            }

            bins(binId) = NumericBin(curBinLower, curBinUpper)
            curBinLower = curBinUpper
            binId += 1
          }

          featureBins += NumericBins(bins)
        }
      } else {
        // For categorical features, the maximum value + 1 is the cardinality.
        featureBins += CategoricalBins(featMax.toInt + 1)
      }

      featIdx += 1
    }

    (minMaxValues._1, featureBins.toArray)
  }
}
