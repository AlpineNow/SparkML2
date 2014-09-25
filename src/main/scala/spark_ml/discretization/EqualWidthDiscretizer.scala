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
import scala.collection.mutable
import spire.implicits._

/**
 * To store a mutable triple.
 */
case class MutableTriple[@specialized(Int, Long, Double, Char, Boolean /* , AnyRef */ ) T1, @specialized(Int, Long, Double, Char, Boolean /* , AnyRef */ ) T2, @specialized(Int, Long, Double, Char, Boolean /* , AnyRef */ ) T3](var _1: T1, var _2: T2, var _3: T3) {
  /** No-arg constructor for serialization */
  def this() = this(null.asInstanceOf[T1], null.asInstanceOf[T2], null.asInstanceOf[T3])

  /** Updates this triple with new values and returns itself */
  def update(n1: T1, n2: T2, n3: T3): MutableTriple[T1, T2, T3] = {
    _1 = n1
    _2 = n2
    _3 = n3
    this
  }

  override def toString = "(" + _1 + "," + _2 + "," + _3 + ")"
  override def canEqual(that: Any): Boolean = that.isInstanceOf[MutableTriple[_, _, _]]
}

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

    // Find the maximum label value and the minimum, the maximum and the existence of NaN for each feature.
    val minMaxNaNValues = input.aggregate(
      (Double.NegativeInfinity, Array.fill[(MutableTriple[Double, Double, Boolean])](numFeatures)(MutableTriple[Double, Double, Boolean](Double.PositiveInfinity, Double.NegativeInfinity, false))))(
        (partitionMinMaxValues, row) => {
          val label = row._1
          val features = row._2
          cfor(0)(_ < numFeatures, _ + 1)(
            featIdx => {
              val curMin = partitionMinMaxValues._2(featIdx)._1
              val curMax = partitionMinMaxValues._2(featIdx)._2
              val curNaN = partitionMinMaxValues._2(featIdx)._3
              val featValue = features(featIdx)
              if (featValue.isNaN) {
                partitionMinMaxValues._2(featIdx).update(curMin, curMax, true)
              } else {
                // If it's a categorical feature value, make sure that it's a non-negative integer value.
                // Also, the maximum number should be smaller than the maximum cardinality.
                if (categoricalFeatureIndices.contains(featIdx)) {
                  if (featValue < 0.0 || featValue.toInt.toDouble != featValue) {
                    throw InvalidCategoricalValueException(featValue + " is not a valid categorical value.")
                  }

                  if (featValue >= maxCardinality) {
                    throw CardinalityOverLimitException("The categorical feature " + featIdx + " has a cardinality that exceeds the limit " + maxCardinality)
                  }
                }

                partitionMinMaxValues._2(featIdx).update(math.min(curMin, featValue), math.max(curMax, featValue), curNaN)
              }
            }
          )

          if (labelIsCategorical && (label < 0.0 || label.toInt.toDouble != label)) {
            throw InvalidCategoricalValueException(label + " is not a valid categorical label.")
          }

          (math.max(label, partitionMinMaxValues._1), partitionMinMaxValues._2)
        },

        (minMaxValuesA, minMaxValuesB) => {
          cfor(0)(_ < numFeatures, _ + 1)(
            featIdx => {
              val aMin = minMaxValuesA._2(featIdx)._1
              val aMax = minMaxValuesA._2(featIdx)._2
              val bMin = minMaxValuesB._2(featIdx)._1
              val bMax = minMaxValuesB._2(featIdx)._2
              val aNan = minMaxValuesA._2(featIdx)._3
              val bNan = minMaxValuesB._2(featIdx)._3
              minMaxValuesA._2(featIdx).update(math.min(aMin, bMin), math.max(aMax, bMax), aNan || bNan)
            }
          )

          (math.max(minMaxValuesA._1, minMaxValuesB._1), minMaxValuesA._2)
        }
      )

    // Now, let's create equi-width numeric bins.
    val featureBins = new mutable.ArrayBuffer[Bins]()
    cfor(0)(_ < numFeatures, _ + 1)(
      featId => {
        val featMin = minMaxNaNValues._2(featId)._1
        val featMax = minMaxNaNValues._2(featId)._2
        val featNaNExists = minMaxNaNValues._2(featId)._3
        if (!categoricalFeatureIndices.contains(featId)) { // For numeric features.
          if (featMax == featMin) { // A single bin case (unless there's a NaN).
            val missingValueIdx = if (featNaNExists) {
              1
            } else {
              -1
            }

            // In this case, only one bin necessary.
            featureBins += NumericBins(Array[NumericBin](NumericBin(Double.NegativeInfinity, Double.PositiveInfinity)), missingValueIdx)
          } else { // Multiple bins case.
            val numBinsForThis = numBins - (if (featNaNExists) 1 else 0) // If this feature contains NaN, one bin is reserved for it.
            val binWidth = (featMax - featMin) / numBinsForThis.toDouble
            val bins = new Array[NumericBin](numBinsForThis)
            var curBinLower = Double.NegativeInfinity
            var curBinUpper = Double.NegativeInfinity
            cfor(0)(_ < numBinsForThis, _ + 1)(
              binId => {
                curBinUpper = featMin + (binId + 1).toDouble * binWidth
                if (binId == (numBinsForThis - 1)) {
                  curBinUpper = Double.PositiveInfinity
                }

                bins(binId) = NumericBin(curBinLower, curBinUpper)
                curBinLower = curBinUpper
              }
            )

            val missingValueIdx = if (featNaNExists) {
              bins.length
            } else {
              -1
            }

            featureBins += NumericBins(bins, missingValueIdx)
          }
        } else { // For categorical features.
          val missingValueIdx = if (featNaNExists) {
            featMax.toInt + 1
          } else {
            -1
          }

          if (missingValueIdx >= maxCardinality) {
            throw CardinalityOverLimitException("The categorical feature " + featId + " has a cardinality that exceeds the limit " + maxCardinality + " because of NaN.")
          }

          // For categorical features, the maximum value + 1 is the cardinality.
          featureBins += CategoricalBins(featMax.toInt + 1, missingValueIdx)
        }
      }
    )

    (minMaxNaNValues._1, featureBins.toArray)
  }
}
