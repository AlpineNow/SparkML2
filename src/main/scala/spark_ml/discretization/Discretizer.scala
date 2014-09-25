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

import spire.implicits._

/**
 * A numeric bin with a lower bound and an upper bound.
 * The lower bound is inclusive. The upper bound is exclusive.
 * @param lower Lower bound
 * @param upper Upper bound
 */
case class NumericBin(lower: Double, upper: Double) {
  def contains(value: Double): Boolean = {
    value >= lower && value < upper
  }
}

/**
 * For both categorical and numerical bins.
 */
trait Bins {
  def getCardinality: Int
  def findBinIdx(value: Double): Int
  def getMissingValueBinIdx: Int
}

/**
 * In case a categorical feature value is not an integer value (e.g., can't be 1.1. or 2.4).
 * @param msg String message to include in the exception.
 */
case class InvalidCategoricalValueException(msg: String) extends Exception(msg)

/**
 * An exception to throw if the cardinality of a feature exceeds the limit.
 * @param msg String message to include.
 */
case class CardinalityOverLimitException(msg: String) extends Exception(msg)

/**
 * For categorical bins, we simply map the value to the integer value.
 * We expect the categorical values to go from 0 to Cardinality in an incremental fashion.
 * @param cardinality The cardinality of the feature.
 */
case class CategoricalBins(cardinality: Int, missingValueBinIdx: Int = -1) extends Bins {
  def getCardinality: Int = {
    cardinality
  }

  def findBinIdx(value: Double): Int = {
    if (value.toInt.toDouble != value) {
      throw InvalidCategoricalValueException(value + " is not a valid categorical value.")
    }

    if (value >= cardinality) {
      throw CardinalityOverLimitException(value + " is above the cardinality of this feature " + cardinality)
    }

    value.toInt
  }

  def getMissingValueBinIdx: Int = missingValueBinIdx
}

/**
 * A discretized type.
 * @param bins Bins of the discretization.
 */
case class NumericBins(bins: Array[NumericBin], missingValueBinIdx: Int = -1) extends Bins {
  /**
   * The cardinality of the bins (the number of bins).
   * @return The number of bins.
   */
  def getCardinality: Int = {
    bins.length
  }

  /**
   * Find the index of the bin that the value belongs to.
   * @param value The numeric value we want to search.
   * @return The index of the bin that contains the given numeric value.
   */
  def findBinIdx(value: Double): Int = {
    // Binary search.
    var s = 0
    var e = bins.length - 1
    var cur = (s + e) / 2
    var found = bins(cur).contains(value)
    while (!found) {
      if (bins(cur).lower > value) {
        e = cur - 1
      } else if (bins(cur).upper <= value) {
        s = cur + 1
      }

      cur = (s + e) / 2
      found = bins(cur).contains(value)
    }

    cur
  }

  def getMissingValueBinIdx: Int = missingValueBinIdx
}

/**
 * A trait for all the discretizers of continuous types.
 */
trait Discretizer {
  /**
   * Compute bins of features.
   * @param input RDD of labeled point.
   * @param categoricalFeatureIndices The indices of the categorical features. All others are assumed to be numerical.
   * @param labelIsCategorical Indicates whether the label is expected to be categorical.
   * @param config Whatever configuration might be needed for a particular discretizer.
   * @return A pair : The first number is the maximum label value (useful to figure out classification target cardinality). The second one is an array of feature bins.
   */
  def discretizeFeatures(
    input: RDD[(Double, Array[Double])],
    categoricalFeatureIndices: Set[Int],
    labelIsCategorical: Boolean,
    config: Map[String, String]): (Double, Array[Bins])
}

/**
 * In case the BinId of a feature bin is larger than what we want (e.g., 256 for Byte types).
 * @param msg String message to include in the exception.
 */
case class BinIdOutOfRangeException(msg: String) extends Exception(msg)

object Discretizer {
  /**
   * Get the value of an unsigned encoded Byte value.
   * @param byteValue Byte value that encodes an unsigned value.
   * @return An integer
   */
  def readUnsignedByte(byteValue: Byte): Int = {
    byteValue.toInt + 128
  }

  /**
   * Convert the given integer value into an unsigned byte.
   * It won't catch over/under-flows.
   * @param value The value to convert.
   * @return A converted Byte value.
   */
  def convertToUnsignedByte(value: Int): Byte = {
    (value - 128).toByte
  }

  /**
   * Get the value of an unsigned encoded Short value.
   * @param shortValue Short value that encodes an unsigned value.
   * @return An integer
   */
  def readUnsignedShort(shortValue: Short): Int = {
    shortValue.toInt + 32768
  }

  /**
   * Convert the given integer value into an unsigned short.
   * It won't catch over/under-flows.
   * @param value The value to convert.
   * @return A converted Short value.
   */
  def convertToUnsignedShort(value: Int): Short = {
    (value - 32768).toShort
  }

  /**
   * Transform the features in the given labeled point row into an array of Bin IDs that could be Unsigned Byte, Unsigned Short, Unsigned Int.
   * @param row The labeled point row that we want to transform.
   * @param featureBins An array of corresponding feature bins.
   * @return A transformed row.
   */
  private def transformToUnsignedByteBinIds(featureBins: Array[Bins])(row: (Double, Array[Double])): (Double, Array[Byte]) = {
    val label = row._1
    val features = row._2
    val transformed = Array.fill[Byte](features.length)(0)
    val unsignedByteMax = 255

    cfor(0)(_ < features.length, _ + 1)(
      featIdx => {
        val featureVal = features(featIdx)
        val binId = if (featureVal.isNaN) {
          featureBins(featIdx).getMissingValueBinIdx
        } else {
          featureBins(featIdx).findBinIdx(featureVal)
        }

        if (binId > unsignedByteMax) {
          throw BinIdOutOfRangeException("The bin index " + binId + " for the feature " + featIdx + " does not fall within the unsigned Byte range (0 to " + unsignedByteMax + ").")
        }

        transformed(featIdx) = convertToUnsignedByte(binId) // We simulate unsigned by mapping the minimum type value to 0.
      }
    )

    (label, transformed)
  }

  /**
   * Transform the features in the given labeled point row into an array of Bin IDs that could be Unsigned Byte, Unsigned Short, Unsigned Int.
   * @param row The labeled point row that we want to transform.
   * @param featureBins An array of corresponding feature bins.
   * @return A transformed row.
   */
  private def transformToUnsignedShortBinIds(featureBins: Array[Bins])(row: (Double, Array[Double])): (Double, Array[Short]) = {
    val label = row._1
    val features = row._2
    val transformed = Array.fill[Short](features.length)(0)
    val unsignedShortMax = 65535

    cfor(0)(_ < features.length, _ + 1)(
      featIdx => {
        val featureVal = features(featIdx)
        val binId = if (featureVal.isNaN) {
          featureBins(featIdx).getMissingValueBinIdx
        } else {
          featureBins(featIdx).findBinIdx(featureVal)
        }

        if (binId > unsignedShortMax) {
          throw BinIdOutOfRangeException("The bin index " + binId + " for the feature " + featIdx + " does not fall within the unsigned Short range (0 to " + unsignedShortMax + ").")
        }

        transformed(featIdx) = convertToUnsignedShort(binId) // We simulate unsigned by mapping the minimum type value to 0.
      }
    )

    (label, transformed)
  }

  /**
   * Transform the features of the labeled data RDD into unsigned Byte bin Ids.
   * @param input An RDD of Double label and Double feature values.
   * @param featureBins Discretized bins of features.
   * @return A new RDD that has all the features transformed into Bin Ids.
   */
  def transformFeaturesToUnsignedByteBinIds(input: RDD[(Double, Array[Double])], featureBins: Array[Bins]): RDD[(Double, Array[Byte])] = {
    input.map(transformToUnsignedByteBinIds(featureBins))
  }

  /**
   * Transform the features of the labeled data RDD into unsigned Short bin Ids.
   * @param input An RDD of Double label and Double feature values.
   * @param featureBins Discretized bins of features.
   * @return A new RDD that has all the features transformed into Bin Ids.
   */
  def transformFeaturesToUnsignedShortBinIds(input: RDD[(Double, Array[Double])], featureBins: Array[Bins]): RDD[(Double, Array[Short])] = {
    input.map(transformToUnsignedShortBinIds(featureBins))
  }
}
