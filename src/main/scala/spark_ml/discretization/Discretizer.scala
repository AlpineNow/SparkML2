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

import scala.reflect.ClassTag

import org.apache.spark.rdd.RDD
import spark_ml.util.DiscretizedFeatureHandler

/**
 * Available discretization types.
 */
object DiscType extends Enumeration {
  type DiscType = Value
  val EqualFrequency = Value(0)
  val EqualWidth = Value(1)
  val MinimumEntropy = Value(2)
  val MinimumVariance = Value(3)
}

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

  override def toString: String = {
    "[" + lower.toString + "," + upper.toString + ")"
  }
}

/**
 * For both categorical and numerical bins.
 */
trait Bins extends Serializable {
  /**
   * Get the number of bins (including the missing-value bin) in case of numeric
   * bins.
   * @return The number of bins.
   */
  def getCardinality: Int

  /**
   * Given the raw feature value, find the bin Id.
   * @param value Raw feature value.
   * @return The corresponding bin Id.
   */
  def findBinIdx(value: Double): Int
}

/**
 * An exception to throw n case a categorical feature value is not an integer
 * value (e.g., can't be 1.1. or 2.4).
 * @param msg String message to include in the exception.
 */
case class InvalidCategoricalValueException(msg: String) extends Exception(msg)

/**
 * An exception to throw if the cardinality of a feature exceeds the limit.
 * @param msg String message to include.
 */
case class CardinalityOverLimitException(msg: String) extends Exception(msg)

/**
 * An exception to throw if the label has unexpected values.
 * @param msg String message to include.
 */
case class InvalidLabelException(msg: String) extends Exception(msg)

/**
 * Numeric bins.
 * @param bins An array of ordered non-NaN numeric bins.
 * @param missingValueBinIdx The optional bin Id for the NaN values.
 */
case class NumericBins(
  bins: Seq[NumericBin],
  missingValueBinIdx: Option[Int] = None) extends Bins {
  /**
   * The cardinality of the bins (the number of bins).
   * @return The number of bins.
   */
  def getCardinality: Int =
    bins.length + (if (missingValueBinIdx.isDefined) 1 else 0)

  /**
   * Find the index of the bin that the value belongs to.
   * @param value The numeric value we want to search.
   * @return The index of the bin that contains the given numeric value.
   */
  def findBinIdx(value: Double): Int = {
    if (value.isNaN) {
      missingValueBinIdx.get
    } else {
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
  }
}

/**
 * For categorical bins, the raw feature value is simply the categorical bin Id.
 * We expect the categorical values to go from 0 to (Cardinality - 1) in an
 * incremental fashion.
 * @param cardinality The cardinality of the feature.
 */
case class CategoricalBins(cardinality: Int) extends Bins {
  /**
   * @return The number of bins.
   */
  def getCardinality: Int = {
    cardinality
  }

  /**
   * Find the bin Id.
   * @param value Raw feature value.
   * @return The corresponding bin Id.
   */
  def findBinIdx(value: Double): Int = {
    if (value.toInt.toDouble != value) {
      throw InvalidCategoricalValueException(
        value + " is not a valid categorical value."
      )
    }

    if (value >= cardinality) {
      throw CardinalityOverLimitException(
        value + " is above the cardinality of this feature " + cardinality
      )
    }

    value.toInt
  }
}

object Discretizer {
  /**
   * Transform the features in the given labeled point row into an array of Bin
   * IDs that could be Unsigned Byte or Unsigned Short.
   * @param featureBins A sequence of feature bin definitions.
   * @param featureHandler A handler for discretized features (unsigned
   *                       Byte/Short).
   * @param row The labeled point row that we want to transform.
   * @return A transformed array of features.
   */
  private def transformFeatures[@specialized(Byte, Short) T: ClassTag](
    featureBins: Seq[Bins],
    featureHandler: DiscretizedFeatureHandler[T])(row: (Double, Array[Double])): Array[T] = {
    val (_, features) = row
    features.zipWithIndex.map {
      case (featureVal, featIdx) => featureHandler.convertToType(featureBins(featIdx).findBinIdx(featureVal))
    }
  }

  /**
   * Transform the features of the labeled data RDD into bin Ids of either
   * unsigned Byte/Short.
   * @param input An RDD of Double label and Double feature values.
   * @param featureBins A sequence of feature bin definitions.
   * @param featureHandler A handler for discretized features (unsigned
   *                       Byte/Short).
   * @return A new RDD that has all the features transformed into Bin Ids.
   */
  def transformFeatures[@specialized(Byte, Short) T: ClassTag](
    input: RDD[(Double, Array[Double])],
    featureBins: Seq[Bins],
    featureHandler: DiscretizedFeatureHandler[T]): RDD[Array[T]] = {
    input.map(transformFeatures(featureBins, featureHandler))
  }
}
