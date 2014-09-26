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

import scala.collection.mutable

import spire.implicits._

import org.apache.spark.rdd.RDD
import scala.util.Random
import spark_ml.util.RandomSet

/**
 * Generate bins with an approximately equal number of elements in each bin for numeric features.
 * For categorical features, it's one to one mapping.
 * This will throw an exception if a categorical feature has a higher cardinality than the MaxNumBins option.
 */
object EqualFrequencyDiscretizer extends Discretizer {
  private case class PartitionSample(
      numFeatures: Int,
      maxNumericFeatureSamples: Int,
      categoricalFeatureIndices: Set[Int],
      maxCategoricalCardinality: Int,
      labelIsCategorical: Boolean,
      seed: Int) {
    var maxLabelValue: Double = Double.NegativeInfinity
    val numericSamples = mutable.Map[Int, mutable.ArrayBuffer[Double]]()
    val nanFound = mutable.Map[Int, Boolean]()
    val categoricalMaxVals = mutable.Map[Int, Int]()
    var numSamplesSeen = 0

    {
      // Initialize the structures.
      cfor(0)(_ < numFeatures, _ + 1)(
        featId => {
          if (!categoricalFeatureIndices.contains(featId)) {
            numericSamples.put(featId, new mutable.ArrayBuffer[Double]())
          } else {
            categoricalMaxVals.put(featId, 0)
          }

          nanFound.put(featId, false)
        }
      )
    }

    // We don't need to serialize this.
    @transient private var randGen: Random = new Random(seed)

    /**
     * Add a sample of features.
     * This object will automatically determine whether to sample the given features or not.
     * @param sample The feature set we are adding.
     * @return This
     */
    def addSample(sample: (Double, Array[Double])): this.type = {
      val label = sample._1
      val features = sample._2
      val numFeatures = features.length
      var featId = 0
      while (featId < numFeatures) {
        val value = features(featId)
        if (value.isNaN) {
          nanFound.put(featId, true)
        } else {
          if (!categoricalFeatureIndices.contains(featId)) {
            val sampleCount = numericSamples(featId).length
            if (sampleCount < maxNumericFeatureSamples) {
              // If there aren't enough samples, just add the value to the array.
              numericSamples(featId) += value
            } else {
              // If the samples are filled, then we'll sample and replace one of the previously selected values.
              // This does M random samples from a stream.
              // Each new sample from the stream is chosen with the M / N probability where N is the total number of samples seen so far.
              // Then, if the sample is chosen, one of the previously chosen samples is randomly chosen and replaced with the new one.
              // This guarantees a random M samples from a stream.
              val chosenIdx = randGen.nextInt(sampleCount + 1)
              if (chosenIdx < maxNumericFeatureSamples) {
                numericSamples(featId)(chosenIdx) = value
              }
            }
          } else {
            // If it's a categorical feature value, make sure that it's a non-negative integer value.
            // Also, the maximum number should be smaller than the maximum cardinality.
            if (value < 0.0 || value.toInt.toDouble != value) {
              throw InvalidCategoricalValueException(value + " is not a valid categorical value.")
            }

            if (value >= maxCategoricalCardinality) {
              throw CardinalityOverLimitException("The categorical feature " + featId + " has a cardinality that exceeds the limit " + maxCategoricalCardinality + " : " + value.toInt.toString)
            }

            categoricalMaxVals.put(featId, math.max(value.toInt, categoricalMaxVals(featId)))
          }
        }

        featId += 1
      }

      if (labelIsCategorical && (label < 0.0 || label.toInt.toDouble != label)) {
        throw InvalidCategoricalValueException(label + " is not a valid categorical label.")
      }

      maxLabelValue = math.max(label, maxLabelValue)

      numSamplesSeen += 1

      this
    }

    /**
     * Merge samples together in place.
     * @param another Another sample that we want to merge with.
     * @return This
     */
    def mergeInPlace(another: PartitionSample): this.type = {
      val randGen = new Random(seed + another.seed)
      numericSamples.foreach(featId_sample => {
        val featId = featId_sample._1
        val mySample = featId_sample._2
        val mySampleCount = mySample.length

        val theirSample = another.numericSamples(featId)
        val theirSampleCount = theirSample.length
        val totalSampleCount = mySampleCount + theirSampleCount

        val mergedSampleCount = math.min(totalSampleCount, maxNumericFeatureSamples)

        val merged = if (mergedSampleCount < totalSampleCount) {
          val mergedSample = new mutable.ArrayBuffer[Double](mergedSampleCount)
          val selectedSampleIndices = RandomSet.nChooseK(mergedSampleCount, totalSampleCount, randGen)
          cfor(0)(_ < selectedSampleIndices.length, _ + 1)(
            i => {
              val sampleIdx = selectedSampleIndices(i)
              if (sampleIdx < mySampleCount) {
                mergedSample += mySample(sampleIdx)
              } else {
                mergedSample += theirSample(sampleIdx - mySampleCount)
              }
            }
          )

          mergedSample
        } else {
          mySample ++ theirSample
        }

        numericSamples(featId) = merged
      })

      categoricalMaxVals.keys.foreach(featId => categoricalMaxVals(featId) = math.max(categoricalMaxVals(featId), another.categoricalMaxVals(featId)))
      nanFound.keys.foreach(featId => nanFound(featId) = nanFound(featId) || another.nanFound(featId))
      maxLabelValue = math.max(maxLabelValue, another.maxLabelValue)
      this
    }
  }

  override def discretizeFeatures(
    input: RDD[(Double, Array[Double])],
    categoricalFeatureIndices: Set[Int],
    labelIsCategorical: Boolean,
    config: Map[String, String]): (Double, Array[Bins]) = {
    // Numeric feature options.
    val numBins = config(StringConstants.NumBins_Numeric).toInt
    val subSampleCount = config(StringConstants.SubSampleCount_Numeric).toInt

    // Categorical feature options.
    val maxCardinality = config(StringConstants.MaxCardinality_Categoric).toInt

    // Seed for the random number generator.
    val seed = config.getOrElse(StringConstants.RandomSeed, "1").toInt

    val numFeatures = input.first()._2.length

    // We will first find unique value counts for each numeric feature.
    // This will fail if the cardinality is too large for a numeric feature or a categorical feature.
    val overallSample = input.mapPartitionsWithIndex((index, rows) => {
      val partitionSample = PartitionSample(numFeatures, subSampleCount, categoricalFeatureIndices, maxCardinality, labelIsCategorical, seed + index)
      while (rows.hasNext) {
        val row = rows.next()
        partitionSample.addSample(row)
      }

      Array(partitionSample).toIterator
    }).reduce((sampleA, sampleB) => sampleA.mergeInPlace(sampleB))

    // Now, let's create equi-frequency numeric bins.
    val featureBins = new mutable.ArrayBuffer[Bins]()
    cfor(0)(_ < numFeatures, _ + 1)(
      featId => {
        if (!categoricalFeatureIndices.contains(featId)) {
          val numBinsForThis = numBins - (if (overallSample.nanFound(featId)) 1 else 0)
          val minBinWeight = math.ceil(overallSample.numericSamples(featId).length.toDouble / numBinsForThis.toDouble).toInt
          val numericSample = overallSample.numericSamples(featId)
          val sortedNumericSample = numericSample.sorted

          // Create bins from left to right, and we'll move onto a new bin once the current bin is filled.
          // This guarantees that we'll have at most numBins bins and not more.
          val bins = new mutable.ArrayBuffer[NumericBin]()
          var sampleIdx = 0
          var curBinLower = Double.NegativeInfinity
          var curBinUpper = Double.NegativeInfinity
          var curBinWeight = 0
          while (sampleIdx < sortedNumericSample.length) {
            var endReached = sampleIdx == (sortedNumericSample.length - 1)
            val curVal = sortedNumericSample(sampleIdx)
            curBinWeight += 1

            // Skip the same values.
            while (!endReached && curVal == sortedNumericSample(sampleIdx + 1)) {
              curBinWeight += 1
              sampleIdx += 1
              endReached = sampleIdx == (sortedNumericSample.length - 1)
            }

            if (endReached) {
              curBinUpper = Double.PositiveInfinity
            } else {
              curBinUpper = (curVal + sortedNumericSample(sampleIdx + 1)) / 2.0
            }

            if (curBinWeight >= minBinWeight || endReached) {
              bins += NumericBin(curBinLower, curBinUpper)
              curBinLower = curBinUpper
              curBinWeight = 0
            }

            sampleIdx += 1
          }

          val missingValueIdx = if (overallSample.nanFound(featId)) {
            bins.length
          } else {
            -1
          }

          featureBins += NumericBins(bins.toArray, missingValueIdx)
        } else {
          val missingValueIdx = if (overallSample.nanFound(featId)) {
            overallSample.categoricalMaxVals(featId) + 1
          } else {
            -1
          }

          if (missingValueIdx >= maxCardinality) {
            throw CardinalityOverLimitException("The categorical feature " + featId + " has a cardinality that exceeds the limit " + maxCardinality + " because of NaN.")
          }

          // For categorical features, the maximum value + 1 is the cardinality.
          featureBins += CategoricalBins(overallSample.categoricalMaxVals(featId) + 1, missingValueIdx)
        }
      }
    )

    (overallSample.maxLabelValue, featureBins.toArray)
  }
}
