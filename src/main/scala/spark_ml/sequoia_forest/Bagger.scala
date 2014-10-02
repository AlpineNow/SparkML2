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

import org.apache.spark.rdd.RDD
import spark_ml.util.Poisson
import scala.util.Random

object SamplingType extends Enumeration {
  type SamplingType = Value
  val SampleWithReplacement = Value(0)
  val SampleWithoutReplacement = Value(1)
}

/**
 * Bag the training data.
 */
object Bagger {
  /**
   * Bag the given RDD and add a sample count for each tree to each row.
   * @param data An RDD of label, discretized features pairs.
   * @param numTrees Number of trees that we are doing the bagging for.
   * @param samplingType Sampling type (with/without replacements).
   * @param samplingRate Sampling rate (between 0 and 1).
   * @param seed Seed for the random number generators.
   * @tparam T Type for features - Should be either Byte or Short.
   * @return An RDD of data rows, tagged with bagged sample counts for trees.
   */
  def bagRDD[@specialized(Byte, Short) T](
    data: RDD[(Double, Array[T])],
    numTrees: Int,
    samplingType: SamplingType.SamplingType,
    samplingRate: Double,
    seed: Int): RDD[(Double, Array[T], Array[Byte])] = {
    data.mapPartitionsWithIndex((index, rows) => {
      val poisson = Poisson(samplingRate, seed + index)
      val rng = new Random(seed + index)
      rows.map(
        row => {
          val counts = Array.fill[Byte](numTrees)(0)
          var treeId = 0
          while (treeId < numTrees) {
            val sampleCount = samplingType match {
              case SamplingType.SampleWithReplacement => poisson.sample()
              case SamplingType.SampleWithoutReplacement => if (rng.nextDouble() <= samplingRate) 1 else 0
            }

            // We only allow a sample count value upto 127 since it's represented as a Byte.
            // This shouldn't matter in practice since the probability of that happening should be close to 0.
            counts(treeId) = math.min(sampleCount, 127).toByte
            treeId += 1
          }

          (row._1, row._2, counts)
        })
    })
  }

  /**
   * Bag the given array and add a sample count for each tree to each row.
   * @param data An array of label, discretized features pairs.
   * @param numTrees Number of trees that we are doing the bagging for.
   * @param samplingType Sampling type (with/without replacements).
   * @param samplingRate Sampling rate (between 0 and 1).
   * @param seed Seed for the random number generators.
   * @tparam T Type for features - Should be either Byte or Short.
   * @return An array of data rows, tagged with bagged sample counts for trees.
   */
  def bagArray[@specialized(Byte, Short) T](
    data: Array[(Double, Array[T])],
    numTrees: Int,
    samplingType: SamplingType.SamplingType,
    samplingRate: Double,
    seed: Int): Array[(Double, Array[T], Array[Byte])] = {
    val poisson = Poisson(samplingRate, seed)
    val rng = new Random(seed)
    data.map(row => {
      val counts = Array.fill[Byte](numTrees)(0)
      var treeId = 0
      while (treeId < numTrees) {
        val sampleCount = samplingType match {
          case SamplingType.SampleWithReplacement => poisson.sample()
          case SamplingType.SampleWithoutReplacement => if (rng.nextDouble() <= samplingRate) 1 else 0
        }

        // We only allow a sample count value upto 127 since it's represented as a Byte.
        // This shouldn't matter in practice since the probability of that happening should be close to 0.
        counts(treeId) = math.min(sampleCount, 127).toByte
        treeId += 1
      }

      (row._1, row._2, counts)
    })
  }
}
