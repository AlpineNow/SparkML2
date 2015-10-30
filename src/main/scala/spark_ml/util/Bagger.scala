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

package spark_ml.util

import scala.util.Random

import org.apache.spark.rdd.RDD
import spire.implicits._

/**
 * Available bagging types.
 */
object BaggingType extends Enumeration {
  type BaggingType = Value
  val WithReplacement = Value(0)
  val WithoutReplacement = Value(1)
}

/**
 * Bagger.
 */
object Bagger {
  /**
   * Create an RDD of bagging info. Each row of the return value is an array of
   * sample counts for a corresponding data row.
   * The number of samples per row is set by numSamples.
   * @param data An RDD of data points.
   * @param numSamples Number of samples we want to get per row.
   * @param baggingType Bagging type.
   * @param baggingRate Bagging rate.
   * @param seed Random seed.
   * @tparam T Data row type.
   * @return An RDD of an array of sample counts.
   */
  def getBagRdd[T](
    data: RDD[T],
    numSamples: Int,
    baggingType: BaggingType.BaggingType,
    baggingRate: Double,
    seed: Int): RDD[Array[Byte]] = {
    data.mapPartitionsWithIndex(
      (index, rows) => {
        val poisson = Poisson(baggingRate, seed + index)
        val rng = new Random(seed + index)
        rows.map(
          row => {
            val counts = Array.fill[Byte](numSamples)(0)
            cfor(0)(_ < numSamples, _ + 1)(
              sampleId => {
                val sampleCount =
                  if (baggingType == BaggingType.WithReplacement) {
                    poisson.sample()
                  } else {
                    if (rng.nextDouble() <= baggingRate) 1 else 0
                  }

                // Only allow a sample count value upto 127
                // to save space.
                counts(sampleId) = math.min(sampleCount, 127).toByte
              }
            )

            counts
          }
        )
      }
    )
  }
}
