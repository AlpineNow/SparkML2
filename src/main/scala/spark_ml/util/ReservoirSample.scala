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

/**
 * Reservoir sample.
 * @param maxSample The maximum sample size.
 */
class ReservoirSample(val maxSample: Int) extends Serializable {
  var sample = Array.fill[Double](maxSample)(0.0)
  var numSamplePoints: Int = 0
  var numPointsSeen: Double = 0.0

  def doReservoirSampling(point: Double, rng: Random): Unit = {
    numPointsSeen += 1.0
    if (numSamplePoints < maxSample) {
      sample(numSamplePoints) = point
      numSamplePoints += 1
    } else {
      val randomNumber = math.floor(rng.nextDouble() * numPointsSeen)
      if (randomNumber < maxSample.toDouble) {
        sample(randomNumber.toInt) = point
      }
    }
  }
}

object ReservoirSample {
  /**
   * Merge two reservoir samples and make sure that each sample retains
   * uniformness.
   * @param a A reservoir sample.
   * @param b A reservoir sample.
   * @param maxSample Maximum number of reservoir samples we want.
   * @param rng A random number generator.
   * @return Merged sample.
   */
  def mergeReservoirSamples(
    a: ReservoirSample,
    b: ReservoirSample,
    maxSample: Int,
    rng: Random): ReservoirSample = {

    assert(maxSample == a.maxSample)
    assert(a.maxSample == b.maxSample)

    // Merged samples.
    val mergedSample = new ReservoirSample(maxSample)

    // Find out which one has seen more samples.
    val (largerSample, smallerSample) =
      if (a.numPointsSeen > b.numPointsSeen) {
        (a, b)
      } else {
        (b, a)
      }

    // First, fill in the merged samples with the samples that had 'lower' prob
    // of being selected. I.e., the sample that has seen more points.
    var i = 0
    while (i < largerSample.numSamplePoints) {
      mergedSample.sample(i) = largerSample.sample(i)
      i += 1
    }
    mergedSample.numSamplePoints = largerSample.numSamplePoints
    mergedSample.numPointsSeen = largerSample.numPointsSeen

    // Now, add smaller sample points with probabilities so that they become
    // uniform.
    var j = 0 // The smaller sample index.
    val probSmaller = smallerSample.numPointsSeen / (largerSample.numPointsSeen + smallerSample.numPointsSeen)
    while (j < smallerSample.numSamplePoints) {
      val samplePoint = smallerSample.sample(j)
      if (mergedSample.numSamplePoints > smallerSample.numSamplePoints) {
        mergedSample.doReservoirSampling(samplePoint, rng)
      } else {
        val rnd = rng.nextDouble()
        if (rnd < probSmaller) {
          mergedSample.sample(j) = samplePoint
        }
      }

      j += 1
    }

    mergedSample
  }
}