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

import scala.util.Random

import org.scalatest.FunSuite
import spark_ml.util._
import spire.implicits.cfor

/**
 * Test various components.
 */
class ComponentSuite extends FunSuite with LocalSparkContext {
  test("Test ReservoirSample") {
    val a = Array.fill[Double](1000)(0.0)
    val b = Array.fill[Double](1000)(1.0)
    val c = Array.fill[Double](1000)(2.0)

    // Let's test through bootstrapping.
    var numSamples = 0.0
    var avgNumZeroes = 0.0
    var avgNumOnes = 0.0
    var avgNumTwos = 0.0
    val rng = new Random()
    val data = a ++ b ++ c
    while (numSamples < 100) {
      val reservoirSample = new ReservoirSample(100)
      data.foreach(point => reservoirSample.doReservoirSampling(point, rng))
      val numZeroes = reservoirSample.sample.filter(_ == 0.0).length
      val numOnes = reservoirSample.sample.filter(_ == 1.0).length
      val numTwos = reservoirSample.sample.filter(_ == 2.0).length

      numSamples += 1.0
      avgNumZeroes = ((numSamples - 1.0) * avgNumZeroes + numZeroes.toDouble) / numSamples
      avgNumOnes = ((numSamples - 1.0) * avgNumOnes + numOnes.toDouble) / numSamples
      avgNumTwos = ((numSamples - 1.0) * avgNumTwos + numTwos.toDouble) / numSamples
    }

    println("avgNumZeroes is " + avgNumZeroes)
    println("avgNumOnes is " + avgNumOnes)
    println("avgNumTwos is " + avgNumTwos)

    assert(avgNumZeroes <= 35.0 && avgNumZeroes >= 31.0)
    assert(avgNumOnes <= 35.0 && avgNumOnes >= 31.0)
    assert(avgNumTwos <= 35.0 && avgNumTwos >= 31.0)
  }

  def testReservoirMerge(
    numZeroPoints: Int,
    numOnePoints: Int,
    avgZeroLower: Double,
    avgZeroUpper: Double,
    avgOneLower: Double,
    avgOneUpper: Double): Unit = {
    val rng = new Random()
    // Now, let's test merge.
    val sample1 = new ReservoirSample(100)
    val sample2 = new ReservoirSample(100)
    (0 to numZeroPoints - 1).foreach { _ =>
      sample1.doReservoirSampling(0.0, rng)
    }
    (0 to numOnePoints - 1).foreach { _ =>
      sample2.doReservoirSampling(1.0, rng)
    }
    var numSamples = 0.0
    var avgNumZeroes = 0.0
    var avgNumOnes = 0.0
    while (numSamples < 100) {
      val mergedSample = ReservoirSample.mergeReservoirSamples(
        sample1,
        sample2,
        100,
        rng
      )

      val numZeroes = mergedSample.sample.filter(_ == 0.0).length
      val numOnes = mergedSample.sample.filter(_ == 1.0).length
      numSamples += 1.0
      avgNumZeroes = ((numSamples - 1.0) * avgNumZeroes + numZeroes.toDouble) / numSamples
      avgNumOnes = ((numSamples - 1.0) * avgNumOnes + numOnes.toDouble) / numSamples
    }

    println("avgNumZeroes is " + avgNumZeroes)
    println("avgNumOnes is " + avgNumOnes)

    assert(avgNumZeroes <= avgZeroUpper && avgNumZeroes >= avgZeroLower)
    assert(avgNumOnes <= avgOneUpper && avgNumOnes >= avgOneLower)
  }

  test("Test Merge Samples") {
    testReservoirMerge(1000, 2000, 31.0, 35.0, 64.0, 68.0)
    testReservoirMerge(150, 50, 73.0, 77.0, 23.0, 27.0)
    testReservoirMerge(50, 150, 23.0, 27.0, 73.0, 77.0)
    testReservoirMerge(90, 45, 64.0, 68.0, 31.0, 35.0)
  }

  test("Test quickSelect") {
    val a = Array(-1.0, 2.0, -152.0, 123.0, 11.0, 11.0, 323.0, -12.0)
    assert(
      Selection.quickSelect(
        a,
        0,
        a.length,
        0,
        new Random()
      ) === -152.0)
    assert(
      Selection.quickSelect(
        a,
        0,
        a.length,
        1,
        new Random()
      ) === -12.0)
    assert(
      Selection.quickSelect(
        a,
        0,
        a.length,
        2,
        new Random()
      ) === -1.0)
    assert(
      Selection.quickSelect(
        a,
        0,
        a.length,
        3,
        new Random()
      ) === 2.0)
    assert(
      Selection.quickSelect(
        a,
        0,
        a.length,
        4,
        new Random()
      ) === 11.0)
    assert(
      Selection.quickSelect(
        a,
        0,
        a.length,
        5,
        new Random()
      ) === 11.0)
    assert(
      Selection.quickSelect(
        a,
        0,
        a.length,
        6,
        new Random()
      ) === 123.0)
    assert(
      Selection.quickSelect(
        a,
        0,
        a.length,
        7,
        new Random()
      ) === 323.0)
  }

  test("Test MapWithSequentialIntKeys") {
    // Test various scenarios.
    val map = new MapWithSequentialIntKeys[Int](
      initCapacity = 1
    )

    cfor(0)(_ < 10, _ + 1)(
      testKey => {
        var exceptionThrown = false
        try {
          map.get(testKey)
        } catch {
          case UnexpectedKeyException(_) => exceptionThrown = true
        }
        assert(exceptionThrown)
      }
    )

    // Add a key.
    map.put(5, 2)
    val value = map.get(5)
    assert(value === 2)

    {
      var exceptionThrown = false
      try {
        map.put(7, 3)
      } catch {
        case UnexpectedKeyException(_) => exceptionThrown = true
      }
      assert(exceptionThrown)
    }

    cfor(6)(_ <= 10, _ + 1)(
      testKey => {
        map.put(testKey, testKey + 1)
      }
    )

    cfor(6)(_ <= 10, _ + 1)(
      testKey => {
        val value = map.get(testKey)
        assert(value === testKey + 1)
      }
    )

    val keyRange = map.getKeyRange
    assert(keyRange._1 === 5)
    assert(keyRange._2 === 10)

    assert(!map.contains(4))
    assert(map.contains(5))
    assert(map.contains(6))
    assert(map.contains(7))
    assert(map.contains(8))
    assert(map.contains(9))
    assert(map.contains(10))
    assert(!map.contains(11))

    {
      var exceptionThrown = false
      try {
        map.remove(6)
      } catch {
        case UnexpectedKeyException(_) => exceptionThrown = true
      }
      assert(exceptionThrown)
    }

    map.remove(5)
    map.remove(6)
    map.remove(7)
    map.remove(8)
    map.remove(9)
    map.remove(10)
  }

  test("Test Bagger") {
    val numRows = 10000
    val rawData = Array.fill[Int](numRows)(0)
    val testDataRDD = sc.parallelize(rawData, 3)

    // Test bagging with replacement.
    var sampleCountRdd = Bagger.getBagRdd(
      data = testDataRDD,
      numSamples = 1,
      baggingType = BaggingType.WithReplacement,
      baggingRate = 0.7,
      seed = 5
    )

    var sampleCountRaw = sampleCountRdd.collect()

    // Make sure that you get roughly the right number of samples.
    // Also make sure that you get with-replacements.
    assert(sampleCountRaw.length === numRows)
    var sum = 0
    var largerThanOneFound = false
    cfor(0)(_ < numRows, _ + 1)(
      i => {
        val sampleRow = sampleCountRaw(i)
        assert(sampleRow.length === 1)
        sum += sampleRow(0).toInt
        if (sampleRow(0).toInt > 1) {
          largerThanOneFound = true
        }
      }
    )

    assert(largerThanOneFound)
    var expectedSum = numRows.toDouble * 0.7
    var expectedSumLowBound = expectedSum * 0.9
    var expectedSumUpperBound = expectedSum * 1.1
    assert((sum >= expectedSumLowBound) && (sum <= expectedSumUpperBound))

    // Test bagging with replacement with multiple samples.
    sampleCountRdd = Bagger.getBagRdd(
      data = testDataRDD,
      numSamples = 3,
      baggingType = BaggingType.WithReplacement,
      baggingRate = 0.7,
      seed = 6
    )

    sampleCountRaw = sampleCountRdd.collect()

    // Make sure that you get roughly the right number of samples.
    // Make sure that you get with replacements.
    // Make sure that the samples are not the same.
    assert(sampleCountRaw.length === numRows)
    sum = 0
    var sum2 = 0
    var sum3 = 0
    largerThanOneFound = false
    var largerThanOneFound2 = false
    var largerThanOneFound3 = false
    var differenceFound1And2 = false
    var differenceFound2And3 = false
    var differenceFound1And3 = false
    cfor(0)(_ < numRows, _ + 1)(
      i => {
        val sampleRow = sampleCountRaw(i)
        assert(sampleRow.length === 3)
        sum += sampleRow(0).toInt
        sum2 += sampleRow(1).toInt
        sum3 += sampleRow(2).toInt
        if (sampleRow(0).toInt > 1) {
          largerThanOneFound = true
        }

        if (sampleRow(1).toInt > 1) {
          largerThanOneFound2 = true
        }

        if (sampleRow(2).toInt > 1) {
          largerThanOneFound3 = true
        }

        if (sampleRow(0) != sampleRow(1)) {
          differenceFound1And2 = true
        }

        if (sampleRow(1) != sampleRow(2)) {
          differenceFound2And3 = true
        }

        if (sampleRow(0) != sampleRow(2)) {
          differenceFound1And3 = true
        }
      }
    )

    assert(largerThanOneFound)
    assert(largerThanOneFound2)
    assert(largerThanOneFound3)
    assert((sum >= expectedSumLowBound) && (sum <= expectedSumUpperBound))
    assert((sum2 >= expectedSumLowBound) && (sum2 <= expectedSumUpperBound))
    assert((sum3 >= expectedSumLowBound) && (sum3 <= expectedSumUpperBound))
    assert(differenceFound1And2)
    assert(differenceFound2And3)
    assert(differenceFound1And3)

    // Test bagging without replacement with multiple samples.
    sampleCountRdd = Bagger.getBagRdd(
      data = testDataRDD,
      numSamples = 3,
      baggingType = BaggingType.WithoutReplacement,
      baggingRate = 0.7,
      seed = 7
    )

    sampleCountRaw = sampleCountRdd.collect()

    // Make sure that you get roughly the right number of samples.
    // Make sure that you get without replacements.
    // Make sure that the samples are not the same.
    sum = 0
    sum2 = 0
    sum3 = 0
    largerThanOneFound = false
    largerThanOneFound2 = false
    largerThanOneFound3 = false
    differenceFound1And2 = false
    differenceFound2And3 = false
    differenceFound1And3 = false
    cfor(0)(_ < numRows, _ + 1)(
      i => {
        val sampleRow = sampleCountRaw(i)
        assert(sampleRow.length === 3)
        sum += sampleRow(0).toInt
        sum2 += sampleRow(1).toInt
        sum3 += sampleRow(2).toInt
        if (sampleRow(0).toInt > 1) {
          largerThanOneFound = true
        }

        if (sampleRow(1).toInt > 1) {
          largerThanOneFound2 = true
        }

        if (sampleRow(2).toInt > 1) {
          largerThanOneFound3 = true
        }

        if (sampleRow(0) != sampleRow(1)) {
          differenceFound1And2 = true
        }

        if (sampleRow(1) != sampleRow(2)) {
          differenceFound2And3 = true
        }

        if (sampleRow(0) != sampleRow(2)) {
          differenceFound1And3 = true
        }
      }
    )

    assert(!largerThanOneFound)
    assert(!largerThanOneFound2)
    assert(!largerThanOneFound3)
    assert((sum >= expectedSumLowBound) && (sum <= expectedSumUpperBound))
    assert((sum2 >= expectedSumLowBound) && (sum2 <= expectedSumUpperBound))
    assert((sum3 >= expectedSumLowBound) && (sum3 <= expectedSumUpperBound))
    assert(differenceFound1And2)
    assert(differenceFound2And3)
    assert(differenceFound1And3)
  }

  test("Test Feature Handler") {
    val unsignedByteHandler = new UnsignedByteHandler
    assert(unsignedByteHandler.convertToInt((-128).toByte) === 0)
    assert(unsignedByteHandler.convertToInt(0.toByte) === 128)
    assert(unsignedByteHandler.convertToInt(127.toByte) === 255)

    assert(unsignedByteHandler.convertToType(0) === (-128).toByte)
    assert(unsignedByteHandler.convertToType(128) === 0.toByte)
    assert(unsignedByteHandler.convertToType(255) === 127.toByte)

    assert(unsignedByteHandler.getMinValue === 0)
    assert(unsignedByteHandler.getMaxValue === 255)

    val unsignedShortHandler = new UnsignedShortHandler

    assert(unsignedShortHandler.convertToInt((-32768).toShort) === 0)
    assert(unsignedShortHandler.convertToInt(0.toShort) === 32768)
    assert(unsignedShortHandler.convertToInt(32767.toShort) === 65535)

    assert(unsignedShortHandler.convertToType(0) === (-32768).toShort)
    assert(unsignedShortHandler.convertToType(32768) === 0.toShort)
    assert(unsignedShortHandler.convertToType(65535) === 32767.toShort)

    assert(unsignedShortHandler.getMinValue === 0)
    assert(unsignedShortHandler.getMaxValue === 65535)
  }

  test("Test DataFrame Transformation") {
  }

  test("Test RandomSet") {
  }

  test("Test IdCache") {
  }

  test("Test IdLookupForNodeStats") {
  }

  test("Test IdLookupForSubTreeInfo") {
  }

  test("Test IdLookupForUpdaters") {
  }

  test("Test InfoGainNodeStats") {
  }

  test("Test VarianceNodeStats") {
  }
}
