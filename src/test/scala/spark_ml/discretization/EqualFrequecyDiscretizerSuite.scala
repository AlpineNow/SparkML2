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

import spark_ml.util._
import org.scalatest.FunSuite
import org.apache.spark.SparkException

/**
 * Test equal frequency discretization.
 */
class EqualFrequecyDiscretizerSuite extends FunSuite with LocalSparkContext {
  test("Test the equi-frequency discretizer 1") {
    val rawData = TestDataGenerator.labeledData1
    val testDataRDD = sc.parallelize(rawData, 3).cache()

    var exceptionThrown = false
    try {
      // First, let's make sure that we can't discretize if categorical features have strange values.
      EqualFrequencyDiscretizer.discretizeFeatures(
        testDataRDD,
        Set[Int](2),
        labelIsCategorical = true,
        Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.SubSampleCount_Numeric -> "1000", StringConstants.MaxCardinality_Categoric -> "1000")
      )
    } catch {
      case e: InvalidCategoricalValueException => {
        println(e.toString)
        exceptionThrown = true
      }

      case e: SparkException => {
        if (e.getMessage.contains("InvalidCategoricalValueException")) {
          exceptionThrown = true
        }
      }
    }

    assert(exceptionThrown)

    exceptionThrown = false
    val rawDataRegression = TestDataGenerator.labeledData3
    val testDataRDDRegression = sc.parallelize(rawDataRegression, 3)

    val rawDataRegression2 = TestDataGenerator.labeledData4
    val testDataRDDRegression2 = sc.parallelize(rawDataRegression2, 3)
    try {
      EqualFrequencyDiscretizer.discretizeFeatures(
        testDataRDDRegression,
        Set[Int](2),
        labelIsCategorical = true,
        Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.SubSampleCount_Numeric -> "1000", StringConstants.MaxCardinality_Categoric -> "1000")
      )
    } catch {
      case e: InvalidCategoricalValueException => {
        println(e.toString)
        exceptionThrown = true
      }

      case e: SparkException => {
        if (e.getMessage.contains("InvalidCategoricalValueException")) {
          exceptionThrown = true
        }
      }
    }

    assert(exceptionThrown) // The label should trigger an exception

    // This should not trigger an error.
    val (maxLabelValueRegression, binsRegression) = EqualFrequencyDiscretizer.discretizeFeatures(
      testDataRDDRegression,
      Set[Int](2),
      labelIsCategorical = false,
      Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.SubSampleCount_Numeric -> "1000", StringConstants.MaxCardinality_Categoric -> "1000")
    )

    assert(maxLabelValueRegression === 3.5)
    assert(binsRegression.length === 2)

    exceptionThrown = false
    try {
      EqualFrequencyDiscretizer.discretizeFeatures(
        testDataRDDRegression2,
        Set[Int](2),
        labelIsCategorical = true,
        Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.SubSampleCount_Numeric -> "1000", StringConstants.MaxCardinality_Categoric -> "1000")
      )
    } catch {
      case e: InvalidCategoricalValueException => {
        println(e.toString)
        exceptionThrown = true
      }

      case e: SparkException => {
        if (e.getMessage.contains("InvalidCategoricalValueException")) {
          exceptionThrown = true
        }
      }
    }

    assert(exceptionThrown) // The label should trigger an exception

    // This should not trigger an error.
    val (maxLabelValueRegression2, binsRegression2) = EqualFrequencyDiscretizer.discretizeFeatures(
      testDataRDDRegression2,
      Set[Int](2),
      labelIsCategorical = false,
      Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.SubSampleCount_Numeric -> "1000", StringConstants.MaxCardinality_Categoric -> "1000")
    )

    assert(maxLabelValueRegression2 === 3.5)
    assert(binsRegression2.length === 2)

    exceptionThrown = false
    try {
      EqualFrequencyDiscretizer.discretizeFeatures(
        testDataRDD,
        Set[Int](0),
        labelIsCategorical = true,
        Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.SubSampleCount_Numeric -> "1000", StringConstants.MaxCardinality_Categoric -> "3")
      )
    } catch {
      case e: SparkException => {
        if (e.getMessage.contains("CardinalityOverLimitException")) {
          exceptionThrown = true
        }
      }
    }

    assert(exceptionThrown)

    val (maxLabelValue, bins) = EqualFrequencyDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](0),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.SubSampleCount_Numeric -> "1000", StringConstants.MaxCardinality_Categoric -> "5")
    )

    assert(maxLabelValue === 3.0)
    assert(bins(0).isInstanceOf[CategoricalBins])
    assert(bins(0).getCardinality === 5)
    assert(bins(0).findBinIdx(0) === 0)
    assert(bins(0).findBinIdx(1) === 1)
    assert(bins(0).findBinIdx(2) === 2)
    assert(bins(0).findBinIdx(3) === 3)
    assert(bins(0).findBinIdx(4) === 4)

    assert(bins(1).isInstanceOf[NumericBins])
    assert(bins(1).getCardinality === 2)
    assert(bins(1).findBinIdx(0) === 0)
    assert(bins(1).findBinIdx(1) === 0)
    assert(bins(1).findBinIdx(2) === 1)

    assert(bins(2).isInstanceOf[NumericBins])
    assert(bins(2).getCardinality === 5)
    val numericBins = bins(2).asInstanceOf[NumericBins].bins
    assert(numericBins.length === 5)
    assert(numericBins(0).lower.isNegInfinity)
    assert(numericBins(0).upper === -65.145)
    assert(numericBins(1).lower === -65.145)
    assert(numericBins(1).upper === -16.005)
    assert(numericBins(2).lower === -16.005)
    assert(numericBins(2).upper === 24.235)
    assert(numericBins(3).lower === 24.235)
    assert(numericBins(3).upper === 59.8)
    assert(numericBins(4).lower === 59.8)
    assert(numericBins(4).upper.isPosInfinity)

    val binCount = Array.fill[Int](5)(0)
    var sampleIdx = 0
    while (sampleIdx < rawData.length) {
      binCount(bins(2).findBinIdx(rawData(sampleIdx)._2(2))) += 1
      sampleIdx += 1
    }

    // The counts should be the same in all the bins.
    assert(binCount(0) === 6)
    assert(binCount(0) === binCount(1))
    assert(binCount(1) === binCount(2))
    assert(binCount(2) === binCount(3))
    assert(binCount(3) === binCount(4))
  }

  test("Test the equi-frequency RDD transformation 1") {
    val rawData = TestDataGenerator.labeledData1
    val testDataRDD = sc.parallelize(rawData, 3).cache()

    val (maxLabelValue, bins) = EqualFrequencyDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](0),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.SubSampleCount_Numeric -> "1000", StringConstants.MaxCardinality_Categoric -> "5")
    )

    assert(maxLabelValue === 3.0)

    // Test Byte transformations.
    val transformedRDD_Byte = Discretizer.transformFeaturesToUnsignedByteBinIds(testDataRDD, bins).cache()
    val transformedRaw_Byte = transformedRDD_Byte.collect()

    var sampleIdx = 0
    while (sampleIdx < rawData.length) {
      val rawRow = rawData(sampleIdx)
      val convertedRow = transformedRaw_Byte(sampleIdx)
      val binId0 = Discretizer.readUnsignedByte(convertedRow._2(0))
      val binId1 = Discretizer.readUnsignedByte(convertedRow._2(1))
      val binId2 = Discretizer.readUnsignedByte(convertedRow._2(2))
      assert(binId0 >= 0)
      assert(binId1 >= 0)
      assert(binId2 >= 0)
      assert(rawRow._2(0).toInt === binId0)
      assert(bins(1).findBinIdx(rawRow._2(1)) === binId1)
      assert(bins(2).findBinIdx(rawRow._2(2)) === binId2)

      sampleIdx += 1
    }

    val transformedRDD_Short = Discretizer.transformFeaturesToUnsignedShortBinIds(testDataRDD, bins)
    val transformedRaw_Short = transformedRDD_Short.collect()

    sampleIdx = 0
    while (sampleIdx < rawData.length) {
      val rawRow = rawData(sampleIdx)
      val convertedRow = transformedRaw_Short(sampleIdx)
      val binId0 = Discretizer.readUnsignedShort(convertedRow._2(0))
      val binId1 = Discretizer.readUnsignedShort(convertedRow._2(1))
      val binId2 = Discretizer.readUnsignedShort(convertedRow._2(2))
      assert(binId0 >= 0)
      assert(binId1 >= 0)
      assert(binId2 >= 0)
      assert(rawRow._2(0).toInt === binId0)
      assert(bins(1).findBinIdx(rawRow._2(1)) === binId1)
      assert(bins(2).findBinIdx(rawRow._2(2)) === binId2)

      sampleIdx += 1
    }
  }
}
