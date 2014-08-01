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
 * Test equal width discretization.
 */
class EqualWidthDiscretizerSuite extends FunSuite with LocalSparkContext {
  test("Test the equi-width discretizer 1") {
    val rawData = TestDataGenerator.labeledData1
    val testDataRDD = sc.parallelize(rawData, 3).cache()

    var exceptionThrown = false
    try {
      // First, let's make sure that we can't discretize if categorical features have strange values.
      EqualWidthDiscretizer.discretizeFeatures(
        testDataRDD,
        Set[Int](2),
        labelIsCategorical = true,
        Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.MaxCardinality_Categoric -> "1000")
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
      EqualWidthDiscretizer.discretizeFeatures(
        testDataRDDRegression,
        Set[Int](2),
        labelIsCategorical = true,
        Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.MaxCardinality_Categoric -> "1000")
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
    val (maxLabelValueRegression, binsRegression) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDDRegression,
      Set[Int](2),
      labelIsCategorical = false,
      Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.MaxCardinality_Categoric -> "1000")
    )

    assert(maxLabelValueRegression === 3.5)
    assert(binsRegression.length === 2)

    exceptionThrown = false
    try {
      EqualWidthDiscretizer.discretizeFeatures(
        testDataRDDRegression2,
        Set[Int](2),
        labelIsCategorical = true,
        Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.MaxCardinality_Categoric -> "1000")
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
    val (maxLabelValueRegression2, binsRegression2) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDDRegression2,
      Set[Int](2),
      labelIsCategorical = false,
      Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.MaxCardinality_Categoric -> "1000")
    )

    assert(maxLabelValueRegression2 === 3.5)
    assert(binsRegression2.length === 2)

    exceptionThrown = false
    try {
      EqualWidthDiscretizer.discretizeFeatures(
        testDataRDD,
        Set[Int](0),
        labelIsCategorical = true,
        Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.MaxCardinality_Categoric -> "3")
      )
    } catch {
      case e: SparkException => {
        if (e.getMessage.contains("CardinalityOverLimitException")) {
          exceptionThrown = true
        }
      }
    }

    assert(exceptionThrown)

    val (maxLabelValue, bins) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](0),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.MaxCardinality_Categoric -> "1000")
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
    assert(bins(1).getCardinality === 5)
    assert(bins(1).findBinIdx(0) === 0)
    assert(bins(1).findBinIdx(1) === 2)
    assert(bins(1).findBinIdx(2) === 4)

    val lastFeature = rawData.map(row => row._2(2)).toArray
    val minFeatVal = lastFeature.min
    val maxFeatVal = lastFeature.max
    val interval = (maxFeatVal - minFeatVal) / 5.0

    assert(bins(2).isInstanceOf[NumericBins])
    assert(bins(2).getCardinality === 5)
    val numericBins = bins(2).asInstanceOf[NumericBins].bins
    assert(numericBins.length === 5)

    var curBinIdx = 0
    while (curBinIdx < numericBins.length) {
      if (curBinIdx == 0) {
        assert(numericBins(curBinIdx).lower.isNegInfinity)
      } else {
        assert(compareDouble(numericBins(curBinIdx).lower, minFeatVal + curBinIdx.toDouble * interval))
      }

      if (curBinIdx == (numericBins.length - 1)) {
        assert(numericBins(curBinIdx).upper.isPosInfinity)
      } else {
        assert(compareDouble(numericBins(curBinIdx).upper, minFeatVal + (curBinIdx.toDouble + 1.0) * interval))
      }

      curBinIdx += 1
    }
  }

  test("Test the equi-width RDD transformation 1") {
    val rawData = TestDataGenerator.labeledData1
    val testDataRDD = sc.parallelize(rawData, 3).cache()

    val (maxLabelValue256, bins256) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](0),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "256", StringConstants.MaxCardinality_Categoric -> "1000")
    )

    val (maxLabelValue1000, bins1000) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](0),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "1000", StringConstants.MaxCardinality_Categoric -> "1000")
    )

    val (maxLabelValue65536, bins65536) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](0),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "65536", StringConstants.MaxCardinality_Categoric -> "1000")
    )

    val (maxLabelValue70000, bins70000) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](0),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "70000", StringConstants.MaxCardinality_Categoric -> "1000")
    )

    assert(maxLabelValue256 === 3.0)
    assert(maxLabelValue1000 === 3.0)
    assert(maxLabelValue65536 === 3.0)
    assert(maxLabelValue70000 === 3.0)

    // First let's make sure that we can get the proper exceptions in case the Bin IDs go over the type limits.
    var exceptionThrown = false
    try {
      val txData = Discretizer.transformFeaturesToUnsignedByteBinIds(testDataRDD, bins1000)
      txData.collect()
    } catch {
      case e: SparkException => {
        if (e.getMessage.contains("BinIdOutOfRangeException")) {
          exceptionThrown = true
        }
      }
    }

    assert(exceptionThrown)

    exceptionThrown = false
    try {
      val txData = Discretizer.transformFeaturesToUnsignedShortBinIds(testDataRDD, bins70000)
      txData.collect()
    } catch {
      case e: SparkException => {
        if (e.getMessage.contains("BinIdOutOfRangeException")) {
          exceptionThrown = true
        }
      }
    }

    assert(exceptionThrown)

    val txData1 = Discretizer.transformFeaturesToUnsignedByteBinIds(testDataRDD, bins256)
    val rawTxData1 = txData1.collect()
    var sampleIdx = 0
    while (sampleIdx < rawTxData1.length) {
      val rawRow = rawData(sampleIdx)
      val convertedRow = rawTxData1(sampleIdx)
      val binId0 = Discretizer.readUnsignedByte(convertedRow._2(0))
      val binId1 = Discretizer.readUnsignedByte(convertedRow._2(1))
      val binId2 = Discretizer.readUnsignedByte(convertedRow._2(2))
      assert(binId0 >= 0)
      assert(binId1 >= 0)
      assert(binId2 >= 0)
      assert(rawRow._2(0).toInt === binId0)
      assert(bins256(1).findBinIdx(rawRow._2(1)) === binId1)
      assert(bins256(2).findBinIdx(rawRow._2(2)) === binId2)

      sampleIdx += 1
    }

    val txData2 = Discretizer.transformFeaturesToUnsignedShortBinIds(testDataRDD, bins65536)
    val rawTxData2 = txData2.collect()
    sampleIdx = 0
    while (sampleIdx < rawTxData2.length) {
      val rawRow = rawData(sampleIdx)
      val convertedRow = rawTxData2(sampleIdx)
      val binId0 = Discretizer.readUnsignedShort(convertedRow._2(0))
      val binId1 = Discretizer.readUnsignedShort(convertedRow._2(1))
      val binId2 = Discretizer.readUnsignedShort(convertedRow._2(2))
      assert(binId0 >= 0)
      assert(binId1 >= 0)
      assert(binId2 >= 0)
      assert(rawRow._2(0).toInt === binId0)
      assert(bins65536(1).findBinIdx(rawRow._2(1)) === binId1)
      assert(bins65536(2).findBinIdx(rawRow._2(2)) === binId2)

      sampleIdx += 1
    }
  }
}
