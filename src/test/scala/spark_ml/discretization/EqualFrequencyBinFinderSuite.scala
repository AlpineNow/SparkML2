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

import scala.util.{Failure, Success, Try}

import org.scalatest.FunSuite
import spark_ml.util._

/**
 * Test equal frequency bin finders.
 */
class EqualFrequencyBinFinderSuite extends FunSuite with LocalSparkContext {
  test("Test the equal frequency bin finder 1") {
    val rawData1 = TestDataGenerator.labeledData1
    val testDataRDD1 = sc.parallelize(rawData1, 3).cache()

    val (labelSummary1, bins1) =
      new EqualFrequencyBinFinderFromSample(
        maxSampleSize = 1000,
        seed = 0
      ).findBins(
        data = testDataRDD1,
        columnNames = ("Label", Array("Col1", "Col2", "Col3")),
        catIndices = Set(1),
        maxNumBins = 8,
        expectedLabelCardinality = Some(4),
        notifiee = new ConsoleNotifiee
      )

    assert(labelSummary1.restCount === 0L)
    assert(labelSummary1.catCounts.get.length === 4)
    assert(labelSummary1.catCounts.get(0) === 7L)
    assert(labelSummary1.catCounts.get(1) === 8L)
    assert(labelSummary1.catCounts.get(2) === 11L)
    assert(labelSummary1.catCounts.get(3) === 4L)
    assert(bins1.length === 3)
    assert(bins1(0).getCardinality === 5)
    assert(bins1(1).getCardinality === 3)
    assert(bins1(2).getCardinality === 8)

    BinsTestUtil.validateNumericalBins(
      bins1(0).asInstanceOf[NumericBins],
      Array((Double.NegativeInfinity, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, Double.PositiveInfinity)),
      None
    )

    assert(bins1(0).findBinIdx(0.0) === 0)
    assert(bins1(0).findBinIdx(0.5) === 0)
    assert(bins1(0).findBinIdx(1.0) === 1)
    assert(bins1(0).findBinIdx(3.99999) === 3)
    assert(bins1(0).findBinIdx(4.0) === 4)
    assert(bins1(0).findBinIdx(10.0) === 4)

    assert(bins1(1).isInstanceOf[CategoricalBins])

    Try(bins1(1).findBinIdx(1.1)) match {
      case Success(idx) => fail("CategoricalBins findBinIdx should've thrown an exception.")
      case Failure(ex) => assert(ex.isInstanceOf[InvalidCategoricalValueException])
    }

    Try(bins1(1).findBinIdx(10.0)) match {
      case Success(idx) => fail("CategoricalBins findBinIdx should've thrown an exception.")
      case Failure(ex) => assert(ex.isInstanceOf[CardinalityOverLimitException])
    }

    BinsTestUtil.validateNumericalBins(
      bins1(2).asInstanceOf[NumericBins],
      Array(
        (Double.NegativeInfinity, -72.87),
        (-72.87, -52.28),
        (-52.28, -5.63),
        (-5.63, 20.88),
        (20.88, 25.89),
        (25.89, 59.07),
        (59.07, 81.67),
        (81.67, Double.PositiveInfinity)
      ),
      None
    )

    val rawData3 = TestDataGenerator.labeledData3
    val testDataRDD3 = sc.parallelize(rawData3, 3).cache()

    val (labelSummary3, bins3) =
      new EqualFrequencyBinFinderFromSample(
        maxSampleSize = 1000,
        seed = 0
      ).findBins(
        data = testDataRDD3,
        columnNames = ("Label", Array("Col1", "Col2")),
        catIndices = Set(),
        maxNumBins = 8,
        expectedLabelCardinality = None,
        notifiee = new ConsoleNotifiee
      )

    assert(labelSummary3.expectedCardinality.isEmpty)
    assert(labelSummary3.catCounts.isEmpty)
    assert(labelSummary3.restCount === 30L)
    assert(bins3.length === 2)
    assert(bins3(0).getCardinality === 5)
    assert(bins3(1).getCardinality === 3)

    BinsTestUtil.validateNumericalBins(
      bins3(0).asInstanceOf[NumericBins],
      Array((Double.NegativeInfinity, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, Double.PositiveInfinity)),
      None
    )

    BinsTestUtil.validateNumericalBins(
      bins3(1).asInstanceOf[NumericBins],
      Array((Double.NegativeInfinity, 1.0), (1.0, 2.0), (2.0, Double.PositiveInfinity)),
      None
    )

    val rawData6 = TestDataGenerator.labeledData6
    val testDataRDD6 = sc.parallelize(rawData6, 3).cache()

    val (labelSummary6, bins6) =
      new EqualFrequencyBinFinderFromSample(
        maxSampleSize = 1000,
        seed = 0
      ).findBins(
        data = testDataRDD6,
        columnNames = ("Label", Array("Col1", "Col2")),
        catIndices = Set(),
        maxNumBins = 8,
        expectedLabelCardinality = Some(3),
        notifiee = new ConsoleNotifiee
      )

    assert(labelSummary6.restCount === 4L)
    assert(labelSummary6.catCounts.get(0) === 7L)
    assert(labelSummary6.catCounts.get(1) === 8L)
    assert(labelSummary6.catCounts.get(2) === 11L)
    assert(bins6.length === 2)
    assert(bins6(0).getCardinality === 6)
    assert(bins6(1).getCardinality === 4)

    BinsTestUtil.validateNumericalBins(
      bins6(0).asInstanceOf[NumericBins],
      Array((Double.NegativeInfinity, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, Double.PositiveInfinity)),
      Option(5)
    )

    BinsTestUtil.validateNumericalBins(
      bins6(1).asInstanceOf[NumericBins],
      Array((Double.NegativeInfinity, 1.0), (1.0, 2.0), (2.0, Double.PositiveInfinity)),
      Option(3)
    )
  }

  test("Test the equal frequency RDD transformation 1") {
    val rawData1 = TestDataGenerator.labeledData1
    val testDataRDD1 = sc.parallelize(rawData1, 1).cache()

    val (labelSummary1, bins1) =
      new EqualFrequencyBinFinderFromSample(
        maxSampleSize = 1000,
        seed = 0
      ).findBins(
        data = testDataRDD1,
        columnNames = ("Label", Array("Col1", "Col2", "Col3")),
        catIndices = Set(1),
        maxNumBins = 8,
        expectedLabelCardinality = Some(4),
        notifiee = new ConsoleNotifiee
      )

    val featureHandler = new UnsignedByteHandler
    val transformedFeatures1 = Discretizer.transformFeatures(
      input = testDataRDD1,
      featureBins = bins1,
      featureHandler = featureHandler
    ).collect()

    assert(featureHandler.convertToInt(transformedFeatures1(0)(0)) === 0)
    assert(featureHandler.convertToInt(transformedFeatures1(0)(1)) === 0)
    assert(featureHandler.convertToInt(transformedFeatures1(0)(2)) === 2)

    assert(featureHandler.convertToInt(transformedFeatures1(17)(0)) === 2)
    assert(featureHandler.convertToInt(transformedFeatures1(17)(1)) === 2)
    assert(featureHandler.convertToInt(transformedFeatures1(17)(2)) === 4)

    val rawData6 = TestDataGenerator.labeledData6
    val testDataRDD6 = sc.parallelize(rawData6, 1).cache()

    val (labelSummary6, bins6) =
      new EqualFrequencyBinFinderFromSample(
        maxSampleSize = 1000,
        seed = 0
      ).findBins(
        data = testDataRDD6,
        columnNames = ("Label", Array("Col1", "Col2")),
        catIndices = Set(),
        maxNumBins = 8,
        expectedLabelCardinality = Some(4),
        notifiee = new ConsoleNotifiee
      )

    val transformedFeatures6 = Discretizer.transformFeatures(
      input = testDataRDD6,
      featureBins = bins6,
      featureHandler = featureHandler
    ).collect()

    assert(featureHandler.convertToInt(transformedFeatures6(2)(0)) === 2)
    assert(featureHandler.convertToInt(transformedFeatures6(2)(1)) === 3)

    assert(featureHandler.convertToInt(transformedFeatures6(7)(0)) === 5)
    assert(featureHandler.convertToInt(transformedFeatures6(7)(1)) === 1)
  }
}
