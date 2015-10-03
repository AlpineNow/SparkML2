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

import org.apache.spark.SparkException
import org.scalatest.FunSuite
import spark_ml.util._

/**
 * Test equal width discretization.
 */
class EqualWidthBinFinderSuite extends FunSuite with LocalSparkContext {
  test("Test the equal width bin finder 1") {
    val rawData1 = TestDataGenerator.labeledData1
    val testDataRDD1 = sc.parallelize(rawData1, 3).cache()

    val (labelSummary1, bins1) =
      new EqualWidthBinFinder().findBins(
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
    assert(bins1(0).getCardinality === 8)
    assert(bins1(1).getCardinality === 3)
    assert(bins1(2).getCardinality === 8)

    assert(bins1(0).findBinIdx(0.0) === 0)
    assert(bins1(0).findBinIdx(0.5) === 1)
    assert(bins1(0).findBinIdx(1.0) === 2)
    assert(bins1(0).findBinIdx(4.0) === 7)

    assert(bins1(1).isInstanceOf[CategoricalBins])

    assert(bins1(2).findBinIdx(-80.0) === 0)
    assert(bins1(2).findBinIdx(-60.0) === 0)
    assert(bins1(2).findBinIdx(-58.0) === 1)

    val rawData6 = TestDataGenerator.labeledData6
    val testDataRDD6 = sc.parallelize(rawData6, 3).cache()

    val (labelSummary6, bins6) =
      new EqualWidthBinFinder().findBins(
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
    assert(bins6(0).getCardinality === 8)
    assert(bins6(1).getCardinality === 8)
    assert(bins6(0).asInstanceOf[NumericBins].missingValueBinIdx === Some(7))
    assert(bins6(1).asInstanceOf[NumericBins].missingValueBinIdx === Some(7))
  }
}
