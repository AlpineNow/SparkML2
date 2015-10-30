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

import org.apache.spark.rdd.RDD
import spark_ml.util.ProgressNotifiee

/**
 * Compute equal width bins for each numeric column.
 */
class EqualWidthBinFinder extends BinFinder {
  def findBins(
    data: RDD[(Double, Array[Double])],
    columnNames: (String, Array[String]),
    catIndices: Set[Int],
    maxNumBins: Int,
    expectedLabelCardinality: Option[Int],
    notifiee: ProgressNotifiee
  ): (LabelSummary, Seq[Bins]) = {
    val numFeatures = columnNames._2.length

    // Find the label summary as well as the existence of feature values and the
    // min/max values.
    val (labelSummary, featureHasNan, minMaxValues) =
      this.getDataSummary(
        data,
        numFeatures,
        expectedLabelCardinality
      )

    val numericFeatureBins = featureHasNan.zipWithIndex.map {
      case (featHasNan, featIdx) if !catIndices.contains(featIdx) =>
        val minMax = minMaxValues(featIdx)
        if (minMax.minValue.isPosInfinity || (minMax.minValue == minMax.maxValue)) {
          NumericBins(
            Seq(
              NumericBin(lower = Double.NegativeInfinity, upper = Double.PositiveInfinity)
            ),
            if (featHasNan) Some(1) else None
          )
        } else {
          val nonNaNMaxNumBins = maxNumBins - (if (featHasNan) 1 else 0)
          val binWidth = (minMax.maxValue - minMax.minValue) / nonNaNMaxNumBins.toDouble
          NumericBins(
            (0 to (nonNaNMaxNumBins - 1)).map {
              binIdx => {
                binIdx match {
                  case 0 => NumericBin(lower = Double.NegativeInfinity, upper = binWidth + minMax.minValue)
                  case x if x == (nonNaNMaxNumBins - 1) => NumericBin(lower = minMax.maxValue - binWidth, upper = Double.PositiveInfinity)
                  case _ => NumericBin(lower = minMax.minValue + binWidth * binIdx.toDouble, upper = minMax.minValue + binWidth * (binIdx + 1).toDouble)
                }
              }
            },
            if (featHasNan) Some(nonNaNMaxNumBins) else None
          )
        }
      case (featHasNan, featIdx) if catIndices.contains(featIdx) =>
        CategoricalBins((minMaxValues(featIdx).maxValue + 1.0).toInt)
    }

    (labelSummary, numericFeatureBins)
  }
}
