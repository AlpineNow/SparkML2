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

case class MinMaxPair(
  var minValue: Double,
  var maxValue: Double
)

class LabelSummary(val expectedCardinality: Option[Int]) extends Serializable {
  val catCounts: Option[Array[Long]] =
    if (expectedCardinality.isDefined) {
      Some(
        Array.fill[Long](expectedCardinality.get)(0L)
      )
    } else {
      None
    }

  var restCount: Long = 0L
  var hasNaN: Boolean = false

  var totalCount: Double = 0.0
  var runningAvg: Double = 0.0
  var runningSqrAvg: Double = 0.0

  private def updateRunningAvgs(
    bTotalCount: Double,
    bRunningAvg: Double,
    bRunningSqrAvg: Double): Unit = {
    if (totalCount > 0.0 || bTotalCount > 0.0) {
      val newTotalCount = totalCount + bTotalCount
      val aRatio = totalCount / newTotalCount
      val bRatio = bTotalCount / newTotalCount
      val newAvg = aRatio * runningAvg + bRatio * bRunningAvg
      val newSqrAvg = aRatio * runningSqrAvg + bRatio * bRunningSqrAvg

      totalCount = newTotalCount
      runningAvg = newAvg
      runningSqrAvg = newSqrAvg
    }
  }

  def addLabel(label: Double): Unit = {
    if (label.isNaN) {
      hasNaN = true
    } else {
      // Keep track of the running averages.
      updateRunningAvgs(1.0, label, label * label)

      if (expectedCardinality.isDefined && label.toLong.toDouble == label) {
        val cat = label.toInt
        if (cat >= 0 && cat < expectedCardinality.get) {
          catCounts.get(cat) += 1L
        } else {
          restCount += 1L
        }
      } else {
        restCount += 1L
      }
    }
  }

  def mergeInPlace(b: LabelSummary): this.type = {
    if (catCounts.isDefined) {
      var i = 0
      while (i < expectedCardinality.get) {
        catCounts.get(i) += b.catCounts.get(i)
        i += 1
      }
    }

    updateRunningAvgs(b.totalCount, b.runningAvg, b.runningSqrAvg)

    restCount += b.restCount
    hasNaN ||= b.hasNaN
    this
  }
}

/**
 * Classes implementing this trait is used to find bins in the dataset.
 */
trait BinFinder {
  def findBins(
    data: RDD[(Double, Array[Double])],
    columnNames: (String, Array[String]),
    catIndices: Set[Int],
    maxNumBins: Int,
    expectedLabelCardinality: Option[Int],
    notifiee: ProgressNotifiee
  ): (LabelSummary, Seq[Bins])

  def getDataSummary(
    data: RDD[(Double, Array[Double])],
    numFeatures: Int,
    expectedLabelCardinality: Option[Int]
  ): (LabelSummary, Array[Boolean], Array[MinMaxPair]) = {
    val (ls, hn, mm) = data.mapPartitions(
      itr => {
        val labelSummary = new LabelSummary(expectedLabelCardinality)
        val nanExists = Array.fill[Boolean](numFeatures)(false)
        val minMaxes = Array.fill[MinMaxPair](numFeatures)(
          MinMaxPair(minValue = Double.PositiveInfinity, maxValue = Double.NegativeInfinity)
        )
        while (itr.hasNext) {
          val (label, features) = itr.next()
          labelSummary.addLabel(label)
          features.zipWithIndex.foreach {
            case (featValue, featIdx) =>
              nanExists(featIdx) = nanExists(featIdx) || featValue.isNaN
              if (!featValue.isNaN) {
                minMaxes(featIdx).minValue = math.min(featValue, minMaxes(featIdx).minValue)
                minMaxes(featIdx).maxValue = math.max(featValue, minMaxes(featIdx).maxValue)
              }
          }
        }

        Array((labelSummary, nanExists, minMaxes)).iterator
      }
    ).reduce {
      case ((labelSummary1, nanExists1, minMaxes1), (labelSummary2, nanExists2, minMaxes2)) =>
        (
          labelSummary1.mergeInPlace(labelSummary2),
          nanExists1.zip(nanExists2).map {
            case (nan1, nan2) => nan1 || nan2
          },
          minMaxes1.zip(minMaxes2).map {
            case (minMax1, minMax2) =>
              MinMaxPair(
                minValue = math.min(minMax1.minValue, minMax2.minValue),
                maxValue = math.max(minMax1.maxValue, minMax2.maxValue)
              )
          }
        )
    }

    // Labels shouldn't have NaN. So throw an exception if we found NaN for the
    // label.
    if (ls.hasNaN) {
      throw InvalidLabelException("Found NaN value for the label.")
    }

    (ls, hn, mm)
  }
}
