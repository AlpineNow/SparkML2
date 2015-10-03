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

import scala.collection.mutable

import org.apache.spark.rdd.RDD
import spark_ml.util.{ProgressNotifiee, Sorting}

/**
 * Find bins that minimizes the loss that can be defined by child classes.
 * E.g., this can be used to calculate bins that satisfy roughly equal
 * frequencies, minimizes entropies, minimizes variances, etc.
 * @param lossName The name of the loss that this is trying to minimize.
 * @param maxSampleSize The maximum sample size.
 * @param seed The seed to use with random sampling.
 */
abstract class NumericBinFinderFromSample(
  val lossName: String,
  val maxSampleSize: Int,
  val seed: Int) extends BinFinder {
  /**
   * Given a segment of a numeric feature array, find a binary partition point
   * that minimizes whatever loss that this finder object wants to minimize.
   * @param labelValues A sample label values.
   * @param featureValues A sample feature values.
   * @param labelSummary This is useful if there's a need to find the label
   *                     class cardinality.
   * @param segLoss The segment loss.
   * @param s The segment starting index (inclusive).
   * @param e The segment ending index (exclusive).
   * @return An optional (left loss, right loss, partition point index) triple.
   *         I.e., the left side (exclusive of the index) is one partition
   *         (starting from s) and the right side (inclusive of the index) is
   *         the other. If there's no suitable partition point (e.g., no
   *         division further reduces the loss to a satisfactory degree), then
   *         the function may return None.
   */
  def findLossMinimizingBinaryPartitions(
    labelValues: Array[Double],
    featureValues: Array[Double],
    labelSummary: LabelSummary,
    segLoss: Double,
    s: Int,
    e: Int): Option[(Double, Double, Int)] = {
    // The default implementation simply goes through possible partitions
    // sequentially and compute weighted losses.
    val segSize = (e - s).toDouble
    var optLoss = segLoss
    var optSplit = s

    var leftLoss = 0.0
    var rightLoss = segLoss
    var curSplit = s + 1
    while (curSplit < e) {
      if (checkSplittability(featureValues, curSplit)) {
        val ll = findSegmentLoss(labelValues, featureValues, labelSummary, s, curSplit)
        val rl = findSegmentLoss(labelValues, featureValues, labelSummary, curSplit, e)
        val loss = (leftLoss * (curSplit - s).toDouble + rightLoss * (e - curSplit).toDouble) / segSize
        if (loss < optLoss) {
          optLoss = loss
          optSplit = curSplit
          leftLoss = ll
          rightLoss = rl
        }
      }

      curSplit += 1
    }

    optSplit match {
      case x if x > s => Some((leftLoss, rightLoss, x))
      case _ => None
    }
  }

  /**
   * Find the loss of the given segment.
   * @param labelValues A sample label values.
   * @param featureValues A sample feature values.
   * @param labelSummary This is useful if there's a need to find the label
   *                     class cardinality.
   * @param s The segment starting index (inclusive).
   * @param e The segment ending index (exclusive).
   * @return The loss of the segment.
   */
  def findSegmentLoss(
    labelValues: Array[Double],
    featureValues: Array[Double],
    labelSummary: LabelSummary,
    s: Int,
    e: Int): Double

  /**
   * Find the numeric bins that minimize the loss for the given label/feature
   * pair with the constraint that the number of bins has to be less than or
   * equal to maxNumBins. This is done in a greedy fashion through continuous
   * binary splits.
   * @param sortedFeatureValues A sample feature values that are sorted.
   * @param matchingLabelValues A sample label values that correspond to the
   *                            feature values position by position.
   * @param labelSummary This is useful if there's a need to find the label
   *                     class cardinality.
   * @param labelName The name of the label column.
   * @param featureName The name of the feature column.
   * @param maxNumBins The maximum number of bins to get.
   * @param hasNan Whether the bins should contain a separate NaN bin.
   * @param notifiee The progress notifiee.
   * @return Bins.
   */
  def findNumericBins(
    sortedFeatureValues: Array[Double],
    matchingLabelValues: Array[Double],
    labelSummary: LabelSummary,
    labelName: String,
    featureName: String,
    maxNumBins: Int,
    hasNan: Boolean,
    notifiee: ProgressNotifiee
  ): Bins = {

    val numPoints = sortedFeatureValues.length
    numPoints match {
      case 0 =>
        NumericBins(
          Seq(NumericBin(Double.NegativeInfinity, Double.PositiveInfinity)),
          if (hasNan) Some(1) else None
        )
      case _ =>
        // Calculate initial loss value without segmentations.
        val initialLoss =
          findSegmentLoss(matchingLabelValues, sortedFeatureValues, labelSummary, 0, numPoints)

        notifiee.newStatusMessage(
          "The current '" + lossName + "' loss for the feature '" + featureName +
            "' and the label '" + labelName + "' is " + initialLoss.toString
        )

        val undivisibleSegments = new mutable.ArrayBuffer[(Double, Int, Int)]()
        val segmentsToExamine = new mutable.Queue[(Double, Int, Int)]()
        segmentsToExamine.enqueue(Tuple3(initialLoss, 0, numPoints))
        while (
          segmentsToExamine.nonEmpty &&
            (undivisibleSegments.length + segmentsToExamine.length) < (maxNumBins - (if (hasNan) 1 else 0))
        ) {
          val (segLoss, segStart, segEnd) = segmentsToExamine.dequeue()
          val division =
            findLossMinimizingBinaryPartitions(
              matchingLabelValues,
              sortedFeatureValues,
              labelSummary,
              segLoss,
              segStart,
              segEnd
            )

          if (division.isEmpty) {
            // If there's no further division, put the segment into the undivisible
            // segments buffer.
            undivisibleSegments += Tuple3(segLoss, segStart, segEnd)
          } else {
            val (leftLoss, rightLoss, divIdx) = division.get
            segmentsToExamine.enqueue(Tuple3(leftLoss, segStart, divIdx))
            segmentsToExamine.enqueue(Tuple3(rightLoss, divIdx, segEnd))
          }
        }

        // Add remaining segments to the undivisible segment buffer.
        undivisibleSegments ++= segmentsToExamine

        // Sort the segments.
        val sortedSegments = scala.util.Sorting.stableSort(
          undivisibleSegments,
          (e1: (Double, Int, Int), e2: (Double, Int, Int)) => e1._2 < e2._2
        )

        // Number of non NaN bins.
        val numBins = sortedSegments.length

        // Let's create the bins object containing bin segments.
        NumericBins(
          sortedSegments.foldLeft(mutable.ArrayBuffer[NumericBin]()) {
            case (binsBuffer, seg) =>
              val (_, lidx, ridx) = seg
              val lower = if (lidx == 0) Double.NegativeInfinity else sortedFeatureValues(lidx)
              val upper = if (ridx == numPoints) Double.PositiveInfinity else sortedFeatureValues(ridx)
              binsBuffer += NumericBin(lower, upper)
              binsBuffer
          },
          // If we want to indicate that a feature's bins contain a separate bin
          // for NaN values.
          if (hasNan) Some(numBins) else None
        )
    }
  }

  /**
   * Find 'bins' from the given data RDD. I.e., numeric columns would be
   * divided into multiple sequential bins via loss minimizations. Individual
   * values of Categorical columns would be simply mapped to their own bins as
   * long as the number of unique categorical values would fall under
   * 'maxNumBins.'
   * @param data The RDD of data.
   * @param columnNames Column names, including the label name and feature names.
   * @param catIndices Indices of categorical features.
   * @param maxNumBins Maximum number of bins that we want to allow.
   * @param expectedLabelCardinality Expected label cardinality. None if
   *                                 regression.
   * @param notifiee A notifiee object used to channel the progress messages
   *                 through.
   * @return The maximum label value and a sequence of feature bins
   */
  def findBins(
    data: RDD[(Double, Array[Double])],
    columnNames: (String, Array[String]),
    catIndices: Set[Int],
    maxNumBins: Int,
    expectedLabelCardinality: Option[Int],
    notifiee: ProgressNotifiee
  ): (LabelSummary, Seq[Bins]) = {

    val numFeatures = columnNames._2.length

    // Find the label summary as well as the existence of feature values.
    val (labelSummary, featureHasNan, minMaxes) = this.getDataSummary(
      data,
      numFeatures,
      expectedLabelCardinality
    )

    // Get a sample of the data.
    val dataSize = data.count()
    val sampleFraction =
      math.min(maxSampleSize.toDouble / dataSize.toDouble, 1.0)
    val sampleData = data.sample(
      withReplacement = false,
      fraction = sampleFraction,
      seed = seed.toLong
    ).collect()
    val sampleSize = sampleData.length

    // Now, go through the features and partition them according to the loss
    // criteria.
    val labelValues = sampleData.map(_._1).toSeq.toArray
    val numericFeatureBins = featureHasNan.zipWithIndex.map {
      case (featHasNan, featIdx) if !catIndices.contains(featIdx) =>
        val featureValues = sampleData.map(_._2(featIdx)).toSeq.toArray
        val validIndices = (0 to (sampleSize - 1)).filter(
          idx => !(featureValues(idx).isNaN || labelValues(idx).isNaN)
        ).toArray
        Sorting.quickSort[Int](validIndices)(Ordering.by[Int, Double](featureValues(_)))
        val sortedFeatureValues = validIndices.map(featureValues(_))
        val matchingLabelValues = validIndices.map(labelValues(_))
        findNumericBins(
          sortedFeatureValues,
          matchingLabelValues,
          labelSummary,
          columnNames._1,
          columnNames._2(featIdx),
          maxNumBins,
          featHasNan,
          notifiee
        )
      case (featHasNan, featIdx) if catIndices.contains(featIdx) =>
        CategoricalBins((minMaxes(featIdx).maxValue + 1.0).toInt)
    }

    (labelSummary, numericFeatureBins)
  }

  /**
   * Check whether values can be split (i.e. the values change at the split
   * point).
   * @param values An array of double values that we want to check for
   *               splittability at the given idx.
   * @param idx The point that we are checking for.
   * @return true if it's splittable, false otherwise.
   */
  protected def checkSplittability(values: Array[Double], idx: Int): Boolean = {
    val leftIdx = math.max(idx - 1, 0)
    (idx != leftIdx) && (values(idx) != values(leftIdx))
  }
}
