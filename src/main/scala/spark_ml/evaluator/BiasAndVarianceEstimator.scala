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

package spark_ml.evaluator

import scala.collection.mutable
import scala.util.Random

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import spark_ml.model._
import spark_ml.util._

trait ErrorBiasAndVariance extends Serializable {
  def addNewPrediction(newPrediction: Double): Unit
  def mergeInPlace(b: ErrorBiasAndVariance): Unit
  def getPredictionCount: Double
  def getNoiseCoefficient: Double
  def getBias: Double
  def getBiasCoefficient: Double
  def getVariance: Double
  def getVarianceCoefficient: Double
}

/**
 * The bias and variance error are kept track of per row.
 * @param optimalLabelValue The optimal label value for this row. This might be
 *                          different from the actual label if labels are noisy.
 */
class MSEBiasAndVariance(optimalLabelValue: Double) extends ErrorBiasAndVariance {
  private var predictionCount: Double = 0.0
  private var predictionSum: Double = 0.0
  private var predictionSqrSum: Double = 0.0

  def addNewPrediction(newPrediction: Double): Unit = {
    predictionCount += 1.0
    predictionSum += newPrediction
    predictionSqrSum += newPrediction * newPrediction
  }

  def mergeInPlace(b: ErrorBiasAndVariance): Unit = {
    predictionCount += b.asInstanceOf[MSEBiasAndVariance].predictionCount
    predictionSum += b.asInstanceOf[MSEBiasAndVariance].predictionSum
    predictionSqrSum += b.asInstanceOf[MSEBiasAndVariance].predictionSqrSum
  }

  def getPredictionCount: Double = predictionCount

  def getNoiseCoefficient: Double = 1.0

  def getBias: Double = {
    val diff = optimalLabelValue - predictionSum / predictionCount
    diff * diff
  }

  def getBiasCoefficient: Double = 1.0

  def getVariance: Double = {
    val avgPred = predictionSum / predictionCount
    predictionSqrSum / predictionCount - avgPred * avgPred
  }

  def getVarianceCoefficient: Double = 1.0
}

/**
 * The bias and variance error are kept track of per row. This is to be used
 * only for binary classification.
 * @param optimalLabelValue The optimal label value for this row. This might be
 *                          different from the actual label if labels are noisy.
 */
class MMEBiasAndVariance(optimalLabelValue: Double) extends ErrorBiasAndVariance {
  private var pred0Cnt: Double = 0.0
  private var pred1Cnt: Double = 0.0

  def addNewPrediction(newPrediction: Double): Unit = {
    if (newPrediction == 0.0) {
      pred0Cnt += 1.0
    } else {
      pred1Cnt += 1.0
    }
  }

  def mergeInPlace(b: ErrorBiasAndVariance): Unit = {
    pred0Cnt += b.asInstanceOf[MMEBiasAndVariance].pred0Cnt
    pred1Cnt += b.asInstanceOf[MMEBiasAndVariance].pred1Cnt
  }

  def getPredictionCount: Double = pred0Cnt + pred1Cnt

  def getNoiseCoefficient: Double = {
    val totalCnt = pred0Cnt + pred1Cnt
    val (optimalPredCnt, nonOptimalPredCnt) =
      if (optimalLabelValue == 0.0) (pred0Cnt, pred1Cnt) else (pred1Cnt, pred0Cnt)
    val optimalPredProb = optimalPredCnt / totalCnt
    val nonOptimalPredProb = nonOptimalPredCnt / totalCnt
    optimalPredProb - nonOptimalPredProb
  }

  def getBias: Double = {
    if (pred0Cnt > pred1Cnt) {
      if (optimalLabelValue == pred0Cnt) {
        0.0
      } else {
        1.0
      }
    } else {
      if (optimalLabelValue == pred1Cnt) {
        0.0
      } else {
        1.0
      }
    }
  }

  def getBiasCoefficient: Double = 1.0

  def getVariance: Double = {
    val totalCnt = pred0Cnt + pred1Cnt
    val meanPred = if (pred0Cnt > pred1Cnt) 0.0 else 1.0
    if (meanPred == 0.0) {
      pred1Cnt / totalCnt
    } else {
      pred0Cnt / totalCnt
    }
  }

  def getVarianceCoefficient: Double = {
    1.0 - 2.0 * getBias
  }
}

trait LabelNoiseStatsAggregator {
  def addLabel(label: Double): Unit
  def getLabelCnt: Double
  def computeNoise: Double
  def computeOptimalLabel: Double

  def reset(): Unit
}

class SquaredErrorLabelNoiseStatsAggregator extends LabelNoiseStatsAggregator {
  var labelSum: Double = 0.0
  var sqrLabelSum: Double = 0.0
  var labelCnt: Double = 0.0

  def addLabel(label: Double): Unit = {
    labelSum += label
    sqrLabelSum += label * label
    labelCnt += 1.0
  }

  def getLabelCnt: Double = labelCnt

  def computeNoise: Double = {
    val avg = labelSum / labelCnt
    sqrLabelSum / labelCnt - avg * avg
  }

  def computeOptimalLabel: Double = labelSum / labelCnt

  def reset(): Unit = {
    labelSum = 0.0
    sqrLabelSum = 0.0
    labelCnt = 0.0
  }
}

class BinaryMisclassificationLabelNoiseStatsAggregator extends LabelNoiseStatsAggregator {
  var label0Cnt: Double = 0.0
  var label1Cnt: Double = 0.0

  def addLabel(label: Double): Unit = {
    if (label == 0.0) {
      label0Cnt += 1.0
    } else {
      label1Cnt += 1.0
    }
  }

  def getLabelCnt: Double = label0Cnt + label1Cnt

  def computeNoise: Double = {
    val totalCnt = label0Cnt + label1Cnt
    if (label0Cnt > label1Cnt) {
      label1Cnt / totalCnt
    } else {
      label0Cnt / totalCnt
    }
  }

  def computeOptimalLabel: Double = if (label0Cnt > label1Cnt) 0.0 else 1.0

  def reset(): Unit = {
    label0Cnt = 0.0
    label1Cnt = 0.0
  }
}

object BiasAndVarianceEstimator {
  /**
   * Using repeated bootstraps on a dataset, estimate the average generalization
   * errors and their bias and variance breakdowns.
   * @param data The dataset that we want to perform bootstrapping on and train
   *             and validate.
   * @param catFeatureIndices Categorical feature indices.
   * @param trainer Training function that takes in a labeled training dataset
   *                and categorical feature indices then returns a model object
   *                that can do predictions.
   * @param labelCardinality Label cardinality if the trainer returns a
   *                         classification model. If this is None, it's assumed
   *                         that we are computing variance/bias for regression.
   * @param numIterations The number of bootstrap iterations that we want to
   *                      perform to estimate the variance and bias measures.
   * @param seed Random seed to use for repeated bootstrap sampling.
   * @param notifiee Progress notifiee.
   * @return A triple of noise, bias and variance.
   */
  def estimatePredictionErrorBiasAndVariance(
    data: RDD[(Double, Array[Double])],
    catFeatureIndices: Set[Int],
    trainer: (RDD[(Double, Array[Double])], Set[Int]) => Model,
    labelCardinality: Option[Int],
    numIterations: Int,
    seed: Int,
    notifiee: ProgressNotifiee
  ): (Double, Double, Double) = {
    val rng = new Random(seed)

    // Sort the dataset by features so that the same exact feature values would
    // be in consecutive rows. This is important later when computing the noise
    // portion of the error (same features having different labels).
    val sortedData =
      data.map { row => (row._2.map(_.hashCode()).sum, row) }.groupByKey().flatMap {
        case (_, groupedRows) =>
          val sortedGroupedRows = groupedRows.toArray
          Sorting.quickSort[(Double, Array[Double])](sortedGroupedRows)(
            new Ordering[(Double, Array[Double])] {
              override def compare(x: (Double, Array[Double]), y: (Double, Array[Double])): Int = {
                var i = 0
                while (i < x._2.length) {
                  val xElem = x._2(i)
                  val yElem = y._2(i)

                  // Since early termination would be frequent, we don't use
                  // foldLeft or other functional method.
                  val compareToValue = xElem.compareTo(yElem)
                  if (compareToValue != 0) {
                    return compareToValue
                  }

                  i += 1
                }

                x._1.compareTo(y._1)
              }
            }
          )

          sortedGroupedRows
      }

    // Let's checkpoint. Assume that checkpointing is already enabled.
    sortedData.checkpoint()

    // Compute the noise component and optimal label values for rows -- i.e. the
    // label variance (or loss for the optimal value) for data points that have
    // the same features.
    val labelNoisesAndOptimalValues = sortedData.mapPartitions {
      case rowItr =>
        var curFeatureSet: Option[Array[Double]] = None
        val noiseStatsAggregator: LabelNoiseStatsAggregator =
          if (labelCardinality.isDefined) {
            if (labelCardinality.get == 2) {
              new BinaryMisclassificationLabelNoiseStatsAggregator
            } else {
              throw new UnsupportedOperationException(
                "Multi-class error variance and bias calculations are not supported."
              )
            }
          } else {
            new SquaredErrorLabelNoiseStatsAggregator
          }

        val output = new mutable.ListBuffer[(Double, Double)]()

        while (rowItr.hasNext) {
          val row = rowItr.next()
          if (curFeatureSet.isDefined) {
            // Find out if the current feature set is different from the new
            // row's feature set. Or whether we've gone through the entire row
            // set.
            if (
                curFeatureSet.get.zip(row._2).exists{ case (v1, v2) => !v1.equals(v2) }
            ) {
              // Compute the noise from the statistics that have been collected.
              val noise = noiseStatsAggregator.computeNoise
              val optimalValue = noiseStatsAggregator.computeOptimalLabel
              val labelCnt = noiseStatsAggregator.getLabelCnt

              output ++= mutable.ListBuffer.fill[(Double, Double)](labelCnt.toInt)((noise, optimalValue))

              noiseStatsAggregator.reset()
            }
          }

          curFeatureSet = Some(row._2)
          noiseStatsAggregator.addLabel(row._1)
        }

        // Get the final row sequence noise aggregator.
        val noise = noiseStatsAggregator.computeNoise
        val optimalValue = noiseStatsAggregator.computeOptimalLabel
        val labelCnt = noiseStatsAggregator.getLabelCnt

        output ++= mutable.ListBuffer.fill[(Double, Double)](labelCnt.toInt)((noise, optimalValue))

        output.iterator
    }

    // Get initial prediction statistics for individual data points.
    // Initially, all prediction statistics should be zeroes.
    var curPredStatistics =
      labelNoisesAndOptimalValues.map {
        case ((noise, optimalValue)) =>
          if (labelCardinality.isDefined) {
            new MMEBiasAndVariance(optimalValue)
          } else {
            new MSEBiasAndVariance(optimalValue)
          }
      }

    curPredStatistics.persist(StorageLevel.MEMORY_AND_DISK)

    // Now, bootstrap train, predict, collect statistics, and repeat.
    var iter = 0
    while (iter < numIterations) {
      val (trainingData, validationFlags) =
        Bagger.getBootstrapSampleAndOOBFlags(
          data = sortedData,
          baggingType = BaggingType.WithReplacement,
          baggingRate = 1.0,
          seed = rng.nextInt()
        )

      val model = trainer(trainingData, catFeatureIndices)

      // Now predict on OOB samples and collect generalization error statistics.
      val dataToPredictOn = sortedData.zip(validationFlags)

      // Update the prediction statistics.
      curPredStatistics = dataToPredictOn.zip(curPredStatistics).map {
        case (((label, features), isValidation), biasAndVariance) =>
          if (isValidation) {
            val newPrediction = model.predict(features)
            if (labelCardinality.isDefined) {
              if (newPrediction(0)._2 > newPrediction(1)._2) {
                biasAndVariance.addNewPrediction(newPrediction(0)._1)
              } else {
                biasAndVariance.addNewPrediction(newPrediction(1)._1)
              }
            } else {
              biasAndVariance.addNewPrediction(newPrediction(0)._1)
            }
          }

          biasAndVariance
      }

      curPredStatistics.persist(StorageLevel.MEMORY_AND_DISK)

      notifiee.newStatusMessage("Finished " + iter + " iterations for the bias-variance estimator.")

      iter += 1
    }

    // Merge prediction statistics from the rows that contain the same set of
    // features.
    val biasAndVariancePerFeatureSet =
      sortedData.zip(labelNoisesAndOptimalValues).zip(curPredStatistics).mapPartitions {
        case rowItr =>
          var curFeatureSet: Option[Array[Double]] = None
          var curErrorBiasAndVariance: Option[ErrorBiasAndVariance] = None
          var curNoise: Option[Double] = None
          val output = new mutable.ListBuffer[(Double, Double, Double, Double)]()
          while (rowItr.hasNext) {
            val (((_, features), (noise, _)), errorBiasAndVariance) = rowItr.next()
            if (curFeatureSet.isDefined) {
              if (
                curFeatureSet.get.zip(features).exists{ case (v1, v2) => !v1.equals(v2) }
              ) {
                output +=
                  ((
                    curErrorBiasAndVariance.get.getPredictionCount,
                    curErrorBiasAndVariance.get.getNoiseCoefficient * curNoise.get,
                    curErrorBiasAndVariance.get.getBiasCoefficient * curErrorBiasAndVariance.get.getBias,
                    curErrorBiasAndVariance.get.getVarianceCoefficient * curErrorBiasAndVariance.get.getVariance
                  ))
                curFeatureSet = Some(features)
                curErrorBiasAndVariance = Some(errorBiasAndVariance)
                curNoise = Some(noise)
              } else {
                curErrorBiasAndVariance.get.mergeInPlace(errorBiasAndVariance)
              }
            } else {
              curFeatureSet = Some(features)
              curErrorBiasAndVariance = Some(errorBiasAndVariance)
              curNoise = Some(noise)
            }
          }

          output.iterator
      }

    val r = biasAndVariancePerFeatureSet.reduce {
      case ((cnt1, noise1, bias1, variance1), (cnt2, noise2, bias2, variance2)) =>
        val totalCnt = cnt1 + cnt2
        val noise = cnt1 / totalCnt * noise1 + cnt2 / totalCnt * noise2
        val bias = cnt1 / totalCnt * bias1 + cnt2 / totalCnt * bias2
        val variance = cnt1 / totalCnt * variance1 + cnt2 / totalCnt * variance2
        (totalCnt, noise, bias, variance)
    }

    (r._2, r._3, r._4)
  }
}
