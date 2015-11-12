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

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import spark_ml.discretization.{DiscType, DiscretizationOptions}
import spark_ml.gradient_boosting.loss.defaults.{GaussianLossFunction, LogLossFunction}
import spark_ml.gradient_boosting.{DelimitedTextFileConfig, GradientBoosting, GradientBoostingOptions}
import spark_ml.model.gb.GradientBoostedTreesFactoryDefault
import spark_ml.transformation.{DataTransformationUtils, StorageFormat}
import spark_ml.tree_ensembles.CatSplitType
import spark_ml.util.{BaggingType, ConsoleNotifiee, ProgressNotifiee}

object TrainingFunction extends Enumeration {
  type TrainingFunction = Value
  val RandomForest1, GradientBoosting1, GradientBoosting2, GradientBoosting3 = Value
}

case class BiasAndVarianceEstimatorConfig(
  inputPath: Option[String] = None,
  inputFormat: StorageFormat.StorageFormat = StorageFormat.DelimitedText,
  delimitedTextFileConfig: Option[DelimitedTextFileConfig] = None,
  labelCardinality: Option[Int] = None,
  labelColIndex: Int = 0,
  catColIndices: Set[Int] = Set[Int](),
  colsToIgnoreIndices: Set[Int] = Set[Int](),
  maxCatCardinality: Int = 256,
  useFeatureHashingForCat: Boolean = true,
  repartitionSize: Option[Int] = None,
  checkpointDir: Option[String] = None,
  trainingFunction: TrainingFunction.TrainingFunction = TrainingFunction.RandomForest1,
  numIterations: Int = 100,
  verbose: Boolean = false
)

object BiasAndVarianceEstimatorRunner {
  def main(args: Array[String]): Unit = {
    val defaultConfig = BiasAndVarianceEstimatorConfig()

    val parser = new OptionParser[BiasAndVarianceEstimatorConfig](
      "BiasAndVarianceEstimatorRunner"
    ) {
      head("BiasAndVarianceEstimatorRunner: Estimate the expected errors and their bias/variance breakdowns for particular algorithms.")
      opt[String]("inputPath")
        .text("Path to the dataset to use for estimating expected errors and bias/variance breakdowns.")
        .required()
        .action((x, c) => c.copy(inputPath = Some(x)))
      opt[String]("inputFormat")
        .text("The input dataset format. The default is delimited text.")
        .action((x, c) => c.copy(inputFormat = StorageFormat.withName(x)))
      opt[String]("delimiter")
        .text("The delimiter for the input dataset if the input is a delimited text dataset. The default is \"\\t\"")
        .action((x, c) =>
          c.copy(delimitedTextFileConfig =
            c.delimitedTextFileConfig match {
              case None => Some(DelimitedTextFileConfig(delimiter = x))
              case Some(dc) => Some(dc.copy(delimiter = x))
            }
          )
        )
      opt[String]("escapeStr")
        .text("The escape str for the input dataset if the input is a delimited text dataset. The default is \\")
        .action((x, c) =>
          c.copy(delimitedTextFileConfig =
            c.delimitedTextFileConfig match {
              case None => Some(DelimitedTextFileConfig(escapeStr = x))
              case Some(dc) => Some(dc.copy(escapeStr = x))
            }
          )
        )
      opt[String]("quoteStr")
        .text("The quote str for the input dataset if the input is a delimited text dataset. The default is \"")
        .action((x, c) =>
          c.copy(delimitedTextFileConfig =
            c.delimitedTextFileConfig match {
              case None => Some(DelimitedTextFileConfig(quoteStr = x))
              case Some(dc) => Some(dc.copy(quoteStr = x))
            }
          )
        )
      opt[Boolean]("headerExists")
        .text("Whether a header line exists if the input is a delimited text dataset. The default is false")
        .action((x, c) =>
          c.copy(delimitedTextFileConfig =
            c.delimitedTextFileConfig match {
              case None => Some(DelimitedTextFileConfig(headerExists = x))
              case Some(dc) => Some(dc.copy(headerExists = x))
            }
          )
        )
      opt[Int]("labelCardinality")
        .text("If the label is categorical (classification), this has to be set. Otherwise, it's assumed to be regression.")
        .action((x, c) => c.copy(labelCardinality = Some(x)))
      opt[Int]("labelColIndex")
        .text("The index of the column to use as the label. The default is 0.")
        .action((x, c) => c.copy(labelColIndex = x))
      opt[String]("catColIndices")
        .text("A comma separated indices for categorical features. The default is empty.")
        .action((x, c) => c.copy(
          catColIndices = x.split(",").map(value => value.toInt).toSet
        ))
      opt[String]("colsToIgnoreIndices")
        .text("A comma separated indices for columns to ignore. Ignored columns won't be used as features. The default is empty.")
        .action((x, c) => c.copy(
          colsToIgnoreIndices = x.split(",").map(value => value.toInt).toSet
        ))
      opt[Int]("maxCatCardinality")
        .text("Maximum allowed categorical cardinality. Default is 256.")
        .action((x, c) => c.copy(maxCatCardinality = x))
      opt[Boolean]("useFeatureHashingForCat")
        .text("Use feature hashing for categorical features if the cardinality of the feature exceeds to allowed limit.")
        .action((x, c) => c.copy(useFeatureHashingForCat = x))
      opt[Int]("repartitionSize")
        .text("RDD repartition size. If this is set, the input data set would be split into this many partitions.")
        .action((x, c) => c.copy(repartitionSize = Some(x)))
      opt[String]("checkpointDir")
        .text("The checkpointing directory. The checkpoints are used to store " +
          "intermediate id and prediction values. The default is None.")
        .action((x, c) => c.copy(checkpointDir = Some(x)))
      opt[String]("trainingFunction")
        .text("Id of the training function to use.")
        .action((x, c) => c.copy(trainingFunction = TrainingFunction.withName(x)))
      opt[Int]("numIterations")
        .text("Number of iterations to do bootstrapping or resampling.")
        .action((x, c) => c.copy(numIterations = x))
      opt[Boolean]("verbose")
        .text("Whether to verbosely print messages to console or not.")
        .action((x, c) => c.copy(verbose = x))
    }

    parser.parse(args, defaultConfig).map {
      config => {
        val conf = new SparkConf().setAppName("BiasAndVarianceEstimatorRunner")
        val sc = new SparkContext(conf)
        run(sc, config, notifiee = new ConsoleNotifiee)
      }
    }.getOrElse {
      Thread.sleep(10000)
      sys.exit(1)
    }
  }

  def run(
    sc: SparkContext,
    config: BiasAndVarianceEstimatorConfig,
    notifiee: ProgressNotifiee,
    inputSchema: Option[StructType] = None
  ): (Double, Double, Double) = {
    val delimitedTextConfig = config.delimitedTextFileConfig match {
      case None => DelimitedTextFileConfig()
      case Some(dc) => dc
    }

    val (labelFeatureRdd, labelTransformer, featureTransformers, labelName, featureNames, catFeatureIndices) =
      DataTransformationUtils.loadDataFileAndGetLabelFeatureRdd(
        filePath = config.inputPath.get,
        sc = sc,
        dataSchema = inputSchema,
        storageFormat = config.inputFormat,
        repartitionSize = config.repartitionSize,
        delimitedTextFileConfig = delimitedTextConfig,
        maxCatCardinality = config.maxCatCardinality,
        useFeatureHashingForCat = config.useFeatureHashingForCat,
        catColIndices = config.catColIndices,
        colsToIgnoreIndices = config.colsToIgnoreIndices,
        labelColIndex = config.labelColIndex,
        checkpointDir = config.checkpointDir,
        notifiee = notifiee,
        verbose = config.verbose
      )

    val trainer = config.trainingFunction match {
      case TrainingFunction.GradientBoosting1 =>
        (trainingData: RDD[(Double, Array[Double])], catFeatureIndices: Set[Int]) => {
          val gbtFactory = new GradientBoostedTreesFactoryDefault
          gbtFactory.setColumnTransformers(
            labelTransformer = labelTransformer,
            featureTransformers = featureTransformers
          )

          gbtFactory.setColumnNamesAndTypes(
            labelName = labelName,
            labelIsCat = config.labelCardinality.isDefined,
            featureNames = featureNames.toArray,
            featureIsCat = featureNames.zipWithIndex.map {
              case (featName, featIdx) => catFeatureIndices.contains(featIdx)
            }.toArray
          )

          GradientBoosting.train_unsignedByte(
            input = trainingData,
            fracTraining = 1.0,
            columnNames = (labelName, featureNames.toArray),
            catIndices = catFeatureIndices,
            gbOpts =
              GradientBoostingOptions(
                numTrees = 100,
                maxTreeDepth = 3,
                minSplitSize = 10,
                lossFunction =
                  if (config.labelCardinality.isDefined) {
                    new LogLossFunction()
                  } else {
                    new GaussianLossFunction()
                  },
                catSplitType = CatSplitType.OrderedBinarySplit,
                baggingRate = 1.0,
                baggingType = BaggingType.WithoutReplacement,
                shrinkage = 0.1,
                fineTuneTerminalNodes = true,
                checkpointDir = config.checkpointDir,
                predCheckpointInterval = 5,
                idCacheCheckpointInterval = 10,
                verbose = false
              ),
            discOpts =
              DiscretizationOptions(
                discType = DiscType.EqualFrequency,
                maxNumericBins = 256,
                maxCatCardinality = config.maxCatCardinality,
                useFeatureHashingOnCat = config.useFeatureHashingForCat,
                maxSampleSizeForDisc = 5000
              ),
            gbtFactory = gbtFactory,
            notifiee = notifiee
          )
        }

      case _ => throw new UnsupportedOperationException("Other trainers are not supported yet.")
    }

    if (config.checkpointDir.isDefined) {
      sc.setCheckpointDir(config.checkpointDir.get)
    }

    val (noise, bias, variance) =
      BiasAndVarianceEstimator.estimatePredictionErrorBiasAndVariance(
        data = labelFeatureRdd,
        catFeatureIndices = catFeatureIndices,
        trainer = trainer,
        labelCardinality = config.labelCardinality,
        numIterations = config.numIterations,
        seed = 0,
        notifiee = notifiee
      )

    notifiee.newStatusMessage("noise : " + noise)
    notifiee.newStatusMessage("bias : " + bias)
    notifiee.newStatusMessage("variance : " + variance)

    (noise, bias, variance)
  }
}
