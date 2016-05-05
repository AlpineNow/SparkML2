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

package spark_ml.gradient_boosting

import com.databricks.spark.csv._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.StructType
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser
import spark_ml.discretization.{CardinalityOverLimitException, DiscType, DiscretizationOptions}
import spark_ml.gradient_boosting.loss.LossFunction
import spark_ml.gradient_boosting.loss.defaults.GaussianLossFunction
import spark_ml.model.gb.{GradientBoostedTrees, GradientBoostedTreesFactory, GradientBoostedTreesFactoryDefault}
import spark_ml.transformation.{ColumnTransformer, DataTransformationUtils, DistinctValueCounter}
import spark_ml.tree_ensembles.CatSplitType
import spark_ml.util.{BaggingType, ConsoleNotifiee, ProgressNotifiee}

/**
 * The storage format for the input dataset.
 */
object StorageFormat extends Enumeration {
  type StorageFormat = Value
  val DelimitedText = Value(0)
  val Parquet = Value(1)
  val Avro = Value(2)
}

/**
 * Delimited file config options.
 * @param delimiter What the delimiter is.
 * @param escapeStr The escape str.
 * @param quoteStr The quote str.
 * @param headerExists Whether the delimited file contains a header.
 */
case class DelimitedTextFileConfig(
  delimiter: String = "\t",
  escapeStr: String = "\\",
  quoteStr: String = "\"",
  headerExists: Boolean = false
)

/**
 * Runner options.
 * @param inputPath Input data file path in HDFS.
 * @param outputPath Output (model) file path in HDFS.
 * @param inputFormat The input data format.
 * @param delimitedTextFileConfig If the input is a delimited text file, this
 *                                option is used to parse it.
 * @param fracTraining Fraction of the input data to use for training. The rest
 *                     will be used for validation.
 * @param labelColIndex Label column index.
 * @param catColIndices Categorical column indices.
 * @param colsToIgnoreIndices Indices of columns to ignore.
 * @param gbOpts Gradient boosting options.
 * @param discOpts Discretization options.
 * @param repartitionSize Optional repartition size. If repartitioning is to be
 *                        triggered, checkpointing should also be enabled.
 * @param pauseDuration Seconds to pause after finishing the job. A non-zero
 *                      value could be useful if one wants to look at stdout/
 *                      stderr logs after the job is finished (some clusters
 *                      by default might not keep these logs).
 */
case class GradientBoostingRunnerConfig(
  inputPath: Option[String] = None,
  outputPath: Option[String] = None,
  inputFormat: StorageFormat.StorageFormat = StorageFormat.DelimitedText,
  delimitedTextFileConfig: Option[DelimitedTextFileConfig] = None,
  fracTraining: Double = 0.8, // Fraction of the entire data to use for training.
  labelColIndex: Int = 0,
  catColIndices: Set[Int] = Set[Int](),
  colsToIgnoreIndices: Set[Int] = Set[Int](),
  gbOpts: GradientBoostingOptions =
    GradientBoostingOptions(
      numTrees = 100,
      maxTreeDepth = 3,
      minSplitSize = 10,
      lossFunction = new GaussianLossFunction(),
      catSplitType = CatSplitType.OrderedBinarySplit,
      baggingRate = 0.5,
      baggingType = BaggingType.WithoutReplacement,
      shrinkage = 0.001,
      fineTuneTerminalNodes = true,
      checkpointDir = None,
      predCheckpointInterval = 5,
      idCacheCheckpointInterval = 10,
      verbose = false
    ),
  discOpts: DiscretizationOptions =
    DiscretizationOptions(
      discType = DiscType.EqualFrequency,
      maxNumericBins = 256,
      maxCatCardinality = 256,
      useFeatureHashingOnCat = true,
      maxSampleSizeForDisc = 5000
    ),
  repartitionSize: Option[Int] = None,
  pauseDuration: Int = 0
)

/**
 * The over-arching trainer that performs data parsing, discretization,
 * training, validations, and outputs the resulting model.
 */
object GradientBoostingRunner {
  def main(args: Array[String]) {
    val defaultConfig = GradientBoostingRunnerConfig()

    val parser = new OptionParser[GradientBoostingRunnerConfig](
      "GradientBoostingRunner"
    ) {
      head("GradientBoostingRunner: Train Gradient Boosted Trees.")
      opt[String]("inputPath")
        .text("Path to the dataset to use as a training set.")
        .required()
        .action((x, c) => c.copy(inputPath = Some(x)))
      opt[String]("outputPath")
        .text("Output path where the trained model will be stored.")
        .required()
        .action((x, c) => c.copy(outputPath = Some(x)))
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
      opt[Double]("fracTraining")
        .text("Fraction of the input data to use for training. The rest is used for validations. The default is 0.8")
        .action((x, c) => c.copy(fracTraining = x))
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
      opt[Int]("numTrees")
        .text("The number of trees to train. The default is 100.")
        .action((x, c) => c.copy(gbOpts = c.gbOpts.copy(numTrees = x)))
      opt[Int]("maxTreeDepth")
        .text("Maximum depth per tree. The default is 3.")
        .action((x, c) => c.copy(gbOpts = c.gbOpts.copy(maxTreeDepth = x)))
      opt[Int]("minSplitSize")
        .text("Minimum node size to be eligible for splits. The default is 10.")
        .action((x, c) => c.copy(gbOpts = c.gbOpts.copy(minSplitSize = x)))
      opt[String]("lossFunction")
        .text("The loss function name. The default is GaussianLossFunction.")
        .action((x, c) => c.copy(gbOpts = c.gbOpts.copy(lossFunction = Class.forName("spark_ml.gradient_boosting.loss.defaults." + x).newInstance().asInstanceOf[LossFunction])))
      opt[String]("catSplitType")
        .text("Type of categorical feature splits to perform. " +
        "Available types are " +
        "OrderedBinarySplit, RandomBinarySplit, and MultiwaySplit. " +
        "The default is OrderedBinarySplit.")
        .action((x, c) => c.copy(gbOpts = c.gbOpts.copy(catSplitType = CatSplitType.withName(x))))
      opt[Double]("baggingRate")
        .text("Bagging rate for each tree. The default is 0.5.")
        .action((x, c) => c.copy(gbOpts = c.gbOpts.copy(baggingRate = x)))
      opt[Boolean]("bagWithReplacement")
        .text("Whether bagging will be done with or without replacements. The default is false (without replacement).")
        .action((x, c) => c.copy(gbOpts = c.gbOpts.copy(baggingType = if (x) BaggingType.WithReplacement else BaggingType.WithoutReplacement)))
      opt[Double]("shrinkage")
        .text("Shrinkage to multiply to each additional tree. " +
          "The default is 0.001.")
        .action((x, c) => c.copy(gbOpts = c.gbOpts.copy(shrinkage = x)))
      opt[Boolean]("fineTuneTerminalNodes")
        .text("Whether to fine tune tree's terminal nodes so that their values are directly optimizing against the loss function, rather than averaging gradients. The default is true.")
        .action((x, c) => c.copy(gbOpts = c.gbOpts.copy(fineTuneTerminalNodes = x)))
      opt[String]("checkpointDir")
        .text("The checkpointing directory. The checkpoints are used to store " +
        "intermediate id and prediction values. The default is None.")
        .action((x, c) => c.copy(gbOpts = c.gbOpts.copy(checkpointDir = Some(x))))
      opt[Int]("predCheckpointInterval")
        .text("How often to checkpoint intermediate predictions. The default is 5.")
        .action((x, c) => c.copy(gbOpts = c.gbOpts.copy(predCheckpointInterval = x)))
      opt[Int]("idCacheCheckpointInterval")
        .text("How often to checkpoint intermediate node Id values. The default is 10.")
        .action((x, c) => c.copy(gbOpts = c.gbOpts.copy(idCacheCheckpointInterval = x)))
      opt[Boolean]("verbose")
        .text("Print as much information as possible. The default is false.")
        .action((x, c) => c.copy(gbOpts = c.gbOpts.copy(verbose = x)))
      opt[String]("discType")
        .text("Type of discretization to perform on numeric features. " +
          "Available types are EqualFrequency, EqualWidth, MinimumEntropy and MinimumVariance. " +
          "The default is EqualFrequency.")
        .action((x, c) => c.copy(discOpts = c.discOpts.copy(discType = DiscType.withName(x))))
      opt[Int]("maxNumericBins")
        .text("Maximum number of numeric bins we'll allow. The default is 256.")
        .action((x, c) => c.copy(discOpts = c.discOpts.copy(maxNumericBins = x)))
      opt[Int]("maxCatCardinality")
        .text("Maximum number of categorical feature cardinality we'll allow. " +
          "The default is 256.")
        .action((x, c) => c.copy(discOpts = c.discOpts.copy(maxCatCardinality = x)))
      opt[Boolean]("useFeatureHashingOnCat")
        .text("Whether to perform feature hashing on categorical features whose cardinalities are larger than maxCatCardinality. " +
          "The default is true.")
        .action((x, c) => c.copy(discOpts = c.discOpts.copy(useFeatureHashingOnCat = x)))
      opt[Int]("maxSampleSizeForDisc")
        .text("Maximum size of a sample to use for certain discretizations (e.g. equal frequency discretizations")
        .action((x, c) => c.copy(discOpts = c.discOpts.copy(maxSampleSizeForDisc = x)))
      opt[Int]("repartitionSize")
        .text("Number of partitions to repartition the data to. The checkpoint directory should be specified, too.")
        .action((x, c) => c.copy(repartitionSize = Some(x)))
      opt[Int]("pauseDuration")
        .text("Seconds to pause after finishing the job. The default is 0.")
        .action((x, c) => c.copy(pauseDuration = x))
    }

    // Parse the arguments and then run.
    parser.parse(args, defaultConfig).map {
      config => {
        val conf = new SparkConf().setAppName("GradientBoostingRunner")
        val sc = new SparkContext(conf)
        run(sc, config, gbtFactory = new GradientBoostedTreesFactoryDefault, notifiee = new ConsoleNotifiee)
      }
    }.getOrElse {
      // Sleep for 10 seconds so that people may see error messages.
      Thread.sleep(10000)
      sys.exit(1)
    }
  }

  /**
   * Train GBT using the given config.
   * @param config Configuration for the training algorithm.
   * @param gbtFactory The factory for the return model. The developer can
   *                   return his/her own GBT implementation through this.
   * @param notifiee Progress notifiee.
   * @param inputSchema Optional schema. E.g., if one already knows the schema
   *                    of a text input file.
   * @return Trained gradient boosted trees.
   */
  def run(
    sc: SparkContext,
    config: GradientBoostingRunnerConfig,
    gbtFactory: GradientBoostedTreesFactory,
    notifiee: ProgressNotifiee,
    inputSchema: Option[StructType] = None
  ): GradientBoostedTrees = {
    val pauseDuration = config.pauseDuration
    val Some(inputPath) = config.inputPath
    val delimitedTextConfig = config.delimitedTextFileConfig match {
      case None => DelimitedTextFileConfig()
      case Some(dc) => dc
    }
    val maxCatCardinality = config.discOpts.maxCatCardinality
    val useFeatureHashingForCat = config.discOpts.useFeatureHashingOnCat
    val catColIndices = config.catColIndices
    val sortedCatColIndices = catColIndices.toSeq.sorted
    val colsToIgnoreIndices = config.colsToIgnoreIndices
    val labelColIndex = config.labelColIndex
    val fracTraining = config.fracTraining

    try {
      // Load the data through Spark's SQLContext.
      val sqlContext = new SQLContext(sc)

      notifiee.newStatusMessage("Loading the inputPath " + inputPath)
      val loadedDf = config.inputFormat match {
        case StorageFormat.DelimitedText =>
          val parser =
            new CsvParser()
              .withUseHeader(delimitedTextConfig.headerExists)
              .withDelimiter(delimitedTextConfig.delimiter(0))
              .withQuoteChar(delimitedTextConfig.quoteStr(0))
              .withEscape(delimitedTextConfig.escapeStr(0))

          (
            inputSchema match {
              case Some(structType) => parser.withSchema(structType)
              case None => parser.withInferSchema(true)
            }
          ).csvFile(sqlContext, inputPath)

        case StorageFormat.Avro => sqlContext.read.format("com.databricks.spark.avro").load(inputPath)
        case StorageFormat.Parquet => sqlContext.read.load(inputPath)
      }

      val dataFrame = config.repartitionSize match {
        case Some(numParts) =>
          if (config.gbOpts.checkpointDir.isDefined) {
            val repartitioned = loadedDf.repartition(numParts)
            repartitioned.rdd.sparkContext.setCheckpointDir(config.gbOpts.checkpointDir.get)
            repartitioned.rdd.checkpoint()
            repartitioned
          } else {
            loadedDf
          }
        case None => loadedDf
      }

      if (config.gbOpts.verbose) {
        notifiee.newStatusMessage("The dataset " + inputPath + " was successfully loaded.")
        notifiee.newStatusMessage("There are " + dataFrame.count() + " rows.")
        notifiee.newStatusMessage("There are " + dataFrame.columns.length + " columns.")
      }

      notifiee.newStatusMessage("Successfully loaded the dataset.")

      // Get the columns.
      val columns = dataFrame.columns
      val numCols = columns.length

      notifiee.newStatusMessage("Finding distinct values for categorical features.")

      // Get distinct categorical values.
      val catDistinctVals = DistinctValueCounter.getDistinctValues(
        dataFrame,
        catColIndices,
        maxCatCardinality
      )

      notifiee.newStatusMessage("Found distinct values for categorical features.")

      // Compute the cardinalities of each categorical column.
      val catColCardinalities =
        sortedCatColIndices.map(colIdx => colIdx -> catDistinctVals(colIdx).size).toMap

      // See if any categorical column has a cardinality that larger than
      // the maximum. If so, see if feature hashing is allowed. Otherwise,
      // we'll simply throw an exception.
      catColCardinalities.foreach {
        case (idx, cardinality) =>
          if (cardinality > maxCatCardinality && !useFeatureHashingForCat) {
            throw CardinalityOverLimitException(
              "The categorical column " + columns(idx) + " has a cardinality of " +
              cardinality.toString + " that is higher than the limit " + maxCatCardinality.toString +
              ". Use the feature hashing option if you want to use this as a feature."
            )
          }
      }

      // Map unique strings to numbers.
      val catDistinctValToInt = DistinctValueCounter.mapDistinctValuesToIntegers(
        catDistinctVals,
        useEmptyString = true
      )

      if (config.gbOpts.verbose) {
        notifiee.newStatusMessage("The categorical columns are : ")
        notifiee.newStatusMessage(catColIndices.map(columns(_)).mkString(" "))
        notifiee.newStatusMessage("The maximum allowed cardinality is " + maxCatCardinality)
        catDistinctValToInt.foreach {
          case (key, distinctValCounts) =>
            notifiee.newStatusMessage(columns(key) + " has " + distinctValCounts.size + " unique values : ")
            if (distinctValCounts.size > maxCatCardinality) {
              notifiee.newStatusMessage(columns(key) + "'s distinct value count exceed the limit.")
            } else {
              notifiee.newStatusMessage(
                distinctValCounts.map {
                  case (distinctVal, mappedInt) => distinctVal + "->" + mappedInt.toString
                }.mkString(",")
              )
            }
        }
      }

      notifiee.newStatusMessage("Transforming the dataset into an RDD of label, feature-vector pairs.")

      // Now convert the data set into a form usable by the algorithm.
      // It'll be a pair of a label and a feature vector.
      val (labelFeatureRdd, (labelTransformer, featureTransformers)) =
        DataTransformationUtils.transformDataFrameToLabelFeatureRdd(
          dataFrame,
          labelColIndex,
          catDistinctValToInt,
          colsToIgnoreIndices,
          maxCatCardinality,
          useFeatureHashingForCat
        )

      // Remember the transformers for the final model.
      gbtFactory.setColumnTransformers(
        labelTransformer = labelTransformer,
        featureTransformers = featureTransformers
      )

      // Map categorical column indices to categorical feature indices.
      // Feature indices exclude the label column and the columns to ignore.
      val (featureNames, catFeatureIndices, _) =
        (0 to (numCols - 1)).foldLeft((Seq[String](), Set[Int](), 0)) {
          case ((featNames, catFeatIndices, curFeatIdx), curColIdx) =>
            curColIdx match {
              case _ if labelColIndex == curColIdx || colsToIgnoreIndices.contains(curColIdx) => (featNames, catFeatIndices, curFeatIdx)
              case _ if catColIndices.contains(curColIdx) => (featNames ++ Seq(columns(curColIdx)), catFeatIndices + curFeatIdx, curFeatIdx + 1)
              case _ => (featNames ++ Seq(columns(curColIdx)), catFeatIndices, curFeatIdx + 1)
            }
        }

      // Set column names and types for the final model creator.
      gbtFactory.setColumnNamesAndTypes(
        labelName = columns(labelColIndex),
        labelIsCat = catColIndices.contains(labelColIndex),
        featureNames = featureNames.toArray,
        featureIsCat = featureNames.zipWithIndex.map {
          case (featName, featIdx) => catFeatureIndices.contains(featIdx)
        }.toArray
      )

      if (config.gbOpts.verbose) {
        // Transformed features.
        val labelFeatureRddSample = labelFeatureRdd.take(50)

        notifiee.newStatusMessage("Transformed column names are :")
        notifiee.newStatusMessage(columns(labelColIndex) + ", (" + featureNames.mkString(",") + ")")
        notifiee.newStatusMessage("Categorical features are :")
        notifiee.newStatusMessage(
          featureNames.zipWithIndex.filter {
            case (featName, featIdx) => catFeatureIndices.contains(featIdx)
          }.mkString(",")
        )
        notifiee.newStatusMessage("Transformed label feature samples are :")
        labelFeatureRddSample.foreach {
          case (sampleLabel, sampleFeatures) =>
            notifiee.newStatusMessage(sampleLabel.toString + ", (" + sampleFeatures.mkString(",") + ")")
        }

        val getColumnTransformerDesc =
          (columnTransformer: ColumnTransformer) => {
            if (columnTransformer.distinctValToInt.isDefined) {
              "CatToIntMapper"
            } else if (columnTransformer.maxCardinality.isDefined) {
              "CatHasher"
            } else {
              "NumericPassThrough"
            }
          }

        notifiee.newStatusMessage("ColumnTransfomer descriptions :")
        notifiee.newStatusMessage(
          getColumnTransformerDesc(labelTransformer) + ", (" +
          featureTransformers.map(getColumnTransformerDesc(_)).mkString(",") + ")"
        )
      }

      val gbOpts = config.gbOpts
      val discOpts = config.discOpts

      // Depending on the number of maximum bins in either
      // numeric or categorical features, we might discretize
      // the features into Byte or Short.
      val model: GradientBoostedTrees =
        if (discOpts.maxNumericBins <= 256 && discOpts.maxCatCardinality <= 256) {
          GradientBoosting.train_unsignedByte(
            input = labelFeatureRdd,
            fracTraining = fracTraining,
            columnNames = (columns(labelColIndex), featureNames.toArray),
            catIndices = catFeatureIndices,
            gbOpts = gbOpts,
            discOpts = discOpts,
            gbtFactory = gbtFactory,
            notifiee = notifiee
          )
        } else if (discOpts.maxNumericBins <= 65536 && discOpts.maxCatCardinality <= 65536) {
          GradientBoosting.train_unsignedShort(
            input = labelFeatureRdd,
            fracTraining = fracTraining,
            columnNames = (columns(labelColIndex), featureNames.toArray),
            catIndices = catFeatureIndices,
            gbOpts = gbOpts,
            discOpts = discOpts,
            gbtFactory = gbtFactory,
            notifiee = notifiee
          )
        } else {
          throw new CardinalityOverLimitException(
            "Currently, GBT does not support discretization with bin counts " +
              "over 65536."
          )
        }

      // TODO: Currently we are not storing the model in the output path.
      // TODO: Need to do this in a proper serialization format.

      model
    } finally {
      Thread.sleep(pauseDuration * 1000)
    }
  }
}
