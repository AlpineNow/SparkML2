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

package spark_ml.sequoia_forest

import scala.collection.mutable

import scopt.OptionParser
import org.apache.spark.{ SparkContext, SparkConf }

/**
 * The forest type (e.g. info gain classification, variance regression).
 */
object ForestType extends Enumeration {
  type ForestType = Value
  val InfoGain = Value(0)
  val Variance = Value(1)
}

case class ValidationOptions(
  validationPath: String = null,
  useLogLossForValidation: Boolean = false)

case class FilePathOptions(
  inputPath: String = null,
  outputPath: String = null,
  checkpointDir: String = null,
  checkpointInterval: Int = 10)

/**
 * Config object to be set by command line arguments.
 */
case class RunnerConfig(
  filePathOptions: FilePathOptions = FilePathOptions(),
  numTrees: Int = 10,
  numPartitions: Int = -1,
  delimiter: String = "\t",
  labelIndex: Int = 0,
  categoricalFeatureIndices: Set[Int] = Set[Int](),
  indicesToIgnore: Set[Int] = Set[Int](),
  forestType: ForestType.ForestType = ForestType.InfoGain,
  discretizationType: DiscretizationType.DiscretizationType = DiscretizationType.EqualFrequency,
  maxNumNumericBins: Int = 256,
  maxNumCategoricalBins: Int = 256,
  sampleWithReplacement: Boolean = true,
  sampleRate: Double = 1.0,
  mtry: Int = -1,
  minSplitSize: Long = -1,
  maxDepth: Int = -1,
  numRowFiltersPerIter: Int = -1,
  subTreeThreshold: Int = -1,
  numSubTreesPerIter: Int = -1,
  pauseDuration: Int = 0,
  validationOptions: ValidationOptions = ValidationOptions(),
  distributedNodeSplits: Boolean = false)

/**
 * Wraps around everything to provide a simple command line interface for running Sequoia Forest.
 */
object SequoiaForestRunner {
  def main(args: Array[String]) {
    val defaultConfig = new RunnerConfig()

    // Command line argument parser.
    val parser = new OptionParser[RunnerConfig]("SequoiaForestRunner") {
      head("SequoiaForestRunner: Train multiple trees on large data sets.")
      opt[String]("inputPath")
        .text("Path to delimited text file(s) (e.g. csv/tsv) to be used as a training input. All the used fields should be numeric (ignored columns can be anything). Categorical fields should be enumerated from 0 to K-1 where K is the cardinality of the field.")
        .required()
        .action((x, c) => c.copy(filePathOptions = c.filePathOptions.copy(inputPath = x)))
      opt[String]("outputPath")
        .text("Output path (directory) where the trained trees will be stored as binary files.")
        .required()
        .action((x, c) => c.copy(filePathOptions = c.filePathOptions.copy(outputPath = x)))
      opt[String]("checkpointDir")
        .text("Checkpoint directory for an intermediate RDD.")
        .action((x, c) => c.copy(filePathOptions = c.filePathOptions.copy(checkpointDir = x)))
      opt[Int]("checkpointInterval")
        .text("Checkpoint interval for an imtermediate RDD.")
        .action((x, c) => c.copy(filePathOptions = c.filePathOptions.copy(checkpointInterval = x)))
      opt[Boolean]("distributedNodeSplits")
        .text("Whether to perform distributed node splits.")
        .action((x, c) => c.copy(distributedNodeSplits = x))
      opt[Int]("numTrees")
        .text("Number of trees to train.")
        .required()
        .action((x, c) => c.copy(numTrees = x))
      opt[Int]("numPartitions")
        .text("Number of partitions to divide the data into. Recommended be the same as the number of executors submitted through spark-submit.")
        .action((x, c) => c.copy(numPartitions = x))
      opt[String]("validationPath")
        .text("Optional path to delimited text file(s) (e.g. csv/tsv) to be used for validation. All the fields should be numeric (ignored columns can be anything). Categorical fields should be enumerated from 0 to K-1 where K is the cardinality of the field.")
        .action((x, c) => c.copy(validationOptions = c.validationOptions.copy(validationPath = x)))
      opt[String]("delimiter")
        .text("Delimiter string for the input data. The default is \"\\t\"")
        .action((x, c) => c.copy(delimiter = x))
      opt[Int]("labelIndex")
        .text("Label column index in training/validation data. The default is 0. All the other columns are used as features.")
        .action((x, c) => c.copy(labelIndex = x))
      opt[String]("categoricalFeatureIndices")
        .text("A comma separated indices for categorical features in training/validation data. Categorical fields should be enumerated from 0 to K-1 where K is the cardinality of the field. The default is empty (no features are categorical).")
        .action((x, c) => c.copy(categoricalFeatureIndices = x.split(",").map(value => value.toInt).toSet))
      opt[String]("indicesToIgnore")
        .text("A comma separated indices of columns to be ignored during training. The default is empty (no columns are ignored).")
        .action((x, c) => c.copy(indicesToIgnore = x.split(",").map(value => value.toInt).toSet))
      opt[String]("forestType")
        .text("The forest type can be either InfoGain (for classification) or Variance (for regression). The default is InfoGain.")
        .action((x, c) => c.copy(forestType = ForestType.withName(x)))
      opt[String]("discretizationType")
        .text("Type of discretization to do on features. Supports EqualWidth or EqualFrequency. The default is EqualFrequency.")
        .action((x, c) => c.copy(discretizationType = DiscretizationType.withName(x)))
      opt[Int]("maxNumNumericBins")
        .text("Maximum number of bins when quantizing numeric features. If both numeric bin count and categorical cardinality are between 0 and 256, Byte is used to represent features. Otherwise, Short is used. The maximum value is 65536. A smaller number would speed up the process. The default is 256.")
        .action((x, c) => c.copy(maxNumNumericBins = x))
      opt[Int]("maxCategoricalCardinality")
        .text("Maximum cardinality allowed for categorical features. If both numeric bin count and categorical cardinality are between 0 and 256, Byte is used to represent features. Otherwise, Short is used. The maximum value is 65536. The default is 256.")
        .action((x, c) => c.copy(maxNumCategoricalBins = x))
      opt[Boolean]("sampleWithReplacement")
        .text("Whether to do bagging with or without replacement. The default is true.")
        .action((x, c) => c.copy(sampleWithReplacement = x))
      opt[Double]("sampleRate")
        .text("The bagging sampling rate. Should be between 0 and 1. The default is 1 (100% sampling).")
        .action((x, c) => c.copy(sampleRate = x))
      opt[Int]("mtry")
        .text("Number of random features to use per tree node. The default value -1 means it'll be automatically determined. For classification, sqrt(numFeatures) is used. For regression, numFeatures / 3 is used.")
        .action((x, c) => c.copy(mtry = x))
      opt[Int]("minSplitSize")
        .text("The minimum number of samples that a node should see to be eligible for splitting. The default is 2 (means trees will be fully grown) for classification and 10 for regression (regression seems to benefit from some form of regularization).")
        .action((x, c) => c.copy(minSplitSize = x))
      opt[Int]("maxDepth")
        .text("The maximum depth of the tree to be trained. The default value -1 means that there's no limit.")
        .action((x, c) => c.copy(maxDepth = x))
      opt[Int]("numRowFiltersPerIter")
        .text("[Advanced] Number of row filters per iteration. The more row filters per iteration, the more distributed node splits can be performed per iteration. The default value -1 means automatic determination.")
        .action((x, c) => c.copy(numRowFiltersPerIter = x))
      opt[Int]("subTreeThreshold")
        .text("[Advanced] The threshold on the number of samples that a node should see before the node is trained as a sub-tree locally in an executor. The default value -1 means automatic determination (currently defaults to 60000).")
        .action((x, c) => c.copy(subTreeThreshold = x))
      opt[Int]("numSubTreesPerIter")
        .text("[Advanced] Number of sub trees to train per iteration. It could speed up the training process if more sub trees are trained per iteration. The default value -1 means automatic determination (depends on the number of executors).")
        .action((x, c) => c.copy(numSubTreesPerIter = x))
      opt[Int]("pauseDuration")
        .text("Time to pause after finished with training in seconds. This is useful for some Yarn clusters where the log messages are not stored after jobs are finished. The default is 0 (no pause).")
        .action((x, c) => c.copy(pauseDuration = x))
      opt[Boolean]("useLogLossForValidation")
        .text("Whether to use log loss for validation.")
        .action((x, c) => c.copy(validationOptions = c.validationOptions.copy(useLogLossForValidation = x)))
      checkConfig(config =>
        if (config.numTrees <= 0) failure("The number of trees must be greater than 0.")
        else if (config.labelIndex < 0) failure("labelIndex " + config.labelIndex + " is invalid.")
        else if (config.categoricalFeatureIndices.foldLeft(false)((invalid, value) => value < 0 || invalid)) failure("The categorical feature indices contain invalid values.")
        else if (config.indicesToIgnore.foldLeft(false)((invalid, value) => value < 0 || invalid)) failure("Indices-to-ignore contain invalid values.")
        else if (config.maxNumNumericBins <= 1 || config.maxNumNumericBins > 65536) failure("The maximum number of numeric bins " + config.maxNumNumericBins + " is invalid.")
        else if (config.maxNumCategoricalBins <= 1 || config.maxNumCategoricalBins > 65536) failure("The maximum categorical cardinality " + config.maxNumCategoricalBins + " is invalid.")
        else if (config.sampleRate <= 0.0 || config.sampleRate > 1.0) failure("The sample rate " + config.sampleRate + " is not valid.")
        else if (config.mtry <= -2 || config.mtry == 0) failure("mtry " + config.mtry + " is invalid.")
        else if (config.minSplitSize < 2 && config.minSplitSize != -1) failure("minSplitSize " + config.minSplitSize + " is invalid. It should be greater than 1.")
        else if (config.numRowFiltersPerIter <= -2 || config.numRowFiltersPerIter == 0) failure("The number of row filters per iter should be either -1 (automatic) or greater than 0.")
        else if (config.subTreeThreshold <= -2) failure("subTreeThreshold " + config.subTreeThreshold + " is invalid.")
        else if (config.numSubTreesPerIter <= -2) failure("numSubTreesPerIter " + config.numSubTreesPerIter + " is invalid.")
        else if (config.pauseDuration < 0) failure("The pause duration should not be negative.")
        else success)
    }

    // Parse the argument and then run.
    parser.parse(args, defaultConfig).map { config =>
      run(config)
    }.getOrElse {
      Thread.sleep(10000) // Sleep for 10 seconds so that people may see error messages in Yarn clusters where logs are not stored.
      sys.exit(1)
    }
  }

  /**
   * Run training.
   * @param config Configuration for the algorithm.
   */
  def run(config: RunnerConfig): Unit = {
    val conf = new SparkConf().setAppName("SequoiaForestRunner")
    val sc = new SparkContext(conf)

    try {
      val indicesToIgnore = config.indicesToIgnore
      val labelIndex = config.labelIndex

      val delimiter = config.delimiter
      val lineParser = (line: String) => {
        val elems = line.split(delimiter)
        val label = elems(labelIndex).toDouble
        val features = new Array[Double](elems.length - 1 - indicesToIgnore.size)
        var col = 0
        var featId = 0
        while (col < elems.length) {
          if (col != labelIndex && !indicesToIgnore.contains(col)) {
            // Feature indices get shifted after subtracting the label index and the ignored columns.
            val featureValue = if (elems(col) == "") {
              Double.NaN
            } else {
              elems(col).toDouble
            }

            features(featId) = featureValue
            featId += 1
          }

          col += 1
        }

        (label, features)
      }

      val trainingRDD = sc.textFile(config.filePathOptions.inputPath).map(lineParser)

      // Find the total number of columns in the raw data.
      val numColumns = trainingRDD.first()._2.length + 1 + indicesToIgnore.size
      val colIdxToFeatIdxMap = mutable.Map[Int, Int]() // Map from original column indices to feature indices.
      var colIdx = 0
      var featIdx = 0
      // Map column indices to feature indices.
      while (colIdx < numColumns) {
        if (colIdx != labelIndex && !indicesToIgnore.contains(colIdx)) {
          colIdxToFeatIdxMap.put(colIdx, featIdx)
          featIdx += 1
        }

        colIdx += 1
      }

      // We need to re-calculate categorical feature indices after the label column and the ignored columns are removed.
      val categoricalFeatureIndices = config.categoricalFeatureIndices.map(index => colIdxToFeatIdxMap(index)).toSet

      val validationData = config.validationOptions.validationPath match {
        case path if path != null => Some(sc.textFile(config.validationOptions.validationPath).map(lineParser).collect())
        case _ => None
      }

      val notifiee = new ConsoleNotifiee

      val inputRDD = if (config.numPartitions > 1) {
        notifiee.newStatusMessage("Repartitioning the input across " + config.numPartitions + " partitions.")
        val repartitionedRDD = trainingRDD.repartition(config.numPartitions)
        repartitionedRDD.sparkContext.setCheckpointDir(config.filePathOptions.checkpointDir)
        repartitionedRDD.checkpoint() // Repartitioned RDD has to be checkpointed. Otherwise, this won't be fault-tolerant because repartition is not deterministic in row orders.
        repartitionedRDD
      } else {
        trainingRDD
      }

      SequoiaForestTrainer.discretizeAndTrain(
        treeType = if (config.forestType == ForestType.InfoGain) TreeType.Classification_InfoGain else TreeType.Regression_Variance,
        input = inputRDD,
        numTrees = config.numTrees,
        outputStorage = new HDFSForestStorage(trainingRDD.sparkContext.hadoopConfiguration, config.filePathOptions.outputPath),
        validationData = validationData,
        categoricalFeatureIndices = categoricalFeatureIndices,
        notifiee = notifiee,
        discretizationType = config.discretizationType,
        maxNumNumericBins = config.maxNumNumericBins,
        maxNumCategoricalBins = config.maxNumCategoricalBins,
        samplingType = if (config.sampleWithReplacement) SamplingType.SampleWithReplacement else SamplingType.SampleWithoutReplacement,
        samplingRate = config.sampleRate,
        mtry = config.mtry,
        minSplitSize = config.minSplitSize,
        maxDepth = config.maxDepth,
        numNodesPerIteration = config.numRowFiltersPerIter,
        localTrainThreshold = config.subTreeThreshold,
        numSubTreesPerIteration = config.numSubTreesPerIter,
        useLogLossForValidation = config.validationOptions.useLogLossForValidation,
        checkpointDir = config.filePathOptions.checkpointDir,
        checkpointInterval = config.filePathOptions.checkpointInterval,
        distributedNodeSplits = config.distributedNodeSplits)
    } catch {
      case e: Exception => println("Exception:" + e.toString)
    } finally {
      Thread.sleep(config.pauseDuration * 1000)
      sc.stop()
    }
  }
}
