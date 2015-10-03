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

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Random

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import spark_ml.discretization._
import spark_ml.gradient_boosting.loss.LossFunction
import spark_ml.model.gb.{GBInternalTree, GradientBoostedTrees, GradientBoostedTreesFactory, GradientBoostedTreesStore}
import spark_ml.tree_ensembles._
import spark_ml.util._
import spire.implicits._

/**
 * GradientBoosting object used for training.
 */
object GradientBoosting {
  /**
   * Train a gradient boosted tree model on the given labeled data points. The
   * features will be converted to unsigned byte bin Ids.
   * @param input A labeled data point with double values for labels and features.
   * @param fracTraining Fraction of the input data to use for training. The
   *                     rest are used for validations.
   * @param columnNames Names of the label and the features.
   * @param catIndices Categorical feature indices.
   * @param gbOpts Options for GB.
   * @param discOpts Discretization options.
   * @param gbtFactory The factory for the return model. The developer can
   *                   return his/her own GBT implementation through this.
   * @param notifiee Notifiee.
   * @param storageLevel Storage level of Spark.
   * @return A trained GB model.
   */
  def train_unsignedByte(
    input: RDD[(Double, Array[Double])],
    fracTraining: Double,
    columnNames: (String, Array[String]),
    catIndices: Set[Int],
    gbOpts: GradientBoostingOptions,
    discOpts: DiscretizationOptions,
    gbtFactory: GradientBoostedTreesFactory,
    notifiee: ProgressNotifiee,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): GradientBoostedTrees = {

    if (gbOpts.verbose) {
      notifiee.newStatusMessage("Training a GBT with the unsigned byte as the discretized feature type.")
    }

    train[Byte](
      input = input,
      fracTraining = fracTraining,
      columnNames = columnNames,
      catIndices = catIndices,
      gbOpts = gbOpts,
      discOpts = discOpts,
      gbtFactory = gbtFactory,
      notifiee = notifiee,
      storageLevel = storageLevel,
      featureHandler = new UnsignedByteHandler)
  }

  /**
   * Train a gradient boosted tree model on the given labeled data points. The
   * features will be converted to unsigned short bin Ids.
   * @param input A labeled data point with double values for labels and features.
   * @param fracTraining Fraction of the input data to use for training. The
   *                     rest are used for validations.
   * @param columnNames Names of the label and the features.
   * @param catIndices Categorical feature indices.
   * @param gbOpts Options for GB.
   * @param discOpts Discretization options.
   * @param gbtFactory The factory for the return model. The developer can
   *                   return his/her own GBT implementation through this.
   * @param notifiee Notifiee.
   * @param storageLevel Storage level of Spark.
   * @return A trained GB model.
   */
  def train_unsignedShort(
    input: RDD[(Double, Array[Double])],
    fracTraining: Double,
    columnNames: (String, Array[String]),
    catIndices: Set[Int],
    gbOpts: GradientBoostingOptions,
    discOpts: DiscretizationOptions,
    gbtFactory: GradientBoostedTreesFactory,
    notifiee: ProgressNotifiee,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): GradientBoostedTrees = {

    if (gbOpts.verbose) {
      notifiee.newStatusMessage("Training a GBT with the unsigned short as the discretized feature type.")
    }

    train[Short](
      input = input,
      fracTraining = fracTraining,
      catIndices = catIndices,
      columnNames = columnNames,
      gbOpts = gbOpts,
      discOpts = discOpts,
      gbtFactory = gbtFactory,
      notifiee = notifiee,
      storageLevel = storageLevel,
      featureHandler = new UnsignedShortHandler)
  }

  /**
   * Train a gradient boosted tree model on the given labeled data points.
   * @param input A labeled data point with double values for labels and features.
   * @param fracTraining Fraction of the input data to use for training. The
   *                     rest are used for validations.
   * @param columnNames Names of the label and the features.
   * @param catIndices Categorical feature indices.
   * @param gbOpts Options for GB.
   * @param discOpts Discretization options.
   * @param gbtFactory The factory for the return model. The developer can
   *                   return his/her own GBT implementation through this.
   * @param notifiee Notifiee.
   * @param storageLevel Storage level of Spark.
   * @param featureHandler Type handler for the features.
   * @tparam T Expected type for the features.
   * @return A trained GB model.
   */
  def train[@specialized(Byte, Short) T: ClassTag](
    input: RDD[(Double, Array[Double])],
    fracTraining: Double,
    columnNames: (String, Array[String]),
    catIndices: Set[Int],
    gbOpts: GradientBoostingOptions,
    discOpts: DiscretizationOptions,
    gbtFactory: GradientBoostedTreesFactory,
    notifiee: ProgressNotifiee,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    featureHandler: DiscretizedFeatureHandler[T],
    seed: Int = 1): GradientBoostedTrees = {

    notifiee.newStatusMessage(
      "Training Gradient Boosted Trees with the following options..."
    )

    notifiee.newStatusMessage(
      "\n" + gbOpts.toString
    )

    notifiee.newStatusMessage(
      "Discretizing the training data with the following options..."
    )

    notifiee.newStatusMessage(
      "\n" + discOpts.toString
    )

    val numFeatures = columnNames._2.length
    notifiee.newStatusMessage("There are total " + numFeatures + " features...")

    notifiee.newStatusMessage("Finding numeric bins for each feature...")

    // Find bins according to the discretization options.
    val (labelSummary, featureBins) =
      (
        discOpts.discType match {
          case DiscType.EqualWidth =>
            new EqualWidthBinFinder()
          case DiscType.EqualFrequency =>
            new EqualFrequencyBinFinderFromSample(
              maxSampleSize = discOpts.maxSampleSizeForDisc,
              seed = seed
            )
          case DiscType.MinimumEntropy =>
            new EntropyMinimizingBinFinderFromSample(
              maxSampleSize = discOpts.maxSampleSizeForDisc,
              seed = seed
            )
          case DiscType.MinimumVariance =>
            new VarianceMinimizingBinFinderFromSample(
              maxSampleSize = discOpts.maxSampleSizeForDisc,
              seed = seed
            )
        }
      ).findBins(
        data = input,
        columnNames = columnNames,
        catIndices = catIndices,
        maxNumBins = discOpts.maxNumericBins,
        expectedLabelCardinality = gbOpts.lossFunction.getLabelCardinality,
        notifiee = notifiee
      )
    val featureBinsAsArray = featureBins.toArray

    val sampleRdds = input.randomSplit(
      Array(fracTraining, 1.0 - fracTraining)
    )

    val trainingData = sampleRdds(0)
    val validationData =
      if (fracTraining < 1.0) {
        Some(sampleRdds(1))
      } else {
        None
      }

    if (gbOpts.verbose) {
      notifiee.newStatusMessage("Training data has " + trainingData.count() + " rows.")
      if (validationData.isDefined) {
        notifiee.newStatusMessage("Validation data has " + validationData.get.count() + " rows.")
      } else {
        notifiee.newStatusMessage("Validation data are undefined.")
      }
    }

    if (gbOpts.verbose) {
      featureBins.zipWithIndex.foreach {
        case (bins, featIdx) =>
          val featureName = columnNames._2(featIdx)
          notifiee.newStatusMessage(
            "The feature " + featureName + " has " + bins.getCardinality + " bins."
          )

          bins match {
            case numericBins: NumericBins =>
              notifiee.newStatusMessage(
                "The feature " + featureName + " is numeric."
              )
              if (numericBins.missingValueBinIdx.isDefined) {
                notifiee.newStatusMessage(
                  "The feature " + featureName + " has a NaN bin " + numericBins.missingValueBinIdx.get
                )
              }
              notifiee.newStatusMessage(
                "The feature " + featureName + " has the following numeric bins :"
              )
              notifiee.newStatusMessage(numericBins.bins.map(_.toString).mkString(" "))
            case catBins: CategoricalBins =>
              notifiee.newStatusMessage(
                "The feature " + featureName + " is categorical."
              )
          }
      }
    }

    // Let the model factory know the feature bins.
    gbtFactory.setFeatureBins(featureBinsAsArray)

    notifiee.newStatusMessage("Finished computing bins for each numeric feature...")
    val lossFunction = gbOpts.lossFunction
    val labelCardinality = gbOpts.lossFunction.getLabelCardinality

    // See that the label is valid for the loss type.
    if (labelCardinality.isDefined) {
      if (labelSummary.restCount > 0) {
        throw InvalidLabelException(
          "The label should have " + labelSummary.expectedCardinality.get +
          " unique values. However, we found more."
        )
      }

      var numLabelCatsThatExist = 0
      labelSummary.catCounts.get.foreach {
        cnt => if (cnt > 0L) { numLabelCatsThatExist += 1 }
      }
      if (numLabelCatsThatExist < labelCardinality.get) {
        throw InvalidLabelException(
          "The label should have " + labelSummary.expectedCardinality.get +
          " unique values. However, we only found " + numLabelCatsThatExist
        )
      }
    } else {
      // Print an error message if we have close to 0 variance in the label for
      // regression but keep continuing.
      if (math.abs(labelSummary.runningSqrAvg - labelSummary.runningAvg * labelSummary.runningAvg) < 1e-19) {
        notifiee.newErrorMessage(
          "The label has an extremely small variance with the average value of " + labelSummary.runningAvg.toString + ". " +
          "The regression algorithm may not learn anything meaningful."
        )
      }
    }

    notifiee.newStatusMessage("Persisting the raw labels...")

    // Persist the raw label since it needs to be reused again and again for
    // training subsequent trees.
    val rawLabelRdd: RDD[Double] = trainingData.map(r => r._1).persist(storageLevel)

    // If the checkpoint directory is present, let's checkpoint the raw labels
    // as well.
    val checkpointDir = gbOpts.checkpointDir
    if (checkpointDir.isDefined) {
      trainingData.sparkContext.setCheckpointDir(checkpointDir.get)
      notifiee.newStatusMessage(
        "Writing the raw label checkpoint to " + checkpointDir.get
      )
      rawLabelRdd.checkpoint()
    }

    notifiee.newStatusMessage("Discretizing the features and persisting them...")

    // Transform the feature values into bin Ids.
    val discFeatureRdd: RDD[Array[T]] =
      Discretizer.transformFeatures(
        trainingData,
        featureBins,
        featureHandler
      ).persist(storageLevel)

    if (gbOpts.verbose) {
      val discFeatureRddSample = discFeatureRdd.take(50)
      notifiee.newStatusMessage("Discretized features look like following :")
      discFeatureRddSample.foreach {
        featureVec =>
          notifiee.newStatusMessage(
            featureVec.map(featureHandler.convertToInt).mkString(",")
          )
      }
    }

    // If the checkpoint is defined, we'll checkpoint this, too.
    if (checkpointDir.isDefined) {
      notifiee.newStatusMessage(
        "Writing the discretized features checkpoint to " + checkpointDir.get
      )
      discFeatureRdd.checkpoint()
    }

    // See if we have validation data. If we do, transform them as well.
    val (valRawLabelRdd, valDiscFeatureRdd) =
      if (validationData.isDefined) {
        notifiee.newStatusMessage("Persisting the validation data raw labels...")
        val valL = validationData.get.map(r => r._1).persist(storageLevel)
        if (checkpointDir.isDefined) {
          notifiee.newStatusMessage(
            "Writing the validation label checkpoint to " + checkpointDir.get
          )
          valL.checkpoint()
        }

        notifiee.newStatusMessage(
          "Discretizing the validation data features and persisting them..."
        )

        val valF = Discretizer.transformFeatures(
          validationData.get,
          featureBins,
          featureHandler
        ).persist(storageLevel)
        if (checkpointDir.isDefined) {
          notifiee.newStatusMessage(
            "Writing the validation features checkpoint to " + checkpointDir.get
          )
          valF.checkpoint()
        }

        (Some(valL), Some(valF))
      } else {
        (None, None)
      }

    // Compute the initial estimate.
    notifiee.newStatusMessage(
      "Computing the model's initial constant estimate..."
    )

    // To compute initial estimates, aggregate the raw label values.
    val initAggregator = rawLabelRdd.mapPartitions(
      samplePoints => {
        val lossAggregator = lossFunction.createAggregator

        while (samplePoints.hasNext) {
          val sampleLabel = samplePoints.next()
          lossAggregator.addSamplePoint(
            sampleLabel,
            1.0,
            0.0
          )
        }

        Array(lossAggregator).toIterator
      }
    ).reduce((a, b) => a.mergeInPlace(b))

    val initValue = initAggregator.computeInitialValue()
    notifiee.newStatusMessage(
      "The initial estimate is " + initValue
    )

    // Update the current predictions and gradients.
    notifiee.newStatusMessage(
      "Setting the initial prediction (initial estimate) and " +
        "the initial gradient and persisting them..."
    )
    val gradientComputer = lossFunction.createAggregator
    val predCache = PredCache.createPredCache(
      rawLabelRdd.map(_ => initValue),
      gbOpts.shrinkage,
      storageLevel,
      gbOpts.checkpointDir,
      gbOpts.predCheckpointInterval
    )
    var curGradRdd = rawLabelRdd.zip(predCache.getRdd).map {
      case (label, curEstimate) => gradientComputer.computeGradient(label, curEstimate)
    }.persist(storageLevel)

    if (gbOpts.verbose) {
      notifiee.newStatusMessage("The initial gradient samples are :")
      val curGradRddSample = curGradRdd.take(50)
      curGradRddSample.foreach {
        curGrad => notifiee.newStatusMessage(curGrad.toString)
      }
    }

    // Likewise, we want to keep track of validation data predictions as well
    // to compute the validation deviance.
    val valPredCache: Option[PredCache] =
      if (validationData.isDefined) {
        Some(
          PredCache.createPredCache(
            valRawLabelRdd.get.map(_ => initValue),
            gbOpts.shrinkage,
            storageLevel,
            gbOpts.checkpointDir,
            gbOpts.predCheckpointInterval
          )
        )
      } else {
        None
      }

    // Compute the current training deviance.
    notifiee.newStatusMessage(
      "Computing the initial training deviance..."
    )
    val trainingDevianceHistory = new mutable.ListBuffer[Double]()
    val initTrainingDeviance = computeDeviance(
      labelRdd = rawLabelRdd,
      predRdd = predCache.getRdd,
      lossFunction = lossFunction
    )
    notifiee.newStatusMessage(
      "The initial training deviance is " + initTrainingDeviance + "..."
    )
    trainingDevianceHistory += initTrainingDeviance

    // Compute the current validation deviance.
    val validationDevianceHistory: Option[mutable.ListBuffer[Double]] =
      if (validationData.isDefined) {
        notifiee.newStatusMessage(
          "Computing the initial validation deviance..."
        )
        val vdh = new mutable.ListBuffer[Double]()
        val initValDeviance = computeDeviance(
          labelRdd = valRawLabelRdd.get,
          predRdd = valPredCache.get.getRdd,
          lossFunction = lossFunction
        )
        notifiee.newStatusMessage(
          "The initial validation deviance is " + initValDeviance + "..."
        )
        vdh += initValDeviance
        Some(vdh)
      } else {
        None
      }

    // Initialize the model store that we can pass into the tree trainer.
    val gbtStore =
      new GradientBoostedTreesStore(
        lossFunction = lossFunction,
        initVal = initValue,
        shrinkage = gbOpts.shrinkage
      )

    // If we have validation data, we can compute the best validation deviance
    // and the optimal number of trees for validation.
    var optimalTreeCnt: Int = 0
    var bestValDeviance: Double = Double.MaxValue

    val rng = new Random(seed)

    // Now, train trees sequentially, using the gradients as the labels.
    var curTreeCnt = 0
    while (curTreeCnt < gbOpts.numTrees) {
      // We need to bag every iteration with or without replacements.
      val curBagRdd = Bagger.getBagRdd(
        data = rawLabelRdd,
        numSamples = 1,
        baggingType = gbOpts.baggingType,
        baggingRate = gbOpts.baggingRate,
        rng.nextInt()).persist(storageLevel)

      // Create an idCache object for tree training.
      val idCache = IdCache.createIdCache(
        numTrees = 1,
        data = rawLabelRdd,
        storageLevel = storageLevel,
        checkpointDir = checkpointDir,
        checkpointInterval = gbOpts.idCacheCheckpointInterval
      )

      // For each tree, the gradient at that moment is the label.
      val treeTrainingRdd = curGradRdd.zip(discFeatureRdd).zip(curBagRdd)
      val quantizedData_Rdd = new QuantizedData_ForTrees_Rdd(
        data = treeTrainingRdd,
        idCache = idCache,
        featureBinsInfo = featureBinsAsArray,
        featureHandler = featureHandler
      )
      val treeOpts = TreeForestTrainerOptions(
        numTrees = 1,
        splitCriteria = SplitCriteria.Regression_Variance,
        mtry = numFeatures,
        maxDepth = gbOpts.maxTreeDepth,
        minSplitSize = gbOpts.minSplitSize,
        catSplitType = gbOpts.catSplitType,
        maxSplitsPerIter = 1024, // We are not likely to train deep trees.
        subTreeWeightThreshold = 0.0, // No need for sub-trees since
                                      // we're only training shallow trees.
        maxSubTreesPerIter = 0,
        numClasses = None,
        verbose = gbOpts.verbose
      )
      gbtStore.initNewTree()
      TreeForestTrainer.train(
        trainingData = quantizedData_Rdd,
        featureBinsInfo = featureBinsAsArray,
        trainingOptions = treeOpts,
        modelStore = gbtStore,
        notifiee = notifiee,
        rng = rng
      )
      curTreeCnt += 1
      idCache.close()
      curGradRdd.unpersist(blocking = true)
      // Get terminal node statistics to update terminal predictions.
      val tree: GBInternalTree = gbtStore.getInternalTree(curTreeCnt - 1)

      if (gbOpts.verbose) {
        // Print the trained tree into an ASCII.
        notifiee.newStatusMessage("The trained tree looks like :")
        notifiee.newStatusMessage(tree.toString)
      }

      if (gbOpts.fineTuneTerminalNodes && lossFunction.canRefineNodeEstimate) {
        // Update the tree's node predictions.
        notifiee.newStatusMessage(
          "Optimizing the latest tree's node predictions..."
        )
        optimizeNodePredictions(
          tree = tree,
          labelRdd = rawLabelRdd,
          predRdd = predCache.getRdd,
          discFeatureRdd = discFeatureRdd,
          bagRdd = curBagRdd,
          lossFunction = lossFunction,
          featureHandler = featureHandler
        )

        if (gbOpts.verbose) {
          notifiee.newStatusMessage("After node prediction refinements, the tree looks like :")
          notifiee.newStatusMessage(tree.toString)
        }
      }

      // Now we can unpersist the previous bag RDD.
      curBagRdd.unpersist(blocking = true)

      notifiee.newStatusMessage(
        "Updating predictions on the data points..."
      )
      predCache.updatePreds(
        discFeatureRdd,
        tree,
        featureHandler
      )

      // Get a new gradient using updated predictions.
      notifiee.newStatusMessage(
        "Updating gradients..."
      )
      curGradRdd = rawLabelRdd.zip(predCache.getRdd).map(
        samplePoint =>
          gradientComputer.computeGradient(samplePoint._1, samplePoint._2)
      ).persist(storageLevel)

      if (gbOpts.verbose) {
        notifiee.newStatusMessage("The current predictions are :")
        val curPredRddSample = predCache.getRdd.take(50)
        curPredRddSample.foreach {
          curPred => notifiee.newStatusMessage(curPred.toString)
        }

        notifiee.newStatusMessage("The current gradients are :")
        val curGradRddSample = curGradRdd.take(50)
        curGradRddSample.foreach {
          curGrad => notifiee.newStatusMessage(curGrad.toString)
        }
      }

      // Compute the training deviance.
      val trainingDeviance = computeDeviance(
        labelRdd = rawLabelRdd,
        predRdd = predCache.getRdd,
        lossFunction = lossFunction
      )
      trainingDevianceHistory += trainingDeviance
      notifiee.newStatusMessage(
        "The training deviance after " + curTreeCnt + " trees is " +
          trainingDeviance + "..."
      )

      val progressMessage = new StringBuilder()
      progressMessage.append("Trained " + curTreeCnt.toString + " trees. ")
      progressMessage.append("Training deviance is " + trainingDeviance.toString + ".")

      if (validationData.nonEmpty) {
        valPredCache.get.updatePreds(
          valDiscFeatureRdd.get,
          tree,
          featureHandler
        )
        // Compute the validation deviance.
        val valDeviance = computeDeviance(
          labelRdd = valRawLabelRdd.get,
          predRdd = valPredCache.get.getRdd,
          lossFunction = lossFunction
        )
        validationDevianceHistory.get += valDeviance
        notifiee.newStatusMessage(
          "The validation deviance after " + curTreeCnt + " trees is " +
            valDeviance + "..."
        )

        progressMessage.append(" Validation deviance is " + valDeviance.toString + ".")

        if (valDeviance < bestValDeviance) {
          bestValDeviance = valDeviance
          optimalTreeCnt = curTreeCnt
        }

        notifiee.newStatusMessage(
          "The optimal number of trees so far is " + optimalTreeCnt +
            " with the validation deviance of " + bestValDeviance + "..."
        )
      }

      notifiee.newProgressMessage(progressMessage.toString())
    }

    // Delete all the RDD cache/checkpoints.
    predCache.close()
    if (valPredCache.isDefined) {
      valPredCache.get.close()
    }

    // Delete all the other checkpoints.
    if (checkpointDir.isDefined) {
      val fs = FileSystem.get(trainingData.sparkContext.hadoopConfiguration)

      if (rawLabelRdd.getCheckpointFile.isDefined) {
        fs.delete(new Path(rawLabelRdd.getCheckpointFile.get), true)
        println("Deleted the raw label checkpoint at " + rawLabelRdd.getCheckpointFile.get)
      }
      rawLabelRdd.unpersist(blocking = true)

      if (discFeatureRdd.getCheckpointFile.isDefined) {
        fs.delete(new Path(discFeatureRdd.getCheckpointFile.get), true)
        println("Deleted the discretized features checkpoint at " + discFeatureRdd.getCheckpointFile.get)
      }
      discFeatureRdd.unpersist(blocking = true)

      if (valRawLabelRdd.isDefined) {
        if (valRawLabelRdd.get.getCheckpointFile.isDefined) {
          fs.delete(new Path(valRawLabelRdd.get.getCheckpointFile.get), true)
          println("Deleted the validation label checkpoint at " + valRawLabelRdd.get.getCheckpointFile.get)
        }
        valRawLabelRdd.get.unpersist(blocking = true)
      }

      if (valDiscFeatureRdd.isDefined) {
        if (valDiscFeatureRdd.get.getCheckpointFile.isDefined) {
          fs.delete(new Path(valDiscFeatureRdd.get.getCheckpointFile.get), true)
          println("Deleted the validation features checkpoint at " + valDiscFeatureRdd.get.getCheckpointFile.get)
        }
        valDiscFeatureRdd.get.unpersist(blocking = true)
      }
    }

    gbtFactory.setTrainingDevianceHistory(trainingDevianceHistory)

    if (validationDevianceHistory.isDefined) {
      gbtFactory.setValidationDevianceHistory(validationDevianceHistory.get)
    }

    if (optimalTreeCnt > 0) {
      gbtFactory.setOptimalTreeCnt(optimalTreeCnt)
    }

    gbtFactory.createGradientBoostedTrees(gbtStore)
  }

  /**
   * Compute deviance given the label and the prediction.
   * @param labelRdd Label RDD.
   * @param predRdd Prediction RDD.
   * @param lossFunction Loss function to use for deviance calculations.
   * @return Deviance given the labels and predictions.
   */
  def computeDeviance(
    labelRdd: RDD[Double],
    predRdd: RDD[Double],
    lossFunction: LossFunction): Double = {
    labelRdd.zip(predRdd).mapPartitions(
      samplePoints => {
        val lossAggregator = lossFunction.createAggregator
        while (samplePoints.hasNext) {
          val (l, p) = samplePoints.next()
          lossAggregator.addSamplePoint(
            label = l,
            weight = 1.0,
            curPred = p
          )
        }

        Array(lossAggregator).toIterator
      }
    ).reduce((a, b) => a.mergeInPlace(b)).computeDeviance()
  }

  /**
   * Update the given tree's node predictions by directly optimizing the
   * predictions to minimize the loss function.
   * @param tree The internal tree whose node predictions we'll optimize.
   * @param labelRdd Label RDD.
   * @param predRdd Current prediction RDD.
   * @param discFeatureRdd Discretized feature RDD.
   * @param bagRdd Bagging RDD.
   * @param lossFunction Loss function.
   * @param featureHandler Feature type handler.
   * @tparam T Feature type.
   */
  def optimizeNodePredictions[@specialized(Byte, Short) T: ClassTag](
    tree: GBInternalTree,
    labelRdd: RDD[Double],
    predRdd: RDD[Double],
    discFeatureRdd: RDD[Array[T]],
    bagRdd: RDD[Array[Byte]],
    lossFunction: LossFunction,
    featureHandler: DiscretizedFeatureHandler[T]): Unit = {
    val nodeAggregators =
      labelRdd.zip(predRdd).zip(discFeatureRdd).zip(bagRdd).mapPartitions(
        samplePoints => {
          // Initialize the tree's nodes with new aggregators.
          tree.initNodeFinetuners(lossFunction)

          while (samplePoints.hasNext) {
            val samplePoint = samplePoints.next()
            if (samplePoint._2(0) > 0) {
              tree.addSamplePointToMatchingNodes(
                samplePoint._1,
                samplePoint._2(0).toDouble,
                featureHandler
              )
            }
          }

          Array(tree.nodeAggregators).toIterator
        }
      ).reduce {
        (aggregators1, aggregators2) =>
          val (startNodeId, endNodeId) = aggregators1.getKeyRange
          cfor(startNodeId)(_ <= endNodeId, _ + 1)(
            nodeId => {
              val aggregator1 = aggregators1.get(nodeId)
              val aggregator2 = aggregators2.get(nodeId)
              aggregator1.mergeInPlace(aggregator2)
            }
          )

          aggregators1
      }

    // Update node predictions for the tree. I.e., this is sort of doing a line
    // search for each terminal node.
    val (startNodeId, endNodeId) = nodeAggregators.getKeyRange
    cfor(startNodeId)(_ <= endNodeId, _ + 1)(
      nodeId => {
        val aggregator = nodeAggregators.get(nodeId)
        val newPrediction = aggregator.computeNodeEstimate()
        tree.updateNodePrediction(nodeId, newPrediction)
      }
    )
  }
}
