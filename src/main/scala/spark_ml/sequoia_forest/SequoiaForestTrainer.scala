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

import scala.util.Random
import scala.collection.mutable

import spark_ml.discretization._
import java.io._
import java.util.Calendar
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * The trainer for Sequoia Forest.
 */
object SequoiaForestTrainer {
  /**
   * Train a Sequoia Forest after discretizing the input.
   * It's recommended that the input is not cached prior to running this, as this will quantize data and then cache them.
   * @param treeType Either classification (infogain) or regression.
   * @param input An RDD of label/features tuple. Label should be from 0...(K - 1) for K target classes. Features should be Doubles for both numeric and categorical features.
   * @param numTrees Number of trees that we want to train.
   * @param outputStorage The storage object where output trees are written to.
   * @param validationData Optional validation data array. If this is not None, then the trained model would also be stored in memory.
   * @param categoricalFeatureIndices Indices of categorical features.
   * @param notifiee A notifiee object that will receive training progress messages.
   * @param discretizationType The discretization type to use for transforming features to Bin IDs.
   * @param maxNumNumericBins Maximum number of bins to use for numeric features during discretization. It should be equal to or smaller than unsigned Byte/Short maximum.
   * @param maxNumCategoricalBins Maximum number of bins to use for categorical features during discretization. The cardinality of categorical variables should be smaller than/equal to this.
   * @param samplingType Sampling type (either with-replacement or without-replacement).
   * @param samplingRate Sampling rate (between 0 and 1).
   * @param mtry Number of random features per node. If this is -1, sqrt would be used.
   * @param minSplitSize The minimum size of nodes that are eligible for splitting. 2 is the lowest number (means fully growing trees).
   * @param maxDepth The maximum depth of the tree. -1 implies that there's no limit.
   * @param numNodesPerIteration The number of row filters to use in each RDD iteration for distributed node splits. -1 means that it'll be automatically determined (based on memory availability).
   * @param localTrainThreshold The number of training samples a node sees before the whole sub-tree is trained locally. -1 means that it'll be automatically determined (based on memory availability).
   * @param numSubTreesPerIteration The number of sub trees to train in each RDD iteration. -1 means that it'll be automatically determined (based on memory availability).
   * @param storageLevel Spark persistence level (whether data are cached to memory, local-disk or both). Defaults to MEMORY_AND_DISK.
   * @return A trained sequoia forest object if there's one in memory. Otherwise, the trees would be stored in the output path only and this would return None.
   */
  def discretizeAndTrain(
    treeType: TreeType.TreeType,
    input: RDD[(Double, Array[Double])],
    numTrees: Int,
    outputStorage: ForestStorage,
    validationData: Option[Array[(Double, Array[Double])]],
    categoricalFeatureIndices: Set[Int],
    notifiee: ProgressNotifiee,
    discretizationType: DiscretizationType.DiscretizationType,
    maxNumNumericBins: Int,
    maxNumCategoricalBins: Int,
    samplingType: SamplingType.SamplingType,
    samplingRate: Double,
    mtry: Int,
    minSplitSize: Long,
    maxDepth: Int,
    numNodesPerIteration: Int,
    localTrainThreshold: Int,
    numSubTreesPerIteration: Int,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): Option[SequoiaForest] = {

    val maxBinCount = math.max(maxNumNumericBins, maxNumCategoricalBins)
    notifiee.newStatusMessage("The maximum number of bins in any feature is " + maxBinCount)
    notifiee.newStatusMessage("Computing bins for each feature...")

    val (maxLabelValue: Double, featureBins: Array[Bins]) = discretizationType match {
      case DiscretizationType.EqualWidth => EqualWidthDiscretizer.discretizeFeatures(
        input,
        categoricalFeatureIndices,
        labelIsCategorical = true,
        Map[String, String](
          StringConstants.NumBins_Numeric -> maxNumNumericBins.toString,
          StringConstants.MaxCardinality_Categoric -> maxNumCategoricalBins.toString))

      case DiscretizationType.EqualFrequency => EqualFrequencyDiscretizer.discretizeFeatures(
        input,
        categoricalFeatureIndices,
        labelIsCategorical = true,
        Map[String, String](
          StringConstants.NumBins_Numeric -> maxNumNumericBins.toString,
          StringConstants.SubSampleCount_Numeric -> "10000", // TODO: Using 10000 samples to find numeric bins but should make this configurable.
          StringConstants.MaxCardinality_Categoric -> maxNumCategoricalBins.toString))

      case _ => throw new UnsupportedOperationException("Currently, only equal-width or equal-frequency discretizations are supported.")
    }

    // If this is classification, the label has to be a non-negative integer.
    if (treeType == TreeType.Classification_InfoGain && (maxLabelValue < 0.0 || maxLabelValue.toInt.toDouble != maxLabelValue)) {
      throw new InvalidCategoricalValueException(maxLabelValue + " is not a valid target class value.")
    }

    notifiee.newStatusMessage("Finished computing bins for each feature...")

    // Now transform data into bin IDs.
    val discretizedBaggedInput: DiscretizedData = maxBinCount match {
      case binCount if binCount <= 256 =>
        notifiee.newStatusMessage("Discretizing the input data features into unsigned Byte bin IDs...")
        val txData = Discretizer.transformFeaturesToUnsignedByteBinIds(input, featureBins)
        notifiee.newStatusMessage("Bagging the input data...")
        val baggedInput = Bagger.bagRDD[Byte](txData, numTrees, samplingType, samplingRate)
        notifiee.newStatusMessage("Caching (and also materializing) the transformed data...")
        baggedInput.persist(storageLevel)
        notifiee.newStatusMessage("Finished caching the transformed data...")
        UnsignedByteRDD(baggedInput)

      case binCount if binCount > 256 && binCount <= 65536 =>
        notifiee.newStatusMessage("Discretizing the input data features into unsigned Short bin IDs...")
        val txData = Discretizer.transformFeaturesToUnsignedShortBinIds(input, featureBins)
        notifiee.newStatusMessage("Bagging the input data...")
        val baggedInput = Bagger.bagRDD[Short](txData, numTrees, samplingType, samplingRate)
        notifiee.newStatusMessage("Caching (and also materializing) the transformed data...")
        baggedInput.persist(storageLevel)
        notifiee.newStatusMessage("Finished caching the transformed data...")
        UnsignedShortRDD(baggedInput)

      case _ => throw new UnsupportedOperationException("Number of bins greater than 65536 is not supported.")
    }

    notifiee.newStatusMessage("Finished transforming the input data into propert training data...")

    // Determine certain parameters automatically.
    val numFeatures = featureBins.length
    val numRandomFeaturesPerNode = mtry match {
      case x if x == -1 =>
        if (treeType == TreeType.Classification_InfoGain) {
          math.ceil(math.sqrt(numFeatures.toDouble)).toInt
        } else {
          math.ceil(numFeatures.toDouble / 3.0).toInt
        }

      case _ => mtry
    }

    val minSplitSizeActual = minSplitSize match {
      case x if x == -1 =>
        if (treeType == TreeType.Classification_InfoGain) {
          2
        } else {
          10
        }

      case _ => minSplitSize
    }

    // We'll store the trained model in memory iff the validation data is not null.
    // Not recommended to use validation unless the models are expected to be small.
    val storeModelInMemory = validationData != None

    // Get the average number of bins across all the features.
    var binCountSum = 0
    var featureId = 0
    while (featureId < numFeatures) {
      notifiee.newStatusMessage("The number of bins in the feature " + featureId + " is " + featureBins(featureId).getCardinality)
      binCountSum += featureBins(featureId).getCardinality
      featureId += 1
    }

    val avgNumBins = binCountSum.toDouble / numFeatures.toDouble
    notifiee.newStatusMessage("The average number of bins is " + avgNumBins)

    // TODO: This should be automatically determined from InfoGainStatistics bin count type.
    val sizeOfLong = 8 // Currently we're using Long to count the number of elements.
    val sizeOfDouble = 8

    // TODO: This is arbitrary. Should do better. Currently, keeping the maximum bytes in transit to be around 256MB.
    // TODO: In practice, the bytes in transit should be much smaller than this thanks to compression.
    val maxBytesInTransit = 256.0 * 1024.0 * 1024.0

    // TODO: Can we use better heuristics to determine these numbers ?
    val nodesPerIter = numNodesPerIteration match {
      case x if x == -1 =>
        val avgNumBytesPerNode = if (treeType == TreeType.Classification_InfoGain) {
          avgNumBins * numRandomFeaturesPerNode.toDouble * (maxLabelValue + 1.0) * sizeOfLong.toDouble
        } else {
          avgNumBins * numRandomFeaturesPerNode.toDouble * 3.0 * sizeOfDouble.toDouble
        }

        math.ceil(maxBytesInTransit / avgNumBytesPerNode / 2.0).toInt

      case _ => numNodesPerIteration
    }

    // TODO: Can we come up with better heuristics and defaults?
    // TODO: By default, we train locally once the number of training samples is less than or equal to 60000.
    val subTreeMaxSamples = localTrainThreshold match {
      case x if x == -1 => 60000
      case _ => localTrainThreshold
    }

    val subTreeTrainers = numSubTreesPerIteration match {
      case x if x == -1 => 400 // Could be even larger as long as we have enough memory in the driver.
      case _ => numSubTreesPerIteration
    }

    val numClasses = if (treeType == TreeType.Classification_InfoGain) {
      Some(maxLabelValue.toInt + 1)
    } else {
      None
    }

    val forest = SequoiaForestTrainer.train(
      discretizedBaggedInput,
      featureBins,
      SequoiaForestOptions(
        numTrees = numTrees,
        treeType = treeType,
        mtry = numRandomFeaturesPerNode,
        minSplitSize = minSplitSizeActual,
        maxDepth = maxDepth,
        numNodesPerIteration = nodesPerIter,
        localTrainThreshold = subTreeMaxSamples,
        numSubTreesPerIteration = subTreeTrainers,
        storeModelInMemory = storeModelInMemory,
        outputStorage = outputStorage,
        numClasses = numClasses
      ),
      notifiee,
      validationData)

    if (storeModelInMemory) {
      Some(forest)
    } else {
      None
    }
  }

  /**
   * Train a forest either locally or using RDD.
   * @param input Training data - could be RDD or local and could use unsigned Byte or Short features.
   * @param featureBins Bin definitions for each feature.
   * @param options Training options.
   * @param notifiee Notifiee object. The progress messages of training would be sent to this object.
   * @param validationData Optional original validation data (untransformed with Double feature values) to use. Using this would mean models would be stored in memory (not recommended for large models).
   * @return Trained Sequoia Forest object.
   */
  private[spark_ml] def train(
    input: DiscretizedData,
    featureBins: Array[Bins],
    options: SequoiaForestOptions,
    notifiee: ProgressNotifiee,
    validationData: Option[Array[(Double, Array[Double])]]): SequoiaForest = {

    // Common structures we need for all types of inputs.
    val randGen = new Random()
    val numFeatures = featureBins.length
    val numBinsPerFeature = Array.fill[Int](numFeatures)(0)
    val numTrees = options.numTrees
    val trees = new Array[SequoiaTree](numTrees)
    val numNodesPerTree = Array.fill[Int](numTrees)(0) // In case we don't store the model in memory, we still want the number of nodes.
    val treeSeeds = new Array[Int](numTrees)
    val scheduledRowFilters = ScheduledRowFilters(numTrees)
    val nextNodeIdsPerTree = Array.fill[Int](numTrees)(2) // Node IDs to assign after the root. Root node Ids will all be 1's.
    val nodeDepths = Array.fill[mutable.Map[Int, Int]](numTrees)(mutable.Map[Int, Int]()) // Keeps track of currently being-trained nodes' depths.
    val varImportance = VarImportance(numFeatures)
    val forest = SequoiaForest(trees, options.treeType, varImportance)

    notifiee.newStatusMessage("Training Sequoia Forest with the following options:")
    notifiee.newStatusMessage("Tree Type : " + options.treeType.toString)
    notifiee.newStatusMessage("Number of Trees : " + numTrees)
    notifiee.newStatusMessage("Number of Features : " + numFeatures)
    notifiee.newStatusMessage("Number of Random Features Per Node : " + options.mtry)
    notifiee.newStatusMessage("Minimum Node Size for Split : " + options.minSplitSize)
    notifiee.newStatusMessage("Maximum Depth for Trees : " + options.maxDepth)
    notifiee.newStatusMessage("Number of Node Splits per Iteration : " + options.numNodesPerIteration)
    notifiee.newStatusMessage("Sub Tree Threshold : " + options.localTrainThreshold)
    notifiee.newStatusMessage("Number of Sub Trees per Iteration : " + options.numSubTreesPerIteration)
    notifiee.newStatusMessage("Output Location : " + options.outputStorage.getLocation)
    if (options.numClasses != None) notifiee.newStatusMessage("Number of Target Classes : " + options.numClasses.get)
    if (options.storeModelInMemory) notifiee.newStatusMessage("Storing the trained models in memory.") else notifiee.newStatusMessage("Trained models will not be stored in memory.")

    var featId = 0
    while (featId < numFeatures) {
      numBinsPerFeature(featId) = featureBins(featId).getCardinality
      featId += 1
    }

    options.outputStorage.initialize(numTrees, options.treeType)

    // First, we need to queue the requests for the root node split.
    var treeId = 0
    while (treeId < numTrees) {
      trees(treeId) = SequoiaTree(treeId)
      treeSeeds(treeId) = randGen.nextInt()
      scheduledRowFilters.addRowFilter(treeId, new RootGetter())
      nodeDepths(treeId).put(1, 1) // The root node always has the depth 1.
      treeId += 1
    }

    // Initialize the node Ids attached to each training row.
    input.initializeRowNodeIds(numTrees)

    // Iterate and train.
    var iter = 0
    while (scheduledRowFilters.size > 0) { // We'll repeat until we run out of nodes to split.
      // Take the node splits from the queue and schedule them to be processed during this iteration.
      val rowFilterLookup = scheduledRowFilters.popRowFilters(options.numNodesPerIteration)

      notifiee.newStatusMessage("Aggregating statistics. The number of row filters is " + rowFilterLookup.nodeSplitCount)

      // Aggregate statistics for scheduled nodes.
      val aggregationT0 = System.currentTimeMillis()
      val aggregatedStats: AggregatedStatistics = input.applyRowFiltersAndAggregateStatistics(rowFilterLookup, treeSeeds, numBinsPerFeature, options)
      val aggregationDuration = System.currentTimeMillis() - aggregationT0

      notifiee.newStatusMessage("Finished statistics aggregation. Time it took (seconds) : " + aggregationDuration.toDouble / 1000.0)

      // Set up row filters to used to find rows that can be shuffled to perform local sub-tree training.
      // Only to be used for RDD data sources.
      val subTreeTrainingRowFilters = if (input.isLocal) null else ScheduledRowFilters(numTrees)

      // Train and find node splits based on the aggregated statistics.
      val trainedNodes = aggregatedStats.computeNodePredictionsAndSplits(
        featureBins,
        nextNodeIdsPerTree,
        nodeDepths,
        options)

      // Create tree nodes and process splits if they exist.
      while (trainedNodes.hasNext) {
        val trainedNodeInfo = trainedNodes.next()
        val treeNode = processTrainedNode(
          trainedNodeInfo,
          scheduledRowFilters,
          featureBins,
          subTreeTrainingRowFilters,
          options.localTrainThreshold)

        numNodesPerTree(trainedNodeInfo.treeId) += 1

        // We store the node in the matching tree if we are storing the models in memory.
        if (options.storeModelInMemory) {
          forest.trees(trainedNodeInfo.treeId).addNode(treeNode)
        }

        // If the output streams object exists, we also write the node to the matching stream.
        options.outputStorage.writeNode(trainedNodeInfo.treeId, trainedNodeInfo.depth, treeNode)

        // Add variable importance from split nodes.
        if (treeNode.splitImpurity != None) forest.varImportance.addVarImportance(treeNode)
      }

      // Now, let's see if there's any sub-tree training to do.
      while (subTreeTrainingRowFilters != null && subTreeTrainingRowFilters.size > 0) {
        val subTreeLookup = subTreeTrainingRowFilters.popRowFiltersForSubTreeTraining(options.numSubTreesPerIteration)

        notifiee.newStatusMessage("Training " + subTreeLookup.subTreeCount + " sub trees.")

        val subTreeT0 = System.currentTimeMillis()
        val subTrees = input.trainSubTreesLocally(subTreeLookup, featureBins, nodeDepths, options)
        val subTreeDuration = System.currentTimeMillis() - subTreeT0

        notifiee.newStatusMessage("Finished training sub trees. Time it took (seconds) : " + subTreeDuration.toDouble / 1000.0)

        while (subTrees.hasNext) {
          val (parentTreeId, subTree) = subTrees.next()

          numNodesPerTree(parentTreeId) += subTree.getNodeCount

          if (options.storeModelInMemory) {
            forest.trees(parentTreeId).addSubTree(subTree)
          }

          val subTreeDepth = nodeDepths(parentTreeId)(subTree.treeId)
          nodeDepths(parentTreeId).remove(subTree.treeId)
          options.outputStorage.writeSubTree(parentTreeId, subTreeDepth, subTree)

          // Add variable importance from each sub tree split node.
          subTree.nodes.values.foreach(node => if (node.splitImpurity != None) forest.varImportance.addVarImportance(node))
        }
      }

      iter += 1

      // Show the number of nodes we have trained so far.
      var emptyTreesExist = false
      treeId = 0
      while (treeId < numTrees) {
        val nodeCount = numNodesPerTree(treeId)
        if (nodeCount == 0) emptyTreesExist = true
        notifiee.newStatusMessage("Tree " + treeId + " node count : " + nodeCount)
        treeId += 1
      }

      // Measure accuracy against validation data.
      if (options.storeModelInMemory && !emptyTreesExist && validationData != None) {
        notifiee.newStatusMessage("Validation after " + iter + " iterations.")
        validate(forest, validationData.get, notifiee)
      }

      // Show memory usages.
      notifiee.newStatusMessage("Maximum memory : " + (Runtime.getRuntime.maxMemory().toDouble / 1024.0 / 1024.0) + " MB")
      notifiee.newStatusMessage("Used memory : " + ((Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()).toDouble / 1024.0 / 1024.0) + " MB")
      notifiee.newStatusMessage("Free memory : " + (Runtime.getRuntime.freeMemory().toDouble / 1024.0 / 1024.0) + " MB")
    }

    // Do final validation if necessary.
    if (options.storeModelInMemory && validationData != None) {
      notifiee.newStatusMessage("Validation after full training.")
      validate(forest, validationData.get, notifiee)
    }

    // Close the output streams if they exist.
    options.outputStorage.writeVarImportance(forest.varImportance)
    options.outputStorage.close()
    forest
  }

  /**
   * Process trained node information and convert it into an actual tree node.
   * Additionally, add future node split and sub-tree training tasks into matching queues.
   * @param trainedNode Information about the trained node.
   * @param scheduledRowFilters Queue of scheduled row filters to add child nodes to (for future statistics collections).
   * @param featureBins Bin descriptions for features.
   * @param subTreeTrainingRowFilters Queue of sub-tree training for RDD sources.
   * @param subTreeTrainThreshold Threshold to initiate sub-tree training for RDD sources.
   * @return A tree node
   */
  private def processTrainedNode(
    trainedNode: TrainedNodeInfo,
    scheduledRowFilters: ScheduledRowFilters,
    featureBins: Array[Bins],
    subTreeTrainingRowFilters: ScheduledRowFilters,
    subTreeTrainThreshold: Int): SequoiaNode = {
    val treeId = trainedNode.treeId
    val nodeId = trainedNode.nodeId
    val prediction = trainedNode.prediction
    val weight = trainedNode.weight
    val impurity = trainedNode.impurity

    if (trainedNode.nodeSplit != None) {
      val splitImpurity = trainedNode.splitImpurity.get
      val nodeSplit = trainedNode.nodeSplit.get
      val childNodeIds = nodeSplit.getOrderedChildNodeIds
      var numUnderweightNodes = 0
      if (subTreeTrainingRowFilters != null) {
        // If the input is RDD, let's find out if there's a reason to perform local sub-tree training.
        var childNodeIdx = 0
        while (childNodeIdx < childNodeIds.length) {
          val childNodeId = childNodeIds(childNodeIdx)
          val childNodeWeight = nodeSplit.getChildNodeWeight(childNodeId)
          if (childNodeWeight <= subTreeTrainThreshold.toDouble) {
            numUnderweightNodes += 1
            nodeSplit.setSubTreeHash(childNodeId, 0) // This makes sure that this child will be trained as a local sub-tree.
          }

          childNodeIdx += 1
        }

        if (numUnderweightNodes > 0) {
          subTreeTrainingRowFilters.addRowFilter(treeId, nodeSplit)
        }
      }

      // If the number of child nodes to use in sub-tree training is not equal to the number of all children,
      // we should also add the node split to the standard row filter queue.
      if (numUnderweightNodes < childNodeIds.length) {
        scheduledRowFilters.addRowFilter(treeId, nodeSplit)
      }

      // Create an inner node.
      val split: NodeSplit = trainedNode.nodeSplit.get match {
        case nodeSplitOnBinId: NumericSplitOnBinId =>
          val nodeSplitOnBinId = trainedNode.nodeSplit.get.asInstanceOf[NumericSplitOnBinId]
          val featureId = nodeSplitOnBinId.featureId
          val splitValue = featureBins(featureId).asInstanceOf[NumericBins].bins(nodeSplitOnBinId.splitBinId).lower
          NumericSplit(
            featureId,
            splitValue,
            nodeSplitOnBinId.leftId,
            nodeSplitOnBinId.rightId)

        case nodeSplitOnBinId: CategoricalSplitOnBinId =>
          val nodeSplitOnBinId = trainedNode.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId]
          val featureId = nodeSplitOnBinId.featureId
          CategoricalSplit(
            featureId,
            nodeSplitOnBinId.binIdToNodeIdMap)
      }

      SequoiaNode(nodeId, prediction, impurity, weight, Some(splitImpurity), Some(split))
    } else {
      // Create a leaf node.
      SequoiaNode(nodeId, prediction, impurity, weight, None, None)
    }
  }

  /**
   * Validate the model against the given in memory data.
   * @param forest The forest model we want to validate.
   * @param data The in memory validation data.
   * @param notifiee The notifiee object.
   * @return Performance measure (e.g. accuracy or MSE).
   */
  def validate(
    forest: SequoiaForest,
    data: Array[(Double, Array[Double])],
    notifiee: ProgressNotifiee): Double = {
    if (forest.treeType == TreeType.Classification_InfoGain) {
      var numCorrect = 0
      var numRows = 0
      while (numRows < data.length) {
        val row = data(numRows)
        val prediction = forest.predict(row._2)
        if (prediction == row._1) numCorrect += 1
        numRows += 1
      }

      val accuracy = numCorrect.toDouble / numRows.toDouble
      notifiee.newStatusMessage("Num Correct : " + numCorrect)
      notifiee.newStatusMessage("Num Rows : " + numRows)
      notifiee.newStatusMessage("Accuracy : " + numCorrect.toDouble / numRows.toDouble)
      accuracy
    } else {
      var squaredErrorSum = 0.0
      var numRows = 0
      while (numRows < data.length) {
        val row = data(numRows)
        val error = row._1 - forest.predict(row._2)
        squaredErrorSum += error * error
        numRows += 1
      }

      val mse = squaredErrorSum / numRows.toDouble
      notifiee.newStatusMessage("Num Rows : " + numRows)
      notifiee.newStatusMessage("Mean Squared Error : " + mse)
      mse
    }
  }
}

/**
 * Options for the forest.
 * @param numTrees Number of trees in the forest.
 * @param treeType Whether it's classification or regression.
 * @param mtry Number of random features per node.
 * @param minSplitSize Minimum node size eligible for splitting (limits tree growth).
 * @param numNodesPerIteration Number of distributed node splits to perform per RDD iteration.
 * @param localTrainThreshold Number of samples to see at a node before training the rest locally.
 * @param numSubTreesPerIteration Number of sub trees to train per iteration.
 * @param storeModelInMemory Whether to store the trained model in memory.
 * @param outputStorage Where to store the output model.
 * @param numClasses Number of classes, if this is a classification model (required for classification).
 */
case class SequoiaForestOptions(
  numTrees: Int,
  treeType: TreeType.TreeType,
  mtry: Int,
  minSplitSize: Long,
  maxDepth: Int,
  numNodesPerIteration: Int,
  localTrainThreshold: Int,
  numSubTreesPerIteration: Int,
  storeModelInMemory: Boolean,
  outputStorage: ForestStorage,
  numClasses: Option[Int]) // Only for classification.

/**
 * This is used to filter training rows to matching Node Ids.
 * In each iteration of distributed node splits, a certain number of row filters are used to filter rows to matching Nodes,
 * and then they are aggregated to matching nodes' statistics.
 * @param numTrees Number of trees in the forest.
 */
case class ScheduledRowFilters(numTrees: Int) {
  var totalRowFilterCount = 0
  val rowFiltersPerTree = Array.fill[mutable.Queue[NodeSplitOnBinId]](numTrees)(mutable.Queue[NodeSplitOnBinId]())

  /**
   * It's expected that the row filters that are added should have both parent and child node Ids
   * in increasing orders. This happens naturally when we do breadth-first training.
   * @param treeId Id of the tree that we are adding the filter for.
   * @param filter The filter (NodeSplit).
   */
  def addRowFilter(treeId: Int, filter: NodeSplitOnBinId): Unit = {
    rowFiltersPerTree(treeId).enqueue(filter)
    totalRowFilterCount += 1
  }

  /**
   * The total number of row filters in this object.
   * @return The total number of row filters in this object.
   */
  def size: Int = totalRowFilterCount

  /**
   * Pop row filters in the queues, and create a fast lookup object to be used by the trainer.
   * @param maxRowFilters The maximum number of row filters we want to pop.
   * @return A look up object for popped row filters.
   */
  def popRowFilters(maxRowFilters: Int): ScheduledNodeSplitLookup = {
    val r = ScheduledNodeSplitLookup.createLookupForNodeSplits(rowFiltersPerTree, maxRowFilters)
    totalRowFilterCount -= r.nodeSplitCount
    r
  }

  /**
   * Pop row filters in the queues, and create a fast lookup object to be used by the trainer to select samples for local sub-tree training.
   * @param maxSubTrees The maximum number of sub-tree training to be done with this.
   * @return A look up object for popped row filters.
   */
  def popRowFiltersForSubTreeTraining(maxSubTrees: Int): ScheduledNodeSplitLookup = {
    val r = ScheduledNodeSplitLookup.createLookupForSubTreeTraining(rowFiltersPerTree, maxSubTrees)
    totalRowFilterCount -= r.nodeSplitCount
    r
  }
}

/**
 * Simple console notifiee.
 * Prints messages to the stdout.
 */
class ConsoleNotifiee extends ProgressNotifiee {
  def newStatusMessage(status: String): Unit = {
    println("[Status] [" + Calendar.getInstance().getTime.toString + "] " + status)
  }

  def newErrorMessage(error: String): Unit = {
    println("[Error] [" + Calendar.getInstance().getTime.toString + "] " + error)
  }
}

/**
 * An object to funnel progress reports to.
 */
trait ProgressNotifiee extends Serializable {
  def newStatusMessage(status: String): Unit
  def newErrorMessage(error: String): Unit
}

/**
 * Different discretization types to use.
 */
object DiscretizationType extends Enumeration {
  type DiscretizationType = Value
  val EqualWidth = Value(0)
  val EqualFrequency = Value(1)
}
