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

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import spark_ml.discretization.{ CategoricalBins, Bins, Discretizer }
import spire.implicits._

import spark_ml.util.Sorting._
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path

/**
 * Various data types for Sequoia Forest - can be RDD or local array. The features must be discretized into unsigned Byte or Short.
 * For all data types, each row will be a tuple of numbers.
 * The first element (Double) of a row should be the label (for categorical labels, it should be like 0.0, 1.0, 2.0...
 * The second element should be an array of feature Bin IDs (should have gone through conversion already into unsigned Byte or Short).
 * The last element is the sample count for each tree after bagging (bagging should be run before calling train). It's using Byte and capped at 127.
 * There will also be an appended array of integers that represent node ID that the row belongs to for a particular tree.
 */
trait DiscretizedData {
  /**
   * Whether this data source is a local array or not.
   * @return true if it's a local array, false if it's RDD.
   */
  def isLocal: Boolean = true

  /**
   * In case of RDD data source, we might want to get SparckContext of the RDD.
   * @return SparkContext of the RDD if it's available.
   */
  def getSparkContext: SparkContext = null

  def setCheckpointDir(dir: String): Unit = {}
  def setCheckpointInterval(interval: Int): Unit = {}

  /**
   * Initialize node IDs of rows to zeroes.
   * @param numTrees Number of trees that we have.
   */
  def initializeRowNodeIds(numTrees: Int): Unit

  /**
   * Train sub trees locally by shuffling data.
   * Only applies to RDD sources.
   * @param subTreeLookup A look up table to find rows that will be used for sub-tree training.
   * @param featureBins Feature bin definitions.
   * @param nodeDepths Currently being-trained nodes' depths.
   * @param options Forest training options.
   * @param treeSeeds To generate a seed for the sub-tree (useful for testing).
   * @return An iterator of trained trees, the first element is the parent tree ID. Sub-trees will have IDs that match the child ID of the parent tree.
   */
  def trainSubTreesLocally(
    subTreeLookup: ScheduledNodeSplitLookup,
    featureBins: Array[Bins],
    nodeDepths: Array[mutable.Map[Int, Int]],
    options: SequoiaForestOptions,
    treeSeeds: Array[Int]): Iterator[(Int, SequoiaTree)]

  /**
   * Apply row filters on rows, and collect/aggregate bin statistics on matching rows and nodes.
   * This will also update the node ID tags on individual rows.
   * @param rowFilterLookup A fast lookup for matching row filters for particular nodes.
   * @param treeSeeds Random seeds to use per tree. This is used in selecting a random number of features per node.
   * @param numBinsPerFeature Number of bins per feature. This is used to initialize the statistics object.
   * @param options The options to be used in Sequoia Forest.
   * @return Aggregated statistics for tree/node/feature/bin combinations.
   */
  def applyRowFiltersAndAggregateStatistics(
    rowFilterLookup: ScheduledNodeSplitLookup,
    treeSeeds: Array[Int],
    numBinsPerFeature: Array[Int],
    options: SequoiaForestOptions): AggregatedStatistics

  /**
   * Similar to applyRowFiltersAndAggregateStatistics.
   * In addition to performing distributed statistics aggregations, this will also perform
   * distributed node splits (node splits for different trees will be done by different machines),
   * making it more efficient.
   * @param rowFilterLookup A fast lookup for matching row filters for particular nodes.
   * @param treeSeeds Random seeds to use per tree. This is used in selecting a random number of features per node.
   * @param numBinsPerFeature Number of bins per feature. This is used to initialize the statistics object.
   * @param options The options to be used in Sequoia Forest.
   * @param featureBins Feature bin descriptions.
   * @param nextNodeIdsPerTree Next node Ids to assign per tree.
   * @param nodeDepths Depths of nodes that are currently being trained.
   * @return An iterator of trained node info objects. The trainer will use this to build trees.
   */
  def performDistributedNodeSplits(
    rowFilterLookup: ScheduledNodeSplitLookup,
    treeSeeds: Array[Int],
    numBinsPerFeature: Array[Int],
    options: SequoiaForestOptions,
    featureBins: Array[Bins],
    nextNodeIdsPerTree: Array[Int],
    nodeDepths: Array[mutable.Map[Int, Int]]): Iterator[TrainedNodeInfo]
}

/**
 * We can have either unsigned Byte or Short features. We want to have consistent ways of handling these types.
 * @tparam T In our cases, either Byte or Short.
 */
trait FeatureHandler[@specialized(Byte, Short) T] extends Serializable {
  /**
   * Clone myself.
   * @return A clone
   */
  def cloneMyself: FeatureHandler[T]

  /**
   * Convert the feature value into an integer value.
   * @param value The value that we want to convert to integer.
   * @return The feature value converted to integer.
   */
  def convertToInt(value: T): Int

  /**
   * Aggregate the values in the row to the given statistics object.
   * @param stats The statistics object that we want to aggregate the sample row into.
   * @param treeId The ID of the tree that we want to aggregate for.
   * @param nodeId The ID of the node that we want to aggregate for.
   * @param row The row whose label/feature values we want to add into the aggregated statistics.
   */
  def addRowToStats(stats: AggregatedStatistics, treeId: Int, nodeId: Int, row: (Double, Array[T], Array[Byte])): Unit

  def addRowToStats(stats: AggregatedStatistics2, treeId: Int, nodeId: Int, row: (Double, Array[T], Array[Byte])): Unit

  /**
   * Use this to create a local discretized data object from a generic typed object.
   * @param array Array of data.
   * @return The data wrapped under the DiscretizedData interface.
   */
  def createDiscretizedDataLocal(array: Array[((Double, Array[T], Array[Byte]), Array[Int])]): DiscretizedData
}

/**
 * Feature handler for unsigned Byte features.
 */
class UnsignedByteFeatureHandler extends FeatureHandler[Byte] {
  /**
   * Clone myself.
   * @return A clone
   */
  override def cloneMyself: FeatureHandler[Byte] = new UnsignedByteFeatureHandler

  /**
   * Convert the feature value into an integer value.
   * @param value The value that we want to convert to integer.
   * @return The feature value converted to integer.
   */
  def convertToInt(value: Byte): Int = {
    Discretizer.readUnsignedByte(value)
  }

  /**
   * Aggregate the values in the row to the given statistics object.
   * @param stats The statistics object that we want to aggregate the sample row into.
   * @param treeId The ID of the tree that we want to aggregate for.
   * @param nodeId The ID of the node that we want to aggregate for.
   * @param row The row whose label/feature values we want to add into the aggregated statistics.
   */
  def addRowToStats(stats: AggregatedStatistics, treeId: Int, nodeId: Int, row: (Double, Array[Byte], Array[Byte])): Unit = {
    stats.addUnsignedByteSample(treeId, nodeId, row)
  }

  def addRowToStats(stats: AggregatedStatistics2, treeId: Int, nodeId: Int, row: (Double, Array[Byte], Array[Byte])): Unit = {
    stats.addUnsignedByteSample(treeId, nodeId, row)
  }

  /**
   * Use this to create a local discretized data object from a generic typed object.
   * @param array Array of data.
   * @return The data wrapped under the DiscretizedData interface.
   */
  def createDiscretizedDataLocal(array: Array[((Double, Array[Byte], Array[Byte]), Array[Int])]): DiscretizedData = {
    UnsignedByteLocal(array)
  }
}

/**
 * Feature handler for unsigned Short features.
 */
class UnsignedShortFeatureHandler extends FeatureHandler[Short] {
  /**
   * Clone myself.
   * @return A clone
   */
  override def cloneMyself: FeatureHandler[Short] = new UnsignedShortFeatureHandler

  /**
   * Convert the feature value into an integer value.
   * @param value The value that we want to convert to integer.
   * @return The feature value converted to integer.
   */
  def convertToInt(value: Short): Int = {
    Discretizer.readUnsignedShort(value)
  }

  /**
   * Aggregate the values in the row to the given statistics object.
   * @param stats The statistics object that we want to aggregate the sample row into.
   * @param treeId The ID of the tree that we want to aggregate for.
   * @param nodeId The ID of the node that we want to aggregate for.
   * @param row The row whose label/feature values we want to add into the aggregated statistics.
   */
  def addRowToStats(stats: AggregatedStatistics, treeId: Int, nodeId: Int, row: (Double, Array[Short], Array[Byte])): Unit = {
    stats.addUnsignedShortSample(treeId, nodeId, row)
  }

  def addRowToStats(stats: AggregatedStatistics2, treeId: Int, nodeId: Int, row: (Double, Array[Short], Array[Byte])): Unit = {
    stats.addUnsignedShortSample(treeId, nodeId, row)
  }

  /**
   * Use this to create a local discretized data object from a generic typed object.
   * @param array Array of data.
   * @return The data wrapped under the DiscretizedData interface.
   */
  def createDiscretizedDataLocal(array: Array[((Double, Array[Short], Array[Byte]), Array[Int])]): DiscretizedData = {
    UnsignedShortLocal(array)
  }
}

/**
 * A parent type for RDD data sources.
 */
class DiscretizedDataRDD[@specialized(Byte, Short) T](data: RDD[(Double, Array[T], Array[Byte])])(featureHandler: FeatureHandler[T]) extends DiscretizedData {
  var nodeIdRDD: RDD[Array[Int]] = null
  var checkpointRootDir: String = null
  var checkpointInterval: Int = 10

  var prevNodeIdRDD: RDD[Array[Int]] = null

  var nodeIdRDDUpdateCount: Int = 0
  val checkpointQueue: mutable.Queue[RDD[Array[Int]]] = mutable.Queue[RDD[Array[Int]]]() // To keep track of last checkpointed RDDs.

  override def isLocal: Boolean = false

  /**
   * @return SparkContext of the RDD.
   */
  override def getSparkContext: SparkContext = data.sparkContext

  override def setCheckpointDir(dir: String): Unit = {
    checkpointRootDir = dir
    data.sparkContext.setCheckpointDir(checkpointRootDir)
  }

  override def setCheckpointInterval(interval: Int): Unit = {
    checkpointInterval = interval
  }

  /**
   * Initialize the node IDs for training rows.
   * @param numTrees Number of trees that we have.
   */
  override def initializeRowNodeIds(numTrees: Int): Unit = {
    nodeIdRDD = data.map(row => Array.fill[Int](numTrees)(0))
  }

  /**
   * Train sub trees locally by shuffling data.
   * Only applies to RDD sources.
   * @param subTreeLookup A look up table to find rows that will be used for sub-tree training.
   * @param featureBins Feature bin definitions.
   * @param nodeDepths Currently being-trained nodes' depths.
   * @param options Forest training options.
   * @param treeSeeds To generate a seed for the sub-tree (useful for testing).
   * @return An iterator of trained trees, the first element is the parent tree ID. Sub-trees will have IDs that match the child ID of the parent tree.
   */
  override def trainSubTreesLocally(
    subTreeLookup: ScheduledNodeSplitLookup,
    featureBins: Array[Bins],
    nodeDepths: Array[mutable.Map[Int, Int]],
    options: SequoiaForestOptions,
    treeSeeds: Array[Int]): Iterator[(Int, SequoiaTree)] = {
    val featureHandlerLocal = featureHandler.cloneMyself // To avoid serializing the entire DiscretizedData object.
    val numTrees = options.numTrees
    val treeType = options.treeType
    val mtry = options.mtry
    val minSplitSize = options.minSplitSize
    val depthLimit = options.maxDepth match {
      case x if x == -1 => Int.MaxValue
      case x => x
    }

    val imputationType = options.imputationType

    val numNodesPerIteration = options.numNodesPerIteration
    val numClasses = options.numClasses
    val numSubTreesToTrain = subTreeLookup.subTreeCount
    val shuffledRows = data.zip(nodeIdRDD).flatMap(row => {
      val output = new mutable.ArrayBuffer[(Int, Int, Int, ((Double, Array[T], Array[Byte]), Array[Int]))]()
      var treeId = 0
      while (treeId < numTrees) {
        val curNodeId = row._2(treeId)
        val rowCntByte = row._1._3(treeId)
        val rowCnt = rowCntByte.toInt
        if (rowCnt > 0 && curNodeId >= 0) {
          val nodeSplit = subTreeLookup.getNodeSplit(treeId, curNodeId)
          if (nodeSplit != null) {
            val childNodeId = nodeSplit.selectChildNode(featureHandlerLocal.convertToInt(row._1._2(nodeSplit.featureId)))
            val subTreeHash = nodeSplit.getSubTreeHash(childNodeId)
            if (subTreeHash >= 0) {
              val label = row._1._1
              val features = row._1._2

              // For shuffled data, we only need one tree appendage.
              output += ((subTreeHash, treeId, childNodeId, ((label, features, Array[Byte](rowCntByte)), Array[Int](0))))
            }
          }
        }

        treeId += 1
      }

      output.toIterator
    }).groupBy((row: (Int, Int, Int, ((Double, Array[T], Array[Byte]), Array[Int]))) => row._1, numSubTreesToTrain)

    val trainedSubTrees = shuffledRows.map(rowSet => {
      val rows = rowSet._2
      var parentTreeId = 0
      var nodeId = 0
      val arrayData = rows.map(row => {
        parentTreeId = row._2
        nodeId = row._3
        row._4
      }).toArray

      // Find the depth limit for the sub tree.
      val subTreeMaxDepth = depthLimit - nodeDepths(parentTreeId)(nodeId) + 1

      // Train locally.
      val forest = SequoiaForestTrainer.train(
        featureHandlerLocal.createDiscretizedDataLocal(arrayData),
        featureBins,
        SequoiaForestOptions(
          numTrees = 1,
          treeType = treeType,
          mtry = mtry,
          minSplitSize = minSplitSize,
          maxDepth = subTreeMaxDepth,
          numNodesPerIteration = numNodesPerIteration,
          localTrainThreshold = 0,
          numSubTreesPerIteration = 0,
          storeModelInMemory = true,
          outputStorage = new NullSinkForestStorage,
          numClasses = numClasses,
          imputationType = imputationType,
          distributedNodeSplits = false),
        new ConsoleNotifiee,
        None,
        useLogLossForValidation = false,
        randGen = new scala.util.Random(treeSeeds(parentTreeId) + nodeId))

      val tree = forest.trees(0)
      tree.treeId = nodeId

      (parentTreeId, tree)
    }).collect()

    // Update the Node IDs for rows to reflect local sub-tree training.
    // TODO: Figure out whether we can optimize this further.
    updateNodeIdRDD(subTreeLookup, numTrees, markSubTreesOnly = true)

    trainedSubTrees.toIterator
  }

  /**
   * Apply row filters on rows, and collect/aggregate bin statistics on matching rows and nodes.
   * This will also update the node ID tags on individual rows.
   * @param rowFilterLookup A fast lookup for matching row filters for particular nodes.
   * @param treeSeeds Random seeds to use per tree. This is used in selecting a random number of features per node.
   * @param numBinsPerFeature Number of bins per feature. This is used to initialize the statistics object.
   * @param options The options to be used in Sequoia Forest.
   * @return Aggregated statistics for tree/node/feature/bin combinations.
   */
  override def applyRowFiltersAndAggregateStatistics(
    rowFilterLookup: ScheduledNodeSplitLookup,
    treeSeeds: Array[Int],
    numBinsPerFeature: Array[Int],
    options: SequoiaForestOptions): AggregatedStatistics = {
    // TODO: This seems stupid.
    val featureHandlerLocal = featureHandler.cloneMyself // To avoid serializing the entire DiscretizedData object.

    // First aggregate bin statistics across all the partitions.
    val numTrees = options.numTrees
    val treeType = options.treeType
    val mtry = options.mtry
    val numClasses = options.numClasses
    val aggregatedArray = data.zip(nodeIdRDD).mapPartitions(rows => {
      val partitionStats = if (treeType == TreeType.Classification_InfoGain) {
        new InfoGainStatistics(rowFilterLookup, numBinsPerFeature, treeSeeds, mtry, numClasses.get)
      } else {
        new VarianceStatistics(rowFilterLookup, numBinsPerFeature, treeSeeds, mtry)
      }

      while (rows.hasNext) {
        val row = rows.next()
        cfor(0)(_ < numTrees, _ + 1)(
          treeId => {
            val curNodeId = row._2(treeId)
            val rowCnt = row._1._3(treeId).toInt
            if (rowCnt > 0 && curNodeId >= 0) { // It can be -1 if the sample was used in a node that was trained locally as a sub-tree.
              val nodeSplit = rowFilterLookup.getNodeSplit(treeId, curNodeId)
              if (nodeSplit != null) {
                val childNodeId = nodeSplit.selectChildNode(featureHandlerLocal.convertToInt(row._1._2(nodeSplit.featureId)))
                featureHandlerLocal.addRowToStats(partitionStats, treeId, childNodeId, row._1)
              }
            }
          }
        )
      }

      Array(partitionStats.binStatsArray).toIterator
    }).reduce((a, b) => a.mergeInPlace(b))

    // Update the nodeIdRDDs to reflect new Node IDs.
    updateNodeIdRDD(rowFilterLookup, numTrees)

    val totalStats = if (treeType == TreeType.Classification_InfoGain) {
      new InfoGainStatistics(rowFilterLookup, numBinsPerFeature, treeSeeds, mtry, numClasses.get)
    } else {
      new VarianceStatistics(rowFilterLookup, numBinsPerFeature, treeSeeds, mtry)
    }

    totalStats.binStatsArray = aggregatedArray

    // Return the aggregated statistics.
    totalStats
  }

  /**
   * Similar to applyRowFiltersAndAggregateStatistics.
   * In addition to performing distributed statistics aggregations, this will also perform
   * distributed node splits (node splits for different trees will be done by different machines),
   * making it more efficient.
   * @param rowFilterLookup A fast lookup for matching row filters for particular nodes.
   * @param treeSeeds Random seeds to use per tree. This is used in selecting a random number of features per node.
   * @param numBinsPerFeature Number of bins per feature. This is used to initialize the statistics object.
   * @param options The options to be used in Sequoia Forest.
   * @param featureBins Feature bin descriptions.
   * @param nextNodeIdsPerTree Next node Ids to assign per tree.
   * @param nodeDepths Depths of nodes that are currently being trained.
   * @return An iterator of trained node info objects. The trainer will use this to build trees.
   */
  override def performDistributedNodeSplits(
    rowFilterLookup: ScheduledNodeSplitLookup,
    treeSeeds: Array[Int],
    numBinsPerFeature: Array[Int],
    options: SequoiaForestOptions,
    featureBins: Array[Bins],
    nextNodeIdsPerTree: Array[Int],
    nodeDepths: Array[mutable.Map[Int, Int]]): Iterator[TrainedNodeInfo] = {
    // TODO: This seems stupid.
    val featureHandlerLocal = featureHandler.cloneMyself // To avoid serializing the entire DiscretizedData object.

    // First aggregate bin statistics across all the partitions.
    val numTrees = options.numTrees
    val treeType = options.treeType
    val mtry = options.mtry
    val numClasses = options.numClasses
    val nodeSplitCount = rowFilterLookup.nodeSplitCount

    val numFeatures = featureBins.length
    val categoricalFeatureFlags = Array.fill[Boolean](numFeatures)(false)
    val featureMissingValueBinIds = Array.fill[Int](numFeatures)(-1)
    val imputationType = options.imputationType
    val minSplitSize = options.minSplitSize
    val depthLimit = options.maxDepth match {
      case x if x == -1 => Int.MaxValue
      case x => x
    }

    cfor(0)(_ < numFeatures, _ + 1)(
      featId => {
        categoricalFeatureFlags(featId) = featureBins(featId).isInstanceOf[CategoricalBins]
        featureMissingValueBinIds(featId) = featureBins(featId).getMissingValueBinIdx
      }
    )

    val trainedNodes = data.zip(nodeIdRDD).mapPartitions(rows => {
      val partitionStats = if (treeType == TreeType.Classification_InfoGain) {
        new InfoGainStatistics2(rowFilterLookup, numBinsPerFeature, treeSeeds, mtry, numClasses.get)
      } else {
        new VarianceStatistics2(rowFilterLookup, numBinsPerFeature, treeSeeds, mtry)
      }

      while (rows.hasNext) {
        val row = rows.next()
        cfor(0)(_ < numTrees, _ + 1)(
          treeId => {
            val curNodeId = row._2(treeId)
            val rowCnt = row._1._3(treeId).toInt
            if (rowCnt > 0 && curNodeId >= 0) { // It can be -1 if the sample was used in a node that was trained locally as a sub-tree.
              val nodeSplit = rowFilterLookup.getNodeSplit(treeId, curNodeId)
              if (nodeSplit != null) {
                val childNodeId = nodeSplit.selectChildNode(featureHandlerLocal.convertToInt(row._1._2(nodeSplit.featureId)))
                featureHandlerLocal.addRowToStats(partitionStats, treeId, childNodeId, row._1)
              }
            }
          }
        )
      }

      partitionStats.toNodeStatisticsIterator
    }).groupBy((nodeStats: (Int, NodeStatistics)) => nodeStats._1, nodeSplitCount).map(nodeStat => {
      val stats = nodeStat._2
      var mergedStats: NodeStatistics = null
      stats.foreach(stat => {
        if (mergedStats == null) {
          mergedStats = stat._2
        } else {
          mergedStats.binStatsArray.mergeInPlace(stat._2.binStatsArray)
        }
      })

      val treeId = mergedStats.treeId
      val nodeId = mergedStats.nodeId

      mergedStats.computeNodePredictionAndSplit(
        categoricalFeatureFlags = categoricalFeatureFlags,
        featureMissingValueBinIds = featureMissingValueBinIds,
        numBinsPerFeature = numBinsPerFeature,
        seed = treeSeeds(treeId) + nodeId,
        minSplitSize = minSplitSize,
        imputationType = imputationType)
    }).collect()

    // Update the nodeIdRDDs to reflect new Node IDs.
    updateNodeIdRDD(rowFilterLookup, numTrees)

    val finalTrainedNodes = new Array[TrainedNodeInfo](trainedNodes.length)
    quickSort[TrainedNodeInfo](trainedNodes)(Ordering.by[TrainedNodeInfo, Int](trainedNode => trainedNode.nodeId))
    cfor(0)(_ < trainedNodes.length, _ + 1)(
      i => {
        val trainedNode = trainedNodes(i)
        val treeId = trainedNode.treeId
        val nodeId = trainedNode.nodeId
        val nodeSplit = trainedNode.nodeSplit
        val nodeDepth = nodeDepths(treeId)(nodeId)
        nodeDepths(treeId).remove(nodeId)

        val splitImpurity = if (nodeDepth >= depthLimit) {
          None
        } else {
          trainedNode.splitImpurity
        }

        val newNodeSplit = if (nodeSplit != None && nodeDepth < depthLimit) {
          if (nodeSplit.get.isInstanceOf[CategoricalSplitOnBinId]) {
            val catSplit = nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId]
            val binIdToNodeIdMap = mutable.Map[Int, Int]()
            val nodeWeights = mutable.Map[Int, Double]()

            var leftChildNodeId = -1
            if (catSplit.nodeWeights.contains(0)) {
              leftChildNodeId = nextNodeIdsPerTree(treeId)
              nodeDepths(treeId).put(leftChildNodeId, nodeDepth + 1)
              val leftChildWeight = catSplit.nodeWeights(0)
              nodeWeights.put(leftChildNodeId, leftChildWeight)
              nextNodeIdsPerTree(treeId) += 1
            }

            var rightChildNodeId = -1
            if (catSplit.nodeWeights.contains(1)) {
              rightChildNodeId = nextNodeIdsPerTree(treeId)
              nodeDepths(treeId).put(rightChildNodeId, nodeDepth + 1)
              val rightChildWeight = catSplit.nodeWeights(1)
              nodeWeights.put(rightChildNodeId, rightChildWeight)
              nextNodeIdsPerTree(treeId) += 1
            }

            var nanChildNodeId = -1
            if (catSplit.nodeWeights.contains(2)) {
              nanChildNodeId = nextNodeIdsPerTree(treeId)
              nodeDepths(treeId).put(nanChildNodeId, nodeDepth + 1)
              val nanChildWeight = catSplit.nodeWeights(2)
              nodeWeights.put(nanChildNodeId, nanChildWeight)
              nextNodeIdsPerTree(treeId) += 1
            }

            catSplit.binIdToNodeIdMap.foreach(binId_stubNodeId => {
              val binId = binId_stubNodeId._1
              val stubNodeId = binId_stubNodeId._2
              val nodeId = stubNodeId match {
                case 0 => leftChildNodeId
                case 1 => rightChildNodeId
                case 2 => nanChildNodeId
              }

              binIdToNodeIdMap.put(binId, nodeId)
            })

            Some(CategoricalSplitOnBinId(
              parentNodeId = nodeId,
              featureId = catSplit.featureId,
              binIdToNodeIdMap = binIdToNodeIdMap,
              nodeWeights = nodeWeights
            ))
          } else {
            val numSplit = nodeSplit.get.asInstanceOf[NumericSplitOnBinId]
            var leftChildNodeId = -1
            if (numSplit.leftId == 0) {
              leftChildNodeId = nextNodeIdsPerTree(treeId)
              nodeDepths(treeId).put(leftChildNodeId, nodeDepth + 1)
              nextNodeIdsPerTree(treeId) += 1
            }

            var rightChildNodeId = -1
            if (numSplit.rightId == 1) {
              rightChildNodeId = nextNodeIdsPerTree(treeId)
              nodeDepths(treeId).put(rightChildNodeId, nodeDepth + 1)
              nextNodeIdsPerTree(treeId) += 1
            }

            var nanChildNodeId = -1
            if (numSplit.nanNodeId == 2) {
              nanChildNodeId = nextNodeIdsPerTree(treeId)
              nodeDepths(treeId).put(nanChildNodeId, nodeDepth + 1)
              nextNodeIdsPerTree(treeId) += 1
            }

            Some(NumericSplitOnBinId(
              parentNodeId = nodeId,
              featureId = numSplit.featureId,
              splitBinId = numSplit.splitBinId,
              leftId = leftChildNodeId,
              rightId = rightChildNodeId,
              leftWeight = numSplit.leftWeight,
              rightWeight = numSplit.rightWeight,
              leftSubTreeHash = numSplit.leftSubTreeHash,
              rightSubTreeHash = numSplit.rightSubTreeHash,
              nanBinId = numSplit.nanBinId,
              nanNodeId = nanChildNodeId,
              nanWeight = numSplit.nanWeight,
              nanSubTreeHash = numSplit.nanSubTreeHash
            ))
          }
        } else {
          None
        }

        finalTrainedNodes(i) = TrainedNodeInfo(
          treeId = trainedNode.treeId,
          nodeId = trainedNode.nodeId,
          prediction = trainedNode.prediction,
          depth = nodeDepth,
          weight = trainedNode.weight,
          impurity = trainedNode.impurity,
          splitImpurity = splitImpurity,
          nodeSplit = newNodeSplit
        )
      }
    )

    finalTrainedNodes.toIterator
  }

  /**
   * Update the nodeID RDD using the row filter lookup.
   * Each row filter maps from a parent Node to a child Node.
   * @param rowFilterLookup The row filter lookup that contains the row filters that we want to apply
   * @param numTrees The number of trees that we are training.
   */
  private def updateNodeIdRDD(rowFilterLookup: ScheduledNodeSplitLookup, numTrees: Int, markSubTreesOnly: Boolean = false): Unit = {
    // TODO: This seems stupid.
    val featureHandlerLocal = featureHandler.cloneMyself // To avoid serializing the entire DiscretizedData object.
    if (prevNodeIdRDD != null) {
      // Unpersist the previous one if one exists.
      prevNodeIdRDD.unpersist()
    }

    // The current one becomes the previous one before we update.
    prevNodeIdRDD = nodeIdRDD

    // Now, update all rows' node Ids.
    val newNodeIdRDD = data.zip(nodeIdRDD).map(row => {
      val nodeIds = row._2
      cfor(0)(_ < numTrees, _ + 1)(
        treeId => {
          val curNodeId = row._2(treeId)
          val rowCnt = row._1._3(treeId).toInt
          if (rowCnt > 0 && curNodeId >= 0) { // It can be -1 if the sample was used in a node that was trained locally as a sub-tree.
            val nodeSplit = rowFilterLookup.getNodeSplit(treeId, curNodeId)
            if (nodeSplit != null) {
              val childNodeId = nodeSplit.selectChildNode(featureHandlerLocal.convertToInt(row._1._2(nodeSplit.featureId)))
              // If the sub tree has is non-negative, it means that this child node has been scheduled to train as a sub-tree.
              // We then set the appended nodeID to -1 to prevent this row from being used in future training.
              if (nodeSplit.getSubTreeHash(childNodeId) >= 0) {
                nodeIds(treeId) = -1
              } else if (!markSubTreesOnly) {
                nodeIds(treeId) = childNodeId
              }
            }
          }
        }
      )

      nodeIds
    })

    newNodeIdRDD.persist(data.getStorageLevel)

    if (checkpointRootDir != null) {
      // Check the checkpoint queue and delete old unneeded ones.
      var canDelete = true
      while (checkpointQueue.size > 1 && canDelete) {
        if (checkpointQueue.get(1).get.getCheckpointFile != None) { // The second to the last one must have been checkpointed for us to delete the last one.
          val old = checkpointQueue.dequeue
          val fs = FileSystem.get(old.sparkContext.hadoopConfiguration)
          println("Deleting the old checkpointed folder " + old.getCheckpointFile.get)
          fs.delete(new Path(old.getCheckpointFile.get), true)
        } else {
          canDelete = false
        }
      }

      nodeIdRDDUpdateCount += 1
      if (nodeIdRDDUpdateCount >= checkpointInterval) {
        newNodeIdRDD.checkpoint()
        checkpointQueue.enqueue(newNodeIdRDD)
        nodeIdRDDUpdateCount = 0
      }
    }

    nodeIdRDD = newNodeIdRDD
  }
}

/**
 * A parent type for local array data sources.
 */
class DiscretizedDataLocal[@specialized(Byte, Short) T](data: Array[((Double, Array[T], Array[Byte]), Array[Int])])(featureHandler: FeatureHandler[T]) extends DiscretizedData {
  /**
   * Initialize the node IDs for training rows.
   * @param numTrees Number of trees that we have.
   */
  override def initializeRowNodeIds(numTrees: Int): Unit = {
    cfor(0)(_ < data.length, _ + 1)(
      rowId => cfor(0)(_ < numTrees, _ + 1)(treeId => data(rowId)._2(treeId) = 0)
    )
  }

  /**
   * This doesn't mean anything for local data sources.
   */
  override def trainSubTreesLocally(
    subTreeLookup: ScheduledNodeSplitLookup,
    featureBins: Array[Bins],
    nodeDepths: Array[mutable.Map[Int, Int]],
    options: SequoiaForestOptions,
    treeSeeds: Array[Int]): Iterator[(Int, SequoiaTree)] = {
    throw new UnsupportedOperationException("SubTree training is not meant for local data sources.")
  }

  /**
   * Apply row filters on rows, and collect/aggregate bin statistics on matching rows and nodes.
   * This will also update the node ID tags on individual rows.
   * @param rowFilterLookup A fast lookup for matching row filters for particular nodes.
   * @param treeSeeds Random seeds to use per tree. This is used in selecting a random number of features per node.
   * @param numBinsPerFeature Number of bins per feature. This is used to initialize the statistics object.
   * @param options The options to be used in Sequoia Forest.
   * @return Aggregated statistics for tree/node/feature/bin combinations.
   */
  override def applyRowFiltersAndAggregateStatistics(
    rowFilterLookup: ScheduledNodeSplitLookup,
    treeSeeds: Array[Int],
    numBinsPerFeature: Array[Int],
    options: SequoiaForestOptions): AggregatedStatistics = {
    val aggregatedStats: AggregatedStatistics = if (options.treeType == TreeType.Classification_InfoGain) {
      new InfoGainStatistics(rowFilterLookup, numBinsPerFeature, treeSeeds, options.mtry, options.numClasses.get)
    } else {
      new VarianceStatistics(rowFilterLookup, numBinsPerFeature, treeSeeds, options.mtry)
    }

    cfor(0)(_ < data.length, _ + 1)(
      rowId => {
        val row = data(rowId)
        cfor(0)(_ < options.numTrees, _ + 1)(
          treeId => {
            val curNodeId = row._2(treeId)
            val rowCnt = row._1._3(treeId).toInt
            if (rowCnt > 0 && curNodeId >= 0) {
              val nodeSplit = rowFilterLookup.getNodeSplit(treeId, curNodeId)
              if (nodeSplit != null) {
                val childNodeId = nodeSplit.selectChildNode(featureHandler.convertToInt(row._1._2(nodeSplit.featureId)))
                featureHandler.addRowToStats(aggregatedStats, treeId, childNodeId, row._1)
                row._2(treeId) = childNodeId
              }
            }
          }
        )
      }
    )

    aggregatedStats
  }

  /**
   * This doesn't mean anything for local data sources.
   */
  def performDistributedNodeSplits(
    rowFilterLookup: ScheduledNodeSplitLookup,
    treeSeeds: Array[Int],
    numBinsPerFeature: Array[Int],
    options: SequoiaForestOptions,
    featureBins: Array[Bins],
    nextNodeIdsPerTree: Array[Int],
    nodeDepths: Array[mutable.Map[Int, Int]]): Iterator[TrainedNodeInfo] = {
    throw new UnsupportedOperationException("Distributed node splits is not meant for local data sources.")
  }
}

/**
 * An RDD data source with unsigned Bytes as features.
 * @param data The RDD of training rows with unsigned Byte for features.
 */
case class UnsignedByteRDD(data: RDD[(Double, Array[Byte], Array[Byte])]) extends DiscretizedDataRDD[Byte](data)(new UnsignedByteFeatureHandler)

/**
 * An RDD data source with unsigned Short as features.
 * @param data The RDD of training rows with unsigned Short for features.
 */
case class UnsignedShortRDD(data: RDD[(Double, Array[Short], Array[Byte])]) extends DiscretizedDataRDD[Short](data)(new UnsignedShortFeatureHandler)

/**
 * An array data source with unsigned Byte as features.
 * @param data The array of training rows with unsigned Byte for features.
 */
case class UnsignedByteLocal(data: Array[((Double, Array[Byte], Array[Byte]), Array[Int])]) extends DiscretizedDataLocal[Byte](data)(new UnsignedByteFeatureHandler)

/**
 * An array data source with unsigned Short as features.
 * @param data The array of training rows with unsigned Short for features.
 */
case class UnsignedShortLocal(data: Array[((Double, Array[Short], Array[Byte]), Array[Int])]) extends DiscretizedDataLocal[Short](data)(new UnsignedShortFeatureHandler)
