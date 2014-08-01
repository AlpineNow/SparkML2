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
import spark_ml.discretization.{ Bins, Discretizer }

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
   * @return An iterator of trained trees, the first element is the parent tree ID. Sub-trees will have IDs that match the child ID of the parent tree.
   */
  def trainSubTreesLocally(
    subTreeLookup: ScheduledNodeSplitLookup,
    featureBins: Array[Bins],
    nodeDepths: Array[mutable.Map[Int, Int]],
    options: SequoiaForestOptions): Iterator[(Int, SequoiaTree)]

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
  override def isLocal: Boolean = false

  /**
   * @return SparkContext of the RDD.
   */
  override def getSparkContext: SparkContext = data.sparkContext

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
   * @return An iterator of trained trees, the first element is the parent tree ID. Sub-trees will have IDs that match the child ID of the parent tree.
   */
  override def trainSubTreesLocally(
    subTreeLookup: ScheduledNodeSplitLookup,
    featureBins: Array[Bins],
    nodeDepths: Array[mutable.Map[Int, Int]],
    options: SequoiaForestOptions): Iterator[(Int, SequoiaTree)] = {
    val featureHandlerLocal = featureHandler.cloneMyself // To avoid serializing the entire DiscretizedData object.
    val numTrees = options.numTrees
    val treeType = options.treeType
    val mtry = options.mtry
    val minSplitSize = options.minSplitSize
    val depthLimit = options.maxDepth match {
      case x if x == -1 => Int.MaxValue
      case x => x
    }

    val numNodesPerIteration = options.numNodesPerIteration
    val numClasses = options.numClasses
    val numSubTreesToTrain = subTreeLookup.subTreeCount
    val shuffledRows = data.zip(nodeIdRDD).flatMap(row => {
      val output = new mutable.ArrayBuffer[(Int, Int, Int, ((Double, Array[T], Array[Byte]), Array[Int]))]()
      var treeId = 0
      while (treeId < numTrees) {
        val curNodeId = row._2(treeId)
        if (curNodeId >= 0) {
          val nodeSplit = subTreeLookup.getNodeSplit(treeId, curNodeId)
          if (nodeSplit != null) {
            val childNodeId = nodeSplit.selectChildNode(featureHandlerLocal.convertToInt(row._1._2(nodeSplit.featureId)))
            val subTreeHash = nodeSplit.getSubTreeHash(childNodeId)
            if (subTreeHash >= 0) {
              output += ((subTreeHash, treeId, childNodeId, row))
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
          outputPath = None,
          numClasses = numClasses),
        new ConsoleNotifiee,
        None)

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
    val featureHandlerLocal = featureHandler.cloneMyself // To avoid serializing the entire DiscretizedData object.

    // First aggregate bin statistics across all the partitions.
    val numTrees = options.numTrees
    val treeType = options.treeType
    val mtry = options.mtry
    val numClasses = options.numClasses
    val aggregatedArray = data.zip(nodeIdRDD).mapPartitions(rows => {
      val partitionStats = if (treeType == TreeType.Classification_InfoGain) {
        InfoGainStatistics(rowFilterLookup, numBinsPerFeature, treeSeeds, mtry, numClasses.get)
      } else {
        VarianceStatistics(rowFilterLookup, numBinsPerFeature, treeSeeds, mtry)
      }

      while (rows.hasNext) {
        val row = rows.next()
        var treeId = 0
        while (treeId < numTrees) {
          val curNodeId = row._2(treeId)
          if (curNodeId >= 0) { // It can be -1 if the sample was used in a node that was trained locally as a sub-tree.
            val nodeSplit = rowFilterLookup.getNodeSplit(treeId, curNodeId)
            if (nodeSplit != null) {
              val childNodeId = nodeSplit.selectChildNode(featureHandlerLocal.convertToInt(row._1._2(nodeSplit.featureId)))
              featureHandlerLocal.addRowToStats(partitionStats, treeId, childNodeId, row._1)
            }
          }

          treeId += 1
        }
      }

      Array(partitionStats.binStatsArray).toIterator
    }).reduce((a, b) => a.mergeInPlace(b))

    // Update the nodeIdRDDs to reflect new Node IDs.
    updateNodeIdRDD(rowFilterLookup, numTrees)

    val totalStats = if (treeType == TreeType.Classification_InfoGain) {
      InfoGainStatistics(rowFilterLookup, numBinsPerFeature, treeSeeds, mtry, numClasses.get)
    } else {
      VarianceStatistics(rowFilterLookup, numBinsPerFeature, treeSeeds, mtry)
    }

    totalStats.binStatsArray = aggregatedArray

    // Return the aggregated statistics.
    totalStats
  }

  /**
   * Update the nodeID RDD using the row filter lookup.
   * Each row filter maps from a parent Node to a child Node.
   * @param rowFilterLookup The row filter lookup that contains the row filters that we want to apply
   * @param numTrees The number of trees that we are training.
   */
  private def updateNodeIdRDD(rowFilterLookup: ScheduledNodeSplitLookup, numTrees: Int, markSubTreesOnly: Boolean = false): Unit = {
    val featureHandlerLocal = featureHandler.cloneMyself // To avoid serializing the entire DiscretizedData object.

    // Now, update all rows' node Ids.
    nodeIdRDD = data.zip(nodeIdRDD).map(row => {
      val nodeIds = row._2
      var treeId = 0
      while (treeId < numTrees) {
        val curNodeId = row._2(treeId)
        if (curNodeId >= 0) { // It can be -1 if the sample was used in a node that was trained locally as a sub-tree.
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

        treeId += 1
      }

      nodeIds
    })

    // To improve performance.
    // nodeIdRDD.checkpoint()
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
    var rowId = 0
    while (rowId < data.length) {
      var treeId = 0
      while (treeId < numTrees) {
        data(rowId)._2(treeId) = 0
        treeId += 1
      }

      rowId += 1
    }
  }

  /**
   * This doesn't mean anything for local data sources.
   */
  override def trainSubTreesLocally(
    subTreeLookup: ScheduledNodeSplitLookup,
    featureBins: Array[Bins],
    nodeDepths: Array[mutable.Map[Int, Int]],
    options: SequoiaForestOptions): Iterator[(Int, SequoiaTree)] = {
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
      InfoGainStatistics(rowFilterLookup, numBinsPerFeature, treeSeeds, options.mtry, options.numClasses.get)
    } else {
      VarianceStatistics(rowFilterLookup, numBinsPerFeature, treeSeeds, options.mtry)
    }

    var rowId = 0
    while (rowId < data.length) {
      val row = data(rowId)
      var treeId = 0
      while (treeId < options.numTrees) {
        val curNodeId = row._2(treeId)
        if (curNodeId >= 0) {
          val nodeSplit = rowFilterLookup.getNodeSplit(treeId, curNodeId)
          if (nodeSplit != null) {
            val childNodeId = nodeSplit.selectChildNode(featureHandler.convertToInt(row._1._2(nodeSplit.featureId)))
            featureHandler.addRowToStats(aggregatedStats, treeId, childNodeId, row._1)
            row._2(treeId) = childNodeId
          }
        }

        treeId += 1
      }
      rowId += 1
    }

    aggregatedStats
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
