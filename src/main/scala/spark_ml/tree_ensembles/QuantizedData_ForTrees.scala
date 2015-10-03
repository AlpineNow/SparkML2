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

package spark_ml.tree_ensembles

import scala.collection.mutable
import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import spark_ml.discretization.Bins
import spark_ml.util.Sorting._
import spark_ml.util.{ConsoleNotifiee, DiscretizedFeatureHandler, MapWithSequentialIntKeys}
import spire.implicits._

/**
 * Options that will be used by the distributed dataset to propagate options
 * across the driver and executors.
 * @param numTrees Number of trees.
 * @param treeType Tree type (split criteria).
 * @param mtry Number of random features to analyze per node.
 *             Useful for random forest.
 * @param maxDepth The maximum allowed depth for trees.
 * @param minSplitSize The minimum size of the node to be eligible for splits.
 * @param treeSeeds Random seeds to be used for random feature selection for
 *                  each tree.
 * @param featureBinsInfo Discretization bin info for each tree.
 * @param catSplitType Categorical feature split types. It can be binary or
 *                     multi-way splits.
 * @param numClasses Optional number of target classes, iff the tree type is a
 *                   classification type.
 * @param verbose If true, the algorithm will print as much information through
 *                the notifiee as possible, including many intermediate
 *                computation values, etc.
 */
case class BroadcastOptions(
  numTrees: Int,
  treeType: SplitCriteria.SplitCriteria,
  mtry: Int,
  maxDepth: Int,
  minSplitSize: Int,
  treeSeeds: Array[Int],
  featureBinsInfo: Array[Bins],
  catSplitType: CatSplitType.CatSplitType,
  numClasses: Option[Int],
  verbose: Boolean)

/**
 * Quantized training data trait for tree ensembles.
 * These can be a wrapper for either an RDD or a local array.
 *
 * Feature values must be quantized into
 * either unsigned Byte or Short.
 */
trait QuantizedData_ForTrees {
  /**
   * Whether this quantized data is a local array.
   * @return true if it's a local array. false otherwise.
   */
  def isLocal: Boolean

  /**
   * In case this is not local, we might want to get the Spark context.
   * @return SparkContext.
   */
  def getSparkContext: SparkContext

  /**
   * Set options for the tree ensemble.
   * @param options Options for the tree ensemble.
   * @param treeSeeds Random seeds for trees.
   */
  def setOptions(options: TreeForestTrainerOptions, treeSeeds: Array[Int]): Unit

  /**
   * Aggregate statistics and then train/split nodes that are specified in the
   * look up object.
   * @param idLookUp The look up object that contains the nodes that we want
   *                 to aggregate statistics for and then split.
   * @return An ordered array of node info, containing aggregated values
   *         and potential split info.
   */
  def aggregateAndSplit(idLookUp: IdLookupForNodeStats): Array[NodeInfo]

  /**
   * Assign proper split/node Ids to child nodes and also queue eligible child
   * nodes to the split queue.
   * @param splitQueue Split queue that we want to add to.
   * @param nodeInfoArray An array of trained nodes.
   * @param nextSplitIdPerTree Next split Ids to assign to eligible children.
   * @param nextNodeIdPerTree Next node Ids to assign to children.
   * @param splitIdToNodeId Map from split Ids to node Ids.
   */
  def updateIdsAndQueues_local(
    splitQueue: mutable.Queue[(Int, Int, Int)],
    nodeInfoArray: Array[NodeInfo],
    nextSplitIdPerTree: Array[Int],
    nextNodeIdPerTree: Array[Int],
    splitIdToNodeId: Array[MapWithSequentialIntKeys[Int]]): Unit

  /**
   * Assign proper split/sub-tree/node Ids to child nodes and also queue
   * eligible child nodes to the split/sub-tree queues.
   * @param splitQueue Split queue that we want to add to.
   * @param subTreeQueue Sub tree queue that we want to add to.
   * @param nodeInfoArray An array of trained nodes.
   * @param nextSplitIdPerTree Next split Ids to assign to eligible children.
   * @param nextSubTreeIdPerTree Next sub-tree Ids to assign to
   *                             eligible children.
   * @param nextNodeIdPerTree Next node Ids to assign to children.
   * @param splitIdToNodeId Map from split Ids to node Ids.
   * @param subTreeIdToNodeId Map from sub-tree Ids to node Ids.
   */
  def updateIdsAndQueues_distributed(
    splitQueue: mutable.Queue[(Int, Int, Int)],
    subTreeQueue: mutable.Queue[(Int, Int, Int)],
    nodeInfoArray: Array[NodeInfo],
    nextSplitIdPerTree: Array[Int],
    nextSubTreeIdPerTree: Array[Int],
    nextNodeIdPerTree: Array[Int],
    splitIdToNodeId: Array[MapWithSequentialIntKeys[Int]],
    subTreeIdToNodeId: Array[MapWithSequentialIntKeys[Int]]): Unit

  /**
   * Train sub-trees and output their nodes.
   * @param subTreeLookup A look up object to find data points matching
   *                      sub-trees to train.
   * @param nextNodeIdPerTree Next node Ids to assign to sub-tree nodes.
   * @param subTreeIdToNodeId Sub tree Id to node Id.
   * @return An iterator of trained sub-tree nodes. They should have node Ids
   *         that are properly adjusted to be inserted into the parent trees.
   */
  def trainSubTrees(
    subTreeLookup: IdLookupForSubTreeInfo,
    nextNodeIdPerTree: Array[Int],
    subTreeIdToNodeId: Array[MapWithSequentialIntKeys[Int]]): Iterator[NodeInfo]
}

/**
 * Quantized training data in RDD. The features are quantized into either Byte
 * or Short values.
 * @param data A bagged quantized dataset.
 *             The first Double value is the label.
 *             The second array is the quantized features.
 *             The last array is the sample count per tree (in Byte).
 * @param idCache Initial id cache to keep track of internal split/node Ids.
 * @param featureBinsInfo Discretized feature info.
 * @param featureHandler The feature type handler
 *                      (to handle conversion to/from unsigned Byte/Short and Int)
 * @tparam T The feature type.
 */
class QuantizedData_ForTrees_Rdd[@specialized(Byte, Short) T](
  data: RDD[((Double, Array[T]), Array[Byte])],
  idCache: IdCache,
  featureBinsInfo: Array[Bins],
  featureHandler: DiscretizedFeatureHandler[T]) extends QuantizedData_ForTrees {

  // To be set by the setOptions function.
  var broadcastOpts: Broadcast[BroadcastOptions] = null
  var subTreeWeightThreshold: Double = 0

  /**
   * This is a distributed RDD, so it's false.
   * @return false
   */
  def isLocal: Boolean = false

  /**
   * Return the spark context of the RDD.
   * @return SparkContext.
   */
  def getSparkContext: SparkContext = data.sparkContext

  /**
   * Set the options that will be used across the driver and executors.
   * @param options Options for the tree ensemble.
   * @param treeSeeds Random seeds for trees.
   */
  def setOptions(options: TreeForestTrainerOptions, treeSeeds: Array[Int]): Unit = {
    val bOpts = BroadcastOptions(
      numTrees = options.numTrees,
      treeType = options.splitCriteria,
      mtry = options.mtry,
      maxDepth = options.maxDepth,
      minSplitSize = options.minSplitSize,
      treeSeeds = treeSeeds,
      featureBinsInfo = featureBinsInfo,
      catSplitType = options.catSplitType,
      numClasses = options.numClasses,
      verbose = options.verbose
    )

    broadcastOpts = data.sparkContext.broadcast(bOpts)
    subTreeWeightThreshold = options.subTreeWeightThreshold
  }

  /**
   * Aggregate statistics and then train/split nodes that are specified in the
   * look up object.
   * @param idLookUp The look up object that contains the nodes that we want to
   *                 aggregate statistics for and then split.
   * @return An ordered array of node info, containing aggregated values and
   *         potential split info.
   */
  def aggregateAndSplit(idLookUp: IdLookupForNodeStats): Array[NodeInfo] = {
    val fh = featureHandler
    val bOpts = broadcastOpts
    val splitNodes = data.zip(idCache.getRdd).mapPartitions(
      rows => {
        val numTrees = bOpts.value.numTrees
        val treeType = bOpts.value.treeType
        val featureBinsInfo = bOpts.value.featureBinsInfo
        val treeSeeds = bOpts.value.treeSeeds
        val mtry = bOpts.value.mtry
        val numClasses = bOpts.value.numClasses
        idLookUp.initNodeStats(
          treeType = treeType,
          featureBinsInfo = featureBinsInfo,
          treeSeeds = treeSeeds,
          mtry = mtry,
          numClasses = numClasses
        )

        while (rows.hasNext) {
          val row = rows.next()
          cfor(0)(_ < numTrees, _ + 1)(
            treeId => {
              val curId = row._2(treeId)
              val baggedRow = row._1
              val rowCnt = baggedRow._2(treeId).toInt
              if (rowCnt > 0 && curId >= 1) {
                val nodeStats = idLookUp.get(treeId = treeId, id = curId)
                if (nodeStats != null) {
                  val (label, features) = baggedRow._1
                  nodeStats.addSample(
                    label = label,
                    features = features,
                    sampleCnt = rowCnt,
                    featureHandler = fh
                  )
                }
              }
            }
          )
        }

        idLookUp.toHashedNodeStatsIterator._1
      }
    ).reduceByKey(
        (nodeStats1, nodeStats2) => {
          // Merge the matching node statistics from individual partitions
          // to get the final aggregated statistics for different nodes.
          // This is being done across multiple executors.
          nodeStats1.mergeInPlace(nodeStats2)
          nodeStats1
        }
    ).map {
      case (_, nodeStats) =>
        val treeSeeds = bOpts.value.treeSeeds
        val nodeSeed = treeSeeds(nodeStats.treeId) + nodeStats.nodeId
        val minSplitSize = bOpts.value.minSplitSize
        val maxDepth = bOpts.value.maxDepth
        val catSplitType = bOpts.value.catSplitType
        nodeStats.splitNode(
          nodeSeed = nodeSeed,
          minSplitSize = minSplitSize,
          maxDepth = maxDepth,
          catSplitType = catSplitType
        )
    }.collect()

    // Sort the split nodes by the node Id so that we are training in a
    // breadth-first fashion.
    quickSort[NodeInfo](splitNodes)(
      Ordering.by[NodeInfo, Int](splitNode => splitNode.nodeId)
    )

    splitNodes
  }

  /*
   * The local updateIdsAndQueues function is not implemented for this one.
   */
  def updateIdsAndQueues_local(
    splitQueue: mutable.Queue[(Int, Int, Int)],
    nodeInfoArray: Array[NodeInfo],
    nextSplitIdPerTree: Array[Int],
    nextNodeIdPerTree: Array[Int],
    splitIdToNodeId: Array[MapWithSequentialIntKeys[Int]]): Unit = {
    throw new UnsupportedOperationException(
      "updateIdsAndQueues_local is not implemented."
    )
  }

  /**
   * Assign proper split/sub-tree/node Ids to child nodes and also queue
   * eligible child nodes to the split/sub-tree queues.
   * @param splitQueue Split queue that we want to add to.
   * @param subTreeQueue Sub tree queue that we want to add to.
   * @param nodeInfoArray An array of trained nodes.
   * @param nextSplitIdPerTree Next split Ids to assign to eligible children.
   * @param nextSubTreeIdPerTree Next sub-tree Ids to assign to eligible
   *                             children.
   * @param nextNodeIdPerTree Next node Ids to assign to children.
   * @param splitIdToNodeId Map from split Ids to node Ids.
   * @param subTreeIdToNodeId Map from sub-tree Ids to node Ids.
   */
  def updateIdsAndQueues_distributed(
    splitQueue: mutable.Queue[(Int, Int, Int)],
    subTreeQueue: mutable.Queue[(Int, Int, Int)],
    nodeInfoArray: Array[NodeInfo],
    nextSplitIdPerTree: Array[Int],
    nextSubTreeIdPerTree: Array[Int],
    nextNodeIdPerTree: Array[Int],
    splitIdToNodeId: Array[MapWithSequentialIntKeys[Int]],
    subTreeIdToNodeId: Array[MapWithSequentialIntKeys[Int]]): Unit = {
    val numTrees = broadcastOpts.value.numTrees
    val maxDepth = broadcastOpts.value.maxDepth
    val minSplitSize = broadcastOpts.value.minSplitSize
    val idRanges = Array.fill[IdRange](numTrees)(null)
    val updaters = Array.fill[MapWithSequentialIntKeys[IdUpdater]](numTrees)(
      null
    )
    val numSplitNodes = nodeInfoArray.length
    val avgSplitNodesPerTree = math.ceil(
      numSplitNodes.toDouble / numTrees.toDouble
    ).toInt

    // Keep track of split node child node Ids.
    // The split node Id would be the split Id.
    // The second map Id is the child index of the split (ordered).
    // The last value is the assigned 'real' node Id for that child.
    val splitNodeChildNodeIds =
      Array.fill[MapWithSequentialIntKeys[MapWithSequentialIntKeys[Int]]](numTrees)(
        new MapWithSequentialIntKeys[MapWithSequentialIntKeys[Int]](
          initCapacity = avgSplitNodesPerTree * 2
        )
      )

    // Find child nodes that are to be split further (either through distributed
    // splits or sub-tree training) and update their Ids.
    cfor(0)(_ < numSplitNodes, _ + 1)(
      i => {
        val splitNode = nodeInfoArray(i)
        val treeId = splitNode.treeId
        val id = splitNode.nodeId
        if (idRanges(treeId) == null) {
          idRanges(treeId) = IdRange(id, id)
          updaters(treeId) = new MapWithSequentialIntKeys[IdUpdater](
            initCapacity = avgSplitNodesPerTree * 2
          )
        } else {
          idRanges(treeId).endId += 1
          if (id != idRanges(treeId).endId) {
            throw new AssertionError(
              "The expected split Id was " + idRanges(treeId).endId +
                " but we found " + id
            )
          }
        }

        // For this split node, we need another map to keep track of the
        // assigned child node Ids.
        splitNodeChildNodeIds(treeId).put(id,
          new MapWithSequentialIntKeys[Int](
            initCapacity = 4
          )
        )

        val splitInfo: NodeSplitInfo =
          if (splitNode.splitInfo.nonEmpty) {
            val si = splitNode.splitInfo.get
            // We have to find out whether the child node can be further
            // split or it's terminal.
            val children = si.getOrderedChildNodes
            val numChildren = children.length
            cfor(0)(_ < numChildren, _ + 1)(
              j => {
                val child = children(j)

                // This is equivalent to assigning 'real' node Ids to trees in
                // a breadth-first manner. Both split Ids and node Ids are
                // assigned in the same manner, except that the split Ids will
                // skip terminal children.
                splitNodeChildNodeIds(treeId).get(id).put(j, nextNodeIdPerTree(treeId))

                if (child.isTerminal(maxDepth, minSplitSize)) {
                  // If the child is terminal, the Id in the RDD would be
                  // updated to 0's first.
                  child.nodeId = 0
                } else {
                  // Else, see if this one satisfies the sub-tree condition.
                  if (child.weight <= subTreeWeightThreshold) {
                    // If it does, assign a new sub tree Id
                    // so that the Id cache can be updated properly.
                    val subTreeId = nextSubTreeIdPerTree(treeId)

                    // Negative Ids mean sub-tree Ids. Only 0's mean finished
                    // terminal nodes.
                    child.nodeId = -subTreeId
                    subTreeQueue.enqueue((treeId, subTreeId, child.depth))
                    subTreeIdToNodeId(treeId).put(
                      subTreeId,
                      nextNodeIdPerTree(treeId)
                    )

                    nextSubTreeIdPerTree(treeId) += 1
                  } else {
                    // Otherwise, assign a new split Id so that the Id cache can
                    // be updated properly.
                    val splitId = nextSplitIdPerTree(treeId)
                    child.nodeId = splitId
                    splitQueue.enqueue((treeId, splitId, child.depth))
                    splitIdToNodeId(treeId).put(
                      splitId,
                      nextNodeIdPerTree(treeId)
                    )

                    nextSplitIdPerTree(treeId) += 1
                  }
                }

                // Now update the next assignable node Id.
                nextNodeIdPerTree(treeId) += 1
              }
            )
            si
          } else {
            // This means the node is terminal, and thus we have to update the
            // updater to indicate terminality.
            null
          }
        updaters(treeId).put(
          id,
          IdUpdater(
            if (splitInfo != null) {
              // Need to use a copy since idCache update is a lazy operation.
              splitInfo.copy
            } else {
              null
            }
          )
        )
      }
    )
    val idLookupForUpdaters = IdLookupForUpdaters.createIdLookupForUpdaters(
      idRanges = idRanges,
      updaterMaps = updaters
    )
    // Update idCache.
    idCache.updateIds(
      data = data,
      idLookupForUpdaters = idLookupForUpdaters,
      featureHandler = featureHandler
    )
    // Now update node Ids of split nodes to actual tree node Ids.
    cfor(0)(_ < numSplitNodes, _ + 1)(
      i => {
        val splitNode = nodeInfoArray(i)
        val treeId = splitNode.treeId
        val splitId = splitNode.nodeId
        // We have kept the split Id to node Id map.
        splitNode.nodeId = splitIdToNodeId(treeId).get(splitId)
        splitIdToNodeId(treeId).remove(splitId)
        if (splitNode.splitInfo.nonEmpty) {
          // Also, update child node ids to proper ones before we write this to
          // the store object.
          val si = splitNode.splitInfo.get
          val children = si.getOrderedChildNodes
          val numChildren = children.length
          cfor(0)(_ < numChildren, _ + 1)(
            j => {
              val child = children(j)
              if (child.isTerminal(maxDepth, minSplitSize)) {
                child.nodeId = splitNodeChildNodeIds(treeId).get(splitId).get(j)
              } else {
                if (child.nodeId < 0) {
                  // The actual node Id can be found in the sub-tree map.
                  val subTreeId = -child.nodeId
                  child.nodeId = subTreeIdToNodeId(treeId).get(subTreeId)
                } else {
                  // The actual node Id can be found in the split map.
                  val childSplitId = child.nodeId
                  child.nodeId = splitIdToNodeId(treeId).get(childSplitId)

                  // For sanity check, also compare with the split node child node
                  // Id map.
                  assert(
                    child.nodeId == splitNodeChildNodeIds(treeId).get(splitId).get(j),
                    "Inconsistent child node Ids : " + child.nodeId + " and " +
                      splitNodeChildNodeIds(treeId).get(splitId).get(j)
                  )
                }
              }
            }
          )
        }
      }
    )
  }

  /**
   * Train sub-trees and output their nodes.
   * @param subTreeLookup A look up object to find data points matching
   *                      sub-trees to train.
   * @param nextNodeIdPerTree Next node Ids to assign to sub-tree nodes.
   * @param subTreeIdToNodeId Sub tree Id to node Id.
   * @return An iterator of trained sub-tree nodes. They should have node Ids
   *         that are properly adjusted to be inserted into the parent trees.
   */
  def trainSubTrees(
    subTreeLookup: IdLookupForSubTreeInfo,
    nextNodeIdPerTree: Array[Int],
    subTreeIdToNodeId: Array[MapWithSequentialIntKeys[Int]]): Iterator[NodeInfo] = {
    val fh = featureHandler
    val bOpts = broadcastOpts
    val subTreeThreshold = subTreeWeightThreshold.toInt
    val subTrees = data.zip(idCache.getRdd).flatMap(
      row => {
        val numTrees = bOpts.value.numTrees
        val output =
          new mutable.ListBuffer[(SubTreeInfo, (Double, Array[T], Byte))]()
        cfor(0)(_ < numTrees, _ + 1)(
          treeId => {
            val curIdsArray = row._2
            val baggedPoint = row._1
            val curId = curIdsArray(treeId)
            val rowCnt = baggedPoint._2(treeId)
            if (rowCnt > 0 && curId < 0) {
              val subTreeInfo = subTreeLookup.get(treeId, -curId)
              if (subTreeInfo != null) {
                val dataPoint = baggedPoint._1
                val label = dataPoint._1
                val features = dataPoint._2
                output += ((subTreeInfo, (label, features, rowCnt)))
              }
            }
          }
        )

        output.toIterator
      }
    ).groupByKey().map {
      case (subTreeInfo, rows) =>
        val parentTreeId = subTreeInfo.parentTreeId
        val subTreeId = subTreeInfo.id
        val subTreeDepth = subTreeInfo.depth
        val arrayData = rows.map(row => ((row._1, row._2), Array(row._3))).toArray
        val maxDepth = bOpts.value.maxDepth
        val minSplitSize = bOpts.value.minSplitSize
        val subTreeMaxDepth = maxDepth - subTreeDepth + 1
        val treeType = bOpts.value.treeType
        val catSplitType = bOpts.value.catSplitType
        val mtry = bOpts.value.mtry
        val numClasses = bOpts.value.numClasses
        val verbose = bOpts.value.verbose
        val featureBinsInfo = bOpts.value.featureBinsInfo
        val options = TreeForestTrainerOptions(
          numTrees = 1,
          splitCriteria = treeType,
          mtry = mtry,
          maxDepth = subTreeMaxDepth,
          minSplitSize = minSplitSize,
          catSplitType = catSplitType,
          maxSplitsPerIter = subTreeThreshold,
          subTreeWeightThreshold = 0.0,
          maxSubTreesPerIter = 0,
          numClasses = numClasses,
          verbose = verbose
        )
        val subTreeStore = new SubTreeStore(
          parentTreeId,
          subTreeId,
          subTreeDepth)
        val localData = new QuantizedData_ForTrees_Local(
          numTrees = 1,
          data = arrayData,
          featureBinsInfo = featureBinsInfo,
          typeHandler = fh
        )
        TreeForestTrainer.train(
          trainingData = localData,
          featureBinsInfo = featureBinsInfo,
          trainingOptions = options,
          modelStore = subTreeStore,
          notifiee = new ConsoleNotifiee,
          rng = new Random(parentTreeId + subTreeId)
        )
        subTreeStore.subTreeDesc
    }.collect()

    val subTreeNodes = new mutable.ListBuffer[NodeInfo]()
    val numSubTrees = subTrees.length
    if (numSubTrees != subTreeLookup.objCnt) {
      // The number of sub trees should be the same as the expected count.
      throw new AssertionError(
        "The expected number of sub-trees was " + subTreeLookup.objCnt +
          " but we found " + numSubTrees + " sub-trees."
      )
    }
    // Update sub-tree nodes with proper depths and node Ids that adhere to the
    // parent tree.
    cfor(0)(_ < numSubTrees, _ + 1)(
      i => {
        val subTree = subTrees(i)
        val parentTreeId = subTree.parentTreeId
        val subTreeId = subTree.subTreeId
        val rootNodeId = subTreeIdToNodeId(parentTreeId).get(subTreeId)
        val lastUsedNodeId = subTree.updateIdsAndDepths(
          rootId = rootNodeId,
          startChildNodeId = nextNodeIdPerTree(parentTreeId)
        )
        nextNodeIdPerTree(parentTreeId) = lastUsedNodeId + 1
        subTreeNodes ++= subTree.getOrderedNodeInfoSeq
      }
    )

    subTreeNodes.iterator
  }
}

/**
 * Quantized training data for a local array of data.
 * @param numTrees Number of trees to train locally.
 * @param data A bagged quantized dataset.
 *             The first Double value is the label.
 *             The second array is the quantized features.
 *             The last array is the sample count per tree (in Byte).
 * @param featureBinsInfo Discretized feature info.
 * @param typeHandler The feature type handler
 *                    (to handle conversion to/from unsigned Byte/Short and Int)
 * @tparam T The feature type.
 */
class QuantizedData_ForTrees_Local[@specialized(Byte, Short) T](
    numTrees: Int,
    data: Array[((Double, Array[T]), Array[Byte])],
    featureBinsInfo: Array[Bins],
    typeHandler: DiscretizedFeatureHandler[T]) extends QuantizedData_ForTrees {

  // To be set by the setOptions function.
  var opts: TreeForestTrainerOptions = null
  var tSeeds: Array[Int] = null
  val numRows = data.length
  // Initialize the Ids of rows to the root nodes.
  val idCache = Array.fill[Array[Int]](numRows)(Array.fill[Int](numTrees)(1))

  /**
   * This is local, so true.
   * @return true
   */
  def isLocal: Boolean = true

  /**
   * This is local, so null.
   * @return null
   */
  def getSparkContext: SparkContext = null

  /**
   * Set the options.
   * @param options Options for the tree ensemble.
   * @param treeSeeds Random seeds for trees.
   */
  def setOptions(options: TreeForestTrainerOptions, treeSeeds: Array[Int]): Unit = {
    this.opts = options
    this.tSeeds = treeSeeds
  }

  /**
   * Aggregate statistics and then train/split nodes that are specified in the
   * look up object.
   * @param idLookUp The look up object that contains the nodes that we want to
   *                 aggregate statistics for and then split.
   * @return An ordered array of node info, containing aggregated values and
   *         potential split info.
   */
  def aggregateAndSplit(idLookUp: IdLookupForNodeStats): Array[NodeInfo] = {
    // Initialize the look up object.
    idLookUp.initNodeStats(
      treeType = opts.splitCriteria,
      featureBinsInfo = featureBinsInfo,
      treeSeeds = tSeeds,
      mtry = opts.mtry,
      numClasses = opts.numClasses
    )
    cfor(0)(_ < numRows, _ + 1)(
      i => {
        cfor(0)(_ < numTrees, _ + 1)(
          treeId => {
            val curId = idCache(i)(treeId)
            val baggedRow = data(i)
            val rowCnt = baggedRow._2(treeId).toInt
            if (rowCnt > 0 && curId >= 1) {
              val nodeStats = idLookUp.get(treeId = treeId, id = curId)
              if (nodeStats != null) {
                val labelAndFeatures = baggedRow._1
                val label = labelAndFeatures._1
                val features = labelAndFeatures._2
                nodeStats.addSample(
                  label = label,
                  features = features,
                  sampleCnt = rowCnt,
                  featureHandler = typeHandler
                )
              }
            }
          }
        )
      }
    )

    // Process all the node stats.
    // The iterator is already ordered by Ids, so no need to order them again.
    val treeSeeds = this.tSeeds
    val minSplitSize = this.opts.minSplitSize
    val maxDepth = this.opts.maxDepth
    val catSplitType = this.opts.catSplitType
    val (nodeStatsItr, numNodeStats) = idLookUp.toHashedNodeStatsIterator
    val splitResults = new Array[NodeInfo](numNodeStats)
    while (nodeStatsItr.hasNext) {
      val (index, nodeStats) = nodeStatsItr.next()
      val nodeSeed = treeSeeds(nodeStats.treeId) + nodeStats.nodeId
      splitResults(index) = nodeStats.splitNode(
        nodeSeed = nodeSeed,
        minSplitSize = minSplitSize,
        maxDepth = maxDepth,
        catSplitType = catSplitType
      )
    }

    splitResults
  }

  /**
   * Assign proper split/node Ids to child nodes and also queue eligible child
   * nodes to the split queue.
   * @param splitQueue Split queue that we want to add to.
   * @param nodeInfoArray An array of trained nodes.
   * @param nextSplitIdPerTree Next split Ids to assign to eligible children.
   * @param nextNodeIdPerTree Next node Ids to assign to children.
   * @param splitIdToNodeId Map from split Ids to node Ids.
   */
  def updateIdsAndQueues_local(
    splitQueue: mutable.Queue[(Int, Int, Int)],
    nodeInfoArray: Array[NodeInfo],
    nextSplitIdPerTree: Array[Int],
    nextNodeIdPerTree: Array[Int],
    splitIdToNodeId: Array[MapWithSequentialIntKeys[Int]]): Unit = {
    val maxDepth = opts.maxDepth
    val minSplitSize = opts.minSplitSize
    val idRanges = Array.fill[IdRange](numTrees)(null)
    val updaters = Array.fill[MapWithSequentialIntKeys[IdUpdater]](numTrees)(
      null
    )
    val numSplitNodes = nodeInfoArray.length
    val avgSplitNodesPerTree = math.ceil(
      numSplitNodes.toDouble / numTrees.toDouble
    ).toInt

    // Keep track of split node child node Ids.
    // The split node Id would be the split Id.
    // The second map Id is the child index of the split (ordered).
    // The last value is the assigned 'real' node Id for that child.
    val splitNodeChildNodeIds =
      Array.fill[MapWithSequentialIntKeys[MapWithSequentialIntKeys[Int]]](numTrees)(
        new MapWithSequentialIntKeys[MapWithSequentialIntKeys[Int]](
          initCapacity = avgSplitNodesPerTree * 2
        )
      )

    // Find child nodes that are to be split further and update their Ids.
    cfor(0)(_ < numSplitNodes, _ + 1)(
      i => {
        val splitNode = nodeInfoArray(i)
        val treeId = splitNode.treeId
        val id = splitNode.nodeId
        if (idRanges(treeId) == null) {
          idRanges(treeId) = IdRange(id, id)
          updaters(treeId) = new MapWithSequentialIntKeys[IdUpdater](
            initCapacity = avgSplitNodesPerTree * 2
          )
        } else {
          idRanges(treeId).endId += 1
          if (id != idRanges(treeId).endId) {
            throw new AssertionError(
              "The expected split Id was " + idRanges(treeId).endId +
                " but we found " + id
            )
          }
        }

        // For this split node, we need another map to keep track of the
        // assigned child node Ids.
        splitNodeChildNodeIds(treeId).put(id,
          new MapWithSequentialIntKeys[Int](
            initCapacity = 4
          )
        )

        val splitInfo: NodeSplitInfo =
          if (splitNode.splitInfo.nonEmpty) {
            val si = splitNode.splitInfo.get
            // We have to find out whether the child node can be further split
            // or it's terminal.
            val children = si.getOrderedChildNodes
            val numChildren = children.length
            cfor(0)(_ < numChildren, _ + 1)(
              j => {
                val child = children(j)

                // This is equivalent to assigning 'real' node Ids to trees in
                // a breadth-first manner. Both split Ids and node Ids are
                // assigned in the same manner, except that the split Ids will
                // skip terminal children.
                splitNodeChildNodeIds(treeId).get(id).put(j, nextNodeIdPerTree(treeId))

                if (child.isTerminal(maxDepth, minSplitSize)) {
                  // If the child is terminal, the Id in the RDD would be
                  // updated to 0's first.
                  child.nodeId = 0
                } else {
                  // Otherwise, assign a new split Id so that the Id cache can
                  // be updated properly.
                  val splitId = nextSplitIdPerTree(treeId)
                  child.nodeId = splitId
                  splitQueue.enqueue((treeId, splitId, child.depth))
                  splitIdToNodeId(treeId).put(
                    splitId,
                    nextNodeIdPerTree(treeId)
                  )

                  nextSplitIdPerTree(treeId) += 1
                }

                // Now update the next assignable node Id.
                nextNodeIdPerTree(treeId) += 1
              }
            )
            si
          } else {
            // This means the node is terminal, and thus we have to update the
            // Id to 0 (accomplished by setting the updater to null) to indicate
            // terminalness.
            null
          }
        updaters(treeId).put(id, IdUpdater(splitInfo))
      }
    )
    val idLookupForUpdaters = IdLookupForUpdaters.createIdLookupForUpdaters(
      idRanges = idRanges,
      updaterMaps = updaters
    )
    // Update idCache.
    cfor(0)(_ < numRows, _ + 1)(
      i => {
        cfor(0)(_ < numTrees, _ + 1)(
          treeId => {
            val curId = idCache(i)(treeId)
            val baggedRow = data(i)
            val rowCnt = baggedRow._2(treeId).toInt
            if (rowCnt > 0 && curId >= 1) {
              val updater = idLookupForUpdaters.get(
                treeId = treeId,
                id = curId)
              if (updater != null) {
                val features = baggedRow._1._2
                idCache(i)(treeId) = updater.updateId(
                  features = features,
                  featureHandler = typeHandler
                )
              }
            }
          }
        )
      }
    )
    // Now update node Ids of split nodes to actual tree node Ids.
    cfor(0)(_ < numSplitNodes, _ + 1)(
      i => {
        val splitNode = nodeInfoArray(i)
        val treeId = splitNode.treeId
        val splitId = splitNode.nodeId
        // We have kept the split Id to node Id map.
        splitNode.nodeId = splitIdToNodeId(treeId).get(splitId)
        splitIdToNodeId(treeId).remove(splitId)
        if (splitNode.splitInfo.nonEmpty) {
          // Also, update child node ids to proper ones before we write this to
          // the store object.
          val si = splitNode.splitInfo.get
          val children = si.getOrderedChildNodes
          val numChildren = children.length
          cfor(0)(_ < numChildren, _ + 1)(
            j => {
              val child = children(j)
              if (child.isTerminal(maxDepth, minSplitSize)) {
                child.nodeId = splitNodeChildNodeIds(treeId).get(splitId).get(j)
              } else {
                // The actual node Id can be found in the split map.
                val childSplitId = child.nodeId
                child.nodeId = splitIdToNodeId(treeId).get(childSplitId)

                // For sanity check, also compare with the split node child node
                // Id map.
                assert(
                  child.nodeId == splitNodeChildNodeIds(treeId).get(splitId).get(j),
                  "Inconsistent child node Ids : " + child.nodeId + " and " +
                  splitNodeChildNodeIds(treeId).get(splitId).get(j)
                )
              }
            }
          )
        }
      }
    )
  }

  /*
   * updateIdsAndQueues_distributed is not needed for local datasets.
   */
  def updateIdsAndQueues_distributed(
    splitQueue: mutable.Queue[(Int, Int, Int)],
    subTreeQueue: mutable.Queue[(Int, Int, Int)],
    nodeInfoArray: Array[NodeInfo],
    nextSplitIdPerTree: Array[Int],
    nextSubTreeIdPerTree: Array[Int],
    nextNodeIdPerTree: Array[Int],
    splitIdToNodeId: Array[MapWithSequentialIntKeys[Int]],
    subTreeIdToNodeId: Array[MapWithSequentialIntKeys[Int]]): Unit = {
    throw new UnsupportedOperationException(
      "updateIdsAndQueues_distributed is not implemented."
    )
  }

  /*
   * trainSubTrees is not needed for local datasets.
   */
  def trainSubTrees(
    subTreeLookup: IdLookupForSubTreeInfo,
    nextNodeIdPerTree: Array[Int],
    subTreeIdToNodeId: Array[MapWithSequentialIntKeys[Int]]): Iterator[NodeInfo] = {
    throw new UnsupportedOperationException(
      "trainSubTrees is not implemented."
    )
  }
}
