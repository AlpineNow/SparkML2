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

import spark_ml.discretization.Bins
import spark_ml.util.{MapWithSequentialIntKeys, ProgressNotifiee, UnexpectedKeyException}
import spire.implicits._

/**
 * Split criteria to use for individual trees.
 */
object SplitCriteria extends Enumeration {
  type SplitCriteria = Value
  val Classification_InfoGain = Value(0)
  val Regression_Variance = Value(1)
}

/**
 * Categorical split type enumeration.
 */
object CatSplitType extends Enumeration {
  type CatSplitType = Value

  /**
   * Ordered binary split means that categorical values will be ordered
   * according to some label criteria and then a binary split will be performed
   * just like the numerical features. E.g., for regression, the categorical
   * values will be sorted by the average label values.
   */
  val OrderedBinarySplit = Value(0)

  /**
   * Random binary split means that the categorical values will be randomly
   * ordered and then a binary split will be performed.
   */
  val RandomBinarySplit = Value(1)

  /**
   * This means that categorical values will be split K-ways.
   */
  val MultiwaySplit = Value(2)
}

/**
 * Options to pass onto the forest trainer.
 * @param numTrees Number of independent trees to train.
 * @param splitCriteria Split criteria.
 * @param mtry Number of random features to analyze per node.
 *             Useful for random forest.
 * @param maxDepth The maximum depth of trees.
 * @param minSplitSize The minimum size of the node to be eligible for splits.
 * @param catSplitType Categorical feature split types. It can be binary or
 *                     multi-way splits.
 * @param maxSplitsPerIter Maximum number of splits to do per iteration.
 * @param subTreeWeightThreshold Sub-tree weight threshold. Only for distributed
 *                               training.
 * @param maxSubTreesPerIter Maximum number of sub-trees to train per iteration.
 * @param numClasses Optional number of target classes, iff the split type is a
 *                   classification type.
 * @param verbose If true, the algorithm will print as much information through
 *                the notifiee as possible, including many intermediate
 *                computation values, etc.
 */
case class TreeForestTrainerOptions(
  numTrees: Int,
  splitCriteria: SplitCriteria.SplitCriteria,
  mtry: Int,
  maxDepth: Int,
  minSplitSize: Int,
  catSplitType: CatSplitType.CatSplitType,
  maxSplitsPerIter: Int,
  subTreeWeightThreshold: Double,
  maxSubTreesPerIter: Int,
  numClasses: Option[Int],
  verbose: Boolean
)

/**
 * Tree forest trainer. I.e., this can be used to train multiple trees in
 * parallel, a la Random Forest.
 * It can also be used to train single trees for other meta algorithms such as
 * Gradient Boosting.
 */
object TreeForestTrainer {
  /**
   * Train tree(s).
   * Multiple tree training is equivalent to random forest training.
   * This assumes that the training data have already been bagged.
   * This is used by both distributed trainer as well as local trainers.
   * @param trainingData The training data that have been processed specifically
   *                     for tree training, such as feature binning, bagging, etc.
   * @param featureBinsInfo Feature discretization info.
   * @param trainingOptions Training options.
   * @param modelStore The trainer doesn't have any explicit idea about tree
   *                   representations. It's the caller's responsibility to use
   *                   a proper storage model. Additionally, it's the model
   *                   store's responsibility to use the proper tree
   *                   representations.
   * @param notifiee Progress notifiee.
   * @param rng Random number generator.
   */
  def train(
    trainingData: QuantizedData_ForTrees,
    featureBinsInfo: Array[Bins],
    trainingOptions: TreeForestTrainerOptions,
    modelStore: TreeEnsembleStore,
    notifiee: ProgressNotifiee,
    rng: Random): Unit = {

    // Extract all the options from the options case class.
    // This is needed to prevent serializing the entire options object.
    val numTrees: Int = trainingOptions.numTrees
    val splitCriteria: SplitCriteria.SplitCriteria = trainingOptions.splitCriteria
    val mtry: Int = trainingOptions.mtry
    val maxDepth: Int = trainingOptions.maxDepth
    val minSplitSize: Int = trainingOptions.minSplitSize
    val catSplitType: CatSplitType.CatSplitType = trainingOptions.catSplitType
    val maxSplitsPerIter: Int = trainingOptions.maxSplitsPerIter
    val subTreeWeightThreshold: Double = trainingOptions.subTreeWeightThreshold
    val maxSubTreesPerIter: Int = trainingOptions.maxSubTreesPerIter
    val numClasses: Option[Int] = trainingOptions.numClasses

    // Write the option values to the notifiee.
    notifiee.newStatusMessage("===================================================")
    notifiee.newStatusMessage("Training a tree ensemble with the following options")
    notifiee.newStatusMessage("===================================================")
    notifiee.newStatusMessage("numTrees                   : " + numTrees)
    notifiee.newStatusMessage("splitCriteria              : " + splitCriteria.toString)
    notifiee.newStatusMessage("mtry                       : " + mtry)
    notifiee.newStatusMessage("maxDepth                   : " + maxDepth)
    notifiee.newStatusMessage("minSplitSize               : " + minSplitSize)
    notifiee.newStatusMessage("catSplitType               : " + catSplitType.toString)
    notifiee.newStatusMessage("maxSplitsPerIter           : " + maxSplitsPerIter)
    notifiee.newStatusMessage("subTreeWeightThreshold     : " + subTreeWeightThreshold)
    notifiee.newStatusMessage("maxSubTreesPerIter         : " + maxSubTreesPerIter)
    if (numClasses.nonEmpty) {
      notifiee.newStatusMessage("numClasses                 : " + numClasses.get)
    } else {
      notifiee.newStatusMessage("numClasses                 : None (Regression)")
    }

    if (trainingData.isLocal) {
      notifiee.newStatusMessage("Training the forest in a single machine.")
    } else {
      notifiee.newStatusMessage("Training the forest in a distributed fashion.")
    }

    val modelWriter: TreeEnsembleWriter = modelStore.getWriter

    // Generate random seeds for trees.
    // Used for synchronize random feature selections per partition.
    val treeSeeds = Array.fill[Int](numTrees)(0)
    cfor(0)(_ < numTrees, _ + 1)(
      treeId => {
        treeSeeds(treeId) = rng.nextInt()
      }
    )

    // The options should be also set in the quantized training data object.
    // This is because actual split methods are members of the training data.
    trainingData.setOptions(
      options = trainingOptions,
      treeSeeds = treeSeeds
    )

    // We keep track of separate Ids for tree nodes and rows to be split.
    // The reason we have a separation between node Id and split Id is
    // because we want to keep a compact and incrementing Ids for splits
    // for efficiency.

    // Set up arrays of node Ids to assign for the next tree nodes.
    // Since the root has the node Id '1', the next ones will be '2'.
    // The index is used to point to the corresponding tree.
    val nextNodeIdPerTree: Array[Int] = Array.fill[Int](numTrees)(2)

    // Set up arrays of split Ids to assign to each row.
    // A split Id belongs to a node that is eligible for splits.
    // This increments independently from nextNodeIdPerTree,
    // as it will only be assigned to nodes that could be split,
    // and not terminal nodes.
    val nextSplitIdPerTree: Array[Int] = Array.fill[Int](numTrees)(2)

    // Set up arrays of sub tree Ids to assign to each row.
    // A sub-tree belongs to a node.
    // This is only used in distributed training to call sub-tree training
    // with a particular tree Id.
    val nextSubTreeIdPerTree: Array[Int] = Array.fill[Int](numTrees)(1)

    // How many splits per iteration we are likely to do per tree.
    val avgMaxSplitsPerTree = math.ceil(
      maxSplitsPerIter.toDouble / numTrees.toDouble
    ).toInt

    // We need to maintain a map from split Id to node Ids.
    val splitIdToNodeId =
      Array.fill[MapWithSequentialIntKeys[Int]](numTrees)(
        new MapWithSequentialIntKeys[Int](
          initCapacity = avgMaxSplitsPerTree * 10
        )
      )

    // Set up the split queue.
    // Splits will be processed (either split or trained as a sub-tree) FIFO.
    // In the triple, the first element is the treeId.
    // The second element is the split Id (corresponding to a node).
    // The last element is the node depth of the node to be split.
    val splitQueue: mutable.Queue[(Int, Int, Int)] = new mutable.Queue[(Int, Int, Int)]()
    cfor(0)(_ < numTrees, _ + 1)(
      treeId => {
        splitQueue.enqueue((treeId, 1, 1))
        splitIdToNodeId(treeId).put(1, 1)
      }
    )

    // The following function is used to dequeue splits and get a corresponding
    // aggregator lookup object.
    def dequeueSplits(
      splitQueue: mutable.Queue[(Int, Int, Int)],
      maxSplits: Int): (IdLookupForNodeStats, Int) = {
      val idRanges = Array.fill[IdRange](numTrees)(null)

      // A map from split Id to the node depth.
      val nodeDepths =
        Array.fill[MapWithSequentialIntKeys[Int]](numTrees)(null)
      var splitCnt = 0
      while (splitCnt < maxSplits && splitQueue.nonEmpty) {
        val splitInfo = splitQueue.dequeue()
        splitCnt += 1
        val treeId = splitInfo._1
        val splitId = splitInfo._2
        val nodeDepth = splitInfo._3
        if (idRanges(treeId) == null) {
          idRanges(treeId) = IdRange(splitId, splitId)
          nodeDepths(treeId) = new MapWithSequentialIntKeys[Int](
            initCapacity = avgMaxSplitsPerTree * 2
          )
        } else {
          idRanges(treeId).endId += 1
          if (idRanges(treeId).endId != splitId) {
            // Something is wrong in this case.
            throw UnexpectedKeyException(
              "The split Ids contained in the split queue are not " +
                "in an expected order. For the tree " + treeId +
                ", a split Id " + idRanges(treeId).endId +
                " was expected but " + splitId + " was found."
            )
          }
        }

        nodeDepths(treeId).put(splitId, nodeDepth)
      }

      // Create a IdLookup object for node stat aggregators.
      (
        new IdLookupForNodeStats(
          idRanges = idRanges,
          nodeDepths = nodeDepths
        ),
        splitCnt
      )
    }

    // The following function is used to dequeue subtrees to train and get a
    // corresponding sub-tree lookup object.
    def dequeueSubTrees(
      subTreeQueue: mutable.Queue[(Int, Int, Int)],
      maxSubTrees: Int): (IdLookupForSubTreeInfo, Int) = {
      val idRanges = Array.fill[IdRange](numTrees)(null)
      val subTreeInfoMaps =
        Array.fill[MapWithSequentialIntKeys[SubTreeInfo]](numTrees)(
          null
        )
      var subTreeCnt = 0
      while (subTreeCnt < maxSubTrees && subTreeQueue.nonEmpty) {
        val subTreeInfo = subTreeQueue.dequeue()
        val treeId = subTreeInfo._1
        val subTreeId = subTreeInfo._2
        val subTreeDepth = subTreeInfo._3
        if (idRanges(treeId) == null) {
          idRanges(treeId) = IdRange(subTreeId, subTreeId)
          subTreeInfoMaps(treeId) =
            new MapWithSequentialIntKeys[SubTreeInfo](
              initCapacity = avgMaxSplitsPerTree * 2
            )
        } else {
          idRanges(treeId).endId += 1
          if (idRanges(treeId).endId != subTreeId) {
            // Something is wrong in this case.
            throw UnexpectedKeyException(
              "The subtree Ids contained in the subtree queue are not " +
                "in an expected order. For the tree " + treeId +
                ", a subtree Id " + idRanges(treeId).endId +
                " was expected but " + subTreeId + " was found."
            )
          }
        }

        subTreeInfoMaps(treeId).put(
          subTreeId,
          SubTreeInfo(
            id = subTreeId,
            hash = subTreeCnt,
            depth = subTreeDepth,
            parentTreeId = treeId
          )
        )

        subTreeCnt += 1
      }

      (
        IdLookupForSubTreeInfo.createIdLookupForSubTreeInfo(
          idRanges = idRanges,
          subTreeMaps = subTreeInfoMaps
        ),
        subTreeCnt
      )
    }

    // Keep track of the number of iterations.
    // Additionally, we'll run as long as there are nodes to be processed in the
    // split queue.
    var iter = 0
    while (splitQueue.nonEmpty) {
      notifiee.newStatusMessage("=============================")
      notifiee.newStatusMessage("Starting the iteration " + (iter + 1))
      notifiee.newStatusMessage("=============================")

      // Deque the splits and get a corresponding node stat aggregator lookup
      // object.
      val (lookup, splitCnt) = dequeueSplits(splitQueue, maxSplitsPerIter)
      notifiee.newStatusMessage(
        "The lookup object contains " + splitCnt + " splits to do."
      )

      // Split nodes.
      // Either of the following is possible:
      // * Distributed splits (RDD).
      // * Local splits. (for use within local sub-tree training).
      // The output array should be ordered by split Ids.
      notifiee.newStatusMessage(
        "Aggregating node statistics and computing optimal splits."
      )
      val nodeInfoArray = trainingData.aggregateAndSplit(lookup)

      // Making sure that the nodeInfoArray contains the expected number of
      // nodes.
      if (nodeInfoArray.length != splitCnt) {
        throw new AssertionError(
          "The expected number of NodeInfo objects was " + splitCnt +
            ". But we only have " + nodeInfoArray.length + " objects."
        )
      }

      // We need to keep track of sub-tree stuffs for distributed training.
      val subTreeQueue: Option[mutable.Queue[(Int, Int, Int)]] =
        if (trainingData.isLocal) {
          None
        } else {
          Some(new mutable.Queue[(Int, Int, Int)]())
        }
      val subTreeIdToNodeId: Option[Array[MapWithSequentialIntKeys[Int]]] =
        if (trainingData.isLocal) {
          None
        } else {
          Some(
            Array.fill[MapWithSequentialIntKeys[Int]](numTrees)(
              new MapWithSequentialIntKeys[Int](
                initCapacity = avgMaxSplitsPerTree * 2
              )
            )
          )
        }

      notifiee.newStatusMessage(
        "Assigning split/node Ids and queueing subsequent splits to perform."
      )
      if (trainingData.isLocal) {
        trainingData.updateIdsAndQueues_local(
          splitQueue = splitQueue,
          nodeInfoArray = nodeInfoArray,
          nextSplitIdPerTree = nextSplitIdPerTree,
          nextNodeIdPerTree = nextNodeIdPerTree,
          splitIdToNodeId = splitIdToNodeId
        )
      } else {
        trainingData.updateIdsAndQueues_distributed(
          splitQueue = splitQueue,
          subTreeQueue = subTreeQueue.get,
          nodeInfoArray = nodeInfoArray,
          nextSplitIdPerTree = nextSplitIdPerTree,
          nextSubTreeIdPerTree = nextSubTreeIdPerTree,
          nextNodeIdPerTree = nextNodeIdPerTree,
          splitIdToNodeId = splitIdToNodeId,
          subTreeIdToNodeId = subTreeIdToNodeId.get
        )
      }

      // Iterate through nodeInfoArray and update the model.
      notifiee.newStatusMessage("Adding trained nodes to the model...")
      cfor(0)(_ < splitCnt, _ + 1)(
        i => {
          // By now, nodeInfo should contain the actual node Ids, instead of
          // temporary internal split Ids.
          val nodeInfo = nodeInfoArray(i)
          modelWriter.writeNodeInfo(nodeInfo)

          // Also write the child nodes if they are terminal.
          // These child nodes should also contain correct Ids.
          if (nodeInfo.splitInfo.nonEmpty) {
            val childNodes: Array[NodeInfo] =
              nodeInfo.splitInfo.get.getOrderedChildNodes
            cfor(0)(_ < childNodes.length, _ + 1)(
              j => {
                val childNode = childNodes(j)
                if (childNode.isTerminal(maxDepth, minSplitSize)) {
                  modelWriter.writeNodeInfo(childNode)
                }
              }
            )
          }
        }
      )

      // Train sub-trees if this is distributed and the queue is not empty.
      if (!trainingData.isLocal) {
        notifiee.newStatusMessage(
          "There are total " + subTreeQueue.get.size + " sub-trees " +
            "to train."
        )
        while (subTreeQueue.get.nonEmpty) {
          val (subTreeLookup, subTreeCnt) =
            dequeueSubTrees(subTreeQueue.get, maxSubTreesPerIter)
          notifiee.newStatusMessage(
            "Training " + subTreeCnt + " sub-trees..."
          )
          val subTreeNodeItr = trainingData.trainSubTrees(
            subTreeLookup = subTreeLookup,
            nextNodeIdPerTree = nextNodeIdPerTree,
            subTreeIdToNodeId = subTreeIdToNodeId.get
          )
          // Write the subtree nodes to the model.
          notifiee.newStatusMessage("Adding subtree nodes to the model...")
          while (subTreeNodeItr.hasNext) {
            val nodeInfo = subTreeNodeItr.next()
            modelWriter.writeNodeInfo(nodeInfo)
          }
        }
      }

      iter += 1
      notifiee.newStatusMessage("Finished " + iter + " iterations.")
    }

    notifiee.newStatusMessage("Finished TreeEnsemble::train.")
  }
}
