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

import spark_ml.util._
import org.scalatest.FunSuite
import spark_ml.discretization._
import spark_ml.discretization.NumericBins
import spark_ml.discretization.NumericBin

/**
 * Test various components of Sequoia Forest.
 */
class SequoiaForestComponentSuite extends FunSuite with LocalSparkContext {
  test("Test scheduled node split lookup") {
    // Let's test a failing case.
    var numTrees = 13
    var nodeSplitsPerTree: Array[mutable.Queue[NodeSplitOnBinId]] = Array.fill[mutable.Queue[NodeSplitOnBinId]](numTrees)(mutable.Queue[NodeSplitOnBinId]())

    var treeId = 0
    while (treeId < numTrees) {
      nodeSplitsPerTree(treeId).enqueue(NumericSplitOnBinId(2, 3, 5, 13, 14, 2.0, 2.0), NumericSplitOnBinId(1, 2, 1, 10, 11, 3.0, 2.0))
      treeId += 1
    }

    // This should throw an exception.
    var exceptionThrown = false
    try {
      ScheduledNodeSplitLookup.createLookupForNodeSplits(nodeSplitsPerTree, 100)
    } catch {
      case e: AssertionError => exceptionThrown = true
    }

    assert(exceptionThrown)

    // Now test that it works as expected when it's properly constructed.
    numTrees = 2
    nodeSplitsPerTree = Array.fill[mutable.Queue[NodeSplitOnBinId]](numTrees)(mutable.Queue[NodeSplitOnBinId]())

    nodeSplitsPerTree(0).enqueue(NumericSplitOnBinId(2, 3, 5, 13, 14, 2.0, 2.0), NumericSplitOnBinId(5, 1, 2, 15, 16, 3.0, 6.0))
    nodeSplitsPerTree(1).enqueue(NumericSplitOnBinId(3, 4, 2, 7, 8, 2.0, 5.0), CategoricalSplitOnBinId(4, 3, mutable.Map[Int, Int](1 -> 10, 4 -> 11), mutable.Map[Int, Double](10 -> 3.0, 11 -> 2.0)))

    val scheduledLookup = ScheduledNodeSplitLookup.createLookupForNodeSplits(nodeSplitsPerTree, 100)
    assert(scheduledLookup.numTrees === numTrees)

    assert(scheduledLookup.parentNodeLookup.length == numTrees)
    assert(scheduledLookup.parentNodeLookup(0)(0) === 2)
    assert(scheduledLookup.parentNodeLookup(0)(1) === 4)
    assert(scheduledLookup.parentNodeLookup(1)(0) === 3)
    assert(scheduledLookup.parentNodeLookup(1)(1) === 2)

    assert(scheduledLookup.nodeSplitTable.length === numTrees)

    assert(scheduledLookup.nodeSplitTable(0).length === 4)

    val split1 = scheduledLookup.nodeSplitTable(0)(0)
    assert(split1.isInstanceOf[NumericSplitOnBinId])
    assert(split1.asInstanceOf[NumericSplitOnBinId].parentNodeId === 2)
    assert(split1.asInstanceOf[NumericSplitOnBinId].featureId === 3)
    assert(split1.asInstanceOf[NumericSplitOnBinId].splitBinId === 5)
    assert(split1.asInstanceOf[NumericSplitOnBinId].leftId === 13)
    assert(split1.asInstanceOf[NumericSplitOnBinId].rightId === 14)
    assert(split1.asInstanceOf[NumericSplitOnBinId].leftWeight === 2.0)
    assert(split1.asInstanceOf[NumericSplitOnBinId].rightWeight === 2.0)

    assert(scheduledLookup.nodeSplitTable(0)(1) === null)
    assert(scheduledLookup.nodeSplitTable(0)(2) === null)

    val split2 = scheduledLookup.nodeSplitTable(0)(3)
    assert(split2.isInstanceOf[NumericSplitOnBinId])
    assert(split2.asInstanceOf[NumericSplitOnBinId].parentNodeId === 5)
    assert(split2.asInstanceOf[NumericSplitOnBinId].featureId === 1)
    assert(split2.asInstanceOf[NumericSplitOnBinId].splitBinId === 2)
    assert(split2.asInstanceOf[NumericSplitOnBinId].leftId === 15)
    assert(split2.asInstanceOf[NumericSplitOnBinId].rightId === 16)
    assert(split2.asInstanceOf[NumericSplitOnBinId].leftWeight === 3.0)
    assert(split2.asInstanceOf[NumericSplitOnBinId].rightWeight === 6.0)

    assert(scheduledLookup.nodeSplitTable(1).length === 2)

    val split3 = scheduledLookup.nodeSplitTable(1)(0)
    assert(split3.isInstanceOf[NumericSplitOnBinId])
    assert(split3.asInstanceOf[NumericSplitOnBinId].parentNodeId === 3)
    assert(split3.asInstanceOf[NumericSplitOnBinId].featureId === 4)
    assert(split3.asInstanceOf[NumericSplitOnBinId].splitBinId === 2)
    assert(split3.asInstanceOf[NumericSplitOnBinId].leftId === 7)
    assert(split3.asInstanceOf[NumericSplitOnBinId].rightId === 8)
    assert(split3.asInstanceOf[NumericSplitOnBinId].leftWeight === 2.0)
    assert(split3.asInstanceOf[NumericSplitOnBinId].rightWeight === 5.0)

    val split4 = scheduledLookup.nodeSplitTable(1)(1)
    assert(split4.isInstanceOf[CategoricalSplitOnBinId])
    assert(split4.asInstanceOf[CategoricalSplitOnBinId].parentNodeId === 4)
    assert(split4.asInstanceOf[CategoricalSplitOnBinId].featureId === 3)
    assert(split4.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap.size === 2)
    assert(split4.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(1) === 10)
    assert(split4.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(4) === 11)
    assert(split4.asInstanceOf[CategoricalSplitOnBinId].nodeWeights.size === 2)
    assert(split4.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(10) === 3.0)
    assert(split4.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(11) === 2.0)

    val split5 = scheduledLookup.getNodeSplit(0, 2)
    assert(split5.isInstanceOf[NumericSplitOnBinId])
    assert(scheduledLookup.getNodeSplit(0, 3) === null)
    assert(scheduledLookup.getNodeSplit(0, 4) === null)
    assert(scheduledLookup.getNodeSplit(0, 6) === null)
    assert(scheduledLookup.getNodeSplit(0, -1) === null)

    val split6 = scheduledLookup.getNodeSplit(0, 5)
    assert(split6.isInstanceOf[NumericSplitOnBinId])

    val split7 = scheduledLookup.getNodeSplit(1, 3)
    assert(split7.isInstanceOf[NumericSplitOnBinId])

    val split8 = scheduledLookup.getNodeSplit(1, 4)
    assert(split8.isInstanceOf[CategoricalSplitOnBinId])

    assert(scheduledLookup.getNodeSplit(1, 2) === null)
    assert(scheduledLookup.getNodeSplit(1, 5) === null)
  }

  test("Test InfoGainStatistics") {
    val numTrees = 2
    val nodeSplitsPerTree = Array.fill[mutable.Queue[NodeSplitOnBinId]](numTrees)(mutable.Queue[NodeSplitOnBinId]())

    nodeSplitsPerTree(0).enqueue(NumericSplitOnBinId(2, 3, 5, 13, 14, 2.0, 2.0, -1, -1), NumericSplitOnBinId(5, 1, 2, 15, 16, 3.0, 6.0, 0, -1))
    nodeSplitsPerTree(1).enqueue(NumericSplitOnBinId(3, 4, 2, 7, 8, 2.0, 5.0, -1, -1), CategoricalSplitOnBinId(4, 3, mutable.Map[Int, Int](1 -> 10, 4 -> 11), mutable.Map[Int, Double](10 -> 3.0, 11 -> 2.0), mutable.Map[Int, Int]()))

    val nodeDepths = Array.fill[mutable.Map[Int, Int]](2)(mutable.Map[Int, Int]())
    nodeDepths(0).put(13, 5)
    nodeDepths(0).put(14, 5)
    nodeDepths(0).put(16, 7)
    nodeDepths(1).put(7, 3)
    nodeDepths(1).put(8, 3)
    nodeDepths(1).put(10, 6)
    nodeDepths(1).put(11, 6)

    val scheduledLookup = ScheduledNodeSplitLookup.createLookupForNodeSplits(nodeSplitsPerTree, 100)
    val treeSeeds = new Array[Int](numTrees)
    var treeId = 0
    val randGen = scala.util.Random
    while (treeId < numTrees) {
      treeSeeds(treeId) = randGen.nextInt()
      treeId += 1
    }

    val mtry = 4
    val numClasses = 5
    val numBinsPerFeature = Array[Int](10, 11, 17, 13, 15, 9)

    val featureBins = new Array[Bins](6)
    var featId = 0
    while (featId < numBinsPerFeature.length) {
      if (featId != 1) {
        val bins = mutable.ArrayBuffer[NumericBin]()
        bins += NumericBin(Double.NegativeInfinity, 0)
        while (bins.length < numBinsPerFeature(featId)) {
          bins += NumericBin(bins.length - 1, bins.length)
        }

        featureBins(featId) = NumericBins(bins.toArray)
      } else {
        featureBins(featId) = CategoricalBins(numBinsPerFeature(featId))
      }

      featId += 1
    }

    // Create an information gain statistics object.
    val infoGainStats = new InfoGainStatistics(
      scheduledLookup,
      numBinsPerFeature,
      treeSeeds,
      mtry,
      numClasses)

    assert(infoGainStats.numTrees === 2)
    assert(infoGainStats.numFeatures === 6)
    assert(infoGainStats.numSelectedFeaturesPerNode === 4)

    assert(infoGainStats.startNodeIds(0) === 13)
    assert(infoGainStats.startNodeIds(1) === 7)
    assert(infoGainStats.numNodes === 7)

    assert(infoGainStats.selectedFeaturesLookup.length === 2)
    assert(infoGainStats.selectedFeaturesLookup(0).length === 4)
    assert(infoGainStats.selectedFeaturesLookup(1).length === 5)

    assert(infoGainStats.offsetLookup.length === 2)
    assert(infoGainStats.offsetLookup(0).length === 4)
    assert(infoGainStats.offsetLookup(1).length === 5)

    var expectedBinStatsLength = 0

    treeId = 0
    while (treeId < numTrees) {
      var nodeIdx = 0
      while (nodeIdx < infoGainStats.selectedFeaturesLookup(treeId).length) {
        if ((treeId == 0 && nodeIdx == 2) || (treeId == 1 && nodeIdx == 2)) {
          assert(infoGainStats.selectedFeaturesLookup(treeId)(nodeIdx).length === 0)
          assert(infoGainStats.offsetLookup(treeId)(nodeIdx).length === 0)
        } else {
          assert(infoGainStats.selectedFeaturesLookup(treeId)(nodeIdx).length === mtry)
          assert(infoGainStats.offsetLookup(treeId)(nodeIdx).length === mtry)

          var featureIdx = 0
          while (featureIdx < mtry) {
            val featureId = infoGainStats.selectedFeaturesLookup(treeId)(nodeIdx)(featureIdx)
            val numBins = numBinsPerFeature(featureId)
            assert(infoGainStats.offsetLookup(treeId)(nodeIdx)(featureIdx) === expectedBinStatsLength)
            expectedBinStatsLength += numBins * numClasses
            featureIdx += 1
          }
        }

        nodeIdx += 1
      }

      treeId += 1
    }

    assert(infoGainStats.binStatsArray.asInstanceOf[ClassificationStatisticsArray].binStats.length === expectedBinStatsLength)

    // Create an information gain statistics object with all the features per node.
    val infoGainStats2 = new InfoGainStatistics(
      scheduledLookup,
      numBinsPerFeature,
      treeSeeds,
      numBinsPerFeature.length,
      numClasses)

    // Add some samples and make sure that we get the right statistics accumulated.
    infoGainStats2.addUnsignedByteSample(
      0,
      13,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(4), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(15), Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5)), Array[Byte](2, 3)))

    infoGainStats2.addUnsignedByteSample(
      0,
      14,
      (2.0, Array[Byte](Discretizer.convertToUnsignedByte(4), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(15), Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5)), Array[Byte](2, 3)))

    infoGainStats2.addUnsignedByteSample(
      0,
      16,
      (4.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(11), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(6), Discretizer.convertToUnsignedByte(7)), Array[Byte](5, 1)))

    infoGainStats2.addUnsignedByteSample(
      0,
      16,
      (4.0, Array[Byte](Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(4)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      8,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(8)), Array[Byte](4, 2)))

    infoGainStats2.addUnsignedByteSample(
      1,
      7,
      (3.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      10,
      (3.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      10,
      (4.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      10,
      (3.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      11,
      (3.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      11,
      (2.0, Array[Byte](Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 2)))

    assert(infoGainStats2.getBinLabelWeight(0, 14, 0, 5, 2) === 0)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 0, 4, 2) === 2)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 1, 4, 2) === 0)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 1, 1, 2) === 2)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 2, 4, 2) === 0)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 2, 15, 2) === 2)
    assert(infoGainStats2.getBinLabelWeight(0, 16, 1, 4, 2) === 0)
    assert(infoGainStats2.getBinLabelWeight(0, 16, 1, 7, 4) === 8)

    assert(infoGainStats2.getBinLabelWeight(1, 8, 3, 7, 4) === 0)
    assert(infoGainStats2.getBinLabelWeight(1, 8, 3, 7, 1) === 2)

    val nextNodeIdsPerTree = Array[Int](21, 23)

    // Now, get the splits.
    val splits = infoGainStats2.computeNodePredictionsAndSplits(
      featureBins,
      nextNodeIdsPerTree,
      nodeDepths,
      SequoiaForestOptions(
        numTrees = 2,
        treeType = TreeType.Classification_InfoGain,
        mtry = 6,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 100,
        localTrainThreshold = 100000,
        numSubTreesPerIteration = 3,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(numClasses),
        imputationType = ImputationType.SplitOnMissing),
      new scala.util.Random(17))

    assert(nodeDepths(0).size === 0)
    assert(nodeDepths(1).size === 4) // There should be 4 child nodes (due to all binary splits).

    val split1 = splits.next()
    assert(split1.prediction === 1.0)
    assert(split1.nodeId === 13)
    assert(split1.impurity === 0.0)
    assert(split1.treeId === 0)
    assert(split1.splitImpurity === None)
    assert(split1.nodeSplit === None)

    val split2 = splits.next()
    assert(split2.prediction === 2.0)
    assert(split2.nodeId === 14)
    assert(split2.impurity === 0.0)
    assert(split2.treeId === 0)
    assert(split2.splitImpurity === None)
    assert(split2.nodeSplit === None)

    val split3 = splits.next()
    assert(split3.prediction === 4.0)
    assert(split3.nodeId === 16)
    assert(split3.impurity === 0.0)
    assert(split3.treeId === 0)
    assert(split3.splitImpurity === None)
    assert(split3.nodeSplit === None)

    val split4 = splits.next()
    assert(split4.prediction === 3.0)
    assert(split4.nodeId === 7)
    assert(split4.impurity === 0.0)
    assert(split4.treeId === 1)
    assert(split4.splitImpurity === None)
    assert(split4.nodeSplit === None)

    val split5 = splits.next()
    assert(split5.prediction === 1.0)
    assert(split5.nodeId === 8)
    assert(split5.impurity === 0.0)
    assert(split5.treeId === 1)
    assert(split5.splitImpurity === None)
    assert(split5.nodeSplit === None)

    val split6 = splits.next()
    assert(split6.prediction === 3.0)
    assert(split6.nodeId === 10)
    assert(compareDouble(split6.impurity, 0.9183))
    assert(split6.treeId === 1)
    assert(compareDouble(split6.splitImpurity.get, 0.6666))
    assert(split6.nodeSplit.get.isInstanceOf[CategoricalSplitOnBinId])
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].parentNodeId === 10)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].featureId === 1)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap.size === 3)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(1) === 24)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(3) === 24)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(5) === 23)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights.size === 2)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(23) === 5)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(24) === 10)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeSubTreeHash.size === 0)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(23) === -1)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(24) === -1)

    val split7 = splits.next()
    assert(split7.prediction === 3.0)
    assert(split7.nodeId === 11)
    assert(compareDouble(split7.impurity, 0.8631))
    assert(split7.treeId === 1)
    assert(split7.splitImpurity.get === 0.0)
    assert(split7.nodeSplit.get.isInstanceOf[NumericSplitOnBinId])
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].parentNodeId === 11)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].featureId === 0)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].splitBinId === 3)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftId === 25)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightId === 26)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftWeight === 5)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightWeight === 2)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftSubTreeHash === -1)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightSubTreeHash === -1)
  }

  test("Test InfoGainStatistics with Unsigned Short Features") {
    val numTrees = 2
    val nodeSplitsPerTree = Array.fill[mutable.Queue[NodeSplitOnBinId]](numTrees)(mutable.Queue[NodeSplitOnBinId]())

    nodeSplitsPerTree(0).enqueue(NumericSplitOnBinId(2, 3, 5, 13, 14, 2.0, 2.0, -1, -1), NumericSplitOnBinId(5, 1, 2, 15, 16, 3.0, 6.0, 0, -1))
    nodeSplitsPerTree(1).enqueue(NumericSplitOnBinId(3, 4, 2, 7, 8, 2.0, 5.0, -1, -1), CategoricalSplitOnBinId(4, 3, mutable.Map[Int, Int](1 -> 10, 4 -> 11), mutable.Map[Int, Double](10 -> 3.0, 11 -> 2.0), mutable.Map[Int, Int]()))

    val nodeDepths = Array.fill[mutable.Map[Int, Int]](2)(mutable.Map[Int, Int]())
    nodeDepths(0).put(13, 5)
    nodeDepths(0).put(14, 5)
    nodeDepths(0).put(16, 7)
    nodeDepths(1).put(7, 3)
    nodeDepths(1).put(8, 3)
    nodeDepths(1).put(10, 6)
    nodeDepths(1).put(11, 6)

    val scheduledLookup = ScheduledNodeSplitLookup.createLookupForNodeSplits(nodeSplitsPerTree, 100)
    val treeSeeds = new Array[Int](numTrees)
    var treeId = 0
    val randGen = scala.util.Random
    while (treeId < numTrees) {
      treeSeeds(treeId) = randGen.nextInt()
      treeId += 1
    }

    val numClasses = 5
    val numBinsPerFeature = Array[Int](10, 11, 17, 13, 15, 9)

    val featureBins = new Array[Bins](6)
    var featId = 0
    while (featId < numBinsPerFeature.length) {
      if (featId != 1) {
        val bins = mutable.ArrayBuffer[NumericBin]()
        bins += NumericBin(Double.NegativeInfinity, 0)
        while (bins.length < numBinsPerFeature(featId)) {
          bins += NumericBin(bins.length - 1, bins.length)
        }

        featureBins(featId) = NumericBins(bins.toArray)
      } else {
        featureBins(featId) = CategoricalBins(numBinsPerFeature(featId))
      }

      featId += 1
    }

    // Create an information gain statistics object.
    val infoGainStats = new InfoGainStatistics(
      scheduledLookup,
      numBinsPerFeature,
      treeSeeds,
      numBinsPerFeature.length,
      numClasses)

    // Add some samples and make sure that we get the right statistics accumulated.
    infoGainStats.addUnsignedShortSample(
      0,
      13,
      (1.0, Array[Short](Discretizer.convertToUnsignedShort(4), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(15), Discretizer.convertToUnsignedShort(10), Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(5)), Array[Byte](2, 3)))

    infoGainStats.addUnsignedShortSample(
      0,
      14,
      (2.0, Array[Short](Discretizer.convertToUnsignedShort(4), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(15), Discretizer.convertToUnsignedShort(10), Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(5)), Array[Byte](2, 3)))

    infoGainStats.addUnsignedShortSample(
      0,
      16,
      (4.0, Array[Short](Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(11), Discretizer.convertToUnsignedShort(5), Discretizer.convertToUnsignedShort(6), Discretizer.convertToUnsignedShort(7)), Array[Byte](5, 1)))

    infoGainStats.addUnsignedShortSample(
      0,
      16,
      (4.0, Array[Short](Discretizer.convertToUnsignedShort(5), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(4)), Array[Byte](3, 5)))

    infoGainStats.addUnsignedShortSample(
      1,
      8,
      (1.0, Array[Short](Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(5), Discretizer.convertToUnsignedShort(8)), Array[Byte](4, 2)))

    infoGainStats.addUnsignedShortSample(
      1,
      7,
      (3.0, Array[Short](Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(5), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(2)), Array[Byte](3, 5)))

    infoGainStats.addUnsignedShortSample(
      1,
      10,
      (3.0, Array[Short](Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(5), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(2)), Array[Byte](3, 5)))

    infoGainStats.addUnsignedShortSample(
      1,
      10,
      (4.0, Array[Short](Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(2)), Array[Byte](3, 5)))

    infoGainStats.addUnsignedShortSample(
      1,
      10,
      (3.0, Array[Short](Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(2)), Array[Byte](3, 5)))

    infoGainStats.addUnsignedShortSample(
      1,
      11,
      (3.0, Array[Short](Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(5), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(2)), Array[Byte](3, 5)))

    infoGainStats.addUnsignedShortSample(
      1,
      11,
      (2.0, Array[Short](Discretizer.convertToUnsignedShort(5), Discretizer.convertToUnsignedShort(5), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(2)), Array[Byte](3, 2)))

    assert(infoGainStats.getBinLabelWeight(0, 14, 0, 5, 2) === 0)
    assert(infoGainStats.getBinLabelWeight(0, 14, 0, 4, 2) === 2)
    assert(infoGainStats.getBinLabelWeight(0, 14, 1, 4, 2) === 0)
    assert(infoGainStats.getBinLabelWeight(0, 14, 1, 1, 2) === 2)
    assert(infoGainStats.getBinLabelWeight(0, 14, 2, 4, 2) === 0)
    assert(infoGainStats.getBinLabelWeight(0, 14, 2, 15, 2) === 2)
    assert(infoGainStats.getBinLabelWeight(0, 16, 1, 4, 2) === 0)
    assert(infoGainStats.getBinLabelWeight(0, 16, 1, 7, 4) === 8)

    assert(infoGainStats.getBinLabelWeight(1, 8, 3, 7, 4) === 0)
    assert(infoGainStats.getBinLabelWeight(1, 8, 3, 7, 1) === 2)

    val nextNodeIdsPerTree = Array[Int](21, 23)

    // Now, get the splits.
    val splits = infoGainStats.computeNodePredictionsAndSplits(
      featureBins,
      nextNodeIdsPerTree,
      nodeDepths,
      SequoiaForestOptions(
        numTrees = 2,
        treeType = TreeType.Classification_InfoGain,
        mtry = 6,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 100,
        localTrainThreshold = 100000,
        numSubTreesPerIteration = 3,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(numClasses),
        imputationType = ImputationType.SplitOnMissing),
      new scala.util.Random(17))

    assert(nodeDepths(0).size === 0)
    assert(nodeDepths(1).size === 4)

    val split1 = splits.next()
    assert(split1.prediction === 1.0)
    assert(split1.nodeId === 13)
    assert(split1.impurity === 0.0)
    assert(split1.treeId === 0)
    assert(split1.splitImpurity === None)
    assert(split1.nodeSplit === None)

    val split2 = splits.next()
    assert(split2.prediction === 2.0)
    assert(split2.nodeId === 14)
    assert(split2.impurity === 0.0)
    assert(split2.treeId === 0)
    assert(split2.splitImpurity === None)
    assert(split2.nodeSplit === None)

    val split3 = splits.next()
    assert(split3.prediction === 4.0)
    assert(split3.nodeId === 16)
    assert(split3.impurity === 0.0)
    assert(split3.treeId === 0)
    assert(split3.splitImpurity === None)
    assert(split3.nodeSplit === None)

    val split4 = splits.next()
    assert(split4.prediction === 3.0)
    assert(split4.nodeId === 7)
    assert(split4.impurity === 0.0)
    assert(split4.treeId === 1)
    assert(split4.splitImpurity === None)
    assert(split4.nodeSplit === None)

    val split5 = splits.next()
    assert(split5.prediction === 1.0)
    assert(split5.nodeId === 8)
    assert(split5.impurity === 0.0)
    assert(split5.treeId === 1)
    assert(split5.splitImpurity === None)
    assert(split5.nodeSplit === None)

    val split6 = splits.next()
    assert(split6.prediction === 3.0)
    assert(split6.nodeId === 10)
    assert(compareDouble(split6.impurity, 0.9183))
    assert(split6.treeId === 1)
    assert(compareDouble(split6.splitImpurity.get, 0.6666))
    assert(split6.nodeSplit.get.isInstanceOf[CategoricalSplitOnBinId])
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].parentNodeId === 10)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].featureId === 1)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap.size === 3)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(1) === 24)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(3) === 24)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(5) === 23)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights.size === 2)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(23) === 5)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(24) === 10)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeSubTreeHash.size === 0)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(23) === -1)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(24) === -1)

    val split7 = splits.next()
    assert(split7.prediction === 3.0)
    assert(split7.nodeId === 11)
    assert(compareDouble(split7.impurity, 0.8631))
    assert(split7.treeId === 1)
    assert(split7.splitImpurity.get === 0.0)
    assert(split7.nodeSplit.get.isInstanceOf[NumericSplitOnBinId])
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].parentNodeId === 11)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].featureId === 0)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].splitBinId === 3)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftId === 25)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightId === 26)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftWeight === 5)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightWeight === 2)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftSubTreeHash === -1)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightSubTreeHash === -1)
  }

  test("Test InfoGainStatistics for Binary Classification") {
    val numTrees = 2
    val nodeSplitsPerTree = Array.fill[mutable.Queue[NodeSplitOnBinId]](numTrees)(mutable.Queue[NodeSplitOnBinId]())

    nodeSplitsPerTree(0).enqueue(NumericSplitOnBinId(2, 3, 5, 13, 14, 2.0, 2.0, -1, -1), NumericSplitOnBinId(5, 1, 2, 15, 16, 3.0, 6.0, 0, -1))
    nodeSplitsPerTree(1).enqueue(NumericSplitOnBinId(3, 4, 2, 7, 8, 2.0, 5.0, -1, -1), CategoricalSplitOnBinId(4, 3, mutable.Map[Int, Int](1 -> 10, 4 -> 11), mutable.Map[Int, Double](10 -> 3.0, 11 -> 2.0), mutable.Map[Int, Int]()))

    val nodeDepths = Array.fill[mutable.Map[Int, Int]](2)(mutable.Map[Int, Int]())
    nodeDepths(0).put(13, 5)
    nodeDepths(0).put(14, 5)
    nodeDepths(0).put(16, 7)
    nodeDepths(1).put(7, 3)
    nodeDepths(1).put(8, 3)
    nodeDepths(1).put(10, 6)
    nodeDepths(1).put(11, 6)

    val scheduledLookup = ScheduledNodeSplitLookup.createLookupForNodeSplits(nodeSplitsPerTree, 100)
    val treeSeeds = new Array[Int](numTrees)
    var treeId = 0
    val randGen = scala.util.Random
    while (treeId < numTrees) {
      treeSeeds(treeId) = randGen.nextInt()
      treeId += 1
    }

    val numClasses = 2
    val numBinsPerFeature = Array[Int](10, 11, 17, 13, 15, 9)

    val featureBins = new Array[Bins](6)
    var featId = 0
    while (featId < numBinsPerFeature.length) {
      if (featId != 1) {
        val bins = mutable.ArrayBuffer[NumericBin]()
        bins += NumericBin(Double.NegativeInfinity, 0)
        while (bins.length < numBinsPerFeature(featId)) {
          bins += NumericBin(bins.length - 1, bins.length)
        }

        featureBins(featId) = NumericBins(bins.toArray)
      } else {
        featureBins(featId) = CategoricalBins(numBinsPerFeature(featId))
      }

      featId += 1
    }

    // Create an information gain statistics object with all the features per node.
    val infoGainStats2 = new InfoGainStatistics(
      scheduledLookup,
      numBinsPerFeature,
      treeSeeds,
      numBinsPerFeature.length,
      numClasses)

    // Add some samples and make sure that we get the right statistics accumulated.
    infoGainStats2.addUnsignedByteSample(
      0,
      13,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(4), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(15), Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5)), Array[Byte](2, 3)))

    infoGainStats2.addUnsignedByteSample(
      0,
      14,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(4), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(15), Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5)), Array[Byte](2, 3)))

    infoGainStats2.addUnsignedByteSample(
      0,
      16,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(11), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(6), Discretizer.convertToUnsignedByte(7)), Array[Byte](5, 1)))

    infoGainStats2.addUnsignedByteSample(
      0,
      16,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(4)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      8,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(8)), Array[Byte](4, 2)))

    infoGainStats2.addUnsignedByteSample(
      1,
      7,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      10,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      10,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      10,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      11,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      11,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 2)))

    assert(infoGainStats2.getBinLabelWeight(0, 14, 0, 5, 1) === 0)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 0, 4, 1) === 2)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 1, 4, 1) === 0)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 1, 1, 1) === 2)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 2, 4, 1) === 0)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 2, 15, 1) === 2)
    assert(infoGainStats2.getBinLabelWeight(0, 16, 1, 4, 1) === 0)
    assert(infoGainStats2.getBinLabelWeight(0, 16, 1, 7, 0) === 8)

    assert(infoGainStats2.getBinLabelWeight(1, 8, 3, 7, 4) === 0)
    assert(infoGainStats2.getBinLabelWeight(1, 8, 3, 7, 1) === 2)

    val nextNodeIdsPerTree = Array[Int](21, 23)

    // Now, get the splits.
    val splits = infoGainStats2.computeNodePredictionsAndSplits(
      featureBins,
      nextNodeIdsPerTree,
      nodeDepths,
      SequoiaForestOptions(
        numTrees = 2,
        treeType = TreeType.Classification_InfoGain,
        mtry = 6,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 100,
        localTrainThreshold = 100000,
        numSubTreesPerIteration = 3,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(numClasses),
        imputationType = ImputationType.SplitOnMissing),
      new scala.util.Random(17))

    assert(nodeDepths(0).size === 0)
    assert(nodeDepths(1).size === 4) // There should be 4 child nodes (due to all binary splits).

    val split1 = splits.next()
    assert(split1.prediction === 1.0)
    assert(split1.nodeId === 13)
    assert(split1.impurity === 0.0)
    assert(split1.treeId === 0)
    assert(split1.splitImpurity === None)
    assert(split1.nodeSplit === None)

    val split2 = splits.next()
    assert(split2.prediction === 1.0)
    assert(split2.nodeId === 14)
    assert(split2.impurity === 0.0)
    assert(split2.treeId === 0)
    assert(split2.splitImpurity === None)
    assert(split2.nodeSplit === None)

    val split3 = splits.next()
    assert(split3.prediction === 0.0)
    assert(split3.nodeId === 16)
    assert(split3.impurity === 0.0)
    assert(split3.treeId === 0)
    assert(split3.splitImpurity === None)
    assert(split3.nodeSplit === None)

    val split4 = splits.next()
    assert(split4.prediction === 0.0)
    assert(split4.nodeId === 7)
    assert(split4.impurity === 0.0)
    assert(split4.treeId === 1)
    assert(split4.splitImpurity === None)
    assert(split4.nodeSplit === None)

    val split5 = splits.next()
    assert(split5.prediction === 1.0)
    assert(split5.nodeId === 8)
    assert(split5.impurity === 0.0)
    assert(split5.treeId === 1)
    assert(split5.splitImpurity === None)
    assert(split5.nodeSplit === None)

    val split6 = splits.next()
    assert(split6.prediction === 0.0)
    assert(split6.nodeId === 10)
    assert(compareDouble(split6.impurity, 0.9183))
    assert(split6.treeId === 1)
    assert(split6.splitImpurity.get === 0.0)
    assert(split6.nodeSplit.get.isInstanceOf[CategoricalSplitOnBinId])
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].parentNodeId === 10)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].featureId === 1)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap.size === 3)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(1) === 23)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(3) === 24)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(5) === 23)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights.size === 2)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(23) === 10)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(24) === 5)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeSubTreeHash.size === 0)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(23) === -1)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(24) === -1)

    val split7 = splits.next()
    assert(split7.prediction === 0.0)
    assert(split7.nodeId === 11)
    assert(compareDouble(split7.impurity, 0.8631))
    assert(split7.treeId === 1)
    assert(split7.splitImpurity.get === 0.0)
    assert(split7.nodeSplit.get.isInstanceOf[NumericSplitOnBinId])
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].parentNodeId === 11)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].featureId === 0)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].splitBinId === 3)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftId === 25)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightId === 26)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftWeight === 5)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightWeight === 2)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftSubTreeHash === -1)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightSubTreeHash === -1)
  }

  test("Test InfoGainStatistics with Missing Values") {
    val numTrees = 2
    val nodeSplitsPerTree = Array.fill[mutable.Queue[NodeSplitOnBinId]](numTrees)(mutable.Queue[NodeSplitOnBinId]())

    nodeSplitsPerTree(0).enqueue(NumericSplitOnBinId(2, 3, 5, 13, 14, 2.0, 2.0, -1, -1), NumericSplitOnBinId(5, 1, 2, 15, 16, 3.0, 6.0, 0, -1))
    nodeSplitsPerTree(1).enqueue(NumericSplitOnBinId(3, 4, 2, 7, 8, 2.0, 5.0, -1, -1), CategoricalSplitOnBinId(4, 3, mutable.Map[Int, Int](1 -> 10, 4 -> 11), mutable.Map[Int, Double](10 -> 3.0, 11 -> 2.0), mutable.Map[Int, Int]()))

    val nodeDepths = Array.fill[mutable.Map[Int, Int]](2)(mutable.Map[Int, Int]())
    nodeDepths(0).put(13, 5)
    nodeDepths(0).put(14, 5)
    nodeDepths(0).put(16, 7)
    nodeDepths(1).put(7, 3)
    nodeDepths(1).put(8, 3)
    nodeDepths(1).put(10, 6)
    nodeDepths(1).put(11, 6)

    val scheduledLookup = ScheduledNodeSplitLookup.createLookupForNodeSplits(nodeSplitsPerTree, 100)
    val treeSeeds = new Array[Int](numTrees)
    var treeId = 0
    val randGen = scala.util.Random
    while (treeId < numTrees) {
      treeSeeds(treeId) = randGen.nextInt()
      treeId += 1
    }

    val numClasses = 2
    val numBinsPerFeature = Array[Int](11, 12, 17, 13, 15, 9) // Features 0, 1 have NaN values.

    val featureBins = new Array[Bins](6)
    var featId = 0
    while (featId < numBinsPerFeature.length) {
      if (featId != 1) {
        val bins = mutable.ArrayBuffer[NumericBin]()
        bins += NumericBin(Double.NegativeInfinity, 0)
        val numBinsWithoutNaN = if (featId == 0) {
          numBinsPerFeature(featId) - 1
        } else {
          numBinsPerFeature(featId)
        }

        while (bins.length < numBinsWithoutNaN) {
          bins += NumericBin(bins.length - 1, bins.length)
        }

        if (featId == 0) {
          featureBins(featId) = NumericBins(bins.toArray, numBinsPerFeature(featId) - 1)
        } else {
          featureBins(featId) = NumericBins(bins.toArray)
        }
      } else {
        featureBins(featId) = CategoricalBins(numBinsPerFeature(featId) - 1, 11) // The last one is missing value bin.
      }

      featId += 1
    }

    // Create an information gain statistics object with all the features per node.
    val infoGainStats2 = new InfoGainStatistics(
      scheduledLookup,
      numBinsPerFeature,
      treeSeeds,
      numBinsPerFeature.length,
      numClasses)

    // Add some samples and make sure that we get the right statistics accumulated.
    infoGainStats2.addUnsignedByteSample(
      0,
      13,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(4), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(15), Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5)), Array[Byte](2, 3)))

    infoGainStats2.addUnsignedByteSample(
      0,
      14,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(4), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(15), Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5)), Array[Byte](2, 3)))

    infoGainStats2.addUnsignedByteSample(
      0,
      16,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(11), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(6), Discretizer.convertToUnsignedByte(7)), Array[Byte](5, 1)))

    infoGainStats2.addUnsignedByteSample(
      0,
      16,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(4)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      8,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(8)), Array[Byte](4, 2)))

    infoGainStats2.addUnsignedByteSample(
      1,
      7,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      10,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      10,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(11), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      10,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      11,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      11,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      11,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 2)))

    assert(infoGainStats2.getBinLabelWeight(0, 14, 0, 5, 1) === 0)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 0, 4, 1) === 2)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 1, 4, 1) === 0)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 1, 1, 1) === 2)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 2, 4, 1) === 0)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 2, 15, 1) === 2)
    assert(infoGainStats2.getBinLabelWeight(0, 16, 1, 4, 1) === 0)
    assert(infoGainStats2.getBinLabelWeight(0, 16, 1, 7, 0) === 8)

    assert(infoGainStats2.getBinLabelWeight(1, 8, 3, 7, 4) === 0)
    assert(infoGainStats2.getBinLabelWeight(1, 8, 3, 7, 1) === 2)

    val nextNodeIdsPerTree = Array[Int](21, 23)

    // Now, get the splits.
    val splits = infoGainStats2.computeNodePredictionsAndSplits(
      featureBins,
      nextNodeIdsPerTree,
      nodeDepths,
      SequoiaForestOptions(
        numTrees = 2,
        treeType = TreeType.Classification_InfoGain,
        mtry = 6,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 100,
        localTrainThreshold = 100000,
        numSubTreesPerIteration = 3,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(numClasses),
        imputationType = ImputationType.SplitOnMissing),
      new scala.util.Random(17))

    assert(nodeDepths(0).size === 0)
    assert(nodeDepths(1).size === 6) // There should be 6 child nodes (due to all binary splits and NaN bins).

    val split1 = splits.next()
    assert(split1.prediction === 1.0)
    assert(split1.nodeId === 13)
    assert(split1.impurity === 0.0)
    assert(split1.treeId === 0)
    assert(split1.splitImpurity === None)
    assert(split1.nodeSplit === None)

    val split2 = splits.next()
    assert(split2.prediction === 1.0)
    assert(split2.nodeId === 14)
    assert(split2.impurity === 0.0)
    assert(split2.treeId === 0)
    assert(split2.splitImpurity === None)
    assert(split2.nodeSplit === None)

    val split3 = splits.next()
    assert(split3.prediction === 0.0)
    assert(split3.nodeId === 16)
    assert(split3.impurity === 0.0)
    assert(split3.treeId === 0)
    assert(split3.splitImpurity === None)
    assert(split3.nodeSplit === None)

    val split4 = splits.next()
    assert(split4.prediction === 0.0)
    assert(split4.nodeId === 7)
    assert(split4.impurity === 0.0)
    assert(split4.treeId === 1)
    assert(split4.splitImpurity === None)
    assert(split4.nodeSplit === None)

    val split5 = splits.next()
    assert(split5.prediction === 1.0)
    assert(split5.nodeId === 8)
    assert(split5.impurity === 0.0)
    assert(split5.treeId === 1)
    assert(split5.splitImpurity === None)
    assert(split5.nodeSplit === None)

    val split6 = splits.next()
    assert(split6.prediction === 0.0)
    assert(split6.nodeId === 10)
    assert(compareDouble(split6.impurity, 0.9183))
    assert(split6.treeId === 1)
    assert(split6.splitImpurity.get === 0.0)
    assert(split6.nodeSplit.get.isInstanceOf[CategoricalSplitOnBinId])
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].parentNodeId === 10)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].featureId === 1)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap.size === 3)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(11) === 25)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(3) === 24)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(5) === 23)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights.size === 3)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(23) === 5)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(24) === 5)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(25) === 5)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeSubTreeHash.size === 0)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(23) === -1)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(24) === -1)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(25) === -1)

    val split7 = splits.next()
    assert(split7.prediction === 0.0)
    assert(split7.nodeId === 11)
    assert(compareDouble(split7.impurity, 0.6500224))
    assert(split7.treeId === 1)
    assert(split7.splitImpurity.get === 0.0)
    assert(split7.nodeSplit.get.isInstanceOf[NumericSplitOnBinId])
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].parentNodeId === 11)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].featureId === 0)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].splitBinId === 3)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftId === 26)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightId === 27)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftWeight === 5)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightWeight === 2)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftSubTreeHash === -1)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightSubTreeHash === -1)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].nanBinId === 10)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].nanNodeId === 28)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].nanWeight === 5)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].nanSubTreeHash === -1)
  }

  test("Test InfoGainStatistics with Missing Values Degenerate Cases") {
    val numTrees = 2
    val nodeSplitsPerTree = Array.fill[mutable.Queue[NodeSplitOnBinId]](numTrees)(mutable.Queue[NodeSplitOnBinId]())

    nodeSplitsPerTree(0).enqueue(NumericSplitOnBinId(2, 3, 5, 13, 14, 2.0, 2.0, -1, -1), NumericSplitOnBinId(5, 1, 2, 15, 16, 3.0, 6.0, 0, -1))
    nodeSplitsPerTree(1).enqueue(NumericSplitOnBinId(3, 4, 2, 7, 8, 2.0, 5.0, -1, -1), CategoricalSplitOnBinId(4, 3, mutable.Map[Int, Int](1 -> 10, 4 -> 11), mutable.Map[Int, Double](10 -> 3.0, 11 -> 2.0), mutable.Map[Int, Int]()))

    val nodeDepths = Array.fill[mutable.Map[Int, Int]](2)(mutable.Map[Int, Int]())
    nodeDepths(0).put(13, 5)
    nodeDepths(0).put(14, 5)
    nodeDepths(0).put(16, 7)
    nodeDepths(1).put(7, 3)
    nodeDepths(1).put(8, 3)
    nodeDepths(1).put(10, 6)
    nodeDepths(1).put(11, 6)

    val scheduledLookup = ScheduledNodeSplitLookup.createLookupForNodeSplits(nodeSplitsPerTree, 100)
    val treeSeeds = new Array[Int](numTrees)
    var treeId = 0
    val randGen = scala.util.Random
    while (treeId < numTrees) {
      treeSeeds(treeId) = randGen.nextInt()
      treeId += 1
    }

    val numClasses = 2
    val numBinsPerFeature = Array[Int](11, 12, 17, 13, 15, 9) // Features 0, 1 have NaN values.

    val featureBins = new Array[Bins](6)
    var featId = 0
    while (featId < numBinsPerFeature.length) {
      if (featId != 1) {
        val bins = mutable.ArrayBuffer[NumericBin]()
        bins += NumericBin(Double.NegativeInfinity, 0)
        val numBinsWithoutNaN = if (featId == 0) {
          numBinsPerFeature(featId) - 1
        } else {
          numBinsPerFeature(featId)
        }

        while (bins.length < numBinsWithoutNaN) {
          bins += NumericBin(bins.length - 1, bins.length)
        }

        if (featId == 0) {
          featureBins(featId) = NumericBins(bins.toArray, numBinsPerFeature(featId) - 1)
        } else {
          featureBins(featId) = NumericBins(bins.toArray)
        }
      } else {
        featureBins(featId) = CategoricalBins(numBinsPerFeature(featId) - 1, 11) // The last one is missing value bin.
      }

      featId += 1
    }

    // Create an information gain statistics object with all the features per node.
    val infoGainStats2 = new InfoGainStatistics(
      scheduledLookup,
      numBinsPerFeature,
      treeSeeds,
      numBinsPerFeature.length,
      numClasses)

    // Add some samples and make sure that we get the right statistics accumulated.
    infoGainStats2.addUnsignedByteSample(
      0,
      13,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(4), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(15), Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5)), Array[Byte](2, 3)))

    infoGainStats2.addUnsignedByteSample(
      0,
      14,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(4), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(15), Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5)), Array[Byte](2, 3)))

    infoGainStats2.addUnsignedByteSample(
      0,
      16,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(11), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(6), Discretizer.convertToUnsignedByte(7)), Array[Byte](5, 1)))

    infoGainStats2.addUnsignedByteSample(
      0,
      16,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(4)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      8,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(8)), Array[Byte](4, 2)))

    infoGainStats2.addUnsignedByteSample(
      1,
      7,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(11), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      10,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      10,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(11), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      11,
      (0.0, Array[Byte](Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    infoGainStats2.addUnsignedByteSample(
      1,
      11,
      (1.0, Array[Byte](Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 2)))

    assert(infoGainStats2.getBinLabelWeight(0, 14, 0, 5, 1) === 0)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 0, 4, 1) === 2)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 1, 4, 1) === 0)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 1, 1, 1) === 2)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 2, 4, 1) === 0)
    assert(infoGainStats2.getBinLabelWeight(0, 14, 2, 15, 1) === 2)
    assert(infoGainStats2.getBinLabelWeight(0, 16, 1, 4, 1) === 0)
    assert(infoGainStats2.getBinLabelWeight(0, 16, 1, 7, 0) === 8)

    assert(infoGainStats2.getBinLabelWeight(1, 8, 3, 7, 4) === 0)
    assert(infoGainStats2.getBinLabelWeight(1, 8, 3, 7, 1) === 2)

    val nextNodeIdsPerTree = Array[Int](21, 23)

    // Now, get the splits.
    val splits = infoGainStats2.computeNodePredictionsAndSplits(
      featureBins,
      nextNodeIdsPerTree,
      nodeDepths,
      SequoiaForestOptions(
        numTrees = 2,
        treeType = TreeType.Classification_InfoGain,
        mtry = 6,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 100,
        localTrainThreshold = 100000,
        numSubTreesPerIteration = 3,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(numClasses),
        imputationType = ImputationType.SplitOnMissing),
      new scala.util.Random(17))

    assert(nodeDepths(0).size === 0)
    assert(nodeDepths(1).size === 4) // There should be 6 child nodes (due to all binary splits and NaN bins).

    val split1 = splits.next()
    assert(split1.prediction === 1.0)
    assert(split1.nodeId === 13)
    assert(split1.impurity === 0.0)
    assert(split1.treeId === 0)
    assert(split1.splitImpurity === None)
    assert(split1.nodeSplit === None)

    val split2 = splits.next()
    assert(split2.prediction === 1.0)
    assert(split2.nodeId === 14)
    assert(split2.impurity === 0.0)
    assert(split2.treeId === 0)
    assert(split2.splitImpurity === None)
    assert(split2.nodeSplit === None)

    val split3 = splits.next()
    assert(split3.prediction === 0.0)
    assert(split3.nodeId === 16)
    assert(split3.impurity === 0.0)
    assert(split3.treeId === 0)
    assert(split3.splitImpurity === None)
    assert(split3.nodeSplit === None)

    val split4 = splits.next()
    assert(split4.prediction === 0.0)
    assert(split4.nodeId === 7)
    assert(split4.impurity === 0.0)
    assert(split4.treeId === 1)
    assert(split4.splitImpurity === None)
    assert(split4.nodeSplit === None)

    val split5 = splits.next()
    assert(split5.prediction === 1.0)
    assert(split5.nodeId === 8)
    assert(split5.impurity === 0.0)
    assert(split5.treeId === 1)
    assert(split5.splitImpurity === None)
    assert(split5.nodeSplit === None)

    val split6 = splits.next()
    assert(split6.prediction === 0.0)
    assert(split6.nodeId === 10)
    assert(compareDouble(split6.impurity, 1.0))
    assert(split6.treeId === 1)
    assert(split6.splitImpurity.get === 0.0)
    assert(split6.nodeSplit.get.isInstanceOf[CategoricalSplitOnBinId])
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].parentNodeId === 10)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].featureId === 1)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap.size === 2)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(11) === 24)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(5) === 23)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights.size === 2)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(23) === 5)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(24) === 5)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeSubTreeHash.size === 0)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(23) === -1)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(24) === -1)

    val split7 = splits.next()
    assert(split7.prediction === 0.0)
    assert(split7.nodeId === 11)
    assert(compareDouble(split7.impurity, 0.8631206))
    assert(split7.treeId === 1)
    assert(split7.splitImpurity.get === 0.0)
    assert(split7.nodeSplit.get.isInstanceOf[NumericSplitOnBinId])
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].parentNodeId === 11)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].featureId === 0)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].splitBinId === 0)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftId === -1)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightId === 25)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftWeight === 0.0)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightWeight === 2.0)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftSubTreeHash === -1)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightSubTreeHash === -1)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].nanBinId === 10)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].nanNodeId === 26)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].nanWeight === 5.0)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].nanSubTreeHash === -1)
  }

  test("Test VarianceStatistics") {
    val numTrees = 2
    val nodeSplitsPerTree = Array.fill[mutable.Queue[NodeSplitOnBinId]](numTrees)(mutable.Queue[NodeSplitOnBinId]())

    nodeSplitsPerTree(0).enqueue(NumericSplitOnBinId(2, 3, 5, 13, 14, 2.0, 2.0, -1, -1), NumericSplitOnBinId(5, 1, 2, 15, 16, 3.0, 6.0, 0, -1))
    nodeSplitsPerTree(1).enqueue(NumericSplitOnBinId(3, 4, 2, 7, 8, 2.0, 5.0, -1, -1), CategoricalSplitOnBinId(4, 3, mutable.Map[Int, Int](1 -> 10, 4 -> 11), mutable.Map[Int, Double](10 -> 3.0, 11 -> 2.0), mutable.Map[Int, Int]()))

    val nodeDepths = Array.fill[mutable.Map[Int, Int]](2)(mutable.Map[Int, Int]())
    nodeDepths(0).put(13, 5)
    nodeDepths(0).put(14, 5)
    nodeDepths(0).put(16, 7)
    nodeDepths(1).put(7, 3)
    nodeDepths(1).put(8, 3)
    nodeDepths(1).put(10, 6)
    nodeDepths(1).put(11, 6)

    val scheduledLookup = ScheduledNodeSplitLookup.createLookupForNodeSplits(nodeSplitsPerTree, 100)
    val treeSeeds = new Array[Int](numTrees)
    var treeId = 0
    val randGen = scala.util.Random
    while (treeId < numTrees) {
      treeSeeds(treeId) = randGen.nextInt()
      treeId += 1
    }

    val mtry = 4
    val numBinsPerFeature = Array[Int](10, 11, 17, 13, 15, 9)

    val featureBins = new Array[Bins](6)
    var featId = 0
    while (featId < numBinsPerFeature.length) {
      if (featId != 1) {
        val bins = mutable.ArrayBuffer[NumericBin]()
        bins += NumericBin(Double.NegativeInfinity, 0)
        while (bins.length < numBinsPerFeature(featId)) {
          bins += NumericBin(bins.length - 1, bins.length)
        }

        featureBins(featId) = NumericBins(bins.toArray)
      } else {
        featureBins(featId) = CategoricalBins(numBinsPerFeature(featId))
      }

      featId += 1
    }

    // Create a variance statistics object.
    val varStats = new VarianceStatistics(
      scheduledLookup,
      numBinsPerFeature,
      treeSeeds,
      mtry)

    assert(varStats.numTrees === 2)
    assert(varStats.numFeatures === 6)
    assert(varStats.numSelectedFeaturesPerNode === 4)

    assert(varStats.startNodeIds(0) === 13)
    assert(varStats.startNodeIds(1) === 7)
    assert(varStats.numNodes === 7)

    assert(varStats.selectedFeaturesLookup.length === 2)
    assert(varStats.selectedFeaturesLookup(0).length === 4)
    assert(varStats.selectedFeaturesLookup(1).length === 5)

    assert(varStats.offsetLookup.length === 2)
    assert(varStats.offsetLookup(0).length === 4)
    assert(varStats.offsetLookup(1).length === 5)

    var expectedBinStatsLength = 0

    treeId = 0
    while (treeId < numTrees) {
      var nodeIdx = 0
      while (nodeIdx < varStats.selectedFeaturesLookup(treeId).length) {
        if ((treeId == 0 && nodeIdx == 2) || (treeId == 1 && nodeIdx == 2)) {
          assert(varStats.selectedFeaturesLookup(treeId)(nodeIdx).length === 0)
          assert(varStats.offsetLookup(treeId)(nodeIdx).length === 0)
        } else {
          assert(varStats.selectedFeaturesLookup(treeId)(nodeIdx).length === mtry)
          assert(varStats.offsetLookup(treeId)(nodeIdx).length === mtry)

          var featureIdx = 0
          while (featureIdx < mtry) {
            val featureId = varStats.selectedFeaturesLookup(treeId)(nodeIdx)(featureIdx)
            val numBins = numBinsPerFeature(featureId)
            assert(varStats.offsetLookup(treeId)(nodeIdx)(featureIdx) === expectedBinStatsLength)
            expectedBinStatsLength += numBins * 3
            featureIdx += 1
          }
        }

        nodeIdx += 1
      }

      treeId += 1
    }

    assert(varStats.binStatsArray.asInstanceOf[RegressionStatisticsArray].binStats.length === expectedBinStatsLength)

    // Create a variance statistics object with all the features per node.
    val varStats2 = new VarianceStatistics(
      scheduledLookup,
      numBinsPerFeature,
      treeSeeds,
      numBinsPerFeature.length)

    // Add some samples and make sure that we get the right statistics accumulated.
    varStats2.addUnsignedByteSample(
      treeId = 0,
      nodeId = 13,
      sample = (1.0, Array[Byte](Discretizer.convertToUnsignedByte(4), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(15), Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5)), Array[Byte](2, 3)))

    varStats2.addUnsignedByteSample(
      treeId = 0,
      nodeId = 14,
      sample = (2.0, Array[Byte](Discretizer.convertToUnsignedByte(4), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(15), Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5)), Array[Byte](2, 3)))

    varStats2.addUnsignedByteSample(
      treeId = 0,
      nodeId = 16,
      sample = (4.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(11), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(6), Discretizer.convertToUnsignedByte(7)), Array[Byte](5, 1)))

    varStats2.addUnsignedByteSample(
      treeId = 0,
      nodeId = 16,
      sample = (4.0, Array[Byte](Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(4)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 8,
      sample = (1.0, Array[Byte](Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(8)), Array[Byte](4, 2)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 7,
      sample = (3.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 10,
      sample = (3.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 10,
      sample = (4.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 10,
      sample = (3.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 11,
      sample = (3.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 11,
      sample = (2.0, Array[Byte](Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 2)))

    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 0)._1 === 0.0)
    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 0)._2 === 0.0)
    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 0)._3 === 0.0)
    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 4)._1 === 1.0 * 2.0)
    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 4)._2 === 1.0 * 2.0)
    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 4)._3 === 2.0)

    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 0)._1 === 0.0)
    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 0)._2 === 0.0)
    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 0)._3 === 0.0)
    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 1)._1 === 3.0 * 5.0)
    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 1)._2 === 9.0 * 5.0)
    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 1)._3 === 5.0)

    val nextNodeIdsPerTree = Array[Int](21, 23)

    // Now, get the splits.
    val splits = varStats2.computeNodePredictionsAndSplits(
      featureBins,
      nextNodeIdsPerTree,
      nodeDepths,
      SequoiaForestOptions(
        numTrees = 2,
        treeType = TreeType.Regression_Variance,
        mtry = 6,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 100,
        localTrainThreshold = 100000,
        numSubTreesPerIteration = 3,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = None,
        imputationType = ImputationType.SplitOnMissing),
      new scala.util.Random(17))

    assert(nodeDepths(0).size === 0)
    assert(nodeDepths(1).size === 4)

    val split1 = splits.next()
    assert(split1.treeId === 0)
    assert(split1.nodeId === 13)
    assert(split1.prediction === 1.0)
    assert(split1.weight === 2.0)
    assert(split1.impurity === 0.0)
    assert(split1.splitImpurity === None)
    assert(split1.nodeSplit === None)

    val split2 = splits.next()
    assert(split2.treeId === 0)
    assert(split2.nodeId === 14)
    assert(split2.prediction === 2.0)
    assert(split2.weight === 2.0)
    assert(split2.impurity === 0.0)
    assert(split2.splitImpurity === None)
    assert(split2.nodeSplit === None)

    val split3 = splits.next()
    assert(split3.treeId === 0)
    assert(split3.nodeId === 16)
    assert(split3.prediction === 4.0)
    assert(split3.weight === 8.0)
    assert(split3.impurity === 0.0)
    assert(split3.splitImpurity === None)
    assert(split3.nodeSplit === None)

    val split4 = splits.next()
    assert(split4.treeId === 1)
    assert(split4.nodeId === 7)
    assert(split4.prediction === 3.0)
    assert(split4.weight === 5.0)
    assert(split4.impurity === 0.0)
    assert(split4.splitImpurity === None)
    assert(split4.nodeSplit === None)

    val split5 = splits.next()
    assert(split5.treeId === 1)
    assert(split5.nodeId === 8)
    assert(split5.prediction === 1.0)
    assert(split5.weight === 2.0)
    assert(split5.impurity === 0.0)
    assert(split5.splitImpurity === None)
    assert(split5.nodeSplit === None)

    val split6 = splits.next()
    assert(split6.treeId === 1)
    assert(split6.nodeId === 10)
    assert(compareDouble(split6.prediction, 3.33333))
    assert(split6.weight === 15.0)
    assert(compareDouble(split6.impurity, 0.22222))
    assert(split6.splitImpurity.get === 0.0)
    assert(split6.nodeSplit.get.featureId === 1)
    assert(split6.nodeSplit.get.parentNodeId === 10)
    assert(split6.nodeSplit.get.isInstanceOf[CategoricalSplitOnBinId])
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap.size === 3)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(1) === 24)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(3) === 23)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(5) === 23)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights.size === 2)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(23) === 10)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(24) === 5)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeSubTreeHash.size === 0)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(23) === -1)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(24) === -1)

    val split7 = splits.next()
    assert(compareDouble(split7.prediction, 2.714286))
    assert(split7.nodeId === 11)
    assert(compareDouble(split7.impurity, 0.2040816))
    assert(split7.treeId === 1)
    assert(split7.splitImpurity.get === 0.0)
    assert(split7.nodeSplit.get.isInstanceOf[NumericSplitOnBinId])
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].parentNodeId === 11)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].featureId === 0)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].splitBinId === 3)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftId === 25)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightId === 26)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftWeight === 5)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightWeight === 2)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftSubTreeHash === -1)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightSubTreeHash === -1)
  }

  test("Test VarianceStatistics with Unsigned Short Features") {
    val numTrees = 2
    val nodeSplitsPerTree = Array.fill[mutable.Queue[NodeSplitOnBinId]](numTrees)(mutable.Queue[NodeSplitOnBinId]())

    nodeSplitsPerTree(0).enqueue(NumericSplitOnBinId(2, 3, 5, 13, 14, 2.0, 2.0, -1, -1), NumericSplitOnBinId(5, 1, 2, 15, 16, 3.0, 6.0, 0, -1))
    nodeSplitsPerTree(1).enqueue(NumericSplitOnBinId(3, 4, 2, 7, 8, 2.0, 5.0, -1, -1), CategoricalSplitOnBinId(4, 3, mutable.Map[Int, Int](1 -> 10, 4 -> 11), mutable.Map[Int, Double](10 -> 3.0, 11 -> 2.0), mutable.Map[Int, Int]()))

    val nodeDepths = Array.fill[mutable.Map[Int, Int]](2)(mutable.Map[Int, Int]())
    nodeDepths(0).put(13, 5)
    nodeDepths(0).put(14, 5)
    nodeDepths(0).put(16, 7)
    nodeDepths(1).put(7, 3)
    nodeDepths(1).put(8, 3)
    nodeDepths(1).put(10, 6)
    nodeDepths(1).put(11, 6)

    val scheduledLookup = ScheduledNodeSplitLookup.createLookupForNodeSplits(nodeSplitsPerTree, 100)
    val treeSeeds = new Array[Int](numTrees)
    var treeId = 0
    val randGen = scala.util.Random
    while (treeId < numTrees) {
      treeSeeds(treeId) = randGen.nextInt()
      treeId += 1
    }

    val numBinsPerFeature = Array[Int](10, 11, 17, 13, 15, 9)

    val featureBins = new Array[Bins](6)
    var featId = 0
    while (featId < numBinsPerFeature.length) {
      if (featId != 1) {
        val bins = mutable.ArrayBuffer[NumericBin]()
        bins += NumericBin(Double.NegativeInfinity, 0)
        while (bins.length < numBinsPerFeature(featId)) {
          bins += NumericBin(bins.length - 1, bins.length)
        }

        featureBins(featId) = NumericBins(bins.toArray)
      } else {
        featureBins(featId) = CategoricalBins(numBinsPerFeature(featId))
      }

      featId += 1
    }

    // Create an information gain statistics object.
    val varStats = new VarianceStatistics(
      scheduledLookup,
      numBinsPerFeature,
      treeSeeds,
      numBinsPerFeature.length)

    // Add some samples and make sure that we get the right statistics accumulated.
    varStats.addUnsignedShortSample(
      0,
      13,
      (1.0, Array[Short](Discretizer.convertToUnsignedShort(4), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(15), Discretizer.convertToUnsignedShort(10), Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(5)), Array[Byte](2, 3)))

    varStats.addUnsignedShortSample(
      0,
      14,
      (2.0, Array[Short](Discretizer.convertToUnsignedShort(4), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(15), Discretizer.convertToUnsignedShort(10), Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(5)), Array[Byte](2, 3)))

    varStats.addUnsignedShortSample(
      0,
      16,
      (4.0, Array[Short](Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(11), Discretizer.convertToUnsignedShort(5), Discretizer.convertToUnsignedShort(6), Discretizer.convertToUnsignedShort(7)), Array[Byte](5, 1)))

    varStats.addUnsignedShortSample(
      0,
      16,
      (4.0, Array[Short](Discretizer.convertToUnsignedShort(5), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(4)), Array[Byte](3, 5)))

    varStats.addUnsignedShortSample(
      1,
      8,
      (1.0, Array[Short](Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(5), Discretizer.convertToUnsignedShort(8)), Array[Byte](4, 2)))

    varStats.addUnsignedShortSample(
      1,
      7,
      (3.0, Array[Short](Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(5), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(2)), Array[Byte](3, 5)))

    varStats.addUnsignedShortSample(
      1,
      10,
      (3.0, Array[Short](Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(5), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(2)), Array[Byte](3, 5)))

    varStats.addUnsignedShortSample(
      1,
      10,
      (4.0, Array[Short](Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(2)), Array[Byte](3, 5)))

    varStats.addUnsignedShortSample(
      1,
      10,
      (3.0, Array[Short](Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(2)), Array[Byte](3, 5)))

    varStats.addUnsignedShortSample(
      1,
      11,
      (3.0, Array[Short](Discretizer.convertToUnsignedShort(2), Discretizer.convertToUnsignedShort(5), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(2)), Array[Byte](3, 5)))

    varStats.addUnsignedShortSample(
      1,
      11,
      (2.0, Array[Short](Discretizer.convertToUnsignedShort(5), Discretizer.convertToUnsignedShort(5), Discretizer.convertToUnsignedShort(7), Discretizer.convertToUnsignedShort(1), Discretizer.convertToUnsignedShort(3), Discretizer.convertToUnsignedShort(2)), Array[Byte](3, 2)))

    assert(varStats.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 0)._1 === 0.0)
    assert(varStats.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 0)._2 === 0.0)
    assert(varStats.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 0)._3 === 0.0)
    assert(varStats.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 4)._1 === 1.0 * 2.0)
    assert(varStats.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 4)._2 === 1.0 * 2.0)
    assert(varStats.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 4)._3 === 2.0)

    assert(varStats.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 0)._1 === 0.0)
    assert(varStats.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 0)._2 === 0.0)
    assert(varStats.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 0)._3 === 0.0)
    assert(varStats.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 1)._1 === 3.0 * 5.0)
    assert(varStats.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 1)._2 === 9.0 * 5.0)
    assert(varStats.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 1)._3 === 5.0)

    val nextNodeIdsPerTree = Array[Int](21, 23)

    // Now, get the splits.
    val splits = varStats.computeNodePredictionsAndSplits(
      featureBins,
      nextNodeIdsPerTree,
      nodeDepths,
      SequoiaForestOptions(
        numTrees = 2,
        treeType = TreeType.Regression_Variance,
        mtry = 6,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 100,
        localTrainThreshold = 100000,
        numSubTreesPerIteration = 3,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = None,
        imputationType = ImputationType.SplitOnMissing),
      new scala.util.Random(17))

    assert(nodeDepths(0).size === 0)
    assert(nodeDepths(1).size === 4)

    val split1 = splits.next()
    assert(split1.treeId === 0)
    assert(split1.nodeId === 13)
    assert(split1.prediction === 1.0)
    assert(split1.weight === 2.0)
    assert(split1.impurity === 0.0)
    assert(split1.splitImpurity === None)
    assert(split1.nodeSplit === None)

    val split2 = splits.next()
    assert(split2.treeId === 0)
    assert(split2.nodeId === 14)
    assert(split2.prediction === 2.0)
    assert(split2.weight === 2.0)
    assert(split2.impurity === 0.0)
    assert(split2.splitImpurity === None)
    assert(split2.nodeSplit === None)

    val split3 = splits.next()
    assert(split3.treeId === 0)
    assert(split3.nodeId === 16)
    assert(split3.prediction === 4.0)
    assert(split3.weight === 8.0)
    assert(split3.impurity === 0.0)
    assert(split3.splitImpurity === None)
    assert(split3.nodeSplit === None)

    val split4 = splits.next()
    assert(split4.treeId === 1)
    assert(split4.nodeId === 7)
    assert(split4.prediction === 3.0)
    assert(split4.weight === 5.0)
    assert(split4.impurity === 0.0)
    assert(split4.splitImpurity === None)
    assert(split4.nodeSplit === None)

    val split5 = splits.next()
    assert(split5.treeId === 1)
    assert(split5.nodeId === 8)
    assert(split5.prediction === 1.0)
    assert(split5.weight === 2.0)
    assert(split5.impurity === 0.0)
    assert(split5.splitImpurity === None)
    assert(split5.nodeSplit === None)

    val split6 = splits.next()
    assert(split6.treeId === 1)
    assert(split6.nodeId === 10)
    assert(compareDouble(split6.prediction, 3.33333))
    assert(split6.weight === 15.0)
    assert(compareDouble(split6.impurity, 0.22222))
    assert(split6.splitImpurity.get === 0.0)
    assert(split6.nodeSplit.get.featureId === 1)
    assert(split6.nodeSplit.get.parentNodeId === 10)
    assert(split6.nodeSplit.get.isInstanceOf[CategoricalSplitOnBinId])
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap.size === 3)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(1) === 24)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(3) === 23)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(5) === 23)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights.size === 2)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(23) === 10)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(24) === 5)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeSubTreeHash.size === 0)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(23) === -1)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(24) === -1)

    val split7 = splits.next()
    assert(compareDouble(split7.prediction, 2.714286))
    assert(split7.nodeId === 11)
    assert(compareDouble(split7.impurity, 0.2040816))
    assert(split7.treeId === 1)
    assert(split7.splitImpurity.get === 0.0)
    assert(split7.nodeSplit.get.isInstanceOf[NumericSplitOnBinId])
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].parentNodeId === 11)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].featureId === 0)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].splitBinId === 3)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftId === 25)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightId === 26)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftWeight === 5)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightWeight === 2)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftSubTreeHash === -1)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightSubTreeHash === -1)
  }

  test("Test VarianceStatistics with Missing Values") {
    val numTrees = 2
    val nodeSplitsPerTree = Array.fill[mutable.Queue[NodeSplitOnBinId]](numTrees)(mutable.Queue[NodeSplitOnBinId]())

    nodeSplitsPerTree(0).enqueue(NumericSplitOnBinId(2, 3, 5, 13, 14, 2.0, 2.0, -1, -1), NumericSplitOnBinId(5, 1, 2, 15, 16, 3.0, 6.0, 0, -1))
    nodeSplitsPerTree(1).enqueue(NumericSplitOnBinId(3, 4, 2, 7, 8, 2.0, 5.0, -1, -1), CategoricalSplitOnBinId(4, 3, mutable.Map[Int, Int](1 -> 10, 4 -> 11), mutable.Map[Int, Double](10 -> 3.0, 11 -> 2.0), mutable.Map[Int, Int]()))

    val nodeDepths = Array.fill[mutable.Map[Int, Int]](2)(mutable.Map[Int, Int]())
    nodeDepths(0).put(13, 5)
    nodeDepths(0).put(14, 5)
    nodeDepths(0).put(16, 7)
    nodeDepths(1).put(7, 3)
    nodeDepths(1).put(8, 3)
    nodeDepths(1).put(10, 6)
    nodeDepths(1).put(11, 6)

    val scheduledLookup = ScheduledNodeSplitLookup.createLookupForNodeSplits(nodeSplitsPerTree, 100)
    val treeSeeds = new Array[Int](numTrees)
    var treeId = 0
    val randGen = scala.util.Random
    while (treeId < numTrees) {
      treeSeeds(treeId) = randGen.nextInt()
      treeId += 1
    }

    val mtry = 4
    val numBinsPerFeature = Array[Int](11, 12, 17, 13, 15, 9)

    val featureBins = new Array[Bins](6)
    var featId = 0
    while (featId < numBinsPerFeature.length) {
      if (featId != 1) {
        val bins = mutable.ArrayBuffer[NumericBin]()
        bins += NumericBin(Double.NegativeInfinity, 0)
        val numBinsWithoutNaN = if (featId == 0) {
          numBinsPerFeature(featId) - 1
        } else {
          numBinsPerFeature(featId)
        }

        while (bins.length < numBinsWithoutNaN) {
          bins += NumericBin(bins.length - 1, bins.length)
        }

        if (featId == 0) {
          featureBins(featId) = NumericBins(bins.toArray, numBinsPerFeature(featId) - 1)
        } else {
          featureBins(featId) = NumericBins(bins.toArray)
        }
      } else {
        featureBins(featId) = CategoricalBins(numBinsPerFeature(featId) - 1, 11) // The last one is missing value bin.
      }

      featId += 1
    }

    // Create a variance statistics object with all the features per node.
    val varStats2 = new VarianceStatistics(
      scheduledLookup,
      numBinsPerFeature,
      treeSeeds,
      numBinsPerFeature.length)

    // Add some samples and make sure that we get the right statistics accumulated.
    varStats2.addUnsignedByteSample(
      treeId = 0,
      nodeId = 13,
      sample = (1.0, Array[Byte](Discretizer.convertToUnsignedByte(4), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(15), Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5)), Array[Byte](2, 3)))

    varStats2.addUnsignedByteSample(
      treeId = 0,
      nodeId = 14,
      sample = (2.0, Array[Byte](Discretizer.convertToUnsignedByte(4), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(15), Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5)), Array[Byte](2, 3)))

    varStats2.addUnsignedByteSample(
      treeId = 0,
      nodeId = 16,
      sample = (4.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(11), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(6), Discretizer.convertToUnsignedByte(7)), Array[Byte](5, 1)))

    varStats2.addUnsignedByteSample(
      treeId = 0,
      nodeId = 16,
      sample = (4.0, Array[Byte](Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(4)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 8,
      sample = (1.0, Array[Byte](Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(8)), Array[Byte](4, 2)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 7,
      sample = (3.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 10,
      sample = (3.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 10,
      sample = (4.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(11), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 10,
      sample = (3.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 11,
      sample = (3.0, Array[Byte](Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 11,
      sample = (3.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 11,
      sample = (2.0, Array[Byte](Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 2)))

    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 0)._1 === 0.0)
    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 0)._2 === 0.0)
    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 0)._3 === 0.0)
    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 4)._1 === 1.0 * 2.0)
    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 4)._2 === 1.0 * 2.0)
    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 4)._3 === 2.0)

    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 0)._1 === 0.0)
    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 0)._2 === 0.0)
    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 0)._3 === 0.0)
    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 1)._1 === 3.0 * 5.0)
    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 1)._2 === 9.0 * 5.0)
    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 1)._3 === 5.0)

    val nextNodeIdsPerTree = Array[Int](21, 23)

    // Now, get the splits.
    val splits = varStats2.computeNodePredictionsAndSplits(
      featureBins,
      nextNodeIdsPerTree,
      nodeDepths,
      SequoiaForestOptions(
        numTrees = 2,
        treeType = TreeType.Regression_Variance,
        mtry = 6,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 100,
        localTrainThreshold = 100000,
        numSubTreesPerIteration = 3,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = None,
        imputationType = ImputationType.SplitOnMissing),
      new scala.util.Random(17))

    assert(nodeDepths(0).size === 0)
    assert(nodeDepths(1).size === 5) // There should be 6 child nodes (due to all binary splits and NaN bins).

    val split1 = splits.next()
    assert(split1.treeId === 0)
    assert(split1.nodeId === 13)
    assert(split1.prediction === 1.0)
    assert(split1.weight === 2.0)
    assert(split1.impurity === 0.0)
    assert(split1.splitImpurity === None)
    assert(split1.nodeSplit === None)

    val split2 = splits.next()
    assert(split2.treeId === 0)
    assert(split2.nodeId === 14)
    assert(split2.prediction === 2.0)
    assert(split2.weight === 2.0)
    assert(split2.impurity === 0.0)
    assert(split2.splitImpurity === None)
    assert(split2.nodeSplit === None)

    val split3 = splits.next()
    assert(split3.treeId === 0)
    assert(split3.nodeId === 16)
    assert(split3.prediction === 4.0)
    assert(split3.weight === 8.0)
    assert(split3.impurity === 0.0)
    assert(split3.splitImpurity === None)
    assert(split3.nodeSplit === None)

    val split4 = splits.next()
    assert(split4.treeId === 1)
    assert(split4.nodeId === 7)
    assert(split4.prediction === 3.0)
    assert(split4.weight === 5.0)
    assert(split4.impurity === 0.0)
    assert(split4.splitImpurity === None)
    assert(split4.nodeSplit === None)

    val split5 = splits.next()
    assert(split5.treeId === 1)
    assert(split5.nodeId === 8)
    assert(split5.prediction === 1.0)
    assert(split5.weight === 2.0)
    assert(split5.impurity === 0.0)
    assert(split5.splitImpurity === None)
    assert(split5.nodeSplit === None)

    val split6 = splits.next()
    assert(split6.treeId === 1)
    assert(split6.nodeId === 10)
    assert(compareDouble(split6.prediction, 3.33333))
    assert(split6.weight === 15.0)
    assert(compareDouble(split6.impurity, 0.22222))
    assert(split6.splitImpurity.get === 0.0)
    assert(split6.nodeSplit.get.featureId === 1)
    assert(split6.nodeSplit.get.parentNodeId === 10)
    assert(split6.nodeSplit.get.isInstanceOf[CategoricalSplitOnBinId])
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap.size === 3)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(11) === 24)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(3) === 23)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(5) === 23)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights.size === 2)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(23) === 10)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(24) === 5)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeSubTreeHash.size === 0)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(23) === -1)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(24) === -1)

    val split7 = splits.next()
    assert(compareDouble(split7.prediction, 2.83333))
    assert(split7.nodeId === 11)
    assert(compareDouble(split7.impurity, 0.138889))
    assert(split7.treeId === 1)
    assert(split7.splitImpurity.get === 0.0)
    assert(split7.nodeSplit.get.isInstanceOf[NumericSplitOnBinId])
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].parentNodeId === 11)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].featureId === 0)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].splitBinId === 3)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftId === 25)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightId === 26)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftWeight === 5)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightWeight === 2)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftSubTreeHash === -1)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightSubTreeHash === -1)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].nanBinId === 10)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].nanNodeId === 27)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].nanWeight === 5)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].nanSubTreeHash === -1)
  }

  test("Test VarianceStatistics with Missing Values Degenerate Cases") {
    val numTrees = 2
    val nodeSplitsPerTree = Array.fill[mutable.Queue[NodeSplitOnBinId]](numTrees)(mutable.Queue[NodeSplitOnBinId]())

    nodeSplitsPerTree(0).enqueue(NumericSplitOnBinId(2, 3, 5, 13, 14, 2.0, 2.0, -1, -1), NumericSplitOnBinId(5, 1, 2, 15, 16, 3.0, 6.0, 0, -1))
    nodeSplitsPerTree(1).enqueue(NumericSplitOnBinId(3, 4, 2, 7, 8, 2.0, 5.0, -1, -1), CategoricalSplitOnBinId(4, 3, mutable.Map[Int, Int](1 -> 10, 4 -> 11), mutable.Map[Int, Double](10 -> 3.0, 11 -> 2.0), mutable.Map[Int, Int]()))

    val nodeDepths = Array.fill[mutable.Map[Int, Int]](2)(mutable.Map[Int, Int]())
    nodeDepths(0).put(13, 5)
    nodeDepths(0).put(14, 5)
    nodeDepths(0).put(16, 7)
    nodeDepths(1).put(7, 3)
    nodeDepths(1).put(8, 3)
    nodeDepths(1).put(10, 6)
    nodeDepths(1).put(11, 6)

    val scheduledLookup = ScheduledNodeSplitLookup.createLookupForNodeSplits(nodeSplitsPerTree, 100)
    val treeSeeds = new Array[Int](numTrees)
    var treeId = 0
    val randGen = scala.util.Random
    while (treeId < numTrees) {
      treeSeeds(treeId) = randGen.nextInt()
      treeId += 1
    }

    val mtry = 4
    val numBinsPerFeature = Array[Int](11, 12, 17, 13, 15, 9)

    val featureBins = new Array[Bins](6)
    var featId = 0
    while (featId < numBinsPerFeature.length) {
      if (featId != 1) {
        val bins = mutable.ArrayBuffer[NumericBin]()
        bins += NumericBin(Double.NegativeInfinity, 0)
        val numBinsWithoutNaN = if (featId == 0) {
          numBinsPerFeature(featId) - 1
        } else {
          numBinsPerFeature(featId)
        }

        while (bins.length < numBinsWithoutNaN) {
          bins += NumericBin(bins.length - 1, bins.length)
        }

        if (featId == 0) {
          featureBins(featId) = NumericBins(bins.toArray, numBinsPerFeature(featId) - 1)
        } else {
          featureBins(featId) = NumericBins(bins.toArray)
        }
      } else {
        featureBins(featId) = CategoricalBins(numBinsPerFeature(featId) - 1, 11) // The last one is missing value bin.
      }

      featId += 1
    }

    // Create a variance statistics object with all the features per node.
    val varStats2 = new VarianceStatistics(
      scheduledLookup,
      numBinsPerFeature,
      treeSeeds,
      numBinsPerFeature.length)

    // Add some samples and make sure that we get the right statistics accumulated.
    varStats2.addUnsignedByteSample(
      treeId = 0,
      nodeId = 13,
      sample = (1.0, Array[Byte](Discretizer.convertToUnsignedByte(4), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(15), Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5)), Array[Byte](2, 3)))

    varStats2.addUnsignedByteSample(
      treeId = 0,
      nodeId = 14,
      sample = (2.0, Array[Byte](Discretizer.convertToUnsignedByte(4), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(15), Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5)), Array[Byte](2, 3)))

    varStats2.addUnsignedByteSample(
      treeId = 0,
      nodeId = 16,
      sample = (4.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(11), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(6), Discretizer.convertToUnsignedByte(7)), Array[Byte](5, 1)))

    varStats2.addUnsignedByteSample(
      treeId = 0,
      nodeId = 16,
      sample = (4.0, Array[Byte](Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(4)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 8,
      sample = (1.0, Array[Byte](Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(8)), Array[Byte](4, 2)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 7,
      sample = (3.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 10,
      sample = (3.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 10,
      sample = (4.0, Array[Byte](Discretizer.convertToUnsignedByte(2), Discretizer.convertToUnsignedByte(11), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 11,
      sample = (3.0, Array[Byte](Discretizer.convertToUnsignedByte(10), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 5)))

    varStats2.addUnsignedByteSample(
      treeId = 1,
      nodeId = 11,
      sample = (2.0, Array[Byte](Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(5), Discretizer.convertToUnsignedByte(7), Discretizer.convertToUnsignedByte(1), Discretizer.convertToUnsignedByte(3), Discretizer.convertToUnsignedByte(2)), Array[Byte](3, 2)))

    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 0)._1 === 0.0)
    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 0)._2 === 0.0)
    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 0)._3 === 0.0)
    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 4)._1 === 1.0 * 2.0)
    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 4)._2 === 1.0 * 2.0)
    assert(varStats2.getBinStats(treeId = 0, nodeId = 13, featureId = 0, binId = 4)._3 === 2.0)

    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 0)._1 === 0.0)
    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 0)._2 === 0.0)
    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 0)._3 === 0.0)
    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 1)._1 === 3.0 * 5.0)
    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 1)._2 === 9.0 * 5.0)
    assert(varStats2.getBinStats(treeId = 1, nodeId = 7, featureId = 3, binId = 1)._3 === 5.0)

    val nextNodeIdsPerTree = Array[Int](21, 23)

    // Now, get the splits.
    val splits = varStats2.computeNodePredictionsAndSplits(
      featureBins,
      nextNodeIdsPerTree,
      nodeDepths,
      SequoiaForestOptions(
        numTrees = 2,
        treeType = TreeType.Regression_Variance,
        mtry = 6,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 100,
        localTrainThreshold = 100000,
        numSubTreesPerIteration = 3,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = None,
        imputationType = ImputationType.SplitOnMissing),
      new scala.util.Random(17))

    assert(nodeDepths(0).size === 0)
    assert(nodeDepths(1).size === 4) // There should be 6 child nodes (due to all binary splits and NaN bins).

    val split1 = splits.next()
    assert(split1.treeId === 0)
    assert(split1.nodeId === 13)
    assert(split1.prediction === 1.0)
    assert(split1.weight === 2.0)
    assert(split1.impurity === 0.0)
    assert(split1.splitImpurity === None)
    assert(split1.nodeSplit === None)

    val split2 = splits.next()
    assert(split2.treeId === 0)
    assert(split2.nodeId === 14)
    assert(split2.prediction === 2.0)
    assert(split2.weight === 2.0)
    assert(split2.impurity === 0.0)
    assert(split2.splitImpurity === None)
    assert(split2.nodeSplit === None)

    val split3 = splits.next()
    assert(split3.treeId === 0)
    assert(split3.nodeId === 16)
    assert(split3.prediction === 4.0)
    assert(split3.weight === 8.0)
    assert(split3.impurity === 0.0)
    assert(split3.splitImpurity === None)
    assert(split3.nodeSplit === None)

    val split4 = splits.next()
    assert(split4.treeId === 1)
    assert(split4.nodeId === 7)
    assert(split4.prediction === 3.0)
    assert(split4.weight === 5.0)
    assert(split4.impurity === 0.0)
    assert(split4.splitImpurity === None)
    assert(split4.nodeSplit === None)

    val split5 = splits.next()
    assert(split5.treeId === 1)
    assert(split5.nodeId === 8)
    assert(split5.prediction === 1.0)
    assert(split5.weight === 2.0)
    assert(split5.impurity === 0.0)
    assert(split5.splitImpurity === None)
    assert(split5.nodeSplit === None)

    val split6 = splits.next()
    assert(split6.treeId === 1)
    assert(split6.nodeId === 10)
    assert(compareDouble(split6.prediction, 3.5))
    assert(split6.weight === 10.0)
    assert(compareDouble(split6.impurity, 0.25))
    assert(split6.splitImpurity.get === 0.0)
    assert(split6.nodeSplit.get.featureId === 1)
    assert(split6.nodeSplit.get.parentNodeId === 10)
    assert(split6.nodeSplit.get.isInstanceOf[CategoricalSplitOnBinId])
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap.size === 2)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(11) === 24)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].binIdToNodeIdMap(5) === 23)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights.size === 2)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(23) === 5)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeWeights(24) === 5)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].nodeSubTreeHash.size === 0)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(23) === -1)
    assert(split6.nodeSplit.get.asInstanceOf[CategoricalSplitOnBinId].getSubTreeHash(24) === -1)

    val split7 = splits.next()
    assert(compareDouble(split7.prediction, 2.714286))
    assert(split7.nodeId === 11)
    assert(compareDouble(split7.impurity, 0.2040801))
    assert(split7.treeId === 1)
    assert(split7.splitImpurity.get === 0.0)
    assert(split7.nodeSplit.get.isInstanceOf[NumericSplitOnBinId])
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].parentNodeId === 11)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].featureId === 0)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].splitBinId === 0)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftId === -1)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightId === 25)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftWeight === 0.0)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightWeight === 2)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].leftSubTreeHash === -1)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].rightSubTreeHash === -1)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].nanBinId === 10)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].nanNodeId === 26)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].nanWeight === 5)
    assert(split7.nodeSplit.get.asInstanceOf[NumericSplitOnBinId].nanSubTreeHash === -1)
  }
}
