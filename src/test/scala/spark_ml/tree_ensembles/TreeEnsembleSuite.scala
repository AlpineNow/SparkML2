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

import scala.reflect.ClassTag

import org.apache.spark.storage.StorageLevel
import org.scalatest.FunSuite
import spark_ml.discretization.{BinFinder, Discretizer, EqualWidthBinFinder}
import spark_ml.model.rf.{RandomForest, RandomForestStore}
import spark_ml.model.{DecisionTree, DecisionTreeNode}
import spark_ml.util._

/**
 * Test TreeEnsemble Trainer.
 */
class TreeEnsembleSuite extends FunSuite with LocalSparkContext {
  def trainForests[T: ClassTag](
    data: Array[(Double, Array[Double])],
    columnNames: (String, Array[String]),
    catIndices: Set[Int],
    binFinder: BinFinder,
    maxNumBins: Int,
    featureHandler: DiscretizedFeatureHandler[T],
    expectedLabelCardinality: Option[Int],
    splitCriteria: SplitCriteria.SplitCriteria,
    numClasses: Option[Int],
    catSplitType: CatSplitType.CatSplitType,
    subTreeWeightThreshold: Double
  ): (RandomForest, RandomForest, RandomForest) = {
    val testDataRdd = sc.parallelize(data, 3)
    val (labelSummary, featureBins) = binFinder.findBins(
      data = testDataRdd,
      columnNames = columnNames,
      catIndices = catIndices,
      maxNumBins = maxNumBins,
      expectedLabelCardinality = expectedLabelCardinality,
      notifiee = new ConsoleNotifiee
    )
    val featureBinsAsArray = featureBins.toArray
    assert(labelSummary.expectedCardinality.equals(expectedLabelCardinality))
    val labelRdd = testDataRdd.map(_._1)
    val discretizedFeatureRdd =
      Discretizer.transformFeatures[T](
        input = testDataRdd,
        featureBins = featureBins,
        featureHandler = featureHandler
      )
    val bagRdd =
      Bagger.getBagRdd(
        data = discretizedFeatureRdd,
        numSamples = 1,
        baggingType = BaggingType.WithoutReplacement,
        baggingRate = 1.0,
        seed = 0
      )
    val idCache = IdCache.createIdCache(
      numTrees = 1,
      data = labelRdd,
      storageLevel = StorageLevel.MEMORY_AND_DISK,
      checkpointDir = None,
      checkpointInterval = 1
    )
    val idCacheForSubTrees = IdCache.createIdCache(
      numTrees = 1,
      data = labelRdd,
      storageLevel = StorageLevel.MEMORY_AND_DISK,
      checkpointDir = None,
      checkpointInterval = 1
    )
    // Test both local and distributed training.
    val quantizedData_Local = new QuantizedData_ForTrees_Local(
      numTrees = 1,
      data = labelRdd.zip(discretizedFeatureRdd).zip(bagRdd).collect(),
      featureBinsInfo = featureBinsAsArray,
      typeHandler = featureHandler
    )
    val quantizedData_Rdd = new QuantizedData_ForTrees_Rdd(
      data = labelRdd.zip(discretizedFeatureRdd).zip(bagRdd),
      idCache = idCache,
      featureBinsInfo = featureBinsAsArray,
      featureHandler = featureHandler
    )
    val quantizedDataForSubTrees_Rdd = new QuantizedData_ForTrees_Rdd(
      data = labelRdd.zip(discretizedFeatureRdd).zip(bagRdd),
      idCache = idCacheForSubTrees,
      featureBinsInfo = featureBinsAsArray,
      featureHandler = featureHandler
    )
    val rfStore_Local = new RandomForestStore(
      splitCriteria = splitCriteria,
      featureNames = columnNames._2,
      featureBins = featureBinsAsArray
    )
    val rfStore = new RandomForestStore(
      splitCriteria = splitCriteria,
      featureNames = columnNames._2,
      featureBins = featureBinsAsArray
    )
    val rfStoreWithSubTrees = new RandomForestStore(
      splitCriteria = splitCriteria,
      featureNames = columnNames._2,
      featureBins = featureBinsAsArray
    )
    val treeOpts = TreeForestTrainerOptions(
      numTrees = 1,
      splitCriteria = splitCriteria,
      mtry = 100,
      minSplitSize = 2,
      maxDepth = Int.MaxValue,
      catSplitType = catSplitType,
      maxSplitsPerIter = 2,
      subTreeWeightThreshold = 0.0,
      maxSubTreesPerIter = 0,
      numClasses = numClasses,
      verbose = true
    )
    val treeOptsWithSubTrees = TreeForestTrainerOptions(
      numTrees = 1,
      splitCriteria = splitCriteria,
      mtry = 100,
      minSplitSize = 2,
      maxDepth = Int.MaxValue,
      catSplitType = catSplitType,
      maxSplitsPerIter = 2,
      subTreeWeightThreshold = subTreeWeightThreshold,
      maxSubTreesPerIter = 2,
      numClasses = numClasses,
      verbose = true
    )

    // Local training.
    TreeForestTrainer.train(
      trainingData = quantizedData_Local,
      featureBinsInfo = featureBinsAsArray,
      trainingOptions = treeOpts,
      modelStore = rfStore_Local,
      notifiee = new ConsoleNotifiee,
      rng = new scala.util.Random(16)
    )

    // Distributed training.
    TreeForestTrainer.train(
      trainingData = quantizedData_Rdd,
      featureBinsInfo = featureBinsAsArray,
      trainingOptions = treeOpts,
      modelStore = rfStore,
      notifiee = new ConsoleNotifiee,
      rng = new scala.util.Random(16)
    )

    // Distributed training with sub-trees.
    TreeForestTrainer.train(
      trainingData = quantizedDataForSubTrees_Rdd,
      featureBinsInfo = featureBinsAsArray,
      trainingOptions = treeOptsWithSubTrees,
      modelStore = rfStoreWithSubTrees,
      notifiee = new ConsoleNotifiee,
      rng = new scala.util.Random(16)
    )

    (
      rfStore_Local.createRandomForest,
      rfStore.createRandomForest,
      rfStoreWithSubTrees.createRandomForest
    )
  }

  def assertTwoTreeNodeEquality(
    nodes1: Map[java.lang.Integer, DecisionTreeNode],
    nodes2: Map[java.lang.Integer, DecisionTreeNode],
    treeNode1: DecisionTreeNode,
    treeNode2: DecisionTreeNode): Unit = {
    assert(treeNode1.featIdx.equals(treeNode2.featIdx))
    assert(treeNode1.impurity === treeNode2.impurity)
    assert(treeNode1.isFeatNumeric.equals(treeNode2.isFeatNumeric))
    assert(treeNode1.nodeWeight === treeNode2.nodeWeight)
    assert(treeNode1.prediction === treeNode2.prediction)
    assert(treeNode1.splitImpurity.equals(treeNode2.splitImpurity))
    assert(treeNode1.split.equals(treeNode2.split))
    if (treeNode1.leftChild.isDefined) {
      assertTwoTreeNodeEquality(nodes1, nodes2, nodes1(treeNode1.leftChild.get), nodes2(treeNode2.leftChild.get))
    }
    if (treeNode1.rightChild.isDefined) {
      assertTwoTreeNodeEquality(nodes1, nodes2, nodes1(treeNode1.rightChild.get), nodes2(treeNode2.rightChild.get))
    }
    if (treeNode1.nanChild.isDefined) {
      assertTwoTreeNodeEquality(nodes1, nodes2, nodes1(treeNode1.nanChild.get), nodes2(treeNode2.nanChild.get))
    }
    if (treeNode1.children.isDefined) {
      assert(treeNode1.children.get.size === treeNode2.children.get.size)
      treeNode1.children.get.keys.foreach {
        tn1cKey =>
          assert(treeNode2.children.get.contains(tn1cKey))
          assertTwoTreeNodeEquality(nodes1, nodes2, nodes1(treeNode1.children.get(tn1cKey)), nodes2(treeNode2.children.get(tn1cKey)))
      }
    }
  }

  def assertTwoTreeEquality(
    tree1: DecisionTree,
    tree2: DecisionTree): Unit = {
    assert(tree1.nodeCount === tree2.nodeCount)
    assertTwoTreeNodeEquality(tree1.nodes, tree2.nodes, tree1.nodes(1), tree2.nodes(1))
  }

  def assertTwoForestEquality(
    model1: RandomForest,
    model2: RandomForest): Unit = {
    assert(model1.splitCriteriaStr.equals(model2.splitCriteriaStr))
    assert(model1.trees.length === model2.trees.length)
    model1.sampleCounts.zip(model2.sampleCounts).foreach {
      case (model1Cnt, model2Cnt) => assert(model1Cnt === model2Cnt)
    }
    model1.sortedVarImportance.zip(model2.sortedVarImportance).foreach {
      case ((model1VarName, model1VarImp), (model2VarName, model2VarImp)) =>
        assert(model1VarName.equals(model2VarName))
        assert(model1VarImp === model2VarImp)
    }
    model1.trees.zip(model2.trees).foreach {
      case (model1Tree, model2Tree) => assertTwoTreeEquality(model1Tree, model2Tree)
    }
  }

  test("Test InfoGain Classification") {
    val labeledData2 = TestDataGenerator.labeledData2
    val (model_local, model, modelWithSubTrees) =
      trainForests(
        data = labeledData2,
        columnNames = ("testLabel", Array("testCol1", "testCol2")),
        catIndices = Set[Int](1),
        binFinder = new EqualWidthBinFinder,
        maxNumBins = 5,
        featureHandler = new UnsignedByteHandler,
        expectedLabelCardinality = Some(4),
        splitCriteria = SplitCriteria.Classification_InfoGain,
        numClasses = Some(4),
        catSplitType = CatSplitType.MultiwaySplit,
        subTreeWeightThreshold = 10.0
      )

    assertTwoForestEquality(model_local, model)
    assertTwoForestEquality(model_local, modelWithSubTrees)

    assert(model_local.trees.length === 1)
    assert(model_local.trees(0).nodeCount === 16)
    assert(model_local.trees(0).nodes(1).prediction === 2.0)
    assert(numbersAreEqual(model_local.trees(0).nodes(1).impurity, 1.916716186961402))
    TestDataGenerator.labeledData2.foreach(row => assert(model_local.predict(row._2)(0)._1 === row._1))

    assert(model.trees.length === 1)
    assert(model.trees(0).nodeCount === 16)
    assert(model.trees(0).nodes(1).prediction === 2.0)
    assert(numbersAreEqual(model.trees(0).nodes(1).impurity, 1.916716186961402))
    labeledData2.foreach(row => assert(model.predict(row._2)(0)._1 === row._1))

    assert(modelWithSubTrees.trees.length === 1)
    assert(modelWithSubTrees.trees(0).nodeCount === 16)
    assert(modelWithSubTrees.trees(0).nodes(1).prediction === 2.0)
    assert(numbersAreEqual(modelWithSubTrees.trees(0).nodes(1).impurity, 1.916716186961402))
    labeledData2.foreach(row => assert(modelWithSubTrees.predict(row._2)(0)._1 === row._1))

    val (model_local_binaryCatSplits, model_binaryCatSplits, modelWithSubTrees_binaryCatSplits) =
      trainForests(
        data = labeledData2,
        columnNames = ("testLabel", Array("testCol1", "testCol2")),
        catIndices = Set[Int](1),
        binFinder = new EqualWidthBinFinder,
        maxNumBins = 5,
        featureHandler = new UnsignedByteHandler,
        expectedLabelCardinality = Some(4),
        splitCriteria = SplitCriteria.Classification_InfoGain,
        numClasses = Some(4),
        catSplitType = CatSplitType.RandomBinarySplit,
        subTreeWeightThreshold = 10.0
      )

    assertTwoForestEquality(model_local_binaryCatSplits, model_binaryCatSplits)

    assert(model_local_binaryCatSplits.trees.length === 1)
    assert(model_local_binaryCatSplits.trees(0).nodeCount === 21)
    assert(model_local_binaryCatSplits.trees(0).nodes(1).prediction === 2.0)
    assert(numbersAreEqual(model_local_binaryCatSplits.trees(0).nodes(1).impurity, 1.916716186961402))
    labeledData2.foreach(row => assert(model_local_binaryCatSplits.predict(row._2)(0)._1 === row._1))

    assert(model_binaryCatSplits.trees.length === 1)
    assert(model_binaryCatSplits.trees(0).nodeCount === 21)
    assert(model_binaryCatSplits.trees(0).nodes(1).prediction === 2.0)
    assert(numbersAreEqual(model_binaryCatSplits.trees(0).nodes(1).impurity, 1.916716186961402))
    labeledData2.foreach(row => assert(model_binaryCatSplits.predict(row._2)(0)._1 === row._1))

    assert(modelWithSubTrees_binaryCatSplits.trees.length === 1)
    assert(modelWithSubTrees_binaryCatSplits.trees(0).nodeCount === 23)
    assert(modelWithSubTrees_binaryCatSplits.trees(0).nodes(1).prediction === 2.0)
    assert(numbersAreEqual(modelWithSubTrees_binaryCatSplits.trees(0).nodes(1).impurity, 1.916716186961402))
    labeledData2.foreach(row => assert(modelWithSubTrees_binaryCatSplits.predict(row._2)(0)._1 === row._1))
  }

  test("Test Variance Regression") {
    val labeledData2 = TestDataGenerator.labeledData2
    val (model_local, model, modelWithSubTrees) =
      trainForests(
        data = labeledData2,
        columnNames = ("testLabel", Array("testCol1", "testCol2")),
        catIndices = Set[Int](1),
        binFinder = new EqualWidthBinFinder,
        maxNumBins = 5,
        featureHandler = new UnsignedByteHandler,
        expectedLabelCardinality = Some(4),
        splitCriteria = SplitCriteria.Regression_Variance,
        numClasses = None,
        catSplitType = CatSplitType.MultiwaySplit,
        subTreeWeightThreshold = 10.0
      )

    assertTwoForestEquality(model_local, model)
    assertTwoForestEquality(model_local, modelWithSubTrees)

    assert(model_local.trees.length === 1)
    assert(model_local.trees(0).nodeCount === 16)
    assert(numbersAreEqual(model_local.trees(0).nodes(1).prediction, 1.4))
    assert(numbersAreEqual(model_local.trees(0).nodes(1).impurity, 0.9733333))
    assert(model_local.trees(0).nodes(1).nodeWeight === 30)
    labeledData2.foreach(row => assert(model_local.predict(row._2)(0)._1 === row._1))

    assert(model.trees.length === 1)
    assert(model.trees(0).nodeCount === 16)
    assert(numbersAreEqual(model.trees(0).nodes(1).prediction, 1.4))
    assert(numbersAreEqual(model.trees(0).nodes(1).impurity, 0.9733333))
    assert(model.trees(0).nodes(1).nodeWeight === 30)
    labeledData2.foreach(row => assert(model.predict(row._2)(0)._1 === row._1))

    assert(modelWithSubTrees.trees.length === 1)
    assert(modelWithSubTrees.trees(0).nodeCount === 16)
    assert(numbersAreEqual(modelWithSubTrees.trees(0).nodes(1).prediction, 1.4))
    assert(numbersAreEqual(modelWithSubTrees.trees(0).nodes(1).impurity, 0.9733333))
    assert(modelWithSubTrees.trees(0).nodes(1).nodeWeight === 30)
    labeledData2.foreach(row => assert(modelWithSubTrees.predict(row._2)(0)._1 === row._1))

    val (model_local_orderedCatSplits, model_orderedCatSplits, modelWithSubTrees_orderedCatSplits) =
      trainForests(
        data = labeledData2,
        columnNames = ("testLabel", Array("testCol1", "testCol2")),
        catIndices = Set[Int](1),
        binFinder = new EqualWidthBinFinder,
        maxNumBins = 5,
        featureHandler = new UnsignedByteHandler,
        expectedLabelCardinality = Some(4),
        splitCriteria = SplitCriteria.Regression_Variance,
        numClasses = None,
        catSplitType = CatSplitType.OrderedBinarySplit,
        subTreeWeightThreshold = 10.0
      )

    assertTwoForestEquality(model_local_orderedCatSplits, model_orderedCatSplits)
    assertTwoForestEquality(model_local_orderedCatSplits, modelWithSubTrees_orderedCatSplits)

    assert(model_local_orderedCatSplits.trees.length === 1)
    assert(model_local_orderedCatSplits.trees(0).nodeCount === 19)
    assert(numbersAreEqual(model_local_orderedCatSplits.trees(0).nodes(1).prediction, 1.4))
    assert(numbersAreEqual(model_local_orderedCatSplits.trees(0).nodes(1).impurity, 0.9733333))
    assert(model_local_orderedCatSplits.trees(0).nodes(1).nodeWeight === 30.0)
    labeledData2.foreach(row => assert(model_local_orderedCatSplits.predict(row._2)(0)._1 === row._1))

    assert(model_orderedCatSplits.trees.length === 1)
    assert(model_orderedCatSplits.trees(0).nodeCount === 19)
    assert(numbersAreEqual(model_orderedCatSplits.trees(0).nodes(1).prediction, 1.4))
    assert(numbersAreEqual(model_orderedCatSplits.trees(0).nodes(1).impurity, 0.9733333))
    assert(model_orderedCatSplits.trees(0).nodes(1).nodeWeight === 30.0)
    labeledData2.foreach(row => assert(model_orderedCatSplits.predict(row._2)(0)._1 === row._1))

    assert(modelWithSubTrees_orderedCatSplits.trees.length === 1)
    assert(modelWithSubTrees_orderedCatSplits.trees(0).nodeCount === 19)
    assert(numbersAreEqual(modelWithSubTrees_orderedCatSplits.trees(0).nodes(1).prediction, 1.4))
    assert(numbersAreEqual(modelWithSubTrees_orderedCatSplits.trees(0).nodes(1).impurity, 0.9733333))
    assert(modelWithSubTrees_orderedCatSplits.trees(0).nodes(1).nodeWeight === 30.0)
    labeledData2.foreach(row => assert(modelWithSubTrees_orderedCatSplits.predict(row._2)(0)._1 === row._1))
  }
}
