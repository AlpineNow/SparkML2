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

import spark_ml.util._
import org.scalatest.FunSuite
import spark_ml.discretization._
import java.io.{ ByteArrayInputStream, ByteArrayOutputStream }

/**
 * Test Sequoia Forest training.
 */
class SequoiaForestSuite extends FunSuite with LocalSparkContext {
  test("Train a tree 1 - unsigned Byte features RDD") {
    val testDataRDD = sc.parallelize(TestDataGenerator.labeledData2, 3)
    val testDataRaw = testDataRDD.collect()
    val (maxLabelValue, bins) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](1),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.MaxCardinality_Categoric -> "1000"))

    assert(maxLabelValue === 3.0)

    val txData = Discretizer.transformFeaturesToUnsignedByteBinIds(testDataRDD, bins)

    // No bagging (100% sampling without replacement).
    val inputRDD = Bagger.bagRDD[Byte](txData, 1, SamplingType.SampleWithoutReplacement, 1.0, 0)

    // Train without local sub-tree training.
    val forest = SequoiaForestTrainer.train(
      UnsignedByteRDD(inputRDD),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Classification_InfoGain,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 2,
        localTrainThreshold = 0,
        numSubTreesPerIteration = 0,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(4),
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest.trees.length === 1)
    assert(forest.trees(0).getNodeCount === 23)
    assert(forest.trees(0).subTrees.size === 0)
    assert(forest.trees(0).nodes(1).prediction === 2.0)
    assert(compareDouble(forest.trees(0).nodes(1).impurity, 1.916716186961402))

    testDataRaw.foreach(row => assert(forest.predict(row._2)(0)._1 === row._1))

    // Train with local sub-tree training.
    val forest2 = SequoiaForestTrainer.train(
      UnsignedByteRDD(inputRDD),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Classification_InfoGain,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 1000,
        localTrainThreshold = 10,
        numSubTreesPerIteration = 1000,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(4),
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest2.trees.length === 1)
    assert(forest2.trees(0).getNodeCount === 21)
    assert(forest2.trees(0).subTrees.size === 4)
    assert(forest2.trees(0).nodes(1).prediction === 2.0)
    assert(compareDouble(forest2.trees(0).nodes(1).impurity, 1.916716186961402))

    testDataRaw.foreach(row => assert(forest2.predict(row._2)(0)._1 === row._1))

    // Train without local sub-tree training.
    val forest3 = SequoiaForestTrainer.train(
      UnsignedByteRDD(inputRDD),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Classification_InfoGain,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = 3,
        numNodesPerIteration = 2,
        localTrainThreshold = 0,
        numSubTreesPerIteration = 0,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(4),
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest3.trees.length === 1)
    assert(forest3.trees(0).getNodeCount === 7)
    assert(forest3.trees(0).subTrees.size === 0)
    assert(forest3.trees(0).nodes(1).prediction === 2.0)
    assert(compareDouble(forest3.trees(0).nodes(1).impurity, 1.916716186961402))

    // Train with local sub-tree training.
    val forest4 = SequoiaForestTrainer.train(
      UnsignedByteRDD(inputRDD),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Classification_InfoGain,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = 3,
        numNodesPerIteration = 1000,
        localTrainThreshold = 10,
        numSubTreesPerIteration = 1000,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(4),
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest4.trees.length === 1)
    assert(forest4.trees(0).getNodeCount === 7)
    assert(forest4.trees(0).subTrees.size === 2)
    assert(forest4.trees(0).nodes(1).prediction === 2.0)
    assert(compareDouble(forest4.trees(0).nodes(1).impurity, 1.916716186961402))

    // This is no longer true because of random sorting for multi-class classification for categorical features.
    // testDataRaw.foreach(row => assert(forest3.predict(row._2)(0)._1 === forest4.predict(row._2)(0)._1))

    // Write tree to a local output stream.
    val outputStream = new ByteArrayOutputStream()
    SequoiaForestWriter.writeTree(forest4.trees(0), outputStream)
    val treeBytes = outputStream.toByteArray
    outputStream.close()

    val inputStream = new ByteArrayInputStream(treeBytes)
    val readTree = SequoiaForestReader.readTree(inputStream)
    inputStream.close()

    assert(forest4.trees(0).nodes.size === readTree.nodes.size)
    assert(forest4.trees(0).subTrees.size === readTree.subTrees.size)

    testDataRaw.foreach(row => assert(forest4.trees(0).predict(row._2) === readTree.predict(row._2)))

    // Make sure that the variable importances are as expected.
    assert(compareDouble(forest.varImportance.featureImportance(0), 34.8883))
    assert(compareDouble(forest.varImportance.featureImportance(1), 22.6132))
    assert(compareDouble(forest2.varImportance.featureImportance(0), 34.8883))
    assert(compareDouble(forest2.varImportance.featureImportance(1), 22.6132))
    assert(compareDouble(forest3.varImportance.featureImportance(0), 7.2368))
    assert(compareDouble(forest3.varImportance.featureImportance(1), 15.1613))
    assert(compareDouble(forest4.varImportance.featureImportance(0), 7.2368))
    assert(compareDouble(forest4.varImportance.featureImportance(1), 17.9161))
  }

  test("Train a tree 2 - unsigned Short features RDD") {
    val testDataRDD = sc.parallelize(TestDataGenerator.labeledData2, 3)
    val testDataRaw = testDataRDD.collect()
    val (maxLabelValue, bins) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](1),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.MaxCardinality_Categoric -> "1000"))

    assert(maxLabelValue === 3.0)

    val txData = Discretizer.transformFeaturesToUnsignedShortBinIds(testDataRDD, bins)

    // No bagging (100% sampling without replacement).
    val inputRDD = Bagger.bagRDD[Short](txData, 1, SamplingType.SampleWithoutReplacement, 1.0, 0)

    // Train without local sub-tree training.
    val forest = SequoiaForestTrainer.train(
      UnsignedShortRDD(inputRDD),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Classification_InfoGain,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 2,
        localTrainThreshold = 0,
        numSubTreesPerIteration = 0,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(4),
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest.trees.length === 1)
    assert(forest.trees(0).getNodeCount === 23)
    assert(forest.trees(0).subTrees.size === 0)
    assert(forest.trees(0).nodes(1).prediction === 2.0)
    assert(compareDouble(forest.trees(0).nodes(1).impurity, 1.916716186961402))

    testDataRaw.foreach(row => assert(forest.predict(row._2)(0)._1 === row._1))

    // Train with local sub-tree training.
    val forest2 = SequoiaForestTrainer.train(
      UnsignedShortRDD(inputRDD),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Classification_InfoGain,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 1000,
        localTrainThreshold = 10,
        numSubTreesPerIteration = 1000,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(4),
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest2.trees.length === 1)
    assert(forest2.trees(0).getNodeCount === 21)
    assert(forest2.trees(0).subTrees.size === 4)
    assert(forest2.trees(0).nodes(1).prediction === 2.0)
    assert(compareDouble(forest2.trees(0).nodes(1).impurity, 1.916716186961402))

    testDataRaw.foreach(row => assert(forest2.predict(row._2)(0)._1 === row._1))

    // Make sure that the variable importances are as expected.
    assert(compareDouble(forest.varImportance.featureImportance(0), 34.8883))
    assert(compareDouble(forest.varImportance.featureImportance(1), 22.6132))
    assert(compareDouble(forest2.varImportance.featureImportance(0), 34.8883))
    assert(compareDouble(forest2.varImportance.featureImportance(1), 22.6132))
  }

  test("Train a tree 3 - unsigned Byte features Local") {
    val testDataRDD = sc.parallelize(TestDataGenerator.labeledData2, 3)
    val testDataRaw = testDataRDD.collect()
    val (maxLabelValue, bins) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](1),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.MaxCardinality_Categoric -> "1000"))

    assert(maxLabelValue === 3.0)

    val txDataLocal = Discretizer.transformFeaturesToUnsignedByteBinIds(testDataRDD, bins).collect()

    // No bagging (100% sampling without replacement).
    val baggedInput = Bagger.bagArray[Byte](txDataLocal, 1, SamplingType.SampleWithoutReplacement, 1.0, 0)
    val inputLocal = baggedInput.map(row => (row, Array.fill[Int](1)(0)))

    // Train without local sub-tree training.
    val forest = SequoiaForestTrainer.train(
      UnsignedByteLocal(inputLocal),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Classification_InfoGain,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 2,
        localTrainThreshold = 0,
        numSubTreesPerIteration = 0,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(4),
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest.trees.length === 1)
    assert(forest.trees(0).getNodeCount === 23)
    assert(forest.trees(0).subTrees.size === 0)
    assert(forest.trees(0).nodes(1).prediction === 2.0)
    assert(compareDouble(forest.trees(0).nodes(1).impurity, 1.916716186961402))

    testDataRaw.foreach(row => assert(forest.predict(row._2)(0)._1 === row._1))

    // Train with local sub-tree training.
    val forest2 = SequoiaForestTrainer.train(
      UnsignedByteLocal(inputLocal),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Classification_InfoGain,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 1000,
        localTrainThreshold = 10,
        numSubTreesPerIteration = 1000,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(4),
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest2.trees.length === 1)
    assert(forest2.trees(0).getNodeCount === 23)
    assert(forest2.trees(0).subTrees.size === 0)
    assert(forest2.trees(0).nodes(1).prediction === 2.0)
    assert(compareDouble(forest2.trees(0).nodes(1).impurity, 1.916716186961402))

    testDataRaw.foreach(row => assert(forest2.predict(row._2)(0)._1 === row._1))

    // Make sure that the variable importances are as expected.
    assert(compareDouble(forest.varImportance.featureImportance(0), 34.8883))
    assert(compareDouble(forest.varImportance.featureImportance(1), 22.6132))
    assert(compareDouble(forest2.varImportance.featureImportance(0), 34.8883))
    assert(compareDouble(forest2.varImportance.featureImportance(1), 22.6132))
  }

  test("Train a tree 4 - unsigned Short features Local") {
    val testDataRDD = sc.parallelize(TestDataGenerator.labeledData2, 3)
    val testDataRaw = testDataRDD.collect()
    val (maxLabelValue, bins) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](1),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.MaxCardinality_Categoric -> "1000"))

    assert(maxLabelValue === 3.0)

    val txDataLocal = Discretizer.transformFeaturesToUnsignedShortBinIds(testDataRDD, bins).collect()

    // No bagging (100% sampling without replacement).
    val baggedInput = Bagger.bagArray[Short](txDataLocal, 1, SamplingType.SampleWithoutReplacement, 1.0, 0)
    val inputLocal = baggedInput.map(row => (row, Array.fill[Int](1)(0)))

    // Train without local sub-tree training.
    val forest = SequoiaForestTrainer.train(
      UnsignedShortLocal(inputLocal),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Classification_InfoGain,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 2,
        localTrainThreshold = 0,
        numSubTreesPerIteration = 0,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(4),
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest.trees.length === 1)
    assert(forest.trees(0).getNodeCount === 23)
    assert(forest.trees(0).subTrees.size === 0)
    assert(forest.trees(0).nodes(1).prediction === 2.0)
    assert(compareDouble(forest.trees(0).nodes(1).impurity, 1.916716186961402))

    testDataRaw.foreach(row => assert(forest.predict(row._2)(0)._1 === row._1))

    // Train with local sub-tree training.
    val forest2 = SequoiaForestTrainer.train(
      UnsignedShortLocal(inputLocal),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Classification_InfoGain,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 1000,
        localTrainThreshold = 10,
        numSubTreesPerIteration = 1000,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(4),
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest2.trees.length === 1)
    assert(forest2.trees(0).getNodeCount === 23)
    assert(forest2.trees(0).subTrees.size === 0)
    assert(forest2.trees(0).nodes(1).prediction === 2.0)
    assert(compareDouble(forest2.trees(0).nodes(1).impurity, 1.916716186961402))

    testDataRaw.foreach(row => assert(forest2.predict(row._2)(0)._1 === row._1))

    // Make sure that the variable importances are as expected.
    assert(compareDouble(forest.varImportance.featureImportance(0), 34.8883))
    assert(compareDouble(forest.varImportance.featureImportance(1), 22.6132))
    assert(compareDouble(forest2.varImportance.featureImportance(0), 34.8883))
    assert(compareDouble(forest2.varImportance.featureImportance(1), 22.6132))
  }

  test("Train a tree 5 - unsigned Byte features RDD with NaN features") {
    val testDataRDD = sc.parallelize(TestDataGenerator.labeledData6, 3)
    val testDataRaw = testDataRDD.collect()
    val (maxLabelValue, bins) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](1),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "6", StringConstants.MaxCardinality_Categoric -> "1000"))

    assert(maxLabelValue === 3.0)

    val txData = Discretizer.transformFeaturesToUnsignedByteBinIds(testDataRDD, bins)

    // No bagging (100% sampling without replacement).
    val inputRDD = Bagger.bagRDD[Byte](txData, 1, SamplingType.SampleWithoutReplacement, 1.0, 0)

    // Train without local sub-tree training.
    val forest = SequoiaForestTrainer.train(
      UnsignedByteRDD(inputRDD),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Classification_InfoGain,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 2,
        localTrainThreshold = 0,
        numSubTreesPerIteration = 0,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(4),
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest.trees.length === 1)
    assert(forest.trees(0).getNodeCount === 18)
    assert(forest.trees(0).subTrees.size === 0)
    assert(forest.trees(0).nodes(1).prediction === 2.0)
    assert(compareDouble(forest.trees(0).nodes(1).impurity, 1.916716186961402))

    testDataRaw.foreach(row => assert(forest.predict(row._2)(0)._1 === row._1))

    // Train with local sub-tree training.
    val forest2 = SequoiaForestTrainer.train(
      UnsignedByteRDD(inputRDD),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Classification_InfoGain,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 1000,
        localTrainThreshold = 10,
        numSubTreesPerIteration = 1000,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = Some(4),
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest2.trees.length === 1)
    assert(forest2.trees(0).getNodeCount === 18)
    assert(forest2.trees(0).subTrees.size === 7)
    assert(forest2.trees(0).nodes(1).prediction === 2.0)
    assert(compareDouble(forest2.trees(0).nodes(1).impurity, 1.916716186961402))

    testDataRaw.foreach(row => assert(forest2.predict(row._2)(0)._1 === row._1))

    // Make sure that the variable importances are as expected.
    assert(compareDouble(forest.varImportance.featureImportance(0), 35.0603))
    assert(compareDouble(forest.varImportance.featureImportance(1), 22.4412))
    assert(compareDouble(forest2.varImportance.featureImportance(0), 35.0603))
    assert(compareDouble(forest2.varImportance.featureImportance(1), 22.4412))
  }

  test("Train a regression tree 1 - unsigned Byte features RDD") {
    val testDataRDD = sc.parallelize(TestDataGenerator.labeledData2, 3)
    val testDataRaw = testDataRDD.collect()
    val (maxLabelValue, bins) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](1),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.MaxCardinality_Categoric -> "1000"))

    assert(maxLabelValue === 3.0)

    val txData = Discretizer.transformFeaturesToUnsignedByteBinIds(testDataRDD, bins)

    // No bagging (100% sampling without replacement).
    val inputRDD = Bagger.bagRDD[Byte](txData, 1, SamplingType.SampleWithoutReplacement, 1.0, 0)

    // Train without local sub-tree training.
    val forest = SequoiaForestTrainer.train(
      UnsignedByteRDD(inputRDD),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Regression_Variance,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 2,
        localTrainThreshold = 0,
        numSubTreesPerIteration = 0,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = None,
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest.trees.length === 1)
    assert(forest.trees(0).getNodeCount === 19)
    assert(forest.trees(0).subTrees.size === 0)
    assert(compareDouble(forest.trees(0).nodes(1).prediction, 1.4))
    assert(compareDouble(forest.trees(0).nodes(1).impurity, 0.9733333))
    assert(forest.trees(0).nodes(1).weight === 30)

    testDataRaw.foreach(row => assert(forest.predict(row._2)(0)._1 === row._1))

    // Train with local sub-tree training.
    val forest2 = SequoiaForestTrainer.train(
      UnsignedByteRDD(inputRDD),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Regression_Variance,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 1000,
        localTrainThreshold = 10,
        numSubTreesPerIteration = 1000,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = None,
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest2.trees.length === 1)
    assert(forest2.trees(0).getNodeCount === 19)
    assert(forest2.trees(0).subTrees.size === 4)
    assert(compareDouble(forest2.trees(0).nodes(1).prediction, 1.4))
    assert(compareDouble(forest2.trees(0).nodes(1).impurity, 0.9733333))

    testDataRaw.foreach(row => assert(forest2.predict(row._2)(0)._1 === row._1))

    // Train without local sub-tree training.
    val forest3 = SequoiaForestTrainer.train(
      UnsignedByteRDD(inputRDD),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Regression_Variance,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = 3,
        numNodesPerIteration = 2,
        localTrainThreshold = 0,
        numSubTreesPerIteration = 0,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = None,
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest3.trees.length === 1)
    assert(forest3.trees(0).getNodeCount === 7)
    assert(forest3.trees(0).subTrees.size === 0)
    assert(compareDouble(forest3.trees(0).nodes(1).prediction, 1.4))
    assert(compareDouble(forest3.trees(0).nodes(1).impurity, 0.9733333))

    // Train with local sub-tree training.
    val forest4 = SequoiaForestTrainer.train(
      UnsignedByteRDD(inputRDD),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Regression_Variance,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = 3,
        numNodesPerIteration = 1000,
        localTrainThreshold = 10,
        numSubTreesPerIteration = 1000,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = None,
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest4.trees.length === 1)
    assert(forest4.trees(0).getNodeCount === 7)
    assert(forest4.trees(0).subTrees.size === 2)
    assert(compareDouble(forest4.trees(0).nodes(1).prediction, 1.4))
    assert(compareDouble(forest4.trees(0).nodes(1).impurity, 0.9733333))

    testDataRaw.foreach(row => assert(forest3.predict(row._2)(0)._1 === forest4.predict(row._2)(0)._1))

    // Compare the variable importance with expected values.
    assert(compareDouble(forest.varImportance.featureImportance(0), 23.05))
    assert(compareDouble(forest.varImportance.featureImportance(1), 6.15))
    assert(compareDouble(forest2.varImportance.featureImportance(0), 23.05))
    assert(compareDouble(forest2.varImportance.featureImportance(1), 6.15))
    assert(compareDouble(forest3.varImportance.featureImportance(0), 18.0083))
    assert(compareDouble(forest3.varImportance.featureImportance(1), 5.4))
    assert(compareDouble(forest4.varImportance.featureImportance(0), 18.0083))
    assert(compareDouble(forest4.varImportance.featureImportance(1), 5.4))
  }

  test("Train a regression tree 2 - unsigned Short features RDD") {
    val testDataRDD = sc.parallelize(TestDataGenerator.labeledData2, 3)
    val testDataRaw = testDataRDD.collect()
    val (maxLabelValue, bins) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](1),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.MaxCardinality_Categoric -> "1000"))

    assert(maxLabelValue === 3.0)

    val txData = Discretizer.transformFeaturesToUnsignedShortBinIds(testDataRDD, bins)

    // No bagging (100% sampling without replacement).
    val inputRDD = Bagger.bagRDD[Short](txData, 1, SamplingType.SampleWithoutReplacement, 1.0, 0)

    // Train without local sub-tree training.
    val forest = SequoiaForestTrainer.train(
      UnsignedShortRDD(inputRDD),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Regression_Variance,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 2,
        localTrainThreshold = 0,
        numSubTreesPerIteration = 0,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = None,
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest.trees.length === 1)
    assert(forest.trees(0).getNodeCount === 19)
    assert(forest.trees(0).subTrees.size === 0)
    assert(compareDouble(forest.trees(0).nodes(1).prediction, 1.4))
    assert(compareDouble(forest.trees(0).nodes(1).impurity, 0.9733333))
    assert(forest.trees(0).nodes(1).weight === 30)

    testDataRaw.foreach(row => assert(forest.predict(row._2)(0)._1 === row._1))

    // Train with local sub-tree training.
    val forest2 = SequoiaForestTrainer.train(
      UnsignedShortRDD(inputRDD),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Regression_Variance,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 1000,
        localTrainThreshold = 10,
        numSubTreesPerIteration = 1000,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = None,
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest2.trees.length === 1)
    assert(forest2.trees(0).getNodeCount === 19)
    assert(forest2.trees(0).subTrees.size === 4)
    assert(compareDouble(forest2.trees(0).nodes(1).prediction, 1.4))
    assert(compareDouble(forest2.trees(0).nodes(1).impurity, 0.9733333))

    testDataRaw.foreach(row => assert(forest2.predict(row._2)(0)._1 === row._1))

    // Compare the variable importance with expected values.
    assert(compareDouble(forest.varImportance.featureImportance(0), 23.05))
    assert(compareDouble(forest.varImportance.featureImportance(1), 6.15))
    assert(compareDouble(forest2.varImportance.featureImportance(0), 23.05))
    assert(compareDouble(forest2.varImportance.featureImportance(1), 6.15))
  }

  test("Train a regression tree 3 - unsigned Byte features Local") {
    val testDataRDD = sc.parallelize(TestDataGenerator.labeledData2, 3)
    val testDataRaw = testDataRDD.collect()
    val (maxLabelValue, bins) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](1),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.MaxCardinality_Categoric -> "1000"))

    assert(maxLabelValue === 3.0)

    val txDataLocal = Discretizer.transformFeaturesToUnsignedByteBinIds(testDataRDD, bins).collect()

    // No bagging (100% sampling without replacement).
    val baggedInput = Bagger.bagArray[Byte](txDataLocal, 1, SamplingType.SampleWithoutReplacement, 1.0, 0)
    val inputLocal = baggedInput.map(row => (row, Array.fill[Int](1)(0)))

    // Train without local sub-tree training.
    val forest = SequoiaForestTrainer.train(
      UnsignedByteLocal(inputLocal),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Regression_Variance,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 2,
        localTrainThreshold = 0,
        numSubTreesPerIteration = 0,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = None,
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest.trees.length === 1)
    assert(forest.trees(0).getNodeCount === 19)
    assert(forest.trees(0).subTrees.size === 0)
    assert(compareDouble(forest.trees(0).nodes(1).prediction, 1.4))
    assert(compareDouble(forest.trees(0).nodes(1).impurity, 0.9733333))
    assert(forest.trees(0).nodes(1).weight === 30)

    testDataRaw.foreach(row => assert(forest.predict(row._2)(0)._1 === row._1))

    // Train with local sub-tree training.
    val forest2 = SequoiaForestTrainer.train(
      UnsignedByteLocal(inputLocal),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Regression_Variance,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 1000,
        localTrainThreshold = 10,
        numSubTreesPerIteration = 1000,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = None,
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest2.trees.length === 1)
    assert(forest2.trees(0).getNodeCount === 19)
    assert(forest2.trees(0).subTrees.size === 0)
    assert(compareDouble(forest2.trees(0).nodes(1).prediction, 1.4))
    assert(compareDouble(forest2.trees(0).nodes(1).impurity, 0.9733333))

    testDataRaw.foreach(row => assert(forest2.predict(row._2)(0)._1 === row._1))

    // Compare the variable importance with expected values.
    assert(compareDouble(forest.varImportance.featureImportance(0), 23.05))
    assert(compareDouble(forest.varImportance.featureImportance(1), 6.15))
    assert(compareDouble(forest2.varImportance.featureImportance(0), 23.05))
    assert(compareDouble(forest2.varImportance.featureImportance(1), 6.15))
  }

  test("Train a regression tree 4 - unsigned Short features Local") {
    val testDataRDD = sc.parallelize(TestDataGenerator.labeledData2, 3)
    val testDataRaw = testDataRDD.collect()
    val (maxLabelValue, bins) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](1),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "5", StringConstants.MaxCardinality_Categoric -> "1000"))

    assert(maxLabelValue === 3.0)

    val txDataLocal = Discretizer.transformFeaturesToUnsignedShortBinIds(testDataRDD, bins).collect()

    // No bagging (100% sampling without replacement).
    val baggedInput = Bagger.bagArray[Short](txDataLocal, 1, SamplingType.SampleWithoutReplacement, 1.0, 0)
    val inputLocal = baggedInput.map(row => (row, Array.fill[Int](1)(0)))

    // Train without local sub-tree training.
    val forest = SequoiaForestTrainer.train(
      UnsignedShortLocal(inputLocal),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Regression_Variance,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 2,
        localTrainThreshold = 0,
        numSubTreesPerIteration = 0,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = None,
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest.trees.length === 1)
    assert(forest.trees(0).getNodeCount === 19)
    assert(forest.trees(0).subTrees.size === 0)
    assert(compareDouble(forest.trees(0).nodes(1).prediction, 1.4))
    assert(compareDouble(forest.trees(0).nodes(1).impurity, 0.9733333))
    assert(forest.trees(0).nodes(1).weight === 30)

    testDataRaw.foreach(row => assert(forest.predict(row._2)(0)._1 === row._1))

    // Train with local sub-tree training.
    val forest2 = SequoiaForestTrainer.train(
      UnsignedShortLocal(inputLocal),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Regression_Variance,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 1000,
        localTrainThreshold = 10,
        numSubTreesPerIteration = 1000,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = None,
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest2.trees.length === 1)
    assert(forest2.trees(0).getNodeCount === 19)
    assert(forest2.trees(0).subTrees.size === 0)
    assert(compareDouble(forest2.trees(0).nodes(1).prediction, 1.4))
    assert(compareDouble(forest2.trees(0).nodes(1).impurity, 0.9733333))

    testDataRaw.foreach(row => assert(forest2.predict(row._2)(0)._1 === row._1))

    // Compare the variable importance with expected values.
    assert(compareDouble(forest.varImportance.featureImportance(0), 23.05))
    assert(compareDouble(forest.varImportance.featureImportance(1), 6.15))
    assert(compareDouble(forest2.varImportance.featureImportance(0), 23.05))
    assert(compareDouble(forest2.varImportance.featureImportance(1), 6.15))
  }

  test("Train a regression tree 5 - unsigned Byte features RDD with NaN features") {
    val testDataRDD = sc.parallelize(TestDataGenerator.labeledData6, 3)
    val testDataRaw = testDataRDD.collect()
    val (maxLabelValue, bins) = EqualWidthDiscretizer.discretizeFeatures(
      testDataRDD,
      Set[Int](1),
      labelIsCategorical = true,
      Map[String, String](StringConstants.NumBins_Numeric -> "6", StringConstants.MaxCardinality_Categoric -> "1000"))

    assert(maxLabelValue === 3.0)

    val txData = Discretizer.transformFeaturesToUnsignedByteBinIds(testDataRDD, bins)

    // No bagging (100% sampling without replacement).
    val inputRDD = Bagger.bagRDD[Byte](txData, 1, SamplingType.SampleWithoutReplacement, 1.0, 0)

    // Train without local sub-tree training.
    val forest = SequoiaForestTrainer.train(
      UnsignedByteRDD(inputRDD),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Regression_Variance,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 2,
        localTrainThreshold = 0,
        numSubTreesPerIteration = 0,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = None,
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest.trees.length === 1)
    assert(forest.trees(0).getNodeCount === 23)
    assert(forest.trees(0).subTrees.size === 0)
    assert(compareDouble(forest.trees(0).nodes(1).prediction, 1.4))
    assert(compareDouble(forest.trees(0).nodes(1).impurity, 0.9733333))
    assert(forest.trees(0).nodes(1).weight === 30)

    testDataRaw.foreach(row => assert(forest.predict(row._2)(0)._1 === row._1))

    // Train with local sub-tree training.
    val forest2 = SequoiaForestTrainer.train(
      UnsignedByteRDD(inputRDD),
      bins,
      SequoiaForestOptions(
        numTrees = 1,
        treeType = TreeType.Regression_Variance,
        mtry = 100,
        minSplitSize = 2,
        maxDepth = -1,
        numNodesPerIteration = 1000,
        localTrainThreshold = 10,
        numSubTreesPerIteration = 1000,
        storeModelInMemory = true,
        outputStorage = new NullSinkForestStorage,
        numClasses = None,
        imputationType = ImputationType.SplitOnMissing,
        distributedNodeSplits = false),
      new ConsoleNotifiee,
      None,
      useLogLossForValidation = false,
      new scala.util.Random(17))

    assert(forest2.trees.length === 1)
    assert(forest2.trees(0).getNodeCount === 23)
    assert(forest2.trees(0).subTrees.size === 7)
    assert(compareDouble(forest2.trees(0).nodes(1).prediction, 1.4))
    assert(compareDouble(forest2.trees(0).nodes(1).impurity, 0.9733333))

    testDataRaw.foreach(row => assert(forest2.predict(row._2)(0)._1 === row._1))

    // Compare the variable importance with expected values.
    assert(compareDouble(forest.varImportance.featureImportance(0), 14.5232))
    assert(compareDouble(forest.varImportance.featureImportance(1), 14.6768))
    assert(compareDouble(forest2.varImportance.featureImportance(0), 14.5232))
    assert(compareDouble(forest2.varImportance.featureImportance(1), 14.6768))
  }
}
