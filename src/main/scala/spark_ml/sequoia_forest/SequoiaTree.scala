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

import java.io._
import java.nio.ByteBuffer
import scala.Serializable

import spire.implicits._

/**
 * Supported types of trees.
 */
object TreeType extends Enumeration {
  type TreeType = Value
  val Classification_InfoGain = Value(0)
  val Regression_Variance = Value(1)
}

/**
 * This is used to select the child of a node based on a feature value.
 */
trait NodeSplit extends Serializable {
  val featureId: Int
  def selectChildNode(featureVal: Double): Int
  def getChildNodeIds: Seq[Int]
}

/**
 * Numeric split.
 * @param featureId The ID of the feature to split on.
 * @param splitValue The value of the feature to split on.
 * @param leftId The ID of the left child.
 * @param rightId The ID of the right child.
 * @param missingValueId The ID of the child to follow if the feature value is NaN.
 */
case class NumericSplit(
    featureId: Int,
    splitValue: Double,
    leftId: Int,
    rightId: Int,
    missingValueId: Int = -1) extends NodeSplit {
  /**
   * If no child is found, this will return -1.
   * @param featureVal The feature value that we want to split on.
   * @return The ID of the child node that we want to go to.
   */
  def selectChildNode(featureVal: Double): Int = {
    if (featureVal.isNaN) {
      missingValueId
    } else if (featureVal < splitValue) {
      leftId
    } else {
      rightId
    }
  }

  def getChildNodeIds: Seq[Int] = {
    if (missingValueId == -1) {
      Seq[Int](leftId, rightId)
    } else {
      Seq[Int](leftId, rightId, missingValueId)
    }
  }
}

/**
 * Categorical split.
 * @param featureId The ID of the feature to split on (should be a categorical feature).
 * @param featValToNodeId The map from feature values to child nodes.
 * @param missingValueId The ID of the child to follow if the feature value is NaN.
 */
case class CategoricalSplit(
    featureId: Int,
    featValToNodeId: mutable.Map[Int, Int],
    missingValueId: Int = -1) extends NodeSplit {
  def selectChildNode(featureVal: Double): Int = {
    if (featureVal.isNaN) {
      missingValueId
    } else {
      // If the mapped node is not found, return -1.
      featValToNodeId.getOrElse(featureVal.toInt, -1)
    }
  }

  def getChildNodeIds: Seq[Int] = {
    val r = featValToNodeId.values.toSet
    if (missingValueId == -1) {
      r.toSeq
    } else {
      r.toSeq ++ Seq[Int](missingValueId)
    }
  }
}

/**
 * A tree node.
 * @param nodeId Node ID.
 * @param prediction Prediction at this node.
 * @param impurity Impurity at this node.
 * @param weight Weight (number of samples) at this node.
 * @param splitImpurity Weighted impurity after the split (if there's one).
 * @param split Split (None means leaf)
 */
case class SequoiaNode(
  nodeId: Int,
  prediction: Double,
  impurity: Double,
  weight: Double,
  splitImpurity: Option[Double],
  split: Option[NodeSplit]) extends Serializable

/**
 * A tree.
 * It can contains nodes and sub-trees.
 */
case class SequoiaTree(var treeId: Int) {
  val nodes = mutable.Map[Int, SequoiaNode]()

  // TODO: Merge nodes of sub-trees into nodes and remove the notion of subTrees.
  val subTrees = mutable.Map[Int, SequoiaTree]()

  /**
   * Get a node count.
   * @return The total number of nodes in this tree, including nodes of sub trees.
   */
  def getNodeCount: Int = nodes.size + subTrees.values.foldLeft(0)((curCount, tree) => curCount + tree.getNodeCount)

  /**
   * Add a node.
   * @param node A node to add.
   */
  def addNode(node: SequoiaNode): Unit = {
    nodes.put(node.nodeId, node)
  }

  /**
   * Add a sub-tree.
   * @param tree A sub-tree to add.
   */
  def addSubTree(tree: SequoiaTree): Unit = {
    subTrees.put(tree.treeId, tree)
  }

  /**
   * Predict the output from the given features - features should be in the same order as the ones that the tree trained on.
   * @param features Feature values.
   * @return Prediction and weight (Double, Double)
   */
  def predict(features: Array[Double]): (Double, Double) = {
    var curNode = nodes(1) // Root node is always 1.
    while (curNode.split != None) {
      val split = curNode.split.get
      var childId = curNode.split.get.selectChildNode(features(split.featureId))
      if (childId == -1) {
        // If the child is not found, we'll go down the path of the child node with the largest weight.
        val allChildIds = curNode.split.get.getChildNodeIds
        var maxWeight = 0.0
        allChildIds.foreach(ci => {
          val childWeight = if (nodes.contains(ci)) {
            nodes(ci).weight
          } else {
            subTrees(ci).nodes(1).weight
          }

          if (childWeight > maxWeight) {
            maxWeight = childWeight
            childId = ci
          }
        })
      }

      if (!nodes.contains(childId)) {
        if (!subTrees.contains(childId)) {
          return (curNode.prediction, curNode.weight)
        } else {
          return subTrees(childId).predict(features)
        }
      } else {
        curNode = nodes(childId)
      }
    }

    (curNode.prediction, curNode.weight)
  }
}

/**
 * A forest is simply a collection of trees.
 * @param trees Trees in the forest.
 * @param treeType Type of trees (classification or regression).
 * @param varImportance Variable importance tracker.
 * @param sampleCounts The number of training samples per tree.
 */
case class SequoiaForest(
    trees: Array[SequoiaTree],
    treeType: TreeType.TreeType,
    varImportance: VarImportance,
    sampleCounts: Array[Long]) {
  /**
   * Predict from the features.
   * @param features A double array of features.
   * @return Prediction(s) and corresponding weight(s) (e.g. probabilities or variances of predictions, etc.)
   */
  def predict(features: Array[Double]): Array[(Double, Double)] = {
    treeType match {
      case TreeType.Classification_InfoGain => predictClass(features)
      case TreeType.Regression_Variance => predictRegression(features)
    }
  }

  /**
   * Predict the class ouput from the given features - features should be in the same order as the ones that the tree trained on.
   * @param features Feature values.
   * @return Predictions and their probabilities an array of (Double, Double)
   */
  private def predictClass(features: Array[Double]): Array[(Double, Double)] = {
    val predictions = mutable.Map[Double, Double]() // Predicted label and its count.
    var treeId = 0
    while (treeId < trees.length) {
      val tree = trees(treeId)
      val (prediction, _) = tree.predict(features)
      predictions.getOrElseUpdate(prediction, 0)
      predictions(prediction) += 1.0

      treeId += 1
    }

    // Sort the predictions by the number of occurrences.
    // The first element has the highest number of occurrences.
    val sortedPredictions = predictions.toArray.sorted(Ordering.by[(Double, Double), Double](-_._2))
    sortedPredictions.map(p => (p._1, p._2 / trees.length.toDouble)).toArray
  }

  /**
   * Predict a continuous ouput from the given features - features should be in the same order as the ones that the tree trained on.
   * @param features Feature values.
   * @return Prediction and its variance (a single element array of (Double, Double))
   */
  private def predictRegression(features: Array[Double]): Array[(Double, Double)] = {
    var predictionSum = 0.0
    var predictionSqrSum = 0.0
    var treeId = 0
    while (treeId < trees.length) {
      val tree = trees(treeId)
      val (prediction, _) = tree.predict(features)
      predictionSum += prediction
      predictionSqrSum += prediction * prediction

      treeId += 1
    }

    val predAvg = predictionSum / trees.length.toDouble
    val predVar = predictionSqrSum / trees.length.toDouble - predAvg * predAvg
    Array[(Double, Double)]((predAvg, predVar))
  }
}

/**
 * Use this to write binary tree information to a stream object.
 */
object SequoiaForestWriter {
  /**
   * Write forest information to a stream.
   * @param numTrees Number of trees in the forest.
   * @param treeType Type of the trees in the forest (either classification or regression).
   * @param outputStream Output stream object.
   */
  def writeForestInfo(numTrees: Int, treeType: TreeType.TreeType, outputStream: OutputStream): Unit = {
    outputStream.write((numTrees.toString + "\n").getBytes)
    outputStream.write((treeType.toString + "\n").getBytes)
  }

  /**
   * Write the variable importance object to a stream.
   * @param varImportance The variable importance object to write.
   * @param outputStream Output stream object.
   */
  def writeVariableImportance(varImportance: VarImportance, outputStream: OutputStream): Unit = {
    writeInt(varImportance.numFeatures, outputStream) // Write the number of features.
    cfor(0)(_ < varImportance.numFeatures, _ + 1)(
      featId => writeDouble(varImportance.featureImportance(featId), outputStream)
    )
  }

  /**
   * Write the header to the beginning of a tree.
   * Every tree (whether it's the main tree or a sub-tree) must start with a header.
   * @param treeId The ID of the tree.
   * @param outputStream The output stream to write to.
   */
  def writeTreeHeader(treeId: Int, outputStream: OutputStream): Unit = {
    val treeStr = "Tree"
    val header = new Array[Byte](treeStr.length + 4)
    treeStr.getBytes.copyToArray(header, 0)
    ByteBuffer.wrap(header, treeStr.length, 4).putInt(treeId)
    outputStream.write(header)
  }

  /**
   * Write a footer to mark the end of a tree.
   * @param outputStream The output stream to write to.
   */
  def writeTreeEnd(outputStream: OutputStream): Unit = {
    outputStream.write("Tend".getBytes)
  }

  /**
   * Write a double value to the output stream.
   * @param value The value to write.
   * @param outputStream The output stream to write to.
   */
  def writeDouble(value: Double, outputStream: OutputStream): Unit = {
    val array = new Array[Byte](8)
    ByteBuffer.wrap(array, 0, 8).putDouble(value)
    outputStream.write(array)
  }

  /**
   * Write an integer value to the output stream.
   * @param value The value to write.
   * @param outputStream The output stream to write to.
   */
  def writeInt(value: Int, outputStream: OutputStream): Unit = {
    val array = new Array[Byte](4)
    ByteBuffer.wrap(array, 0, 4).putInt(value)
    outputStream.write(array)
  }

  /**
   * Write a numeric split to the output stream.
   * @param split The split to write.
   * @param outputStream The output stream to write to.
   */
  def writeNumericSplit(split: NumericSplit, outputStream: OutputStream): Unit = {
    writeInt(0, outputStream) // 0 means numeric split.
    writeInt(split.featureId, outputStream)
    writeDouble(split.splitValue, outputStream)
    writeInt(split.leftId, outputStream)
    writeInt(split.rightId, outputStream)
    writeInt(split.missingValueId, outputStream)
  }

  /**
   * Write a categorical split to the output stream.
   * @param split The split to write.
   * @param outputStream The output stream to write to.
   */
  def writeCategoricalSplit(split: CategoricalSplit, outputStream: OutputStream): Unit = {
    writeInt(1, outputStream) // 1 means categorical split.
    writeInt(split.featureId, outputStream)
    writeInt(split.featValToNodeId.size, outputStream) // Number of children.
    split.featValToNodeId.foreach(key_value => {
      writeInt(key_value._1, outputStream)
      writeInt(key_value._2, outputStream)
    })

    writeInt(split.missingValueId, outputStream)
  }

  /**
   * Write a node to the ouput stream.
   * @param node The node to write.
   * @param outputStream The output stream to write to.
   */
  def writeNode(node: SequoiaNode, outputStream: OutputStream): Unit = {
    val nodeStr = "Node"
    val header = new Array[Byte](nodeStr.length + 4)
    nodeStr.getBytes.copyToArray(header, 0)
    ByteBuffer.wrap(header, nodeStr.length, 4).putInt(node.nodeId)
    outputStream.write(header)

    writeDouble(node.prediction, outputStream)
    writeDouble(node.impurity, outputStream)
    writeDouble(node.weight, outputStream)

    if (node.split != None) {
      // 1 means that this node has a split.
      writeInt(1, outputStream)
      writeDouble(node.splitImpurity.get, outputStream)

      node.split.get match {
        case split: NumericSplit => writeNumericSplit(split, outputStream)
        case split: CategoricalSplit => writeCategoricalSplit(split, outputStream)
      }
    } else {
      // 0 means there's nothing more in this node.
      writeInt(0, outputStream)
    }
  }

  /**
   * Write a sub tree to the output stream.
   * @param subTree The sub tree to write.
   * @param outputStream The output stream to write to.
   */
  def writeSubTree(subTree: SequoiaTree, outputStream: OutputStream): Unit = {
    writeTreeHeader(subTree.treeId, outputStream)
    subTree.nodes.values.foreach(node => writeNode(node, outputStream))
    writeTreeEnd(outputStream)
  }

  /**
   * Write a tree to the output stream.
   * @param tree The tree to write.
   * @param outputStream The output stream to write to.
   */
  def writeTree(tree: SequoiaTree, outputStream: OutputStream): Unit = {
    writeTreeHeader(tree.treeId, outputStream)
    tree.nodes.values.foreach(node => writeNode(node, outputStream))
    tree.subTrees.values.foreach(subTree => writeSubTree(subTree, outputStream))
    writeTreeEnd(outputStream)
  }
}

/**
 * Throw this exception if the tree stream that we are reading from is invalid.
 * @param msg String message to include.
 */
case class InvalidTreeStreamException(msg: String) extends Exception(msg)

/**
 * Use this to read saved Sequoia Forest info and trees.
 */
object SequoiaForestReader {
  /**
   * Read a forest stored in the hadoop file system.
   * @param forestPath Path (directory) where the forest is stored.
   * @param hadoopConf Hadoop configuration.
   * @return Forest
   */
  def readForest(forestPath: String, hadoopConf: org.apache.hadoop.conf.Configuration): SequoiaForest = {
    val hdfs = org.apache.hadoop.fs.FileSystem.get(hadoopConf)

    val forestInfoStream = hdfs.open(new org.apache.hadoop.fs.Path(forestPath, "forestInfo"))
    val (numTrees, treeType) = SequoiaForestReader.readForestInfo(forestInfoStream)
    forestInfoStream.close()
    println("Number of trees in the forest : " + numTrees)
    println("Tree type is " + treeType.toString)

    val trees = new Array[SequoiaTree](numTrees)

    val fileStatuses = hdfs.listStatus(new org.apache.hadoop.fs.Path(forestPath))
    var i = 0
    while (i < fileStatuses.length) {
      if (fileStatuses(i).getPath.getName.startsWith("tree")) {
        println("Parsing " + fileStatuses(i).getPath.toString)
        val inputStream = hdfs.open(fileStatuses(i).getPath)
        val tree = SequoiaForestReader.readTree(inputStream)
        trees(tree.treeId) = tree
      }

      i += 1
    }

    val varImpStream = hdfs.open(new org.apache.hadoop.fs.Path(forestPath, "varImp"))
    val varImportance = SequoiaForestReader.readVarImportance(varImpStream)
    varImpStream.close()

    // TODO: Properly write/load numbers of training samples.
    SequoiaForest(trees, treeType, varImportance, Array.fill[Long](numTrees)(0))
  }

  /**
   * Read the variable importance object from a stream.
   * @param is Input stream.
   * @return A variable importance object.
   */
  private[spark_ml] def readVarImportance(is: InputStream): VarImportance = {
    val numFeatures = readInt(is)
    val varImportance = VarImportance(numFeatures)
    cfor(0)(_ < numFeatures, _ + 1)(
      featId => {
        val varImp = readDouble(is)
        varImportance.addVarImportance(featId, varImp)
      }
    )

    varImportance
  }

  /**
   * Use this to get read bytes in a guaranteed fashion.
   * In Java, there's no guarantee that InputStream.read(buf, 0, 10) will read 10 bytes,
   * even if 10 bytes really exist. We may have to do multiple reads.
   * Warning: If you try to read more than what are available, this may time out (in case of network stream, it'll depend on network timeout).
   */
  private[spark_ml] def readExactBytes(is: InputStream, buf: Array[Byte], off: Int, len: Int): Unit = {
    var readCnt = 0
    while (readCnt < len) {
      readCnt += is.read(buf, off + readCnt, len - readCnt)
    }
  }

  /**
   * Read forest info from an input stream.
   * @param inputStream The input stream from which to read forest info.
   * @return Number of trees and the tree type.
   */
  private[spark_ml] def readForestInfo(inputStream: InputStream): (Int, TreeType.TreeType) = {
    val br = new BufferedReader(new InputStreamReader(inputStream))
    (br.readLine().toInt, TreeType.withName(br.readLine()))
  }

  /**
   * Read an integer.
   * @param inputStream Input stream from which to read.
   * @return An integer value read from the stream.
   */
  private[spark_ml] def readInt(inputStream: InputStream): Int = {
    val intBytes = new Array[Byte](4)
    readExactBytes(inputStream, intBytes, 0, 4)
    ByteBuffer.wrap(intBytes).getInt
  }

  /**
   * Read a double.
   * @param inputStream Input stream from which to read.
   * @return A double value read from the stream.
   */
  private[spark_ml] def readDouble(inputStream: InputStream): Double = {
    val doubleBytes = new Array[Byte](8)
    readExactBytes(inputStream, doubleBytes, 0, 8)
    ByteBuffer.wrap(doubleBytes).getDouble
  }

  /**
   * Read a Sequoia Tree from the input stream.
   * @param inputStream Input stream from which to read.
   * @param skipPrefix Whether we should skip tree header prefix ("Tree").
   * @return A tree object.
   */
  private[spark_ml] def readTree(inputStream: InputStream, skipPrefix: Boolean = false): SequoiaTree = {
    if (!skipPrefix) {
      val treePrefixBytes = new Array[Byte](4)
      readExactBytes(inputStream, treePrefixBytes, 0, 4)
      val treePrefix = new String(treePrefixBytes)
      if (treePrefix != "Tree") {
        throw new InvalidTreeStreamException("Expected to find 'Tree' but instead found '" + treePrefix + "'")
      }
    }

    val treeId = readInt(inputStream)
    val tree = SequoiaTree(treeId)

    val entityTypeBytes = new Array[Byte](4)
    readExactBytes(inputStream, entityTypeBytes, 0, 4)
    var entityType = new String(entityTypeBytes)
    while (entityType != "Tend") {
      if (entityType == "Tree") {
        val subTree = readTree(inputStream, skipPrefix = true)
        tree.addSubTree(subTree)
      } else if (entityType == "Node") {
        val node = readNode(inputStream)
        tree.addNode(node)
      } else {
        throw new InvalidTreeStreamException(entityType + " is not a valid entity type.")
      }

      readExactBytes(inputStream, entityTypeBytes, 0, 4)
      entityType = new String(entityTypeBytes)
    }

    tree
  }

  /**
   * Read a node.
   * @param inputStream Input stream from which to read.
   * @return Node that we read.
   */
  private[spark_ml] def readNode(inputStream: InputStream): SequoiaNode = {
    val nodeId = readInt(inputStream)
    val prediction = readDouble(inputStream)
    val impurity = readDouble(inputStream)
    val weight = readDouble(inputStream)

    val splitIndicator = readInt(inputStream)
    if (splitIndicator == 0) {
      SequoiaNode(nodeId, prediction, impurity, weight, None, None)
    } else {
      val splitImpurity = Some(readDouble(inputStream))
      val splitType = readInt(inputStream)
      val split: NodeSplit = if (splitType == 0) {
        val featId = readInt(inputStream)
        val splitValue = readDouble(inputStream)
        val leftId = readInt(inputStream)
        val rightId = readInt(inputStream)
        val missingValueId = readInt(inputStream)

        NumericSplit(featureId = featId, splitValue = splitValue, leftId = leftId, rightId = rightId, missingValueId = missingValueId)
      } else {
        val featId = readInt(inputStream)
        val featValToNodeId = mutable.Map[Int, Int]()
        val numMappings = readInt(inputStream)
        var i = 0
        while (i < numMappings) {
          val key = readInt(inputStream)
          val value = readInt(inputStream)
          featValToNodeId.put(key, value)
          i += 1
        }

        val missingValueId = readInt(inputStream)

        CategoricalSplit(featureId = featId, featValToNodeId = featValToNodeId, missingValueId = missingValueId)
      }

      SequoiaNode(nodeId, prediction, impurity, weight, splitImpurity, Some(split))
    }
  }
}
