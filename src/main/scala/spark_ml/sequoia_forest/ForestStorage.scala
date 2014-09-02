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

import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream

import spire.implicits._

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{ FileSystem, Path }

/**
 * The trainer will write forest trees to the underlying storage represented by this object.
 */
trait ForestStorage {
  /**
   * Get the location of the storage. (E.g. the directory in a file system).
   * @return The location string.
   */
  def getLocation: String

  /**
   * Initialize this storage object.
   * @param numTrees The number of trees in the forest.
   * @param treeType The type of the tree.
   */
  def initialize(numTrees: Int, treeType: TreeType.TreeType): Unit

  /**
   * Write a tree node.
   * @param treeId Id of the tree.
   * @param depth Depth of the node.
   * @param node The node to write to the storage.
   */
  def writeNode(treeId: Int, depth: Int, node: SequoiaNode): Unit

  /**
   * Write a sub tree.
   * @param treeId Id of the tree.
   * @param depth Depth of the sub-tree's root in the parent tree.
   * @param subTree The sub tree to write to the storage.
   */
  def writeSubTree(treeId: Int, depth: Int, subTree: SequoiaTree): Unit

  /**
   * Write a variable importance object.
   * @param varImportance The variable importance object to write to the storage.
   */
  def writeVarImportance(varImportance: VarImportance): Unit

  /**
   * Write the numbers of training samples for trees.
   * @param sampleCounts The numbers of training samples for trees.
   */
  def writeSampleCounts(sampleCounts: Array[Long]): Unit

  /**
   * Close the storage.
   */
  def close(): Unit
}

/**
 * The base class for file system based forest storage types.
 * Both local file system and HDFS storage types should be derived from this.
 */
abstract class FSForestStorage extends ForestStorage {
  protected var outputStreams: Array[OutputStream] = _
  protected var forestPath: String = _

  /**
   * Get the path in HDFS under which the forest is stored.
   * @return The location string.
   */
  override def getLocation: String = forestPath

  /**
   * Write a tree node.
   * @param treeId Id of the tree.
   * @param depth Depth of the node.
   * @param node The node to write to the storage.
   */
  override def writeNode(treeId: Int, depth: Int, node: SequoiaNode): Unit = {
    SequoiaForestWriter.writeNode(node, outputStreams(treeId))
  }

  /**
   * Write a sub tree.
   * @param treeId Id of the tree.
   * @param depth Depth of the sub-tree's root in the parent tree.
   * @param subTree The sub tree to write to the storage.
   */
  override def writeSubTree(treeId: Int, depth: Int, subTree: SequoiaTree): Unit = {
    SequoiaForestWriter.writeSubTree(subTree, outputStreams(treeId))
  }

  /**
   * Close the storage.
   */
  override def close(): Unit = {
    cfor(0)(_ < outputStreams.length, _ + 1)(
      treeId => {
        SequoiaForestWriter.writeTreeEnd(outputStreams(treeId))
        outputStreams(treeId).flush()
        outputStreams(treeId).close()
      }
    )
  }
}

/**
 * Implement an HDFS forest storage.
 * This stores the trees in an HDFS directory.
 */
class HDFSForestStorage(hadoopConf: Configuration, path: String) extends FSForestStorage {
  private val hdfs = FileSystem.get(hadoopConf)
  forestPath = path

  /**
   * Initialize the forest streams.
   * @param numTrees The number of trees in the forest.
   * @param treeType The type of the tree.
   */
  override def initialize(numTrees: Int, treeType: TreeType.TreeType): Unit = {
    // First, write forest info into a separate file.
    val forestInfoStream = hdfs.create(new Path(path, "forestInfo"))
    SequoiaForestWriter.writeForestInfo(numTrees, treeType, forestInfoStream)
    forestInfoStream.close()

    // Now, open tree output streams.
    outputStreams = new Array[OutputStream](numTrees)
    cfor(0)(_ < numTrees, _ + 1)(
      treeId => {
        outputStreams(treeId) = hdfs.create(new Path(path, "tree" + treeId))
        SequoiaForestWriter.writeTreeHeader(treeId, outputStreams(treeId))
      }
    )
  }

  /**
   * Write a variable importance object.
   * @param varImportance The variable importance object to write to the storage.
   */
  override def writeVarImportance(varImportance: VarImportance): Unit = {
    val varImpStream = hdfs.create(new Path(path, "varImp"))
    SequoiaForestWriter.writeVariableImportance(varImportance, varImpStream)
    varImpStream.close()
  }

  /**
   * Write the numbers of training samples for trees.
   * @param sampleCounts The numbers of training samples for trees.
   */
  override def writeSampleCounts(sampleCounts: Array[Long]): Unit = {}
}

/**
 * Implement a local forest storage.
 * This stores the trees in a local file system's directory.
 */
class LocalFSForestStorage(path: String) extends FSForestStorage {
  forestPath = path

  /**
   * Initialize the forest streams.
   * @param numTrees The number of trees in the forest.
   * @param treeType The type of the tree.
   */
  override def initialize(numTrees: Int, treeType: TreeType.TreeType): Unit = {
    // First, write forest info into a separate file.
    val forestInfoStream = new FileOutputStream(new File(path, "forestInfo"))
    SequoiaForestWriter.writeForestInfo(numTrees, treeType, forestInfoStream)
    forestInfoStream.close()

    // Now, open tree output streams.
    outputStreams = new Array[OutputStream](numTrees)
    cfor(0)(_ < numTrees, _ + 1)(
      treeId => {
        outputStreams(treeId) = new FileOutputStream(new File(path, "tree" + treeId))
        SequoiaForestWriter.writeTreeHeader(treeId, outputStreams(treeId))
      }
    )
  }

  /**
   * Write a variable importance object.
   * @param varImportance The variable importance object to write to the storage.
   */
  override def writeVarImportance(varImportance: VarImportance): Unit = {
    val varImpStream = new FileOutputStream(new File(path, "varImp"))
    SequoiaForestWriter.writeVariableImportance(varImportance, varImpStream)
    varImpStream.close()
  }

  /**
   * Write the numbers of training samples for trees.
   * @param sampleCounts The numbers of training samples for trees.
   */
  override def writeSampleCounts(sampleCounts: Array[Long]): Unit = {}
}

/**
 * A simple null sink.
 * The calls don't do anything.
 */
class NullSinkForestStorage extends ForestStorage {
  override def getLocation: String = "Null"
  override def initialize(numTrees: Int, treeType: TreeType.TreeType): Unit = {}
  override def writeNode(treeId: Int, depth: Int, node: SequoiaNode): Unit = {}
  override def writeSubTree(treeId: Int, depth: Int, subTree: SequoiaTree): Unit = {}
  override def writeVarImportance(varImportance: VarImportance): Unit = {}
  override def writeSampleCounts(sampleCounts: Array[Long]): Unit = {}
  override def close(): Unit = {}
}
