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

import spark_ml.util.Sorting
import spire.implicits._

/**
 * This object contains the descriptions of all the nodes in this sub-tree.
 * @param parentTreeId Id of the parent tree.
 * @param subTreeId Sub tree Id.
 * @param subTreeDepth Depth of the sub-tree from the parent tree's perspective.
 */
class SubTreeDesc(
  val parentTreeId: Int,
  val subTreeId: Int,
  val subTreeDepth: Int) extends Serializable {
  // Store the nodes in a mutable map.
  var nodes = mutable.Map[Int, NodeInfo]()

  /**
   * Add a new trained node info object.
   * It's expected that the node Id's increment monotonically, one by one.
   * @param nodeInfo Info about the next trained node.
   */
  def addNodeInfo(nodeInfo: NodeInfo): Unit = {
    assert(
      !nodes.contains(nodeInfo.nodeId),
      "A node with the node Id " + nodeInfo.nodeId + " already exists in the " +
      "sub tree " + subTreeId + " that belongs to the parent tree " + parentTreeId
    )

    nodes.put(nodeInfo.nodeId, nodeInfo)
  }

  /**
   * Update the tree/node Ids and the node depth to reflect the parent tree's
   * reality.
   * @param rootId The root node Id will be updated to this.
   * @param startChildNodeId The descendant nodes will have updated Ids,
   *                         starting from this number.
   * @return The last descendant node Id that was used for this sub-tree.
   */
  def updateIdsAndDepths(
    rootId: Int,
    startChildNodeId: Int): Int = {
    val itr = nodes.values.iterator
    val updatedNodes = mutable.Map[Int, NodeInfo]()
    var largestUpdatedNodeId = 0
    while (itr.hasNext) {
      val nodeInfo = itr.next()
      val updatedNodeInfo = nodeInfo.copy
      updatedNodeInfo.treeId = parentTreeId
      if (updatedNodeInfo.nodeId == 1) {
        updatedNodeInfo.nodeId = rootId
      } else {
        // Start node Id is used from the first child node of the root node,
        // which should have the Id of 2. Therefore, update the node Ids by
        // subtracting 2 and then adding startChildNodeId.
        updatedNodeInfo.nodeId = updatedNodeInfo.nodeId - 2 + startChildNodeId
      }
      largestUpdatedNodeId = math.max(updatedNodeInfo.nodeId, largestUpdatedNodeId)
      updatedNodeInfo.depth = updatedNodeInfo.depth - 1 + subTreeDepth
      if (updatedNodeInfo.splitInfo.nonEmpty) {
        val si = updatedNodeInfo.splitInfo.get
        val children = si.getOrderedChildNodes
        val numChildren = children.length
        cfor(0)(_ < numChildren, _ + 1)(
          i => {
            val child = children(i)
            child.treeId = parentTreeId
            child.nodeId = child.nodeId - 2 + startChildNodeId
            child.depth = child.depth - 1 + subTreeDepth
          }
        )
      }

      updatedNodes.put(updatedNodeInfo.nodeId, updatedNodeInfo)
    }

    this.nodes = updatedNodes

    largestUpdatedNodeId
  }

  /**
   * Get a sequence of nodeInfo objects ordered by the node Id.
   * @return A sequence of nodeInfo objects ordered by the node Id.
   */
  def getOrderedNodeInfoSeq: Seq[NodeInfo] = {
    val nodeInfoArray = this.nodes.values.toArray
    Sorting.quickSort[NodeInfo](nodeInfoArray)(
      Ordering.by[NodeInfo, Int](_.nodeId)
    )

    nodeInfoArray
  }
}

/**
 * Local subtree store. Used for locally training sub-trees.
 * @param parentTreeId Id of the parent tree.
 * @param subTreeId Sub tree Id.
 * @param subTreeDepth Depth of the sub-tree from the parent tree's perspective.
 */
class SubTreeStore(
  parentTreeId: Int,
  subTreeId: Int,
  subTreeDepth: Int) extends TreeEnsembleStore {
  val subTreeDesc = new SubTreeDesc(
    parentTreeId = parentTreeId,
    subTreeId = subTreeId,
    subTreeDepth = subTreeDepth
  )

  /**
   * Get a sub tree writer.
   * @return Sub tree writer.
   */
  def getWriter: TreeEnsembleWriter = new SubTreeWriter(this)
}

/**
 * Sub tree writer.
 * @param subTreeStore The sub tree store this belongs to.
 */
class SubTreeWriter(subTreeStore: SubTreeStore) extends TreeEnsembleWriter {
  val subTreeDesc = subTreeStore.subTreeDesc

  /**
   * Write node info.
   * @param nodeInfo Node info to write.
   */
  def writeNodeInfo(nodeInfo: NodeInfo): Unit = {
    subTreeDesc.addNodeInfo(nodeInfo)
  }
}
