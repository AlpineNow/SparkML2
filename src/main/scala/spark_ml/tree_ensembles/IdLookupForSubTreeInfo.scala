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

import spark_ml.util.MapWithSequentialIntKeys

/**
 * Sub tree info used during sub-tree training.
 * @param id Id of the subtree.
 * @param hash Hash value for the sub-tree (used to push hash sub-tree data
 *             evenly to different executors).
 * @param depth Depth of the sub-tree from the parent tree perspective.
 * @param parentTreeId Parent tree Id.
 */
case class SubTreeInfo(
  id: Int,
  hash: Int,
  depth: Int,
  parentTreeId: Int) {
  /**
   * Override the hashCode to return the subTreeHash value.
   * @return The subTreeHash value.
   */
  override def hashCode: Int = {
    hash
  }
}

/**
 * Sub tree info lookup used to find matching data points for each sub tree.
 * @param idRanges Id ranges for each tree.
 */
class IdLookupForSubTreeInfo(
    idRanges: Array[IdRange]
) extends IdLookup[SubTreeInfo](idRanges)

object IdLookupForSubTreeInfo {
  /**
   * Create a new Id lookup object for sub-trees.
   * @param idRanges Id ranges of sub trees for parent trees.
   * @param subTreeMaps Sub tree info maps for parent trees.
   * @return Id lookup object for sub-trees.
   */
  def createIdLookupForSubTreeInfo(
    idRanges: Array[IdRange],
    subTreeMaps: Array[MapWithSequentialIntKeys[SubTreeInfo]]): IdLookupForSubTreeInfo = {
    val lookup = new IdLookupForSubTreeInfo(idRanges)
    lookup.initLookUpObjs(
      (treeId: Int, id: Int) => subTreeMaps(treeId).get(id)
    )

    lookup
  }
}
