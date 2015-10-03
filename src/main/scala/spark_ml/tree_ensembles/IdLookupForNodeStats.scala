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
import spark_ml.util.{MapWithSequentialIntKeys, RandomSet}
import spire.implicits._

/**
 * Id lookup for aggregating node statistics during training.
 * @param idRanges Id range for nodes to look up.
 * @param nodeDepths A map containing depths of all the nodes in the ranges.
 */
class IdLookupForNodeStats(
  idRanges: Array[IdRange],
  nodeDepths: Array[MapWithSequentialIntKeys[Int]]
) extends IdLookup[NodeStats](idRanges) {

  /**
   * Initialize this look up object.
   * @param treeType Tree type by the split criteria (e.g. classification based
   *                 on info-gain or regression based on variance.)
   * @param featureBinsInfo Feature discretization info.
   * @param treeSeeds Random seeds for trees.
   * @param mtry mtry.
   * @param numClasses Optional number of target classes (for classifications).
   */
  def initNodeStats(
    treeType: SplitCriteria.SplitCriteria,
    featureBinsInfo: Array[Bins],
    treeSeeds: Array[Int],
    mtry: Int,
    numClasses: Option[Int]): Unit = {
    val numFeatures = featureBinsInfo.length
    def createNodeStats(treeId: Int, id: Int): NodeStats = {
      val mtryFeatureIds = RandomSet.nChooseK(
        k = mtry,
        n = numFeatures,
        rnd = new Random(treeSeeds(treeId) + id)
      )
      NodeStats.createNodeStats(
        treeId = treeId,
        nodeId = id,
        nodeDepth = nodeDepths(treeId).get(id),
        treeType = treeType,
        featureBinsInfo = featureBinsInfo,
        mtryFeatureIds = mtryFeatureIds,
        numClasses = numClasses
      )
    }

    initLookUpObjs(createNodeStats)
  }

  /**
   * Get an iterator of all the nodestats. Each nodestats gets assigned an
   * incrementing hash value. This is useful to even distribute aggregated
   * nodestats to different machines to perform distributed splits.
   * @return An iterator of pairs of (hashValue, nodestats) and the count of
   *         node stats.
   */
  def toHashedNodeStatsIterator: (Iterator[(Int, NodeStats)], Int) = {
    val numTrees = idRanges.length
    var curHash = 0
    val output = new mutable.ListBuffer[(Int, NodeStats)]()
    cfor(0)(_ < numTrees, _ + 1)(
      treeId => {
        if (idRanges(treeId) != null) {
          val numIds = idRanges(treeId).endId - idRanges(treeId).startId + 1
          cfor(0)(_ < numIds, _ + 1)(
            i => {
              val nodeStats = lookUpObjs(treeId)(i)
              output += ((curHash, nodeStats))
              curHash += 1
            }
          )
        }
      }
    )

    (output.toIterator, curHash)
  }
}
