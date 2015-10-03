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

import spark_ml.util.{DiscretizedFeatureHandler, MapWithSequentialIntKeys}

/**
 * Update the Id of the node that a row belongs to via the split information.
 * @param splitInfo split information that will determine the new Id for
 *                  the given features.
 */
case class IdUpdater(splitInfo: NodeSplitInfo) {
  def updateId[@specialized(Byte, Short) T](
    features: Array[T],
    featureHandler: DiscretizedFeatureHandler[T]): Int = {
    if (splitInfo == null) {
      0 // 0 indicates that the data point has reached a terminal node.
    } else {
      val binId = featureHandler.convertToInt(features(splitInfo.featureId))

      // The nodeId here is usually not the same as the final tree's node Id.
      // This is more of a temporary value to refer split Ids during training.
      splitInfo.chooseChildNode(binId).nodeId
    }
  }
}

/**
 * A look up for Id updaters. This is used to update Ids of nodes that
 * data points belong to during training.
 */
class IdLookupForUpdaters(
  idRanges: Array[IdRange]
) extends IdLookup[IdUpdater](idRanges)

object IdLookupForUpdaters {
  /**
   * Create a new Id lookup object for updaters.
   * @param idRanges Id ranges of updaters.
   * @param updaterMaps Maps of Id updaters.
   * @return Id lookup object for updaters.
   */
  def createIdLookupForUpdaters(
    idRanges: Array[IdRange],
    updaterMaps: Array[MapWithSequentialIntKeys[IdUpdater]]
  ): IdLookupForUpdaters = {
    val lookup = new IdLookupForUpdaters(idRanges)
    lookup.initLookUpObjs(
      (treeId: Int, id: Int) => updaterMaps(treeId).get(id)
    )

    lookup
  }
}