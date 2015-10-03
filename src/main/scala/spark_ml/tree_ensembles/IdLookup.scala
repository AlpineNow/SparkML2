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

import spire.implicits._

/**
 * The Id lookup object will contain sequential Ids from startId to endId.
 * @param startId Start Id.
 * @param endId End Id.
 */
case class IdRange(
  var startId: Int,
  var endId: Int)

/**
 * Each data point belongs to a particular tree's node and is assigned the Id of
 * the node (the Id is only for training). This structure is used to find any
 * object (e.g. aggregator) that corresponds to the node that the point belongs
 * to.
 * @param idRanges Id ranges for each tree.
 * @tparam T The type of the look up objects.
 */
abstract class IdLookup[T: ClassTag](idRanges: Array[IdRange])
    extends Serializable {
  // This is the array that contains the corresponding objects for each Id.
  protected var lookUpObjs: Array[Array[T]] = null
  var objCnt: Int = 0

  // This is used to initialize the look up obj for tree/node pairs.
  protected def initLookUpObjs(
    initLookUpObj: (Int, Int) => T): Unit = {
    val numTrees = idRanges.length
    lookUpObjs = Array.fill[Array[T]](numTrees)(null)

    // Initialize the look up objects for tree/nodes.
    // There's one array per tree.
    cfor(0)(_ < numTrees, _ + 1)(
      treeId => {
        val idRange = idRanges(treeId)
        if (idRange != null) {
          val numIds = idRange.endId - idRange.startId + 1
          lookUpObjs(treeId) = Array.fill[T](numIds)(null.asInstanceOf[T])
          cfor(idRange.startId)(_ <= idRange.endId, _ + 1)(
            id => {
              val lookUpObjIdx = id - idRange.startId
              lookUpObjs(treeId)(lookUpObjIdx) = initLookUpObj(treeId, id)
              objCnt += 1
            }
          )
        }
      }
    )
  }

  /**
   * Get Id ranges.
   * @return Id ranges.
   */
  def getIdRanges: Array[IdRange] = idRanges

  /**
   * Get a look up object for the given tree/node.
   * @param treeId Tree Id.
   * @param id Id representing the node.
   *           During training, nodes can be assigned arbitrary Ids.
   *           They are not necessarily the final model node Ids.
   * @return The corresponding look up object.
   */
  def get(treeId: Int, id: Int): T = {
    val idRange = idRanges(treeId)
    if (idRange != null) {
      val startId = idRange.startId
      val endId = idRange.endId
      if (id >= startId && id <= endId) {
        lookUpObjs(treeId)(id - startId)
      } else {
        null.asInstanceOf[T]
      }
    } else {
      null.asInstanceOf[T]
    }
  }
}
