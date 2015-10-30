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

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import spark_ml.util.DiscretizedFeatureHandler
import spire.implicits._

object IdCache {
  /**
   * Create a new Id cache object, filled with initial Ids (1's).
   * @param numTrees Number of trees we expect.
   * @param data A data RDD or an RDD that contains the same number of rows as
   *             the data.
   * @param storageLevel Cache storage level.
   * @param checkpointDir Checkpoint directory for intermediate checkpointing.
   * @param checkpointInterval Checkpointing interval.
   * @tparam T Type of data rows.
   * @return A new IdCache object.
   */
  def createIdCache[T](
    numTrees: Int,
    data: RDD[T],
    storageLevel: StorageLevel,
    checkpointDir: Option[String],
    checkpointInterval: Int): IdCache = {
    new IdCache(
      // All the Ids start with '1', meaning that all the rows are
      // assigned to the root node.
      curIds = data.map(_ => Array.fill[Int](numTrees)(1)),
      storageLevel = storageLevel,
      checkpointDir = checkpointDir,
      checkpointInterval = checkpointInterval
    )
  }
}

/**
 * Id cache to keep track of Ids of rows to indicate which tree nodes that
 * rows to belong to.
 * @param curIds RDD of the current Ids per data row. Each row of the RDD
 *               is an array of Ids, each element an Id for a tree.
 * @param storageLevel Cache storage level.
 * @param checkpointDir Checkpoint directory.
 * @param checkpointInterval Checkpoint interval.
 */
class IdCache(
  var curIds: RDD[Array[Int]],
  storageLevel: StorageLevel,
  checkpointDir: Option[String],
  checkpointInterval: Int) {
  private var prevIds: RDD[Array[Int]] = null
  private var updateCount: Int = 0

  // To keep track of last checkpointed RDDs.
  private val checkpointQueue = new mutable.Queue[RDD[Array[Int]]]()

  // Persist the initial Ids.
  curIds = curIds.persist(storageLevel)

  // If a checkpoint directory is given, and there's no prior checkpoint
  // directory, then set the checkpoint directory with the given one.
  if (checkpointDir.isDefined && curIds.sparkContext.getCheckpointDir.isEmpty) {
    curIds.sparkContext.setCheckpointDir(checkpointDir.get)
  }

  /**
   * Get the current Id RDD.
   * @return curIds RDD.
   */
  def getRdd: RDD[Array[Int]] = curIds

  /**
   * Update Ids that are stored in the cache RDD.
   * @param data RDD of data rows needed to find the updated Ids.
   * @param idLookupForUpdaters Id updaters.
   * @param featureHandler Data row feature type handler.
   * @tparam T Type of feature.
   */
  def updateIds[@specialized(Byte, Short) T](
    data: RDD[((Double, Array[T]), Array[Byte])],
    idLookupForUpdaters: IdLookupForUpdaters,
    featureHandler: DiscretizedFeatureHandler[T]): Unit = {
    if (prevIds != null) {
      // Unpersist the previous one if one exists.
      prevIds.unpersist(blocking = true)
    }

    prevIds = curIds

    // Update Ids.
    curIds = data.zip(curIds).map {
      case (((label, features), baggedCounts), nodeIds) =>
        val numTrees = nodeIds.length
        cfor(0)(_ < numTrees, _ + 1)(
          treeId => {
            val curNodeId = nodeIds(treeId)
            val rowCnt = baggedCounts(treeId)
            if (rowCnt > 0 && curNodeId != 0) {
              val idUpdater = idLookupForUpdaters.get(
                treeId = treeId,
                id = curNodeId
              )
              if (idUpdater != null) {
                nodeIds(treeId) = idUpdater.updateId(features = features, featureHandler = featureHandler)
              }
            }
          }
        )

        nodeIds
    }.persist(storageLevel)

    updateCount += 1

    // Handle checkpointing if the directory is not None.
    if (curIds.sparkContext.getCheckpointDir.isDefined &&
      (updateCount % checkpointInterval) == 0) {
      // See if we can delete previous checkpoints.
      var canDelete = true
      while (checkpointQueue.size > 1 && canDelete) {
        // We can delete the oldest checkpoint iff the next checkpoint actually
        // exists in the file system.
        if (checkpointQueue.get(1).get.getCheckpointFile.isDefined) {
          val old = checkpointQueue.dequeue()

          // Since the old checkpoint is not deleted by Spark, we'll manually
          // delete it here.
          val fs = FileSystem.get(old.sparkContext.hadoopConfiguration)
          println("Deleting a stale IdCache RDD checkpoint at " + old.getCheckpointFile.get)
          fs.delete(new Path(old.getCheckpointFile.get), true)
        } else {
          canDelete = false
        }
      }

      curIds.checkpoint()
      checkpointQueue.enqueue(curIds)
    }
  }

  /**
   * Unpersist all the RDDs stored internally.
   */
  def close(): Unit = {
    // Unpersist and delete all the checkpoints.
    curIds.unpersist(blocking = true)
    if (prevIds != null) {
      prevIds.unpersist(blocking = true)
    }

    while (checkpointQueue.nonEmpty) {
      val old = checkpointQueue.dequeue()
      if (old.getCheckpointFile.isDefined) {
        val fs = FileSystem.get(old.sparkContext.hadoopConfiguration)
        println("Deleting a stale IdCache RDD checkpoint at " + old.getCheckpointFile.get)
        fs.delete(new Path(old.getCheckpointFile.get), true)
      }
    }
  }
}
