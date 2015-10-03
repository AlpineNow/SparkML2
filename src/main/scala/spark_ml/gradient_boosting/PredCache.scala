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

package spark_ml.gradient_boosting

import scala.collection.mutable

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import spark_ml.model.gb.GBInternalTree
import spark_ml.util.DiscretizedFeatureHandler

object PredCache {
  def createPredCache(
    initPreds: RDD[Double],
    shrinkage: Double,
    storageLevel: StorageLevel,
    checkpointDir: Option[String],
    checkpointInterval: Int): PredCache = {
    new PredCache(
      curPreds = initPreds,
      shrinkage = shrinkage,
      storageLevel = storageLevel,
      checkpointDir = checkpointDir,
      checkpointInterval = checkpointInterval
    )
  }
}

class PredCache(
  var curPreds: RDD[Double],
  shrinkage: Double,
  storageLevel: StorageLevel,
  checkpointDir: Option[String],
  checkpointInterval: Int) {
  private var prevPreds: RDD[Double] = null
  private var updateCount: Int = 0

  private val checkpointQueue = new mutable.Queue[RDD[Double]]()

  // Persist the initial predictions.
  curPreds = curPreds.persist(storageLevel)

  if (checkpointDir.isDefined && curPreds.sparkContext.getCheckpointDir.isEmpty) {
    curPreds.sparkContext.setCheckpointDir(checkpointDir.get)
  }

  def getRdd: RDD[Double] = curPreds

  def updatePreds[@specialized(Byte, Short) T](
    discFeatData: RDD[Array[T]],
    tree: GBInternalTree,
    featureHandler: DiscretizedFeatureHandler[T]): Unit = {
    if (prevPreds != null) {
      // Unpersist the previous one if one exists.
      prevPreds.unpersist(blocking = true)
    }

    prevPreds = curPreds

    // Need to do this since we don't want to serialize this object.
    val shk = shrinkage
    curPreds = discFeatData.zip(curPreds).map {
      case (features, curPred) =>
        curPred + shk * tree.predict(features, featureHandler)
    }.persist(storageLevel)

    updateCount += 1

    // Handle checkpointing if the directory is not None.
    if (curPreds.sparkContext.getCheckpointDir.isDefined &&
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
          println("Deleting a stale PredCache RDD checkpoint at " + old.getCheckpointFile.get)
          fs.delete(new Path(old.getCheckpointFile.get), true)
        } else {
          canDelete = false
        }
      }

      curPreds.checkpoint()
      checkpointQueue.enqueue(curPreds)
    }
  }

  def close(): Unit = {
    // Unpersist and delete all the checkpoints.
    curPreds.unpersist(blocking = true)
    if (prevPreds != null) {
      prevPreds.unpersist(blocking = true)
    }

    while (checkpointQueue.nonEmpty) {
      val old = checkpointQueue.dequeue()
      if (old.getCheckpointFile.isDefined) {
        val fs = FileSystem.get(old.sparkContext.hadoopConfiguration)
        println("Deleting a stale PredCache RDD checkpoint at " + old.getCheckpointFile.get)
        fs.delete(new Path(old.getCheckpointFile.get), true)
      }
    }
  }
}
