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

import spark_ml.gradient_boosting.loss.LossFunction
import spark_ml.tree_ensembles.CatSplitType
import spark_ml.util.BaggingType

/**
 * Gradient boosting trainer options.
 * @param numTrees Number of trees to build. The algorithm can also determine
 *                 the optimal number of trees if a validation data is provided.
 * @param maxTreeDepth Maximum tree depth allowed per tree.
 * @param minSplitSize Min split size allowed per tree.
 * @param lossFunction The loss function object to use.
 * @param catSplitType How to split categorical features.
 * @param baggingRate Bagging rate.
 * @param baggingType Whether to bag with/without replacements.
 * @param shrinkage Shrinkage.
 * @param fineTuneTerminalNodes Whether to fine-tune tree's terminal nodes so
 *                              that their values are directly optimizing
 *                              against the loss function.
 * @param checkpointDir Checkpoint directory.
 * @param predCheckpointInterval Intermediate prediction checkpointing interval.
 * @param idCacheCheckpointInterval Id cache checkpointing interval.
 * @param verbose If true, the algorithm will print as much information through
 *                the notifiee as possible, including many intermediate
 *                computation values, etc.
 */
case class GradientBoostingOptions(
  numTrees: Int,
  maxTreeDepth: Int,
  minSplitSize: Int,
  lossFunction: LossFunction,
  catSplitType: CatSplitType.CatSplitType,
  baggingRate: Double,
  baggingType: BaggingType.BaggingType,
  shrinkage: Double,
  fineTuneTerminalNodes: Boolean,
  checkpointDir: Option[String],
  predCheckpointInterval: Int,
  idCacheCheckpointInterval: Int,
  verbose: Boolean) {
  override def toString: String = {
    "=========================" + "\n" +
      "Gradient Boosting Options" + "\n" +
      "=========================" + "\n" +
      "numTrees                   : " + numTrees + "\n" +
      "maxTreeDepth               : " + maxTreeDepth + "\n" +
      "minSplitSize               : " + minSplitSize + "\n" +
      "lossFunction               : " + lossFunction.getClass.getSimpleName + "\n" +
      "catSplitType               : " + catSplitType.toString + "\n" +
      "baggingRate                : " + baggingRate + "\n" +
      "baggingType                : " + baggingType.toString + "\n" +
      "shrinkage                  : " + shrinkage + "\n" +
      "fineTuneTerminalNodes      : " + fineTuneTerminalNodes + "\n" +
      "checkpointDir              : " + (checkpointDir match { case None => "None" case Some(dir) => dir }) + "\n" +
      "predCheckpointInterval     : " + predCheckpointInterval + "\n" +
      "idCacheCheckpointInterval  : " + idCacheCheckpointInterval + "\n" +
      "verbose                    : " + verbose + "\n"
  }
}
