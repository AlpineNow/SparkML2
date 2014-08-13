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

/**
 * Variable importance of each feature.
 */
case class VarImportance(numFeatures: Int) {
  val featureImportance = Array.fill[Double](numFeatures)(0.0)

  /**
   * Add variable importance of a feature.
   * @param featId The feature index.
   * @param varImp Importance to add for the feature.
   */
  def addVarImportance(featId: Int, varImp: Double): Unit = {
    featureImportance(featId) += varImp
  }

  /**
   * Extract variable importance from a split node.
   * And add the measure to this object.
   * @param node A split tree node from which to extract variable importance.
   */
  def addVarImportance(node: SequoiaNode): Unit = {
    val impurityReduction = node.impurity - node.splitImpurity.get
    val weight = node.weight
    val featId = node.split.get.featureId

    addVarImportance(featId, impurityReduction * weight)
  }
}
