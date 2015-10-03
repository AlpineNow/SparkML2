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

package spark_ml.model

import scala.collection.mutable

import spark_ml.discretization.{Bins, NumericBins}
import spark_ml.tree_ensembles.{CatNodeSplitInfo, NodeInfo, NumericNodeSplitInfo}

/**
 * A DecisionTreeNode splits on either a numeric feature or a categorical
 * feature.
 * @param nodeId Node ID.
 * @param nodeWeight Node Weight. This is useful when a never-before-seen
 *                   feature value (e.g. a new categorical value) is detected.
 *                   In such cases, a heavier child's prediction may be used.
 * @param prediction Prediction at this node.
 * @param impurity The node impurity.
 * @param featIdx The index of the feature to use.
 * @param isFeatNumeric Whether the feature is numeric or not.
 * @param splitImpurity The impurity after the split.
 * @param split An optional numeric feature split point.
 * @param leftChild An optional numeric feature left child Id.
 * @param rightChild An optional numeric feature right child Id.
 * @param nanChild An optional numeric feature 'NaN' child Id.
 * @param children An optional collection of categorical feature children.
 */
case class DecisionTreeNode(
  nodeId: Int,
  nodeWeight: Double,
  prediction: Double,
  impurity: Double,

  // These exist if there is a split. If this is a terminal node, all of the
  // following would be None.
  featIdx: Option[java.lang.Integer] = None,
  isFeatNumeric: Option[java.lang.Boolean] = None,
  splitImpurity: Option[java.lang.Double] = None,

  // Numeric split information. Only if this is a numeric split.
  split: Option[java.lang.Double] = None,
  leftChild: Option[java.lang.Integer] = None,
  rightChild: Option[java.lang.Integer] = None,
  nanChild: Option[java.lang.Integer] = None,

  // Categorical split information. Only if this is a categorical split.
  children: Option[Map[java.lang.Integer, java.lang.Integer]] = None) {
}

/**
 * Decision Tree.
 * @param nodes All the nodes in the tree.
 * @param nodeCount Number of nodes in the tree.
 */
case class DecisionTree(
  nodes: Map[java.lang.Integer, DecisionTreeNode],
  nodeCount: Int
) {
  def predict(features: Array[Double]): Double = predict(nodes(1), features)

  /**
   * Predict a value based on the given features.
   * @param node Decision tree node.
   * @param features Features.
   * @return Prediction.
   */
  def predict(node: DecisionTreeNode, features: Array[Double]): Double = {
    node.isFeatNumeric match {
      case Some(java.lang.Boolean.TRUE) => predictOnNumericFeature(node, features)
      case Some(java.lang.Boolean.FALSE) => predictOnCategoricalFeature(node, features)
      case None => node.prediction
    }
  }

  /**
   * Predict a value based on a numeric feature value.
   * @param node Decision tree node.
   * @param features Features.
   * @return Prediction.
   */
  def predictOnNumericFeature(node: DecisionTreeNode, features: Array[Double]): Double = {
    features(node.featIdx.get) match {
      case x if x.isNaN =>
        node.nanChild match {
          case Some(ncId) => predict(nodes(ncId), features)

          case None =>
            // This is an unusual situation. Print a message.
            println(
              "The numeric feature " + node.featIdx.get + " has a NaN value " +
                "but there's no NaN child node for the node " + node.nodeId
            )
            val leftChild = nodes(node.leftChild.get)
            val rightChild = nodes(node.rightChild.get)
            if (leftChild.nodeWeight > rightChild.nodeWeight) {
              predict(leftChild, features)
            } else {
              predict(rightChild, features)
            }
        }
      case x if !x.isNaN =>
        node.split match {
          case Some(splitValue) =>
            val leftChild = nodes(node.leftChild.get)
            val rightChild = nodes(node.rightChild.get)
            if (splitValue > x) {
              predict(leftChild, features)
            } else {
              predict(rightChild, features)
            }
          case None =>
            // This is an unusual situation. Print a message.
            println(
              "The node " + node.nodeId + " doesn't have a valid numeric split. " +
                "so it'll select the sole non-NaN child node."
            )
            if (node.leftChild.isDefined) {
              val leftChild = nodes(node.leftChild.get)
              predict(leftChild, features)
            } else {
              val rightChild = nodes(node.rightChild.get)
              predict(rightChild, features)
            }
        }
    }
  }

  /**
   * Predict a value based on a categorical feature value.
   * @param node Decision tree node.
   * @param features Features.
   * @return Prediction.
   */
  def predictOnCategoricalFeature(node: DecisionTreeNode, features: Array[Double]): Double = {
    val cat = new java.lang.Double(features(node.featIdx.get)).intValue()
    if (node.children.get.contains(cat)) {
      predict(nodes(node.children.get(cat)), features)
    } else {
      println("cat " + cat.toString + " for " + node.featIdx.get.toString + " has never been seen before")
      val (_, maxWeightChild) =
        node.children.get.values.foldLeft((0.0, Option[DecisionTreeNode](null))) {
          case ((curMaxWeight, curMaxWeightChild), childId) =>
            val child = nodes(childId)
            if (child.nodeWeight > curMaxWeight) {
              (child.nodeWeight, Some(child))
            } else {
              (curMaxWeight, curMaxWeightChild)
            }
        }

      predict(maxWeightChild.get, features)
    }
  }
}

object DecisionTreeUtil {
  /**
   * Create a DecisionTreeNode that corresponds to nodeInfo. This will also
   * create child nodes recursively, and also compute feature importance along
   * the way.
   * @param nodeInfo The internal nodeInfo that we want to create a
   *                 DecisionTreeNode from.
   * @param nodes A map of all the nodes in the tree.
   * @param featureImportance We calculate feature importance as well.
   * @param featureBins Feature bin information needed to convert numeric bin
   *                    Ids to actual boundary values.
   * @param nodesSoFar Nodes that have been created so far within this tree.
   *                   This is needed so as to not to create the same node
   *                   twice for categorical split mapping.
   * @return A DecisionTreeTreeNode that corresponds to nodeInfo.
   */
  def createDecisionTreeNode(
    nodeInfo: NodeInfo,
    nodes: mutable.Map[Int, NodeInfo],
    featureImportance: Array[Double],
    featureBins: Array[Bins],
    nodesSoFar: mutable.Map[java.lang.Integer, DecisionTreeNode]
  ): DecisionTreeNode = {
    val nodeId = nodeInfo.nodeId
    if (nodesSoFar.contains(nodeId)) {
      // If the node corresponding to this node has already been created, simply
      // retrieve it.
      return nodesSoFar(nodeId)
    }

    val nodeWeight = nodeInfo.weight
    val prediction = nodeInfo.prediction
    val impurity = nodeInfo.impurity
    if (nodeInfo.splitInfo.isDefined) {
      val featIdx = Some(nodeInfo.splitInfo.get.featureId)
      val isFeatNumeric =
        Some(nodeInfo.splitInfo.get.isInstanceOf[NumericNodeSplitInfo])
      featureImportance(featIdx.get) +=
        (nodeInfo.impurity - nodeInfo.splitInfo.get.splitImpurity) * nodeWeight
      if (isFeatNumeric.get) {
        val numericNodeSplitInfo =
          nodeInfo.splitInfo.get.asInstanceOf[NumericNodeSplitInfo]
        val split =
          if (numericNodeSplitInfo.leftChildNode != null && numericNodeSplitInfo.rightChildNode != null) {
            Some(
              featureBins(featIdx.get).asInstanceOf[NumericBins].bins(
                numericNodeSplitInfo.splitBinId
              ).lower
            )
          } else {
            None
          }
        val leftChild: Option[java.lang.Integer] =
          if (numericNodeSplitInfo.leftChildNode != null) {
            Some(
              this.createDecisionTreeNode(
                nodes(numericNodeSplitInfo.leftChildNode.nodeId),
                nodes,
                featureImportance,
                featureBins,
                nodesSoFar
              ).nodeId
            )
          } else {
            None
          }
        val rightChild: Option[java.lang.Integer] =
          if (numericNodeSplitInfo.rightChildNode != null) {
            Some(
              this.createDecisionTreeNode(
                nodes(numericNodeSplitInfo.rightChildNode.nodeId),
                nodes,
                featureImportance,
                featureBins,
                nodesSoFar
              ).nodeId
            )
          } else {
            None
          }
        val nanChild: Option[java.lang.Integer] =
          if (numericNodeSplitInfo.nanChildNode.isDefined) {
            Some(
              this.createDecisionTreeNode(
                nodes(numericNodeSplitInfo.nanChildNode.get.nodeId),
                nodes,
                featureImportance,
                featureBins,
                nodesSoFar
              ).nodeId
            )
          } else {
            None
          }
        val newNode = DecisionTreeNode(
          nodeId = nodeId,
          nodeWeight = nodeWeight,
          prediction = prediction,
          impurity = impurity,
          featIdx = featIdx.map(new java.lang.Integer(_)),
          isFeatNumeric = isFeatNumeric.map(new java.lang.Boolean(_)),
          splitImpurity = Some(new java.lang.Double(numericNodeSplitInfo.splitImpurity)),
          split = split.map(new java.lang.Double(_)),
          leftChild = leftChild,
          rightChild = rightChild,
          nanChild = nanChild
        )
        nodesSoFar.put(nodeId, newNode)
        newNode
      } else {
        val catNodeSplitInfo =
          nodeInfo.splitInfo.get.asInstanceOf[CatNodeSplitInfo]
        val newNode = DecisionTreeNode(
          nodeId = nodeId,
          nodeWeight = nodeWeight,
          prediction = prediction,
          impurity = impurity,
          featIdx = featIdx.map(new java.lang.Integer(_)),
          isFeatNumeric = isFeatNumeric.map(new java.lang.Boolean(_)),
          splitImpurity = Some(new java.lang.Double(catNodeSplitInfo.splitImpurity)),
          children = Some(
            // This assumes that the 'map' function is not parallel, but
            // sequential.
            catNodeSplitInfo.binIdToChildNode.map {
              case (binId, childIdx) =>
                new java.lang.Integer(binId) ->
                  new java.lang.Integer(
                    this.createDecisionTreeNode(
                      nodes(catNodeSplitInfo.orderedChildNodes(childIdx).nodeId),
                      nodes,
                      featureImportance,
                      featureBins,
                      nodesSoFar
                    ).nodeId
                  )
            }.toMap
          )
        )
        nodesSoFar.put(nodeId, newNode)
        newNode
      }
    } else {
      val newNode = DecisionTreeNode(
        nodeId = nodeId,
        nodeWeight = nodeWeight,
        prediction = prediction,
        impurity = impurity
      )
      nodesSoFar.put(nodeId, newNode)
      newNode
    }
  }
}
