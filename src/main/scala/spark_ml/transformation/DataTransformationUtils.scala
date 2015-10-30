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

package spark_ml.transformation

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import spark_ml.discretization.CardinalityOverLimitException

/**
 * A class that encodes the transformation to a numeric value for a column.
 * @param distinctValToInt If this is defined, then the column value would be
 *                         transformed to a mapped integer.
 * @param maxCardinality If this is defined, then the column value would be
 *                       transformed into an integer value in
 *                       [0, maxCardinality) through hashing.
 * @return a transformed numeric value for a column value. If neither
 *         distinctValToInt nor maxCardinality is defined, then the value
 *         would be assumed to be a numeric value already and passed through.
 */
case class ColumnTransformer(
  // We are using java.lang.Integer so that this can be easily serialized by
  // Gson.
  distinctValToInt: Option[Map[String, java.lang.Integer]],
  maxCardinality: Option[java.lang.Integer]) {
  def transform(value: String): Double = {
    if (distinctValToInt.isDefined) {
      val nonNullValue =
        if (value == null) {
          ""
        } else {
          value
        }
      distinctValToInt.get(nonNullValue).toDouble
    } else if (maxCardinality.isDefined) {
      val nonNullValue =
        if (value == null) {
          ""
        } else {
          value
        }
      DataTransformationUtils.getSimpleHashedValue(nonNullValue, maxCardinality.get)
    } else {
      if (value == null) {
        Double.NaN
      } else {
        value.toDouble
      }
    }
  }
}

object DataTransformationUtils {
  /**
   * Convert the given data frame into an RDD of label, feature vector pairs.
   * All label/feature values are also converted into Double.
   * @param dataFrame Spark Data Frame that we want to convert.
   * @param labelColIndex Label column index.
   * @param catDistinctValToInt Categorical column distinct value to int maps.
   *                            This is used to map distinct string values to
   *                            numeric values (doubles).
   * @param colsToIgnoreIndices Indices of columns in the data frame to be
   *                            ignored (not used as features or label).
   * @param maxCatCardinality Maximum categorical cardinality we allow. If the
   *                          cardinality goes over this, feature hashing might
   *                          be used (or will simply throw an exception).
   * @param useFeatureHashing Whether feature hashing should be used on
   *                          categorical columns whose unique value counts
   *                          exceed the maximum cardinality.
   * @return An RDD of label/feature-vector pairs and column transformer definitions.
   */
  def transformDataFrameToLabelFeatureRdd(
    dataFrame: DataFrame,
    labelColIndex: Int,
    catDistinctValToInt: Map[Int, Map[String, java.lang.Integer]],
    colsToIgnoreIndices: Set[Int],
    maxCatCardinality: Int,
    useFeatureHashing: Boolean): (RDD[(Double, Array[Double])], (ColumnTransformer, Array[ColumnTransformer])) = {
    val transformedRDD = dataFrame.map(row => {
      val labelValue = row.get(labelColIndex)

      Tuple2(
        if (catDistinctValToInt.contains(labelColIndex)) {
          val nonNullLabelValue =
            if (labelValue == null) {
              ""
            } else {
              labelValue.toString
            }
          mapCategoricalValueToNumericValue(
            labelColIndex,
            nonNullLabelValue,
            catDistinctValToInt(labelColIndex),
            maxCatCardinality,
            useFeatureHashing
          )
        } else {
          if (labelValue == null) {
            Double.NaN
          } else {
            labelValue.toString.toDouble
          }
        },
        row.toSeq.zipWithIndex.flatMap {
          case (colVal, idx) =>
            if (colsToIgnoreIndices.contains(idx) || (labelColIndex == idx)) {
              Array[Double]().iterator
            } else {
              if (catDistinctValToInt.contains(idx)) {
                val nonNullColVal =
                  if (colVal == null) {
                    ""
                  } else {
                    colVal.toString
                  }
                Array(
                  mapCategoricalValueToNumericValue(
                    idx,
                    nonNullColVal,
                    catDistinctValToInt(idx),
                    maxCatCardinality,
                    useFeatureHashing
                  )
                ).iterator
              } else {
                val nonNullColVal =
                  if (colVal == null) {
                    Double.NaN
                  } else {
                    colVal.toString.toDouble
                  }
                Array(nonNullColVal).iterator
              }
            }
        }.toArray
      )
    })

    val numCols = dataFrame.columns.length
    val colTransformers = Tuple2(
      if (catDistinctValToInt.contains(labelColIndex)) {
        if (catDistinctValToInt(labelColIndex).size <= maxCatCardinality) {
          ColumnTransformer(Some(catDistinctValToInt(labelColIndex)), None)
        } else {
          ColumnTransformer(None, Some(maxCatCardinality))
        }
      } else {
        ColumnTransformer(None, None)
      },
      (0 to (numCols - 1)).flatMap {
        case (colIdx) =>
          if (colsToIgnoreIndices.contains(colIdx) || (labelColIndex == colIdx)) {
            Array[ColumnTransformer]().iterator
          } else {
            if (catDistinctValToInt.contains(colIdx)) {
              if (catDistinctValToInt(colIdx).size <= maxCatCardinality) {
                Array(ColumnTransformer(Some(catDistinctValToInt(colIdx)), None)).iterator
              } else {
                Array(ColumnTransformer(None, Some(maxCatCardinality))).iterator
              }
            } else {
              Array(ColumnTransformer(None, None)).iterator
            }
          }
      }.toArray
    )

    (transformedRDD, colTransformers)
  }

  /**
   * Map a categorical value (string) to a numeric value (double).
   * @param categoryId Equal to the column index.
   * @param catValue Categorical value that we want to map to a numeric value.
   * @param catValueToIntMap A map from categorical value to integers. If the
   *                         cardinality of the category is less than or equal
   *                         to maxCardinality, this is used.
   * @param maxCardinality The maximum allowed cardinality.
   * @param useFeatureHashing Whether feature hashing should be performed if the
   *                          cardinality of the category exceeds the maximum
   *                          cardinality.
   * @return A mapped double value.
   */
  def mapCategoricalValueToNumericValue(
    categoryId: Int,
    catValue: String,
    catValueToIntMap: Map[String, java.lang.Integer],
    maxCardinality: Int,
    useFeatureHashing: Boolean): Double = {
    if (catValueToIntMap.size <= maxCardinality) {
      catValueToIntMap(catValue).toDouble
    } else {
      if (useFeatureHashing) {
        getSimpleHashedValue(catValue, maxCardinality)
      } else {
        throw new CardinalityOverLimitException(
          "The categorical column with the index " + categoryId +
            " has a cardinality that exceeds the limit " + maxCardinality)
      }
    }
  }

  /**
   * Compute a simple hash of a categorical value.
   * @param catValue Categorical value that we want to compute a simple numeric
   *                 hash for. The hash value would be an integer between 0 and
   *                 (maxCardinality - 1).
   * @param maxCardinality The value of the hash is limited within
   *                       [0, maxCardinality).
   * @return A hashed value of double.
   */
  def getSimpleHashedValue(catValue: String, maxCardinality: Int): Double = {
    ((catValue.hashCode.toLong - Int.MinValue.toLong) % maxCardinality.toLong).toDouble
  }
}
