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

import scala.collection.mutable

import org.apache.spark.sql.DataFrame

object DistinctValueCounter {
  /**
   * Get distinct values of categorical columns from the given data frame.
   * @param dataFrame Data frame from whose columns we want to gather distinct
   *                  values from.
   * @param catColIndices Categorical column indices.
   * @param maxCardinality The maximum number of unique values per column.
   * @return A map of column index and distinct value sets.
   */
  def getDistinctValues(
    dataFrame: DataFrame,
    catColIndices: Set[Int],
    maxCardinality: Int): Map[Int, mutable.Set[String]] = {
    dataFrame.mapPartitions(
      rowItr => {
        val distinctVals = mutable.Map[Int, mutable.Set[String]]()
        while (rowItr.hasNext) {
          val row = rowItr.next()
          row.toSeq.zipWithIndex.map {
            case (colVal, colIdx) =>
              if (catColIndices.contains(colIdx)) {
                val colDistinctVals = distinctVals.getOrElseUpdate(colIdx, mutable.Set[String]())

                // We don't care to count all the unique values if the distinct
                // count goes over the given limit.
                if (colDistinctVals.size <= maxCardinality) {
                  val nonNullColVal =
                    if (colVal == null) {
                      ""
                    } else {
                      colVal.toString
                    }
                  colDistinctVals.add(nonNullColVal)
                }
              }
          }
        }

        distinctVals.toIterator
      }
    ).reduceByKey {
      (colDistinctVals1, colDistinctVals2) =>
        (colDistinctVals1 ++ colDistinctVals2).splitAt(maxCardinality + 1)._1
    }.collect().toMap
  }

  /**
   * Map a set of distinct values to an increasing non-negative numbers.
   * E.g., {'Women' -> 0, 'Men' -> 1}, etc.
   * @param distinctValues A set of distinct values for different columns (first
   *                       key is index to a column).
   * @param useEmptyString Whether an empty string should be used as a distinct
   *                       value.
   * @return A map of distinct values to integers for different columns (first
   *         key is index to a column).
   */
  def mapDistinctValuesToIntegers(
    distinctValues: Map[Int, mutable.Set[String]],
    useEmptyString: Boolean
  ): Map[Int, Map[String, java.lang.Integer]] = {
    distinctValues.map {
      case (colIndex, values) =>
        colIndex -> values.filter(value => !(value == "" && !useEmptyString)).zipWithIndex.map {
          case (value, mappedVal) => value -> new java.lang.Integer(mappedVal)
        }.toMap
    }
  }
}
