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

package spark_ml.transformation.category_transformation

import org.apache.spark.rdd.RDD

import spire.implicits._

/**
 * Transform categorical features to an integer through feature hashing.
 */
object FeatureHashing {
  /**
   * Hash categorical feature values (strings) to an integer value. This is usually useful for large-cardinality categorical features.
   * @param data The raw RDD of strings. Each string line is expected to be a delimited row of column values.
   * @param categoricalFeatureNames Column names of the categorical features.
   * @param hashFunction The hash function to apply to string values.
   * @param leaveEmptyStrings Whether to leave empty strings unprocessed, or to treat them as a categorical value.
   * @param delimiter Delimiter to split lines into columns.
   * @param header The header of column names.
   * @param headerExists Whether the header also exists in the RDD or not.
   * @return A transformed RDD.
   */
  def hashCategoricalFeatures(
    data: RDD[String],
    categoricalFeatureNames: Set[String],
    hashFunction: String => Int,
    leaveEmptyStrings: Boolean,
    delimiter: String,
    header: Array[String],
    headerExists: Boolean): RDD[String] = {
    data.mapPartitionsWithIndex((index: Int, lines: Iterator[String]) => {
      var headerWritten = false
      lines.map(line => {
        if (index == 0 && headerExists && !headerWritten) {
          headerWritten = true
          line
        } else {
          val lineElems = line.split(delimiter, -1)
          var outputLine = ""
          cfor(0)(_ < header.length, _ + 1)(
            idx => {
              val colName = header(idx)
              val outputDelimiter = if (idx == 0) "" else delimiter
              if (categoricalFeatureNames.contains(colName)) {
                outputLine += outputDelimiter + hashFunction(lineElems(idx)).toString
              } else {
                outputLine += outputDelimiter + lineElems(idx)
              }
            }
          )

          outputLine
        }
      })
    })
  }
}

/**
 * A simple hash function.
 * @param maxCardinality The maximum size of the hash space.
 */
case class SimpleHasher(maxCardinality: Int) {
  def hashString(string: String): Int = {
    ((string.hashCode.toLong - Int.MinValue.toLong) % maxCardinality.toLong).toInt
  }
}
