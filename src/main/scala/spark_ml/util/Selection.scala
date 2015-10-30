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

package spark_ml.util

import scala.util.Random

/**
 * Selection algorithm utils.
 * Adopted with some modifications from the java code at :
 * http://blog.teamleadnet.com/2012/07/quick-select-algorithm-find-kth-element.html
 */
object Selection {
  /**
   * A quick select algorithm used to select the n'th number from an array.
   * @param array Array of Doubles.
   * @param s The starting index of the segment that we are looking at.
   * @param e The ending position (ending index + 1) of the segment that we are
   *          looking at.
   * @param n We're looking for the n'th number if the array is sorted.
   * @param rng A random number generator.
   * @return The n'th number in the array.
   */
  def quickSelect(
    array: Array[Double],
    s: Int,
    e: Int,
    n: Int,
    rng: Random): Double = {

    var from = s
    var to = e - 1
    while (from < to) {
      var r = from
      var w = to
      val pivotIdx = rng.nextInt(to - from + 1) + from
      val pivotVal = array(pivotIdx)
      while (r < w) {
        if (array(r) >= pivotVal) {
          val tmp = array(w)
          array(w) = array(r)
          array(r) = tmp
          w -= 1
        } else {
          r += 1
        }
      }

      if (array(r) > pivotVal) {
        r -= 1
      }

      if (n <= r) {
        to = r
      } else {
        from = r + 1
      }
    }

    array(n)
  }
}
