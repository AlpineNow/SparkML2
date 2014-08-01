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

import scala.collection.mutable

/**
 * Helper functions to select random samples.
 */
object RandomSet {
  def nChooseK(k: Int, n: Int, rnd: scala.util.Random): Array[Int] = {
    val indices = new mutable.ArrayBuilder.ofInt
    var remains = k

    for (i <- 0 to n - 1) {
      if (rnd.nextInt(n - i) < remains) {
        indices += i
        remains -= 1
      }
    }

    indices.result()
  }
}
