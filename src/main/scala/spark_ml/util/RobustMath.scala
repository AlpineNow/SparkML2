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

/**
 * Some math functions that don't give NaN's for edge cases (e.g. log of 0's) or
 * too large/small numbers of exp's.
 * These are used for loss function calculations.
 */
object RobustMath {
  private val minExponent = -19.0
  private val maxExponent = 19.0
  private val expPredLowerLimit = math.exp(minExponent)
  private val expPredUpperLimit = math.exp(maxExponent)

  def log(value: Double): Double = {
    if (value == 0.0) {
      minExponent
    } else {
      math.min(math.max(math.log(value), minExponent), maxExponent)
    }
  }

  def exp(value: Double): Double = {
    math.min(math.max(math.exp(value), expPredLowerLimit), expPredUpperLimit)
  }
}
