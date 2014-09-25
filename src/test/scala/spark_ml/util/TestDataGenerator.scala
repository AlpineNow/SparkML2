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
 * Generate test data.
 */
object TestDataGenerator {
  def labeledData1: Array[(Double, Array[Double])] = {
    Array(
      (0.0, Array(0.0, 0.0, -52.28)),
      (0.0, Array(1.0, 0.0, -32.16)),
      (1.0, Array(2.0, 0.0, -73.68)),
      (1.0, Array(3.0, 0.0, -26.38)),
      (2.0, Array(4.0, 0.0, 13.69)),
      (2.0, Array(0.0, 1.0, 42.07)),
      (2.0, Array(1.0, 1.0, 22.96)),
      (3.0, Array(2.0, 1.0, -33.43)),
      (3.0, Array(3.0, 1.0, -61.80)),
      (0.0, Array(4.0, 1.0, -81.34)),
      (2.0, Array(0.0, 1.0, -68.49)),
      (2.0, Array(1.0, 1.0, 64.17)),
      (3.0, Array(2.0, 1.0, 20.88)),
      (3.0, Array(3.0, 1.0, 27.75)),
      (0.0, Array(4.0, 1.0, 59.07)),
      (0.0, Array(0.0, 2.0, -53.55)),
      (1.0, Array(1.0, 2.0, 25.89)),
      (1.0, Array(2.0, 2.0, 22.62)),
      (2.0, Array(3.0, 2.0, -5.63)),
      (2.0, Array(4.0, 2.0, 81.67)),
      (0.0, Array(0.0, 2.0, -72.87)),
      (1.0, Array(1.0, 2.0, 25.51)),
      (1.0, Array(2.0, 2.0, 43.14)),
      (2.0, Array(3.0, 2.0, 60.53)),
      (2.0, Array(4.0, 2.0, 88.94)),
      (0.0, Array(0.0, 2.0, 17.08)),
      (1.0, Array(1.0, 2.0, 69.48)),
      (1.0, Array(2.0, 2.0, -76.47)),
      (2.0, Array(3.0, 2.0, 90.90)),
      (2.0, Array(4.0, 2.0, -79.67))
    )
  }

  def labeledData2: Array[(Double, Array[Double])] = {
    Array(
      (0.0, Array(0.0, 0.0)),
      (0.0, Array(1.0, 0.0)),
      (1.0, Array(2.0, 0.0)),
      (1.0, Array(3.0, 0.0)),
      (2.0, Array(4.0, 0.0)),
      (2.0, Array(0.0, 1.0)),
      (2.0, Array(1.0, 1.0)),
      (3.0, Array(2.0, 1.0)),
      (3.0, Array(3.0, 1.0)),
      (0.0, Array(4.0, 1.0)),
      (2.0, Array(0.0, 1.0)),
      (2.0, Array(1.0, 1.0)),
      (3.0, Array(2.0, 1.0)),
      (3.0, Array(3.0, 1.0)),
      (0.0, Array(4.0, 1.0)),
      (0.0, Array(0.0, 2.0)),
      (1.0, Array(1.0, 2.0)),
      (1.0, Array(2.0, 2.0)),
      (2.0, Array(3.0, 2.0)),
      (2.0, Array(4.0, 2.0)),
      (0.0, Array(0.0, 2.0)),
      (1.0, Array(1.0, 2.0)),
      (1.0, Array(2.0, 2.0)),
      (2.0, Array(3.0, 2.0)),
      (2.0, Array(4.0, 2.0)),
      (0.0, Array(0.0, 2.0)),
      (1.0, Array(1.0, 2.0)),
      (1.0, Array(2.0, 2.0)),
      (2.0, Array(3.0, 2.0)),
      (2.0, Array(4.0, 2.0))
    )
  }

  def labeledData3: Array[(Double, Array[Double])] = {
    Array(
      (0.0, Array(0.0, 0.0)),
      (0.0, Array(1.0, 0.0)),
      (1.0, Array(2.0, 0.0)),
      (1.1, Array(3.0, 0.0)),
      (2.0, Array(4.0, 0.0)),
      (2.3, Array(0.0, 1.0)),
      (2.0, Array(1.0, 1.0)),
      (3.0, Array(2.0, 1.0)),
      (3.5, Array(3.0, 1.0)),
      (0.0, Array(4.0, 1.0)),
      (2.0, Array(0.0, 1.0)),
      (2.0, Array(1.0, 1.0)),
      (3.0, Array(2.0, 1.0)),
      (3.2, Array(3.0, 1.0)),
      (0.0, Array(4.0, 1.0)),
      (0.0, Array(0.0, 2.0)),
      (1.0, Array(1.0, 2.0)),
      (1.0, Array(2.0, 2.0)),
      (2.0, Array(3.0, 2.0)),
      (2.0, Array(4.0, 2.0)),
      (0.0, Array(0.0, 2.0)),
      (1.0, Array(1.0, 2.0)),
      (1.0, Array(2.0, 2.0)),
      (2.0, Array(3.0, 2.0)),
      (2.0, Array(4.0, 2.0)),
      (0.0, Array(0.0, 2.0)),
      (1.0, Array(1.0, 2.0)),
      (1.0, Array(2.0, 2.0)),
      (2.0, Array(3.0, 2.0)),
      (2.0, Array(4.0, 2.0))
    )
  }

  def labeledData4: Array[(Double, Array[Double])] = {
    Array(
      (-1.0, Array(0.0, 0.0)),
      (0.0, Array(1.0, 0.0)),
      (1.0, Array(2.0, 0.0)),
      (1.1, Array(3.0, 0.0)),
      (2.0, Array(4.0, 0.0)),
      (2.3, Array(0.0, 1.0)),
      (2.0, Array(1.0, 1.0)),
      (3.0, Array(2.0, 1.0)),
      (3.5, Array(3.0, 1.0)),
      (0.0, Array(4.0, 1.0)),
      (2.0, Array(0.0, 1.0)),
      (2.0, Array(1.0, 1.0)),
      (3.0, Array(2.0, 1.0)),
      (3.2, Array(3.0, 1.0)),
      (0.0, Array(4.0, 1.0)),
      (0.0, Array(0.0, 2.0)),
      (1.0, Array(1.0, 2.0)),
      (1.0, Array(2.0, 2.0)),
      (2.0, Array(3.0, 2.0)),
      (2.0, Array(4.0, 2.0)),
      (0.0, Array(0.0, 2.0)),
      (1.0, Array(1.0, 2.0)),
      (1.0, Array(2.0, 2.0)),
      (2.0, Array(3.0, 2.0)),
      (2.0, Array(4.0, 2.0)),
      (0.0, Array(0.0, 2.0)),
      (1.0, Array(1.0, 2.0)),
      (1.0, Array(2.0, 2.0)),
      (2.0, Array(3.0, 2.0)),
      (2.0, Array(4.0, 2.0))
    )
  }

  def labeledData5: Array[(Double, Array[Double])] = {
    Array(
      (0.0, Array(0.0, 0.0, -52.28)),
      (0.0, Array(1.0, 0.0, -32.16)),
      (1.0, Array(2.0, 0.0, -73.68)),
      (1.0, Array(3.0, 0.0, -26.38)),
      (2.0, Array(Double.NaN, 0.0, 13.69)),
      (2.0, Array(0.0, 1.0, 42.07)),
      (2.0, Array(1.0, 1.0, 22.96)),
      (3.0, Array(2.0, 1.0, -33.43)),
      (3.0, Array(3.0, 1.0, -61.80)),
      (0.0, Array(4.0, 1.0, -81.34)),
      (2.0, Array(0.0, 1.0, -68.49)),
      (2.0, Array(1.0, 1.0, 64.17)),
      (3.0, Array(2.0, Double.NaN, 20.88)),
      (3.0, Array(3.0, 1.0, 27.75)),
      (0.0, Array(4.0, 1.0, 59.07)),
      (0.0, Array(0.0, 2.0, -53.55)),
      (1.0, Array(1.0, 2.0, 25.89)),
      (1.0, Array(2.0, 2.0, 22.62)),
      (2.0, Array(3.0, 2.0, -5.63)),
      (2.0, Array(4.0, 2.0, 81.67)),
      (0.0, Array(0.0, 2.0, -72.87)),
      (1.0, Array(1.0, 2.0, 25.51)),
      (1.0, Array(2.0, 2.0, 43.14)),
      (2.0, Array(3.0, 2.0, 60.53)),
      (2.0, Array(4.0, 2.0, 88.94)),
      (0.0, Array(0.0, 2.0, 17.08)),
      (1.0, Array(1.0, 2.0, 69.48)),
      (1.0, Array(2.0, 2.0, -76.47)),
      (2.0, Array(3.0, 2.0, 90.90)),
      (2.0, Array(4.0, 2.0, -79.67))
    )
  }

  def labeledData6: Array[(Double, Array[Double])] = {
    Array(
      (0.0, Array(0.0, 0.0)),
      (0.0, Array(1.0, 0.0)),
      (1.0, Array(2.0, Double.NaN)),
      (1.0, Array(3.0, Double.NaN)),
      (2.0, Array(4.0, 0.0)),
      (2.0, Array(0.0, 1.0)),
      (2.0, Array(1.0, 1.0)),
      (3.0, Array(Double.NaN, 1.0)),
      (3.0, Array(Double.NaN, 1.0)),
      (0.0, Array(4.0, 1.0)),
      (2.0, Array(0.0, 1.0)),
      (2.0, Array(1.0, 1.0)),
      (3.0, Array(2.0, 1.0)),
      (3.0, Array(3.0, 1.0)),
      (0.0, Array(4.0, 1.0)),
      (0.0, Array(0.0, 2.0)),
      (1.0, Array(1.0, 2.0)),
      (1.0, Array(2.0, 2.0)),
      (2.0, Array(3.0, 2.0)),
      (2.0, Array(4.0, 2.0)),
      (0.0, Array(0.0, 2.0)),
      (1.0, Array(1.0, 2.0)),
      (1.0, Array(2.0, 2.0)),
      (2.0, Array(3.0, 2.0)),
      (2.0, Array(Double.NaN, 2.0)),
      (0.0, Array(0.0, 2.0)),
      (1.0, Array(1.0, 2.0)),
      (1.0, Array(2.0, 2.0)),
      (2.0, Array(3.0, 2.0)),
      (2.0, Array(4.0, 2.0))
    )
  }
}
