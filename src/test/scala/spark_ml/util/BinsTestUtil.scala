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

import org.scalatest.Assertions._
import spark_ml.discretization.NumericBins

object BinsTestUtil {
  def validateNumericalBins(
    bins: NumericBins,
    boundaries: Array[(Double, Double)],
    missingBinId: Option[Int]): Unit = {
    assert(bins.getCardinality === (boundaries.length + (if (missingBinId.isDefined) 1 else 0)))
    assert(bins.missingValueBinIdx === missingBinId)
    bins.bins.zip(boundaries).foreach {
      case (numericBin, (l, r)) =>
        assert(numericBin.lower === l)
        assert(numericBin.upper === r)
    }
  }
}
