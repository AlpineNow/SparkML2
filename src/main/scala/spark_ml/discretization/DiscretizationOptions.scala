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

package spark_ml.discretization

/**
 * Discretization options.
 * @param discType Discretization type (Equal-Frequency/Width).
 * @param maxNumericBins Maximum number of numerical bins.
 * @param maxCatCardinality Maximum number of categorical bins.
 * @param useFeatureHashingOnCat Whether we should perform feature hashing on
 *                               categorical features whose cardinality goes
 *                               over 'maxCatCardinality'.
 * @param maxSampleSizeForDisc Maximum number of sample rows for certain numeric
 *                             discretizations. E.g., equal frequency
 *                             discretization uses a sample, instead of the
 *                             entire data.
 */
case class DiscretizationOptions(
  discType: DiscType.DiscType,
  maxNumericBins: Int,
  maxCatCardinality: Int,
  useFeatureHashingOnCat: Boolean,
  maxSampleSizeForDisc: Int) {
  override def toString: String = {
    "=========================" + "\n" +
      "Discretization Options" + "\n" +
      "=========================" + "\n" +
      "discType                   : " + discType.toString + "\n" +
      "maxNumericBins             : " + maxNumericBins + "\n" +
      "maxCatCardinality          : " + maxCatCardinality + "\n" +
      "useFeatureHashingOnCat     : " + useFeatureHashingOnCat.toString + "\n" +
      "maxSampleSizeForDisc       : " + maxSampleSizeForDisc.toString + "\n"
  }
}
