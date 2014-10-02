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
import scala.util.Random

/**
 * Poisson sampler.
 * @param lambda The average count of the Poisson distribution.
 * @param seed The random number generator seed.
 */
case class Poisson(lambda: Double, seed: Int) {
  private val rng = new Random(seed)
  private val tolerance: Double = 0.00001

  private var pdf: Array[Double] = _
  private var cdf: Array[Double] = _

  {
    val pdfBuilder = new mutable.ArrayBuilder.ofDouble
    val cdfBuilder = new mutable.ArrayBuilder.ofDouble

    val expPart: Double = math.exp(-lambda)
    var curCDF: Double = expPart
    var curPDF: Double = expPart

    cdfBuilder += curCDF
    pdfBuilder += curPDF

    var i: Double = 1.0
    while (curCDF < (1.0 - tolerance)) {
      curPDF *= lambda / i
      curCDF += curPDF

      cdfBuilder += curCDF
      pdfBuilder += curPDF

      i += 1.0
    }

    pdf = pdfBuilder.result()
    cdf = cdfBuilder.result()
  }

  /**
   * Sample a poisson distributed value.
   * @return A sampled integer value.
   */
  def sample(): Int = {
    val rnd = rng.nextDouble()
    for (i <- 0 to cdf.length - 1) {
      if (rnd <= cdf(i)) {
        return i
      }
    }

    cdf.length - 1
  }
}
