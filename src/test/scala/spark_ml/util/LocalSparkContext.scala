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

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, Suite}

/**
 * Start a local spark context for unit testing.
 */
trait LocalSparkContext extends BeforeAndAfterAll { self: Suite =>
  @transient var sc: SparkContext = _

  override def beforeAll() {
    super.beforeAll()
    // http://stackoverflow.com/questions/27781187/how-to-stop-messages-displaying-on-spark-console
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    Thread.sleep(100L)
    val conf = new SparkConf()
      .setMaster("local[3]")
      .setAppName("test")
    sc = new SparkContext(conf)
  }

  override def afterAll() {
    if (sc != null) {
      sc.stop()
      sc = null
    }
    super.afterAll()
  }

  def numbersAreEqual(x: Double, y: Double, tol: Double = 1E-3): Boolean = {
    math.abs(x - y) / (math.abs(y) + 1e-15) < tol
  }
}
