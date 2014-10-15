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

import org.apache.spark.{ SparkContext, SparkConf }
import spire.implicits._
import scopt.OptionParser

case class FeatureHashingRunnerConfig(
  inputPath: String = null,
  outputPath: String = null,
  delimiter: String = ",",
  headerExists: Boolean = false,
  colIndices: Array[Int] = Array[Int](),
  hashSpaceSize: Int = 256,
  leaveEmptyString: Boolean = true)

object FeatureHashingRunner {
  def main(args: Array[String]) {
    val defaultConfig = FeatureHashingRunnerConfig()

    val parser = new OptionParser[FeatureHashingRunnerConfig]("FeatureHashingRunnerConfig") {
      head("FeatureHashingRunnerConfig: Hash categorical feature values to integers.")
      opt[String]("inputPath")
        .text("Path to delimited text file(s) (e.g. csv/tsv) to be used as an input.")
        .required()
        .action((x, c) => c.copy(inputPath = x))
      opt[String]("outputPath")
        .text("Output path (directory) where the converted files will be stored.")
        .required()
        .action((x, c) => c.copy(outputPath = x))
      opt[String]("delimiter")
        .text("Delimiter string for the input data. The default is \",\"")
        .action((x, c) => c.copy(delimiter = x))
      opt[Boolean]("headerExists")
        .text("Whether a header exists in the input data. The default is false.")
        .action((x, c) => c.copy(headerExists = x))
      opt[Boolean]("leaveEmptyString")
        .text("Whether to leave empty strings as they are or to treat them as categorical values. The default value is true.")
        .action((x, c) => c.copy(leaveEmptyString = x))
      opt[String]("colIndices")
        .text("A comma separated indices for categorical columns in the input.")
        .required()
        .action((x, c) => c.copy(colIndices = x.split(",").map(value => value.toInt)))
      opt[Int]("hashSpaceSize")
        .text("The maximum possible hash value + 1. The default value is 256.")
        .action((x, c) => c.copy(hashSpaceSize = x))
    }

    parser.parse(args, defaultConfig).map {
      config => run(config)
    }.getOrElse {
      Thread.sleep(10000)
      sys.exit(1)
    }
  }

  def run(config: FeatureHashingRunnerConfig): Unit = {
    val conf = new SparkConf().setAppName("FeatureHashingRunner")
    val sc = new SparkContext(conf)

    try {
      val inputRDD = sc.textFile(config.inputPath)
      val firstLine = inputRDD.first()
      val firstLineSplit = firstLine.split(config.delimiter, -1)
      val numColumns = firstLineSplit.length
      val headerExists = config.headerExists
      val delimiter = config.delimiter
      val header = if (config.headerExists) {
        firstLineSplit
      } else {
        // Still, we want to figure out the number of columns and assign arbitrary column names.
        val colNames = new Array[String](numColumns)
        cfor(0)(_ < numColumns, _ + 1)(
          columnId => colNames(columnId) = "Col" + columnId
        )

        colNames
      }

      val transformedRDD = FeatureHashing.hashCategoricalFeatures(
        inputRDD,
        config.colIndices.map(header(_)).toSet,
        SimpleHasher(config.hashSpaceSize).hashString,
        config.leaveEmptyString,
        delimiter,
        header,
        headerExists)

      transformedRDD.saveAsTextFile(config.outputPath)
    } catch {
      case e: Exception => println("Exception:" + e.toString)
    } finally {
      Thread.sleep(10000)
      sc.stop()
    }
  }
}
