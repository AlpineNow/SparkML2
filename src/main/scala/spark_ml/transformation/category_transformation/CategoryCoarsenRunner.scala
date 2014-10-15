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

import scala.collection.mutable

import org.apache.spark.{ SparkContext, SparkConf }
import org.apache.hadoop.fs.{ Path, FileSystem }
import spire.implicits._
import scopt.OptionParser
import java.io.{ ObjectOutputStream, ObjectInputStream }

case class CategoryCoarsenRunnerConfig(
  inputPath: String = null,
  outputPath: String = null,
  mapPath: String = null,
  readPreviousMap: Boolean = false,
  delimiter: String = "\t",
  headerExists: Boolean = false,
  colIndices: Array[Int] = Array[Int](),
  maxCardinality: Int = 256)

// TODO: Does not work with categorical features that have empty strings.
object CategoryCoarsenRunner {
  def main(args: Array[String]) {
    val defaultConfig = CategoryCoarsenRunnerConfig()

    val parser = new OptionParser[CategoryCoarsenRunnerConfig]("CategoryCoarsenRunner") {
      head("CategoryCoarsenRunner: Reduce cardinality of categorical columns.")
      opt[String]("inputPath")
        .text("Path to delimited text file(s) (e.g. csv/tsv) to be used as an input.")
        .required()
        .action((x, c) => c.copy(inputPath = x))
      opt[String]("outputPath")
        .text("Output path (directory) where the converted files will be stored.")
        .required()
        .action((x, c) => c.copy(outputPath = x))
      opt[String]("mapPath")
        .text("Map path (directory) where the map structure exists or will be stored.")
        .required()
        .action((x, c) => c.copy(mapPath = x))
      opt[Boolean]("readPreviousMap")
        .text("Whether we want to read a previous map and apply it.")
        .action((x, c) => c.copy(readPreviousMap = x))
      opt[String]("delimiter")
        .text("Delimiter string for the input data. The default is \"\\t\"")
        .action((x, c) => c.copy(delimiter = x))
      opt[Boolean]("headerExists")
        .text("Whether a header exists in the input data. The default is false.")
        .action((x, c) => c.copy(headerExists = x))
      opt[String]("colIndices")
        .text("A comma separated indices for categorical columns in the input.")
        .required()
        .action((x, c) => c.copy(colIndices = x.split(",").map(value => value.toInt)))
      opt[Int]("maxCardinality")
        .text("Maximum cardinality allowed for categorical columns. If the cardinality is larger, it will be reduced to be smaller than or equal to this number using some heuristic.")
        .action((x, c) => c.copy(maxCardinality = x))
    }

    parser.parse(args, defaultConfig).map { config =>
      run(config)
    }.getOrElse {
      Thread.sleep(10000)
      sys.exit(1)
    }
  }

  def run(config: CategoryCoarsenRunnerConfig): Unit = {
    val conf = new SparkConf().setAppName("CategoryCoarsenRunner")
    val sc = new SparkContext(conf)
    val hdfs = FileSystem.get(sc.hadoopConfiguration)

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

      val catValMap = if (config.readPreviousMap) {
        val mapInputStream = hdfs.open(new Path(config.mapPath))
        val objectInput = new ObjectInputStream(mapInputStream)
        val m = objectInput.readObject().asInstanceOf[mutable.Map[String, mutable.Map[Int, Int]]]
        objectInput.close()
        mapInputStream.close()
        m
      } else {
        // Print the current memory usage.
        println("Memory usage before performing category value counting.")
        println("Maximum memory : " + (Runtime.getRuntime.maxMemory().toDouble / 1024.0 / 1024.0) + " MB")
        println("Used memory : " + ((Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()).toDouble / 1024.0 / 1024.0) + " MB")
        println("Free memory : " + (Runtime.getRuntime.freeMemory().toDouble / 1024.0 / 1024.0) + " MB")

        val catValCounts = CategoryCoarsener.collectCatValueCounts(
          inputRDD,
          config.delimiter,
          config.headerExists,
          config.colIndices)

        // Print the current memory usage.
        println("Memory usage after performing category value counting.")
        println("Maximum memory : " + (Runtime.getRuntime.maxMemory().toDouble / 1024.0 / 1024.0) + " MB")
        println("Used memory : " + ((Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()).toDouble / 1024.0 / 1024.0) + " MB")
        println("Free memory : " + (Runtime.getRuntime.freeMemory().toDouble / 1024.0 / 1024.0) + " MB")

        val m = CategoryCoarsener.shrinkCardinalityToMostCommonOnes(catValCounts, config.maxCardinality)
        val colNameToM = mutable.Map[String, mutable.Map[Int, Int]]()
        cfor(0)(_ < config.colIndices.length, _ + 1)(
          i => {
            val colIdx = config.colIndices(i)
            colNameToM.put(header(colIdx), m(colIdx))
          }
        )

        val mapOutputStream = hdfs.create(new Path(config.mapPath))
        val objectOutput = new ObjectOutputStream(mapOutputStream)
        objectOutput.writeObject(colNameToM)
        objectOutput.flush()
        objectOutput.close()
        mapOutputStream.close()

        colNameToM
      }

      val transformedRDD = CategoryMapper.mapCategories(
        inputRDD,
        sc.broadcast(catValMap),
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
