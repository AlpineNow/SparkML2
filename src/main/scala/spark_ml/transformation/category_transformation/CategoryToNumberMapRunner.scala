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

import spire.implicits._

import scopt.OptionParser
import org.apache.spark.{ SparkContext, SparkConf }
import org.apache.hadoop.fs.{ Path, FileSystem }
import java.io.{ ObjectOutputStream, ObjectInputStream }
import spark_ml.transformation.DistinctValueCounter

case class CategoryToNumberMapRunnerConfig(
  inputPath: String = null,
  outputPath: String = null,
  mapPath: String = null,
  readPreviousMap: Boolean = false,
  delimiter: String = ",",
  headerExists: Boolean = false,
  colIndices: Array[Int] = Array[Int]())

/**
 * This is a command line tool that either:
 * 1. Finds all the distinct values for selected columns and maps them to non-negative integer numbers.
 * 2. Load previous distinct value to non-negative integer mapping and applies them.
 */
object CategoryToNumberMapRunner {
  def main(args: Array[String]) {
    val defaultConfig = CategoryToNumberMapRunnerConfig()
    val parser = new OptionParser[CategoryToNumberMapRunnerConfig]("CategoryToNumberMapRunner") {
      head("CategoryToNumberMapRunner: Map strings in columns to numbers.")
      opt[String]("inputPath")
        .text("Path to delimited text file(s) (e.g. csv/tsv) to be used as an input.")
        .required()
        .action((x, c) => c.copy(inputPath = x))
      opt[String]("outputPath")
        .text("Output path (directory) where the converted structure will be stored.")
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
        .text("Delimiter string for the input data. The default is \",\"")
        .action((x, c) => c.copy(delimiter = x))
      opt[Boolean]("headerExists")
        .text("Whether a header exists in the input data. The default is false.")
        .action((x, c) => c.copy(headerExists = x))
      opt[String]("colIndices")
        .text("A comma separated indices for columns in the input that we want to convert.")
        .required()
        .action((x, c) => c.copy(colIndices = x.split(",").map(value => value.toInt)))
    }

    parser.parse(args, defaultConfig).map { config =>
      run(config)
    }.getOrElse {
      Thread.sleep(10000)
      sys.exit(1)
    }
  }

  def run(config: CategoryToNumberMapRunnerConfig): Unit = {
    val conf = new SparkConf().setAppName("CategoryToNumberMapRunnerConfig")
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

      // Print the current memory usage.
      println("Memory stats before performing distinct value counting.")
      println("Maximum memory : " + (Runtime.getRuntime.maxMemory().toDouble / 1024.0 / 1024.0) + " MB")
      println("Used memory : " + ((Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()).toDouble / 1024.0 / 1024.0) + " MB")
      println("Free memory : " + (Runtime.getRuntime.freeMemory().toDouble / 1024.0 / 1024.0) + " MB")

      val distinctValueMap: mutable.Map[String, mutable.Map[String, Int]] = if (config.readPreviousMap) {
        // Read the map that's already in the hadoop file system.
        val mapInputStream = hdfs.open(new Path(config.mapPath))
        val objectInput = new ObjectInputStream(mapInputStream)
        val m = objectInput.readObject().asInstanceOf[mutable.Map[String, mutable.Map[String, Int]]]
        objectInput.close()
        mapInputStream.close()
        m
      } else {
        // Otherwise, compute a new mapping.
        val m = DistinctValueCounter.mapDistinctValuesToIntegers(DistinctValueCounter.getDistinctValues(inputRDD, config.delimiter, config.headerExists, config.colIndices))
        val colNameToM = mutable.Map[String, mutable.Map[String, Int]]()
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

      // Print the current memory usage.
      println("Memory stats after performing distinct value counting.")
      println("Maximum memory : " + (Runtime.getRuntime.maxMemory().toDouble / 1024.0 / 1024.0) + " MB")
      println("Used memory : " + ((Runtime.getRuntime.totalMemory() - Runtime.getRuntime.freeMemory()).toDouble / 1024.0 / 1024.0) + " MB")
      println("Free memory : " + (Runtime.getRuntime.freeMemory().toDouble / 1024.0 / 1024.0) + " MB")

      val broadcastMap = sc.broadcast(distinctValueMap)

      // Now that we have the distinct value map, apply it.
      val convertedRDD = inputRDD.mapPartitionsWithIndex((partitionIdx: Int, lines: Iterator[String]) => {
        var headerWritten = false

        lines.map(line => {
          if (partitionIdx == 0 && headerExists && !headerWritten) {
            headerWritten = true
            line
          } else {
            val lineElems = line.split(delimiter, -1)
            var outputLine = ""
            cfor(0)(_ < header.length, _ + 1)(
              idx => {
                val colName = header(idx)
                val outputDelimiter = if (idx == 0) "" else delimiter

                // TODO: Handling a previously unseen categorical feature value is very ad-hoc.
                if (broadcastMap.value.contains(colName)) {
                  if (broadcastMap.value(colName).contains(lineElems(idx))) {
                    outputLine += outputDelimiter + broadcastMap.value(colName)(lineElems(idx)).toString
                  } else if (broadcastMap.value(colName).contains("")) { // Treating a new categorical feature value as an empty string.
                    outputLine += outputDelimiter + broadcastMap.value(colName)("").toString
                  } else { // Otherwise, the unknown feature value is treated as a new categorical value.
                    outputLine += outputDelimiter + broadcastMap.value(colName).size.toString
                  }
                } else {
                  outputLine += outputDelimiter + lineElems(idx)
                }
              }
            )

            outputLine
          }
        })
      })

      convertedRDD.saveAsTextFile(config.outputPath)
    } catch {
      case e: Exception => println("Exception:" + e.toString)
    } finally {
      Thread.sleep(10000)
      sc.stop()
    }
  }
}
