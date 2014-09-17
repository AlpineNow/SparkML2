package spark_ml.transformation.category_transformation

import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import scala.collection.mutable

import spire.implicits._

object CategoryMapper {
  /**
   * Apply the categorical feature mapping to the original data and get a transformed data set.
   * @param data Original data.
   * @param transformation The categorical feature mapping. If a mapping is not found, 0 is used.
   * @return Transformed RDD.
   */
  def mapCategories(
    data: RDD[String],
    transformation: Broadcast[mutable.Map[String, mutable.Map[Int, Int]]],
    delimiter: String,
    header: Array[String],
    headerExists: Boolean): RDD[String] = {
    data.mapPartitionsWithIndex((partitionIdx: Int, lines: Iterator[String]) => {
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
              if (transformation.value.contains(colName)) {
                // If the transformation is not found in the map, it becomes 0.
                outputLine += outputDelimiter + transformation.value(colName).getOrElse(lineElems(idx).toInt, 0).toString
              } else {
                outputLine += outputDelimiter + lineElems(idx)
              }
            }
          )

          outputLine
        }
      })
    })
  }
}

/**
 * Transformers based on this trait is used to 'coarsen' fine-grained categorical features with many unique values into
 * more coarse grained categorical features with smaller number of unique values.
 */
object CategoryCoarsener {
  // Collect the number of each unique categorical value for each categorical column.
  // Assume that the category values go from 0 to K - 1 (where K is the number of unique values for that column).
  def collectCatValueCounts(
    data: RDD[String],
    delimiter: String,
    headerExistsInRDD: Boolean,
    colIndices: Array[Int]): mutable.Map[Int, mutable.ArrayBuffer[Long]] = {
    data.mapPartitionsWithIndex((partitionIdx: Int, lines: Iterator[String]) => {
      val valueCountsPerCol = mutable.Map[Int, mutable.ArrayBuffer[Long]]()
      cfor(0)(_ < colIndices.length, _ + 1)(
        i => valueCountsPerCol.put(colIndices(i), new mutable.ArrayBuffer[Long]())
      )

      if (headerExistsInRDD && partitionIdx == 0) lines.drop(1)

      lines.foreach((line: String) => {
        val lineElems = line.split(delimiter, -1)
        cfor(0)(_ < colIndices.length, _ + 1)(
          i => {
            val colVal = lineElems(colIndices(i)).trim.toInt
            val arrayBuf = valueCountsPerCol(colIndices(i))
            if (colVal >= arrayBuf.length) {
              val appendCount = colVal - arrayBuf.length + 1
              arrayBuf ++= mutable.ArrayBuffer.fill[Long](appendCount)(0)
            }

            arrayBuf(colVal) += 1
          }
        )
      })

      Array[mutable.Map[Int, mutable.ArrayBuffer[Long]]](valueCountsPerCol).toIterator
    }).reduce((m1: mutable.Map[Int, mutable.ArrayBuffer[Long]], m2: mutable.Map[Int, mutable.ArrayBuffer[Long]]) => {
      m2.keys.foreach((idx: Int) => {
        val arrayBuf1 = m1(idx)
        val arrayBuf2 = m2(idx)
        cfor(0)(_ < arrayBuf2.length, _ + 1)(
          i => {
            if (i >= arrayBuf1.length) {
              val appendCount = i - arrayBuf1.length + 1
              arrayBuf1 ++= mutable.ArrayBuffer.fill[Long](appendCount)(0)
            }

            arrayBuf1(i) += arrayBuf2(i)
          }
        )
      })

      m1
    })
  }

  /**
   * Shrink the cardinality of each categorical feature to the max value.
   * The most common ones are preserved as a single category whereas less common ones will eventually be aggregated into a single categorical value.
   */
  def shrinkCardinalityToMostCommonOnes(catValCounts: mutable.Map[Int, mutable.ArrayBuffer[Long]], maxCardinality: Int): mutable.Map[Int, mutable.Map[Int, Int]] = {
    val transformation = mutable.Map[Int, mutable.Map[Int, Int]]()
    catValCounts.foreach(key_value => {
      val colIdx = key_value._1
      val counts = key_value._2

      val cardinality = counts.length
      val indices = (0 to cardinality - 1).toArray
      val sortedIndices = indices.sorted(Ordering.by[Int, Long](counts(_)))

      println("For feature " + colIdx + " : ")

      transformation.put(colIdx, mutable.Map[Int, Int]())
      val maxCatVal = math.min(maxCardinality - 1, sortedIndices.length - 1)
      cfor(sortedIndices.length - 1)(_ > (sortedIndices.length - maxCatVal - 1), _ - 1)(
        i => {
          val prevCatVal = sortedIndices(i)
          val newCatVal = i - (sortedIndices.length - 1) + (maxCatVal - 1)
          transformation(colIdx).put(prevCatVal, newCatVal)

          println("Categorical value '" + prevCatVal + "' has occurred " + counts(prevCatVal) + " times.")
        }
      )
    })

    transformation
  }
}
