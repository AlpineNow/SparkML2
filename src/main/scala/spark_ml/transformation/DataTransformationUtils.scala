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

package spark_ml.transformation

import com.databricks.spark.csv.CsvParser
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, SQLContext}
import spark_ml.discretization.CardinalityOverLimitException
import spark_ml.gradient_boosting.DelimitedTextFileConfig
import spark_ml.util.ProgressNotifiee

/**
  * The storage format for the input dataset.
  */
object StorageFormat extends Enumeration {
  type StorageFormat = Value
  val DelimitedText = Value(0)
  val Parquet = Value(1)
  val Avro = Value(2)
}

/**
 * A class that encodes the transformation to a numeric value for a column.
 * @param distinctValToInt If this is defined, then the column value would be
 *                         transformed to a mapped integer.
 * @param maxCardinality If this is defined, then the column value would be
 *                       transformed into an integer value in
 *                       [0, maxCardinality) through hashing.
 * @return a transformed numeric value for a column value. If neither
 *         distinctValToInt nor maxCardinality is defined, then the value
 *         would be assumed to be a numeric value already and passed through.
 */
case class ColumnTransformer(
  // We are using java.lang.Integer so that this can be easily serialized by
  // Gson.
  distinctValToInt: Option[Map[String, java.lang.Integer]],
  maxCardinality: Option[java.lang.Integer]) {
  def transform(value: String): Double = {
    if (distinctValToInt.isDefined) {
      val nonNullValue =
        if (value == null) {
          ""
        } else {
          value
        }
      distinctValToInt.get(nonNullValue).toDouble
    } else if (maxCardinality.isDefined) {
      val nonNullValue =
        if (value == null) {
          ""
        } else {
          value
        }
      DataTransformationUtils.getSimpleHashedValue(nonNullValue, maxCardinality.get)
    } else {
      if (value == null) {
        Double.NaN
      } else {
        value.toDouble
      }
    }
  }
}

object DataTransformationUtils {
  /**
   * Load the given data file path and get a labeled dataset RDD with a pair
   * (label, feature-array).
   * @param filePath Data file path.
   * @param sc Spark context.
   * @param dataSchema Optional input schema for the dataset, if known in
   *                    advance.
   * @param storageFormat The data file storage format.
   * @param repartitionSize Optional repartition size. If this is not None, then
   *                        the RDD will get repartitioned.
   * @param delimitedTextFileConfig Delimited text file configuration.
   * @param maxCatCardinality Maximum categorical column cardinality that's
   *                          allowed. If the categorical cardinality exceeds
   *                          this, then feature hashing might get performed,
   *                          depending on the following option value.
   * @param useFeatureHashingForCat Whether to convert large-cardinality
   *                                categorical features through feature hashing.
   * @param catColIndices Categorical column indices.
   * @param colsToIgnoreIndices Columns that should be ignored (won't be used
   *                            for either label or features).
   * @param labelColIndex Label column index.
   * @param checkpointDir Checkpoint directory (optional).
   * @param verbose Whether to be verbose during the load and transform phase.
   * @return An RDD of pairs (label, feature-array), label-column-transformer,
   *         feature-column-transformers, label-name, feature-names, and
   *         categorical feature indices (different from the categorical column
   *         indices).
   */
  def loadDataFileAndGetLabelFeatureRdd(
    filePath: String,
    sc: SparkContext,
    dataSchema: Option[StructType],
    storageFormat: StorageFormat.StorageFormat,
    repartitionSize: Option[Int],
    delimitedTextFileConfig: DelimitedTextFileConfig,
    maxCatCardinality: Int,
    useFeatureHashingForCat: Boolean,
    catColIndices: Set[Int],
    colsToIgnoreIndices: Set[Int],
    labelColIndex: Int,
    checkpointDir: Option[String],
    notifiee: ProgressNotifiee,
    verbose: Boolean):
    (
      RDD[(Double, Array[Double])],
      ColumnTransformer,
      Array[ColumnTransformer],
      String,
      Seq[String],
      Set[Int]
    ) = {
    val sortedCatColIndices = catColIndices.toSeq.sorted
    val sqlContext = new SQLContext(sc)

    notifiee.newStatusMessage("Loading the inputPath " + filePath)

    // First, load the dataset as a dataframe.
    val loadedDf = storageFormat match {
      case StorageFormat.DelimitedText =>
        val parser =
          new CsvParser()
            .withUseHeader(delimitedTextFileConfig.headerExists)
            .withDelimiter(delimitedTextFileConfig.delimiter(0))
            .withQuoteChar(delimitedTextFileConfig.quoteStr(0))
            .withEscape(delimitedTextFileConfig.escapeStr(0))

        (
          dataSchema match {
            case Some(structType) => parser.withSchema(structType)
            case None => parser.withInferSchema(true)
          }
          ).csvFile(sqlContext, filePath)

      case StorageFormat.Avro => sqlContext.load(filePath, "com.databricks.spark.avro")
      case StorageFormat.Parquet => sqlContext.load(filePath)
    }

    val dataFrame = repartitionSize match {
      case Some(numParts) =>
        if (checkpointDir.isDefined) {
          val repartitioned = loadedDf.repartition(numParts)
          repartitioned.rdd.sparkContext.setCheckpointDir(checkpointDir.get)
          repartitioned.rdd.checkpoint()
          repartitioned
        } else {
          loadedDf
        }
      case None => loadedDf
    }

    if (verbose) {
      notifiee.newStatusMessage("The dataset " + filePath + " was successfully loaded.")
      notifiee.newStatusMessage("There are " + dataFrame.count() + " rows.")
      notifiee.newStatusMessage("There are " + dataFrame.columns.length + " columns.")
    }

    notifiee.newStatusMessage("Successfully loaded the dataset.")

    // Get the columns.
    val columns = dataFrame.columns
    val numCols = columns.length

    notifiee.newStatusMessage("Finding distinct values for categorical features.")

    // Get distinct categorical values.
    val catDistinctVals = DistinctValueCounter.getDistinctValues(
      dataFrame,
      catColIndices,
      maxCatCardinality
    )

    notifiee.newStatusMessage("Found distinct values for categorical features.")

    // Compute the cardinalities of each categorical column.
    val catColCardinalities =
      sortedCatColIndices.map(colIdx => colIdx -> catDistinctVals(colIdx).size).toMap

    // See if any categorical column has a cardinality that larger than
    // the maximum. If so, see if feature hashing is allowed. Otherwise,
    // we'll simply throw an exception.
    catColCardinalities.foreach {
      case (idx, cardinality) =>
        if (cardinality > maxCatCardinality && !useFeatureHashingForCat) {
          throw CardinalityOverLimitException(
            "The categorical column " + columns(idx) + " has a cardinality of " +
              cardinality.toString + " that is higher than the limit " + maxCatCardinality.toString +
              ". Use the feature hashing option if you want to use this as a feature."
          )
        }
    }

    // Map unique strings to numbers.
    val catDistinctValToInt = DistinctValueCounter.mapDistinctValuesToIntegers(
      catDistinctVals,
      useEmptyString = true
    )

    if (verbose) {
      notifiee.newStatusMessage("The categorical columns are : ")
      notifiee.newStatusMessage(catColIndices.map(columns(_)).mkString(" "))
      notifiee.newStatusMessage("The maximum allowed cardinality is " + maxCatCardinality)
      catDistinctValToInt.foreach {
        case (key, distinctValCounts) =>
          notifiee.newStatusMessage(columns(key) + " has " + distinctValCounts.size + " unique values : ")
          if (distinctValCounts.size > maxCatCardinality) {
            notifiee.newStatusMessage(columns(key) + "'s distinct value count exceed the limit.")
          } else {
            notifiee.newStatusMessage(
              distinctValCounts.map {
                case (distinctVal, mappedInt) => distinctVal + "->" + mappedInt.toString
              }.mkString(",")
            )
          }
      }
    }

    notifiee.newStatusMessage("Transforming the dataset into an RDD of label, feature-vector pairs.")

    // Now convert the data set into a form usable by the algorithm.
    // It'll be a pair of a label and a feature vector.
    val (labelFeatureRdd, (labelTransformer, featureTransformers)) =
      DataTransformationUtils.transformDataFrameToLabelFeatureRdd(
        dataFrame,
        labelColIndex,
        catDistinctValToInt,
        colsToIgnoreIndices,
        maxCatCardinality,
        useFeatureHashingForCat
      )

    // Map categorical column indices to categorical feature indices.
    // Feature indices exclude the label column and the columns to ignore.
    val (featureNames, catFeatureIndices, _) =
      (0 to (numCols - 1)).foldLeft((Seq[String](), Set[Int](), 0)) {
        case ((featNames, catFeatIndices, curFeatIdx), curColIdx) =>
          curColIdx match {
            case _ if labelColIndex == curColIdx || colsToIgnoreIndices.contains(curColIdx) => (featNames, catFeatIndices, curFeatIdx)
            case _ if catColIndices.contains(curColIdx) => (featNames ++ Seq(columns(curColIdx)), catFeatIndices + curFeatIdx, curFeatIdx + 1)
            case _ => (featNames ++ Seq(columns(curColIdx)), catFeatIndices, curFeatIdx + 1)
          }
      }

    if (verbose) {
      // Transformed features.
      val labelFeatureRddSample = labelFeatureRdd.take(50)

      notifiee.newStatusMessage("Transformed column names are :")
      notifiee.newStatusMessage(columns(labelColIndex) + ", (" + featureNames.mkString(",") + ")")
      notifiee.newStatusMessage("Categorical features are :")
      notifiee.newStatusMessage(
        featureNames.zipWithIndex.filter {
          case (featName, featIdx) => catFeatureIndices.contains(featIdx)
        }.mkString(",")
      )
      notifiee.newStatusMessage("Transformed label feature samples are :")
      labelFeatureRddSample.foreach {
        case (sampleLabel, sampleFeatures) =>
          notifiee.newStatusMessage(sampleLabel.toString + ", (" + sampleFeatures.mkString(",") + ")")
      }

      val getColumnTransformerDesc =
        (columnTransformer: ColumnTransformer) => {
          if (columnTransformer.distinctValToInt.isDefined) {
            "CatToIntMapper"
          } else if (columnTransformer.maxCardinality.isDefined) {
            "CatHasher"
          } else {
            "NumericPassThrough"
          }
        }

      notifiee.newStatusMessage("ColumnTransfomer descriptions :")
      notifiee.newStatusMessage(
        getColumnTransformerDesc(labelTransformer) + ", (" +
          featureTransformers.map(getColumnTransformerDesc(_)).mkString(",") + ")"
      )
    }

    (
      labelFeatureRdd,
      labelTransformer,
      featureTransformers,
      columns(labelColIndex),
      featureNames,
      catFeatureIndices
    )
  }

  /**
   * Convert the given data frame into an RDD of label, feature vector pairs.
   * All label/feature values are also converted into Double.
   * @param dataFrame Spark Data Frame that we want to convert.
   * @param labelColIndex Label column index.
   * @param catDistinctValToInt Categorical column distinct value to int maps.
   *                            This is used to map distinct string values to
   *                            numeric values (doubles).
   * @param colsToIgnoreIndices Indices of columns in the data frame to be
   *                            ignored (not used as features or label).
   * @param maxCatCardinality Maximum categorical cardinality we allow. If the
   *                          cardinality goes over this, feature hashing might
   *                          be used (or will simply throw an exception).
   * @param useFeatureHashing Whether feature hashing should be used on
   *                          categorical columns whose unique value counts
   *                          exceed the maximum cardinality.
   * @return An RDD of label/feature-vector pairs and column transformer definitions.
   */
  def transformDataFrameToLabelFeatureRdd(
    dataFrame: DataFrame,
    labelColIndex: Int,
    catDistinctValToInt: Map[Int, Map[String, java.lang.Integer]],
    colsToIgnoreIndices: Set[Int],
    maxCatCardinality: Int,
    useFeatureHashing: Boolean): (RDD[(Double, Array[Double])], (ColumnTransformer, Array[ColumnTransformer])) = {
    val transformedRDD = dataFrame.map(row => {
      val labelValue = row.get(labelColIndex)

      Tuple2(
        if (catDistinctValToInt.contains(labelColIndex)) {
          val nonNullLabelValue =
            if (labelValue == null) {
              ""
            } else {
              labelValue.toString
            }
          mapCategoricalValueToNumericValue(
            labelColIndex,
            nonNullLabelValue,
            catDistinctValToInt(labelColIndex),
            maxCatCardinality,
            useFeatureHashing
          )
        } else {
          if (labelValue == null) {
            Double.NaN
          } else {
            labelValue.toString.toDouble
          }
        },
        row.toSeq.zipWithIndex.flatMap {
          case (colVal, idx) =>
            if (colsToIgnoreIndices.contains(idx) || (labelColIndex == idx)) {
              Array[Double]().iterator
            } else {
              if (catDistinctValToInt.contains(idx)) {
                val nonNullColVal =
                  if (colVal == null) {
                    ""
                  } else {
                    colVal.toString
                  }
                Array(
                  mapCategoricalValueToNumericValue(
                    idx,
                    nonNullColVal,
                    catDistinctValToInt(idx),
                    maxCatCardinality,
                    useFeatureHashing
                  )
                ).iterator
              } else {
                val nonNullColVal =
                  if (colVal == null) {
                    Double.NaN
                  } else {
                    colVal.toString.toDouble
                  }
                Array(nonNullColVal).iterator
              }
            }
        }.toArray
      )
    })

    val numCols = dataFrame.columns.length
    val colTransformers = Tuple2(
      if (catDistinctValToInt.contains(labelColIndex)) {
        if (catDistinctValToInt(labelColIndex).size <= maxCatCardinality) {
          ColumnTransformer(Some(catDistinctValToInt(labelColIndex)), None)
        } else {
          ColumnTransformer(None, Some(maxCatCardinality))
        }
      } else {
        ColumnTransformer(None, None)
      },
      (0 to (numCols - 1)).flatMap {
        case (colIdx) =>
          if (colsToIgnoreIndices.contains(colIdx) || (labelColIndex == colIdx)) {
            Array[ColumnTransformer]().iterator
          } else {
            if (catDistinctValToInt.contains(colIdx)) {
              if (catDistinctValToInt(colIdx).size <= maxCatCardinality) {
                Array(ColumnTransformer(Some(catDistinctValToInt(colIdx)), None)).iterator
              } else {
                Array(ColumnTransformer(None, Some(maxCatCardinality))).iterator
              }
            } else {
              Array(ColumnTransformer(None, None)).iterator
            }
          }
      }.toArray
    )

    (transformedRDD, colTransformers)
  }

  /**
   * Map a categorical value (string) to a numeric value (double).
   * @param categoryId Equal to the column index.
   * @param catValue Categorical value that we want to map to a numeric value.
   * @param catValueToIntMap A map from categorical value to integers. If the
   *                         cardinality of the category is less than or equal
   *                         to maxCardinality, this is used.
   * @param maxCardinality The maximum allowed cardinality.
   * @param useFeatureHashing Whether feature hashing should be performed if the
   *                          cardinality of the category exceeds the maximum
   *                          cardinality.
   * @return A mapped double value.
   */
  def mapCategoricalValueToNumericValue(
    categoryId: Int,
    catValue: String,
    catValueToIntMap: Map[String, java.lang.Integer],
    maxCardinality: Int,
    useFeatureHashing: Boolean): Double = {
    if (catValueToIntMap.size <= maxCardinality) {
      catValueToIntMap(catValue).toDouble
    } else {
      if (useFeatureHashing) {
        getSimpleHashedValue(catValue, maxCardinality)
      } else {
        throw new CardinalityOverLimitException(
          "The categorical column with the index " + categoryId +
            " has a cardinality that exceeds the limit " + maxCardinality)
      }
    }
  }

  /**
   * Compute a simple hash of a categorical value.
   * @param catValue Categorical value that we want to compute a simple numeric
   *                 hash for. The hash value would be an integer between 0 and
   *                 (maxCardinality - 1).
   * @param maxCardinality The value of the hash is limited within
   *                       [0, maxCardinality).
   * @return A hashed value of double.
   */
  def getSimpleHashedValue(catValue: String, maxCardinality: Int): Double = {
    ((catValue.hashCode.toLong - Int.MinValue.toLong) % maxCardinality.toLong).toDouble
  }
}
