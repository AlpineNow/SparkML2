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
 * Discretized features can be either unsigned Byte or Short. This trait is
 * responsible for handling the actual data types, freeing the aggregation logic
 * from handling different types.
 * @tparam T Type of the features. Either Byte or Short.
 */
trait DiscretizedFeatureHandler[@specialized(Byte, Short) T] extends Serializable {
  /**
   * Convert the given value of type T to an integer value.
   * @param value The value that we want to convert.
   * @return The integer value.
   */
  def convertToInt(value: T): Int

  /**
   * Convert the given integer value to the type.
   * @param value Integer value that we want to convert.
   * @return The converted value.
   */
  def convertToType(value: Int): T

  /**
   * Get the minimum value you can get for this type.
   * @return The minimum value for this type.
   */
  def getMinValue: Int

  /**
   * get the maximum value you can get for this type.
   * @return The maximum value for this type.
   */
  def getMaxValue: Int
}

/**
 * Handle unsigned byte features.
 */
class UnsignedByteHandler extends DiscretizedFeatureHandler[Byte] {
  /**
   * Convert the given value of unsigned Byte to an integer value.
   * @param value The value that we want to convert.
   * @return The integer value.
   */
  def convertToInt(value: Byte): Int = {
    value.toInt + 128
  }

  /**
   * Convert the given integer value to the type.
   * @param value Integer value that we want to convert.
   * @return The converted value.
   */
  def convertToType(value: Int): Byte = {
    (value - 128).toByte
  }

  /**
   * Get the minimum value you can get for this type.
   * @return The minimum value for this type.
   */
  def getMinValue: Int = 0

  /**
   * get the maximum value you can get for this type.
   * @return The maximum value for this type.
   */
  def getMaxValue: Int = 255
}

/**
 * Handle unsigned short features.
 */
class UnsignedShortHandler extends DiscretizedFeatureHandler[Short] {
  /**
   * Convert the given value of unsigned Short to an integer value.
   * @param value The value that we want to convert.
   * @return The integer value.
   */
  def convertToInt(value: Short): Int = {
    value.toInt + 32768
  }

  /**
   * Convert the given integer value to the type.
   * @param value Integer value that we want to convert.
   * @return The converted value.
   */
  def convertToType(value: Int): Short = {
    (value - 32768).toShort
  }

  /**
   * Get the minimum value you can get for this type.
   * @return The minimum value for this type.
   */
  def getMinValue: Int = 0

  /**
   * get the maximum value you can get for this type.
   * @return The maximum value for this type.
   */
  def getMaxValue: Int = 65535
}
