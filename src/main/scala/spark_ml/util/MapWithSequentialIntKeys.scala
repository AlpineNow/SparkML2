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

import scala.reflect.ClassTag

import spire.implicits._

/**
 * Exception to be thrown in case of unexpected behavior of the following map
 * class.
 * @param msg Exception msg.
 */
case class UnexpectedKeyException(msg: String) extends Exception(msg)

/**
 * A map with integer keys that are guaranteed to be incrementing one by one.
 * A next insert is expected to use a key that's equal to 1 + prev put key.
 * A next remove is expected to use a key that's equal to the beginning key.
 * @param initCapacity Initial capacity.
 */
class MapWithSequentialIntKeys[@specialized(Int) T: ClassTag](initCapacity: Int)
  extends Serializable {
  private var values = new Array[T](initCapacity)
  private var capacity = initCapacity
  private var size = 0
  private var putCursor = 0
  private var expectedPutKey = 0
  private var firstGetPos = 0
  private var firstGetKey = 0

  /**
   * Put the next key value. The key is expected to be 1 + the previous one
   * unless it's the very first key.
   * @param key The key value.
   * @param value The value corresponding to the key.
   */
  def put(key: Int, value: T): Unit = {
    // Make sure that the key is as expected.
    if (size > 0 && key != expectedPutKey) {
      throw UnexpectedKeyException(
        "The put key " + key +
          " is different from the expected key " + expectedPutKey
      )
    }

    // If the array is full, we need to get a new array.
    if (size >= capacity) {
      capacity *= 2
      val newValues = new Array[T](capacity)
      cfor(0)(_ < size, _ + 1)(
        i => {
          newValues(i) = values(firstGetPos)
          firstGetPos += 1
          if (firstGetPos >= size) {
            firstGetPos = 0
          }
        }
      )

      values = newValues
      putCursor = size
      firstGetPos = 0
    }

    values(putCursor) = value
    putCursor += 1
    if (putCursor >= capacity) {
      putCursor = 0
    }

    if (size == 0) {
      firstGetKey = key
    }

    size += 1
    expectedPutKey = key + 1
  }

  /**
   * Get the value corresponding to the key.
   * @param key The integer key value.
   * @return The corresponding value.
   */
  def get(key: Int): T = {
    val getOffset = key - firstGetKey

    // Make sure that it's within the expected range.
    if (getOffset >= size || getOffset < 0) {
      throw UnexpectedKeyException(
        "The get key " + key +
          " is not within [" + firstGetKey + ", " + (firstGetKey + size - 1) + "]"
      )
    }

    var idx = firstGetPos + getOffset
    if (idx >= capacity) {
      idx -= capacity
    }

    values(idx)
  }

  /**
   * Remove a key.
   * @param key The key we want to remove.
   */
  def remove(key: Int): Unit = {
    // We expect the key to firstGetKey.
    // Otherwise, this is not being used as expected.
    if (key != firstGetKey) {
      throw UnexpectedKeyException(
        "The remove key " + key +
          " is different from the expected key " + firstGetKey
      )
    }

    size -= 1
    firstGetKey += 1
    firstGetPos += 1
    if (firstGetPos >= capacity) {
      firstGetPos = 0
    }
  }

  /**
   * Get key range in the object.
   * @return A pair of (startKey, endKey).
   */
  def getKeyRange: (Int, Int) = {
    (firstGetKey, firstGetKey + size - 1)
  }

  /**
   * Whether this map contains the key.
   * @param key The integer key that we want to check for.
   * @return true if the key is contained. false otherwise.
   */
  def contains(key: Int): Boolean = {
    val (startKey, endKey) = getKeyRange
    (key >= startKey) && (key <= endKey)
  }
}
