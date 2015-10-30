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

import java.io.Serializable
import java.util.Calendar

/**
 * An object to funnel progress reports to.
 */
trait ProgressNotifiee extends Serializable {
  def newProgressMessage(progress: String): Unit
  def newStatusMessage(status: String): Unit
  def newErrorMessage(error: String): Unit
}

/**
 * Simple console notifiee.
 * Prints messages to stdout.
 */
class ConsoleNotifiee extends ProgressNotifiee {
  def newProgressMessage(progress: String): Unit = {
    println(
      "[Progress] [" + Calendar.getInstance().getTime.toString + "] " + progress
    )
  }

  def newStatusMessage(status: String): Unit = {
    println(
      "[Status] [" + Calendar.getInstance().getTime.toString + "] " + status
    )
  }

  def newErrorMessage(error: String): Unit = {
    println(
      "[Error] [" + Calendar.getInstance().getTime.toString + "] " + error
    )
  }
}
