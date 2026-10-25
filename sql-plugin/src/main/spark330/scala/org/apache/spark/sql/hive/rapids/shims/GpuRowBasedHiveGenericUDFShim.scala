/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*** spark-rapids-shim-json-lines
{"spark": {"introduced": "330"}}
{"spark": "330db"}
{"spark": "352"}
{"spark": "353"}
{"spark": "354"}
{"spark": "355"}
{"spark": "356"}
{"spark": "357"}
{"spark": {"introduced": "358"}}
{"spark": {"introduced": "400"}}
{"spark": {"introduced": "402"}}
{"spark": {"introduced": "420"}}
spark-rapids-shim-json-lines ***/

package org.apache.spark.sql.hive.rapids.shims

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.SpecializedGetters
import org.apache.spark.sql.hive.DeferredObjectAdapter

object GpuRowBasedHiveGenericUDFShim {
  def setDeferredObject(
      o: DeferredObjectAdapter,
      childRowAccessors: Array[SpecializedGetters => Any],
      idx: Int,
      childrenRow: InternalRow): Unit = {
    o.set(() => childRowAccessors(idx)(childrenRow))
  }
}