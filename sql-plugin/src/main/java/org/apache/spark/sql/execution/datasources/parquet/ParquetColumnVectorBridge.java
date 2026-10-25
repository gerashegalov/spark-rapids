/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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

package org.apache.spark.sql.execution.datasources.parquet;

import java.util.List;

import org.apache.spark.sql.execution.vectorized.WritableColumnVector;

/** Root-loaded accessors for Spark's package-private ParquetColumnVector. */
public final class ParquetColumnVectorBridge {
  private ParquetColumnVectorBridge() {
  }

  private static ParquetColumnVector asVector(Object vector) {
    return (ParquetColumnVector) vector;
  }

  public static ParquetColumn getColumn(Object vector) {
    return asVector(vector).getColumn();
  }

  @SuppressWarnings("unchecked")
  public static List<Object> getChildren(Object vector) {
    return (List<Object>) (List<?>) asVector(vector).getChildren();
  }

  @SuppressWarnings("unchecked")
  public static List<Object> getLeaves(Object vector) {
    return (List<Object>) (List<?>) asVector(vector).getLeaves();
  }

  public static void reset(Object vector) {
    asVector(vector).reset();
  }

  public static VectorizedColumnReader getColumnReader(Object vector) {
    return asVector(vector).getColumnReader();
  }

  public static void setColumnReader(Object vector, VectorizedColumnReader reader) {
    asVector(vector).setColumnReader(reader);
  }

  public static WritableColumnVector getValueVector(Object vector) {
    return asVector(vector).getValueVector();
  }

  public static WritableColumnVector getRepetitionLevelVector(Object vector) {
    return asVector(vector).getRepetitionLevelVector();
  }

  public static WritableColumnVector getDefinitionLevelVector(Object vector) {
    return asVector(vector).getDefinitionLevelVector();
  }

  public static void assemble(Object vector) {
    asVector(vector).assemble();
  }
}
