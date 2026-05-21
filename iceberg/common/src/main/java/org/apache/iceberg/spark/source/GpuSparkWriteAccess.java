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

package org.apache.iceberg.spark.source;

import org.apache.iceberg.DataFile;
import org.apache.iceberg.io.DeleteWriteResult;
import org.apache.iceberg.io.WriteResult;
import org.apache.spark.sql.connector.write.Write;
import org.apache.spark.sql.connector.write.WriterCommitMessage;

/**
 * Package-local access to Iceberg Spark write classes.
 *
 * <p>Iceberg keeps SparkWrite and SparkPositionDeltaWrite package-private. Resolve those
 * classes from a conventional-root helper so runtime package access is checked in the same
 * class loader as Iceberg itself.
 */
public final class GpuSparkWriteAccess {
  private GpuSparkWriteAccess() {
  }

  public static boolean supports(Class<? extends Write> cpuClass) {
    return SparkWrite.class.isAssignableFrom(cpuClass)
        || SparkPositionDeltaWrite.class.isAssignableFrom(cpuClass);
  }

  public static String sparkWriteClassName() {
    return SparkWrite.class.getName();
  }

  public static WriterCommitMessage taskCommit(DataFile[] files) {
    SparkWrite.TaskCommit commit = new SparkWrite.TaskCommit(files);
    commit.reportOutputMetrics();
    return commit;
  }

  public static DataFile[] taskCommitFiles(WriterCommitMessage message) {
    return ((SparkWrite.TaskCommit) message).files();
  }

  public static WriterCommitMessage deltaTaskCommit(WriteResult result) {
    return new SparkPositionDeltaWrite.DeltaTaskCommit(result);
  }

  public static WriterCommitMessage deltaTaskCommit(DeleteWriteResult result) {
    return new SparkPositionDeltaWrite.DeltaTaskCommit(result);
  }
}
