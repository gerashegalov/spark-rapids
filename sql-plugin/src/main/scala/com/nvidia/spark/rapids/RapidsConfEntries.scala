/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.
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
package com.nvidia.spark.rapids

import org.apache.spark.network.util.ByteUnit

private[rapids] trait RapidsConfEntries extends RapidsConfShuffleEntries {

  // USER FACING DEBUG CONFIGS

  val SHUFFLE_COMPRESSION_MAX_BATCH_MEMORY =
    conf("spark.rapids.shuffle.compression.maxBatchMemory")
      .internal()
      .bytesConf(ByteUnit.BYTE)
      .createWithDefault(1024 * 1024 * 1024)

  val EXPLAIN = conf("spark.rapids.sql.explain")
    .doc("Explain why some parts of a query were not placed on a GPU or not. Possible " +
      "values are ALL: print everything, NONE: print nothing, NOT_ON_GPU: print only parts of " +
      "a query that did not go on the GPU")
    .commonlyUsed()
    .stringConf
    .createWithDefault("NOT_ON_GPU")

  val SHIMS_PROVIDER_OVERRIDE = conf("spark.rapids.shims-provider-override")
    .internal()
    .startupOnly()
    .doc("Overrides the automatic Spark shim detection logic and forces a specific shims " +
      "provider class to be used. Set to the fully qualified shims provider class to use. " +
      "If you are using a custom Spark version such as Spark 3.2.0 then this can be used to " +
      "specify the shims provider that matches the base Spark version of Spark 3.2.0, i.e.: " +
      "com.nvidia.spark.rapids.shims.spark320.SparkShimServiceProvider. If you modified Spark " +
      "then there is no guarantee the cuDF plugin will function properly." +
      "When tested in a combined jar with other Shims, it's expected that the provided " +
      "implementation follows the same convention as existing Spark shims. If its class" +
      " name has the form com.nvidia.spark.rapids.shims.<shimId>.YourSparkShimServiceProvider. " +
      "The last package name component, i.e., shimId, can be used in the combined jar as the root" +
      " directory /shimId for any incompatible classes. When tested in isolation, no special " +
      "jar root is required"
    )
    .stringConf
    .createOptional

  val CUDF_VERSION_OVERRIDE = conf("spark.rapids.cudfVersionOverride")
    .internal()
    .startupOnly()
    .doc("Overrides the cudf version compatibility check between cudf jar and cuDF plugin " +
      "jar. If you are sure that the cudf jar which is mentioned in the classpath is compatible " +
      "with the cuDF plugin version, then set this to true.")
    .booleanConf
    .createWithDefault(false)

  object AllowMultipleJars extends Enumeration {
    val ALWAYS, SAME_REVISION, NEVER = Value
  }

  val ALLOW_MULTIPLE_JARS = conf("spark.rapids.sql.allowMultipleJars")
    .startupOnly()
    .doc("Allow multiple rapids-4-spark, spark-rapids-jni, and cudf jars on the classpath. " +
      "Spark will take the first one it finds, so the version may not be expected. Possisble " +
      "values are ALWAYS: allow all jars, SAME_REVISION: only allow jars with the same " +
      "revision, NEVER: do not allow multiple jars at all.")
    .stringConf
    .transform(_.toUpperCase(java.util.Locale.ROOT))
    .checkValues(AllowMultipleJars.values.map(_.toString))
    .createWithDefault(AllowMultipleJars.SAME_REVISION.toString)

  val ALLOW_DISABLE_ENTIRE_PLAN = conf("spark.rapids.allowDisableEntirePlan")
    .internal()
    .doc("The plugin has the ability to detect possibe incompatibility with some specific " +
      "queries and cluster configurations. In those cases the plugin will disable GPU support " +
      "for the entire query. Set this to false if you want to override that behavior, but use " +
      "with caution.")
    .booleanConf
    .createWithDefault(true)

  val OPTIMIZER_ENABLED = conf("spark.rapids.sql.optimizer.enabled")
      .internal()
      .doc("Enable cost-based optimizer that will attempt to avoid " +
          "transitions to GPU for operations that will not result in improved performance " +
          "over CPU")
      .booleanConf
      .createWithDefault(false)

  val OPTIMIZER_EXPLAIN = conf("spark.rapids.sql.optimizer.explain")
      .internal()
      .doc("Explain why some parts of a query were not placed on a GPU due to " +
          "optimization rules. Possible values are ALL: print everything, NONE: print nothing")
      .stringConf
      .createWithDefault("NONE")

  val OPTIMIZER_DEFAULT_ROW_COUNT = conf("spark.rapids.sql.optimizer.defaultRowCount")
    .internal()
    .doc("The cost-based optimizer uses estimated row counts to calculate costs and sometimes " +
      "there is no row count available so we need a default assumption to use in this case")
    .longConf
    .createWithDefault(1000000)

  val OPTIMIZER_CLASS_NAME = conf("spark.rapids.sql.optimizer.className")
    .internal()
    .doc("Optimizer implementation class name. The class must implement the " +
      "com.nvidia.spark.rapids.Optimizer trait")
    .stringConf
    .createWithDefault("com.nvidia.spark.rapids.CostBasedOptimizer")

  val OPTIMIZER_DEFAULT_CPU_OPERATOR_COST = conf("spark.rapids.sql.optimizer.cpu.exec.default")
    .internal()
    .doc("Default per-row CPU cost of executing an operator, in seconds")
    .doubleConf
    .createWithDefault(0.0002)

  val OPTIMIZER_DEFAULT_CPU_EXPRESSION_COST = conf("spark.rapids.sql.optimizer.cpu.expr.default")
    .internal()
    .doc("Default per-row CPU cost of evaluating an expression, in seconds")
    .doubleConf
    .createWithDefault(0.0)

  val OPTIMIZER_DEFAULT_GPU_OPERATOR_COST = conf("spark.rapids.sql.optimizer.gpu.exec.default")
      .internal()
      .doc("Default per-row GPU cost of executing an operator, in seconds")
      .doubleConf
      .createWithDefault(0.0001)

  val OPTIMIZER_DEFAULT_GPU_EXPRESSION_COST = conf("spark.rapids.sql.optimizer.gpu.expr.default")
      .internal()
      .doc("Default per-row GPU cost of evaluating an expression, in seconds")
      .doubleConf
      .createWithDefault(0.0)

  val OPTIMIZER_CPU_READ_SPEED = conf(
    "spark.rapids.sql.optimizer.cpuReadSpeed")
      .internal()
      .doc("Speed of reading data from CPU memory in GB/s")
      .doubleConf
      .createWithDefault(30.0)

  val OPTIMIZER_CPU_WRITE_SPEED = conf(
    "spark.rapids.sql.optimizer.cpuWriteSpeed")
    .internal()
    .doc("Speed of writing data to CPU memory in GB/s")
    .doubleConf
    .createWithDefault(30.0)

  val OPTIMIZER_GPU_READ_SPEED = conf(
    "spark.rapids.sql.optimizer.gpuReadSpeed")
    .internal()
    .doc("Speed of reading data from GPU memory in GB/s")
    .doubleConf
    .createWithDefault(320.0)

  val OPTIMIZER_GPU_WRITE_SPEED = conf(
    "spark.rapids.sql.optimizer.gpuWriteSpeed")
    .internal()
    .doc("Speed of writing data to GPU memory in GB/s")
    .doubleConf
    .createWithDefault(320.0)

  val USE_ARROW_OPT = conf("spark.rapids.arrowCopyOptimizationEnabled")
    .doc("Option to turn off using the optimized Arrow copy code when reading from " +
      "ArrowColumnVector in HostColumnarToGpu. Left as internal as user shouldn't " +
      "have to turn it off, but its convenient for testing.")
    .internal()
    .booleanConf
    .createWithDefault(true)

  val SPARK_GPU_RESOURCE_NAME = conf("spark.rapids.gpu.resourceName")
    .doc("The name of the Spark resource that represents a GPU that you want the plugin to use " +
      "if using custom resources with Spark.")
    .startupOnly()
    .stringConf
    .createWithDefault("gpu")

  val SUPPRESS_PLANNING_FAILURE = conf("spark.rapids.sql.suppressPlanningFailure")
    .doc("Option to fallback an individual query to CPU if an unexpected condition prevents the " +
      "query plan from being converted to a GPU-enabled one. Note this is different from " +
      "a normal CPU fallback for a yet-to-be-supported Spark SQL feature. If this happens " +
      "the error should be reported and investigated as a GitHub issue.")
    .booleanConf
    .createWithDefault(value = false)

  val ENABLE_FAST_SAMPLE = conf("spark.rapids.sql.fast.sample")
    .doc("Option to turn on fast sample. If enable it is inconsistent with CPU sample " +
      "because of GPU sample algorithm is inconsistent with CPU.")
    .booleanConf
    .createWithDefault(value = false)

  val DETECT_DELTA_LOG_QUERIES = conf("spark.rapids.sql.detectDeltaLogQueries")
    .doc("Queries against Delta Lake _delta_log JSON files are not efficient on the GPU. When " +
      "this option is enabled, the plugin will attempt to detect these queries and fall back " +
      "to the CPU.")
    .booleanConf
    .createWithDefault(value = true)

  val DETECT_DELTA_CHECKPOINT_QUERIES = conf("spark.rapids.sql.detectDeltaCheckpointQueries")
    .doc("Queries against Delta Lake _delta_log checkpoint Parquet files are not efficient on " +
      "the GPU. When this option is enabled, the plugin will attempt to detect these queries " +
      "and fall back to the CPU.")
    .booleanConf
    .createWithDefault(value = true)

  val NUM_FILES_FILTER_PARALLEL = conf("spark.rapids.sql.coalescing.reader.numFilterParallel")
    .doc("This controls the number of files the coalescing reader will run " +
      "in each thread when it filters blocks for reading. If this value is greater than zero " +
      "the files will be filtered in a multithreaded manner where each thread filters " +
      "the number of files set by this config. If this is set to zero the files are " +
      "filtered serially. This uses the same thread pool as the multithreaded reader, " +
      s"see $MULTITHREAD_READ_NUM_THREADS.")
    .integerConf
    .createWithDefault(value = 0)

  val CONCURRENT_WRITER_PARTITION_FLUSH_SIZE =
    conf("spark.rapids.sql.concurrentWriterPartitionFlushSize")
        .doc("The flush size of the concurrent writer cache in bytes for each partition. " +
            "If specified spark.sql.maxConcurrentOutputFileWriters, use concurrent writer to " +
            "write data. Concurrent writer first caches data for each partition and begins to " +
            "flush the data if it finds one partition with a size that is greater than or equal " +
            "to this config. The default value is 0, which will try to select a size based off " +
            "of file type specific configs. E.g.: It uses `write.parquet.row-group-size-bytes` " +
            "config for Parquet type and `orc.stripe.size` config for Orc type. " +
            "If the value is greater than 0, will use this positive value." +
            "Max value may get better performance but not always, because concurrent writer uses " +
            "spillable cache and big value may cause more IO swaps.")
        .bytesConf(ByteUnit.BYTE)
        .createWithDefault(0L)

  val NUM_SUB_PARTITIONS = conf("spark.rapids.sql.join.hash.numSubPartitions")
    .doc("The number of partitions for the repartition in each partition for big hash join. " +
      "GPU will try to repartition the data into smaller partitions in each partition when the " +
      "data from the build side is too large to fit into a single batch.")
    .internal()
    .integerConf
    .createWithDefault(16)

  val SIZED_JOIN_PARTITION_AMPLIFICATION =
    conf("spark.rapids.sql.join.sizedJoin.buildPartitionNumberAmplification")
      .doc("In sized join, by default we'll use bytes_of_build_size/batch_size + 1 as the number " +
        "of partitions for the build side. This config is used to amplify the number of " +
        "partitions for the build side. The default value is 1, which means we'll use the " +
        "default number of partitions. If the value is greater than 1, we'll amplify the " +
        "number of partitions by this value.")
      .internal()
      .doubleConf
      .checkValue(v => v >= 1, "The amplification factor must be greater than or equal to 1")
      .createWithDefault(1)

  val ENABLE_AQE_EXCHANGE_REUSE_FIXUP = conf("spark.rapids.sql.aqeExchangeReuseFixup.enable")
      .doc("Option to turn on the fixup of exchange reuse when running with " +
          "adaptive query execution.")
      .internal()
      .booleanConf
      .createWithDefault(true)

  val ENABLE_NON_AQE_BROADCAST_REUSE_FIXUP =
    conf("spark.rapids.sql.nonAqeBroadcastReuseFixup.enable")
      .doc("Option to turn on the fixup of broadcast exchange reuse for DPP " +
          "subqueries when AQE is disabled. The DPP-side GpuBroadcastExchange is built " +
          "during GpuOverrides and bypasses GpuTransitionOverrides, so it does not match " +
          "the join-side broadcast canonically. This fixup builds a per-query signature map " +
          "of join-side GpuBroadcastExchangeExec nodes in the main plan and rewrites a " +
          "matching DPP-side broadcast to ReusedExchangeExec.")
      .internal()
      .booleanConf
      .createWithDefault(true)

  val CHUNKED_PACK_POOL_SIZE = conf("spark.rapids.sql.chunkedPack.poolSize")
      .doc("Amount of GPU memory (in bytes) to set aside at startup for the chunked pack " +
           "scratch space, needed during spill from GPU to host memory. As a rule of thumb, each " +
           "column should see around 200B that will be allocated from this pool. " +
           "With the default of 10MB, a table of ~60,000 columns can be spilled using only this " +
           "pool. If this config is 0B, or if allocations fail, the plugin will retry with " +
           "the regular GPU memory resource.")
      .internal()
      .bytesConf(ByteUnit.BYTE)
      .createWithDefault(10L*1024*1024)

  val CHUNKED_PACK_BOUNCE_BUFFER_SIZE = conf("spark.rapids.sql.chunkedPack.bounceBufferSize")
      .doc("Amount of GPU memory (in bytes) to set aside at startup per chunked pack " +
          "bounce buffer, needed during spill from GPU to host memory. ")
      .internal()
      .bytesConf(ByteUnit.BYTE)
      .checkValue(v => v >= 1L*1024*1024,
        "The chunked pack bounce buffer must be at least 1MB in size")
      .createWithDefault(32L * 1024 * 1024)

  val CHUNKED_PACK_BOUNCE_BUFFER_COUNT = conf("spark.rapids.sql.chunkedPack.bounceBuffers")
    .doc("Number of chunked pack bounce buffers, needed during spill from GPU to host memory. ")
    .internal()
    .integerConf
    .checkValue(v => v >= 1,
      "The chunked pack bounce buffer count must be at least 1")
    .createWithDefault(4)

  val SPILL_TO_DISK_BOUNCE_BUFFER_SIZE =
    conf("spark.rapids.memory.host.spillToDiskBounceBufferSize")
      .doc("Amount of host memory (in bytes) to set aside at startup for the " +
        "bounce buffer used for gpu to disk spill that bypasses the host store.")
      .internal()
      .bytesConf(ByteUnit.BYTE)
      .checkValue(v => v >= 1,
        "The gpu to disk spill bounce buffer must have a positive size")
      .createWithDefault(32L * 1024 * 1024)

  val SPILL_TO_DISK_BOUNCE_BUFFER_COUNT =
    conf("spark.rapids.memory.host.spillToDiskBounceBuffers")
      .doc("Number of bounce buffers used for gpu to disk spill that bypasses the host store.")
      .internal()
      .integerConf
      .checkValue(v => v >= 1,
        "The gpu to disk spill bounce buffer count must be positive")
      .createWithDefault(4)

  val SPLIT_UNTIL_SIZE_OVERRIDE = conf("spark.rapids.sql.test.overrides.splitUntilSize")
      .doc("Only for tests: override the value of GpuDeviceManager.splitUntilSize")
      .internal()
      .longConf
      .createOptional

  val PROJECT_SPLIT_RETRY_ENABLED = conf("spark.rapids.sql.projectExec.splitRetry.enabled")
      .doc("When true, GpuProjectExec uses split-and-retry on GPU OOM for retryable " +
          "projections: the input batch is halved by rows and the projection is re-run on " +
          "each half. Projections that include non-retryable expressions fall back to the " +
          "existing withRetryNoSplit path because those expressions cannot be safely " +
          "re-evaluated on row-split inputs. Disable this to revert to the prior behavior.")
      .internal()
      .booleanConf
      .createWithDefault(true)

  val TEST_IO_ENCRYPTION = conf("spark.rapids.test.io.encryption")
    .doc("Only for tests: verify for IO encryption")
    .internal()
    .booleanConf
    .createOptional

  val SKIP_GPU_ARCH_CHECK = conf("spark.rapids.skipGpuArchitectureCheck")
    .doc("When true, skips GPU architecture compatibility check. Note that this check " +
      "might still be present in cuDF.")
    .internal()
    .booleanConf
    .createWithDefault(false)

  val TEST_GET_JSON_OBJECT_SAVE_PATH = conf("spark.rapids.sql.expression.GetJsonObject.debugPath")
    .doc("Only for tests: specify a directory to save CSV debug output for get_json_object " +
      "if the output differs from the CPU version. Multiple files may be saved")
    .internal()
    .stringConf
    .createOptional

  val TEST_GET_JSON_OBJECT_SAVE_ROWS =
    conf("spark.rapids.sql.expression.GetJsonObject.debugSaveRows")
      .doc("Only for tests: when a debugPath is provided this is the number " +
        "of rows that is saved per file. There may be multiple files if there " +
        "are multiple tasks or multiple batches within a task")
      .internal()
      .integerConf
      .createWithDefault(1024)

  val DELTA_LOW_SHUFFLE_MERGE_SCATTER_DEL_VECTOR_BATCH_SIZE =
    conf("spark.rapids.sql.delta.lowShuffleMerge.deletion.scatter.max.size")
      .doc("Option to set max batch size when scattering deletion vector")
      .internal()
      .integerConf
      .createWithDefault(32 * 1024)

  val DELTA_LOW_SHUFFLE_MERGE_DEL_VECTOR_BROADCAST_THRESHOLD =
    conf("spark.rapids.sql.delta.lowShuffleMerge.deletionVector.broadcast.threshold")
      .doc("Currently we need to broadcast deletion vector to all executors to perform low " +
        "shuffle merge. When we detect the deletion vector broadcast size is larger than this " +
        "value, we will fallback to normal shuffle merge.")
      .bytesConf(ByteUnit.BYTE)
      .createWithDefault(20 * 1024 * 1024)

  val ENABLE_DELTA_LOW_SHUFFLE_MERGE =
    conf("spark.rapids.sql.delta.lowShuffleMerge.enabled")
    .doc("Option to turn on the low shuffle merge for Delta Lake. Currently there are some " +
      "limitations for this feature: " +
      "1. We only support Delta Lake 2.4. " +
      s"2. The file scan mode must be set to ${RapidsReaderType.PERFILE} " +
      "3. The deletion vector size must be smaller than " +
      s"${DELTA_LOW_SHUFFLE_MERGE_DEL_VECTOR_BROADCAST_THRESHOLD.key} ")
    .booleanConf
    .createWithDefault(false)

    val DELTA_DELETION_VECTOR_PREDICATE_PUSHDOWN =
    conf("spark.rapids.sql.delta.deletionVectors.predicatePushdown.enabled")
      .doc("When true, the deletion vector processing will be pushed down to " +
        "the GPU Delta Lake scans. The result of the scan will contain only the rows " +
        "that are not deleted according to the deletion vector. When false, " +
        "the deletion vectors will be materialized as a boolean column and " +
        "the GPU filter operator will process it together with other filters. " +
        "This setting is effective only when " +
        "spark.databricks.delta.deletionVectors.useMetadataRowIndex is true. " +
        "Otherwise, this setting is fixed to false regardless of its actual value.")
      .internal()
      .booleanConf
      .createWithDefault(true)

  val ENABLE_HASH_FUNCTION_IN_PARTITIONING =
    conf("spark.rapids.sql.partitioning.hashFunction.enabled")
      .doc("When false, Only Murmur3Hash is used for GPU hash partitioning to " +
        "align with the regular Spark. When enabled, GPU will try to infer the hash " +
        "function from the CPU hash partitioning and use the same one. This is for " +
        "a customized Spark supporting multiple hash functions in hash partitioning. " +
        "So far only 'HiveHash' and 'Murmur3Hash' are supported on GPU. This requires " +
        s"'${INCOMPATIBLE_OPS.key}' to also be set to true.")
      .internal()
      .booleanConf
      .createWithDefault(true)

  val TAG_LORE_ID_ENABLED = conf("spark.rapids.sql.lore.tag.enabled")
    .doc("Enable add a LORE id to each gpu plan node")
    .internal()
    .booleanConf
    .createWithDefault(true)

  val LORE_DUMP_IDS = conf("spark.rapids.sql.lore.idsToDump")
    .doc("Specify the LORE ids of operators to dump. The format is a comma separated list of " +
      "LORE ids. For example: \"1[0]\" will dump partition 0 of input of gpu operator " +
      "with lore id 1. For more details, please refer to " +
      "[the LORE documentation](../dev/lore.md). If this is not set, no data will be dumped.")
    .stringConf
    .createOptional

  val LORE_DUMP_PATH = conf("spark.rapids.sql.lore.dumpPath")
    .doc(s"The path to dump the LORE nodes' input data. This must be set if ${LORE_DUMP_IDS.key} " +
      "has been set. The data of each LORE node will be dumped to a subfolder with name " +
      "'loreId-<LORE id>' under this path. For more details, please refer to " +
      "[the LORE documentation](../dev/lore.md).")
    .stringConf
    .createOptional

  val LORE_SKIP_DUMPING_PLAN = conf("spark.rapids.sql.lore.skip.plan.dump")
    .doc("Skip dumping plan metadata when doing lore dump")
    .internal()
    .booleanConf
    .createWithDefault(false)

  val LORE_NON_STRICT_MODE = conf("spark.rapids.sql.lore.nonStrictMode.enabled")
    .doc("Allow LoRE dumping to continue when a selected lore id fails. When enabled, failing " +
      "lore ids are skipped with a warning, previously dumped data under the dump path is kept, " +
      "and the rest of the query continues executing.")
    .booleanConf
    .createWithDefault(false)

  val OP_TIME_TRACKING_RDD_ENABLED = conf("spark.rapids.sql.exec.opTimeTrackingRDD.enabled")
    .doc("Enable OpTimeTrackingRDD for all GPU operations. When true, OpTimeTrackingRDD " +
      "wrappers will be created to track operation time. When false, can improve " +
      "performance by avoiding overhead of operation time tracking.")
    .booleanConf
    .createWithDefault(true)

  val LORE_PARQUET_USE_ORIGINAL_NAMES =
    conf("spark.rapids.sql.lore.parquet.useOriginalSchemaNames")
      .doc("When enabled, LORE writes Parquet files using the original Spark schema names " +
        "instead of auto-generated type-based names. This makes the dumped Parquet data " +
        "easier to consume directly via Spark/other tools.")
      .booleanConf
      .createWithDefault(true)

  val CASE_WHEN_FUSE =
    conf("spark.rapids.sql.case_when.fuse")
      .doc("If when branches is greater than 2 and all then/else values in case when are string " +
        "scalar, fuse mode improves the performance. By default this is enabled.")
      .internal()
      .booleanConf
      .createWithDefault(true)

  val TRACE_TASK_GPU_OWNERSHIP = conf("spark.rapids.sql.nvtx.traceTaskGpuOwnership")
    .doc("Enable tracing of the GPU ownership of tasks. This can be useful for debugging " +
      "deadlocks and other issues related to GPU semaphore.")
    .internal()
    .booleanConf
    .createWithDefault(false)

  val ENABLE_ASYNC_OUTPUT_WRITE =
    conf("spark.rapids.sql.asyncWrite.queryOutput.enabled")
      .doc("Option to turn on the async query output write. During the final output write, the " +
        "task first copies the output to the host memory, and then writes it into the storage. " +
        "When this option is enabled, the task will asynchronously write the output in the host " +
        "memory to the storage. Only the Parquet and ORC formats are supported currently.")
      .internal()
      .booleanConf
      .createWithDefault(false)

  val ASYNC_QUERY_OUTPUT_WRITE_HOLD_GPU_IN_TASK =
    conf("spark.rapids.sql.queryOutput.holdGpuInTask")
      .doc("Option to hold GPU semaphore between batch processing during the final output write. " +
        "This option could degrade query performance if it is enabled without the async query " +
        "output write. It is recommended to consider enabling this option only when " +
        s"${ENABLE_ASYNC_OUTPUT_WRITE.key} is set. This option is off by default when the async " +
        "query output write is disabled; otherwise, it is on.")
      .internal()
      .booleanConf
      .createOptional

  val ASYNC_WRITE_MAX_IN_FLIGHT_HOST_MEMORY_BYTES =
    conf("spark.rapids.sql.asyncWrite.maxInFlightHostMemoryBytes")
      .doc("Maximum number of host memory bytes per executor that can be in-flight for async " +
        "write. Tasks may be blocked if the total host memory bytes in-flight " +
        "exceeds this value. Today this config only covers file output write, but in future" +
        "it may cover other writes like shuffle write as well. If set to <= 0 it means unlimited " +
        "memory")
      .internal()
      .bytesConf(ByteUnit.BYTE)
      .createWithDefault(2L * 1024 * 1024 * 1024)

  val ASYNC_READ_MAX_IN_FLIGHT_HOST_MEMORY_BYTES =
    conf("spark.rapids.sql.asyncRead.maxInFlightHostMemoryBytes")
      .doc("Maximum number of host memory bytes per executor that can be in-flight for async " +
        "read. Tasks may be blocked if the total host memory bytes in-flight " +
        "exceeds this value. Today this config only covers shuffle read, but in future" +
        "it may cover other reads like file read as well. If set to <= 0 it means unlimited " +
        "memory")
      .internal()
      .bytesConf(ByteUnit.BYTE)
      // Why by default set to unlimited? The reasons are:
      // 1. For async shuffle read (done) or async file read (already there but need to integrate
      // into this unified throttling), the host memory usage is bounded: N * batchSize * 2, where N
      // is the number of total task number in executor, and "*2" is for prefetch + concatenation.
      // 2. Even without asyncRead, today each task is already allowed to use host memory to prepare
      // its data in CPU before it acquires GPU semaphore. Take shuffle read for example
      // the host memory usage is already high, actually async shuffle read is adding just a
      // little more memory pressure (concurrentGpuTasks * batchSize * 2), note concurrentGpuTasks
      // is typically much smaller than N.
      // 3. The read in data will be spillable, so it won't have deadly consequences.
      // 4. We have not yet implemented unified thread priority, so there's deadlock risks if this
      // value is improperly set.
      .createWithDefault(-1)
}
