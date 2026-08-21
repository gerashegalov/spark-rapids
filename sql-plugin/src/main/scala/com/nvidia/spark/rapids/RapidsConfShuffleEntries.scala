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

import com.nvidia.spark.rapids.jni.kudo.DumpOption

import org.apache.spark.network.util.ByteUnit

private[rapids] trait RapidsConfShuffleEntries extends RapidsConfFormatEntries {

  // INTERNAL TEST AND DEBUG CONFIGS

  val TEST_RETRY_OOM_INJECTION_MODE = conf("spark.rapids.sql.test.injectRetryOOM")
    .doc("Only to be used in tests. If `true` the retry iterator will inject a GpuRetryOOM " +
         "or CpuRetryOOM once per invocation. Furthermore an extended config " +
         "`num_ooms=<int>,skip=<int>,type=CPU|GPU|CPU_OR_GPU,split=<bool>` can be provided to " +
         "specify the number of OOMs to generate; how many to skip before doing so; whether to " +
         "filter by allocation events by host(CPU), device(GPU), or both (CPU_OR_GPU); " +
         "whether to inject *SplitAndRetryOOM instead of plain *RetryOOM exceptions." +
         "Note *SplitAndRetryOOM exceptions are not always handled - use with care.")
    .internal()
    .stringConf
    .createWithDefault(false.toString)

  val FOLDABLE_NON_LIT_ALLOWED = conf("spark.rapids.sql.test.isFoldableNonLitAllowed")
    .doc("Only to be used in tests. If `true` the foldable expressions that are not literals " +
      "will be allowed")
    .internal()
    .booleanConf
    .createWithDefault(false)

  val TEST_RETRY_CONTEXT_CHECK_ENABLED = conf("spark.rapids.sql.test.retryContextCheck.enabled")
    .doc("Only to be used in tests. When set to true, enable the context check for " +
      "GPU nondeterministic expressions but declaring to be retryable. A GPU retryable " +
      "nondeterministic expression should run inside a checkpoint-restore context. And it " +
      "will blow up when the context does not satisfy.")
    .internal()
    .booleanConf
    .createWithDefault(false)

  val TEST_CONF = conf("spark.rapids.sql.test.enabled")
    .doc("Intended to be used by unit tests, if enabled all operations must run on the " +
      "GPU or an error happens.")
    .internal()
    .booleanConf
    .createWithDefault(false)

  val TEST_ALLOWED_NONGPU = conf("spark.rapids.sql.test.allowedNonGpu")
    .doc("Comma separate string of exec or expression class names that are allowed " +
      "to not be GPU accelerated for testing.")
    .internal()
    .stringConf
    .toSequence
    .createWithDefault(Nil)

  val TEST_VALIDATE_EXECS_ONGPU = conf("spark.rapids.sql.test.validateExecsInGpuPlan")
    .doc("Comma separate string of exec class names to validate they " +
      "are GPU accelerated. Used for testing.")
    .internal()
    .stringConf
    .toSequence
    .createWithDefault(Nil)

  val HASH_SUB_PARTITION_TEST_ENABLED = conf("spark.rapids.sql.test.subPartitioning.enabled")
    .doc("Setting to true will force hash joins to use the sub-partitioning algorithm if " +
      s"${TEST_CONF.key} is also enabled, while false means always disabling it. This is " +
      "intended for tests. Do not set any value under production environments, since it " +
      "will override the default behavior that will choose one automatically according to " +
      "the input batch size")
    .internal()
    .booleanConf
    .createOptional

  val LOG_TRANSFORMATIONS = conf("spark.rapids.sql.debug.logTransformations")
    .doc("When enabled, all query transformations will be logged.")
    .internal()
    .booleanConf
    .createWithDefault(false)

  val OUTPUT_DEBUG_DUMP_PREFIX = conf("spark.rapids.sql.output.debug.dumpPrefix")
    .doc("A path prefix where data that is intended to be written out as the result " +
      "of a query should be dumped for debugging. The format of this is based on " +
      "JCudfSerialization and is trying to capture the underlying table so that if " +
      "there are errors in the output format we can try to recreate it.")
    .internal()
    .stringConf
    .createOptional

  val PARQUET_DEBUG_DUMP_PREFIX = conf("spark.rapids.sql.parquet.debug.dumpPrefix")
    .doc("A path prefix where Parquet split file data is dumped for debugging.")
    .internal()
    .stringConf
    .createOptional

  val PARQUET_DEBUG_DUMP_ALWAYS = conf("spark.rapids.sql.parquet.debug.dumpAlways")
    .doc(s"This only has an effect if $PARQUET_DEBUG_DUMP_PREFIX is set. If true then " +
      "Parquet data is dumped for every read operation otherwise only on a read error.")
    .internal()
    .booleanConf
    .createWithDefault(false)

  val ORC_DEBUG_DUMP_PREFIX = conf("spark.rapids.sql.orc.debug.dumpPrefix")
    .doc("A path prefix where ORC split file data is dumped for debugging.")
    .internal()
    .stringConf
    .createOptional

  val ORC_DEBUG_DUMP_ALWAYS = conf("spark.rapids.sql.orc.debug.dumpAlways")
    .doc(s"This only has an effect if $ORC_DEBUG_DUMP_PREFIX is set. If true then " +
      "ORC data is dumped for every read operation otherwise only on a read error.")
    .internal()
    .booleanConf
    .createWithDefault(false)

  val AVRO_DEBUG_DUMP_PREFIX = conf("spark.rapids.sql.avro.debug.dumpPrefix")
    .doc("A path prefix where AVRO split file data is dumped for debugging.")
    .internal()
    .stringConf
    .createOptional

  val AVRO_DEBUG_DUMP_ALWAYS = conf("spark.rapids.sql.avro.debug.dumpAlways")
    .doc(s"This only has an effect if $AVRO_DEBUG_DUMP_PREFIX is set. If true then " +
      "Avro data is dumped for every read operation otherwise only on a read error.")
    .internal()
    .booleanConf
    .createWithDefault(false)

  val HYBRID_PARQUET_READER = conf("spark.rapids.sql.hybrid.parquet.enabled")
    .doc("Use HybridScan to read Parquet data using CPUs. The underlying implementation " +
      "leverages both Gluten and Velox. Supports Spark 3.2.2, 3.3.1, 3.4.2, and 3.5.1 " +
      "as Gluten does, also supports other versions but not fully tested.")
    .internal()
    .booleanConf
    .createWithDefault(false)

  val HYBRID_PARQUET_PRELOAD_CAP = conf("spark.rapids.sql.hybrid.parquet.numPreloadedBatches")
    .doc("Preloading capacity of HybridParquetScan. If > 0, will enable preloading" +
      " the result of HybridParquetScan asynchronously in a separate thread")
    .internal()
    .integerConf
    .createWithDefault(0)

  // This config name is the same as HybridPluginWrapper in Hybrid jar,
  // can not refer to Hybrid jar because of the jar is optional.
  val LOAD_HYBRID_BACKEND = conf("spark.rapids.sql.hybrid.loadBackend")
    .doc("Load hybrid backend as an extra plugin of cuDF plugin during launch time")
    .internal()
    .startupOnly()
    .booleanConf
    .createWithDefault(false)

  object HybridFilterPushdownType extends Enumeration {
    val CPU, GPU, OFF = Value
  }

  val PUSH_DOWN_FILTERS_TO_HYBRID = conf("spark.rapids.sql.hybrid.parquet.filterPushDown")
    .doc("Push down all supported filters to CPU if set to CPU. " +
      "If set to GPU, no filters will be pushed down so all filters are on the GPU. " +
      "If set to OFF, filters will be both pushed down and keeped on the GPU. " +
      "OFF is to make the behavior same as before.")
    .internal()
    .stringConf
    .transform(_.toUpperCase(java.util.Locale.ROOT))
    .checkValues(HybridFilterPushdownType.values.map(_.toString))
    .createWithDefault(HybridFilterPushdownType.CPU.toString)

  val HYBRID_EXPRS_WHITELIST = conf("spark.rapids.sql.hybrid.whitelistExprs")
    .doc("White list of expressions that can be pushed down to CPU. " +
      "The expressions are separated by comma.")
    .internal()
    .stringConf
    .createWithDefault("")

  val HASH_AGG_REPLACE_MODE = conf("spark.rapids.sql.hashAgg.replaceMode")
    .doc("Only when hash aggregate exec has these modes (\"all\" by default): " +
      "\"all\" (try to replace all aggregates, default), " +
      "\"complete\" (exclusively replace complete aggregates), " +
      "\"partial\" (exclusively replace partial aggregates), " +
      "\"final\" (exclusively replace final aggregates)." +
      " These modes can be connected with &(AND) or |(OR) to form sophisticated patterns.")
    .internal()
    .stringConf
    .createWithDefault("all")

  val PARTIAL_MERGE_DISTINCT_ENABLED = conf("spark.rapids.sql.partialMerge.distinct.enabled")
    .doc("Enables aggregates that are in PartialMerge mode to run on the GPU if true")
    .internal()
    .booleanConf
    .createWithDefault(true)

  val SHUFFLE_MANAGER_ENABLED = conf("spark.rapids.shuffle.enabled")
    .doc("Enable or disable the RAPIDS Shuffle Manager implementation at runtime. On supported " +
      "Spark versions, including Spark 4.0.0 and later, the " +
      "[RAPIDS Shuffle Manager](https://docs.nvidia.com/spark-rapids/user-guide/latest" +
      "/additional-functionality/rapids-shuffle.html) is configured automatically unless " +
      "spark.shuffle.manager is explicitly set. Note: Databricks runtimes may explicitly set " +
      "spark.shuffle.manager, preventing automatic configuration from taking effect. To " +
      "override it, explicitly set spark.shuffle.manager to the RAPIDS Shuffle Manager class. " +
      "This automatic configuration is skipped on Dataproc runtimes that set " +
      "spark.dataproc.engine, including Lightning Engine runtimes; on those runtimes, " +
      "spark.shuffle.manager remains unset unless explicitly configured. " +
      "On earlier Spark versions, the RAPIDS Shuffle Manager must already be configured. When " +
      "set to `false`, the built-in Spark shuffle implementation will be used. ")
    .booleanConf
    .createWithDefault(true)

  object RapidsShuffleManagerMode extends Enumeration {
    val UCX, CACHE_ONLY, MULTITHREADED = Value
  }

  val SHUFFLE_MANAGER_MODE = conf("spark.rapids.shuffle.mode")
    .doc("RAPIDS Shuffle Manager mode. " +
      "\"MULTITHREADED\": shuffle file writes and reads are parallelized using a thread pool. " +
      "\"UCX\": (requires UCX installation) uses accelerated transports for " +
      "transferring shuffle blocks. " +
      "\"CACHE_ONLY\": use when running a single executor, for short-circuit cached " +
      "shuffle (for testing purposes).")
    .startupOnly()
    .stringConf
    .checkValues(RapidsShuffleManagerMode.values.map(_.toString))
    .createWithDefault(RapidsShuffleManagerMode.MULTITHREADED.toString)

  val MULTITHREADED_SHUFFLE_SKIP_MERGE = conf("spark.rapids.shuffle.multithreaded.skipMerge")
    .doc("When using MULTITHREADED shuffle mode, skip merging partial shuffle files and " +
      "instead serve data directly from the MultithreadedShuffleBufferCatalog. " +
      "This avoids I/O overhead from merging but requires: (1) External Shuffle Service (ESS) " +
      "to be disabled, and (2) spark.rapids.memory.host.offHeapLimit.enabled=true (off-heap " +
      "memory limits enabled) to prevent OOM from unbounded buffer growth. " +
      "When set to false (default), partial files will be merged into a single " +
      "shuffle file per map task as in standard Spark shuffle. " +
      "Set to true when both requirements are met and shuffle data is not reused across " +
      "SQL queries (e.g., avoid on Databricks with shuffle reuse enabled).")
    .startupOnly()
    .booleanConf
    .createWithDefault(false)

  val SHUFFLE_TRANSPORT_EARLY_START = conf("spark.rapids.shuffle.transport.earlyStart")
    .doc("Enable early connection establishment for RAPIDS Shuffle")
    .startupOnly()
    .booleanConf
    .createWithDefault(true)

  val SHUFFLE_TRANSPORT_EARLY_START_HEARTBEAT_INTERVAL =
    conf("spark.rapids.shuffle.transport.earlyStart.heartbeatInterval")
      .doc("Shuffle early start heartbeat interval (milliseconds). " +
        "Executors will send a heartbeat RPC message to the driver at this interval")
      .startupOnly()
      .integerConf
      .createWithDefault(5000)

  val SHUFFLE_TRANSPORT_EARLY_START_HEARTBEAT_TIMEOUT =
    conf("spark.rapids.shuffle.transport.earlyStart.heartbeatTimeout")
      .doc(s"Shuffle early start heartbeat timeout (milliseconds). " +
        s"Executors that don't heartbeat within this timeout will be considered stale. " +
        s"This timeout must be higher than the value for " +
        s"${SHUFFLE_TRANSPORT_EARLY_START_HEARTBEAT_INTERVAL.key}")
      .startupOnly()
      .integerConf
      .createWithDefault(10000)

  val SHUFFLE_TRANSPORT_CLASS_NAME = conf("spark.rapids.shuffle.transport.class")
    .doc("The class of the specific RapidsShuffleTransport to use during the shuffle.")
    .internal()
    .startupOnly()
    .stringConf
    .createWithDefault("com.nvidia.spark.rapids.shuffle.ucx.UCXShuffleTransport")

  val SHUFFLE_TRANSPORT_MAX_RECEIVE_INFLIGHT_BYTES =
    conf("spark.rapids.shuffle.transport.maxReceiveInflightBytes")
      .doc("Maximum aggregate amount of bytes that be fetched at any given time from peers " +
        "during shuffle")
      .startupOnly()
      .bytesConf(ByteUnit.BYTE)
      .createWithDefault(1024 * 1024 * 1024)

  val SHUFFLE_UCX_ACTIVE_MESSAGES_FORCE_RNDV =
    conf("spark.rapids.shuffle.ucx.activeMessages.forceRndv")
      .doc("Set to true to force 'rndv' mode for all UCX Active Messages. " +
        "This should only be required with UCX 1.10.x. UCX 1.11.x deployments should " +
        "set to false.")
      .startupOnly()
      .booleanConf
      .createWithDefault(false)

  val SHUFFLE_UCX_USE_WAKEUP = conf("spark.rapids.shuffle.ucx.useWakeup")
    .doc("When set to true, use UCX's event-based progress (epoll) in order to wake up " +
      "the progress thread when needed, instead of a hot loop.")
    .startupOnly()
    .booleanConf
    .createWithDefault(true)

  val SHUFFLE_UCX_LISTENER_START_PORT = conf("spark.rapids.shuffle.ucx.listenerStartPort")
    .doc("Starting port to try to bind the UCX listener.")
    .internal()
    .startupOnly()
    .integerConf
    .createWithDefault(0)

  val SHUFFLE_UCX_MGMT_SERVER_HOST = conf("spark.rapids.shuffle.ucx.managementServerHost")
    .doc("The host to be used to start the management server")
    .startupOnly()
    .stringConf
    .createWithDefault(null)

  val SHUFFLE_UCX_MGMT_CONNECTION_TIMEOUT =
    conf("spark.rapids.shuffle.ucx.managementConnectionTimeout")
    .doc("The timeout for client connections to a remote peer")
    .internal()
    .startupOnly()
    .integerConf
    .createWithDefault(0)

  val SHUFFLE_UCX_BOUNCE_BUFFERS_SIZE = conf("spark.rapids.shuffle.ucx.bounceBuffers.size")
    .doc("The size of bounce buffer to use in bytes. Note that this size will be the same " +
      "for device and host memory")
    .internal()
    .startupOnly()
    .bytesConf(ByteUnit.BYTE)
    .createWithDefault(4 * 1024  * 1024)

  val SHUFFLE_UCX_BOUNCE_BUFFERS_DEVICE_COUNT =
    conf("spark.rapids.shuffle.ucx.bounceBuffers.device.count")
    .doc("The number of bounce buffers to pre-allocate from device memory")
    .internal()
    .startupOnly()
    .integerConf
    .createWithDefault(32)

  val SHUFFLE_UCX_BOUNCE_BUFFERS_HOST_COUNT =
    conf("spark.rapids.shuffle.ucx.bounceBuffers.host.count")
    .doc("The number of bounce buffers to pre-allocate from host memory")
    .internal()
    .startupOnly()
    .integerConf
    .createWithDefault(32)

  val SHUFFLE_MAX_CLIENT_THREADS = conf("spark.rapids.shuffle.maxClientThreads")
    .doc("The maximum number of threads that the shuffle client should be allowed to start")
    .internal()
    .startupOnly()
    .integerConf
    .createWithDefault(50)

  val SHUFFLE_MAX_CLIENT_TASKS = conf("spark.rapids.shuffle.maxClientTasks")
    .doc("The maximum number of tasks shuffle clients will queue before adding threads " +
      s"(up to spark.rapids.shuffle.maxClientThreads), or slowing down the transport")
    .internal()
    .startupOnly()
    .integerConf
    .createWithDefault(100)

  val SHUFFLE_CLIENT_THREAD_KEEPALIVE = conf("spark.rapids.shuffle.clientThreadKeepAlive")
    .doc("The number of seconds that the ThreadPoolExecutor will allow an idle client " +
      "shuffle thread to stay alive, before reclaiming.")
    .internal()
    .startupOnly()
    .integerConf
    .createWithDefault(30)

  val SHUFFLE_MAX_SERVER_TASKS = conf("spark.rapids.shuffle.maxServerTasks")
    .doc("The maximum number of tasks the shuffle server will queue up for its thread")
    .internal()
    .startupOnly()
    .integerConf
    .createWithDefault(1000)

  val SHUFFLE_MAX_METADATA_SIZE = conf("spark.rapids.shuffle.maxMetadataSize")
    .doc("The maximum size of a metadata message that the shuffle plugin will keep in its " +
      "direct message pool. ")
    .internal()
    .startupOnly()
    .bytesConf(ByteUnit.BYTE)
    .createWithDefault(500 * 1024)

  val SHUFFLE_COMPRESSION_CODEC = conf("spark.rapids.shuffle.compression.codec")
    .doc("The GPU codec used to compress shuffle data when using RAPIDS shuffle. " +
      "Supported codecs: zstd, lz4, copy, none")
    .internal()
    .startupOnly()
    .stringConf
    .createWithDefault("none")

val SHUFFLE_COMPRESSION_LZ4_CHUNK_SIZE = conf("spark.rapids.shuffle.compression.lz4.chunkSize")
    .doc("A configurable chunk size to use when compressing with LZ4.")
    .internal()
    .startupOnly()
    .bytesConf(ByteUnit.BYTE)
    .createWithDefault(64 * 1024)

  val SHUFFLE_COMPRESSION_ZSTD_CHUNK_SIZE =
    conf("spark.rapids.shuffle.compression.zstd.chunkSize")
      .doc("A configurable chunk size to use when compressing with ZSTD.")
      .internal()
      .startupOnly()
      .bytesConf(ByteUnit.BYTE)
      .createWithDefault(64 * 1024)

  val FORCE_HIVE_HASH_FOR_BUCKETED_WRITE =
    conf("spark.rapids.sql.format.write.forceHiveHashForBucketing")
      .doc("Hive write commands before Spark 330 use Murmur3Hash for bucketed write. " +
        "When enabled, HiveHash will be always used for this instead of Murmur3. This is " +
        "used to align with some customized Spark binaries before 330.")
      .internal()
      .booleanConf
      .createWithDefault(false)

  val SHUFFLE_MULTITHREADED_MAX_BYTES_IN_FLIGHT =
    conf("spark.rapids.shuffle.multiThreaded.maxBytesInFlight")
      .doc(
        "The size limit, in bytes, that the RAPIDS shuffle manager configured in " +
        "\"MULTITHREADED\" mode will allow to be serialized or deserialized concurrently " +
        "per task. This is also the maximum amount of memory that will be used per task. " +
        "This should be set larger " +
        "than Spark's default maxBytesInFlight (48MB). The larger this setting is, the " +
        "more compressed shuffle chunks are processed concurrently. In practice, " +
        "care needs to be taken to not go over the amount of off-heap memory that Netty has " +
        "available. See https://github.com/NVIDIA/cudf-spark/issues/9153.")
      .startupOnly()
      .bytesConf(ByteUnit.BYTE)
      .createWithDefault(128 * 1024 * 1024)

  val SHUFFLE_MULTITHREADED_WRITER_THREADS =
    conf("spark.rapids.shuffle.multiThreaded.writer.threads")
      .doc("The number of threads to use for writing shuffle blocks per executor in the " +
          "RAPIDS shuffle manager configured in \"MULTITHREADED\" mode. " +
          "There are two special values: " +
          "0 = feature is disabled, falls back to Spark built-in shuffle writer; " +
          "1 = our implementation of Spark's built-in shuffle writer with extra metrics.")
      .startupOnly()
      .integerConf
      .createWithDefault(20)

  val SHUFFLE_PARTITIONING_MAX_CPU_BATCH_SIZE =
    conf("spark.rapids.shuffle.partitioning.maxCpuBatchSize")
      .doc("The maximum size of a sliced batch output to the CPU side " +
        "when GPU partitioning shuffle data. This can be used to limit the peak on-heap memory " +
        "used by CPU to serialize the shuffle data, especially for skew data cases. " +
        "The default value is maximum size of an Array minus 2k overhead (2147483639L - 2048L), " +
        "user should only set a smaller value than default value to avoid subsequent failures.")
      .internal()
      .bytesConf(ByteUnit.BYTE)
      .checkValue(v => v > 0 && v <= 2147483639L - 2048L,
        s"maxCpuBatchSize must be positive and not exceed ${2147483639L - 2048L} bytes.")
      // The maximum size of an Array minus a bit for overhead for metadata
      .createWithDefault(2147483639L - 2048L)

  val SHUFFLE_MULTITHREADED_READER_THREADS =
    conf("spark.rapids.shuffle.multiThreaded.reader.threads")
        .doc("The number of threads to use for reading shuffle blocks per executor in the " +
            "RAPIDS shuffle manager configured in \"MULTITHREADED\" mode. " +
            "There are two special values: " +
            "0 = feature is disabled, falls back to Spark built-in shuffle reader; " +
            "1 = our implementation of Spark's built-in shuffle reader with extra metrics.")
        .startupOnly()
        .integerConf
        .createWithDefault(20)

  val SHUFFLE_KUDO_SERIALIZER_ENABLED = conf("spark.rapids.shuffle.kudo.serializer.enabled")
    .doc("Enable or disable the Kudo serializer for the shuffle.")
    .internal()
    .startupOnly()
    .booleanConf
    .createWithDefault(true)

  object ShuffleKudoMode extends Enumeration {
    val CPU, GPU = Value
  }

  val SHUFFLE_KUDO_WRITE_MODE = conf("spark.rapids.shuffle.kudo.serializer.write.mode")
    .doc("Kudo serializer mode. " +
      "\"CPU\": serialize shuffle outputs on the cpu. " +
      "\"GPU\": serialize shuffle outputs on the gpu. ")
    .internal()
    .startupOnly()
    .stringConf
    .createWithDefault(ShuffleKudoMode.CPU.toString)

  val SHUFFLE_KUDO_READ_MODE = conf("spark.rapids.shuffle.kudo.serializer.read.mode")
    .doc("Kudo serializer read mode. " +
      "\"CPU\": deserialize shuffle inputs on the cpu. " +
      "\"GPU\": deserialize shuffle inputs on the gpu. ")
    .internal()
    .startupOnly()
    .stringConf
    .createWithDefault(ShuffleKudoMode.GPU.toString)

  val SHUFFLE_KUDO_SERIALIZER_MEASURE_BUFFER_COPY_ENABLED =
    conf("spark.rapids.shuffle.kudo.serializer.measure.buffer.copy.enabled")
    .doc("Enable or disable measuring buffer copy time when using Kudo serializer for the shuffle.")
    .internal()
    .startupOnly()
    .booleanConf
    .createWithDefault(false)

  val SHUFFLE_ASYNC_READ_ENABLED = conf("spark.rapids.sql.asyncRead.shuffle.enabled")
    .doc("Enable or disable the asynchronous read for Shuffle. If you turn this on you should " +
      "also consider increasing spark.rapids.sql.asyncRead.maxInFlightHostMemoryBytes so that " +
      "async threads won't be blocked by the memory limit. If By default this in off now.")
    .internal()
    .startupOnly()
    .booleanConf
    .createWithDefault(false)

  // ["NEVER", "ALWAYS", "ONFAILURE"]
  private val KudoDebugModes =
    DumpOption.values.map(_.toString.toUpperCase(java.util.Locale.ROOT)).toSet

  val SHUFFLE_KUDO_SERIALIZER_DEBUG_MODE = conf("spark.rapids.shuffle.kudo.serializer.debug.mode")
    .doc("Debug mode for Kudo serializer for the shuffle. If Always, it will dump the " +
      "kudo tables to a file in spark.rapids.shuffle.kudo.serializer.debug.dump.path.prefix. " +
      "If Never, it will not dump the kudo tables data. If OnFailure, it will only dump the " +
      "kudo tables data when the shuffle fails.")
    .internal()
    .startupOnly()
    .stringConf
    .transform(_.toUpperCase(java.util.Locale.ROOT))
    .checkValues(KudoDebugModes)
    .createWithDefault("NEVER")

  val SHUFFLE_KUDO_SERIALIZER_DEBUG_DUMP_PREFIX =
    conf("spark.rapids.shuffle.kudo.serializer.debug.dump.path.prefix")
    .doc("The path prefix to use for the kudo tables when using Kudo serializer for the shuffle.")
    .internal()
    .startupOnly()
    .stringConf
    .createOptional

}
