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

private[rapids] trait RapidsConfSqlEntries extends RapidsConfResourceEntries {

  // ENABLE/DISABLE PROCESSING

  val SQL_ENABLED = conf("spark.rapids.sql.enabled")
    .doc("Enable (true) or disable (false) sql operations on the GPU")
    .commonlyUsed()
    .booleanConf
    .createWithDefault(true)

  val SQL_MODE = conf("spark.rapids.sql.mode")
    .doc("Set the mode for the cuDF plugin. The supported modes are explainOnly and " +
         "executeOnGPU. This config can not be changed at runtime, you must restart the " +
         "application for it to take affect. The default mode is executeOnGPU, which means " +
         "the cuDF plugin converts Spark operations and executes them on the " +
         "GPU when possible. The explainOnly mode allows running queries on the CPU and the " +
         "cuDF plugin will evaluate the queries as if it was going to run on the GPU. " +
         "The explanations of what would have run on the GPU and why are output in log " +
         "messages. When using explainOnly mode, the default explain output is ALL, this can " +
         "be changed by setting spark.rapids.sql.explain. See that config for more details.")
    .startupOnly()
    .stringConf
    .transform(_.toLowerCase(java.util.Locale.ROOT))
    .checkValues(Set("explainonly", "executeongpu"))
    .createWithDefault("executeongpu")

  val UDF_COMPILER_ENABLED = conf("spark.rapids.sql.udfCompiler.enabled")
    .doc("When set to true, Scala UDFs will be considered for compilation as Catalyst expressions")
    .commonlyUsed()
    .booleanConf
    .createWithDefault(false)

  val DFUDF_ENABLED = conf("spark.rapids.sql.dfudf.enabled")
    .doc("When set to false, the DataFrame UDF plugin is disabled. True enables it.")
    .internal()
    .booleanConf
    .createWithDefault(true)

  val INCOMPATIBLE_OPS = conf("spark.rapids.sql.incompatibleOps.enabled")
    .doc("For operations that work, but are not 100% compatible with the Spark equivalent " +
      "set if they should be enabled by default or disabled by default.")
    .booleanConf
    .createWithDefault(true)

  val INCOMPATIBLE_DATE_FORMATS = conf("spark.rapids.sql.incompatibleDateFormats.enabled")
    .doc("When parsing strings as dates and timestamps in functions like unix_timestamp, some " +
         "formats are fully supported on the GPU and some are unsupported and will fall back to " +
         "the CPU.  Some formats behave differently on the GPU than the CPU.  Spark on the CPU " +
         "interprets date formats with unsupported trailing characters as nulls, while Spark on " +
         "the GPU will parse the date with invalid trailing characters. More detail can be found " +
         "at [parsing strings as dates or timestamps]" +
         "(../compatibility.md#parsing-strings-as-dates-or-timestamps).")
      .booleanConf
      .createWithDefault(false)

  val IMPROVED_FLOAT_OPS = conf("spark.rapids.sql.improvedFloatOps.enabled")
    .doc("For some floating point operations spark uses one way to compute the value " +
      "and the underlying cudf implementation can use an improved algorithm. " +
      "In some cases this can result in cudf producing an answer when spark overflows.")
    .booleanConf
    .createWithDefault(true)

  val NEED_DECIMAL_OVERFLOW_GUARANTEES = conf("spark.rapids.sql.decimalOverflowGuarantees")
      .doc("FOR TESTING ONLY. DO NOT USE IN PRODUCTION. Please see the decimal section of " +
          "the compatibility documents for more information on this config.")
      .booleanConf
      .createWithDefault(true)

  val ENABLE_WINDOW_COLLECT_LIST = conf("spark.rapids.sql.window.collectList.enabled")
      .doc("The output size of collect list for a window operation is proportional to " +
          "the window size squared. The current GPU implementation does not handle this well " +
          "and is disabled by default. If you know that your window size is very small you " +
          "can try to enable it.")
      .booleanConf
      .createWithDefault(false)

  val ENABLE_WINDOW_COLLECT_SET = conf("spark.rapids.sql.window.collectSet.enabled")
      .doc("The output size of collect set for a window operation can be proportional to " +
          "the window size squared. The current GPU implementation does not handle this well " +
          "and is disabled by default. If you know that your window size is very small you " +
          "can try to enable it.")
      .booleanConf
      .createWithDefault(false)

  val ENABLE_WINDOW_UNBOUNDED_AGG = conf("spark.rapids.sql.window.unboundedAgg.enabled")
      .doc("This is a temporary internal config to turn on an unbounded to unbounded " +
          "window optimization that is still a work in progress. It should eventually replace " +
          "the double pass window exec.")
      .internal()
      .booleanConf
      .createWithDefault(false)

  val ENABLE_WINDOW_GROUP_LIMIT_OPT = conf("spark.rapids.sql.window.groupLimit.opt.enabled")
      .doc("When enabled, the plugin will skip redundant Final WindowGroupLimit operators " +
          "when they are followed by a WindowExec and FilterExec that perform the same " +
          "rank computation and filtering. This avoids computing the rank twice and " +
          "improves performance for queries with rank-based window limits.")
      .internal()
      .booleanConf
      .createWithDefault(true)

  val ENABLE_FLOAT_AGG = conf("spark.rapids.sql.variableFloatAgg.enabled")
    .doc("Spark assumes that all operations produce the exact same result each time. " +
      "This is not true for some floating point aggregations, which can produce slightly " +
      "different results on the GPU as the aggregation is done in parallel.  This can enable " +
      "those operations if you know the query is only computing it once.")
    .booleanConf
    .createWithDefault(true)

  val ENABLE_REPLACE_SORTMERGEJOIN = conf("spark.rapids.sql.replaceSortMergeJoin.enabled")
    .doc("Allow replacing sortMergeJoin with HashJoin")
    .booleanConf
    .createWithDefault(true)

  val ENABLE_HASH_OPTIMIZE_SORT = conf("spark.rapids.sql.hashOptimizeSort.enabled")
    .doc("Whether sorts should be inserted after some hashed operations to improve " +
      "output ordering. This can improve output file sizes when saving to columnar formats.")
    .booleanConf
    .createWithDefault(false)

  val ENABLE_FOLD_LOCAL_AGGREGATE = conf("spark.rapids.sql.foldLocalAggregate.enabled")
    .doc("Whether to fold two-stages local aggregate into one-shot Complete aggregate.")
    .internal()
    .booleanConf
    .createWithDefault(true)

  val ENABLE_CAST_FLOAT_TO_DECIMAL = conf("spark.rapids.sql.castFloatToDecimal.enabled")
    .doc("Casting from floating point types to decimal on the GPU returns results that have " +
      "tiny difference compared to results returned from CPU.")
    .booleanConf
    .createWithDefault(true)

  val ENABLE_CAST_FLOAT_TO_STRING = conf("spark.rapids.sql.castFloatToString.enabled")
    .doc("Casting from floating point types to string on the GPU returns results that have " +
      "a different precision than the default results of Spark.")
    .booleanConf
    .createWithDefault(true)

  val ENABLE_FLOAT_FORMAT_NUMBER = conf("spark.rapids.sql.formatNumberFloat.enabled")
    .doc("format_number with floating point types on the GPU returns results that have " +
      "a different precision than the default results of Spark.")
    .booleanConf
    .createWithDefault(true)

  val ENABLE_CAST_FLOAT_TO_INTEGRAL_TYPES =
    conf("spark.rapids.sql.castFloatToIntegralTypes.enabled")
      .doc("Casting from floating point types to integral types on the GPU supports a " +
          "slightly different range of values when using Spark 3.1.0 or later. Refer to the CAST " +
          "documentation for more details.")
      .booleanConf
      .createWithDefault(true)

  val ENABLE_CAST_DECIMAL_TO_FLOAT = conf("spark.rapids.sql.castDecimalToFloat.enabled")
      .doc("Casting from decimal to floating point types on the GPU returns results that have " +
          "tiny difference compared to results returned from CPU.")
      .booleanConf
      .createWithDefault(true)

  val ENABLE_CAST_STRING_TO_FLOAT = conf("spark.rapids.sql.castStringToFloat.enabled")
    .doc("When set to true, enables casting from strings to float types (float, double) " +
      "on the GPU. Currently hex values aren't supported on the GPU. Also note that casting from " +
      "string to float types on the GPU returns incorrect results when the string represents any " +
      "number \"1.7976931348623158E308\" <= x < \"1.7976931348623159E308\" " +
      "and \"-1.7976931348623158E308\" >= x > \"-1.7976931348623159E308\" in both these cases " +
      "the GPU returns Double.MaxValue while CPU returns \"+Infinity\" and \"-Infinity\" " +
      "respectively")
    .booleanConf
    .createWithDefault(true)

  val ENABLE_CAST_STRING_TO_TIMESTAMP = conf("spark.rapids.sql.castStringToTimestamp.enabled")
    .doc("When set to false, this disables casting from string to timestamp on the GPU.")
    .booleanConf
    .createWithDefault(true)

  val HAS_EXTENDED_YEAR_VALUES = conf("spark.rapids.sql.hasExtendedYearValues")
      .doc("Spark 3.2.0+ extended parsing of years in dates and " +
          "timestamps to support the full range of possible values. Prior " +
          "to this it was limited to a positive 4 digit year. The Accelerator does not " +
          "support the extended range yet. This config indicates if your data includes " +
          "this extended range or not, or if you don't care about getting the correct " +
          "values on values with the extended range.")
      .booleanConf
      .createWithDefault(true)

  val ENABLE_INNER_JOIN = conf("spark.rapids.sql.join.inner.enabled")
      .doc("When set to true inner joins are enabled on the GPU")
      .booleanConf
      .createWithDefault(true)

  val ENABLE_CROSS_JOIN = conf("spark.rapids.sql.join.cross.enabled")
      .doc("When set to true cross joins are enabled on the GPU")
      .booleanConf
      .createWithDefault(true)

  val ENABLE_LEFT_OUTER_JOIN = conf("spark.rapids.sql.join.leftOuter.enabled")
      .doc("When set to true left outer joins are enabled on the GPU")
      .booleanConf
      .createWithDefault(true)

  val ENABLE_RIGHT_OUTER_JOIN = conf("spark.rapids.sql.join.rightOuter.enabled")
      .doc("When set to true right outer joins are enabled on the GPU")
      .booleanConf
      .createWithDefault(true)

  val ENABLE_FULL_OUTER_JOIN = conf("spark.rapids.sql.join.fullOuter.enabled")
      .doc("When set to true full outer joins are enabled on the GPU")
      .booleanConf
      .createWithDefault(true)

  val ENABLE_LEFT_SEMI_JOIN = conf("spark.rapids.sql.join.leftSemi.enabled")
      .doc("When set to true left semi joins are enabled on the GPU")
      .booleanConf
      .createWithDefault(true)

  val ENABLE_LEFT_ANTI_JOIN = conf("spark.rapids.sql.join.leftAnti.enabled")
      .doc("When set to true left anti joins are enabled on the GPU")
      .booleanConf
      .createWithDefault(true)

  val ENABLE_EXISTENCE_JOIN = conf("spark.rapids.sql.join.existence.enabled")
      .doc("When set to true existence joins are enabled on the GPU")
      .booleanConf
      .createWithDefault(true)

  val ENABLE_PROJECT_AST = conf("spark.rapids.sql.projectAstEnabled")
      .doc("Enable project operations to use cudf AST expressions when possible.")
      .internal()
      .booleanConf
      .createWithDefault(false)

  val ENABLE_TIERED_PROJECT = conf("spark.rapids.sql.tiered.project.enabled")
      .doc("Enable tiered projections.")
      .internal()
      .booleanConf
      .createWithDefault(true)

  val ENABLE_COMBINED_EXPR_PREFIX = "spark.rapids.sql.expression.combined."

  val ENABLE_COMBINED_EXPRESSIONS = conf("spark.rapids.sql.combined.expressions.enabled")
    .doc("For some expressions it can be much more efficient to combine multiple " +
      "expressions together into a single kernel call. This enables or disables that " +
      s"feature. Note that this requires that $ENABLE_TIERED_PROJECT is turned on or " +
      "else there is no performance improvement. You can also disable this feature for " +
      "expressions that support it. Each config is expression specific and starts with " +
      s"$ENABLE_COMBINED_EXPR_PREFIX followed by the name of the GPU expression class " +
      s"similar to what we do for enabling converting individual expressions to the GPU.")
    .internal()
    .booleanConf
    .createWithDefault(true)

  val ENABLE_RLIKE_REGEX_REWRITE = conf("spark.rapids.sql.rLikeRegexRewrite.enabled")
      .doc("Enable the optimization to rewrite rlike regex to contains in some cases.")
      .internal()
      .booleanConf
      .createWithDefault(true)

  // FILE FORMATS
  val MULTITHREAD_READ_NUM_THREADS = conf("spark.rapids.sql.multiThreadedRead.numThreads")
      .doc("The maximum number of threads on each executor to use for reading small " +
        "files in parallel. This can not be changed at runtime after the executor has " +
        "started. Used with COALESCING and MULTITHREADED readers, see " +
        "spark.rapids.sql.format.parquet.reader.type, " +
        "spark.rapids.sql.format.orc.reader.type, or " +
        "spark.rapids.sql.format.avro.reader.type for a discussion of reader types. " +
        "If it is not set explicitly and spark.executor.cores is set, it will be tried to " +
        "assign value of `max(MULTITHREAD_READ_NUM_THREADS_DEFAULT, spark.executor.cores)`, " +
        s"where MULTITHREAD_READ_NUM_THREADS_DEFAULT = $MULTITHREAD_READ_NUM_THREADS_DEFAULT" +
        ".")
      .startupOnly()
      .commonlyUsed()
      .integerConf
      .checkValue(v => v > 0, "The thread count must be greater than zero.")
      .createWithDefault(MULTITHREAD_READ_NUM_THREADS_DEFAULT)

  // Memory-bounded multithreaded reading
  val MULTITHREAD_READ_MEMORY_LIMIT_ENABLED = {
    // TODO: This conf should be adjusted during the runtime
    conf("spark.rapids.sql.multiThreadedRead.memoryLimit.enabled")
        .doc("Enable memory-bounded multi-threaded reading. When enabled, the concurrency of " +
            "multi-threaded reading will be dynamically controlled based on the available memory " +
            s"budget per Spark executor.")
        .startupOnly()
        .internal()
        .booleanConf
        .createWithDefault(false)
  }

  val MULTITHREAD_READ_MEMORY_LIMIT_SIZE = {
    // TODO: This conf should be adjusted during the runtime
    conf("spark.rapids.sql.multiThreadedRead.memoryLimit.size")
        .doc("The maximum memory capacity in bytes to use for reading files in parallel. " +
            "This can not be changed at runtime after the executor has started. And if 0, it " +
            "will be set with 90% of executor's off-heap memory. This config only takes effect " +
            s"when ${MULTITHREAD_READ_MEMORY_LIMIT_ENABLED.key} is enabled.")
        .startupOnly()
        .internal()
        .bytesConf(ByteUnit.BYTE)
        .checkValue(v => v >= 0, s"The memory capacity must be greatThanOrEqual zero")
        .createWithDefault(0)
  }

  val MULTITHREAD_READ_MEMORY_LIMIT_ACQUISITION_TIMEOUT  = {
    conf("spark.rapids.sql.multiThreadedRead.memoryLimit.acquisitionTimeout")
        .doc("The maximum time in milliseconds to wait for a task to acquire required resource " +
            "before giving up and put off the run, by re-appending the task back into the queue " +
            "with a priority penality.")
        .startupOnly()
        .internal()
        .longConf
        .checkValue(v => v >= 0, "The timeout must be greater than zero")
        .createWithDefault(30 * 1000L)
  } // 30 seconds

  val MULTITHREAD_READ_MEMORY_LIMIT_TEST_PER_STAGE_POOL = {
    conf("spark.rapids.sql.multiThreadedRead.memoryLimit.tests.perStagePool")
        .doc("Enable test mode for the multi-threaded read. This will create different " +
            "threadpools for each stage, so as to verify threadpools with different configs " +
            "as independent test cases which can be run in parallel.")
        .startupOnly()
        .internal()
        .booleanConf
        .createWithDefault(false)
  }
}
