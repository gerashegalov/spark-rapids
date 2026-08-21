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

package com.nvidia.spark.rapids

import com.nvidia.spark.rapids.GpuOverrides._
import com.nvidia.spark.rapids.shims._
import com.nvidia.spark.rapids.window.GpuWindowExecMeta

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.adaptive._
import org.apache.spark.sql.execution.aggregate._
import org.apache.spark.sql.execution.columnar.InMemoryTableScanExec
import org.apache.spark.sql.execution.command.{DataWritingCommandExec, ExecutedCommandExec}
import org.apache.spark.sql.execution.datasources.v2._
import org.apache.spark.sql.execution.exchange._
import org.apache.spark.sql.execution.joins._
import org.apache.spark.sql.execution.python._
import org.apache.spark.sql.execution.window.WindowExec
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids._
import org.apache.spark.sql.rapids.execution._
import org.apache.spark.sql.rapids.execution.python._
import org.apache.spark.sql.rapids.shims.GpuMapInPandasExecMeta

private[rapids] object GpuExecOverrides {
  val rules: Seq[ExecRule[_ <: SparkPlan]] = Seq(
    exec[GenerateExec] (
      "The backend for operations that generate more output rows than input rows like explode",
      ExecChecks(
        (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 + TypeSig.BINARY +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP).nested(),
        TypeSig.all),
      (gen, conf, p, r) => new GpuGenerateExecSparkPlanMeta(gen, conf, p, r)),
    exec[ProjectExec](
      "The backend for most select, withColumn and dropColumn statements",
      ExecChecks(
        (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.STRUCT + TypeSig.MAP +
            TypeSig.ARRAY + TypeSig.DECIMAL_128 + TypeSig.BINARY +
            GpuTypeShims.additionalCommonOperatorSupportedTypes).nested(),
        TypeSig.all),
      (proj, conf, p, r) => new GpuProjectExecMeta(proj, conf, p, r)),
    exec[RangeExec](
      "The backend for range operator",
      ExecChecks(TypeSig.LONG, TypeSig.LONG),
      (range, conf, p, r) => {
        new SparkPlanMeta[RangeExec](range, conf, p, r) {
          override def convertToGpu(): GpuExec =
            GpuRangeExec(range.start, range.end, range.step, range.numSlices, range.output,
              this.conf.gpuTargetBatchSizeBytes)
        }
      }),
    exec[BatchScanExec](
      "The backend for most file input",
      ExecChecks(
        (TypeSig.commonCudfTypes + TypeSig.STRUCT + TypeSig.MAP + TypeSig.ARRAY +
          TypeSig.DECIMAL_128 + TypeSig.BINARY).nested(),
        TypeSig.all),
      (p, conf, parent, r) => new BatchScanExecMeta(p, conf, parent, r)),
    exec[CoalesceExec](
      "The backend for the dataframe coalesce method",
      ExecChecks((gpuCommonTypes + TypeSig.STRUCT + TypeSig.ARRAY +
          TypeSig.MAP + TypeSig.BINARY + GpuTypeShims.additionalArithmeticSupportedTypes).nested(),
        TypeSig.all),
      (coalesce, conf, parent, r) => new SparkPlanMeta[CoalesceExec](coalesce, conf, parent, r) {
        override def convertToGpu(): GpuExec =
          GpuCoalesceExec(coalesce.numPartitions, childPlans.head.convertIfNeeded())
      }),
    exec[DataWritingCommandExec](
      "Writing data",
      ExecChecks((TypeSig.commonCudfTypes + TypeSig.DECIMAL_128.withPsNote(
          TypeEnum.DECIMAL, "128bit decimal only supported for Orc and Parquet") +
          TypeSig.STRUCT + TypeSig.MAP + TypeSig.ARRAY + TypeSig.BINARY +
          GpuTypeShims.additionalCommonOperatorSupportedTypes).nested(),
        TypeSig.all),
      (p, conf, parent, r) => new SparkPlanMeta[DataWritingCommandExec](p, conf, parent, r) {
        override val childDataWriteCmds: scala.Seq[DataWritingCommandMeta[_]] =
          Seq(GpuOverrides.wrapDataWriteCmds(p.cmd, this.conf, Some(this)))

        override def convertToGpu(): GpuExec =
          GpuDataWritingCommandExec(childDataWriteCmds.head.convertToGpu(),
            childPlans.head.convertIfNeeded())
      }),
    exec[ExecutedCommandExec](
      "Eagerly executed commands",
      ExecChecks(TypeSig.all, TypeSig.all),
      (p, conf, parent, r) => new ExecutedCommandExecMeta(p, conf, parent, r)),
    exec[TakeOrderedAndProjectExec](
      "Take the first limit elements as defined by the sortOrder, and do projection if needed",
      // The SortOrder TypeSig will govern what types can actually be used as sorting key data type.
      // The types below are allowed as inputs and outputs.
      ExecChecks((pluginSupportedOrderableSig +
          TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP).nested(), TypeSig.all),
      GpuTakeOrderedAndProjectExecMeta),
    exec[LocalLimitExec](
      "Per-partition limiting of results",
      ExecChecks((TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
          TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP).nested(),
        TypeSig.all),
      (localLimitExec, conf, p, r) =>
        new SparkPlanMeta[LocalLimitExec](localLimitExec, conf, p, r) {
          override def convertToGpu(): GpuExec =
            GpuLocalLimitExec(localLimitExec.limit, childPlans.head.convertIfNeeded())
        }),
    exec[GlobalLimitExec](
      "Limiting of results across partitions",
      ExecChecks((TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
          TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP).nested(),
        TypeSig.all),
      (globalLimitExec, conf, p, r) =>
        new SparkPlanMeta[GlobalLimitExec](globalLimitExec, conf, p, r) {
          override def convertToGpu(): GpuExec =
            GpuGlobalLimitExec(globalLimitExec.limit, childPlans.head.convertIfNeeded(), 0)
        }),
    exec[CollectLimitExec](
      "Reduce to single partition and apply limit",
      ExecChecks((TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
          TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP).nested(),
        TypeSig.all),
      (collectLimitExec, conf, p, r) => new GpuCollectLimitMeta(collectLimitExec, conf, p, r))
        .disabledByDefault("Collect Limit replacement can be slower on the GPU, if huge number " +
            "of rows in a batch it could help by limiting the number of rows transferred from " +
            "GPU to CPU"),
    exec[FilterExec](
      "The backend for most filter statements",
      ExecChecks((TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.STRUCT + TypeSig.MAP +
          TypeSig.ARRAY + TypeSig.DECIMAL_128 + TypeSig.BINARY +
          GpuTypeShims.additionalCommonOperatorSupportedTypes).nested(), TypeSig.all),
      GpuFilterExecMeta),
    exec[ShuffleExchangeExec](
      "The backend for most data being exchanged between processes",
      ExecChecks((TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 + TypeSig.BINARY +
          TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP +
          GpuTypeShims.additionalArithmeticSupportedTypes).nested()
          .withPsNote(TypeEnum.STRUCT, "Round-robin partitioning is not supported for nested " +
              s"structs if ${SQLConf.SORT_BEFORE_REPARTITION.key} is true")
          .withPsNote(
            Seq(TypeEnum.MAP),
            "Round-robin partitioning is not supported if " +
              s"${SQLConf.SORT_BEFORE_REPARTITION.key} is true"),
        TypeSig.all),
      (shuffle, conf, p, r) => new GpuShuffleMeta(shuffle, conf, p, r)),
    exec[UnionExec](
      "The backend for the union operator",
      ExecChecks((TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
          TypeSig.MAP + TypeSig.ARRAY + TypeSig.STRUCT).nested()
        .withPsNote(TypeEnum.STRUCT,
          "unionByName will not optionally impute nulls for missing struct fields " +
          "when the column is a struct and there are non-overlapping fields"), TypeSig.all),
      (union, conf, p, r) => new SparkPlanMeta[UnionExec](union, conf, p, r) {
        override def convertToGpu(): GpuExec =
          GpuUnionExec(childPlans.map(_.convertIfNeeded()))
      }),
    exec[BroadcastExchangeExec](
      "The backend for broadcast exchange of data",
      ExecChecks((TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 + TypeSig.BINARY +
          TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP).nested(TypeSig.commonCudfTypes +
          TypeSig.NULL + TypeSig.DECIMAL_128 + TypeSig.STRUCT),
        TypeSig.all),
      (exchange, conf, p, r) => new GpuBroadcastMeta(exchange, conf, p, r)),
    exec[BroadcastHashJoinExec](
      "Implementation of join using broadcast data",
      JoinTypeChecks.equiJoinExecChecks,
      (join, conf, p, r) => new GpuBroadcastHashJoinMeta(join, conf, p, r)),
    exec[BroadcastNestedLoopJoinExec](
      "Implementation of join using brute force. Full outer joins and joins where the " +
          "broadcast side matches the join side (e.g.: LeftOuter with left broadcast) are not " +
          "supported",
      JoinTypeChecks.nonEquiJoinChecks,
      (join, conf, p, r) => new GpuBroadcastNestedLoopJoinMeta(join, conf, p, r)),
    exec[CartesianProductExec](
      "Implementation of join using brute force",
      ExecChecks((TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 + TypeSig.BINARY +
          TypeSig.ARRAY + TypeSig.MAP + TypeSig.STRUCT)
          .nested(TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 + TypeSig.BINARY +
              TypeSig.ARRAY + TypeSig.MAP + TypeSig.STRUCT),
        TypeSig.all),
      (join, conf, p, r) => new SparkPlanMeta[CartesianProductExec](join, conf, p, r) {
        val condition: Option[BaseExprMeta[_]] =
          join.condition.map(GpuOverrides.wrapExpr(_, this.conf, Some(this)))

        override val childExprs: Seq[BaseExprMeta[_]] = condition.toSeq

        override def convertToGpu(): GpuExec = {
          val Seq(left, right) = childPlans.map(_.convertIfNeeded())
          val joinExec = GpuCartesianProductExec(
            left,
            right,
            None,
            this.conf.gpuTargetBatchSizeBytes)
          // The GPU does not yet support conditional joins, so conditions are implemented
          // as a filter after the join when possible.
          condition.map(c => GpuFilterExec(c.convertToGpu(),
            joinExec)()).getOrElse(joinExec)
        }
      }),
    exec[HashAggregateExec](
      "The backend for hash based aggregations",
      ExecChecks(
        (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 + TypeSig.BINARY +
          TypeSig.MAP + TypeSig.ARRAY + TypeSig.STRUCT)
            .nested()
            .withPsNote(TypeEnum.MAP,
              "not allowed for grouping expressions")
            .withPsNote(TypeEnum.ARRAY,
              "not allowed for grouping expressions if containing Struct as child")
            .withPsNote(TypeEnum.STRUCT,
              "not allowed for grouping expressions if containing Array, Map, or Binary as child"),
        TypeSig.all),
      (agg, conf, p, r) => new GpuHashAggregateMeta(agg, conf, p, r)),
    exec[ObjectHashAggregateExec](
      "The backend for hash based aggregations supporting TypedImperativeAggregate functions",
      ExecChecks(
        // note that binary input is allowed here but there are additional checks later on to
        // check that we have can support binary in the context of aggregate buffer conversions
        (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
          TypeSig.MAP + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.BINARY)
            .nested()
            .withPsNote(TypeEnum.BINARY, "not allowed for grouping expressions and " +
              "only allowed when aggregate buffers can be converted between CPU and GPU")
            .withPsNote(Seq(TypeEnum.ARRAY, TypeEnum.MAP),
              "not allowed for grouping expressions")
            .withPsNote(TypeEnum.STRUCT,
              "not allowed for grouping expressions if containing Array, Map, or Binary as child"),
        TypeSig.all),
      (agg, conf, p, r) => new GpuObjectHashAggregateExecMeta(agg, conf, p, r)),
    exec[ShuffledHashJoinExec](
      "Implementation of join using hashed shuffled data",
      JoinTypeChecks.equiJoinExecChecks,
      (join, conf, p, r) => new GpuShuffledHashJoinMeta(join, conf, p, r)),
    exec[SortAggregateExec](
      "The backend for sort based aggregations",
      ExecChecks(
        (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            TypeSig.MAP + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.BINARY)
            .nested()
            .withPsNote(Seq(TypeEnum.ARRAY, TypeEnum.MAP),
              "not allowed for grouping expressions")
            .withPsNote(TypeEnum.STRUCT,
              "not allowed for grouping expressions if containing Array, Map, or Binary as child"),
        TypeSig.all),
      (agg, conf, p, r) => new GpuSortAggregateExecMeta(agg, conf, p, r)),
    exec[SortExec](
      "The backend for the sort operator",
      // The SortOrder TypeSig will govern what types can actually be used as sorting key data type.
      // The types below are allowed as inputs and outputs.
      ExecChecks((pluginSupportedOrderableSig + TypeSig.ARRAY +
          TypeSig.STRUCT +TypeSig.MAP + TypeSig.BINARY).nested(), TypeSig.all),
      (sort, conf, p, r) => new GpuSortMeta(sort, conf, p, r)),
    exec[SortMergeJoinExec](
      "Sort merge join, replacing with shuffled hash join",
      JoinTypeChecks.equiJoinExecChecks,
      (join, conf, p, r) => new GpuSortMergeJoinMeta(join, conf, p, r)),
    exec[ExpandExec](
      "The backend for the expand operator",
      ExecChecks(
        (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP + TypeSig.BINARY).nested(),
        TypeSig.all),
      (expand, conf, p, r) => new GpuExpandExecMeta(expand, conf, p, r)),
    exec[WindowExec](
      "Window-operator backend",
      ExecChecks((TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
          TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP + TypeSig.BINARY).nested(),
        TypeSig.all,
        Map("partitionSpec" ->
            InputCheck(
                TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
                TypeSig.STRUCT.nested(TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128),
            TypeSig.all))),
      (windowOp, conf, p, r) =>
        new GpuWindowExecMeta(windowOp, conf, p, r)
    ),
    exec[SampleExec](
      "The backend for the sample operator",
      ExecChecks((TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.STRUCT + TypeSig.MAP +
        TypeSig.ARRAY + TypeSig.DECIMAL_128 + GpuTypeShims.additionalCommonOperatorSupportedTypes)
          .nested(), TypeSig.all),
      (sample, conf, p, r) => new GpuSampleExecMeta(sample, conf, p, r)
    ),
    exec[SubqueryBroadcastExec](
      "Plan to collect and transform the broadcast key values",
      ExecChecks(TypeSig.all, TypeSig.all),
      (s, conf, p, r) => new GpuSubqueryBroadcastMeta(s, conf, p, r)
    ),
    SparkShimImpl.aqeShuffleReaderExec,
    // AggregateInPandasExec renamed to ArrowAggregatePythonExec in Spark 4.1.0
    AggregateInPandasExecShims.execRule.orNull,
    exec[ArrowEvalPythonExec](
      "The backend of the Scalar Pandas UDFs. Accelerates the data transfer between the" +
        " Java process and the Python process. It also supports scheduling GPU resources" +
        " for the Python process when enabled",
      ExecChecks(
        (TypeSig.commonCudfTypes + TypeSig.ARRAY + TypeSig.STRUCT).nested(),
        TypeSig.all),
      (e, conf, p, r) =>
        new SparkPlanMeta[ArrowEvalPythonExec](e, conf, p, r) {
          val udfs: Seq[BaseExprMeta[PythonUDF]] =
            e.udfs.map(GpuOverrides.wrapExpr(_, this.conf, Some(this)))
          val resultAttrs: Seq[BaseExprMeta[Attribute]] =
            e.resultAttrs.map(GpuOverrides.wrapExpr(_, this.conf, Some(this)))
          override val childExprs: Seq[BaseExprMeta[_]] = udfs ++ resultAttrs

          override def replaceMessage: String = "partially run on GPU"
          override def noReplacementPossibleMessage(reasons: String): String =
            s"cannot run even partially on the GPU because $reasons"

          override def convertToGpu(): GpuExec =
            GpuArrowEvalPythonExec(udfs.map(_.convertToGpu()).asInstanceOf[Seq[GpuPythonUDF]],
              resultAttrs.map(_.convertToGpu()).asInstanceOf[Seq[Attribute]],
              childPlans.head.convertIfNeeded(),
              e.evalType)
        }),
    exec[FlatMapCoGroupsInPandasExec](
      "The backend for CoGrouped Aggregation Pandas UDF. Accelerates the data transfer" +
        " between the Java process and the Python process. It also supports scheduling GPU" +
        " resources for the Python process when enabled.",
      ExecChecks(TypeSig.commonCudfTypes, TypeSig.all),
      (flatCoPy, conf, p, r) => new GpuFlatMapCoGroupsInPandasExecMeta(flatCoPy, conf, p, r))
        .disabledByDefault("Performance is not ideal with many small groups"),
    exec[FlatMapGroupsInPandasExec](
      "The backend for Flat Map Groups Pandas UDF, Accelerates the data transfer between the" +
        " Java process and the Python process. It also supports scheduling GPU resources" +
        " for the Python process when enabled.",
      ExecChecks(TypeSig.commonCudfTypes, TypeSig.all),
      (flatPy, conf, p, r) => new GpuFlatMapGroupsInPandasExecMeta(flatPy, conf, p, r)),
    exec[MapInPandasExec](
      "The backend for Map Pandas Iterator UDF. Accelerates the data transfer between the" +
        " Java process and the Python process. It also supports scheduling GPU resources" +
        " for the Python process when enabled.",
      ExecChecks((TypeSig.commonCudfTypes + TypeSig.ARRAY + TypeSig.STRUCT).nested(),
        TypeSig.all),
      (mapPy, conf, p, r) => new GpuMapInPandasExecMeta(mapPy, conf, p, r)),
    exec[InMemoryTableScanExec](
      "Implementation of InMemoryTableScanExec to use GPU accelerated caching",
      ExecChecks((TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 + TypeSig.STRUCT +
          TypeSig.ARRAY + TypeSig.MAP + GpuTypeShims.additionalCommonOperatorSupportedTypes)
          .nested(), TypeSig.all),
      (scan, conf, p, r) => new InMemoryTableScanMeta(scan, conf, p, r)),
    neverReplaceExec[AlterNamespaceSetPropertiesExec]("Namespace metadata operation"),
    neverReplaceExec[CreateNamespaceExec]("Namespace metadata operation"),
    neverReplaceExec[DescribeNamespaceExec]("Namespace metadata operation"),
    neverReplaceExec[DropNamespaceExec]("Namespace metadata operation"),
    neverReplaceExec[SetCatalogAndNamespaceExec]("Namespace metadata operation"),
    SparkShimImpl.neverReplaceShowCurrentNamespaceCommand,
    ShowNamespacesExecShims.neverReplaceExec.orNull,
    neverReplaceExec[AlterTableExec]("Table metadata operation"),
    neverReplaceExec[CreateTableExec]("Table metadata operation"),
    neverReplaceExec[DeleteFromTableExec]("Table metadata operation"),
    neverReplaceExec[DescribeTableExec]("Table metadata operation"),
    neverReplaceExec[DropTableExec]("Table metadata operation"),
    neverReplaceExec[AtomicReplaceTableExec]("Table metadata operation"),
    neverReplaceExec[RefreshTableExec]("Table metadata operation"),
    neverReplaceExec[RenameTableExec]("Table metadata operation"),
    neverReplaceExec[ReplaceTableExec]("Table metadata operation"),
    neverReplaceExec[ShowTablePropertiesExec]("Table metadata operation"),
    neverReplaceExec[ShowTablesExec]("Table metadata operation"),
    neverReplaceExec[AdaptiveSparkPlanExec]("Wrapper for adaptive query plan"),
    neverReplaceExec[BroadcastQueryStageExec]("Broadcast query stage"),
    neverReplaceExec[ShuffleQueryStageExec]("Shuffle query stage")
  )
}
