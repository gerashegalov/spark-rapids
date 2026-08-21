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
import com.nvidia.spark.rapids.window._

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.rapids._

private[rapids] object GpuCoreExpressionOverrides {
  val rules: Seq[ExprRule[_ <: Expression]] = Seq(
    expr[Literal](
      "Holds a static value from the query",
      ExprChecks.projectAndAst(
        TypeSig.astTypes,
        (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 + TypeSig.CALENDAR
          + TypeSig.BINARY + TypeSig.ARRAY + TypeSig.MAP + TypeSig.STRUCT + TypeSig.ansiIntervals)
          .nested(TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 + TypeSig.BINARY +
            TypeSig.ARRAY + TypeSig.MAP + TypeSig.STRUCT),
        TypeSig.all),
      (lit, conf, p, r) => new LiteralExprMeta(lit, conf, p, r)),
    expr[Signum](
      "Returns -1.0, 0.0 or 1.0 as expr is negative, 0 or positive",
      ExprChecks.mathUnary,
      (a, conf, p, r) => new UnaryExprMeta[Signum](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuSignum(child)
      }),
    expr[Alias](
      "Gives a column a name",
      ExprChecks.unaryProjectAndAstInputMatchesOutput(
        TypeSig.astTypes + GpuTypeShims.additionalCommonOperatorSupportedTypes,
        (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.MAP + TypeSig.ARRAY + TypeSig.STRUCT
            + TypeSig.DECIMAL_128 + TypeSig.BINARY
            + GpuTypeShims.additionalCommonOperatorSupportedTypes).nested(),
        TypeSig.all),
      (a, conf, p, r) => new UnaryAstExprMeta[Alias](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuAlias(child, a.name)(a.exprId, a.qualifier, a.explicitMetadata)
      }),
    expr[BoundReference](
      "Reference to a bound variable",
      ExprChecks.projectAndAst(
        TypeSig.astTypes + GpuTypeShims.additionalCommonOperatorSupportedTypes,
        (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.MAP + TypeSig.ARRAY + TypeSig.STRUCT +
          TypeSig.DECIMAL_128 + TypeSig.BINARY +
          GpuTypeShims.additionalCommonOperatorSupportedTypes).nested(),
        TypeSig.all),
      (currentRow, conf, p, r) => new ExprMeta[BoundReference](currentRow, conf, p, r) {
        // BoundReference should not be directly wrapped in a bridge (unit test compatibility)
        override def isBridgeCompatible: Boolean = false

        override def convertToGpuImpl(): GpuExpression = GpuBoundReference(
          currentRow.ordinal, currentRow.dataType, currentRow.nullable)(
          NamedExpression.newExprId, "")
      }),
    expr[AttributeReference](
      "References an input column",
      ExprChecks.projectAndAst(
        TypeSig.astTypes + GpuTypeShims.additionalCommonOperatorSupportedTypes,
        (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.MAP + TypeSig.ARRAY +
            TypeSig.STRUCT + TypeSig.DECIMAL_128 + TypeSig.BINARY +
            GpuTypeShims.additionalCommonOperatorSupportedTypes).nested(),
        TypeSig.all),
        (att, conf, p, r) => new BaseExprMeta[AttributeReference](att, conf, p, r) {
          // This is the only NOOP operator.  It goes away when things are bound
          override def convertToGpuImpl(): Expression = att

          // There are so many of these that we don't need to print them out, unless it
          // will not work on the GPU
          override def print(append: StringBuilder, depth: Int, all: Boolean): Unit = {
            if (!this.canThisBeReplaced || cannotRunOnGpuBecauseOfSparkPlan) {
              super.print(append, depth, all)
            }
          }
        }),

    expr[ToDegrees](
      "Converts radians to degrees",
      ExprChecks.mathUnary,
      (a, conf, p, r) => new UnaryExprMeta[ToDegrees](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuToDegrees = GpuToDegrees(child)
      }),
    expr[ToRadians](
      "Converts degrees to radians",
      ExprChecks.mathUnary,
      (a, conf, p, r) => new UnaryExprMeta[ToRadians](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuToRadians = GpuToRadians(child)
      }),
    expr[Bin](
      "Returns the string representation of the long value `expr` represented in binary",
      ExprChecks.unaryProject(TypeSig.STRING, TypeSig.STRING, TypeSig.LONG, TypeSig.LONG),
      (a, conf, p, r) => new UnaryExprMeta[Bin](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuBin = GpuBin(child)
      }),
    expr[Hex](
      "Returns the hex string representation of a value",
      ExprChecks.unaryProject(TypeSig.STRING, TypeSig.STRING,
        TypeSig.LONG + TypeSig.STRING + TypeSig.BINARY,
        TypeSig.LONG + TypeSig.STRING + TypeSig.BINARY),
      (a, conf, p, r) => new UnaryExprMeta[Hex](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuHex(child)
      }),
    expr[WindowExpression](
      "Calculates a return value for every input row of a table based on a group (or " +
        "\"window\") of rows",
      ExprChecks.windowOnly(
        TypeSig.all,
        TypeSig.all,
        Seq(ParamCheck("windowFunction", TypeSig.all, TypeSig.all),
          ParamCheck("windowSpec",
            TypeSig.CALENDAR + TypeSig.NULL + TypeSig.integral + TypeSig.DECIMAL_64,
            TypeSig.numericAndInterval))),
      (windowExpression, conf, p, r) => new GpuWindowExpressionMeta(windowExpression, conf, p, r)),
    expr[SpecifiedWindowFrame](
      "Specification of the width of the group (or \"frame\") of input rows " +
        "around which a window function is evaluated",
      ExprChecks.projectOnly(
        TypeSig.CALENDAR + TypeSig.NULL + TypeSig.integral,
        TypeSig.numericAndInterval,
        Seq(
          ParamCheck("lower",
            TypeSig.CALENDAR + TypeSig.NULL + TypeSig.integral + TypeSig.DECIMAL_128 +
              TypeSig.FLOAT + TypeSig.DOUBLE,
            TypeSig.numericAndInterval),
          ParamCheck("upper",
            TypeSig.CALENDAR + TypeSig.NULL + TypeSig.integral + TypeSig.DECIMAL_128 +
              TypeSig.FLOAT + TypeSig.DOUBLE,
            TypeSig.numericAndInterval))),
      (windowFrame, conf, p, r) => new GpuSpecifiedWindowFrameMeta(windowFrame, conf, p, r) ),
    expr[WindowSpecDefinition](
      "Specification of a window function, indicating the partitioning-expression, the row " +
        "ordering, and the width of the window",
      WindowSpecCheck,
      (windowSpec, conf, p, r) => new GpuWindowSpecDefinitionMeta(windowSpec, conf, p, r)),
    expr[CurrentRow.type](
      "Special boundary for a window frame, indicating stopping at the current row",
      ExprChecks.projectOnly(TypeSig.NULL, TypeSig.NULL),
      (currentRow, conf, p, r) => new ExprMeta[CurrentRow.type](currentRow, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = GpuSpecialFrameBoundary(currentRow)
      }),
    expr[UnboundedPreceding.type](
      "Special boundary for a window frame, indicating all rows preceding the current row",
      ExprChecks.projectOnly(TypeSig.NULL, TypeSig.NULL),
      (unboundedPreceding, conf, p, r) =>
        new ExprMeta[UnboundedPreceding.type](unboundedPreceding, conf, p, r) {
          override def convertToGpuImpl(): GpuExpression =
            GpuSpecialFrameBoundary(unboundedPreceding)
        }),
    expr[UnboundedFollowing.type](
      "Special boundary for a window frame, indicating all rows preceding the current row",
      ExprChecks.projectOnly(TypeSig.NULL, TypeSig.NULL),
      (unboundedFollowing, conf, p, r) =>
        new ExprMeta[UnboundedFollowing.type](unboundedFollowing, conf, p, r) {
          override def convertToGpuImpl(): GpuExpression =
            GpuSpecialFrameBoundary(unboundedFollowing)
        }),
    expr[RowNumber](
      "Window function that returns the index for the row within the aggregation window",
      ExprChecks.windowOnly(TypeSig.INT, TypeSig.INT,
        repeatingParamCheck =
          Some(RepeatingParamCheck("ordering",
            TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL,
            TypeSig.all))),
      (rowNumber, conf, p, r) => new ExprMeta[RowNumber](rowNumber, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = GpuRowNumber
      }),
    expr[Rank](
      "Window function that returns the rank value within the aggregation window",
      ExprChecks.windowOnly(TypeSig.INT, TypeSig.INT,
        repeatingParamCheck =
          Some(RepeatingParamCheck("ordering",
            TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL,
            TypeSig.all))),
      (rank, conf, p, r) => new ExprMeta[Rank](rank, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = GpuRank(childExprs.map(_.convertToGpu()))
      }),
    expr[DenseRank](
      "Window function that returns the dense rank value within the aggregation window",
      ExprChecks.windowOnly(TypeSig.INT, TypeSig.INT,
        repeatingParamCheck =
          Some(RepeatingParamCheck("ordering",
            TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL,
            TypeSig.all))),
      (denseRank, conf, p, r) => new ExprMeta[DenseRank](denseRank, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression =
          GpuDenseRank(childExprs.map(_.convertToGpu()))
      }),
    expr[PercentRank](
      "Window function that returns the percent rank value within the aggregation window",
      ExprChecks.windowOnly(TypeSig.DOUBLE, TypeSig.DOUBLE,
        repeatingParamCheck =
          Some(RepeatingParamCheck("ordering",
            TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL,
            TypeSig.all))),
      (percentRank, conf, p, r) => new ExprMeta[PercentRank](percentRank, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression =
          GpuPercentRank(childExprs.map(_.convertToGpu()))
      }),
    expr[NTile](
      "Window function that divides the rows in each partition into buckets",
      ExprChecks.windowOnly(TypeSig.INT, TypeSig.INT,
        Seq(ParamCheck("buckets", TypeSig.lit(TypeEnum.INT), TypeSig.lit(TypeEnum.INT)))),
      GpuNTileMeta),
    expr[Lead](
      "Window function that returns N entries ahead of this one",
      ExprChecks.windowOnly(
        (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
          TypeSig.ARRAY + TypeSig.STRUCT).nested(),
        TypeSig.all,
        Seq(
          ParamCheck("input",
            (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
              TypeSig.NULL + TypeSig.ARRAY + TypeSig.STRUCT).nested(),
            TypeSig.all),
          ParamCheck("offset", TypeSig.INT, TypeSig.INT),
          ParamCheck("default",
            (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
              TypeSig.ARRAY + TypeSig.STRUCT).nested(),
            TypeSig.all)
        )
      ),
      (lead, conf, p, r) => new OffsetWindowFunctionMeta[Lead](lead, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression =
          GpuLead(input.convertToGpu(), offset.convertToGpu(), default.convertToGpu())
      }),
    expr[Lag](
      "Window function that returns N entries behind this one",
      ExprChecks.windowOnly(
        (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
          TypeSig.ARRAY + TypeSig.STRUCT).nested(),
        TypeSig.all,
        Seq(
          ParamCheck("input",
            (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
              TypeSig.NULL + TypeSig.ARRAY + TypeSig.STRUCT).nested(),
            TypeSig.all),
          ParamCheck("offset", TypeSig.INT, TypeSig.INT),
          ParamCheck("default",
            (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
              TypeSig.ARRAY + TypeSig.STRUCT).nested(),
            TypeSig.all)
        )
      ),
      (lag, conf, p, r) => new OffsetWindowFunctionMeta[Lag](lag, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression =
          GpuLag(input.convertToGpu(), offset.convertToGpu(), default.convertToGpu())
      }),
  )
}
