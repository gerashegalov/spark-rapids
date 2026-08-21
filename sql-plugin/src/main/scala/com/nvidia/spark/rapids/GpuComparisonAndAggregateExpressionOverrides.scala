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

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids._
import org.apache.spark.sql.rapids.aggregate._
import org.apache.spark.sql.types._


private[rapids] object GpuComparisonAndAggregateExpressionOverrides {
  val rules: Seq[ExprRule[_ <: Expression]] = Seq(
    expr[Pmod](
      "Pmod",
      // Decimal support disabled https://github.com/NVIDIA/spark-rapids/issues/7553
      ExprChecks.binaryProject(TypeSig.integral + TypeSig.fp, TypeSig.cpuNumeric,
        ("lhs", (TypeSig.integral + TypeSig.fp).withPsNote(TypeEnum.DECIMAL,
          s"decimals with precision ${DecimalType.MAX_PRECISION} are not supported"),
            TypeSig.cpuNumeric),
        ("rhs", (TypeSig.integral + TypeSig.fp), TypeSig.cpuNumeric)),
      (a, conf, p, r) => new BinaryExprMeta[Pmod](a, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          a.dataType match {
            case dt: DecimalType if dt.precision == DecimalType.MAX_PRECISION =>
              willNotWorkOnGpu("pmod at maximum decimal precision is not supported")
            case _ =>
          }
        }
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuPmod(lhs, rhs)
      }),
    expr[Add](
      "Addition",
      ExprChecks.binaryProjectAndAst(
        TypeSig.implicitCastsAstTypes,
        TypeSig.gpuNumeric + GpuTypeShims.additionalArithmeticSupportedTypes,
        TypeSig.numericAndInterval,
        ("lhs", TypeSig.gpuNumeric + GpuTypeShims.additionalArithmeticSupportedTypes,
            TypeSig.numericAndInterval),
        ("rhs", TypeSig.gpuNumeric + GpuTypeShims.additionalArithmeticSupportedTypes,
            TypeSig.numericAndInterval)),
      (a, conf, p, r) => new BinaryAstExprMeta[Add](a, conf, p, r) {
        private val ansiEnabled = SQLConf.get.ansiEnabled

        override def tagExprForGpu(): Unit = {
          // Check if this Add expression is in TRY mode context
          if (TryModeShim.isTryMode(a)) {
            willNotWorkOnGpu("try_add is not supported on GPU")
          }
        }

        override def tagSelfForAst(): Unit = {
          if (ansiEnabled && GpuAnsi.needBasicOpOverflowCheck(a.dataType)) {
            willNotWorkInAst("AST Addition does not support ANSI mode.")
          }
        }

        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuAdd(lhs, rhs, ansiEnabled)(a.origin)
      }),
    expr[Subtract](
      "Subtraction",
      ExprChecks.binaryProjectAndAst(
        TypeSig.implicitCastsAstTypes,
        TypeSig.gpuNumeric + GpuTypeShims.additionalArithmeticSupportedTypes,
        TypeSig.numericAndInterval,
        ("lhs", TypeSig.gpuNumeric + GpuTypeShims.additionalArithmeticSupportedTypes,
            TypeSig.numericAndInterval),
        ("rhs", TypeSig.gpuNumeric + GpuTypeShims.additionalArithmeticSupportedTypes,
            TypeSig.numericAndInterval)),
      (a, conf, p, r) => new BinaryAstExprMeta[Subtract](a, conf, p, r) {
        private val ansiEnabled = SQLConf.get.ansiEnabled

        override def tagExprForGpu(): Unit = {
          // Check if this Subtract expression is in TRY mode context
          if (TryModeShim.isTryMode(a)) {
            willNotWorkOnGpu("try_subtract is not supported on GPU")
          }
        }

        override def tagSelfForAst(): Unit = {
          if (ansiEnabled && GpuAnsi.needBasicOpOverflowCheck(a.dataType)) {
            willNotWorkInAst("AST Subtraction does not support ANSI mode.")
          }
        }

        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuSubtract(lhs, rhs, ansiEnabled)(a.origin)
      }),
    expr[And](
      "Logical AND",
      ExprChecks.binaryProjectAndAst(TypeSig.BOOLEAN, TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        ("lhs", TypeSig.BOOLEAN, TypeSig.BOOLEAN),
        ("rhs", TypeSig.BOOLEAN, TypeSig.BOOLEAN)),
      (a, conf, p, r) => new BinaryExprMeta[And](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuAnd(lhs, rhs)
      }),
    expr[Or](
      "Logical OR",
      ExprChecks.binaryProjectAndAst(TypeSig.BOOLEAN, TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        ("lhs", TypeSig.BOOLEAN, TypeSig.BOOLEAN),
        ("rhs", TypeSig.BOOLEAN, TypeSig.BOOLEAN)),
      (a, conf, p, r) => new BinaryExprMeta[Or](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuOr(lhs, rhs)
      }),
    expr[EqualNullSafe](
      "Check if the values are equal including nulls <=>",
      ExprChecks.binaryProject(
        TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        ("lhs", (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            GpuTypeShims.additionalPredicateSupportedTypes + TypeSig.STRUCT).nested(),
            TypeSig.comparable),
        ("rhs", (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            GpuTypeShims.additionalPredicateSupportedTypes + TypeSig.STRUCT).nested(),
            TypeSig.comparable)),
      (a, conf, p, r) => new BinaryExprMeta[EqualNullSafe](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuEqualNullSafe(lhs, rhs)
      }),
    expr[EqualTo](
      "Check if the values are equal",
      ExprChecks.binaryProjectAndAst(
        TypeSig.comparisonAstTypes,
        TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        ("lhs", (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            GpuTypeShims.additionalPredicateSupportedTypes + TypeSig.STRUCT).nested(),
            TypeSig.comparable),
        ("rhs", (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            GpuTypeShims.additionalPredicateSupportedTypes + TypeSig.STRUCT).nested(),
            TypeSig.comparable)),
      (a, conf, p, r) => new BinaryAstExprMeta[EqualTo](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuEqualTo(lhs, rhs)
      }),
    expr[GreaterThan](
      "> operator",
      ExprChecks.binaryProjectAndAst(
        TypeSig.comparisonAstTypes,
        TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        ("lhs", (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            GpuTypeShims.additionalPredicateSupportedTypes + TypeSig.STRUCT).nested(),
            TypeSig.orderable),
        ("rhs", (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            GpuTypeShims.additionalPredicateSupportedTypes + TypeSig.STRUCT).nested(),
            TypeSig.orderable)),
      (a, conf, p, r) => new BinaryAstExprMeta[GreaterThan](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuStringInstr.optimizeContains(GpuGreaterThan(lhs, rhs))
      }),
    expr[GreaterThanOrEqual](
      ">= operator",
      ExprChecks.binaryProjectAndAst(
        TypeSig.comparisonAstTypes,
        TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        ("lhs", (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            GpuTypeShims.additionalPredicateSupportedTypes + TypeSig.STRUCT).nested(),
            TypeSig.orderable),
        ("rhs", (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            GpuTypeShims.additionalPredicateSupportedTypes + TypeSig.STRUCT).nested(),
            TypeSig.orderable)),
      (a, conf, p, r) => new BinaryAstExprMeta[GreaterThanOrEqual](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuStringInstr.optimizeContains(GpuGreaterThanOrEqual(lhs, rhs))
      }),
    expr[In](
      "IN operator",
      ExprChecks.projectOnly(TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        Seq(ParamCheck("value", TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128,
          TypeSig.comparable)),
        Some(RepeatingParamCheck("list",
          (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128).withAllLit(),
          TypeSig.comparable))),
      (in, conf, p, r) => new ExprMeta[In](in, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression =
          GpuInSet(childExprs.head.convertToGpu(), in.list.asInstanceOf[Seq[Literal]].map(_.value))
      }),
    expr[InSet](
      "INSET operator",
      ExprChecks.unaryProject(TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128, TypeSig.comparable),
      (in, conf, p, r) => new ExprMeta[InSet](in, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression =
          GpuInSet(childExprs.head.convertToGpu(), in.hset.toSeq)
      }),
    expr[LessThan](
      "< operator",
      ExprChecks.binaryProjectAndAst(
        TypeSig.comparisonAstTypes,
        TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        ("lhs", (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            GpuTypeShims.additionalPredicateSupportedTypes + TypeSig.STRUCT).nested(),
            TypeSig.orderable),
        ("rhs", (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            GpuTypeShims.additionalPredicateSupportedTypes + TypeSig.STRUCT).nested(),
            TypeSig.orderable)),
      (a, conf, p, r) => new BinaryAstExprMeta[LessThan](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuStringInstr.optimizeContains(GpuLessThan(lhs, rhs))
      }),
    expr[LessThanOrEqual](
      "<= operator",
      ExprChecks.binaryProjectAndAst(
        TypeSig.comparisonAstTypes,
        TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        ("lhs", (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            GpuTypeShims.additionalPredicateSupportedTypes + TypeSig.STRUCT).nested(),
            TypeSig.orderable),
        ("rhs", (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            GpuTypeShims.additionalPredicateSupportedTypes + TypeSig.STRUCT).nested(),
            TypeSig.orderable)),
      (a, conf, p, r) => new BinaryAstExprMeta[LessThanOrEqual](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuStringInstr.optimizeContains(GpuLessThanOrEqual(lhs, rhs))
      }),
    expr[CaseWhen](
      "CASE WHEN expression",
      CaseWhenCheck,
      GpuCaseWhenMeta),
    expr[If](
      "IF expression",
      ExprChecks.projectOnly(
        (gpuCommonTypes + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP +
            TypeSig.BINARY + GpuTypeShims.additionalCommonOperatorSupportedTypes).nested(),
        TypeSig.all,
        Seq(ParamCheck("predicate", TypeSig.BOOLEAN, TypeSig.BOOLEAN),
          ParamCheck("trueValue",
            (gpuCommonTypes + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP +
                TypeSig.BINARY + GpuTypeShims.additionalCommonOperatorSupportedTypes).nested(),
            TypeSig.all),
          ParamCheck("falseValue",
            (gpuCommonTypes + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP +
                TypeSig.BINARY + GpuTypeShims.additionalCommonOperatorSupportedTypes).nested(),
            TypeSig.all))),
      GpuIfMeta),
    expr[Pow](
      "lhs ^ rhs",
      ExprChecks.binaryProjectAndAst(
        TypeSig.implicitCastsAstTypes, TypeSig.DOUBLE, TypeSig.DOUBLE,
        ("lhs", TypeSig.DOUBLE, TypeSig.DOUBLE),
        ("rhs", TypeSig.DOUBLE, TypeSig.DOUBLE)),
      (a, conf, p, r) => new BinaryAstExprMeta[Pow](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuPow(lhs, rhs)
      }),
    expr[AggregateExpression](
      "Aggregate expression",
      // Let the underlying expression checks decide whether this can be on the GPU.
      ExprChecks.fullAgg(
        TypeSig.all,
        TypeSig.all,
        Seq(ParamCheck("aggFunc", TypeSig.all, TypeSig.all)),
        Some(RepeatingParamCheck("filter", TypeSig.BOOLEAN, TypeSig.BOOLEAN))),
      GpuAggregateExpressionMeta),
    expr[SortOrder](
      "Sort order",
      ExprChecks.projectOnly(
        pluginSupportedOrderableSig + TypeSig.ARRAY.nested(gpuCommonTypes)
            .withPsNote(TypeEnum.ARRAY, "STRUCT is not supported as a child type for ARRAY"),
        TypeSig.orderable,
        Seq(ParamCheck(
          "input",
          pluginSupportedOrderableSig + TypeSig.ARRAY.nested(gpuCommonTypes)
             .withPsNote(TypeEnum.ARRAY, "STRUCT is not supported as a child type for ARRAY"),
          TypeSig.orderable))),
      GpuSortOrderMeta),
    expr[PivotFirst](
      "PivotFirst operator",
      ExprChecks.reductionAndGroupByAgg(
        TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
          TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128),
        TypeSig.all,
        Seq(ParamCheck(
          "pivotColumn",
          (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128),
          TypeSig.all),
          ParamCheck("valueColumn",
          TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128,
          TypeSig.all))),
      (pivot, conf, p, r) => new ImperativeAggExprMeta[PivotFirst](pivot, conf, p, r) {
        override def tagAggForGpu(): Unit = {
          pivot.pivotColumn.dataType match {
            // `StringType` is the UTF8_BINARY singleton, while `st` may be another
            // StringType instance whose collation differs from UTF8_BINARY in Spark 4.x.
            case st: StringType if st != StringType =>
              willNotWorkOnGpu(
                "PivotFirst does not support non-UTF8_BINARY string collations on the GPU")
            case _ =>
          }
          // If pivotColumnValues doesn't have distinct values, fall back to CPU
          if (pivot.pivotColumnValues.distinct.lengthCompare(pivot.pivotColumnValues.length) != 0) {
            willNotWorkOnGpu("PivotFirst does not work on the GPU when there are duplicate" +
                " pivot values provided")
          }
        }
        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression = {
          val Seq(pivotColumn, valueColumn) = childExprs
          GpuPivotFirst(pivotColumn, valueColumn, pivot.pivotColumnValues)
        }

        // Pivot does not overflow, so it doesn't need the ANSI check
        override val needsAnsiCheck: Boolean = false
      }),
    expr[Count](
      "Count aggregate operator",
      ExprChecks.fullAgg(
        TypeSig.LONG, TypeSig.LONG,
        repeatingParamCheck = Some(RepeatingParamCheck(
          "input", TypeSig.all, TypeSig.all))),
      (count, conf, p, r) => new AggExprMeta[Count](count, conf, p, r) {

        // Spark Count agg returns Long and does not check Ansi mode and overflow
        override def needsAnsiCheck: Boolean = false

        override def tagAggForGpu(): Unit = {
          if (count.children.size > 1) {
            willNotWorkOnGpu("count of multiple columns not supported")
          }
        }
        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression =
          GpuCount(childExprs)
      }),
    expr[Max](
      "Max aggregate operator",
      ExprChecksImpl(
        ExprChecks.reductionAndGroupByAgg(
          (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL + TypeSig.STRUCT +
            TypeSig.ARRAY).nested(),
          TypeSig.orderable,
          Seq(ParamCheck("input",
            (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL + TypeSig.STRUCT +
              TypeSig.ARRAY).nested(),
            TypeSig.orderable))).asInstanceOf[ExprChecksImpl].contexts
          ++
          ExprChecks.windowOnly(
            (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
            TypeSig.orderable,
            Seq(ParamCheck("input",
              (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
              TypeSig.orderable))).asInstanceOf[ExprChecksImpl].contexts),
      (max, conf, p, r) => new AggExprMeta[Max](max, conf, p, r) {
        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression =
          GpuMax(childExprs.head)

        // Max does not overflow, so it doesn't need the ANSI check
        override val needsAnsiCheck: Boolean = false
      }),
    expr[Min](
      "Min aggregate operator",
      ExprChecksImpl(
        ExprChecks.reductionAndGroupByAgg(
          (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL + TypeSig.STRUCT +
              TypeSig.ARRAY).nested(),
          TypeSig.orderable,
          Seq(ParamCheck("input",
            (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL + TypeSig.STRUCT +
              TypeSig.ARRAY).nested(),
            TypeSig.orderable))).asInstanceOf[ExprChecksImpl].contexts
          ++
          ExprChecks.windowOnly(
            (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
            TypeSig.orderable,
            Seq(ParamCheck("input",
              (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
              TypeSig.orderable))).asInstanceOf[ExprChecksImpl].contexts),
      (a, conf, p, r) => new AggExprMeta[Min](a, conf, p, r) {
        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression =
          GpuMin(childExprs.head)

        // Min does not overflow, so it doesn't need the ANSI check
        override val needsAnsiCheck: Boolean = false
      }),
    expr[Sum](
      "Sum aggregate operator",
      ExprChecks.fullAgg(
        TypeSig.LONG + TypeSig.DOUBLE + TypeSig.DECIMAL_128,
        TypeSig.LONG + TypeSig.DOUBLE + TypeSig.DECIMAL_128,
        Seq(ParamCheck("input", TypeSig.gpuNumeric, TypeSig.cpuNumeric))),
      (a, conf, p, r) => new AggExprMeta[Sum](a, conf, p, r) {
        override def tagAggForGpu(): Unit = {
          val inputDataType = a.child.dataType
          checkAndTagFloatAgg(inputDataType, this.conf, this)

          // Check if this Sum expression is in TRY mode context
          if (TryModeShim.isTryMode(a)) {
            willNotWorkOnGpu("try_sum is not supported on GPU")
          }
        }

        override def needsAnsiCheck: Boolean = false

        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression =
          GpuSum(childExprs.head, a.dataType)
      }),
    expr[NthValue](
      "nth window operator",
      ExprChecks.windowOnly(
        (TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP + TypeSig.BINARY +
            TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128).nested(),
        TypeSig.all,
        Seq(ParamCheck("input",
          (TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP + TypeSig.BINARY +
              TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128).nested(),
          TypeSig.all),
          ParamCheck("offset", TypeSig.lit(TypeEnum.INT), TypeSig.lit(TypeEnum.INT)))
      ),
      (a, conf, p, r) => new AggExprMeta[NthValue](a, conf, p, r) {
        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression =
          GpuNthValue(childExprs.head, a.offset, a.ignoreNulls)

        // nth does not overflow, so it doesn't need the ANSI check
        override val needsAnsiCheck: Boolean = false
      }),
    expr[First](
      "first aggregate operator",
      ExprChecks.fullAgg(
        (TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP + TypeSig.BINARY +
            TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128).nested(),
        TypeSig.all,
        Seq(ParamCheck("input",
          (TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP + TypeSig.BINARY +
              TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128).nested(),
          TypeSig.all))
      ),
      (a, conf, p, r) => new AggExprMeta[First](a, conf, p, r) {
        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression =
          GpuFirst(childExprs.head, a.ignoreNulls)

        // First does not overflow, so it doesn't need the ANSI check
        override val needsAnsiCheck: Boolean = false
      }),
    expr[Last](
    "last aggregate operator",
      ExprChecks.fullAgg(
        (TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP + TypeSig.BINARY +
            TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128).nested(),
        TypeSig.all,
        Seq(ParamCheck("input",
          (TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP + TypeSig.BINARY +
              TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128).nested(),
          TypeSig.all))
      ),
      (a, conf, p, r) => new AggExprMeta[Last](a, conf, p, r) {
        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression =
          GpuLast(childExprs.head, a.ignoreNulls)

        // Last does not overflow, so it doesn't need the ANSI check
        override val needsAnsiCheck: Boolean = false
      }),
    expr[MaxBy](
      "MaxBy aggregate operator. It may produce different results than CPU when " +
        "multiple rows in a group have same minimum value in the ordering column and " +
        "different associated values in the value column.",
      ExprChecks.reductionAndGroupByAgg(
        (TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP + TypeSig.BINARY +
          TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128).nested(),
        TypeSig.all,
        Seq(
          ParamCheck("value", (TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP + TypeSig.BINARY
            + TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128).nested(),
            TypeSig.all),
          ParamCheck("ordering", (TypeSig.commonCudfTypes - TypeSig.fp + TypeSig.DECIMAL_128 +
            TypeSig.NULL + TypeSig.STRUCT + TypeSig.ARRAY).nested(
              TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
              TypeSig.NULL + TypeSig.STRUCT + TypeSig.ARRAY),
            TypeSig.orderable))
      ),
      (maxBy, conf, p, r) => new AggExprMeta[MaxBy](maxBy, conf, p, r) {

        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression = {
          // Only two children (value expression, ordering expression)
          require(childExprs.length == 2)
          GpuMaxBy(childExprs.head, childExprs.last)
        }

        // MaxBy does not overflow, so it doesn't need the ANSI check
        override val needsAnsiCheck: Boolean = false
      }),
    expr[MinBy](
      "MinBy aggregate operator. It may produce different results than CPU when " +
        "multiple rows in a group have same minimum value in the ordering column and " +
        "different associated values in the value column.",
      ExprChecks.reductionAndGroupByAgg(
        (TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP + TypeSig.BINARY +
          TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128).nested(),
        TypeSig.all,
        Seq(
          ParamCheck("value", (TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP + TypeSig.BINARY
            + TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128).nested(),
            TypeSig.all),
          ParamCheck("ordering", (TypeSig.commonCudfTypes - TypeSig.fp + TypeSig.DECIMAL_128 +
            TypeSig.NULL + TypeSig.STRUCT + TypeSig.ARRAY).nested(
              TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
              TypeSig.NULL + TypeSig.STRUCT + TypeSig.ARRAY),
            TypeSig.orderable))
      ),
      (minBy, conf, p, r) => new AggExprMeta[MinBy](minBy, conf, p, r) {

        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression = {
          // Only two children (value expression, ordering expression)
          require(childExprs.length == 2)
          GpuMinBy(childExprs.head, childExprs.last)
        }

        // MinBy does not overflow, so it doesn't need the ANSI check
        override val needsAnsiCheck: Boolean = false
      }),
    expr[BRound](
      "Round an expression to d decimal places using HALF_EVEN rounding mode",
      ExprChecks.binaryProject(
        TypeSig.gpuNumeric, TypeSig.cpuNumeric,
        ("value", TypeSig.gpuNumeric +
            TypeSig.psNote(TypeEnum.FLOAT, "result may round slightly differently") +
            TypeSig.psNote(TypeEnum.DOUBLE, "result may round slightly differently"),
            TypeSig.cpuNumeric),
        ("scale", TypeSig.lit(TypeEnum.INT), TypeSig.lit(TypeEnum.INT))),
      (a, conf, p, r) => new GpuBRoundMeta(a, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          a.child.dataType match {
            case FloatType | DoubleType if !this.conf.isIncompatEnabled =>
              willNotWorkOnGpu("rounding floating point numbers may be slightly off " +
                  s"compared to Spark's result, to enable set ${RapidsConf.INCOMPATIBLE_OPS}")
            case _ => // NOOP
          }
        }
      }),
    expr[Round](
      "Round an expression to d decimal places using HALF_UP rounding mode",
      ExprChecks.binaryProject(
        TypeSig.gpuNumeric, TypeSig.cpuNumeric,
        ("value", TypeSig.gpuNumeric +
            TypeSig.psNote(TypeEnum.FLOAT, "result may round slightly differently") +
            TypeSig.psNote(TypeEnum.DOUBLE, "result may round slightly differently"),
            TypeSig.cpuNumeric),
        ("scale", TypeSig.lit(TypeEnum.INT), TypeSig.lit(TypeEnum.INT))),
      (a, conf, p, r) => new GpuRoundMeta(a, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          a.child.dataType match {
            case FloatType | DoubleType if !this.conf.isIncompatEnabled =>
              willNotWorkOnGpu("rounding floating point numbers may be slightly off " +
                  s"compared to Spark's result, to enable set ${RapidsConf.INCOMPATIBLE_OPS}")
            case _ => // NOOP
          }
        }
      }),
  )
}
