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

import ai.rapids.cudf.DType
import com.nvidia.spark.rapids.GpuOverrides._
import com.nvidia.spark.rapids.shims._

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.catalyst.optimizer.NormalizeNaNAndZero
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids._
import org.apache.spark.sql.rapids.aggregate._
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.CalendarInterval


private[rapids] object GpuMathAndDateExpressionOverrides {
  val rules: Seq[ExprRule[_ <: Expression]] = Seq(
    expr[PreciseTimestampConversion](
      "Expression used internally to convert the TimestampType to Long and back without losing " +
          "precision, i.e. in microseconds. Used in time windowing",
      ExprChecks.unaryProject(
        TypeSig.TIMESTAMP + TypeSig.LONG,
        TypeSig.TIMESTAMP + TypeSig.LONG,
        TypeSig.TIMESTAMP + TypeSig.LONG,
        TypeSig.TIMESTAMP + TypeSig.LONG),
      (a, conf, p, r) => new UnaryExprMeta[PreciseTimestampConversion](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuPreciseTimestampConversion(child, a.fromType, a.toType)
      }),
    expr[UnaryMinus](
      "Negate a numeric value",
      ExprChecks.unaryProjectAndAstInputMatchesOutput(
        TypeSig.implicitCastsAstTypes,
        TypeSig.gpuNumeric + GpuTypeShims.additionalArithmeticSupportedTypes,
        TypeSig.numericAndInterval),
      (a, conf, p, r) => new UnaryAstExprMeta[UnaryMinus](a, conf, p, r) {
        val ansiEnabled = SQLConf.get.ansiEnabled

        override def tagSelfForAst(): Unit = {
          if (ansiEnabled && GpuAnsi.needBasicOpOverflowCheck(a.dataType)) {
            willNotWorkInAst("AST unary minus does not support ANSI mode.")
          }
        }

        override def convertToGpu(child: Expression): GpuExpression =
          GpuUnaryMinus(child, ansiEnabled)
      }),
    expr[UnaryPositive](
      "A numeric value with a + in front of it",
      ExprChecks.unaryProjectAndAstInputMatchesOutput(
        TypeSig.astTypes + GpuTypeShims.additionalArithmeticSupportedTypes,
        TypeSig.gpuNumeric + GpuTypeShims.additionalArithmeticSupportedTypes,
        TypeSig.numericAndInterval),
      (a, conf, p, r) => new ExprMeta[UnaryPositive](a, conf, p, r) {
        override val isFoldableNonLitAllowed: Boolean = true
        override def convertToGpuImpl(): GpuExpression =
          GpuUnaryPositive(childExprs.head.convertToGpu())
      }),
    expr[Year](
      "Returns the year from a date or timestamp",
      ExprChecks.unaryProject(TypeSig.INT, TypeSig.INT, TypeSig.DATE, TypeSig.DATE),
      (a, conf, p, r) => new UnaryExprMeta[Year](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuYear(child)
      }),
    expr[Month](
      "Returns the month from a date or timestamp",
      ExprChecks.unaryProject(TypeSig.INT, TypeSig.INT, TypeSig.DATE, TypeSig.DATE),
      (a, conf, p, r) => new UnaryExprMeta[Month](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuMonth(child)
      }),
    expr[Quarter](
      "Returns the quarter of the year for date, in the range 1 to 4",
      ExprChecks.unaryProject(TypeSig.INT, TypeSig.INT, TypeSig.DATE, TypeSig.DATE),
      (a, conf, p, r) => new UnaryExprMeta[Quarter](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuQuarter(child)
      }),
    expr[DayOfMonth](
      "Returns the day of the month from a date or timestamp",
      ExprChecks.unaryProject(TypeSig.INT, TypeSig.INT, TypeSig.DATE, TypeSig.DATE),
      (a, conf, p, r) => new UnaryExprMeta[DayOfMonth](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuDayOfMonth(child)
      }),
    expr[DayOfYear](
      "Returns the day of the year from a date or timestamp",
      ExprChecks.unaryProject(TypeSig.INT, TypeSig.INT, TypeSig.DATE, TypeSig.DATE),
      (a, conf, p, r) => new UnaryExprMeta[DayOfYear](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuDayOfYear(child)
      }),
    expr[SecondsToTimestamp](
      "Converts the number of seconds from unix epoch to a timestamp",
      ExprChecks.unaryProject(TypeSig.TIMESTAMP, TypeSig.TIMESTAMP,
      TypeSig.gpuNumeric, TypeSig.cpuNumeric),
      (a, conf, p, r) => new UnaryExprMeta[SecondsToTimestamp](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuSecondsToTimestamp(child)
      }),
    expr[MillisToTimestamp](
      "Converts the number of milliseconds from unix epoch to a timestamp",
      ExprChecks.unaryProject(TypeSig.TIMESTAMP, TypeSig.TIMESTAMP,
      TypeSig.integral, TypeSig.integral),
      (a, conf, p, r) => new UnaryExprMeta[MillisToTimestamp](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuMillisToTimestamp(child)
      }),
    expr[MicrosToTimestamp](
      "Converts the number of microseconds from unix epoch to a timestamp",
      ExprChecks.unaryProject(TypeSig.TIMESTAMP, TypeSig.TIMESTAMP,
      TypeSig.integral, TypeSig.integral),
      (a, conf, p, r) => new UnaryExprMeta[MicrosToTimestamp](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuMicrosToTimestamp(child)
      }),
    expr[Acos](
      "Inverse cosine",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Acos](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuAcos(child)
      }),
    expr[Acosh](
      "Inverse hyperbolic cosine",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Acosh](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          if (this.conf.includeImprovedFloat) {
            GpuAcoshImproved(child)
          } else {
            GpuAcoshCompat(child)
          }
      }),
    expr[Asin](
      "Inverse sine",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Asin](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuAsin(child)
      }),
    expr[Asinh](
      "Inverse hyperbolic sine",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Asinh](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          if (this.conf.includeImprovedFloat) {
            GpuAsinhImproved(child)
          } else {
            GpuAsinhCompat(child)
          }

        override def tagSelfForAst(): Unit = {
          if (!this.conf.includeImprovedFloat) {
            // AST is not expressive enough yet to implement the conditional expression needed
            // to emulate Spark's behavior
            willNotWorkInAst("asinh is not AST compatible unless " +
                s"${RapidsConf.IMPROVED_FLOAT_OPS.key} is enabled")
          }
        }
      }),
    expr[Sqrt](
      "Square root",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Sqrt](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuSqrt(child)
      }),
    expr[Cbrt](
      "Cube root",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Cbrt](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuCbrt(child)
      }),
    expr[Hypot](
      "Pythagorean addition (Hypotenuse) of real numbers",
      ExprChecks.binaryProject(
        TypeSig.DOUBLE,
        TypeSig.DOUBLE,
        ("lhs", TypeSig.DOUBLE, TypeSig.DOUBLE),
        ("rhs", TypeSig.DOUBLE, TypeSig.DOUBLE)),
      (a, conf, p, r) => new BinaryExprMeta[Hypot](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuHypot(lhs, rhs)
      }),
    expr[Floor](
      "Floor of a number",
      ExprChecks.unaryProjectInputMatchesOutput(
        TypeSig.DOUBLE + TypeSig.LONG + TypeSig.DECIMAL_128,
        TypeSig.DOUBLE + TypeSig.LONG + TypeSig.DECIMAL_128),
      (a, conf, p, r) => new UnaryExprMeta[Floor](a, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          a.dataType match {
            case dt: DecimalType =>
              val precision = GpuFloorCeil.unboundedOutputPrecision(dt)
              if (precision > DType.DECIMAL128_MAX_PRECISION) {
                willNotWorkOnGpu(s"output precision $precision would require overflow " +
                    s"checks, which are not supported yet")
              }
            case _ => // NOOP
          }
        }

        override def convertToGpu(child: Expression): GpuExpression = {
          // use Spark `Floor.dataType` to keep consistent between Spark versions.
          GpuFloor(child, a.dataType)
        }
      }),
    expr[Ceil](
      "Ceiling of a number",
      ExprChecks.unaryProjectInputMatchesOutput(
        TypeSig.DOUBLE + TypeSig.LONG + TypeSig.DECIMAL_128,
        TypeSig.DOUBLE + TypeSig.LONG + TypeSig.DECIMAL_128),
      (a, conf, p, r) => new UnaryExprMeta[Ceil](a, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          a.dataType match {
            case dt: DecimalType =>
              val precision = GpuFloorCeil.unboundedOutputPrecision(dt)
              if (precision > DType.DECIMAL128_MAX_PRECISION) {
                willNotWorkOnGpu(s"output precision $precision would require overflow " +
                    s"checks, which are not supported yet")
              }
            case _ => // NOOP
          }
        }

        override def convertToGpu(child: Expression): GpuExpression = {
          // use Spark `Ceil.dataType` to keep consistent between Spark versions.
          GpuCeil(child, a.dataType)
        }
      }),
    expr[Not](
      "Boolean not operator",
      ExprChecks.unaryProjectAndAstInputMatchesOutput(
        TypeSig.astTypes, TypeSig.BOOLEAN, TypeSig.BOOLEAN),
      (a, conf, p, r) => new UnaryAstExprMeta[Not](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuNot(child)
      }),
    expr[IsNull](
      "Checks if a value is null",
      ExprChecks.unaryProjectAndAst(TypeSig.astTypes, TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.MAP + TypeSig.ARRAY +
            TypeSig.STRUCT + TypeSig.DECIMAL_128 + TypeSig.BINARY +
            GpuTypeShims.additionalPredicateSupportedTypes).nested(),
        TypeSig.all),
      (a, conf, p, r) => new UnaryAstExprMeta[IsNull](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuIsNull(child)
      }),
    expr[IsNotNull](
      "Checks if a value is not null",
      ExprChecks.unaryProjectAndAst(TypeSig.astTypes, TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.MAP + TypeSig.ARRAY +
            TypeSig.STRUCT + TypeSig.DECIMAL_128 + TypeSig.BINARY +
            GpuTypeShims.additionalPredicateSupportedTypes).nested(),
        TypeSig.all),
      (a, conf, p, r) => new UnaryAstExprMeta[IsNotNull](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuIsNotNull(child)
      }),
    expr[IsNaN](
      "Checks if a value is NaN",
      ExprChecks.unaryProject(TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        TypeSig.DOUBLE + TypeSig.FLOAT, TypeSig.DOUBLE + TypeSig.FLOAT),
      (a, conf, p, r) => new UnaryExprMeta[IsNaN](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuIsNan(child)
      }),
    expr[Rint](
      "Rounds up a double value to the nearest double equal to an integer",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Rint](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuRint(child)
      }),
    expr[AtLeastNNonNulls](
      "Checks if number of non null/Nan values is greater than a given value",
      ExprChecks.projectOnly(TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        repeatingParamCheck = Some(RepeatingParamCheck("input",
          (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 + TypeSig.BINARY +
              TypeSig.MAP + TypeSig.ARRAY + TypeSig.STRUCT).nested(),
          TypeSig.all))),
      (a, conf, p, r) => new ExprMeta[AtLeastNNonNulls](a, conf, p, r) {
        def convertToGpuImpl(): GpuExpression =
          GpuAtLeastNNonNulls(a.n, childExprs.map(_.convertToGpu()))
      }),
    expr[DateAdd](
      "Returns the date that is num_days after start_date",
      ExprChecks.binaryProject(TypeSig.DATE, TypeSig.DATE,
        ("startDate", TypeSig.DATE, TypeSig.DATE),
        ("days",
            TypeSig.INT + TypeSig.SHORT + TypeSig.BYTE,
            TypeSig.INT + TypeSig.SHORT + TypeSig.BYTE)),
      (a, conf, p, r) => new BinaryExprMeta[DateAdd](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuDateAdd(lhs, rhs)
      }),
    expr[DateSub](
      "Returns the date that is num_days before start_date",
      ExprChecks.binaryProject(TypeSig.DATE, TypeSig.DATE,
        ("startDate", TypeSig.DATE, TypeSig.DATE),
        ("days",
            TypeSig.INT + TypeSig.SHORT + TypeSig.BYTE,
            TypeSig.INT + TypeSig.SHORT + TypeSig.BYTE)),
      (a, conf, p, r) => new BinaryExprMeta[DateSub](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuDateSub(lhs, rhs)
      }),
    expr[NaNvl](
      "Evaluates to `left` iff left is not NaN, `right` otherwise",
      ExprChecks.binaryProject(TypeSig.fp, TypeSig.fp,
        ("lhs", TypeSig.fp, TypeSig.fp),
        ("rhs", TypeSig.fp, TypeSig.fp)),
      (a, conf, p, r) => new BinaryExprMeta[NaNvl](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuNaNvl(lhs, rhs)
      }),
    expr[ShiftLeft](
      "Bitwise shift left (<<)",
      ExprChecks.binaryProject(TypeSig.INT + TypeSig.LONG, TypeSig.INT + TypeSig.LONG,
        ("value", TypeSig.INT + TypeSig.LONG, TypeSig.INT + TypeSig.LONG),
        ("amount", TypeSig.INT, TypeSig.INT)),
      (a, conf, p, r) => new BinaryExprMeta[ShiftLeft](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuShiftLeft(lhs, rhs)
      }),
    expr[ShiftRight](
      "Bitwise shift right (>>)",
      ExprChecks.binaryProject(TypeSig.INT + TypeSig.LONG, TypeSig.INT + TypeSig.LONG,
        ("value", TypeSig.INT + TypeSig.LONG, TypeSig.INT + TypeSig.LONG),
        ("amount", TypeSig.INT, TypeSig.INT)),
      (a, conf, p, r) => new BinaryExprMeta[ShiftRight](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuShiftRight(lhs, rhs)
      }),
    expr[ShiftRightUnsigned](
      "Bitwise unsigned shift right (>>>)",
      ExprChecks.binaryProject(TypeSig.INT + TypeSig.LONG, TypeSig.INT + TypeSig.LONG,
        ("value", TypeSig.INT + TypeSig.LONG, TypeSig.INT + TypeSig.LONG),
        ("amount", TypeSig.INT, TypeSig.INT)),
      (a, conf, p, r) => new BinaryExprMeta[ShiftRightUnsigned](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuShiftRightUnsigned(lhs, rhs)
      }),
    expr[BitwiseAnd](
      "Returns the bitwise AND of the operands",
      ExprChecks.binaryProjectAndAst(
        TypeSig.implicitCastsAstTypes, TypeSig.integral, TypeSig.integral,
        ("lhs", TypeSig.integral, TypeSig.integral),
        ("rhs", TypeSig.integral, TypeSig.integral)),
      (a, conf, p, r) => new BinaryAstExprMeta[BitwiseAnd](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuBitwiseAnd(lhs, rhs)
      }),
    expr[BitwiseOr](
      "Returns the bitwise OR of the operands",
      ExprChecks.binaryProjectAndAst(
        TypeSig.implicitCastsAstTypes, TypeSig.integral, TypeSig.integral,
        ("lhs", TypeSig.integral, TypeSig.integral),
        ("rhs", TypeSig.integral, TypeSig.integral)),
      (a, conf, p, r) => new BinaryAstExprMeta[BitwiseOr](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuBitwiseOr(lhs, rhs)
      }),
    expr[BitwiseXor](
      "Returns the bitwise XOR of the operands",
      ExprChecks.binaryProjectAndAst(
        TypeSig.implicitCastsAstTypes, TypeSig.integral, TypeSig.integral,
        ("lhs", TypeSig.integral, TypeSig.integral),
        ("rhs", TypeSig.integral, TypeSig.integral)),
      (a, conf, p, r) => new BinaryAstExprMeta[BitwiseXor](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuBitwiseXor(lhs, rhs)
      }),
    expr[BitwiseNot](
      "Returns the bitwise NOT of the operands",
      ExprChecks.unaryProjectAndAstInputMatchesOutput(
        TypeSig.implicitCastsAstTypes, TypeSig.integral, TypeSig.integral),
      (a, conf, p, r) => new UnaryAstExprMeta[BitwiseNot](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuBitwiseNot(child)
      }),
    expr[BitwiseCount](
      "Returns the number of bits that are set in the input as unsigned 64-bit integer",
      ExprChecks.unaryProject(
        TypeSig.INT, TypeSig.INT,
        TypeSig.integral + TypeSig.BOOLEAN, TypeSig.integral + TypeSig.BOOLEAN),
      (a, conf, p, r) => new UnaryExprMeta[BitwiseCount](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuBitwiseCount(child)
      }),
    expr[BitAndAgg](
      "Returns the bitwise AND of all non-null input values",
      ExprChecks.reductionAndGroupByAgg(
        TypeSig.integral, TypeSig.integral,
        Seq(ParamCheck("input", TypeSig.integral, TypeSig.integral))),
      (a, conf, p, r) => new AggExprMeta[BitAndAgg](a, conf, p, r) {
        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression =
          GpuBitAndAgg(childExprs.head)

        override def needsAnsiCheck: Boolean = false
      }),
    expr[BitOrAgg](
      "Returns the bitwise OR of all non-null input values",
      ExprChecks.reductionAndGroupByAgg(
        TypeSig.integral, TypeSig.integral,
        Seq(ParamCheck("input", TypeSig.integral, TypeSig.integral))),
      (a, conf, p, r) => new AggExprMeta[BitOrAgg](a, conf, p, r) {
        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression =
          GpuBitOrAgg(childExprs.head)

        override def needsAnsiCheck: Boolean = false
      }),
    expr[BitXorAgg](
      "Returns the bitwise XOR of all non-null input values",
      ExprChecks.reductionAndGroupByAgg(
        TypeSig.integral, TypeSig.integral,
        Seq(ParamCheck("input", TypeSig.integral, TypeSig.integral))),
      (a, conf, p, r) => new AggExprMeta[BitXorAgg](a, conf, p, r) {
        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression =
          GpuBitXorAgg(childExprs.head)

        override def needsAnsiCheck: Boolean = false
      }),
    expr[Coalesce] (
      "Returns the first non-null argument if exists. Otherwise, null",
      ExprChecks.projectOnly(
        (gpuCommonTypes + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.BINARY +
          TypeSig.MAP + GpuTypeShims.additionalArithmeticSupportedTypes).nested(),
        TypeSig.all,
        repeatingParamCheck = Some(RepeatingParamCheck("param",
          (gpuCommonTypes + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.BINARY +
            TypeSig.MAP + GpuTypeShims.additionalArithmeticSupportedTypes).nested(),
          TypeSig.all))),
      (a, conf, p, r) => new ExprMeta[Coalesce](a, conf, p, r) {
        // Allow foldable non-literal Coalesce (e.g. coalesce(cast(null as bigint), -1001)):
        // AQE can regenerate these after ConstantFolding ran; GpuCoalesce evaluates them on GPU.
        override val isFoldableNonLitAllowed: Boolean = true
        override def convertToGpuImpl(): GpuExpression =
          GpuCoalesce(childExprs.map(_.convertToGpu()))
      }),
    expr[Least] (
      "Returns the least value of all parameters, skipping null values",
      ExprChecks.projectOnly(
        TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128, TypeSig.orderable,
        repeatingParamCheck = Some(RepeatingParamCheck("param",
          TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128,
          TypeSig.orderable))),
      (a, conf, p, r) => new ExprMeta[Least](a, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = GpuLeast(childExprs.map(_.convertToGpu()))
      }),
    expr[Greatest] (
      "Returns the greatest value of all parameters, skipping null values",
      ExprChecks.projectOnly(
        TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128, TypeSig.orderable,
        repeatingParamCheck = Some(RepeatingParamCheck("param",
          TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128,
          TypeSig.orderable))),
      (a, conf, p, r) => new ExprMeta[Greatest](a, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression =
          GpuGreatest(childExprs.map(_.convertToGpu()))
      }),
    expr[Atan](
      "Inverse tangent",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Atan](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuAtan(child)
      }),
    expr[Atanh](
      "Inverse hyperbolic tangent",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Atanh](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuAtanh(child)
      }),
    expr[Cos](
      "Cosine",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Cos](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuCos(child)
      }),
    expr[Exp](
      "Euler's number e raised to a power",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Exp](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuExp(child)
      }),
    expr[Expm1](
      "Euler's number e raised to a power minus 1",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Expm1](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuExpm1(child)
      }),
    expr[InitCap](
      "Returns str with the first letter of each word in uppercase. " +
      "All other letters are in lowercase",
      ExprChecks.unaryProjectInputMatchesOutput(TypeSig.STRING, TypeSig.STRING),
      (a, conf, p, r) => new UnaryExprMeta[InitCap](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuInitCap(child)
      }).incompat(CASE_MODIFICATION_INCOMPAT),
    expr[Log](
      "Natural log",
      ExprChecks.mathUnary,
      (a, conf, p, r) => new UnaryExprMeta[Log](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuLog(child)
      }),
    expr[Log1p](
      "Natural log 1 + expr",
      ExprChecks.mathUnary,
      (a, conf, p, r) => new UnaryExprMeta[Log1p](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = {
          // No need for overflow checking on the GpuAdd in Double as Double handles overflow
          // the same in all modes.
          GpuLog(GpuAdd(child, GpuLiteral(1d, DataTypes.DoubleType), false)())
        }
      }),
    expr[Log2](
      "Log base 2",
      ExprChecks.mathUnary,
      (a, conf, p, r) => new UnaryExprMeta[Log2](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuLogarithm(child, GpuLiteral(2d, DataTypes.DoubleType))
      }),
    expr[Log10](
      "Log base 10",
      ExprChecks.mathUnary,
      (a, conf, p, r) => new UnaryExprMeta[Log10](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuLogarithm(child, GpuLiteral(10d, DataTypes.DoubleType))
      }),
    expr[Logarithm](
      "Log variable base",
      ExprChecks.binaryProject(TypeSig.DOUBLE, TypeSig.DOUBLE,
        ("value", TypeSig.DOUBLE, TypeSig.DOUBLE),
        ("base", TypeSig.DOUBLE, TypeSig.DOUBLE)),
      (a, conf, p, r) => new BinaryExprMeta[Logarithm](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          // the order of the parameters is transposed intentionally
          GpuLogarithm(rhs, lhs)
      }),
    expr[Sin](
      "Sine",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Sin](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuSin(child)
      }),
    expr[Sinh](
      "Hyperbolic sine",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Sinh](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuSinh(child)
      }),
    expr[Cosh](
      "Hyperbolic cosine",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Cosh](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuCosh(child)
      }),
    expr[Cot](
      "Cotangent",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Cot](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuCot(child)
      }),
    expr[Tanh](
      "Hyperbolic tangent",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Tanh](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuTanh(child)
      }),
    expr[Tan](
      "Tangent",
      ExprChecks.mathUnaryWithAst,
      (a, conf, p, r) => new UnaryAstExprMeta[Tan](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuTan(child)
      }),
    expr[NormalizeNaNAndZero](
      "Normalize NaN and zero",
      ExprChecks.unaryProjectInputMatchesOutput(
        TypeSig.DOUBLE + TypeSig.FLOAT,
        TypeSig.DOUBLE + TypeSig.FLOAT),
      (a, conf, p, r) => new UnaryExprMeta[NormalizeNaNAndZero](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuNormalizeNaNAndZero(child)
      }),
    expr[KnownFloatingPointNormalized](
      "Tag to prevent redundant normalization",
      ExprChecks.unaryProjectInputMatchesOutput(TypeSig.all, TypeSig.all),
      (a, conf, p, r) => new UnaryExprMeta[KnownFloatingPointNormalized](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuKnownFloatingPointNormalized(child)
      }),
    expr[KnownNotNull](
      "Tag an expression as known to not be null",
      ExprChecks.unaryProjectInputMatchesOutput(
        (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.BINARY + TypeSig.CALENDAR +
          TypeSig.ARRAY + TypeSig.MAP + TypeSig.STRUCT).nested(), TypeSig.all),
      (k, conf, p, r) => new UnaryExprMeta[KnownNotNull](k, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuKnownNotNull(child)
      }),
    expr[DateDiff](
      "Returns the number of days from startDate to endDate",
      ExprChecks.binaryProject(TypeSig.INT, TypeSig.INT,
        ("lhs", TypeSig.DATE, TypeSig.DATE),
        ("rhs", TypeSig.DATE, TypeSig.DATE)),
      (a, conf, p, r) => new BinaryExprMeta[DateDiff](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression = {
          GpuDateDiff(lhs, rhs)
        }
    }),
    // TimeAdd moved to TimeAddShims to handle API differences across Spark versions
    expr[DateAddInterval](
      "Adds interval to date",
      ExprChecks.binaryProject(TypeSig.DATE, TypeSig.DATE,
        ("start", TypeSig.DATE, TypeSig.DATE),
        ("interval", TypeSig.lit(TypeEnum.CALENDAR)
          .withPsNote(TypeEnum.CALENDAR, "month intervals are not supported"),
          TypeSig.CALENDAR)),
      (dateAddInterval, conf, p, r) =>
        new BinaryExprMeta[DateAddInterval](dateAddInterval, conf, p, r) {
          override def tagExprForGpu(): Unit = {
            GpuOverrides.extractLit(dateAddInterval.interval).foreach { lit =>
              val intvl = lit.value.asInstanceOf[CalendarInterval]
              if (intvl.months != 0) {
                willNotWorkOnGpu("interval months isn't supported")
              }
            }
          }

          override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
            GpuDateAddInterval(lhs, rhs)
        }),
    expr[DateFormatClass](
      "Converts timestamp to a value of string in the format specified by the date format",
      ExprChecks.binaryProject(TypeSig.STRING, TypeSig.STRING,
        ("timestamp", TypeSig.TIMESTAMP, TypeSig.TIMESTAMP),
        ("strfmt", TypeSig.lit(TypeEnum.STRING)
            .withPsNote(TypeEnum.STRING, "A limited number of formats are supported"),
            TypeSig.STRING)),
      (a, conf, p, r) => new UnixTimeExprMeta[DateFormatClass](a, conf, p, r) {
        override def isTimeZoneSupported = true
        override protected def allowLegacyFormattingOnlyFormats: Boolean = true
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuDateFormatClass(lhs, rhs, strfFormat, a.timeZoneId)
      }
    ),
    expr[ToUnixTimestamp](
      "Returns the UNIX timestamp of the given time",
      ExprChecks.binaryProject(TypeSig.LONG, TypeSig.LONG,
        ("timeExp",
            TypeSig.STRING + TypeSig.DATE + TypeSig.TIMESTAMP,
            TypeSig.STRING + TypeSig.DATE + TypeSig.TIMESTAMP),
        ("format", TypeSig.lit(TypeEnum.STRING)
            .withPsNote(TypeEnum.STRING, "A limited number of formats are supported"),
            TypeSig.STRING)),
      (a, conf, p, r) => new UnixTimeExprMeta[ToUnixTimestamp](a, conf, p, r) {
        // String type is not supported yet for non-UTC timezone.
        override def isTimeZoneSupported = true
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression = {
          GpuToUnixTimestamp(lhs, rhs, sparkFormat, strfFormat, a.timeZoneId)
        }
      }),
    expr[UnixTimestamp](
      "Returns the UNIX timestamp of current or specified time",
      ExprChecks.binaryProject(TypeSig.LONG, TypeSig.LONG,
        ("timeExp",
            TypeSig.STRING + TypeSig.DATE + TypeSig.TIMESTAMP,
            TypeSig.STRING + TypeSig.DATE + TypeSig.TIMESTAMP),
        ("format", TypeSig.lit(TypeEnum.STRING)
            .withPsNote(TypeEnum.STRING, "A limited number of formats are supported"),
            TypeSig.STRING)),
      (a, conf, p, r) => new UnixTimeExprMeta[UnixTimestamp](a, conf, p, r) {
        override def isTimeZoneSupported = true
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression = {
          GpuUnixTimestamp(lhs, rhs, sparkFormat, strfFormat, a.timeZoneId)
        }
      }),
    expr[Hour](
      "Returns the hour component of the string/timestamp",
      ExprChecks.unaryProject(TypeSig.INT, TypeSig.INT,
        TypeSig.TIMESTAMP, TypeSig.TIMESTAMP),
      (hour, conf, p, r) => new UnaryExprMeta[Hour](hour, conf, p, r) {
        override def isTimeZoneSupported = true
        override def convertToGpu(expr: Expression): GpuExpression = GpuHour(expr, hour.timeZoneId)
      }),
    expr[Minute](
      "Returns the minute component of the string/timestamp",
      ExprChecks.unaryProject(TypeSig.INT, TypeSig.INT,
        TypeSig.TIMESTAMP, TypeSig.TIMESTAMP),
      (minute, conf, p, r) => new UnaryExprMeta[Minute](minute, conf, p, r) {
        override def isTimeZoneSupported = true
        override def convertToGpu(expr: Expression): GpuExpression =
          GpuMinute(expr, minute.timeZoneId)
      }),
    expr[Second](
      "Returns the second component of the string/timestamp",
      ExprChecks.unaryProject(TypeSig.INT, TypeSig.INT,
        TypeSig.TIMESTAMP, TypeSig.TIMESTAMP),
      (second, conf, p, r) => new UnaryExprMeta[Second](second, conf, p, r) {
        override def isTimeZoneSupported = true
        override def convertToGpu(expr: Expression): GpuExpression =
          GpuSecond(expr, second.timeZoneId)
      }),
    expr[WeekDay](
      "Returns the day of the week (0 = Monday...6=Sunday)",
      ExprChecks.unaryProject(TypeSig.INT, TypeSig.INT,
        TypeSig.DATE, TypeSig.DATE),
      (a, conf, p, r) => new UnaryExprMeta[WeekDay](a, conf, p, r) {
        override def convertToGpu(expr: Expression): GpuExpression =
          GpuWeekDay(expr)
      }),
    expr[DayOfWeek](
      "Returns the day of the week (1 = Sunday...7=Saturday)",
      ExprChecks.unaryProject(TypeSig.INT, TypeSig.INT,
        TypeSig.DATE, TypeSig.DATE),
      (a, conf, p, r) => new UnaryExprMeta[DayOfWeek](a, conf, p, r) {
        override def convertToGpu(expr: Expression): GpuExpression =
          GpuDayOfWeek(expr)
      }),
    expr[LastDay](
      "Returns the last day of the month which the date belongs to",
      ExprChecks.unaryProjectInputMatchesOutput(TypeSig.DATE, TypeSig.DATE),
      (a, conf, p, r) => new UnaryExprMeta[LastDay](a, conf, p, r) {
        override def convertToGpu(expr: Expression): GpuExpression =
          GpuLastDay(expr)
      }),
    expr[FromUnixTime](
      "Get the string from a unix timestamp",
      ExprChecks.binaryProject(TypeSig.STRING, TypeSig.STRING,
        ("sec", TypeSig.LONG, TypeSig.LONG),
        ("format", TypeSig.lit(TypeEnum.STRING)
            .withPsNote(TypeEnum.STRING, "Only a limited number of formats are supported"),
            TypeSig.STRING)),
      (a, conf, p, r) => new FromUnixTimeMeta(a ,conf ,p ,r)),
    expr[FromUTCTimestamp](
      "Render the input UTC timestamp in the input timezone",
      ExprChecks.binaryProject(TypeSig.TIMESTAMP, TypeSig.TIMESTAMP,
        ("timestamp", TypeSig.TIMESTAMP, TypeSig.TIMESTAMP),
        ("timezone", TypeSig.lit(TypeEnum.STRING)
          .withPsNote(TypeEnum.STRING,
            "Only non-DST(Daylight Savings Time) timezones are supported"),
          TypeSig.lit(TypeEnum.STRING))),
      (a, conf, p, r) => new FromUTCTimestampExprMeta(a, conf, p, r)
    ),
    expr[ToUTCTimestamp](
      "Render the input timestamp in UTC",
      ExprChecks.binaryProject(TypeSig.TIMESTAMP, TypeSig.TIMESTAMP,
        ("timestamp", TypeSig.TIMESTAMP, TypeSig.TIMESTAMP),
        ("timezone", TypeSig.lit(TypeEnum.STRING)
          .withPsNote(TypeEnum.STRING,
            "Only non-DST(Daylight Savings Time) timezones are supported"),
          TypeSig.lit(TypeEnum.STRING))),
      (a, conf, p, r) => new ToUTCTimestampExprMeta(a, conf, p, r)
    ),
    expr[MonthsBetween](
      "If `timestamp1` is later than `timestamp2`, then the result " +
        "is positive. If `timestamp1` and `timestamp2` are on the same day of month, or both " +
        "are the last day of month, time of day will be ignored. Otherwise, the difference is " +
        "calculated based on 31 days per month, and rounded to 8 digits unless roundOff=false.",
      ExprChecks.projectOnly(TypeSig.DOUBLE, TypeSig.DOUBLE,
        Seq(ParamCheck("timestamp1", TypeSig.TIMESTAMP, TypeSig.TIMESTAMP),
          ParamCheck("timestamp2", TypeSig.TIMESTAMP, TypeSig.TIMESTAMP),
          ParamCheck("round", TypeSig.lit(TypeEnum.BOOLEAN), TypeSig.BOOLEAN))),
      (a, conf, p, r) => new MonthsBetweenExprMeta(a, conf, p, r)
    ),
    expr[TruncDate](
      "Truncate the date to the unit specified by the given string format",
      ExprChecks.binaryProject(TypeSig.DATE, TypeSig.DATE,
        ("date", TypeSig.DATE, TypeSig.DATE),
        ("format", TypeSig.STRING, TypeSig.STRING)),
      (a, conf, p, r) => new TruncDateExprMeta(a, conf, p, r)
    ),
    expr[TruncTimestamp](
      "Truncate the timestamp to the unit specified by the given string format",
      ExprChecks.binaryProject(TypeSig.TIMESTAMP, TypeSig.TIMESTAMP,
        ("format", TypeSig.STRING, TypeSig.STRING),
        ("date", TypeSig.TIMESTAMP, TypeSig.TIMESTAMP)),
      (a, conf, p, r) => new TruncTimestampExprMeta(a, conf, p, r)
    ),
  )
}
