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
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids._
import org.apache.spark.sql.rapids.catalyst.expressions.GpuRand
import org.apache.spark.sql.rapids.execution.python.GpuPythonUDF
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

private[rapids] object GpuCollectionExpressionOverrides {
  val rules: Seq[ExprRule[_ <: Expression]] = Seq(
    expr[PythonUDF](
      "UDF run in an external python process. Does not actually run on the GPU, but " +
          "the transfer of data to/from it can be accelerated",
      ExprChecks.fullAggAndProject(
        // Different types of Pandas UDF support different sets of output type. Please refer to
        //   https://github.com/apache/spark/blob/master/python/pyspark/sql/udf.py#L98
        // for more details.
        // It is impossible to specify the exact type signature for each Pandas UDF type in a single
        // expression 'PythonUDF'.
        // So use the 'unionOfPandasUdfOut' to cover all types for Spark. The type signature of
        // plugin is also an union of all the types of Pandas UDF.
        (TypeSig.commonCudfTypes + TypeSig.ARRAY).nested() + TypeSig.STRUCT,
        TypeSig.unionOfPandasUdfOut,
        repeatingParamCheck = Some(RepeatingParamCheck(
          "param",
          (TypeSig.commonCudfTypes + TypeSig.ARRAY + TypeSig.STRUCT).nested(),
          TypeSig.all))),
      (a, conf, p, r) => new ExprMeta[PythonUDF](a, conf, p, r) {
        override def replaceMessage: String = "not block GPU acceleration"
        override def noReplacementPossibleMessage(reasons: String): String =
          s"blocks running on GPU because $reasons"

        override def convertToGpuImpl(): GpuExpression =
          GpuPythonUDF(a.name, a.func, a.dataType,
            childExprs.map(_.convertToGpu()),
            a.evalType, a.udfDeterministic, a.resultId)
        }),
    GpuScalaUDFMeta.exprMeta,
    expr[Rand](
      "Generate a random column with i.i.d. uniformly distributed values in [0, 1)",
      ExprChecks.projectOnly(TypeSig.DOUBLE, TypeSig.DOUBLE,
        Seq(ParamCheck("seed",
          (TypeSig.INT + TypeSig.LONG).withAllLit(),
          (TypeSig.INT + TypeSig.LONG).withAllLit()))),
      (a, conf, p, r) => new UnaryExprMeta[Rand](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuRand(child, this.conf.isRetryContextCheckEnabled)
      }),
    expr[SparkPartitionID] (
      "Returns the current partition id",
      ExprChecks.projectOnly(TypeSig.INT, TypeSig.INT),
      (a, conf, p, r) => new ExprMeta[SparkPartitionID](a, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = GpuSparkPartitionID()
      }),
    expr[MonotonicallyIncreasingID] (
      "Returns monotonically increasing 64-bit integers",
      ExprChecks.projectOnly(TypeSig.LONG, TypeSig.LONG),
      (a, conf, p, r) => new ExprMeta[MonotonicallyIncreasingID](a, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = GpuMonotonicallyIncreasingID()
      }),
    expr[InputFileName] (
      "Returns the name of the file being read, or empty string if not available",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING),
      (a, conf, p, r) => new ExprMeta[InputFileName](a, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = GpuInputFileName()
      }),
    expr[InputFileBlockStart] (
      "Returns the start offset of the block being read, or -1 if not available",
      ExprChecks.projectOnly(TypeSig.LONG, TypeSig.LONG),
      (a, conf, p, r) => new ExprMeta[InputFileBlockStart](a, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = GpuInputFileBlockStart()
      }),
    expr[InputFileBlockLength] (
      "Returns the length of the block being read, or -1 if not available",
      ExprChecks.projectOnly(TypeSig.LONG, TypeSig.LONG),
      (a, conf, p, r) => new ExprMeta[InputFileBlockLength](a, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = GpuInputFileBlockLength()
      }),
    expr[Md5] (
      "MD5 hash operator",
      ExprChecks.unaryProject(TypeSig.STRING, TypeSig.STRING,
        TypeSig.BINARY, TypeSig.BINARY),
      (a, conf, p, r) => new UnaryExprMeta[Md5](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuMd5(child)
      }),
    expr[Sha1] (
      "Sha1 hash operator",
      ExprChecks.unaryProject(TypeSig.STRING, TypeSig.STRING,
        TypeSig.BINARY, TypeSig.BINARY),
      (a, conf, p, r) => new UnaryExprMeta[Sha1](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuSha1(child)
      }),
    expr[Sha2] (
      "Sha2 hash operator",
      ExprChecks.binaryProject(
        TypeSig.STRING,
        TypeSig.STRING,
        ("input", TypeSig.BINARY, TypeSig.BINARY),
        ("bitLength", TypeSig.lit(TypeEnum.INT), TypeSig.lit(TypeEnum.INT))),
      (a, conf, p, r) => new GpuSha2.Meta(a, conf, p, r)
    ),
    expr[Upper](
      "String uppercase operator",
      ExprChecks.unaryProjectInputMatchesOutput(TypeSig.STRING, TypeSig.STRING),
      (a, conf, p, r) => new UnaryExprMeta[Upper](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuUpper(child)
      })
      .incompat(CASE_MODIFICATION_INCOMPAT),
    expr[Lower](
      "String lowercase operator",
      ExprChecks.unaryProjectInputMatchesOutput(TypeSig.STRING, TypeSig.STRING),
      (a, conf, p, r) => new UnaryExprMeta[Lower](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuLower(child)
      })
      .incompat(CASE_MODIFICATION_INCOMPAT),
    expr[StringLPad](
      "Pad a string on the left",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING,
        Seq(ParamCheck("str", TypeSig.STRING, TypeSig.STRING),
          ParamCheck("len", TypeSig.lit(TypeEnum.INT), TypeSig.INT),
          ParamCheck("pad", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING))),
      (in, conf, p, r) => new TernaryExprMeta[StringLPad](in, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          extractLit(in.pad).foreach { padLit =>
            if (padLit.value != null &&
                padLit.value.asInstanceOf[UTF8String].toString.length != 1) {
              willNotWorkOnGpu("only a single character is supported for pad")
            }
          }
        }
        override def convertToGpu(
            str: Expression,
            width: Expression,
            pad: Expression): GpuExpression =
          GpuStringLPad(str, width, pad)
      }),
    expr[StringRPad](
      "Pad a string on the right",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING,
        Seq(ParamCheck("str", TypeSig.STRING, TypeSig.STRING),
          ParamCheck("len", TypeSig.lit(TypeEnum.INT), TypeSig.INT),
          ParamCheck("pad", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING))),
      (in, conf, p, r) => new TernaryExprMeta[StringRPad](in, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          extractLit(in.pad).foreach { padLit =>
            if (padLit.value != null &&
                padLit.value.asInstanceOf[UTF8String].toString.length != 1) {
              willNotWorkOnGpu("only a single character is supported for pad")
            }
          }
        }
        override def convertToGpu(
            str: Expression,
            width: Expression,
            pad: Expression): GpuExpression =
          GpuStringRPad(str, width, pad)
      }),
    expr[StringSplit](
       "Splits `str` around occurrences that match `regex`",
      // Java's split API produces different behaviors than cudf when splitting with empty pattern
      ExprChecks.projectOnly(TypeSig.ARRAY.nested(TypeSig.STRING),
        TypeSig.ARRAY.nested(TypeSig.STRING),
        Seq(ParamCheck("str", TypeSig.STRING, TypeSig.STRING),
          ParamCheck("regexp", TypeSig.lit(TypeEnum.STRING)
              .withPsNote(TypeEnum.STRING, "very limited subset of regex supported"),
            TypeSig.STRING),
          ParamCheck("limit", TypeSig.lit(TypeEnum.INT), TypeSig.INT))),
      (in, conf, p, r) => new GpuStringSplitMeta(in, conf, p, r)),
    expr[GetStructField](
      "Gets the named field of the struct",
      ExprChecks.unaryProject(
        (TypeSig.commonCudfTypes + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP + TypeSig.NULL +
            TypeSig.DECIMAL_128 + TypeSig.BINARY).nested(),
        TypeSig.all,
        TypeSig.STRUCT.nested(TypeSig.commonCudfTypes + TypeSig.ARRAY +
            TypeSig.STRUCT + TypeSig.MAP + TypeSig.NULL + TypeSig.DECIMAL_128 + TypeSig.BINARY),
        TypeSig.STRUCT.nested(TypeSig.all)),
      (expr, conf, p, r) => new UnaryExprMeta[GetStructField](expr, conf, p, r) {
        override def convertToGpu(arr: Expression): GpuExpression =
          GpuGetStructField(arr, expr.ordinal, expr.name)
      }),
    expr[GetArrayItem](
      "Gets the field at `ordinal` in the Array",
      ExprChecks.binaryProject(
        (TypeSig.commonCudfTypes + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.NULL +
            TypeSig.DECIMAL_128 + TypeSig.MAP + TypeSig.BINARY).nested(),
        TypeSig.all,
        ("array", TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.ARRAY +
            TypeSig.STRUCT + TypeSig.NULL + TypeSig.DECIMAL_128 + TypeSig.MAP + TypeSig.BINARY),
            TypeSig.ARRAY.nested(TypeSig.all)),
        ("ordinal", TypeSig.integral, TypeSig.integral)),
      (in, conf, p, r) => new BinaryExprMeta[GetArrayItem](in, conf, p, r) {
        override def convertToGpu(arr: Expression, ordinal: Expression): GpuExpression =
          GpuGetArrayItem(arr, ordinal, in.failOnError)
      }),
    expr[GetMapValue](
      "Gets Value from a Map based on a key",
      ExprChecks.binaryProject(
        (TypeSig.commonCudfTypes + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.NULL +
          TypeSig.DECIMAL_128 + TypeSig.MAP + TypeSig.BINARY).nested(),
        TypeSig.all,
        ("map", TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.ARRAY + TypeSig.STRUCT +
          TypeSig.NULL + TypeSig.DECIMAL_128 + TypeSig.MAP + TypeSig.BINARY),
          TypeSig.MAP.nested(TypeSig.all)),
        ("key", TypeSig.commonCudfTypes + TypeSig.DECIMAL_128, TypeSig.all)),
      (in, conf, p, r) => new GetMapValueMeta(in, conf, p, r){}),
    GpuElementAtMeta.elementAtRule(false),
    expr[MapKeys](
      "Returns an unordered array containing the keys of the map",
      ExprChecks.unaryProject(
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.BINARY).nested(),
        TypeSig.ARRAY.nested(TypeSig.all - TypeSig.MAP), // Maps cannot have other maps as keys
        TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP + TypeSig.BINARY),
        TypeSig.MAP.nested(TypeSig.all)),
      (in, conf, p, r) => new UnaryExprMeta[MapKeys](in, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuMapKeys(child)
      }),
    expr[MapValues](
      "Returns an unordered array containing the values of the map",
      ExprChecks.unaryProject(
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP + TypeSig.BINARY),
        TypeSig.ARRAY.nested(TypeSig.all),
        TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP + TypeSig.BINARY),
        TypeSig.MAP.nested(TypeSig.all)),
      (in, conf, p, r) => new UnaryExprMeta[MapValues](in, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuMapValues(child)
      }),
    expr[MapEntries](
      "Returns an unordered array of all entries in the given map",
      ExprChecks.unaryProject(
        // Technically the return type is an array of struct, but we cannot really express that
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP + TypeSig.BINARY),
        TypeSig.ARRAY.nested(TypeSig.all),
        TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP + TypeSig.BINARY),
        TypeSig.MAP.nested(TypeSig.all)),
      (in, conf, p, r) => new UnaryExprMeta[MapEntries](in, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuMapEntries(child)
      }),
    expr[MapFromEntries](
      "Creates a map from an array of entries (structs of key-value pairs)",
      ExprChecks.unaryProject(
        TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP + TypeSig.BINARY),
        TypeSig.MAP.nested(TypeSig.all),
        // Input is an array of struct<key, value>
        // The struct fields can contain the supported types
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP + TypeSig.BINARY),
        TypeSig.ARRAY.nested(TypeSig.all)),
      (in, conf, p, r) => new UnaryExprMeta[MapFromEntries](in, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          // Spark 4.1+ returns an enum value instead of String, so use toString first
          SQLConf.get.getConf(SQLConf.MAP_KEY_DEDUP_POLICY).toString.toUpperCase match {
            case "EXCEPTION" | "LAST_WIN" => // Good we can support this
            case other =>
              willNotWorkOnGpu(s"$other is not supported for config setting" +
                  s" ${SQLConf.MAP_KEY_DEDUP_POLICY.key}")
          }
        }
        override def convertToGpu(child: Expression): GpuExpression =
          GpuMapFromEntries(child)
      }),
    expr[StringToMap](
      "Creates a map after splitting the input string into pairs of key-value strings",
      // Java's split API produces different behaviors than cudf when splitting with empty pattern
      ExprChecks.projectOnly(TypeSig.MAP.nested(TypeSig.STRING), TypeSig.MAP.nested(TypeSig.STRING),
        Seq(ParamCheck("str", TypeSig.STRING, TypeSig.STRING),
          ParamCheck("pairDelim", TypeSig.lit(TypeEnum.STRING), TypeSig.lit(TypeEnum.STRING)),
          ParamCheck("keyValueDelim", TypeSig.lit(TypeEnum.STRING), TypeSig.lit(TypeEnum.STRING)))),
      (in, conf, p, r) => new GpuStringToMapMeta(in, conf, p, r)),
    expr[ArrayMin](
      "Returns the minimum value in the array",
      ExprChecks.unaryProject(
        TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL,
        TypeSig.orderable,
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
        TypeSig.ARRAY.nested(TypeSig.orderable)),
      (in, conf, p, r) => new UnaryExprMeta[ArrayMin](in, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuArrayMin(child)
      }),
    expr[ArrayMax](
      "Returns the maximum value in the array",
      ExprChecks.unaryProject(
        TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL,
        TypeSig.orderable,
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
        TypeSig.ARRAY.nested(TypeSig.orderable)),
      (in, conf, p, r) => new UnaryExprMeta[ArrayMax](in, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuArrayMax(child)
      }),
    expr[ArrayRepeat](
      "Returns the array containing the given input value (left) count (right) times",
      ExprChecks.binaryProject(
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL
          + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
        TypeSig.ARRAY.nested(TypeSig.all),
        ("left", (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL
          + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP).nested(), TypeSig.all),
        ("right", TypeSig.integral, TypeSig.integral)),
      (in, conf, p, r) => new BinaryExprMeta[ArrayRepeat](in, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuArrayRepeat(lhs, rhs)
      }
    ),
    expr[CreateNamedStruct](
      "Creates a struct with the given field names and values",
      CreateNamedStructCheck,
      (in, conf, p, r) => new ExprMeta[CreateNamedStruct](in, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression =
          GpuCreateNamedStruct(childExprs.map(_.convertToGpu()))
      }),
    expr[ArrayContains](
      "Returns a boolean if the array contains the passed in key",
      ExprChecks.binaryProject(
        TypeSig.BOOLEAN,
        TypeSig.BOOLEAN,
        ("array", TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.NULL),
          TypeSig.ARRAY.nested(TypeSig.all)),
        ("key", TypeSig.commonCudfTypes, TypeSig.all)),
      (in, conf, p, r) => new BinaryExprMeta[ArrayContains](in, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuArrayContains(lhs, rhs)
      }),
    expr[ArrayPosition](
      "Returns the (1-based) index of the first matching element of the array as long, " +
        "or 0 if no match is found.",
      ExprChecks.binaryProject(
        TypeSig.LONG,
        TypeSig.LONG,
        ("array", TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL
            + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.BINARY),
          TypeSig.ARRAY.nested(TypeSig.orderable)),
        ("key", (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.BINARY).nested(),
          TypeSig.orderable)),
      (in, conf, p, r) => new GpuArrayPositionMeta(in, conf, p, r)),
    expr[SortArray](
      "Returns a sorted array with the input array and the ascending / descending order",
      ExprChecks.binaryProject(
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            TypeSig.STRUCT),
        TypeSig.ARRAY.nested(TypeSig.all),
        ("array", TypeSig.ARRAY.nested(
          TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 + TypeSig.STRUCT),
            TypeSig.ARRAY.nested(TypeSig.all)),
        ("ascendingOrder", TypeSig.lit(TypeEnum.BOOLEAN), TypeSig.lit(TypeEnum.BOOLEAN))),
      (sortExpression, conf, p, r) => new BinaryExprMeta[SortArray](sortExpression, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression = {
          GpuSortArray(lhs, rhs)
        }
      }
    ),
    expr[ArraySort](
      "Sorts the input array in ascending order with nulls last according to the natural " +
          "ordering of the elements (the default comparator). A custom comparator, or a nested " +
          "element type (array or struct), falls back to the CPU",
      ExprChecks.projectOnly(
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128),
        TypeSig.ARRAY.nested(TypeSig.all),
        Seq(
          ParamCheck("array",
            TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128),
            TypeSig.ARRAY.nested(TypeSig.all)))),
      (in, conf, p, r) => new ExprMeta[ArraySort](in, conf, p, r) {
        // Wrap only the array; listSortRows applies the default comparator natively, so the
        // comparator lambda is excluded from children (neither converted nor type-checked).
        // This must wrap exactly one child to stay 1:1 with the single ParamCheck above, which
        // the framework pairs to childExprs positionally.
        override val childExprs: Seq[BaseExprMeta[_]] =
          Seq(GpuOverrides.wrapExpr(in.arguments.head, this.conf, Some(this)))

        override def tagExprForGpu(): Unit = {
          if (!GpuArraySort.isDefaultComparator(in)) {
            willNotWorkOnGpu("array_sort with a custom comparator function is not supported " +
                "on the GPU; only the default ordering (ascending, nulls last) is supported")
          }
        }
        override def convertToGpuImpl(): GpuExpression =
          GpuArraySort(childExprs.head.convertToGpu())
      }),
    expr[CreateArray](
      "Returns an array with the given elements",
      ExprChecks.projectOnly(
        TypeSig.ARRAY.nested(TypeSig.gpuNumeric +
          TypeSig.NULL + TypeSig.STRING + TypeSig.BOOLEAN + TypeSig.DATE + TypeSig.TIMESTAMP +
          TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.BINARY),
        TypeSig.ARRAY.nested(TypeSig.all),
        repeatingParamCheck = Some(RepeatingParamCheck("arg",
          TypeSig.gpuNumeric + TypeSig.NULL + TypeSig.STRING +
              TypeSig.BOOLEAN + TypeSig.DATE + TypeSig.TIMESTAMP + TypeSig.STRUCT + TypeSig.BINARY +
              TypeSig.ARRAY.nested(TypeSig.gpuNumeric + TypeSig.NULL + TypeSig.STRING +
                TypeSig.BOOLEAN + TypeSig.DATE + TypeSig.TIMESTAMP + TypeSig.STRUCT +
                  TypeSig.ARRAY + TypeSig.BINARY),
          TypeSig.all))),
      (in, conf, p, r) => new ExprMeta[CreateArray](in, conf, p, r) {

        override def tagExprForGpu(): Unit = {
          wrapped.dataType match {
            case ArrayType(ArrayType(ArrayType(_, _), _), _) =>
              willNotWorkOnGpu("Only support to create array or array of array, Found: " +
                s"${wrapped.dataType}")
            case _ =>
          }
        }

        override def convertToGpuImpl(): GpuExpression =
          GpuCreateArray(childExprs.map(_.convertToGpu()), wrapped.useStringTypeWhenEmpty)
      }),
    expr[ArrayDistinct](
      "Removes duplicate values from the array",
      ExprChecks.unaryProject(
        TypeSig.ARRAY.nested(TypeSig.orderable),
        TypeSig.ARRAY.nested(TypeSig.orderable),
        TypeSig.ARRAY.nested(TypeSig.orderable),
        TypeSig.ARRAY.nested(TypeSig.orderable)),
      GpuArrayDistinctMeta),
    expr[Flatten](
      "Creates a single array from an array of arrays",
      ExprChecks.unaryProject(
        TypeSig.ARRAY.nested(TypeSig.all),
        TypeSig.ARRAY.nested(TypeSig.all),
        TypeSig.ARRAY.nested(TypeSig.all),
        TypeSig.ARRAY.nested(TypeSig.all)),
      (a, conf, p, r) => new UnaryExprMeta[Flatten](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuFlattenArray(child)
      }),
    expr[LambdaFunction](
      "Holds a higher order SQL function",
      ExprChecks.projectOnly(
        (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL + TypeSig.BINARY +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP).nested(),
        TypeSig.all,
        Seq(ParamCheck("function",
          (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL + TypeSig.BINARY +
              TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP).nested(),
          TypeSig.all)),
        Some(RepeatingParamCheck("arguments",
          (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL + TypeSig.BINARY +
              TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP).nested(),
          TypeSig.all))),
      (in, conf, p, r) => new ExprMeta[LambdaFunction](in, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = {
          val func = childExprs.head
          val args = childExprs.tail
          GpuLambdaFunction(func.convertToGpu(),
            args.map(_.convertToGpu().asInstanceOf[NamedExpression]),
            in.hidden)
        }
      }),
    expr[NamedLambdaVariable](
      "A parameter to a higher order SQL function",
      ExprChecks.projectOnly(
        (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL + TypeSig.BINARY +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP).nested(),
        TypeSig.all),
      (in, conf, p, r) => new ExprMeta[NamedLambdaVariable](in, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = {
          GpuNamedLambdaVariable(in.name, in.dataType, in.nullable, in.exprId)
        }
      }),
    expr[ArrayTransform](
      "Transform elements in an array using the transform function. This is similar to a `map` " +
          "in functional programming",
      ExprChecks.projectOnly(TypeSig.ARRAY.nested(TypeSig.commonCudfTypes +
        TypeSig.DECIMAL_128 + TypeSig.NULL + TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT +
        TypeSig.MAP),
        TypeSig.ARRAY.nested(TypeSig.all),
        Seq(
          ParamCheck("argument",
            TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
                TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
            TypeSig.ARRAY.nested(TypeSig.all)),
          ParamCheck("function",
            (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
                TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP).nested(),
            TypeSig.all))),
      (in, conf, p, r) => new ExprMeta[ArrayTransform](in, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = {
          GpuArrayTransform(childExprs.head.convertToGpu(), childExprs(1).convertToGpu())
        }
      }),
     expr[ArrayExists](
      "Return true if any element satisfies the predicate LambdaFunction",
      ExprChecks.projectOnly(TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        Seq(
          ParamCheck("argument",
            TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
                TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
            TypeSig.ARRAY.nested(TypeSig.all)),
          ParamCheck("function", TypeSig.BOOLEAN, TypeSig.BOOLEAN))),
      (in, conf, p, r) => new ExprMeta[ArrayExists](in, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = {
          GpuArrayExists(
            childExprs.head.convertToGpu(),
            childExprs(1).convertToGpu(),
            SQLConf.get.getConf(SQLConf.LEGACY_ARRAY_EXISTS_FOLLOWS_THREE_VALUED_LOGIC)
          )
        }
      }),
    expr[ArrayFilter](
      "Filter an input array using a given predicate",
      ExprChecks.projectOnly(TypeSig.ARRAY.nested(TypeSig.commonCudfTypes +
        TypeSig.DECIMAL_128 + TypeSig.NULL + TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT +
        TypeSig.MAP),
        TypeSig.ARRAY.nested(TypeSig.all),
        Seq(
          ParamCheck("argument",
            TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
              TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
            TypeSig.ARRAY.nested(TypeSig.all)),
          ParamCheck("function", TypeSig.BOOLEAN, TypeSig.BOOLEAN))),
      (in, conf, p, r) => new ExprMeta[ArrayFilter](in, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = {
          GpuArrayFilter(
            childExprs.head.convertToGpu(),
            childExprs(1).convertToGpu()
          )
        }
      }),
    expr[ArrayAggregate](
      "Aggregate elements in an array using an accumulator function and finishing " +
          "transformation. Currently only lambdas of the form (acc, x) -> op(acc, g(x)) with " +
          "an identity finish are executed on the GPU, where op is one of SUM/PRODUCT/MAX/" +
          "MIN/ALL/ANY. If/CaseWhen branches are accepted as long as each branch is itself " +
          "op-of-acc (or bare acc) with op consistent across branches; other shapes fall " +
          "back to CPU. SUM/PRODUCT on float/double share the same parallel-reduction " +
          "non-determinism as GpuSum and are gated by " +
          "spark.rapids.sql.variableFloatAgg.enabled. SUM/PRODUCT on integer/decimal in " +
          "ANSI mode fall back to CPU because cuDF segmented reduce wraps on overflow " +
          "instead of raising.",
      ExprChecks.projectOnly(
        TypeSig.commonCudfTypes + TypeSig.DECIMAL_128,
        TypeSig.all,
        Seq(
          ParamCheck("argument",
            TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
                TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
            TypeSig.ARRAY.nested(TypeSig.all)),
          ParamCheck("zero",
            TypeSig.commonCudfTypes + TypeSig.DECIMAL_128,
            TypeSig.all),
          ParamCheck("merge",
            TypeSig.commonCudfTypes + TypeSig.DECIMAL_128,
            TypeSig.all),
          ParamCheck("finish",
            TypeSig.commonCudfTypes + TypeSig.DECIMAL_128,
            TypeSig.all))),
      (in, conf, p, r) => new GpuArrayAggregateMeta(in, conf, p, r)),
    // TODO: fix the signature https://github.com/NVIDIA/spark-rapids/issues/5327
    expr[ArraysZip](
      "Returns a merged array of structs in which the N-th struct contains" +
        " all N-th values of input arrays.",
      ExprChecks.projectOnly(TypeSig.ARRAY.nested(
        TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL + TypeSig.BINARY +
          TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
        TypeSig.ARRAY.nested(TypeSig.all),
        repeatingParamCheck = Some(RepeatingParamCheck("children",
          TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
            TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
          TypeSig.ARRAY.nested(TypeSig.all)))),
      (in, conf, p, r) => new ExprMeta[ArraysZip](in, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = {
          GpuArraysZip(childExprs.map(_.convertToGpu()))
        }
      }
    ),
    expr[ArrayExcept](
      "Returns an array of the elements in array1 but not in array2, without duplicates",
      ExprChecks.binaryProject(
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
        TypeSig.ARRAY.nested(TypeSig.all),
        ("array1",
            TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
            TypeSig.ARRAY.nested(TypeSig.all)),
        ("array2",
            TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
            TypeSig.ARRAY.nested(TypeSig.all))),
      (in, conf, p, r) => new BinaryExprMeta[ArrayExcept](in, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression = {
          GpuArrayExcept(lhs, rhs)
        }
      }
    ).incompat("the GPU implementation treats -0.0 and 0.0 as equal, but the CPU " +
        "implementation currently does not (see SPARK-39845). Also, Apache Spark " +
        "3.1.3 fixed issue SPARK-36741 where NaNs in these set like operators were " +
        "not treated as being equal. We have chosen to break with compatibility for " +
        "the older versions of Spark in this instance and handle NaNs the same as 3.1.3+"),
    expr[ArrayIntersect](
      "Returns an array of the elements in the intersection of array1 and array2, without" +
        " duplicates",
      ExprChecks.binaryProject(
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
        TypeSig.ARRAY.nested(TypeSig.all),
        ("array1",
            TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
            TypeSig.ARRAY.nested(TypeSig.all)),
        ("array2",
            TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
            TypeSig.ARRAY.nested(TypeSig.all))),
      (in, conf, p, r) => new BinaryExprMeta[ArrayIntersect](in, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression = {
          GpuArrayIntersect(lhs, rhs)
        }
      }
    ).incompat("the GPU implementation treats -0.0 and 0.0 as equal, but the CPU " +
        "implementation currently does not (see SPARK-39845). Also, Apache Spark " +
        "3.1.3 fixed issue SPARK-36741 where NaNs in these set like operators were " +
        "not treated as being equal. We have chosen to break with compatibility for " +
        "the older versions of Spark in this instance and handle NaNs the same as 3.1.3+"),
    expr[ArrayUnion](
      "Returns an array of the elements in the union of array1 and array2, without duplicates.",
      ExprChecks.binaryProject(
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
        TypeSig.ARRAY.nested(TypeSig.all),
        ("array1",
            TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
            TypeSig.ARRAY.nested(TypeSig.all)),
        ("array2",
            TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
            TypeSig.ARRAY.nested(TypeSig.all))),
      (in, conf, p, r) => new BinaryExprMeta[ArrayUnion](in, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression = {
          GpuArrayUnion(lhs, rhs)
        }
      }
    ).incompat("the GPU implementation treats -0.0 and 0.0 as equal, but the CPU " +
        "implementation currently does not (see SPARK-39845). Also, Apache Spark " +
        "3.1.3 fixed issue SPARK-36741 where NaNs in these set like operators were " +
        "not treated as being equal. We have chosen to break with compatibility for " +
        "the older versions of Spark in this instance and handle NaNs the same as 3.1.3+"),
    expr[ArraysOverlap](
      "Returns true if a1 contains at least a non-null element present also in a2. If the arrays " +
      "have no common element and they are both non-empty and either of them contains a null " +
      "element null is returned, false otherwise.",
      ExprChecks.binaryProject(TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        ("array1",
            TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
            TypeSig.ARRAY.nested(TypeSig.all)),
        ("array2",
            TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL),
            TypeSig.ARRAY.nested(TypeSig.all))),
      (in, conf, p, r) => new BinaryExprMeta[ArraysOverlap](in, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression = {
          GpuArraysOverlap(lhs, rhs)
        }
      }
    ).incompat("the GPU implementation treats -0.0 and 0.0 as equal, but the CPU " +
        "implementation currently does not (see SPARK-39845). Also, Apache Spark " +
        "3.1.3 fixed issue SPARK-36741 where NaNs in these set like operators were " +
        "not treated as being equal. We have chosen to break with compatibility for " +
        "the older versions of Spark in this instance and handle NaNs the same as 3.1.3+"),
    expr[ArrayRemove](
      "Returns the array after removing all elements that equal to the input element (right) " +
      "from the input array (left)",
      ExprChecks.binaryProject(
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
          TypeSig.STRUCT),
        TypeSig.ARRAY.nested(TypeSig.all),
        ("array",
          TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
            TypeSig.STRUCT),
          TypeSig.all),
        ("element",
          (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
            TypeSig.STRUCT).nested(),
          TypeSig.all)),
      (in, conf, p, r) => new BinaryExprMeta[ArrayRemove](in, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuArrayRemove(lhs, rhs)
      }
    ),
    expr[MapFromArrays](
      "Creates a new map from two arrays",
      ExprChecks.binaryProject(
        TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
          TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
        // Map values may themselves be maps. Map keys remain constrained by the key-array check.
        TypeSig.MAP.nested(TypeSig.all),
        ("keys",
          TypeSig.ARRAY.nested(
            TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
              TypeSig.ARRAY + TypeSig.STRUCT),
          TypeSig.ARRAY.nested(TypeSig.all - TypeSig.MAP)),
        ("values",
          TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
          TypeSig.ARRAY.nested(TypeSig.all))),
      GpuMapFromArraysMeta
    ),
    expr[TransformKeys](
      "Transform keys in a map using a transform function",
      ExprChecks.projectOnly(TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
          TypeSig.NULL + TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
        TypeSig.MAP.nested(TypeSig.all),
        Seq(
          ParamCheck("argument",
            TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
                TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
            TypeSig.MAP.nested(TypeSig.all)),
          ParamCheck("function",
            // We need to be able to check for duplicate keys (equality)
            TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL,
            TypeSig.all - TypeSig.MAP.nested()))),
      (in, conf, p, r) => new ExprMeta[TransformKeys](in, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          // Spark 4.1+ returns an enum value instead of String, so use toString first
          SQLConf.get.getConf(SQLConf.MAP_KEY_DEDUP_POLICY).toString.toUpperCase match {
            case "EXCEPTION"| "LAST_WIN" => // Good we can support this
            case other =>
              willNotWorkOnGpu(s"$other is not supported for config setting" +
                  s" ${SQLConf.MAP_KEY_DEDUP_POLICY.key}")
          }
        }
        override def convertToGpuImpl(): GpuExpression = {
          GpuTransformKeys(childExprs.head.convertToGpu(), childExprs(1).convertToGpu())
        }
      }),
    expr[TransformValues](
      "Transform values in a map using a transform function",
      ExprChecks.projectOnly(TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
          TypeSig.NULL + TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
        TypeSig.MAP.nested(TypeSig.all),
        Seq(
          ParamCheck("argument",
            TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
                TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
            TypeSig.MAP.nested(TypeSig.all)),
          ParamCheck("function",
            (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
                TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP).nested(),
            TypeSig.all))),
      (in, conf, p, r) => new ExprMeta[TransformValues](in, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = {
          GpuTransformValues(childExprs.head.convertToGpu(), childExprs(1).convertToGpu())
        }
      }),
    expr[MapZipWith](
      "Filters entries in a map using the function",
      ExprChecks.projectOnly(TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
          TypeSig.NULL + TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
        TypeSig.MAP.nested(TypeSig.all),
        Seq(
          ParamCheck("argument1",
            TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
                TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
            TypeSig.MAP.nested(TypeSig.all)),
          ParamCheck("argument2",
            TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
                TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
            TypeSig.MAP.nested(TypeSig.all)),
          ParamCheck("function",
            (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
                TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP).nested(),
            TypeSig.all))),
      (in, conf, p, r) => new ExprMeta[MapZipWith](in, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = {
          GpuMapZipWith(childExprs.head.convertToGpu(),
          childExprs(1).convertToGpu(), childExprs(2).convertToGpu())
        }
      }),
    expr[MapFilter](
      "Filters entries in a map using the function",
      ExprChecks.projectOnly(TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
          TypeSig.NULL + TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
        TypeSig.MAP.nested(TypeSig.all),
        Seq(
          ParamCheck("argument",
            TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
                TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
            TypeSig.MAP.nested(TypeSig.all)),
          ParamCheck("function", TypeSig.BOOLEAN, TypeSig.BOOLEAN))),
      (in, conf, p, r) => new ExprMeta[MapFilter](in, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression = {
          GpuMapFilter(childExprs.head.convertToGpu(), childExprs(1).convertToGpu())
        }
      }),
  )
}
