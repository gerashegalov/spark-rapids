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
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids._
import org.apache.spark.sql.rapids.aggregate._
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

private[rapids] object GpuStringAndAggregateExpressionOverrides {
  val rules: Seq[ExprRule[_ <: Expression]] = Seq(
    expr[StringLocate](
      "Substring search operator",
      ExprChecks.projectOnly(TypeSig.INT, TypeSig.INT,
        Seq(ParamCheck("substr", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING),
          ParamCheck("str", TypeSig.STRING, TypeSig.STRING),
          ParamCheck("start", TypeSig.lit(TypeEnum.INT), TypeSig.INT))),
      (in, conf, p, r) => new TernaryExprMeta[StringLocate](in, conf, p, r) {
        override def convertToGpu(
            val0: Expression,
            val1: Expression,
            val2: Expression): GpuExpression =
          GpuStringLocate(val0, val1, val2)
      }),
    expr[StringInstr](
      "Instr string operator",
      ExprChecks.projectOnly(TypeSig.INT, TypeSig.INT,
        Seq(ParamCheck("str", TypeSig.STRING, TypeSig.STRING),
            ParamCheck("substr", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING))),
      (in, conf, p, r) => new BinaryExprMeta[StringInstr](in, conf, p, r) {
        override def convertToGpu(
            str: Expression,
            substr: Expression): GpuExpression =
          GpuStringInstr(str, substr)
      }),
    expr[Substring](
      "Substring operator",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING + TypeSig.BINARY,
        Seq(ParamCheck("str", TypeSig.STRING, TypeSig.STRING + TypeSig.BINARY),
          ParamCheck("pos", TypeSig.INT, TypeSig.INT),
          ParamCheck("len", TypeSig.INT, TypeSig.INT))),
      (in, conf, p, r) => new TernaryExprMeta[Substring](in, conf, p, r) {
        override def convertToGpu(
            column: Expression,
            position: Expression,
            length: Expression): GpuExpression =
          GpuSubstring(column, position, length)
      }),
    expr[SubstringIndex](
      "substring_index operator",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING,
        Seq(ParamCheck("str", TypeSig.STRING, TypeSig.STRING),
          ParamCheck("delim", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING),
          ParamCheck("count", TypeSig.lit(TypeEnum.INT), TypeSig.INT))),
      (in, conf, p, r) => new SubstringIndexMeta(in, conf, p, r)),
    expr[StringRepeat](
      "StringRepeat operator that repeats the given strings with numbers of times " +
        "given by repeatTimes",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING,
        Seq(ParamCheck("input", TypeSig.STRING, TypeSig.STRING),
          ParamCheck("repeatTimes", TypeSig.INT, TypeSig.INT))),
      (in, conf, p, r) => new BinaryExprMeta[StringRepeat](in, conf, p, r) {
        override def convertToGpu(
            input: Expression,
            repeatTimes: Expression): GpuExpression = GpuStringRepeat(input, repeatTimes)
      }),
    expr[StringReplace](
      "StringReplace operator",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING,
        Seq(ParamCheck("src", TypeSig.STRING, TypeSig.STRING),
          ParamCheck("search", TypeSig.STRING, TypeSig.STRING),
          ParamCheck("replace", TypeSig.STRING, TypeSig.STRING))),
      (in, conf, p, r) => new TernaryExprMeta[StringReplace](in, conf, p, r) {
        override def convertToGpu(
            column: Expression,
            target: Expression,
            replace: Expression): GpuExpression =
          GpuStringReplace(column, target, replace)
      }),
    expr[StringTrim](
      "StringTrim operator",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING,
        Seq(ParamCheck("src", TypeSig.STRING, TypeSig.STRING)),
        // Should really be an OptionalParam
        Some(RepeatingParamCheck("trimStr", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING))),
      (in, conf, p, r) => new String2TrimExpressionMeta[StringTrim](in, conf, p, r) {
        override def convertToGpu(
            column: Expression,
            target: Option[Expression] = None): GpuExpression =
          GpuStringTrim(column, target)
      }),
    expr[StringTrimLeft](
      "StringTrimLeft operator",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING,
        Seq(ParamCheck("src", TypeSig.STRING, TypeSig.STRING)),
        // Should really be an OptionalParam
        Some(RepeatingParamCheck("trimStr", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING))),
      (in, conf, p, r) =>
        new String2TrimExpressionMeta[StringTrimLeft](in, conf, p, r) {
          override def convertToGpu(
            column: Expression,
            target: Option[Expression] = None): GpuExpression =
            GpuStringTrimLeft(column, target)
        }),
    expr[StringTrimRight](
      "StringTrimRight operator",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING,
        Seq(ParamCheck("src", TypeSig.STRING, TypeSig.STRING)),
        // Should really be an OptionalParam
        Some(RepeatingParamCheck("trimStr", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING))),
      (in, conf, p, r) =>
        new String2TrimExpressionMeta[StringTrimRight](in, conf, p, r) {
          override def convertToGpu(
              column: Expression,
              target: Option[Expression] = None): GpuExpression =
            GpuStringTrimRight(column, target)
        }),
    expr[StringTranslate](
      "StringTranslate operator",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING,
        Seq(ParamCheck("input", TypeSig.STRING, TypeSig.STRING),
          ParamCheck("from", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING),
          ParamCheck("to", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING))),
      (in, conf, p, r) => new TernaryExprMeta[StringTranslate](in, conf, p, r) {
        override def convertToGpu(
            input: Expression,
            from: Expression,
            to: Expression): GpuExpression =
          GpuStringTranslate(input, from, to)
      }).incompat("the GPU implementation supports all unicode code points. In Spark versions " +
          "< 3.2.0, translate() does not support unicode characters with code point >= U+10000 " +
          "(See SPARK-34094)"),
    expr[StartsWith](
      "Starts with",
      ExprChecks.binaryProject(TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        ("src", TypeSig.STRING, TypeSig.STRING),
        ("search", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING)),
      (a, conf, p, r) => new BinaryExprMeta[StartsWith](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuStartsWith(lhs, rhs)
      }),
    expr[EndsWith](
      "Ends with",
      ExprChecks.binaryProject(TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        ("src", TypeSig.STRING, TypeSig.STRING),
        ("search", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING)),
      (a, conf, p, r) => new BinaryExprMeta[EndsWith](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuEndsWith(lhs, rhs)
      }),
    expr[Concat](
      "List/String concatenate",
      ExprChecks.projectOnly((TypeSig.STRING + TypeSig.ARRAY).nested(
        TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP + TypeSig.BINARY),
        (TypeSig.STRING + TypeSig.BINARY + TypeSig.ARRAY).nested(TypeSig.all),
        repeatingParamCheck = Some(RepeatingParamCheck("input",
          (TypeSig.STRING + TypeSig.ARRAY).nested(
            TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
                TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP + TypeSig.BINARY),
          (TypeSig.STRING + TypeSig.BINARY + TypeSig.ARRAY).nested(TypeSig.all)))),
      (a, conf, p, r) => new ComplexTypeMergingExprMeta[Concat](a, conf, p, r) {
        override def convertToGpu(child: Seq[Expression]): GpuExpression = GpuConcat(child)
      }),
    expr[Conv](
      desc = "Convert string representing a number from one base to another",
      pluginChecks = ExprChecks.projectOnly(
        outputCheck = TypeSig.STRING,
        paramCheck = Seq(
          ParamCheck(
            name = "num",
            cudf = TypeSig.STRING,
            spark = TypeSig.STRING),
          ParamCheck(
            name = "from_base",
            cudf = TypeSig.INT,
            spark = TypeSig.INT),
          ParamCheck(
            name = "to_base",
            cudf = TypeSig.INT,
            spark = TypeSig.INT)),
        sparkOutputSig = TypeSig.STRING),
      (convExpr, conf, parentMetaOpt, dataFromReplacementRule) =>
        new GpuConvMeta(convExpr, conf, parentMetaOpt, dataFromReplacementRule)
    ),
    expr[FormatNumber](
      "Formats the number x like '#,###,###.##', rounded to d decimal places.",
      ExprChecks.binaryProject(TypeSig.STRING, TypeSig.STRING,
        ("x", TypeSig.gpuNumeric, TypeSig.cpuNumeric),
        ("d", TypeSig.lit(TypeEnum.INT), TypeSig.INT+TypeSig.STRING)),
      (in, conf, p, r) => new BinaryExprMeta[FormatNumber](in, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          in.children.head.dataType match {
            case FloatType | DoubleType if !this.conf.isFloatFormatNumberEnabled =>
              willNotWorkOnGpu("format_number with floating point types on the GPU returns " +
                  "results that have a different precision than the default results of Spark. " +
                  "To enable this operation on the GPU, set" +
                  s" ${RapidsConf.ENABLE_FLOAT_FORMAT_NUMBER} to true.")
            case _ =>
          }
        }
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuFormatNumber(lhs, rhs)
      }
    ),
    expr[MapConcat](
      "Returns the union of all the given maps",
      ExprChecks.projectOnly(TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
          TypeSig.NULL + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
        TypeSig.MAP.nested(TypeSig.all),
        repeatingParamCheck = Some(RepeatingParamCheck("input",
          TypeSig.MAP.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
          TypeSig.NULL + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
          TypeSig.MAP.nested(TypeSig.all)))),
      (a, conf, p, r) => new ComplexTypeMergingExprMeta[MapConcat](a, conf, p, r) {
        override def convertToGpu(child: Seq[Expression]): GpuExpression = GpuMapConcat(child)
      }),
    expr[Slice](
      "Subsets array x starting from index start (array indices start at 1, " +
        "or starting from the end if start is negative) with the specified length.",
      ExprChecks.projectOnly(TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
          TypeSig.NULL + TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
        TypeSig.ARRAY.nested(TypeSig.all),
        Seq(
          ParamCheck("x",
            TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
                TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
            TypeSig.ARRAY.nested(TypeSig.all)),
          ParamCheck("start", TypeSig.INT, TypeSig.INT),
          ParamCheck("length", TypeSig.INT, TypeSig.INT))),
      (in, conf, p, r) => new TernaryExprMeta[Slice](in, conf, p, r) {
        override def convertToGpu(
            x: Expression,
            start: Expression,
            length: Expression): GpuExpression =
          GpuSlice(x, start, length)
      }),
    expr[ArrayJoin](
      "Concatenates the elements of the given array using the delimiter and an optional " +
        "string to replace nulls. If no value is set for nullReplacement, any null value " +
        "is filtered.",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING,
        Seq(ParamCheck("array",
          TypeSig.ARRAY.nested(TypeSig.STRING),
          TypeSig.ARRAY.nested(TypeSig.STRING)),
          ParamCheck("delimiter",
            TypeSig.STRING,
            TypeSig.STRING)),
        repeatingParamCheck = Some(RepeatingParamCheck("nullReplacement",
          TypeSig.lit(TypeEnum.STRING),
          TypeSig.STRING))),
      (a, conf, p, r) => new ExprMeta[ArrayJoin](a, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          if (a.children.size > 3) {
            willNotWorkOnGpu(s"array_join has more parameters than we expected " +
              s"to see. Found ${a.children.size}")
          }
        }
        override def convertToGpuImpl(): GpuExpression =
          GpuArrayJoin(childExprs.map(_.convertToGpu()))
      }
    ),
    expr[ConcatWs](
      "Concatenates multiple input strings or array of strings into a single " +
        "string using a given separator",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING,
        repeatingParamCheck = Some(RepeatingParamCheck("input",
          (TypeSig.STRING + TypeSig.ARRAY).nested(TypeSig.STRING),
          (TypeSig.STRING + TypeSig.ARRAY).nested(TypeSig.STRING)))),
      (a, conf, p, r) => new ExprMeta[ConcatWs](a, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          if (a.children.size <= 1) {
            // If only a separator specified and its a column, Spark returns an empty
            // string for all entries unless they are null, then it returns null.
            // This seems like edge case so instead of handling on GPU just fallback.
            willNotWorkOnGpu("Only specifying separator column not supported on GPU")
          }
        }
        override final def convertToGpuImpl(): GpuExpression =
          GpuConcatWs(childExprs.map(_.convertToGpu()))
      }),
    expr[Murmur3Hash](
      "Murmur3 hash operator",
      HashExprChecks.murmur3ProjectChecks,
      Murmur3HashExprMeta.apply),
    expr[XxHash64](
      "xxhash64 hash operator",
      HashExprChecks.xxhash64ProjectChecks,
      XxHash64ExprMeta.apply),
    expr[HiveHash](
      "hive hash operator",
      ExprChecks.projectOnly(TypeSig.INT, TypeSig.INT,
        repeatingParamCheck = Some(RepeatingParamCheck("input",
          (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.STRUCT + TypeSig.ARRAY).nested() +
              TypeSig.psNote(TypeEnum.ARRAY, "The nesting depth has a certain limit") +
              TypeSig.psNote(TypeEnum.STRUCT, "The nesting depth has a certain limit"),
          TypeSig.all))),
      (a, conf, p, r) => new ExprMeta[HiveHash](a, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          def getMaxStackDepth(inputType: DataType): Int = {
            inputType match {
              case at: ArrayType => 1 + getMaxStackDepth(at.elementType)
              case st: StructType =>
                1 + st.map(f => getMaxStackDepth(f.dataType)).max
              case _ => 0 // primitive types
            }
          }
          val maxDepth = a.children.map(c => getMaxStackDepth(c.dataType)).max
          val supportedDepth = XxHash64Utils.MAX_STACK_DEPTH
          if (maxDepth > supportedDepth) {
            willNotWorkOnGpu(s"the data type requires a stack size of $maxDepth, " +
              s"which exceeds the GPU limit of $supportedDepth")
          }
        }

        def convertToGpuImpl(): GpuExpression =
          GpuHiveHash(childExprs.map(_.convertToGpu()))
      }),
    expr[Contains](
      "Contains",
      ExprChecks.binaryProject(TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        ("src", TypeSig.STRING, TypeSig.STRING),
        ("search", TypeSig.STRING, TypeSig.STRING)),
      (a, conf, p, r) => new BinaryExprMeta[Contains](a, conf, p, r) {
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuContains(lhs, rhs)
      }),
    expr[Like](
      "Like",
      ExprChecks.binaryProject(TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        ("src", TypeSig.STRING, TypeSig.STRING),
        ("search", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING)),
      (a, conf, p, r) => new BinaryExprMeta[Like](a, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          a.right match {
            case Literal(v: UTF8String, _) =>
              val pattern = v.toString
              val esc = a.escapeChar
              var i = 0
              while (i < pattern.length) {
                if (pattern.charAt(i) == esc) {
                  val j = i + 1
                  if (j >= pattern.length) {
                    willNotWorkOnGpu(
                      "invalid LIKE escape pattern")
                    return
                  }
                  val c = pattern.charAt(j)
                  if (c != '_' && c != '%' && c != esc) {
                    willNotWorkOnGpu(
                      "invalid LIKE escape pattern")
                    return
                  }
                  i = j + 1
                } else {
                  i += 1
                }
              }
            case _ =>
          }
        }
        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuLike(lhs, rhs, a.escapeChar)
      }),
    expr[RLike](
      "Regular expression version of Like",
      ExprChecks.binaryProject(TypeSig.BOOLEAN, TypeSig.BOOLEAN,
        ("str", TypeSig.STRING, TypeSig.STRING),
        ("regexp", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING)),
      (a, conf, p, r) => new GpuRLikeMeta(a, conf, p, r)),
    expr[RegExpReplace](
      "String replace using a regular expression pattern",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING,
        Seq(ParamCheck("str", TypeSig.STRING, TypeSig.STRING),
          ParamCheck("regex", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING),
          ParamCheck("rep", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING),
          ParamCheck("pos", TypeSig.lit(TypeEnum.INT)
              .withPsNote(TypeEnum.INT, "only a value of 1 is supported"),
            TypeSig.lit(TypeEnum.INT)))),
      (a, conf, p, r) => new GpuRegExpReplaceMeta(a, conf, p, r)),
    expr[RegExpExtract](
      "Extract a specific group identified by a regular expression",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING,
        Seq(ParamCheck("str", TypeSig.STRING, TypeSig.STRING),
          ParamCheck("regexp", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING),
          ParamCheck("idx", TypeSig.lit(TypeEnum.INT),
            TypeSig.lit(TypeEnum.INT)))),
      (a, conf, p, r) => new GpuRegExpExtractMeta(a, conf, p, r)),
    expr[RegExpExtractAll](
      "Extract all strings matching a regular expression corresponding to the regex group index",
      ExprChecks.projectOnly(TypeSig.ARRAY.nested(TypeSig.STRING),
        TypeSig.ARRAY.nested(TypeSig.STRING),
        Seq(ParamCheck("str", TypeSig.STRING, TypeSig.STRING),
          ParamCheck("regexp", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING),
          ParamCheck("idx", TypeSig.lit(TypeEnum.INT), TypeSig.INT))),
      (a, conf, p, r) => new GpuRegExpExtractAllMeta(a, conf, p, r)),
    expr[ParseUrl](
      "Extracts a part from a URL",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING,
        Seq(ParamCheck("url", TypeSig.STRING, TypeSig.STRING),
          ParamCheck("partToExtract", TypeSig.lit(TypeEnum.STRING).withPsNote(
            TypeEnum.STRING, "only support partToExtract = PROTOCOL | HOST | QUERY | PATH"),
            TypeSig.STRING)),
          // Should really be an OptionalParam
          Some(RepeatingParamCheck("key", TypeSig.STRING, TypeSig.STRING))),
      (a, conf, p, r) => new ExprMeta[ParseUrl](a, conf, p, r) {
        override def tagExprForGpu(): Unit = {

          extractStringLit(a.children(1)) match {
            // In Spark, the key in parse_url could act like a regex, but GPU will match the key
            // exactly. When key is literal, GPU will check if the key contains regex special and
            // fallbcak to CPU if it does, but we are not able to fallback when key is column.
            // see Spark issue: https://issues.apache.org/jira/browse/SPARK-44500
            case Some("QUERY") if (a.children.size == 3) => {
              extractLit(a.children(2)).foreach { key =>
                if (key.value != null) {
                  val keyStr = key.value.asInstanceOf[UTF8String].toString
                  if (regexMetaChars.exists(keyStr.contains(_))) {
                    willNotWorkOnGpu(s"Key $keyStr could act like a regex which is not " +
                        "supported on GPU")
                  }
                }
              }
            }
            case Some(part) if GpuParseUrl.isSupportedPart(part) =>
            case Some(other) =>
              willNotWorkOnGpu(s"Part to extract $other is not supported on GPU")
            case None =>
              // Should never get here, but just in case
              willNotWorkOnGpu("GPU only supports a literal for the part to extract")
          }
        }

        override def convertToGpuImpl(): GpuExpression = {
          GpuParseUrl(childExprs.map(_.convertToGpu()), a.failOnError)
        }
      }),
    expr[Length](
      "String character length or binary byte length",
      ExprChecks.unaryProject(TypeSig.INT, TypeSig.INT,
        TypeSig.STRING, TypeSig.STRING + TypeSig.BINARY),
      (a, conf, p, r) => new UnaryExprMeta[Length](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuLength(child)
      }),
    expr[Size](
      "The size of an array or a map",
      ExprChecks.unaryProject(TypeSig.INT, TypeSig.INT,
        (TypeSig.ARRAY + TypeSig.MAP).nested(TypeSig.commonCudfTypes + TypeSig.NULL
            + TypeSig.DECIMAL_128 + TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
        (TypeSig.ARRAY + TypeSig.MAP).nested(TypeSig.all)),
      (a, conf, p, r) => new UnaryExprMeta[Size](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuSize(child, a.legacySizeOfNull)
      }),
    expr[Reverse](
      "Returns a reversed string or an array with reverse order of elements",
      ExprChecks.unaryProject(TypeSig.STRING + TypeSig.ARRAY.nested(TypeSig.all),
        TypeSig.STRING + TypeSig.ARRAY.nested(TypeSig.all),
        TypeSig.STRING + TypeSig.ARRAY.nested(TypeSig.all),
        TypeSig.STRING + TypeSig.ARRAY.nested(TypeSig.all)),
      (a, conf, p, r) => new UnaryExprMeta[Reverse](a, conf, p, r) {
        override def convertToGpu(input: Expression): GpuExpression =
          GpuReverse(input)
      }),
    expr[UnscaledValue](
      "Convert a Decimal to an unscaled long value for some aggregation optimizations",
      ExprChecks.unaryProject(TypeSig.LONG, TypeSig.LONG,
        TypeSig.DECIMAL_64, TypeSig.DECIMAL_128),
      (a, conf, p, r) => new UnaryExprMeta[UnscaledValue](a, conf, p, r) {
        override val isFoldableNonLitAllowed: Boolean = true
        override def convertToGpu(child: Expression): GpuExpression = GpuUnscaledValue(child)
      }),
    expr[MakeDecimal](
      "Create a Decimal from an unscaled long value for some aggregation optimizations",
      ExprChecks.unaryProject(TypeSig.DECIMAL_64, TypeSig.DECIMAL_128,
        TypeSig.LONG, TypeSig.LONG),
      (a, conf, p, r) => new UnaryExprMeta[MakeDecimal](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression =
          GpuMakeDecimal(child, a.precision, a.scale, a.nullOnOverflow)
      }),
    expr[Explode](
      "Given an input array produces a sequence of rows for each value in the array",
      ExprChecks.unaryProject(
        // Here is a walk-around representation, since multi-level nested type is not supported yet.
        // related issue: https://github.com/NVIDIA/spark-rapids/issues/1901
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
        TypeSig.ARRAY.nested(TypeSig.all),
        (TypeSig.ARRAY + TypeSig.MAP).nested(TypeSig.commonCudfTypes + TypeSig.NULL +
            TypeSig.DECIMAL_128 + TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
        (TypeSig.ARRAY + TypeSig.MAP).nested(TypeSig.all)),
      (a, conf, p, r) => new GeneratorExprMeta[Explode](a, conf, p, r) {
        override val supportOuter: Boolean = true
        override def convertToGpuImpl(): GpuExpression = GpuExplode(childExprs.head.convertToGpu())
      }),
    expr[PosExplode](
      "Given an input array produces a sequence of rows for each value in the array",
      ExprChecks.unaryProject(
        // Here is a walk-around representation, since multi-level nested type is not supported yet.
        // related issue: https://github.com/NVIDIA/spark-rapids/issues/1901
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
        TypeSig.ARRAY.nested(TypeSig.all),
        (TypeSig.ARRAY + TypeSig.MAP).nested(TypeSig.commonCudfTypes + TypeSig.NULL +
            TypeSig.DECIMAL_128 + TypeSig.BINARY + TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
        (TypeSig.ARRAY + TypeSig.MAP).nested(TypeSig.all)),
      (a, conf, p, r) => new GeneratorExprMeta[PosExplode](a, conf, p, r) {
        override val supportOuter: Boolean = true
        override def convertToGpuImpl(): GpuExpression =
          GpuPosExplode(childExprs.head.convertToGpu())
      }),
    expr[Stack](
      "Separates expr1, ..., exprk into n rows.",
      ExprChecks.projectOnly(
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP),
        TypeSig.ARRAY.nested(TypeSig.all),
        Seq(ParamCheck("n", TypeSig.lit(TypeEnum.INT), TypeSig.INT)),
        Some(RepeatingParamCheck("expr",
          (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
              TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP).nested(),
          TypeSig.all))),
      (a, conf, p, r) => new GpuStackMeta(a, conf, p, r)
    ),
    expr[ReplicateRows](
      "Given an input row replicates the row N times",
      ExprChecks.projectOnly(
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
            TypeSig.ARRAY + TypeSig.STRUCT),
        TypeSig.ARRAY.nested(TypeSig.all),
        repeatingParamCheck = Some(RepeatingParamCheck("input",
          (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 +
              TypeSig.ARRAY + TypeSig.STRUCT).nested(),
          TypeSig.all))),
      (a, conf, p, r) => new ReplicateRowsExprMeta[ReplicateRows](a, conf, p, r) {
        override def convertToGpu(childExpr: Seq[Expression]): GpuExpression =
          GpuReplicateRows(childExpr)
      }),
    expr[CollectList](
      "Collect a list of non-unique elements, not supported in reduction",
      ExprChecks.fullAgg(
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.BINARY +
            TypeSig.NULL + TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP)
            .withPsNote(TypeEnum.ARRAY, "window operations are disabled by default due " +
                "to extreme memory usage"),
        TypeSig.ARRAY.nested(TypeSig.all),
        Seq(ParamCheck("input",
          (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.BINARY +
              TypeSig.NULL + TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP).nested(),
          TypeSig.all))),
      (c, conf, p, r) => new TypedImperativeAggExprMeta[CollectList](c, conf, p, r) {
        override def tagAggForGpu(): Unit = {
          if (context == WindowAggExprContext && !this.conf.isWindowCollectListEnabled) {
            willNotWorkOnGpu("collect_list is disabled for window operations because " +
                "the output explodes in size proportional to the window size squared. If " +
                "you know the window is small you can try it by setting " +
                s"${RapidsConf.ENABLE_WINDOW_COLLECT_LIST} to true")
          }
        }

        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression =
          GpuCollectList(childExprs.head, c.mutableAggBufferOffset, c.inputAggBufferOffset,
            TypeUtilsShims.collectListIgnoreNulls(c))

        override def aggBufferAttribute: AttributeReference = {
          val aggBuffer = c.aggBufferAttributes.head
          aggBuffer.copy(dataType = c.dataType)(aggBuffer.exprId, aggBuffer.qualifier)
        }

        override def createCpuToGpuBufferConverter(): CpuToGpuAggregateBufferConverter =
          new CpuToGpuCollectBufferConverter(c.child.dataType,
            !TypeUtilsShims.collectListIgnoreNulls(c))

        override def createGpuToCpuBufferConverter(): GpuToCpuAggregateBufferConverter =
          new GpuToCpuCollectBufferConverter()

        override val supportBufferConversion: Boolean = true

        // Last does not overflow, so it doesn't need the ANSI check
        override val needsAnsiCheck: Boolean = false
      }),
    expr[CollectSet](
      "Collect a set of unique elements, not supported in reduction",
      ExprChecks.fullAgg(
        TypeSig.ARRAY.nested(TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
            TypeSig.NULL + TypeSig.STRUCT + TypeSig.ARRAY)
            .withPsNote(TypeEnum.ARRAY, "window operations are disabled by default due " +
                "to extreme memory usage"),
        TypeSig.ARRAY.nested(TypeSig.all),
        Seq(ParamCheck("input",
          (TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 +
            TypeSig.NULL +
            TypeSig.STRUCT +
            TypeSig.ARRAY).nested(),
          TypeSig.all))),
      (c, conf, p, r) => new TypedImperativeAggExprMeta[CollectSet](c, conf, p, r) {
        override def tagAggForGpu(): Unit = {
          if (context == WindowAggExprContext && !this.conf.isWindowCollectSetEnabled) {
            willNotWorkOnGpu("collect_set is disabled for window operations because " +
                "the output can explode in size proportional to the window size squared. If " +
                "you know the window is small you can try it by setting " +
                s"${RapidsConf.ENABLE_WINDOW_COLLECT_SET} to true")
          }
        }

        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression =
          GpuCollectSet(childExprs.head, c.mutableAggBufferOffset, c.inputAggBufferOffset,
            TypeUtilsShims.collectSetIgnoreNulls(c))

        override def aggBufferAttribute: AttributeReference = {
          val aggBuffer = c.aggBufferAttributes.head
          // Match Spark 4.2+ CollectSet buffer layout for float/double (normalized bit keys).
          val ignoreNulls = TypeUtilsShims.collectSetIgnoreNulls(c)
          val bufferElementType =
            TypeUtilsShims.collectSetCpuBufferElementType(c.child.dataType)
          aggBuffer.copy(dataType = ArrayType(bufferElementType, !ignoreNulls))(
            aggBuffer.exprId, aggBuffer.qualifier)
        }

        override def createCpuToGpuBufferConverter(): CpuToGpuAggregateBufferConverter = {
          val ignoreNulls = TypeUtilsShims.collectSetIgnoreNulls(c)
          new CpuToGpuCollectBufferConverter(
            TypeUtilsShims.collectSetCpuBufferElementType(c.child.dataType),
            !ignoreNulls)
        }

        override def createGpuToCpuBufferConverter(): GpuToCpuAggregateBufferConverter =
          new GpuToCpuCollectBufferConverter()

        override val supportBufferConversion: Boolean = true

        // Last does not overflow, so it doesn't need the ANSI check
        override val needsAnsiCheck: Boolean = false
      }),
    expr[StddevPop](
      "Aggregation computing population standard deviation",
      ExprChecks.groupByOnly(
        TypeSig.DOUBLE, TypeSig.DOUBLE,
        Seq(ParamCheck("input", TypeSig.DOUBLE, TypeSig.DOUBLE))),
      (a, conf, p, r) => new AggExprMeta[StddevPop](a, conf, p, r) {
        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression = {
          val legacyStatisticalAggregate = SQLConf.get.legacyStatisticalAggregate
          GpuStddevPop(childExprs.head, !legacyStatisticalAggregate)
        }
      }),
    expr[StddevSamp](
      "Aggregation computing sample standard deviation",
      ExprChecks.fullAgg(
          TypeSig.DOUBLE, TypeSig.DOUBLE,
          Seq(ParamCheck("input", TypeSig.DOUBLE,
            TypeSig.DOUBLE))),
        (a, conf, p, r) => new AggExprMeta[StddevSamp](a, conf, p, r) {
          override def convertToGpu(childExprs: Seq[Expression]): GpuExpression = {
            val legacyStatisticalAggregate = SQLConf.get.legacyStatisticalAggregate
            GpuStddevSamp(childExprs.head, !legacyStatisticalAggregate)
          }
        }),
    expr[VariancePop](
      "Aggregation computing population variance",
      ExprChecks.groupByOnly(
        TypeSig.DOUBLE, TypeSig.DOUBLE,
        Seq(ParamCheck("input", TypeSig.DOUBLE, TypeSig.DOUBLE))),
      (a, conf, p, r) => new AggExprMeta[VariancePop](a, conf, p, r) {
        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression = {
          val legacyStatisticalAggregate = SQLConf.get.legacyStatisticalAggregate
          GpuVariancePop(childExprs.head, !legacyStatisticalAggregate)
        }
      }),
    expr[VarianceSamp](
      "Aggregation computing sample variance",
      ExprChecks.groupByOnly(
        TypeSig.DOUBLE, TypeSig.DOUBLE,
        Seq(ParamCheck("input", TypeSig.DOUBLE, TypeSig.DOUBLE))),
      (a, conf, p, r) => new AggExprMeta[VarianceSamp](a, conf, p, r) {
        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression = {
          val legacyStatisticalAggregate = SQLConf.get.legacyStatisticalAggregate
          GpuVarianceSamp(childExprs.head, !legacyStatisticalAggregate)
        }
      }),
    expr[Percentile](
      "Aggregation computing exact percentile",
      ExprChecks.reductionAndGroupByAgg(
        // The output can be a single number or array depending on whether percentiles param
        // is a single number or an array.
        TypeSig.DOUBLE + TypeSig.ARRAY.nested(TypeSig.DOUBLE),
        TypeSig.DOUBLE + TypeSig.ARRAY.nested(TypeSig.DOUBLE),
        Seq(
          // ANSI interval types are new in Spark 3.2.0 and are not yet supported by the
          // current GPU implementation.
          ParamCheck("input", TypeSig.integral + TypeSig.fp, TypeSig.integral + TypeSig.fp),
          ParamCheck("percentage",
            TypeSig.lit(TypeEnum.DOUBLE) + TypeSig.ARRAY.nested(TypeSig.lit(TypeEnum.DOUBLE)),
            TypeSig.DOUBLE + TypeSig.ARRAY.nested(TypeSig.DOUBLE)),
          ParamCheck("frequency",
            TypeSig.LONG + TypeSig.ARRAY.nested(TypeSig.LONG),
            TypeSig.LONG + TypeSig.ARRAY.nested(TypeSig.LONG)))),
      (c, conf, p, r) => new TypedImperativeAggExprMeta[Percentile](c, conf, p, r) {
        override def tagAggForGpu(): Unit = {
          // Check if the input percentage can be supported on GPU.
          GpuOverrides.extractLit(childExprs(1).wrapped.asInstanceOf[Expression]) match {
            case None =>
              willNotWorkOnGpu("percentile on GPU only supports literal percentages")
            case Some(Literal(null, _)) =>
              willNotWorkOnGpu("percentile on GPU only supports non-null literal percentages")
            case Some(Literal(a: ArrayData, _)) => {
              if((0 until a.numElements).exists(a.isNullAt)) {
                willNotWorkOnGpu(
                  "percentile on GPU does not support percentage arrays containing nulls")
              }
              if (a.toDoubleArray().exists(percentage => percentage < 0.0 || percentage > 1.0)) {
                willNotWorkOnGpu(
                  "percentile requires the input percentages given in the range [0, 1]")
              }
            }
            case Some(_) => // This is fine
          }
        }

        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression = {
          val exprMeta = p.get.asInstanceOf[BaseExprMeta[_]]
          val isReduction = exprMeta.context match {
            case ReductionAggExprContext => true
            case GroupByAggExprContext => false
            case _ => throw new IllegalStateException(
              s"Invalid aggregation context: ${exprMeta.context}")
          }
          GpuPercentile(childExprs.head, childExprs(1).asInstanceOf[GpuLiteral], childExprs(2),
            isReduction)
        }
        // Declare the data type of the internal buffer so it can be serialized and
        // deserialized correctly during shuffling.
        override def aggBufferAttribute: AttributeReference = {
          val aggBuffer = c.aggBufferAttributes.head
          val dataType: DataType = ArrayType(StructType(Seq(
            StructField("value", childExprs.head.dataType),
            StructField("frequency", LongType))), containsNull = false)
          aggBuffer.copy(dataType = dataType)(aggBuffer.exprId, aggBuffer.qualifier)
        }

        override val needsAnsiCheck: Boolean = false
        override val supportBufferConversion: Boolean = true
        override def createCpuToGpuBufferConverter(): CpuToGpuAggregateBufferConverter =
          CpuToGpuPercentileBufferConverter(childExprs.head.dataType)
        override def createGpuToCpuBufferConverter(): GpuToCpuAggregateBufferConverter =
          GpuToCpuPercentileBufferConverter(childExprs.head.dataType)
      }),
    expr[ApproximatePercentile](
      "Approximate percentile",
      ExprChecks.reductionAndGroupByAgg(
        // note that output can be single number or array depending on whether percentiles param
        // is a single number or an array
        TypeSig.gpuNumeric +
            TypeSig.ARRAY.nested(TypeSig.gpuNumeric),
        TypeSig.cpuNumeric + TypeSig.DATE + TypeSig.TIMESTAMP + TypeSig.ARRAY.nested(
          TypeSig.cpuNumeric + TypeSig.DATE + TypeSig.TIMESTAMP),
        Seq(
          ParamCheck("input",
            TypeSig.gpuNumeric,
            TypeSig.cpuNumeric + TypeSig.DATE + TypeSig.TIMESTAMP),
          ParamCheck("percentage",
            TypeSig.DOUBLE + TypeSig.ARRAY.nested(TypeSig.DOUBLE),
            TypeSig.DOUBLE + TypeSig.ARRAY.nested(TypeSig.DOUBLE)),
          ParamCheck("accuracy", TypeSig.INT, TypeSig.INT))),
      (c, conf, p, r) => new TypedImperativeAggExprMeta[ApproximatePercentile](c, conf, p, r) {

        override def tagAggForGpu(): Unit = {
          // check if the percentile expression can be supported on GPU
          childExprs(1).wrapped match {
            case lit: Literal => lit.value match {
              case null =>
                willNotWorkOnGpu(
                  "approx_percentile on GPU only supports non-null literal percentiles")
              case a: ArrayData if a.numElements == 0 =>
                willNotWorkOnGpu(
                  "approx_percentile on GPU does not support empty percentiles arrays")
              case a: ArrayData if (0 until a.numElements).exists(a.isNullAt) =>
                willNotWorkOnGpu(
                  "approx_percentile on GPU does not support percentiles arrays containing nulls")
              case _ =>
                // this is fine
            }
            case _ =>
              willNotWorkOnGpu("approx_percentile on GPU only supports literal percentiles")
          }
        }

        override def convertToGpu(childExprs: Seq[Expression]): GpuExpression =
          GpuApproximatePercentile(childExprs.head,
              childExprs(1).asInstanceOf[GpuLiteral],
              childExprs(2).asInstanceOf[GpuLiteral])

        override def aggBufferAttribute: AttributeReference = {
          // Spark's ApproxPercentile has an aggregation buffer named "buf" with type "BinaryType"
          // so we need to replace that here with the GPU aggregation buffer reference, which is
          // a t-digest type
          val aggBuffer = c.aggBufferAttributes.head
          aggBuffer.copy(dataType = CudfTDigest.dataType)(aggBuffer.exprId, aggBuffer.qualifier)
        }
      }).incompat("the GPU implementation of approx_percentile is not bit-for-bit " +
          s"compatible with Apache Spark"),
  )
}
