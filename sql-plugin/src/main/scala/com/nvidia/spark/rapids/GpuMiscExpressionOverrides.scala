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
import org.apache.spark.sql.catalyst.expressions.objects.StaticInvoke
import org.apache.spark.sql.rapids._
import org.apache.spark.sql.rapids.aggregate.GpuHyperLogLogPlusPlus
import org.apache.spark.sql.rapids.shims.GpuAscii

private[rapids] object GpuMiscExpressionOverrides {
  val rules: Seq[ExprRule[_ <: Expression]] = Seq(
    expr[GetJsonObject](
      "Extracts a json object from path",
      ExprChecks.projectOnly(
        TypeSig.STRING, TypeSig.STRING, Seq(ParamCheck("json", TypeSig.STRING, TypeSig.STRING),
          ParamCheck("path", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING))),
      (a, conf, p, r) => new GpuGetJsonObjectMeta(a, conf, p, r)),
    expr[JsonToStructs](
      "Returns a struct value with the given `jsonStr` and `schema`",
      ExprChecks.projectOnly(
        TypeSig.STRUCT.nested(jsonStructReadTypes) +
          TypeSig.MAP.nested(TypeSig.STRING + TypeSig.ARRAY.nested(TypeSig.STRING))
            .withPsNote(TypeEnum.MAP,
          "MAP only supports keys of STRING type and values that are of STRING type " +
            "or ARRAY of STRING type, and is only supported at the top level"),
        (TypeSig.STRUCT + TypeSig.MAP + TypeSig.ARRAY).nested(TypeSig.all),
        Seq(ParamCheck("jsonStr", TypeSig.STRING, TypeSig.STRING))),
      GpuJsonToStructsMeta),
    expr[StructsToJson](
      "Converts structs to JSON text format",
      ExprChecks.projectOnly(
        TypeSig.STRING,
        TypeSig.STRING,
        Seq(ParamCheck("struct",
          (TypeSig.BOOLEAN + TypeSig.STRING + TypeSig.integral + TypeSig.FLOAT +
            TypeSig.DOUBLE + TypeSig.DATE + TypeSig.TIMESTAMP +
            TypeSig.DECIMAL_128 +
            TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP).nested(),
          (TypeSig.BOOLEAN + TypeSig.STRING + TypeSig.integral + TypeSig.FLOAT +
            TypeSig.DOUBLE + TypeSig.DATE + TypeSig.TIMESTAMP +
            TypeSig.DECIMAL_128 +
            TypeSig.STRUCT + TypeSig.ARRAY + TypeSig.MAP).nested()
        ))),
      (a, conf, p, r) => new GpuStructsToJsonMeta(a, conf, p, r))
        .disabledByDefault("it is currently in beta and undergoes continuous enhancements."+
      " Please consult the "+
      "[compatibility documentation](../compatibility.md#json-supporting-types)"+
      " to determine whether you can enable this configuration for your use case"),
    expr[JsonTuple](
      "Returns a tuple like the function get_json_object, but it takes multiple names. " +
        "All the input parameters and output column types are string.",
      ExprChecks.projectOnly(
        TypeSig.ARRAY.nested(TypeSig.STRUCT + TypeSig.STRING),
        TypeSig.ARRAY.nested(TypeSig.STRUCT + TypeSig.STRING),
        Seq(ParamCheck("json", TypeSig.STRING, TypeSig.STRING)),
        Some(RepeatingParamCheck("field", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING))),
      GpuJsonTupleMeta
    ),
    expr[org.apache.spark.sql.execution.ScalarSubquery](
      "Subquery that will return only one row and one column",
      ExprChecks.projectOnly(
        (TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL_128 + TypeSig.BINARY +
            TypeSig.ARRAY + TypeSig.MAP + TypeSig.STRUCT).nested(),
        TypeSig.all,
        Nil, None),
      (a, conf, p, r) =>
        new ExprMeta[org.apache.spark.sql.execution.ScalarSubquery](a, conf, p, r) {
          override def convertToGpuImpl(): GpuExpression = GpuScalarSubquery(a.plan, a.exprId)
        }
    ),
    expr[CreateMap](
      desc = "Create a map",
      CreateMapCheck,
      (a, conf, p, r) => new ExprMeta[CreateMap](a, conf, p, r) {
        override def convertToGpuImpl(): GpuExpression =
          GpuCreateMap(childExprs.map(_.convertToGpu()))
      }
    ),
    expr[Sequence](
      desc = "Sequence",
      ExprChecks.projectOnly(
        TypeSig.ARRAY.nested(TypeSig.integral), TypeSig.ARRAY.nested(TypeSig.integral +
          TypeSig.TIMESTAMP + TypeSig.DATE),
        Seq(ParamCheck("start", TypeSig.integral, TypeSig.integral + TypeSig.TIMESTAMP +
          TypeSig.DATE),
          ParamCheck("stop", TypeSig.integral, TypeSig.integral + TypeSig.TIMESTAMP +
            TypeSig.DATE)),
        Some(RepeatingParamCheck("step", TypeSig.integral, TypeSig.integral + TypeSig.CALENDAR))),
      (a, conf, p, r) => new GpuSequenceMeta(a, conf, p, r)
    ),
    expr[BitLength](
      "The bit length of string data",
      ExprChecks.unaryProject(
        TypeSig.INT, TypeSig.INT,
        TypeSig.STRING, TypeSig.STRING + TypeSig.BINARY),
      (a, conf, p, r) => new UnaryExprMeta[BitLength](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuBitLength(child)
      }),
    expr[OctetLength](
      "The byte length of string data",
      ExprChecks.unaryProject(
        TypeSig.INT, TypeSig.INT,
        TypeSig.STRING, TypeSig.STRING + TypeSig.BINARY),
      (a, conf, p, r) => new UnaryExprMeta[OctetLength](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuOctetLength(child)
      }),
    expr[Ascii](
      "The numeric value of the first character of string data.",
      ExprChecks.unaryProject(TypeSig.INT, TypeSig.INT, TypeSig.STRING, TypeSig.STRING),
      (a, conf, p, r) => new UnaryExprMeta[Ascii](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = GpuAscii(child)
      }).disabledByDefault("it only supports strings starting with ASCII or Latin-1 characters " +
        "after Spark 3.2.3, 3.3.1 and 3.4.0. Otherwise the results will not match the CPU."),
    expr[GetArrayStructFields](
      "Extracts the `ordinal`-th fields of all array elements for the data with the type of" +
        " array of struct",
      ExprChecks.unaryProject(
        TypeSig.ARRAY.nested((TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP + TypeSig.BINARY).nested()),
        TypeSig.ARRAY.nested(TypeSig.all),
        // we should allow all supported types for the children types signature of the nested
        // struct, even only a struct child is allowed for the array here. Since TypeSig supports
        // only one level signature for nested type.
        TypeSig.ARRAY.nested((TypeSig.commonCudfTypes + TypeSig.DECIMAL_128 + TypeSig.NULL +
            TypeSig.ARRAY + TypeSig.STRUCT + TypeSig.MAP + TypeSig.BINARY).nested()),
        TypeSig.ARRAY.nested(TypeSig.all)),
      (e, conf, p, r) => new GpuGetArrayStructFieldsMeta(e, conf, p, r)
    ),
    expr[DynamicPruningExpression](
      "Dynamic pruning expression marker",
      ExprChecks.unaryProject(TypeSig.all, TypeSig.all, TypeSig.BOOLEAN, TypeSig.BOOLEAN),
      (a, conf, p, r) => new UnaryExprMeta[DynamicPruningExpression](a, conf, p, r) {
        override def convertToGpu(child: Expression): GpuExpression = {
          GpuDynamicPruningExpression(child)
        }
      }),
    expr[HyperLogLogPlusPlus](
      "Aggregation approximate count distinct",
      ExprChecks.reductionAndGroupByAgg(TypeSig.LONG, TypeSig.LONG,
        // HyperLogLogPlusPlus depends on Xxhash64
        // HyperLogLogPlusPlus supports all the types that Xxhash 64 supports
        Seq(ParamCheck("input",XxHash64Shims.supportedTypes, TypeSig.all))),
      (a, conf, p, r) => new UnaryExprMeta[HyperLogLogPlusPlus](a, conf, p, r) {

        // It's the same as Xxhash64
        override def tagExprForGpu(): Unit = {
          val maxDepth = a.children.map(
            c => XxHash64Utils.computeMaxStackSize(c.dataType)).max
          if (maxDepth > XxHash64Utils.MAX_STACK_DEPTH) {
            willNotWorkOnGpu(s"The data type requires a stack depth of $maxDepth, " +
                s"which exceeds the GPU limit of ${XxHash64Utils.MAX_STACK_DEPTH}. " +
                "The algorithm to calculate stack depth: " +
                "1: Primitive type counts 1 depth; " +
                "2: Array of Structure counts:  1  + depthOf(Structure); " +
                "3: Array of Other counts: depthOf(Other); " +
                "4: Structure counts: 1 + max of depthOf(child); " +
                "5: Map counts: 2 + max(depthOf(key), depthOf(value)); "
            )
          }
          val precision = GpuHyperLogLogPlusPlus.computePrecision(a.relativeSD)
          // Spark supports precision range: [4, Infinity)
          // cuCollection supports precision range: [4, 18]
          // Spark-Rapids only supports precision range: [4, 14],
          // GPU does not perform well for precision > 14.
          if (precision < 4 || precision > 14) {
            willNotWorkOnGpu(s"The precision $precision from relativeSD ${a.relativeSD} is out of" +
              s" range, GPU only supports precision range [4, 14].")
          }
        }

        override def convertToGpu(child: Expression): GpuExpression = {
          GpuHyperLogLogPlusPlus(child, a.relativeSD)
        }
      }
    ),
    expr[Uuid](
      desc = "Uuid",
      ExprChecks.projectOnly(TypeSig.STRING, TypeSig.STRING),
      (a, conf, p, r) => new GpuUuidMeta(a, conf, p, r)
    ),
    expr[StaticInvoke](
      desc = "StaticInvoke",
      StaticInvokeCheck,
      (a, conf, p, r) => new StaticInvokeMeta(a, conf, p, r)
    ).note("The supported types are not deterministic since it's a dynamic expression"),
    SparkShimImpl.ansiCastRule
  )
}
