#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compile fixtures that enforce the repository's scoped deprecation policy."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path


MAVEN_NAMESPACE = {"m": "http://maven.apache.org/POM/4.0.0"}


def compiler_configuration(pom_path):
    root = ElementTree.parse(pom_path).getroot()
    scala_version = root.findtext("m:properties/m:scala.version", namespaces=MAVEN_NAMESPACE)
    if not scala_version:
        raise RuntimeError(f"Could not find scala.version in {pom_path}")
    plugin_paths = (
        "m:build/m:plugins/m:plugin",
        "m:build/m:pluginManagement/m:plugins/m:plugin",
    )
    plugins = (
        plugin
        for plugin_path in plugin_paths
        for plugin in root.findall(plugin_path, MAVEN_NAMESPACE)
    )
    for plugin in plugins:
        artifact_id = plugin.findtext("m:artifactId", namespaces=MAVEN_NAMESPACE)
        if artifact_id == "scala-maven-plugin":
            args = [
                argument.text
                for argument in plugin.findall("m:configuration/m:args/m:arg", MAVEN_NAMESPACE)
                if argument.text
            ]
            if not args:
                raise RuntimeError(f"scala-maven-plugin has no compiler arguments in {pom_path}")
            return scala_version, args
    raise RuntimeError(f"Could not find scala-maven-plugin in {pom_path}")


def scala_compiler_classpath(maven_repo, scala_version):
    scala_root = Path(maven_repo) / "org" / "scala-lang"
    jars = [
        scala_root / artifact / scala_version / f"{artifact}-{scala_version}.jar"
        for artifact in ("scala-compiler", "scala-library", "scala-reflect")
    ]
    missing = [str(jar) for jar in jars if not jar.is_file()]
    if missing:
        raise RuntimeError("Missing Scala compiler dependencies: " + ", ".join(missing))
    return jars


def run_command(command):
    return subprocess.run(command, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)


def write_fixtures(root):
    sources = {
        "ai/rapids/cudf/fixture/NvidiaApi.java": """
package ai.rapids.cudf.fixture;
public final class NvidiaApi {
  private NvidiaApi() {}
  @Deprecated public static void oldApi() {}
}
""",
        "org/example/fixture/ThirdPartyApi.java": """
package org.example.fixture;
public final class ThirdPartyApi {
  private ThirdPartyApi() {}
  @Deprecated public static void oldApi() {}
}
""",
        "com/nvidia/spark/rapids/jni/fixture/JniApi.java": """
package com.nvidia.spark.rapids.jni.fixture;
public final class JniApi {
  private JniApi() {}
  @Deprecated public static void oldApi() {}
}
""",
        "com/nvidia/spark/rapids/optimizer/fixture/PrivateApi.java": """
package com.nvidia.spark.rapids.optimizer.fixture;
public final class PrivateApi {
  private PrivateApi() {}
  @Deprecated public static void oldApi() {}
}
""",
        "org/apache/spark/sql/rapids/internal/fixture/PrivateApi.java": """
package org.apache.spark.sql.rapids.internal.fixture;
public final class PrivateApi {
  private PrivateApi() {}
  @Deprecated public static void oldApi() {}
}
""",
        "org/apache/spark/sql/execution/aggregate/PartialAggUtils.java": """
package org.apache.spark.sql.execution.aggregate;
public final class PartialAggUtils {
  private PartialAggUtils() {}
  @Deprecated public static void oldApi() {}
}
""",
        "org/apache/spark/sql/execution/aggregate/PartialAggUtilsNeighbor.java": """
package org.apache.spark.sql.execution.aggregate;
public final class PartialAggUtilsNeighbor {
  private PartialAggUtilsNeighbor() {}
  @Deprecated public static void oldApi() {}
}
""",
        "org/apache/spark/sql/execution/aggregate/SparkApi.java": """
package org.apache.spark.sql.execution.aggregate;
public final class SparkApi {
  private SparkApi() {}
  @Deprecated public static void oldApi() {}
}
""",
        "NvidiaCall.scala": """
object NvidiaCall {
  def call(): Unit = ai.rapids.cudf.fixture.NvidiaApi.oldApi()
}
""",
        "JniCall.scala": """
object JniCall {
  def call(): Unit = com.nvidia.spark.rapids.jni.fixture.JniApi.oldApi()
}
""",
        "PrivateComNvidiaCall.scala": """
object PrivateComNvidiaCall {
  def call(): Unit = com.nvidia.spark.rapids.optimizer.fixture.PrivateApi.oldApi()
}
""",
        "PrivateRapidsCall.scala": """
object PrivateRapidsCall {
  def call(): Unit = org.apache.spark.sql.rapids.internal.fixture.PrivateApi.oldApi()
}
""",
        "PrivateSparkBridgeCall.scala": """
object PrivateSparkBridgeCall {
  def call(): Unit = org.apache.spark.sql.execution.aggregate.PartialAggUtils.oldApi()
}
""",
        "PartialAggUtilsNeighborCall.scala": """
object PartialAggUtilsNeighborCall {
  def call(): Unit = org.apache.spark.sql.execution.aggregate.PartialAggUtilsNeighbor.oldApi()
}
""",
        "SparkCall.scala": """
object SparkCall {
  def call(): Unit = org.apache.spark.sql.execution.aggregate.SparkApi.oldApi()
}
""",
        "ThirdPartyCall.scala": """
object ThirdPartyCall {
  def call(): Unit = org.example.fixture.ThirdPartyApi.oldApi()
}
""",
    }
    for relative_path, source in sources.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source.lstrip(), encoding="utf-8")


def check_policy(pom_path, maven_repo):
    scala_version, compiler_args = compiler_configuration(pom_path)
    compiler_jar, library_jar, reflect_jar = scala_compiler_classpath(
        maven_repo, scala_version)
    javac = shutil.which("javac")
    java = shutil.which("java")
    if not javac or not java:
        raise RuntimeError("Both java and javac are required for the deprecation policy check")

    with tempfile.TemporaryDirectory(prefix="cudf-spark-deprecation-policy-") as temp_dir:
        fixture_root = Path(temp_dir)
        classes = fixture_root / "classes"
        classes.mkdir()
        write_fixtures(fixture_root)
        java_compile = run_command([
            javac, "-d", str(classes),
            str(fixture_root / "ai/rapids/cudf/fixture/NvidiaApi.java"),
            str(fixture_root / "com/nvidia/spark/rapids/jni/fixture/JniApi.java"),
            str(fixture_root / "com/nvidia/spark/rapids/optimizer/fixture/PrivateApi.java"),
            str(fixture_root / "org/apache/spark/sql/rapids/internal/fixture/PrivateApi.java"),
            str(fixture_root / "org/apache/spark/sql/execution/aggregate/PartialAggUtils.java"),
            str(fixture_root /
                "org/apache/spark/sql/execution/aggregate/PartialAggUtilsNeighbor.java"),
            str(fixture_root / "org/apache/spark/sql/execution/aggregate/SparkApi.java"),
            str(fixture_root / "org/example/fixture/ThirdPartyApi.java"),
        ])
        if java_compile.returncode:
            raise RuntimeError("Could not compile Java fixtures:\n" + java_compile.stdout)

        compiler_classpath = os.pathsep.join(map(str, (compiler_jar, library_jar, reflect_jar)))
        source_classpath = os.pathsep.join(map(str, (classes, library_jar)))

        def compile_scala(source):
            return run_command([
                java, "-cp", compiler_classpath, "scala.tools.nsc.Main",
                "-classpath", source_classpath, "-d", str(classes),
                *compiler_args, str(fixture_root / source),
            ])

        nvidia_sources = (
            ("cuDF Java", "NvidiaCall.scala"),
            ("cudf-spark-jni", "JniCall.scala"),
            ("cudf-spark-private com.nvidia namespace", "PrivateComNvidiaCall.scala"),
            ("cudf-spark-private RAPIDS namespace", "PrivateRapidsCall.scala"),
            ("cudf-spark-private Spark-package bridge", "PrivateSparkBridgeCall.scala"),
        )
        for api_name, source in nvidia_sources:
            nvidia_compile = compile_scala(source)
            if nvidia_compile.returncode or "deprecated" not in nvidia_compile.stdout.lower():
                raise RuntimeError(
                    f"{api_name} deprecation must be visible and nonfatal, "
                    "but compilation produced:\n" + nvidia_compile.stdout)

        fatal_sources = (
            ("Third-party", "ThirdPartyCall.scala"),
            ("Apache Spark sibling", "SparkCall.scala"),
            ("PartialAggUtils prefix neighbor", "PartialAggUtilsNeighborCall.scala"),
        )
        for api_name, source in fatal_sources:
            fatal_compile = compile_scala(source)
            if fatal_compile.returncode == 0 or "deprecated" not in fatal_compile.stdout.lower():
                raise RuntimeError(
                    f"{api_name} deprecation must be visible and fatal, "
                    "but compilation produced:\n" + fatal_compile.stdout)

    print(f"Deprecation policy check passed for Scala {scala_version}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pom", required=True)
    parser.add_argument("--maven-repo", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        check_policy(args.pom, args.maven_repo)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
