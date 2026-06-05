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

"""Build dist/target/parallel-world directly for repeated unshim analysis.

This mirrors the analyzer-relevant part of dist/maven-antrun/build-parallel-worlds.xml
without starting a final Maven dist generate-resources invocation. It assumes buildall
has already built the per-shim sql-plugin-api and aggregator jars under target/sparkXYZ.
"""

import argparse
import fnmatch
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ARTIFACTS = ("sql-plugin-api", "aggregator")


def read_patterns(path):
    with path.open() as fh:
        return [
            line.strip()
            for line in fh
            if line.strip() and not line.lstrip().startswith("#")
        ]


def has_fnmatch_magic(pattern):
    return any(ch in pattern for ch in "*?[")


def matching_members(namelist, patterns):
    names_by_entry = {}
    for name in namelist:
        names_by_entry.setdefault(name, []).append(name)

    matches = []
    for pattern in patterns:
        if has_fnmatch_magic(pattern):
            matches.extend(fnmatch.filter(namelist, pattern))
        else:
            matches.extend(names_by_entry.get(pattern, []))
    return matches


def safe_extract(zip_handle, destination, members=None):
    destination = destination.resolve()
    for member in members if members is not None else zip_handle.namelist():
        target = (destination / member).resolve()
        if not str(target).startswith(str(destination) + os.sep):
            raise RuntimeError("refusing to extract outside destination: %s" % member)
        zip_handle.extract(member, destination)


def clean_output(target_dir):
    for dirname in ("parallel-world", "deps", "extra-resources"):
        path = target_dir / dirname
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for jar_path in target_dir.glob("*.jar"):
        jar_path.unlink()


def artifact_jar(base_dir, artifact, scala_binary_version, project_version, buildver):
    artifact_id = "rapids-4-spark-%s_%s" % (artifact, scala_binary_version)
    classifier = "spark%s" % buildver
    jar_name = "%s-%s-%s.jar" % (artifact_id, project_version, classifier)
    jar_path = base_dir / artifact / "target" / classifier / jar_name
    if not jar_path.is_file():
        raise FileNotFoundError(
            "expected built %s jar missing: %s" % (artifact, jar_path))
    return jar_path


def copy_and_extract_jars(
        base_dir,
        target_dir,
        scala_binary_version,
        project_version,
        buildvers,
        from_single_shim,
        from_each):
    deps_dir = target_dir / "deps"
    parallel_world = target_dir / "parallel-world"
    sorted_buildvers = sorted(buildvers, reverse=True)
    root_buildver = sorted_buildvers[0]

    for buildver in sorted_buildvers:
        classifier = "spark%s" % buildver
        for artifact in ARTIFACTS:
            jar_path = artifact_jar(
                base_dir, artifact, scala_binary_version, project_version, buildver)
            deps_jar = deps_dir / jar_path.name
            shutil.copy2(jar_path, deps_jar)

            with zipfile.ZipFile(deps_jar) as zip_handle:
                safe_extract(zip_handle, parallel_world / classifier)
                if buildver == root_buildver and artifact == "sql-plugin-api":
                    safe_extract(zip_handle, parallel_world)

                patterns = from_each
                if buildver == root_buildver:
                    patterns = from_single_shim + from_each
                members = matching_members(zip_handle.namelist(), patterns)
                safe_extract(zip_handle, parallel_world, members)


def run_checked(command, cwd, env=None):
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def remove_allowlisted_from_spark_shared(parallel_world, from_single_shim):
    shared_dir = parallel_world / "spark-shared"
    if not shared_dir.is_dir():
        return

    for pattern in from_single_shim:
        if has_fnmatch_magic(pattern):
            for path in shared_dir.rglob("*"):
                if path.is_file() and fnmatch.fnmatch(path.relative_to(shared_dir).as_posix(), pattern):
                    path.unlink()
        else:
            path = shared_dir / pattern
            if path.is_file():
                path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvn-base-dir", required=True,
        help="Maven build root containing module target directories")
    parser.add_argument("--source-dir", required=True,
        help="Top-level spark-rapids source directory")
    parser.add_argument("--project-version", required=True)
    parser.add_argument("--scala-binary-version", required=True)
    parser.add_argument("--buildvers", required=True,
        help="Comma-separated Spark build versions, for example 350,411")
    args = parser.parse_args()

    base_dir = Path(args.mvn_base_dir).resolve()
    source_dir = Path(args.source_dir).resolve()
    dist_dir = source_dir / "dist"
    target_dir = base_dir / "dist" / "target"
    parallel_world = target_dir / "parallel-world"
    buildvers = [item.strip() for item in args.buildvers.split(",") if item.strip()]

    if len(buildvers) == 0:
        raise RuntimeError("no build versions were supplied")

    from_single_shim = read_patterns(dist_dir / "unshimmed-common-from-single-shim.txt")
    from_each = read_patterns(dist_dir / "unshimmed-from-each-spark3xx.txt")

    print("Direct unshim parallel-world assembly for Spark versions: %s" %
          ", ".join(buildvers),
          flush=True)
    clean_output(target_dir)
    copy_and_extract_jars(
        base_dir,
        target_dir,
        args.scala_binary_version,
        args.project_version,
        buildvers,
        from_single_shim,
        from_each)

    run_checked(
        [str(dist_dir / "scripts" / "check-shims-revisions.sh"), ",".join(buildvers)],
        cwd=target_dir)

    dedupe_env = os.environ.copy()
    dedupe_env["UNSHIM_FAST"] = "1"
    dedupe_env["UNSHIMMED_COMMON_FROM_SINGLE_SHIM_TXT"] = str(
        dist_dir / "unshimmed-common-from-single-shim.txt")
    run_checked([str(dist_dir / "scripts" / "binary-dedupe.sh")],
                cwd=target_dir,
                env=dedupe_env)
    remove_allowlisted_from_spark_shared(parallel_world, from_single_shim)

    print("Direct unshim parallel-world output: %s" % parallel_world, flush=True)


if __name__ == "__main__":
    sys.exit(main())
