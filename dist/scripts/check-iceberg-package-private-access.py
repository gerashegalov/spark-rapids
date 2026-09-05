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

"""Require Iceberg package-private callers to be stored at the dist jar root."""

from __future__ import print_function

import argparse
import collections
import os
import re
import sys

from java.io import DataInputStream, FileInputStream, IOException
from java.util.jar import JarFile
from javassist.bytecode import AccessFlag, ClassFile, ConstPool


ICEBERG_PREFIX = "org/apache/iceberg/"
ICEBERG_SHADED_PREFIX = "org/apache/iceberg/shaded/"
SHIM_DIR_RE = re.compile(r"^spark[0-9][0-9a-z]*$")
DESCRIPTOR_CLASS_RE = re.compile(r"L([^;<>()\[\]]+);")
RUNTIME_JAR_RE = re.compile(
    r"^iceberg-spark-runtime-(3\.5|4\.0|4\.1)_2\.(?:12|13)-([^/]+)\.jar$")
NO_ICEBERG_BUILD_RE = re.compile(r"^(?:33[0-4]|34[0-4]|350db143|400db173|420|500)$")
MEMBER_VISIBILITY = AccessFlag.PUBLIC | AccessFlag.PRIVATE | AccessFlag.PROTECTED

Member = collections.namedtuple("Member", "access name descriptor")
MemberRef = collections.namedtuple("MemberRef", "kind owner name descriptor")
ClassInfo = collections.namedtuple(
    "ClassInfo", "name access super_name interfaces fields methods class_refs member_refs")
Finding = collections.namedtuple("Finding", "entry caller target reason runtime")
RuntimeSpec = collections.namedtuple(
    "RuntimeSpec", "build_version spark_version iceberg_version")
RuntimeSelection = collections.namedtuple("RuntimeSelection", "spec path")


def _internal_name(name):
    return name.replace(".", "/") if name else None


def _descriptor_classes(descriptor):
    return set(DESCRIPTOR_CLASS_RE.findall(descriptor or ""))


def _normalized_class_names(name):
    internal_name = _internal_name(name)
    if internal_name.startswith("["):
        return _descriptor_classes(internal_name)
    return {internal_name}


def _parse_class(stream):
    class_file = ClassFile(DataInputStream(stream))
    pool = class_file.getConstPool()
    class_refs = set()
    for name in pool.getClassNames():
        class_refs.update(_normalized_class_names(name))
    member_refs = []

    for index in range(1, pool.getSize()):
        tag = pool.getTag(index)
        if tag == ConstPool.CONST_Fieldref:
            ref = MemberRef("field", pool.getFieldrefClassName(index),
                            pool.getFieldrefName(index), pool.getFieldrefType(index))
        elif tag == ConstPool.CONST_Methodref:
            ref = MemberRef("method", pool.getMethodrefClassName(index),
                            pool.getMethodrefName(index), pool.getMethodrefType(index))
        elif tag == ConstPool.CONST_InterfaceMethodref:
            ref = MemberRef("method", pool.getInterfaceMethodrefClassName(index),
                            pool.getInterfaceMethodrefName(index),
                            pool.getInterfaceMethodrefType(index))
        else:
            if tag == ConstPool.CONST_MethodType:
                class_refs.update(_descriptor_classes(
                    pool.getUtf8Info(pool.getMethodTypeInfo(index))))
            elif tag == ConstPool.CONST_Dynamic:
                class_refs.update(_descriptor_classes(pool.getDynamicType(index)))
            elif tag == ConstPool.CONST_InvokeDynamic:
                class_refs.update(_descriptor_classes(pool.getInvokeDynamicType(index)))
            continue
        ref = MemberRef(ref.kind, _internal_name(ref.owner), ref.name, ref.descriptor)
        member_refs.append(ref)
        class_refs.add(ref.owner)
        class_refs.update(_descriptor_classes(ref.descriptor))

    fields = tuple(Member(field.getAccessFlags(), field.getName(), field.getDescriptor())
                   for field in class_file.getFields())
    methods = tuple(Member(method.getAccessFlags(), method.getName(), method.getDescriptor())
                    for method in class_file.getMethods())
    for member in fields + methods:
        class_refs.update(_descriptor_classes(member.descriptor))

    name = _internal_name(class_file.getName())
    class_refs.discard(name)
    inner_access = class_file.getInnerAccessFlags()
    access = inner_access if inner_access >= 0 else class_file.getAccessFlags()
    return ClassInfo(name, access,
                     _internal_name(class_file.getSuperclass()),
                     tuple(_internal_name(name) for name in class_file.getInterfaces()),
                     fields, methods, frozenset(class_refs), tuple(member_refs))


def _is_class_entry(entry):
    return (entry.endswith(".class") and not entry.endswith("/module-info.class") and
            not entry.startswith("META-INF/versions/"))


def has_class_entries(path):
    if not os.path.exists(path):
        return False
    repository = LazyClassRepository(path, lambda entry: True)
    try:
        return bool(repository)
    finally:
        repository.close()


def load_classes(path, entry_predicate):
    repository = LazyClassRepository(path, entry_predicate)
    try:
        return list(repository.items())
    finally:
        repository.close()


class LazyClassRepository(object):
    """Index a class layout and parse classes only when resolution needs them."""

    def __init__(self, path, entry_predicate):
        if not os.path.exists(path):
            raise IOError("class layout does not exist: %s" % path)
        self.path = path
        self.archive = JarFile(path) if os.path.isfile(path) else None
        self.entries = {}
        self.cache = {}
        if self.archive:
            entries = self.archive.entries()
            while entries.hasMoreElements():
                item = entries.nextElement()
                entry = item.getName()
                if not item.isDirectory() and _is_class_entry(entry) and entry_predicate(entry):
                    self.entries[entry[:-6]] = item
        else:
            for root, _, files in os.walk(path):
                for file_name in files:
                    full_path = os.path.join(root, file_name)
                    entry = os.path.relpath(full_path, path).replace(os.sep, "/")
                    if _is_class_entry(entry) and entry_predicate(entry):
                        self.entries[entry[:-6]] = full_path

    def __len__(self):
        return len(self.entries)

    def get(self, name, default=()):
        if name not in self.entries:
            return default
        return (self._load(name),)

    def _load(self, name):
        if name not in self.cache:
            source = self.entries[name]
            stream = (self.archive.getInputStream(source) if self.archive else
                      FileInputStream(source))
            try:
                self.cache[name] = _parse_class(stream)
            finally:
                stream.close()
        return self.cache[name]

    def items(self):
        for name in sorted(self.entries):
            yield name + ".class", self._load(name)

    def close(self):
        if self.archive:
            self.archive.close()


def _is_runtime_entry(entry):
    return entry.startswith(ICEBERG_PREFIX) and not entry.startswith(ICEBERG_SHADED_PREFIX)


def _is_plugin_entry(entry):
    return ICEBERG_PREFIX in entry and ICEBERG_SHADED_PREFIX not in entry


def _classes_by_name(entries):
    result = collections.defaultdict(list)
    for _, info in entries:
        result[info.name].append(info)
    return result


def _entry_world(entry):
    first = entry.split("/", 1)[0]
    if first == "spark-shared":
        return "shared"
    if SHIM_DIR_RE.match(first):
        return first[len("spark"):]
    return "root"


def _plugin_world(layout_entries, build_version):
    if build_version is None:
        return layout_entries
    visible = []
    class_names = set()
    for world in ("root", "shared", build_version):
        for entry, info in layout_entries:
            if _entry_world(entry) == world and info.name not in class_names:
                visible.append((entry, info))
                class_names.add(info.name)
    return visible


def _find_members(runtime_classes, plugin_classes, owner, reference):
    pending = [owner]
    visited = set()
    while pending:
        next_pending = []
        found = set()
        for current in pending:
            if current in visited:
                continue
            visited.add(current)
            infos = (list(runtime_classes.get(current, ())) +
                     list(plugin_classes.get(current, ())))
            for info in infos:
                members = info.fields if reference.kind == "field" else info.methods
                for member in members:
                    if member.name == reference.name and member.descriptor == reference.descriptor:
                        found.add((info.name, member))
                if info.super_name:
                    next_pending.append(info.super_name)
                next_pending.extend(info.interfaces)
        if found:
            return found
        pending = next_pending
    return set()


def _package_name(class_name):
    return class_name.rsplit("/", 1)[0]


def _is_root_entry(entry):
    first = entry.split("/", 1)[0]
    return first != "spark-shared" and not SHIM_DIR_RE.match(first)


def _runtime_key(path):
    match = RUNTIME_JAR_RE.match(os.path.basename(path))
    return match.groups() if match else None


def _iceberg_versions(values):
    versions = {}
    for value in values:
        family, separator, version = value.partition("=")
        if not separator or family not in ("16", "19", "110", "111") or not version:
            raise RuntimeError("invalid Iceberg version mapping: %s" % value)
        versions[family] = version
    return versions


def runtime_specs(build_versions, iceberg_versions):
    specs = []
    for build_version in re.split(r"[,\s]+", build_versions.strip()):
        if not build_version:
            continue
        if re.match(r"^35[0-3]$", build_version):
            families = (("3.5", "16"),)
        elif re.match(r"^35[4-9]$", build_version):
            families = (("3.5", "19"), ("3.5", "110"))
        elif re.match(r"^40[01]$", build_version):
            families = (("4.0", "110"),)
        elif re.match(r"^40[2-4]$", build_version):
            families = (("4.0", "110"), ("4.0", "111"))
        elif re.match(r"^41[1-3]$", build_version):
            families = (("4.1", "111"),)
        elif not NO_ICEBERG_BUILD_RE.match(build_version):
            raise RuntimeError("unknown build version in Iceberg audit: %s" % build_version)
        else:
            families = ()
        for spark_version, family in families:
            if family not in iceberg_versions:
                raise RuntimeError("missing Iceberg %s.x version mapping" % family)
            specs.append(RuntimeSpec(
                build_version, spark_version, iceberg_versions[family]))
    return specs


def select_runtime_paths(paths, specs):
    if specs is None:
        if not paths:
            raise RuntimeError("no Iceberg audit runtime was provided")
        return [RuntimeSelection(RuntimeSpec(None, None, None), path)
                for path in sorted(paths)]
    by_key = dict((_runtime_key(path), path) for path in paths if _runtime_key(path))
    missing = set((spec.spark_version, spec.iceberg_version)
                  for spec in specs).difference(by_key)
    if missing:
        raise RuntimeError("missing Iceberg audit runtime(s): %s" % sorted(missing))
    return [RuntimeSelection(spec, by_key[(spec.spark_version, spec.iceberg_version)])
            for spec in specs]


def maven_runtime_paths(repository, scala_binary_version, specs):
    paths = []
    for spec in specs:
        artifact = "iceberg-spark-runtime-%s_%s" % (
            spec.spark_version, scala_binary_version)
        paths.append(os.path.join(
            repository, "org", "apache", "iceberg", artifact, spec.iceberg_version,
            "%s-%s.jar" % (artifact, spec.iceberg_version)))
    return paths


def find_package_private_access(layout_entries, runtime_classes, runtime_label):
    plugin_classes = _classes_by_name(layout_entries)
    findings = set()
    callers = set()
    for entry, caller in layout_entries:
        caller_findings = set()
        for target_name in caller.class_refs:
            for target in runtime_classes.get(target_name, ()):
                if not target.access & AccessFlag.PUBLIC:
                    caller_findings.add((target_name, "package-private class"))
        for reference in caller.member_refs:
            if not reference.owner.startswith(ICEBERG_PREFIX):
                continue
            for declaring_class, member in _find_members(
                    runtime_classes, plugin_classes, reference.owner, reference):
                if (not member.access & MEMBER_VISIBILITY and
                        _package_name(caller.name) == _package_name(declaring_class)):
                    separator = ":" if reference.kind == "field" else ""
                    target = "%s.%s%s%s" % (declaring_class, member.name,
                                             separator, member.descriptor)
                    caller_findings.add((target, "package-private %s" % reference.kind))
        if caller_findings:
            callers.add((entry, caller.name))
            for target, reason in caller_findings:
                findings.add(Finding(entry, caller.name, target, reason, runtime_label))
    return callers, findings


def _runtime_paths(args):
    paths = list(args.iceberg_runtime)
    for directory in args.runtime_directory:
        paths.extend(os.path.join(directory, name) for name in os.listdir(directory)
                     if name.endswith(".jar"))
    specs = None
    if args.build_versions:
        specs = runtime_specs(args.build_versions, _iceberg_versions(args.iceberg_version))
    if args.maven_repository:
        if not args.build_versions or not args.scala_binary_version:
            raise RuntimeError("Maven repository lookup requires build and Scala versions")
        paths.extend(maven_runtime_paths(
            args.maven_repository, args.scala_binary_version, specs))
    return select_runtime_paths(paths, specs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout", help="assembled dist jar or parallel-world directory")
    parser.add_argument("iceberg_runtime", nargs="*", help="Iceberg runtime jar(s)")
    parser.add_argument("--runtime-directory", action="append", default=[])
    parser.add_argument("--maven-repository")
    parser.add_argument("--scala-binary-version", choices=("2.12", "2.13"))
    parser.add_argument("--build-versions")
    parser.add_argument("--iceberg-version", action="append", default=[])
    args = parser.parse_args(argv)

    try:
        if not has_class_entries(args.layout):
            raise RuntimeError("assembled layout is missing or contains no class files: %s" %
                               args.layout)
        layout_entries = load_classes(args.layout, _is_plugin_entry)
        runtime_selections = _runtime_paths(args)
        if runtime_selections and not layout_entries:
            raise RuntimeError("assembled layout contains no Iceberg plugin classes")
        callers = set()
        findings = set()
        for selection in runtime_selections:
            runtime_path = selection.path
            runtime_classes = LazyClassRepository(runtime_path, _is_runtime_entry)
            try:
                if not runtime_classes:
                    raise RuntimeError("runtime contains no Iceberg classes: %s" % runtime_path)
                world_entries = _plugin_world(
                    layout_entries, selection.spec.build_version)
                runtime_label = os.path.basename(runtime_path)
                if selection.spec.build_version:
                    runtime_label = "spark%s; %s" % (
                        selection.spec.build_version, runtime_label)
                runtime_callers, runtime_findings = find_package_private_access(
                    world_entries, runtime_classes, runtime_label)
                callers.update(runtime_callers)
                findings.update(runtime_findings)
            finally:
                runtime_classes.close()
    except (IOError, OSError, IOException, RuntimeError) as error:
        print("Iceberg package-private access audit failed: %s" % error, file=sys.stderr)
        return 2

    violations = sorted(finding for finding in findings if not _is_root_entry(finding.entry))
    if violations:
        print("Iceberg package-private access must be root-safe; found non-root callers:",
              file=sys.stderr)
        for entry in sorted(set(finding.entry for finding in violations)):
            entry_findings = [finding for finding in violations if finding.entry == entry]
            print("  %s (%s)" % (entry, entry_findings[0].caller.replace("/", ".")),
                  file=sys.stderr)
            for finding in entry_findings:
                print("    -> %s [%s; %s]" % (finding.target.replace("/", "."),
                                               finding.reason, finding.runtime),
                      file=sys.stderr)
        print("Move each caller to a module listed in dist/root-safe-module-classes.txt "
              "or promote it with dist/unshimmed-common-from-single-shim.txt.", file=sys.stderr)
        return 1

    print("Iceberg package-private access audit passed: %d caller classes are at the jar root "
          "across %d runtime world(s)" % (len(callers), len(runtime_selections)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
