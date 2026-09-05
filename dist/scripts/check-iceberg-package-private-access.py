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
    r"^iceberg-spark-runtime-(3\.5|4\.0|4\.1)_2\.(?:12|13)-([0-9.]+)\.jar$")
NO_ICEBERG_BUILD_RE = re.compile(r"^(?:33[0-4]|34[0-4]|350db143|400db173|420|500)$")
MEMBER_VISIBILITY = AccessFlag.PUBLIC | AccessFlag.PRIVATE | AccessFlag.PROTECTED

Member = collections.namedtuple("Member", "access name descriptor")
MemberRef = collections.namedtuple("MemberRef", "kind owner name descriptor")
ClassInfo = collections.namedtuple(
    "ClassInfo", "name access super_name interfaces fields methods class_refs member_refs")
Finding = collections.namedtuple("Finding", "entry caller target reason runtime")


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
    if os.path.isfile(path):
        archive = JarFile(path)
        try:
            entries = archive.entries()
            while entries.hasMoreElements():
                if _is_class_entry(entries.nextElement().getName()):
                    return True
            return False
        finally:
            archive.close()
    for _, _, files in os.walk(path):
        if any(file_name.endswith(".class") for file_name in files):
            return True
    return False


def load_classes(path, entry_predicate):
    if not os.path.exists(path):
        raise IOError("class layout does not exist: %s" % path)
    loaded = []
    if os.path.isfile(path):
        archive = JarFile(path)
        try:
            entries = archive.entries()
            while entries.hasMoreElements():
                item = entries.nextElement()
                entry = item.getName()
                if not item.isDirectory() and _is_class_entry(entry) and entry_predicate(entry):
                    stream = archive.getInputStream(item)
                    try:
                        loaded.append((entry, _parse_class(stream)))
                    finally:
                        stream.close()
        finally:
            archive.close()
        return loaded

    for root, _, files in os.walk(path):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            entry = os.path.relpath(full_path, path).replace(os.sep, "/")
            if _is_class_entry(entry) and entry_predicate(entry):
                stream = FileInputStream(full_path)
                try:
                    loaded.append((entry, _parse_class(stream)))
                finally:
                    stream.close()
    return loaded


def _is_runtime_entry(entry):
    return entry.startswith(ICEBERG_PREFIX) and not entry.startswith(ICEBERG_SHADED_PREFIX)


def _is_plugin_entry(entry):
    return ICEBERG_PREFIX in entry and ICEBERG_SHADED_PREFIX not in entry


def _classes_by_name(entries):
    result = collections.defaultdict(list)
    for _, info in entries:
        result[info.name].append(info)
    return result


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


def expected_runtime_keys(build_versions):
    keys = set()
    for build_version in re.split(r"[,\s]+", build_versions.strip()):
        if not build_version:
            continue
        if re.match(r"^35[0-3]$", build_version):
            keys.add(("3.5", "1.6.1"))
        elif re.match(r"^35[4-9]$", build_version):
            keys.update((("3.5", "1.9.2"), ("3.5", "1.10.1")))
        elif re.match(r"^40[01]$", build_version):
            keys.add(("4.0", "1.10.1"))
        elif re.match(r"^40[2-4]$", build_version):
            keys.update((("4.0", "1.10.1"), ("4.0", "1.11.0")))
        elif re.match(r"^41[1-3]$", build_version):
            keys.add(("4.1", "1.11.0"))
        elif not NO_ICEBERG_BUILD_RE.match(build_version):
            raise RuntimeError("unknown build version in Iceberg audit: %s" % build_version)
    return keys


def select_runtime_paths(paths, build_versions):
    if build_versions is None:
        if not paths:
            raise RuntimeError("no Iceberg audit runtime was provided")
        return sorted(paths)
    by_key = dict((_runtime_key(path), path) for path in paths if _runtime_key(path))
    expected = expected_runtime_keys(build_versions)
    missing = expected.difference(by_key)
    if missing:
        raise RuntimeError("missing Iceberg audit runtime(s): %s" % sorted(missing))
    return [by_key[key] for key in sorted(expected)]


def find_package_private_access(layout_entries, runtime_entries, runtime_label):
    plugin_classes = _classes_by_name(layout_entries)
    runtime_classes = _classes_by_name(runtime_entries)
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
    return select_runtime_paths(paths, args.build_versions)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout", help="assembled dist jar or parallel-world directory")
    parser.add_argument("iceberg_runtime", nargs="*", help="Iceberg runtime jar(s)")
    parser.add_argument("--runtime-directory", action="append", default=[])
    parser.add_argument("--build-versions")
    args = parser.parse_args(argv)

    try:
        if not has_class_entries(args.layout):
            raise RuntimeError("assembled layout is missing or contains no class files: %s" %
                               args.layout)
        layout_entries = load_classes(args.layout, _is_plugin_entry)
        runtime_paths = _runtime_paths(args)
        if args.build_versions and expected_runtime_keys(args.build_versions) and not layout_entries:
            raise RuntimeError("assembled layout contains no Iceberg plugin classes")
        callers = set()
        findings = set()
        for runtime_path in runtime_paths:
            runtime_entries = load_classes(runtime_path, _is_runtime_entry)
            if not runtime_entries:
                raise RuntimeError("runtime contains no Iceberg classes: %s" % runtime_path)
            runtime_callers, runtime_findings = find_package_private_access(
                layout_entries, runtime_entries, os.path.basename(runtime_path))
            callers.update(runtime_callers)
            findings.update(runtime_findings)
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
          "across %d runtime(s)" % (len(callers), len(runtime_paths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
