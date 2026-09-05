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

"""Require Iceberg package-private callers to be stored at the dist jar root."""

import argparse
import collections
import os
import re
import struct
import sys
import zipfile


ACC_PUBLIC = 0x0001
ACC_PRIVATE = 0x0002
ACC_PROTECTED = 0x0004
MEMBER_VISIBILITY = ACC_PUBLIC | ACC_PRIVATE | ACC_PROTECTED
ICEBERG_PREFIX = "org/apache/iceberg/"
ICEBERG_SHADED_PREFIX = "org/apache/iceberg/shaded/"
SHIM_DIR_RE = re.compile(r"^spark[0-9][0-9a-z]*$")
DESCRIPTOR_CLASS_RE = re.compile(r"L([^;<>()\[\]]+);")
MAX_ZIP_CLASS_ENTRY_BYTES = 64 * 1024 * 1024
MAX_ZIP_TOTAL_CLASS_BYTES = 1024 * 1024 * 1024


Member = collections.namedtuple("Member", ("access", "name", "descriptor"))
MemberRef = collections.namedtuple("MemberRef", ("kind", "owner", "name", "descriptor"))
ClassInfo = collections.namedtuple(
    "ClassInfo",
    ("name", "access", "super_name", "interfaces", "fields", "methods", "class_refs",
     "member_refs"))
Finding = collections.namedtuple("Finding", ("entry", "caller", "target", "reason"))


def _read_u1(data, offset):
    return data[offset], offset + 1


def _read_u2(data, offset):
    return struct.unpack_from(">H", data, offset)[0], offset + 2


def _read_u4(data, offset):
    return struct.unpack_from(">I", data, offset)[0], offset + 4


def _skip_attributes(data, offset):
    attribute_count, offset = _read_u2(data, offset)
    for _ in range(attribute_count):
        _, offset = _read_u2(data, offset)
        length, offset = _read_u4(data, offset)
        offset += length
    return offset


def _parse_members(data, offset, constant_pool):
    members = []
    member_count, offset = _read_u2(data, offset)
    for _ in range(member_count):
        access, offset = _read_u2(data, offset)
        name_index, offset = _read_u2(data, offset)
        descriptor_index, offset = _read_u2(data, offset)
        members.append(Member(access, constant_pool[name_index], constant_pool[descriptor_index]))
        offset = _skip_attributes(data, offset)
    return members, offset


def _class_name(constant_pool, class_index):
    if class_index == 0:
        return None
    tag, name_index = constant_pool[class_index]
    if tag != "class":
        raise ValueError("invalid class constant")
    return constant_pool[name_index]


def _descriptor_classes(descriptor):
    return set(DESCRIPTOR_CLASS_RE.findall(descriptor))


def _normalized_class_names(value):
    if value is None:
        return set()
    if value.startswith("["):
        return _descriptor_classes(value)
    return {value}


def parse_class_file(data):
    magic, offset = _read_u4(data, 0)
    if magic != 0xCAFEBABE:
        raise ValueError("not a class file")

    # minor_version, major_version
    _, offset = _read_u2(data, offset)
    _, offset = _read_u2(data, offset)

    constant_pool_count, offset = _read_u2(data, offset)
    constant_pool = [None] * constant_pool_count
    index = 1
    while index < constant_pool_count:
        tag, offset = _read_u1(data, offset)
        if tag == 1:  # CONSTANT_Utf8
            length, offset = _read_u2(data, offset)
            raw = data[offset:offset + length]
            offset += length
            constant_pool[index] = raw.decode("utf-8", errors="replace")
        elif tag in (3, 4):  # Integer, Float
            offset += 4
        elif tag in (5, 6):  # Long, Double
            offset += 8
            index += 1
        elif tag == 7:  # Class
            name_index, offset = _read_u2(data, offset)
            constant_pool[index] = ("class", name_index)
        elif tag == 8:  # String
            offset += 2
        elif tag in (9, 10, 11):  # Fieldref, Methodref, InterfaceMethodref
            class_index, offset = _read_u2(data, offset)
            name_and_type_index, offset = _read_u2(data, offset)
            constant_pool[index] = ("member", tag, class_index, name_and_type_index)
        elif tag == 12:  # NameAndType
            name_index, offset = _read_u2(data, offset)
            descriptor_index, offset = _read_u2(data, offset)
            constant_pool[index] = ("name-and-type", name_index, descriptor_index)
        elif tag in (17, 18):  # Dynamic, InvokeDynamic
            bootstrap_index, offset = _read_u2(data, offset)
            name_and_type_index, offset = _read_u2(data, offset)
            constant_pool[index] = ("dynamic", bootstrap_index, name_and_type_index)
        elif tag == 15:  # MethodHandle
            offset += 3
        elif tag == 16:  # MethodType
            descriptor_index, offset = _read_u2(data, offset)
            constant_pool[index] = ("method-type", descriptor_index)
        elif tag in (19, 20):  # Module, Package
            offset += 2
        else:
            raise ValueError("unknown constant pool tag %s" % tag)
        index += 1

    access, offset = _read_u2(data, offset)
    this_class_index, offset = _read_u2(data, offset)
    super_class_index, offset = _read_u2(data, offset)
    name = _class_name(constant_pool, this_class_index)
    super_name = _class_name(constant_pool, super_class_index)

    interface_count, offset = _read_u2(data, offset)
    interfaces = []
    for _ in range(interface_count):
        interface_index, offset = _read_u2(data, offset)
        interfaces.append(_class_name(constant_pool, interface_index))

    fields, offset = _parse_members(data, offset, constant_pool)
    methods, offset = _parse_members(data, offset, constant_pool)

    class_refs = set()
    member_refs = []
    for value in constant_pool:
        if not isinstance(value, tuple):
            continue
        if value[0] == "class":
            class_refs.update(_normalized_class_names(constant_pool[value[1]]))
        elif value[0] == "member":
            _, member_tag, owner_index, name_and_type_index = value
            owner = _class_name(constant_pool, owner_index)
            name_and_type = constant_pool[name_and_type_index]
            member_name = constant_pool[name_and_type[1]]
            descriptor = constant_pool[name_and_type[2]]
            kind = "field" if member_tag == 9 else "method"
            member_refs.append(MemberRef(kind, owner, member_name, descriptor))
            class_refs.add(owner)
            class_refs.update(_descriptor_classes(descriptor))
        elif value[0] in ("name-and-type", "method-type"):
            descriptor_index = value[-1]
            class_refs.update(_descriptor_classes(constant_pool[descriptor_index]))
        elif value[0] == "dynamic":
            name_and_type = constant_pool[value[2]]
            class_refs.update(_descriptor_classes(constant_pool[name_and_type[2]]))

    for member in fields + methods:
        class_refs.update(_descriptor_classes(member.descriptor))
    class_refs.discard(name)

    return ClassInfo(
        name, access, super_name, tuple(interfaces), tuple(fields), tuple(methods),
        frozenset(class_refs), tuple(member_refs))


def iter_class_entries(path):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            total_class_bytes = 0
            for info in archive.infolist():
                entry = info.filename
                if (not entry.endswith(".class") or entry.endswith("/module-info.class") or
                        entry.startswith("META-INF/versions/")):
                    continue
                if info.file_size > MAX_ZIP_CLASS_ENTRY_BYTES:
                    raise RuntimeError(
                        "refusing to read oversized class entry %s (%s bytes)" %
                        (entry, info.file_size))
                total_class_bytes += info.file_size
                if total_class_bytes > MAX_ZIP_TOTAL_CLASS_BYTES:
                    raise RuntimeError("refusing to read oversized zip class payload")
                yield entry, archive.read(info)
        return

    for root, _, files in os.walk(path):
        for file_name in files:
            if not file_name.endswith(".class") or file_name == "module-info.class":
                continue
            full_path = os.path.join(root, file_name)
            entry = os.path.relpath(full_path, path).replace(os.sep, "/")
            with open(full_path, "rb") as class_file:
                yield entry, class_file.read()


def load_runtime_classes(paths):
    classes = {}
    for path in paths:
        for _, data in iter_class_entries(path):
            info = parse_class_file(data)
            if info.name.startswith(ICEBERG_PREFIX) and not info.name.startswith(
                    ICEBERG_SHADED_PREFIX):
                classes[info.name] = info
    return classes


def _find_member(classes, owner, name, descriptor, kind):
    pending = [owner]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        info = classes.get(current)
        if info is None:
            continue
        members = info.fields if kind == "field" else info.methods
        for member in members:
            if member.name == name and member.descriptor == descriptor:
                return current, member
        if info.super_name:
            pending.append(info.super_name)
        pending.extend(info.interfaces)
    return None, None


def _is_root_entry(entry):
    first = entry.split("/", 1)[0]
    return first != "spark-shared" and not SHIM_DIR_RE.match(first)


def find_package_private_access(layout_path, runtime_classes):
    findings = set()
    callers = set()
    for entry, data in iter_class_entries(layout_path):
        caller = parse_class_file(data)
        caller_findings = set()
        for target_name in caller.class_refs:
            target = runtime_classes.get(target_name)
            if target is not None and not target.access & ACC_PUBLIC:
                caller_findings.add((target_name, "package-private class"))

        for reference in caller.member_refs:
            if not reference.owner.startswith(ICEBERG_PREFIX):
                continue
            declaring_class, member = _find_member(
                runtime_classes, reference.owner, reference.name, reference.descriptor,
                reference.kind)
            if member is not None and not member.access & MEMBER_VISIBILITY:
                target = "%s.%s%s" % (
                    declaring_class, member.name,
                    (":" if reference.kind == "field" else "") + member.descriptor)
                caller_findings.add((target, "package-private %s" % reference.kind))

        if caller_findings:
            callers.add((entry, caller.name))
            for target, reason in caller_findings:
                findings.add(Finding(entry, caller.name, target, reason))
    return sorted(callers), sorted(findings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "layout", help="assembled dist jar or dist/target/parallel-world directory")
    parser.add_argument(
        "iceberg_runtime", nargs="+", help="Iceberg Spark runtime jar(s) to resolve access against")
    args = parser.parse_args()

    runtime_classes = load_runtime_classes(args.iceberg_runtime)
    if not runtime_classes:
        parser.error("no org.apache.iceberg runtime classes found")

    callers, findings = find_package_private_access(args.layout, runtime_classes)
    if not callers:
        print("No Iceberg package-private callers were detected; refusing to pass an empty audit",
              file=sys.stderr)
        return 2
    violations = [(entry, caller) for entry, caller in callers if not _is_root_entry(entry)]

    if violations:
        print("Iceberg package-private access must be root-safe; found non-root callers:",
              file=sys.stderr)
        for entry, caller in violations:
            print("  %s (%s)" % (entry, caller.replace("/", ".")), file=sys.stderr)
            caller_findings = [finding for finding in findings if finding.entry == entry]
            for finding in caller_findings:
                print("    -> %s [%s]" %
                      (finding.target.replace("/", "."), finding.reason), file=sys.stderr)
        print("Move each caller to a module listed in dist/root-safe-module-classes.txt "
              "or promote it with dist/unshimmed-common-from-single-shim.txt.", file=sys.stderr)
        return 1

    print("Iceberg package-private access audit passed: %d caller classes are at the jar root" %
          len(callers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
