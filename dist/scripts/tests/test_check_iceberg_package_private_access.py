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

from __future__ import print_function

import contextlib
import imp
import os
import shutil
import sys
import tempfile
import unittest

from java.io import DataOutputStream, FileOutputStream
from javassist.bytecode import AccessFlag, ClassFile, FieldInfo, MethodInfo


SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                      "check-iceberg-package-private-access.py")
LINT = imp.load_source("check_iceberg_package_private_access", SCRIPT)


@contextlib.contextmanager
def temporary_directory():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path)


class OutputSink(object):
    def __init__(self):
        self.parts = []

    def write(self, value):
        self.parts.append(value)

    def flush(self):
        pass

    def getvalue(self):
        return "".join(self.parts)


@contextlib.contextmanager
def captured_stream(name):
    output = OutputSink()
    original = getattr(sys, name)
    setattr(sys, name, output)
    try:
        yield output
    finally:
        setattr(sys, name, original)


def write_class(root, entry, class_name, super_name="java.lang.Object",
                methods=(), fields=(), references=(), class_references=(),
                access=AccessFlag.PUBLIC):
    class_file = ClassFile(False, class_name, super_name)
    class_file.setAccessFlags(access)
    pool = class_file.getConstPool()
    for target in class_references:
        pool.addClassInfo(target)
    for access, name, descriptor in methods:
        method = MethodInfo(pool, name, descriptor)
        method.setAccessFlags(access)
        class_file.addMethod(method)
    for access, name, descriptor in fields:
        field = FieldInfo(pool, name, descriptor)
        field.setAccessFlags(access)
        class_file.addField(field)
    for kind, owner, name, descriptor in references:
        owner_index = pool.addClassInfo(owner)
        if kind == "field":
            pool.addFieldrefInfo(owner_index, name, descriptor)
        else:
            pool.addMethodrefInfo(owner_index, name, descriptor)

    path = os.path.join(root, *entry.split("/"))
    parent = os.path.dirname(path)
    if not os.path.isdir(parent):
        os.makedirs(parent)
    output = DataOutputStream(FileOutputStream(path))
    try:
        class_file.write(output)
    finally:
        output.close()


def write_runtime(runtime, method_access=0):
    write_class(runtime, "org/apache/iceberg/p/Base.class", "org.apache.iceberg.p.Base",
                methods=((method_access, "hidden", "()V"),))


def write_inherited_caller(layout, prefix):
    write_class(layout, "org/apache/iceberg/p/GpuChild.class",
                "org.apache.iceberg.p.GpuChild", "org.apache.iceberg.p.Base")
    write_class(layout, prefix + "org/apache/iceberg/p/Caller.class",
                "org.apache.iceberg.p.Caller",
                references=(("method", "org.apache.iceberg.p.GpuChild", "hidden", "()V"),))


class IcebergPackagePrivateAccessTest(unittest.TestCase):
    def test_root_inherited_package_private_caller_passes(self):
        with temporary_directory() as root:
            layout = os.path.join(root, "layout")
            runtime = os.path.join(root, "runtime")
            write_runtime(runtime)
            write_inherited_caller(layout, "")
            with captured_stream("stdout") as stdout:
                result = LINT.main([layout, runtime])
            self.assertEqual(0, result)
            self.assertIn("1 caller classes", stdout.getvalue())

    def test_non_root_inherited_package_private_caller_fails(self):
        with temporary_directory() as root:
            layout = os.path.join(root, "layout")
            runtime = os.path.join(root, "runtime")
            write_runtime(runtime)
            write_inherited_caller(layout, "spark-shared/")
            with captured_stream("stderr") as stderr:
                result = LINT.main([layout, runtime])
            self.assertEqual(1, result)
            self.assertIn("Caller.class", stderr.getvalue())
            self.assertIn("package-private method", stderr.getvalue())

    def test_runtime_versions_are_audited_independently(self):
        with temporary_directory() as root:
            layout = os.path.join(root, "layout")
            public_runtime = os.path.join(root, "runtime-public")
            private_runtime = os.path.join(root, "runtime-private")
            write_runtime(public_runtime, AccessFlag.PUBLIC)
            write_runtime(private_runtime)
            write_inherited_caller(layout, "spark-shared/")
            with captured_stream("stderr") as stderr:
                result = LINT.main([layout, public_runtime, private_runtime])
            self.assertEqual(1, result)
            self.assertIn("runtime-private", stderr.getvalue())

    def test_array_reference_to_package_private_class_fails(self):
        with temporary_directory() as root:
            layout = os.path.join(root, "layout")
            runtime = os.path.join(root, "runtime")
            write_class(runtime, "org/apache/iceberg/p/Hidden.class",
                        "org.apache.iceberg.p.Hidden", access=0)
            write_class(layout, "spark-shared/org/apache/iceberg/p/Caller.class",
                        "org.apache.iceberg.p.Caller",
                        class_references=("[Lorg.apache.iceberg.p.Hidden;",))
            with captured_stream("stderr") as stderr:
                result = LINT.main([layout, runtime])
            self.assertEqual(1, result)
            self.assertIn("package-private class", stderr.getvalue())

    def test_package_private_field_fails(self):
        with temporary_directory() as root:
            layout = os.path.join(root, "layout")
            runtime = os.path.join(root, "runtime")
            write_class(runtime, "org/apache/iceberg/p/Base.class",
                        "org.apache.iceberg.p.Base", fields=((0, "hidden", "I"),))
            write_class(layout, "spark-shared/org/apache/iceberg/p/Caller.class",
                        "org.apache.iceberg.p.Caller",
                        references=(("field", "org.apache.iceberg.p.Base", "hidden", "I"),))
            with captured_stream("stderr") as stderr:
                result = LINT.main([layout, runtime])
            self.assertEqual(1, result)
            self.assertIn("package-private field", stderr.getvalue())

    def test_scala_synthetic_caller_fails(self):
        with temporary_directory() as root:
            layout = os.path.join(root, "layout")
            runtime = os.path.join(root, "runtime")
            write_runtime(runtime)
            write_class(layout, "spark-shared/org/apache/iceberg/p/Caller$anon$1.class",
                        "org.apache.iceberg.p.Caller$anon$1",
                        references=(("method", "org.apache.iceberg.p.Base",
                                     "hidden", "()V"),))
            with captured_stream("stderr") as stderr:
                result = LINT.main([layout, runtime])
            self.assertEqual(1, result)
            self.assertIn("Caller$anon$1.class", stderr.getvalue())

    def test_public_override_stops_inherited_member_resolution(self):
        with temporary_directory() as root:
            layout = os.path.join(root, "layout")
            runtime = os.path.join(root, "runtime")
            write_runtime(runtime)
            write_class(layout, "org/apache/iceberg/p/GpuChild.class",
                        "org.apache.iceberg.p.GpuChild", "org.apache.iceberg.p.Base",
                        methods=((AccessFlag.PUBLIC, "hidden", "()V"),))
            write_class(layout, "spark-shared/org/apache/iceberg/p/Caller.class",
                        "org.apache.iceberg.p.Caller",
                        references=(("method", "org.apache.iceberg.p.GpuChild",
                                     "hidden", "()V"),))
            with captured_stream("stdout"):
                self.assertEqual(0, LINT.main([layout, runtime]))

    def test_missing_and_empty_layouts_fail(self):
        with temporary_directory() as root:
            runtime = os.path.join(root, "runtime")
            write_runtime(runtime)
            missing = os.path.join(root, "missing")
            empty = os.path.join(root, "empty")
            os.makedirs(empty)
            with captured_stream("stderr"):
                self.assertEqual(2, LINT.main([missing, runtime]))
                self.assertEqual(2, LINT.main([empty, runtime]))

    def test_build_versions_require_every_runtime(self):
        paths = (
            "/tmp/iceberg-spark-runtime-3.5_2.13-1.9.2.jar",
            "/tmp/iceberg-spark-runtime-3.5_2.13-1.10.1.jar",
        )
        self.assertEqual(sorted(paths), LINT.select_runtime_paths(paths, "359"))
        with self.assertRaises(RuntimeError):
            LINT.select_runtime_paths(paths[:1], "359")
        with self.assertRaises(RuntimeError):
            LINT.select_runtime_paths(paths, "future-version")

    def test_no_runtime_fails(self):
        with self.assertRaises(RuntimeError):
            LINT.select_runtime_paths([], None)

    def test_malformed_class_fails_closed(self):
        with temporary_directory() as root:
            layout = os.path.join(root, "layout")
            runtime = os.path.join(root, "runtime")
            write_runtime(runtime)
            path = os.path.join(layout, "spark-shared/org/apache/iceberg/p/Broken.class")
            os.makedirs(os.path.dirname(path))
            output = FileOutputStream(path)
            try:
                output.write(bytearray([0, 1, 2, 3]))
            finally:
                output.close()
            with captured_stream("stderr"):
                self.assertEqual(2, LINT.main([layout, runtime]))


if __name__ == "__main__":
    unittest.main()
