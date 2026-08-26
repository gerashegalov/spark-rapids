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

import collections
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "check_with_resource_nesting.py"
SPEC = importlib.util.spec_from_file_location("check_with_resource_nesting", SCRIPT)
LINT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = LINT
SPEC.loader.exec_module(LINT)


def nested_source(depth):
    body = "result"
    for index in reversed(range(depth)):
        body = f"withResource(make{index}()) {{ resource{index} =>\n{body}\n}}"
    return body


class WithResourceNestingLintSuite(unittest.TestCase):
    def test_allows_nesting_at_limit(self):
        result = LINT.scan_source("Test.scala", nested_source(4), 4)
        self.assertEqual((), result.violations)

    def test_reports_each_scope_beyond_limit(self):
        result = LINT.scan_source("Test.scala", nested_source(6), 4)
        self.assertEqual([5, 6], [violation.depth for violation in result.violations])

    def test_ignores_comments_and_literals(self):
        source = '''
          // withResource(commented()) {
          val text = "withResource(notCode()) { }"
          val regex = """withResource(notCodeEither()) { }"""
        ''' + nested_source(4)
        result = LINT.scan_source("Test.scala", source, 4)
        self.assertEqual((), result.violations)

    def test_handles_blocks_inside_resource_argument(self):
        source = """
          withResource(values.map { value => convert(value) }) { first =>
            withResource(make2()) { second =>
              withResource(make3()) { third =>
                withResource(make4()) { fourth =>
                  withResource(make5()) { fifth => fifth }
                }
              }
            }
          }
        """
        result = LINT.scan_source("Test.scala", source, 4)
        self.assertEqual([5], [violation.depth for violation in result.violations])

    def test_justified_directive_exempts_subtree(self):
        source = """
          // with-resource-lint: allow-deep-nesting -- required by https://github.com/NVIDIA/cudf-spark/issues/11713
        """ + nested_source(6)
        result = LINT.scan_source("Test.scala", source, 4)
        self.assertEqual((), result.violations)
        self.assertEqual((), result.directive_errors)

    def test_directive_requires_a_reason(self):
        source = """
          // with-resource-lint: allow-deep-nesting -- short
        """ + nested_source(5)
        result = LINT.scan_source("Test.scala", source, 4)
        self.assertEqual(1, len(result.directive_errors))

    def test_directive_requires_an_issue_link(self):
        source = """
          // with-resource-lint: allow-deep-nesting -- required by the native API
        """ + nested_source(5)
        result = LINT.scan_source("Test.scala", source, 4)
        self.assertEqual(1, len(result.directive_errors))

    def test_baseline_allows_only_recorded_occurrences(self):
        result = LINT.scan_source("Test.scala", nested_source(5), 4)
        violation = result.violations[0]
        baseline = collections.Counter({violation.baseline_key: 1})
        self.assertEqual([], LINT.new_violations(result.violations, baseline))

        doubled = list(result.violations) + list(result.violations)
        self.assertEqual(1, len(LINT.new_violations(doubled, baseline)))

    def test_detects_resolved_baseline_entries(self):
        result = LINT.scan_source("Test.scala", nested_source(5), 4)
        violation = result.violations[0]
        baseline = collections.Counter({violation.baseline_key: 2})
        stale = LINT.stale_baseline_entries(result.violations, baseline)
        self.assertEqual(1, sum(stale.values()))

    def test_fingerprint_does_not_depend_on_depth(self):
        deep = LINT.scan_source("Test.scala", nested_source(5), 4).violations[0]
        shallower = LINT.scan_source("Test.scala", nested_source(5), 3).violations[-1]
        self.assertEqual(deep.fingerprint, shallower.fingerprint)

    def test_command_fails_for_new_violation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "module" / "src" / "main" / "scala"
            source_dir.mkdir(parents=True)
            (source_dir / "Test.scala").write_text(nested_source(5), encoding="utf-8")
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps({
                "version": 1,
                "maxDepth": 4,
                "trackingIssue": "https://github.com/NVIDIA/cudf-spark/issues/11713",
                "entries": [],
            }), encoding="utf-8")

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = LINT.main([
                    "--root", str(root),
                    "--baseline", str(baseline),
                ])

            self.assertEqual(1, exit_code)
            self.assertIn("nesting depth 5 exceeds 4", stderr.getvalue())

    def test_command_accepts_justified_exemption(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "module" / "src" / "main" / "scala"
            source_dir.mkdir(parents=True)
            source = (
                "// with-resource-lint: allow-deep-nesting -- required by "
                "https://github.com/NVIDIA/cudf-spark/issues/11713\n" +
                nested_source(5))
            (source_dir / "Test.scala").write_text(source, encoding="utf-8")
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps({
                "version": 1,
                "maxDepth": 4,
                "trackingIssue": "https://github.com/NVIDIA/cudf-spark/issues/11713",
                "entries": [],
            }), encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = LINT.main([
                    "--root", str(root),
                    "--baseline", str(baseline),
                ])

            self.assertEqual(0, exit_code)
            self.assertIn("lint passed", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
