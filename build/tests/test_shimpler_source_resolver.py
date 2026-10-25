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

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'build'))

from shimpler_source_graph import load_source_graph
from shimpler_source_resolver import (candidate_inherited_sources, discover_shim_modules,
                                     resolve_sources, scan_module_sources, validate_legacy_shadow)
from shimpler_source_selectors import SourceSelector


LEGACY_330_331_ONLY = {
    'sql-plugin/src/main/spark330/scala/com/nvidia/spark/rapids/shims/parquet/'
    'ParquetLegacyNanoAsLongShims.scala',
    'sql-plugin/src/main/spark330/scala/org/apache/spark/sql/rapids/shims/'
    'RapidsHadoopWriterUtils.scala',
}

LEGACY_330_THROUGH_333_ONLY = {
    'sql-plugin/src/main/spark330/scala/com/nvidia/spark/rapids/shims/'
    'GetSequenceSize.scala',
    'sql-plugin/src/main/spark330/scala/org/apache/spark/sql/rapids/shims/'
    'RapidsErrorUtils.scala',
}

# Pre-conversion selections are stored as digests to keep the oracle compact.
LEGACY_SOURCE_SNAPSHOTS = {
    '340': (208, '890009dc3321362baa0d1af71961d7af628750c0c5d99c65ed49b7b64bf6ab89'),
    '341': (208, '95b0ec66c0b12a44a0a41d599d007cbefc0f62f06bf507f53a2317bd02fae614'),
    '342': (209, '0e6a2e412dda75bc0f2ba7177c9c07c09c0f8885453bb40f610584b7a1e591b2'),
    '343': (209, '5359b8509ab16e8f377f387c8b1132ad9f1088799f661a9a518efeeac660fd23'),
    '344': (209, '2aaa6988c70386a74c6f7ba5f8328e2b6863934aba9de2d6e7e78767b5b4d44a'),
    '351': (253, 'fd894e32839a336d7adf67c0bf0735bc1a0342873018d0a5ef67dbbcb5168847'),
    '352': (253, 'c9f27ab13c1aa246520c9ede2f2c43a1affff31b0bbdda5fd72373d1aca50b3b'),
    '353': (253, 'b5a38b550f5b0b70488c9a59330fbd8640ff8518f2c9b45865bb267441ab4933'),
    '354': (252, '89bca34554423999c417befe326d7670495864978e11d210cb0a8f6745660c9d'),
    '355': (251, '2e39c89505f80abdbb47c2cc41a89dec6b23c926035d7b8d63dcff86cb2782a7'),
    '356': (251, 'ea363a06407b273379bfc65f6ec28126790eca69e3202c807c7fbc567251a849'),
    '357': (251, 'dab6ab2e33b38ac17ca85e4a279e19b72a675ae45b6bee39000511a252123582'),
    '358': (251, '1606d7d71fb13b7c2aecb2413104311b26ceaaf40000fb8beea266fef2ebb79d'),
    '359': (255, 'ab579153d940febdf8ccae8ca7cd1f69a8ecb00312c8fb0d3be8a090c16d7921'),
    '401': (248, '58b828c5f62785661bca6857dba24c7c1338aa002e31336d80d36478ee999c49'),
    '403': (249, '9658703e1a146054d1abe20a6c1ca093797128af9534292f1cdef3f0ce5d705b'),
    '404': (249, '01544872e4b3e4b9c4b8d20558d03df29344c7b7032b7422e37c9c822d4b2822'),
    '411': (257, 'd9783d12727d99efc9eb484f4f9cee0d083bbbd8094c20a298cbfb1b73c9bcb2'),
    '412': (248, '083e1f47f810e160cf296bd91451f69cc014770ce0e02c98c1f38a3a911ff0d0'),
    '413': (249, '380f2aa75afe86dd06ace15921c48ff5b8c2f95d7a5497939a9a25671622eb40'),
    '500': (239, 'a35c3ab8861526e0caeb8f6d16efb08e21ad156d26e0f252db8b73067a18c431'),
}


class SourceResolverTest(unittest.TestCase):
    def setUp(self):
        self.graph = {'355': None, '356': '355', '357': '356',
                      '420': None, '500': '420'}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.module = Path(self.temp.name) / 'nested' / 'module'

    def source(self, owner, name, selectors, language='scala', scope='main'):
        path = self.module / 'src' / scope / ('spark' + owner) / language / 'pkg' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if selectors is None:
            text = 'package pkg\nclass Example\n'
        else:
            text = ('/*** spark-rapids-shim-json-lines\n%s\n'
                    'spark-rapids-shim-json-lines ***/\npackage pkg\nclass Example\n'
                    % '\n'.join(selectors))
        path.write_text(text, encoding='utf-8')
        return str(path)

    def scan(self):
        return scan_module_sources(str(self.module), self.graph)

    def test_scan_scala_java_and_unannotated(self):
        scala = self.source('355', 'Example.scala', ['{"spark":"355"}'])
        java = self.source('355', 'Other.java', ['{"spark":{"only":"355"}}'], 'java', 'test')
        unannotated = self.source('355', 'Skipped.scala', None)
        variants, skipped = self.scan()
        self.assertEqual(set(variant.physical_path for variant in variants), {scala, java})
        self.assertEqual(skipped, [unannotated])
        self.assertEqual({variant.logical_path for variant in variants},
                         {'src/main/scala/pkg/Example.scala', 'src/test/java/pkg/Other.java'})
        self.assertEqual(discover_shim_modules(self.temp.name), [str(self.module)])

    def test_scan_rejects_unsorted_or_duplicate_selectors(self):
        for selectors in (['{"spark":"356"}', '{"spark":"355"}'],
                          ['{"spark":"355"}', '{"spark":{"introduced":"355"}}']):
            self.source('355', 'Example.scala', selectors)
            with self.assertRaisesRegex(ValueError, 'unique and sorted'):
                self.scan()

    def test_scan_rejects_multiple_selector_blocks(self):
        source = self.source('355', 'Example.scala', ['{"spark":"355"}'])
        with open(source, 'a', encoding='utf-8') as output:
            output.write('/*** spark-rapids-shim-json-lines\n'
                         '{"spark":"356"}\nspark-rapids-shim-json-lines ***/\n')
        with self.assertRaisesRegex(ValueError, 'Malformed shim selector block'):
            self.scan()

    def test_nearest_overlay_and_exact_only(self):
        first = self.source('355', 'Example.scala', ['{"spark":{"introduced":"355"}}'])
        overlay = self.source('357', 'Example.scala', ['{"spark":{"introduced":"357"}}'])
        only = self.source('500', 'Other.scala', ['{"spark":{"only":"500"}}'])
        variants, _ = self.scan()
        logical = 'src/main/scala/pkg/Example.scala'
        self.assertEqual(resolve_sources(self.graph, variants, '356')[logical].physical_path, first)
        selected = resolve_sources(self.graph, variants, '357')[logical]
        self.assertEqual(selected.physical_path, overlay)
        self.assertEqual(selected.ancestry_path, ('357',))
        self.assertEqual(resolve_sources(self.graph, variants, '500')
                         ['src/main/scala/pkg/Other.scala'].physical_path, only)
        self.assertNotIn('src/main/scala/pkg/Other.scala',
                         resolve_sources(self.graph, variants, '420'))

    def test_multiple_introduction_roots_and_exact_override(self):
        base = self.source('355', 'Example.scala',
                           ['{"spark":{"introduced":["355","420"]}}'])
        override = self.source('500', 'Example.scala', ['{"spark":{"only":"500"}}'])
        variants, _ = self.scan()
        logical = 'src/main/scala/pkg/Example.scala'
        self.assertEqual(resolve_sources(self.graph, variants, '357')[logical].physical_path, base)
        self.assertEqual(resolve_sources(self.graph, variants, '420')[logical].physical_path, base)
        self.assertEqual(resolve_sources(self.graph, variants, '500')[logical].physical_path,
                         override)

    def test_ambiguous_variants_rejected(self):
        self.source('355', 'Example.scala', ['{"spark":"356"}'])
        self.source('356', 'Example.scala', ['{"spark":"356"}'])
        variants, _ = self.scan()
        with self.assertRaisesRegex(ValueError, 'Ambiguous shim sources for 356'):
            resolve_sources(self.graph, variants, '356')

    def test_nearer_override_wins_regardless_of_variant_order(self):
        self.source('355', 'Example.scala', ['{"spark":{"introduced":"355"}}'])
        self.source('356', 'Example.scala', ['{"spark":{"introduced":"355"}}'])
        override = self.source('357', 'Example.scala', ['{"spark":{"only":"357"}}'])
        variants, _ = self.scan()
        logical = 'src/main/scala/pkg/Example.scala'
        for ordered in (variants, list(reversed(variants))):
            self.assertEqual(resolve_sources(self.graph, ordered, '357')[logical].physical_path,
                             override)

    def test_shadow_keeps_exact_removals(self):
        exact = self.source('420', 'Example.scala', ['{"spark":"420"}'])
        variants, _ = self.scan()
        self.assertEqual(validate_legacy_shadow(self.graph, variants), (exact,))
        candidates, _ = candidate_inherited_sources(self.graph, variants)
        self.assertEqual(resolve_sources(self.graph, candidates, '500'), {})

    def test_shadow_accepts_descendant_closed_membership(self):
        self.source('355', 'Example.scala',
                    ['{"spark":"355"}', '{"spark":"356"}', '{"spark":"357"}'])
        variants, _ = self.scan()
        self.assertEqual(validate_legacy_shadow(self.graph, variants), ())

    def test_versioned_package_path_stays_exact(self):
        source = self.source('357', 'spark357/Provider.scala', ['{"spark":"357"}'])
        variants, _ = self.scan()
        self.assertEqual(validate_legacy_shadow(self.graph, variants), (source,))

    def test_shadow_requires_oracle_for_converted_sources(self):
        self.source('355', 'Example.scala', ['{"spark":{"introduced":"355"}}'])
        variants, _ = self.scan()
        with self.assertRaisesRegex(ValueError, 'pre-conversion oracle is required'):
            validate_legacy_shadow(self.graph, variants)

    def test_repository_shadow_matches_pre_conversion_exact_selection(self):
        graphs = (load_source_graph(str(ROOT / 'pom.xml')),
                  load_source_graph(str(ROOT / 'scala2.13' / 'pom.xml')))
        targets = set(graphs[0]) | set(graphs[1])
        modules = discover_shim_modules(str(ROOT))
        self.assertIn(str(ROOT / 'delta-lake' / 'common'), modules)
        for module in modules:
            variants, _ = scan_module_sources(module, targets)
            legacy_variants = []
            for variant in variants:
                selectors = []
                physical = Path(variant.physical_path).relative_to(ROOT).as_posix()
                for selector in variant.selectors:
                    if selector.kind == 'introduced' and selector.targets == ('330',):
                        # These sources have version-specific replacements after their last
                        # legacy exact selector.
                        if physical in LEGACY_330_331_ONLY:
                            legacy_targets = ('330', '331')
                        elif physical in LEGACY_330_THROUGH_333_ONLY:
                            legacy_targets = ('330', '331', '332', '333')
                        else:
                            legacy_targets = ('330', '331', '332', '333', '334')
                        selectors.extend(SourceSelector('exact', (target,), {})
                                         for target in legacy_targets)
                    elif selector.kind == 'introduced' and selector.targets == ('331',):
                        selectors.extend(SourceSelector('exact', (target,), {})
                                         for target in ('331', '332', '333', '334'))
                    elif selector.kind == 'introduced' and selector.targets == ('332',):
                        selectors.extend(SourceSelector('exact', (target,), {})
                                         for target in ('332', '333', '334'))
                    elif selector.kind == 'introduced' and selector.targets == ('350',):
                        selectors.extend(SourceSelector('exact', (target,), {})
                                         for target in ('350', '351'))
                    elif selector.kind == 'introduced' and selector.targets == ('358',):
                        selectors.extend(SourceSelector('exact', (target,), {})
                                         for target in ('358', '359'))
                    elif selector.kind == 'introduced' and selector.targets == ('400',):
                        selectors.extend(SourceSelector('exact', (target,), {})
                                         for target in ('400', '401'))
                    elif selector.kind == 'introduced' and selector.targets == ('402',):
                        selectors.extend(SourceSelector('exact', (target,), {})
                                         for target in ('402', '403'))
                    elif selector.kind == 'introduced' and selector.targets == ('403',):
                        selectors.extend(SourceSelector('exact', (target,), {})
                                         for target in ('403', '404'))
                    elif selector.kind == 'introduced' and selector.targets == ('411',):
                        selectors.extend(SourceSelector('exact', (target,), {})
                                         for target in ('411', '412'))
                    elif selector.kind == 'introduced' and selector.targets == ('412',):
                        selectors.extend(SourceSelector('exact', (target,), {})
                                         for target in ('412', '413'))
                    elif selector.kind == 'introduced' and selector.targets == ('420',):
                        selectors.extend(SourceSelector('exact', (target,), {})
                                         for target in ('420', '500'))
                    else:
                        selectors.append(selector)
                legacy_variants.append(variant._replace(selectors=tuple(selectors)))
            for graph in graphs:
                shadow_targets = tuple(target for target in graph
                                       if target not in LEGACY_SOURCE_SNAPSHOTS)
                validate_legacy_shadow(graph, legacy_variants, shadow_targets)
                for target in graph:
                    if target in LEGACY_SOURCE_SNAPSHOTS:
                        continue
                    actual = resolve_sources(graph, variants, target)
                    legacy = resolve_sources(graph, legacy_variants, target)
                    self.assertEqual({key: value.physical_path for key, value in actual.items()},
                                     {key: value.physical_path for key, value in legacy.items()},
                                     '%s at %s' % (module, target))

    def test_repository_sources_match_pre_conversion_snapshots(self):
        graph = load_source_graph(str(ROOT / 'pom.xml'))
        scala_213_graph = load_source_graph(str(ROOT / 'scala2.13' / 'pom.xml'))
        targets = set(graph) | set(scala_213_graph)
        modules = []
        for module in discover_shim_modules(str(ROOT)):
            variants, _ = scan_module_sources(module, targets)
            modules.append((Path(module).relative_to(ROOT).as_posix(), variants))
        for target, expected in LEGACY_SOURCE_SNAPSHOTS.items():
            target_graph = graph if target in graph else scala_213_graph
            rows = []
            for module_path, variants in modules:
                for logical, selected in resolve_sources(target_graph, variants, target).items():
                    physical = Path(selected.physical_path).relative_to(ROOT).as_posix()
                    rows.append('%s\t%s\t%s' % (module_path, logical, physical))
            digest = hashlib.sha256(('\n'.join(sorted(rows)) + '\n').encode()).hexdigest()
            self.assertEqual((len(rows), digest), expected, target)

    def test_repository_rapids_error_utils_owners(self):
        graphs = (load_source_graph(str(ROOT / 'pom.xml')),
                  load_source_graph(str(ROOT / 'scala2.13' / 'pom.xml')))
        targets = set(graphs[0]) | set(graphs[1])
        variants, _ = scan_module_sources(str(ROOT / 'sql-plugin'), targets)
        logical = 'src/main/scala/org/apache/spark/sql/rapids/shims/RapidsErrorUtils.scala'
        expected = {
            '350db143': 'spark350db143',
            '400db173': 'spark350db143',
            '410db183': 'spark350db143',
            '400': 'spark400',
            '401': 'spark400',
        }
        for graph in graphs:
            for target, owner in expected.items():
                if target in graph:
                    selected = resolve_sources(graph, variants, target)[logical]
                    self.assertIn('/src/main/%s/' % owner, selected.physical_path)

    def test_source_inspector_reports_actual_selection_and_infers_scala_tree(self):
        command = [str(ROOT / 'build' / 'shimpler'), 'sources', '--target', '359',
                   '--module', 'sql-plugin', '--format=json']
        output = subprocess.check_output(command, universal_newlines=True)
        result = json.loads(output)
        self.assertEqual(result['scala'], '2.12')
        self.assertTrue(any(row['selector_kind'] == 'introduced' for row in result['sources']))
        self.assertTrue(any(row['selector_kind'] == 'exact' for row in result['sources']))
        self.assertTrue(any(row['physical_path'].endswith('PartitionedFileUtilsShim.scala')
                            and row['introduction_target'] == '358'
                            for row in result['sources']))
        command[3] = '500'
        result = json.loads(subprocess.check_output(command, universal_newlines=True))
        self.assertEqual(result['scala'], '2.13')
        self.assertTrue(result['sources'])


if __name__ == '__main__':
    unittest.main()
