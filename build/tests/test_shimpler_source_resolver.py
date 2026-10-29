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

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'build'))

from shimpler_source_graph import load_source_graph
from shimpler_source_generator import generate_sources
from shimpler_source_resolver import (candidate_inherited_sources, discover_shim_modules,
                                     resolve_sources, scan_module_sources, validate_legacy_shadow)
from shimpler_source_selectors import SourceSelector, selector_membership
from shimpler_source_targets import EXTERNAL_EXACT_TARGETS


def selector(descendants, *targets):
    return json.dumps({'spark': targets, 'carryForward': 'all' if descendants else 'none'},
                      separators=(',', ':'))


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
    '340': (210, '8f1d84e9caedcba0a7f3651bed361d2eb6f816e4e2f0a8e8f1e20a1ccf6be8f1'),
    '341': (210, 'c99320c37bc0a21e538a86e9689ad9dffea038808215292027abd3c872288873'),
    '342': (211, 'fcee0e42a14166c4f924e5670a188e823cd959c93a6d6c4ca82ddbb95e3f5a33'),
    '343': (211, '71fd41b0fd2976364f17e30d758d0e8cf656aa9cfc5ee82efe94423b46d6d504'),
    '344': (211, '0e84e31107a97e537dce4b08e2b1327e0645b5ac9e41c419cf2cce0411f3f5a2'),
    '351': (256, 'f945f231ca237937a0a5ca153d1db397514f7db1f90d3a4de6bec2833f13b3be'),
    '352': (256, '19af00721eeb0c745a3ffc517c8725a801e2b453ae976ae1851c42090f930eb4'),
    '353': (256, 'd92f98c92afc6470388155c786a2b1eb94791732b45c69403b50b15438c0b747'),
    '354': (255, '2d274df92efa3e346e272119c4232dd8c7fc5d05db2442f9547c6a8beb761670'),
    '355': (254, 'ffef48789d55ea0c29f94dec4892209a424ec58d0e8c289a6e05ee6857a9510d'),
    '356': (254, '246f81abc77c11e6ff45b2e265736c131da60c5fd564cf3ec0409fad820d38f1'),
    '357': (254, 'cb4c2629a774ad32f3f5278fda5db7c4ed5246240363ba8788e359a48c74c956'),
    '358': (254, '4873162316b4df19d1836b612f668e3c534e3287e4f616f1599935e21b618a28'),
    '359': (258, 'e9a9f9de357fa03ca35e2285769a3de4b98f79ed005231b6bf7af96f1a7ca966'),
    '401': (254, '97a23fdbff4e18131de714445aa3dc6cdb67b4b0e82f83511e820b7fe1e1bc50'),
    '403': (254, 'c0f73caf31ef2f158ce86d668fb8879a7cf2c6ec7c4678596e7e76ea9209e56d'),
    '404': (254, '85702e5812014c8f94c80b349939033088d24533c8e78affaeb3bd40d0046654'),
    '411': (264, 'b1f04abaa3ce43528d545326e61ebcf98b32dee61c83a56c0d8f9ead629f7047'),
    '412': (255, 'b0717277f362f739731302811c1bdbe8b00e84a6be6a69e721c3e11f8b5479fc'),
    '413': (255, '994c62e7cf4634293763aab520e8b2de5e5a5f033f04e1e95859debed30d30d4'),
    '500': (241, '1361b486c4bb3041d76acdc84538ad7cc9272dfe98f36936fef9578504d6c73e'),
}


class SourceResolverTest(unittest.TestCase):
    def test_vendored_tools_match_pin(self):
        manifest = ROOT / 'build' / 'shimpler_source_tools.sha256'
        for entry in manifest.read_text().splitlines():
            if not entry or entry.startswith('#'):
                continue
            expected, filename = entry.split('  ', 1)
            actual = hashlib.sha256((ROOT / 'build' / filename).read_bytes()).hexdigest()
            self.assertEqual(expected, actual, filename)

    def test_generation_accepts_other_scala_profiles(self):
        graphs = (load_source_graph(str(ROOT / 'pom.xml')),
                  load_source_graph(str(ROOT / 'scala2.13' / 'pom.xml')))
        known_targets = set().union(*graphs)
        with tempfile.TemporaryDirectory() as temp:
            for target in ('400', '500'):
                generated = Path(temp) / target
                selected = generate_sources(str(ROOT / 'sql-plugin'), str(generated),
                                            graphs[1], target, known_targets,
                                            EXTERNAL_EXACT_TARGETS)
                self.assertTrue(selected)

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
        scala = self.source('355', 'Example.scala', [selector(False, '355')])
        java = self.source('355', 'Other.java', [selector(False, '355')], 'java', 'test')
        unannotated = self.source('355', 'Skipped.scala', None)
        variants, skipped = self.scan()
        self.assertEqual(set(variant.physical_path for variant in variants), {scala, java})
        self.assertEqual(skipped, [unannotated])
        self.assertEqual({variant.logical_path for variant in variants},
                         {'src/main/scala/pkg/Example.scala', 'src/test/java/pkg/Other.java'})
        self.assertEqual(discover_shim_modules(self.temp.name), [str(self.module)])

    def test_scan_rejects_unsorted_or_duplicate_selectors(self):
        for selectors in ([selector(False, '356'), selector(False, '355')],
                          [selector(False, '355'), selector(True, '355')]):
            self.source('355', 'Example.scala', selectors)
            with self.assertRaisesRegex(ValueError, 'unique and sorted'):
                self.scan()

    def test_scan_rejects_multiple_selector_blocks(self):
        source = self.source('355', 'Example.scala', [selector(False, '355')])
        with open(source, 'a', encoding='utf-8') as output:
            output.write('/*** spark-rapids-shim-json-lines\n'
                         '%s\nspark-rapids-shim-json-lines ***/\n' % selector(False, '356'))
        with self.assertRaisesRegex(ValueError, 'Malformed shim selector block'):
            self.scan()

    def test_nearest_overlay_and_exact_only(self):
        first = self.source('355', 'Example.scala', [selector(True, '355')])
        overlay = self.source('357', 'Example.scala', [selector(True, '357')])
        only = self.source('500', 'Other.scala', [selector(False, '500')])
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
        base = self.source('355', 'Example.scala', [selector(True, '355', '420')])
        override = self.source('500', 'Example.scala', [selector(False, '500')])
        variants, _ = self.scan()
        logical = 'src/main/scala/pkg/Example.scala'
        self.assertEqual(resolve_sources(self.graph, variants, '357')[logical].physical_path, base)
        self.assertEqual(resolve_sources(self.graph, variants, '420')[logical].physical_path, base)
        self.assertEqual(resolve_sources(self.graph, variants, '500')[logical].physical_path,
                         override)

    def test_ambiguous_variants_rejected(self):
        self.source('355', 'Example.scala', [selector(False, '356')])
        self.source('356', 'Example.scala', [selector(False, '356')])
        variants, _ = self.scan()
        with self.assertRaisesRegex(ValueError, 'Ambiguous shim sources for 356'):
            resolve_sources(self.graph, variants, '356')

    def test_nearer_override_wins_regardless_of_variant_order(self):
        self.source('355', 'Example.scala', [selector(True, '355')])
        self.source('356', 'Example.scala', [selector(True, '355')])
        override = self.source('357', 'Example.scala', [selector(False, '357')])
        variants, _ = self.scan()
        logical = 'src/main/scala/pkg/Example.scala'
        for ordered in (variants, list(reversed(variants))):
            self.assertEqual(resolve_sources(self.graph, ordered, '357')[logical].physical_path,
                             override)

    def test_shadow_keeps_exact_removals(self):
        exact = self.source('420', 'Example.scala', [selector(False, '420')])
        variants, _ = self.scan()
        self.assertEqual(validate_legacy_shadow(self.graph, variants), (exact,))
        candidates, _ = candidate_inherited_sources(self.graph, variants)
        self.assertEqual(resolve_sources(self.graph, candidates, '500'), {})

    def test_shadow_accepts_descendant_closed_membership(self):
        self.source('355', 'Example.scala', [selector(False, '355', '356', '357')])
        variants, _ = self.scan()
        self.assertEqual(validate_legacy_shadow(self.graph, variants), ())

    def test_versioned_package_path_stays_exact(self):
        source = self.source('357', 'spark357/Provider.scala', [selector(False, '357')])
        variants, _ = self.scan()
        self.assertEqual(validate_legacy_shadow(self.graph, variants), (source,))

    def test_shadow_requires_oracle_for_converted_sources(self):
        self.source('355', 'Example.scala', [selector(True, '355')])
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
            variants, _ = scan_module_sources(module, targets, EXTERNAL_EXACT_TARGETS)
            legacy_variants = []
            for variant in variants:
                selectors = []
                physical = Path(variant.physical_path).relative_to(ROOT).as_posix()
                expanded = []
                for selector in variant.selectors:
                    if selector.kind == 'descendants' and len(selector.targets) > 1:
                        expanded.extend(selector._replace(targets=(target,))
                                        for target in selector.targets)
                    else:
                        expanded.append(selector)
                for selector in expanded:
                    if selector.kind == 'minor':
                        # Preserve today's exact membership in the legacy oracle.
                        selectors.extend(SourceSelector('exact', (target,))
                                         for target in sorted(selector_membership(graph, selector)))
                    elif selector.kind == 'descendants' and selector.targets == ('330',):
                        # These sources have version-specific replacements after their last
                        # legacy exact selector.
                        if physical in LEGACY_330_331_ONLY:
                            legacy_targets = ('330', '331')
                        elif physical in LEGACY_330_THROUGH_333_ONLY:
                            legacy_targets = ('330', '331', '332', '333')
                        else:
                            legacy_targets = ('330', '331', '332', '333', '334')
                        selectors.extend(SourceSelector('exact', (target,))
                                         for target in legacy_targets)
                    elif selector.kind == 'descendants' and selector.targets == ('331',):
                        selectors.extend(SourceSelector('exact', (target,))
                                         for target in ('331', '332', '333', '334'))
                    elif selector.kind == 'descendants' and selector.targets == ('332',):
                        selectors.extend(SourceSelector('exact', (target,))
                                         for target in ('332', '333', '334'))
                    elif selector.kind == 'descendants' and selector.targets == ('350',):
                        selectors.extend(SourceSelector('exact', (target,))
                                         for target in ('350', '351'))
                    elif selector.kind == 'descendants' and selector.targets == ('358',):
                        selectors.extend(SourceSelector('exact', (target,))
                                         for target in ('358', '359'))
                    elif selector.kind == 'descendants' and selector.targets == ('400',):
                        selectors.extend(SourceSelector('exact', (target,))
                                         for target in ('400', '401'))
                    elif selector.kind == 'descendants' and selector.targets == ('402',):
                        selectors.extend(SourceSelector('exact', (target,))
                                         for target in ('402', '403'))
                    elif selector.kind == 'descendants' and selector.targets == ('403',):
                        selectors.extend(SourceSelector('exact', (target,))
                                         for target in ('403', '404'))
                    elif selector.kind == 'descendants' and selector.targets == ('411',):
                        selectors.extend(SourceSelector('exact', (target,))
                                         for target in ('411', '412'))
                    elif selector.kind == 'descendants' and selector.targets == ('412',):
                        selectors.extend(SourceSelector('exact', (target,))
                                         for target in ('412', '413'))
                    elif selector.kind == 'descendants' and selector.targets == ('420',):
                        selectors.extend(SourceSelector('exact', (target,))
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
            variants, _ = scan_module_sources(module, targets, EXTERNAL_EXACT_TARGETS)
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
        variants, _ = scan_module_sources(str(ROOT / 'sql-plugin'), targets,
                                          EXTERNAL_EXACT_TARGETS)
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
        self.assertTrue(any(row['selector_kind'] == 'descendants' for row in result['sources']))
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
