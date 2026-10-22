"""Offline contract tests using synthetic files; no policy, renderer or simulator."""
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import instruction_control_evaluate as metrics
import instruction_control_report as report
import instruction_disk_guard as disk
import instruction_storage_contract as storage


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


class InstructionControlTests(unittest.TestCase):
    def test_any_disk_api_at_floor_stops(self):
        # Each native/shutil reading independently vetoes an otherwise safe sample.
        high = disk.FLOOR + 1024
        for low in [None, 'available', 'free', 'clusters', 'shutil']:
            with self.subTest(low=low):
                values = {k: disk.FLOOR if k == low else high
                          for k in ['available', 'free', 'clusters', 'shutil']}

                def extended(path, available, total, free):
                    available._obj.value = values['available']
                    total._obj.value = high * 2
                    free._obj.value = values['free']
                    return True

                def legacy(path, sectors, size, free, total):
                    sectors._obj.value = size._obj.value = 1
                    free._obj.value = values['clusters']
                    total._obj.value = high
                    return True

                kernel = SimpleNamespace(GetDiskFreeSpaceExW=MagicMock(side_effect=extended),
                                         GetDiskFreeSpaceW=MagicMock(side_effect=legacy))
                with patch.object(disk.ctypes, 'WinDLL', return_value=kernel, create=True), \
                     patch.object(disk.shutil, 'disk_usage', return_value=SimpleNamespace(free=values['shutil'])):
                    sample = disk.sample(Path.cwd())
                self.assertEqual(sample['above_floor'], low is None)
                record = {'paths': {'C': sample, 'D': {'above_floor': True}}}
                if low is None:
                    disk.require_safe(record)
                else:
                    with self.assertRaises(AssertionError):
                        disk.require_safe(record)

    def test_episode_path_rejects_escape_and_nested_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).absolute() / 'evaluation'
            with patch.object(storage, 'evaluation_root', return_value=root):
                self.assertEqual(storage.episode_path(root / 'A-ID-42'), root / 'A-ID-42')
                for path in [root / '..' / 'escape', root / 'nested' / 'cell', root.parent / 'other']:
                    with self.subTest(path=path), self.assertRaises(AssertionError):
                        storage.episode_path(path)

    def test_contact_categories_and_reopening_are_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            def state(left, right):
                return {'fridge': {'qpos': {'left': math.radians(left), 'right': math.radians(right)}},
                        'action': [0] * 12}
            write_json(folder / 'reset.json', state(90, 60))
            (folder / 'steps.jsonl').write_text('\n'.join(json.dumps(s) for s in
                                                         [state(30, 20), state(45, 10)]))
            write_json(folder / 'geometry-catalog.json', [
                {'id': 0, 'name': 'robot', 'robot': True},
                {'id': 1, 'name': 'fridge', 'robot': False},
                {'id': 2, 'name': 'other_robot', 'robot': True},
                {'id': 3, 'name': 'table', 'robot': False}])
            write_json(folder / 'study-fixture.json', {'fridge_contact_geoms': ['fridge']})
            write_json(folder / 'summary.json', {'inference_seconds': [.2, .3]})
            def contact(ids, force, address=0):
                return {'geom_ids': ids, 'force_N_contact_frame': [force, 0, 0],
                        'constraint_address': address}
            contacts = [
                {'step': 1, 'samples': [{'robot_contacts': [
                    contact([0, 1], 2), contact([0, 1], 5), contact([0, 2], 3),
                    contact([0, 3], .3), contact([0, 1], 9999, -1), contact([0, 1], -9)]}]},
                {'step': 2, 'samples': [{'robot_contacts': [contact([0, 1], 10)]}]}]
            (folder / 'contacts.jsonl').write_text('\n'.join(map(json.dumps, contacts)))
            with patch.object(metrics, 'episode_metrics', return_value={'success': False}):
                row = metrics.detailed_metrics(folder)
            self.assertFalse(row['success'])
            for actual, expected in zip(row['per_door_progress_deg'], [45, 50]):
                self.assertAlmostEqual(actual, expected)
            for actual, expected in zip(row['per_door_best_progress_deg'], [60, 50]):
                self.assertAlmostEqual(actual, expected)
            self.assertEqual(row['contacts']['fridge'],
                             {'peak_N': 10, 'loaded_actions': 2, 'loaded_action_indices': [1, 2]})
            self.assertEqual(row['contacts']['self']['loaded_actions'], 1)
            self.assertEqual(row['contacts']['other']['loaded_actions'], 0)
            self.assertAlmostEqual(row['contacts']['other']['peak_N'], .3)

    def test_partial_report_keeps_invalid_unrun_and_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); evaluation = root / 'evaluation'; evaluation.mkdir()
            run = evaluation / 'report'; run.mkdir()
            row = dict(seed=42, model='A', condition='ID', path='evaluation/A-ID-42',
                       success=True, valid=True, termination='success', actions=2,
                       door_count=2, worker_seconds=1., rollout_seconds=.5,
                       total_inference_seconds=.1, contacts={
                           c: {'peak_N': 0, 'loaded_actions': 0} for c in ['fridge', 'self', 'other']})
            write_json(run / 'results.json', [row])
            ledger = [dict(seed=s, model=m, condition=c, status='unrun')
                      for s in range(42, 52) for m in 'ABC' for c in ['ID', 'NL']]
            for cell in ledger:
                if (cell['seed'], cell['model'], cell['condition']) == (42, 'A', 'ID'):
                    cell['status'] = 'completed'
                if (cell['seed'], cell['model'], cell['condition']) == (50, 'C', 'NL'):
                    cell['status'] = 'invalid'
            write_json(run / 'ledger.json', ledger)
            write_json(run / 'stopped.json', {'reason': 'timeout'})
            invalid = evaluation / 'C-NL-50'; invalid.mkdir()
            write_json(invalid / 'run-started.json', {})
            (invalid / 'steps.jsonl').write_text('{}\n{}\n')
            with patch.object(report, 'ROOT', root), \
                 patch.object(report, 'evaluation_root', return_value=evaluation), \
                 patch.object(report, 'physical_phases', return_value=[]), \
                 patch.object(sys, 'argv', ['report', '--results-directory', str(run)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                report.main()
            summary = json.loads((run / 'comparison-summary.json').read_text())
            self.assertFalse(summary['complete'])
            a = next(x for x in summary['aggregates'] if x['model'] == 'A' and x['condition'] == 'ID')
            self.assertEqual((a['successes'], a['valid']), (1, 1))
            self.assertEqual(a['double'], {'successes': 1, 'count': 1})
            with (run / 'all-60-cells.csv').open(encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 60)
            cells = {(r['seed'], r['model'], r['condition']): r for r in rows}
            self.assertEqual(cells['50', 'C', 'NL']['status'], 'invalid')
            self.assertEqual(cells['50', 'C', 'NL']['actions'], '2')
            self.assertEqual(cells['50', 'C', 'ID']['status'], 'unrun')
            self.assertFalse((run / 'complete.json').exists())


if __name__ == '__main__':
    unittest.main()
