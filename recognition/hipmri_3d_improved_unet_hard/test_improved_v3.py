"""CPU synthetic tests: continuation, selection, evidence logging and isolation."""
import copy
import json
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import train as base
from . import train_improved_v2 as v2
from . import train_improved_v3 as v3
from . import validate as evaluation
from .dataset import APPROVED_PATIENTS, VolumePair
from .improved_modules import ImprovedUNet3D
from .test_improved_v2 import config as parent_config


def pairs():
    names = sorted(APPROVED_PATIENTS['validation'])
    return tuple(VolumePair(names[i % 6], i // 6, Path(f'/fake/{i}.nii.gz'),
                            Path(f'/fake/{i}_mask.nii.gz')) for i in range(38))


def records():
    result = []
    for p in pairs():
        counts = np.eye(6, dtype=np.int64) * 100
        result.append(dict(event='volume', case=f'{p.patient_id}_Week{p.week}',
                           patient_id=p.patient_id, week=p.week, split='validation', status='success',
                           confusion_counts=counts.tolist(), metrics=evaluation.metrics_from_confusion(counts),
                           peak_gpu_allocated_bytes=0, peak_gpu_reserved_bytes=0))
    return result


def boundary(**extra):
    # Synthetic boundary metadata, never claimed to be 143 real updates.
    return dict(epoch_completed=True, processed_steps=143, expected_steps=143, **extra)


def initial_selection():
    return {'epoch': 10, 'score': sum(v3.settings()['v2_patient_macro_dice'][c] for c in (2, 4, 5)) / 3}


class V3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        base.configure_reproducibility(3710, torch.device('cpu'))

    def model_opt(self):
        m = ImprovedUNet3D(base_channels=1)
        return m, torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.00001, foreach=False, fused=False)

    def batch(self):
        return {'mri': torch.randn(1, 1, 16, 16, 16),
                'mask': torch.arange(4096).reshape(1, 16, 16, 16).long() % 6}

    def test_shared_loss_and_unchanged_epoch_implementation(self):
        self.assertIs(v3.v2.run_epoch, v2.run_epoch)
        self.assertIs(v3.v2.combined_loss, v2.combined_loss)
        self.assertIs(v3.evaluation._process_pair, evaluation._process_pair)
        self.assertIs(v3.evaluation.summarize, evaluation.summarize)

    def test_real_v2_to_v3_optimizer_rng_and_next_update_parity(self):
        model, opt = self.model_opt()
        batch = self.batch()
        v2.run_epoch(model, [batch], opt, torch.device('cpu'))
        gen = torch.Generator().manual_seed(3710)
        cfg = v3.make_config(parent_config(), pairs())
        with tempfile.TemporaryDirectory() as d:
            old, new = Path(d)/'v2.pt', Path(d)/'v3.pt'
            base.atomic_checkpoint(old, model, opt, completed_epochs=10, total_steps=1430,
                config=cfg['parent_config'], generator=gen, device=torch.device('cpu'), epoch_result=boundary())
            base.atomic_checkpoint(new, model, opt, completed_epochs=11, total_steps=1573,
                config=cfg, generator=gen, device=torch.device('cpu'), epoch_result=boundary(selection=initial_selection()))
            v2.resume_checkpoint(old, model, opt, config=cfg['parent_config'], generator=gen, device=torch.device('cpu'))
            expected_rng = (random.random(), np.random.random(), torch.rand(5), torch.randperm(143, generator=gen))
            reference = v2.run_epoch(model, [batch], opt, torch.device('cpu'))
            restored, optimizer = self.model_opt()
            self.assertEqual(v3.resume_checkpoint(new, restored, optimizer, config=cfg, generator=gen,
                                                 device=torch.device('cpu'))[:2], (11, 1573))
            actual_rng = (random.random(), np.random.random(), torch.rand(5), torch.randperm(143, generator=gen))
            self.assertEqual(expected_rng[:2], actual_rng[:2])
            for a, b in zip(expected_rng[2:], actual_rng[2:]):
                self.assertTrue(torch.equal(a, b))
            actual = v2.run_epoch(restored, [batch], optimizer, torch.device('cpu'))
            self.assertEqual(actual['mean_loss'], reference['mean_loss'])
            for a, b in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(optimizer.param_groups[0]['lr'], .001)
            self.assertTrue(optimizer.state)

    def test_validation_restores_all_rng_and_model_mode_even_on_error(self):
        m, _ = self.model_opt()
        g = torch.Generator().manual_seed(9)
        state = base._rng_state(g, torch.device('cpu'))
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            with v3.preserve_rng(g, torch.device('cpu'), m):
                random.random(); np.random.random(); torch.rand(4); torch.rand(4, generator=g)
                m.eval()
                raise RuntimeError('synthetic')
        now = base._rng_state(g, torch.device('cpu'))
        self.assertEqual(state['python'], now['python'])
        self.assertEqual(state['numpy'], now['numpy'])
        for k in ('torch_cpu', 'loader'):
            self.assertTrue(torch.equal(state[k], now[k]))
        self.assertTrue(m.training)

    def test_full_validation_reuses_production_and_restores_rng(self):
        m, _ = self.model_opt()
        g = torch.Generator().manual_seed(9)
        before = torch.get_rng_state().clone()
        generated, out = iter(records()), []
        def process(*args):
            self.assertFalse(m.training)
            self.assertTrue(torch.is_inference_mode_enabled())
            torch.rand(4)
            return next(generated)
        with patch.object(evaluation, '_process_pair', side_effect=process) as p:
            result = v3.full_validation(m, pairs(), torch.device('cpu'), g, out.append, 12)
        self.assertEqual(p.call_count, 38)
        self.assertEqual(len(out), 39)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(m.training)
        self.assertEqual(result['foreground_mean_pooled_dice'], 1.)

    def test_confusion_derived_precision_recall_and_macro_parity(self):
        rows = records()
        matrix = np.array(rows[0]['confusion_counts'])
        matrix[2, 1] = 50
        matrix[1, 2] = 20
        rows[0]['confusion_counts'] = matrix.tolist()
        rows[0]['metrics'] = evaluation.metrics_from_confusion(matrix)
        v3.validate_records(rows, pairs())
        summary = v3.enrich_summary(rows)
        standard = evaluation.summarize(rows)
        self.assertEqual(summary['overall_patient_macro'], standard['overall_patient_macro'])
        self.assertEqual(summary['overall_volume_macro'], standard['overall_volume_macro'])
        bones = summary['overall_pooled'][2]
        self.assertEqual(bones['false_positives'], 20)
        self.assertEqual(bones['false_negatives'], 50)
        self.assertAlmostEqual(bones['precision'], 3800/3820)
        self.assertAlmostEqual(bones['recall'], 3800/3850)

    def test_selection_minority_score_guards_and_earliest_tie(self):
        good = v3.enrich_summary(records())
        best, decision = v3.selection(good, initial_selection(), 12)
        self.assertEqual(best['epoch'], 12)
        self.assertTrue(decision['meaningful_improvement'])
        self.assertTrue(decision['all_class_pooled_dice_at_least_070'])
        self.assertEqual(v3.selection(good, best, 14)[0], best)
        for c in range(6):
            bad = copy.deepcopy(good)
            bad['overall_pooled'][c]['dice'] = 0
            retained, decision = v3.selection(bad, initial_selection(), 12)
            self.assertEqual(retained, initial_selection())
            self.assertIn(f'class_{c}_regression', decision['rejection_reasons'])
        bad = copy.deepcopy(good)
        bad['overall_pooled'][5]['tp_positive_volumes'] = 37
        self.assertFalse(v3.selection(bad, initial_selection(), 12)[1]['selected'])

    def test_selection_rejects_incomplete_nan_or_absent_metrics(self):
        for bad in (None, float('nan')):
            s = v3.enrich_summary(records())
            s['overall_pooled'][2]['dice'] = bad
            with self.assertRaises(ValueError):
                v3.selection(s, initial_selection(), 12)
        s = v3.enrich_summary(records())
        s['successful'] = 37
        with self.assertRaises(ValueError):
            v3.selection(s, initial_selection(), 12)

    def test_duplicate_failed_train_and_bad_confusion_rejected(self):
        for change in (lambda r: r.__setitem__(-1, r[0]), lambda r: r[0].update(split='train'),
                       lambda r: r[0].update(status='failed'), lambda r: r[0]['confusion_counts'][0].__setitem__(0, -1)):
            r = records(); change(r)
            with self.assertRaises(ValueError):
                v3.validate_records(r, pairs())

    def test_authenticated_baseline_log_and_hash_rejection(self):
        cfg = parent_config()
        rows = records()
        summary = evaluation.summarize(rows)
        setup = dict(event='setup', experiment_id=v2.EXPERIMENT_ID, stride=32, patch_shape_dhw=[64]*3,
                     checkpoint_metadata=dict(config_sha256=cfg['config_sha256'], completed_epochs=10, total_steps=1430))
        events = [setup, *rows, dict(summary, event='summary', successful=38, failed=0,
                                   full_38_volume_validation_completed=True), dict(event='result', status='PASS')]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'baseline.jsonl'
            p.write_text(''.join(json.dumps(r)+'\n' for r in events))
            policy = v3.settings()
            policy.update(v2_pooled_dice=[1.]*6, v2_patient_macro_dice=[1.]*6)
            with patch.object(v3, 'VALIDATION_SHA256', v3.sha256(p)), patch.object(v3, 'settings', return_value=policy):
                self.assertEqual(v3.authenticate_validation(p, pairs(), cfg)['successful'], 38)
                p.write_bytes(p.read_bytes()+b'\n')
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    v3.authenticate_validation(p, pairs(), cfg)

    def test_parent_byte_authentication_and_protected_loader(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'parent.pt'
            p.write_bytes(b'not a checkpoint')
            with self.assertRaisesRegex(ValueError, 'Parent bytes'):
                v3.read_parent(p)
            with patch.object(v3, 'PARENT_SHA256', v3.sha256(p)), \
                    patch.object(v3.v2_evaluation, 'load_checkpoint_model', side_effect=ValueError('protected rejection')) as loader:
                with self.assertRaisesRegex(ValueError, 'protected rejection'):
                    v3.read_parent(p)
                loader.assert_called_once()

    def test_v3_config_and_source_mismatch_rejected(self):
        cfg = v3.make_config(parent_config(), pairs())
        v3.validate_config(cfg)
        for change in (lambda c: c.update(experiment_id='v2'), lambda c: c.update(parent_sha256='bad'),
                       lambda c: c['source_sha256'].update({'train.py': 'bad'}),
                       lambda c: c['parent_config']['policies']['patch_dataset'].update(seed=9)):
            bad = copy.deepcopy(cfg); change(bad); bad['config_sha256'] = v2.config_hash(bad)
            with self.assertRaises(ValueError):
                v3.validate_config(bad)

    def test_incompatible_resume_and_missing_validation_rejected(self):
        cfg = v3.make_config(parent_config(), pairs())
        for epoch, result in ((10, boundary(selection=initial_selection())),
                              (12, boundary(selection=initial_selection())),
                              (11, boundary(selection={'epoch': 12, 'score': .9}))):
            payload = dict(completed_epochs=epoch, epoch_result=result)
            with patch.object(torch, 'load', return_value=payload):
                with self.assertRaises(ValueError):
                    v3.resume_checkpoint('fake', None, None, config=cfg, generator=None, device=torch.device('cpu'))

    def test_journal_exclusive_no_overwrite_and_hash_chain(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'events.jsonl'
            j = v3.Journal(p)
            j.emit({'event':'setup'}); j.emit({'event':'result', 'status':'PASS'}); j.close()
            self.assertEqual(len(v3.audit_journal(p)), 2)
            before = p.read_bytes()
            with self.assertRaises(FileExistsError):
                v3.Journal(p)
            self.assertEqual(p.read_bytes(), before)
            p.write_bytes(before.replace(b'PASS', b'FAIL'))
            with self.assertRaises(ValueError):
                v3.audit_journal(p)

    def test_journal_detects_truncate_nul_and_partial_record(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'events.jsonl'; j = v3.Journal(p)
            try:
                j.emit({'event':'setup'}); p.write_bytes(b'')
                with self.assertRaises(RuntimeError):
                    j.emit({'event':'next'})
            finally:
                j.close()
            for bad in (b'\0\n', b'{"event":'):
                p.write_bytes(bad)
                with self.assertRaises(ValueError):
                    v3.audit_journal(p)

    def orchestration(self, directory, diagnostic=False, fail_validation=False):
        model, opt = self.model_opt()
        dataset = types.SimpleNamespace(epochs=[], set_epoch=lambda e: dataset.epochs.append(e))
        loader = types.SimpleNamespace(dataset=dataset)
        cfg = v3.make_config(parent_config(), pairs())
        parent = Path(directory)/'parent.pt'; parent.write_bytes(b'protected')
        events=[]
        def fake_epoch(*a, **kw):
            kw['progress']({'event':'step', 'step':1})
            return dict(boundary(), **{}) if not diagnostic else dict(epoch_completed=False, processed_steps=1, expected_steps=143)
        def validation(*a):
            if fail_validation:
                raise RuntimeError('synthetic validation failure')
            return v3.enrich_summary(records())
        with patch.object(v2, 'run_epoch', side_effect=fake_epoch), patch.object(v3, 'full_validation', side_effect=validation):
            result=v3.execute(model,opt,loader,torch.device('cpu'),torch.Generator(),cfg,Path(directory),parent,
                              10,1430,initial_selection(),pairs(),12,diagnostic,events.append)
        return result,dataset.epochs,events

    def test_bounded_epochs_selection_checkpoint_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            result, epochs, events=self.orchestration(d)
            self.assertEqual(result[:2], (12,1716))
            self.assertEqual(epochs, [10,11])
            self.assertEqual(result[2]['epoch'],12)
            p=Path(d)/'improved_unet_v3_epoch12.pt'
            payload=torch.load(p,weights_only=True)
            self.assertEqual(payload['epoch_result']['selection']['epoch'],12)
            self.assertIn('validation',payload['epoch_result'])
            self.assertEqual(json.loads((Path(d)/'selection.json').read_text())['checkpoint_sha256'],v3.sha256(p))
            before=v3.sha256(p)
            with self.assertRaises(FileExistsError):
                self.orchestration(d)
            self.assertEqual(v3.sha256(p),before)

    def test_diagnostic_writes_no_checkpoint_or_selection(self):
        with tempfile.TemporaryDirectory() as d:
            result,epochs,events=self.orchestration(d,diagnostic=True)
            self.assertEqual(result[:2],(10,1430))
            self.assertEqual(epochs,[10])
            self.assertEqual([p.name for p in Path(d).iterdir()],['parent.pt'])
            self.assertFalse(events[-1]['checkpoint_saved'])

    def test_failed_validation_retains_previous_epoch_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'validation failure'):
                self.orchestration(d,fail_validation=True)
            self.assertTrue((Path(d)/'improved_unet_v3_epoch11.pt').is_file())
            self.assertFalse((Path(d)/'improved_unet_v3_epoch12.pt').exists())

    def test_cli_rejects_test_and_unbounded_requests(self):
        required=['--parent-checkpoint','parent','--v2-validation-log','log']
        for extra in (['--all-test'], ['--stop-after-epoch','100'], [], ['--preflight-only','--resume','old']):
            with self.assertRaises(SystemExit):
                v3.main(required+extra)

    def test_actual_main_restores_parent_state_before_diagnostic(self):
        """CPU orchestration with mocked remote data/CUDA; real save and resume."""
        model, optimizer = self.model_opt()
        v2.run_epoch(model, [self.batch()], optimizer, torch.device('cpu'))
        cfg = parent_config()
        generator = torch.Generator().manual_seed(44)
        torch.randperm(143, generator=generator)
        expected_rng = torch.get_rng_state().clone()
        expected_loader = generator.get_state().clone()
        expected_model = copy.deepcopy(model.state_dict())
        cpu = torch.device('cpu')
        fake_dataset = types.SimpleNamespace()
        captured = {}
        def execute(m, opt, loader, device, gen, config, run_dir, parent, completed, steps,
                    selected, pairs, stop_epoch, diagnostic, emit):
            self.assertEqual((completed, steps, stop_epoch, diagnostic), (10, 1430, 12, True))
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            self.assertTrue(torch.equal(gen.get_state(), expected_loader))
            for name, value in m.state_dict().items():
                self.assertTrue(torch.equal(value, expected_model[name]))
            self.assertTrue(opt.state)
            captured['restored'] = True
            return completed, steps, selected
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); old=root/'v2'; run=root/'v3'; old.mkdir(); run.mkdir()
            parent=old/'improved_unet_v2_last.pt'
            base.atomic_checkpoint(parent, model, optimizer, completed_epochs=10, total_steps=1430,
                config=cfg, generator=generator, device=cpu, epoch_result=boundary())
            stub=types.SimpleNamespace(HipMRIPatchDataset=lambda *a:fake_dataset)
            with patch.dict('sys.modules', {__package__+'.patch_dataset':stub}), \
                    patch.dict('os.environ', {'SLURM_JOB_ID':'synthetic'}), \
                    patch.object(v3, 'PARENT_SHA256', v3.sha256(parent)), \
                    patch.object(v3, 'read_parent', return_value=(model,cfg)), \
                    patch.object(v3, 'runtime_contract'), \
                    patch.object(v3, 'discover_dataset', return_value={'validation':pairs()}), \
                    patch.object(v3, 'authenticate_validation', return_value=v3.enrich_summary(records())), \
                    patch.object(v3, 'DataLoader', return_value=types.SimpleNamespace(dataset=fake_dataset)), \
                    patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(torch.cuda, 'get_device_name', return_value='synthetic'), \
                    patch.object(torch, 'device', return_value=cpu), \
                    patch.object(v3, 'execute', side_effect=execute):
                v3.main(['--parent-checkpoint',str(parent),'--v2-validation-log','synthetic',
                         '--run-dir',str(run),'--diagnostic-one-step'])
            self.assertTrue(captured['restored'])
            self.assertFalse(list(run.glob('*.pt')))
            self.assertFalse((run/'active.lock').exists())
            rows=v3.audit_journal(next(run.glob('events_*.jsonl')))
            self.assertEqual(rows[-1]['status'],'PASS')


if __name__=='__main__':
    unittest.main()
