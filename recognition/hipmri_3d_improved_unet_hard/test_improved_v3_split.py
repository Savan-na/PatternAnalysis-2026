"""CPU synthetic boundaries; no claim of real NIfTI or CUDA equivalence."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import run_improved_v3_split as split
from .test_improved_v3 import pairs, records, boundary
from .test_improved_v2 import config as parent_config


class Samples(Dataset):
    def __len__(self):
        return 143

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        return torch.tensor([float(index)/143, float(self.epoch)/12]), index


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        split.v3.base.configure_reproducibility(3710, self.device)
        self.cfg = split.config_for(parent_config(), pairs())

    def objects(self):
        m = torch.nn.Linear(2, 1)
        o = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.00001,
                             foreach=False, fused=False)
        g = torch.Generator().manual_seed(3710)
        return m, o, g

    def test_two_epoch_split_exact_cpu_update_rng_and_shuffle_parity(self):
        """286 real small CPU AdamW updates, with a serialization boundary at 143."""
        m, o, g = self.objects()

        def epoch(m, o, g, e):
            ds = Samples(); ds.set_epoch(e)
            loader = DataLoader(ds, batch_size=1, shuffle=True, generator=g, num_workers=0)
            order = []
            for x, i in loader:
                order.append(int(i.item()))
                # Exercise global and loader streams, not just a fixed input.
                x = x + random.random() + float(np.random.random()) + torch.rand_like(x)
                o.zero_grad(set_to_none=True)
                m(x).square().mean().backward(); o.step()
            return order

        a = epoch(m,o,g,10)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'boundary.pt'
            split.v3.base.atomic_checkpoint(path,m,o,completed_epochs=11,total_steps=1573,
                config=self.cfg,generator=g,device=self.device,
                epoch_result=boundary(validation_pending=True))
            b = epoch(m,o,g,11)
            expected = copy.deepcopy(m.state_dict())
            expected_opt = copy.deepcopy(o.state_dict())
            expected_next = (random.random(), np.random.random(), torch.rand(3), torch.rand(3,generator=g))
            restored, opt, gen = self.objects()
            self.assertEqual(split.restore_boundary(path,11,restored,opt,self.cfg,gen,self.device),(11,1573))
            self.assertEqual(epoch(restored,opt,gen,11),b)
            for k,v in expected.items():
                self.assertTrue(torch.equal(v,restored.state_dict()[k]))
            for k,values in expected_opt['state'].items():
                for name,value in values.items():
                    self.assertTrue(torch.equal(value,opt.state_dict()['state'][k][name]))
            actual = (random.random(), np.random.random(), torch.rand(3), torch.rand(3,generator=gen))
            self.assertEqual(expected_next[:2],actual[:2])
            self.assertTrue(torch.equal(expected_next[2],actual[2]))
            self.assertTrue(torch.equal(expected_next[3],actual[3]))
        self.assertEqual(len(set(a)),143)

    def test_boundaries_reject_wrong_epoch_config_and_missing_state(self):
        m,o,g=self.objects()
        m(torch.ones(1,2)).sum().backward(); o.step()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'b.pt'
            split.v3.base.atomic_checkpoint(path,m,o,completed_epochs=12,total_steps=1716,
                config=self.cfg,generator=g,device=self.device,
                epoch_result=boundary(validation_pending=True))
            original=torch.load(path,weights_only=True)
            for mutate in (
                lambda p:p.update(completed_epochs=11),
                lambda p:p['config'].update(experiment_id='wrong'),
                lambda p:p['epoch_result'].update(validation_pending=False),
                lambda p:p.update(optimizer={'state':{}}),
                lambda p:p['rng'].pop('loader'),
            ):
                bad=copy.deepcopy(original); mutate(bad); torch.save(bad,path)
                with self.assertRaises(ValueError):
                    split.restore_boundary(path,12,m,o,self.cfg,g,self.device)

    def test_frozen_policy_and_source_hash_rejection(self):
        split.validate_config(self.cfg)
        bad=copy.deepcopy(self.cfg); bad['source_sha256']['run_improved_v3_split.py']='bad'
        bad['config_sha256']=split.v3.v2.config_hash(bad)
        with self.assertRaises(ValueError): split.validate_config(bad)
        self.assertEqual(self.cfg['protocol']['settings'],split.v3.settings())

    def test_train_one_no_validation_immutable_boundary_and_epoch(self):
        m,o,g=self.objects()
        ds=Samples(); loader=DataLoader(ds,batch_size=1)
        events=[]
        def fake(*args,**kw):
            for i in range(1,144): kw['progress'](dict(event='step',step=i))
            return boundary()
        with tempfile.TemporaryDirectory() as d, patch.object(split.v3.v2,'run_epoch',side_effect=fake), \
                patch.object(split.v3,'full_validation') as validation:
            path=Path(d)/'boundary.pt'
            split.train_one(m,o,loader,self.device,g,self.cfg,path,12,events.append)
            self.assertEqual(ds.epoch,11)
            self.assertEqual(events[0]['global_step'],1574)
            self.assertEqual(events[-2]['global_step'],1716)
            validation.assert_not_called()
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):
                split.train_one(m,o,loader,self.device,g,self.cfg,path,12,events.append)
            self.assertEqual(before,path.read_bytes())

    def test_journal_boundary_authentication_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'A').mkdir()
            path=split.checkpoint_path(root,'A'); path.write_bytes(b'synthetic')
            j=split.v3.Journal(root/'A'/'events_test.jsonl')
            j.emit(dict(event='setup',stage='A',config=self.cfg,starting_completed_epochs=10,starting_total_steps=1430))
            for i in range(1,144):
                j.emit(dict(event='step',global_step=1430+i,human_epoch=11,
                            loss=1.,weighted_ce=1.,ordinary_ce=1.,soft_dice_loss=1.))
            j.emit(dict(event='checkpoint',path=str(path),sha256=split.v3.sha256(path)))
            j.emit(dict(event='result',status='PASS',parent_unchanged=True,test_arrays_decoded=0)); j.close()
            split.audit_stage(root,'A',self.cfg)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError): split.audit_stage(root,'A',self.cfg)

    def test_original_loader_rejects_pending_e12_and_metrics_unchanged(self):
        summary=split.v3.enrich_summary(records())
        baseline={'epoch':10,'score':0.4}
        self.assertEqual(split.v3.selection(summary,baseline,12)[0]['epoch'],12)
        payload=dict(completed_epochs=12,epoch_result=boundary(validation_pending=True))
        with patch.object(torch,'load',return_value=payload):
            with self.assertRaises(ValueError):
                split.v3.resume_checkpoint('fake',None,None,config=self.cfg['protocol'],
                    generator=None,device=self.device)

    def test_complete_three_stage_receipt_recomputed_and_corruption_rejected(self):
        m,o,g=self.objects()
        m(torch.ones(1,2)).sum().backward(); o.step()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); inputs={}
            for stage,epoch in [('A',11),('B',12)]:
                (root/stage).mkdir()
                p=split.checkpoint_path(root,stage)
                split.v3.base.atomic_checkpoint(p,m,o,completed_epochs=epoch,total_steps=epoch*143,
                    config=self.cfg,generator=g,device=self.device,
                    epoch_result=boundary(validation_pending=True))
                log=root/stage/'events_test.jsonl'; j=split.v3.Journal(log)
                j.emit(dict(event='setup',stage=stage,config=self.cfg,input_journals=dict(inputs),
                    starting_completed_epochs=epoch-1,starting_total_steps=(epoch-1)*143))
                for i in range(1,144):
                    j.emit(dict(event='step',human_epoch=epoch,global_step=(epoch-1)*143+i,
                        loss=1.,weighted_ce=1.,ordinary_ce=1.,soft_dice_loss=1.))
                j.emit(dict(event='checkpoint',path=str(p),sha256=split.v3.sha256(p)))
                j.emit(dict(event='result',status='PASS',parent_unchanged=True,test_arrays_decoded=0)); j.close()
                inputs[stage]=split.v3.sha256(log)
            (root/'C').mkdir(); j=split.v3.Journal(root/'C'/'events_test.jsonl')
            j.emit(dict(event='setup',stage='C',config=self.cfg,input_journals=inputs,
                        starting_completed_epochs=12,starting_total_steps=1716))
            for r in records(): j.emit(dict(r,completed_epochs=12))
            summary=split.v3.enrich_summary(records())
            j.emit(dict(summary,event='validation_summary',completed_epochs=12))
            policy=split.v3.settings()
            best,decision=split.v3.selection(summary,dict(epoch=10,score=sum(
                policy['v2_patient_macro_dice'][c] for c in (2,4,5))/3),12)
            receipt=dict(best,decision=decision,checkpoint=str(split.checkpoint_path(root,'B')),
                checkpoint_sha256=split.v3.sha256(split.checkpoint_path(root,'B')),
                config_sha256=self.cfg['config_sha256'],validation_completed=True,test_evaluated=False)
            split.v3.atomic_json(root/'C'/'selection.json',receipt)
            j.emit(dict(receipt,event='selection'))
            j.emit(dict(event='result',status='PASS',parent_unchanged=True,test_arrays_decoded=0)); j.close()
            self.assertEqual(split.audit_complete(root,'C')['status'],'PASS')
            receipt['score']=0
            split.v3.atomic_json(root/'C'/'selection.json',receipt)
            with self.assertRaises(ValueError): split.audit_complete(root,'C')

    def test_cli_rejects_unbounded_or_test_requests(self):
        for args in (['--stage','D'], ['--stage','A','--test'], ['--stop-after-epoch','14']):
            with self.assertRaises(SystemExit): split.main(args)


if __name__ == '__main__':
    unittest.main()
