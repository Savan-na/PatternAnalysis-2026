import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from hipmri_rewrite.data import Volume
from hipmri_rewrite.splits import Case
from hipmri_rewrite.sampling import crop_for_3d_model
from hipmri_rewrite.augmentation_v4 import paired_patch, learning_rate, make_optimizer


class V4Tests(unittest.TestCase):
    def volume(self):
        grid = np.indices((90, 90, 60))
        mask = (((grid[0]-45)**2 + (grid[1]-45)**2 + (grid[2]-30)**2) < 100).astype(np.uint8)
        return Volume(Case('B006', 0, 'train', Path('a'), Path('b')), mask.astype(np.float32),
                      mask, np.diag([1., 2., 3., 1.]), (1., 2., 3.))

    def test_identity_and_physical_pair_and_stateless_replay(self):
        v = self.volume(); origin = (10, 10, 10)
        kw = dict(seed=123, epoch=7, draw=2, enabled=True)
        im, ma, _ = paired_patch(v, origin, **kw, force=([0,0,0], 1.))
        a, b = crop_for_3d_model(v, origin)
        np.testing.assert_allclose(im, a, atol=1e-6); np.testing.assert_array_equal(ma, b)
        im, ma, record = paired_patch(v, origin, **kw, force=([7,-8,9], 1.05))
        # Same synthetic anatomical object remains aligned under unequal spacing.
        dice = 2 * ((im[0] > .5) & (ma == 1)).sum() / ((im[0] > .5).sum() + (ma == 1).sum())
        self.assertGreater(dice, .95)
        first = paired_patch(v, origin, **kw)
        np.random.seed(999); np.random.random(20)
        second = paired_patch(v, origin, **kw)
        np.testing.assert_array_equal(first[0], second[0]); self.assertEqual(first[2], second[2])
        self.assertTrue(set(np.unique(ma)) <= {0,1})
        invalid = Volume(Case('B006',0,'validation',Path('a'),Path('b')),v.image,v.mask,v.affine,v.spacing_mm)
        with self.assertRaises(ValueError): paired_patch(invalid, origin, **kw)

    def test_adamw_cosine_checkpoint_resume_matches_uninterrupted(self):
        torch.set_num_threads(1); torch.manual_seed(7)
        model = torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.LayerNorm(4),torch.nn.Linear(4,1))
        other = torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.LayerNorm(4),torch.nn.Linear(4,1))
        other.load_state_dict(model.state_dict())
        opt = make_optimizer(model); opt2 = make_optimizer(other)
        x = torch.arange(12, dtype=torch.float32).reshape(4,3)/10
        def step(m,o,t):
            self.assertAlmostEqual(o.param_groups[0]['lr'], learning_rate(t), places=15)
            o.zero_grad(); m(x).square().mean().backward(); o.step()
            for g in o.param_groups: g['lr'] = learning_rate(t+1)
        for t in range(4): step(model,opt,t)
        for t in range(2): step(other,opt2,t)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'state.pt'; torch.save({'model':other.state_dict(),'opt':opt2.state_dict(),'updates':2},p)
            d=torch.load(p,weights_only=False); other.load_state_dict(d['model']); opt2.load_state_dict(d['opt'])
            for t in range(d['updates'],4): step(other,opt2,t)
        for a,b in zip(model.parameters(),other.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertAlmostEqual(learning_rate(6500),1e-5)
        self.assertEqual([g['weight_decay'] for g in opt.param_groups],[1e-5,0.])


if __name__ == '__main__': unittest.main()
