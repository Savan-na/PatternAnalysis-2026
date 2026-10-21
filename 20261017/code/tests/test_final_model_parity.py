"""Require independent final components to preserve frozen model semantics."""
import ast
import importlib.util
from pathlib import Path
import unittest
import torch
from hipmri_rewrite.models.baseline_3d import UNet3D
from hipmri_rewrite.improved_v3 import ImprovedUNet3DV3

MODULE=Path(__file__).resolve().parents[3]/'recognition/HipMRI_UNet_s4906923/modules.py'
spec=importlib.util.spec_from_file_location('final_components',MODULE)
final=importlib.util.module_from_spec(spec)
spec.loader.exec_module(final)
PAIRS=[(UNet3D,final.StandardUNet3D,1412550),(ImprovedUNet3DV3,final.ImprovedUNet3DV3,2366946)]


class FinalParityTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_standalone_imports_only_torch_and_future(self):
        for node in ast.walk(ast.parse(MODULE.read_text())):
            if isinstance(node,ast.Import):
                self.assertTrue(all(n.name=='torch' for n in node.names))
            if isinstance(node,ast.ImportFrom):
                self.assertIn(node.module,('__future__','torch','torch.nn'))
                self.assertEqual(node.level,0)

    def test_default_initialization_parameter_keys_values_and_rng(self):
        for old,new,count in PAIRS:
            torch.manual_seed(371017)
            reference=old()
            rng=torch.get_rng_state()
            torch.manual_seed(371017)
            candidate=new()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(sum(p.numel() for p in candidate.parameters()),count)
            self.assertEqual(list(reference.state_dict()),list(candidate.state_dict()))
            for name,value in reference.state_dict().items():
                self.assertTrue(torch.equal(value,candidate.state_dict()[name]),name)

    def test_strict_state_loading_and_eval_logits(self):
        x=torch.randn(2,1,16,16,16)
        for old,new,_ in PAIRS:
            reference=old().eval()
            candidate=new().eval()
            candidate.load_state_dict(reference.state_dict(),strict=True)
            with torch.no_grad():
                self.assertTrue(torch.equal(reference(x),candidate(x)))
        combined,heads=candidate(x,return_components=True)
        self.assertTrue(torch.equal(combined,heads[0]+heads[1]+heads[2]))
        self.assertEqual(sum(isinstance(m,torch.nn.GroupNorm) for m in candidate.modules()),14)
        self.assertFalse(any(isinstance(m,torch.nn.InstanceNorm3d) for m in candidate.modules()))

    def test_training_dropout_gradients_and_adam_update_match(self):
        x=torch.randn(2,1,16,16,16)
        y=torch.randint(0,6,(2,16,16,16))
        for old,new,_ in PAIRS:
            reference=old().train()
            candidate=new().train()
            candidate.load_state_dict(reference.state_dict(),strict=True)
            rng=torch.get_rng_state()
            a=torch.optim.Adam(reference.parameters(),lr=.001)
            b=torch.optim.Adam(candidate.parameters(),lr=.001)
            torch.nn.functional.cross_entropy(reference(x),y).backward()
            a.step()
            torch.set_rng_state(rng)
            torch.nn.functional.cross_entropy(candidate(x),y).backward()
            b.step()
            for name,value in reference.state_dict().items():
                self.assertTrue(torch.equal(value,candidate.state_dict()[name]),name)


if __name__=='__main__':unittest.main()
