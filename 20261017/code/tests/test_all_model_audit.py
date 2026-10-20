"""Check geometry and metric distance definitions without reading dataset files."""
import unittest
from types import SimpleNamespace
import numpy as np
from tools.audit_all_models import geometry, diagnostics


class GeneralizationAuditTests(unittest.TestCase):
    def test_six_connectivity_gaps_and_physical_volume(self):
        mask=np.zeros((5,5,5),dtype=bool)
        mask[1,1,1]=True;mask[1,1,3]=True
        result=geometry(mask,(2.,3.,4.))
        self.assertEqual(result['components_6'],2)
        self.assertEqual(result['internal_empty_z_slices'],1)
        self.assertEqual(result['largest_component_fraction'],.5)
        self.assertEqual(result['volume_mm3'],48.)

    def test_distance_20mm_not_20voxels_and_empty_surface_status(self):
        truth=np.zeros((20,6,6),dtype=np.uint8);truth[10:12,0:2,0:2]=1
        prediction=truth.copy();prediction[0,0,0]=1;prediction[1,0,0]=1
        volume=SimpleNamespace(mask=truth,spacing_mm=(2.,1.,1.),affine=np.diag([2.,1.,1.,1.]))
        result=diagnostics(volume,prediction)
        self.assertEqual(result['far_background_to_body_voxels'],1)
        self.assertEqual(result['surfaces']['Prostate']['status'],'both_empty')
        self.assertIsNone(result['surfaces']['Prostate']['hd95_mm'])
        self.assertIsNone(result['prostate_regions'])


if __name__=='__main__':unittest.main()
