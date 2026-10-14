"""Independent checks for anisotropic common diagnostics and input contracts."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from nfgs.controlled import evaluate,load,match,cci_errors
from nfgs.diagnostics.common_fd2 import curl,divergence
from nfgs.paths import project_root


class ControlledEvaluationTests(unittest.TestCase):
    def bundle(self,root):
        x=np.linspace(-1,1,9);y=np.linspace(-2,2,7);z=np.linspace(0,3,5)
        zz,yy,xx=np.meshgrid(z,y,x,indexing='ij')
        field={'Bx':-yy,'By':xx,'Bz':1+xx*yy,'x':x,'y':y,'z':z,
               'field_unit':'benchmark','length_unit':'normalized'}
        np.savez(root/'field.npz',**field)
        boundary={k:field[k][0] for k in ('Bx','By','Bz')}
        boundary.update({k:field[k] for k in ('x','y','field_unit','length_unit')})
        np.savez(root/'boundary.npz',**boundary)
        return field

    def test_anisotropic_polynomial_curl_and_divergence(self):
        with tempfile.TemporaryDirectory() as d:
            b=self.bundle(Path(d));j=curl(b,(.25,4/6,.75))
            xx=np.broadcast_to(b['x'][None,None,:],b['Bx'].shape)
            yy=np.broadcast_to(b['y'][None,:,None],b['Bx'].shape)
            np.testing.assert_allclose(j['Jx'],xx,atol=2e-14)
            np.testing.assert_allclose(j['Jy'],-yy,atol=2e-14)
            np.testing.assert_allclose(j['Jz'],2,atol=2e-14)
            np.testing.assert_allclose(divergence(b,(.25,4/6,.75)),0,atol=2e-14)

    def test_identity_and_actual_observation_target(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.bundle(root)
            config=project_root()/'configs/controlled/evaluation_example.json'
            args=(root/'field.npz',root/'field.npz',root/'boundary.npz',config)
            result=evaluate(*args)['metrics']
            self.assertAlmostEqual(result['C_vec'],1,12)
            self.assertAlmostEqual(result['Eprime_n'],1,12)
            self.assertAlmostEqual(result['E_rel_Bgt'],0,12)
            obs=load(root/'boundary.npz',True);obs['Bz']+=.1*np.outer(np.sin(obs['y']),np.cos(obs['x']))
            np.savez(root/'boundary.npz',**obs)
            noisy=evaluate(*args)['metrics']
            self.assertGreater(noisy['E_rel_Bgt_L2'],0)
            self.assertAlmostEqual(noisy['clean_reference_CCI']['E_rel_Bgt_L2'],0,12)

    def test_coordinate_unit_and_zero_imprint_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.bundle(root);a=load(root/'field.npz');b=load(root/'field.npz')
            b['x']+=1
            with self.assertRaisesRegex(ValueError,'Coordinate mismatch'):match(a,b)
            b=load(root/'field.npz');b['field_unit']='T'
            with self.assertRaisesRegex(ValueError,'Unit mismatch'):match(a,b)
            result=cci_errors(np.zeros((3,7,9)),np.ones((3,7,9)),(slice(None),)*3,[1e-8,.01,.05])
            self.assertIsNone(result['E_rel_Bgt_L2']);self.assertIsNone(result['E_rel_Bgt'])


if __name__=='__main__':unittest.main()
