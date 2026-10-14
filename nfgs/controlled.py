"""Scalar diagnostics for future controlled experiments; no solver-specific J input.

NPZ contract: Bx/By/Bz in z,y,x order (y,x for the observed boundary),
1-D x/y/z coordinates, and scalar field_unit/length_unit strings. Cropping
always follows full-field differentiation and full-footprint GS separation.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from nfgs.diagnostics.common_fd2 import curl, divergence
from nfgs._backend.scripts.recompute_td_referee_diagnostics import compute_metrics
from nfgs._backend.scripts.plot_nfgs_td_robin_metrics_vs_er import (
    make_bgt_helper_solver, bottom_plane_bgt_nfgs,
)

KEYS = ('Bx','By','Bz')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, boundary=False):
    with np.load(path, allow_pickle=False) as data:
        axes=('x','y') if boundary else ('x','y','z')
        required=set(KEYS+axes+('field_unit','length_unit'))
        if required-set(data.files):
            raise ValueError(f'{path}: missing coordinate/unit contract {required-set(data.files)}')
        result={k:np.asarray(data[k],dtype=np.float64) for k in KEYS+axes}
        result.update({k:str(data[k].item()) for k in ('field_unit','length_unit')})
    expected=tuple(len(result[k]) for k in axes[::-1])
    for k in KEYS:
        if result[k].shape!=expected or not np.all(np.isfinite(result[k])):
            raise ValueError(f'{path}: invalid {k} shape or nonfinite data')
    for k in axes:
        a=result[k]
        if a.ndim!=1 or len(a)<3 or not np.all(np.isfinite(a)) or not np.all(np.diff(a)>0):
            raise ValueError(f'{path}: {k} must have at least 3 increasing coordinates')
        if not np.allclose(np.diff(a),np.diff(a)[0],rtol=1e-7,atol=1e-12):
            raise ValueError(f'{path}: nonuniform {k} is not supported by FD2/GS contract')
    return result


def match(reference,candidate,boundary=False):
    for k in ('field_unit','length_unit'):
        if reference[k]!=candidate[k]:raise ValueError(f'Unit mismatch: {k}')
    for k in ('x','y') if boundary else ('x','y','z'):
        if not np.array_equal(reference[k],candidate[k]):
            raise ValueError(f'Coordinate mismatch: {k}; no implicit resampling is allowed')


def cci_errors(target,candidate,slices,cutoffs):
    _,sy,sx=slices
    b=target[:,sy,sx];B=candidate[:,sy,sx]
    mag=np.linalg.norm(b,axis=0);diff=np.linalg.norm(B-b,axis=0)
    denominator=float(np.linalg.norm(b))
    result={'E_rel_Bgt_L2':float(np.linalg.norm(B-b)/denominator) if denominator>0 else None,
            'CCI_pixels_total':int(mag.size),'CCI_cutoff_results':{}}
    for cutoff in cutoffs:
        mask=mag>float(cutoff)*float(mag.max())
        result['CCI_cutoff_results'][str(cutoff)]={
            'E_rel_Bgt':float(np.mean(diff[mask]/mag[mask])) if mask.any() else None,
            'retained_pixels':int(mask.sum()),'retained_fraction':float(mask.mean())}
    result['E_rel_Bgt']=result['CCI_cutoff_results'][str(cutoffs[0])]['E_rel_Bgt']
    return result


def evaluate(reference_path,field_path,boundary_path,config_path):
    config=json.loads(Path(config_path).read_text())
    if config.get('schema_version')!=1:raise ValueError('Unsupported evaluation config')
    ref=load(reference_path);field=load(field_path);obs=load(boundary_path,True)
    match(ref,field);match(ref,obs,True)
    if ref['z'][0]!=0:raise ValueError('Photospheric z=0 must be the first plane')
    if any(field[k].shape!=ref[k].shape for k in KEYS):raise ValueError('Field shape mismatch')
    selections=config['scoring_slices_zyx'];slices=tuple(slice(*s) for s in selections)
    if len(slices)!=3:raise ValueError('Three z,y,x scoring slices are required')
    for n,spec in zip(ref['Bx'].shape,selections):
        start,stop=spec[:2];step=spec[2] if len(spec)>2 else 1
        if not 0<=start<stop<=n or step not in (None,1):raise ValueError('Scoring box is outside the field or is strided')
    spacing=tuple(float(np.diff(ref[k])[0]) for k in ('x','y','z'))
    j=curl(field,spacing);div=divergence(field,spacing)
    # 6/h_eff = 2*(1/dx+1/dy+1/dz): exact legacy convention for isotropic TD.
    h_eff=3/sum(1/h for h in spacing)
    metrics=compute_metrics(ref,field,j,div,slices,h_eff)
    gs=config['gaussian_separation']
    args=SimpleNamespace(**gs)
    helper=make_bgt_helper_solver(nx=len(ref['x']),ny=len(ref['y']),
            dx=spacing[0],dy=spacing[1],dz=spacing[2],device='cpu',dtype=torch.float64,args=args)
    observation={k:obs[k][None,...] for k in KEYS}
    target=bottom_plane_bgt_nfgs(helper,observation)
    candidate=bottom_plane_bgt_nfgs(helper,field)
    metrics.update(cci_errors(target,candidate,slices,config['CCI_cutoff_fractions']))
    # Supplementary clean-reference CCI is separate from fidelity to the actual input.
    clean=bottom_plane_bgt_nfgs(helper,ref)
    clean_errors=cci_errors(clean,candidate,slices,config['CCI_cutoff_fractions'])
    metrics['clean_reference_CCI']=clean_errors
    _,sy,sx=slices
    metrics['bottom_Bz_RMSE_to_input']=float(np.sqrt(np.mean((field['Bz'][0,sy,sx]-obs['Bz'][sy,sx])**2)))
    jobs=curl({k:np.repeat(obs[k][None],3,axis=0) for k in KEYS},spacing)['Jz'][0]
    metrics['bottom_Jz_RMSE_to_input_fd2']=float(np.sqrt(np.mean((j['Jz'][0,sy,sx]-jobs[sy,sx])**2)))
    return {'schema_version':1,'case':config['case'],'method':config['method'],
            'current_operator':'common_fd2_float64_full_volume_then_crop',
            'CCI_target':'actual_observed_input_full_footprint_then_crop',
            'spacing_xyz':spacing,'scoring_slices_zyx':selections,
            'field_unit':ref['field_unit'],'length_unit':ref['length_unit'],
            'source_sha256':{k:sha(v) for k,v in {'reference':reference_path,'field':field_path,
                'boundary':boundary_path,'config':config_path}.items()},
            'metrics':metrics,'diagnostics_not_computed':['FastQSL','twist','six_face_energy','helicity'],
            'controlled_acquisition_verified':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','field','boundary','config','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=evaluate(a.reference,a.field,a.boundary,a.config)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(a.output),'method':result['method'],'metrics':result['metrics']},indent=2))
