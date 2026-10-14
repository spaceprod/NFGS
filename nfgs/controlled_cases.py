"""Immutable coordinate-explicit cases for live controlled comparisons."""
from pathlib import Path
import json,hashlib,os,copy
import numpy as np
from nfgs.paths import project_root
from nfgs._backend.scripts.run_nfgs_analytic_validation import make_td_case
from nfgs._backend.util.lnl_io import get_analytic_b_field,LOW_LOU_FAMILY
from nfgs._backend.scripts.run_analytic_validation import TD_TOPOLOGY
from nfgs._backend.td import generate_td_model


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def prepare_case(root,spec):
    folder=Path(root)/'cases'/spec['id'];meta=folder/'case.json'
    if meta.exists():
        result=json.loads(meta.read_text())
        if result['spec']!=spec:raise RuntimeError('Immutable case specification changed')
        if all((folder/name).is_file() for name in result['sha256']):return folder
        raise RuntimeError('Case cache is incomplete')
    folder.mkdir(parents=True,exist_ok=True)
    base=spec['base'];factor=spec.get('spacing_factor',1.0)
    if spec.get('noise'):
        clean_spec={k:v for k,v in spec.items() if k!='noise'};clean_spec['id']=base
        clean=prepare_case(root,clean_spec);result=copy.deepcopy(json.loads((clean/'case.json').read_text()))
        with np.load(clean/'boundary.npz') as data:boundary={k:data[k].copy() for k in ['Bx','By','Bz']};other={k:data[k] for k in ['x','y','field_unit','length_unit']}
        noise=spec['noise'];rng=np.random.default_rng(noise['seed']);rms=np.sqrt(sum(np.mean(v*v) for v in boundary.values()))
        for k in boundary:boundary[k]+=rng.normal(0,rms*(noise['Bz'] if k=='Bz' else noise['Bh']),boundary[k].shape)
        os.link(clean/'reference.npz',folder/'reference.npz')
        np.savez_compressed(folder/'boundary.npz',**boundary,**other)
        np.savez_compressed(folder/'nf2_boundary_xy.npz',**{k.lower():v.T.astype(np.float32) for k,v in boundary.items()})
        result.update(case=spec['id'],spec=spec,NF2_field_normalization=float(np.sqrt(sum(np.mean(v*v) for v in boundary.values()))))
        result['noise_absolute_sigma']={'Bz':rms*noise['Bz'],'Bh':rms*noise['Bh'],'normalization':'clean boundary vector RMS'}
        result['sha256']={name:sha(folder/name) for name in ['reference.npz','boundary.npz','nf2_boundary_xy.npz']}
        dump(meta,result);return folder
    if base.startswith('td'):
        case=make_td_case(base,calculation_domain_mode='mudpack_legacy',chunk_depth=4,n_test_x=66,n_test_y=66,n_test_z=30)
        nx,ny,nz,dx,dy,dz=case['dims'];coords={k:np.linspace(*case[k+'_bounds'],n,dtype=np.float64) for k,n in [('x',nx),('y',ny),('z',nz)]}
        fields=case['reference'];slices=case['metric_slices'];params=case['geometry']['parameters']
        physical_score={k:[float(coords[k][s.start or 0]),float(coords[k][s.stop-1])] for k,s in zip(['z','y','x'],slices)}
        if factor!=1:
            coords={k:np.linspace(v[0],v[-1],max(3,int(round((len(v)-1)/factor))+1),dtype=np.float64) for k,v in coords.items()}
            nx,ny,nz=[len(coords[k]) for k in ['x','y','z']];fields={k:np.empty((nz,ny,nx),np.float32) for k in ['Bx','By','Bz']}
            for z0 in range(0,nz,4):
                xx,yy,zz=np.meshgrid(coords['x'],coords['y'],coords['z'][z0:z0+4],indexing='ij')
                values=generate_td_model(xx,yy,zz,R=params['R'],a=params['a'],L=params['L'],d=params['d'],q=params['q'],I_0=-params['I0'],l_i=params['l_i'],p_Bt_ratio=params['p_Bt_ratio'],elliptic_invalid='nan')
                for key,value in zip(fields,values):fields[key][z0:z0+4]=value.transpose(2,1,0)
            selections=[]
            for axis in ['z','y','x']:
                indexes=np.flatnonzero((coords[axis]>=physical_score[axis][0]-1e-10)&(coords[axis]<=physical_score[axis][1]+1e-10))
                selections.append(slice(int(indexes[0]),int(indexes[-1])+1))
            slices=tuple(selections)
        length_unit='Mm'
    else:
        legacy=spec.get('grid_kind')=='paper_anisotropic';shape=(80,80,72) if legacy else (80,80,80)
        coords={'x':np.linspace(-1,1,shape[0]),'y':np.linspace(-1,1,shape[1]),'z':np.linspace(0,2,shape[2])}
        values=get_analytic_b_field(**LOW_LOU_FAMILY[base],resolution=shape,dtype=np.float64)
        fields={k:values[...,i].transpose(2,1,0).copy() for i,k in enumerate(['Bx','By','Bz'])}
        # Preserve the original physical scoring height as closely as this common native grid permits.
        zstop=64 if legacy else int(np.searchsorted(coords['z'],63*2/71+1e-12,side='right'))
        slices=(slice(0,zstop),slice(8,72),slice(8,72));params=LOW_LOU_FAMILY[base];length_unit='normalized'
        physical_score={k:[float(coords[k][s.start]),float(coords[k][s.stop-1])] for k,s in zip(['z','y','x'],slices)}
    boundary={k:np.asarray(v[0],np.float64).copy() for k,v in fields.items()}
    noise=spec.get('noise')
    if noise:
        rng=np.random.default_rng(noise['seed']);rms=np.sqrt(sum(np.mean(v*v) for v in boundary.values()))
        for k in boundary:boundary[k]+=rng.normal(0,rms*(noise['Bz'] if k=='Bz' else noise['Bh']),boundary[k].shape)
    payload={**fields,**coords,'field_unit':'benchmark','length_unit':length_unit}
    np.savez_compressed(folder/'reference.npz',**payload)
    np.savez_compressed(folder/'boundary.npz',**boundary,x=coords['x'],y=coords['y'],field_unit='benchmark',length_unit=length_unit)
    np.savez_compressed(folder/'nf2_boundary_xy.npz',**{k.lower():v.T.astype(np.float32) for k,v in boundary.items()})
    spacing=[float(np.diff(coords[k])[0]) for k in ['x','y','z']]
    result={'schema_version':1,'case':spec['id'],'spec':spec,'grid_xyz':[len(coords[k]) for k in ['x','y','z']],
      'spacing_xyz':spacing,'bounds_xyz':{k:[float(v[0]),float(v[-1])] for k,v in coords.items()},
      'scoring_slices_zyx':[[s.start,s.stop,1] for s in slices],'physical_scoring_bounds':physical_score,
      'source_parameters':params,'length_unit':length_unit,'field_unit':'benchmark',
      'fortran_uniform_scale_supported':bool(np.allclose(spacing,spacing[0],rtol=1e-8)),
      'NF2_length_normalization':float((coords['x'][-1]-coords['x'][0])/2),
      'NF2_field_normalization':float(np.sqrt(sum(np.mean(v*v) for v in boundary.values()))),
      'side_top_reference_field_supplied_to_solvers':False,
      'sha256':{name:sha(folder/name) for name in ['reference.npz','boundary.npz','nf2_boundary_xy.npz']}}
    dump(meta,result);return folder
