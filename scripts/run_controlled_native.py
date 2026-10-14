#!/usr/bin/env python3
"""Run a full-budget NFGS or Python NFPT job on an immutable case."""
from pathlib import Path
import argparse,json,time,sys,hashlib
import numpy as np
import torch
from nfgs.paths import project_root
from nfgs.controlled_cases import prepare_case,dump,sha
from nfgs_solver import NFGSSolverGPU,NFPTSolverGPU
from nfgs.controlled import cci_errors
from nfgs.diagnostics.common_fd2 import curl
from nfgs._backend.scripts.plot_nfgs_td_robin_metrics_vs_er import make_bgt_helper_solver,bottom_plane_bgt_nfgs
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    job=json.loads(a.job.read_text());out=a.root/'runs'/job['id'];out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();started_wall=time.time()
    folder=prepare_case(a.root,job['case_spec']);meta=json.loads((folder/'case.json').read_text())
    with np.load(folder/'boundary.npz') as data:boundary={k:np.asarray(data[k]) for k in ['Bx','By','Bz']}
    nx,ny,nz=meta['grid_xyz'];dx,dy,dz=meta['spacing_xyz'];settings=job['settings'];solver_diagnostic=job.get('stage')!='timing';diagnostic=bool(settings.get('iteration_observers',solver_diagnostic))
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cls=NFPTSolverGPU if job['method']=='NFPT_GPU' else NFGSSolverGPU
    kwargs=dict(device='cuda',dtype=torch.float32,lateral_wall='paper_bc1',pin_potential_boundaries=False,buffer_thickness=0,padded_buffer_zone=True,pad_calculation_domain=True,calculation_domain_mode=settings.get('padding_mode','mudpack_legacy'))
    if cls is NFGSSolverGPU:
        kwargs.update(source_partition_mode='legacy_single_field' if job['method']=='NFGS_legacy' else 'gaussian_separated',
            bottom_divergence_robin_weight=settings.get('rho',0),mean_bz_policy='uniform_open',
            horizontal_zero_mode_policy='omit' if job['method']=='NFGS_legacy' else 'frozen_linear_harmonic',d0_zero_mode_policy='project',source_zero_mode_policy='project',
            harmonic_residual_policy='diagnose',poloidal_update_mode='direct',vertical_current_operator='staggered_face',record_iteration_diagnostics=solver_diagnostic)
    explicit_padding=settings.get('explicit_padding_cells')
    constructor_xyz=(nx,ny,nz)
    if explicit_padding is not None:
        left,right,front,back,top=[int(explicit_padding[k]) for k in ['x_low','x_high','y_low','y_high','z_top']]
        if min(left,right,front,back,top)<0 or left!=right or front!=back:raise ValueError('Explicit padding currently requires nonnegative symmetric lateral pads')
        constructor_xyz=(nx+left+right,ny+front+back,nz+top)
        kwargs.update(pad_calculation_domain=False,padded_buffer_zone=True)
    torch.cuda.synchronize();prepare_start=time.perf_counter();solver=cls(*constructor_xyz,dx,dy,dz,**kwargs)
    if explicit_padding is not None:
        # Metadata tells the existing zero-extension boundary loader and native
        # monitor which nodes belong to the observed box. No PDE/operator change.
        solver._input_shape_zyx=(nz,ny,nx)
        solver._calculation_pad_xy=(left,front)
        solver._calculation_domain_mode='explicit_author_padding'
        solver._auto_padded_calculation_domain=constructor_xyz!=(nx,ny,nz)
    solver.set_bottom_vector_boundary(boundary['Bx'],boundary['By'],boundary['Bz'],zero_net_current=False,smooth_passes=0,ijzbcor=3)
    if explicit_padding is not None:solver._observed_shape_zyx=(nz,ny,nx)
    torch.cuda.synchronize();prepare_seconds=time.perf_counter()-prepare_start
    monitor_helper=None;target_cci=None;slices=tuple(slice(*v) for v in meta['scoring_slices_zyx'])
    if diagnostic:
        args=SimpleNamespace(mean_bz_policy='uniform_open',horizontal_zero_mode_policy='frozen_linear_harmonic',d0_zero_mode_policy='project',source_zero_mode_policy='project',harmonic_residual_policy='diagnose',mean_mode_tolerance=1e-6,source_zero_mode_tolerance=.01,gaussian_horizontal_operator='spectral')
        monitor_helper=make_bgt_helper_solver(nx=nx,ny=ny,dx=dx,dy=dy,dz=dz,device='cpu',dtype=torch.float64,args=args)
        target_cci=bottom_plane_bgt_nfgs(monitor_helper,{k:v[None] for k,v in boundary.items()})
    events=[];previous=None;monitor_seconds=0.
    sy,sx=solver.fov_slice_yx
    def safe(value):
        if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [safe(v) for v in value]
        if isinstance(value,(np.floating,np.integer)):return safe(value.item())
        if isinstance(value,float) and not np.isfinite(value):return None
        return value
    def update(n,sigma,B,J):
        nonlocal previous,monitor_seconds
        event={'iteration':n,'native_sigma_monitor':float(sigma),'elapsed_solver_s':time.perf_counter()-t0}
        if diagnostic:
            if previous is not None:event['relative_B_state_change']=float(torch.linalg.vector_norm(B-previous)/torch.linalg.vector_norm(previous).clamp_min(1e-30))
            previous=B.clone()
            event.update({k:float(v) for k,v in (getattr(solver,'_last_iteration_diagnostics',{}) or {}).items() if isinstance(v,(float,int))})
        if diagnostic and (n==1 or n%10==0):
            diagnostic_start=time.perf_counter()
            fields={k:B[i,:nz,sy,sx].detach().cpu().numpy().astype(np.float64) for i,k in enumerate(['Bx','By','Bz'])}
            candidate=bottom_plane_bgt_nfgs(monitor_helper,fields)
            errors=cci_errors(target_cci,candidate,slices,[1e-8,.01,.05])
            event['E_rel_Bgt']=errors['E_rel_Bgt'];event['E_rel_Bgt_L2']=errors['E_rel_Bgt_L2']
            current=curl(fields,(dx,dy,dz));b=np.stack([fields[k][slices] for k in ['Bx','By','Bz']]);j=np.stack([current[k][slices] for k in ['Jx','Jy','Jz']])
            event['common_sigma_J']=float(np.sum(np.linalg.norm(np.cross(j,b,axisa=0,axisb=0,axisc=0),axis=0)/np.maximum(np.linalg.norm(b,axis=0),1e-30))/max(float(np.sum(np.linalg.norm(j,axis=0))),1e-30))
            monitor_seconds+=time.perf_counter()-diagnostic_start
            event['diagnostic_overhead_cumulative_s']=monitor_seconds
        events.append(event)
        if n==1 or n%10==0:dump(out/'iteration_diagnostics.json',safe(events))
        if n==1 or n%10==0:dump(out/'progress.json',{'status':'running','method':job['method'],'iteration':n,'maximum_iterations':3000,'elapsed_s':time.perf_counter()-start,'native_sigma_monitor':float(sigma),'updated_at':time.time()})
    torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
    init='curl_free_current' if cls is NFPTSolverGPU else 'cfcf' if job['method']=='NFGS_legacy' else 'gs_cfcf'
    with torch.inference_mode():history=solver.iterate(max_iter=3000,tol=settings.get('tolerance',1e-5),mu=1,patience=settings.get('patience',40),sigma_improvement_tol=settings.get('improvement_tolerance',1e-5),init_mode=init,per_step_callback=update,verbose=True)
    torch.cuda.synchronize();solver_seconds=time.perf_counter()-t0
    field={k:v for k,v in solver.fields_to_numpy().items() if k in ['Bx','By','Bz']}
    with np.load(folder/'reference.npz') as data:
        field.update({k:data[k] for k in ['x','y','z','field_unit','length_unit']})
    np.savez_compressed(out/'field.npz',**field)
    end_to_end_seconds=time.perf_counter()-start
    def safe(value):
        if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [safe(v) for v in value]
        if isinstance(value,(np.floating,np.integer)):return safe(value.item())
        if isinstance(value,float) and not np.isfinite(value):return None
        return value
    dump(out/'history.json',safe(history));dump(out/'iteration_diagnostics.json',safe(events))
    dump(out/'result.json',{'status':'solver_complete','job':job,'case_sha256':meta['sha256'],
        'torch_version':torch.__version__,'dtype':'float32','gpu':'RTX 4070 Ti SUPER',
        'calculation_grid_xyz':[solver.Nx,solver.Ny,solver.Nz],'padding_xy':list(solver._calculation_pad_xy),
        'explicit_padding_cells':explicit_padding,'observed_grid_xyz':[nx,ny,nz],'observed_shape_metadata_zyx':list(solver._observed_shape_zyx),'observed_fov_slices_yx':[[v.start,v.stop] for v in solver.fov_slice_yx],'initialization':init,'solver_only_s':solver_seconds,'preparation_s':prepare_seconds,'end_to_end_s':end_to_end_seconds,'end_to_end_clock':'CLOCK_MONOTONIC','measurement_started_at_wall':started_wall,
        'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'native_stop_reason':history.get('stop_reason'),
        'iterations':len(history.get('sigma',[])),'best_iteration':history.get('best_iter'),
        'diagnostic_overhead_s':monitor_seconds,'timing_instrumented':diagnostic,'end_to_end_scope':'cached canonical input load through synchronized field computation and NPZ export; excludes common evaluation and case generation','CUDA_synchronization_bracketed':True,'field_sha256':sha(out/'field.npz'),
        'solver_source_sha256':sha(project_root()/'nfgs_solver.py')})
    dump(out/'progress.json',{'status':'solver_complete','iteration':len(history.get('sigma',[])),'updated_at':time.time()})


if __name__=='__main__':main()
