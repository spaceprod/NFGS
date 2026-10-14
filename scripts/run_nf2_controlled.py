#!/usr/bin/env python3
"""Run an externally installed pinned NF2 with offline logging and explicit coordinates.

NF2 retains its upstream GPL license; no upstream source is vendored here.
"""
from pathlib import Path
import argparse,json,time,os,sys
os.environ['WANDB_MODE']='offline';os.environ['WANDB_SILENT']='true';os.environ['WANDB_DISABLE_CODE']='true'
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from lightning.pytorch import seed_everything,Callback
from nfgs.comparison_policy import dump_json as dump, file_sha as sha
import nf2.extrapolate as extrapolate
from nf2.train.config import normalize_config


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();job=json.loads(a.job.read_text());out=a.root/'runs'/job['id'];out.mkdir(parents=True,exist_ok=True)
    folder=a.root/'cases'/job['case_spec']['id'];meta=json.loads((folder/'case.json').read_text());start=time.perf_counter();started_wall=time.time()
    seed_everything(int(job['settings'].get('seed',0)),workers=True)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    template=json.loads((ROOT/'configs/controlled/nf2_bottom_only_template.json').read_text())
    template['path']=str(out/'nf2');template['work_path']=str(out/'nf2_work')
    template['logging']={'mode':'offline','project':'nfgs-controlled','name':job['id'],'log_model':False}
    L=meta['NF2_length_normalization'];bscale=meta['NF2_field_normalization'];template['data']['normalization']={'Mm_per_ds':L,'Gauss_per_dB':bscale}
    ds={'id':'boundary','type':'numpy','data_path':str(folder/'nf2_boundary_xy.npz'),
        'Mm_per_pixel':meta['spacing_xyz'][0],'coordinate_center':[(v[0]+v[1])/2 for k,v in meta['bounds_xyz'].items() if k in ('x','y')],
        'batch_size':512,'plot':False}
    template['data']['boundaries']=[ds];template['data']['validation']=[dict(ds)];template['data']['z_range']=meta['bounds_xyz']['z'];template['data']['num_workers']=0
    template['losses'][0]['weight']=float(job['settings'].get('boundary_weight_multiplier',1))
    if job['settings'].get('potential_boundary'):
        template['data']['potential_boundary']={'type':'potential','strides':4}
        template['losses'].insert(1,{'type':'boundary','name':'potential_boundary','weight':10,'datasets':['potential']})
    # User explicitly chose direct production, no pilot-driven budget adjustment.
    template['training']['trainer'].update(max_steps=15000,precision='32-true',enable_progress_bar=False,enable_model_summary=False,deterministic=False)
    template['training']['validation_interval']=1000
    resolved=normalize_config(template);dump(out/'nf2_public_config.json',template);dump(out/'nf2_resolved_config.json',resolved)
    # The upstream numpy reader assumes one pixel spacing. Supply explicit observed
    # physical coordinates through its supported MapDataset coords argument for
    # exact-box resolution variants with dx != dy, without changing upstream code.
    if not np.isclose(meta['spacing_xyz'][0],meta['spacing_xyz'][1],rtol=1e-7):
        import nf2.loader.cartesian as cartesian
        OriginalNumpyDataset=cartesian.NumpyDataset
        with np.load(folder/'boundary.npz') as data:observed_x=data['x'].copy();observed_y=data['y'].copy()
        xx,yy=np.meshgrid(observed_x,observed_y,indexing='ij');observed_coords=np.stack([xx,yy,np.zeros_like(xx)],axis=-1).astype(np.float32)
        class ExplicitCoordinateNumpyDataset(OriginalNumpyDataset):
            def __init__(self,*args,**kwargs):kwargs['coords']=observed_coords;super().__init__(*args,**kwargs)
        # torch serializes the data module: register this local wrapper under a
        # module-level name so pickle can reconstruct it within this process.
        ExplicitCoordinateNumpyDataset.__qualname__='ExplicitCoordinateNumpyDataset'
        globals()['ExplicitCoordinateNumpyDataset']=ExplicitCoordinateNumpyDataset
        cartesian.NumpyDataset=ExplicitCoordinateNumpyDataset
        dump(out/'explicit_coordinate_adapter.json',{'purpose':'anisotropic observed x-y coordinates; no reference-field constraint','shape':list(observed_coords.shape),'input_sha256':meta['sha256']['boundary.npz']})
    fit_start=[None];fit_end=[None];actual_steps=[0]
    class Progress(Callback):
        def on_train_start(self,trainer,module):
            torch.cuda.synchronize();fit_start[0]=time.perf_counter()
        def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
            n=int(trainer.global_step)
            if n==1 or n%100==0:
                torch.cuda.synchronize();elapsed=time.perf_counter()-fit_start[0]
                dump(out/'progress.json',{'status':'running','phase':'training','method':'NF2','step':n,'total_steps':15000,
                     'training_elapsed_s':elapsed,'step_rate':n/max(elapsed,1e-9),'updated_at':time.time()})
            if n%1000==0 and job['stage']!='timing':
                checkpoint=out/'model_snapshots'/f'step_{n:05d}.pt';checkpoint.parent.mkdir(exist_ok=True)
                torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in module.model.state_dict().items()},'step':n,'training_elapsed_s':time.perf_counter()-fit_start[0]},checkpoint)
        def on_train_end(self,trainer,module):
            torch.cuda.synchronize();fit_end[0]=time.perf_counter();actual_steps[0]=int(trainer.global_step)
    OriginalTrainer=extrapolate.Trainer
    class ObservedTrainer(OriginalTrainer):
        def __init__(self,**kwargs):kwargs['callbacks'].append(Progress());super().__init__(**kwargs)
    extrapolate.Trainer=ObservedTrainer
    torch.cuda.reset_peak_memory_stats();extrapolate.run(**resolved)
    if actual_steps[0]!=15000:raise RuntimeError(f'Fixed production budget not fulfilled: {actual_steps[0]} steps')
    checkpoint=out/'nf2/extrapolation_result.nf2';state=torch.load(checkpoint,map_location='cuda',weights_only=False);model=state['model'];model.eval()
    with np.load(folder/'reference.npz') as data:coords={k:data[k] for k in ['x','y','z']};units={k:data[k] for k in ['field_unit','length_unit']}
    shape=(len(coords['z']),len(coords['y']),len(coords['x']));values=np.empty((3,*shape),np.float32)
    torch.cuda.synchronize();inference_start=time.perf_counter()
    for iz,z in enumerate(coords['z']):
        xx,yy=np.meshgrid(coords['x']/L,coords['y']/L,indexing='xy');points=np.stack([xx,yy,np.full_like(xx,z/L)],axis=-1).reshape(-1,3)
        chunks=[]
        for k in range(0,len(points),4096):
            pts=torch.tensor(points[k:k+4096],dtype=torch.float32,device='cuda',requires_grad=True)
            model.zero_grad();b=model(pts,compute_jacobian=False)['b'];chunks.append(b.detach().cpu().numpy()*bscale)
        values[:,iz]=np.concatenate(chunks).reshape(shape[1],shape[2],3).transpose(2,0,1)
        if iz%10==0:dump(out/'progress.json',{'status':'running','phase':'inference','method':'NF2','planes_done':iz+1,'planes_total':shape[0],'updated_at':time.time()})
    torch.cuda.synchronize();inference_seconds=time.perf_counter()-inference_start
    np.savez_compressed(out/'field.npz',**{k:values[i] for i,k in enumerate(['Bx','By','Bz'])},**coords,**units)
    end_to_end_seconds=time.perf_counter()-start
    dump(out/'result.json',{'status':'solver_complete','job':job,'case_sha256':meta['sha256'],
         'public_adapter':'scripts/run_nf2_controlled.py','NF2_commit':'e7b3ae97dcb37161aea6cd9510d9235de396270a','torch_version':torch.__version__,
         'precision':'float32','seed':job['settings'].get('seed',0),'training_steps':actual_steps[0],
         'selected_state':'last_fixed_budget_checkpoint','reference_field_used_in_training_or_selection':False,
         'training_s':fit_end[0]-fit_start[0],'solver_only_s':fit_end[0]-fit_start[0],
         'inference_s':inference_seconds,'end_to_end_s':end_to_end_seconds,'end_to_end_clock':'CLOCK_MONOTONIC','measurement_started_at_wall':started_wall,
         'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated(),'field_sha256':sha(out/'field.npz'),
         'end_to_end_scope':'cached canonical input load through synchronized field computation and NPZ export; excludes common evaluation and case generation','CUDA_synchronization_bracketed':True,'logging':'offline_only','case_normalizations':{'L':L,'B':bscale}})
    dump(out/'progress.json',{'status':'solver_complete','updated_at':time.time()})


if __name__=='__main__':main()
