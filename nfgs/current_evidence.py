#!/usr/bin/env python3
"""Build manuscript evidence from one complete reconstruction campaign.

All diagnostics remain attached to their input fields; no old NFGS results are
merged into a new campaign. Published NFPT comparison rows are not modified.
"""
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import torch

from nfgs.paths import project_root
ROOT=project_root()
sys.path.insert(0,str(ROOT))


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def write_table(path,columns,caption,label,heads,rows,comment=''):
    lines=[r'\begin{deluxetable*}{'+columns+'}']
    if len(heads)>=9:lines.extend([r'\tabletypesize{\scriptsize}',r'\tablewidth{\textwidth}'])
    lines += [r'\tablecaption{'+caption+r'\label{'+label+'}}',
           r'\tablehead{'+'&'.join(r'\colhead{'+h+'}' for h in heads)+'}',r'\startdata']
    lines+=[' & '.join(row)+r'\\' for row in rows]
    lines += [r'\enddata']
    if comment:lines += [r'\tablecomments{'+comment+'}']
    lines += [r'\end{deluxetable*}']
    Path(path).write_text('\n'.join(lines)+'\n')


def build_low_lou_cci(data,selected_rho):
    from nfgs._backend.experiments.nfgs_paper.analytic_nlfff.benchmarks import make_low_lou_case
    from nfgs._backend.scripts.plot_nfgs_td_robin_metrics_vs_er import make_bgt_helper_solver,bottom_plane_bgt_nfgs
    from nfgs.diagnostics.common_fd2 import sigma_j
    cases={name:make_low_lou_case('mudpack',name) for name in ['low_lou_ll1','low_lou_ll2','low_lou_c2']}
    result=[]
    for rho in sorted({0.,.5,1.,selected_rho}):
        base=data/('low_lou_family_rho0p5/float32' if rho==.5 else f'low_lou_baselines/rho{rho:g}')
        records=json.loads((base/'metrics.json').read_text())
        for record in records:
            name=record['benchmark'];case=cases[name]
            nx,ny,nz,dx,dy,dz=case['dims']
            args=SimpleNamespace(mean_bz_policy='uniform_open',horizontal_zero_mode_policy='frozen_linear_harmonic',d0_zero_mode_policy='project',source_zero_mode_policy='project',harmonic_residual_policy='diagnose',mean_mode_tolerance=1e-6,source_zero_mode_tolerance=.01,gaussian_horizontal_operator='spectral')
            helper=make_bgt_helper_solver(nx=nx,ny=ny,dx=dx,dy=dy,dz=dz,device='cpu',dtype=torch.float64,args=args)
            target=bottom_plane_bgt_nfgs(helper,{k:v[None] for k,v in case['boundary'].items()})
            with np.load(base/'fields'/f'{name}__nfgs__gs_cfcf.npz') as f:field={k:f[k] for k in ['Bx','By','Bz']}
            candidate=bottom_plane_bgt_nfgs(helper,field)
            _,sy,sx=case['metric_slices'];b=target[:,sy,sx];B=candidate[:,sy,sx]
            mag=np.linalg.norm(b,axis=0);err=np.linalg.norm(B-b,axis=0);mask=mag>1e-8*mag.max()
            result.append({k:record[k] for k in ['benchmark','C_vec','C_CS','Eprime_n','Eprime_m','epsilon','n_iters','best_iter','stop_reason']}
                          | dict(rho=rho,E_rel_Bgt=float(np.mean(err[mask]/mag[mask])),
                                 E_rel_Bgt_L2=float(np.linalg.norm(B-b)/np.linalg.norm(b)),
                                 sigma_J=sigma_j(field,(dx,dy,dz),case['metric_slices'])))
    dump(data/'low_lou_comparison.json',result)
    return result


def ensure_structure_weights(data):
    """Trace the selected compromise as an independent structural check."""
    rows=[r for r in read_csv(data/'common_diagnostics/headline_common_fd2_protocol.csv') if r['rho']]
    values=np.array([[float(r['E_rel_Bgt_domain']),float(r['sigma_J'])] for r in rows])
    distances=np.linalg.norm((values-values.min(0))/np.ptp(values,axis=0),axis=1)
    weights=sorted({float(r['rho']) for r in rows})
    chosen=min(weights,key=lambda w:np.mean([d for d,r in zip(distances,rows) if float(r['rho'])==w]))
    traces=sorted({0.,.1,.5,1.,chosen});helicity=sorted({0.,.5,1.,chosen})
    nfpt=ROOT/'data/nfpt'
    specifications=[
        ('quantitative_topology',traces,['--resume']),
        ('twist_apex',traces,['--resume','--omit-nfpt']),
        ('six_face_helicity',helicity,['--omit-nfpt'])]
    for name,wanted,extra in specifications:
        existing=read_csv(data/name/'metrics.csv')
        have={(r['benchmark'],r['method']) for r in existing}
        if all((c,f'nfgs_rho_{w:.1f}') in have for c in ['td1','td2','td3'] for w in wanted):continue
        cmd=[sys.executable,'-m',f'nfgs._backend.experiments.nfgs_paper.{name}.run',
             '--field-dir',str(data/'rho_full/fields'),'--nfpt-field-dir',str(nfpt),
             '--output-dir',str(data/name),'--weights',','.join(f'{w:g}' for w in wanted),*extra]
        subprocess.run(cmd,cwd=ROOT,check=True)
    if chosen not in [0.,.5,1.]:
        folder=data/f'low_lou_baselines/rho{chosen:g}'
        if not (folder/'metrics.json').exists():
            cmd=[sys.executable,'-m','nfgs._backend.experiments.nfgs_paper.analytic_nlfff.run',
                '--output-dir',str(folder),'--field-cache-dir',str(folder/'fields'),
                '--mixed-robin-weight',f'{chosen:g}','--device','cuda','--dtype','float32',
                '--max-iter-lnl','3000','--stop-mode','sigma_best','--tol','1e-5',
                '--patience','40','--sigma-improvement-tol','1e-5','--record-iteration-diagnostics']
            subprocess.run(cmd,cwd=ROOT,check=True)
    return chosen


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign-root',type=Path,required=True);a=p.parse_args()
    data=a.campaign_root.resolve();gen=data/'figures';gen.mkdir(exist_ok=True)
    selected_rho=ensure_structure_weights(data)
    spec=importlib.util.spec_from_file_location('reader_evidence',Path(__file__).resolve().parent/'_backend/scripts/build_nfgs_second_readability_evidence.py')
    evidence=importlib.util.module_from_spec(spec);spec.loader.exec_module(evidence)
    evidence.DATA=data;evidence.OUT=data/'second_reader_revision';evidence.GEN=gen;evidence.OUT.mkdir(exist_ok=True)
    evidence.repair_low_lou();_,budgets=evidence.td_profiles(selected_weight=selected_rho)
    thresholds=evidence.q_thresholds()
    from nfgs.diagnostics.topology_metrics import q_map_metrics
    # The main structure table also includes rho=0.1 and any additional
    # selected-weight diagnostic, beyond the three threshold-study weights.
    traced=read_csv(data/'quantitative_topology/metrics.csv')
    for case in ['td1','td2','td3']:
        with np.load(data/'quantitative_topology'/f'{case}__analytic_reference.npz') as f:reference=f['q'].copy()
        for row in traced:
            if row['benchmark']!=case or not row['method'].startswith('nfgs_rho_'):continue
            rho=float(row['method'].split('_')[-1])
            if rho in [0.,.5,1.]:continue
            with np.load(data/'quantitative_topology'/f'{case}__{row["method"]}.npz') as f:candidate=f['q'].copy()
            for threshold in [2.,2.5,3.]:thresholds.append(dict(benchmark=case,rho=rho,high_logq=threshold,**q_map_metrics(reference,candidate,high_logq=threshold)))
    rows=read_csv(data/'common_diagnostics/headline_common_fd2_protocol.csv')
    nfgs=[r for r in rows if r['method'].startswith('nfgs_rho_')]
    assert len(nfgs)==33
    values=np.array([[float(r['E_rel_Bgt_domain']),float(r['sigma_J'])] for r in nfgs])
    scaled=(values-values.min(0))/np.ptp(values,axis=0)
    selection=[]
    for weight in [.5,1.,2.]:
        ds=np.sqrt(weight*scaled[:,0]**2+scaled[:,1]**2)
        for rho in sorted({float(r['rho']) for r in nfgs}):
            selection.append(dict(cci_squared_term_weight=weight,rho=rho,mean_distance=float(np.mean([d for r,d in zip(nfgs,ds) if float(r['rho'])==rho]))))
    evidence.write_csv(evidence.OUT/'selection_sensitivity.csv',selection)
    evidence.write_latex_tables(thresholds,selection)
    for r,d in zip(nfgs,np.linalg.norm(scaled,axis=1)):r['distance']=float(d)
    raw={}
    for path in (data/'rho_full').glob('robin_w*.json'):
        for r in json.loads(path.read_text()):raw[(r['benchmark'],round(float(r['robin_weight']),1))]=r
    joined=[]
    for r in nfgs:
        base=raw[(r['benchmark'],round(float(r['rho']),1))]
        joined.append(dict(benchmark=r['benchmark'],robin_weight=float(r['rho']),E_rel_Bgt=float(r['E_rel_Bgt_domain']),
            **{k:float(r[k]) for k in ['C_vec','C_CS','Eprime_n','Eprime_m','epsilon','sigma_J']},
            **{k:base[k] for k in ['best_sigma','best_iter','n_iters','stop_reason','wall_time_s','dtype','init_mode','metric_domain']},
            current_operator='common_fd2_full_volume_then_crop'))
    with (data/'rho_full/metrics_with_erel_bgt.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(joined[0]));writer.writeheader();writer.writerows(joined)
    selected=min((r for r in selection if r['cci_squared_term_weight']==1.),key=lambda r:r['mean_distance'])
    optima={case:min((r for r in nfgs if r['benchmark']==case),key=lambda r:r['distance']) for case in ['td1','td2','td3']}
    keys=['E_rel_Bgt_domain','C_vec','C_CS','Eprime_n','Eprime_m','epsilon','sigma_J']
    heads=[r'$E^{\rm rel}_{B^>}$',r'$C_{\rm vec}$',r'$C_{\rm CS}$',r"$E'_n$",r"$E'_m$",r'$\epsilon$',r'$\sigma_J$']
    data_rows=[[r['benchmark'].upper(),f"{float(r['rho']):g}"]+[f'{float(r[k]):.6f}' for k in keys] for r in sorted(nfgs,key=lambda r:(r['benchmark'],float(r['rho'])))]
    write_table(gen/'table_full_rho_scan.tex','lrrrrrrrr','Complete TD weight scan','tab:full_rho_scan',['Case',r'$\rho$']+heads,data_rows,'All rows use the fixed primary scoring volumes and the common FD2 current; CCI separation precedes the footprint crop.')
    write_table(gen/'table_case_rho_optima.tex','lrrrr','Case-specific minima of the diagnostic objective','tab:case_rho_optima',['Case',r'$\rho_{\rm min}$',r'$d_b$',heads[0],heads[-1]],
        [[case.upper(),f"{float(r['rho']):g}",f"{r['distance']:.5f}",f"{float(r['E_rel_Bgt_domain']):.5f}",f"{float(r['sigma_J']):.5f}"] for case,r in optima.items()],
        'The same min/max scales are used over all 33 NFGS states. These diagnostic minima are not physical optima.')
    precision=[]
    for dtype in ['float32','float64']:
        precision += [dict(r,precision=dtype) for r in json.loads((data/f'low_lou_family_rho0p5/{dtype}/metrics.json').read_text())]
    label={'low_lou_ll1':'LL1','low_lou_ll2':'LL2','low_lou_c2':'LL3'}
    write_table(gen/'table_precision_comparison.tex','llrrrrrrr','Low--Lou precision comparison at '+r'$\rho=0.5$','tab:precision_comparison',
        ['Case','Precision',heads[1],heads[3],heads[4],heads[5],heads[6],'Best','Steps'],
        [[label[r['benchmark']],'FP32' if r['precision']=='float32' else 'FP64']+[f'{float(r[k]):.6f}' for k in ['C_vec','Eprime_n','Eprime_m','epsilon','sigma_J_common_fd2']]+[str(r['best_iter']),str(r['n_iters'])] for r in sorted(precision,key=lambda r:(label[r['benchmark']],r['precision']))],
        'The same meshes and stopping rules are used. The common FD2 curl uses separate horizontal and vertical spacings. Negative normalized mean-vector-error scores are retained.')
    low=build_low_lou_cci(data,selected_rho)
    summary=dict(common_choice=selected,case_optima=optima,selection_sensitivity=selection,
        primary_rows=rows,low_lou=low,height_profile_rho=selected_rho,budgets=[r for r in budgets if r['method']==f'nfgs_rho_{selected_rho:.1f}'],
        topology=read_csv(data/'quantitative_topology/metrics.csv'),twist=read_csv(data/'twist_apex/metrics.csv'),
        helicity=read_csv(data/'six_face_helicity/metrics.csv'),q_thresholds=thresholds,
        precision_differences={k:max(abs(float(next(x for x in precision if x['benchmark']==c and x['precision']=='float32')[k])-float(next(x for x in precision if x['benchmark']==c and x['precision']=='float64')[k])) for c in label) for k in ['C_vec','Eprime_m','sigma_J_common_fd2']})
    dump(data/'evidence_summary.json',summary)
    print('Common choice:',selected,'case optima:',{k:v['rho'] for k,v in optima.items()},flush=True)


if __name__=='__main__':main()
