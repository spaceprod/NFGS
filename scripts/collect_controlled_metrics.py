#!/usr/bin/env python3
"""Apply the same scalar operators, raw FastQSL protocol and six-face diagnostics."""
from pathlib import Path
import argparse,json,numpy as np
from nfgs.controlled import evaluate
from nfgs.controlled_cases import dump,sha
from nfgs.comparison_policy import read_policy
from nfgs.diagnostics.topology_metrics import q_map_metrics,endpoint_pair_metrics
from nfgs.diagnostics.volume_diagnostics import cell_center_potential,energy_diagnostics
from nfgs.diagnostics.helicity_diagnostics import relative_helicity_diagnostics
from nfgs._backend.util.fastqsl_plane import compute_bottom_qmap,compute_y0_qmap


def safe(value):
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(np.integer,np.floating)):return safe(value.item())
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def bundle(path):
    with np.load(path,allow_pickle=False) as d:return {k:np.asarray(d[k]) for k in d.files}


def get_topology(path,field):
    if path.exists():return bundle(path)
    bottom=compute_bottom_qmap(field['Bx'],field['By'],field['Bz'],step_len=1/8,tol=.1,b_smooth_sigma_cells=0)
    mid=compute_y0_qmap(field['Bx'],field['By'],field['Bz'],step_len=1/8,tol=.1,b_smooth_sigma_cells=0,include_twist=True)
    values={k:v for k,v in bottom.items() if k!='metadata'};values.update({'q_y0':mid['q'],'twist_y0':mid['twist']})
    np.savez_compressed(path,**values);dump(path.with_suffix('.json'),{'bottom':bottom['metadata'],'y0':mid['metadata']});return values


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();job=json.loads(a.job.read_text());out=a.root/'runs'/job['id'];folder=a.root/'cases'/job['case_spec']['id'];meta=json.loads((folder/'case.json').read_text())
    config={'schema_version':1,'case':meta['case'],'method':job['method'],'scoring_slices_zyx':meta['scoring_slices_zyx'],
      'CCI_cutoff_fractions':[1e-8,.01,.05],'gaussian_separation':{'mean_bz_policy':'uniform_open','horizontal_zero_mode_policy':'frozen_linear_harmonic',
       'd0_zero_mode_policy':'project','source_zero_mode_policy':'project','harmonic_residual_policy':'diagnose','mean_mode_tolerance':1e-6,'source_zero_mode_tolerance':.01,'gaussian_horizontal_operator':'spectral'}}
    dump(out/'evaluation_config.json',config)
    record=evaluate(folder/'reference.npz',out/'field.npz',folder/'boundary.npz',out/'evaluation_config.json')
    result=json.loads((out/'result.json').read_text());assert result['case_sha256']==meta['sha256'];assert result['field_sha256']==sha(out/'field.npz')
    assert record['source_sha256']['reference']==meta['sha256']['reference.npz'], 'Immutable reference hash changed'
    assert record['source_sha256']['boundary']==meta['sha256']['boundary.npz'], 'Immutable observed input hash changed'
    policy=read_policy(a.root)
    record['evidence_origin']='new_execution'
    record['evaluation_protocol']='common_fd2_gs_v1'
    record['comparison_policy_id']=policy.get('policy_id') if policy else None
    record['publication_equivalence_verified']=False
    record['controlled_acquisition_verified']=True;record['acquisition_controls']={'input_and_coordinates_sha_verified':True,'same_scoring_and_FD2_GS':True,'numerical_closures_identical':False,'environment_and_precision_differences_recorded':True}
    dump(out/'metrics.json',safe(record))
    if job['stage']!='primary':return
    isotropic=np.allclose(meta['spacing_xyz'],meta['spacing_xyz'][0],rtol=1e-8)
    if not isotropic:
        record['diagnostics_not_computed']=['FastQSL isotropic-kernel requirement','six-face isotropic backend requirement'];dump(out/'metrics.json',safe(record));return
    field=bundle(out/'field.npz');ref=bundle(folder/'reference.npz');h=meta['spacing_xyz'][0];extended={}
    # Six-face quantities use the full identical physical box, not padding or a solver's own potential.
    def energetics(b):
        bp,flux=cell_center_potential(b,h);energy=energy_diagnostics(b,h,potential=bp,flux_imbalance=flux)
        energy.update(relative_helicity_diagnostics(np.stack([b[k] for k in ['Bx','By','Bz']]),bp,h));return energy
    refenergyfile=folder/'reference_energy_helicity.json'
    if refenergyfile.exists():refenergy=json.loads(refenergyfile.read_text())
    else:refenergy=safe(energetics(ref));dump(refenergyfile,refenergy)
    extended['energy_helicity']=energetics(field);extended['reference_energy_helicity']=refenergy
    reference=get_topology(folder/'reference_topology.npz',ref);candidate=get_topology(out/'topology.npz',field)
    _,sy,sx=[slice(*s) for s in meta['scoring_slices_zyx']]
    extended['topology']={}
    for threshold in [2.,2.5,3.]:extended['topology'][f'Q_threshold_{threshold}']=q_map_metrics(reference['q'][sy,sx],candidate['q'][sy,sx],high_logq=threshold)
    extended['topology']['endpoints']=endpoint_pair_metrics(reference['start_xyz'][sy,sx],reference['end_xyz'][sy,sx],candidate['start_xyz'][sy,sx],candidate['end_xyz'][sy,sx],reference_flag_start=reference['flag_start'][sy,sx],reference_flag_end=reference['flag_end'][sy,sx],candidate_flag_start=candidate['flag_start'][sy,sx],candidate_flag_end=candidate['flag_end'][sy,sx],spacing_xyz=tuple(meta['spacing_xyz']))
    extended['topology']['twist_y0_peak_abs']=float(np.nanmax(np.abs(candidate['twist_y0'])))
    extended['topology']['reference_twist_y0_peak_abs']=float(np.nanmax(np.abs(reference['twist_y0'])))
    extended['topology']['protocol']={'raw_field_smoothing_sigma':0,'step_len_cells':1/8,'trace_tolerance':.1,'seed_plane_z_cells':.5,'common_full_physical_box':True}
    record['extended']=extended;record['diagnostics_not_computed']=[];dump(out/'metrics.json',safe(record))


if __name__=='__main__':main()
