#!/usr/bin/env python3
"""Report live campaign progress and ETA from observed method runtimes."""
from pathlib import Path
import argparse,json,time,statistics,os,psutil
ROOT=Path(__file__).resolve().parents[1]


def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def report(root):
    root=Path(root);state=read(root/'campaign_state.json',{});jobs=state.get('jobs',[]);now=time.time();observed={}
    pause=read(root/'pause_record.json',{})
    paused=pause.get('status')=='paused_by_user'
    pause_history=read(root/'pause_history.json',[])
    paused_total=sum(float(v['duration_s']) for v in pause_history)+(now-pause['paused_at'] if paused else 0.)
    def completed_job_pause_seconds(job):return sum(float(v['duration_s']) for v in pause_history if v.get('job')==job['id'])
    for job in jobs:
        if job.get('status')=='completed':
            key=(job['method'],job['case_spec']['base']);observed.setdefault(key,[]).append(max(0.,float(job['duration_s'])-completed_job_pause_seconds(job)))
    current=next((j for j in jobs if j.get('status')=='running'),None);progress=read(root/'runs'/current['id']/'progress.json',{}) if current else {}
    if current and progress.get('training_elapsed_s') is not None:
        progress['recorded_training_elapsed_s_including_pause']=progress['training_elapsed_s']
        progress['training_elapsed_s']=max(0.,progress['training_elapsed_s']-completed_job_pause_seconds(current))
        if progress.get('phase')=='training' and progress.get('step'):
            progress['recorded_step_rate_including_pause']=progress.get('step_rate')
            progress['step_rate']=progress['step']/max(progress['training_elapsed_s'],1e-9)
    estimates={};basis={}
    for job in jobs:
        key=(job['method'],job['case_spec']['base']);values=observed.get(key,[])
        if values:
            duration=statistics.median(values);basis[str(key)]='observed_median'
        elif any(method==job['method'] for method,base in observed):
            pool=[v for (method,base),values in observed.items() if method==job['method'] for v in values]
            duration=statistics.median(pool);basis[str(key)]='observed_same_method_other_case_preliminary'
        else:duration=float(job.get('prior_duration_s',180));basis[str(key)]='initial_prior_not_measured'
        factor=float(job.get('runtime_scale',1))
        if factor!=1:duration*=factor
        estimates[job['id']]=duration
    outstanding=[j for j in jobs if j.get('status') in ('pending','running','retry')]
    remaining=sum(estimates[j['id']] for j in outstanding)
    if current:
        elapsed=now-current.get('started_at',now)-completed_job_pause_seconds(current)-(now-pause['paused_at'] if paused and pause.get('current_job')==current['id'] else 0.);remaining=max(0,remaining-min(elapsed,estimates[current['id']]))
    errors=[{'id':j['id'],'error':j.get('error')} for j in jobs if j.get('status')=='failed']
    pid=state.get('pid');alive=False
    if pid:
        try:alive=psutil.Process(pid).is_running() and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:pass
    scientific=[j for j in jobs if j.get('role','science')=='science']
    return {'status':state.get('status','not_initialized'),'pid':pid,'process_alive':alive,'root':str(root),
      'elapsed_hours':(now-state.get('started_at',now)-paused_total)/3600,'paused_hours':paused_total/3600,'currently_paused':paused,'total_jobs':len(jobs),
      'completed':sum(j.get('status')=='completed' for j in jobs),'failed':len(errors),
      'pending_or_running':len(outstanding),'scientific_jobs':len(scientific),'current_job':current['id'] if current else None,'current_progress':progress,
      'remaining_estimated_hours':remaining/3600 if jobs else None,'estimated_finish_unix':now+remaining if jobs and not paused else None,
      'ETA_basis':basis,'failed_jobs_excluded_from_ETA':len(errors),'ETA_is_preliminary':any(v!='observed_median' for v in basis.values()),
      'does_not_include_external_Zenodo_or_funding_wait':True,'failed_jobs_require_attention':errors[:10],'comparison_policy_id':state.get('comparison_policy_id'),'archived_jobs_outside_active_queue':len(read(root/'policy_revisions/published_nfpt_fixed_v2/excluded_jobs.json',[]))}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path);a=p.parse_args()
    root=a.root or Path((ROOT/'.local/controlled/active_root').read_text().strip());print(json.dumps(report(root),indent=2))


if __name__=='__main__':main()
