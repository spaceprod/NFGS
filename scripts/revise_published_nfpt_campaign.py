#!/usr/bin/env python3
"""Promote a paused campaign to published NFPT policy while retaining its live NF2 worker."""
from pathlib import Path
import argparse,copy,json,os,signal,subprocess,time
import psutil
from nfgs.controlled_cases import dump,sha
from nfgs.comparison_policy import POLICY_ID,ACTIVE_METHODS,classify_job
ROOT=Path(__file__).resolve().parents[1]

def prepare(root):
    root=Path(root);state=json.loads((root/'campaign_state.json').read_text());pause=json.loads((root/'pause_record.json').read_text())
    if pause['status']!='paused_by_user':raise RuntimeError('Migration requires the original user-paused campaign')
    active=[copy.deepcopy(j) for j in state['jobs'] if classify_job(j)=='active_controlled']
    excluded=[copy.deepcopy(j) for j in state['jobs'] if classify_job(j)!='active_controlled']
    if not any(j['id']==pause['current_job'] and j['method']=='NF2' for j in active):raise RuntimeError('Live NF2 worker is not retained')
    current=next(j for j in active if j['id']==pause['current_job'])
    job_path=root/'jobs'/(current['id']+'.json');spec=json.loads(job_path.read_text())
    for key in ['id','method','case_spec','settings']:
        if current[key]!=spec[key]:raise RuntimeError('Live training specification was changed: '+key)
    return state,pause,active,excluded,{'policy_id':POLICY_ID,'original_jobs':len(state['jobs']),'active_jobs':len(active),
        'reused_completed':sum(j['status']=='completed' for j in active),'archived_jobs':len(excluded),'reused_live_job':current['id'],
        'live_job_file_sha256':sha(job_path),'existing_cases':{p.name:sha(p/'case.json') for p in (root/'cases').iterdir() if (p/'case.json').exists()},
        'source_solver_sha256':sha(ROOT/'nfgs_solver.py'),'published_scores_sha256':sha(ROOT/'reference_results/published_methods.csv'),
        'training_budgets_or_active_scientific_settings_changed':False,'created_at':time.time()}

def apply(root):
    root=Path(root).resolve();state,pause,active,excluded,ledger=prepare(root)
    identities={p['pid']:p for p in pause['processes']}
    # WSL can adjust btime after host sleep. Require identical commands, stopped
    # states, relative creation times and parent links; record stable kernel ticks.
    observed=[]
    for saved in pause['processes']:
        process=psutil.Process(saved['pid'])
        if process.cmdline()!=saved['command'] or process.status()!=psutil.STATUS_STOPPED:
            raise RuntimeError('Process identity/status mismatch: '+str(saved['pid']))
        observed.append({'pid':saved['pid'],'create_time':process.create_time(),
            'offset_s':process.create_time()-saved['create_time'],'ppid':process.ppid(),
            'starttime_ticks':Path(f'/proc/{saved["pid"]}/stat').read_text().rsplit(')',1)[1].split()[19]})
    clock_offset=observed[0]['offset_s']
    if any(abs(v['offset_s']-clock_offset)>.1 for v in observed):raise RuntimeError('Nonuniform creation shift suggests PID reuse')
    worker=next(p for p in pause['processes'] if 'run_nf2_private.py' in ' '.join(p['command']))
    actual_worker=next(v for v in observed if v['pid']==worker['pid'])
    if actual_worker['ppid']!=pause['campaign_pid']:raise RuntimeError('Training parent relation changed')
    for saved in pause['processes']:
        if 'wandb-core' in ' '.join(saved['command']) and psutil.Process(saved['pid']).ppid()!=worker['pid']:raise RuntimeError('Offline logger parent changed')
    ledger['process_identity_checks']=observed;ledger['WSL_btime_clock_offset_s']=clock_offset
    revision=root/'policy_revisions'/POLICY_ID;revision.mkdir(parents=True,exist_ok=False)
    import shutil
    for name in ['plan.json','campaign_state.json','pause_record.json','bootstrap.json']:
        if (root/name).exists():shutil.copy2(root/name,revision/name)
    dump(revision/'excluded_jobs.json',excluded);dump(revision/'reuse_ledger.json',ledger)
    shutil.copy2(ROOT/'reference_results/published_methods.csv',root/'published_baseline.csv')
    policy={'schema_version':2,'policy_id':POLICY_ID,'active_methods':list(ACTIVE_METHODS),
        'new_results_protocol':'common_fd2_gs_v1','published_protocol_match_verified':False,
        'published_baseline':{'snapshot':'published_baseline.csv','sha256':ledger['published_scores_sha256'],'source':'Yi et al. 2022, Tables 2-4'},
        'local_solver_jobs':'archived in policy_revisions/'+POLICY_ID+'/excluded_jobs.json; never substitute for published rows',
        'reference_gate':{'status':'awaiting_verified_original_protocol','missing':['original reference/input coordinates','original scoring indices and metrics code']},
        'completed_external_evidence':{'author_padding':'results/padding_sensitivity/author_confirmed_20261014/report','TD3_padding_mechanism':'results/padding_sensitivity/td3_padding_mechanism_20261014','NFPT_reproduction':'results/nfpt_padding_validation/author_confirmed_20261014/report'},
        'active_NFGS_padding':'unchanged mudpack_legacy; authors padding is independent sensitivity evidence',
        'worker_budget_steps':15000,'NF2_seeds':[0,1,2,3,4],'published_NFPT_CCI_and_runtime_substitution_allowed':False}
    dump(root/'comparison_policy.json',policy)
    # The stopped dispatchers hold obsolete lists in memory. Retire only them;
    # the optimizer, CUDA context and offline logging children are untouched.
    for pid in [pause['recovery_watch_pid'],pause['campaign_pid']]:
        expected=next(v for v in observed if v['pid']==pid)['starttime_ticks']
        if Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]!=expected:raise RuntimeError('Kernel process identity changed')
        os.kill(pid,signal.SIGKILL)
    now=time.time();interval={'start':pause['paused_at']+clock_offset,'original_wall_start':pause['paused_at'],'WSL_btime_clock_offset_s':clock_offset,'end':now,'duration_s':now-pause['paused_at']-clock_offset,'reason':'user pause, then explicit authorization to resume under revised published-NFPT policy'}
    histpath=root/'pause_history.json';hist=json.loads(histpath.read_text()) if histpath.exists() else []
    hist.append({**interval,'job':pause['current_job']});dump(histpath,hist)
    intervals_path=root/'runs'/pause['current_job']/'pause_intervals.json';intervals=json.loads(intervals_path.read_text()) if intervals_path.exists() else []
    intervals.append(interval);dump(intervals_path,intervals)
    newstate={**state,'jobs':active,'status':'resuming_policy_revision','pid':None,'worker_pid':worker['pid'],'comparison_policy_id':POLICY_ID,'policy_revision_at':now}
    if clock_offset:
        newstate['original_started_at_wall']=newstate['started_at'];newstate['started_at']+=clock_offset
        for job in active:
            if job['id']==pause['current_job']:
                job['original_started_at_wall']=job['started_at'];job['started_at']+=clock_offset
    newstate['events']=list(state.get('events',[]))+[{'event':'published_nfpt_policy_revision','time':now,'reuse_ledger':str(revision/'reuse_ledger.json')}]
    dump(root/'campaign_state.json',newstate)
    oldplan=json.loads((revision/'plan.json').read_text());dump(root/'plan.json',{**oldplan,'jobs':active,'comparison_policy_id':POLICY_ID,'policy_revised_at':now,'source_initial_plan':'policy_revisions/'+POLICY_ID+'/plan.json'})
    dump(root/'bootstrap.json',{'job_id':pause['current_job'],'pid':worker['pid'],'create_time':actual_worker['create_time'],'starttime_ticks':actual_worker['starttime_ticks'],'adopt_exact_live_worker':True})
    env={**os.environ,'PYTHONPATH':str(ROOT),'PYTHONUNBUFFERED':'1'}
    log=(root/'background_published_nfpt_v2.log').open('ab')
    dispatcher=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/run_controlled_background.py'),'--root',str(root)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    watcher=subprocess.Popen([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/retry_failed_after_campaign.py'),'--root',str(root),'--campaign-pid',str(dispatcher.pid)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=(root/'recovery_published_nfpt_v2.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True,env=env)
    (ROOT/'.local/controlled/campaign_pid').write_text(str(dispatcher.pid)+'\n');(ROOT/'.local/controlled/recovery_watch_pid').write_text(str(watcher.pid)+'\n')
    pause.update(status='resumed',resumed_at=now,comparison_policy_id=POLICY_ID,new_campaign_pid=dispatcher.pid,new_recovery_watch_pid=watcher.pid,retired_dispatcher_pids=[pause['campaign_pid'],pause['recovery_watch_pid']])
    dump(root/'pause_record.json',pause)
    retired=set(pause['retired_dispatcher_pids'])
    for saved in reversed(pause['processes']):
        if saved['pid'] not in retired:
            expected=next(v for v in observed if v['pid']==saved['pid'])['starttime_ticks']
            if Path(f'/proc/{saved["pid"]}/stat').read_text().rsplit(')',1)[1].split()[19]!=expected:raise RuntimeError('Worker kernel identity changed')
            os.kill(saved['pid'],signal.SIGCONT)
    ledger.update(dispatcher_pid=dispatcher.pid,recovery_watch_pid=watcher.pid,retained_worker_pid=worker['pid'],resumed_at=now,pause_duration_s=interval['duration_s'])
    dump(revision/'migration_completed.json',ledger)
    return ledger

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--apply',action='store_true');a=p.parse_args()
    result=apply(a.root) if a.apply else prepare(a.root)[-1]
    print(json.dumps(result,indent=2))
