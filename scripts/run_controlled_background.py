#!/usr/bin/env python3
"""Detached, resumable serial controlled campaign with durable state and ETA."""
from pathlib import Path
import argparse,json,time,subprocess,os,sys,traceback,shutil,hashlib
import psutil
from nfgs.controlled_cases import prepare_case,dump
from nfgs.comparison_policy import read_policy,require_allowed_job,normalize_pause_timings
ROOT=Path(__file__).resolve().parents[1]
CORE=ROOT/'.venv/bin/python'


def active(pid):
    try:return psutil.Process(pid).is_running() and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False


def execute(command,log,env,heartbeat=None):
    with log.open('a') as stream:
        stream.write('\nCOMMAND '+json.dumps([str(v) for v in command])+'\n');stream.flush()
        p=subprocess.Popen([str(v) for v in command],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        while p.poll() is None:
            if heartbeat:heartbeat(p.pid)
            time.sleep(5)
        return p.returncode


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--retry-failed',action='store_true');a=p.parse_args();root=a.root.resolve();plan=json.loads((root/'plan.json').read_text())
    statepath=root/'campaign_state.json';state=json.loads(statepath.read_text()) if statepath.exists() else {'started_at':time.time(),'jobs':plan['jobs'],'events':[]}
    if a.retry_failed:
        for j in state['jobs']:
            if j.get('status')=='failed':j['status']='pending';j['retry_of_previous_failure']=True
    policy=read_policy(root)
    state.update(pid=os.getpid(),status='running');dump(statepath,state)
    def heartbeat(pid):
        state['worker_pid']=pid;state['updated_at']=time.time();dump(statepath,state)
    env=dict(os.environ,PYTHONPATH=str(ROOT),MPLCONFIGDIR=str(ROOT/'.local/matplotlib'),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',CUDA_VISIBLE_DEVICES='0',WANDB_MODE='offline',WANDB_SILENT='true',WANDB_DISABLE_CODE='true')
    bootstrap=json.loads((root/'bootstrap.json').read_text()) if (root/'bootstrap.json').exists() else {}
    for job in state['jobs']:
        if job.get('status')=='completed':continue
        if job.get('status')=='failed':continue
        path=root/'jobs'/(job['id']+'.json');out=root/'runs'/job['id'];out.mkdir(parents=True,exist_ok=True)
        while shutil.disk_usage(root).free<8*1024**3:
            state['status']='waiting_for_disk_space';state['updated_at']=time.time();dump(statepath,state);time.sleep(30)
        adopting=bootstrap.get('job_id')==job['id'] and not (out/'result.json').exists() and active(bootstrap.get('pid',0))
        job.update(status='running',started_at=job.get('started_at',time.time()) if adopting else time.time());state['status']='running';state['updated_at']=time.time();dump(statepath,state)
        try:
            require_allowed_job(job,policy)
            prepare_case(root,job['case_spec'])
            if adopting:
                if bootstrap.get('starttime_ticks') is not None:
                    if Path(f'/proc/{bootstrap["pid"]}/stat').read_text().rsplit(')',1)[1].split()[19]!=bootstrap['starttime_ticks']:raise RuntimeError('Bootstrap PID was reused')
                elif bootstrap.get('create_time') is not None and abs(psutil.Process(bootstrap['pid']).create_time()-bootstrap['create_time'])>.1:raise RuntimeError('Bootstrap PID was reused')
                while active(bootstrap.get('pid',0)):
                    heartbeat(bootstrap['pid']);time.sleep(5)
                if not (out/'result.json').exists():raise RuntimeError('Adopted live worker exited without a result; state preserved for retry')
            if not (out/'result.json').exists():
                if job['method']=='NFPT_Fortran':command=[CORE,ROOT/'.local/controlled/run_nfpt_private.py','--job',path,'--root',root]
                else:command=[CORE,ROOT/'scripts/run_controlled_native.py','--job',path,'--root',root]
                code=execute(command,out/'worker.log',env,heartbeat)
                if code:raise RuntimeError(f'Solver exited {code}; see {out}/worker.log')
            normalize_pause_timings(root,job)
            code=execute([CORE,ROOT/'scripts/collect_controlled_metrics.py','--job',path,'--root',root],out/'evaluation.log',env,heartbeat)
            if code:raise RuntimeError(f'Common evaluation exited {code}; scalar metrics may exist; see {out}/evaluation.log')
            job.pop('error',None);job.update(status='completed',finished_at=time.time(),duration_s=time.time()-job['started_at']);state['events'].append({'job':job['id'],'event':'completed','time':time.time()})
        except Exception as error:
            job.update(status='failed',finished_at=time.time(),error=str(error),duration_s=time.time()-job['started_at']);state['events'].append({'job':job['id'],'event':'failed','error':str(error),'time':time.time()})
            with (root/'failures.log').open('a') as log:log.write('\n'+job['id']+'\n'+traceback.format_exc())
        state['worker_pid']=None;state['updated_at']=time.time();dump(statepath,state)
    state['status']='experiments_complete' if not any(j['status']=='failed' for j in state['jobs']) else 'experiments_finished_with_failures';state['finished_at']=time.time();dump(statepath,state)
    execute([CORE,ROOT/'scripts/build_live_comparison_report.py','--root',root],root/'report.log',env)


if __name__=='__main__':main()
