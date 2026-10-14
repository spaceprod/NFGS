#!/usr/bin/env python3
"""Generate a draft job matrix and count; this tool does not launch solvers."""
import argparse, hashlib, itertools, json
from pathlib import Path


def plan(config):
    rho=config['NFGS']['rho_grid'];weights=config['NF2']['boundary_weight_multipliers'];seeds=config['NF2']['seeds']
    jobs=[]
    def add(stage,case,method,**settings):
        jobs.append({'job_id':f'job_{len(jobs)+1:04d}','stage':stage,'case':case,'method':method,
                     'settings':settings,'status':'planned_not_executed'})
    for case in config['TD_cases']:
        for method in (() if config.get('comparison_policy',{}).get('published_NFPT_immutable') else ('NFPT_Fortran','NFPT_GPU')):add('E1',case,method)
        for value in rho:add('E1_E2',case,'NFGS',rho=value)
        for weight,seed in itertools.product(weights,seeds):add('E2',case,'NF2',boundary_weight_multiplier=weight,seed=seed)
    for case in config['Low_Lou_cases']:
        for method in (() if config.get('comparison_policy',{}).get('published_NFPT_immutable') else ('NFPT_Fortran','NFPT_GPU')):add('E3',case,method)
        for value in config['NFGS']['low_lou_rho_grid']:add('E3',case,'NFGS',rho=value)
        for seed in seeds:add('E3',case,'NF2',boundary_weight_multiplier=1.0,seed=seed)
    return {'status':'draft_design_not_execution_ready','config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
            'primary_job_count':len(jobs),'NF2_training_count':sum(j['method']=='NF2' for j in jobs),
            'jobs':jobs,'additional_not_counted':['pilot','default_potential_boundaries','timing_repeats','scaling','E5_sensitivity'],
            'execution_gates':config['execution_gates']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/controlled_published_nfpt_20261014.json'));p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=plan(json.loads(a.config.read_text()));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','primary_job_count','NF2_training_count','additional_not_counted')},indent=2))


if __name__=='__main__':main()
