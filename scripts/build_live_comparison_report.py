#!/usr/bin/env python3
"""Aggregate executed controlled runs; never substitute archived or missing rows."""
from pathlib import Path
import argparse,json,csv,math,statistics,time
import numpy as np
from nfgs.controlled_cases import dump
from nfgs.comparison_policy import read_policy,published_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    state=json.loads((root/'campaign_state.json').read_text());destination=root/'report';destination.mkdir(exist_ok=True)
    policy=read_policy(root)
    historical=published_rows(root)
    if historical:
        with (destination/'published_scores_unchanged.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,list(historical[0]));writer.writeheader();writer.writerows(historical)
    if policy:
        repository=Path(__file__).resolve().parents[1];supplements=[]
        for name,relative in policy.get('completed_external_evidence',{}).items():
            path=repository/relative;entry={'name':name,'source':str(path),'exists':path.exists(),'evidence_role':'separate_supplement','included_in_new_primary_scores':False}
            for candidate in [path/'completion_manifest.json',path/'state.json']:
                if candidate.exists():entry['manifest']=json.loads(candidate.read_text());break
            supplements.append(entry)
        excluded_file=root/'policy_revisions/published_nfpt_fixed_v2/excluded_jobs.json'
        if excluded_file.exists():
            excluded=json.loads(excluded_file.read_text())
            supplements.append({'name':'original_local_PT_jobs','preserved_manifest':str(excluded_file),'preserved_jobs':len(excluded),'completed_preserved':sum(j['status']=='completed' for j in excluded),'included_in_new_primary_scores':False})
        dump(destination/'supplementary_evidence_index.json',supplements)
    rows=[];extended=[]
    for job in state['jobs']:
        if job['method'] not in ['NFGS','NFPT_Fortran','NFPT_GPU','NFGS_legacy']:continue
        out=root/'runs'/job['id']
        if not (out/'metrics.json').exists() or not (out/'result.json').exists():continue
        result=json.loads((out/'result.json').read_text());record=json.loads((out/'metrics.json').read_text())
        row={'id':job['id'],'status':job['status'],'stage':job['stage'],'role':job['role'],'case':job['case_spec']['id'],'base_case':job['case_spec']['base'],'method':job['method'],**job['settings']}
        row.update(evidence_origin='new_execution',evaluation_protocol='common_fd2_gs_v1',publication_equivalence_verified=False)
        row.update({k:v for k,v in record['metrics'].items() if isinstance(v,(int,float,str)) or v is None})
        for k in ['solver_only_s','training_s','inference_s','end_to_end_s','preparation_s','iterations','best_iteration','native_stop_reason','precision','dtype','torch_version']:
            row[k]=result.get(k)
        for cutoff,values in record['metrics'].get('CCI_cutoff_results',{}).items():
            for name,v in values.items():row[name+'_cutoff_'+cutoff]=v
        rows.append(row);extended.append({'job':job,'metrics':record,'result':result})
    columns=sorted({key for r in rows for key in r})
    with (destination/'all_executed_scalar_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,columns);writer.writeheader();writer.writerows(rows)
    dump(destination/'all_executed_diagnostics.json',extended)
    groups={}
    for r in rows:
        if r['status']=='completed' and r['role']=='timing':groups.setdefault((r['case'],r['method'],json.dumps({k:r[k] for k in ['rho','seed','boundary_weight_multiplier','explicit_padding_cells'] if k in r},sort_keys=True)),[]).append(r)
    timing=[]
    for (case,method,settings_json),values in groups.items():
        summary={'case':case,'method':method,'settings':json.loads(settings_json),'measured_repeats':len(values),'required_repeats':5,'complete':len(values)==5}
        for key in ['solver_only_s','training_s','inference_s','end_to_end_s']:
            numbers=[float(r[key]) for r in values if r.get(key) is not None]
            if numbers:summary[key+'_mean']=statistics.mean(numbers);summary[key+'_sd']=statistics.stdev(numbers) if len(numbers)>1 else None
        timing.append(summary)
    dump(destination/'timing_mean_sd.json',timing)
    # Raw paired points: no method is declared a universal winner and no metric count is causal evidence.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary=[r for r in rows if r['stage']=='primary' and r['status']=='completed']
    for case in sorted({r['case'] for r in primary}):
        fig,ax=plt.subplots(figsize=(6,4.5))
        for method,marker in ([('NFGS','o')] if policy else [('NFGS','o'),('NFPT_Fortran','*'),('NFPT_GPU','x')]):
            data=[r for r in primary if r['case']==case and r['method']==method]
            valid=[r for r in data if r.get('E_rel_Bgt') is not None and r.get('sigma_J') is not None]
            if not valid:continue
            ax.scatter([r['E_rel_Bgt'] for r in valid],[r['sigma_J'] for r in valid],label=method,marker=marker,alpha=.8)
            if method=='NFGS':
                for r in valid:ax.annotate(str(r['rho']),(r['E_rel_Bgt'],r['sigma_J']),fontsize=8)
        ax.set(xlabel='CCI relative error (actual input)',ylabel=r'$\sigma_J$',title=case);ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(destination/(case+'_CCI_force_free.png'),dpi=180);plt.close(fig)
    failures=[{'job':j['id'],'error':j.get('error')} for j in state['jobs'] if j['status']=='failed']
    summary={'created_at':time.time(),'total_planned':len(state['jobs']),'completed':sum(j['status']=='completed' for j in state['jobs']),'failed':failures,'reported_executed_rows':len(rows),'complete_for_manuscript':all(j['status']=='completed' for j in state['jobs']), 'scientific_claims_require_review':True,'comparison_policy_id':policy.get('policy_id') if policy else None,'published_scores_preserved':bool(historical),'published_NFPT_CCI_or_runtime_substituted':False,'original_publication_protocol_verified':False}
    dump(destination/'report_manifest.json',summary)
    text=['# Classical comparison 실행 결과','',
          '신규 실행은 공통 FD2/GS 평가 결과이며 출판 NFPT 수치를 대체하지 않는다.',
          'CCI 제어의 효과는 같은 NFGS 설정에서 rho=0 대비 변화로 해석한다.',
          '방법별 경계, 정밀도, 계산 영역 및 시간 측정 차이를 함께 보고한다.',
          'NFPT Fortran 소스와 실행 파일은 공개하지 않는다.']
    if failures:text.extend(['','## 실패 작업',*['- '+f['job']+': '+str(f['error']) for f in failures]])
    (destination/'manuscript_update_evidence_ko.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
