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
        for method,marker in ([('NFGS','o'),('NF2','s')] if policy else [('NFGS','o'),('NF2','s'),('NFPT_Fortran','*'),('NFPT_GPU','x')]):
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
    text=['# Controlled comparison 실행 결과','',f"완료: {summary['completed']} / {summary['total_planned']}. 실패: {len(failures)}.",'','동일한 입력 좌표·FD2 curl·GS 연산자로 재평가한 신규 결과만 수록한다. 기존 논문의 인용 값은 대체 데이터로 사용하지 않는다.','', 'NFGS ρ=0, 0.5, 1과 NFPT를 함께 읽어 GS/구현 변화와 CCI 제어의 효과를 분리해야 한다. NFPT_GPU는 Python 기준선이며 원형 Fortran과의 수치적 동일성을 가정하지 않는다.','', '원고 반영: all_executed_scalar_metrics.csv의 primary 행으로 TD 및 Low–Lou 비교표와 전체 ρ 표를 작성한다. NF2는 5 seed 분포를 보고하고 마지막 고정 예산 상태만 사용한다. timing_mean_sd.json에서 warm-up을 제외한 5회 측정만 속도 주장에 사용한다.','', '누락·실패 또는 미달 예산이 있으면 원고에 완전한 controlled comparison이라고 쓰지 않는다. 무힘 잔차의 비용, 노이즈·정지·해상도 민감도, Fortran FP64와 GPU FP32 및 라이브러리 차이를 함께 보고한다.','', 'Low–Lou 주 비교는 등방 80³이며 기존 원고의 80×80×72는 별도 보조 집합이다. 주 표에 혼합하지 않는다.','', '완료 후 영어·한국어 원고의 제목/초록/결과/Discussion/Table 6 및 Low–Lou 표를 동일한 수치로 갱신하고 투고용 결론은 근거를 확인한 뒤 확정한다. 공개 저장소에는 NFPT Fortran 소스·실행 파일·로컬 어댑터를 포함하지 않는다.']
    if policy:
        text=['# Published NFPT 고정 방침: 신규 NFGS–NF2 실행 결과','',f"완료: {summary['completed']} / {summary['total_planned']}. 실패: {len(failures)}.",'',
        'published_scores_unchanged.csv는 Yi et al. (2022)의 수치를 원래 소수 문자열 그대로 보존한다. 로컬 NFPT의 CCI/시간을 그 행에 결합하지 않는다.',
        'all_executed_scalar_metrics.csv와 Pareto는 현재 동일 canonical 입력·좌표·공통 FD2/GS로 얻은 NFGS–NF2 신규 실행만 포함한다. 원 NFPT 논문 프로토콜과의 동일성은 아직 확인되지 않았다.',
        'Table 6은 published NFPT 행을 유지한다. 원래 reference/좌표/mask/curl 코드가 검증되기 전에는 본 캠페인의 신규 결과를 동일 조건의 NFPT 대비 순위나 CCI 제어의 인과 증거로 쓰지 않는다.',
        'CCI 제어 효과는 같은 NFGS 설정에서 rho=0 대비 변화로, NFGS–NF2 차이는 같은 실제 입력과 평가 연산자의 method comparison으로 보고한다.',
        '속도 주 비교는 NFGS–NF2이며, 고정 학습 예산의 training+inference 비용과 warm-up 제외 5회 측정 분포를 각각 표기한다. 설정이 다른 timing 실행을 한 그룹으로 합치지 않는다.',
        '기존 로컬 PT 자료는 root/policy_revisions/ 아래 제외 작업 원장을 통해 보존한다. author-confirmed padding 99건과 TD3 보조 3건, NFPT 재현 검사 7건도 별도 근거로 재사용한다.',
        'Low–Lou rho=0/0.5/1, noise, resolution, stopping, precision 및 CCI cutoff/L2는 유지하며, 불리한 결과·비수렴·미달 예산은 그대로 보고한다.',
        '영어/한글 원고의 정량 갱신은 신규 결과가 완료·검증된 뒤 한다. 원고에 NF2 결과나 speedup을 미리 만들어 넣지 않는다. Fortran 자료는 공개하지 않는다.','']
        dump(destination/'comparison_protocol_gate.json',{'published_NFPT':'fixed_original_scores','new_results':'canonical_common_fd2_gs_v1','original_protocol_verified':False,'missing_evidence':['verified original arrays/coordinates','original scoring indices','original sigma_J code'],'local_CCI_not_assigned_to_published_runs':True})
    if failures:text.extend(['','## 실패 작업',*['- '+f['job']+': '+str(f['error']) for f in failures]])
    (destination/'manuscript_update_evidence_ko.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
