#!/usr/bin/env python3
"""Build bilingual tables from explicitly archived metrics; no new reconstruction."""
from pathlib import Path
import argparse,csv,hashlib,json,math

ROOT=Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def table(caption,label,head,body,comment,columns):
    return '\\begin{deluxetable*}{'+columns+'}\n\\tablecaption{'+caption+'\\label{'+label+'}}\n\\tablehead{'+ '&'.join('\\colhead{'+h+'}' for h in head)+'}\n\\startdata\n'+'\\\\\n'.join(' & '.join(r) for r in body)+'\n\\enddata\n\\tablecomments{'+comment+'}\n\\end{deluxetable*}\n'


def build(output):
    source=ROOT/'reference_results';path=source/'common_diagnostics/headline_common_fd2_protocol.csv'
    data=rows(path);scan=[r for r in data if r['method'].startswith('nfgs_rho_')]
    assert len(scan)==33
    sigma=[float(r['sigma_J']) for r in scan];err=[float(r['E_rel_Bgt_domain']) for r in scan]
    lows=(min(sigma),min(err));ranges=(max(sigma)-lows[0],max(err)-lows[1])
    opt=[];body=[]
    for case in ('td1','td2','td3'):
        group=sorted((r for r in scan if r['benchmark']==case),key=lambda r:float(r['rho']))
        assert len(group)==11 and all(abs(float(r['rho'])-i/10)<1e-12 for i,r in enumerate(group))
        for r in group:
            r['selection_d']=math.hypot((float(r['sigma_J'])-lows[0])/ranges[0],(float(r['E_rel_Bgt_domain'])-lows[1])/ranges[1])
            body.append([case.upper(),f"{float(r['rho']):.1f}"]+[f'{float(r[k]):.5f}' for k in ('E_rel_Bgt_domain','C_vec','C_CS','Eprime_n','Eprime_m','epsilon','sigma_J')])
        best=min(group,key=lambda r:r['selection_d']);opt.append({k:best[k] for k in ('benchmark','rho','selection_d','E_rel_Bgt_domain','sigma_J')})
    head=['Case',r'\(\rho\)',r'\(E^{\rm rel}_{B^>}\)',r'\(C_{\rm vec}\)',r'\(C_{\rm CS}\)',r'\(E_n^\prime\)',r'\(E_m^\prime\)',r'\(\epsilon\)',r'\(\sigma_J\)']
    pairs=[]
    pairs.append(('table_full_rho_scan',table('Full TD control-weight scan','tab:full_rho_scan',head,body,'Archived final-field diagnostics on the unchanged primary scoring boxes; all current metrics use the common FD2 curl. These are the same states used in the main-text selection, not newly executed controlled comparisons.','lrrrrrrrr'),table('TD 제어 가중치 전체 스캔','tab:full_rho_scan',['사례']+head[1:],body,'기존 최종 자기장을 동일 주 평가 영역에서 진단한 값이다. 전류 지표는 공통 FD2 curl을 사용한다. 본문 선택에 사용한 상태이며 신규 통제 비교 실행값이 아니다.','lrrrrrrrr')))
    selection_body=[[r['benchmark'].upper(),f"{float(r['rho']):.1f}",f"{r['selection_d']:.5f}",f"{float(r['E_rel_Bgt_domain']):.5f}",f"{float(r['sigma_J']):.5f}"] for r in opt]
    sh=['Case',r'\(\rho_{\rm min}\)',r'\(d_b\)',r'\(E^{\rm rel}_{B^>}\)',r'\(\sigma_J\)']
    pairs.append(('table_case_rho_optima',table('Case-specific minima of the stated diagnostic objective','tab:case_rho_optima',sh,selection_body,'The shared min/max scales are fixed over all 33 NFGS states, as in the main-text objective. Individual minima need not equal the mean-distance minimum at 0.5 and are not physical optima.','lrrrr'),table('명시한 진단 목적함수의 사례별 최소점','tab:case_rho_optima',['사례']+sh[1:],selection_body,'본문 목적함수와 같이 33개 NFGS 상태 전체에서 공통 최솟값/최댓값 척도를 고정한다. 개별 최소점은 평균 거리 최소점 0.5와 다를 수 있으며 물리적 최적값이 아니다.','lrrrr')))
    precision=[];inputs=[path]
    for label,name in [('LL1','low_lou_ll1'),('LL2','low_lou_ll2'),('LL3','low_lou_c2')]:
        for precision_name in ('float32','float64'):
            p=source/f'low_lou_family_rho0p5/{precision_name}/metrics.csv';inputs.append(p)
            r=next(r for r in rows(p) if r['benchmark']==name)
            precision.append([label,precision_name.replace('float','FP')]+[f'{float(r[k]):.6f}' for k in ('C_vec','Eprime_n','Eprime_m','epsilon','sigma_J_common_fd2')]+[r['best_iter'],r['n_iters']])
    ph=['Case','Precision',r'\(C_{\rm vec}\)',r'\(E_n^\prime\)',r'\(E_m^\prime\)',r'\(\epsilon\)',r'\(\sigma_J\)','Best','Steps']
    pairs.append(('table_precision_comparison',table('Low--Lou single- and double-precision results at \\(\\rho=0.5\\)','tab:precision_comparison',ph,precision,'The same meshes, scoring boxes and stopping parameters are used; best-monitor state selection may differ. \\(\\sigma_J\\) uses the common anisotropic FD2 curl. Negative \\(E_m^\\prime\\) values are retained. Best and Steps denote the selected-state iteration and total iterations.','llrrrrrrr'),table('Low--Lou \\(\\rho=0.5\\)의 단정밀도와 배정밀도 결과','tab:precision_comparison',['사례','정밀도']+ph[2:7]+['선택','반복'],precision,'동일 격자, 평가 영역과 정지 매개변수를 사용하며 최선 모니터 상태 선택은 달라질 수 있다. \\(\\sigma_J\\)는 방향별 간격을 반영한 공통 FD2 curl을 사용한다. 음의 \\(E_m^\\prime\\)도 보존한다. 선택과 반복은 선택 상태 반복 번호와 전체 반복 횟수이다.','llrrrrrrr')))
    output.mkdir(parents=True,exist_ok=True)
    for name,en,ko in pairs:
        if name=='table_full_rho_scan':
            en=en.replace('\\tablecaption{','\\tabletypesize{\\scriptsize}\n\\tablewidth{\\textwidth}\n\\tablecaption{',1)
            ko=ko.replace('\\tablecaption{','\\tabletypesize{\\scriptsize}\n\\tablewidth{\\textwidth}\n\\tablecaption{',1)
        (output/(name+'.tex')).write_text(en);(output/(name+'_ko.tex')).write_text(ko)
    record={'status':'archived_diagnostic_report_not_new_solver_runs','case_optima':opt,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in set(inputs)}}
    (output/'comparison_report.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT/'results/controlled_design/report');a=p.parse_args();print(json.dumps(build(a.output_dir),indent=2))


if __name__=='__main__':main()
