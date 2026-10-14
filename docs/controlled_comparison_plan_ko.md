# Published NFPT 고정 비교와 NFGS–NF2 실행·원고 개편안

개편일: 2026-10-14. 사용자 승인: NFPT 논문 메트릭과 결과는 보존하고, 기존 실험을 최대한 재사용하여 새 방침으로 큐를 이어 재개한다.

## 1. 세 가지 근거를 구분한다

1. **Published benchmark**: Yi et al. (2022) Tables 2–4의 NFPT/VFVP/SSW 점수는 원래 소수 문자열 그대로 고정한다. NFPT의 missing CCI와 현재 머신 runtime을 로컬 계산으로 채우지 않는다. 새 Table 6의 published 행을 로컬 Fortran 또는 Python 결과로 대체하지 않는다.
2. **현재 동일 입력의 신규 비교**: NFGS–NF2는 기존 canonical input, coordinate, scoring box와 공통 FD2/GS를 유지한다. 이는 실제 두 방법 사이의 controlled-input/common-evaluation 비교이며 published NFPT와 동일 프로토콜이라는 주장과는 구분한다.
3. **기존 구현·패딩 근거**: 로컬 Fortran/Python PT 출력, author-confirmed padding 연구 99건+TD3 보조 3건, NFPT 재현 검사 7건은 별도 보조 자료로 보존한다. NFPT 논문 당시 계산본으로 표시하지 않는다.

현재 입력에서 원저자 제공 TD2 bcor와 생성 reference의 차이가 확인되었다. 같은 TD 파라미터, 배열 크기, 16셀 패딩만으로 원 논문과 동일한 벤치마크를 입증할 수 없다. 원래 자료의 좌표·정규화·평가 영역·curl 코드 확인이 필요하다. 결과를 published 점수에 맞추기 위해 mask나 iteration을 사후 선택하지 않는다.

## 2. 큐 개편과 재사용

- 기존 644개 manifest와 state, pause record, bootstrap은 `policy_revisions/published_nfpt_fixed_v2`에 보존한다.
- NFGS 221개와 NF2 291개, **총 512개**를 활성 큐로 유지한다. 완료된 **25개**를 재계산하지 않는다.
- 로컬 NFPT Fortran/GPU 129개와 legacy PT 3개, 총 132개는 별도 원장으로 이동한다. 그중 완료된 2개와 실패 기록, field 및 로그를 삭제하지 않는다. 남은 작업은 주 실험의 필수 실행에서 제외한다.
- 현재 NF2 TD1 boundary weight multiplier=0.025118864315095794, seed=4 학습은 기존 PID의 optimizer/CUDA 상태를 그대로 유지해 약 2300 step에서 이어간다. 학습을 새로 시작하거나 checkpoint로 되돌리지 않는다.
- 중지된 옛 작업 배분기와 실패 복구 감시기만 종료하고, 새 배분기가 기존 학습 프로세스의 완료를 인계받는다. obsolete in-memory 큐가 이후 NFPT를 재실행하는 일을 막는다.
- 새 policy가 활성 큐에서 NFPT 실행을 거부한다. published snapshot hash가 달라지면 집계를 중단한다.
- 중지 시간은 raw 기록과 interval을 보존한 채 NF2 training/solver/end-to-end 비용에서 제외한다. inference에는 적용하지 않는다. 재시도 실행의 시간에서 이전 학습의 중지 시간을 빼지 않는다.

## 3. 수치 조건은 보존한다

별도 파일럿을 추가하지 않는다. NF2는 기존 pinned commit, bottom-only, vector-potential 표현, FP32, TF32 off, 15,000 optimizer steps, 11 boundary weights, seeds 0–4를 유지한다. 마지막 고정 예산 상태를 반환하며 reference-field metric으로 checkpoint를 선택하지 않는다. NFGS는 기존 11 rho와 native mudpack_legacy 계산 padding을 유지한다. Author-confirmed padding은 이미 완료한 별도 민감도 자료로 재사용한다. 이를 현재 큐에 몰래 덮어쓰지 않는다.

CCI는 전체 footprint에 동일 GS를 적용한 다음 scoring footprint를 자른다. sigma_J는 전체 field의 FD2 curl 후 scoring volume을 자른다. FastQSL/energy/helicity의 공통 프로토콜도 유지한다. NF2 seed 분포·고정-budget 실패·미도달·불리한 점을 모두 보고한다. TD3의 낮은 정확도나 noise 민감도를 삭제하지 않는다.

Low–Lou rho=0/0.5/1, resolution, noise, patience/threshold, FP32/64 및 CCI cutoff/L2 진단은 유지한다. 기존 Low–Lou 80×80×72와 현재 등방 80³은 다른 근거로 분리한다. 시간 측정은 warm-up 1회와 측정 5회를 유지하고, 다른 rho/seed/weight를 같은 반복 그룹으로 합치지 않는다.

## 4. 원래 NFPT 프로토콜로 맞추는 추가 단계

현재 큐의 재개는 다음 자료 대기로 막지 않는다. 원래 자료가 확보되면 별도 입력·평가 프로파일을 등록한다.

| 필요한 자료 | 확인 | 후속 작업 |
|---|---|---|
| 당시 B0와 analytic reference 배열 | 실제 bytes/SHA, 단위, 축 순서 | 같은 배열을 NFGS/NF2에 공급 |
| x/y/z coordinates | origin, spacing, endpoints, node/cell 해석 | observed/solved/scoring grid를 구분 |
| 평가 영역 인덱스와 metric 코드 | 원래 출력에서 published 3자리 점수 재현 | 동일 코드를 신규 결과에 적용 |
| 당시 field/J 원본 | 논문 계산본인지 검증 | 원본 B는 보존한 채 추가 CCI를 별도 열로 계산 가능 |
| padding fill/closure | zero/potential/analytic fill을 구분 | method-intrinsic 차이를 ledger로 명시 |

평가법만 바뀌면 저장된 field를 후처리한다. 입력·좌표·계산 도메인·padding이 바뀌면 해당 케이스만 새 ID로 재실험한다. NF2의 입력이 바뀌면 기존 optimizer를 해당 새 입력에 이어 붙이지 않고 독립 학습한다. 아직 없는 프로토콜을 검증된 것으로 표시하거나 published CCI/runtime을 만들어 넣지 않는다.

## 5. 보고서·노트북

- `published_scores_unchanged.csv`: 고정된 문헌 값, 원래 protocol 표시, CCI/runtime 미제시.
- `all_executed_scalar_metrics.csv`: NFGS–NF2 신규 실행, source/input/config SHA와 실제 protocol 표시.
- Pareto 그림: 새 NFGS–NF2 CCI–sigma_J만 표시. Published NFPT의 missing CCI는 점으로 그리지 않는다.
- Timing: NFGS–NF2 solver/training+inference/end-to-end를 분리하고 설정별 5회 통계를 산출한다. 로컬 NFPT 시간을 원 논문 solver의 현재 머신 속도로 인용하지 않는다.
- 외부 패딩·재현 검사와 보존된 로컬 PT 자료는 별도 supplement index에서 접근한다.
- `comparison_protocol_gate.json`: 현재 검증 수준과 원 NFPT 프로토콜의 미확인 항목을 기계 판독 가능한 상태로 남긴다.
- 공개 저장소에는 Fortran 소스·실행파일·빌드·실행 adapter를 포함하지 않는다. DOI와 funding은 실제 값이 확인된 뒤 추가한다.

## 6. 원고 수정 단계

### 지금 반영

영어·한국어 원고의 published NFPT 행을 그대로 보존한다. TD reference의 파라미터와 명목 격자는 공저자 제공 설정에 기반하지만 원 논문의 배열과의 bitwise 동일성은 확인되지 않았다고 범위를 명확히 한다. 로컬 archived PT 진단은 보조 appendix로 분리한다. 표의 bold는 같은 protocol의 NFGS 내부 비교에만 적용한다. TD1의 rho=0부터 존재하는 이득을 CCI 제어에 귀속하지 않고, TD2 0.986 대 0.985와 sigma_J 비용을 균형 있게 서술한다.

### 활성 큐 완료·검증 후 반영

- Results: NFGS–NF2 통제 조건 표, weight/seed Pareto, Low–Lou baseline과 CCI 진단 추가.
- Table 6: 원래 NFPT 값과 source 표시 유지. 신규 결과의 protocol을 별도로 명기한다. 원래 protocol gate가 통과되기 전에는 published 행과 신규 행 전체의 best-score bold를 쓰지 않는다.
- 귀속: 같은 NFGS 구현에서 rho=0 대비 효과를 정량화한다. GS/구현 변경에 따른 총 차이를 CCI의 효과로 합치지 않는다.
- 속도: 같은 GPU, 측정 구간, 고정 예산 및 품질 범위를 함께 보고한다. NF2 training과 inference를 분리한다. NFPT published 결과로 직접 time speedup을 만들지 않는다. 제목은 근거 검증 전 GPU-Accelerated를 유지한다.
- Discussion: sigma_J, noise, padding/resolution 의존성, seed spread, 비수렴, case별 제한을 실제 결과에 맞춰 보강한다.
- Appendix: 전체 rho, pause-corrected timing ledger, convergence/patience, resolution/noise, precision 및 cutoff/L2 자료 유지.
- 양 언어 원고의 숫자는 같은 검증된 artifact에서 생성한다. 새 결과나 speedup을 완료 전에 본문에 미리 넣지 않는다.

완료 조건은 큐 실행 완료와 과학적 근거 검토, 원고의 양 언어 수치 동기화 및 PDF 확인을 구분한다. 중간 보고의 계산 ETA는 원고 편집/외부 DOI·funding 대기를 포함하지 않는다.
