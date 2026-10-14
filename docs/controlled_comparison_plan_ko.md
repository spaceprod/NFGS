# NFGS–NFPT–NF2 통제 비교와 원고 보강 실행 설계

상태: 설계 및 공통 평가 기반 마련. 이 문서의 신규 NF2/NFPT 실험 결과는 아직 존재하지 않는다. 기존 결과를 신규 controlled 결과로 표시하지 않는다.

## 1. 검증할 주장과 즉시 교정

연구 질문은 (i) 동일 입력·표본·평가 연산자에서 CCI–force-freeness 절충과 참조장 복원이 어떻게 다른가, (ii) 같은 시스템에서 실제 solution time은 얼마인가, (iii) 제어 손잡이를 바꾸면 경계 충실도가 얼마나 변하는가이다. 전체 지표의 승수나 한 방법의 보편적 우월성을 주 주장으로 삼지 않는다.

기존 자료로 먼저 다음을 반영한다.

- Table 6에 NFGS ρ=0과 0.5를 함께 넣고, published 값과 보관 자기장의 재평가를 서로 다른 행으로 표시한다. TD1의 이미 존재하는 ρ=0 이득을 CCI 제어에 귀속하지 않는다. TD2의 Cvec 0.986 대 0.985는 해상도·반복 변동을 검증하기 전 실질적 우위로 주장하지 않는다.
- Iyer et al. (2026)이 관측 복원 방법 간 CCI 보존과 force-freeness의 차이를 보였다는 것으로 동기를 한정한다. 이번 연구가 모든 기존 방법의 CCI 손실을 입증했다고 쓰지 않는다.
- TD1에서 ρ=0→0.5의 Cvec 증가 약 0.002와 σJ 증가 0.092→0.212를 함께 보고한다. 3D field agreement와 force-freeness만으로는 0.5의 비용이 정당화되지 않는다. CCI 오차 감소는 별도 이득이므로 모든 목적에서의 ‘순손실’이라고 단정하지 않는다.
- 동일한 기존 전역 정규화 기준에서 사례별 최소점은 TD1/TD2/TD3 각각 ρ=0.3/0.6/0.7이다. 이는 물리적 최적값이 아니다. 공통 가중치 0.5와 구별한다.
- 전체 11점 표와 실제 FP32/FP64 차이를 부록에 제시한다. ‘무시할 만함’ 대신 지표별 차이를 보고한다.
- 속도 상대 비교가 없으므로 현재 제목은 GPU-Accelerated로 낮춘다. E4 완료 이후 Fast 복구 여부를 판정한다.

## 2. 사전 확인 결과와 실행 전 gate

NF2 공식 저장소: https://github.com/RobertJaro/NF2
검토 commit: `e7b3ae97dcb37161aea6cd9510d9235de396270a` (현재 공개 구현, 2023 논문의 원래 실행 버전과 동일하다고 간주하지 않음).

`nf2/loader/cartesian.py`는 `potential_boundary: {type: none}`를 지원한다. `nf2/loader/cartesian_datasets.py`의 numpy 입력은 `bx/by/bz`를 읽으며, `MapDataset`는 **x,y** 배열 및 실제 좌표를 사용한다. NFGS의 **z,y,x** 자료를 명시적으로 전치해야 한다. 검토한 `analytical_case2.yaml`은 `boundary: full`이므로 주 비교에 그대로 쓰지 않는다. reference volume은 학습·early stopping·loss 선택에 전달하지 않는다. 비교 모델은 이 commit의 명시적 vector-potential/SIREN 구성으로 고정하고, 이후 upstream 변경을 따라가지 않는다.

NFPT: 기존 드라이버가 가리키는 경로는 사라졌지만 확장 검색에서 `Archive/NFPTgpu/nfpt_fortran/nfpt`의 원본 소스를 찾았다. `nfpt_1.0.2.f90`와 `nfptpack_1.0.0.f90`의 체크섬을 config/provenance에 기록한다. 비공개 `.local`의 로컬 빌드 설정이 `NFPT_SOURCE_ROOT`로 지정한 외부 소스를 편집하지 않고 빌드한다. **사용자 지시에 따라 NFPT Fortran 소스·실행파일·로컬 빌드 스크립트는 공개 저장소, Release 및 Zenodo에 포함하지 않는다.** 소스의 `real(kind=8)`은 native FP64를 뜻한다. 현재 소스가 실제 비교 실행에 쓴 revision인지, 입력/폐쇄/전류 보정/정지 세팅은 공저자 Sibaek Yi와 확인한다. 공개할 것은 설정, 체크섬, 평가 코드와 허용된 비교 결과이며 Fortran 솔버는 공개하지 않는다. 원본 Fortran 솔버는 비공개 외부 의존성이며, 해당 비교의 solver 실행까지는 공개 자료만으로 재현할 수 없음을 명시한다. GPU Python NFPT는 별도의 implementation이며 원래 Fortran과 자동으로 동일하다고 표시하지 않는다.

추가 확인: funding 실제 기관/과제번호, CPU 모델·OpenMP thread 수, NF2 별도 환경, Zenodo 계정/보존 DOI. 확인되지 않은 정보는 원고에 만들어 넣지 않는다.

## 3. 통제 조건과 두 비교 수준

| 항목 | 기본 프로토콜 | 반드시 기록할 차이 |
|---|---|---|
| 입력 | 한 번 생성한 Bx,By,Bz 및 동일한 noise realization, SHA-256 고정 | NFPT/NFGS 내부 Jz 보정과 endpoint projection 전후 오차 |
| 학습/구동 정보 | bottom-only; 참조장의 side/top 정보 금지 | 각 방법의 side/top closure는 다름 |
| strict NFPT ablation | Fortran이 실제 지원하는 계산 격자와 원점을 먼저 확정, GPU NFPT와 NFGS에도 동일 적용 | 기존 NFGS symmetric padding의 짝수 격자를 Fortran에 억지로 적용하지 않음 |
| cross-method sampling | 같은 실제 x,y,z 좌표; curl은 full B에서 계산 후 같은 scoring box로 crop | 보간은 본 비교에서 허용하지 않음; 필요하면 별도 interpolation-sensitivity 결과 |
| 기존 원고와 연결 | 현 원고 grid/scoring도 별도 재현 | strict-grid 결과와 기존 grid 결과를 섞지 않음 |
| 정밀도 | NFGS/NF2 FP32 저장; 공통 진단 FP64 | Fortran native precision 확인 후 별도 보고; 모두 FP32라고 허위 표기하지 않음 |
| 시스템 | RTX 4070 Ti SUPER 한 장; 직렬 실행, CPU 모델·threads·software 기록 | CPU/GPU 차이를 algorithm speedup과 구별 |
| 불확실성 | NF2 seeds 0–4; deterministic 방법 반복 timing 5회 | seed spread와 paired method difference를 구별 |

P0의 계약은 `Bx,By,Bz`(z,y,x), 실제 x/y/z 좌표, length/field unit, observation boundary 및 immutable case metadata이다. 출력 shape만 같아도 원점·단위가 다르면 평가를 거부한다. 광구 reflection은 display-only이며 scoring에 적용하지 않는다.

CCI는 **전체 원래 footprint에 GS를 적용한 후** scoring footprint를 자른다. crop한 경계에 먼저 GS를 적용하지 않는다. σJ는 동일 FD2 curl, full-volume 후 crop. vector metrics/energy ratio도 같은 evaluator를 쓴다. FastQSL·twist·six-face energy/helicity는 모두 같은 선택 state와 seed/threshold/gauge를 쓰는 후속 backend이며 P0 scalar evaluator가 완료됐다는 이유만으로 해당 진단까지 완료로 표시하지 않는다. Low–Lou의 anisotropic spacing을 모든 potential/helicity backend가 처리하는지 별도 gate로 검증한다.

## 4. 실험 행렬

### E1 — NFPT 재실행과 귀속 분리 (최우선)

TD1–3마다 Fortran NFPT, GPU NFPT baseline, NFGS ρ=0/0.5/1을 동일 입력·strict calculation-grid·scoring으로 실행한다. initialization, bottom Jz balancing, curl/source operators, zero mode, top/side closures 및 stopping 차이를 ledger로 남긴다. Fortran↔GPU 차이는 implementation, GPU NFPT↔GS-only는 GS partition/initialization, GS-only↔mixed row는 **동일 구현 내** CCI 조건의 효과로 구분한다. ρ=0 결과가 NFPT와 비슷하다는 사실만으로 인과 귀속이 증명되지는 않는다.

가능한 차이 하나만 바꾸는 ablation을 만들지 못하면 비교를 controlled-input performance comparison이라고 쓰고 causal isolation이라고 부르지 않는다. published NFPT 숫자는 부록에 남기고 새 Table 6과 섞지 않는다. 모든 returned-state의 native monitor, reported σJ, CCI error 및 state-change/PDE/mixed-row residual을 함께 저장한다.

### E2 — NF2 Pareto 비교

주 비교는 bottom-only. potential side/top을 쓰는 upstream 기본 boundary 구성은 별도 supplementary 행으로 표시한다(관측에서 유도된 경계임을 밝힘). 기본 loss 가중치는 상수로 고정하고 boundary weight multiplier `10**linspace(-2,2,11)`의 11점, seeds 0–4를 사용한다. NFGS 역시 11개 ρ를 탐색한다. boundary loss는 CCI slope와 동일한 constraint가 아니라 전체 벡터 충실도의 간접 손잡이이므로 NF2의 CCI가 단조 개선한다고 가정하지 않는다. 가중치별 mean±seed spread 및 개별 seed 점을 표시하며 CCI–σJ plane에서 nondominated states를 구한다.

최신 pinned analytical template의 기준은 boundary=1, force_free=1e-3, data iterations=1000, epochs=15(총 optimizer steps는 실제 dataloader/trainer에서 검증), learning rate 5e-4→5e-5(1e5-step decay), vector-potential/SIREN 256 units×8 layers, w0_init=5, w0=1이다. pinned NF2의 default model/loss normalization을 모두 resolved config에 저장한다. FP32 비교의 matmul precision은 highest로 고정하여 TF32 사용을 제한하고 실제 backend flag도 기록한다. 이것은 Jarolim (2023)의 unpublished ‘정확한 원래 설정’이라고 부르지 않는다. 기본 single case에서 native duration·memory·step count를 먼저 확인하고 **scientific execution 전에** 예산 및 학습 step 수를 확정한다. checkpoint는 마지막 고정-budget state이며 ref Cvec로 선택하지 않는다. train/validation에는 coronal reference field를 전달하지 않는다.

최대 TD 학습 수는 3×11×5=165회다. 3×5×5는 75회이며 45회가 아니다. 파일럿 1회 wall time을 t시간이라고 하면 TD 비용은 165t GPU시간. 부족하면 결과를 보기 전에 balanced 3 settings×3 seeds로 변경하고 NFGS도 main tuning comparison은 동일 3 settings(ρ=0/0.5/1)로 제한한다. 기존 11점 NFGS scan은 supplementary로 유지한다. pilot의 참조장 점수는 축소 여부 판단에 사용하지 않는다.

### E3 — Low–Lou 기준선

LL1–3에 NFGS ρ=0/0.5/1, NFPT CPU/GPU, NF2 bottom-only 기본값×5 seeds. NF2 Low–Lou full 11-point sweep는 확장 실험으로 따로 표시한다. 기존 80×80×72와 central64×64×bottom64, 각 방향 spacing 유지. Erel(pixel), Erel(L2), Cvec, CCS, E'n, E'm, energy ratio, σJ, divergence를 모두 보고한다. 기존 Schrijver/Jarolim 결과는 six-face 조건·mesh 차이를 명시한 context일 뿐 controlled 행으로 사용하지 않는다.

### E4 — 속도와 scaling

warm-up 1회 + 측정 5회, 모든 GPU timing 구간 앞뒤 `cuda.synchronize()`. training/inference/solver-only/end-to-end/export/diagnostics를 구분한다. end-to-end는 prepared identical input을 읽는 시점부터 동일 output tensor가 완성되는 시점까지(수학적으로 같은 전처리 포함), dataset generation·최종 공통평가·그림은 별도. cold context startup 포함 시간도 별도 보고한다. peak allocated/reserved GPU memory와 process CPU RSS를 기록한다.

NF2는 매 case 새 학습이며 training+동일 grid inference가 reconstruction time이다. seed 5개의 비용을 보고하되 loss-weight scan 전체 비용도 따로 합산한다. 짧은 GPU primitive warm-up과 실제 학습 warm-up을 구분한다. 반복으로 새 학습을 수행할 경우 예산에 5회가 추가됨을 명시한다. 후보 예산안은 main default의 5-seed 학습을 speed distribution으로 쓰고, matched 5-repeat timing은 대표 case/seed에 별도 수행하는 것이다.

time-to-quality는 사전 고정한 복수 (CCI,σJ,Cvec) 허용 영역에 **동시에 처음 진입한 checkpoint**의 시간. 미도달은 censored로 보고하며 최종값에 맞춰 threshold를 바꾸지 않는다. threshold는 아직 확정 전이므로 time-to-quality 주장은 gate 전까지 원고에 넣지 않는다. Fast는 동일 품질/비슷한 지표 수준에서의 유의한 solution-time 이득과 scaling이 확인될 때만 복구한다. CPU Fortran vs GPU 비교만으로 알고리즘 자체의 speedup을 주장하지 않는다.

Scaling은 TD1 같은 물리적 도메인, 간격 factor=2,1,0.5의 3개 + 중간 factor 1.5의 4개 후보. exact endpoints를 유지하는 node counts와 native grid feasibility를 사전 확정한다. Fourier/collocation resolution 및 samples당 비용을 같다고 가정하지 않는다. 고해상도 OOM도 실제 limit로 보고한다.

### E5 — 한계와 민감도

- 해상도: TD1 3개 간격×11ρ, physical scoring box 고정. NF2는 같은 checkpoint를 다른 grid에서 읽는 **sampling** 실험과 새 collocation/training 해상도 실험을 구별한다. γ=ρ/[(1−ρ)Δz]를 같게 유지하는 scan도 비교한다. endpoints ρ=0,1은 따로 처리한다. 작은 변화에서 ρ'/[1−ρ']=Δz'/Δz·ρ/[1−ρ]. 단순히 ρ∝1/Δz라고 쓰지 않는다.
- Noise: dimensionless synthetic boundary σz=0/0.005/0.02×B_rms 및 σh=0/0.02/0.10×B_rms, paired seeds 0–4. 세 방법에 동일 배열·전류 계산 적용. 실측 HMI uncertainty라고 부르지 않음; 실제 G 단위 테스트는 B unit calibration/HMI uncertainty 출처를 확인한 뒤 추가. CCI fidelity-to-noisy-input와 fidelity-to-clean-input 모두 저장하고 Cvec는 clean 3D reference에 대해 평가. conservation correction이 noise를 제거/변형하면 전후 입력도 저장.
- stopping: ρ=0.5의 TD1–3에서 patience20/40/80와 monitor thresholds0.1×/1×/10×. maximum budget3000 유지. native monitor·common σJ·Erel·relative state change를 checkpoint 시간에 기록; 매 step GS 평가 overhead는 instrumented diagnostic run에만 포함하고 speed run과 구별. TD3 87회 종료를 수학적 수렴이라고 쓰지 않는다.
- 작은 CCI 분모: 동일 기존 footprint에서 threshold1e-8/0.01/0.05×local max와 retained-pixel count, plus 전체 footprint L2 normalized error. 기준이 fixed reference/input field에 달려야 candidate에 따라 pixel 집합이 바뀌지 않는다. L2의 분모 0이면 undefined/NA를 반환한다.
- topology 이상값: TD3ρ=0.1의 endpoint RMSE187.5Mm는 fieldline exit-face 이동, valid-trace mask 변화, seed matching, endpoint error 분포를 함께 조사. reference/모든 후보에 공통인 valid seeds에서 다시 계산하고 원래 pairwise-mask 값도 보존. 원인은 확인 전 단정하지 않는다.
- padding: 현재 축별 크기 선택 규칙과 offset을 공개하고 strict legacy vs symmetric 및 확대 buffer에서 TD1/TD2 대표ρ 비교. side closure와 padding을 동시에 바꾸지 않는다.
- precision: FP32/64 실제 metric 및 차이, chosen state/iteration/monitor 함께 표기. negative E'm도 삭제하지 않는다.

## 5. 단계·산출물·완료 조건

| 단계 | 산출물 | 원고 반영 전 완료 조건 |
|---|---|---|
| P0 | `evaluate.py`, immutable case bundle/schema, independent analytic tests | 모든 방법에 동일 curl·GS·mask, 좌표/단위 mismatch 거부; input/output hashes |
| P1 | NF2 commit+resolved config+환경, NFPT source/exe manifest | bottom-only 학습 dataset inspection, NF2 coordinate round-trip, Fortran actual grid/precision 확인 |
| P2 | strict E1 metrics/ablation ledger | 같은 입력·좌표 및 전처리, GPU NFPT↔Fortran 차이 기록; 새 Table6 근거 |
| P3 | E2/E3 field/checkpoints, seed summaries, Pareto | 사전 freeze된 설정/예산; failed seeds 포함; LL endpoints/CCI 포함 |
| P4 | timing JSONL,5-repeat mean±SD,scaling | warm-up/sync/timer scope 및 quality threshold 통과; CPU threads 명시 |
| P5 | sensitivity tables/figures | fixed scoring physics; noise pairing; no fabricated convergence/causality |
| P6 | 영어·한글 원고, reproducibility notebook, DOI-backed release | table rows와field/config hashes 직접 연결; bibliography/fig order/bold/PDF 시각 검증 |

공식 execution config는 현재 `status: draft_design`이다. `plan_controlled_comparison.py`는 job manifest와 예상 run 수를 생성하는 **설계 도구**이며 solver launch를 흉내 내지 않는다. 다음 실행 담당자는 P0/P1을 완성하고 draft를 freeze한 뒤 신규 run을 시작한다. 현재 제공된 evaluator는 scalar field/CCI 공통 평가를 구현한다; topology/energy/helicity와 method adapters는 미완료 gate로 명시한다.

## 6. 원고 변경 위치와 결과 판정

| 원고 | 지금 반영 | 신규 근거 후 반영 |
|---|---|---|
| 제목·초록 | GPU-Accelerated; Iyer 범위 한정; TD1 force-free 비용·noise 미검증 | E4 통과 시 Fast; NF2/NFPT controlled 위치와 실제 수치 |
| Introduction | NLFFF review 인용; CCI 개선의 인과 과장 제거 | boundary-loss vs mixed-row 독립 질문과 pinned implementations |
| Sec3 controlled comparison | archived 자료는 따로 표시 | 통제 조건 표, source/precision/closures, matched-input ablation |
| Table6 | NFGS0/0.5와published/archived 구분 | fresh common-evaluator Table, 인용값은 appendix |
| Pareto figure | 기존NFGS33점·case minima 보존 | NF2 weight×seed cloud/curves, NFPTCPU/GPU points; dominated cases 포함 |
| Low–Lou | 현재0.5만으로CCI 기여 입증 불가 명시 | ρ0/0.5/1 및방법별CCI/3D metric |
| runtime | 현 singles 불확실성; padding rule | solver/e2e/train/inference 반복통계와scaling |
| Discussion | TD1 작은3D 이득 대비큰σJ 비용, local minima와common0.5 구분 | 불리한noise/FF 결과도보고; 적용용도별권고는결과가뒷받침할때만 |
| Appendix | fullρ, precision, case optimum, padding | convergence/patience,resolution/noise,L2/cutoff,TD3exit-face 분석 |
| Data/code | 실제GitHub 유지; public script명 | version DOI+config/seeds/checkpoints/rawtiming,실제funding |

NF2가 CCI와σJ 모두 우수하면 그 사실을 중심에 쓰며 NFGS 속도도 별도 검증된 경우만 장점으로 보고한다. NFGS가 빨라도 정확도가 크게 낮으면 ‘동일 품질 speedup’이라고 쓰지 않는다. noisy boundary에서 NFGS 손실이 크면 robustness 제한과 관측 적용 전 noise treatment 필요성을 강화한다. Pareto domination, missing/censored runs, seed variability를 숨기지 않는다.

## 7. 배포·라이선스·행정

NFGS 자체 BSD-3, FastQSL CC BY-NC-SA4.0(별도 subtree·수정 소스/고지 보존), NF2 GPL3(별도 환경/프로세스·NPZ 파일 교환, NF2 source를 BSD로 재배포하지 않음). 전체 repository가 단일 BSD 라이선스라고 표시하지 않는다. FastQSL 파생 수정은 원라이선스 유지, NC 조건 때문에 저장소 전체가 상업 사용 가능하다고 주장하지 않는다. NFPT Fortran 소스·실행파일·빌드 스크립트는 사용자 지시에 따라 공개하지 않는다. 공개 저장소에는 비공개 외부 의존성과 실행 재현성의 제한을 명시하고 source/executable 체크섬과 설정만 공개한다. 이는 배포 구조와 라이선스 고지 확인이며 모든 법적 상황의 호환성에 대한 보증은 아니다.

Zenodo: GitHub integration enable 후 **새** release를 보존하거나 해당 버전을 직접 업로드한다. GitHub 첨부된550MB NFPT field tar가 자동으로 보존되는지 추정하지 않고 실제 Zenodo files와SHA 확인한다. 필요하면 field dataset을 별도 record로 올려 related identifiers로 연결한다. version DOI와concept DOI를 구별하고 원고에는 실제 보존된 코드 version DOI를 넣는다. DOI와funding은 사용자 제공/실제발급 전 placeholder로본문에 넣지 않는다.

원고 영어/한글 유지, inactive old draft는 scientific evidence로 사용하지 않음. 스크립트 공용 이름은 `run_experiments.py`, `evaluate.py`, `build_comparison_report.py`로 하고 내부 migrated backend의 역사적 이름은 provenance에서만 관리한다.

## 확인한 1차 자료

- NF2 pinned source: https://github.com/RobertJaro/NF2/tree/e7b3ae97dcb37161aea6cd9510d9235de396270a
- Wiegelmann & Sakurai (2021): https://doi.org/10.1007/s41116-020-00027-4
- Zenodo GitHub 연결: https://help.zenodo.org/docs/github/enable-repository/
- FastQSL bundled original license: `third_party/FastQSL/LICENSE`
