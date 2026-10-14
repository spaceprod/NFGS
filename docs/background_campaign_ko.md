# Published NFPT 고정 방침의 백그라운드 실행

2026-10-14 사용자 지시에 따라 기존 큐를 published_nfpt_fixed_v2로 개편해 재개했다.
활성 큐는 NFGS 221개와 NF2 291개, 총 512개다. 완료된 25개와 중지된 NF2 optimizer를 그대로 재사용했다. 원래 644개 manifest, 상태, 제외한 로컬 PT 132개 및 로그는 policy_revisions에 보존한다.

NF2는 기존 pinned commit, 15,000 optimizer steps, weight 11점, seeds 0–4를 유지한다. 별도 파일럿이나 reference 기반 checkpoint 선택을 추가하지 않는다. Published NFPT/VFVP/SSW 값은 해시로 고정하고 새 결과의 CCI·시간을 그 행에 붙이지 않는다. 로컬 Fortran 실행은 활성 큐에서 금지한다.

## 상태와 ETA

```bash
PYTHONPATH=. .venv/bin/python scripts/run_controlled_status.py
```

중간 보고 요청 시 호스트에서 현재 상태를 읽어 완료/전체, 현재 step, 실패 여부, 남은 계산 ETA와 KST 완료 추정을 반환한다. 중지 시간은 보정하고 raw 시간과 interval도 보존한다. 실패 작업을 ETA에서 제외했는지 명기한다. ETA에는 원고 편집, 원래 NFPT 프로토콜 자료 확인, 외부 DOI/funding 대기가 포함되지 않는다.

## 실행·집계

```bash
PYTHONPATH=. .venv/bin/python scripts/run_controlled_background.py --root results/controlled/live_20261014 --retry-failed
PYTHONPATH=. .venv/bin/python scripts/build_live_comparison_report.py --root results/controlled/live_20261014
```

실행 중인 배분기와 중복 실행하지 않는다. 새 배분기는 기존 NF2 PID의 완료를 기다린 뒤 같은 입력 hash/좌표로 평가하고 다음 작업으로 진행한다. 재시도에서는 기존 field를 재사용할 수 있으며, finished budget이 확인되지 않은 seed를 성공으로 표시하지 않는다.

published_scores_unchanged.csv는 원 논문 protocol, all_executed_scalar_metrics.csv는 실제 common_fd2_gs_v1 protocol로 명시한다. Pareto는 NFGS–NF2 신규 결과만 사용한다. 설정별 warm-up 제외 5회 timing을 집계하며 다른 rho/seed/weight를 한 그룹으로 합치지 않는다. 원 NFPT benchmark의 original coordinate/mask/curl gate가 통과되기 전에는 이 캠페인을 published NFPT와 동일 조건 비교라고 쓰지 않는다.

저장 위치는 .local/controlled/active_root, 상태는 campaign_state.json, 각 작업은 progress/result/metrics JSON이다. 기존 NFGS padding은 mudpack_legacy로 유지하며 이미 완료된 author-confirmed padding 99건, TD3 보조 3건, NFPT 재현 검사 7건은 별도 근거로 재사용한다. 관측 잡음으로 확인되지 않은 synthetic noise를 HMI uncertainty로 표시하지 않는다.

Fortran 소스·실행파일·빌드 및 실행 adapter는 비공개다. 원고에 없는 결과·DOI·funding을 만들어 넣지 않는다.
