The standalone NFGS solver, experiment scripts and configurations, integrated reproduction runner, and results-overview notebook are publicly available at https://github.com/spaceprod/NFGS. Release v0.1.0 includes checksum-verified archived NFPT comparison fields and the manuscript result tables. The repository documents the numerical environment, benchmark parameters, scoring domains, and commands required to reproduce the reconstructions and diagnostics.

독립 실행 가능한 NFGS 풀이기, 실험별 스크립트와 설정 파일, 통합 재현 실행기, 결과 개요 노트북은 https://github.com/spaceprod/NFGS 에 공개되어 있다. v0.1.0 배포본에는 체크섬으로 검증하는 보관된 NFPT 비교 자기장과 논문의 결과표가 포함되어 있다. 저장소에는 복원과 진단을 재현하는 데 필요한 수치 환경, 검증 매개변수, 평가 영역 및 실행 명령을 제시한다.

The NFPT Fortran solver is a private external dependency; its source, executables and local build scripts are excluded from public releases. Only the archived comparison fields and public evaluation tools are provided. Repeating NFPT solver execution requires separate access to the private code.

NFPT Fortran 풀이기는 비공개 외부 의존성으로 소스, 실행파일 및 로컬 빌드 스크립트는 공개 배포본에서 제외한다. 보관된 비교 자기장과 공개 평가 도구만 제공하며, NFPT 실행을 반복하려면 비공개 코드에 별도로 접근할 수 있어야 한다.
