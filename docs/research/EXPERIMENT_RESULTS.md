# 실험 결과 기록 — Frozen π₀.₅ / Representation Probe / Residual Control

> **문서 성격:** 대화 중 공유된 RLinf 로그·CSV·Probe 요약에 근거한 연구 기록 초안. 서버의 원본 파일과 직접 대조한 확정본은 아니다. `AUDIT` 항목은 Codex로 재검증할 것. 기록일: 2026-11-24.

## 0. 결과를 읽는 원칙

- **Success Once:** 80-step episode 중 한 번이라도 성공 조건을 만족한 비율.
- **Success At End:** episode 종료 시점에 성공 조건을 만족한 비율. Once와 혼동하지 않는다.
- **Return:** episode reward 합계. 성공률을 대체하지 않는다.
- 20개 episode 평가에서는 **한 건이 5%p**다. 단일 실행의 10~20%p 차이를 통계적으로 검증된 개선이라 부르지 않는다.
- `EVAL_FIXED_RESET_IDS=true`는 고정 초기 상태 사용을 의도하지만, 서로 다른 실행의 모든 난수·trajectory가 일치함을 보장하지는 않는다.
- 50-episode Baseline과 20-episode 실험을 구분한다.
- Main은 `obj_set=train`, 선택한 OOD 환경은 `obj_set=test`이므로 차이가 순수한 Shift 효과만을 반영한다고 단정하지 않는다.

## 1. Frozen π₀.₅ Baseline — 왜 먼저 측정했나?

Residual RL이 50% 성공했다고 해도 Frozen이 70%라면 개선이 아니다. 따라서 모든 실험의 기준점으로 원본 Frozen VLA를 별도 측정했다.

### 초기 평가 — 각 50 Episodes

| 환경 | Success Once | 성공 수 | 비고 |
|---|---:|---:|---|
| Main | **66%** | 33/50 | 기본 분포 |
| VisionTexture05 | **34%** | 17/50 | 외관·텍스처 변화 |
| EEPose | **26%** | 13/50 | 초기 EE Pose 변화 |

### 후속 비교용 20 Episodes

| 환경 | Once | At End | Return | 성공 수 |
|---|---:|---:|---:|---:|
| Main | 70% | 70% | 0.850 | 14/20 |
| Vision | 30% | 30% | 0.415 | 6/20 |

**해석:** Distribution Shift에서 작업 성공률이 낮아지는 현상을 관찰했다. 다만 50개와 20개 평가는 짝지어진 동일 표본 비교가 아니다.

## 2. Geometry Probe — 왜 RL에 앞서 표현을 분석했나?

연구 가설은 'VLA가 실패해도 내부에는 제어에 유용한 공간 정보가 남아 있을 수 있다'는 것이다. 이를 RL과 분리하여 확인하기 위해 `object_position - gripper_position` 3D Vector의 예측 오차를 측정했다.

- π₀.₅ VLM Prefix Feature: Masked-Mean Pooling, 2,048D.
- 데이터: 환경별 **20 episodes × 16 chunks = 320 samples**.
- 학습·평가 분리: episode 단위 Grouping을 사용한 것으로 기록되어 있으나 정확한 CV, Probe Estimator와 Aggregation은 저장소에서 재검증 필요.
- 낮을수록 좋은 지표: 상대 위치 예측 오차(cm).

### ID: Relative Position Error

| 입력 | 오차 (cm) |
|---|---:|
| **π₀.₅ + Robot State** | **2.87** |
| π₀.₅ Feature | 2.90 |
| Robot State | 3.68 |
| DINOv2 Global + State | 4.51 |
| SigLIP Spatial + State | 4.53 |
| SigLIP Global + State | 4.55 |
| DINOv2 Spatial + State | 4.70 |

### OOD: Main에서 학습한 Probe를 Shift에 적용

| 입력 | Vision (cm) | Execution (cm) |
|---|---:|---:|
| Robot State | 5.03 | 8.10 |
| **π₀.₅ + State** | **4.52** | **4.86** |

**관찰:** π₀.₅ Feature에서 상대 위치 정보를 복원할 수 있으며, 선택한 OOD 평가에서도 State-only 대비 오차가 낮다.

**주장하지 않은 것:** 이 결과는 **VLA Latent를 이용한 RL의 성공**을 의미하지 않는다. π₀.₅ Feature와 순수 Visual Encoder는 입력 정보량이 다를 수 있으므로 동일 조건에서의 공정한 Architecture 비교도 아니다. Label의 물체 원점은 실제 Grasp Point가 아닐 수 있다.

## 3. TD3 v1/v2 — GT Pose Positive Control

### 왜 GT Pose를 넣었나?

VLA Latent RL을 바로 시작하면 실패 원인을 '표현 부족 / RL 실패 / 보정 불가능'으로 분리하기 어렵다. 먼저 GT Object·Gripper·Plate 위치를 제공하는 **Positive Control**로 학습과 행동 경로 자체를 시험했다. GT 기반 RL의 실패는 **표현 가설의 반증이 아니다**.

| Main 20 Episodes | Once | At End | Return | 구현/학습 상태 |
|---|---:|---:|---:|---|
| Frozen π₀.₅ | **70%** | 70% | 0.850 | 원본 |
| TD3 v1 (14D) | 45% | 40% | — | 초기 설계 |
| TD3 v2 (21D, 10 Rollouts 완료) | 65% | 65% | — | 3,200 Chunk Transitions |
| TD3 v2 (`train_50` 미완료, λ=1) | 35% | 25% | 0.345 | 실제 Replay 1,180 |

**중요한 기록 정정:** `train_50`은 디렉터리 이름과 요청 설정이 50 Rollout이라는 뜻이지 50 Rollout 완료가 아니다. 로그는 `3/50`까지 진행된 사실을 보여주며 CSV 60행, Replay 1,180, TD3 Updates 1,060을 기록했다. **학습량이 많아져 성능이 악화했다는 결론을 철회했다.** 다른 실행의 초기화·수집 경로도 다르다.

### 중단 체크포인트의 Residual Scale Ablation (Main 20)

| λ | Once | At End | Return |
|---:|---:|---:|---:|
| 0 — Frozen | 70% | 70% | 0.850 |
| 0.1 | 55% | 55% | 0.685 |
| 0.25 | 60% | 60% | 0.745 |
| 0.5 | 65% | 55% | 0.705 |
| 1.0 | 35% | 25% | 0.345 |

큰 잔차가 기본 VLA의 동작을 해칠 가능성이 관찰됐지만, 결과가 λ에 따라 단조롭지 않고 어느 설정도 Frozen을 넘지 않았다. λ 조절만으로 유용한 Policy가 학습되었다고 볼 수 없다.

### Constant / Random 비교 (Main 20, λ=0.5)

| 정책 | Once | At End | Return |
|---|---:|---:|---:|
| Frozen | 70% | 70% | 0.850 |
| TD3 | 65% | 55% | 0.705 |
| Constant raw `[-2,-2,+2]mm` (실제 `[-1,-1,+1]mm`) | 35% | 30% | 0.415 |
| Random Uniform Residual | 65% | 65% | 0.800 |

**왜 했나?** 학습된 정책이 무학습 Baseline보다 실제 가치가 있는지 확인하기 위해서다. TD3가 특정 상수보다는 나았지만 Random보다 낫다는 근거는 확보하지 못했다. Random은 Seed 1개이므로 일반적 우위는 주장하지 않는다.

## 4. Conservative TD3 — Residual Action Penalty

### 가설

VLA는 Main에서 이미 70% 성공한다. 모든 Chunk에서 큰 보정을 더하면 성공하던 궤적도 망가질 수 있다. 따라서 Normalized Action Penalty로 과도한 수정을 억제해 보았다.

`R'_chunk = R_chunk - β · mean((residual / 0.002)^2)`; `β=0.02`, 최대 보정 ±2mm/axis.

### 학습 로그

| 지표 | 값 |
|---|---:|
| 완료 Rollouts | 10/10 |
| 기록 Chunks | 160 |
| Replay Size | 3,200 |
| 소요 시간 | 약 13분 24초 |
| 후반 Mean Absolute Residual | 약 1.80mm |
| 후반 Mean Penalty | 약 0.0166 |

### 성능

| 환경 / 정책 | Once | At End | Return | Episodes |
|---|---:|---:|---:|---:|
| Main Frozen | 70% | 70% | 0.850 | 20 |
| Main Conservative TD3 | 70% | 70% | 0.845 | 20 |
| Vision Frozen | 30% | 30% | 0.415 | 20 |
| Vision Constant `[+2,+2,+2]mm` | 35% | 35% | 0.455 | 20 |
| Vision Conservative TD3 | 50% | 35% | 0.470 | 20 |
| Execution Conservative TD3 | 20% | 10% | 0.190 | 20 |

**주의:** Frozen Execution **26%는 50 Episodes**이므로 Conservative 20 Episodes 결과와 짝지어 비교하지 않는다. Vision의 `50% vs 30%`는 탐색적 차이이지 RL로 인한 인과적 개선으로 확정되지 않는다.

## 5. Actor Trace — 성능만으로는 보이지 않던 정책 붕괴

Vision에서 16 Chunks × 20 Environments = **320개 Residual Vector**와 21D Input State를 저장했다.

| 항목 | 측정값 |
|---|---:|
| `states.shape` | `(16,20,21)` |
| `residuals.shape` | `(16,20,3)` |
| X/Y/Z 평균 | 1.99996722 / 1.99999881 / 1.99999714 mm |
| X/Y/Z 표준편차 | 0.00008490 / 0.00000242 / 0.00000505 mm |
| Constant `[2,2,2]mm`와 평균 L2 거리 | **0.00003379 mm** |
| Constant와 최대 거리 | **0.00069877 mm** |
| Constant와 0.001mm 이내 | **100%** |
| State 변화가 관찰된 차원 | 21/21 |

**핵심 관찰:** 입력 State가 변해도 실제 Actor 출력은 사실상 고정된 최대 양의 보정이다. 이로부터 '상태 의존적인 유용한 제어가 학습됐다'는 주장은 할 수 없다. 반면 왜 Actor가 포화되었는지(Pre-tanh, Critic Q Shape, 데이터/스케일)는 아직 검증하지 않았다.

Vision에서 TD3가 Constant보다 15%p 높았더라도, 위 Action Trace 때문에 그 차이를 상태별 Action 조절 효과로 해석할 수 없다.

## 6. GT Scripted Control — RL 문제와 Action 문제를 분리

`e = p_object - p_gripper`, `δ = clip(0.25·e, -max_mm/1000, +max_mm/1000)`을 축별 계산. `0.04m < ||e|| < 0.25m`일 때만 보정. 5 Step Chunk에 `δ/5`씩 적용. `world` Frame을 선택했지만 Controller Frame과의 정확한 일치는 아직 코드 감사 필요.

### Vision 20 Episodes — 최종 결과

| 정책 | Once | At End | Return | 성공 수 (Once) |
|---|---:|---:|---:|---:|
| Frozen π₀.₅ | **30%** | **30%** | 0.415 | 6/20 |
| Scripted GT 2mm | **15%** | 10% | 0.165 | 3/20 |
| Scripted GT 5mm | **40%** | 35% | 0.465 | 8/20 |
| Scripted GT 10mm | **45%** | 30% | 0.405 | 9/20 |
| Conservative TD3 (참고) | 50% | 35% | 0.470 | 10/20 |

### 무엇을 배웠나?

1. 2mm 보정은 Frozen보다 오히려 나쁘다. **보정이 작다는 사실은 보정이 안전하다는 뜻이 아니다.**
2. 5mm·10mm에서 Once가 높아지는 양상을 관찰했으므로 Translation Residual Action Path가 무의미하다고 단정할 수 없다.
3. 10mm에서는 Once 45%였지만 At End는 Frozen과 같은 30%다. **도달과 유지**를 분리해야 한다.
4. 객체 중심은 반드시 Grasp Target이 아니며 4–25cm 거리 조건은 실제 접근 Phase Detector가 아니다.
5. 축별 ±10mm 제한은 Vector Norm의 10mm 제한이 아니다.
6. 개별 Episode별 동일 Reset State에 대한 성공→실패 전환을 추적하지 않았으므로, '실패 3개를 복구했다'는 인과적 표현은 금지한다.

### 대표 로그

```text
5mm / chunk=1: active=20/20, mean_distance=0.1524m, mean_abs_residual_mm=4.7235
  first_error=[ 0.03163528, 0.01100731,-0.13285172]
  first_residual=[ 0.005, 0.00275183,-0.005]
10mm / chunk=1: active=20/20, mean_abs_residual_mm=8.7099
  first_residual=[ 0.00790882, 0.00275183,-0.01]
```

## 7. 최종 결과와 다음 단계

- **확인:** Frozen VLA 추론과 ManiSkill 연결, Feature/GT 수집, 3,200-Transition TD3 Online 학습, 출력 포화 진단, Scripted GT Residual 실험.
- **긍정적 신호:** π₀.₅ 내부 표현에 OOD 공간 정보가 일부 유지된다. 5/10mm GT 보정에서 Once 상승이 관찰됐다.
- **부정적 신호:** TD3는 상태별 유효 보정을 학습했다는 증거가 없으며 Actor가 거의 상수 최대 출력에 포화되었다.
- **미실시:** VLA Internal Latent를 직접 입력으로 받는 Residual RL. 따라서 'VLA Latent가 RL을 개선했다'는 주장을 하지 않는다.
- **연구 의사결정:** 주제를 바꿀 게 아니라 GT Pose Positive Control을 먼저 안정화해야 한다. 다만 KIST 지원 목적에서는 현재의 구현·실험·진단 내용을 정확히 문서화하고 추가 GPU 실험은 보류한다.
