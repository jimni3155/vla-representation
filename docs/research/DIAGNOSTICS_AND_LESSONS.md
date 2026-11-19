# 실험 실패 분석과 판단 기록 — 무엇이 잘못됐고 무엇을 배웠는가

> **기록 목적:** 결과가 좋지 않았을 때 단순히 Hyperparameter를 바꾼 것이 아니라, 어떤 가설을 세웠고 어떤 반례로 해석을 고쳤는지 보존한다. 관찰·추론·미확인을 구분한다.

## 1. 가장 중요한 구분: Positive Control 실패 ≠ VLA Latent 가설 실패

원래 질문: **Do VLA Representations Help RL under Distribution Shift?**

완료한 관련 실험은 (1) VLA Geometry Probe, (2) GT Pose 기반 Residual TD3 Positive Control이다. **π₀.₅ Latent를 직접 넣어 TD3를 학습한 실험은 없다.**

따라서 GT 기반 Residual TD3가 성공하지 못한 현재 상태를 'VLA Representation 연구 실패'라고 부르면 연구 단계와 검증 대상을 혼동하게 된다. 이 판단이 특히 중요해, 이후 Gate 기반 새로운 주제로 옮기는 제안을 보류하고 **GT Scripted 보정을 통해 Action 자체를 검증**했다.

## 2. Failure Timeline

| 단계 | 관찰 | 당시 가설/판단 | 이후 수정한 해석 |
|---|---|---|---|
| Baseline | Main 70%, Vision 30%(20ep) | OOD에서 정책이 취약 | 객체 집합도 다르므로 순수 Shift 효과로만 단정 금지 |
| Probe | VLA+State 오차가 State-only보다 낮음 | 내부 공간 정보가 있음 | 실제 RL 유용성은 별개 |
| TD3 v1 | Main 45% | Actor/Action 문제 가능 | 14D 초기 실험으로 성공 미입증 |
| TD3 v2 | Main 65% | VLA Action 7D 추가의 가능성 | Frozen 70% 미달; 변형 효과 미분리 |
| `train_50` | Main 35% | 처음에는 '오래 학습해 악화' 의심 | CSV 60 Chunks, Replay 1,180로 **중단된 실행**; 과학적 해석 철회 |
| Scale Ablation | λ=0.25/0.5에서 부분 회복 | Residual 크기가 문제일 수 있음 | 모든 λ에서 Frozen보다 낮음; 단조 관계 아님 |
| Constant/Random | Random Once=TD3 Once=65% | TD3만의 학습 효과 불분명 | Random Seed 1개라 일반 결론 금지 |
| Conservative | Main 70%, Actor 경계 포화 | 작은 β로 보정 절제 가능? | β=0.02는 포화 억제 못함 |
| OOD Vision | TD3 50% vs Frozen 30% | OOD 복구 가능성 | 20ep 단일 실행, Actor 사실상 상수; RL 인과 효과 미입증 |
| Actor Trace | 320개 Vector가 상수 +2mm에 거의 동일 | State-conditioned Actor 학습 실패 | 원인은 Critic/Pre-tanh 미측정 |
| GT Scripted | 2mm 15%, 5mm 40%, 10mm 45% Once | Action Space에 개선 가능성 | End 유지 불안정·좌표계 미검증·paired effect 미분리 |

## 3. 가장 큰 실수: 중단된 학습을 완료된 학습으로 오인

실험 이름 `train_50` 때문에 처음에는 50 Rollouts을 완료한 Checkpoint라고 해석했다. 하지만 실제 로그와 CSV:

```text
Evaluating Rollout Epochs: 3/50  (progress shown)
CSV recorded chunks: 60
Replay size: 1180
TD3 updates: 1060
```

**정정:** 50-Rollout 학습은 완료되지 않았다. 10-Rollout 완료 실험과 단순히 데이터 규모만 비교할 수 없으며 '더 오래 학습하면 성능이 나빠졌다'는 주장은 폐기했다.

**교훈:** 실행 디렉터리 이름이나 Checkpoint 존재 여부가 아니라 `rollout progress`, `chunks`, `replay size`, `checkpoint timestamp`, `worker log`를 확인한다.

## 4. TD3 Reward와 Transition 디버깅

### 점검한 내용

1. `use_rel_reward=true`에서 보상은 Reward Potential의 차이로 계산된다.
2. `_reset_metrics()`는 `prev_step_reward`를 전체/부분 Reset 시 0으로 초기화했다.
3. Chunk Reward는 5 Step Discounted Sum으로 구성했다.
4. 21D State의 다음 VLA Action을 포함하기 위해 이전 Nonterminal Transition을 Pending으로 보류했다.
5. `done = terminated | truncated`; Truncation도 Terminal로 처리한 Pilot 구현이다.

### 판단

공유 코드에서 **즉시 확인되는 치명적인 Reward Reset 버그는 발견되지 않았다**. Pending 연결 역시 설계 의도는 일관됐으나 모든 Auto-reset 및 Stage 경계조건이 형식적으로 검증된 것은 아니다. '완벽하게 검증했다'고 말하지 않는다.

## 5. Actor Saturation이 중요한 이유

Conservative TD3 Trace 데이터:

- Inputs: `(16,20,21)`, 21개 State Feature 차원 모두 변화 관찰.
- Outputs: `(16,20,3)`, 320개의 Residual Vector.
- X/Y/Z Mean: `1.99996722 / 1.99999881 / 1.99999714 mm`.
- Constant `[2,2,2]mm`과 Mean Distance: **0.00003379mm**.
- Max Distance: **0.00069877mm**.
- 0.001mm 이내 출력: **320/320 = 100%**.

**관찰:** 다양한 State에도 출력은 사실상 `[+2,+2,+2]mm`로 포화.

**과도한 표현:** 'Actor가 내부적으로 State를 완전히 무시했다'는 것은 확정할 수 없다. tanh 출력이 포화되면 내부 Feature가 달라도 실제 제어 출력은 거의 변하지 않는다.

**기록상 원인 후보:** 잘못된 Q Gradient/오추정, 부족한 Transition, 입력 Feature 스케일, Action Limit 2mm, Reward/Success의 불일치. **Pre-tanh 값, Q(s,δ) 격자, Held-out MC Return Correlation은 미측정**이다.

## 6. 왜 'TD3 50% vs Constant 35%'를 RL 성공이라고 부르면 안 되나?

Vision 20ep에서 TD3 Once=50%, Constant Once=35%였지만, 실제 TD3 Action이 Constant와 수십 나노미터 수준에 가깝도록 포화되었다. 따라서 15%p의 Run-to-run 차이를 **상태 의존 제어의 학습 효과**로 귀속할 근거가 부족하다.

- Fixed Reset State IDs는 전체 환경/모델 추론의 완전한 재현성을 보증하지 않는다.
- 20ep에서는 3개 성공 차이가 15%p다.
- Per-episode 결과를 비교하는 Paired Test/반복 Seed가 없었다.
- Success Once는 도달 여부만 나타내며 End 유지와 별개다.

## 7. 왜 RL을 제거하고 GT Scripted를 시험했나?

TD3가 실패한 이유가 Action Correction Path 자체의 한계인지, RL이 적절한 Action을 찾지 못한 것인지 분리하기 위해서다. GT 상대 위치를 직접 Action으로 변환하면 Actor/Critic/Exploration/Replay 변수가 사라진다.

- `δ = clip(0.25 * (p_object-p_gripper), ± max_mm)`
- 거리 기반 Gate: 4–25cm.
- Vision 20ep: 2mm 15%, 5mm 40%, 10mm 45% (Frozen 30%).

### 의외의 결과와 해석

- **2mm가 가장 나쁨:** 작은 보정이라고 항상 안전하지 않다. Timing/Direction/Controller 상호작용이 중요하다.
- **10mm Once 45%, End 30%:** 도달은 늘었지만 최종 유지 증가는 관찰되지 않았다.
- **5mm Once 40%, End 35%:** 일부 개선 신호가 있으나 성공 수 2개 차이를 통계적 증거로 부르지 않는다.
- **좌표계 불확실성:** GT Vector의 World-aligned 해석과 `root_translation` 컨트롤러의 실제 행동 프레임이 일치하는지 검증되지 않았다.
- **Grasp Point 문제:** 물체 중심으로 이동하는 것이 올바른 파지점 접근을 뜻하지 않는다.
- **Stage 문제:** 거리 4–25cm는 Approach Phase Detection이 아니다.

따라서 이 실험이 말해주는 것은 **Translation 보정 경로가 결과에 영향을 줄 수 있다**는 탐색적 수준이다. 'RL이 해결 가능한 Positive Control에 성공했다'는 결론도 아직 과도하다.

## 8. KIST 지원 목적에서의 실질적 판단

논문을 목표로 한다면 다양한 Seed, 같은 초기 상태의 Paired Success/Failure, Action Frame 실측, Critic Q Landscape, Latent RL 등 추가 검증이 필요하다. 하지만 KIST 지원서에 필요한 것은 이 전체 연구를 완성했다는 주장이 아니다.

**지금 확보한 실제 역량 증거:**

1. Frozen π₀.₅ / RLinf / ManiSkill 통합 실행.
2. VLA 내부 Feature 추출과 Geometry Probe 평가.
3. Chunk-based Residual Action 수정 Hook과 TD3 학습 파이프라인 구현.
4. Replay/Reward/Termination 점검 및 중단 학습 기록 정정.
5. Constant/Random Control 및 320-Action Trace로 Policy Saturation 발견.
6. GT Scripted Control로 RL 학습 문제와 제어 보정 경로를 분리해 검증.

KIST용 문서에서는 **실험을 설계하고 구현·검증·디버깅한 과정**을 강조한다. 새로운 SOTA, 유의미하게 검증된 OOD 성공률 개선, 성공적인 Latent RL이라고 쓰지 않는다.

## 9. 재개할 때의 우선순위 (현재는 보류)

1. Controller Frame과 실제 TCP 위치 변화부터 검증.
2. GT Point 대신 Grasp Target, 정확한 Approach Phase 제어.
3. 같은 Reset State의 Per-episode Frozen vs Scripted Paired 결과 확인.
4. Positive Control이 안정적으로 동작하면 Actor/Critic, Q Grid, Pre-tanh 및 Reward 스케일 진단.
5. 그 이후에 VLA Latent vs DINOv2 / State 기반 RL 비교.

**결론:** 연구 방향은 유지하되, GT Positive Control은 아직 완성되지 않았다. KIST 지원을 위해서는 여기까지의 증거를 정확하게 기록하는 것이 우선이다.
