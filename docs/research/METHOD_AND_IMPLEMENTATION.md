# 실험 방법과 구현 기록 — VLA Feature, Residual TD3, GT Scripted Control

> **목적:** 무엇을 왜 구현했으며, 행동·상태·Reward가 어떻게 연결되는지 남긴다. 이 문서는 대화에서 확인된 코드 일부에 기반하므로 코드·설정의 정확한 세부사항은 Codex가 실제 저장소와 대조해야 한다.

## 1. 문제를 두 단계로 나눈 이유

원래 질문은 **Do VLA Representations Help RL under Distribution Shift?**이다. 이 질문은 두 독립적인 하위 실험으로 나누었다.

1. **Representation Probe:** Frozen π₀.₅ 내부 표현에서 조작에 필요한 상대 위치 정보를 읽어낼 수 있는가?
2. **GT-Pose Residual RL Positive Control:** 정확한 상대 위치를 줬을 때 보정 정책 자체가 학습될 수 있는가?

순서를 바꿔 곧바로 Latent 기반 RL을 시작하면, 실패했을 때 VLA 표현이 나쁜지, Critic이 나쁜지, Controller Action이 잘못됐는지 구분하기 어렵다. 이번 실험에서 GT TD3가 실질적 정책으로 발전하지 못했다고 해서 VLA 표현 연구가 반증된 것은 아니다.

## 2. 시스템 구조

```text
Image + Instruction + Robot Observation
               │
               ▼
       Frozen π₀.₅ ManiSkill SFT
               │
               ▼
       VLA Action Chunk [B,5,7]
               │
        ┌──────┴────────────┐
        ▼                   ▼
 Current VLA Action     Original Rotation / Gripper
        │                   │
 GT Object–Gripper / Plate + qpos
        │
        ▼
  TD3 Actor (or GT Scripted Formula)
        │
        ▼
 Residual δ∈R³ (one per chunk)
        │
        ▼
 a'[b,i,:3] = a[b,i,:3] + δ[b,:]/5
        │
        ▼
   RLinf EnvWorker → ManiSkill
        │
        └─ reward / done / next observation → replay / TD3 update
```

### 왜 Frozen VLA인가?

모델 전체를 다시 Fine-tuning하지 않고도 추가 제어기를 시험하려 했다. 큰 모델의 Gradient를 계산하지 않으면서 Residual이 기존 정책에 미치는 영향을 분리하기 위한 선택이다.

### 왜 3D Translation만 수정했나?

7D Action은 앞의 3D Translation, 다음 3D Rotation, 마지막 Gripper Command로 구성됐다. 최초 제어 보정은 상대 위치와 직접 연결되는 Translation만 다뤘다. Rotation/Gripper까지 한 번에 바꾸면 실패 원인을 구분하기 어렵다. 단, 이 선택은 파지 자세·그리퍼 타이밍 실패를 복구하기 어려운 제약을 만든다.

## 3. 시뮬레이션과 Action Interface

| 항목 | 대화에서 확인된 설정 |
|---|---|
| Base Model | Frozen π₀.₅, ManiSkill SFT checkpoint |
| Simulator | ManiSkill 3 |
| Orchestrator | RLinf |
| 기본 Task | `PutOnPlateInScene25Main-v3` |
| Vision OOD | `PutOnPlateInScene25VisionTexture05-v1` |
| Execution OOD | `PutOnPlateInScene25EEPose-v1` |
| Horizon | 80 environment steps |
| Chunk Length | 5 actions; 16 chunks / episode |
| VLA Action Tensor | `[num_envs, 5, 7]` |
| Controller | `PDEEPoseController`로 보고됨 |
| Controller Flags | `normalize_action=False`, `use_target=True`, `use_delta=True`로 보고됨 |
| Frame Descriptor | `root_translation:root_aligned_body_rotation`로 보고됨 |

**아직 미확인:** 이 컨트롤러가 Translation Delta를 실제로 어느 프레임에서 해석하는지, Target 누적과 Low-level Step 사이에서 어떤 변환이 일어나는지. Scripted 실험의 `GT_TRANSLATION_FRAME=world`는 당시 사용한 설정이지, 물리적으로 검증된 정답 프레임이라는 뜻이 아니다.

또한 Residual `2mm/5mm/10mm`는 Controller에 전달한 **명목상 축별 Action 크기**다. 실제 TCP가 그 거리만큼 이동한 것을 측정한 결과가 아니다.

## 4. Geometry Probe 설계

- **Label:** `p_object - p_gripper`, 3D 상대 위치. Object Pose의 기준점은 실제 Grasp Point와 다를 수 있다.
- **VLA Feature:** π₀.₅ VLM Prefix의 Masked-Mean Pooling, 약 2,048D.
- **샘플:** 환경별 20 Episode × 16 Chunk = 320 rows.
- **비교:** π₀.₅ alone; π₀.₅ + robot state; state alone; DINOv2/SigLIP global/spatial feature + state.
- **메트릭:** 상대 위치 예측 오차(cm).
- **데이터 분리:** Episode 단위 Grouping을 사용한 것으로 보고됨. 정확한 `GroupKFold`, 중첩 CV, estimator 및 Seed는 코드 검증 필요.

VLA Prefix Feature에는 이미지뿐 아니라 다른 모달리티의 정보가 들어갈 수 있으므로 일반 Visual Encoder와 입력 정보량이 반드시 같지는 않다. Probe 성능이 RL 성능을 보장하지는 않는다.

## 5. GT Positive Control: State 정의

### v1 — 14D

```text
object - gripper relative xyz  3
plate  - object  relative xyz  3
robot qpos                   8
--------------------------------
TOTAL                       14
```

### v2 / Conservative — 21D

```text
v1 state                    14
current VLA chunk first action 7
--------------------------------
TOTAL                       21
```

**왜 첫 VLA Action을 추가했나?** 같은 GT 상대 위치에서도 VLA가 이미 적절한 방향으로 움직이려는지에 따라 보정 필요성이 달라질 수 있다. 따라서 현재 기본 정책이 '무슨 행동을 하려는지' 입력으로 제공했다.

**중요:** 이 7D는 VLA가 출력한 Action이지, π₀.₅의 **2,048D 내부 Latent가 아니다**. 이번 RL 실험은 Latent 기반 RL이 아니다.

## 6. Chunk-Level Residual Action

한 Chunk에 3D Vector 한 개를 생성한다. v2에서 축별 최대 크기는 ±0.002m였다.

```python
modified = actions.clone()
modified[:, :, :3] += (residual / chunk_len)[:, None, :]
```

평가에서 Residual Scale을 적용할 때:

```python
scale = float(os.environ.get('TD3_RESIDUAL_SCALE', '1.0'))
modified[:, :, :3] += (residual * scale / chunk_len)[:, None, :]
```

- `λ=1`: 출력 그대로 적용.
- `λ=0.5`: 명목상 Chunk Residual 크기를 반으로 줄임.
- Rotation과 Gripper Action은 그대로 둠.
- 각 축의 ±2mm 제한은 Vector Euclidean Norm 2mm 제한이 아니다.
- Per-step Action과 실제 EE Pose 변화는 Controller 구현에 의존한다.

**왜 Chunk마다 한 번만 결정하나?** RL의 제어 주기를 줄이고, 기본 VLA의 시간 구조를 보존하려 했다. 반대로 Chunk 내부 상황 변화에 실시간 대응하기 어렵다는 단점이 있다.

## 7. TD3 — Policy, Replay, 학습 설정

대화에서 사용한 구현 요약:

| 설정 | 값 |
|---|---:|
| Actor/Critic Hidden Width | 128, 128 |
| Actor Output | 3D Residual |
| Replay Capacity | 100,000 |
| Warmup | Replay 256개까지 Random Residual |
| Batch Size | 128 |
| Rollout Batch | 20 Parallel Envs |
| Updates | Chunk당 20회 (`for _ in range(n)`) |
| Exploration Noise | 약 0.0005m |
| TD3 Chunk Discount | 보고된 값 `0.99**5` (실제 코드 확인) |

TD3는 Replay를 재사용하는 Off-policy 연속 제어 알고리즘이다. VLA는 Freeze했고 작은 Actor/Critic만 학습했다. 10 Rollouts × 20 Envs × 16 Chunks = **3,200 Chunk Transitions**이지 3,200 Episode가 아니다.

## 8. Reward: 상대 보상과 Chunk Return

공유된 RLinf 코드의 Default Reward Potential:

```python
reward = (
    0.1 * info['is_src_obj_grasped']
    + 0.1 * info['consecutive_grasp']
    + 1.0 * (info['success'] & info['is_src_obj_grasped'])
)
reward_diff = reward - self.prev_step_reward
self.prev_step_reward = reward
```

`use_rel_reward=true`이므로 `reward_diff`를 학습용 Step Reward로 사용한다. `reset()`에서 호출되는 `_reset_metrics()`는 전체 Reset과 부분 Reset 모두 `prev_step_reward`를 0으로 초기화했다. 따라서 **에피소드 간 이전 보상이 누적된다는 버그는 공유 코드에서 발견되지 않았다.**

한 Chunk의 Return은 다음과 같다.

`R_chunk = Σ(i=0..4) γ^i · r_(t+i)`

```python
weights = GAMMA ** np.arange(chunk_len, dtype=np.float32)
returns = (rewards * weights[None, :]).sum(axis=1)
```

학습 Reward와 Success Metric은 동일하지 않다. 짧은 Chunk Return을 올리는 정책이 전체 80-Step 성공률까지 높인다고 볼 수 없다.

## 9. Pending Transition — 왜 다음 VLA Action이 필요했나?

현재 TD3 State에는 **현재 Chunk의 첫 VLA Action 7D**가 들어 있다. Chunk를 다 실행한 직후에는 다음 Chunk의 VLA Action이 아직 생성되지 않았을 수 있다.

이에 다음 방식으로 Replay를 저장했다.

```text
Chunk k 시작: s_k = GT pose + qpos + VLA action_k
             residual_k 선택
Chunk k 실행: 5 steps, reward R_k 계산
             nonterminal이면 pending에 (s_k, residual_k, R_k) 보관
Chunk k+1:   새로운 VLA action_{k+1}을 포함한 s_{k+1} 계산
             pending을 (s_k, residual_k, R_k, s_{k+1}, done=0)으로 완성
```

단 Terminal Row는 `done=1`로 처리하고 Next State에 0을 저장했다. `done=True`이면 TD3 Target에서 Bootstrap을 하지 않으므로 Next State 값은 보통 사용되지 않는다.

### 아직 해결되지 않은 부분

- `ignore_terminations=true`: Success가 Episode의 즉시 종료를 뜻하지 않는다.
- `truncation`도 Terminal로 처리한 Pilot 구현: 유한 Horizon 정의·Time Limit Bootstrapping 방식과 일치하는지 추가 검토 가능.
- Auto Reset과 Chunk Boundary, Worker Stage 간 Pending dict가 정확하게 정리되는지 모든 경계조건을 검증한 것은 아님.
- 60 Chunks에서 Replay 1,180개는 마지막 Chunk의 Nonterminal Pending 20개와 일치할 수 있으나, 실행 중단 시점의 정확한 Event Sequence까지 증명하지는 못한다.

## 10. Conservative TD3 — 무엇을 수정했나?

기존 TD3 v2를 복사해 Replay에 저장하는 Chunk Reward에서 Residual Penalty를 뺐다.

```python
beta = 0.02
normalized_residual = residual / MAX_RESIDUAL  # MAX_RESIDUAL = 0.002 m
penalty = beta * np.mean(normalized_residual ** 2, axis=1)
returns = (returns - penalty).astype(np.float32)
```

**목적:** Actor가 기존 VLA를 필요 없이 크게 수정하지 않도록 유도.

**결과:** 10 Rollouts를 완료했지만 Actor Output이 세 축 모두 거의 +2mm로 포화되었다. 이 한 설정에서 Penalty가 출력을 효과적으로 억제했다는 근거가 없었을 뿐, Penalty 일반을 반증한 것은 아니다.

## 11. GT Scripted Residual — 왜 RL을 제거했나?

TD3가 포화된 원인이 RL 학습인지, Action Space가 애초에 쓸모없는지 분리하기 위한 검사다.

```python
error = object_xyz - gripper_xyz
active = (np.linalg.norm(error, axis=1) > 0.04) & (
    np.linalg.norm(error, axis=1) < 0.25
)
limit = max_mm / 1000.0
residual = np.clip(0.25 * error, -limit, +limit)
residual[~active] = 0.0
```

GT Error는 `world` 좌표계로 간주해 그대로 Action에 넣었다. 실제 Controller Frame 검증 필요. `active`는 단순 거리 Gate이지 접근/파지/배치 Phase Label이 아니다. 물체 중심이 Grasp Target이 아닐 수 있어 성공을 보장하지 않는다.

사용한 최대 축별 보정: 2mm / 5mm / 10mm. 5개 Action에는 각각 Residual/5를 더했다.

## 12. 구현상 주의사항

- Vendor RLinf 원본을 변경하기보다 별도의 `scripts/*_hook/sitecustomize.py`를 사용했다.
- `sitecustomize.py`가 `PYTHONPATH`에 잡힌 상태로 Conda `(base)` Python을 실행하면 `torch`를 import하지 못하는 경우가 있었다. Offline 수정·컴파일은 `PYTHONPATH= python ...`로 실행했다.
- 코드를 수정할 때 **주석의 정확한 문자열**을 Anchor로 가정해 AssertionError가 발생한 경험이 있다. 이후 실제 Code Anchor와 구조를 확인하도록 바꿨다.
- TD3 평가 로그의 `mean_abs`/`sat_rate`는 Scale 적용 **전의 Actor Raw Output**이었다. 실제 적용 Residual과 혼동하지 않는다.
- 저장된 Checkpoint가 있다고 학습을 완료했다고 판단하지 않는다. Rollout Progress와 Replay/CSV를 함께 확인한다.
- Code Audit 전까지 Controller Frame과 미검증 시나리오에 대한 과도한 주장을 하지 않는다.
