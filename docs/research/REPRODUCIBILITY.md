# 재현성 기록 — 실행 환경, 코드 위치, 결과 파일, 검증 절차

> **상태:** 실제 서버 파일에 직접 접근하지 않고, 대화에서 공유된 명령어·경로·로그를 정리한 초안이다. Codex는 실제 경로를 확인하고 누락/오타를 고쳐야 한다. 이 문서에는 새 GPU 작업을 실행하라는 뜻이 포함되어 있지 않다.

## 1. 저장소와 Python 실행 환경

```text
Project: /home/jimin/vla-representation/vla-representation
RLinf:   /home/jimin/vla-representation/RLinf
RLinf Python: /home/jimin/vla-representation/RLinf/.venv/bin/python
```

보고된 주요 하드웨어는 NVIDIA RTX PRO 5000 Blackwell 48GB다. 환경의 실제 CUDA, Torch, Transformer 버전은 서버 실행 기록으로 검증해야 한다.

### 특히 주의할 점: `sitecustomize.py`

`PYTHONPATH` 맨 앞에 Hook 폴더를 놓으면 Python이 `sitecustomize.py`를 자동으로 import한다. Conda `(base)` Python에서 `torch`가 설치되지 않은 경우 단순 코드 수정 명령에도 `ModuleNotFoundError: No module named 'torch'`가 발생했다.

오프라인 문법 검사 예시:

```bash
cd ~/vla-representation/vla-representation
PYTHONPATH= python -m py_compile scripts/td3_v2_hook/sitecustomize.py
```

Torch가 필요한 진단은 RLinf의 `.venv/bin/python`을 사용했다.

## 2. 실험용 파일 위치 (대화에 나타난 경로)

```text
scripts/run_baseline.sh
scripts/probe_hook/sitecustomize.py
scripts/residual_rl/td3.py
scripts/td3_v2_hook/sitecustomize.py
scripts/td3_compare_hook/sitecustomize.py
scripts/td3_conservative_hook/sitecustomize.py
scripts/td3_positive_constant_hook/sitecustomize.py
scripts/td3_trace_hook/sitecustomize.py
scripts/gt_scripted_hook/sitecustomize.py
scripts/diagnostics/inspect_td3.py
```

추가로 `scripts/residual_hook/`, `scripts/online_td3_hook/`, `scripts/td3_eval_hook/`가 존재했다고 보고됐다. **존재 여부·Git Tracking 여부는 Codex가 직접 확인해야 한다.**

## 3. 보고된 결과 파일 위치

```text
results/baseline/main_50/
results/baseline/vision_50/
results/baseline/execution_50/

results/probe/features/pi05_main_20.npz
results/probe/features/pi05_vision_20.npz
results/probe/features/pi05_execution_20.npz

results/residual/td3_v2/train_01/latest_td3_v2.pt
results/residual/td3_v2/train_50/latest_td3_v2.pt  # INTERRUPTED!
results/residual/td3_v2/eval_scale_025/console.log
results/residual/td3_v2/eval_scale_01/console.log      # 존재 여부 확인
results/residual/td3_v2/eval_scale_05/console.log      # 존재 여부 확인

results/residual/td3_compare/constant_scale05/console.log
results/residual/td3_compare/random_scale05/console.log

results/residual/td3_conservative/train_beta002/latest_td3_v2.pt
results/residual/td3_conservative/eval_main_beta002/console.log
results/residual/td3_conservative/eval_vision_20/console.log
results/residual/td3_conservative/eval_execution_20/console.log
results/residual/td3_conservative/vision_frozen_20/console.log
results/residual/td3_conservative/vision_constant_20/console.log
results/residual/td3_conservative/vision_trace_20/residual_trace.npz

results/residual/gt_scripted/vision_2mm/console.log
results/residual/gt_scripted/vision_5mm/console.log
results/residual/gt_scripted/vision_10mm/console.log
```

**원칙:** `results/`에는 원본 로그와 체크포인트를 보존한다. 큰 파일을 Git에 억지로 올리지 않는다. `docs/research/`의 문서는 상대경로로 원본을 가리키되, GitHub에서 결과 파일이 Ignore되어 있을 수 있음을 설명한다.

## 4. GPU 없이 숫자만 검증하는 명령어

아래는 **읽기 전용** 명령이다.

```bash
cd ~/vla-representation/vla-representation

# GT scripted: 2/5/10mm 평가 로그
for mm in 2 5 10; do
    log="results/residual/gt_scripted/vision_${mm}mm/console.log"
    echo "===== ${mm}mm ====="
    grep -E 'eval/success_once|\[GT-SCRIPT\]' "$log" | tail -5
done

# 50-rollout 요청 실험은 완료되지 않았음을 확인
wc -l results/residual/td3_v2/train_50/worker_*.csv
ls -lh results/residual/td3_v2/train_50/latest_td3_v2.pt
grep 'Evaluating Rollout Epochs:' results/residual/td3_v2/train_50/console.log | tail -4

# Conservative 10-rollout 완료/Replay 확인
tail -2 results/residual/td3_conservative/train_beta002/worker_*.csv

# Trace Shape/통계는 기존 스크립트를 읽고 필요하면 CPU만 사용하여 계산
ls -lh results/residual/td3_conservative/vision_trace_20/residual_trace.npz
```

### Python으로 원본 결과를 재파싱할 때

기존 텍스트를 정규식으로 읽고 `success_once`, `success_at_end`, `return`, `num_trajectories`를 추출하되, 여러 평가 레코드가 하나의 `console.log`에 들어 있으면 정확히 어느 Epoch를 비교하는지 표기한다.

## 5. 과거 평가 명령 템플릿 — **기본적으로 실행하지 말 것**

아래는 당시 사용한 평가 설정을 보존하는 예시이며, `run_baseline.sh`의 실제 구현·새로운 Output 경로를 점검하지 않고 재실행하지 않는다.

```bash
cd ~/vla-representation/vla-representation

export TD3_V2_MODE=eval
export TD3_V2_CHECKPOINT="$PWD/results/residual/td3_conservative/train_beta002/latest_td3_v2.pt"
export TD3_RESIDUAL_SCALE=1.0
export EVAL_ROLLOUT_EPOCHS=1
export EVAL_FIXED_RESET_IDS=true
export EVAL_SAVE_VIDEO=false

export PYTHONPATH="$PWD/scripts/td3_conservative_hook:$PWD/scripts/residual_rl:$HOME/vla-representation/RLinf"
export EMBODIED_PATH="$HOME/vla-representation/RLinf/examples/embodiment"
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"

# 새 폴더와 명시적 승인 없이 평가를 시작하지 말 것.
# export TD3_V2_DIR="$PWD/results/residual/td3_conservative/NEW_DIRECTORY"
# export EVAL_OUTPUT_DIR="$TD3_V2_DIR"
# bash scripts/run_baseline.sh vision 20 > "$TD3_V2_DIR/console.log" 2>&1
```

## 6. 재현성 체크리스트

- [ ] 모든 Result Table 수치를 실제 `console.log`/CSV와 비교했다.
- [ ] 50-episode와 20-episode 결과를 분리했다.
- [ ] Frozen과 Residual 평가의 실제 Reset State IDs를 확인했다.
- [ ] `train_50`을 **50-Rollout 완료 실험**이라고 부르지 않는다.
- [ ] Probe가 어떤 Split, Estimator, Normalization, Label Unit을 사용했는지 코드로 확인했다.
- [ ] Action Controller Frame과 실제 TCP Move를 코드·필요 시 실험으로 검증했다.
- [ ] Residual Scale 전/후 Log가 어떤 값을 보여주는지 구분했다.
- [ ] `done`, `truncated`, `ignore_terminations`, `auto_reset`의 상호작용을 확인했다.
- [ ] `VLA Latent RL`을 수행했다고 쓰지 않는다.
- [ ] 필요 이상으로 다시 학습하거나 평가하지 않고 읽기 전용 감사를 우선한다.

## 7. 결과 Provenance 표기 규칙

실험 표에는 가능한 한 다음을 남긴다.

`run_name | environment | policy/checkpoint | rollout_epochs | episodes | seed/reset IDs | lambda | beta | max_mm | once | end | return | log_path | status`

값을 모르면 추정해서 채우지 않고 `not verified`로 둔다. 특히 **'실패한 결과'도 삭제하지 않고 기록**한다. 실험 설계 의도를 이해하는 데 중요하기 때문이다.
