# FUTURE_WORK: DAMT 연구 방향

> **최종 수정일**: 2026-03-20
> **현재 상태**: 9개 pretext task (rot/loc/**VICReg**/atlas/feat/texture/mim/msn/asym) 구현 완료. linear_probe.py 완료. 30-epoch 대조 실험 분석 완료.
> ⚠️ **age/sex pretext task 제거** — downstream 평가 target과 동일한 신호를 pretraining에 사용하면 label leakage 발생. **SimCLR → VICReg 교체** 완료.

---

## 0. 30-epoch 실험 결과 분석 (2026-03-20)

### 0.1 실험 개요

| 실험 | 구성 | 최종 loss (epoch 29) |
|------|------|---------------------|
| runs_test_baseline | 7 tasks (MSN 없음) | **3.580** |
| runs_test_msn | 8 tasks (MSN 포함) | **3.773** |

두 run의 loss 차이(0.193)는 MSN loss의 잔여값(0.328 × weight)에서 비롯된 것으로, MSN이 다른 task 수렴에 미치는 영향은 거의 없다.

### 0.2 각 task별 loss 추이 (epoch 0 → epoch 29, baseline 기준)

| Task | Epoch 0 | Epoch 10 | Epoch 29 | 수렴 상태 |
|------|---------|----------|----------|----------|
| rot | 2.36 | 0.77 | **0.21** | ✅ 빠르게 수렴 |
| loc | 1.84 | 1.19 | **0.74** | 🔶 느리게 수렴 중 |
| contrastive | 1.16 | 1.06 | **0.76** | 🔶 plateau |
| atlas | 4.28 | 1.72 | **1.11** | 🔶 가장 큰 잔여 loss |
| feat | 0.53 | 0.37 | **0.34** | ✅ 수렴 근접 |
| texture | 0.68 | 0.50 | **0.45** | ✅ 수렴 근접 |
| mim | 0.92 | 0.47 | **0.42** | 🔶 느리게 감소 중 |
| msn (MSN run) | 0.55 | 0.33 | **0.33** | ❌ epoch 5부터 plateau |

### 0.3 MSN loss 평가

**요약**: MSN loss는 epoch 5에서 이미 0.339에 도달한 후 epoch 29까지 거의 변화 없음 (0.328).
동시에 `w_msn`(AutoWeightedLoss weight)은 1.0 → 1.257로 계속 증가 — 모델이 MSN task에 더 높은 confidence를 부여하지만 raw loss를 줄이지 못하는 상태.

**해석**: CLS token(768-dim global average pooling)은 1891개 MSN edge를 예측하기에 정보가 불충분하다.
MSN은 뇌 영역별 세밀한 공간 정보를 요구하는 반면, CLS token은 전역 요약만 담기 때문이다.

**긍정적 신호**: MSN run에서 contrastive loss가 baseline 대비 약간 더 낮게 수렴 (epoch 29: 0.750 vs 0.755).
MSN task가 CLS token을 더 풍부한 전역 표현으로 유도하는 부수 효과가 있을 수 있다.

**결론**: MSN 자체는 유지하되 **아키텍처 개선이 필수**. 현재 구현은 성능 상한이 낮다.

### 0.4 가장 문제가 되는 task: Contrastive (SimCLR)

SimCLR은 같은 batch 내의 다른 샘플을 negative로 사용한다.
현재 설정: 4 GPU × batch_size=2 = **8 samples/step → negative 수 6개**.
이 정도로는 충분한 학습 신호를 얻기 어렵고, epoch 29에서도 0.76으로 수렴이 정체된다.

---

## 1. 최우선 과제: linear_probe.py 작성

**현재 상태**: 사전학습 품질을 정량화할 평가 코드가 없다.
loss 수치만으로는 어떤 task 조합이 downstream에 유익한지 판단할 수 없다.

**다음 30-epoch 실험 전에 반드시 구현해야 할 것**:

```python
# 개요: pretrained checkpoint → freeze encoder → linear head fine-tune → 평가
python linear_probe.py \
    --checkpoint runs_test_msn/checkpoint.pth \
    --data-path /path/to/labeled_data \
    --task age_regression     # 또는 diagnosis_classification
    --epochs 50 \
    --lr 1e-3
```

**권장 downstream task (현실적 접근 순서)**:

| 우선순위 | Task | Dataset | 이유 |
|---------|------|---------|------|
| 1순위 | 뇌 나이 예측 (회귀) | fomo60k metadata (age) | label이 이미 있을 가능성 높음 |
| 2순위 | 조현병 분류 (CN vs SCZ) | fomo60k SCZ data | 최종 목표 |
| 3순위 | 알츠하이머 분류 | ADNI (CN/MCI/AD) | 공개 데이터 |

---

## 2. Loss 조합 검토: 추가 vs 제거

### 2.1 가장 즉각적인 개선: Contrastive 교체

**현 문제**: SimCLR with batch=8 → negative 수 6개 → 학습 신호 약함. 0.76에서 plateau.

**대안 A — BYOL / SimSiam (negative-free)**:
- negative 없이 두 뷰의 표현 일관성만 학습
- batch size에 무관하게 안정적 학습
- Collapse 방지: BYOL은 momentum encoder, SimSiam은 stop-gradient

**대안 B — VICReg**:
- Variance-Invariance-Covariance regularization
- batch 크기 무관, collapse 없음
- 코드 변경 규모: loss.py에 VICReg 클래스 추가 (~50줄)

**대안 C — MoCo (memory bank)**:
- 큰 memory bank(e.g. 4096)로 효과적인 negative 확보
- 기존 SimCLR 구조를 최대한 유지

**권장**: 구현 용이성 기준 VICReg > SimSiam > MoCo. 논문 기여 측면에서는 VICReg가 뇌 MRI에 적용된 사례가 드물어 novelty가 있다.

### 2.2 신규 Loss 후보: 뇌 나이 예측 (Brain Age Regression)

**근거**:
- Brain age gap (예측 나이 - 실제 나이)은 신경정신과 질환의 강력한 바이오마커
- 조현병 환자는 평균적으로 뇌 나이가 실제보다 높게 추정됨 (accelerated aging)
- 나이 label은 별도의 진단 label 없이 메타데이터에서 바로 얻을 수 있음 (약한 지도 학습)

**구현**:
```python
# head: CLS token → Linear(768, 1)
# loss: SmoothL1(predicted_age, chronological_age)
# target: normalized age ((age - mean) / std)
```

**논문 포인트**: "We leverage chronological age as a weak supervision signal to guide the encoder toward clinically meaningful representations." → downstream SCZ 분류 성능 향상을 함께 보여주면 강력한 contribution.

### 2.3 신규 Loss 후보: 반구 비대칭 예측 (Hemisphere Asymmetry)

**근거**:
- 뇌는 구조적으로 좌우 비대칭이며, 언어 기능 측화 등이 조현병과 관련
- FreeSurfer stats에서 LH/RH를 분리하면 비대칭 지수를 레이블 없이 계산 가능

**구현 아이디어**:
```python
# LH features와 RH features를 각각 추출 (split atlas)
# Asymmetry Index (AI) = (LH - RH) / (0.5 * (LH + RH)) per region per feature
# head: CLS token → predict AI vector (n_regions × n_features)
```

**우선순위**: 중간. 뇌 나이보다 구현이 복잡하고 novelty가 제한적.

### 2.4 제거 고려 대상

| Task | 현황 | 판단 |
|------|------|------|
| rot (rotation) | 0.21까지 수렴, 사실상 포화 | **유지** — 나쁜 3D 표현 방지. 제거해도 큰 변화 없을 것 |
| loc (location) | 0.74로 여전히 높음 | **유지** — 공간 인식 여전히 학습 중 |
| atlas | 1.11로 가장 큰 잔여 | **유지하되 개선** — 101-class를 coarse parcellation으로 줄이는 것 고려 |

**현 시점 결론**: 제거보다 downstream 평가 우선. 평가 없이 제거/추가 판단은 추측에 불과.

---

## 3. MSN 아키텍처 개선

### 3.1 즉각 적용 가능: Multi-scale Feature

현재 CLS token = `AdaptiveAvgPool3d(x4)` = 768-dim.
x4는 4³ 해상도로 이미 공간 정보가 거의 없다.

```python
# x2: (B, 192, 16, 16, 16) → pool → (B, 192)
# x3: (B, 384, 8, 8, 8)   → pool → (B, 384)
# x4: (B, 768, 4, 4, 4)   → pool → (B, 768)  (현재 CLS)
x_ms = torch.cat([
    F.adaptive_avg_pool3d(x2, 1).flatten(1),
    F.adaptive_avg_pool3d(x3, 1).flatten(1),
    cls_token,
], dim=1)  # 1344-dim
```

이 1344-dim feature를 msn_head, glo_feat_head, texture_head의 입력으로 사용하면 각 head가 더 풍부한 정보를 활용할 수 있다.

### 3.2 중기 개선: 압축된 MSN 예측

1891-dim 전체를 예측하는 대신, 정보가 가장 많은 성분만 예측:
- **PCA 압축**: 학습 전 전체 subject의 MSN에 PCA 수행 → top-32 PC 예측
  - 1891 → 32 차원으로 target 축소
  - Loss scale 및 learning signal 명확
- **Top-K 연결 선택**: 개인 간 분산이 가장 큰 K개 edge만 예측 (K=200 등)

### 3.3 장기 개선: Regional Feature 기반 MSN 예측

CLS token 대신 atlas segmentation으로 ROI 단위 feature를 추출해서 MSN 예측:
```
x4 (4,4,4 feature map) + atlas mask → ROI-wise average pooling → (n_regions, 768) → MSN
```
이 방향은 GNN decoder와 결합하면 해석 가능성도 높아진다.

---

## 4. Atlas Loss 개선

Atlas (FreeSurfer aparc+aseg) 101-class segmentation은 30 epoch 후에도 1.11로 가장 크다.

**원인**: 101개 클래스 중 다수가 작은 subcortical 구조물로 voxel 수가 극히 적음 → class imbalance.

**개선 방향**:

| 방향 | 설명 | 효과 |
|------|------|------|
| Coarse parcellation | 101-class → 7 Yeo networks + subcortical | Loss 수렴 빠름, 정보량 감소 |
| Weighted CE | 희귀 class에 높은 weight | Class imbalance 완화 |
| DiceCE hybrid | CE + Dice 결합 | 작은 구조물 recall 향상 |
| Hierarchical loss | Coarse(7-class)→Fine(101-class) curriculum | 안정적 수렴 |

**권장**: 가장 간단한 **Weighted CrossEntropy**를 먼저 시도. 각 class의 inverse frequency를 weight으로.

---

## 5. 학습 전략

### 5.1 Curriculum Learning (장기)

8개 task를 단계적으로 활성화:

```
Phase 1 (epoch 0~30):   rot + mim                           → 기본 3D 구조 학습
Phase 2 (epoch 30~100): + loc + contrastive (or BYOL)       → 공간/전역 표현
Phase 3 (epoch 100~):   + atlas + feat + texture + msn + age → 도메인 지식 통합
```

### 5.2 데이터 증강 강화

현재 증강: spatial crop + 3D rotation (10 orientations).

추가 권장:
- **Intensity bias field**: MRI scanner 아티팩트 시뮬레이션 → scanner 간 robustness
- **Elastic deformation**: 해부학적 다양성 시뮬레이션 (MONAI의 `Rand3DElastic` 사용 가능)

---

## 6. 다운스트림 task와의 연계 (조현병 예측)

fomo60k_wo_scz는 조현병 제외 데이터로 pre-training 중.
최종 목표는 이 encoder로 조현병 분류를 fine-tune하는 것.

**조현병과 관련된 pretext task 설계 관점**:

| Pretext Task | 조현병과의 관련성 |
|-------------|----------------|
| feat (FreeSurfer features) | 피질 두께, 피질 표면적 이상이 조현병의 핵심 바이오마커 |
| MSN | 조현병에서 frontal-temporal 연결성 이상 잘 알려져 있음 |
| Brain age (신규) | 조현병 환자에서 accelerated brain aging 관찰됨 |
| Contrastive | 진단에 관계없이 일반적 표현 학습 |
| rot / loc / mim | 진단과 직접 관련 낮음 — 일반 표현 품질 향상 역할 |

**권장 전략**: feat + MSN + brain_age 세 가지 domain-specific task의 기여도를 ablation으로 측정하면, "어떤 사전 지식이 조현병 예측에 중요한가"를 보여줄 수 있어 논문의 핵심 contribution이 될 수 있다.

---

## 7. 논문 실험 계획 (업데이트)

### 7.1 Ablation Study

| 실험 | 구성 | 목적 |
|------|------|------|
| Full | rot+loc+contrastive+atlas+feat+texture+mim+msn | 전체 모델 |
| w/o MSN | MSN 제거 | MSN 기여도 |
| w/o Domain | feat+texture+msn 제거 | 도메인 지식 기여도 |
| w/o Geometry | rot+loc 제거 | 기하 학습 기여도 |
| w/o Contrastive | contrastive 제거 | 전역 표현 기여도 |
| MIM only | rot+loc+contrastive+atlas+feat+texture+msn 제거 | 순수 MAE 베이스라인 |
| + Brain Age | Full + brain_age | 약한 지도학습 효과 |

### 7.2 비교 모델

| 방법 | 코드 | 비고 |
|------|------|------|
| MAE 3D | 공개 | 가장 가까운 SSL 베이스라인 |
| SwinMM | 공개 | 의료 3D SSL |
| Models Genesis | 공개 | 의료 3D SSL |
| UniMiSS | 공개 | 의료 3D SSL |
| 원 논문 (DAMT) | 본 코드 | 개선 대상 |

### 7.3 스케일링 실험

사전학습 데이터 크기에 따른 downstream 성능:
- 1k / 5k / 10k / 30k / 60k subjects
- 이 곡선 하나만으로도 강력한 Figure 생성 가능

---

## 8. 코드베이스 우선순위

| 항목 | 상태 | 우선순위 |
|------|------|---------|
| **linear_probe.py** | ✅ 완료 | — |
| **hemisphere asymmetry pretext task** | ✅ 완료 | — |
| **age/sex pretext task 제거** (label leakage 방지) | ✅ 완료 | — |
| **SimCLR → VICReg 교체** | ✅ 완료 | — |
| **downstream 평가 실행** (linear_probe.py) | 대기 중 | 🔴 높음 — loss 비교 근거 확보 |
| MSN multi-scale feature | 없음 | 🟡 중간 — MSN plateau 해결 필요 |
| Atlas weighted CE | 없음 | 🟡 중간 — 최대 잔여 loss 개선 |
| Hydra config 도입 | 없음 | 🟡 중간 |
| MSN PCA 압축 target | 없음 | 🟢 낮음 |
| Docker/Singularity 환경 고정 | 없음 | 🟢 낮음 |

---

## 9. 데이터 관련

### 9.1 MSN 사전 생성

```bash
python extract/msn_feat.py \
    --data-path /NFS/Users/kimyw/data/fomo60k_wo_scz \
    --data fomo60k_wo_scz \
    --features SurfArea GrayVol ThickAvg MeanCurv GausCurv
```

출력: `{data_path}/results/msn_sa_gv_ta_mc_gc/` (피험자당 .mat).
MSN 파일이 없는 피험자는 zero vector → `remove_zerotensor`로 자동 스킵.

### 9.2 QC 필터링

현재 fomo60k_wo_scz (9,153명)는 QC 전 데이터. QC 후 재학습 예정.
QC 기준: 움직임 아티팩트, FreeSurfer 처리 실패, 이상 뇌 부피 등.

---

## 10. 참고 문헌

- Kendall et al. (2018). Multi-task learning using uncertainty to weigh losses. CVPR.
- He et al. (2022). Masked autoencoders are scalable vision learners. CVPR.
- Caron et al. (2021). Emerging properties in self-supervised vision transformers. ICCV.
- Tang et al. (2022). Self-supervised pre-training of swin transformers for 3d medical image analysis. CVPR.
- Bardes et al. (2022). VICReg: Variance-Invariance-Covariance Regularization. ICLR.
- Grill et al. (2020). Bootstrap Your Own Latent — a new approach to self-supervised learning. NeurIPS.
- Cole et al. (2017). Predicting brain age with deep learning. NeuroImage. (Brain age regression 원조)
- Franke et al. (2019). Brain-age: a useful biomarker for psychiatric disorders. Neuroscience & Biobehavioral Reviews.
- Yeo et al. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. J Neurophysiology.
- Seidlitz et al. (2018). Morphometric Similarity Networks Detect Microscale Cortical Organization. Neuron.
