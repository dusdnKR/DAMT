# FUTURE_WORK: DAMT 연구 방향

> **최종 수정일**: 2026-03-19
> **현재 상태**: 8개 pretext task (rot/loc/contrastive/atlas/feat/texture/mim/msn) 구현 완료. MSN 생성 스크립트 완료. fomo60k_wo_scz 학습 진행 중.

---

## 1. 즉시 수행 가능한 실험

### 1.1 Loss 조합 탐색 (최우선)

MSN loss 추가를 계기로 어떤 task 조합이 downstream 성능에 기여하는지 ablation으로 확인한다.

| 실험 ID | 활성 task | 목적 |
|---------|-----------|------|
| A | 7개 기존 task (MSN 없음) | 베이스라인 |
| B | 7개 + MSN | 전체 모델 |
| C | MSN만 | MSN 단독 기여도 |
| D | contrastive + feat + msn | 최소 핵심 조합 |

**권장 순서**: A → B를 먼저 비교. B가 A보다 downstream 성능이 높으면 MSN이 유효한 신호. C는 MSN이 독립적으로 의미있는 사전학습인지 확인용 (논문 ablation에 포함 가능).

**평가 지표**: downstream linear probing (알츠하이머 분류, 뇌 나이 예측 등) 또는 loss curve 수렴 속도 비교.

### 1.2 MSN feature 조합 탐색

MSN 생성에 사용할 feature 종류가 결과에 미치는 영향:

| MSN 변형 | feature | 약어 |
|----------|---------|------|
| MSN-5 (기본) | SurfArea, GrayVol, ThickAvg, MeanCurv, GausCurv | sa_gv_ta_mc_gc |
| MSN-T | ThickAvg, ThickStd만 | ta_ts |
| MSN-8 | 전체 8종 | sa_gv_ta_ts_mc_gc_fi_ci |
| MSN-2 | GrayVol, ThickAvg만 | gv_ta |

가장 downstream과 상관성이 높은 feature 조합을 찾는 것이 목표.

---

## 2. 모델 아키텍처 개선

### 2.1 MSN 예측 헤드 고도화

현재: CLS token → 2-layer MLP → 1891-dim upper triangle
**문제**: CLS token은 global pooling으로 압축된 정보. 뇌 영역별 공간 정보가 손실됨.

개선 방향:
- **Regional feature 기반 MSN 예측**: CLS token 대신 atlas segmentation으로 ROI를 마스킹한 region-level feature로 예측 (해석 가능성 ↑)
- **GNN decoder**: (n_regions) 노드에 region feature를 부여하고 GNN으로 MSN 엣지를 예측
- **Contrastive MSN**: 두 subject의 MSN 유사도를 contrastive loss로 학습 (동일 진단군끼리 가깝게)

### 2.2 인코더 스케일링

현재 Swin-T (feature_size=48, ~27M params)는 60k 규모 데이터에 비해 capacity가 작을 수 있다.

| 모델 | embed_dim | 파라미터 (추정) | 메모리/GPU |
|------|-----------|----------------|------------|
| Swin-T (현재) | 48 | ~27M | ~22GB |
| Swin-S | 96 | ~50M | ~30GB (추정, OOM 위험) |
| Swin-B | 128 | ~88M | DeepSpeed 필요 |

**현실적 방향**: batch_size_per_gpu=1 + gradient accumulation으로 Swin-S 테스트 가능.

### 2.3 Multi-scale Feature Aggregation

현재 feature/texture/msn head는 최종 레이어 x4만 사용. FPN 스타일로 다중 스케일 결합:

```python
x_ms = torch.cat([
    F.adaptive_avg_pool3d(x2, 1).flatten(1),  # 192-dim
    F.adaptive_avg_pool3d(x3, 1).flatten(1),  # 384-dim
    cls_token,                                  # 768-dim
], dim=1)  # 1344-dim
```

MSN, feat, texture head 모두 이 multi-scale feature를 입력으로 받으면 품질 향상 예상.

---

## 3. 학습 전략

### 3.1 Curriculum Learning

7+1개 task를 단계적으로 활성화하면 초반 불안정한 MSN loss 수렴을 방지할 수 있다:

```
Phase 1 (epoch 0~30):   rot + mim                      → 기본 시각 구조 학습
Phase 2 (epoch 30~100): + loc + contrastive             → 공간/글로벌 표현 학습
Phase 3 (epoch 100~):   + atlas + feat + texture + msn  → 도메인 지식 학습
```

AutoWeightedLoss가 자동으로 균형을 잡지만, 초기 MSN loss 스케일이 feat/texture와 크게 다를 경우 curriculum이 안정성을 높인다.

### 3.2 Loss 스케일 모니터링

MSN upper triangle은 [-1, 1] 범위의 Pearson 상관값이므로 L1 loss 스케일은 작다 (0~2 이하 예상). AutoWeightedLoss가 자동 조정하지만, W&B의 `iter/msn_loss`와 `w_msn` 가중치를 초기에 주의 깊게 모니터링할 것.

### 3.3 Teacher-Student (DINO 방식)

현재 SimCLR 방식 contrastive를 DINO/iBOT 방식으로 교체하면:
- Collapse 없는 학습 (prototypes 활용)
- CLS token 품질 향상 → MSN 예측 품질도 간접적으로 향상
- 코드 변경 규모: contrastive head + loss 교체 (~200줄 예상)

---

## 4. 평가 파이프라인

### 4.1 현재 없는 것: downstream evaluation

사전학습 품질을 정량화할 코드가 없다. 최소한으로 필요한 것:

```
pretrained_model → freeze encoder → linear head → fine-tune on labeled data
```

권장 downstream dataset:
- **알츠하이머 분류**: ADNI (CN vs MCI vs AD), label 구하기 쉬움
- **뇌 나이 예측**: 정상 피험자 age label, 회귀 문제
- **진단 분류**: ABIDE (자폐), ADHD-200

### 4.2 MSN 특화 평가

MSN loss의 효과를 직접 평가하는 방법:
1. **MSN 재구성 정확도**: 예측된 MSN vs 실제 MSN의 Pearson r
2. **MSN 기반 클러스터링**: 예측 MSN으로 피험자 클러스터링 → 진단군 분리 품질 (NMI, ARI)
3. **Graph downstream**: 예측 MSN을 GNN 입력으로 쓰는 graph-level 분류

### 4.3 빠른 품질 확인 (Linear Probing)

```bash
# 체크포인트에서 인코더만 추출 → linear probing
python linear_probe.py \
    --checkpoint runs_dict/checkpoint.pth \
    --data-path /path/to/downstream_data \
    --task age_regression
```

이 스크립트는 아직 없음. 필요시 작성 필요.

---

## 5. 데이터 관련

### 5.1 QC 필터링

현재 fomo60k_wo_scz (9,153명)는 QC 전 데이터. QC 후 데이터로 재학습 예정.
QC 기준: 움직임 아티팩트, FreeSurfer 실패, 이상 뇌 부피 등.

### 5.2 MSN 사전 생성

학습 전에 모든 subject의 MSN을 미리 생성해두는 것이 좋다:

```bash
python extract/msn_feat.py \
    --data-path /NFS/Users/kimyw/data/fomo60k_wo_scz \
    --data fomo60k_wo_scz \
    --features SurfArea GrayVol ThickAvg MeanCurv GausCurv
```

출력: `{data_path}/results/msn_sa_gv_ta_mc_gc/` (피험자당 .mat 파일).
MSN 파일이 없는 피험자는 zero 벡터로 대체되어 `msn_loss`가 자동 스킵된다.

### 5.3 데이터 증강 강화

현재 3D 증강이 crop + rotation으로 단순함. 추가 가능:
- Elastic deformation (해부학적 다양성)
- Random intensity bias field (MRI 아티팩트 시뮬레이션)

---

## 6. 코드베이스 정리

| 항목 | 상태 | 우선순위 |
|------|------|---------|
| linear_probe.py 작성 | 없음 | 🔴 높음 (논문 필수) |
| Hydra config 도입 | 없음 | 🟡 중간 |
| Docker/Singularity 환경 고정 | 없음 | 🟡 중간 |
| swin_unetr.py dead code 정리 | 미감사 | 🟢 낮음 |
| unit test 작성 | 없음 | 🟢 낮음 |

---

## 7. 논문 실험 계획

### 7.1 Ablation Study (최소 필요)

| 실험 | 제거 | 목적 |
|------|------|------|
| Full (A+MSN) | — | 최종 모델 |
| w/o MSN | msn_loss | MSN 기여도 |
| w/o Domain (feat+texture) | feat+texture | 도메인 지식 전반 기여도 |
| w/o Geometry (rot+loc) | rot+loc | 기하 학습 기여도 |
| w/o Contrastive | contrastive | 글로벌 표현 기여도 |
| MIM only | 나머지 전부 | 순수 MAE 베이스라인 |

### 7.2 비교 모델

| 방법 | 코드 | 비고 |
|------|------|------|
| MAE 3D | 공개 | 가장 가까운 베이스라인 |
| SwinMM | 공개 | 의료 3D SSL |
| Models Genesis | 공개 | 의료 3D SSL |
| UniMiSS | 공개 | 의료 3D SSL |

### 7.3 스케일링 실험

사전학습 데이터 크기에 따른 성능 변화:
- 1k / 5k / 10k / 30k / 60k subjects
- 이 실험만으로도 논문의 강력한 그림 하나

---

## 8. 참고 문헌

- Kendall et al. (2018). Multi-task learning using uncertainty to weigh losses. CVPR.
- He et al. (2022). Masked autoencoders are scalable vision learners. CVPR.
- Caron et al. (2021). Emerging properties in self-supervised vision transformers. ICCV.
- Tang et al. (2022). Self-supervised pre-training of swin transformers for 3d medical image analysis. CVPR.
- Yeo et al. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. J Neurophysiology. (MSN 개념의 기반)
- Seidlitz et al. (2018). Morphometric Similarity Networks Detect Microscale Cortical Organization. Neuron. (MSN 원조 논문)
