# FUTURE_WORK: DAMT 개선 및 발전 방안

> **작성일**: 2026년 2월 26일  
> **최종 수정일**: 2026년 2월 26일  
> **배경**: DAMT 코드 리뷰 및 검증 완료 후, NeurIPS 2026 제출을 위한 연구 방향 정리  
> **현재 상태**: AutoWeightedLoss 및 per-task 모니터링 적용 완료, 4-GPU 학습 검증 통과

---

## 1. 즉시 적용 가능한 개선사항

### 1.1 ~~Loss 함수 통합 및 리팩토링~~ ✅ 완료 (2026-02-26)

**구현 내용**:
- `loss.py`에 `AutoWeightedLoss` 클래스 구현 (Uncertainty Weighting, Kendall et al. 2018)
- 7가지 task별 학습 가능한 `log_vars` 파라미터로 자동 가중치 밸런싱
- 기존 `Loss` 클래스를 제거하고 `AutoWeightedLoss`로 대체
- `loss_function.state_dict()`를 체크포인트에 저장/복원하여 학습 재개 시 가중치 유지
- `AutoWeightedLoss` 파라미터를 optimizer param group에 추가

**검증 결과** (3 에폭, 4-GPU):
- texture loss(~230)가 지배적이지만, AutoWeightedLoss가 자동으로 가중치를 조절:
  - `texture`: 0.9989 (↓), `feat`: 1.0011 (↑), `mim`: 1.0013 (↑)
- 총 loss: 240.7 → 239.6 → 238.3 (안정적 감소)

**추후 고려**: GradNorm 방식으로 전환 가능성 검토 (gradient 기반 동적 밸런싱)

### 1.2 데이터셋 경로 설정 복원

- `datasets.py`에서 하드코딩된 테스트 경로를 제거하고 `args.data_path`를 사용하도록 복원
- 전체 데이터셋(60k)으로 전환 시 메모리/속도 확인 필요

### 1.3 ~~Loss 모니터링 개선~~ ✅ 완료 (2026-02-26)

**구현 내용**:
- `train_one_epoch`에서 7가지 task별 loss를 `metric_logger`에 개별 기록
- 기록 항목: `rot_loss`, `loc_loss`, `contrastive_loss`, `atlas_loss`, `feat_loss`, `texture_loss`, `mim_loss`
- 에폭 종료 시 `AutoWeightedLoss`의 학습된 가중치(`w_*`)도 로그에 저장
- `log.txt`에 JSON 형식으로 모든 per-task loss + 가중치 기록

**로그 예시** (epoch 0):
```json
{"train_loss": 240.70, "train_rot_loss": 2.42, "train_loc_loss": 1.89,
 "train_contrastive_loss": 1.21, "train_atlas_loss": 4.32,
 "train_feat_loss": 0.79, "train_texture_loss": 229.23,
 "train_mim_loss": 0.86, "train_w_texture": 0.9996, ...}
```

**추후 고려**: TensorBoard 또는 WandB 연동으로 실시간 시각화

---

## 2. 모델 아키텍처 개선

### 2.1 인코더 개선

#### a) 더 큰 Swin Transformer

현재 `feature_size=48`은 Swin-Tiny 수준이다. 데이터가 충분하다면:

| 모델 | embed_dim | depths | 파라미터 수 (추정) |
|------|-----------|--------|-------------------|
| Swin-T (현재) | 48 | [2,2,18,2] | ~27M |
| Swin-S | 48 | [2,2,18,2] + wider | ~50M |
| Swin-B | 96 | [2,2,18,2] | ~88M |

**주의**: RTX 4090 24GB × 4 환경에서 batch_size=2로 이미 ~16GB 사용 중이므로, 모델 크기 증가 시 gradient checkpointing 최적화 또는 DeepSpeed/FSDP 도입이 필요하다.

#### b) 하이브리드 인코더

- **CNN stem + Swin**: 초기 레이어를 CNN으로 대체하여 local feature 추출 강화
- **3D ConvNeXt V2 + MAE**: 최신 CNN 아키텍처와 MAE를 결합

#### c) Relative Position Bias 개선

현재 Swin Transformer는 고정된 relative position bias를 사용한다. 이를 **conditional position encoding** (CPE)이나 **rotary position embedding** (RoPE)으로 교체하면, 다양한 입력 크기에 더 유연하게 대응할 수 있다.

### 2.2 디코더 개선

#### a) Atlas 분할 디코더

현재 단순 Conv + Upsample 구조를 **nnU-Net 스타일 디코더**나 **Swin UNETR 디코더**로 교체할 수 있다. 이미 `swin_unetr.py`에 `SwinUNETR` 구현이 포함되어 있으므로 활용 가능하다.

#### b) MIM 디코더

현재 Conv3d + PixelShuffle 구조이다. **경량 Transformer 디코더** (MAE 스타일)를 사용하면 복원 품질이 개선될 수 있다:
- 인코더 출력 토큰 + 마스크 토큰 → 소규모 Transformer 블록 → 복원

### 2.3 Multi-scale Feature 활용

현재 feature/texture regression은 `x4` (최종 레이어)만 사용한다. **Feature Pyramid Network (FPN)** 방식으로 다중 스케일 특징을 결합하면 더 풍부한 representation이 가능하다:

```python
# 예시: multi-scale feature aggregation
x_multi = torch.cat([
    F.adaptive_avg_pool3d(x2, 1).flatten(1),
    F.adaptive_avg_pool3d(x3, 1).flatten(1),
    F.adaptive_avg_pool3d(x4, 1).flatten(1),
], dim=1)
```

---

## 3. Pretext Task 개선 및 추가

### 3.1 기존 Task 개선

#### a) 위치 예측 고도화

**현재**: 3×3=9 위치 (2D 그리드, 3번째 축 무시)

**개선**:
- **3×3×3=27 위치**: 세 번째 공간축도 포함하여 완전한 3D 위치 예측
- **연속 좌표 회귀**: 이산적 bin 대신 정규화된 (x, y, z) 좌표를 직접 회귀
- **상대 위치 예측**: 두 local crop 간의 상대적 위치를 예측

#### b) 회전 예측 고도화

**현재**: 10-class 이산 회전

**개선**:
- **연속 회전 회귀**: 쿼터니언 또는 rotation matrix 예측
- **3D Jigsaw Puzzle**: 볼륨을 여러 블록으로 분할하고 원래 순서를 예측

#### c) 대조 학습 개선

**현재**: SimCLR 스타일 (batch 내 negative)

**개선**:
- **DINO/iBOT**: Teacher-Student 구조 + EMA 업데이트로 collapse 없는 학습
- **VICReg**: Variance-Invariance-Covariance 정규화로 trivial solution 방지
- **Barlow Twins**: cross-correlation matrix를 identity에 가깝게 만드는 방식

### 3.2 새로운 Pretext Task 추가

#### a) Cross-modal Prediction

뇌 MRI의 **다른 modality를 예측**하는 task:
- T1 → T2 변환 예측 (데이터 확보 시)
- 구조적 MRI에서 DTI metric (FA, MD) 예측

#### b) Anatomical Consistency

- **좌우 대칭 예측**: 뇌의 좌반구와 우반구 간 대칭 관계 학습
- **볼륨 비율 예측**: 각 뇌 영역의 상대적 크기 분포 예측

#### c) Temporal Consistency (종단 데이터 활용 시)

- 같은 피험자의 다른 시점 스캔 간 일관성 학습
- 나이 예측 또는 뇌 변화 방향 예측

#### d) Graph-based Pretext Task

- 뇌 영역 간의 **구조적 연결성 그래프** 예측
- GNN을 보조 모듈로 활용하여 영역 간 관계 학습

---

## 4. 학습 전략 개선

### 4.1 Curriculum Learning

7가지 task를 **단계적으로 활성화**:

```
Phase 1 (epoch 0-50):   Rotation + MIM (기본 시각 특징 학습)
Phase 2 (epoch 50-150): + Contrastive + Location (공간 관계 학습)
Phase 3 (epoch 150-301): + Atlas + Feature + Texture (도메인 지식 학습)
```

이렇게 하면 초기에 기본적인 시각 표현을 안정적으로 학습한 뒤, 점진적으로 복잡한 도메인 지식을 추가할 수 있다.

### 4.2 Progressive Resizing

- 초기: 작은 crop 크기 (96³/48³)로 빠른 학습
- 중기: 현재 크기 (128³/64³)
- 후기: 더 큰 crop 크기 (160³/80³)로 fine-grained 학습

### 4.3 EMA (Exponential Moving Average) 모델

- Teacher 모델을 EMA로 유지하여 더 안정적인 representation 제공
- DINO/BYOL 방식의 self-distillation 적용 가능

### 4.4 Batch Size 최적화

현재 batch=2/GPU (total=8)는 대조 학습에 부족할 수 있다.

**대안**:
- **Memory Bank**: 이전 배치의 representation을 저장하여 사실상의 batch size 증가
- **Gradient Accumulation**: 여러 step의 gradient를 누적하여 effective batch size 증가
- **MoCo 방식 Queue**: momentum encoder의 출력을 queue에 저장

---

## 5. 데이터 관련 개선

### 5.1 데이터 증강 강화

현재 증강은 비교적 단순하다 (crop + rotation + intensity jitter). 추가 가능한 3D 증강:

- **Elastic deformation**: 비선형 변형으로 해부학적 다양성 증가
- **Random bias field**: MRI 특유의 intensity inhomogeneity 시뮬레이션
- **Mixup/CutMix 3D 버전**: 볼륨 간 혼합으로 정규화 효과
- **Random erasing/inpainting**: MIM과 시너지

### 5.2 Feature 품질 개선

현재 CSV 기반 feature (566-dim, 72-dim)는 사전 추출된 정적 값이다:

- 학습 중 **online feature extraction**으로 전환 가능성 검토
- 더 풍부한 radiomics feature 추출 (PyRadiomics 활용)
- Feature 정규화 전략 (z-score, quantile transform 등)

### 5.3 대규모 데이터셋 활용

- 현재 테스트 데이터: 366명
- 목표: 60,000명 (`fomo60k_wo_scz`)
- **대규모 데이터로의 전환 시 고려사항**:
  - DataLoader의 `num_workers` 최적화
  - 데이터 캐싱 전략 (MONAI `PersistentDataset` 또는 `CacheDataset`)
  - 분산 파일시스템 I/O 병목 확인

---

## 6. 평가 및 다운스트림 파이프라인

### 6.1 사전학습 품질 평가

SSL 사전학습의 품질을 정량적으로 평가하기 위한 프로토콜:

- **Linear probing**: 인코더를 고정하고 linear classifier만 학습
- **k-NN evaluation**: feature 공간에서 k-NN 분류 정확도
- **t-SNE/UMAP 시각화**: 학습된 representation의 클러스터링 품질 확인

### 6.2 다운스트림 태스크

사전학습된 백본을 활용할 수 있는 주요 태스크:

| 태스크 | 유형 | 데이터셋 예시 |
|--------|------|-------------|
| 뇌 나이 예측 | 회귀 | 자체 데이터 |
| 질환 분류 | 분류 | ADNI (알츠하이머), ABIDE (자폐) |
| 뇌 분할 | 분할 | FreeSurfer 대비 |
| 이상 탐지 | 비지도 | 정상 vs 병리 |

### 6.3 Few-shot / Zero-shot 평가

- 소량 레이블 데이터(1%, 10%)로 fine-tuning 시 성능 비교
- Task-agnostic representation의 범용성 입증

---

## 7. 논문 작성을 위한 실험 계획

### 7.1 Ablation Study

각 pretext task의 기여도를 검증하기 위한 체계적 실험:

| 실험 | 제거 task | 목적 |
|------|-----------|------|
| Full model | - | 기준선 |
| w/o Rotation | Rotation | 3D 방향 인식의 기여도 |
| w/o Location | Location | 공간 위치 인식의 기여도 |
| w/o Contrastive | Contrastive | 글로벌 representation의 기여도 |
| w/o Atlas | Atlas | 해부학적 지식의 기여도 |
| w/o Feature | Feature regression | 형태학적 특징의 기여도 |
| w/o Texture | Texture regression | 텍스처 특징의 기여도 |
| w/o MIM | MIM | 저수준 시각 특징의 기여도 |

### 7.2 비교 실험

다른 SSL 방법론과의 비교:

- **MAE 3D** (He et al., 2022의 3D 확장)
- **DINO 3D** (Caron et al., 2021의 3D 확장)
- **SwinMM** (Wang et al., 2023)
- **UniMiSS** (Xie et al., 2022)
- **Models Genesis** (Zhou et al., 2021)

### 7.3 스케일링 실험

- 사전학습 데이터 크기에 따른 성능 변화 (1k, 5k, 10k, 30k, 60k)
- 모델 크기에 따른 성능 변화 (Swin-T, Swin-S, Swin-B)
- 학습 에폭에 따른 수렴 분석

---

## 8. 코드 품질 개선

### 8.1 설정 관리
- `argparse` 대신 **Hydra** 또는 **dataclass 기반 config** 도입
- YAML 파일로 실험 설정 관리

### 8.2 실험 추적
- **WandB** 또는 **MLflow** 연동으로 실험 자동 추적
- 하이퍼파라미터, loss curve, 체크포인트를 한 곳에서 관리

### 8.3 테스트 코드
- 각 모듈별 unit test 작성
- 데이터 파이프라인 검증 스크립트 별도 관리

### 8.4 재현성
- Docker/Singularity 컨테이너로 환경 고정
- `requirements.txt` 또는 `conda env export` 명확히 관리
- 시드 고정 검증 (현재 `seed_everything` 함수 존재하나 실제 호출 여부 확인 필요)

---

## 9. 우선순위 정리

| 우선순위 | 항목 | 난이도 | 영향력 | 비고 |
|----------|------|--------|--------|------|
| ✅ 완료 | Loss 자동 가중치 밸런싱 | 중 | 높음 | AutoWeightedLoss 구현 완료 (2026-02-26) |
| ✅ 완료 | Task별 loss 모니터링 | 하 | 높음 | 7개 task + 가중치 로깅 완료 (2026-02-26) |
| 🔴 높음 | 전체 데이터셋 전환 | 하 | 높음 | 경로 변경만으로 가능 |
| 🟡 중간 | 대조 학습 → DINO/iBOT | 중 | 높음 | 논문 novelty |
| 🟡 중간 | 위치 예측 3D 확장 (27-class) | 하 | 중 | 간단한 개선 |
| 🟡 중간 | Ablation study 자동화 | 중 | 높음 | 논문 필수 |
| 🟡 중간 | 데이터 증강 강화 | 중 | 중 | elastic deform 등 |
| 🟢 낮음 | Hydra config 도입 | 중 | 낮음 | 코드 품질 |
| 🟢 낮음 | WandB 연동 | 하 | 중 | 편의성 |
| 🟢 낮음 | 모델 크기 스케일링 | 높 | 중 | 리소스 필요 |

---

## 10. 참고 문헌

- Kendall, A. et al. (2018). "Multi-task learning using uncertainty to weigh losses." CVPR.
- Chen, Z. et al. (2018). "GradNorm: Gradient normalization for adaptive loss balancing." ICML.
- He, K. et al. (2022). "Masked autoencoders are scalable vision learners." CVPR.
- Caron, M. et al. (2021). "Emerging properties in self-supervised vision transformers." ICCV.
- Tang, Y. et al. (2022). "Self-supervised pre-training of swin transformers for 3d medical image analysis." CVPR.
- Wang, Z. et al. (2023). "SwinMM: Masked multi-view with swin transformers for 3d medical image segmentation." MICCAI.
