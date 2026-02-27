# DAMT: Domain Aware Multi-task Self-Supervised Pretraining of 3D Swin Transformer

> **작성일**: 2026년 2월 26일  
> **목적**: DAMT 코드베이스의 전체 작동 방식 요약  
> **대상 독자**: 본 프로젝트 연구자 (김영우)

---

## 1. 프로젝트 개요

DAMT는 뇌 MRI 영상에 대한 **Self-Supervised Learning (SSL) 백본 사전학습** 프레임워크이다. 3D Swin Transformer를 인코더로 사용하며, **7가지 pretext task**를 동시에 학습하는 **multi-task** 방식으로 설계되었다.

핵심 아이디어: 레이블 없이도 뇌 MRI의 구조적·통계적 특성을 다각도로 학습하여, 다운스트림 태스크(분류, 분할 등)에 강건한 representation을 획득한다.

```
┌─────────────────────────────────────────────────────────┐
│                    DAMT Pipeline                        │
│                                                         │
│  Brain MRI ──→ Data Augmentation ──→ 3D Swin Encoder    │
│                                          │              │
│           ┌──────────────────────────────┤              │
│           ▼          ▼         ▼         ▼              │
│      Rotation   Location  Contrast    MIM               │
│      (10cls)    (9cls)    (SimCLR)  (MAE-like)          │
│           ▼          ▼         ▼         ▼              │
│      Atlas Seg  Feature Reg  Texture Reg                │
│      (101cls)   (566-dim)    (72-dim)                   │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 데이터 입력 파이프라인

### 2.1 데이터 구조

각 피험자(subject) 디렉토리에는 FreeSurfer 전처리 결과물이 들어 있다:

| 파일 | 설명 |
|------|------|
| `mri/brainmask.nii.gz` | 뇌 마스크가 적용된 T1 MRI 볼륨 |
| `mri/aparc+aseg.nii.gz` | FreeSurfer parcellation (101개 영역 레이블) |

이에 더해, 프로젝트 루트에 3개의 CSV 파일이 존재한다:

| CSV 파일 | 차원 | 설명 |
|----------|------|------|
| `nfeats_global.csv` | 440 | 전역 형태학적 특징 (cortical thickness, volume 등) |
| `nfeats_local.csv` | 126 | 국소 형태학적 특징 |
| `radiomics_texture.csv` | 72 | 라디오믹스 텍스처 특징 |

→ Global features = 440 + 126 = **566차원**, Radiomics = **72차원**

### 2.2 데이터 로딩 (`datasets.py`)

```python
# 각 subject에 대해 다음 딕셔너리를 구성
{
    "image": "path/to/brainmask.nii.gz",
    "label": "path/to/aparc+aseg.nii.gz",
    "features": np.array([566-dim]),    # global + local features
    "radiomics": np.array([72-dim]),    # radiomics textures
}
```

- MONAI `Dataset`을 사용하여 `transform`(=`DataAugmentation`)을 lazy하게 적용
- atlas 파일이 없거나, CSV에 해당 subject가 없으면 자동으로 건너뜀

### 2.3 데이터 증강 (`DataAugmentation` 클래스, `main.py`)

DataAugmentation은 하나의 subject 딕셔너리를 받아 **6개 값의 튜플**을 반환한다:

```
(images, atlases, masks, radiomics, features, loc_true)
```

#### 전처리 파이프라인 (`load_image`)

```
LoadImaged → EnsureChannelFirstd → Lambdad(x[0:1]) → EnsureTyped
→ CropForegroundd → Spacingd(1.25mm³) → SpatialPadd(128³)
→ ScaleIntensityRangePercentilesd(0.05~99.95 → [0,1])
→ Lambdad(remap_atlas_labels)
```

**핵심 단계 설명**:
1. **`Lambdad(x[0:1])`**: MONAI가 자동으로 추가하는 채널을 1채널로 고정
2. **`CropForegroundd`**: 뇌 영역만 잘라내어 불필요한 배경 제거
3. **`Spacingd(1.25mm³)`**: 모든 볼륨을 동일 해상도로 리샘플링
4. **`SpatialPadd(128³)`**: 최소 128³이 되도록 패딩
5. **`remap_atlas_labels`**: FreeSurfer의 비연속적 레이블(0~2035, 101개)을 연속 인덱스 [0, 100]으로 매핑

#### 크롭 전략

| 크롭 유형 | 크기 | 용도 |
|-----------|------|------|
| **Global crop** | 128³ | 회전 예측, 대조 학습, 특징 회귀, 텍스처 회귀, Atlas 분할, MIM |
| **Local crop** | 56³ → resize 64³ | 위치 예측, Atlas 분할, MIM |

#### 위치 레이블 계산 (`_crop_local_with_location`)

Local crop의 중심 좌표를 기준으로 볼륨을 **3×3 그리드**로 나누어 **0~8의 위치 레이블**을 부여한다:

```
┌───┬───┬───┐
│ 0 │ 1 │ 2 │   첫 번째 공간축 (D)  
├───┼───┼───┤   × 
│ 3 │ 4 │ 5 │   두 번째 공간축 (H)
├───┼───┼───┤
│ 6 │ 7 │ 8 │
└───┴───┴───┘
```

`loc_label = bin_D × 3 + bin_H`

#### MIM 마스크 생성 (`MaskGenerator`)

- **Global**: 128³ 볼륨 → 16³ 패치 그리드 → 8³ = 512 토큰 중 75% 마스킹
- **Local**: 64³ 볼륨 → 16³ 패치 그리드 → 4³ = 64 토큰 중 75% 마스킹
- 마스킹 패턴은 `model_patch_size=2`에 맞게 업스케일됨

---

## 3. 모델 아키텍처 (`models.py`)

### 3.1 전체 구조

```
SSLHead_Swin
├── SwinViT (3D Swin Transformer 인코더)
│   ├── patch_embed: 2³ 패치 임베딩 → 48-dim
│   ├── layers1: depth=2, 48-dim → 96-dim
│   ├── layers2: depth=2, 96-dim → 192-dim
│   ├── layers3: depth=18, 192-dim → 384-dim  ← 가장 깊은 레이어
│   └── layers4: depth=2, 384-dim → 768-dim
│
├── Task-specific Heads
│   ├── rotation_head: Linear(768 → 10)
│   ├── location_head: Linear(768 → 9)
│   ├── contrastive_head: Linear(768 → 512)
│   ├── glo_feat_head: Conv3d+FC(768 → 566)
│   ├── texture_head: Conv3d+FC(768 → 72)
│   ├── decoder_mim: Conv3d → PixelShuffle3d
│   └── ConvDecoder + UnetOutBlock(48 → 101)
│
└── mask_token: 마스킹된 토큰을 대체할 학습 가능 파라미터
```

### 3.2 인코더: 3D Swin Transformer

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `in_chans` | 1 | 입력 채널 (T1 MRI) |
| `embed_dim` | 48 | 기본 임베딩 차원 |
| `patch_size` | 2³ | 패치 크기 |
| `window_size` | 7³ | 윈도우 어텐션 크기 |
| `depths` | [2, 2, 18, 2] | 각 스테이지의 Swin Block 수 |
| `num_heads` | [3, 6, 12, 24] | 각 스테이지의 어텐션 헤드 수 |
| `use_checkpoint` | True | Gradient checkpointing (메모리 절약) |

**출력**: 5단계의 hidden states `[x0, x1, x2, x3, x4]`
- `x0`: (B, 48, 64, 64, 64) — 패치 임베딩 직후
- `x1`: (B, 96, 32, 32, 32)
- `x2`: (B, 192, 16, 16, 16)
- `x3`: (B, 384, 8, 8, 8)
- `x4`: (B, 768, 4, 4, 4) — 최종 특징 맵

**CLS 토큰**: `x4`에 `AdaptiveAvgPool3d(1)`을 적용하여 (B, 768) 벡터 생성

### 3.3 두 가지 인코딩 경로

#### `encode(x)` — 일반 인코딩
```
입력 이미지 → SwinViT → [hidden_states, cls_token]
```

#### `encode_mask(x, mask)` — 마스크 인코딩 (MIM용)
```
입력 이미지 → patch_embed → 마스크 토큰으로 패치 교체
→ pos_drop → 각 layers 수동 통과 → [hidden_states, cls_token]
```

마스킹된 위치는 학습 가능한 `mask_token` 파라미터로 대체된다. 일반 `encode()`와 같은 SwinViT 가중치를 공유하지만, 마스킹 삽입을 위해 forward를 수동으로 진행한다.

### 3.4 디코더: ConvDecoder (Atlas 분할용)

U-Net 스타일의 skip connection 구조:

```
x4 (768) → encoder10 → decoder5 ──concat──→ x3 (384)
                                   │
                         decoder4 ──concat──→ x2 (192)
                                   │
                         decoder3 ──concat──→ x1 (96)
                                   │
                         decoder2 ──concat──→ x0 (48)
                                   │
                         decoder1 → (48-dim feature map)
                                   │
                         UnetOutBlock → (101-class logits)
```

최종 출력: (B, 101, D, H, W) — 각 복셀의 101-class parcellation 예측

### 3.5 MIM 디코더

```
x4 (768, 4, 4, 4) → Conv3d(768 → 32768) → PixelShuffle3d(scale=32)
→ 복원된 볼륨 (1, 128, 128, 128)
```

`PixelShuffle3d`는 채널 차원의 정보를 공간 차원으로 재배치하는 3D 확장 버전이다.

---

## 4. 학습 과정 (`main.py: train_one_epoch`)

### 4.1 데이터 흐름

```
DataLoader로부터 배치:
  (images, atlases, masks, radiomics, features, loc_trues)
       │          │        │        │          │        │
       ▼          ▼        ▼        ▼          ▼        ▼
  [glo_img,   [glo_atlas, [glo_mask,  (B,72)  (B,566)  (B,)
   loc_img]   loc_atlas]  loc_mask]
```

### 4.2 회전 증강

학습 전에 입력 이미지와 atlas에 **랜덤 3D 회전**을 적용한다 (`rot_rand`):

- 10가지 회전 중 하나를 랜덤 선택 (0=원본, 1~3=축1 회전, 4~6=축2 회전, 7~9=축3 회전)
- 이미지와 atlas에 동일한 회전을 적용
- 적용된 회전 방향이 rotation prediction의 정답 레이블이 됨

```python
x1, a1, rot1 = rot_rand(args, glo_x, glo_atlas)   # global view 1
x2, _ , _    = rot_rand(args, glo_x, glo_atlas)   # global view 2 (대조학습용)
x3, a3, rot2 = rot_rand(args, loc_x1, loc_atlas)  # local view
```

### 4.3 Forward Pass

모든 연산은 `torch.cuda.amp.autocast` (fp16 mixed precision) 내에서 수행된다.

#### Global Forward
```python
hidden_states_out1, cls_token1 = model.encode(x1)          # 회전된 global
hidden_states_out2, cls_token2 = model.encode_mask(x2, mask) # 마스킹된 global

rot1_p      = model.forward_rot(cls_token1)           # 회전 예측
texture_p   = model.forward_texture(hidden_states_out1[4])  # 텍스처 예측
glo_feat_p  = model.forward_global(hidden_states_out1[4])   # 특징 예측
glo_atlas_p = model.forward_decoder(hidden_states_out1)     # Atlas 분할
contrastive1_p = model.forward_contrastive(cls_token1)      # 대조 임베딩 1
contrastive2_p = model.forward_contrastive(cls_token2)      # 대조 임베딩 2
```

#### Local Forward
```python
hidden_states_out3, cls_token3 = model.encode(x3)           # 회전된 local
hidden_states_out4, _          = model.encode_mask(loc_x1, mask) # 마스킹된 local

rot2_p      = model.forward_rot(cls_token3)           # 회전 예측
loc_p       = model.forward_loc(cls_token3)           # 위치 예측
loc_atlas_p = model.forward_decoder(hidden_states_out3)     # Atlas 분할
```

### 4.4 제로 텐서 제거

atlas가 없는(전체가 0인) 샘플이나, feature/radiomics가 비어있는 샘플이 존재할 수 있다. `remove_zerotensor` 함수가 이들을 필터링하여 loss 계산에서 제외한다:

```python
glo_atlas_p, glo_atlas = remove_zerotensor(glo_atlas_p, a1)
loc_atlas_p, loc_atlas = remove_zerotensor(loc_atlas_p, a3)
texture_p, glo_radi    = remove_zerotensor(texture_p, glo_radi)
glo_feat_p, glo_feat   = remove_zerotensor(glo_feat_p, glo_feat)
```

---

## 5. Loss 함수 상세

### 5.1 7가지 Pretext Task Loss

| # | Task | Loss 함수 | 가중치 | 설명 |
|---|------|-----------|--------|------|
| 1 | **Rotation Prediction** | `CrossEntropyLoss` | 1.0 | 10-class 분류 (global + local 결합) |
| 2 | **Location Prediction** | `CrossEntropyLoss` | 1.0 | 9-class 분류 (local crop의 공간 위치) |
| 3 | **Contrastive Learning** | `NT-Xent (SimCLR)` | 1.0 | 같은 이미지의 두 view 간 유사도 최대화 |
| 4 | **Atlas Segmentation** | `CrossEntropyLoss` | 0.5 | 101-class 복셀 분할 (global + local 평균) |
| 5 | **Feature Regression** | `L1Loss × 5` | 5.0 | 566-dim 전역 특징 벡터 회귀 |
| 6 | **Texture Regression** | `L1Loss × 5` | 5.0 | 72-dim 라디오믹스 텍스처 회귀 |
| 7 | **Masked Image Modeling** | `L1Loss (masked)` | 1.0 | 마스킹된 패치만 복원 |

### 5.2 총 Loss

```python
loss = rot_loss + loc_loss + contrastive_loss + atlas_loss + feat_loss + texture_loss + mim_loss
```

- `atlas_loss = 0.5 × (glo_atlas_loss + loc_atlas_loss)`
- `feat_loss = 5 × L1(pred, target)` (sum이 0이면 skip)
- `texture_loss = 5 × L1(pred, target)` (sum이 0이면 skip)
- `mim_loss = mim_global + mim_local` (마스킹 영역만 L1 계산)

### 5.3 대조 학습 Loss (`Contrast` 클래스, `loss.py`)

NT-Xent (Normalized Temperature-scaled Cross Entropy) 방식:

```
z_i, z_j = L2_normalize(cls_token1), L2_normalize(cls_token2)
sim_matrix = cosine_similarity(z, z)  # z = [z_i; z_j]
loss = -log( exp(sim_pos/τ) / Σ exp(sim_neg/τ) )
```

- 온도 파라미터 τ = 0.5
- batch 내 다른 샘플은 모두 negative로 취급

### 5.4 MIM Loss (`forward_mim`)

```python
x_rec = decoder_mim(z4)           # 마스킹된 인코딩에서 원본 복원
loss = L1(x, x_rec) * mask        # 마스킹된 위치만 복원 손실 계산
loss = loss.sum() / (mask.sum() + 1e-5)
```

---

## 6. 최적화 전략

### 6.1 옵티마이저

- **AdamW** (weight_decay=0.04 → 0.4, cosine schedule)
- Linear scaling rule: `lr = base_lr × (batch_size × num_gpus) / 256`

### 6.2 스케줄러

| 항목 | 방식 | 시작 값 | 종료 값 |
|------|------|---------|---------|
| Learning Rate | Cosine + Warmup | 5e-4 (scaled) | 1e-7 |
| Weight Decay | Cosine | 0.04 | 0.4 |

- Warmup: 첫 5 에폭 동안 1e-6에서 목표 LR까지 선형 증가

### 6.3 Mixed Precision (FP16)

- `torch.cuda.amp.GradScaler`로 자동 스케일링
- NaN loss 발생 시 해당 배치를 건너뛰고 계속 학습 (`continue`)

### 6.4 Gradient Clipping

- 기본값 `--clip-grad=1.0`으로 gradient explosion 방지

### 6.5 분산 학습

- `torch.distributed.launch` + `DistributedDataParallel`
- `DistributedSampler`로 데이터 분할
- `find_unused_parameters=True`로 미사용 파라미터 허용

### 6.6 체크포인트

- 매 에폭: `checkpoint.pth` (최신 상태 덮어쓰기)
- 매 `saveckp_freq` 에폭(기본 20): `checkpoint{epoch:04}.pth`
- 학습 재개 시 `restart_from_checkpoint`로 자동 복원

---

## 7. 파일 구조 요약

| 파일 | 역할 | 핵심 함수/클래스 |
|------|------|------------------|
| `main.py` | 진입점, 학습 루프, 데이터 증강 | `main()`, `train_one_epoch()`, `DataAugmentation`, `MaskGenerator` |
| `models.py` | 모델 정의 | `SSLHead_Swin`, `ConvDecoder`, `PixelShuffle3d` |
| `datasets.py` | 데이터 로딩 | `get_brain_dataet()` |
| `loss.py` | Loss 함수 정의 | `Contrast`, `WeightedMSELoss`, `Loss` (미사용) |
| `ops.py` | 데이터 증강 연산 | `rot_rand()`, `aug_rand()`, `patch_rand_drop()` |
| `utils.py` | 유틸리티 (DDP, 스케줄러) | `cosine_scheduler()`, `MetricLogger`, `init_distributed_mode()` |
| `swin_unetr.py` | 3D Swin Transformer 구현 | `SwinTransformer`, `SwinUNETR` |

---

## 8. 학습 실행 방법

```bash
# 4-GPU 분산 학습 (RTX 4090 × 4 기준)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m torch.distributed.launch --nproc_per_node=4 main.py \
    --epochs 301 \
    --batch_size_per_gpu 2 \
    --warmup-epochs 5 \
    --data-path /path/to/dataset
```

### 검증된 학습 결과 (샘플 데이터 366명, 3 에폭)

| 에폭 | Loss | NaN 배치 | 비고 |
|------|------|----------|------|
| 0 | 1160.33 | 2 | |
| 1 | 1156.22 | 2 | |
| 2 | 1153.02 | 2 | 안정적 감소 확인 |

- GPU 메모리: ~16GB/GPU (batch=2)
- 체크포인트 크기: ~849MB

### 8.2 본격 학습 (Session 6, 2026-02-27)

**데이터**: fomo60k_wo_scz (9,153 subjects, QC 전 데이터)

| 항목 | 값 |
|------|------|
| GPU | 4×RTX4090 |
| batch_size_per_gpu | 2 |
| total_batch | 8 |
| epochs | 301 |
| iterations/epoch | ~1,145 |
| lr (effective) | 1.5625e-5 (5e-4 × 8/256) |
| warmup | 5 epochs |
| 모니터링 | W&B (project: DAMT-SSL, run: fomo60k_wo_scz_v1) |
| output_dir | ./runs_fomo60k |
| GPU 메모리 | ~22GB/GPU |

**Feature 추출**: `extract/extract_all_features.py` 스크립트를 통해 자동 생성
- nfeats_global.csv (9,153 × 440 features)
- nfeats_local.csv (9,153 × 126 features)
- radiomics_texture.csv (9,153 × 72 features)

**초기 loss 분포** (iteration 0):
| Task | Loss |
|------|------|
| rotation | 2.82 |
| location | 2.22 |
| contrastive | 1.46 |
| atlas | 4.72 |
| feature | 0.72 |
| texture | 252.18 |
| mim | 1.05 |
| **total** | **265.17** |

> texture_loss가 높은 이유: 정규화되지 않은 radiomics 특성의 스케일 차이. AutoWeightedLoss가 자동으로 가중치를 조정할 것으로 예상.

---

## 9. 주의사항 및 알려진 이슈

1. ~~**`loss.py`의 `Loss` 클래스는 현재 미사용**~~: Session 5에서 삭제됨. `AutoWeightedLoss`로 대체.

2. **NaN 배치**: FP16 학습에서 에폭당 ~2개 배치에서 NaN loss가 발생한다. 현재는 해당 배치를 건너뛰는 방식으로 처리 중이며, 학습 안정성에는 영향 없다.

3. ~~**`datasets.py` 경로 하드코딩**~~: Session 6에서 `--data-path` 인자 기본값을 fomo60k_wo_scz로 수정. `args.data_path`를 사용.

4. **`trainer.py` 미사용**: import 어디에서도 되지 않으며, 삭제 후보.

5. **Feature NaN 값**: 정규화 과정에서 분산이 0인 열에 NaN 발생 (global: ~26K, local: ~7.5K). `fillna(0)` + `nan_to_num()`으로 처리.

6. **W&B 모니터링**: `wandb` 패키지 설치 필요. main.py에서 rank 0에서만 init/log/finish 호출.

7. **Contrastive Loss 동적 배치 처리** (Session 7에서 수정): `Contrast` 클래스의 `neg_mask`가 고정 `batch_size`로 사전 계산되어, 에폭 마지막 배치(9153/4=2288.25 → 나머지 배치)에서 차원 불일치 RuntimeError 발생. `forward()`에서 `N = z_i.shape[0]`으로 동적 계산하도록 수정하여 가변 배치 크기를 지원. (commit: c37c31c)
