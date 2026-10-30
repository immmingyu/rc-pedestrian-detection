"""
RC카 보행자 탐지용 커스텀 디텍터 학습 스크립트.

데이터셋: archive/{train,valid,test}/{같은 이름}/ 아래 이미지 + _annotations.csv
          (filename,width,height,class,xmin,ymin,xmax,ymax / pascal_voc 절대좌표)

핵심 수정 사항 (원본 노트북 대비):
  - build_target / loss / decode의 wh 인코딩을 log(bw/W) 기준으로 통일.
    (원본은 target에 bw/W를 저장하고 loss는 sigmoid(pred)와 비교했는데,
     추론 decode는 exp(pred)*W로 복원해서 학습-추론 수식이 어긋났음.
     그 결과 박스 크기가 신뢰할 수 없게 나와 몸통/얼굴이 따로 잡히는 원인이 됨.
     지금은 loss도 raw pred와 log(bw/W)를 그대로 비교하므로 decode의 exp가
     정확한 역함수가 됨)
  - PersonDataset이 매 __getitem__마다 전체 DataFrame을 스캔하지 않도록
    파일명 -> 박스 배열 dict를 미리 그룹핑.
  - mAP@0.5 평가지표 도입 (기존엔 conf=0.25 한 점에서의 recall만 측정).
  - ImageNet 사전학습 MobileNetV3-Small 백본 옵션 (--backbone mobilenet),
    사전학습 백본에는 차등 학습률(backbone lr * --backbone-lr-mult) 적용.
    mobilenet 백본일 때만 입력에 ImageNet 정규화(mean/std) 적용 (custom은 0~1 유지).
  - TorchScript 변환 시 example 입력 shape 버그 수정 (H, W 순서).
  - [2단계] 다중 양성 셀 할당: GT박스 중심 셀 + 인접 최대 2셀(가로/세로 각 1개)에 동시에
    정답 할당 (YOLOv5 스타일). 오프셋 범위가 [0,1) -> [-0.5,1.5)로 확장되어 loss/decode
    모두 sigmoid(x)*2-0.5 사용 (detect_video.py도 동일하게 맞춰야 함).
  - [2단계] RandomSizedBBoxSafeCrop으로 스케일 증강 추가 (클로즈업/큰 스케일 사람 비중 증가).
"""
import argparse
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import albumentations as A

# 백본 출력 1셀이 입력 몇 px에 대응하는지(기본값). 8로 낮추면 그리드가 가로세로 2배
# (셀 4배)로 촘촘해져 작은 사람의 중심을 더 정확히 잡을 수 있지만 연산량이 늘어난다
# (mobilenet 백본 320x240 실측: 38.7M -> 72.8M MACs = 1.88배, 파라미터 수는 동일).
GRID_STRIDE = 16

# mobilenet 백본은 ImageNet 사전학습 가중치를 쓰므로 그 학습 당시 분포(정규화)를 맞춰야
# 사전학습 feature가 제대로 작동함. custom 백본은 처음부터 랜덤 초기화라 0~1 그대로 사용.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------
# 데이터셋
# --------------------------------------------------------------------------
def split_paths(data_root, split):
    d = os.path.join(data_root, split, split)
    return d, os.path.join(d, "_annotations.csv")


def unsharp_mask(image, sigma=1.0, alpha=0.5):
    """언샤프 마스크: 원본 + alpha * (원본 - 가우시안블러).

    블러로 저주파 성분을 뽑아내 원본에서 빼면 윤곽(고주파)만 남고,
    그걸 원본에 더해 경계를 또렷하게 만든다. alpha가 크면 윤곽이 강해지지만
    노이즈도 같이 증폭된다.

    학습(PersonDataset)과 추론(detect_video.py)에서 반드시 같은 sigma/alpha를
    써야 한다. 한쪽만 적용하면 입력 분포가 어긋나 성능이 떨어진다.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + alpha, blurred, -alpha, 0)


def build_train_transform(img_w, img_h, scale_crop_p=0.3):
    """scale_crop_p=0으로 주면 RandomSizedBBoxSafeCrop을 빼고 구성한다.
    Mosaic이 적용된 샘플은 이미 스케일이 크게 흔들린 상태라 확대 크롭까지 겹치면
    과해지므로, Mosaic 경로에는 이 값을 낮추거나 0으로 준 파이프라인을 쓴다."""
    augs = [A.HorizontalFlip(p=0.5)]

    if scale_crop_p > 0:
        # 클로즈업(큰 스케일) 사람 비중을 늘리기 위한 확대 크롭: 박스를 보존하는 랜덤
        # 크기로 자른 뒤 원래 해상도로 다시 리사이즈 -> 사람이 화면에서 차지하는 비율이 커짐
        augs.append(A.RandomSizedBBoxSafeCrop(height=img_h, width=img_w, erosion_rate=0.0, p=scale_crop_p))

    augs += [
        A.RandomBrightnessContrast(p=0.3),
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-5, 5), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 20), hole_width_range=(8, 20), p=0.2),
    ]

    return A.Compose(augs, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))


class PersonDataset(Dataset):
    def __init__(self, image_dir, csv_file, img_w, img_h, max_images=None, transform=None,
                 normalize=False, mosaic_prob=0.0, transform_mosaic=None,
                 unsharp=False, unsharp_sigma=1.0, unsharp_alpha=0.5):
        self.image_dir = image_dir
        self.img_w = img_w
        self.img_h = img_h
        self.transform = transform
        self.normalize = normalize
        # 언샤프 마스크 전처리. train/valid 양쪽 + detect_video.py까지 같은 값을 써야 한다.
        self.unsharp = unsharp
        self.unsharp_sigma = unsharp_sigma
        self.unsharp_alpha = unsharp_alpha
        # Mosaic 적용 확률. transform_mosaic이 주어지면 Mosaic 샘플엔 그 파이프라인을 쓴다.
        self.mosaic_prob = mosaic_prob
        self.transform_mosaic = transform_mosaic

        df = pd.read_csv(csv_file)
        df = df[df["class"] == "person"]

        # 파일명별로 미리 그룹핑 -> __getitem__마다 O(N) 스캔하는 것을 방지
        self.boxes_by_file = {
            fname: g[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float32)
            for fname, g in df.groupby("filename")
        }
        self.image_list = sorted(self.boxes_by_file.keys())

        if max_images is not None:
            random.seed(32)
            self.image_list = random.sample(self.image_list, min(max_images, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def _load_resized(self, idx):
        """이미지 1장을 읽어 (img_w, img_h)로 리사이즈하고 박스도 같은 비율로 변환."""
        filename = self.image_list[idx]
        boxes = self.boxes_by_file[filename].copy()

        path = os.path.join(self.image_dir, filename)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # [13단계] 언샤프 마스크는 리사이즈 전, 원본 해상도에서 적용한다.
        # 5단계에서는 리사이즈(320x240) 후에 적용했었는데, 그러면 sigma=1.0인
        # 가우시안 블러의 반경이 원본 기준으로는 훨씬 큰 영역에 걸쳐 저주파를 뽑아내는
        # 셈이 되어(다운스케일로 이미 뭉개진 픽셀 단위에서 블러를 또 하는 것) 실제 텍스처가
        # 아니라 리사이즈 자체가 만든 앨리어싱을 강조했을 가능성이 있다. 원본에서
        # 샤프닝한 뒤 리사이즈하면 사람 윤곽 같은 진짜 고주파 성분을 원래 스케일에서
        # 강조하고, 그 다음 다운스케일이 자연스러운 안티앨리어싱 역할을 한다.
        if self.unsharp:
            image = unsharp_mask(image, self.unsharp_sigma, self.unsharp_alpha)

        valid = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
        boxes = boxes[valid]

        # 원본(최대 수천px) 대신 320x240으로 먼저 줄인 뒤 augmentation을 적용함.
        # 순서를 바꾸면 (1) augmentation 비용이 원본 해상도에 비례해 커지는 문제가 없어지고,
        # (2) CoarseDropout의 절대 픽셀(8~20px) 구멍 크기가 최종 학습 해상도 기준으로
        #     의도한 만큼(작은 가림)의 비중을 가지게 됨 (원본 대형 이미지 기준이면 사실상 안 보이는 크기였음)
        h, w = image.shape[:2]
        sx, sy = self.img_w / w, self.img_h / h
        image = cv2.resize(image, (self.img_w, self.img_h))
        if len(boxes) > 0:
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] * sx, 0, self.img_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] * sy, 0, self.img_h)

        return image, boxes

    def _build_mosaic(self, idx):
        """4장을 2x2로 붙인 뒤 다시 (img_w, img_h)로 축소한다.
        결과적으로 각 사람이 절반 크기로 줄어들어 작은 객체 학습 샘플이 늘어난다."""
        indices = [idx] + [random.randrange(len(self.image_list)) for _ in range(3)]
        W, H = self.img_w, self.img_h

        canvas = np.zeros((H * 2, W * 2, 3), dtype=np.uint8)
        # (x_offset, y_offset): 좌상, 우상, 좌하, 우하
        offsets = [(0, 0), (W, 0), (0, H), (W, H)]
        merged = []

        for i, (ox, oy) in zip(indices, offsets):
            img_i, boxes_i = self._load_resized(i)
            canvas[oy:oy + H, ox:ox + W] = img_i
            if len(boxes_i) > 0:
                shifted = boxes_i.copy()
                shifted[:, [0, 2]] += ox
                shifted[:, [1, 3]] += oy
                merged.append(shifted)

        # 2배 캔버스를 원래 크기로 축소 -> 박스 좌표도 정확히 절반
        image = cv2.resize(canvas, (W, H))
        if merged:
            boxes = np.concatenate(merged, axis=0) * 0.5
            # 축소로 1px 미만이 된 박스는 학습에 해로우므로 제거
            keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
            boxes = boxes[keep].astype(np.float32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)

        return image, boxes

    def __getitem__(self, idx):
        use_mosaic = self.mosaic_prob > 0 and random.random() < self.mosaic_prob

        if use_mosaic:
            image, boxes = self._build_mosaic(idx)
            transform = self.transform_mosaic if self.transform_mosaic is not None else self.transform
        else:
            image, boxes = self._load_resized(idx)
            transform = self.transform

        # 언샤프 마스크는 _load_resized 안, 리사이즈 전(원본 해상도)에서 이미 적용됨.
        # Mosaic 경로도 _build_mosaic이 조각마다 _load_resized를 호출하므로 각 조각이
        # 자기 원본 해상도에서 한 번씩 샤프닝된다.

        if transform is not None and len(boxes) > 0:
            transformed = transform(image=image, bboxes=boxes, labels=np.ones(len(boxes)))
            image = transformed["image"]
            boxes = np.array(transformed["bboxes"], dtype=np.float32)

        image = image.astype(np.float32) / 255.0
        if self.normalize:
            image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = np.transpose(image, (2, 0, 1))

        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(boxes, dtype=torch.float32)


def collate_fn(batch):
    images = torch.stack([img for img, _ in batch], dim=0)
    boxes_list = [boxes for _, boxes in batch]
    return images, boxes_list


# --------------------------------------------------------------------------
# 타겟 생성 (그리드 좌표 인코딩)
# --------------------------------------------------------------------------
def get_grid_size(img_w, img_h, stride=GRID_STRIDE):
    return img_w // stride, img_h // stride


# 2레벨 FPN: MobileNetV3-Small features의 어느 인덱스를 뽑을지와 그 stride.
# features[3] -> /8 (24채널), features[4] -> /16 (40채널).
FPN_TAPS = (3, 4)
FPN_STRIDES = (8, 16)


def get_level_grids(img_w, img_h, fpn, grid_stride):
    """레벨별 (grid_w, grid_h) 리스트. 단일 스케일이면 원소 1개."""
    strides = FPN_STRIDES if fpn else (grid_stride,)
    return [get_grid_size(img_w, img_h, s) for s in strides]


def split_boxes_by_level(boxes, boundaries):
    """박스 높이(px)로 담당 레벨을 정한다. boundaries=[64.0]이면 [<64, >=64] 2그룹.

    FCOS 관행대로 크기 기준으로 나눈다. 작은 사람은 셀이 촘촘한 고해상도 레벨(/8)이,
    큰 사람은 수용 영역이 넓은 저해상도 레벨(/16)이 담당하게 해서 레벨마다 역할을
    특화시킨다. 경계를 잘못 잡으면 특정 크기대가 어느 레벨에서도 학습되지 않으므로
    학습 중 레벨별 양성 셀 수를 반드시 확인할 것.
    """
    n_levels = len(boundaries) + 1
    if len(boxes) == 0:
        return [boxes.new_zeros((0, 4)) for _ in range(n_levels)]

    heights = boxes[:, 3] - boxes[:, 1]
    groups = []
    lo = 0.0
    for hi in list(boundaries) + [float("inf")]:
        groups.append(boxes[(heights >= lo) & (heights < hi)])
        lo = hi
    return groups


def build_targets_multi(boxes, grids, img_w, img_h, boundaries):
    """레벨별 타겟 텐서 리스트. 단일 스케일(grids 길이 1)이면 boundaries가 비어
    기존 build_target과 완전히 동일한 결과를 낸다."""
    groups = split_boxes_by_level(boxes, boundaries)
    return [build_target(g, gw, gh, img_w, img_h) for g, (gw, gh) in zip(groups, grids)]


def stack_level_targets(boxes_list, grids, img_w, img_h, boundaries, device):
    """배치 안 각 샘플의 레벨별 타겟을 레벨 축으로 묶어 [level] -> (B,5,H,W)로 변환."""
    per_sample = [build_targets_multi(b, grids, img_w, img_h, boundaries) for b in boxes_list]
    return [
        torch.stack([s[level] for s in per_sample]).to(device)
        for level in range(len(grids))
    ]


def as_levels(x):
    """단일 텐서를 원소 1개짜리 튜플로 정규화 (단일 스케일/FPN 코드 경로 통일용)."""
    return x if isinstance(x, (list, tuple)) else (x,)


def build_target(boxes, grid_w, grid_h, img_w, img_h):
    """다중 양성 셀 할당(YOLOv5 스타일): 중심 셀 + 인접 최대 2셀(가로/세로 각 1개, 박스
    중심이 셀의 어느 반쪽에 있는지로 결정)에 동시에 정답을 할당함. 양성 샘플 수가
    늘어나 recall이 개선됨. 인접 셀은 자기 기준 오프셋이 [0,1) 밖(약 -0.5~1.5)으로
    나갈 수 있어서, decode/loss 쪽도 sigmoid(x)*2-0.5로 범위를 맞춰야 함 (decode_prediction 참고)."""
    target = torch.zeros((5, grid_h, grid_w), dtype=torch.float32)
    occupied = torch.zeros((grid_h, grid_w), dtype=torch.bool)

    cell_w = img_w / grid_w
    cell_h = img_h / grid_h

    for box in boxes:
        xmin, ymin, xmax, ymax = box.tolist()
        bw = xmax - xmin
        bh = ymax - ymin
        if bw <= 1 or bh <= 1:
            continue

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        gx_f = cx / cell_w
        gy_f = cy / cell_h
        gx, gy = int(gx_f), int(gy_f)
        if gx < 0 or gy < 0 or gx >= grid_w or gy >= grid_h:
            continue

        frac_x = gx_f - gx
        frac_y = gy_f - gy

        cells = [(gx, gy)]
        if frac_x < 0.5 and gx - 1 >= 0:
            cells.append((gx - 1, gy))
        elif frac_x >= 0.5 and gx + 1 < grid_w:
            cells.append((gx + 1, gy))
        if frac_y < 0.5 and gy - 1 >= 0:
            cells.append((gx, gy - 1))
        elif frac_y >= 0.5 and gy + 1 < grid_h:
            cells.append((gx, gy + 1))

        for cgx, cgy in cells:
            if occupied[cgy, cgx]:
                continue
            occupied[cgy, cgx] = True

            target[0, cgy, cgx] = 1.0
            target[1, cgy, cgx] = gx_f - cgx
            target[2, cgy, cgx] = gy_f - cgy
            # log 인코딩: decode의 exp(pred)*W와 정확히 역함수 관계가 되도록 함
            target[3, cgy, cgx] = math.log(bw / img_w)
            target[4, cgy, cgx] = math.log(bh / img_h)

    return target


# --------------------------------------------------------------------------
# 모델
# --------------------------------------------------------------------------
class DSConv(nn.Module):
    """Depthwise Separable Conv: 엣지 디바이스(RPi4)를 겨냥한 경량 conv 블록."""

    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out, inplace=True)


#  MACs 실측(320x240, stride16, neck 재투영 비용 포함):
#    (32,64,128,256) 기본값        1490.84M  (팀원 모델과 동일 체급)
#    (16,24,32,64)   "lite" 후보     141.56M  (9단계 FPN 52.51M의 2.70배)
CUSTOM_CHANNELS_DEFAULT = (32, 64, 128, 256)


class CustomBackbone(nn.Module):
    """DSConv + ResidualBlock. channels=(stem, layer1, layer2, layer3=layer4=out) 채널 구성.
    stride 16이면 320x240 -> 20x15, stride 8이면 layer3의 다운샘플을 빼서 40x30.

    fpn=True면 layer2(/8)와 layer4(/16) 두 지점을 리스트로 반환한다(MobileNetBackbone의
    taps와 같은 역할). 이 경우 두 해상도를 모두 얻어야 하므로 stride 인자와 무관하게
    layer3은 항상 다운샘플한다."""

    def __init__(self, stride=16, channels=CUSTOM_CHANNELS_DEFAULT, fpn=False):
        super().__init__()
        self.fpn = fpn
        if not fpn and stride not in (8, 16):
            raise ValueError(f"지원하지 않는 stride: {stride} (8 또는 16)")
        s0, s1, s2, s3 = channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, s0, 3, 2, 1, bias=False),
            nn.BatchNorm2d(s0),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(DSConv(s0, s1, 2), ResidualBlock(s1))
        self.layer2 = nn.Sequential(DSConv(s1, s2, 2), ResidualBlock(s2))
        # 단일 스케일 stride 8: 여기서 한 번 덜 줄여서 출력 해상도를 2배로 유지.
        # FPN은 /8, /16을 둘 다 뽑아야 하므로 항상 다운샘플(stride 2)한다.
        layer3_stride = 2 if (fpn or stride == 16) else 1
        self.layer3 = nn.Sequential(DSConv(s2, s3, layer3_stride), ResidualBlock(s3))
        self.layer4 = nn.Sequential(DSConv(s3, s3, 1), ResidualBlock(s3))
        self.out_channels = [s2, s3] if fpn else s3

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        if self.fpn:
            l4 = self.layer4(self.layer3(l2))
            return [l2, l4]
        x = self.layer3(l2)
        x = self.layer4(x)
        return x


def _download_with_certifi_fallback(download_fn):
    """torch.hub 다운로드는 stdlib urllib을 쓰기 때문에 pip-system-certs(=requests만 패치)로는
    고쳐지지 않는 Windows 인증서 저장소 파싱 버그(ASN1: NOT_ENOUGH_DATA)가 있음.
    실패 시 certifi CA 번들로 HTTPS 컨텍스트를 한 번 교체해서 재시도한다."""
    try:
        return download_fn()
    except Exception as e:  # noqa: BLE001 - SSL/네트워크 오류를 폭넓게 흡수 후 한 번 더 시도
        try:
            import ssl
            import certifi

            ctx = ssl.create_default_context(cafile=certifi.where())
            ssl._create_default_https_context = lambda: ctx
            print(f"[안내] 기본 SSL 컨텍스트 실패({e!r}) -> certifi CA 번들로 재시도합니다.")
            return download_fn()
        except Exception:
            raise e


class MobileNetBackbone(nn.Module):
    """ImageNet 사전학습 MobileNetV3-Small의 앞 5개 블록만 사용 (stride 16, 40채널).

    stride 8을 요청하면 features[4]의 depthwise conv만 stride 2->1로 바꾼다.
    블록을 잘라내는 대신(features[:4]는 24채널로 얇아짐) stride만 바꾸는 이유는
    커널 자체는 그대로라 ImageNet 사전학습 가중치가 전부 유효하게 남기 때문.
    다만 이 블록 이후(SE/projection/neck/head)의 수용 영역이 입력 픽셀 기준으로
    절반이 되므로, 큰 사람에 대해서는 셀 하나가 보는 범위가 좁아지는 trade-off가 있다.
    """

    def __init__(self, pretrained=True, pretrained_path=None, stride=16, taps=None):
        super().__init__()
        if stride not in (8, 16):
            raise ValueError(f"지원하지 않는 stride: {stride} (8 또는 16)")
        import torchvision
        from torchvision.models import MobileNet_V3_Small_Weights

        base = None
        if pretrained_path is not None:
            base = torchvision.models.mobilenet_v3_small(weights=None)
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            base.load_state_dict(state)
        elif pretrained:
            try:
                base = _download_with_certifi_fallback(
                    lambda: torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
                )
            except Exception as e:  # noqa: BLE001 - 네트워크/인증서 문제 등 환경 이슈를 폭넓게 흡수
                print(f"[경고] ImageNet 사전학습 가중치 다운로드 실패 ({e!r}).")
                print("       --pretrained-path로 로컬 가중치를 지정하거나 랜덤 초기화로 계속 진행합니다.")
        if base is None:
            base = torchvision.models.mobilenet_v3_small(weights=None)

        # features[0:5] -> 240x320 입력 기준 (40, 15, 20) : stride 16과 정확히 일치
        self.features = base.features[:5]
        self.out_channels = 40
        self.taps = tuple(taps) if taps is not None else None

        if self.taps is not None:
            # FPN 모드: 여러 지점의 feature를 뽑아 리스트로 반환한다. stride 패치는 하지 않음
            # (레벨을 만드는 게 목적이라 다운샘플이 그대로 있어야 /8과 /16이 둘 다 나온다).
            self.out_channels = self._probe_tap_channels()
            return

        if stride == 8:
            # features[4]는 5x5 depthwise가 stride 2인 유일한 블록. 그것만 1로 바꾸면
            # 출력이 (40, 30, 40)이 된다. 대상이 정확히 1개인지 확인 후 바꿔서
            # torchvision 버전이 바뀌어 구조가 달라지면 조용히 넘어가지 않고 터지게 한다.
            targets = [m for m in self.features[4].modules()
                       if isinstance(m, nn.Conv2d) and m.stride == (2, 2)]
            if len(targets) != 1:
                raise RuntimeError(
                    f"features[4]에서 stride 2 conv를 {len(targets)}개 찾음 (1개를 기대). "
                    "torchvision 버전에 따라 MobileNetV3 구조가 다를 수 있습니다."
                )
            targets[0].stride = (1, 1)

    def _probe_tap_channels(self):
        """tap 지점의 채널 수를 하드코딩 대신 더미 forward로 실측한다.
        eval()로 돌려 BatchNorm running stats가 오염되지 않게 한다."""
        was_training = self.features.training
        self.features.eval()
        chs = []
        with torch.no_grad():
            h = torch.zeros(1, 3, 64, 64)
            for i, layer in enumerate(self.features):
                h = layer(h)
                if i in self.taps:
                    chs.append(h.shape[1])
        self.features.train(was_training)
        return chs

    def forward(self, x):
        if self.taps is None:
            return self.features(x)

        feats = []
        h = x
        for i, layer in enumerate(self.features):
            h = layer(h)
            if i in self.taps:
                feats.append(h)
        return feats


class FPNNeck(nn.Module):
    """top-down FPN. 각 레벨을 1x1로 같은 채널로 맞춘 뒤, 저해상도 레벨을 업샘플해
    고해상도 레벨에 더해 내려온다. 이렇게 하면 고해상도 레벨(/8)이 위치 정밀도를
    유지하면서도 저해상도 레벨(/16)의 넓은 문맥 정보를 함께 갖게 된다.

    in_channels는 고해상도 -> 저해상도 순서로 준다.
    """

    def __init__(self, in_channels, fpn_ch=64):
        super().__init__()
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, fpn_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_ch),
            )
            for c in in_channels
        ])
        # 업샘플로 생긴 격자무늬(aliasing)를 눌러주는 역할
        self.smooth = nn.ModuleList([DSConv(fpn_ch, fpn_ch, stride=1) for _ in in_channels])

    def forward(self, feats):
        lats = [layer(f) for layer, f in zip(self.lateral, feats)]
        for i in range(len(lats) - 2, -1, -1):
            up = F.interpolate(lats[i + 1], size=lats[i].shape[-2:], mode="nearest")
            lats[i] = lats[i] + up
        return [s(l) for s, l in zip(self.smooth, lats)]


class PersonDetector(nn.Module):
    def __init__(self, backbone="custom", pretrained=True, pretrained_path=None, grid_stride=GRID_STRIDE,
                 fpn=False, fpn_ch=64, custom_channels=CUSTOM_CHANNELS_DEFAULT):
        super().__init__()
        self.fpn = fpn
        if fpn:
            if backbone == "mobilenet":
                self.backbone = MobileNetBackbone(
                    pretrained=pretrained, pretrained_path=pretrained_path, taps=FPN_TAPS
                )
            elif backbone == "custom":
                self.backbone = CustomBackbone(channels=custom_channels, fpn=True)
            else:
                raise ValueError(f"FPN 모드는 mobilenet/custom 백본만 지원합니다: {backbone}")
            self.neck = FPNNeck(self.backbone.out_channels, fpn_ch=fpn_ch)
            # head를 레벨마다 따로 두는 이유: head 안 DSConv에 BatchNorm이 있는데
            # 레벨별로 feature 통계가 크게 달라서, 공유하면 BN running stats가 섞여 망가진다.
            # 연산량은 공유하든 말든 동일하고 파라미터만 조금 늘어난다.
            self.heads = nn.ModuleList([self._make_head(fpn_ch) for _ in FPN_TAPS])
            return

        if backbone == "custom":
            self.backbone = CustomBackbone(stride=grid_stride, channels=custom_channels)
        elif backbone == "mobilenet":
            self.backbone = MobileNetBackbone(
                pretrained=pretrained, pretrained_path=pretrained_path, stride=grid_stride
            )
        else:
            raise ValueError(f"알 수 없는 backbone: {backbone}")

        backbone_ch = self.backbone.out_channels
        # custom backbone은 이미 256채널이라 neck이 항등함수가 되어 기존 head와 완전히 동일하게 유지됨
        if backbone_ch != 256:
            self.neck = nn.Sequential(
                nn.Conv2d(backbone_ch, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
        else:
            self.neck = nn.Identity()

        self.head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            DSConv(64, 64, stride=1),
            nn.Conv2d(64, 5, kernel_size=1),
        )

    @staticmethod
    def _make_head(in_ch):
        """단일 스케일 head와 같은 구성(1x1 -> DSConv -> 1x1 5채널)이되 입력 채널만 다름."""
        return nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            DSConv(64, 64, stride=1),
            nn.Conv2d(64, 5, kernel_size=1),
        )

    def forward(self, x):
        if self.fpn:
            feats = self.neck(self.backbone(x))
            # 고해상도(/8) 먼저, 저해상도(/16) 나중 — FPN_STRIDES와 순서가 반드시 같아야 함
            return tuple(head(f) for head, f in zip(self.heads, feats))

        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """0/1 타겟은 물론 IoU 같은 연속값 타겟(soft target)도 처리하는 Focal Loss.

    변조항으로 |target - p|^gamma를 쓴다. target이 0 또는 1이면 기존 (1-pt)^gamma와
    완전히 동일한 값이 되므로 IoU-aware를 끄면 이전 실험과 정확히 같게 동작하고,
    target이 0~1 사이 연속값일 때도 그대로 일반화된다 (Quality Focal Loss 형태).
    기존 구현은 `target == 1` 정확 일치로 양성을 판별해서, soft target을 주면
    양성 셀이 전부 음성으로 취급되는 문제가 있었다.
    """

    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p = torch.sigmoid(pred)
        modulating = (target - p).abs().pow(self.gamma)
        alpha_factor = torch.where(target > 0, self.alpha, 1 - self.alpha)
        loss = alpha_factor * modulating * bce
        return loss.sum()


def complete_iou(pred_xyxy, gt_xyxy, eps=1e-7):
    """CIoU = IoU - 중심점거리항 - 종횡비항. 반환값 범위는 대략 (-1, 1].

    SmoothL1이 인코딩된 값(오프셋/로그비)을 따로따로 비교하는 것과 달리,
    CIoU는 박스를 실제 픽셀 좌표로 되돌린 뒤 겹침 정도를 직접 최적화한다.
    """
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_xyxy.unbind(-1)

    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    gw = (gx2 - gx1).clamp(min=eps)
    gh = (gy2 - gy1).clamp(min=eps)

    inter = ((torch.min(px2, gx2) - torch.max(px1, gx1)).clamp(min=0)
             * (torch.min(py2, gy2) - torch.max(py1, gy1)).clamp(min=0))
    union = pw * ph + gw * gh - inter + eps
    iou = inter / union

    # 두 박스를 모두 감싸는 최소 사각형의 대각선 길이 제곱
    cw = (torch.max(px2, gx2) - torch.min(px1, gx1)).clamp(min=eps)
    ch = (torch.max(py2, gy2) - torch.min(py1, gy1)).clamp(min=eps)
    c2 = cw ** 2 + ch ** 2 + eps

    # 중심점 사이 거리 제곱
    rho2 = ((px1 + px2 - gx1 - gx2) ** 2 + (py1 + py2 - gy1 - gy2) ** 2) / 4.0

    # 종횡비 일치도
    v = (4.0 / (math.pi ** 2)) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        # alpha는 논문 정의상 상수 취급(그래디언트 전파 안 함)
        alpha = v / (1.0 - iou + v + eps)

    return iou - rho2 / c2 - alpha * v


class DetectionLoss(nn.Module):
    """objectness는 FocalLoss 그대로, 박스 회귀만 SmoothL1 -> CIoU로 교체."""

    def __init__(self, img_w, img_h, box_gain=5.0, iou_aware=False, iou_aware_warmup=0):
        super().__init__()
        self.obj_loss = FocalLoss()
        self.img_w = img_w
        self.img_h = img_h
        # SmoothL1(인코딩 공간)과 CIoU(0~2 범위)는 스케일이 달라서 가중치도 다시 잡음
        self.box_gain = box_gain
        # True면 양성 셀의 objectness 타겟을 고정 1.0 대신 그 시점 예측 박스와 GT의 CIoU로 사용
        self.iou_aware = iou_aware
        # 초반 N epoch은 고정 1.0 타겟을 유지한다. 학습 시작 시점엔 박스가 엉망이라
        # CIoU가 0에 가깝고, 그걸 그대로 objectness 타겟으로 쓰면 "모든 셀에서 0을 예측"하는
        # 퇴화 해로 빠져 탐지가 통째로 죽는다(cold-start). 박스가 자리잡은 뒤 전환한다.
        self.iou_aware_warmup = iou_aware_warmup
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """warmup 판정을 위해 학습 루프에서 매 epoch 호출 (0-index)."""
        self.current_epoch = epoch

    @property
    def _use_iou_target(self):
        return self.iou_aware and self.current_epoch >= self.iou_aware_warmup

    def _decode_to_xyxy(self, box, gx, gy, cell_w, cell_h, is_pred):
        """(N,4) 인코딩 값 -> (N,4) 픽셀 좌표 xyxy. decode_prediction과 동일한 수식."""
        if is_pred:
            # 다중 양성 셀 할당으로 오프셋 범위가 [-0.5,1.5)까지 확장됨
            tx = torch.sigmoid(box[:, 0]) * 2.0 - 0.5
            ty = torch.sigmoid(box[:, 1]) * 2.0 - 0.5
            tw = box[:, 2].clamp(max=WH_LOG_CLAMP)
            th = box[:, 3].clamp(max=WH_LOG_CLAMP)
        else:
            tx, ty = box[:, 0], box[:, 1]
            tw, th = box[:, 2], box[:, 3]

        cx = (gx + tx) * cell_w
        cy = (gy + ty) * cell_h
        bw = torch.exp(tw) * self.img_w
        bh = torch.exp(th) * self.img_h

        return torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=-1)

    def forward(self, preds, targets):
        """preds/targets는 텐서 1개(단일 스케일) 또는 레벨별 리스트(FPN) 둘 다 받는다.

        정규화는 레벨별이 아니라 '전체 레벨의 양성 셀 합'으로 한 번에 한다. 레벨마다
        따로 나누면 양성이 몇 개 없는 레벨의 손실이 과도하게 증폭되어, 그 레벨이
        전체 학습을 지배해버린다.
        """
        preds, targets = as_levels(preds), as_levels(targets)
        num_positives = torch.clamp(
            sum((t[:, 0] > 0).sum() for t in targets).float(), min=1.0
        )
        return sum(self._level_loss(p, t, num_positives) for p, t in zip(preds, targets))

    def _level_loss(self, pred, target, num_positives):
        obj_pred = pred[:, 0]
        obj_gt = target[:, 0]

        mask = obj_gt > 0

        # objectness 타겟은 박스 CIoU를 참조할 수 있으므로 박스 계산 뒤로 미룸
        obj_target = obj_gt

        if mask.any():
            B, _, H, W = pred.shape
            cell_w = self.img_w / W
            cell_h = self.img_h / H

            # 양성 셀의 그리드 좌표 (오프셋을 절대 픽셀로 되돌리는 데 필요)
            gy_grid, gx_grid = torch.meshgrid(
                torch.arange(H, device=pred.device),
                torch.arange(W, device=pred.device),
                indexing="ij",
            )
            gx = gx_grid.unsqueeze(0).expand(B, H, W)[mask].float()
            gy = gy_grid.unsqueeze(0).expand(B, H, W)[mask].float()

            # CIoU는 atan/나눗셈이 있어 AMP(fp16)에서 불안정할 수 있으므로 fp32로 계산
            pred_box = pred[:, 1:5].permute(0, 2, 3, 1)[mask].float()
            gt_box = target[:, 1:5].permute(0, 2, 3, 1)[mask].float()

            pred_xyxy = self._decode_to_xyxy(pred_box, gx, gy, cell_w, cell_h, is_pred=True)
            gt_xyxy = self._decode_to_xyxy(gt_box, gx, gy, cell_w, cell_h, is_pred=False)

            ciou = complete_iou(pred_xyxy, gt_xyxy)
            box_loss = (1.0 - ciou).sum()

            if self._use_iou_target:
                # detach 필수: objectness 타겟으로 쓰는 CIoU에 그래디언트가 흐르면
                # "박스를 잘 맞추는 것"과 "타겟을 낮춰서 손실을 줄이는 것"이 뒤섞여
                # 순환 그래디언트가 생긴다. 또 CIoU는 음수(~-1)까지 나올 수 있어
                # BCE 타겟으로 쓰려면 [0,1]로 잘라야 한다.
                obj_target = obj_gt.clone()
                obj_target[mask] = ciou.detach().clamp(0.0, 1.0).to(obj_target.dtype)
        else:
            box_loss = torch.tensor(0.0, device=pred.device)

        obj_loss_sum = self.obj_loss(obj_pred, obj_target)

        return (obj_loss_sum / num_positives) + (self.box_gain * box_loss / num_positives)


# --------------------------------------------------------------------------
# Decode / NMS / IoU
# --------------------------------------------------------------------------
WH_LOG_CLAMP = 4.0  # exp() 오버플로 방지용 (초기/불안정 학습 시 raw pred가 튈 수 있음)


def decode_prediction(pred, img_w, img_h, conf_threshold=0.25):
    boxes = []
    pred = pred.detach().cpu()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    cell_w = img_w / grid_w
    cell_h = img_h / grid_h

    obj = torch.sigmoid(pred[0])
    ys, xs = torch.where(obj > conf_threshold)

    for y, x in zip(ys.tolist(), xs.tolist()):
        # 다중 양성 셀 할당으로 오프셋 범위가 [-0.5,1.5)까지 확장됨 (build_target 참고)
        tx = (torch.sigmoid(pred[1, y, x]) * 2.0 - 0.5).item()
        ty = (torch.sigmoid(pred[2, y, x]) * 2.0 - 0.5).item()
        tw = torch.clamp(pred[3, y, x], max=WH_LOG_CLAMP).item()
        th = torch.clamp(pred[4, y, x], max=WH_LOG_CLAMP).item()

        bw = math.exp(tw) * img_w
        bh = math.exp(th) * img_h

        cx = (x + tx) * cell_w
        cy = (y + ty) * cell_h

        xmin = max(0.0, cx - bw / 2)
        ymin = max(0.0, cy - bh / 2)
        xmax = min(float(img_w), cx + bw / 2)
        ymax = min(float(img_h), cy + bh / 2)

        boxes.append([xmin, ymin, xmax, ymax, obj[y, x].item()])

    return boxes


def decode_levels(preds, img_w, img_h, conf_threshold=0.25):
    """레벨별로 디코드한 박스를 하나의 리스트로 합친다 (NMS는 합친 뒤 호출부에서 1회).

    decode_prediction이 그리드 크기를 텐서 shape에서 직접 읽으므로 레벨마다 셀 크기가
    달라도 그대로 쓸 수 있다. 결과 좌표는 모두 입력 이미지 픽셀 기준이라 바로 합쳐진다.
    """
    boxes = []
    for pred in as_levels(preds):
        boxes.extend(decode_prediction(pred, img_w, img_h, conf_threshold))
    return boxes


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def nms(boxes, iou_threshold=0.3):
    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if calculate_iou(best[:4], b[:4]) < iou_threshold]

    return keep


# --------------------------------------------------------------------------
# mAP@IoU 계산 (VOC2012 all-point 방식, 클래스 1개)
# --------------------------------------------------------------------------
def compute_ap(all_predictions, all_gts, iou_threshold=0.5):
    total_gt = sum(len(v) for v in all_gts.values())
    if total_gt == 0:
        return 0.0
    if not all_predictions:
        return 0.0

    matched = {img_id: [False] * len(boxes) for img_id, boxes in all_gts.items()}
    preds_sorted = sorted(all_predictions, key=lambda p: p[1], reverse=True)

    tp = np.zeros(len(preds_sorted), dtype=np.float64)
    fp = np.zeros(len(preds_sorted), dtype=np.float64)

    for i, (img_id, _conf, box) in enumerate(preds_sorted):
        gts = all_gts.get(img_id, [])
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gts):
            if matched[img_id][j]:
                continue
            iou = calculate_iou(box, gt)
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_j >= 0 and best_iou >= iou_threshold:
            tp[i] = 1.0
            matched[img_id][best_j] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / total_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# --------------------------------------------------------------------------
# 평가
# --------------------------------------------------------------------------
def evaluate(model, loader, criterion, device, img_w, img_h, grids, boundaries,
             conf_threshold=0.25, recall_iou=0.3, nms_iou=0.3, map_iou=0.5):
    model.eval()

    total_loss = 0.0
    detected_objects = 0
    total_objects = 0
    total_predictions = 0

    all_predictions = []
    all_gts = {}
    img_id = 0

    with torch.no_grad():
        for images, boxes_list in loader:
            images = images.to(device)
            targets = stack_level_targets(boxes_list, grids, img_w, img_h, boundaries, device)

            pred = model(images)
            loss = criterion(pred, targets)
            total_loss += loss.item()

            pred_levels = as_levels(pred)
            for i in range(len(images)):
                # 샘플 i의 레벨별 예측을 모아 한 번에 디코드 -> 합친 뒤 NMS 1회
                pred_boxes = decode_levels([p[i] for p in pred_levels], img_w, img_h, conf_threshold)
                pred_boxes = nms(pred_boxes, nms_iou)

                gt_boxes = boxes_list[i].tolist()
                all_gts[img_id] = gt_boxes
                for box in pred_boxes:
                    all_predictions.append((img_id, box[4], box[:4]))

                total_predictions += len(pred_boxes)
                total_objects += len(gt_boxes)
                for gt in gt_boxes:
                    if any(calculate_iou(pb[:4], gt) >= recall_iou for pb in pred_boxes):
                        detected_objects += 1

                img_id += 1

    avg_loss = total_loss / len(loader)
    recall = detected_objects / total_objects if total_objects > 0 else 0.0
    avg_pred = total_predictions / len(loader.dataset)
    mAP = compute_ap(all_predictions, all_gts, iou_threshold=map_iou)

    return {"loss": avg_loss, "recall": recall, "avg_pred": avg_pred, "mAP50": mAP}


# --------------------------------------------------------------------------
# 학습
# --------------------------------------------------------------------------
def build_optimizer(model, args):
    if args.backbone == "mobilenet" and args.backbone_lr_mult != 1.0:
        backbone_params = list(model.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
        param_groups = [
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
            {"params": other_params, "lr": args.lr},
        ]
    else:
        param_groups = model.parameters()

    return torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)


def plot_history(train_losses, val_losses, val_maps, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(train_losses, label="Train Loss")
    axes[0].plot(val_losses, label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(val_maps, label="Validation mAP@0.5")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP@0.5")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close(fig)


def export_torchscript(model, img_w, img_h, device, output_path):
    """항상 CPU로 트레이싱해서 저장한다.

    GPU(cuda)에서 트레이싱하면 저장된 그래프에 CUDA 텐서 할당이 그대로 박혀서,
    CUDA가 없는 환경(RPi4, CPU 전용 LibTorch)에서 torch::jit::load 시
    'Could not run aten::empty_strided with arguments from the CUDA backend'
    에러가 난다. map_location만으로는 못 고치는 문제라(그래프 자체에 박혀 있음)
    트레이싱 시점부터 CPU로 옮겨야 한다. 이 함수는 학습 루프가 전부 끝난 뒤
    (main()의 args.export 블록에서) 한 번만 호출되므로 여기서 CPU로 옮겨도
    이후 학습에 영향이 없다.
    """
    model_cpu = model.to("cpu")
    model_cpu.eval()
    # [버그 수정] 원본은 (1,3,W,H)로 잘못 생성했음. 모델은 (B,C,H,W)를 기대함
    example = torch.randn(1, 3, img_h, img_w, device="cpu")
    scripted = torch.jit.trace(model_cpu, example)
    scripted.save(output_path)
    model.to(device)  # 호출부가 이후 device의 model을 계속 참조할 수 있으므로 원상 복구


def main():
    parser = argparse.ArgumentParser(description="RC카 보행자 탐지 모델 학습")
    parser.add_argument("--data-root", default="./archive")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--backbone", choices=["custom", "mobilenet"], default="custom")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--pretrained-path", default=None, help="로컬에 저장된 mobilenet_v3_small state_dict 경로")
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument("--iou-aware", action="store_true",
                        help="양성 셀의 objectness 타겟을 고정 1.0 대신 예측 박스와 GT의 CIoU(detach)로 사용")
    parser.add_argument("--mosaic-prob", type=float, default=0.0,
                        help="Mosaic 증강 확률(0이면 사용 안 함). 4장을 2x2로 붙여 축소하므로 "
                             "작은 객체 샘플이 늘어난다. YOLOv8 기본값은 1.0이지만 여기서는 "
                             "데이터셋이 작아 0.5 정도부터 시험 권장")
    parser.add_argument("--iou-aware-warmup", type=int, default=5,
                        help="IoU-aware 전환 전 고정 1.0 타겟을 유지할 epoch 수 (cold-start 방지)")
    parser.add_argument("--unsharp", action="store_true",
                        help="리사이즈 직후 언샤프 마스크로 윤곽을 강조 (train/valid 모두 적용). "
                             "추론 시 detect_video.py에도 --unsharp과 동일한 sigma/alpha를 줘야 함")
    parser.add_argument("--unsharp-sigma", type=float, default=1.0,
                        help="언샤프 마스크의 가우시안 블러 sigma(클수록 넓은 범위의 윤곽을 강조, 할로 위험 증가)")
    parser.add_argument("--unsharp-alpha", type=float, default=0.5,
                        help="샤프닝 세기(클수록 또렷해지지만 노이즈도 함께 증폭)")

    parser.add_argument("--fpn", action="store_true",
                        help="2레벨 FPN(/8 + /16) 출력 사용. mobilenet 백본 전용이며 "
                             "--grid-stride는 무시됨(레벨 stride가 FPN_STRIDES로 고정)")
    parser.add_argument("--fpn-ch", type=int, default=64, help="FPN 내부 채널 수")
    parser.add_argument("--fpn-boundaries", type=float, nargs="+", default=[64.0],
                        help="레벨을 나누는 박스 높이(px) 경계. 기본 [64] -> 높이 <64px는 /8, "
                             "그 이상은 /16이 담당. 레벨 수보다 1개 적게 줄 것")

    parser.add_argument("--grid-stride", type=int, choices=[8, 16], default=GRID_STRIDE,
                        help="백본 출력 stride. 8이면 그리드가 4배 촘촘해지고(20x15 -> 40x30) "
                             "연산량은 약 1.9배(mobilenet 기준). 학습 때 쓴 값을 detect_video.py에도 "
                             "동일하게 줘야 함(raw .pth 로드 시)")
    parser.add_argument("--custom-channels", type=int, nargs=4, default=None,
                        metavar=("STEM", "L1", "L2", "L3"),
                        help="--backbone custom 전용 채널 축소. 기본값은 (32,64,128,256) "
                             "(팀원 모델과 동일 체급, 1490.84M MACs). RPi4 배포 체급으로 낮추려면 "
                             "예: --custom-channels 16 24 32 64 (141.56M MACs, 9단계 FPN의 2.70배). "
                             "detect_video.py에도 동일 값을 줘야 함(raw .pth 로드 시)")

    parser.add_argument("--img-width", type=int, default=320)
    parser.add_argument("--img-height", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--recall-iou", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--map-iou", type=float, default=0.5)

    parser.add_argument("--max-train-images", type=int, default=None, help="빠른 스모크 테스트용 서브셋 크기")
    parser.add_argument("--max-val-images", type=int, default=None)

    parser.add_argument("--resume", default=None, help="이어서 학습할 checkpoint(.pth) 경로")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-export", dest="export", action="store_false", default=True)
    args = parser.parse_args()

    # stride를 낮춰도 원래 stride(16)의 배수는 유지해야 32/16 다운샘플 경계에서 어긋나지 않음
    if args.img_width % GRID_STRIDE != 0 or args.img_height % GRID_STRIDE != 0:
        raise ValueError(f"img-width/height는 {GRID_STRIDE}의 배수여야 합니다.")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("DEVICE:", device)

    grids = get_level_grids(args.img_width, args.img_height, args.fpn, args.grid_stride)
    # 마지막 레벨은 상한이 없으므로 경계값은 레벨 수보다 1개 적다
    boundaries = list(args.fpn_boundaries) if args.fpn else []
    if len(boundaries) != len(grids) - 1:
        raise ValueError(
            f"--fpn-boundaries는 {len(grids) - 1}개여야 합니다 (레벨 {len(grids)}개). "
            f"받은 값: {boundaries}"
        )

    if args.fpn:
        strides_desc = ", ".join(
            f"/{s} {gw}x{gh}" for s, (gw, gh) in zip(FPN_STRIDES, grids)
        )
        print(f"FPN 2레벨: {strides_desc} | 레벨 경계(박스 높이): {boundaries} px | ch={args.fpn_ch}")
    else:
        print(f"Grid size: {grids[0][0]}x{grids[0][1]} (stride {args.grid_stride})")

    # ---------------- 데이터 ----------------
    train_dir, train_csv = split_paths(args.data_root, "train")
    valid_dir, valid_csv = split_paths(args.data_root, "valid")

    normalize = args.backbone == "mobilenet"
    train_dataset = PersonDataset(
        train_dir, train_csv, args.img_width, args.img_height,
        max_images=args.max_train_images,
        transform=build_train_transform(args.img_width, args.img_height),
        # Mosaic 샘플은 이미 스케일이 크게 바뀐 상태라 확대 크롭은 빼고 적용
        transform_mosaic=build_train_transform(args.img_width, args.img_height, scale_crop_p=0.0),
        mosaic_prob=args.mosaic_prob,
        normalize=normalize,
        unsharp=args.unsharp, unsharp_sigma=args.unsharp_sigma, unsharp_alpha=args.unsharp_alpha,
    )
    val_dataset = PersonDataset(
        valid_dir, valid_csv, args.img_width, args.img_height,
        max_images=args.max_val_images, transform=None, normalize=normalize,
        # 검증셋에도 동일하게 적용해야 학습/평가 입력 분포가 일치한다
        unsharp=args.unsharp, unsharp_sigma=args.unsharp_sigma, unsharp_alpha=args.unsharp_alpha,
    )
    print(f"Train: {len(train_dataset)}  Valid: {len(val_dataset)}")
    if args.unsharp:
        print(f"Unsharp mask: ON (sigma={args.unsharp_sigma}, alpha={args.unsharp_alpha}) "
              f"-> 추론 시 detect_video.py에 --unsharp --unsharp-sigma {args.unsharp_sigma} "
              f"--unsharp-alpha {args.unsharp_alpha} 를 반드시 동일하게 줄 것")

    # persistent_workers=True: 매 epoch(= DataLoader.__iter__ 호출)마다 worker 프로세스를
    # 새로 spawn/재import하는 오버헤드를 없앰 (Windows spawn 방식에서 특히 비용이 큼)
    persistent = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=persistent,
    )
    valid_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=persistent,
    )

    # ---------------- 모델 ----------------
    custom_channels = tuple(args.custom_channels) if args.custom_channels else CUSTOM_CHANNELS_DEFAULT
    model = PersonDetector(
        backbone=args.backbone, pretrained=args.pretrained, pretrained_path=args.pretrained_path,
        grid_stride=args.grid_stride, fpn=args.fpn, fpn_ch=args.fpn_ch,
        custom_channels=custom_channels,
    ).to(device)
    if args.backbone == "custom" and custom_channels != CUSTOM_CHANNELS_DEFAULT:
        print(f"Custom backbone 채널: {custom_channels} (기본값 {CUSTOM_CHANNELS_DEFAULT} 아님)")

    # 레벨 수와 각 레벨의 그리드가 기대와 정확히 맞는지 검증. FPN은 tap 인덱스 하나만
    # 틀려도 조용히 다른 해상도가 나오므로 여기서 반드시 걸러야 한다.
    with torch.no_grad():
        dummy = torch.randn(1, 3, args.img_height, args.img_width, device=device)
        out_levels = as_levels(model(dummy))
    if len(out_levels) != len(grids):
        raise AssertionError(f"모델 출력 레벨 수({len(out_levels)})가 기대값({len(grids)})과 다릅니다.")
    for lv, (out, (gw, gh)) in enumerate(zip(out_levels, grids)):
        if out.shape[2] != gh or out.shape[3] != gw:
            raise AssertionError(
                f"레벨 {lv} 출력 그리드({out.shape[3]}x{out.shape[2]})가 "
                f"기대값({gw}x{gh})과 다릅니다."
            )

    criterion = DetectionLoss(args.img_width, args.img_height,
                              iou_aware=args.iou_aware,
                              iou_aware_warmup=args.iou_aware_warmup)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    train_losses, val_losses, val_maps, val_recalls = [], [], [], []
    start_epoch = 0
    best_map = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        train_losses = ckpt.get("train_loss", [])
        val_losses = ckpt.get("val_loss", [])
        val_maps = ckpt.get("val_map", [])
        best_map = max(val_maps) if val_maps else 0.0
        print(f"체크포인트에서 재개: {args.resume} (epoch {start_epoch}부터)")

    best_path = os.path.join(args.output_dir, "best_model.pth")
    last_path = os.path.join(args.output_dir, "person_detector.pth")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        criterion.set_epoch(epoch)  # IoU-aware warmup 판정용
        total_loss = 0.0
        # 파일로 리다이렉트된 백그라운드 실행 시 tqdm의 \r 갱신이 로그를 어지럽히지 않도록 비활성화
        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True,
            disable=not sys.stdout.isatty(),
        )

        # 레벨별 양성 셀 수 누적: 한쪽 레벨이 굶으면(0에 가까우면) 경계값이 잘못된 것
        pos_per_level = [0] * len(grids)

        for images, boxes_list in progress_bar:
            images = images.to(device)
            targets = stack_level_targets(
                boxes_list, grids, args.img_width, args.img_height, boundaries, device
            )
            for lv, t in enumerate(targets):
                pos_per_level[lv] += int((t[:, 0] > 0).sum().item())

            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = model(images)
                loss = criterion(pred, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        val_metrics = evaluate(
            model, valid_loader, criterion, device, args.img_width, args.img_height, grids, boundaries,
            conf_threshold=args.conf_threshold, recall_iou=args.recall_iou,
            nms_iou=args.nms_iou, map_iou=args.map_iou,
        )
        scheduler.step(val_metrics["loss"])

        val_losses.append(val_metrics["loss"])
        val_maps.append(val_metrics["mAP50"])
        val_recalls.append(val_metrics["recall"])

        if val_metrics["mAP50"] > best_map:
            best_map = val_metrics["mAP50"]
            torch.save(model.state_dict(), best_path)

        pos_desc = ""
        if len(grids) > 1:
            total_pos = max(sum(pos_per_level), 1)
            parts = [
                f"/{s}:{n:,}({n / total_pos * 100:.1f}%)"
                for s, n in zip(FPN_STRIDES, pos_per_level)
            ]
            pos_desc = "  양성셀 " + " ".join(parts)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {avg_loss:.4f}  Val Loss: {val_metrics['loss']:.4f}  "
            f"Val mAP@0.5: {val_metrics['mAP50']:.4f}  Val Recall: {val_metrics['recall']:.4f}  "
            f"(best mAP: {best_map:.4f}){pos_desc}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_losses,
                "val_loss": val_losses,
                "val_map": val_maps,
                "args": vars(args),
            },
            last_path,
        )

    try:
        plot_history(train_losses, val_losses, val_maps, args.output_dir)
    except Exception as e:  # noqa: BLE001 - 그래프 저장 실패가 학습 결과 자체를 무효화하진 않음
        print(f"[경고] 학습 곡선 저장 실패: {e!r}")

    if args.export:
        # 마지막 epoch 모델을 먼저 내보낸다 (참고용).
        last_export = os.path.join(args.output_dir, "person_detector_script_last.pt")
        export_torchscript(model, args.img_width, args.img_height, device, last_export)
        print(f"TorchScript 저장 완료(마지막 epoch): {last_export}")

        # 기본 이름(person_detector_script.pt)은 best 가중치로 내보낸다.
        # 마지막 epoch이 항상 최고 성능은 아니라서, 별도 지정 없이 쓰는 파일이
        # best가 되도록 하는 편이 실수를 줄인다.
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
            best_export = os.path.join(args.output_dir, "person_detector_script.pt")
            export_torchscript(model, args.img_width, args.img_height, device, best_export)
            print(f"TorchScript 저장 완료(best, mAP={best_map:.4f}): {best_export}")
        else:
            print(f"[경고] best 체크포인트가 없어 best export를 건너뜀: {best_path}")

    print(f"학습 종료. best mAP@0.5 = {best_map:.4f} ({best_path})")


if __name__ == "__main__":
    main()
