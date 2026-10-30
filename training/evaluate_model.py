"""보행자 탐지 모델(TorchScript 또는 ONNX)의 mAP@0.5를 단독으로 평가하는 스크립트.

train_person_detector.py 없이 이 파일 하나만으로 돌아간다 (Colab 등 외부 환경용).
의존성은 torch, opencv-python, numpy, pandas뿐이다(.onnx 모델이면 onnxruntime 추가).

로직(디코드/NMS/AP 계산)은 train_person_detector.py의 decode_prediction / nms /
compute_ap와 완전히 동일한 수식을 그대로 옮겨온 것이다. 원본이 바뀌면 이 파일도
같이 갱신해야 한다. 백엔드(TorchScript/ONNX)는 모델 로드·추론 호출만 다르고
전처리·디코드·AP 계산은 공통이다.

사용법 (CLI):
    python evaluate_model.py --weights ../models/person_detector_lite.onnx --data-root ./archive
    python evaluate_model.py --weights person_detector_script.pt --data-root ./archive

사용법 (Colab 셀):
    아래 CONFIG 섹션의 값만 고쳐서 그대로 실행하면 됨(--weights 등 인자 불필요).
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

# ============================================================================
# CONFIG — Colab 등에서 CLI 인자 없이 쓸 때 여기 값만 고쳐서 실행하면 됨.
# CLI로 --weights 등을 넘기면 그 값이 이 기본값을 덮어씀.
# ============================================================================
CONFIG = {
    # 평가할 모델. .onnx면 ONNX Runtime, 그 외(.pt/.torchscript)는 TorchScript로 로드
    "weights": os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "models", "person_detector_lite.onnx")),
    "backend": "auto",                 # auto / torch / onnx (auto면 확장자로 판별)
    "data_root": "./archive",          # archive/{split}/{split}/_annotations.csv 구조를 기대
    "split": "valid",                  # 보통 valid로 평가. 학습에 쓴 분할과 겹치면 안 됨
    "img_width": 320,
    "img_height": 240,
    "normalize": False,                # mobilenet 백본으로 학습한 모델이면 True
                                        # (ImageNet mean/std 정규화 적용)
    "conf_threshold": 0.05,
    "nms_iou": 0.3,
    "map_iou": 0.5,
    "recall_iou": 0.3,
    "max_images": None,                # 빠른 확인용으로 장수를 제한하고 싶으면 정수로
    "device": None,                    # None이면 cuda 있으면 cuda, 없으면 cpu
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
WH_LOG_CLAMP = 4.0


# --------------------------------------------------------------------------
# 디코드 / NMS / AP — train_person_detector.py와 동일한 수식
# --------------------------------------------------------------------------
def decode_prediction(pred, img_w, img_h, conf_threshold):
    """(5, grid_h, grid_w) raw 출력 -> [xmin,ymin,xmax,ymax,score] 리스트.

    grid_h/grid_w를 pred.shape에서 직접 읽으므로 stride 8/16, FPN 레벨 등
    그리드 크기와 무관하게 동작한다.
    """
    boxes = []
    pred = pred.detach().cpu()
    grid_h, grid_w = pred.shape[1], pred.shape[2]
    cell_w = img_w / grid_w
    cell_h = img_h / grid_h

    obj = torch.sigmoid(pred[0])
    ys, xs = torch.where(obj > conf_threshold)

    for y, x in zip(ys.tolist(), xs.tolist()):
        # 다중 양성 셀 할당으로 오프셋 범위가 [-0.5,1.5)까지 확장됨
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


def decode_levels(preds, img_w, img_h, conf_threshold):
    """FPN처럼 출력이 레벨별 텐서 tuple/list인 경우 각각 디코드해서 합친다.
    단일 스케일 모델은 preds가 텐서 하나이므로 자동으로 원소 1개짜리로 처리됨."""
    levels = preds if isinstance(preds, (list, tuple)) else (preds,)
    boxes = []
    for pred in levels:
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


def nms(boxes, iou_threshold):
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if calculate_iou(best[:4], b[:4]) < iou_threshold]
    return keep


def compute_ap(all_predictions, all_gts, iou_threshold):
    """VOC 방식 all-point AP. all_predictions: [(img_id, conf, [x1,y1,x2,y2]), ...]
    all_gts: {img_id: [[x1,y1,x2,y2], ...]}"""
    total_gt = sum(len(v) for v in all_gts.values())
    if total_gt == 0 or not all_predictions:
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
# 데이터 로드 (archive/{split}/{split}/_annotations.csv 구조)
# --------------------------------------------------------------------------
def load_boxes_by_file(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["class"] == "person"]
    return {
        fname: g[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float32)
        for fname, g in df.groupby("filename")
    }


def load_and_resize(image_dir, filename, boxes, img_w, img_h, normalize):
    path = os.path.join(image_dir, filename)
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = boxes.copy()
    valid = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
    boxes = boxes[valid]

    h, w = image.shape[:2]
    sx, sy = img_w / w, img_h / h
    image = cv2.resize(image, (img_w, img_h))
    if len(boxes) > 0:
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] * sx, 0, img_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] * sy, 0, img_h)

    image = image.astype(np.float32) / 255.0
    if normalize:
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1)))
    return tensor, boxes


# --------------------------------------------------------------------------
# 평가 본체
# --------------------------------------------------------------------------
def run(cfg):
    device = torch.device(cfg["device"] if cfg["device"] else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    backend = cfg["backend"]
    if backend == "auto":
        backend = "onnx" if cfg["weights"].lower().endswith(".onnx") else "torch"
    if backend == "onnx":
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" \
            else ["CPUExecutionProvider"]
        session = ort.InferenceSession(cfg["weights"], providers=providers)
        input_name = session.get_inputs()[0].name

        def model(x):
            # numpy 입출력을 기존 디코드 코드가 기대하는 torch.Tensor로만 변환
            outs = [torch.from_numpy(o) for o in session.run(None, {input_name: x.cpu().numpy()})]
            return outs[0] if len(outs) == 1 else tuple(outs)

        print(f"모델 로드(ONNX): {cfg['weights']} (providers={session.get_providers()})")
    else:
        model = torch.jit.load(cfg["weights"], map_location=device)
        model.eval()
        print(f"모델 로드(TorchScript): {cfg['weights']}")

    image_dir = os.path.join(cfg["data_root"], cfg["split"], cfg["split"])
    csv_path = os.path.join(image_dir, "_annotations.csv")
    boxes_by_file = load_boxes_by_file(csv_path)
    image_list = sorted(boxes_by_file.keys())
    if cfg["max_images"]:
        image_list = image_list[:cfg["max_images"]]
    print(f"평가 이미지: {len(image_list)}장 ({csv_path})")
    print(f"conf={cfg['conf_threshold']}  NMS IoU={cfg['nms_iou']}  mAP IoU={cfg['map_iou']}\n")

    all_predictions = []
    all_gts = {}
    detected_objects = total_objects = total_predictions = 0

    with torch.no_grad():
        for img_id, filename in enumerate(image_list):
            tensor, gt_boxes = load_and_resize(
                image_dir, filename, boxes_by_file[filename],
                cfg["img_width"], cfg["img_height"], cfg["normalize"],
            )
            pred = model(tensor.unsqueeze(0).to(device))
            levels = pred[0] if not isinstance(pred, (list, tuple)) else [p[0] for p in pred]

            boxes = decode_levels(levels, cfg["img_width"], cfg["img_height"], cfg["conf_threshold"])
            boxes = nms(boxes, cfg["nms_iou"])

            gt_list = gt_boxes.tolist()
            all_gts[img_id] = gt_list
            for b in boxes:
                all_predictions.append((img_id, b[4], b[:4]))

            total_predictions += len(boxes)
            total_objects += len(gt_list)
            for gt in gt_list:
                if any(calculate_iou(b[:4], gt) >= cfg["recall_iou"] for b in boxes):
                    detected_objects += 1

            if (img_id + 1) % 200 == 0:
                print(f"  {img_id + 1}/{len(image_list)}장 처리...")

    mAP = compute_ap(all_predictions, all_gts, cfg["map_iou"])
    recall = detected_objects / total_objects if total_objects else 0.0
    avg_pred = total_predictions / len(image_list) if image_list else 0.0

    print(f"\n{'=' * 50}")
    print(f"mAP@{cfg['map_iou']}: {mAP:.4f}")
    print(f"Recall (IoU {cfg['recall_iou']}): {recall:.4f}")
    print(f"이미지당 평균 검출 수: {avg_pred:.2f}")
    print(f"{'=' * 50}")
    return mAP, recall


def parse_args():
    p = argparse.ArgumentParser(description="TorchScript/ONNX 모델 mAP 단독 평가")
    p.add_argument("--weights", default=None, help=".onnx 또는 TorchScript(.pt/.torchscript)")
    p.add_argument("--backend", choices=["auto", "torch", "onnx"], default=None,
                   help="기본값 auto: --weights 확장자로 판별")
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--img-width", type=int, default=None)
    p.add_argument("--img-height", type=int, default=None)
    p.add_argument("--normalize", action="store_true", default=None,
                   help="mobilenet 백본으로 학습한 모델이면 켤 것")
    p.add_argument("--conf-threshold", type=float, default=None)
    p.add_argument("--nms-iou", type=float, default=None)
    p.add_argument("--map-iou", type=float, default=None)
    p.add_argument("--recall-iou", type=float, default=None)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = dict(CONFIG)
    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v
    return cfg


if __name__ == "__main__":
    # Colab의 !python 실행과 노트북 셀 실행(sys.argv에 노트북 관련 인자가 섞여 들어오는 경우) 둘 다
    # 대응하기 위해, 인식 못 하는 인자가 있으면 CONFIG만으로 조용히 진행한다.
    try:
        cfg = parse_args()
    except SystemExit:
        print("[안내] CLI 인자 파싱 실패 -> 파일 상단 CONFIG 값으로 실행합니다.")
        cfg = dict(CONFIG)
    run(cfg)
