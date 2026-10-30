"""
커스텀 grid 보행자 탐지 모델로 웹캠/영상/이미지를 추론하는 스크립트.

RPi4 배포를 염두에 두고 torch, cv2, numpy만 있으면 동작함
(pandas/albumentations 등 학습용 의존성 불필요 — requirements.txt 참고).
기본값은 models/person_detector_lite.onnx(실험 11, 경량 custom 백본)이다.

== --backend (torch 또는 onnx) ==
--weights 확장자(.onnx)로 자동 감지하거나 --backend onnx로 명시할 수 있음.
전처리(grid_preprocess)와 후처리(decode_prediction/decode_levels/nms)는 백엔드와
완전히 무관하게 그대로 재사용한다 - run_inference()가 onnxruntime의 numpy 입출력을
기존 코드가 기대하는 torch.Tensor로 앞뒤에서만 변환해주는 얇은 어댑터 역할이라,
모델 로드/추론 호출부만 바뀐다.

== 모델 입출력 ==
전처리는 letterbox가 아니라 (--img-width, --img-height)로의 단순 resize(종횡비
유지 안 함)이고, 출력은 grid 원시 출력(obj_logit, tx_logit, ty_logit, tw_log, th_log)
이라 sigmoid/exp로 직접 복원한다. training/train_person_detector.py의
decode_prediction/decode_levels와 완전히 동일한 수식 - 한쪽만 고치면 같은 모델인데
결과가 달라지므로 수정 시 같이 볼 것.
FPN 모델은 출력이 텐서 하나가 아니라 레벨별 텐서 tuple이라 decode_levels가 각각
디코드해서 합친다. MobileNet 백본으로 학습한 모델은 --normalize를 반드시 켜야 함
(ImageNet 정규화로 학습됐음); custom 백본 모델(기본 모델 포함)은 끄면 됨(기본값).
"""
import argparse
import math
import os
import time

import cv2
import numpy as np
import torch

DEFAULT_WEIGHTS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "person_detector_lite.onnx"))

GRID_IMG_WIDTH_DEFAULT = 320
GRID_IMG_HEIGHT_DEFAULT = 240
GRID_WH_LOG_CLAMP = 4.0
# train_person_detector.py의 PersonDataset과 반드시 동일해야 함(mobilenet 백본 전용)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def grid_preprocess(frame, img_w, img_h, device, normalize):
    """letterbox 없이 (img_w, img_h)로 단순 resize."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_w, img_h))
    img = img.astype(np.float32) / 255.0
    if normalize:
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor


def decode_prediction(pred, img_w, img_h, conf_threshold):
    """(5, grid_h, grid_w) raw 출력 -> [xmin,ymin,xmax,ymax,score] 리스트.
    train_person_detector.py::decode_prediction과 동일한 수식."""
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
        tw = torch.clamp(pred[3, y, x], max=GRID_WH_LOG_CLAMP).item()
        th = torch.clamp(pred[4, y, x], max=GRID_WH_LOG_CLAMP).item()

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


def resolve_backend(backend, weights_path):
    """backend='auto'면 --weights 확장자로 torch/onnx를 자동 판별."""
    if backend != "auto":
        return backend
    return "onnx" if weights_path.lower().endswith(".onnx") else "torch"


def load_model(weights_path, device, backend):
    """backend='onnx'면 onnxruntime.InferenceSession, 아니면 TorchScript 로드
    (.pt/.torchscript 등 확장자와 무관하게 동작)."""
    if backend == "onnx":
        import onnxruntime as ort

        # CUDAExecutionProvider는 onnxruntime-gpu가 깔려 있어야 실제로 잡힘.
        # 없으면 onnxruntime이 자동으로 CPU로 넘어가므로 굳이 미리 검사하지 않는다.
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" \
            else ["CPUExecutionProvider"]
        session = ort.InferenceSession(weights_path, providers=providers)
        print(f"ONNX 모델 로드: {weights_path} (providers={session.get_providers()})")
        return session

    model = torch.jit.load(weights_path, map_location=device)
    model.eval()
    print(f"TorchScript 모델 로드: {weights_path} (device={device})")
    return model


def run_inference(model, tensor, backend):
    """backend에 따라 추론하고, decode 함수들이 그대로 쓸 수 있게 torch.Tensor로
    통일해서 반환한다. onnxruntime은 numpy 입출력이라 앞뒤로만 변환한다.
    출력이 여러 개인 모델(FPN 등)이면 tuple로 반환해 decode_levels가 처리하게 함."""
    if backend == "onnx":
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: tensor.cpu().numpy()})
        tensors = [torch.from_numpy(o) for o in outputs]
        return tensors[0] if len(tensors) == 1 else tuple(tensors)

    with torch.no_grad():
        return model(tensor)


def run(args):
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = load_model(args.weights, device, args.backend)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"영상 소스를 열 수 없습니다: {args.source}")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (w0, h0))

    prev_time = time.time()
    frame_idx = 0

    # FPS 측정: 처음 --warmup-frames장은 모델/디바이스 워밍업(캐시 미스, CUDA 컨텍스트
    # 초기화 등)으로 느리게 나오므로 평균 계산에서 제외한다. 워밍업이 끝나는 시점부터
    # 시간을 다시 재기 시작해서 순수 추론 구간만 평균을 낸다.
    avg_start_time = None
    avg_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        tensor = grid_preprocess(frame, args.img_width, args.img_height, device, args.normalize)
        output = run_inference(model, tensor, args.backend)

        # 단일 스케일이면 (1,5,H,W) 텐서 하나, FPN이면 레벨별 텐서 tuple ->
        # 배치 차원만 벗겨서 decode_levels에 전달
        levels = output if isinstance(output, (list, tuple)) else (output,)
        boxes = decode_levels([lv[0] for lv in levels], args.img_width, args.img_height, args.conf)
        boxes = nms(boxes, args.nms_iou)

        for xmin, ymin, xmax, ymax, conf in boxes:
            # 단순 resize 좌표계 -> 원본 프레임 좌표계
            sx, sy = w / args.img_width, h / args.img_height
            rx0 = max(0, min(w, int(xmin * sx)))
            ry0 = max(0, min(h, int(ymin * sy)))
            rx1 = max(0, min(w, int(xmax * sx)))
            ry1 = max(0, min(h, int(ymax * sy)))

            cv2.rectangle(frame, (rx0, ry0), (rx1, ry1), (0, 255, 0), 2)
            cv2.putText(frame, f"person {conf:.2f}", (rx0, max(0, ry0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        is_warmup = frame_idx < args.warmup_frames
        if not is_warmup:
            if avg_start_time is None:
                avg_start_time = now  # 워밍업 직후 첫 프레임 시점부터 평균 집계 시작
            else:
                avg_frame_count += 1
        avg_fps = avg_frame_count / max(now - avg_start_time, 1e-6) if avg_start_time else 0.0

        cv2.putText(frame, f"FPS: {fps:.1f} (avg {avg_fps:.1f})  persons: {len(boxes)}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if writer is not None:
            writer.write(frame)
        if not args.no_display:
            cv2.imshow("Person Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if args.print_every and frame_idx % args.print_every == 0:
            tag = " [워밍업]" if is_warmup else ""
            print(f"frame {frame_idx}: FPS={fps:.1f}  avg_FPS={avg_fps:.1f}  "
                  f"persons={len(boxes)}{tag}")

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    if avg_frame_count > 0:
        total_elapsed = time.time() - avg_start_time
        print(f"\n[FPS 요약] 워밍업 {args.warmup_frames}프레임 제외, "
              f"측정 {avg_frame_count}프레임, 총 {total_elapsed:.2f}초 "
              f"-> 평균 {avg_frame_count / total_elapsed:.2f} FPS")
    else:
        print("\n[FPS 요약] 워밍업 이후 프레임이 없어 평균을 계산할 수 없습니다 "
              f"(영상이 --warmup-frames={args.warmup_frames}보다 짧음).")


def main():
    parser = argparse.ArgumentParser(description="커스텀 grid 보행자 탐지 모델 실시간 추론")
    parser.add_argument("--source", default="0", help="웹캠 인덱스(예: 0) 또는 영상/이미지 파일 경로")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="ONNX(.onnx) 또는 TorchScript 모델 경로 (기본값: models/person_detector_lite.onnx)")
    parser.add_argument("--backend", choices=["auto", "torch", "onnx"], default="auto",
                        help="auto(기본값)면 --weights 확장자(.onnx)로 자동 판별. "
                             "명시하려면 --backend onnx / --backend torch")
    parser.add_argument("--img-width", type=int, default=GRID_IMG_WIDTH_DEFAULT,
                        help="모델 학습 시 사용한 입력 너비")
    parser.add_argument("--img-height", type=int, default=GRID_IMG_HEIGHT_DEFAULT,
                        help="모델 학습 시 사용한 입력 높이")
    parser.add_argument("--normalize", action="store_true",
                        help="MobileNet 백본으로 학습한 모델이면 반드시 켤 것. "
                             "custom 백본 모델(기본 모델 포함)은 끄면 됨(기본값)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="objectness 임계값 (기본값 0.25 = RPi4 배포 기본값. "
                             "mAP 평가 때는 0.05를 썼음 - evaluate_model.py 참고)")
    parser.add_argument("--nms-iou", type=float, default=0.3,
                        help="NMS IoU 임계값 (기본값 0.3 = 학습/평가 때 쓴 값)")
    parser.add_argument("--device", default=None, help="미지정 시 cuda 사용 가능하면 cuda, 아니면 cpu")
    parser.add_argument("--save", default=None, help="결과 영상을 저장할 경로 (mp4)")
    parser.add_argument("--no-display", action="store_true", help="GUI 없이 실행 (RPi 헤드리스 환경용)")
    parser.add_argument("--print-every", type=int, default=30,
                        help="콘솔에 FPS를 출력할 프레임 간격 (0이면 출력 안 함). "
                             "--no-display 여부와 무관하게 항상 출력됨")
    parser.add_argument("--warmup-frames", type=int, default=10,
                        help="평균 FPS 계산에서 제외할 초반 프레임 수 (모델/디바이스 초기화 "
                             "오버헤드로 처음 몇 프레임은 느리게 나오므로 평균이 왜곡되는 것을 방지)")
    args = parser.parse_args()

    args.backend = resolve_backend(args.backend, args.weights)
    run(args)


if __name__ == "__main__":
    main()
