# YOLOv8n 베이스라인

COCO 사전학습 YOLOv8n을 같은 데이터셋으로 파인튜닝한 비교 기준입니다. 학습 명령은 각 실행의
`args.yaml`(Ultralytics 8.4.104가 자동 저장)에서 기본값과 다른 항목만 골라 CLI 형태로 옮긴 것입니다.
실제로 CLI와 Python API 중 어느 쪽으로 실행했는지는 기록이 없습니다.

## 1. 라벨 변환

`archive/`(CSV 라벨)가 있는 폴더에서 실행하면 `archive_yolo/{train,valid,test}/{images,labels}`를 만듭니다
(이미지는 하드링크).

```bash
python training/baseline/convert_to_yolo.py
```

`archive_yolo/data.yaml`은 변환 스크립트가 만들지 않으므로 직접 작성합니다(당시에는 `path`에 절대 경로를 썼음).

```yaml
path: ./archive_yolo
train: train/images
val: valid/images
test: test/images
names:
  0: person
```

## 2. 학습

```bash
# 320 입력
yolo detect train model=yolov8n.pt data=archive_yolo/data.yaml imgsz=320 epochs=50 batch=32 \
  patience=15 workers=4 device=0 project=yolo_runs name=yolov8n_person

# 416 입력 (imgsz와 name만 다름)
yolo detect train model=yolov8n.pt data=archive_yolo/data.yaml imgsz=416 epochs=50 batch=32 \
  patience=15 workers=4 device=0 project=yolo_runs name=yolov8n_person_416
```

나머지는 Ultralytics 기본값입니다(`optimizer=auto`, `lr0=0.01`, `mosaic=1.0`, `close_mosaic=10`, `seed=0`).
`optimizer=auto`일 때는 지정한 `lr0`·`momentum`이 무시되고 Ultralytics가 optimizer와 학습률을 다시 정합니다
(두 실행 모두 로그상 `MuSGD(lr=0.01, momentum=0.9)`가 선택됨). 이후 실행(`yolov8n_person_ft3`,
`yolov8n_person_416_wu_sgd`)에서는 `optimizer=SGD`를 명시했습니다.

## 3. 결과 (Ultralytics 평가, 검증 1,370장)

| 실행 | 입력 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| yolov8n_person | 320 | 0.765 | 0.521 | 0.592 | 0.337 |
| yolov8n_person_416 | 416 | 0.801 | 0.572 | 0.650 | 0.385 |

Ultralytics는 conf 0.001, NMS IoU 0.7, 101점 보간 AP로 평가하므로 커스텀 모델의 기준(conf 0.05, NMS IoU 0.3,
VOC all-point AP)과 다릅니다. 같은 기준으로 다시 재면 320 모델은 mAP@0.5 0.553입니다(최상위 README 결과 요약).
