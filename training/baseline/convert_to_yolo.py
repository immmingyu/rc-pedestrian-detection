# YOLOv8n 베이스라인 실험용 라벨 변환
"""CSV(pascal_voc) 어노테이션 -> YOLO 형식(.txt) 변환 + images/labels 디렉터리 구성.
원본 archive/{split}/{split}/*.jpg + _annotations.csv 구조는 건드리지 않고,
archive_yolo/{split}/images(하드링크) + archive_yolo/{split}/labels(.txt)를 새로 만든다.
"""
import os

import cv2
import pandas as pd

SRC_ROOT = "./archive"
DST_ROOT = "./archive_yolo"
SPLITS = ["train", "valid", "test"]


def convert_split(split):
    src_dir = os.path.join(SRC_ROOT, split, split)
    csv_path = os.path.join(src_dir, "_annotations.csv")
    img_dst = os.path.join(DST_ROOT, split, "images")
    lbl_dst = os.path.join(DST_ROOT, split, "labels")
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df["class"] == "person"]

    by_file = df.groupby("filename")
    n_images, n_boxes = 0, 0

    for fname, group in by_file:
        src_img = os.path.join(src_dir, fname)
        if not os.path.exists(src_img):
            continue

        img = cv2.imread(src_img)
        if img is None:
            continue
        h, w = img.shape[:2]

        lines = []
        for _, row in group.iterrows():
            xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
            if xmax <= xmin or ymax <= ymin:
                continue
            cx = (xmin + xmax) / 2 / w
            cy = (ymin + ymax) / 2 / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            continue

        dst_img = os.path.join(img_dst, fname)
        if not os.path.exists(dst_img):
            os.link(src_img, dst_img)  # 하드링크: 복사 없이 같은 볼륨 내 파일 공유

        stem = os.path.splitext(fname)[0]
        with open(os.path.join(lbl_dst, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

        n_images += 1
        n_boxes += len(lines)

    print(f"{split}: images={n_images} boxes={n_boxes}")


if __name__ == "__main__":
    for split in SPLITS:
        convert_split(split)
