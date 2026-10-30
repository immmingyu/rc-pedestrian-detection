"""박스 인코딩(build_target) -> 디코딩(decode_prediction) 왕복 오차 검증.

검증셋의 GT 박스를 학습 코드의 build_target으로 인코딩한 뒤, 그 타깃을 "모델이 완벽히
학습했을 때의 출력"(로짓)으로 바꿔 decode_prediction으로 복원하고 원래 박스와 비교한다.
비교용으로 수정 전 방식(w/h 선형 타깃을 sigmoid 공간에서 학습, 추론은 exp로 복원)도
같은 가정으로 계산해 박스가 얼마나 부풀었는지 보여준다.

사용법:
    python scripts/check_box_roundtrip.py --data-root ./archive
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
import train_person_detector as T  # noqa: E402

OBJ_LOGIT = 20.0  # sigmoid(20) ~= 1: 양성 셀이 확실히 검출되도록
EPS = 1e-12


def logit(p):
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def target_to_logits(target):
    """build_target 결과(디코드 공간 값)를 decode_prediction 입력(raw 출력) 공간으로 변환."""
    pred = torch.full(target.shape, -OBJ_LOGIT, dtype=torch.float64)
    ys, xs = torch.where(target[0] > 0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        pred[0, y, x] = OBJ_LOGIT
        # decode: t = sigmoid(raw) * 2 - 0.5  ->  raw = logit((t + 0.5) / 2)
        pred[1, y, x] = logit((float(target[1, y, x]) + 0.5) / 2.0)
        pred[2, y, x] = logit((float(target[2, y, x]) + 0.5) / 2.0)
        # decode: bw = exp(raw) * W  ->  raw = log(bw / W) = target 그대로
        pred[3, y, x] = float(target[3, y, x])
        pred[4, y, x] = float(target[4, y, x])
    return pred


def main():
    p = argparse.ArgumentParser(description="build_target -> decode_prediction 왕복 오차 검증")
    p.add_argument("--data-root", default="./archive")
    p.add_argument("--split", default="valid")
    p.add_argument("--img-width", type=int, default=320)
    p.add_argument("--img-height", type=int, default=240)
    p.add_argument("--grid-stride", type=int, default=16)
    args = p.parse_args()

    W, H = args.img_width, args.img_height
    gw, gh = T.get_grid_size(W, H, args.grid_stride)
    image_dir, csv_path = T.split_paths(args.data_root, args.split)
    ds = T.PersonDataset(image_dir, csv_path, W, H, transform=None, normalize=False)

    max_err, n_cells, n_gt, n_gt_matched = 0.0, 0, 0, 0
    old_w_ratio, old_h_ratio = [], []
    for i in range(len(ds)):
        _, boxes = ds[i]
        gts = boxes.tolist()
        n_gt += len(gts)

        target = T.build_target(boxes, gw, gh, W, H)
        decoded = T.decode_prediction(target_to_logits(target), W, H, conf_threshold=0.5)
        n_cells += len(decoded)

        matched = [False] * len(gts)
        for d in decoded:
            errs = [max(abs(d[k] - g[k]) for k in range(4)) for g in gts]
            j = int(np.argmin(errs))
            max_err = max(max_err, errs[j])
            if errs[j] < 1e-3:
                matched[j] = True
        n_gt_matched += sum(matched)

        # 수정 전 방식: 타깃 bw/W를 sigmoid 공간에서 맞췄다고 가정 -> raw = logit(bw/W) -> exp(raw)*W
        for x0, y0, x1, y1 in gts:
            bw, bh = x1 - x0, y1 - y0
            if bw <= 1 or bh <= 1:
                continue
            old_w_ratio.append(math.exp(logit(bw / W)) * W / bw)
            old_h_ratio.append(math.exp(logit(bh / H)) * H / bh)

    print(f"images: {len(ds)}  GT boxes: {n_gt}  positive cells decoded: {n_cells}")
    print(f"GT boxes recovered by at least one cell: {n_gt_matched} "
          f"(나머지 {n_gt - n_gt_matched}개는 할당될 셀을 이미 다른 박스가 차지하는 등의 이유로 타깃에 없음)")
    print(f"[현재 방식: log 인코딩 + exp 복원] 최대 좌표 오차: {max_err:.2e} px ({W}x{H} 기준)")
    # 선형 타깃 r을 sigmoid로 맞춘 raw를 exp로 풀면 r/(1-r) -> 박스가 클수록(r->1) 급격히 부풀어 최댓값은 무의미
    for label, ratios in (("폭", np.array(old_w_ratio)), ("높이", np.array(old_h_ratio))):
        q50, q90, q99 = np.percentile(ratios, [50, 90, 99])
        print(f"[수정 전 방식: 선형 타깃 + exp 복원] 복원 {label}/실제 {label}: "
              f"중앙값 {q50:.2f}배, 상위 10% {q90:.2f}배 이상, 상위 1% {q99:.1f}배 이상, "
              f"1.5배 이상 부푼 박스 {np.mean(ratios >= 1.5) * 100:.1f}%")


if __name__ == "__main__":
    main()
