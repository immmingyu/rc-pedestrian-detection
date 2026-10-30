"""TorchScript 모델(.pt)을 ONNX로 변환하고, 두 모델의 출력이 같은지 검증한다.

models/person_detector_lite.onnx를 만들 때 실제로 쓴 변환과 같은 설정이다:
학습 스크립트(training/train_person_detector.py)가 저장한 TorchScript를 그대로 읽어
재학습 없이 export (opset 17, 입력 이름 "input", 출력 이름 "output", 입력 1x3x240x320).

사용법:
    python scripts/export_onnx.py --torchscript runs/11_custom_lite/person_detector_script.pt \
        --output models/person_detector_lite.onnx
"""
import argparse
import inspect

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description="TorchScript -> ONNX 변환 + 출력 일치 검증")
    p.add_argument("--torchscript", required=True, help="학습 후 저장된 TorchScript(.pt) 경로")
    p.add_argument("--output", required=True, help="저장할 .onnx 경로")
    p.add_argument("--img-width", type=int, default=320)
    p.add_argument("--img-height", type=int, default=240)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-verify", action="store_true", help="onnxruntime 출력 비교를 건너뜀")
    args = p.parse_args()

    model = torch.jit.load(args.torchscript, map_location="cpu")
    model.eval()
    dummy = torch.randn(1, 3, args.img_height, args.img_width)

    kwargs = dict(input_names=["input"], output_names=["output"],
                  opset_version=args.opset, do_constant_folding=True)
    # torch 2.5+는 dynamo 기반 exporter가 있고, 2.9+부터 기본값이 됐다. TorchScript 모델은
    # 기존(TorchScript 기반) exporter로 변환해야 하므로 지원하면 명시적으로 끈다.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(model, dummy, args.output, **kwargs)
    print(f"saved: {args.output}")

    if args.no_verify:
        return

    import onnxruntime as ort

    x = torch.rand(1, 3, args.img_height, args.img_width)
    with torch.no_grad():
        ref = model(x)
    ref = [r.numpy() for r in ref] if isinstance(ref, (list, tuple)) else [ref.numpy()]
    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})
    diff = max(float(np.abs(a - b).max()) for a, b in zip(ref, out))
    print(f"TorchScript vs ONNX max abs diff: {diff:.2e} (output shapes: {[o.shape for o in out]})")


if __name__ == "__main__":
    main()
