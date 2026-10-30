#include "PersonDetector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace detection {

namespace {

// Ort::Session 생성자는 Windows에서 wchar_t*, Linux에서 char*를 요구한다(ORTCHAR_T).
// 이 프로젝트의 모델 경로는 전부 ASCII라 단순 폭 변환으로 충분하다.
#ifdef _WIN32
std::wstring ToOrtPath(const std::string& path) {
    return std::wstring(path.begin(), path.end());
}
#else
const std::string& ToOrtPath(const std::string& path) {
    return path;
}
#endif

}  // namespace

PersonDetector::PersonDetector(const std::string& modelPath, int imgWidth, int imgHeight,
    float confThreshold, float nmsThreshold)
    : env_(ORT_LOGGING_LEVEL_WARNING, "PersonDetector"),
    imgWidth_(imgWidth), imgHeight_(imgHeight),
    confThreshold_(confThreshold), nmsThreshold_(nmsThreshold) {
    try {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        auto ortPath = ToOrtPath(modelPath);
        session_ = std::make_unique<Ort::Session>(env_, ortPath.c_str(), sessionOptions);

        Ort::AllocatorWithDefaultOptions allocator;
        inputName_ = session_->GetInputNameAllocated(0, allocator).get();
        size_t outputCount = session_->GetOutputCount();
        for (size_t i = 0; i < outputCount; ++i) {
            outputNames_.push_back(session_->GetOutputNameAllocated(i, allocator).get());
        }

        loaded_ = true;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[PersonDetector] 모델 로드 실패: " << e.what() << std::endl;
        loaded_ = false;
    }
}

float PersonDetector::calculateIou(const DetectionBox& a, const DetectionBox& b) {
    float x1 = std::max(a.xmin, b.xmin);
    float y1 = std::max(a.ymin, b.ymin);
    float x2 = std::min(a.xmax, b.xmax);
    float y2 = std::min(a.ymax, b.ymax);

    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area1 = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    float area2 = (b.xmax - b.xmin) * (b.ymax - b.ymin);
    float unionVal = area1 + area2 - inter;

    return inter / (unionVal + 1e-6f);
}

std::vector<DetectionBox> PersonDetector::nms(std::vector<DetectionBox> boxes, float iouThreshold) {
    if (boxes.empty()) return {};

    std::sort(boxes.begin(), boxes.end(), [](const DetectionBox& a, const DetectionBox& b) {
        return a.conf > b.conf;
        });

    std::vector<DetectionBox> keep;
    while (!boxes.empty()) {
        DetectionBox best = boxes.front();
        boxes.erase(boxes.begin());
        keep.push_back(best);

        std::vector<DetectionBox> remain;
        for (const auto& box : boxes) {
            if (calculateIou(best, box) < iouThreshold) {
                remain.push_back(box);
            }
        }
        boxes = remain;
    }
    return keep;
}

// train_person_detector.py::decode_prediction과 완전히 동일한 수식.
// raw grid 출력(obj_logit, tx_logit, ty_logit, tw_log, th_log)을 sigmoid/exp로
// 복원한다. 한쪽만 고치면 같은 모델인데 결과가 달라지므로 Python 쪽과 반드시 같이 볼 것.
//
//   obj = sigmoid(pred[0])                       -> objectness/confidence
//   tx  = sigmoid(pred[1]) * 2 - 0.5              -> 다중 양성 셀 할당으로 [-0.5,1.5) 범위
//   ty  = sigmoid(pred[2]) * 2 - 0.5
//   tw  = clamp(pred[3], max=4.0)                 -> WH_LOG_CLAMP (exp 오버플로 방지)
//   th  = clamp(pred[4], max=4.0)
//   bw  = exp(tw) * img_w,  bh = exp(th) * img_h  -> log(bw/W) 인코딩의 정확한 역함수
//   cx  = (grid_x + tx) * cell_w,  cy = (grid_y + ty) * cell_h
//
// pred는 (5, gridH, gridW) 레이아웃의 float 버퍼(row-major, contiguous) 하나를
// 가리킨다 - ONNX Runtime의 GetTensorMutableData<float>()가 반환하는 포인터를
// 그대로 넘기면 된다. LibTorch 시절엔 torch::Tensor::accessor<float,3>()로 같은
// 메모리를 인덱싱했을 뿐, 수식과 순회 순서는 동일하다.
std::vector<DetectionBox> PersonDetector::decodePrediction(
    const float* pred, int64_t gridH, int64_t gridW, int imgW, int imgH, float confThreshold) {
    std::vector<DetectionBox> boxes;
    float cellW = static_cast<float>(imgW) / static_cast<float>(gridW);
    float cellH = static_cast<float>(imgH) / static_cast<float>(gridH);

    const int64_t planeSize = gridH * gridW;
    constexpr float kWhLogClamp = 4.0f;  // train_person_detector.py::WH_LOG_CLAMP

    auto sigmoidf = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
    auto at = [&](int64_t channel, int64_t y, int64_t x) {
        return pred[channel * planeSize + y * gridW + x];
        };

    for (int64_t y = 0; y < gridH; ++y) {
        for (int64_t x = 0; x < gridW; ++x) {
            float objScore = sigmoidf(at(0, y, x));
            if (objScore <= confThreshold) {
                continue;
            }

            float tx = sigmoidf(at(1, y, x)) * 2.0f - 0.5f;
            float ty = sigmoidf(at(2, y, x)) * 2.0f - 0.5f;
            float tw = std::min(at(3, y, x), kWhLogClamp);
            float th = std::min(at(4, y, x), kWhLogClamp);

            float bw = std::exp(tw) * static_cast<float>(imgW);
            float bh = std::exp(th) * static_cast<float>(imgH);

            float cx = (static_cast<float>(x) + tx) * cellW;
            float cy = (static_cast<float>(y) + ty) * cellH;

            float xmin = std::max(0.0f, cx - bw / 2.0f);
            float ymin = std::max(0.0f, cy - bh / 2.0f);
            float xmax = std::min(static_cast<float>(imgW), cx + bw / 2.0f);
            float ymax = std::min(static_cast<float>(imgH), cy + bh / 2.0f);

            boxes.push_back({ xmin, ymin, xmax, ymax, objScore });
        }
    }
    return boxes;
}

// FPN(9단계 등)처럼 모델 출력이 여러 개일 수도 있어 세션의 모든 출력 텐서를 순회한다.
// 단일 스케일(11단계)이면 출력이 하나라 루프가 1회만 돈다.
std::vector<DetectionBox> PersonDetector::decodeOutputs(
    const std::vector<Ort::Value>& outputs, int imgW, int imgH, float confThreshold) {
    std::vector<DetectionBox> boxes;

    for (const auto& output : outputs) {
        auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
        // shape: (1, 5, gridH, gridW) 또는 배치 차원 없이 (5, gridH, gridW).
        int64_t gridH, gridW;
        const float* data = output.GetTensorData<float>();
        if (shape.size() == 4) {
            gridH = shape[2];
            gridW = shape[3];
        }
        else if (shape.size() == 3) {
            gridH = shape[1];
            gridW = shape[2];
        }
        else {
            std::cerr << "[PersonDetector] 알 수 없는 출력 텐서 shape입니다 (rank="
                << shape.size() << ").\n";
            continue;
        }

        auto part = decodePrediction(data, gridH, gridW, imgW, imgH, confThreshold);
        boxes.insert(boxes.end(), part.begin(), part.end());
    }

    return boxes;
}

std::vector<DetectionBox> PersonDetector::detect(cv::Mat& frame, bool drawBoxes) {
    if (!loaded_ || frame.empty()) {
        return {};
    }

    int origW = frame.cols;
    int origH = frame.rows;

    // 11단계 등 커스텀 모델은 letterbox가 아니라 종횡비 유지 없는 단순 resize.
    // train_person_detector.py::PersonDataset._load_resized와 동일한 전처리.
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(imgWidth_, imgHeight_));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
    // custom 백본은 ImageNet 정규화를 쓰지 않는다(랜덤 초기화로 학습되어 0~1 그대로 씀).
    // mobilenet 백본 모델(9단계 등)을 나중에 붙이면 여기에 정규화 분기가 필요함 -
    // PersonDataset.normalize / detect_video.py의 --normalize 플래그 참고.

    // HWC(OpenCV Mat) -> CHW(모델 입력) 수동 변환. LibTorch 시절엔
    // torch::from_blob + permute로 했지만, ONNX Runtime은 std::vector<float> 버퍼를
    // Ort::Value::CreateTensor로 감싸는 방식이라 순서를 직접 맞춰 채운다.
    std::vector<float> inputBuffer(static_cast<size_t>(3) * imgHeight_ * imgWidth_);
    const int planeSize = imgHeight_ * imgWidth_;
    for (int y = 0; y < imgHeight_; ++y) {
        const cv::Vec3f* rowPtr = resized.ptr<cv::Vec3f>(y);
        for (int x = 0; x < imgWidth_; ++x) {
            const cv::Vec3f& px = rowPtr[x];
            inputBuffer[0 * planeSize + y * imgWidth_ + x] = px[0];  // R
            inputBuffer[1 * planeSize + y * imgWidth_ + x] = px[1];  // G
            inputBuffer[2 * planeSize + y * imgWidth_ + x] = px[2];  // B
        }
    }

    std::vector<Ort::Value> outputTensors;
    try {
        std::lock_guard<std::mutex> lock(modelMutex_);

        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 4> inputShape{ 1, 3, imgHeight_, imgWidth_ };
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, inputBuffer.data(), inputBuffer.size(), inputShape.data(), inputShape.size());

        const char* inputNames[] = { inputName_.c_str() };
        std::vector<const char*> outputNamesC;
        outputNamesC.reserve(outputNames_.size());
        for (const auto& name : outputNames_) {
            outputNamesC.push_back(name.c_str());
        }

        outputTensors = session_->Run(Ort::RunOptions{ nullptr }, inputNames, &inputTensor, 1,
            outputNamesC.data(), outputNamesC.size());
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[PersonDetector] 추론 실패: " << e.what() << std::endl;
        return {};
    }

    auto boxes = decodeOutputs(outputTensors, imgWidth_, imgHeight_, confThreshold_);
    boxes = nms(std::move(boxes), nmsThreshold_);

    std::vector<DetectionBox> result;
    result.reserve(boxes.size());
    float sx = static_cast<float>(origW) / static_cast<float>(imgWidth_);
    float sy = static_cast<float>(origH) / static_cast<float>(imgHeight_);

    for (const auto& box : boxes) {
        // 단순 resize 좌표계 -> 원본 프레임 좌표계. letterbox와 달리 pad 보정이
        // 없고, 종횡비를 유지하지 않고 늘렸으므로 x/y 배율(sx, sy)이 다를 수 있다.
        float xmin = std::max(0.0f, std::min(box.xmin * sx, static_cast<float>(origW)));
        float ymin = std::max(0.0f, std::min(box.ymin * sy, static_cast<float>(origH)));
        float xmax = std::max(0.0f, std::min(box.xmax * sx, static_cast<float>(origW)));
        float ymax = std::max(0.0f, std::min(box.ymax * sy, static_cast<float>(origH)));

        result.push_back({ xmin, ymin, xmax, ymax, box.conf });

        if (drawBoxes) {
            cv::rectangle(frame, cv::Point(static_cast<int>(xmin), static_cast<int>(ymin)),
                cv::Point(static_cast<int>(xmax), static_cast<int>(ymax)),
                cv::Scalar(0, 255, 0), 3);
            std::string confStr = cv::format("%.2f", box.conf);
            cv::putText(frame, confStr,
                cv::Point(static_cast<int>(xmin), std::max(20, static_cast<int>(ymin) - 10)),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }
    }

    return result;
}

std::vector<FootPoint> PersonDetector::getFootPoints(const std::vector<DetectionBox>& boxes) {
    std::vector<FootPoint> points;
    points.reserve(boxes.size());
    for (const auto& box : boxes) {
        points.push_back({ (box.xmin + box.xmax) / 2.0f, box.ymax, box.conf });
    }
    return points;
}

}  // namespace detection
